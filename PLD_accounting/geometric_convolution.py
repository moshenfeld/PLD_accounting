"""Geometric-grid convolution for privacy loss distributions."""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numpy.typing import NDArray

from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain
from PLD_accounting.distribution_utils import (
    enforce_mass_conservation,
    stable_isclose,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import (
    binary_self_convolve,
    convolve_boundary_masses,
)
from PLD_accounting.validation import validate_bound_type

# Rounding tolerance for grid bin mapping — must stay at machine-epsilon scale
# to avoid misrouting mass between bins.
_GRID_ROUNDING_TOL = 10 * np.finfo(np.float64).eps

# =============================================================================
# PUBLIC API
# =============================================================================


def geometric_convolve(
    *,
    dist_1: DenseDiscreteDist,
    dist_2: DenseDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Convolve two geometric-grid distributions.

    Algorithm 4 (`conv`) in Appendix C wrapper.
    For POSITIVES-domain distributions the 0 atom is neutral (not absorbing),
    so cross-terms (0 + finite and finite + 0) are added to the finite PMF.
    """
    # Input validation
    if not (
        isinstance(dist_1, DenseDiscreteDist)
        and dist_1.spacing_type == SpacingType.GEOMETRIC
        and dist_1.domain == Domain.POSITIVES
    ) or not (
        isinstance(dist_2, DenseDiscreteDist)
        and dist_2.spacing_type == SpacingType.GEOMETRIC
        and dist_2.domain == Domain.POSITIVES
    ):
        raise TypeError(
            "geometric_convolve requires geometric DenseDiscreteDist inputs on "
            f"Domain.POSITIVES; got dist_1={type(dist_1).__name__} "
            f"(spacing={dist_1.spacing_type}, domain={dist_1.domain}), "
            f"dist_2={type(dist_2).__name__} "
            f"(spacing={dist_2.spacing_type}, domain={dist_2.domain})"
        )
    if tail_truncation < 0:
        raise ValueError(f"tail_truncation must be non-negative, got {tail_truncation}")

    # Ensure both inputs share the same growth factor.
    if not stable_isclose(a=dist_1.step, b=dist_2.step):
        raise ValueError(
            f"Grid ratios must match: ratio_1={dist_1.step:.12g}, ratio_2={dist_2.step:.12g}"
        )
    ratio = dist_1.step

    # Core Numeric Convolution
    x_out, pmf_conv = _compute_geometric_convolution(
        x1=dist_1.x_array,
        p1=dist_1.prob_arr,
        x2=dist_2.x_array,
        p2=dist_2.prob_arr,
        r=ratio,
        bound_type=bound_type,
    )

    # Add cross-terms from the 0 atom
    x_out_0 = float(x_out[0])
    pmf_conv = _add_single_zero_atom_cross_term(
        pmf_conv=pmf_conv,
        x_arr=dist_2.x_array,
        prob_arr=dist_2.prob_arr,
        zero_prob=dist_1.p_min,
        x_out_0=x_out_0,
        r=ratio,
        bound_type=bound_type,
    )
    pmf_conv = _add_single_zero_atom_cross_term(
        pmf_conv=pmf_conv,
        x_arr=dist_1.x_array,
        prob_arr=dist_1.prob_arr,
        zero_prob=dist_2.p_min,
        x_out_0=x_out_0,
        r=ratio,
        bound_type=bound_type,
    )

    expected_p_min, expected_p_max = convolve_boundary_masses(
        dist_1.p_min, dist_1.p_max, dist_2.p_min, dist_2.p_max, dist_1.domain
    )

    pmf_conv, p_min, p_max = enforce_mass_conservation(
        prob_arr=pmf_conv,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_min=float(x_out[0]),
        step=ratio,
        prob_arr=pmf_conv,
        p_min=p_min,
        p_max=p_max,
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    ).truncate_edges(tail_truncation, bound_type)


def geometric_self_convolve(
    *,
    dist: DenseDiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Self-convolve distribution T times using binary exponentiation."""
    # Input validation
    if not (isinstance(dist, DenseDiscreteDist) and dist.spacing_type == SpacingType.GEOMETRIC):
        raise TypeError(f"dist must be DenseDiscreteDist, got {type(dist)}")
    validate_bound_type(bound_type)
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if tail_truncation < 0:
        raise ValueError(f"tail_truncation must be non-negative, got {tail_truncation}")

    self_conv = binary_self_convolve(
        dist=dist,
        T=T,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=geometric_convolve,
    )
    if not (
        isinstance(self_conv, DenseDiscreteDist) and self_conv.spacing_type == SpacingType.GEOMETRIC
    ):
        raise TypeError(f"Expected DenseDiscreteDist from self-convolution, got {type(self_conv)}")
    return self_conv


# =============================================================================
# INTERNAL KERNEL IMPLEMENTATION
# =============================================================================


def _compute_geometric_convolution(
    *,
    x1: NDArray[np.float64],
    p1: NDArray[np.float64],
    x2: NDArray[np.float64],
    p2: NDArray[np.float64],
    r: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Align grids, compute bin mapping parameters, and invoke the Numba kernel.

    Algorithm 4 (`conv`) with internal Algorithm 5 (`range-renorm`) in Appendix C.
    """
    # --- A. Standardization (Swap & Pad) ---
    # We normalize such that x_base (x1) starts at the lower value.
    # This ensures scale = x2[0]/x1[0] >= 1, simplifying log calculations.

    # 1. Swap if necessary so x1[0] <= x2[0]
    if x1[0] > x2[0]:
        x1, p1, x2, p2 = x2, p2, x1, p1

    # 2. Calculate Scale (Relative Offset)
    scale = x2[0] / x1[0]

    # 3. Equalize Lengths (Right-Padding)
    # The Numba kernel assumes arrays of equal length 'n'.
    target_n = max(x1.size, x2.size)
    if x1.size < target_n:
        x1, p1 = _pad_right_geometric(
            x=x1,
            p=p1,
            r=r,
            target_n=target_n,
        )
    elif x2.size < target_n:
        x2, p2 = _pad_right_geometric(
            x=x2,
            p=p2,
            r=r,
            target_n=target_n,
        )

    # Convert to float64 for Numba compatibility
    x_base = x1.astype(np.float64, copy=False)
    pmf_base = p1.astype(np.float64, copy=False)
    pmf_scaled = p2.astype(np.float64, copy=False)

    # --- B. Grid Mapping Parameters ---
    n = x_base.size

    # Edge case: Single point
    if n == 1:
        mass = pmf_base[0] * pmf_scaled[0]
        x_out = np.array([(scale + 1.0) * x_base[0]], dtype=np.float64)
        pmf_out = np.array([mass], dtype=np.float64)
        return x_out, pmf_out

    # Calculate shift parameters (delta)
    log_r = np.log(r)
    log_scale = np.log(scale)
    log_ap1 = np.log(scale + 1.0)

    # Vectorized calculation for d=1..n-1
    d_vec = np.arange(n, dtype=np.float64)
    log_r_d = d_vec * log_r

    log_lohi = np.logaddexp(0.0, log_scale + log_r_d)  # log(1 + scale*r^d)
    tau_lohi = (log_lohi - log_ap1) / log_r

    log_hilo = np.logaddexp(log_scale, log_r_d)  # log(scale + r^d)
    tau_hilo = (log_hilo - log_ap1) / log_r

    # Rounding strategy
    delta_lohi = np.zeros(n, dtype=np.int64)
    delta_hilo = np.zeros(n, dtype=np.int64)
    rounding_eps = _GRID_ROUNDING_TOL

    if bound_type == BoundType.DOMINATES:
        # Pessimistic: Round UP
        delta_lohi[1:] = np.ceil(tau_lohi[1:] - rounding_eps).astype(np.int64)
        delta_hilo[1:] = np.ceil(tau_hilo[1:] - rounding_eps).astype(np.int64)
    elif bound_type == BoundType.IS_DOMINATED:
        # Optimistic: Round DOWN
        delta_lohi[1:] = np.floor(tau_lohi[1:] + rounding_eps).astype(np.int64)
        delta_hilo[1:] = np.floor(tau_hilo[1:] + rounding_eps).astype(np.int64)
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    # --- C. Kernel Execution ---
    pmf_out = _numba_geometric_kernel(
        PMF_base=pmf_base,
        PMF_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
    )

    # Construct output X grid: x_out = x_base * (1 + scale)
    x_out = x_base * (scale + 1.0)

    return x_out, pmf_out


def _pad_right_geometric(
    *,
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    r: float,
    target_n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extend grid to the right to reach target_n using ratio r."""
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    n = x.size
    if n >= target_n:
        return x, p

    k = target_n - n
    if r == 1.0:
        tail = np.full(k, x[-1], dtype=np.float64)
    else:
        tail = x[-1] * (r ** np.arange(1, k + 1, dtype=np.float64))

    x_ext = np.concatenate([x, tail])
    p_ext = np.pad(p, (0, k), mode="constant")
    return x_ext, p_ext


@njit(cache=True)
def _numba_geometric_kernel(
    *,
    PMF_base: NDArray[np.float64],
    PMF_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Core convolution loop.

    Calculates Z = X + Y by iterating over the difference 'd' between indices.

    """
    n = PMF_base.size
    pmf_out = np.zeros(n, dtype=np.float64)
    comp = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mass = PMF_base[i] * PMF_scaled[i]
        y = mass - comp[i]
        t = pmf_out[i] + y
        comp[i] = (t - pmf_out[i]) - y
        pmf_out[i] = t

    for d in range(1, n):
        imax = n - d
        kshift1 = int(delta_lohi[d])
        kshift2 = int(delta_hilo[d])

        for i in range(imax):
            k1 = i + kshift1
            mass1 = PMF_base[i] * PMF_scaled[i + d]
            if 0 <= k1 < n:
                y = mass1 - comp[k1]
                t = pmf_out[k1] + y
                comp[k1] = (t - pmf_out[k1]) - y
                pmf_out[k1] = t

            k2 = i + kshift2
            mass2 = PMF_base[i + d] * PMF_scaled[i]
            if 0 <= k2 < n:
                y = mass2 - comp[k2]
                t = pmf_out[k2] + y
                comp[k2] = (t - pmf_out[k2]) - y
                pmf_out[k2] = t

    return pmf_out


def _add_single_zero_atom_cross_term(
    *,
    pmf_conv: NDArray[np.float64],
    x_arr: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    zero_prob: float,
    x_out_0: float,
    r: float,
    bound_type: BoundType,
) -> NDArray[np.float64]:
    """Map one family of 0+finite cross-terms onto the fixed output grid."""
    if zero_prob == 0.0:
        return pmf_conv

    return _numba_add_single_zero_atom_cross_term(
        pmf_out=pmf_conv,
        x_vals=np.asarray(x_arr, dtype=np.float64),
        prob_arr=np.asarray(prob_arr, dtype=np.float64),
        zero_prob=float(zero_prob),
        x_out_0=float(x_out_0),
        log_r=float(math.log(r)),
        dominates=(bound_type == BoundType.DOMINATES),
    )


@njit(cache=True)
def _numba_add_single_zero_atom_cross_term(
    *,
    pmf_out: NDArray[np.float64],
    x_vals: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    zero_prob: float,
    x_out_0: float,
    log_r: float,
    dominates: bool,
) -> NDArray[np.float64]:
    """Core loop for one 0+finite cross-term family."""
    n = pmf_out.size

    for i in range(prob_arr.size):
        weight = prob_arr[i] * zero_prob
        if weight == 0.0:
            continue

        x = x_vals[i]
        if x <= 0.0:
            continue

        frac_k = math.log(x / x_out_0) / log_r
        if dominates:
            k = int(math.ceil(frac_k - _GRID_ROUNDING_TOL))
        else:
            k = int(math.floor(frac_k + _GRID_ROUNDING_TOL))

        if 0 <= k < n:
            pmf_out[k] += weight
        elif k < 0 and dominates:
            # Upper-bound rounding maps sub-grid mass to the first finite bin.
            pmf_out[0] += weight

    return pmf_out
