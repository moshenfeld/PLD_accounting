from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

from PLD_accounting.types import BoundType
from PLD_accounting.discrete_dist import GeometricDiscreteDist
from PLD_accounting.distribution_utils import compute_bin_ratio_two_arrays, enforce_mass_conservation
from PLD_accounting.utils import binary_self_convolve, convolve_infinite_masses


# =============================================================================
# PUBLIC API
# =============================================================================


def geometric_convolve(*,
    dist_1: GeometricDiscreteDist,
    dist_2: GeometricDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
) -> GeometricDiscreteDist:
    """
    Convolve two geometric-grid distributions.

    Algorithm 4 (`conv`) in Appendix C wrapper.
    """
    # Ensure both inputs share the same growth factor.
    ratio = compute_bin_ratio_two_arrays(x_array_1=dist_1.x_array, x_array_2=dist_2.x_array)

    # Core Numeric Convolution
    x_out, pmf_conv = _compute_geometric_convolution(
        x1=dist_1.x_array,
        p1=dist_1.PMF_array,
        x2=dist_2.x_array,
        p2=dist_2.PMF_array,
        r=ratio,
        bound_type=bound_type,
    )

    expected_neg_inf, expected_pos_inf = convolve_infinite_masses(
        p_neg_inf_1=dist_1.p_neg_inf,
        p_pos_inf_1=dist_1.p_pos_inf,
        p_neg_inf_2=dist_2.p_neg_inf,
        p_pos_inf_2=dist_2.p_pos_inf,
    )
    pmf_conv, p_neg_inf, p_pos_inf = enforce_mass_conservation(
        PMF_array=pmf_conv,
        expected_neg_inf=expected_neg_inf,
        expected_pos_inf=expected_pos_inf,
        bound_type=bound_type,
    )

    return GeometricDiscreteDist(
        x_min=float(x_out[0]),
        ratio=ratio,
        PMF_array=pmf_conv,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
    ).truncate_edges(tail_truncation, bound_type)



def geometric_self_convolve(*,
    dist: GeometricDiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> GeometricDiscreteDist:
    """
    Self-convolve distribution T times using binary exponentiation.
    """
    self_conv = binary_self_convolve(
        dist=dist,
        T=T,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=geometric_convolve,
    )
    if not isinstance(self_conv, GeometricDiscreteDist):
        raise TypeError(f"Expected GeometricDiscreteDist from self-convolution, got {type(self_conv)}")
    return self_conv


# =============================================================================
# INTERNAL KERNEL IMPLEMENTATION
# =============================================================================


def _compute_geometric_convolution(*,
    x1: NDArray[np.float64],
    p1: NDArray[np.float64],
    x2: NDArray[np.float64],
    p2: NDArray[np.float64],
    r: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Align grids, compute bin mapping parameters, and invoke the Numba kernel.

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
    rounding_eps = 1e-16

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


def _pad_right_geometric(*,
    x: NDArray[np.float64],
    p: NDArray[np.float64],
    r: float,
    target_n: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Helper: Extend grid to the right to reach target_n using ratio r.
    """
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
def _numba_geometric_kernel(*,
    PMF_base: NDArray[np.float64],
    PMF_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    Core convolution loop.
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
