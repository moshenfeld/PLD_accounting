"""Geometric-grid convolution for privacy loss distributions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain
from PLD_accounting.distribution_utils import (
    enforce_mass_conservation,
    stable_isclose,
)
from PLD_accounting.types import BoundType, SpacingType, has_numba, optional_njit
from PLD_accounting.utils import binary_self_convolve, convolve_boundary_masses
from PLD_accounting.validation import validate_bound_type

# Snap tolerance, in units of output bins, for mapping pairwise sums to grid
# indices. Log-space index computations carry absolute fp noise of up to
# ~|log(x / anchor)| * eps / log(ratio) ~ 1e-9 bins at realistic PLD magnitudes,
# so exact lattice hits need a tolerance above that to stay in their bin. A
# false snap misplaces mass by at most a factor ratio**tol ~ 1 + 1e-12, and an
# unsnapped hit still rounds in the conservative direction for its bound type.
_BIN_SNAP_TOL = 1e-8

# =============================================================================
# PUBLIC API
# =============================================================================


def geometric_convolve(
    *,
    dist_1: DenseDiscreteDist,
    dist_2: DenseDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
    target_anchor: float | None = None,
) -> DenseDiscreteDist:
    """Convolve two geometric-grid distributions.

    A wrapper of Algorithm 4 (`conv`), in
    Appendix C of https://arxiv.org/abs/2602.17284.
    For POSITIVES-domain distributions the 0 atom is neutral (not absorbing),
    so cross-terms (0 + finite and finite + 0) are added to the finite PMF.

    If ``target_anchor`` is provided, the output lies on its ``target_anchor * r**k``
    lattice. Rounding is directional for the bound type either way, so validity never
    requires the inputs to sit on any particular lattice; input alignment only sharpens
    the result by keeping exact-hit sums in their bin.
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
    if not stable_isclose(value_1=dist_1.step, value_2=dist_2.step):
        raise ValueError(
            f"Grid ratios must match: ratio_1={dist_1.step:.12g}, ratio_2={dist_2.step:.12g}"
        )
    ratio = dist_1.step

    if target_anchor is not None and target_anchor <= 0:
        raise ValueError(f"target_anchor must be positive, got {target_anchor}")

    # Convolve all finite-by-finite mass.
    output_origin, pmf_conv = _compute_geometric_convolution(
        origin_1=dist_1.x_0,
        pmf_1=dist_1.prob_arr,
        origin_2=dist_2.x_0,
        pmf_2=dist_2.prob_arr,
        ratio=ratio,
        bound_type=bound_type,
        target_anchor=target_anchor,
    )

    # Add each 0-by-finite cross-term and retain out-of-grid mass by side.
    pmf_conv, omitted_below_1, omitted_above_1 = _add_single_zero_atom_cross_term(
        pmf_conv=pmf_conv,
        x_arr=dist_2.x_array,
        prob_arr=dist_2.prob_arr,
        zero_prob=dist_1.p_min,
        x_out_0=output_origin,
        ratio=ratio,
        bound_type=bound_type,
    )
    pmf_conv, omitted_below_2, omitted_above_2 = _add_single_zero_atom_cross_term(
        pmf_conv=pmf_conv,
        x_arr=dist_1.x_array,
        prob_arr=dist_1.prob_arr,
        zero_prob=dist_2.p_min,
        x_out_0=output_origin,
        ratio=ratio,
        bound_type=bound_type,
    )

    # Account exactly for boundary-by-boundary convolution mass.
    expected_p_min, expected_p_max = convolve_boundary_masses(
        dist_1.p_min, dist_1.p_max, dist_2.p_min, dist_2.p_max, dist_1.domain
    )
    omitted_below = omitted_below_1 + omitted_below_2
    omitted_above = omitted_above_1 + omitted_above_2
    if bound_type == BoundType.DOMINATES:
        # Round lower underflow up to the first finite cell; upper overflow
        # becomes upper-boundary mass.
        pmf_conv[0] += omitted_below
        expected_p_max += omitted_above
    else:
        # Lower underflow becomes lower-boundary mass; round upper overflow
        # down to the last finite cell.
        expected_p_min += omitted_below
        pmf_conv[-1] += omitted_above

    # Repair only numerical drift after all omitted mass is explicit.
    pmf_conv, p_min, p_max = enforce_mass_conservation(
        prob_arr=pmf_conv,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_0=output_origin,
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
    num_convolutions: int,
    tail_truncation: float,
    bound_type: BoundType,
    lattice_anchor: float | None = None,
) -> DenseDiscreteDist:
    """Self-convolve using either ordinary or anchored binary composition.

    When ``lattice_anchor`` is provided, the input grid is interpreted as
    ``lattice_anchor * r**k`` and each intermediate output uses the summed-anchor
    lattice. Precondition: ``dist.x_0`` should lie on that lattice (up to fp noise);
    a misaligned input still yields a valid bound, but the output no longer sits on
    the claimed ``num_convolutions * lattice_anchor * r**k`` grid.
    """
    # Input validation
    if not (isinstance(dist, DenseDiscreteDist) and dist.spacing_type == SpacingType.GEOMETRIC):
        spacing = getattr(dist, "spacing_type", "?")
        raise TypeError(
            "geometric_self_convolve requires DenseDiscreteDist input: "
            "expected DenseDiscreteDist with GEOMETRIC spacing, "
            f"got {type(dist).__name__} with spacing {spacing}"
        )
    validate_bound_type(bound_type)
    if num_convolutions < 1:
        raise ValueError(f"num_convolutions must be >= 1, got {num_convolutions}")
    if tail_truncation < 0:
        raise ValueError(f"tail_truncation must be non-negative, got {tail_truncation}")
    if lattice_anchor is not None and lattice_anchor <= 0:
        raise ValueError(f"lattice_anchor must be positive, got {lattice_anchor}")

    return binary_self_convolve(
        dist=dist,
        num_convolutions=num_convolutions,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=geometric_convolve,
        lattice_anchor=lattice_anchor,
    )


# =============================================================================
# INTERNAL KERNEL IMPLEMENTATION
# =============================================================================


def _compute_geometric_convolution(
    *,
    origin_1: float,
    pmf_1: NDArray[np.float64],
    origin_2: float,
    pmf_2: NDArray[np.float64],
    ratio: float,
    bound_type: BoundType,
    target_anchor: float | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """Align grids, compute bin mapping parameters, and invoke the Numba kernel.

    Algorithm 4 (`conv`), with internal
    Algorithm 5 (`range-renorm`),
    in Appendix C of https://arxiv.org/abs/2602.17284.

    Each geometric support is fully specified by its origin, shared ratio, and
    PMF length, so materialized input grids are unnecessary.
    """
    # --- A. Standardization (Swap & Pad) ---
    # Keep the lower-origin distribution first to match the kernel's two orientations.
    if origin_1 > origin_2:
        origin_1, pmf_1, origin_2, pmf_2 = origin_2, pmf_2, origin_1, pmf_1

    smallest_sum = origin_1 + origin_2
    largest_sum = origin_1 * ratio ** (pmf_1.size - 1) + origin_2 * ratio ** (pmf_2.size - 1)

    # The kernel requires equal PMF lengths; missing high-grid bins have zero mass.
    num_bins = max(pmf_1.size, pmf_2.size)
    pmf_1 = np.pad(pmf_1, (0, num_bins - pmf_1.size), mode="constant")
    pmf_2 = np.pad(pmf_2, (0, num_bins - pmf_2.size), mode="constant")

    # Convert to float64 for Numba compatibility.
    lower_pmf = pmf_1.astype(np.float64, copy=False)
    upper_pmf = pmf_2.astype(np.float64, copy=False)

    # --- B. Grid Mapping Parameters ---
    log_ratio = np.log(ratio)

    # Use one bound-independent grid covering the complete input-grid sum range.
    # When anchored, start at the lattice point at or below the smallest sum.
    # smallest_sum_offset re-expresses the smallest sum in bins from that
    # origin, reusing the same unrounded index so every later snap decision
    # stays consistent with the floor taken here.
    output_origin = smallest_sum
    smallest_sum_offset = 0.0
    if target_anchor is not None:
        smallest_sum_index = math.log(smallest_sum / target_anchor) / log_ratio
        output_index = math.floor(smallest_sum_index + _BIN_SNAP_TOL)
        output_origin = target_anchor * ratio**output_index
        smallest_sum_offset = smallest_sum_index - output_index

    # End at the lattice point at or above the largest sum.
    last_output_index = math.ceil(
        math.log(largest_sum / smallest_sum) / log_ratio + smallest_sum_offset - _BIN_SNAP_TOL
    )
    num_output_bins = last_output_index + 1

    index_differences = np.arange(num_bins, dtype=np.float64)
    log_ratio_offsets = index_differences * log_ratio
    # Normalized origin weights (w_1 + w_2 = 1) keep the log arguments O(1):
    # (origin_1 + origin_2 * r**d) / smallest_sum = w_1 + w_2 * r**d, so the
    # d = 0 diagonal offset is log(w_1 + w_2) ~ 0 to fp precision regardless of
    # the origins' magnitudes, and exact lattice hits stay inside snap range.
    log_weight_1 = np.log(origin_1 / smallest_sum)
    log_weight_2 = np.log(origin_2 / smallest_sum)

    # For an index difference d, the pair can occur in either orientation.
    # Measure both sums in output-grid bins from output_origin.
    low_high_grid_offsets = (
        np.logaddexp(log_weight_1, log_weight_2 + log_ratio_offsets) / log_ratio
        + smallest_sum_offset
    )
    high_low_grid_offsets = (
        np.logaddexp(log_weight_2, log_weight_1 + log_ratio_offsets) / log_ratio
        + smallest_sum_offset
    )

    # Bound type affects only mass allocation, not output-grid construction.
    if bound_type == BoundType.DOMINATES:
        # A dominating discretization rounds each sum up to the next output bin.
        low_high_bin_offsets = np.ceil(low_high_grid_offsets - _BIN_SNAP_TOL).astype(np.int64)
        high_low_bin_offsets = np.ceil(high_low_grid_offsets - _BIN_SNAP_TOL).astype(np.int64)
    elif bound_type == BoundType.IS_DOMINATED:
        # A dominated discretization rounds each sum down to the previous output bin.
        low_high_bin_offsets = np.floor(low_high_grid_offsets + _BIN_SNAP_TOL).astype(np.int64)
        high_low_bin_offsets = np.floor(high_low_grid_offsets + _BIN_SNAP_TOL).astype(np.int64)
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    # --- C. Kernel Execution ---
    pmf_out = _geometric_kernel(
        pmf_base=lower_pmf,
        pmf_scaled=upper_pmf,
        delta_lohi=low_high_bin_offsets,
        delta_hilo=high_low_bin_offsets,
        output_size=num_output_bins,
    )

    return output_origin, pmf_out


def _geometric_kernel(
    *,
    pmf_base: NDArray[np.float64],
    pmf_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
    output_size: int,
) -> NDArray[np.float64]:
    """Dispatch geometric convolution to numba when available, else NumPy."""
    if has_numba():
        return _numba_geometric_kernel(
            pmf_base=pmf_base,
            pmf_scaled=pmf_scaled,
            delta_lohi=delta_lohi,
            delta_hilo=delta_hilo,
            output_size=output_size,
        )
    return _numpy_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        output_size=output_size,
    )


@optional_njit()
def _numba_geometric_kernel(
    *,
    pmf_base: NDArray[np.float64],
    pmf_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
    output_size: int,
) -> NDArray[np.float64]:
    """Core convolution loop with compensated summation.

    The diagonal ``d = 0`` is counted once and shifted to its rounded bin.
    For ``d > 0`` both orderings ``(i, i + d)`` and ``(i + d, i)`` are
    scattered to their rounded output bins.
    """
    n = pmf_base.size
    pmf_out = np.zeros(output_size, dtype=np.float64)
    comp = np.zeros(output_size, dtype=np.float64)

    diagonal_shift = delta_lohi[0]
    for i in range(n):
        k = i + diagonal_shift
        mass = pmf_base[i] * pmf_scaled[i]
        if 0 <= k < output_size:
            y = mass - comp[k]
            t = pmf_out[k] + y
            comp[k] = (t - pmf_out[k]) - y
            pmf_out[k] = t

    for d in range(1, n):
        imax = n - d
        kshift1 = delta_lohi[d]
        kshift2 = delta_hilo[d]

        for i in range(imax):
            k1 = i + kshift1
            mass1 = pmf_base[i] * pmf_scaled[i + d]
            if 0 <= k1 < output_size:
                y = mass1 - comp[k1]
                t = pmf_out[k1] + y
                comp[k1] = (t - pmf_out[k1]) - y
                pmf_out[k1] = t

            k2 = i + kshift2
            mass2 = pmf_base[i + d] * pmf_scaled[i]
            if 0 <= k2 < output_size:
                y = mass2 - comp[k2]
                t = pmf_out[k2] + y
                comp[k2] = (t - pmf_out[k2]) - y
                pmf_out[k2] = t

    return pmf_out


def _numpy_geometric_kernel(
    *,
    pmf_base: NDArray[np.float64],
    pmf_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
    output_size: int,
) -> NDArray[np.float64]:
    """Numpy fallback for the geometric convolution kernel."""
    n = pmf_base.size
    pmf_out = np.zeros(output_size, dtype=np.float64)
    diagonal_indices = np.arange(n) + delta_lohi[0]
    valid_diagonal = (0 <= diagonal_indices) & (diagonal_indices < output_size)
    np.add.at(
        pmf_out,
        diagonal_indices[valid_diagonal],
        (pmf_base * pmf_scaled)[valid_diagonal],
    )
    for d in range(1, n):
        imax = n - d
        base_idx = np.arange(imax)
        k1 = base_idx + delta_lohi[d]
        mass1 = pmf_base[:imax] * pmf_scaled[d:]
        valid1 = (0 <= k1) & (k1 < output_size)
        np.add.at(pmf_out, k1[valid1], mass1[valid1])

        k2 = base_idx + delta_hilo[d]
        mass2 = pmf_base[d:] * pmf_scaled[:imax]
        valid2 = (0 <= k2) & (k2 < output_size)
        np.add.at(pmf_out, k2[valid2], mass2[valid2])
    return pmf_out


def _add_single_zero_atom_cross_term(
    *,
    pmf_conv: NDArray[np.float64],
    x_arr: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    zero_prob: float,
    x_out_0: float,
    ratio: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], float, float]:
    """Map 0+finite cross-terms and return omitted mass below and above the grid."""
    if np.any(x_arr <= 0.0):
        raise ValueError("0+finite cross-term support values must be strictly positive")
    if zero_prob == 0.0:
        return pmf_conv, 0.0, 0.0

    # Each cross-term has probability zero_prob times its finite-atom mass.
    masses = prob_arr * zero_prob
    frac_k = np.log(x_arr / x_out_0) / math.log(ratio)
    if bound_type == BoundType.DOMINATES:
        k = np.ceil(frac_k - _BIN_SNAP_TOL).astype(np.int64)
    elif bound_type == BoundType.IS_DOMINATED:
        k = np.floor(frac_k + _BIN_SNAP_TOL).astype(np.int64)
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    below = k < 0
    above = k >= pmf_conv.size
    in_range = ~(below | above)
    # Accumulate representable cross-terms into their rounded finite bins.
    np.add.at(pmf_conv, k[in_range], masses[in_range])
    omitted_below = math.fsum(map(float, masses[below]))
    omitted_above = math.fsum(map(float, masses[above]))
    return pmf_conv, omitted_below, omitted_above
