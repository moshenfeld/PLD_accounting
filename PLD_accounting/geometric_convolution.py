"""Geometric-grid convolution for privacy loss distributions."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    require_geometric_positives_dist,
)
from PLD_accounting.distribution_utils import (
    enforce_mass_conservation,
)
from PLD_accounting.types import (
    BoundType,
    SpacingType,
    has_numba,
    optional_njit,
    require_bound_type,
)
from PLD_accounting.utils import binary_self_convolve, convolve_boundary_masses
from PLD_accounting.validation import (
    require_nonnegative_real,
    require_positive_int,
    require_positive_real,
)

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

    Algorithm 4 (`conv`) in Appendix C of https://arxiv.org/abs/2602.17284.
    In the POSITIVES domain the lower-boundary atom is at zero, which is neutral
    under addition rather than absorbing; its cross-terms with finite atoms are
    therefore included explicitly.

    If ``target_anchor`` is provided, the output lies on that
    ``target_anchor * r**k`` lattice. Rounding is directional for the bound type
    either way, so validity never requires the inputs to sit on any particular
    lattice; input alignment only sharpens exact-hit sums.
    """
    require_geometric_positives_dist(dist=dist_1, name="dist_1")
    require_geometric_positives_dist(dist=dist_2, name="dist_2")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_bound_type(value=bound_type)
    if target_anchor is not None:
        require_positive_real(value=target_anchor, name="target_anchor")
    # The kernel requires both inputs to use one exact geometric growth rate.
    if dist_1.grid.step != dist_2.grid.step:
        # Compare the stored log spacing: two distinct log steps can exponentiate to the
        # same ratio, which would merge two different lattices.
        raise ValueError(
            "Geometric grids must share one exact log spacing: "
            f"{dist_1.grid.step:.17g} vs {dist_2.grid.step:.17g}"
        )

    # First convolve all finite-by-finite mass.
    output_grid, pmf_conv = _compute_geometric_convolution(
        grid_1=dist_1.grid,
        pmf_1=dist_1.prob_arr,
        grid_2=dist_2.grid,
        pmf_2=dist_2.prob_arr,
        bound_type=bound_type,
        output_anchor=target_anchor,
    )

    # Then add both zero-by-finite cross-terms, retaining overflow by side.
    pmf_conv, omitted_below_1, omitted_above_1 = _add_single_zero_atom_cross_term(
        pmf_conv=pmf_conv,
        x_arr=dist_2.x_array,
        prob_arr=dist_2.prob_arr,
        zero_prob=dist_1.p_min,
        output_grid=output_grid,
        bound_type=bound_type,
    )
    pmf_conv, omitted_below_2, omitted_above_2 = _add_single_zero_atom_cross_term(
        pmf_conv=pmf_conv,
        x_arr=dist_1.x_array,
        prob_arr=dist_1.prob_arr,
        zero_prob=dist_2.p_min,
        output_grid=output_grid,
        bound_type=bound_type,
    )

    # Boundary-by-boundary products are accounted for independently of the kernel.
    expected_p_min, expected_p_max = convolve_boundary_masses(
        p_min_1=dist_1.p_min,
        p_max_1=dist_1.p_max,
        p_min_2=dist_2.p_min,
        p_max_2=dist_2.p_max,
        domain=dist_1.domain,
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
        grid=output_grid.with_n(pmf_conv.size),
        prob_arr=pmf_conv,
        p_min=p_min,
        p_max=p_max,
        domain=Domain.POSITIVES,
    ).truncate_edges(tail_truncation=tail_truncation, bound_type=bound_type)


def geometric_self_convolve(
    *,
    dist: DenseDiscreteDist,
    num_convolutions: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Self-convolve a geometric-grid distribution by binary composition.

    Each intermediate result carries its own ``GridSpec``, so pairwise convolution
    derives the summed-anchor lattice from its two inputs.
    """
    require_geometric_positives_dist(dist=dist, name="dist")
    require_bound_type(value=bound_type)
    require_positive_int(value=num_convolutions, name="num_convolutions")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")

    def _convolve(
        *,
        dist_1: DenseDiscreteDist,
        dist_2: DenseDiscreteDist,
        tail_truncation: float,
        bound_type: BoundType,
    ) -> DenseDiscreteDist:
        return geometric_convolve(
            dist_1=dist_1,
            dist_2=dist_2,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )

    return binary_self_convolve(
        dist=dist,
        num_convolutions=num_convolutions,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=_convolve,
    )


# =============================================================================
# INTERNAL KERNEL IMPLEMENTATION
# =============================================================================


def _compute_geometric_convolution(
    *,
    grid_1: GridSpec,
    pmf_1: NDArray[np.float64],
    grid_2: GridSpec,
    pmf_2: NDArray[np.float64],
    bound_type: BoundType,
    output_anchor: float | None,
) -> tuple[GridSpec, NDArray[np.float64]]:
    """Align grids, compute bin mapping parameters, and invoke the Numba kernel.

    Algorithm 4 (`conv`), with internal
    Algorithm 5 (`range-renorm`),
    in Appendix C of https://arxiv.org/abs/2602.17284.

    The two supports stay structural: their anchors and integer indices define
    the target lattice, while log-space offsets determine only the directional
    placement of pairwise sums. Returns that output ``GridSpec`` (with a
    provisional ``n``) and the finite PMF.
    """
    # --- A. Standardization (Swap & Pad) ---
    # Keep the lower-origin distribution first to match the kernel's two orientations.
    if grid_1.x_0 > grid_2.x_0:
        grid_1, pmf_1, grid_2, pmf_2 = grid_2, pmf_2, grid_1, pmf_1

    log_ratio = grid_1.step
    origin_1 = grid_1.x_0
    origin_2 = grid_2.x_0
    smallest_sum = origin_1 + origin_2

    # The kernel requires equal PMF lengths; missing high-grid bins have zero mass.
    size_1, size_2 = pmf_1.size, pmf_2.size
    num_bins = max(size_1, size_2)
    pmf_1 = np.pad(pmf_1, (0, num_bins - pmf_1.size), mode="constant")
    pmf_2 = np.pad(pmf_2, (0, num_bins - pmf_2.size), mode="constant")

    # The numerical kernels require float64 arrays even if callers use another dtype.
    lower_pmf = pmf_1.astype(np.float64, copy=False)
    upper_pmf = pmf_2.astype(np.float64, copy=False)

    # --- B. Grid Mapping Parameters ---
    # The lattice is the summed anchors; a validated output_anchor is the caller's
    # canonical name for that same lattice. The grid is sized to hold the one-bin shift.
    target_anchor = grid_1.anchor + grid_2.anchor if output_anchor is None else output_anchor
    smallest_sum_index = math.log(smallest_sum / target_anchor) / log_ratio
    output_index_0 = int(math.floor(smallest_sum_index))
    smallest_sum_offset = smallest_sum_index - output_index_0
    diagonal_shifts = None
    if grid_1.index_0 == grid_2.index_0:
        # Same-index sums are (a1 + a2) * r**k in exact arithmetic. Elements a ULP off
        # their knot get their own shift. Off-diagonal terms still use smallest_sum_offset
        # when the target differs from the float-summed anchors.
        output_index_0 = grid_1.index_0
        diagonal_shifts = _diagonal_bin_shifts(
            grid_1=grid_1,
            grid_2=grid_2,
            target_anchor=target_anchor,
            num_diagonal=num_bins,
            bound_type=bound_type,
        )
        if output_anchor is None:
            smallest_sum_offset = 0.0
        else:
            output_origin = target_anchor * math.exp(float(output_index_0) * log_ratio)
            smallest_sum_offset = math.log(smallest_sum / output_origin) / log_ratio

    index_differences = np.arange(num_bins, dtype=np.float64)
    log_ratio_offsets = index_differences * log_ratio
    # Normalized origin weights (w_1 + w_2 = 1) keep the log arguments O(1):
    # (origin_1 + origin_2 * r**d) / smallest_sum = w_1 + w_2 * r**d, so the
    # d = 0 diagonal offset is log(w_1 + w_2) ~ 0 to fp precision regardless of
    # the origins' magnitudes.
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

    # The d = 0 logaddexp of unit weights is a half-ULP off log(1), enough to ceil the
    # whole diagonal. Per-element shifts own that overwrite, so keep it at 0.
    if diagonal_shifts is None:
        low_high_grid_offsets[0] = smallest_sum_offset
        high_low_grid_offsets[0] = smallest_sum_offset
    else:
        low_high_grid_offsets[0] = 0.0
        high_low_grid_offsets[0] = 0.0

    # Size one grid that holds both directional roundings, so upper and lower results
    # stay on comparable supports and no rounded index escapes the kernel's
    # [0, output_size) guard, which would drop that mass silently.
    minimum_offset, num_output_bins = _geometric_output_extent(
        low_high_grid_offsets=low_high_grid_offsets,
        high_low_grid_offsets=high_low_grid_offsets,
        size_1=size_1,
        size_2=size_2,
        index_differences=index_differences,
        diagonal_shifts=diagonal_shifts,
        num_bins=num_bins,
        bound_type=bound_type,
    )
    output_grid = GridSpec(
        step=log_ratio,
        n=num_output_bins,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=target_anchor,
        index_0=output_index_0 + int(minimum_offset),
    )

    # --- C. Kernel Execution ---
    low_high_bin_offsets = _rounded_bin_offsets(
        grid_offsets=low_high_grid_offsets, minimum_offset=minimum_offset, bound_type=bound_type
    )
    high_low_bin_offsets = _rounded_bin_offsets(
        grid_offsets=high_low_grid_offsets, minimum_offset=minimum_offset, bound_type=bound_type
    )
    if diagonal_shifts is None:
        diagonal_bins = np.full(num_bins, low_high_bin_offsets[0], dtype=np.int64)
    else:
        diagonal_bins = low_high_bin_offsets[0] + diagonal_shifts
    pmf_out = _geometric_kernel(
        pmf_base=lower_pmf,
        pmf_scaled=upper_pmf,
        delta_lohi=low_high_bin_offsets,
        delta_hilo=high_low_bin_offsets,
        diagonal_bins=diagonal_bins,
        output_size=num_output_bins,
    )

    return output_grid, pmf_out


def _geometric_output_extent(
    *,
    low_high_grid_offsets: NDArray[np.float64],
    high_low_grid_offsets: NDArray[np.float64],
    size_1: int,
    size_2: int,
    index_differences: NDArray[np.float64],
    diagonal_shifts: NDArray[np.int64] | None,
    num_bins: int,
    bound_type: BoundType,
) -> tuple[int, int]:
    """Return an output range covering every rounded kernel placement.

    Padded input cells are excluded from the reach calculation, while the
    element-specific diagonal shifts are included. This prevents a valid term
    from escaping the kernel's bounds and being silently dropped.
    """
    minimum_offset = math.floor(
        min(float(low_high_grid_offsets.min()), float(high_low_grid_offsets.min()))
    )
    # Highest base index that still pairs with real mass, per orientation: the kernel
    # reads pmf_base[i] * pmf_scaled[i + d] and pmf_base[i + d] * pmf_scaled[i], so the
    # padded tail contributes nothing and must not size the grid.
    difference = index_differences.astype(np.int64)
    highest_index = np.maximum(
        np.minimum(size_1 - 1, size_2 - 1 - difference)
        + np.ceil(low_high_grid_offsets).astype(np.int64),
        np.minimum(size_2 - 1, size_1 - 1 - difference)
        + np.ceil(high_low_grid_offsets).astype(np.int64),
    )
    largest_index = int(np.max(highest_index))
    if diagonal_shifts is not None:
        # Match the kernel: when per-element shifts are active, the shared diagonal
        # grid offset is 0.0, not smallest_sum_offset.
        diagonal_grid_offset = 0.0
        if bound_type == BoundType.DOMINATES:
            diagonal_bin_base = int(math.ceil(diagonal_grid_offset)) - int(minimum_offset)
        else:
            diagonal_bin_base = int(math.floor(diagonal_grid_offset)) - int(minimum_offset)
        diagonal_reach = int(
            np.max(np.arange(num_bins, dtype=np.int64) + diagonal_bin_base + diagonal_shifts)
        )
        largest_index = max(largest_index, diagonal_reach)
        minimum_offset = min(minimum_offset, int(np.min(diagonal_shifts)))
    return int(minimum_offset), largest_index - int(minimum_offset) + 1


def _rounded_bin_offsets(
    *,
    grid_offsets: NDArray[np.float64],
    minimum_offset: int,
    bound_type: BoundType,
) -> NDArray[np.int64]:
    """Round fractional bin offsets in the bound's direction, relative to the grid start."""
    if bound_type == BoundType.DOMINATES:
        rounded = np.ceil(grid_offsets)
    elif bound_type == BoundType.IS_DOMINATED:
        rounded = np.floor(grid_offsets)
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")
    return rounded.astype(np.int64) - minimum_offset


def _diagonal_bin_shifts(
    *,
    grid_1: GridSpec,
    grid_2: GridSpec,
    target_anchor: float,
    num_diagonal: int,
    bound_type: BoundType,
) -> NDArray[np.int64]:
    """Return the per-element bin shift that bounds each same-index sum.

    Zero wherever the summed-anchor knot already bounds the realized sum, and one step in
    the bound's direction for the elements where binary64 puts the sum on the wrong side.
    """
    index = grid_1.index_0 + np.arange(num_diagonal, dtype=np.int64)
    unit = np.exp(index.astype(np.float64) * grid_1.step)
    sums = grid_1.anchor * unit + grid_2.anchor * unit
    knots = target_anchor * unit
    if bound_type == BoundType.DOMINATES:
        return np.where(knots < sums, 1, 0).astype(np.int64)
    return np.where(knots > sums, -1, 0).astype(np.int64)


def _geometric_kernel(
    *,
    pmf_base: NDArray[np.float64],
    pmf_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
    diagonal_bins: NDArray[np.int64],
    output_size: int,
) -> NDArray[np.float64]:
    """Dispatch geometric convolution to numba when available, else NumPy."""
    if has_numba():
        return _numba_geometric_kernel(
            pmf_base=pmf_base,
            pmf_scaled=pmf_scaled,
            delta_lohi=delta_lohi,
            delta_hilo=delta_hilo,
            diagonal_bins=diagonal_bins,
            output_size=output_size,
        )
    return _numpy_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        diagonal_bins=diagonal_bins,
        output_size=output_size,
    )


@optional_njit()
def _numba_geometric_kernel(
    *,
    pmf_base: NDArray[np.float64],
    pmf_scaled: NDArray[np.float64],
    delta_lohi: NDArray[np.int64],
    delta_hilo: NDArray[np.int64],
    diagonal_bins: NDArray[np.int64],
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

    for i in range(n):
        k = i + diagonal_bins[i]
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
    diagonal_bins: NDArray[np.int64],
    output_size: int,
) -> NDArray[np.float64]:
    """Numpy fallback for the geometric convolution kernel.

    Replays the numba kernel's per-bin Kahan updates in the same order, one vectorized
    batch of distinct bins at a time, so both variants return identical arrays. The mass
    repair downstream assumes this compensated accumulation.
    """
    n = pmf_base.size
    pmf_out = np.zeros(output_size, dtype=np.float64)
    comp = np.zeros(output_size, dtype=np.float64)
    base_idx = np.arange(n)
    diagonal_mass = pmf_base * pmf_scaled
    shifts = diagonal_bins[:n]
    # Neighbouring i can share a diagonal bin. numba adds them in increasing i, which is
    # decreasing shift, and one shift value never repeats a bin.
    for shift in np.unique(shifts)[::-1]:
        on_shift = shifts == shift
        _kahan_scatter(
            totals=pmf_out,
            compensations=comp,
            bins=base_idx[on_shift] + shift,
            masses=diagonal_mass[on_shift],
        )
    for d in range(1, n):
        imax = n - d
        batches = [
            (base_idx[:imax] + delta_lohi[d], pmf_base[:imax] * pmf_scaled[d:]),
            (base_idx[:imax] + delta_hilo[d], pmf_base[d:] * pmf_scaled[:imax]),
        ]
        # A bin hit by both orderings receives the smaller i first, as in the numba loop.
        if delta_lohi[d] < delta_hilo[d]:
            batches.reverse()
        for bins, masses in batches:
            _kahan_scatter(totals=pmf_out, compensations=comp, bins=bins, masses=masses)
    return pmf_out


def _kahan_scatter(
    *,
    totals: NDArray[np.float64],
    compensations: NDArray[np.float64],
    bins: NDArray[np.int64],
    masses: NDArray[np.float64],
) -> None:
    """Apply one in-place Kahan update per bin; ``bins`` must not repeat.

    Bins outside ``totals`` are skipped, matching the numba kernel's range guard.
    """
    in_range = (0 <= bins) & (bins < totals.size)
    k = bins[in_range]
    y = masses[in_range] - compensations[k]
    t = totals[k] + y
    compensations[k] = (t - totals[k]) - y
    totals[k] = t


def _add_single_zero_atom_cross_term(
    *,
    pmf_conv: NDArray[np.float64],
    x_arr: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    zero_prob: float,
    output_grid: GridSpec,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], float, float]:
    """Map neutral-zero cross-terms and return their out-of-grid mass by side.

    POSITIVES-domain ``p_min`` is an atom at zero, so ``0 + x`` must retain the
    finite atom ``x``. Directional rounding decides its bin; the caller assigns
    underflow and overflow to the conservative finite or boundary location.
    """
    if np.any(x_arr <= 0.0):
        raise ValueError("0+finite cross-term support values must be strictly positive")
    if zero_prob == 0.0:
        return pmf_conv, 0.0, 0.0

    # Each cross-term has probability zero_prob times its finite-atom mass.
    masses = prob_arr * zero_prob
    log_ratio = output_grid.step
    frac_k = np.log(x_arr / output_grid.x_0) / log_ratio
    if bound_type == BoundType.DOMINATES:
        k = np.ceil(frac_k).astype(np.int64)
    elif bound_type == BoundType.IS_DOMINATED:
        k = np.floor(frac_k).astype(np.int64)
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    below = k < 0
    above = k >= pmf_conv.size
    in_range = ~(below | above)
    # Accumulate representable cross-terms into their directionally rounded bins.
    np.add.at(pmf_conv, k[in_range], masses[in_range])
    omitted_below = math.fsum(map(float, masses[below]))
    omitted_above = math.fsum(map(float, masses[above]))
    return pmf_conv, omitted_below, omitted_above
