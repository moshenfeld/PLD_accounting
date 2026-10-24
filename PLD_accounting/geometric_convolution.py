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
from PLD_accounting.distribution_utils import enforce_mass_conservation, exact_sum
from PLD_accounting.projections import project_grid_offsets, truncate_tails
from PLD_accounting.projections_ctd import project_grid_offsets_ctd
from PLD_accounting.projections_stoch_dom import project_grid_offsets_stoch_dom
from PLD_accounting.types import (
    BoundType,
    ProjectionMethod,
    SpacingType,
    has_numba,
    optional_njit,
    require_projection_method,
)
from PLD_accounting.utils import convolve_boundary_masses
from PLD_accounting.validation import require_nonnegative_real

# =============================================================================
# PUBLIC API
# =============================================================================


def geometric_convolve(
    *,
    dist_1: DenseDiscreteDist,
    dist_2: DenseDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod = ProjectionMethod.STOCH_DOM,
) -> DenseDiscreteDist:
    """Convolve two geometric-grid distributions.

    Algorithm 4 (`conv`) in Appendix C of https://arxiv.org/abs/2602.17284.
    In the POSITIVES domain the lower-boundary atom is at zero, which is neutral
    under addition rather than absorbing; its cross-terms with finite atoms are
    therefore included explicitly.

    The output lies on the lattice of the summed anchors. Rounding is directional
    for the bound type, so validity never requires the inputs to sit on any
    particular lattice; input alignment only sharpens exact-hit sums.

    ``projection_method`` is the projection of a sum that falls between two grid
    points: ``STOCH_DOM`` rounds it to the neighbour the bound requires, ``CTD``
    splits it between both so that its mean is kept and requires
    ``BoundType.DOMINATES``.
    """
    require_geometric_positives_dist(dist=dist_1, name="dist_1")
    require_geometric_positives_dist(dist=dist_2, name="dist_2")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_projection_method(value=projection_method, bound_type=bound_type)
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
        projection_method=projection_method,
    )

    # Boundary-by-boundary products are accounted for independently of the kernel.
    p_min, p_max = convolve_boundary_masses(
        p_min_1=dist_1.p_min,
        p_max_1=dist_1.p_max,
        p_min_2=dist_2.p_min,
        p_max_2=dist_2.p_max,
        domain=dist_1.domain,
    )

    # Then add both zero-by-finite cross-terms, retaining the mass left outside the grid.
    omitted_below = 0.0
    omitted_above = 0.0
    for finite, zero_prob in ((dist_2, dist_1.p_min), (dist_1, dist_2.p_min)):
        output_grid, pmf_conv, below, above = _add_zero_atom_cross_term(
            pmf_conv=pmf_conv,
            x_arr=finite.x_array,
            prob_arr=finite.prob_arr,
            zero_prob=zero_prob,
            output_grid=output_grid,
            bound_type=bound_type,
            projection_method=projection_method,
        )
        omitted_below += below
        omitted_above += above
    if bound_type == BoundType.DOMINATES:
        # Round lower underflow up to the first finite cell; upper overflow
        # becomes upper-boundary mass.
        pmf_conv[0] += omitted_below
        p_max += omitted_above
    else:
        # Lower underflow becomes lower-boundary mass; round upper overflow
        # down to the last finite cell.
        p_min += omitted_below
        pmf_conv[-1] += omitted_above

    # Repair only numerical drift after all omitted mass is explicit. A connect-the-dots
    # sum keeps its bound when mass moves to a smaller value, the direction of a
    # dominated rounding.
    connect_the_dots = projection_method == ProjectionMethod.CTD
    pmf_conv, p_min, p_max = enforce_mass_conservation(
        prob_arr=pmf_conv,
        expected_p_min=p_min,
        expected_p_max=p_max,
        bound_type=BoundType.IS_DOMINATED if connect_the_dots else bound_type,
    )
    # The first moment over the anchor is the share of the mass under P: on
    # unit-anchored factors the anchor of a sum is the number of copies it adds.
    pmf_conv, p_min, p_max, first_kept = truncate_tails(
        values=output_grid.materialize(),
        prob_arr=pmf_conv,
        p_min=p_min,
        p_max=p_max,
        moment_scale=output_grid.anchor,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        projection_method=projection_method,
    )

    return DenseDiscreteDist(
        grid=output_grid.slice(start=first_kept, n=pmf_conv.size),
        prob_arr=pmf_conv,
        p_min=p_min,
        p_max=p_max,
        domain=Domain.POSITIVES,
    )


# =============================================================================
# Convolution Helpers
# =============================================================================


def _compute_geometric_convolution(
    *,
    grid_1: GridSpec,
    pmf_1: NDArray[np.float64],
    grid_2: GridSpec,
    pmf_2: NDArray[np.float64],
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> tuple[GridSpec, NDArray[np.float64]]:
    """Align grids, compute bin mapping parameters, and invoke the Numba kernel.

    Algorithm 4 (`conv`), with internal
    Algorithm 5 (`range-renorm`),
    in Appendix C of https://arxiv.org/abs/2602.17284.

    The two supports stay structural: their anchors and integer indices define
    the target lattice, while log-space offsets determine only the placement of
    pairwise sums: one bin under rounding, a bin and its upper neighbour under
    connect-the-dots. Returns that output ``GridSpec`` and the finite PMF on it.
    """
    # Keep the lower-origin distribution first to match the kernel's two orientations.
    if grid_1.x_0 > grid_2.x_0:
        grid_1, pmf_1, grid_2, pmf_2 = grid_2, pmf_2, grid_1, pmf_1
    log_ratio = grid_1.step
    num_bins = max(pmf_1.size, pmf_2.size)

    # The output lattice is the summed anchors, counted from the index at or below the
    # smallest pairwise sum.
    target_anchor = grid_1.anchor + grid_2.anchor
    if grid_1.index_0 == grid_2.index_0:
        # Same-index sums sit on the knots of the summed anchors.
        output_index_0, smallest_sum_offset = grid_1.index_0, 0.0
    else:
        smallest_sum_index = math.log((grid_1.x_0 + grid_2.x_0) / target_anchor) / log_ratio
        output_index_0 = int(math.floor(smallest_sum_index))
        smallest_sum_offset = smallest_sum_index - output_index_0
    diagonal_shifts = None
    diagonal_offset = smallest_sum_offset
    if grid_1.index_0 == grid_2.index_0 and projection_method == ProjectionMethod.STOCH_DOM:
        # Same-index sums are (a1 + a2) * r**k in exact arithmetic. Elements a ULP off
        # their knot get their own shift, which replaces the shared diagonal offset. A
        # split diagonal needs no shift: an ULP of offset moves an ULP of its mass.
        diagonal_shifts = _diagonal_bin_shifts(
            grid_1=grid_1,
            grid_2=grid_2,
            target_anchor=target_anchor,
            num_diagonal=num_bins,
            bound_type=bound_type,
        )
        diagonal_offset = 0.0

    # Positions of the pairwise sums on that lattice, one row per orientation.
    grid_offsets = _pairwise_sum_grid_offsets(
        origin_1=grid_1.x_0,
        origin_2=grid_2.x_0,
        log_ratio=log_ratio,
        num_bins=num_bins,
        smallest_sum_offset=smallest_sum_offset,
        diagonal_offset=diagonal_offset,
    )

    # Size one grid that holds both directional roundings, so upper and lower results
    # stay on comparable supports and no rounded index escapes the kernel's
    # [0, output_size) guard, which would drop that mass silently. A split writes the
    # floor bin and, for a positive weight, the ceiling bin, so the same grid holds it.
    minimum_offset, num_output_bins = _geometric_output_extent(
        grid_offsets=grid_offsets,
        size_1=pmf_1.size,
        size_2=pmf_2.size,
        diagonal_shifts=diagonal_shifts,
        bound_type=bound_type,
    )
    output_grid = GridSpec(
        step=log_ratio,
        n=num_output_bins,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=target_anchor,
        index_0=output_index_0 + minimum_offset,
    )

    # Projection of each position onto the bins of that grid.
    bin_offsets, weights = _bin_offsets_and_weights(
        grid_offsets=grid_offsets,
        minimum_offset=minimum_offset,
        log_ratio=log_ratio,
        bound_type=bound_type,
        projection_method=projection_method,
    )
    if diagonal_shifts is None:
        diagonal_bins = np.full(num_bins, bin_offsets[0, 0], dtype=np.int64)
    else:
        diagonal_bins = bin_offsets[0, 0] + diagonal_shifts

    # The kernel requires equal-length float64 PMFs; missing high-grid bins have zero mass.
    pmf_base = _pad_pmf(pmf=pmf_1, num_bins=num_bins)
    pmf_scaled = _pad_pmf(pmf=pmf_2, num_bins=num_bins)
    pmf_out = _geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        bin_offsets=bin_offsets,
        weights=weights,
        diagonal_bins=diagonal_bins,
        output_size=num_output_bins,
    )

    return output_grid, pmf_out


def _add_zero_atom_cross_term(
    *,
    pmf_conv: NDArray[np.float64],
    x_arr: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    zero_prob: float,
    output_grid: GridSpec,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> tuple[GridSpec, NDArray[np.float64], float, float]:
    """Add the neutral-zero cross-terms of one input under the requested projection.

    Returns the grid, the PMF on it, and the mass left below and above the grid.
    Rounding keeps the grid and returns the mass outside it; connect-the-dots
    extends the grid and leaves none.
    """
    if projection_method == ProjectionMethod.CTD:
        output_grid, pmf_conv = _add_zero_atom_cross_term_ctd(
            pmf_conv=pmf_conv,
            x_arr=x_arr,
            prob_arr=prob_arr,
            zero_prob=zero_prob,
            output_grid=output_grid,
        )
        return output_grid, pmf_conv, 0.0, 0.0
    pmf_conv, omitted_below, omitted_above = _add_zero_atom_cross_term_stoch_dom(
        pmf_conv=pmf_conv,
        x_arr=x_arr,
        prob_arr=prob_arr,
        zero_prob=zero_prob,
        output_grid=output_grid,
        bound_type=bound_type,
    )
    return output_grid, pmf_conv, omitted_below, omitted_above


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


def _pairwise_sum_grid_offsets(
    *,
    origin_1: float,
    origin_2: float,
    log_ratio: float,
    num_bins: int,
    smallest_sum_offset: float,
    diagonal_offset: float,
) -> NDArray[np.float64]:
    """Return where the pairwise sums fall, in output-grid bins, one row per orientation.

    Column ``d`` holds the pairs whose indices differ by ``d``, measured from the
    lower index of the pair: row 0 has the higher index on the second grid and row 1
    on the first. ``diagonal_offset`` is the value of the ``d = 0`` column.
    """
    smallest_sum = origin_1 + origin_2
    log_ratio_offsets = np.arange(num_bins, dtype=np.float64) * log_ratio
    # Normalized origin weights (w_1 + w_2 = 1) keep the log arguments O(1):
    # (origin_1 + origin_2 * r**d) / smallest_sum = w_1 + w_2 * r**d, so the
    # d = 0 diagonal offset is log(w_1 + w_2) ~ 0 to fp precision regardless of
    # the origins' magnitudes.
    log_weight_1 = np.log(origin_1 / smallest_sum)
    log_weight_2 = np.log(origin_2 / smallest_sum)
    grid_offsets = np.stack(
        (
            np.logaddexp(log_weight_1, log_weight_2 + log_ratio_offsets) / log_ratio
            + smallest_sum_offset,
            np.logaddexp(log_weight_2, log_weight_1 + log_ratio_offsets) / log_ratio
            + smallest_sum_offset,
        )
    )
    # The d = 0 logaddexp of unit weights is a half-ULP off log(1), enough to ceil the
    # whole diagonal, so that column is set and not computed.
    grid_offsets[:, 0] = diagonal_offset
    return grid_offsets


def _geometric_output_extent(
    *,
    grid_offsets: NDArray[np.float64],
    size_1: int,
    size_2: int,
    diagonal_shifts: NDArray[np.int64] | None,
    bound_type: BoundType,
) -> tuple[int, int]:
    """Return an output range covering every rounded kernel placement.

    Padded input cells are excluded from the reach calculation, while the
    element-specific diagonal shifts are included. This prevents a valid term
    from escaping the kernel's bounds and being silently dropped.
    """
    low_high_grid_offsets, high_low_grid_offsets = grid_offsets
    num_bins = low_high_grid_offsets.size
    minimum_offset = math.floor(float(grid_offsets.min()))
    # Highest base index that still pairs with real mass, per orientation: the kernel
    # reads pmf_base[i] * pmf_scaled[i + d] and pmf_base[i + d] * pmf_scaled[i], so the
    # padded tail contributes nothing and must not size the grid.
    difference = np.arange(num_bins, dtype=np.int64)
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


def _bin_offsets_and_weights(
    *,
    grid_offsets: NDArray[np.float64],
    minimum_offset: int,
    log_ratio: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return each offset's output bin and the share of mass sent to the next bin.

    Every pair goes to a lower bin and sends a share of its mass to the next one;
    rounding is the share zero, with the bin chosen by the bound.
    """
    bins, shares = project_grid_offsets(
        grid_offsets=grid_offsets,
        log_step=log_ratio,
        bound_type=bound_type,
        projection_method=projection_method,
    )
    return bins - minimum_offset, shares


def _pad_pmf(*, pmf: NDArray[np.float64], num_bins: int) -> NDArray[np.float64]:
    """Return ``pmf`` as float64, extended with zero mass to ``num_bins`` entries."""
    padded = np.pad(pmf, (0, num_bins - pmf.size), mode="constant")
    return padded.astype(np.float64, copy=False)


def _geometric_kernel(
    *,
    pmf_base: NDArray[np.float64],
    pmf_scaled: NDArray[np.float64],
    bin_offsets: NDArray[np.int64],
    weights: NDArray[np.float64],
    diagonal_bins: NDArray[np.int64],
    output_size: int,
) -> NDArray[np.float64]:
    """Dispatch geometric convolution to numba when available, else NumPy."""
    kernel = _numba_geometric_kernel if has_numba() else _numpy_geometric_kernel
    return kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        bin_offsets=bin_offsets,
        weights=weights,
        diagonal_bins=diagonal_bins,
        output_size=output_size,
    )


def _add_zero_atom_cross_term_ctd(
    *,
    pmf_conv: NDArray[np.float64],
    x_arr: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    zero_prob: float,
    output_grid: GridSpec,
) -> tuple[GridSpec, NDArray[np.float64]]:
    """Split neutral-zero cross-terms between their two neighbouring grid points.

    The lattice is extended to the points a term outside the bins of the finite
    pairs needs. Returns the grid and the PMF on it.
    """
    if np.any(x_arr <= 0.0):
        raise ValueError("0+finite cross-term support values must be strictly positive")
    masses = prob_arr * zero_prob
    live = masses > 0.0
    if not np.any(live):
        return output_grid, pmf_conv

    lower_bins, weights = project_grid_offsets_ctd(
        grid_offsets=np.log(x_arr[live] / output_grid.x_0) / output_grid.step,
        log_step=output_grid.step,
    )
    upper_masses = masses[live] * weights
    split = upper_masses > 0.0
    bins = np.concatenate((lower_bins, lower_bins[split] + 1))
    shares = np.concatenate((masses[live] - upper_masses, upper_masses[split]))

    pad_left = max(0, -int(bins.min()))
    pad_right = max(0, int(bins.max()) - (pmf_conv.size - 1))
    padded = np.concatenate((np.zeros(pad_left), pmf_conv, np.zeros(pad_right)))
    # Offsets a rounding error apart from an integer can put two terms on one bin, so the
    # accumulation must not assume distinct bins.
    np.add.at(padded, bins + pad_left, shares)
    return output_grid.pad(left=pad_left, right=pad_right), padded


def _add_zero_atom_cross_term_stoch_dom(
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

    masses = prob_arr * zero_prob
    log_ratio = output_grid.step
    frac_k = np.log(x_arr / output_grid.x_0) / log_ratio
    k = project_grid_offsets_stoch_dom(grid_offsets=frac_k, bound_type=bound_type)

    below = k < 0
    above = k >= pmf_conv.size
    in_range = ~(below | above)
    np.add.at(pmf_conv, k[in_range], masses[in_range])
    omitted_below = exact_sum(masses[below])
    omitted_above = exact_sum(masses[above])
    return pmf_conv, omitted_below, omitted_above


# =============================================================================
# Compensated Convolution Kernels
# =============================================================================


@optional_njit()
def _numba_geometric_kernel(
    *,
    pmf_base: NDArray[np.float64],
    pmf_scaled: NDArray[np.float64],
    bin_offsets: NDArray[np.int64],
    weights: NDArray[np.float64],
    diagonal_bins: NDArray[np.int64],
    output_size: int,
) -> NDArray[np.float64]:
    """Core convolution loop with compensated summation.

    Row 0 of ``bin_offsets`` and ``weights`` is the orientation ``(i, i + d)`` and
    row 1 the orientation ``(i + d, i)``. A pair goes to bin ``i + bin_offsets[row, d]``
    and sends the share ``weights[row, d]`` of its mass to the next bin; a zero weight
    is the rounding placement. Unsplit diagonal elements go to their own
    ``diagonal_bins`` entry.
    """
    n = pmf_base.size
    pmf_out = np.zeros(output_size, dtype=np.float64)
    comp = np.zeros(output_size, dtype=np.float64)

    if weights[0, 0] == 0.0:
        for i in range(n):
            _numba_kahan_add(pmf_out, comp, i + diagonal_bins[i], pmf_base[i] * pmf_scaled[i])
    else:
        _numba_scatter_run(pmf_out, comp, pmf_base, pmf_scaled, bin_offsets[0, 0], weights[0, 0])

    for d in range(1, n):
        imax = n - d
        if weights[0, d] == 0.0 and weights[1, d] == 0.0:
            kshift1 = bin_offsets[0, d]
            kshift2 = bin_offsets[1, d]
            for i in range(imax):
                _numba_kahan_add(pmf_out, comp, i + kshift1, pmf_base[i] * pmf_scaled[i + d])
                _numba_kahan_add(pmf_out, comp, i + kshift2, pmf_base[i + d] * pmf_scaled[i])
        else:
            _numba_scatter_run(
                pmf_out, comp, pmf_base[:imax], pmf_scaled[d:], bin_offsets[0, d], weights[0, d]
            )
            _numba_scatter_run(
                pmf_out, comp, pmf_base[d:], pmf_scaled[:imax], bin_offsets[1, d], weights[1, d]
            )

    return pmf_out


def _numpy_geometric_kernel(
    *,
    pmf_base: NDArray[np.float64],
    pmf_scaled: NDArray[np.float64],
    bin_offsets: NDArray[np.int64],
    weights: NDArray[np.float64],
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
    if weights[0, 0] == 0.0:
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
    else:
        _numpy_scatter_run(
            totals=pmf_out,
            compensations=comp,
            masses=diagonal_mass,
            shift=int(bin_offsets[0, 0]),
            weight=float(weights[0, 0]),
        )
    for d in range(1, n):
        imax = n - d
        runs = [
            (pmf_base[:imax] * pmf_scaled[d:], bin_offsets[0, d], weights[0, d]),
            (pmf_base[d:] * pmf_scaled[:imax], bin_offsets[1, d], weights[1, d]),
        ]
        # The numba loop interleaves two unsplit orderings, so a bin hit by both receives
        # the smaller i first.
        if weights[0, d] == 0.0 and weights[1, d] == 0.0 and bin_offsets[0, d] < bin_offsets[1, d]:
            runs.reverse()
        for masses, shift, weight in runs:
            _numpy_scatter_run(
                totals=pmf_out,
                compensations=comp,
                masses=masses,
                shift=int(shift),
                weight=float(weight),
            )
    return pmf_out


@optional_njit()
def _numba_kahan_add(
    totals: NDArray[np.float64],
    compensations: NDArray[np.float64],
    k: int,
    value: float,
) -> None:
    """Apply one Kahan update to bin ``k``; a bin outside ``totals`` is skipped."""
    if 0 <= k < totals.size:
        y = value - compensations[k]
        t = totals[k] + y
        compensations[k] = (t - totals[k]) - y
        totals[k] = t


@optional_njit()
def _numba_scatter_run(
    totals: NDArray[np.float64],
    compensations: NDArray[np.float64],
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    shift: int,
    weight: float,
) -> None:
    """Place the pairs ``first[i] * second[i]`` of one run, one Kahan update per bin.

    Pair ``i`` keeps the share ``1 - weight`` at bin ``i + shift`` and sends the rest
    to the next bin, which receives it together with the kept share of pair ``i + 1``.
    """
    count = first.size
    carried = 0.0
    for i in range(count):
        mass = first[i] * second[i]
        upper = weight * mass
        _numba_kahan_add(totals, compensations, i + shift, mass - upper + carried)
        carried = upper
    if weight != 0.0:
        _numba_kahan_add(totals, compensations, count + shift, carried)


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


def _numpy_scatter_run(
    *,
    totals: NDArray[np.float64],
    compensations: NDArray[np.float64],
    masses: NDArray[np.float64],
    shift: int,
    weight: float,
) -> None:
    """Place one run of pair masses with the updates of ``_numba_scatter_run``."""
    bins = np.arange(masses.size, dtype=np.int64) + shift
    if weight == 0.0:
        _kahan_scatter(totals=totals, compensations=compensations, bins=bins, masses=masses)
        return
    upper = weight * masses
    updates = masses - upper
    updates[1:] += upper[:-1]
    _kahan_scatter(totals=totals, compensations=compensations, bins=bins, masses=updates)
    _kahan_scatter(
        totals=totals,
        compensations=compensations,
        bins=np.array([shift + masses.size], dtype=np.int64),
        masses=upper[-1:],
    )
