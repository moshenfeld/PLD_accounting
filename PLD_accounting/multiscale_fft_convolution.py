"""Multiscale FFT convolution of measures on the dyadic grid.

The grid is ``{0}`` with the points ``2**j (1 + i/m)`` for ``i = 0..m-1``: ``m`` uniform
points per octave, so two neighbouring points have ratio at most ``1 + 1/m``. A merge is
the pairwise sum of two measures projected back onto the grid. Pairs are grouped by the
octave of their larger summand; every point at or above ``2**j`` is a multiple of that
octave's spacing, so each group is a convolution on a uniform lattice and one merge costs
``O(J m log m)`` for ``J`` octaves.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.fft import irfft, next_fast_len, rfft

from PLD_accounting.discrete_dist import DyadicDist, Domain, SparseDiscreteDist
from PLD_accounting.distribution_utils import (
    classify_residual,
    exact_sum,
    transform_mass_tolerances,
)
from PLD_accounting.projections import halve_lattice
from PLD_accounting.types import BoundType, ProjectionMethod, require_projection_method
from PLD_accounting.utils import binary_self_convolve, binary_self_convolve_depth
from PLD_accounting.validation import (
    require_nonnegative_int,
    require_nonnegative_real,
    require_positive_int,
    require_positive_real,
)

# Octaves on each side of one that the grid of a one-step input may span; the
# logarithms of its points stay where exp is representable.
MAX_DYADIC_REACH = 1000
# Bytes the transforms of one batch of octaves may hold: the zero-padded real arrays
# and their complex spectra, about 64 bytes per point of a transform.
_CHUNK_BUDGET_BYTES = 128_000_000
_BYTES_PER_TRANSFORM_POINT = 64

# =============================================================================
# Public API
# =============================================================================


def multiscale_convolve(
    *,
    dist_1: DyadicDist,
    dist_2: DyadicDist,
    tail_truncation: float,
    projection_method: ProjectionMethod,
    bound_type: BoundType,
) -> DyadicDist:
    """Return the sum of two dyadic-grid measures, projected back onto the grid.

    In exact arithmetic the result is the pairwise convolution followed by the
    projection: ``CTD`` splits a sum between its two neighbouring grid points with the
    mean kept and requires ``BoundType.DOMINATES``; ``STOCH_DOM`` moves it to the point
    above for ``DOMINATES`` and below for ``IS_DOMINATED``. Both operands must have the
    same ``cells_per_band``. Passing one object as both takes the self-product path,
    which needs half the transforms.

    ``tail_truncation`` is accepted for the shared merge interface and does not
    trim this grid; callers pass zero.

    The result is checked against the two operands: its mass on grid points that no
    sum of their values reaches, which is set to zero, its total mass and its first
    moment. A residual of the floating-point computation above the silent threshold
    warns and one above the ceiling raises. The checks admit or reject a
    computation; they do not certify a bound.
    """
    require_projection_method(value=projection_method, bound_type=bound_type)
    merged = _merge(
        dist_1=dist_1,
        dist_2=dist_2,
        tail_truncation=tail_truncation,
        projection_method=projection_method,
        bound_type=bound_type,
    )
    return _admit_sum(
        summed=merged,
        leaves=[(dist_1, 1), (dist_2, 1)],
        projection_method=projection_method,
        bound_type=bound_type,
    )


def multiscale_self_convolve(
    *,
    dist: DyadicDist,
    num_convolutions: int,
    tail_truncation: float,
    projection_method: ProjectionMethod,
    bound_type: BoundType,
) -> DyadicDist:
    """Return the sum of ``num_convolutions`` copies, projected after every merge.

    Repeated squaring builds powers of two and accumulates the powers selected by
    the binary expansion, from smallest to largest. The maximum depth is
    ``ceil(log2 n)``. No merge trims; each adds at most one octave at the
    top. The sum is checked once against its copies,
    as ``multiscale_convolve`` checks a merge against its operands.
    ``tail_truncation`` has the shared merge convention and callers pass zero.
    """
    require_projection_method(value=projection_method, bound_type=bound_type)
    require_positive_int(value=num_convolutions, name="num_convolutions")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    if num_convolutions == 1:
        return dist
    summed = binary_self_convolve(
        dist=dist,
        num_convolutions=num_convolutions,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=partial(_merge, projection_method=projection_method),
    )
    return _admit_sum(
        summed=summed,
        leaves=[(dist, num_convolutions)],
        projection_method=projection_method,
        bound_type=bound_type,
    )


# =============================================================================
# The Grid of a One-Step Input
# =============================================================================


def dyadic_cells_per_band(*, loss_step: float, max_cells_per_band: int) -> int:
    """Return the points per octave ``m`` of the grid that meets a spacing in the loss.

    The logarithms of an octave's points have the mean spacing ``log(2) / m``. ``m`` is
    the smallest integer with ``log(2) / m <= loss_step``, so the grid has as many
    points per octave as a linear loss grid of step ``loss_step``. The spacing runs from
    ``log(1 + 1/m)``, up to ``loss_step / log(2)``, at the bottom of an octave to half
    of that at its top. ``max_cells_per_band`` is used when it is smaller; a cap that
    leaves the mean spacing above ``loss_step`` warns with the spacing reached.
    """
    require_positive_real(value=loss_step, name="loss_step")
    require_positive_int(value=max_cells_per_band, name="max_cells_per_band")
    cells = min(math.ceil(math.log(2.0) / loss_step), max_cells_per_band)
    achieved = math.log(2.0) / cells
    if achieved > loss_step:
        warnings.warn(
            f"multiscale allocation: max_cells_per_band={max_cells_per_band} limits the dyadic "
            f"grid to a mean loss spacing of {achieved:.6e}, above the requested {loss_step:.6e}. "
            "The bound stays valid and is looser.",
            stacklevel=4,
        )
    return cells


def dyadic_range_points(*, reach: int, cells_per_band: int) -> NDArray[np.float64]:
    """Return the dyadic grid points from ``2**-reach`` to ``2**reach``, both included."""
    require_positive_int(value=reach, name="reach")
    if reach >= MAX_DYADIC_REACH:
        raise ValueError(f"reach must be below {MAX_DYADIC_REACH}, got {reach}")
    require_positive_int(value=cells_per_band, name="cells_per_band")
    octaves = DyadicDist(first_band=-reach, prob_arr=np.zeros((2 * reach + 1, cells_per_band)))
    return octaves.support()[: 2 * reach * cells_per_band + 1]


def dyadic_dist_on_range(loss_dist: SparseDiscreteDist) -> DyadicDist:
    """Return the law of ``exp(L)`` for a loss ``L`` on the dyadic loss points.

    The support of ``loss_dist`` must be the logarithms of ``dyadic_range_points``. Its
    masses stay on their points, its mass at ``-inf`` is the atom at zero and its mass
    at ``+inf`` the absorbing mass at infinity.
    """
    if loss_dist.domain != Domain.REALS:
        raise ValueError(f"loss_dist must be over the reals, got {loss_dist.domain}")
    loss = loss_dist.x_array
    # A loss factor on explicit points carries no lattice, so the reach and the points
    # per octave are read from its coordinates and accepted only if the dyadic points
    # they describe reproduce those coordinates bitwise.
    reach = max(1, round(-float(loss[0]) / math.log(2.0)))
    size = max(1, (loss.size - 1) // (2 * reach))
    range_loss = np.log(dyadic_range_points(reach=reach, cells_per_band=size))
    if not np.array_equal(loss, range_loss):
        raise ValueError(
            f"loss_dist, with {loss.size} points from {loss[0]:.6g}, is not on the "
            "logarithms of the points of a dyadic range"
        )
    prob_arr = loss_dist.prob_arr
    padded = np.zeros((2 * reach + 1) * size, dtype=np.float64)
    padded[: prob_arr.size] = prob_arr
    return DyadicDist(
        first_band=-reach,
        prob_arr=padded.reshape(-1, size),
        p_min=float(loss_dist.p_min),
        p_max=float(loss_dist.p_max),
    )


# =============================================================================
# Merge and Admission
# =============================================================================


def _merge(
    *,
    dist_1: DyadicDist,
    dist_2: DyadicDist,
    tail_truncation: float,
    projection_method: ProjectionMethod,
    bound_type: BoundType,
) -> DyadicDist:
    """Return the projected sum of two measures; the shared tail budget is unused."""
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    size = dist_1.cells_per_band
    if dist_2.cells_per_band != size:
        raise ValueError(
            f"operands must share cells_per_band, got {size} and {dist_2.cells_per_band}"
        )
    first_band = min(dist_1.first_band, dist_2.first_band)
    num_bands = (
        max(dist_1.first_band + dist_1.num_bands, dist_2.first_band + dist_2.num_bands) - first_band
    )
    self_product = dist_1 is dist_2
    blocks_1 = _aligned_band_blocks(dist=dist_1, first_band=first_band, num_bands=num_bands)
    blocks_2 = (
        blocks_1
        if self_product
        else _aligned_band_blocks(dist=dist_2, first_band=first_band, num_bands=num_bands)
    )
    result = _merge_positive_pairs(
        blocks_1=blocks_1,
        blocks_2=blocks_2,
        self_product=self_product,
        band_stride=max(1, _CHUNK_BUDGET_BYTES // (_BYTES_PER_TRANSFORM_POINT * 4 * size)),
        projection_method=projection_method,
        bound_type=bound_type,
    )
    # The atom at zero is neutral under addition, so its cross-terms are the other
    # operand's own octaves, already on the grid.
    if dist_1.p_min > 0.0:
        result[:num_bands] += dist_1.p_min * blocks_2
    if dist_2.p_min > 0.0:
        result[:num_bands] += dist_2.p_min * blocks_1
    # Infinity absorbs: a pair is there when either summand is. The operands are
    # measures, so each atom is weighted by the other operand's own mass.
    p_max = 0.0
    if dist_1.p_max > 0.0 or dist_2.p_max > 0.0:
        finite_1 = dist_1.finite_mass()
        finite_2 = finite_1 if self_product else dist_2.finite_mass()
        p_max = math.fsum(
            (dist_1.p_max * finite_2, dist_2.p_max * finite_1, dist_1.p_max * dist_2.p_max)
        )
    return DyadicDist(
        first_band=first_band,
        prob_arr=result,
        p_min=float(dist_1.p_min) * float(dist_2.p_min),
        p_max=p_max,
    )


def _admit_sum(
    *,
    summed: DyadicDist,
    leaves: Sequence[tuple[DyadicDist, int]],
    projection_method: ProjectionMethod,
    bound_type: BoundType,
) -> DyadicDist:
    """Check a merge tree's result against its leaves and clear unreachable grid points.

    ``leaves`` lists each distinct leaf with its number of copies. The copies of a
    leaf are summed by ``multiscale_self_convolve`` and the sums are merged one after
    the other, which gives ``tree_depth``, the largest number of merges above a leaf.
    Three residuals of the
    floating-point computation are judged against ``_sum_mass_tolerances``: the mass
    on grid points that no sum of leaf values can reach after ``tree_depth``
    projections, which is set to zero; the total mass against the product of the leaf
    masses; and the first moment of the finite part against its exact value, which
    connect-the-dots keeps and rounding moves to one side by at most
    ``(1 + 1/m) ** tree_depth``. A residual above the silent threshold warns and one
    above the ceiling raises. The checks admit or reject a computation; they do not
    certify a bound.
    """
    require_projection_method(value=projection_method, bound_type=bound_type)
    if len(leaves) == 0:
        raise ValueError("leaves must name at least one leaf")
    for _, count in leaves:
        require_positive_int(value=count, name="leaf count")
    tree_depth = max(binary_self_convolve_depth(count) for _, count in leaves) + len(leaves) - 1
    num_leaves = sum(count for _, count in leaves)
    drift_tol, repair_tol = _sum_mass_tolerances(
        fft_length=next_fast_len(4 * summed.cells_per_band), charged_merges=num_leaves - 1
    )
    projection_factor = (1.0 + 1.0 / summed.cells_per_band) ** tree_depth

    # Reachable values: every leaf value lies in [x_low, x_high], and a leaf with an
    # atom at zero can drop out of a sum. A leaf without a positive point gives none.
    spans = [_positive_span(leaf) for leaf, _ in leaves]
    if all(span is not None for span in spans):
        lowest = min(span[0] for span in spans if span is not None)
        highest = max(span[1] for span in spans if span is not None)
        if not any(leaf.p_min > 0.0 for leaf, _ in leaves):
            lowest *= num_leaves
        points = summed.support().reshape(summed.prob_arr.shape)
        unreachable = (points < lowest / projection_factor) | (
            points > num_leaves * highest * projection_factor
        )
        cleared = exact_sum(summed.prob_arr[unreachable])
        classify_residual(
            residual=cleared,
            drift_tol=drift_tol,
            repair_tol=repair_tol,
            context="multiscale sum: mass on unreachable grid points",
            repair="clearing it",
        )
        if cleared > 0.0:
            summed = DyadicDist(
                first_band=summed.first_band,
                prob_arr=np.where(unreachable, 0.0, summed.prob_arr),
                p_min=summed.p_min,
                p_max=summed.p_max,
            )

    exact_mass = math.prod(leaf.total_mass() ** count for leaf, count in leaves)
    classify_residual(
        residual=abs(summed.total_mass() - exact_mass),
        drift_tol=drift_tol,
        repair_tol=repair_tol,
        context="multiscale sum: total mass",
        repair="leaving it to the rescaling of the sum to unit mass",
    )

    exact_moment = _exact_first_moment(leaves)
    if projection_method == ProjectionMethod.CTD:
        moment_low, moment_high = 0.0, 0.0
    elif bound_type == BoundType.DOMINATES:
        moment_low, moment_high = 0.0, exact_moment * (projection_factor - 1.0)
    else:
        moment_low, moment_high = exact_moment * (1.0 / projection_factor - 1.0), 0.0
    moment_residual = summed.first_moment() - exact_moment
    # A moment residual is judged by a moment tolerance: the mass tolerance times the
    # size of the moment.
    moment_scale = max(1.0, exact_moment)
    moment_drift_tol, moment_repair_tol = drift_tol * moment_scale, repair_tol * moment_scale
    classify_residual(
        residual=max(0.0, moment_low - moment_residual, moment_residual - moment_high),
        drift_tol=moment_drift_tol,
        repair_tol=moment_repair_tol,
        context="multiscale sum: first moment",
        repair="accepting the computed first moment",
    )
    return summed


def _aligned_band_blocks(
    *, dist: DyadicDist, first_band: int, num_bands: int
) -> NDArray[np.float64]:
    """Copy a measure's octaves into ``num_bands`` rows that start at ``first_band``."""
    offset = dist.first_band - first_band
    stop = offset + dist.num_bands
    blocks = np.zeros((num_bands, dist.cells_per_band), dtype=np.float64)
    blocks[offset:stop] = dist.prob_arr
    return blocks


def _merge_positive_pairs(
    *,
    blocks_1: NDArray[np.float64],
    blocks_2: NDArray[np.float64],
    self_product: bool,
    band_stride: int,
    projection_method: ProjectionMethod,
    bound_type: BoundType,
) -> NDArray[np.float64]:
    """Return the projected sums of all pairs of positive points, one octave per row.

    The octaves are transformed ``band_stride`` at a time; the result does not depend
    on the stride. The output has one row more than the inputs: a pair whose larger
    summand is in the last octave can reach the next one and no further.
    """
    num_bands, size = blocks_1.shape
    prefixes_1 = _projected_prefixes(
        blocks=blocks_1, projection_method=projection_method, bound_type=bound_type
    )
    prefixes_2 = (
        prefixes_1
        if self_product
        else _projected_prefixes(
            blocks=blocks_2, projection_method=projection_method, bound_type=bound_type
        )
    )
    result = np.zeros((num_bands + 1, size), dtype=np.float64)
    for start in range(0, num_bands, band_stride):
        stop = min(start + band_stride, num_bands)
        products = _band_products(
            blocks_1=blocks_1[start:stop],
            blocks_2=blocks_2[start:stop],
            prefixes_1=prefixes_1[start:stop],
            prefixes_2=prefixes_2[start:stop],
            self_product=self_product,
        )
        _route_band_products(
            result=result,
            products=products,
            first_row=start,
            projection_method=projection_method,
            bound_type=bound_type,
        )
    return result


def _sum_mass_tolerances(*, fft_length: int, charged_merges: int) -> tuple[float, float]:
    """Return ``(drift_tol, repair_tol)`` for the mass of a multiscale sum.

    The transforms of a merge are uncompensated, so one merge leaves a residual of
    order ``log2(fft_length) * eps`` times the mass entering it. A sum that is reused
    is charged in both factors of its parent, so ``charged_merges`` for ``n`` copies
    is ``n - 1`` and the raise threshold is linear in it. Residuals of either sign
    largely cancel, so the silent threshold follows its square root. A count of zero
    is charged as one, for the rounding of the leaves themselves.
    """
    require_positive_int(value=fft_length, name="fft_length")
    require_nonnegative_int(value=charged_merges, name="charged_merges")
    charge = max(charged_merges, 1)
    drift_tol, repair_tol = transform_mass_tolerances(num_bins=fft_length, num_convolutions=charge)
    return drift_tol / math.sqrt(charge), repair_tol


def _positive_span(leaf: DyadicDist) -> tuple[float, float] | None:
    """Return the smallest and the largest positive grid point that carries mass, if any."""
    occupied = np.flatnonzero(leaf.masses())
    if occupied.size == 0:
        return None
    support = leaf.support()
    return float(support[occupied[0]]), float(support[occupied[-1]])


def _exact_first_moment(leaves: Sequence[tuple[DyadicDist, int]]) -> float:
    """Return the first moment of the finite part of the unprojected sum of the leaves.

    With ``f`` the mass of a leaf that is not at infinity and ``mu`` its first moment,
    each copy contributes ``mu`` times the finite mass of every other copy.
    """
    finite = [leaf.finite_mass() for leaf, _ in leaves]
    terms = []
    for index, (leaf, count) in enumerate(leaves):
        others = math.prod(
            finite[other] ** other_count
            for other, (_, other_count) in enumerate(leaves)
            if other != index
        )
        terms.append(count * leaf.first_moment() * finite[index] ** (count - 1) * others)
    return math.fsum(terms)


# =============================================================================
# Merge Kernels
# =============================================================================


def _projected_prefixes(
    *,
    blocks: NDArray[np.float64],
    projection_method: ProjectionMethod,
    bound_type: BoundType,
) -> NDArray[np.float64]:
    """Return, for each octave, the mass below it projected onto that octave's lattice.

    Row ``j`` holds ``m + 1`` masses at the multiples ``0..m`` of the octave's spacing,
    whatever the number of lower octaves. The lattices are nested, so
    ``prefix[j + 1] = halve(prefix[j] + block[j])`` is exact for every projection.
    """
    num_bands, size = blocks.shape
    prefixes = np.zeros((num_bands, size + 1), dtype=np.float64)
    for band in range(num_bands - 1):
        joined = np.zeros(2 * size + 1, dtype=np.float64)
        prefix_stop, band_stop = size + 1, 2 * size
        joined[:prefix_stop] = prefixes[band]
        joined[size:band_stop] += blocks[band]
        prefixes[band + 1] = halve_lattice(
            values=joined, projection_method=projection_method, bound_type=bound_type
        )
    return prefixes


def _band_products(
    *,
    blocks_1: NDArray[np.float64],
    blocks_2: NDArray[np.float64],
    prefixes_1: NDArray[np.float64],
    prefixes_2: NDArray[np.float64],
    self_product: bool,
) -> NDArray[np.float64]:
    """Return, per octave, the sums of the pairs whose larger summand is in that octave.

    In units of the octave's spacing a block occupies indices ``m..2m-1`` and a prefix
    ``0..m``, so the products occupy ``m..4m-2`` and transforms of a fast length of at
    least ``4m`` do not wrap around; the first ``4m`` indices are returned. One inverse
    transform of ``F G + F V + U G`` gives block with block and each block with the
    other operand's prefix; prefix with prefix is absent because such a pair belongs
    to a lower octave.
    """
    chunk, size = blocks_1.shape
    length = next_fast_len(4 * size)
    band_stop, prefix_stop = 2 * size, size + 1
    padded_block_1 = np.zeros((chunk, length), dtype=np.float64)
    padded_block_1[:, size:band_stop] = blocks_1
    padded_prefix_1 = np.zeros((chunk, length), dtype=np.float64)
    padded_prefix_1[:, :prefix_stop] = prefixes_1
    block_spectrum_1 = rfft(padded_block_1, n=length, axis=1)
    prefix_spectrum_1 = rfft(padded_prefix_1, n=length, axis=1)
    if self_product:
        spectrum = block_spectrum_1 * (block_spectrum_1 + 2.0 * prefix_spectrum_1)
    else:
        padded_block_2 = np.zeros((chunk, length), dtype=np.float64)
        padded_block_2[:, size:band_stop] = blocks_2
        padded_prefix_2 = np.zeros((chunk, length), dtype=np.float64)
        padded_prefix_2[:, :prefix_stop] = prefixes_2
        block_spectrum_2 = rfft(padded_block_2, n=length, axis=1)
        prefix_spectrum_2 = rfft(padded_prefix_2, n=length, axis=1)
        spectrum = (
            block_spectrum_1 * block_spectrum_2
            + block_spectrum_1 * prefix_spectrum_2
            + prefix_spectrum_1 * block_spectrum_2
        )
    products = np.asarray(irfft(spectrum, n=length, axis=1, overwrite_x=True), dtype=np.float64)
    return products[:, : 4 * size]


def _route_band_products(
    *,
    result: NDArray[np.float64],
    products: NDArray[np.float64],
    first_row: int,
    projection_method: ProjectionMethod,
    bound_type: BoundType,
) -> None:
    """Add one batch of products to the output octaves, in place.

    A pair with a summand in octave ``j`` sums into ``[2**j, 2**(j+2))``, so indices
    below ``m`` and the last index carry only transform noise and are zeroed, as are
    negative values. Indices ``m..2m-1`` are the points of octave ``j``. Indices from
    ``2m`` lie in octave ``j + 1``, whose points are the even ones.
    """
    chunk, length = products.shape
    size = length // 4
    products[:, :size] = 0.0
    products[:, length - 1] = 0.0
    np.maximum(products, 0.0, out=products)
    row_stop, band_stop, carry_start = first_row + chunk, 2 * size, 2 * size
    result[first_row:row_stop] += products[:, size:band_stop]
    # The halving takes 2m + 1 points. Index 4m - 1 is unreachable, so a zero at 4m
    # completes the shape; its output slot would be octave j + 2 and is dropped.
    upper = np.concatenate(
        (products[:, carry_start:], np.zeros((chunk, 1), dtype=np.float64)), axis=1
    )
    dest_start, dest_stop = first_row + 1, first_row + chunk + 1
    halved = halve_lattice(values=upper, projection_method=projection_method, bound_type=bound_type)
    result[dest_start:dest_stop] += halved[:, :-1]
