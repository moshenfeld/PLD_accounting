"""Multiscale merges on the dyadic grid.

A merge is the pairwise sum of two measures followed by the projection onto the grid,
so every test compares it with the enumerated sum projected by the oracle. The
admission of a sum is tested on the residuals it judges.
"""

from __future__ import annotations

import math
import warnings
from copy import deepcopy
from functools import partial, reduce

import numpy as np
import pytest

from PLD_accounting.convolutions import loss_stage_count, self_convolve
from PLD_accounting.discrete_dist import DyadicDist
from PLD_accounting.multiscale_fft_convolution import (
    _admit_sum,
    _merge_positive_pairs,
    _sum_mass_tolerances,
    multiscale_convolve,
)
from PLD_accounting.types import BoundType, ConvolutionMethod, ProjectionMethod
from PLD_accounting.utils import binary_self_convolve
from tests.unit.allocation_oracles import dyadic_points, project_onto_points

_CTD, _ROUNDING = ProjectionMethod.CTD, ProjectionMethod.STOCH_DOM
_UPPER, _LOWER = BoundType.DOMINATES, BoundType.IS_DOMINATED
# Each projection of a merge with the oracle mode that places a sum the same way.
_PROJECTIONS = [
    pytest.param(_CTD, _UPPER, "split", id="ctd"),
    pytest.param(_ROUNDING, _UPPER, "up", id="rounding-up"),
    pytest.param(_ROUNDING, _LOWER, "down", id="rounding-down"),
]
# Agreement of a transform of a few dozen points with the enumerated products.
_MASS_ATOL = 1e-15
_CELLS = 4
# An FFT library may take a different vectorized path for a batch of rows than for one
# row, so batching changes each output by rounding alone: at most a few ulps of the
# largest output per halving of the transform length. Batches are compared in that unit.
_BATCHING_ROUNDING_ULPS = 16
_COPY_COUNTS = range(1, 34)


def _measure(
    *, seed: int, first_band: int, num_bands: int, p_min: float, p_max: float
) -> DyadicDist:
    """A measure of unit mass with random masses, a third of its grid points empty."""
    rng = np.random.default_rng(seed)
    masses = rng.random((num_bands, _CELLS))
    masses[rng.random((num_bands, _CELLS)) < 0.3] = 0.0
    masses *= (1.0 - p_min - p_max) / masses.sum()
    return DyadicDist(first_band=first_band, prob_arr=masses, p_min=p_min, p_max=p_max)


def _enumerated_merge(*, dist_1: DyadicDist, dist_2: DyadicDist, mode: str) -> DyadicDist:
    # pylint: disable=too-many-locals  # each compared quantity is named
    """The pairwise sums of two measures, each projected onto the grid by the oracle."""
    first_band = min(dist_1.first_band, dist_2.first_band)
    end_band = max(dist_1.first_band + dist_1.num_bands, dist_2.first_band + dist_2.num_bands)
    num_bands = end_band - first_band + 1
    cells = dist_1.cells_per_band
    values_1 = np.concatenate(([0.0], dist_1.support()))
    values_2 = np.concatenate(([0.0], dist_2.support()))
    masses_1 = np.concatenate(([dist_1.p_min], dist_1.masses()))
    masses_2 = np.concatenate(([dist_2.p_min], dist_2.masses()))
    sums = np.add.outer(values_1, values_2).ravel()
    products = np.multiply.outer(masses_1, masses_2).ravel()
    positive = sums > 0.0
    points = dyadic_points(first_band=first_band, num_bands=num_bands, cells_per_band=cells)
    projected = project_onto_points(
        values=sums[positive], masses=products[positive], points=points, mode=mode
    )
    # No sum reaches the point that closes the range.
    assert projected[-1] == 0.0
    prob_arr = projected[:-1].reshape(num_bands, cells)
    return DyadicDist(
        first_band=first_band,
        prob_arr=prob_arr,
        p_min=dist_1.p_min * dist_2.p_min,
        p_max=dist_1.p_max * dist_2.total_mass() + dist_2.p_max * dist_1.finite_mass(),
    )


def _enumerated_sum_of_copies(*, dist: DyadicDist, num_copies: int, mode: str) -> DyadicDist:
    """Enumerate projected powers of two, then merge the selected powers in ascending order."""
    powers = [dist]
    for _ in range(1, num_copies.bit_length()):
        powers.append(_enumerated_merge(dist_1=powers[-1], dist_2=powers[-1], mode=mode))
    selected = [power for bit, power in enumerate(powers) if num_copies & (1 << bit)]
    return reduce(
        lambda left, right: _enumerated_merge(dist_1=left, dist_2=right, mode=mode), selected
    )


def _assert_same_measure(*, actual: DyadicDist, expected: DyadicDist) -> None:
    """Equal masses on the octaves the two share; nothing outside them."""
    assert actual.first_band == expected.first_band
    shared = min(actual.num_bands, expected.num_bands)
    np.testing.assert_allclose(
        actual.prob_arr[:shared], expected.prob_arr[:shared], rtol=0.0, atol=_MASS_ATOL
    )
    assert not np.any(actual.prob_arr[shared:]) and not np.any(expected.prob_arr[shared:])
    assert actual.p_min == pytest.approx(expected.p_min, abs=_MASS_ATOL)
    assert actual.p_max == pytest.approx(expected.p_max, abs=_MASS_ATOL)


# =============================================================================
# The Merge
# =============================================================================


@pytest.mark.parametrize("projection_method, bound_type, mode", _PROJECTIONS)
def test_merge_is_the_enumerated_sum_followed_by_the_projection(
    projection_method: ProjectionMethod, bound_type: BoundType, mode: str
):
    """Two measures on different octaves, with atoms at zero and mass at infinity."""
    dist_1 = _measure(seed=1, first_band=-2, num_bands=3, p_min=0.1, p_max=0.05)
    dist_2 = _measure(seed=2, first_band=0, num_bands=2, p_min=0.2, p_max=0.0)

    merged = multiscale_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        projection_method=projection_method,
        bound_type=bound_type,
        tail_truncation=0.0,
    )

    expected = _enumerated_merge(dist_1=dist_1, dist_2=dist_2, mode=mode)
    _assert_same_measure(actual=merged, expected=expected)


@pytest.mark.parametrize("projection_method, bound_type, mode", _PROJECTIONS)
def test_merge_weights_the_boundary_atoms_by_the_masses_of_measures_that_are_not_unit(
    projection_method: ProjectionMethod, bound_type: BoundType, mode: str
):
    """The mass of a merge is the product of the masses, at infinity as elsewhere."""
    unit_1 = _measure(seed=5, first_band=-1, num_bands=2, p_min=0.1, p_max=0.15)
    unit_2 = _measure(seed=6, first_band=0, num_bands=2, p_min=0.2, p_max=0.1)
    dist_1 = DyadicDist(
        first_band=unit_1.first_band,
        prob_arr=0.3 * unit_1.prob_arr,
        p_min=0.3 * unit_1.p_min,
        p_max=0.3 * unit_1.p_max,
    )
    dist_2 = DyadicDist(
        first_band=unit_2.first_band,
        prob_arr=0.6 * unit_2.prob_arr,
        p_min=0.6 * unit_2.p_min,
        p_max=0.6 * unit_2.p_max,
    )

    merged = multiscale_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        projection_method=projection_method,
        bound_type=bound_type,
        tail_truncation=0.0,
    )
    squared = multiscale_convolve(
        dist_1=dist_1,
        dist_2=dist_1,
        projection_method=projection_method,
        bound_type=bound_type,
        tail_truncation=0.0,
    )

    expected = _enumerated_merge(dist_1=dist_1, dist_2=dist_2, mode=mode)
    _assert_same_measure(actual=merged, expected=expected)
    assert merged.total_mass() == pytest.approx(0.18, abs=_MASS_ATOL)
    assert merged.p_max == pytest.approx(0.18 * (0.15 + 0.1 - 0.015), abs=_MASS_ATOL)
    assert squared.total_mass() == pytest.approx(0.09, abs=_MASS_ATOL)
    assert squared.p_max == pytest.approx(0.09 * (0.3 - 0.0225), abs=_MASS_ATOL)


@pytest.mark.parametrize("cells", [1, 3, 5, 12])
def test_merge_holds_for_any_number_of_points_per_octave(cells: int):
    """The lattices of the octaves are nested whether or not the count is a power of two.

    Only connect-the-dots is compared: such grid points are not exact floats, so a sum
    that lands on one is a rounding away from it for an oracle that rounds.
    """
    rng = np.random.default_rng(12)
    masses_1, masses_2 = rng.random((3, cells)), rng.random((2, cells))
    prob_arr_1 = 0.9 * masses_1 / masses_1.sum()
    dist_1 = DyadicDist(first_band=-2, prob_arr=prob_arr_1, p_min=0.1)
    prob_arr_2 = 0.8 * masses_2 / masses_2.sum()
    dist_2 = DyadicDist(first_band=0, prob_arr=prob_arr_2, p_min=0.2)

    merged = multiscale_convolve(
        dist_1=dist_1, dist_2=dist_2, projection_method=_CTD, bound_type=_UPPER, tail_truncation=0.0
    )

    expected = _enumerated_merge(dist_1=dist_1, dist_2=dist_2, mode="split")
    _assert_same_measure(actual=merged, expected=expected)


@pytest.mark.parametrize("projection_method, bound_type, mode", _PROJECTIONS)
def test_self_product_path_equals_the_general_path(
    projection_method: ProjectionMethod, bound_type: BoundType, mode: str
):
    """One object as both operands gives the merge of two equal measures."""
    del mode
    dist = _measure(seed=3, first_band=-1, num_bands=3, p_min=0.1, p_max=0.05)
    copied_prob_arr = dist.prob_arr.copy()
    copy = DyadicDist(first_band=dist.first_band, prob_arr=copied_prob_arr, p_min=0.1, p_max=0.05)

    self_product = multiscale_convolve(
        dist_1=dist,
        dist_2=dist,
        projection_method=projection_method,
        bound_type=bound_type,
        tail_truncation=0.0,
    )
    general = multiscale_convolve(
        dist_1=dist,
        dist_2=copy,
        projection_method=projection_method,
        bound_type=bound_type,
        tail_truncation=0.0,
    )

    _assert_same_measure(actual=self_product, expected=general)


@pytest.mark.parametrize("projection_method, bound_type, mode", _PROJECTIONS)
def test_result_does_not_depend_on_how_many_octaves_are_transformed_together(
    projection_method: ProjectionMethod, bound_type: BoundType, mode: str
):
    """The octaves are independent rows of the transforms, up to the rounding of a transform."""
    del mode
    blocks_1 = _measure(seed=4, first_band=0, num_bands=7, p_min=0.0, p_max=0.0).prob_arr
    blocks_2 = _measure(seed=5, first_band=0, num_bands=7, p_min=0.0, p_max=0.0).prob_arr

    together, by_three, one_by_one = (
        _merge_positive_pairs(
            blocks_1=blocks_1,
            blocks_2=blocks_2,
            self_product=False,
            band_stride=band_stride,
            projection_method=projection_method,
            bound_type=bound_type,
        )
        for band_stride in (7, 3, 1)
    )

    transform_length = 4 * blocks_1.shape[1]
    atol = (
        _BATCHING_ROUNDING_ULPS
        * math.log2(transform_length)
        * np.finfo(np.float64).eps
        * float(np.abs(together).max())
    )
    np.testing.assert_allclose(by_three, together, rtol=0.0, atol=atol)
    np.testing.assert_allclose(one_by_one, together, rtol=0.0, atol=atol)


def test_merge_rejects_operands_it_cannot_combine():
    """Different points per octave, and a lower CtD bound."""
    dist = _measure(seed=6, first_band=0, num_bands=2, p_min=0.0, p_max=0.0)
    wider_prob_arr = np.full((1, 8), 0.125)
    wider = DyadicDist(first_band=0, prob_arr=wider_prob_arr)

    with pytest.raises(ValueError, match="cells_per_band"):
        multiscale_convolve(
            dist_1=dist,
            dist_2=wider,
            projection_method=_CTD,
            bound_type=_UPPER,
            tail_truncation=0.0,
        )
    with pytest.raises(ValueError, match="DOMINATES only"):
        multiscale_convolve(
            dist_1=dist, dist_2=dist, projection_method=_CTD, bound_type=_LOWER, tail_truncation=0.0
        )


# =============================================================================
# The Sum of Copies
# =============================================================================


@pytest.mark.parametrize("num_copies", [1, 13])
@pytest.mark.parametrize("projection_method, bound_type, mode", _PROJECTIONS)
def test_binary_self_convolve_accepts_a_dyadic_merge_with_zero_tail(
    projection_method: ProjectionMethod, bound_type: BoundType, mode: str, num_copies: int
) -> None:
    """A zero tail budget sums dyadic measures through the shared loop."""
    dist = _measure(seed=7, first_band=-1, num_bands=2, p_min=0.15, p_max=0.0)
    summed = binary_self_convolve(
        dist=dist,
        num_convolutions=num_copies,
        tail_truncation=0.0,
        bound_type=bound_type,
        convolve=partial(multiscale_convolve, projection_method=projection_method),
    )
    expected = _enumerated_sum_of_copies(dist=dist, num_copies=num_copies, mode=mode)
    _assert_same_measure(actual=summed, expected=expected)


@pytest.mark.parametrize("num_copies", [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17, 23, 31, 33])
@pytest.mark.parametrize("projection_method, bound_type, mode", _PROJECTIONS)
def test_sum_of_copies_is_the_enumerated_tree_of_merges(
    projection_method: ProjectionMethod, bound_type: BoundType, mode: str, num_copies: int
):
    """Projected squaring agrees with enumerated merges, including counts with several set bits."""
    dist = _measure(seed=7, first_band=-1, num_bands=2, p_min=0.15, p_max=0.0)

    summed = self_convolve(
        dist=dist,
        num_convolutions=num_copies,
        projection_method=projection_method,
        bound_type=bound_type,
        tail_truncation=0.0,
        convolution_method=ConvolutionMethod.MS_FFT,
    )

    expected = _enumerated_sum_of_copies(dist=dist, num_copies=num_copies, mode=mode)
    _assert_same_measure(actual=summed, expected=expected)


@pytest.mark.parametrize("num_copies", _COPY_COUNTS)
@pytest.mark.parametrize("projection_method, bound_type, mode", _PROJECTIONS)
def test_sum_of_copies_keeps_mass_and_moves_the_first_moment_only_as_its_projection_may(
    projection_method: ProjectionMethod, bound_type: BoundType, mode: str, num_copies: int
):
    """Connect-the-dots keeps the first moment; rounding moves it to its own side."""
    dist = _measure(seed=8, first_band=-3, num_bands=4, p_min=0.1, p_max=0.0)
    drift_tol, _ = _sum_mass_tolerances(fft_length=4 * _CELLS, charged_merges=num_copies - 1)
    exact_moment = num_copies * dist.first_moment()

    summed = self_convolve(
        dist=dist,
        num_convolutions=num_copies,
        projection_method=projection_method,
        bound_type=bound_type,
        tail_truncation=0.0,
        convolution_method=ConvolutionMethod.MS_FFT,
    )

    assert summed.total_mass() == pytest.approx(1.0, abs=drift_tol)
    moment_shift = summed.first_moment() - exact_moment
    widest_shift = exact_moment * ((1.0 + 1.0 / _CELLS) ** (num_copies - 1).bit_length() - 1.0)
    if mode == "split":
        assert abs(moment_shift) <= drift_tol * exact_moment
    elif mode == "up":
        assert -drift_tol * exact_moment <= moment_shift <= widest_shift
    else:
        assert -widest_shift <= moment_shift <= drift_tol * exact_moment


@pytest.mark.parametrize(
    "num_copies, depth", [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4), (1000, 10)]
)
def test_stages_of_a_sum_are_the_two_projections_and_the_depth_of_its_tree(
    num_copies: int, depth: int
):
    """One copy needs no merge; each doubling of the copies adds a level."""
    stages = loss_stage_count(
        num_convolutions=num_copies, convolution_method=ConvolutionMethod.MS_FFT
    )
    assert stages == depth + 2


# =============================================================================
# Admission of a Merge
# =============================================================================


def _merged_pair(
    *, projection_method: ProjectionMethod, bound_type: BoundType
) -> tuple[DyadicDist, DyadicDist, DyadicDist]:
    """A merge of two measures with atoms at zero, and its two operands."""
    dist_1 = _measure(seed=9, first_band=-1, num_bands=2, p_min=0.1, p_max=0.0)
    dist_2 = _measure(seed=12, first_band=0, num_bands=2, p_min=0.05, p_max=0.0)
    merged = multiscale_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        projection_method=projection_method,
        bound_type=bound_type,
        tail_truncation=0.0,
    )
    return merged, dist_1, dist_2


def _with_extra_mass(*, summed: DyadicDist, band: int, mass: float) -> DyadicDist:
    """The measure with ``mass`` added at the first point of an octave below its support."""
    extended = np.vstack((np.zeros((summed.first_band - band, _CELLS)), summed.prob_arr))
    extended[0, 0] = mass
    return DyadicDist(first_band=band, prob_arr=extended, p_min=summed.p_min, p_max=summed.p_max)


@pytest.mark.parametrize("projection_method, bound_type, mode", _PROJECTIONS)
def test_a_merge_the_library_computed_is_admitted_unchanged(
    projection_method: ProjectionMethod, bound_type: BoundType, mode: str
):
    """A merge passes every check without a warning, and a second admission keeps it."""
    del mode
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        merged, dist_1, dist_2 = _merged_pair(
            projection_method=projection_method, bound_type=bound_type
        )
        admitted = _admit_sum(
            summed=merged,
            leaves=[(dist_1, 1), (dist_2, 1)],
            projection_method=projection_method,
            bound_type=bound_type,
        )

    assert admitted is merged


def test_admission_clears_mass_on_points_no_sum_of_the_operands_reaches():
    """Mass below the smallest operand value is cleared: silently, with a warning, or rejected."""
    merged, dist_1, dist_2 = _merged_pair(projection_method=_CTD, bound_type=_UPPER)
    drift_tol, repair_tol = _sum_mass_tolerances(fft_length=4 * _CELLS, charged_merges=1)

    def admit(mass: float) -> DyadicDist:
        extended = _with_extra_mass(summed=merged, band=merged.first_band - 3, mass=mass)
        return _admit_sum(
            summed=extended,
            leaves=[(dist_1, 1), (dist_2, 1)],
            projection_method=_CTD,
            bound_type=_UPPER,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        silently = admit(drift_tol / 4)
    with pytest.warns(RuntimeWarning, match="unreachable grid points"):
        warned = admit((drift_tol + repair_tol) / 2)
    with pytest.raises(ValueError, match="unreachable grid points"):
        admit(2 * repair_tol)

    for admitted in (silently, warned):
        assert not np.any(admitted.prob_arr[:3])
        np.testing.assert_array_equal(admitted.prob_arr[3:], merged.prob_arr)


def test_admission_rejects_a_first_moment_on_the_wrong_side_of_rounding():
    """A merge rounded up has a larger first moment than a merge rounded down may have."""
    merged, dist_1, dist_2 = _merged_pair(projection_method=_ROUNDING, bound_type=_UPPER)

    with pytest.raises(ValueError, match="first moment"):
        _admit_sum(
            summed=merged,
            leaves=[(dist_1, 1), (dist_2, 1)],
            projection_method=_ROUNDING,
            bound_type=_LOWER,
        )


def test_admission_rejects_a_first_moment_that_connect_the_dots_did_not_keep():
    """A merge rounded up is not a connect-the-dots merge of the same operands."""
    merged, dist_1, dist_2 = _merged_pair(projection_method=_ROUNDING, bound_type=_UPPER)

    with pytest.raises(ValueError, match="first moment"):
        _admit_sum(
            summed=merged,
            leaves=[(dist_1, 1), (dist_2, 1)],
            projection_method=_CTD,
            bound_type=_UPPER,
        )


def test_a_merge_with_mass_at_infinity_is_admitted():
    """Copies of one measure and a second one, as in the rounding sum of the remove direction."""
    copies = _measure(seed=10, first_band=-1, num_bands=2, p_min=0.1, p_max=0.0)
    last = _measure(seed=11, first_band=-1, num_bands=2, p_min=0.0, p_max=0.05)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        four_copies = self_convolve(
            dist=copies,
            num_convolutions=4,
            projection_method=_ROUNDING,
            bound_type=_UPPER,
            tail_truncation=0.0,
            convolution_method=ConvolutionMethod.MS_FFT,
        )
        summed = multiscale_convolve(
            dist_1=four_copies,
            dist_2=last,
            projection_method=_ROUNDING,
            bound_type=_UPPER,
            tail_truncation=0.0,
        )

    assert summed.p_max == pytest.approx(0.05, abs=_MASS_ATOL)


def test_a_merge_is_admitted_when_its_finite_mass_is_below_the_rounding_of_its_total():
    """The finite mass of an operand is summed on its own, not as the total less infinity."""
    finite_mass = 1e-17
    leaf = DyadicDist(first_band=100, prob_arr=np.array([[finite_mass, 0.0]]), p_min=0.0, p_max=1.0)
    assert leaf.total_mass() - leaf.p_max == 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        summed = multiscale_convolve(
            dist_1=leaf, dist_2=leaf, projection_method=_CTD, bound_type=_UPPER, tail_truncation=0.0
        )

    assert summed.first_moment() == pytest.approx(2 * leaf.first_moment() * finite_mass)
    assert summed.first_moment() > 1e-4


def test_a_merge_with_an_operand_at_zero_only_keeps_the_other_operand():
    """An atom at zero is neutral under addition, and gives no point to clear against."""
    at_zero = DyadicDist(first_band=0, prob_arr=np.zeros((1, _CELLS)), p_min=1.0)
    other = _measure(seed=13, first_band=-1, num_bands=2, p_min=0.1, p_max=0.0)

    merged = multiscale_convolve(
        dist_1=at_zero, dist_2=other, projection_method=_CTD, bound_type=_UPPER, tail_truncation=0.0
    )

    assert merged.total_mass() == pytest.approx(other.total_mass(), abs=_MASS_ATOL)
    assert merged.first_moment() == pytest.approx(other.first_moment(), abs=_MASS_ATOL)
    assert merged.p_min == pytest.approx(other.p_min, abs=_MASS_ATOL)


def test_mass_tolerances_grow_with_the_transform_length_and_with_the_charge():
    """The raise threshold is linear in the charge; the silent one follows its square root."""
    drift, repair = _sum_mass_tolerances(fft_length=1024, charged_merges=16)
    longer_drift, longer_repair = _sum_mass_tolerances(fft_length=4096, charged_merges=16)
    charged_drift, charged_repair = _sum_mass_tolerances(fft_length=1024, charged_merges=64)

    assert 0.0 < drift < repair
    assert longer_drift > drift and longer_repair > repair
    assert charged_repair == pytest.approx(4.0 * repair)
    assert charged_drift == pytest.approx(2.0 * drift)
    # A sum without a merge is charged as one merge.
    assert _sum_mass_tolerances(fft_length=1024, charged_merges=0) == _sum_mass_tolerances(
        fft_length=1024, charged_merges=1
    )


# =============================================================================
# The Measure Type
# =============================================================================


_ONE_DIMENSIONAL = np.full(4, 0.25)
_NO_OCTAVE = np.zeros((0, 4))
_NO_CELL = np.zeros((2, 0))
_NEGATIVE_MASS = np.array([[0.5, -0.1], [0.3, 0.3]])
_NAN_MASS = np.array([[0.5, np.nan], [0.3, 0.2]])
_INFINITE_MASS = np.array([[0.5, np.inf], [0.3, 0.2]])
_INVALID_MASSES = [
    pytest.param(_ONE_DIMENSIONAL, "shape", id="one-dimensional"),
    pytest.param(_NO_OCTAVE, "shape", id="no-octave"),
    pytest.param(_NO_CELL, "cells_per_band", id="no-cell"),
    pytest.param(_NEGATIVE_MASS, "nonnegative", id="negative-mass"),
    pytest.param(_NAN_MASS, "finite", id="nan-mass"),
    pytest.param(_INFINITE_MASS, "finite", id="infinite-mass"),
]


@pytest.mark.parametrize("prob_arr, message", _INVALID_MASSES)
def test_dyadic_measure_rejects_arrays_that_are_not_masses_on_octaves(prob_arr, message: str):
    """The masses are a finite nonnegative array with one row per octave."""
    with pytest.raises(ValueError, match=message):
        DyadicDist(first_band=0, prob_arr=prob_arr)


def test_dyadic_measure_rejects_a_negative_boundary_mass():
    """The atoms at zero and at infinity are masses too."""
    prob_arr = np.full((1, 2), 0.5)
    with pytest.raises(ValueError, match="p_min"):
        DyadicDist(first_band=0, prob_arr=prob_arr, p_min=-0.1)


def test_dyadic_measure_reports_its_points_mass_and_first_moment():
    """Octave ``j`` holds ``2**j (1 + i/m)``; the atoms at zero and infinity count as mass."""
    prob_arr = np.array([[0.1, 0.2], [0.3, 0.1]])
    dist = DyadicDist(first_band=-1, prob_arr=prob_arr, p_min=0.2, p_max=0.1)

    support = dist.support()
    np.testing.assert_array_equal(support, [0.5, 0.75, 1.0, 1.5])
    points = dyadic_points(first_band=-1, num_bands=2, cells_per_band=2)
    np.testing.assert_array_equal(support, points[:-1])
    assert (dist.cells_per_band, dist.num_bands) == (2, 2)
    assert dist.total_mass() == pytest.approx(1.0, abs=_MASS_ATOL)
    assert dist.first_moment() == pytest.approx(0.05 + 0.15 + 0.3 + 0.15, abs=_MASS_ATOL)


def test_dyadic_measure_detaches_input_before_caching_totals():
    """Mutating the caller's array cannot invalidate a measure's cached totals."""
    masses = np.array([[0.25, 0.75]])
    dist = DyadicDist(first_band=0, prob_arr=masses)
    expected = (dist.total_mass(), dist.finite_mass(), dist.first_moment())

    masses[0, 0] = 1.25

    np.testing.assert_array_equal(dist.masses(), [0.25, 0.75])
    assert (dist.total_mass(), dist.finite_mass(), dist.first_moment()) == expected
    assert masses.flags.writeable


@pytest.mark.parametrize("copy_measure", [False, True], ids=["original", "deepcopy"])
def test_dyadic_measure_exposes_readonly_masses(copy_measure: bool):
    """Neither array accessor can mutate masses, including after a deep copy."""
    original = DyadicDist(first_band=0, prob_arr=np.array([[0.25, 0.75]]))
    original.total_mass()
    dist = deepcopy(original) if copy_measure else original

    if copy_measure:
        assert not np.shares_memory(dist.prob_arr, original.prob_arr)
    with pytest.raises(ValueError, match="read-only"):
        dist.prob_arr[0, 0] = 1.25
    with pytest.raises(ValueError, match="read-only"):
        dist.masses()[0] = 1.25
    assert dist.total_mass() == 1.0
