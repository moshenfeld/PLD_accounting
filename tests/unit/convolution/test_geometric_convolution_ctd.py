"""Connect-the-dots merges on the geometric grid.

The contracts tested here are the three properties a merge needs: it projects the
exact pairwise sum, every sum goes to its two neighbouring grid points, and the
projection keeps the mean. The trim that follows a merge keeps the mean below and
may only lower it above.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from PLD_accounting.convolutions import self_convolve
from PLD_accounting.types import ConvolutionMethod
from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain, GridSpec
from PLD_accounting.geometric_convolution import geometric_convolve
from PLD_accounting.types import BoundType, ProjectionMethod, SpacingType
from tests.unit.allocation_oracles import project_onto_points

_CTD = ProjectionMethod.CTD
# Agreement of two float64 evaluations of the same sums of a few hundred products.
_SUM_RTOL = 1e-13


def _dist(
    *,
    step: float,
    index_0: int,
    masses: list[float],
    p_min: float,
) -> DenseDiscreteDist:
    """A geometric-grid distribution whose finite masses are scaled to total ``1 - p_min``."""
    prob_arr = np.asarray(masses, dtype=np.float64)
    prob_arr = prob_arr * ((1.0 - p_min) / math.fsum(prob_arr.tolist()))
    prob_arr[int(np.argmax(prob_arr))] += 1.0 - p_min - math.fsum(prob_arr.tolist())
    grid = GridSpec(
        step=step, n=prob_arr.size, spacing_type=SpacingType.GEOMETRIC, anchor=1.0, index_0=index_0
    )
    return DenseDiscreteDist(grid=grid, prob_arr=prob_arr, p_min=p_min, domain=Domain.POSITIVES)


def _random_dist(
    *, seed: int, size: int, step: float, index_0: int, p_min: float
) -> DenseDiscreteDist:
    rng = np.random.default_rng(seed)
    masses = list(rng.random(size) + 0.01)
    return _dist(step=step, index_0=index_0, masses=masses, p_min=p_min)


def _enumerated_sum(
    dist_1: DenseDiscreteDist, dist_2: DenseDiscreteDist
) -> tuple[np.ndarray, np.ndarray]:
    """Every positive value of the exact sum with its mass; the atoms at zero take part."""
    values_1 = np.concatenate(([0.0], dist_1.x_array))
    values_2 = np.concatenate(([0.0], dist_2.x_array))
    masses_1 = np.concatenate(([dist_1.p_min], dist_1.prob_arr))
    masses_2 = np.concatenate(([dist_2.p_min], dist_2.prob_arr))
    values = np.add.outer(values_1, values_2).ravel()
    masses = np.multiply.outer(masses_1, masses_2).ravel()
    atoms = (values > 0.0) & (masses > 0.0)
    return values[atoms], masses[atoms]


def _mean(dist: DenseDiscreteDist) -> float:
    return math.fsum((dist.prob_arr * dist.x_array).tolist())


def _merge_cases() -> list:
    """Pairs covering equal and unequal lattices, sums off the lattice, and atoms at zero."""
    step = 0.3
    same_lattice_1 = _random_dist(seed=1, size=7, step=step, index_0=-3, p_min=0.0)
    same_lattice_2 = _random_dist(seed=2, size=7, step=step, index_0=-3, p_min=0.0)
    shifted_1 = _random_dist(seed=3, size=5, step=step, index_0=-4, p_min=0.0)
    shifted_2 = _random_dist(seed=4, size=9, step=step, index_0=2, p_min=0.0)
    with_atom_at_zero = _random_dist(seed=5, size=6, step=step, index_0=-2, p_min=0.2)
    without_atom_at_zero = _random_dist(seed=6, size=4, step=step, index_0=-2, p_min=0.0)
    low_with_atom = _random_dist(seed=7, size=3, step=step, index_0=-9, p_min=0.3)
    high_with_atom = _random_dist(seed=8, size=8, step=step, index_0=1, p_min=0.1)
    single_point_1 = _random_dist(seed=9, size=1, step=step, index_0=0, p_min=0.5)
    single_point_2 = _random_dist(seed=10, size=1, step=step, index_0=5, p_min=0.5)
    return [
        pytest.param(same_lattice_1, same_lattice_2, id="same-lattice"),
        pytest.param(shifted_1, shifted_2, id="shifted-lattices"),
        pytest.param(with_atom_at_zero, without_atom_at_zero, id="one-atom-at-zero"),
        pytest.param(
            low_with_atom,
            high_with_atom,
            id="two-atoms-at-zero-cross-terms-outside-the-pairs",
        ),
        pytest.param(single_point_1, single_point_2, id="single-points"),
    ]


_MERGE_CASES = _merge_cases()


@pytest.mark.parametrize("dist_1, dist_2", _MERGE_CASES)
def test_merge_is_the_split_projection_of_the_exact_sum(dist_1, dist_2):
    """The merge equals the enumerated convolution projected onto the output grid.

    Each exact sum is split between its two neighbouring grid points with the weights
    that keep its mean; a sum with the atom at zero is the finite atom itself.
    """
    merged = geometric_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        projection_method=_CTD,
    )
    values, masses = _enumerated_sum(dist_1, dist_2)

    expected = project_onto_points(
        values=values, masses=masses, points=merged.x_array, mode="split"
    )
    np.testing.assert_allclose(merged.prob_arr, expected, rtol=1e-12, atol=1e-16)
    assert merged.p_min == pytest.approx(dist_1.p_min * dist_2.p_min, abs=1e-16)
    assert merged.p_max == 0.0


@pytest.mark.parametrize("dist_1, dist_2", _MERGE_CASES)
def test_merge_keeps_mass_and_mean(dist_1, dist_2):
    """Without trimming, the total mass is one and the mean is the sum of the means."""
    merged = geometric_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        projection_method=_CTD,
    )

    total = math.fsum(merged.prob_arr.tolist()) + merged.p_min + merged.p_max
    mass_atol = 4 * np.finfo(float).eps
    assert total == pytest.approx(1.0, abs=mass_atol)
    sum_of_means = _mean(dist_1) + _mean(dist_2)
    assert _mean(merged) == pytest.approx(sum_of_means, rel=_SUM_RTOL)


@pytest.mark.parametrize("dist_1, dist_2", _MERGE_CASES)
def test_merge_dominates_the_exact_sum_in_the_convex_order(dist_1, dist_2):
    """Every call and put payoff is worth at least as much under the merge."""
    merged = geometric_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        projection_method=_CTD,
    )
    values, masses = _enumerated_sum(dist_1, dist_2)

    for strike in np.quantile(values, [0.05, 0.3, 0.5, 0.8, 0.97]):
        for sign in (1.0, -1.0):
            exact = math.fsum((masses * np.maximum(sign * (values - strike), 0.0)).tolist())
            if sign < 0.0:
                exact += dist_1.p_min * dist_2.p_min * strike
            projected = math.fsum(
                (merged.prob_arr * np.maximum(sign * (merged.x_array - strike), 0.0)).tolist()
            )
            if sign < 0.0:
                projected += merged.p_min * strike
            assert projected >= exact * (1.0 - _SUM_RTOL)


def test_self_convolution_keeps_the_mean_over_many_merges():
    """Binary self-composition without trimming multiplies the mean by the count."""
    dist = _random_dist(seed=11, size=40, step=0.05, index_0=-20, p_min=0.05)
    summed = self_convolve(
        dist=dist,
        num_convolutions=13,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        projection_method=_CTD,
        convolution_method=ConvolutionMethod.GEOM,
    )

    assert summed.grid.anchor == 13.0
    expected_mean = 13.0 * _mean(dist)
    assert _mean(summed) == pytest.approx(expected_mean, rel=_SUM_RTOL)
    assert summed.p_min == pytest.approx(0.05**13, rel=1e-12)


def _two_tailed_dist() -> DenseDiscreteDist:
    """A bulk with a light tail on each side, on a unit-anchored lattice."""
    index = np.arange(-60, 61)
    return _dist(step=0.1, index_0=-60, masses=list(np.exp(-0.5 * (index / 6.0) ** 2)), p_min=0.0)


def test_mass_repair_of_a_merge_never_raises_the_mean_on_a_wide_lattice():
    """The rounding residual of a merge is repaired by moving mass down only.

    On a lattice that spans many orders of magnitude, a residual of one ulp of mass
    put at the largest sum would raise the mean by that sum's value times the
    residual; taking it from there, or putting it at zero, cannot raise the mean.
    """
    step = 0.05
    offsets = np.arange(-400, 401)
    # A bell in the logarithm with a heavy upper tail: values from exp(-20) to exp(20).
    masses = np.exp(-0.5 * (offsets * step / 4.0) ** 2) * np.exp(-offsets * step)
    dist = _dist(step=step, index_0=-400, masses=list(masses), p_min=0.1)
    count = 64

    summed = self_convolve(
        dist=dist,
        num_convolutions=count,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        projection_method=_CTD,
        convolution_method=ConvolutionMethod.GEOM,
    )

    total = math.fsum(summed.prob_arr.tolist()) + summed.p_min + summed.p_max
    assert total == pytest.approx(1.0, abs=1e-15)
    assert _mean(summed) <= count * _mean(dist) * (1.0 + _SUM_RTOL)


@pytest.mark.parametrize("tail_truncation", [1e-12, 1e-6, 1e-3])
def test_trim_regions_respect_the_tail_share_under_both_charges(tail_truncation: float):
    """Each trimmed region has mass and first moment over the anchor within the share."""
    dist = _two_tailed_dist()
    untrimmed, trimmed = (
        geometric_convolve(
            dist_1=dist,
            dist_2=dist,
            tail_truncation=share,
            bound_type=BoundType.DOMINATES,
            projection_method=_CTD,
        )
        for share in (0.0, tail_truncation)
    )
    points, masses = untrimmed.x_array, untrimmed.prob_arr
    below = points < trimmed.x_array[0] * (1.0 - 1e-12)
    above = points > trimmed.x_array[-1] * (1.0 + 1e-12)

    assert below.any() and above.any()
    for region in (below, above):
        assert math.fsum(masses[region].tolist()) <= tail_truncation
        moment = math.fsum((masses[region] * points[region]).tolist())
        assert moment / untrimmed.grid.anchor <= tail_truncation
    # One more point on either side would break a charge: the trim is the largest.
    next_below = int(np.flatnonzero(below)[-1]) + 1
    next_above = int(np.flatnonzero(above)[0]) - 1
    for region, extra in ((below, next_below), (above, next_above)):
        grown = region.copy()
        grown[extra] = True
        mass = math.fsum(masses[grown].tolist())
        moment = math.fsum((masses[grown] * points[grown]).tolist())
        assert max(mass, moment / untrimmed.grid.anchor) > tail_truncation


def test_lower_trim_keeps_mass_and_mean():
    """Below the first kept point the mass goes to that point and to the atom at zero."""
    # A heavy top point leaves nothing to trim above.
    index = np.arange(-80, 1)
    dist = _dist(step=0.1, index_0=-80, masses=list(np.exp(0.5 * index)), p_min=0.0)
    untrimmed, trimmed = (
        geometric_convolve(
            dist_1=dist,
            dist_2=dist,
            tail_truncation=share,
            bound_type=BoundType.DOMINATES,
            projection_method=_CTD,
        )
        for share in (0.0, 1e-6)
    )

    assert trimmed.x_array[0] > untrimmed.x_array[0]
    assert trimmed.x_array[-1] == untrimmed.x_array[-1]
    untrimmed_mean = _mean(untrimmed)
    assert _mean(trimmed) == pytest.approx(untrimmed_mean, rel=_SUM_RTOL)
    assert 0.0 < trimmed.p_min <= 1e-6
    total = math.fsum(trimmed.prob_arr.tolist()) + trimmed.p_min + trimmed.p_max
    mass_atol = 4 * np.finfo(float).eps
    assert total == pytest.approx(1.0, abs=mass_atol)


def test_upper_trim_keeps_mass_and_never_raises_the_mean():
    """Above the last kept point the mass goes to that point and its excess moment is dropped."""
    # A heavy bottom point leaves nothing to trim below.
    index = np.arange(0, 81)
    dist = _dist(step=0.1, index_0=0, masses=list(np.exp(-1.5 * index)), p_min=0.0)
    untrimmed, trimmed = (
        geometric_convolve(
            dist_1=dist,
            dist_2=dist,
            tail_truncation=share,
            bound_type=BoundType.DOMINATES,
            projection_method=_CTD,
        )
        for share in (0.0, 1e-6)
    )

    assert trimmed.x_array[-1] < untrimmed.x_array[-1]
    assert trimmed.x_array[0] == untrimmed.x_array[0]
    assert trimmed.p_min == 0.0 and trimmed.p_max == 0.0
    mass_atol = 4 * np.finfo(float).eps
    assert math.fsum(trimmed.prob_arr.tolist()) == pytest.approx(1.0, abs=mass_atol)
    dropped = _mean(untrimmed) - _mean(trimmed)
    assert 0.0 < dropped <= 1e-6 * untrimmed.grid.anchor


def test_a_light_point_with_a_large_moment_is_not_trimmed():
    """The moment charge keeps a far point whose mass alone would fit the share."""
    dist = _dist(
        step=1.0, index_0=0, masses=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-9], p_min=0.0
    )
    merged = geometric_convolve(
        dist_1=dist,
        dist_2=dist,
        tail_truncation=1e-6,
        bound_type=BoundType.DOMINATES,
        projection_method=_CTD,
    )

    # The far pairs carry about 2e-9 of mass and 2e-9 * e**8 / 2 > 1e-6 of moment share.
    assert merged.x_array[-1] > math.exp(8.0)
    expected_mean = 2.0 * _mean(dist)
    assert _mean(merged) == pytest.approx(expected_mean, rel=_SUM_RTOL)


def test_connect_the_dots_rejects_a_dominated_bound():
    """Connect-the-dots gives upper bounds only."""
    dist = _two_tailed_dist()
    with pytest.raises(ValueError, match="dominating bounds only"):
        geometric_convolve(
            dist_1=dist,
            dist_2=dist,
            tail_truncation=0.0,
            bound_type=BoundType.IS_DOMINATED,
            projection_method=_CTD,
        )
    with pytest.raises(ValueError, match="dominating bounds only"):
        self_convolve(
            dist=dist,
            num_convolutions=3,
            tail_truncation=0.0,
            bound_type=BoundType.IS_DOMINATED,
            projection_method=_CTD,
            convolution_method=ConvolutionMethod.GEOM,
        )
