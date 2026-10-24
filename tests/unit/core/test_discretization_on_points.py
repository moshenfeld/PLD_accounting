"""Discretization of a one-step input onto given loss points.

The four functions that take loss points share their cell computations with the four
fixed-gap ones, so on the points of a fixed-gap grid the two give the same masses. On
points that are not equally spaced, connect-the-dots keeps the privacy profile at
every point and dominates it between them, and rounding moves each cell to its side.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from PLD_accounting.discrete_dist import GridSpec, SparseDiscreteDist
from PLD_accounting.mechanisms import gaussian_loss_law
from PLD_accounting.projections_ctd import (
    project_continuous_dist_ctd,
    project_continuous_dist_ctd_on_points,
    project_dist_onto_grid_ctd,
    project_discrete_dist_ctd_on_points,
)
from PLD_accounting.projections_stoch_dom import (
    project_continuous_dist_stoch_dom,
    project_continuous_dist_stoch_dom_on_points,
    project_dist_onto_grid_stoch_dom,
    project_discrete_dist_stoch_dom_on_points,
)
from PLD_accounting.types import BoundType
from tests.unit.allocation_oracles import dyadic_points, gaussian_delta, project_onto_points

_UPPER, _LOWER = BoundType.DOMINATES, BoundType.IS_DOMINATED
_BOUNDS = [pytest.param(_UPPER, id="upper"), pytest.param(_LOWER, id="lower")]
_SIGMA = 1.5
# A fixed-gap grid passes its step as every cell width; the points give differences of
# materialized points, which agree with the step to a rounding of the points.
_WIDTH_RTOL = 1e-10
# Agreement of two float64 evaluations of one privacy profile.
_PROFILE_RTOL = 1e-11
_PROFILE_ATOL = 1e-15

# Losses that are not equally spaced: the logarithms of the dyadic points of [1/4, 4].
_LOSS_POINTS = np.log(dyadic_points(first_band=-2, num_bands=4, cells_per_band=8))


def _atoms() -> SparseDiscreteDist:
    """A discrete PLD with atoms inside the loss points, away from the points themselves."""
    rng = np.random.default_rng(0)
    losses = np.sort(rng.uniform(_LOSS_POINTS[0] + 0.01, _LOSS_POINTS[-1] - 0.01, size=40))
    masses = rng.random(40)
    # Scaling the masses of P leaves E[exp(-L)] at 0.8, a valid PLD with mass at +inf.
    masses *= 0.8 / float(np.sum(masses * np.exp(-losses)))
    finite_mass = math.fsum(masses.tolist())
    return SparseDiscreteDist(x_array=losses, prob_arr=masses, p_max=1.0 - finite_mass)


def _delta(*, dist: SparseDiscreteDist, epsilon: float) -> float:
    """``delta(epsilon)`` of a loss distribution: its hockey-stick value."""
    signed_excess = -np.expm1(epsilon - dist.x_array)
    excess = np.maximum(signed_excess, 0.0)
    return math.fsum((dist.prob_arr * excess).tolist()) + dist.p_max


# =============================================================================
# The Points of a Fixed-Gap Grid
# =============================================================================


def test_continuous_ctd_on_the_points_of_a_grid_gives_the_masses_of_the_grid():
    """The Gaussian law on the grid that its own tail budget selects."""
    law = gaussian_loss_law(_SIGMA)
    on_grid = project_continuous_dist_ctd(dist=law, dual_dist=law, tail_truncation=1e-8, step=0.05)

    on_points = project_continuous_dist_ctd_on_points(
        dist=law, dual_dist=law, loss_points=on_grid.x_array
    )

    np.testing.assert_array_equal(on_points.x_array, on_grid.x_array)
    np.testing.assert_allclose(on_points.prob_arr, on_grid.prob_arr, rtol=_WIDTH_RTOL, atol=1e-18)
    assert on_points.p_max == pytest.approx(on_grid.p_max, rel=_WIDTH_RTOL, abs=1e-18)


def test_discrete_ctd_on_the_points_of_a_grid_gives_the_masses_of_the_grid():
    """Atoms projected onto a grid given as a lattice and as its points."""
    source = _atoms()
    grid = GridSpec(step=0.1, n=31, index_0=-15)

    on_grid = project_dist_onto_grid_ctd(dist=source, grid=grid)
    loss_points = grid.materialize()
    on_points = project_discrete_dist_ctd_on_points(dist=source, loss_points=loss_points)

    np.testing.assert_allclose(on_points.prob_arr, on_grid.prob_arr, rtol=_WIDTH_RTOL, atol=1e-18)
    assert on_points.p_max == pytest.approx(on_grid.p_max, rel=_WIDTH_RTOL)


@pytest.mark.parametrize("bound_type", _BOUNDS)
def test_continuous_rounding_on_the_points_of_a_grid_gives_the_masses_of_the_grid(
    bound_type: BoundType,
):
    """On this grid every cell holds more than the tail budget, so the grid batches none."""
    law = stats.norm()
    on_grid = project_continuous_dist_stoch_dom(
        dist=law, tail_truncation=1e-3, bound_type=bound_type, step=0.5
    )

    on_points = project_continuous_dist_stoch_dom_on_points(
        dist=law, loss_points=on_grid.x_array, bound_type=bound_type
    )

    np.testing.assert_array_equal(on_points.prob_arr, on_grid.prob_arr)
    assert (on_points.p_min, on_points.p_max) == (on_grid.p_min, on_grid.p_max)


@pytest.mark.parametrize("bound_type", _BOUNDS)
def test_discrete_rounding_on_the_points_of_a_grid_gives_the_masses_of_the_grid(
    bound_type: BoundType,
):
    """The two rounding functions share one kernel, so the masses are equal bit for bit."""
    atoms = _atoms()
    masses = atoms.prob_arr.copy()
    masses[-1] += atoms.p_max
    source = SparseDiscreteDist(x_array=atoms.x_array, prob_arr=masses)
    grid = GridSpec(step=0.1, n=31, index_0=-15)

    on_grid = project_dist_onto_grid_stoch_dom(dist=source, grid=grid, bound_type=bound_type)
    loss_points = grid.materialize()
    on_points = project_discrete_dist_stoch_dom_on_points(
        dist=source, loss_points=loss_points, bound_type=bound_type
    )

    np.testing.assert_array_equal(on_points.prob_arr, on_grid.prob_arr)
    assert (on_points.p_min, on_points.p_max) == (on_grid.p_min, on_grid.p_max)


# =============================================================================
# Points That Are Not Equally Spaced
# =============================================================================


def test_discrete_ctd_keeps_mass_reciprocal_moment_and_the_profile_at_every_point():
    """Each cell's mass and reciprocal moment go to its two points, so the profile is kept there."""
    source = _atoms()

    projected = project_discrete_dist_ctd_on_points(dist=source, loss_points=_LOSS_POINTS)

    np.testing.assert_array_equal(projected.x_array, _LOSS_POINTS)
    assert projected.p_min == 0.0
    assert projected.p_max == pytest.approx(source.p_max, rel=_PROFILE_RTOL)
    projected_moment = math.fsum((projected.prob_arr * np.exp(-_LOSS_POINTS)).tolist())
    source_moment = math.fsum((source.prob_arr * np.exp(-source.x_array)).tolist())
    assert projected_moment == pytest.approx(source_moment, rel=_PROFILE_RTOL)
    for epsilon in _LOSS_POINTS:
        projected_delta = _delta(dist=projected, epsilon=epsilon)
        source_delta = _delta(dist=source, epsilon=epsilon)
        assert projected_delta == pytest.approx(source_delta, rel=_PROFILE_RTOL, abs=_PROFILE_ATOL)
    for epsilon in (_LOSS_POINTS[:-1] + _LOSS_POINTS[1:]) / 2:
        assert _delta(dist=projected, epsilon=epsilon) >= _delta(dist=source, epsilon=epsilon)


def test_discrete_ctd_tails_follow_the_boundary_rule():
    """Above the points the reciprocal moment stays at the last one; the rest goes to ``+inf``."""
    last = float(_LOSS_POINTS[-1])
    x_array = np.array([0.1, last + 1.0])
    prob_arr = np.array([0.7, 0.2])
    source = SparseDiscreteDist(x_array=x_array, prob_arr=prob_arr, p_max=0.1)

    projected = project_discrete_dist_ctd_on_points(dist=source, loss_points=_LOSS_POINTS)

    kept_at_last = 0.2 * math.exp(-1.0)
    assert projected.prob_arr[-1] == pytest.approx(kept_at_last, rel=_PROFILE_RTOL)
    assert projected.p_max == pytest.approx(0.1 + 0.2 - kept_at_last, rel=_PROFILE_RTOL)
    for epsilon in np.linspace(_LOSS_POINTS[0], last + 2.0, 40):
        assert _delta(dist=projected, epsilon=epsilon) >= _delta(dist=source, epsilon=epsilon) * (
            1.0 - _PROFILE_RTOL
        )


def test_discrete_ctd_puts_the_mass_below_the_points_at_the_first_one_or_at_infinity():
    """An atom below the points is not placed below them, and the bound stays above."""
    first = float(_LOSS_POINTS[0])
    x_array = np.array([first - 1.0, 0.1])
    prob_arr = np.array([0.01, 0.9])
    source = SparseDiscreteDist(x_array=x_array, prob_arr=prob_arr, p_max=0.09)

    projected = project_discrete_dist_ctd_on_points(dist=source, loss_points=_LOSS_POINTS)

    assert projected.p_min == 0.0
    assert math.fsum(projected.prob_arr.tolist()) + projected.p_max == pytest.approx(1.0)
    for epsilon in np.linspace(first - 2.0, _LOSS_POINTS[-1], 40):
        assert _delta(dist=projected, epsilon=epsilon) >= _delta(dist=source, epsilon=epsilon) * (
            1.0 - _PROFILE_RTOL
        )


def test_continuous_ctd_matches_the_gaussian_profile_at_the_points_and_dominates_between():
    """The closed-form Gaussian profile is kept at every point inside the range."""
    law = gaussian_loss_law(_SIGMA)

    projected = project_continuous_dist_ctd_on_points(
        dist=law, dual_dist=law, loss_points=_LOSS_POINTS
    )

    for epsilon in _LOSS_POINTS:
        projected_delta = _delta(dist=projected, epsilon=epsilon)
        exact_delta = gaussian_delta(sigma=_SIGMA, epsilon=epsilon)
        assert projected_delta == pytest.approx(exact_delta, rel=1e-9, abs=_PROFILE_ATOL)
    for epsilon in (_LOSS_POINTS[:-1] + _LOSS_POINTS[1:]) / 2:
        assert _delta(dist=projected, epsilon=epsilon) >= gaussian_delta(
            sigma=_SIGMA, epsilon=epsilon
        )


@pytest.mark.parametrize("bound_type, mode", [(_UPPER, "up"), (_LOWER, "down")])
def test_discrete_rounding_moves_every_atom_to_the_point_on_its_side(
    bound_type: BoundType, mode: str
):
    """Atoms inside the points go up or down; the tail on that side goes to the boundary."""
    first, last = float(_LOSS_POINTS[0]), float(_LOSS_POINTS[-1])
    inside = _atoms()
    scale = 0.9 / (1.0 - inside.p_max)
    x_array = np.concatenate(([first - 1.0], inside.x_array, [last + 1.0]))
    prob_arr = np.concatenate(([0.04], inside.prob_arr * scale, [0.06]))
    source = SparseDiscreteDist(x_array=x_array, prob_arr=prob_arr)

    projected = project_discrete_dist_stoch_dom_on_points(
        dist=source, loss_points=_LOSS_POINTS, bound_type=bound_type
    )

    expected = project_onto_points(
        values=inside.x_array, masses=inside.prob_arr * scale, points=_LOSS_POINTS, mode=mode
    )
    if bound_type == _UPPER:
        expected[0] += 0.04
        boundaries = (0.0, 0.06)
    else:
        expected[-1] += 0.06
        boundaries = (0.04, 0.0)
    np.testing.assert_allclose(projected.prob_arr, expected, rtol=0.0, atol=1e-15)
    assert (projected.p_min, projected.p_max) == pytest.approx(boundaries, abs=1e-15)


@pytest.mark.parametrize("bound_type", _BOUNDS)
def test_continuous_rounding_moves_every_cell_to_the_point_on_its_side(bound_type: BoundType):
    """Each cell probability of the law sits on its upper or its lower point."""
    law = gaussian_loss_law(_SIGMA)
    cells = np.diff(law.cdf(_LOSS_POINTS))
    below, above = float(law.cdf(_LOSS_POINTS[0])), float(law.sf(_LOSS_POINTS[-1]))

    projected = project_continuous_dist_stoch_dom_on_points(
        dist=law, loss_points=_LOSS_POINTS, bound_type=bound_type
    )

    if bound_type == _UPPER:
        expected, boundaries = np.concatenate(([below], cells)), (0.0, above)
    else:
        expected, boundaries = np.concatenate((cells, [above])), (below, 0.0)
    np.testing.assert_allclose(projected.prob_arr, expected, rtol=1e-9, atol=1e-15)
    assert (projected.p_min, projected.p_max) == pytest.approx(boundaries, rel=1e-9)


# =============================================================================
# Validation
# =============================================================================


_ONE_POINT = np.array([0.0])
_TWO_DIMENSIONAL_POINTS = np.array([[0.0, 1.0]])
_REPEATED_POINTS = np.array([0.0, 0.0, 1.0])
_DECREASING_POINTS = np.array([1.0, 0.0])
_INFINITE_POINTS = np.array([0.0, np.inf])
_INVALID_LOSS_POINTS = [
    pytest.param(_ONE_POINT, "at least two", id="one-point"),
    pytest.param(_TWO_DIMENSIONAL_POINTS, "one-dimensional", id="two-dimensional"),
    pytest.param(_REPEATED_POINTS, "strictly increasing", id="repeated"),
    pytest.param(_DECREASING_POINTS, "strictly increasing", id="decreasing"),
    pytest.param(_INFINITE_POINTS, "finite", id="infinite"),
]


@pytest.mark.parametrize("loss_points, message", _INVALID_LOSS_POINTS)
def test_loss_points_must_be_finite_and_strictly_increasing(loss_points, message: str):
    """All four functions reject points that do not delimit cells."""
    law = gaussian_loss_law(_SIGMA)
    source = _atoms()
    with pytest.raises(ValueError, match=message):
        project_continuous_dist_ctd_on_points(dist=law, dual_dist=law, loss_points=loss_points)
    with pytest.raises(ValueError, match=message):
        project_discrete_dist_ctd_on_points(dist=source, loss_points=loss_points)
    with pytest.raises(ValueError, match=message):
        project_continuous_dist_stoch_dom_on_points(
            dist=law, loss_points=loss_points, bound_type=_UPPER
        )
    with pytest.raises(ValueError, match=message):
        project_discrete_dist_stoch_dom_on_points(
            dist=source, loss_points=loss_points, bound_type=_UPPER
        )
