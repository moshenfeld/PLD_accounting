"""Tests for combining distributions on a shared grid."""

import math

import numpy as np
import pytest

from PLD_accounting.discrete_dist import DenseDiscreteDist, GridSpec
from PLD_accounting.types import BoundType
from PLD_accounting.utils import combine_best_of_two_plds, combine_distributions
from tests.test_tolerances import TestTolerances as TOL


def test_combine_distributions_rejects_mismatched_grids():
    """Mismatched support grids are rejected; callers must project first."""
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_max=0.1)
    dist_2 = _make_dist([0.5, 1.5], [0.3, 0.6], p_max=0.1)

    with pytest.raises(ValueError, match="identical support grids"):
        combine_distributions(dist_1=dist_1, dist_2=dist_2, bound_type=BoundType.DOMINATES)


def test_combine_distributions_rejects_close_but_nonidentical_grids() -> None:
    """Support identity is exact and never inferred from ``allclose``."""
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_max=0.1)
    dist_2 = _make_dist([5e-13, 1.0000000000005], [0.4, 0.5], p_max=0.1)

    with pytest.raises(ValueError, match="identical support grids"):
        combine_distributions(dist_1=dist_1, dist_2=dist_2, bound_type=BoundType.DOMINATES)


def test_combine_distributions_same_grid_dominate():
    """DOMINATES combination on a shared grid keeps the grid and valid mass."""
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_max=0.1)
    dist_2 = _make_dist([0.0, 1.0], [0.3, 0.6], p_max=0.1)

    combined = combine_distributions(dist_1=dist_1, dist_2=dist_2, bound_type=BoundType.DOMINATES)

    expected_grid = np.array([0.0, 1.0], dtype=np.float64)
    assert np.allclose(combined.x_array, expected_grid)
    assert combined.p_min == 0.0
    assert np.all(combined.prob_arr >= -TOL.PMF_NONNEGATIVE_SLACK)


def test_combine_distributions_same_grid_is_dominated():
    """IS_DOMINATED combination on a shared grid keeps the grid and valid mass."""
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_min=0.1)
    dist_2 = _make_dist([0.0, 1.0], [0.3, 0.6], p_min=0.1)

    combined = combine_distributions(
        dist_1=dist_1, dist_2=dist_2, bound_type=BoundType.IS_DOMINATED
    )

    expected_grid = np.array([0.0, 1.0], dtype=np.float64)
    assert np.allclose(combined.x_array, expected_grid)
    assert combined.p_max == 0.0
    assert np.all(combined.prob_arr >= -TOL.PMF_NONNEGATIVE_SLACK)


def test_combine_distributions_dominates_takes_exact_min_p_max():
    """DOMINATES combination keeps the exact CCDF-min infinity atom (the min)."""
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_max=0.1)
    dist_2 = _make_dist([0.0, 1.0], [0.35, 0.6], p_max=0.05)

    combined = combine_distributions(dist_1=dist_1, dist_2=dist_2, bound_type=BoundType.DOMINATES)

    assert np.isclose(combined.p_max, 0.05)
    total = math.fsum([combined.p_min, *map(float, combined.prob_arr), combined.p_max])
    assert np.isclose(total, 1.0)


def test_combine_distributions_is_dominated_takes_exact_min_p_min():
    """IS_DOMINATED combination keeps the exact CCDF-max minus-infinity atom (the min)."""
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_min=0.1)
    dist_2 = _make_dist([0.0, 1.0], [0.35, 0.6], p_min=0.05)

    combined = combine_distributions(
        dist_1=dist_1, dist_2=dist_2, bound_type=BoundType.IS_DOMINATED
    )

    assert np.isclose(combined.p_min, 0.05)
    total = math.fsum([combined.p_min, *map(float, combined.prob_arr), combined.p_max])
    assert np.isclose(total, 1.0)


@pytest.mark.parametrize("invalid_input", [0, 1])
@pytest.mark.parametrize(
    ("bound_type", "invalid_boundary", "message"),
    [
        (BoundType.DOMINATES, "p_min", "canonical dominating inputs"),
        (BoundType.IS_DOMINATED, "p_max", "canonical dominated inputs"),
    ],
)
def test_combine_best_rejects_noncanonical_boundary_on_either_input(
    invalid_input: int,
    bound_type: BoundType,
    invalid_boundary: str,
    message: str,
) -> None:
    """Grid-anchor selection cannot decide whether an invalid atom is folded."""
    canonical_boundary = "p_max" if bound_type == BoundType.DOMINATES else "p_min"
    boundaries = [
        {"p_min": 0.0, "p_max": 0.0},
        {"p_min": 0.0, "p_max": 0.0},
    ]
    for values in boundaries:
        values[canonical_boundary] = 0.1
    boundaries[invalid_input][canonical_boundary] = 0.0
    boundaries[invalid_input][invalid_boundary] = 0.1
    dists = [
        _make_dist([0.0, 1.0], [0.45, 0.45], **boundaries[0]),
        _make_dist([0.0, 0.5, 1.0], [0.3, 0.3, 0.3], **boundaries[1]),
    ]

    with pytest.raises(ValueError, match=message):
        combine_best_of_two_plds(
            dist_1=dists[0],
            dist_2=dists[1],
            bound_type=bound_type,
        )


def _make_dist(x_values, probs, p_max=0.0, p_min=0.0):
    x_array = np.array(x_values, dtype=np.float64)
    return DenseDiscreteDist(
        grid=GridSpec(
            step=float(x_array[1] - x_array[0]),
            n=(np.array(probs, dtype=np.float64)).size,
            anchor=float(x_array[0]),
        ),
        prob_arr=np.array(probs, dtype=np.float64),
        p_min=p_min,
        p_max=p_max,
    )
