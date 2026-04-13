"""Tests for combining distributions on a union grid."""

import math

import numpy as np
from PLD_accounting.discrete_dist import SparseDiscreteDist
from PLD_accounting.types import BoundType
from PLD_accounting.utils import _align_distributions_to_union_grid, combine_distributions

from tests.test_tolerances import TestTolerances as TOL


def _make_dist(x_values, probs, p_max=0.0, p_min=0.0):
    return SparseDiscreteDist(
        x_array=np.array(x_values, dtype=np.float64),
        prob_arr=np.array(probs, dtype=np.float64),
        p_min=p_min,
        p_max=p_max,
    )


def test_combine_distributions_aligns_grids_dominate():
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_max=0.1)
    dist_2 = _make_dist([0.5, 1.5], [0.3, 0.6], p_max=0.1)

    combined = combine_distributions(dist_1=dist_1, dist_2=dist_2, bound_type=BoundType.DOMINATES)

    expected_grid = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    assert np.allclose(combined.x_array, expected_grid)
    assert combined.p_min == 0.0
    assert np.all(combined.prob_arr >= -TOL.PMF_NONNEGATIVE_SLACK)


def test_align_distributions_to_union_grid_preserves_mass():
    dist_1 = _make_dist([0.0, 2.0], [0.2, 0.7], p_max=0.1)
    dist_2 = _make_dist([1.0, 3.0], [0.3, 0.6], p_max=0.1)

    aligned_1, aligned_2 = _align_distributions_to_union_grid(
        dist_1=dist_1,
        dist_2=dist_2,
    )
    expected_grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    assert np.allclose(aligned_1.x_array, expected_grid)
    assert np.allclose(aligned_2.x_array, expected_grid)

    assert np.isclose(math.fsum([*map(float, aligned_1.prob_arr), aligned_1.p_max]), 1.0)
    assert np.isclose(math.fsum([*map(float, aligned_2.prob_arr), aligned_2.p_max]), 1.0)


def test_combine_distributions_aligns_grids_is_dominated():
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_min=0.1)
    dist_2 = _make_dist([0.5, 1.5], [0.3, 0.6], p_min=0.1)

    combined = combine_distributions(
        dist_1=dist_1, dist_2=dist_2, bound_type=BoundType.IS_DOMINATED
    )

    expected_grid = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    assert np.allclose(combined.x_array, expected_grid)
    assert combined.p_max == 0.0
    assert np.all(combined.prob_arr >= -TOL.PMF_NONNEGATIVE_SLACK)
