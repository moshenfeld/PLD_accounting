import numpy as np

from PLD_accounting.discrete_dist import GeneralDiscreteDist
from PLD_accounting.types import BoundType
from PLD_accounting.utils import combine_distributions


def _make_dist(x_values, probs, p_pos_inf=0.0, p_neg_inf=0.0):
    return GeneralDiscreteDist(
        x_array=np.array(x_values, dtype=np.float64),
        PMF_array=np.array(probs, dtype=np.float64),
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf
    )


def test_combine_distributions_aligns_grids_dominate():
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_pos_inf=0.1)
    dist_2 = _make_dist([0.5, 1.5], [0.3, 0.6], p_pos_inf=0.1)

    combined = combine_distributions(dist_1, dist_2, bound_type=BoundType.DOMINATES)

    expected_grid = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    assert np.allclose(combined.x_array, expected_grid)
    combined.validate_mass_conservation(BoundType.DOMINATES)
    assert combined.p_neg_inf == 0.0
    assert np.all(combined.PMF_array >= -1e-15)


def test_align_distributions_to_union_grid_preserves_mass():
    from PLD_accounting.utils import _align_distributions_to_union_grid

    dist_1 = _make_dist([0.0, 2.0], [0.2, 0.7], p_pos_inf=0.1)
    dist_2 = _make_dist([1.0, 3.0], [0.3, 0.6], p_pos_inf=0.1)

    aligned_1, aligned_2 = _align_distributions_to_union_grid(dist_1, dist_2)
    expected_grid = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    assert np.allclose(aligned_1.x_array, expected_grid)
    assert np.allclose(aligned_2.x_array, expected_grid)

    assert np.isclose(np.sum(aligned_1.PMF_array) + aligned_1.p_pos_inf, 1.0)
    assert np.isclose(np.sum(aligned_2.PMF_array) + aligned_2.p_pos_inf, 1.0)


def test_combine_distributions_aligns_grids_is_dominated():
    dist_1 = _make_dist([0.0, 1.0], [0.4, 0.5], p_neg_inf=0.1)
    dist_2 = _make_dist([0.5, 1.5], [0.3, 0.6], p_neg_inf=0.1)

    combined = combine_distributions(dist_1, dist_2, bound_type=BoundType.IS_DOMINATED)

    expected_grid = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    assert np.allclose(combined.x_array, expected_grid)
    combined.validate_mass_conservation(BoundType.IS_DOMINATED)
    assert combined.p_pos_inf == 0.0
    assert np.all(combined.PMF_array >= -1e-15)
