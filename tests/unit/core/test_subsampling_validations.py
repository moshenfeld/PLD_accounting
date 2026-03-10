import numpy as np
import pytest

from PLD_accounting.discrete_dist import GeneralDiscreteDist
from PLD_accounting.subsample_PLD import (
    _stable_subsampling_transformation,
    calc_subsampled_grid,
    subsample_PMF,
)
from PLD_accounting.types import BoundType, Direction


def _simple_remove_dist() -> GeneralDiscreteDist:
    return GeneralDiscreteDist(
        x_array=np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float64),
        PMF_array=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        p_neg_inf=0.0,
        p_pos_inf=0.0,
    )


def test_subsample_pmf_rejects_invalid_sampling_probability():
    dist = _simple_remove_dist()
    with pytest.raises(ValueError, match="sampling_prob must be in"):
        subsample_PMF(
            base_pld=dist,
            sampling_prob=0.0,
            direction=Direction.REMOVE,
            bound_type=BoundType.DOMINATES,
        )
    with pytest.raises(ValueError, match="sampling_prob must be in"):
        subsample_PMF(
            base_pld=dist,
            sampling_prob=1.1,
            direction=Direction.REMOVE,
            bound_type=BoundType.DOMINATES,
        )


def test_subsample_pmf_rejects_direction_both():
    dist = _simple_remove_dist()
    with pytest.raises(ValueError, match="Direction BOTH is invalid"):
        subsample_PMF(
            base_pld=dist,
            sampling_prob=0.5,
            direction=Direction.BOTH,
            bound_type=BoundType.DOMINATES,
        )


def test_subsample_pmf_q1_returns_input_distribution():
    dist = _simple_remove_dist()
    result = subsample_PMF(
        base_pld=dist,
        sampling_prob=1.0,
        direction=Direction.REMOVE,
        bound_type=BoundType.DOMINATES,
    )
    assert result is dist


def test_calc_subsampled_grid_rejects_invalid_bucket_count():
    with pytest.raises(ValueError, match="num_buckets must be >= 2"):
        calc_subsampled_grid(
            lower_loss=0.0,
            discretization=0.1,
            num_buckets=1,
            grid_size=0.5,
            direction=Direction.REMOVE,
        )


def test_calc_subsampled_grid_rejects_invalid_grid_size():
    with pytest.raises(ValueError, match="grid_size must be in"):
        calc_subsampled_grid(
            lower_loss=0.0,
            discretization=0.1,
            num_buckets=10,
            grid_size=0.0,
            direction=Direction.REMOVE,
        )


def test_stable_subsampling_transformation_handles_extreme_losses():
    losses = np.array([-750.0, -100.0, -1.0, 0.0, 1.0, 100.0, 750.0], dtype=np.float64)
    transformed_remove = _stable_subsampling_transformation(
        x_array=losses,
        sampling_prob=0.01,
        direction=Direction.REMOVE,
    )
    transformed_add = _stable_subsampling_transformation(
        x_array=losses,
        sampling_prob=0.01,
        direction=Direction.ADD,
    )

    assert np.all(np.isfinite(transformed_remove))
    assert np.all(np.isfinite(transformed_add))
    assert np.all(np.diff(transformed_remove) >= 0.0)
    assert np.all(np.diff(transformed_add) >= 0.0)
