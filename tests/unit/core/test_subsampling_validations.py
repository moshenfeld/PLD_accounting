import math
import numpy as np
import pytest

from PLD_accounting.discrete_dist import GeneralDiscreteDist, PLDRealization
from PLD_accounting.subsample_PLD import (
    _stable_subsampling_transformation,
    _calc_subsampled_grid,
    subsample_PMF,
)
from PLD_accounting.types import Direction


def _simple_remove_dist() -> PLDRealization:
    return PLDRealization(
        x_min=0.0,
        x_gap=0.5,
        PMF_array=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
    )


def test_subsample_pmf_rejects_invalid_sampling_probability():
    dist = _simple_remove_dist()
    with pytest.raises(ValueError, match="sampling_prob must be in"):
        subsample_PMF(
            base_pld=dist,
            sampling_prob=0.0,
            direction=Direction.REMOVE,
        )
    with pytest.raises(ValueError, match="sampling_prob must be in"):
        subsample_PMF(
            base_pld=dist,
            sampling_prob=1.1,
            direction=Direction.REMOVE,
        )


def test_subsample_pmf_rejects_direction_both():
    dist = _simple_remove_dist()
    with pytest.raises(ValueError, match="Direction BOTH is invalid"):
        subsample_PMF(
            base_pld=dist,
            sampling_prob=0.5,
            direction=Direction.BOTH,
        )


def test_subsample_pmf_rejects_non_realization_input():
    dist = GeneralDiscreteDist(
        x_array=np.array([-1.0, -0.5, 0.0, 0.5], dtype=np.float64),
        PMF_array=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
    )
    with pytest.raises(TypeError, match="requires PLDRealization"):
        subsample_PMF(
            base_pld=dist,
            sampling_prob=0.5,
            direction=Direction.REMOVE,
        )


def test_subsample_pmf_q1_returns_input_distribution():
    dist = _simple_remove_dist()
    result = subsample_PMF(
        base_pld=dist,
        sampling_prob=1.0,
        direction=Direction.REMOVE,
    )
    assert result is dist


def test_subsample_pmf_returns_valid_pld_realization_remove():
    dist = _simple_remove_dist()
    result = subsample_PMF(
        base_pld=dist,
        sampling_prob=0.3,
        direction=Direction.REMOVE,
    )
    assert isinstance(result, PLDRealization)
    result.validate_pld_realization()


def test_subsample_pmf_returns_valid_pld_realization_add():
    dist = PLDRealization(
        x_min=0.0,
        x_gap=0.25,
        PMF_array=np.array([0.24, 0.2, 0.18, 0.16, 0.14], dtype=np.float64),
        p_loss_inf=0.08,
    )
    result = subsample_PMF(
        base_pld=dist,
        sampling_prob=0.3,
        direction=Direction.ADD,
    )
    assert isinstance(result, PLDRealization)
    result.validate_pld_realization()


def test_subsample_pmf_add_places_positive_infinity_mass_at_max_add_loss():
    # Regression case: add-direction +inf mass must map to -log(1-q), not to the
    # rightmost transformed finite bin when that bin is below the cap.
    q = 0.3708686650516492
    dist = PLDRealization(
        x_min=-6.305102226697834,
        x_gap=0.02532892132180583,
        PMF_array=np.full(44, 0.0018884873605871 / 44.0, dtype=np.float64),
        p_loss_inf=0.998111512639413,
    )

    result = subsample_PMF(
        base_pld=dist,
        sampling_prob=q,
        direction=Direction.ADD,
    )

    max_add_loss = -math.log1p(-q)
    assert result.x_array[-1] >= max_add_loss
    result.validate_pld_realization()


def test_calc_subsampled_grid_rejects_invalid_bucket_count():
    with pytest.raises(ValueError, match="num_buckets must be >= 2"):
        _calc_subsampled_grid(
            lower_loss=0.0,
            discretization=0.1,
            num_buckets=1,
            grid_size=0.5,
            direction=Direction.REMOVE,
        )


def test_calc_subsampled_grid_rejects_invalid_grid_size():
    with pytest.raises(ValueError, match="grid_size must be in"):
        _calc_subsampled_grid(
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
