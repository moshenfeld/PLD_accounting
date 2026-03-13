"""Integration regressions for general-allocation composition semantics."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest
from scipy import stats

from PLD_accounting import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    PrivacyParams,
    gaussian_allocation_PLD,
    general_allocation_PLD,
)
from PLD_accounting.discrete_dist import LinearDiscreteDist, PLDRealization


@lru_cache(maxsize=None)
def _gaussian_realization(sigma: float, num_points: int = 2200, num_std: float = 8.0) -> PLDRealization:
    """Build a reusable Gaussian PLD realization on a linear loss grid."""
    sigma_loss = 1.0 / sigma
    mean = 1.0 / (2.0 * sigma * sigma)
    loss_values = np.linspace(
        mean - num_std * sigma_loss,
        mean + num_std * sigma_loss,
        num_points,
    )
    loss_gap = loss_values[1] - loss_values[0]
    probabilities = stats.norm.pdf(loss_values, loc=mean, scale=sigma_loss) * loss_gap
    probabilities = probabilities / probabilities.sum()
    dist = LinearDiscreteDist.from_x_array(
        x_array=loss_values,
        PMF_array=probabilities,
    )
    return PLDRealization(
        x_min=dist.x_min,
        x_gap=dist.x_gap,
        PMF_array=dist.PMF_array,
        p_loss_inf=dist.p_pos_inf,
        p_loss_neg_inf=dist.p_neg_inf,
    )


def _comparison_config() -> AllocationSchemeConfig:
    return AllocationSchemeConfig(
        loss_discretization=5e-3,
        tail_truncation=1e-10,
        max_grid_FFT=200_000,
        max_grid_mult=30_000,
        convolution_method=ConvolutionMethod.GEOM,
    )


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_general_allocation_floor_semantics_invariant_for_non_divisible_steps(bound_type: BoundType):
    """Regression: num_steps contributes only through floor(num_steps / num_selected)."""
    config = _comparison_config()
    realization = _gaussian_realization(2.0)
    delta = 1e-5

    pld_steps_6 = general_allocation_PLD(
        num_steps=6,
        num_selected=2,
        num_epochs=2,
        remove_realization=realization,
        add_realization=realization,
        config=config,
        bound_type=bound_type,
    )
    pld_steps_7 = general_allocation_PLD(
        num_steps=7,
        num_selected=2,
        num_epochs=2,
        remove_realization=realization,
        add_realization=realization,
        config=config,
        bound_type=bound_type,
    )

    epsilon_6 = float(pld_steps_6.get_epsilon_for_delta(delta))
    epsilon_7 = float(pld_steps_7.get_epsilon_for_delta(delta))
    assert abs(epsilon_6 - epsilon_7) < 1e-12


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
@pytest.mark.parametrize(
    ("sigma", "num_steps", "num_selected", "num_epochs", "delta", "epsilon_tol"),
    [
        (2.0, 6, 2, 1, 1e-5, 2e-2),
        (2.2, 7, 2, 2, 1e-5, 4e-2),
        (2.8, 15, 5, 1, 1e-6, 6e-2),
    ],
)
def test_general_allocation_composition_semantics_num_selected_gt_1(
    sigma: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    delta: float,
    epsilon_tol: float,
    bound_type: BoundType,
):
    """General realization path should match Gaussian two-stage semantics."""
    config = _comparison_config()
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        delta=delta,
    )

    gaussian_pld = gaussian_allocation_PLD(
        params=params,
        config=config,
        bound_type=bound_type,
    )

    realization = _gaussian_realization(sigma)
    realization_pld = general_allocation_PLD(
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        remove_realization=realization,
        add_realization=realization,
        config=config,
        bound_type=bound_type,
    )

    epsilon_gaussian = float(gaussian_pld.get_epsilon_for_delta(delta))
    epsilon_realization = float(realization_pld.get_epsilon_for_delta(delta))

    assert np.isfinite(epsilon_gaussian)
    assert np.isfinite(epsilon_realization)
    assert abs(epsilon_gaussian - epsilon_realization) < epsilon_tol
