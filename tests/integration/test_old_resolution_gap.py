"""Gap checks for the fixed-resolution epsilon implementation."""

import pytest

from PLD_accounting import AllocationSchemeConfig, PrivacyParams
from PLD_accounting.random_allocation_api import gaussian_allocation_epsilon_extended
from PLD_accounting.types import BoundType

TARGET_FACTOR = 2.0


@pytest.mark.parametrize(
    ("sigma", "delta", "num_steps", "num_selected", "desired_resolution"),
    [
        (0.8, 1e-4, 20, 1, 0.1),
        (1.0, 1e-6, 100, 1, 0.05),
        (1.5, 1e-4, 20, 5, 0.05),
        (2.0, 1e-6, 20, 1, 0.02),
        (2.0, 1e-6, 100, 5, 0.05),
        (3.0, 1e-6, 20, 1, 0.05),
    ],
)
def test_old_epsilon_gap_is_positive_and_below_target(
    sigma: float,
    delta: float,
    num_steps: int,
    num_selected: int,
    desired_resolution: float,
) -> None:
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        delta=delta,
    )
    config = AllocationSchemeConfig(
        loss_discretization=desired_resolution,
        tail_truncation=min(delta / 10.0, 1e-7),
    )

    dominates = gaussian_allocation_epsilon_extended(
        params=params,
        config=config,
        bound_type=BoundType.DOMINATES,
    )
    is_dominated = gaussian_allocation_epsilon_extended(
        params=params,
        config=config,
        bound_type=BoundType.IS_DOMINATED,
    )

    gap = dominates - is_dominated

    assert gap > 0.0
    assert gap < TARGET_FACTOR * desired_resolution
