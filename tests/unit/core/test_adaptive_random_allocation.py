"""Regression tests for adaptive bound refinement."""

import pytest

from PLD_accounting import PrivacyParams
from PLD_accounting import adaptive_random_allocation as adaptive


class _FixedEpsilonPLD:  # pylint: disable=too-few-public-methods
    """Stand-in PLD whose epsilon does not depend on delta or on refinement."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def get_epsilon_for_delta(self, _delta: float) -> float:
        """Return the fixed epsilon regardless of the requested delta."""
        return self.epsilon


def test_adaptive_refinement_rejects_crossed_raw_bounds(monkeypatch):
    """A lower estimate above the upper estimate is diagnosed, never clamped."""
    monkeypatch.setattr(
        adaptive,
        "_build_pld_pair",
        lambda **_kwargs: (_FixedEpsilonPLD(1.0), _FixedEpsilonPLD(1.1)),
    )

    with pytest.raises(RuntimeError, match="dominating bound 1.*below dominated bound 1.1"):
        adaptive.optimize_allocation_epsilon_range(
            params=PrivacyParams(sigma=1.0, num_steps=2, delta=1e-5),
            target_accuracy=0.1,
            initial_discretization=0.1,
            initial_tail_truncation=1e-8,
            pld_builder=lambda **_kwargs: None,  # replaced by the pair stub above
        )
