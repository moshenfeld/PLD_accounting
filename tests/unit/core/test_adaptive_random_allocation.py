"""Regression tests for adaptive bound refinement."""

from __future__ import annotations

import numpy as np
import pytest

from PLD_accounting import AllocationSchemeConfig, PrivacyParams
from PLD_accounting import adaptive_random_allocation as adaptive
from PLD_accounting.types import BoundType


class _FixedEpsilonPLD:  # pylint: disable=too-few-public-methods
    """Stand-in PLD whose epsilon does not depend on delta or on refinement."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def get_epsilon_for_delta(self, _delta: float) -> float:
        """Return the fixed epsilon regardless of the requested delta."""
        return self.epsilon


def test_adaptive_refinement_rejects_crossed_raw_bounds():
    """A lower estimate above the upper estimate is diagnosed, never clamped."""

    def fake_builder(*, params, config, bound_type):
        del params, config
        if bound_type == BoundType.DOMINATES:
            return _FixedEpsilonPLD(1.0)
        return _FixedEpsilonPLD(1.1)

    with pytest.raises(RuntimeError, match="dominating bound 1.*below dominated bound 1.1"):
        adaptive.optimize_allocation_epsilon_range(
            params=PrivacyParams(sigma=1.0, num_steps=2, delta=1e-5),
            target_accuracy=0.1,
            initial_discretization=0.1,
            initial_tail_truncation=1e-8,
            pld_builder=fake_builder,
        )


def test_adaptive_negative_target_stops_on_relative_ratio():
    """A negative target_accuracy stops when upper/lower is within the default ratio."""

    def fake_builder(*, params, config, bound_type):
        del params, config
        if bound_type == BoundType.DOMINATES:
            return _FixedEpsilonPLD(1.05)
        return _FixedEpsilonPLD(1.0)

    result = adaptive.optimize_allocation_epsilon_range(
        params=PrivacyParams(sigma=1.0, num_steps=2, delta=1e-5),
        target_accuracy=-1.0,
        initial_discretization=0.1,
        initial_tail_truncation=1e-8,
        pld_builder=fake_builder,
    )

    assert result.converged
    assert result.iterations == 1
    assert result.target_accuracy == -1.0
    assert result.upper_bound / result.lower_bound <= 1.0 + adaptive.DEFAULT_RELATIVE_ACCURACY


def test_adaptive_reports_last_evaluated_config_on_max_iterations(monkeypatch):
    """Exhausting the iteration cap returns the last evaluated pair, not a further step."""
    monkeypatch.setattr(adaptive, "MAX_ITERATIONS", 3)
    configs: list[AllocationSchemeConfig] = []

    def fake_builder(*, params, config, bound_type):
        del params
        configs.append(config)
        if bound_type == BoundType.DOMINATES:
            return _FixedEpsilonPLD(1.0)
        return _FixedEpsilonPLD(0.0)

    with pytest.warns(RuntimeWarning, match="did not converge after 3 evaluations"):
        result = adaptive.optimize_allocation_epsilon_range(
            params=PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6),
            target_accuracy=1e-12,
            initial_discretization=0.1,
            initial_tail_truncation=1e-4,
            pld_builder=fake_builder,
        )

    last_config = configs[-1]
    assert not result.converged
    assert result.iterations == 3
    assert np.isclose(result.discretization, last_config.loss_discretization)
    assert np.isclose(result.tail_truncation, last_config.tail_truncation)
    assert np.isclose(result.discretization, 0.1 / 4.0)
    assert np.isclose(result.tail_truncation, 1e-6)


def test_adaptive_reports_last_evaluated_config_on_no_change_exit():
    """A clamped no-change exit reports the one pair it evaluated, not the step it skipped."""
    configs: list[AllocationSchemeConfig] = []

    def fake_builder(*, params, config, bound_type):
        del params
        configs.append(config)
        if bound_type == BoundType.DOMINATES:
            return _FixedEpsilonPLD(1.0)
        return _FixedEpsilonPLD(0.0)

    with pytest.warns(RuntimeWarning, match="did not converge after 1 evaluations"):
        result = adaptive.optimize_allocation_epsilon_range(
            params=PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6),
            target_accuracy=1e-12,
            initial_discretization=adaptive.MIN_DISCRETIZATION,
            initial_tail_truncation=adaptive.MIN_TAIL_TRUNCATION,
            pld_builder=fake_builder,
        )

    assert not result.converged
    # One pair, so two builder calls: the skipped refinement must not be counted.
    assert len(configs) == 2
    assert result.iterations == 1
    assert np.isclose(result.discretization, adaptive.MIN_DISCRETIZATION)
    assert np.isclose(result.tail_truncation, adaptive.MIN_TAIL_TRUNCATION)
