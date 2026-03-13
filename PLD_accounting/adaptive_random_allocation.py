"""
Adaptive helpers for random-allocation queries.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np

from PLD_accounting.types import AllocationSchemeConfig, BoundType, ConvolutionMethod, Direction, PrivacyParams
from dp_accounting.pld import privacy_loss_distribution


# Global configuration constants
MAX_ITERATIONS = 10
POISSON_GUESS_DISCRETIZATION = 1e-4
MIN_DISCRETIZATION = 1e-6
MAX_DISCRETIZATION = 1e-1
MIN_TAIL_TRUNCATION = 1e-20
MAX_TAIL_TRUNCATION = 1e-4


@dataclass
class AdaptiveResult:
    """
    Result from adaptive allocation computation.

    Attributes:
        upper_bound: Best upper bound found across all iterations.
        lower_bound: Best lower bound found across all iterations.
        absolute_gap: Final gap between upper and lower bounds.
        converged: Whether the algorithm converged to target accuracy.
        iterations: Number of refinement iterations performed.
        initial_discretization: Starting loss_discretization value.
        discretization: Final loss_discretization value used.
        initial_tail_truncation: Starting tail_truncation value.
        tail_truncation: Final tail_truncation value used.
        target_accuracy: Target absolute gap for convergence.
    """
    upper_bound: float
    lower_bound: float
    absolute_gap: float
    converged: bool
    iterations: int
    initial_discretization: float
    discretization: float
    initial_tail_truncation: float
    tail_truncation: float
    target_accuracy: float

# =============================================================================
# Public Adaptive API
# =============================================================================

def optimize_allocation_epsilon_range(*,
    params: PrivacyParams,
    target_accuracy: float,
    pld_builder: Callable[..., privacy_loss_distribution.PrivacyLossDistribution],
    initial_discretization: float | None = None,
    initial_tail_truncation: float | None = None,
) -> AdaptiveResult:
    """Fixed-schedule adaptive refinement for epsilon bounds."""
    if params.delta is None:
        raise ValueError("optimize_allocation_epsilon_range requires params.delta")
    delta = params.delta

    estimated_epsilon = None
    if target_accuracy < 0.0:
        estimated_epsilon = estimate_poisson_query(
            params=params,
            query_func=lambda pld: float(pld.get_epsilon_for_delta(delta)),
        )
    target_accuracy, auto_target_accuracy = _auto_target_accuracy(
        target_accuracy=target_accuracy,
        estimated_value=estimated_epsilon if estimated_epsilon is not None else target_accuracy,
    )
    if initial_discretization is None:
        initial_discretization = target_accuracy / 2
    if initial_tail_truncation is None:
        initial_tail_truncation = 0.1 * delta

    discretization = _clip_discretization(initial_discretization)
    tail_truncation = _clip_tail_truncation(initial_tail_truncation)
    effective_initial_discretization = discretization
    effective_initial_tail_truncation = tail_truncation

    upper_bound = np.inf
    lower_bound = -np.inf
    converged = False
    iteration = 0

    while not converged and iteration < MAX_ITERATIONS:
        iteration += 1

        config = AllocationSchemeConfig(
            loss_discretization=discretization,
            tail_truncation=tail_truncation,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld_upper, pld_lower = _build_pld_pair(
            params=params,
            config=config,
            pld_builder=pld_builder,
        )
        new_upper = float(pld_upper.get_epsilon_for_delta(delta))
        new_lower = min(
            float(pld_lower.get_epsilon_for_delta(delta - tail_truncation)),
            new_upper,
        )

        if new_upper < new_lower:
            raise RuntimeError(
                "Adaptive refinement produced invalid bounds: dominating bound "
                f"{new_upper:.12g} is below dominated bound {new_lower:.12g}"
            )

        upper_bound = min(upper_bound, new_upper)
        lower_bound = max(lower_bound, new_lower)

        if upper_bound < lower_bound:
            raise RuntimeError(
                "Adaptive refinement produced invalid bounds: dominating bound "
                f"{upper_bound:.12g} is below dominated bound {lower_bound:.12g}"
            )

        if auto_target_accuracy:
            target_accuracy = max(target_accuracy, 0.10 * lower_bound)

        gap = upper_bound - lower_bound
        if gap < target_accuracy:
            converged = True
            break

        discretization, tail_truncation, changed = _apply_refinement_step(
            discretization=discretization,
            tail_truncation=tail_truncation,
        )
        if not changed:
            break

    if not converged:
        warnings.warn(
            f"Adaptive refinement did not converge after {MAX_ITERATIONS} iterations. "
            f"Final gap: {upper_bound - lower_bound:.6e}, target: {target_accuracy:.6e}. "
            f"Returning best bounds found.",
            RuntimeWarning,
        )

    return AdaptiveResult(
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        absolute_gap=upper_bound - lower_bound,
        converged=converged,
        iterations=iteration,
        initial_discretization=effective_initial_discretization,
        discretization=discretization,
        initial_tail_truncation=effective_initial_tail_truncation,
        tail_truncation=tail_truncation,
        target_accuracy=target_accuracy,
    )


def optimize_allocation_delta_range(*,
    params: PrivacyParams,
    target_accuracy: float,
    pld_builder: Callable[..., privacy_loss_distribution.PrivacyLossDistribution],
    initial_discretization: float | None = None,
    initial_tail_truncation: float | None = None,
) -> AdaptiveResult:
    """Fixed-schedule adaptive refinement for delta bounds."""
    if params.epsilon is None:
        raise ValueError("optimize_allocation_delta_range requires params.epsilon")
    epsilon = params.epsilon

    estimated_delta = None
    if target_accuracy < 0.0:
        estimated_delta = estimate_poisson_query(
            params=params,
            query_func=lambda pld: float(pld.get_delta_for_epsilon(epsilon)),
        )
    target_accuracy, auto_target_accuracy = _auto_target_accuracy(
        target_accuracy=target_accuracy,
        estimated_value=estimated_delta if estimated_delta is not None else target_accuracy,
    )
    if initial_discretization is None:
        initial_discretization = 0.1 * epsilon
    if initial_tail_truncation is None:
        initial_tail_truncation = target_accuracy

    discretization = _clip_discretization(initial_discretization)
    tail_truncation = _clip_tail_truncation(initial_tail_truncation)
    effective_initial_discretization = discretization
    effective_initial_tail_truncation = tail_truncation

    upper_bound = np.inf
    lower_bound = -np.inf
    converged = False
    iteration = 0

    while not converged and iteration < MAX_ITERATIONS:
        iteration += 1

        config = AllocationSchemeConfig(
            loss_discretization=discretization,
            tail_truncation=tail_truncation,
            convolution_method=ConvolutionMethod.GEOM,
        )

        pld_upper, pld_lower = _build_pld_pair(
            params=params,
            config=config,
            pld_builder=pld_builder,
        )
        new_upper = float(pld_upper.get_delta_for_epsilon(epsilon))
        new_lower = min(
            float(pld_lower.get_delta_for_epsilon(epsilon - discretization)),
            new_upper,
        )

        if new_upper < new_lower:
            raise RuntimeError(
                "Adaptive refinement produced invalid bounds: dominating bound "
                f"{new_upper:.12g} is below dominated bound {new_lower:.12g}"
            )

        upper_bound = min(upper_bound, new_upper)
        lower_bound = max(lower_bound, new_lower)

        if upper_bound < lower_bound:
            raise RuntimeError(
                "Adaptive refinement produced invalid bounds: dominating bound "
                f"{upper_bound:.12g} is below dominated bound {lower_bound:.12g}"
            )

        if auto_target_accuracy:
            target_accuracy = max(target_accuracy, 0.25 * lower_bound)

        gap = upper_bound - lower_bound
        if gap < target_accuracy:
            converged = True
            break

        discretization, tail_truncation, changed = _apply_refinement_step(
            discretization=discretization,
            tail_truncation=tail_truncation,
        )
        if not changed:
            break

    if not converged:
        warnings.warn(
            f"Adaptive refinement did not converge after {MAX_ITERATIONS} iterations. "
            f"Final gap: {upper_bound - lower_bound:.6e}, target: {target_accuracy:.6e}. "
            f"Returning best bounds found.",
            RuntimeWarning,
        )

    return AdaptiveResult(
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        absolute_gap=upper_bound - lower_bound,
        converged=converged,
        iterations=iteration,
        initial_discretization=effective_initial_discretization,
        discretization=discretization,
        initial_tail_truncation=effective_initial_tail_truncation,
        tail_truncation=tail_truncation,
        target_accuracy=target_accuracy,
    )

# =============================================================================
# Helper Functions
# =============================================================================

def estimate_poisson_query(*,
    params: PrivacyParams,
    query_func: Callable[[privacy_loss_distribution.PrivacyLossDistribution], float],
) -> float:
    """Estimate the query value with a Poisson-subsampled Gaussian approximation."""
    sampling_probability = params.num_selected / params.num_steps
    num_rounds = params.num_selected * params.num_epochs

    pld = privacy_loss_distribution.from_gaussian_mechanism(
        standard_deviation=params.sigma,
        sensitivity=1.0,
        value_discretization_interval=POISSON_GUESS_DISCRETIZATION,
        pessimistic_estimate=True,
        sampling_prob=sampling_probability,
    ).self_compose(num_rounds)

    estimate = float(query_func(pld))
    if not math.isfinite(estimate) or estimate < 0.0:
        raise RuntimeError(
            "Poisson-based adaptive initialization produced an invalid estimate: "
            f"{estimate!r}"
        )
    return estimate


def _clip_discretization(value: float) -> float:
    return min(max(value, MIN_DISCRETIZATION), MAX_DISCRETIZATION)


def _clip_tail_truncation(value: float) -> float:
    return min(max(value, MIN_TAIL_TRUNCATION), MAX_TAIL_TRUNCATION)


def _apply_refinement_step(*,
    discretization: float,
    tail_truncation: float,
) -> tuple[float, float, bool]:
    next_discretization = _clip_discretization(discretization / 2)
    next_tail_truncation = _clip_tail_truncation(tail_truncation / 10)
    changed = (
        next_discretization != discretization
        or next_tail_truncation != tail_truncation
    )
    return next_discretization, next_tail_truncation, changed

def _auto_target_accuracy(*,
    target_accuracy: float,
    estimated_value: float,
) -> tuple[float, bool]:
    if target_accuracy >= 0.0:
        if not math.isfinite(target_accuracy):
            raise RuntimeError(
                "Adaptive refinement received an invalid target accuracy: "
                f"{target_accuracy!r}"
            )
        return float(target_accuracy), False

    auto_target_accuracy = 0.10 * estimated_value
    if not math.isfinite(auto_target_accuracy) or auto_target_accuracy < 0.0:
        raise RuntimeError(
            "Adaptive refinement produced an invalid automatic target accuracy: "
            f"{auto_target_accuracy!r}"
        )
    return auto_target_accuracy, True


def _build_pld_pair(*,
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    pld_builder: Callable[..., privacy_loss_distribution.PrivacyLossDistribution],
) -> tuple[
    privacy_loss_distribution.PrivacyLossDistribution,
    privacy_loss_distribution.PrivacyLossDistribution,
]:
    pld_upper = pld_builder(
        params=params,
        config=config,
        bound_type=BoundType.DOMINATES,
    )
    pld_lower = pld_builder(
        params=params,
        config=config,
        bound_type=BoundType.IS_DOMINATED,
    )
    return pld_upper, pld_lower
