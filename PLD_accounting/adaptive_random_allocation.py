"""Adaptive helpers for random-allocation queries."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable

from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    PrivacyParams,
    require_privacy_params,
)
from PLD_accounting.validation import require_finite_real, require_positive_real

MAX_ITERATIONS = 10
# Default relative ratio bound (upper/lower <= 1 + this value), tail init, and
# Poisson grid seed. Negative target_accuracy selects this ratio; it is not auto-target.
DEFAULT_RELATIVE_ACCURACY = 0.1
POISSON_GUESS_DISCRETIZATION = 1e-4
MIN_DISCRETIZATION = 1e-6
MAX_DISCRETIZATION = 1e-1
MIN_TAIL_TRUNCATION = 1e-20
MAX_TAIL_TRUNCATION = 1e-4

# Per-iteration refinement schedule. The discretization halves (bound error is
# roughly linear in the step, so halving buys a predictable factor per pass)
# while the tail budget drops by a decade, which shrinks the truncation-induced
# term faster than the quantization term it is paired against.
DISCRETIZATION_REFINEMENT_FACTOR = 2.0
TAIL_TRUNCATION_REFINEMENT_FACTOR = 10.0


@dataclass(kw_only=True)
class AdaptiveResult:
    """Result from adaptive allocation computation.

    Attributes:
        upper_bound: Best upper bound found across all iterations.
        lower_bound: Best lower bound found across all iterations.
        absolute_gap: Final gap between upper and lower bounds.
        converged: Whether the algorithm converged to target accuracy.
        iterations: Number of evaluations performed.
        initial_discretization: Starting loss_discretization value.
        discretization: Last evaluated loss_discretization.
        initial_tail_truncation: Starting tail_truncation value.
        tail_truncation: Last evaluated tail_truncation.
        target_accuracy: Requested accuracy. Nonnegative is an absolute gap;
            negative selects the default relative ratio bound.

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


def optimize_allocation_epsilon_range(
    *,
    params: PrivacyParams,
    target_accuracy: float,
    pld_builder: Callable[..., privacy_loss_distribution.PrivacyLossDistribution],
    initial_discretization: float | None = None,
    initial_tail_truncation: float | None = None,
) -> AdaptiveResult:
    """Refine paired epsilon bounds on a fixed numerical schedule.

    Each iteration builds dominating and dominated PLDs with one shared
    configuration, retains the best bounds seen so far, halves the loss step,
    and reduces the tail budget by a decade until the requested accuracy is met.
    A negative ``target_accuracy`` stops when ``upper_bound / lower_bound`` is
    at most ``1 + DEFAULT_RELATIVE_ACCURACY``.
    """
    require_privacy_params(value=params)
    delta = params.require_delta()
    require_finite_real(value=target_accuracy, name="target_accuracy")
    if initial_discretization is not None:
        require_positive_real(value=initial_discretization, name="initial_discretization")
    if initial_tail_truncation is not None:
        require_positive_real(value=initial_tail_truncation, name="initial_tail_truncation")

    if initial_discretization is None:
        accuracy_scale = target_accuracy
        if target_accuracy < 0.0:
            accuracy_scale = DEFAULT_RELATIVE_ACCURACY * estimate_poisson_query(
                params=params,
                query_func=lambda pld: float(pld.get_epsilon_for_delta(delta)),
            )
        initial_discretization = accuracy_scale / DISCRETIZATION_REFINEMENT_FACTOR
    if initial_tail_truncation is None:
        initial_tail_truncation = DEFAULT_RELATIVE_ACCURACY * delta

    discretization = _clip_discretization(initial_discretization)
    tail_truncation = _clip_tail_truncation(initial_tail_truncation)
    effective_initial_discretization = discretization
    effective_initial_tail_truncation = tail_truncation

    upper_bound, lower_bound = _evaluate_pair_epsilons(
        params=params,
        discretization=discretization,
        tail_truncation=tail_truncation,
        pld_builder=pld_builder,
        delta=delta,
    )
    if _has_converged(
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        target_accuracy=target_accuracy,
    ):
        return AdaptiveResult(
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            absolute_gap=upper_bound - lower_bound,
            converged=True,
            iterations=1,
            initial_discretization=effective_initial_discretization,
            discretization=discretization,
            initial_tail_truncation=effective_initial_tail_truncation,
            tail_truncation=tail_truncation,
            target_accuracy=target_accuracy,
        )

    # Counted after the call, so a refinement that clamps to a no-change step exits
    # without claiming an evaluation it never ran.
    evaluations = 1
    converged = False
    for _ in range(1, MAX_ITERATIONS):
        discretization, tail_truncation, changed = _apply_refinement_step(
            discretization=discretization,
            tail_truncation=tail_truncation,
        )
        if not changed:
            break
        new_upper, new_lower = _evaluate_pair_epsilons(
            params=params,
            discretization=discretization,
            tail_truncation=tail_truncation,
            pld_builder=pld_builder,
            delta=delta,
        )
        evaluations += 1
        upper_bound = min(upper_bound, new_upper)
        lower_bound = max(lower_bound, new_lower)
        if upper_bound < lower_bound:
            raise RuntimeError(
                "Adaptive refinement produced invalid bounds: dominating bound "
                f"{upper_bound:.12g} is below dominated bound {lower_bound:.12g}"
            )
        if _has_converged(
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            target_accuracy=target_accuracy,
        ):
            converged = True
            break

    if not converged:
        warnings.warn(
            f"Adaptive refinement did not converge after {evaluations} evaluations. "
            f"Final gap: {upper_bound - lower_bound:.6e}, target: {target_accuracy:.6e}. "
            f"Returning best bounds found.",
            RuntimeWarning,
        )

    return AdaptiveResult(
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        absolute_gap=upper_bound - lower_bound,
        converged=converged,
        iterations=evaluations,
        initial_discretization=effective_initial_discretization,
        discretization=discretization,
        initial_tail_truncation=effective_initial_tail_truncation,
        tail_truncation=tail_truncation,
        target_accuracy=target_accuracy,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def estimate_poisson_query(
    *,
    params: PrivacyParams,
    query_func: Callable[[privacy_loss_distribution.PrivacyLossDistribution], float],
) -> float:
    """Estimate the query value with a Poisson-subsampled Gaussian approximation."""
    require_privacy_params(value=params)

    # Approximate random allocation as Poisson subsampling applied once per
    # allocation step. Each epoch has ``num_steps`` opportunities, and each
    # example participates in ``num_selected`` of them on average, so the
    # per-step sampling probability is ``num_selected / num_steps`` and the
    # total number of Poisson rounds is ``num_steps * num_epochs``.
    sampling_probability = params.num_selected / params.num_steps
    num_rounds = params.num_steps * params.num_epochs

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
            "Poisson-based adaptive initialization produced an invalid estimate: " f"{estimate!r}"
        )
    return estimate


def _clip_discretization(value: float) -> float:
    return min(max(value, MIN_DISCRETIZATION), MAX_DISCRETIZATION)


def _clip_tail_truncation(value: float) -> float:
    return min(max(value, MIN_TAIL_TRUNCATION), MAX_TAIL_TRUNCATION)


def _apply_refinement_step(
    *,
    discretization: float,
    tail_truncation: float,
) -> tuple[float, float, bool]:
    """Tighten both budgets one notch, reporting whether either actually moved.

    The ``changed`` flag lets the caller stop early once both values have hit
    their clamps, instead of burning the remaining iterations on identical work.
    """
    next_discretization = _clip_discretization(discretization / DISCRETIZATION_REFINEMENT_FACTOR)
    next_tail_truncation = _clip_tail_truncation(
        tail_truncation / TAIL_TRUNCATION_REFINEMENT_FACTOR
    )
    changed = next_discretization != discretization or next_tail_truncation != tail_truncation
    return next_discretization, next_tail_truncation, changed


def _has_converged(
    *,
    upper_bound: float,
    lower_bound: float,
    target_accuracy: float,
) -> bool:
    """Return whether the current pair meets the requested accuracy."""
    if target_accuracy < 0.0:
        return lower_bound > 0.0 and upper_bound / lower_bound <= 1.0 + DEFAULT_RELATIVE_ACCURACY
    return upper_bound - lower_bound < target_accuracy


def _evaluate_pair_epsilons(
    *,
    params: PrivacyParams,
    discretization: float,
    tail_truncation: float,
    pld_builder: Callable[..., privacy_loss_distribution.PrivacyLossDistribution],
    delta: float,
) -> tuple[float, float]:
    """Build one dominating/dominated pair and return its epsilon estimates."""
    config = AllocationSchemeConfig(
        loss_discretization=discretization,
        tail_truncation=tail_truncation,
        convolution_method=ConvolutionMethod.GEOM,
    )
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
    new_upper = float(pld_upper.get_epsilon_for_delta(delta))
    new_lower = float(pld_lower.get_epsilon_for_delta(delta))
    if new_upper < new_lower:
        raise RuntimeError(
            "Adaptive refinement produced invalid bounds: dominating bound "
            f"{new_upper:.12g} is below dominated bound {new_lower:.12g}"
        )
    return new_upper, new_lower
