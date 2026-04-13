"""Validation utilities for privacy accounting parameters.

This module provides reusable validation functions to standardize parameter checking
across the codebase and eliminate repetitive validation code.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.types import BoundType, PrivacyParams

# =============================================================================
# Discrete PMF validation
# =============================================================================


def validate_discrete_pmf_and_boundaries(
    prob_arr: NDArray[np.float64],
    p_min: float,
    p_max: float,
) -> None:
    """Validate 1-D nonnegative PMF and nonnegative boundary masses.

    Args:
        prob_arr: Finite-support probability masses.
        p_min: Lower boundary mass (e.g. mass at ``-∞`` or 0).
        p_max: Upper boundary mass (e.g. mass at ``+∞``).

    Raises:
        ValueError: If shape or nonnegativity checks fail.

    """
    prob_arr = np.asarray(prob_arr, dtype=np.float64)
    if prob_arr.ndim != 1:
        raise ValueError("PMF must be 1-D array")
    if np.any(prob_arr < 0.0):
        raise ValueError("PMF must be nonnegative")
    if p_min < 0.0:
        raise ValueError(f"min must be nonnegative, got {p_min:.2e}")
    if p_max < 0.0:
        raise ValueError(f"max must be nonnegative, got {p_max:.2e}")


# =============================================================================
# Privacy Parameter Validation
# =============================================================================


def validate_privacy_params(
    params: PrivacyParams,
    *,
    require_delta: bool = False,
    require_epsilon: bool = False,
) -> None:
    """Validate PrivacyParams object.

    Args:
        params: Privacy parameters to validate.
        require_delta: If True, validate that delta is set and in valid range (0, 1).
        require_epsilon: If True, validate that epsilon is set and positive.

    Raises:
        TypeError: If params is not a PrivacyParams instance.
        ValueError: If any parameter value is invalid.

    """
    if not isinstance(params, PrivacyParams):
        raise TypeError(f"params must be PrivacyParams, got {type(params)}")
    validate_gaussian_params(params.sigma, params.num_steps, params.num_selected, params.num_epochs)
    if require_delta:
        validate_delta(params.delta)
    if require_epsilon:
        validate_epsilon(params.epsilon)


def validate_gaussian_params(
    sigma: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
) -> None:
    """Validate Gaussian allocation parameters.

    Args:
        sigma: Gaussian noise scale.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.

    Raises:
        ValueError: If any parameter value is invalid.

    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    validate_allocation_params(num_steps, num_selected, num_epochs)


def validate_allocation_params(
    num_steps: int,
    num_selected: int,
    num_epochs: int,
) -> None:
    """Validate allocation parameters.

    Args:
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.

    Raises:
        ValueError: If any parameter value is invalid.

    """
    if num_steps < 1 or num_selected < 1 or num_epochs < 1:
        raise ValueError(
            f"num_steps (={num_steps}), num_selected (={num_selected}), "
            f"and num_epochs (={num_epochs}) must be >= 1"
        )
    if num_selected > num_steps:
        raise ValueError(f"num_selected ({num_selected}) cannot exceed num_steps ({num_steps})")


def validate_delta(delta: float | None) -> None:
    """Validate delta value.

    Args:
        delta: Delta value for differential privacy.

    Raises:
        ValueError: If delta is None or not in the valid range (0, 1).

    """
    if delta is None or not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def validate_epsilon(epsilon: float | None) -> None:
    """Validate epsilon value.

    Args:
        epsilon: Epsilon value for differential privacy.

    Raises:
        ValueError: If epsilon is None or not positive.

    """
    if epsilon is None or epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")


# =============================================================================
# Bound Type Validation
# =============================================================================


def validate_bound_type(bound_type: BoundType) -> None:
    """Validate BoundType enum value.

    Args:
        bound_type: The bound type to validate.
        allow_both: If False, raise an error if bound_type is BoundType.BOTH.

    Raises:
        ValueError: If bound_type is invalid or BoundType.BOTH when not allowed.

    """
    if bound_type not in (BoundType.DOMINATES, BoundType.IS_DOMINATED):
        raise ValueError(f"Invalid bound_type: {bound_type}, BOTH is not supported")


# =============================================================================
# Discretization Parameter Validation
# =============================================================================


def validate_discretization_params(
    loss_discretization: float,
    tail_truncation: float,
) -> None:
    """Validate discretization parameters.

    Args:
        loss_discretization: Loss discretization interval.
        tail_truncation: Tail truncation threshold.

    Raises:
        ValueError: If any parameter is invalid.

    """
    if loss_discretization <= 0:
        raise ValueError(f"loss_discretization must be positive, got {loss_discretization}")
    if tail_truncation <= 0:
        raise ValueError(f"tail_truncation must be positive, got {tail_truncation}")


def validate_optional_discretization_params(
    initial_discretization: float | None = None,
    initial_tail_truncation: float | None = None,
) -> None:
    """Validate optional discretization parameters.

    Args:
        initial_discretization: Optional initial loss discretization interval.
        initial_tail_truncation: Optional initial tail truncation threshold.

    Raises:
        ValueError: If any provided parameter is invalid.

    """
    if initial_discretization is not None and initial_discretization <= 0:
        raise ValueError(f"initial_discretization must be positive, got {initial_discretization}")
    if initial_tail_truncation is not None and initial_tail_truncation <= 0:
        raise ValueError(f"initial_tail_truncation must be positive, got {initial_tail_truncation}")
