"""Validation utilities for privacy accounting parameters.

This module provides reusable validation functions to standardize parameter checking
across the codebase and eliminate repetitive validation code.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    PrivacyParams,
)

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
    if prob_arr.size == 0:
        raise ValueError("PMF must contain at least one finite-support bin")
    validate_finite_array(prob_arr, "PMF")
    if np.any(prob_arr < 0.0):
        raise ValueError("PMF must be nonnegative")
    validate_finite_real(p_min, "p_min")
    validate_finite_real(p_max, "p_max")
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
    validate_finite_real(sigma, "sigma")
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
    for value, name in (
        (num_steps, "num_steps"),
        (num_selected, "num_selected"),
        (num_epochs, "num_epochs"),
    ):
        validate_integer(value, name)
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
    if delta is None:
        raise ValueError("delta must be in (0, 1), got None")
    validate_finite_real(delta, "delta")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")


def validate_epsilon(epsilon: float | None) -> None:
    """Validate epsilon value.

    Args:
        epsilon: Epsilon value for differential privacy.

    Raises:
        ValueError: If epsilon is None or not positive.

    """
    if epsilon is None:
        raise ValueError("epsilon must be positive, got None")
    validate_finite_real(epsilon, "epsilon")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")


# =============================================================================
# Bound Type Validation
# =============================================================================


def validate_bound_type(bound_type: BoundType) -> None:
    """Validate BoundType enum value.

    Args:
        bound_type: The bound type to validate.

    Raises:
        ValueError: If bound_type is not DOMINATES or IS_DOMINATED.

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
    validate_finite_real(loss_discretization, "loss_discretization")
    validate_finite_real(tail_truncation, "tail_truncation")
    if loss_discretization <= 0:
        raise ValueError(f"loss_discretization must be positive, got {loss_discretization}")
    if tail_truncation <= 0:
        raise ValueError(f"tail_truncation must be positive, got {tail_truncation}")


def validate_allocation_scheme_config(config: AllocationSchemeConfig) -> None:
    """Validate AllocationSchemeConfig fields.

    Args:
        config: Configuration to validate.

    Raises:
        TypeError: If config is not an AllocationSchemeConfig instance.
        ValueError: If any field value is out of range.

    """
    if not isinstance(config, AllocationSchemeConfig):
        raise TypeError(f"config must be AllocationSchemeConfig, got {type(config)}")
    validate_discretization_params(config.loss_discretization, config.tail_truncation)
    if not isinstance(config.convolution_method, ConvolutionMethod):
        raise TypeError(
            "convolution_method must be ConvolutionMethod, "
            f"got {type(config.convolution_method).__name__}"
        )
    for value, name in (
        (config.max_grid_fft, "max_grid_fft"),
        (config.max_grid_mult, "max_grid_mult"),
        (config.cf_max_grid, "cf_max_grid"),
        (config.cf_refine_factor, "cf_refine_factor"),
    ):
        validate_integer(value, name)
    if config.max_grid_fft <= 0:
        raise ValueError(f"max_grid_fft must be positive, got {config.max_grid_fft}")
    if config.max_grid_mult != -1 and config.max_grid_mult <= 0:
        raise ValueError(
            f"max_grid_mult must be -1 (no limit) or a positive integer, "
            f"got {config.max_grid_mult}"
        )
    if config.cf_max_grid <= 0:
        raise ValueError(f"cf_max_grid must be positive, got {config.cf_max_grid}")
    if config.cf_refine_factor <= 0:
        raise ValueError(f"cf_refine_factor must be positive, got {config.cf_refine_factor}")


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
    if initial_discretization is not None:
        validate_finite_real(initial_discretization, "initial_discretization")
        if initial_discretization <= 0:
            raise ValueError(
                f"initial_discretization must be positive, got {initial_discretization}"
            )
    if initial_tail_truncation is not None:
        validate_finite_real(initial_tail_truncation, "initial_tail_truncation")
        if initial_tail_truncation <= 0:
            raise ValueError(
                f"initial_tail_truncation must be positive, got {initial_tail_truncation}"
            )


def validate_finite_real(value: object, name: str) -> None:
    """Require a non-boolean, finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


def validate_finite_array(values: NDArray[np.float64], name: str) -> None:
    """Require every entry in a numeric array to be finite."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} entries must be finite")


def validate_integer(value: object, name: str) -> None:
    """Require an integer scalar while rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
