"""Core type definitions for privacy accounting."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

try:
    from numba import njit as _NJIT

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    warnings.warn(
        "numba is not installed; some operations will use NumPy fallbacks "
        "and may be slower. Install numba for full performance.",
        ImportWarning,
        stacklevel=2,
    )

from PLD_accounting.validation import (  # noqa: E402  # local after optional numba import
    require_allocation_counts,
    require_enum,
    require_integer,
    require_open_unit_interval,
    require_positive_int,
    require_positive_real,
    require_type,
)


def optional_njit() -> Callable[[Callable], Callable]:
    """Return numba's njit(cache=True) if available, else the identity decorator."""
    if _HAS_NUMBA:
        return _NJIT(cache=True)

    def identity_decorator(function: Callable) -> Callable:
        """Return the function unchanged, so kernels stay callable without Numba."""
        return function

    return identity_decorator


def has_numba() -> bool:
    """Return whether numba JIT support is available."""
    return _HAS_NUMBA


# =============================================================================
# Discrete Distribution Types
# =============================================================================


class BoundType(Enum):
    """Domination orientation for accounting and discretization.

    ``DOMINATES`` is the upper (pessimistic) bound and ``IS_DOMINATED`` is the
    lower (optimistic) bound. ``BOTH`` is a real member, but public operations
    take one orientation.
    """

    DOMINATES = "DOMINATES"
    IS_DOMINATED = "IS_DOMINATED"
    BOTH = "BOTH"


class SpacingType(Enum):
    """Grid spacing_type strategy."""

    LINEAR = "linear"
    GEOMETRIC = "geometric"


class ConvolutionMethod(Enum):
    """Convolution method for numerical stability."""

    GEOM = "geometric"
    FFT = "fft"
    COMBINED = "combined"
    BEST_OF_TWO = "best_of_two"


class Direction(Enum):
    """Enum for direction of privacy analysis."""

    ADD = "add"
    REMOVE = "remove"
    BOTH = "both"


def require_bound_type(*, value: object, name: str = "bound_type") -> BoundType:
    """Require ``DOMINATES`` or ``IS_DOMINATED``.

    ``BOTH`` is a real member, but public operations take one orientation.
    """
    return require_enum(
        value=value,
        enum_cls=BoundType,
        name=name,
        allowed=(BoundType.DOMINATES, BoundType.IS_DOMINATED),
    )


def require_direction(*, value: object, name: str = "direction") -> Direction:
    """Require ``ADD`` or ``REMOVE``.

    ``BOTH`` is a real member, but public operations take one orientation.
    """
    return require_enum(
        value=value,
        enum_cls=Direction,
        name=name,
        allowed=(Direction.ADD, Direction.REMOVE),
    )


# Defaults for AllocationSchemeConfig (independent of REALIZATION_MOMENT_TOL;
# tail budget is a modeling choice).
DEFAULT_LOSS_DISCRETIZATION = 1e-2
DEFAULT_TAIL_TRUNCATION = 1e-12
# Point cap for FFT grids; the multiplicative route is uncapped by default.
DEFAULT_MAX_GRID_FFT = 1_000_000


@dataclass(frozen=True, kw_only=True)
class PrivacyParams:
    """Parameters common to all privacy schemes."""

    sigma: float
    num_steps: int
    num_selected: int = 1
    num_epochs: int = 1
    epsilon: float | None = None
    delta: float | None = None

    def __post_init__(self) -> None:
        """Reject field values that cannot describe a privacy query."""
        require_positive_real(value=self.sigma, name="sigma")
        require_allocation_counts(
            num_steps=self.num_steps, num_selected=self.num_selected, num_epochs=self.num_epochs
        )
        if self.epsilon is not None:
            require_positive_real(value=self.epsilon, name="epsilon")
        if self.delta is not None:
            require_open_unit_interval(value=self.delta, name="delta")

    def require_delta(self) -> float:
        """Return ``delta``, requiring it to be set."""
        if self.delta is None:
            raise ValueError("delta must be in (0, 1), got None")
        return self.delta

    def require_epsilon(self) -> float:
        """Return ``epsilon``, requiring it to be set."""
        if self.epsilon is None:
            raise ValueError("epsilon must be positive, got None")
        return self.epsilon


def require_privacy_params(*, value: object, name: str = "params") -> PrivacyParams:
    """Require a ``PrivacyParams`` instance."""
    return require_type(value=value, expected_type=PrivacyParams, name=name)


@dataclass(frozen=True, kw_only=True)
class AllocationSchemeConfig:
    """Configuration for privacy schemes."""

    loss_discretization: float = DEFAULT_LOSS_DISCRETIZATION
    tail_truncation: float = DEFAULT_TAIL_TRUNCATION
    max_grid_fft: int = DEFAULT_MAX_GRID_FFT
    max_grid_mult: int = -1  # any value <= 0 means no upper limit on grid size
    convolution_method: ConvolutionMethod = ConvolutionMethod.GEOM

    def __post_init__(self) -> None:
        """Reject field values that cannot describe a discretization scheme."""
        require_positive_real(
            value=[self.loss_discretization, self.tail_truncation],
            name=["loss_discretization", "tail_truncation"],
        )
        require_positive_int(value=self.max_grid_fft, name="max_grid_fft")
        require_integer(value=self.max_grid_mult, name="max_grid_mult")
        require_enum(
            value=self.convolution_method, enum_cls=ConvolutionMethod, name="convolution_method"
        )


def require_allocation_config(*, value: object, name: str = "config") -> AllocationSchemeConfig:
    """Require an ``AllocationSchemeConfig`` instance."""
    return require_type(value=value, expected_type=AllocationSchemeConfig, name=name)
