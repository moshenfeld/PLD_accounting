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


def optional_njit() -> Callable[[Callable], Callable]:
    """Return numba's njit(cache=True) if available, else the identity decorator."""
    if _HAS_NUMBA:
        return _NJIT(cache=True)

    def identity_decorator(function: Callable) -> Callable:
        return function

    return identity_decorator


def has_numba() -> bool:
    """Return whether numba JIT support is available."""
    return _HAS_NUMBA


# =============================================================================
# Discrete Distribution Types
# =============================================================================


class BoundType(Enum):
    """Tie-breaking bound_type for discretization."""

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


# Defaults for AllocationSchemeConfig (independent of REALIZATION_MOMENT_TOL;
# tail budget is a modeling choice).
DEFAULT_LOSS_DISCRETIZATION = 1e-2
DEFAULT_TAIL_TRUNCATION = 1e-12
# Point cap for FFT grids; the multiplicative route is uncapped by default.
DEFAULT_MAX_GRID_FFT = 1_000_000


@dataclass(frozen=True)
class PrivacyParams:
    """Parameters common to all privacy schemes."""

    sigma: float
    num_steps: int
    num_selected: int = 1
    num_epochs: int = 1
    epsilon: float | None = None
    delta: float | None = None


@dataclass(frozen=True)
class AllocationSchemeConfig:
    """Configuration for privacy schemes."""

    loss_discretization: float = DEFAULT_LOSS_DISCRETIZATION
    tail_truncation: float = DEFAULT_TAIL_TRUNCATION
    max_grid_fft: int = DEFAULT_MAX_GRID_FFT
    max_grid_mult: int = -1  # any value <= 0 means no upper limit on grid size
    convolution_method: ConvolutionMethod = ConvolutionMethod.GEOM
