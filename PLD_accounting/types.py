from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
    BEST_OF_TWO = "best of two"


class Direction(Enum):
    """Enum for direction of privacy analysis"""
    ADD    = 'add'
    REMOVE = 'remove'
    BOTH   = 'both'


@dataclass
class PrivacyParams:
    """Parameters common to all privacy schemes"""
    sigma:        float
    num_steps:    int
    num_selected: int   = 1
    num_epochs:   int   = 1
    epsilon:      float | None = None
    delta:        float | None = None

@dataclass
class AllocationSchemeConfig:
    """Configuration for privacy schemes"""
    loss_discretization: float = 1e-2
    tail_truncation:     float = 1e-12
    max_grid_FFT:        int   = 1_000_000
    max_grid_mult:       int   = -1
    convolution_method:  ConvolutionMethod = ConvolutionMethod.GEOM