"""Discrete distribution classes and structured grid data types.

Class hierarchy:
- DiscreteDistBase: abstract base for all discrete distributions
- SparseDiscreteDist: arbitrary explicit support (explicit x_array)
- DenseDiscreteDist: regular-grid distribution
  - spacing_type=LINEAR:    x[i] = x_min + i * step
  - spacing_type=GEOMETRIC: x[i] = x_min * step^i
- PLDRealization: DenseDiscreteDist specialised for privacy loss (LINEAR + REALS)

Domain semantics:
- REALS:     p_min = mass at −∞,  p_max = mass at +∞
- POSITIVES: p_min = mass at 0,   p_max = mass at +∞
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from PLD_accounting.distribution_utils import (
    PMF_MASS_TOL,
    compute_bin_ratio,
    compute_bin_width,
    compute_truncation,
    exp_moment_terms,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.validation import validate_discrete_pmf_and_boundaries

REALIZATION_MOMENT_TOL = 1e-12


class Domain(Enum):
    """Domain of a discrete distributsion's support."""

    REALS = "reals"  # p_min = mass at −∞, p_max = mass at +∞
    POSITIVES = "positives"  # p_min = mass at 0,  p_max = mass at +∞


# =============================================================================
# ABSTRACT BASE
# =============================================================================


class DiscreteDistBase(ABC):
    """Abstract base for discrete PMF representations with boundary masses.

    Attributes:
        prob_arr: probability mass on finite support
        p_min: mass at the lower boundary (−∞ for REALS, 0 for POSITIVES)
        p_max: mass at +∞
        domain: whether the support is over the reals or positive numbers
    """

    def __init__(
        self,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Initialize discrete distribution with PMF array and boundary masses."""
        self.prob_arr = np.asarray(prob_arr, dtype=np.float64)
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.domain = domain
        self._validate_basic()

    @abstractmethod
    def get_x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""

    @property
    def x_array(self) -> NDArray[np.float64]:
        """Materialized support."""
        return self.get_x_array()

    def _validate_basic(self) -> None:
        validate_discrete_pmf_and_boundaries(
            self.prob_arr,
            self.p_min,
            self.p_max,
        )

        pmf_sum = math.fsum(map(float, self.prob_arr))
        total_mass = pmf_sum + self.p_min + self.p_max
        mass_error = abs(total_mass - 1.0)
        if mass_error > PMF_MASS_TOL:
            error_msg = "MASS CONSERVATION ERROR"
            error_msg += f": Error={mass_error:.2e} (tolerance={PMF_MASS_TOL:.2e})"
            error_msg += f", PMF sum={pmf_sum:.15f}"
            error_msg += f", min={self.p_min:.2e}"
            error_msg += f", max={self.p_max:.2e}"
            error_msg += f", Total mass={total_mass:.15f}"
            raise ValueError(error_msg)

        # REALS domain: both boundaries being non-zero is not allowed.
        if self.domain == Domain.REALS and self.p_min > PMF_MASS_TOL and self.p_max > PMF_MASS_TOL:
            raise ValueError("REALS domain: p_min and p_max cannot both be non-zero")

    def truncate_edges(self, tail_truncation: float, bound_type: BoundType) -> Self:
        """Truncate distribution edges. Computation lives in distribution_utils."""
        new_PMF, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            self.prob_arr, self.p_min, self.p_max, tail_truncation, bound_type
        )
        return self._create_truncated(new_PMF, new_p_min, new_p_max, min_ind, max_ind)

    @abstractmethod
    def _create_truncated(
        self,
        new_PMF: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> Self:
        """Create truncated instance preserving representation semantics."""

    @abstractmethod
    def copy(self) -> Self:
        """Deep-copy this distribution while preserving representation type."""


# =============================================================================
# GENERAL (EXPLICIT) DISTRIBUTION
# =============================================================================


class SparseDiscreteDist(DiscreteDistBase):
    """General discrete distribution with explicit support values."""

    def __init__(
        self,
        x_array: NDArray[np.float64],
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Initialize general discrete distribution with explicit support points."""
        self._x_array = np.asarray(x_array, dtype=np.float64)
        super().__init__(prob_arr, p_min, p_max, domain)
        self._validate_x_array()

    def _validate_x_array(self) -> None:
        if self._x_array.ndim != 1 or self._x_array.shape != self.prob_arr.shape:
            raise ValueError("x and PMF must be 1-D arrays of equal length")
        if not np.all(np.diff(self._x_array) > 0):
            raise ValueError("x must be strictly increasing")

    def get_x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""
        return self._x_array

    def _create_truncated(
        self,
        new_PMF: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> SparseDiscreteDist:
        return SparseDiscreteDist(
            x_array=self._x_array[slice(min_ind, max_ind + 1)],
            prob_arr=new_PMF,
            p_min=new_p_min,
            p_max=new_p_max,
            domain=self.domain,
        )

    def copy(self) -> SparseDiscreteDist:
        """Create a deep copy of this distribution."""
        return SparseDiscreteDist(
            x_array=self._x_array.copy(),
            prob_arr=self.prob_arr.copy(),
            p_min=self.p_min,
            p_max=self.p_max,
            domain=self.domain,
        )


# =============================================================================
# UNIFIED REGULAR-GRID DISTRIBUTION
# =============================================================================


class DenseDiscreteDist(DiscreteDistBase):
    """Discrete distribution on a regular (linear or geometric) grid.

    spacing_type = LINEAR:    x[i] = x_min + i * step   (step = additive gap > 0)
    spacing_type = GEOMETRIC: x[i] = x_min * step^i     (step = ratio > 1, x_min > 0)

    For geometric grids the domain is always POSITIVES (x_min > 0 enforces positivity).
    """

    def __init__(
        self,
        x_min: float,
        step: float,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        spacing_type: SpacingType = SpacingType.LINEAR,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Initialize regular-grid discrete distribution."""
        self.x_min = float(x_min)
        self.step = float(step)
        self.spacing_type = spacing_type
        super().__init__(prob_arr, p_min, p_max, domain)
        self._validate_grid()

    def _validate_grid(self) -> None:
        if self.spacing_type == SpacingType.LINEAR:
            if self.step <= 0.0:
                raise ValueError("step must be positive for linear grid")
        elif self.spacing_type == SpacingType.GEOMETRIC:
            if self.x_min <= 0.0:
                raise ValueError("x_min must be positive for geometric grid")
            if self.step <= 1.0:
                raise ValueError("step must be > 1 for geometric grid")
            if self.domain != Domain.POSITIVES:
                raise ValueError("Geometric spacing requires domain=Domain.POSITIVES")
        else:
            raise ValueError(f"Unknown SpacingType: {self.spacing_type}")

    @classmethod
    def from_x_array(
        cls,
        x_array: NDArray[np.float64],
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        spacing_type: SpacingType = SpacingType.LINEAR,
        domain: Domain = Domain.REALS,
    ) -> "DenseDiscreteDist":
        """Create DenseDiscreteDist from x_array by extracting x_min and step."""
        if spacing_type == SpacingType.LINEAR:
            step = compute_bin_width(x_array)
        else:
            step = compute_bin_ratio(x_array)
        return cls(
            x_min=float(x_array[0]),
            step=step,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            spacing_type=spacing_type,
            domain=domain,
        )

    def get_x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""
        n = self.prob_arr.size
        if self.spacing_type == SpacingType.LINEAR:
            return self.x_min + np.arange(n, dtype=np.float64) * self.step
        else:
            return self.x_min * np.power(self.step, np.arange(n, dtype=np.float64))

    def _create_truncated(
        self,
        new_PMF: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> "DenseDiscreteDist":
        if self.spacing_type == SpacingType.LINEAR:
            new_x_min = self.x_min + min_ind * self.step
        else:
            new_x_min = self.x_min * (self.step ** float(min_ind))
        return self.__class__(
            x_min=new_x_min,
            step=self.step,
            prob_arr=new_PMF,
            p_min=new_p_min,
            p_max=new_p_max,
            spacing_type=self.spacing_type,
            domain=self.domain,
        )

    def copy(self) -> "DenseDiscreteDist":
        """Create a deep copy of this distribution."""
        return self.__class__(
            x_min=self.x_min,
            step=self.step,
            prob_arr=self.prob_arr.copy(),
            p_min=self.p_min,
            p_max=self.p_max,
            spacing_type=self.spacing_type,
            domain=self.domain,
        )


# =============================================================================
# PLD REALIZATION
# =============================================================================


class PLDRealization(DenseDiscreteDist):
    """Linear-grid PLD realization in loss space."""

    def __init__(
        self,
        x_min: float,
        step: float,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
    ) -> None:
        """Initialize PLD realization with privacy loss values and probabilities."""
        super().__init__(
            x_min=x_min,
            step=step,
            prob_arr=prob_arr,
            p_min=float(p_min),
            p_max=float(p_max),
            spacing_type=SpacingType.LINEAR,
            domain=Domain.REALS,
        )
        self._validate_pld_realization()

    @classmethod
    def from_linear_dist(cls, dist: DenseDiscreteDist) -> "PLDRealization":
        """Build a validated PLD realization from a linear-grid DenseDiscreteDist."""
        if not isinstance(dist, DenseDiscreteDist) or dist.spacing_type != SpacingType.LINEAR:
            raise TypeError(
                f"from_linear_dist requires DenseDiscreteDist with LINEAR spacing, got {type(dist)}"
            )
        return cls(
            x_min=dist.x_min,
            step=dist.step,
            prob_arr=dist.prob_arr,
            p_max=dist.p_max,
            p_min=dist.p_min,
        )

    def _validate_pld_realization(self) -> "PLDRealization":
        """Validate the properties of PLD-realization.

        1. p(-inf) = 0 (p_min = 0).
        2. E[e^(-X)] <= 1.
        """
        # PLD realizations must have zero mass at negative-infinity loss.
        if self.p_min > PMF_MASS_TOL:
            raise ValueError(f"PLD realization requires p_min = 0, got {self.p_min:.2e}")

        exp_moment_val = exp_moment_terms(prob_arr=self.prob_arr, x_vals=self.x_array)
        if np.any(np.isinf(exp_moment_val)):
            raise ValueError(
                "Exponential moment E[exp(-L)] is infinite, not a valid PLD realization"
            )
        exp_moment_total = math.fsum(map(float, exp_moment_val))
        if exp_moment_total > 1.0 + REALIZATION_MOMENT_TOL:
            raise ValueError(
                f"Exponential moment E[exp(-L)] = {exp_moment_total:.15f} > 1.0, "
                "not a valid PLD realization"
            )
        return self

    def copy(self) -> "PLDRealization":
        """Create a deep copy of this PLD realization."""
        return PLDRealization(
            x_min=self.x_min,
            step=self.step,
            prob_arr=self.prob_arr.copy(),
            p_max=self.p_max,
            p_min=self.p_min,
        )

    def _create_truncated(
        self,
        new_PMF: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> "PLDRealization":
        """Create a truncated PLD realization while preserving linear-loss semantics."""
        del max_ind
        return PLDRealization(
            x_min=self.x_min + min_ind * self.step,
            step=self.step,
            prob_arr=new_PMF,
            p_min=new_p_min,
            p_max=new_p_max,
        )
