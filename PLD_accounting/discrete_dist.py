"""Discrete distribution classes and structured grid data types.

Class hierarchy:
- DiscreteDistBase: abstract base for all discrete distributions
- SparseDiscreteDist: arbitrary explicit support (explicit x_array)
- DenseDiscreteDist: regular-grid distribution
  - spacing_type=LINEAR:    x[i] = x_0 + i * step
  - spacing_type=GEOMETRIC: x[i] = x_0 * step^i
- PLDRealization: DenseDiscreteDist specialised for privacy loss (LINEAR + REALS)

Domain semantics:
- REALS:     p_min = mass at −∞,  p_max = mass at +∞
- POSITIVES: p_min = mass at 0,   p_max = mass at +∞
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
from PLD_accounting.validation import (
    validate_discrete_pmf_and_boundaries,
    validate_finite_array,
    validate_finite_real,
)

REALIZATION_MOMENT_TOL = 1e-12


class Domain(Enum):
    """Domain of a discrete distribution's support."""

    REALS = "reals"  # p_min = mass at −∞, p_max = mass at +∞
    POSITIVES = "positives"  # p_min = mass at 0,  p_max = mass at +∞


@dataclass(frozen=True)
class GridSpec:
    """Exact parameters of a regular grid -- the single source of truth for a lattice.

    ``x[i] = x_0 + i * step``     for ``spacing_type == LINEAR``    (i in range(n))
    ``x[i] = x_0 * step ** i``    for ``spacing_type == GEOMETRIC``

    Threaded through the discretization pipeline so the resulting spacing is the
    exact one requested, never re-derived from a materialized array. The object
    is immutable so a validated distribution's support cannot later change.
    """

    x_0: float
    step: float
    n: int
    spacing_type: SpacingType = SpacingType.LINEAR

    def materialize(self) -> NDArray[np.float64]:
        """Materialize the grid points (the canonical support array)."""
        k = np.arange(self.n, dtype=np.float64)
        if self.spacing_type == SpacingType.LINEAR:
            return self.x_0 + k * self.step
        return self.x_0 * np.power(self.step, k)

    def last_point(self) -> float:
        """Last grid point, computed exactly as :meth:`materialize` produces it."""
        k_last = np.float64(self.n - 1)
        if self.spacing_type == SpacingType.LINEAR:
            return float(self.x_0 + k_last * self.step)
        return float(self.x_0 * np.power(self.step, k_last))


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
        validate_discrete_pmf_and_boundaries(
            prob_arr,
            p_min,
            p_max,
        )

        self._prob_arr = np.array(prob_arr, dtype=np.float64, copy=True)
        self._p_min = float(p_min)
        self._p_max = float(p_max)
        self._domain = domain
        self._validate_basic()
        self._prob_arr.setflags(write=False)

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        """Deep-copy while preserving the read-only array invariant."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            setattr(result, name, copy.deepcopy(value, memo))
        result._prob_arr.setflags(write=False)
        if hasattr(result, "_x_arr"):
            result._x_arr.setflags(write=False)
        return result

    @property
    def prob_arr(self) -> NDArray[np.float64]:
        """Read-only finite-support probability masses."""
        return self._prob_arr

    @property
    def p_min(self) -> float:
        """Mass at the lower boundary."""
        return self._p_min

    @property
    def p_max(self) -> float:
        """Mass at the upper boundary."""
        return self._p_max

    @property
    def domain(self) -> Domain:
        """Support-domain semantics."""
        return self._domain

    @property
    @abstractmethod
    def x_array(self) -> NDArray[np.float64]:
        """Materialized support."""

    def truncate_edges(self, tail_truncation: float, bound_type: BoundType) -> Self:
        """Truncate distribution edges. Computation lives in distribution_utils."""
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            self.prob_arr, self.p_min, self.p_max, tail_truncation, bound_type
        )
        return self._create_truncated(new_prob_arr, new_p_min, new_p_max, min_ind, max_ind)

    def with_probabilities(
        self,
        *,
        prob_arr: NDArray[np.float64],
        p_min: float,
        p_max: float,
    ) -> Self:
        """Return a new distribution on the same support with different masses.

        The support grid is preserved exactly, so ``prob_arr`` must keep its
        shape. The result is built through the subclass constructor, which
        re-runs every invariant the type declares.

        Args:
            prob_arr: Replacement finite-support masses, same shape as the current ones.
            p_min: Replacement lower-boundary mass.
            p_max: Replacement upper-boundary mass.

        Returns:
            A new distribution of the same type on the same support.
        """
        prob_arr = np.asarray(prob_arr, dtype=np.float64)
        if prob_arr.shape != self.prob_arr.shape:
            raise ValueError(
                "Replacement PMF must preserve the support shape, got "
                f"{prob_arr.shape} instead of {self.prob_arr.shape}"
            )
        # A full-range "truncation" is exactly a support-preserving rebuild: each
        # subclass already reconstructs itself through its own constructor there.
        return self._create_truncated(prob_arr, float(p_min), float(p_max), 0, prob_arr.size - 1)

    def _validate_basic(self) -> None:
        # fsum avoids floating-point accumulation error, keeping mass sum close to 1.
        pmf_sum = math.fsum(map(float, self.prob_arr))
        total_mass = pmf_sum + self.p_min + self.p_max
        mass_error = abs(total_mass - 1.0)
        if mass_error > PMF_MASS_TOL:
            raise ValueError(
                f"PMF mass does not total 1: error={mass_error:.2e} "
                f"(tolerance={PMF_MASS_TOL:.2e}), PMF sum={pmf_sum:.15f}, "
                f"min={self.p_min:.2e}, max={self.p_max:.2e}, "
                f"total mass={total_mass:.15f}"
            )

        # REALS domain: both boundaries being non-zero is not allowed.
        if self.domain == Domain.REALS and self.p_min > PMF_MASS_TOL and self.p_max > PMF_MASS_TOL:
            raise ValueError("REALS domain: p_min and p_max cannot both be non-zero")

    @abstractmethod
    def _create_truncated(
        self,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> Self:
        """Create truncated instance preserving representation semantics."""


# =============================================================================
# GENERAL (EXPLICIT) DISTRIBUTION
# =============================================================================


class SparseDiscreteDist(DiscreteDistBase):
    """General discrete distribution with explicit support values.

    Attributes:
        x_array: explicit finite support points.
        prob_arr: probability mass on finite support.
        p_min: lower-boundary mass.
        p_max: upper-boundary mass.
        domain: support-domain semantics.
    """

    def __init__(
        self,
        x_array: NDArray[np.float64],
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Initialize general discrete distribution with explicit support points."""
        self._x_arr = np.array(x_array, dtype=np.float64, copy=True)
        super().__init__(prob_arr, p_min, p_max, domain)
        self._validate_x_array()
        self._x_arr.setflags(write=False)

    @property
    def x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""
        return self._x_arr

    def _create_truncated(
        self,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> SparseDiscreteDist:
        return SparseDiscreteDist(
            x_array=self._x_arr[slice(min_ind, max_ind + 1)].copy(),
            prob_arr=new_prob_arr,
            p_min=new_p_min,
            p_max=new_p_max,
            domain=self.domain,
        )

    def _validate_x_array(self) -> None:
        if self._x_arr.ndim != 1 or self._x_arr.shape != self.prob_arr.shape:
            raise ValueError("x and PMF must be 1-D arrays of equal length")
        validate_finite_array(self._x_arr, "x support")
        if not np.all(np.diff(self._x_arr) > 0):
            raise ValueError("x must be strictly increasing")


# =============================================================================
# UNIFIED REGULAR-GRID DISTRIBUTION
# =============================================================================


class DenseDiscreteDist(DiscreteDistBase):
    """Discrete distribution on a regular (linear or geometric) grid.

    spacing_type = LINEAR:    x[i] = x_0 + i * step   (step = additive gap > 0)
    spacing_type = GEOMETRIC: x[i] = x_0 * step^i     (step = ratio > 1, x_0 > 0)

    For geometric grids the domain is always POSITIVES (x_0 > 0 enforces positivity).

    Attributes:
        x_array: materialized finite support points.
        prob_arr: probability mass on finite support.
        p_min: lower-boundary mass.
        p_max: upper-boundary mass.
        domain: support-domain semantics.
    """

    def __init__(
        self,
        x_0: float,
        step: float,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        spacing_type: SpacingType = SpacingType.LINEAR,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Initialize regular-grid discrete distribution."""
        super().__init__(prob_arr, p_min, p_max, domain)
        # The grid identity is held as a GridSpec; n is sourced from prob_arr.
        self._grid = GridSpec(
            x_0=float(x_0),
            step=float(step),
            n=self.prob_arr.size,
            spacing_type=spacing_type,
        )
        self._validate_grid()

    @property
    def grid(self) -> GridSpec:
        """Grid parameters ``(x_0, step, n, spacing_type)`` of this distribution."""
        return self._grid

    @property
    def x_0(self) -> float:
        """Grid origin: the smallest finite support point."""
        return self._grid.x_0

    @property
    def step(self) -> float:
        """Additive bin width (LINEAR) or multiplicative ratio (GEOMETRIC)."""
        return self._grid.step

    @property
    def spacing_type(self) -> SpacingType:
        """Grid spacing family."""
        return self._grid.spacing_type

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
        """Create DenseDiscreteDist from x_array by extracting x_0 and step."""
        if spacing_type == SpacingType.LINEAR:
            step = compute_bin_width(x_array)
        elif spacing_type == SpacingType.GEOMETRIC:
            step = compute_bin_ratio(x_array)
        else:
            raise ValueError(f"Unknown SpacingType: {spacing_type}")
        return cls(
            x_0=float(x_array[0]),
            step=step,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            spacing_type=spacing_type,
            domain=domain,
        )

    @property
    def x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""
        return self._grid.materialize()

    def _validate_grid(self) -> None:
        if self.spacing_type == SpacingType.LINEAR:
            if self.step <= 0.0:
                raise ValueError("step must be positive for linear grid")
        elif self.spacing_type == SpacingType.GEOMETRIC:
            if self.x_0 <= 0.0:
                raise ValueError("x_0 must be positive for geometric grid")
            if self.step <= 1.0:
                raise ValueError("step must be > 1 for geometric grid")
            if self.domain != Domain.POSITIVES:
                raise ValueError("Geometric spacing requires domain=Domain.POSITIVES")
        else:
            raise ValueError(f"Unknown SpacingType: {self.spacing_type}")
        validate_finite_real(self._grid.last_point(), "dense grid last point")

    def _create_truncated(
        self,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> "DenseDiscreteDist":
        if self.spacing_type == SpacingType.LINEAR:
            new_x_0 = self.x_0 + min_ind * self.step
        elif self.spacing_type == SpacingType.GEOMETRIC:
            new_x_0 = self.x_0 * (self.step ** float(min_ind))
        else:
            raise ValueError(f"Unknown SpacingType: {self.spacing_type}")
        return self.__class__(
            x_0=new_x_0,
            step=self.step,
            prob_arr=new_prob_arr,
            p_min=new_p_min,
            p_max=new_p_max,
            spacing_type=self.spacing_type,
            domain=self.domain,
        )


# =============================================================================
# PLD REALIZATION
# =============================================================================


class PLDRealization(DenseDiscreteDist):
    """Linear-grid PLD realization in loss space.

    Attributes:
        x_array: materialized privacy-loss support points.
        prob_arr: probability mass on finite privacy losses.
        p_min: mass at negative-infinity loss, always zero for valid realizations.
        p_max: mass at positive-infinity loss.
    """

    def __init__(
        self,
        x_0: float,
        step: float,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
    ) -> None:
        """Initialize PLD realization with privacy loss values and probabilities."""
        super().__init__(
            x_0=x_0,
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
            x_0=dist.x_0,
            step=dist.step,
            prob_arr=dist.prob_arr,
            p_max=dist.p_max,
            p_min=dist.p_min,
        )

    def truncate_edges(  # type: ignore[override]
        self, tail_truncation: float, bound_type: BoundType
    ) -> DenseDiscreteDist:
        """Trim edge mass, returning a plain dense distribution when needed.

        ``IS_DOMINATED`` truncation can move mass into ``p_min``, which violates
        the PLD-realization invariant ``p_min == 0``. In that case the result is
        intentionally downgraded to ``DenseDiscreteDist``.
        """
        if bound_type == BoundType.IS_DOMINATED:
            # IS_DOMINATED can set p_min > 0, violating PLDRealization.p_min = 0.
            # Delegate through a plain DenseDiscreteDist so the result is not a PLDRealization.
            return DenseDiscreteDist(
                x_0=self.x_0,
                step=self.step,
                prob_arr=self.prob_arr.copy(),
                p_min=self.p_min,
                p_max=self.p_max,
            ).truncate_edges(tail_truncation, bound_type)
        return super().truncate_edges(tail_truncation, bound_type)

    def _validate_pld_realization(self) -> None:
        """Validate the properties of PLD-realization.

        1. p(-inf) = 0 (p_min = 0).
        2. E[e^(-X)] <= 1.
        """
        # PLD realizations must have zero mass at negative-infinity loss.
        if self.p_min != 0.0:
            raise ValueError(f"PLD realization requires p_min = 0, got {self.p_min:.2e}")

        exp_moment_val = exp_moment_terms(prob_arr=self.prob_arr, x_vals=self.x_array)
        if np.any(np.isinf(exp_moment_val)):
            raise ValueError(
                "Exponential moment E[exp(-L)] is infinite, not a valid PLD realization"
            )
        # fsum avoids floating-point accumulation error, keeping mass sum close to 1.
        exp_moment_total = math.fsum(map(float, exp_moment_val))
        if exp_moment_total > 1.0 + REALIZATION_MOMENT_TOL:
            raise ValueError(
                f"Exponential moment E[exp(-L)] = {exp_moment_total:.15f} > 1.0, "
                "not a valid PLD realization"
            )

    def _create_truncated(
        self,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> "PLDRealization":
        """Create a truncated PLD realization while preserving linear-loss semantics."""
        del max_ind  # Unused.
        return PLDRealization(
            x_0=self.x_0 + min_ind * self.step,
            step=self.step,
            prob_arr=new_prob_arr,
            p_min=new_p_min,
            p_max=new_p_max,
        )
