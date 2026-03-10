"""
Discrete distribution classes and structured grid data types.

Class hierarchy:
- DiscreteDistBase: abstract base for all discrete distributions
- GeneralDiscreteDist: arbitrary explicit support
- LinearDiscreteDist: linear grid x[i] = x_min + i * x_gap
- GeometricDiscreteDist: geometric grid x[i] = x_min * ratio^i
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from PLD_accounting.core_utils import (
    PMF_MASS_TOL,
    enforce_mass_conservation,
)
from PLD_accounting.types import BoundType


# =============================================================================
# ABSTRACT BASE
# =============================================================================


class DiscreteDistBase(ABC):
    """
    Abstract base for discrete PMF representations with optional infinite masses.

    Attributes:
        PMF_array: probability mass on finite support
        p_neg_inf: mass at -infinity
        p_pos_inf: mass at +infinity
    """

    def __init__(
        self,
        PMF_array: NDArray[np.float64],
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0,
    ) -> None:
        self.PMF_array = np.asarray(PMF_array, dtype=np.float64)
        self.p_neg_inf = float(p_neg_inf)
        self.p_pos_inf = float(p_pos_inf)
        self._validate_basic()

    @abstractmethod
    def get_x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""

    @abstractmethod
    def get_support_size(self) -> int:
        """Return support size (finite bins)."""

    @property
    def x_array(self) -> NDArray[np.float64]:
        """Materialized support."""
        return self.get_x_array()

    def _validate_basic(self) -> None:
        if self.PMF_array.ndim != 1:
            raise ValueError("PMF must be 1-D array")
        if self.PMF_array.size < 2:
            raise ValueError("x and PMF must contain at least 2 points")
        if np.any(self.PMF_array < -PMF_MASS_TOL):
            raise ValueError("PMF must be nonnegative")
        if self.p_neg_inf < 0:
            raise ValueError("p_neg_inf must be nonnegative")
        if self.p_pos_inf < 0:
            raise ValueError("p_pos_inf must be nonnegative")

    def validate_mass_conservation(self, bound_type: BoundType) -> Self:
        """Validate total mass and bound semantics."""
        self._validate_basic()

        pmf_sum = math.fsum(map(float, self.PMF_array))
        total_mass = pmf_sum + self.p_neg_inf + self.p_pos_inf
        mass_error = abs(total_mass - 1.0)
        if mass_error > PMF_MASS_TOL:
            error_msg = "MASS CONSERVATION ERROR"
            error_msg += f": Error={mass_error:.2e} (tolerance={PMF_MASS_TOL:.2e})"
            error_msg += f", PMF sum={pmf_sum:.15f}"
            error_msg += f", p_neg_inf={self.p_neg_inf:.2e}"
            error_msg += f", p_pos_inf={self.p_pos_inf:.2e}"
            error_msg += f", Total mass={total_mass:.15f}"
            raise ValueError(error_msg)

        if pmf_sum <= PMF_MASS_TOL and (self.p_neg_inf + self.p_pos_inf) >= 1.0 - PMF_MASS_TOL:
            raise ValueError("Distributions with all mass at infinity are not supported")

        if bound_type == BoundType.DOMINATES and self.p_neg_inf > 0:
            raise ValueError("DOMINATES bound_type requires p_neg_inf=0")
        if bound_type == BoundType.IS_DOMINATED and self.p_pos_inf > 0:
            raise ValueError("IS_DOMINATED bound_type requires p_pos_inf=0")

        return self

    def truncate_edges(self, tail_truncation: float, bound_type: BoundType) -> Self:
        """Truncate finite tails and move truncated mass according to bound semantics."""
        if tail_truncation == 0.0:
            nonzero_indices = np.nonzero(self.PMF_array)[0]
            if nonzero_indices.size == 0:
                raise ValueError("Cannot truncate distribution with zero finite mass")
            min_ind = int(nonzero_indices[0])
            max_ind = int(nonzero_indices[-1])
        else:
            cumsum_left = np.cumsum(self.PMF_array, dtype=np.float64)
            cumsum_right = np.cumsum(self.PMF_array[::-1], dtype=np.float64)
            min_ind = int(np.searchsorted(cumsum_left, tail_truncation, side="right"))
            right_cnt = int(np.searchsorted(cumsum_right, tail_truncation, side="right"))
            max_ind = self.PMF_array.size - 1 - right_cnt

        if min_ind == 0 and max_ind == self.PMF_array.size - 1:
            return self

        if min_ind > max_ind:
            finite_mass = math.fsum(map(float, self.PMF_array))
            raise ValueError(
                f"Cannot truncate {tail_truncation:.2e} from each tail: "
                f"finite mass ({finite_mass:.2e}) < 2*tail_truncation ({2*tail_truncation:.2e})"
            )

        if max_ind - min_ind < 1:
            if min_ind > 0:
                min_ind -= 1
            elif max_ind < self.PMF_array.size - 1:
                max_ind += 1
            else:
                raise ValueError("Cannot truncate to fewer than 2 bins")

        left_mass = math.fsum(map(float, self.PMF_array[:min_ind]))
        right_mass = math.fsum(map(float, self.PMF_array[max_ind + 1 :]))

        new_PMF = self.PMF_array[min_ind : max_ind + 1].copy()

        if bound_type == BoundType.DOMINATES:
            new_PMF[0] += left_mass
            new_p_pos_inf = self.p_pos_inf + right_mass
            new_p_neg_inf = self.p_neg_inf
        elif bound_type == BoundType.IS_DOMINATED:
            new_p_neg_inf = self.p_neg_inf + left_mass
            new_PMF[-1] += right_mass
            new_p_pos_inf = self.p_pos_inf
        else:
            raise ValueError(f"Unknown BoundType: {bound_type}")

        new_PMF, new_p_neg_inf, new_p_pos_inf = enforce_mass_conservation(
            PMF_array=new_PMF,
            expected_neg_inf=new_p_neg_inf,
            expected_pos_inf=new_p_pos_inf,
            bound_type=bound_type,
        )

        return self._create_truncated(new_PMF, new_p_neg_inf, new_p_pos_inf, min_ind, max_ind)

    @abstractmethod
    def _create_truncated(
        self,
        new_PMF: NDArray[np.float64],
        new_p_neg_inf: float,
        new_p_pos_inf: float,
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


class GeneralDiscreteDist(DiscreteDistBase):
    """General discrete distribution with explicit support values."""

    def __init__(
        self,
        x_array: NDArray[np.float64],
        PMF_array: NDArray[np.float64],
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0,
    ) -> None:
        self._x_array = np.asarray(x_array, dtype=np.float64)
        super().__init__(PMF_array, p_neg_inf, p_pos_inf)
        self._validate_x_array()

    def _validate_x_array(self) -> None:
        if self._x_array.ndim != 1 or self._x_array.shape != self.PMF_array.shape:
            raise ValueError("x and PMF must be 1-D arrays of equal length")
        if self._x_array.size < 2:
            raise ValueError("x and PMF must contain at least 2 points")
        if not np.all(np.diff(self._x_array) > 0):
            raise ValueError("x must be strictly increasing")

    def get_x_array(self) -> NDArray[np.float64]:
        return self._x_array

    def get_support_size(self) -> int:
        return self._x_array.size

    def _create_truncated(
        self,
        new_PMF: NDArray[np.float64],
        new_p_neg_inf: float,
        new_p_pos_inf: float,
        min_ind: int,
        max_ind: int,
    ) -> GeneralDiscreteDist:
        return GeneralDiscreteDist(
            x_array=self._x_array[min_ind : max_ind + 1],
            PMF_array=new_PMF,
            p_neg_inf=new_p_neg_inf,
            p_pos_inf=new_p_pos_inf,
        )

    def copy(self) -> GeneralDiscreteDist:
        return GeneralDiscreteDist(
            x_array=self._x_array.copy(),
            PMF_array=self.PMF_array.copy(),
            p_neg_inf=self.p_neg_inf,
            p_pos_inf=self.p_pos_inf,
        )


# =============================================================================
# LINEAR DISTRIBUTIONS
# =============================================================================


class LinearDiscreteDist(DiscreteDistBase):
    """Linear grid x[i] = x_min + i * x_gap."""

    def __init__(
        self,
        x_min: float,
        x_gap: float,
        PMF_array: NDArray[np.float64],
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0,
    ) -> None:
        self.x_min = float(x_min)
        self.x_gap = float(x_gap)
        if self.x_gap <= 0.0:
            raise ValueError("x_gap must be positive")
        super().__init__(PMF_array, p_neg_inf, p_pos_inf)

    @classmethod
    def from_x_array(
        cls,
        x_array: NDArray[np.float64],
        PMF_array: NDArray[np.float64],
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0,
    ) -> "LinearDiscreteDist":
        """Create LinearDiscreteDist from x_array by extracting x_min and x_gap.

        Args:
            x_array: Support values (must be uniformly spaced)
            PMF_array: Probability masses
            p_neg_inf: Mass at negative infinity
            p_pos_inf: Mass at positive infinity

        Returns:
            LinearDiscreteDist instance

        Raises:
            ValueError: If x_array doesn't have uniform spacing
        """
        from PLD_accounting.core_utils import compute_bin_width

        x_gap = compute_bin_width(x_array)
        return cls(
            x_min=float(x_array[0]),
            x_gap=x_gap,
            PMF_array=PMF_array,
            p_neg_inf=p_neg_inf,
            p_pos_inf=p_pos_inf,
        )

    def get_x_array(self) -> NDArray[np.float64]:
        return self.x_min + np.arange(self.PMF_array.size, dtype=np.float64) * self.x_gap

    def get_support_size(self) -> int:
        return self.PMF_array.size

    def _create_truncated(
        self,
        new_PMF: NDArray[np.float64],
        new_p_neg_inf: float,
        new_p_pos_inf: float,
        min_ind: int,
        max_ind: int,
    ) -> "LinearDiscreteDist":
        return LinearDiscreteDist(
            x_min=self.x_min + min_ind * self.x_gap,
            x_gap=self.x_gap,
            PMF_array=new_PMF,
            p_neg_inf=new_p_neg_inf,
            p_pos_inf=new_p_pos_inf,
        )

    def copy(self) -> "LinearDiscreteDist":
        return LinearDiscreteDist(
            x_min=self.x_min,
            x_gap=self.x_gap,
            PMF_array=self.PMF_array.copy(),
            p_neg_inf=self.p_neg_inf,
            p_pos_inf=self.p_pos_inf,
        )


# =============================================================================
# GEOMETRIC DISTRIBUTIONS
# =============================================================================


class GeometricDiscreteDist(DiscreteDistBase):
    """Geometric grid x[i] = x_min * ratio^i."""

    def __init__(
        self,
        x_min: float,
        ratio: float,
        PMF_array: NDArray[np.float64],
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0,
    ) -> None:
        self.x_min = float(x_min)
        self.ratio = float(ratio)
        if self.x_min <= 0.0:
            raise ValueError("x_min must be positive for geometric grid")
        if self.ratio <= 1.0:
            raise ValueError("ratio must be > 1 for geometric grid")
        super().__init__(PMF_array, p_neg_inf, p_pos_inf)

    @classmethod
    def from_x_array(
        cls,
        x_array: NDArray[np.float64],
        PMF_array: NDArray[np.float64],
        p_neg_inf: float = 0.0,
        p_pos_inf: float = 0.0,
    ) -> "GeometricDiscreteDist":
        """Create GeometricDiscreteDist from x_array by extracting x_min and ratio.

        Args:
            x_array: Support values (must be geometrically spaced)
            PMF_array: Probability masses
            p_neg_inf: Mass at negative infinity
            p_pos_inf: Mass at positive infinity

        Returns:
            GeometricDiscreteDist instance

        Raises:
            ValueError: If x_array doesn't have geometric spacing
        """
        from PLD_accounting.core_utils import compute_bin_ratio

        ratio = compute_bin_ratio(x_array)
        return cls(
            x_min=float(x_array[0]),
            ratio=ratio,
            PMF_array=PMF_array,
            p_neg_inf=p_neg_inf,
            p_pos_inf=p_pos_inf,
        )

    def get_x_array(self) -> NDArray[np.float64]:
        return self.x_min * np.power(self.ratio, np.arange(self.PMF_array.size, dtype=np.float64))

    def get_support_size(self) -> int:
        return self.PMF_array.size

    def _create_truncated(
        self,
        new_PMF: NDArray[np.float64],
        new_p_neg_inf: float,
        new_p_pos_inf: float,
        min_ind: int,
        max_ind: int,
    ) -> "GeometricDiscreteDist":
        return GeometricDiscreteDist(
            x_min=self.x_min * np.power(self.ratio, float(min_ind)),
            ratio=self.ratio,
            PMF_array=new_PMF,
            p_neg_inf=new_p_neg_inf,
            p_pos_inf=new_p_pos_inf,
        )

    def copy(self) -> "GeometricDiscreteDist":
        return GeometricDiscreteDist(
            x_min=self.x_min,
            ratio=self.ratio,
            PMF_array=self.PMF_array.copy(),
            p_neg_inf=self.p_neg_inf,
            p_pos_inf=self.p_pos_inf,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DiscreteDistBase",
    "GeneralDiscreteDist",
    "LinearDiscreteDist",
    "GeometricDiscreteDist",
]
