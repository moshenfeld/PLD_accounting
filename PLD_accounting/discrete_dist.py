"""Discrete distributions with explicit boundary masses and structural grids.

Boundary mass is never folded into the support; ``Domain`` fixes what the two
boundaries mean. Constructors copy their arrays and mark them read-only. A dense
support is described by a ``GridSpec``, from which ``x_0``, ``step`` and
``x_array`` are derived.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from PLD_accounting.distribution_utils import (
    PMF_TOLERATED_MASS_TOL,
    compute_truncation,
    exp_moment_terms,
    signed_unit_residual,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.validation import (
    require_finite_array,
    require_finite_real,
    require_integer,
    require_nonnegative_int,
    require_nonnegative_masses,
    require_positive_int,
    require_positive_real,
)

# Noise floor of ``1 - E[exp(-L)]``. A one-ulp shift in a loss moves its term by ``|x| eps``,
# so the residual moves by about ``eps * E_dual[|L|]``.
REALIZATION_MOMENT_TOL = float(16 * np.finfo(np.float64).eps)
_INT64_MIN = int(np.iinfo(np.int64).min)
_INT64_MAX = int(np.iinfo(np.int64).max)


class Domain(Enum):
    """Domain of a discrete distribution's support."""

    REALS = "reals"  # p_min = mass at −∞, p_max = mass at +∞
    POSITIVES = "positives"  # p_min = mass at 0,  p_max = mass at +∞


@dataclass(frozen=True, kw_only=True)
class GridSpec:
    """Immutable regular lattice, described by its parameters rather than its points.

    Point ``i`` of ``n`` is ``anchor + k * step`` on a linear lattice and
    ``anchor * exp(k * step)`` on a geometric one, where ``k = index_0 + i`` is the
    lattice index of that point and ``i`` is its position in the materialized array.

    Equality is structural, so lattice operations are integer-index arithmetic and
    retained coordinates never drift.

    Attributes:
        step: Positive spacing. Additive bin width when linear; the **log** ratio when
            geometric, so both families share one index formula. For ratio-shaped
            geometric lattices, use ``GridSpec.geometric``.
        n: Number of points, at least one.
        spacing_type: Selects which point formula applies.
        anchor: Point at lattice index zero. Additive when linear, multiplicative and
            strictly positive when geometric.
        index_0: Lattice index of the first point; may be negative. Slicing changes
            this, not ``anchor``.
    """

    step: float
    n: int
    spacing_type: SpacingType = SpacingType.LINEAR
    anchor: float = 0.0
    index_0: int = 0

    def __post_init__(self) -> None:
        """Reject a parameter set that does not describe a usable lattice.

        Both formulas are monotone in ``k``, so only the first and last cells can
        collapse and only those are checked.
        """
        require_integer(value=[self.n, self.index_0], name=["GridSpec n", "GridSpec index_0"])
        require_positive_int(value=self.n, name="GridSpec n")
        require_positive_real(value=self.step, name="GridSpec step")
        if self.spacing_type == SpacingType.GEOMETRIC:
            require_positive_real(value=self.anchor, name="GridSpec anchor")
        else:
            require_finite_real(value=self.anchor, name="GridSpec anchor")
        last_index = int(self.index_0) + int(self.n) - 1
        if not _INT64_MIN <= int(self.index_0) <= _INT64_MAX or not (
            _INT64_MIN <= last_index <= _INT64_MAX
        ):
            raise ValueError(f"GridSpec index range [{self.index_0}, {last_index}] leaves int64")
        if self.spacing_type not in (SpacingType.LINEAR, SpacingType.GEOMETRIC):
            raise ValueError(f"Unknown SpacingType: {self.spacing_type}")
        if self.n == 1:
            # A singleton has no cell, so ``point(1)`` is off the declared grid; evaluating
            # it can overflow on a lattice whose one point is perfectly representable.
            require_finite_real(value=self.point(0), name="GridSpec point")
            return
        check_indices = (0,) if self.n == 2 else (0, self.n - 2)
        for i in check_indices:
            low, high = self.point(i), self.point(i + 1)
            require_finite_real(value=[low, high], name=["GridSpec point", "GridSpec point"])
            if not low < high:
                raise ValueError(
                    f"GridSpec must be strictly increasing, got {low!r} then {high!r} "
                    f"at index {i}"
                )

    @classmethod
    def geometric(cls, *, ratio: float, n: int, anchor: float, index_0: int = 0) -> GridSpec:
        """Build a geometric lattice ``anchor * ratio ** (index_0 + i)``.

        Takes the multiplicative ratio and stores its log, so a caller thinking in
        ratios cannot silently build a lattice whose spacing is off by an ``exp``.
        """
        if ratio <= 1.0:
            raise ValueError(f"geometric ratio must be > 1, got {ratio}")
        return cls(
            step=math.log(ratio),
            n=n,
            spacing_type=SpacingType.GEOMETRIC,
            anchor=anchor,
            index_0=index_0,
        )

    # -- Materialization ---------------------------------------------------

    def point(self, i: int) -> float:
        """Return point ``i`` alone, through the formula ``materialize`` uses.

        Agrees bitwise with ``materialize()[i]``; the two must not drift apart.
        """
        offset = float(self.index_0 + i) * self.step
        if self.spacing_type == SpacingType.LINEAR:
            return self.anchor + offset
        return self.anchor * math.exp(offset)

    @property
    def x_0(self) -> float:
        """First grid point."""
        return self.point(0)

    @property
    def last_point(self) -> float:
        """Return the last point through the same formula as ``materialize``.

        Keeping the scalar and array paths identical prevents endpoint decisions from
        disagreeing by one ULP.
        """
        return self.point(self.n - 1)

    def materialize(self) -> NDArray[np.float64]:
        """Return all ``n`` points as an array.

        Lattice indices are formed in int64 before any float conversion, so index
        arithmetic never loses precision to the coordinate magnitude.
        """
        k = (self.index_0 + np.arange(self.n, dtype=np.int64)).astype(np.float64)
        if self.spacing_type == SpacingType.LINEAR:
            return self.anchor + k * self.step
        return self.anchor * np.exp(k * self.step)

    # -- Structure-preserving operations -----------------------------------

    def slice(self, *, start: int, n: int) -> GridSpec:
        """Return the child grid retaining ``[start, start + n)`` points."""
        if start < 0 or n < 1 or start + n > self.n:
            raise ValueError(f"Invalid slice: start={start}, n={n}, parent_n={self.n}")
        return replace(self, index_0=self.index_0 + start, n=n)

    def pad(self, *, left: int, right: int) -> GridSpec:
        """Return the grid extended by ``left`` points below and ``right`` above."""
        require_nonnegative_int(value=[left, right], name=["left", "right"])
        return replace(self, index_0=self.index_0 - left, n=self.n + left + right)

    def with_n(self, n: int) -> GridSpec:
        """Return the same lattice with a different point count."""
        return replace(self, n=n)

    def reflect(self) -> GridSpec:
        """Return the lattice of ``-x``, still in increasing order.

        Negation is exact and rounding is symmetric about zero, so reflecting twice
        returns the original coordinates bitwise.
        """
        if self.spacing_type != SpacingType.LINEAR:
            raise ValueError(f"reflect requires a linear grid, got {self.spacing_type}")
        return replace(self, anchor=-self.anchor, index_0=-(self.index_0 + self.n - 1))

    def exp(self) -> GridSpec:
        """Map a linear lattice to the geometric lattice of ``exp(x)``.

        Spacing is unchanged. Coordinates match ``np.exp`` of the linear support only
        when ``anchor == 0``; an affine grid uses ``exp(anchor) * exp(k * step)``.
        """
        if self.spacing_type != SpacingType.LINEAR:
            raise ValueError(f"exp requires a linear grid, got {self.spacing_type}")
        anchor = 1.0 if self.anchor == 0.0 else math.exp(self.anchor)
        return replace(self, spacing_type=SpacingType.GEOMETRIC, anchor=anchor)

    def log(self) -> GridSpec:
        """Map a geometric lattice to the linear lattice of ``log(x)``.

        Inverse of ``exp``. Exact on a unit anchor, where ``log(1)`` is zero.
        """
        if self.spacing_type != SpacingType.GEOMETRIC:
            raise ValueError(f"log requires a geometric grid, got {self.spacing_type}")
        anchor = 0.0 if self.anchor == 1.0 else math.log(self.anchor)
        return replace(self, spacing_type=SpacingType.LINEAR, anchor=anchor)

    def convolve(self, other: GridSpec) -> GridSpec:
        """Return the support lattice of ``X + Y`` for two linear lattices."""
        if self.spacing_type != SpacingType.LINEAR or other.spacing_type != SpacingType.LINEAR:
            raise ValueError(
                f"convolve requires linear grids, got {self.spacing_type} and {other.spacing_type}"
            )
        if self.step != other.step:
            raise ValueError(
                f"Convolution requires one exact step: {self.step:.17g} vs {other.step:.17g}"
            )
        return replace(
            self,
            anchor=self.anchor + other.anchor,
            index_0=self.index_0 + other.index_0,
            n=self.n + other.n - 1,
        )

    def self_convolve(self, num_convolutions: int) -> GridSpec:
        """Return the support lattice of a ``num_convolutions``-fold self-sum."""
        if self.spacing_type != SpacingType.LINEAR:
            raise ValueError(f"self_convolve requires a linear grid, got {self.spacing_type}")
        require_positive_int(value=num_convolutions, name="num_convolutions")
        return replace(
            self,
            anchor=self.anchor * num_convolutions,
            index_0=self.index_0 * num_convolutions,
            n=num_convolutions * (self.n - 1) + 1,
        )


# =============================================================================
# ABSTRACT BASE
# =============================================================================


class DiscreteDistBase(ABC):
    """Discrete PMF with finite-support masses and domain boundary masses."""

    def __init__(
        self,
        *,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Store validated masses, copying ``prob_arr`` and marking it read-only."""
        require_nonnegative_masses(
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
        )

        self._prob_arr = np.array(prob_arr, dtype=np.float64, copy=True)
        self._p_min = float(p_min)
        self._p_max = float(p_max)
        self._domain = domain
        self._require_unit_mass()
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

    def truncate_edges(self, *, tail_truncation: float, bound_type: BoundType) -> Self:
        """Return a copy with up to ``tail_truncation`` mass removed from each edge.

        Removed mass moves to whichever boundary the bound direction makes
        conservative, so the result still bounds the same quantity.
        """
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=self.prob_arr,
            p_min=self.p_min,
            p_max=self.p_max,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
        return self._rebuild_on_range(
            new_prob_arr=new_prob_arr,
            new_p_min=new_p_min,
            new_p_max=new_p_max,
            min_ind=min_ind,
            max_ind=max_ind,
        )

    def with_probabilities(
        self,
        *,
        prob_arr: NDArray[np.float64],
        p_min: float,
        p_max: float,
    ) -> Self:
        """Return a copy on the same support with replacement masses.

        The support shape must remain unchanged. Rebuilding through the concrete
        subclass re-runs its mass and representation invariants.
        """
        prob_arr = np.asarray(prob_arr, dtype=np.float64)
        if prob_arr.shape != self.prob_arr.shape:
            raise ValueError(f"{prob_arr.shape} does not match support shape {self.prob_arr.shape}")
        return self._rebuild_on_range(
            new_prob_arr=prob_arr,
            new_p_min=float(p_min),
            new_p_max=float(p_max),
            min_ind=0,
            max_ind=prob_arr.size - 1,
        )

    def _require_unit_mass(self) -> None:
        """Reject masses that do not total one, or a REALS law with mass at both boundaries."""
        mass_error = abs(
            signed_unit_residual(
                values=self.prob_arr,
                lower_term=self.p_min,
                upper_term=self.p_max,
            )
        )
        if mass_error > PMF_TOLERATED_MASS_TOL:
            pmf_sum = math.fsum(map(float, self.prob_arr))
            raise ValueError(
                f"PMF mass does not total 1: error={mass_error:.2e} "
                f"(tolerance={PMF_TOLERATED_MASS_TOL:.2e}), PMF sum={pmf_sum:.15f}, "
                f"min={self.p_min:.2e}, max={self.p_max:.2e}, "
                f"total mass={pmf_sum + self.p_min + self.p_max:.15f}"
            )

        # Opposing infinite atoms would make later real-domain sums encounter ``-inf + inf``.
        if (
            self.domain == Domain.REALS
            and self.p_min > PMF_TOLERATED_MASS_TOL
            and self.p_max > PMF_TOLERATED_MASS_TOL
        ):
            raise ValueError("REALS domain: p_min and p_max cannot both be non-zero")

    @abstractmethod
    def _rebuild_on_range(
        self,
        *,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> Self:
        """Rebuild this distribution on ``support[min_ind : max_ind + 1]``.

        The one construction hook each representation implements, so mass-changing
        operations need not know how the support is stored. The range is the mass that
        survived: empty edge cells are always dropped, tail truncation may drop more,
        and a caller replacing masses in place passes the full range.
        """


# =============================================================================
# GENERAL (EXPLICIT) DISTRIBUTION
# =============================================================================


class SparseDiscreteDist(DiscreteDistBase):
    """Discrete distribution with an explicit, strictly increasing support."""

    def __init__(
        self,
        *,
        x_array: NDArray[np.float64],
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Store an explicit support and its masses, both copied and marked read-only."""
        self._x_arr = np.array(x_array, dtype=np.float64, copy=True)
        super().__init__(prob_arr=prob_arr, p_min=p_min, p_max=p_max, domain=domain)
        self._require_x_array()
        self._x_arr.setflags(write=False)

    @property
    def x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""
        return self._x_arr

    def _rebuild_on_range(
        self,
        *,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> SparseDiscreteDist:
        """Rebuild on the sliced explicit support."""
        return SparseDiscreteDist(
            x_array=self._x_arr[slice(min_ind, max_ind + 1)].copy(),
            prob_arr=new_prob_arr,
            p_min=new_p_min,
            p_max=new_p_max,
            domain=self.domain,
        )

    def _require_x_array(self) -> None:
        """Reject a support that is not a 1-D, finite, strictly increasing match for the PMF."""
        if self._x_arr.ndim != 1 or self._x_arr.shape != self.prob_arr.shape:
            raise ValueError("x and PMF must be 1-D arrays of equal length")
        require_finite_array(values=self._x_arr, name="x support")
        if not np.all(np.diff(self._x_arr) > 0):
            raise ValueError("x must be strictly increasing")


# =============================================================================
# UNIFIED REGULAR-GRID DISTRIBUTION
# =============================================================================


class DenseDiscreteDist(DiscreteDistBase):
    """Regular linear or geometric grid distribution. Geometric grids are POSITIVES."""

    def __init__(
        self,
        *,
        grid: GridSpec,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Initialize a regular-grid discrete distribution on ``grid``."""
        super().__init__(prob_arr=prob_arr, p_min=p_min, p_max=p_max, domain=domain)
        if grid.n != self.prob_arr.size:
            raise ValueError(
                f"GridSpec n={grid.n} does not match prob_arr size {self.prob_arr.size}"
            )
        self._grid = grid
        self._require_grid()

    @property
    def grid(self) -> GridSpec:
        """Structural lattice of this distribution."""
        return self._grid

    @property
    def x_0(self) -> float:
        """Grid origin: the smallest finite support point."""
        return self._grid.x_0

    @property
    def step(self) -> float:
        """Lattice spacing: additive bin width when linear, log ratio when geometric."""
        return self._grid.step

    @property
    def spacing_type(self) -> SpacingType:
        """Grid spacing family."""
        return self._grid.spacing_type

    @property
    def x_array(self) -> NDArray[np.float64]:
        """Return materialized support points."""
        return self._grid.materialize()

    def _require_grid(self) -> None:
        """Reject grid/domain pairings the package does not define.

        A geometric lattice is positive, so it can only carry POSITIVES semantics.
        """
        if self.spacing_type == SpacingType.GEOMETRIC and self.domain != Domain.POSITIVES:
            raise ValueError("Geometric spacing requires domain=Domain.POSITIVES")
        require_finite_real(value=self._grid.last_point, name="dense grid last point")

    def _rebuild_on_range(
        self,
        *,
        new_prob_arr: NDArray[np.float64],
        new_p_min: float,
        new_p_max: float,
        min_ind: int,
        max_ind: int,
    ) -> Self:
        """Rebuild on the sliced lattice, which the grid derives from ``min_ind``."""
        del max_ind  # Unused; child length is encoded in ``new_prob_arr``.
        return type(self)(
            grid=self._grid.slice(start=min_ind, n=new_prob_arr.size),
            prob_arr=new_prob_arr,
            p_min=new_p_min,
            p_max=new_p_max,
            domain=self.domain,
        )


# =============================================================================
# PLD REALIZATION
# =============================================================================


class PLDRealization(DenseDiscreteDist):
    """Linear-grid PLD realization: ``p_min = 0`` and ``E[exp(-L)] <= 1``."""

    def __init__(
        self,
        *,
        grid: GridSpec,
        prob_arr: NDArray[np.float64],
        p_min: float = 0.0,
        p_max: float = 0.0,
        domain: Domain = Domain.REALS,
    ) -> None:
        """Initialize a linear-grid PLD realization.

        ``domain`` is accepted so base-class operations can rebuild a realization
        through one signature, but only ``Domain.REALS`` is a realization.
        """
        if grid.spacing_type != SpacingType.LINEAR:
            raise ValueError(f"PLDRealization requires a linear grid, got {grid.spacing_type}")
        if domain != Domain.REALS:
            raise ValueError(f"PLDRealization requires Domain.REALS, got {domain}")
        super().__init__(
            grid=grid,
            prob_arr=prob_arr,
            p_min=float(p_min),
            p_max=float(p_max),
            domain=Domain.REALS,
        )
        self._require_pld_realization()

    def truncate_edges(  # type: ignore[override]
        self, *, tail_truncation: float, bound_type: BoundType
    ) -> DenseDiscreteDist:
        """Trim edge mass, returning a plain dense distribution when needed.

        ``IS_DOMINATED`` truncation can move mass into ``p_min``, which violates
        the PLD-realization invariant ``p_min == 0``. In that case the result is
        intentionally downgraded to ``DenseDiscreteDist``.
        """
        if bound_type == BoundType.IS_DOMINATED:
            return DenseDiscreteDist(
                grid=self._grid,
                prob_arr=self.prob_arr.copy(),
                p_min=self.p_min,
                p_max=self.p_max,
            ).truncate_edges(tail_truncation=tail_truncation, bound_type=bound_type)
        return super().truncate_edges(tail_truncation=tail_truncation, bound_type=bound_type)

    def _require_pld_realization(self) -> None:
        """Require ``p_min = 0`` and ``E[exp(-L)] <= 1``."""
        if self.p_min != 0.0:
            raise ValueError(f"PLD realization requires p_min = 0, got {self.p_min:.2e}")

        exp_moment_val = exp_moment_terms(prob_arr=self.prob_arr, x_vals=self.x_array)
        if np.any(np.isinf(exp_moment_val)):
            raise ValueError(
                "Exponential moment E[exp(-L)] is infinite, not a valid PLD realization"
            )
        moment_residual = signed_unit_residual(
            values=exp_moment_val, lower_term=0.0, upper_term=0.0
        )
        if moment_residual < -REALIZATION_MOMENT_TOL:
            raise ValueError(
                f"Exponential moment E[exp(-L)] = {1.0 - moment_residual:.15f} > 1.0, "
                "not a valid PLD realization"
            )


def require_dense_dist(
    *,
    dist: object,
    name: str,
    spacing: SpacingType | None = None,
    domain: Domain | None = None,
) -> DenseDiscreteDist:
    """Require a dense distribution, optionally of one spacing and domain."""
    if not isinstance(dist, DenseDiscreteDist):
        spacing_note = f" with {spacing.name} spacing" if spacing is not None else ""
        raise TypeError(
            f"{name} must be DenseDiscreteDist{spacing_note}, got {type(dist).__name__}"
        )
    if spacing is not None and dist.spacing_type != spacing:
        raise TypeError(
            f"{name}: expected DenseDiscreteDist with {spacing.name} spacing, "
            f"got {type(dist).__name__} with spacing {dist.spacing_type}"
        )
    if domain is not None and dist.domain != domain:
        raise ValueError(f"{name} must use Domain.{domain.name}, got {dist.domain}")
    return dist


def require_linear_reals_dist(*, dist: object, name: str) -> DenseDiscreteDist:
    """Require a linear-grid real-domain dense distribution."""
    return require_dense_dist(dist=dist, name=name, spacing=SpacingType.LINEAR, domain=Domain.REALS)


def require_zero_anchor(*, grid: GridSpec, name: str) -> GridSpec:
    """Require a linear lattice on whole multiples of its step, i.e. ``anchor == 0``.

    A loss lattice is ``k * step``; the offset of its first point belongs in ``index_0``.
    A non-zero anchor is either an exp-space carrier grid, which does not belong here, or
    an index written into the wrong field.
    """
    if grid.spacing_type != SpacingType.LINEAR:
        raise ValueError(f"{name} must be a linear grid, got {grid.spacing_type}")
    if grid.anchor != 0.0:
        raise ValueError(
            f"{name} must lie on whole multiples of its step (anchor == 0), got "
            f"anchor={grid.anchor!r}, step={grid.step!r}; carry the offset in index_0"
        )
    return grid


def require_geometric_positives_dist(*, dist: object, name: str) -> DenseDiscreteDist:
    """Require a geometric-grid positive-domain dense distribution."""
    return require_dense_dist(
        dist=dist, name=name, spacing=SpacingType.GEOMETRIC, domain=Domain.POSITIVES
    )
