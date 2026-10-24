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
)

# Defaults for AllocationSchemeConfig (independent of REALIZATION_MOMENT_TOL;
# tail budget is a modeling choice).
DEFAULT_LOSS_DISCRETIZATION = 1e-2
DEFAULT_TAIL_TRUNCATION = 1e-12
# Resource limits of ConvolutionMethod.MS_FFT: grid points per octave, and points of
# the uniform loss grid that one allocation returns.
DEFAULT_MAX_CELLS_PER_BAND = 1 << 18
DEFAULT_MAX_CLOSING_GRID = 1_000_000
# Smallest cap on the points of a zero-anchored linear grid: one interval of the step
# covers the range, and the alignment to multiples of the step adds up to two points.
MIN_GRID_POINT_CAP = 4


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
    """Convolution method of the allocation routes.

    ``GEOM`` sums likelihood ratios on a geometric grid by direct convolution.
    ``MS_FFT`` sums them on a dyadic grid, uniform inside each octave, by FFTs
    inside each octave.
    """

    GEOM = "geometric"
    MS_FFT = "ms_fft"


class ProjectionMethod(Enum):
    """Projection of a sum of two likelihood ratios back onto the grid at a merge.

    ``STOCH_DOM`` rounds the sum to the neighbouring grid point the bound requires.
    ``CTD`` (connect-the-dots) splits it between both neighbours so that its mean is
    kept, which gives upper bounds only. The method also selects the truncation of
    the tails at every stage after the one-step input. The one-step input itself is
    projected by the bound type: connect-the-dots for an upper bound and rounding
    for a lower bound.
    """

    STOCH_DOM = "stochastic_domination"
    CTD = "connect_the_dots"


class Direction(Enum):
    """Enum for direction of privacy analysis."""

    ADD = "add"
    REMOVE = "remove"
    BOTH = "both"


def require_bound_type(*, value: object, name: str = "bound_type") -> None:
    """Require ``DOMINATES`` or ``IS_DOMINATED``.

    ``BOTH`` is a real member, but public operations take one orientation.
    """
    require_enum(
        value=value,
        enum_cls=BoundType,
        name=name,
        allowed=(BoundType.DOMINATES, BoundType.IS_DOMINATED),
    )


def require_direction(*, value: object, name: str = "direction") -> None:
    """Require ``ADD`` or ``REMOVE``.

    ``BOTH`` is a real member, but public operations take one orientation.
    """
    require_enum(
        value=value,
        enum_cls=Direction,
        name=name,
        allowed=(Direction.ADD, Direction.REMOVE),
    )


def require_projection_method(*, value: object, bound_type: BoundType) -> None:
    """Require a projection compatible with the requested bound.

    Rounding supports either single bound. Connect-the-dots supports
    ``BoundType.DOMINATES`` only. Operations that do not take a bound, such as
    stage counting, validate the enum alone with ``require_enum``.
    """
    require_bound_type(value=bound_type)
    require_enum(value=value, enum_cls=ProjectionMethod, name="projection_method")
    if value == ProjectionMethod.CTD and bound_type == BoundType.IS_DOMINATED:
        raise ValueError(
            "connect-the-dots supports BoundType.DOMINATES only "
            f"(dominating bounds only), got {bound_type}"
        )


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

    def get_delta(self) -> float:
        """Return ``delta``, raising if it is unset."""
        if self.delta is None:
            raise ValueError("delta must be in (0, 1), got None")
        return self.delta

    def get_epsilon(self) -> float:
        """Return ``epsilon``, raising if it is unset."""
        if self.epsilon is None:
            raise ValueError("epsilon must be positive, got None")
        return self.epsilon


@dataclass(frozen=True, kw_only=True)
class AllocationSchemeConfig:
    """Configuration for privacy schemes.

    Attributes:
        loss_discretization: Nominal loss-resolution budget shared by the
            discretizing stages. ``GEOM`` uses fixed loss spacing; ``MS_FFT``
            matches mean loss spacing within each octave. When caps and input-grid
            resolution do not coarsen the requested stages, their worst-case
            accumulated grid displacements are at most this budget and this budget
            divided by ``log(2)``, respectively. This is
            not a bound on the error of an epsilon query. Explicit input
            realizations retain their own prior approximation error.
        tail_truncation: Probability mass that truncation may move over a whole
            computation.
        max_grid_mult: ``GEOM`` only. Largest number of points of a one-step factor
            grid; a binding cap coarsens the step without a warning. Any value
            ``<= 0`` means no cap. A positive value must be at least
            ``MIN_GRID_POINT_CAP`` (4); construction rejects 1 to 3.
        convolution_method: ``GEOM`` sums likelihood ratios by direct convolution on
            a geometric grid; ``MS_FFT`` by FFTs inside each octave of a dyadic grid.
        projection_method: Projection at the merges. ``None`` selects the grid's own:
            ``STOCH_DOM`` (rounding) for ``GEOM`` and ``CTD`` (connect-the-dots) for
            ``MS_FFT``. Rounding gives upper and lower bounds. Connect-the-dots gives
            upper bounds only, so a lower bound with ``MS_FFT`` requires
            ``STOCH_DOM`` explicitly.
        max_cells_per_band: ``MS_FFT`` only. Largest number of dyadic grid points per
            octave. For a stage's share ``h`` of ``loss_discretization``, the count
            is ``ceil(log(2) / h)``, or this cap when smaller; it need not be a power
            of two. A binding cap warns and the bound stays valid and is looser.
        max_closing_grid: ``MS_FFT`` only. Largest number of points of the linear
            loss grid that one allocation returns, at least ``MIN_GRID_POINT_CAP``; a
            binding cap coarsens the step of that grid and warns, and the bound stays
            valid and is looser.
    """

    loss_discretization: float = DEFAULT_LOSS_DISCRETIZATION
    tail_truncation: float = DEFAULT_TAIL_TRUNCATION
    max_grid_mult: int = -1
    convolution_method: ConvolutionMethod = ConvolutionMethod.GEOM
    projection_method: ProjectionMethod | None = None
    max_cells_per_band: int = DEFAULT_MAX_CELLS_PER_BAND
    max_closing_grid: int = DEFAULT_MAX_CLOSING_GRID

    def __post_init__(self) -> None:
        """Reject field values that cannot describe a discretization scheme."""
        require_positive_real(
            value=[self.loss_discretization, self.tail_truncation],
            name=["loss_discretization", "tail_truncation"],
        )
        require_integer(value=self.max_grid_mult, name="max_grid_mult")
        if 0 < self.max_grid_mult < MIN_GRID_POINT_CAP:
            raise ValueError(
                f"max_grid_mult must be <= 0 or >= {MIN_GRID_POINT_CAP}, "
                f"got {self.max_grid_mult!r}"
            )
        require_enum(
            value=self.convolution_method, enum_cls=ConvolutionMethod, name="convolution_method"
        )
        if self.projection_method is not None:
            require_enum(
                value=self.projection_method,
                enum_cls=ProjectionMethod,
                name="projection_method",
            )
        require_positive_int(
            value=[self.max_cells_per_band, self.max_closing_grid],
            name=["max_cells_per_band", "max_closing_grid"],
        )
        if self.max_closing_grid < MIN_GRID_POINT_CAP:
            raise ValueError(
                f"max_closing_grid must be >= {MIN_GRID_POINT_CAP}, "
                f"got {self.max_closing_grid!r}"
            )


def resolve_projection_method(
    *, config: AllocationSchemeConfig, bound_type: BoundType
) -> ProjectionMethod:
    """Return the projection of the merges for this configuration and bound.

    ``None`` selects the grid's own projection: rounding on the geometric grid and
    connect-the-dots on the dyadic grid. Connect-the-dots is rejected with
    ``BoundType.IS_DOMINATED``.
    """
    projection_method = config.projection_method
    if projection_method is None:
        projection_method = (
            ProjectionMethod.CTD
            if config.convolution_method == ConvolutionMethod.MS_FFT
            else ProjectionMethod.STOCH_DOM
        )
    require_projection_method(value=projection_method, bound_type=bound_type)
    return projection_method
