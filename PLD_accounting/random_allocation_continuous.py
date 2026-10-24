"""One-step random-allocation factors from a continuous privacy-loss law and its dual."""

from __future__ import annotations

import math
from functools import partial
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from PLD_accounting.discrete_dist import LossFactor, PLDRealization, SparseDiscreteDist
from PLD_accounting.distribution_utils import joint_source_dual_bounds
from PLD_accounting.multiscale_fft_convolution import (
    MAX_DYADIC_REACH,
    dyadic_cells_per_band,
    dyadic_range_points,
)
from PLD_accounting.projections import (
    project_continuous_dist,
    project_continuous_dist_on_points,
)
from PLD_accounting.projections_stoch_dom import (
    project_continuous_dist_stoch_dom,
    project_continuous_dist_stoch_dom_on_points,
)
from PLD_accounting.types import BoundType, ConvolutionMethod, Direction, require_bound_type
from PLD_accounting.utils import (
    calc_pld_dual,
    grid_step_for_point_cap,
    negate_reverse_distribution,
)
from PLD_accounting.validation import require_integer, require_positive_real

_Projection = Callable[..., LossFactor]


# =============================================================================
# Public API
# =============================================================================


def continuous_base_distributions(
    *,
    law: stats.rv_continuous | rv_frozen[Any, Any],
    dual: stats.rv_continuous | rv_frozen[Any, Any],
    loss_discretizations: tuple[float, float],
    tail_truncation: float,
    bound_type: BoundType,
    convolution_method: ConvolutionMethod,
    max_grid_mult: int,
    max_cells_per_band: int,
) -> tuple[tuple[LossFactor, LossFactor], LossFactor]:
    """Prepare the one-step factors of both directions from a continuous law and its exact dual.

    ``law`` is the REMOVE-direction privacy-loss law and ``dual`` its exact PLD dual,
    the ADD-direction law. Budgets are ordered ``(remove, add)``; ``tail_truncation``
    is the budget of one direction. Returns the remove loss factors
    ``(base, neg_dual_base)`` and the add loss factor ``base`` on the loss points of
    ``convolution_method``. ``GEOM`` takes a fixed-gap grid of at most
    ``max_grid_mult`` points (``<= 0`` disables the cap), as ``DenseDiscreteDist``.
    ``MS_FFT`` takes the logarithms of the dyadic points of a range of likelihood
    ratios, with at most ``max_cells_per_band`` points per octave, negated for the
    add factor, as ``SparseDiscreteDist``.
    """
    require_bound_type(value=bound_type)
    remove_projection, dual_rounding = _loss_projections(
        law=law,
        dual=dual,
        loss_discretization=loss_discretizations[0],
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolution_method=convolution_method,
        max_grid_mult=max_grid_mult,
        max_cells_per_band=max_cells_per_band,
        direction=Direction.REMOVE,
    )
    add_projection, _ = _loss_projections(
        law=dual,
        dual=law,
        loss_discretization=loss_discretizations[1],
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolution_method=convolution_method,
        max_grid_mult=max_grid_mult,
        max_cells_per_band=max_cells_per_band,
        direction=Direction.ADD,
    )

    loss_factor = remove_projection(dist=law, dual_dist=dual, bound_type=bound_type)
    if bound_type == BoundType.DOMINATES:
        # Discretize once and take the exact dual of the discretized factor: a second
        # discretization of the dual is avoided and the upper bound stays valid.
        if not isinstance(loss_factor, (PLDRealization, SparseDiscreteDist)):
            raise TypeError(
                f"dominating projection must return a PLD, got {type(loss_factor).__name__}"
            )
        neg_dual_factor = negate_reverse_distribution(calc_pld_dual(loss_factor))
    else:
        # Lower-bound truncation can create negative-infinity mass, which the exact dual
        # of a discretized factor cannot represent, so the two laws are rounded
        # separately. Rounding the dual up and reflecting rounds the negated dual down.
        neg_dual_factor = negate_reverse_distribution(dual_rounding(dist=dual))
    add_factor = add_projection(dist=dual, dual_dist=law, bound_type=bound_type)
    return (loss_factor, neg_dual_factor), add_factor


# =============================================================================
# Helper Functions
# =============================================================================


def _loss_projections(
    *,
    law: stats.rv_continuous | rv_frozen[Any, Any],
    dual: stats.rv_continuous | rv_frozen[Any, Any],
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    convolution_method: ConvolutionMethod,
    max_grid_mult: int,
    max_cells_per_band: int,
    direction: Direction,
) -> tuple[_Projection, _Projection]:
    """Return the projections of one direction onto the loss points of a convolution method.

    The first projects the law of the direction; the second rounds a law up on the
    reflection of the same points, for the dual of a lower bound.

    ``GEOM`` projects onto a fixed-gap grid. Its step is the loss budget, coarsened
    when the grid over the range the tail budget allows would exceed ``max_grid_mult``
    points (``<= 0`` disables the cap). ``MS_FFT`` projects onto the logarithms of
    dyadic points, with the points per octave that the loss budget gives, up to
    ``max_cells_per_band``. The add ratio is ``exp(-L)``, so the add factor is on the
    negated points.
    """
    projection: _Projection
    dual_rounding: _Projection
    if convolution_method == ConvolutionMethod.GEOM:
        require_integer(value=max_grid_mult, name="max_grid_mult")
        # A lower bound rounds the two remove factors separately; they share the budget.
        if direction == Direction.REMOVE and bound_type == BoundType.IS_DOMINATED:
            tail_truncation /= 2
        step = _geometric_step(
            law=law,
            dual=dual,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            max_grid_mult=max_grid_mult,
        )
        projection = partial(project_continuous_dist, tail_truncation=tail_truncation, step=step)
        dual_rounding = partial(
            project_continuous_dist_stoch_dom,
            tail_truncation=tail_truncation,
            step=step,
            bound_type=BoundType.DOMINATES,
        )
        return projection, dual_rounding
    if convolution_method == ConvolutionMethod.MS_FFT:
        cells_per_band = dyadic_cells_per_band(
            loss_step=loss_discretization, max_cells_per_band=max_cells_per_band
        )
        if direction == Direction.REMOVE:
            loss_points = _dyadic_loss_points(
                law=law, dual=dual, cells_per_band=cells_per_band, tail_truncation=tail_truncation
            )
        else:
            loss_points = -_dyadic_loss_points(
                law=dual, dual=law, cells_per_band=cells_per_band, tail_truncation=tail_truncation
            )[::-1]
        projection = partial(project_continuous_dist_on_points, loss_points=loss_points)
        dual_rounding = partial(
            project_continuous_dist_stoch_dom_on_points,
            loss_points=-loss_points[::-1],
            bound_type=BoundType.DOMINATES,
        )
        return projection, dual_rounding
    raise ValueError(f"Unknown ConvolutionMethod: {convolution_method}")


def _geometric_step(
    *,
    law: stats.rv_continuous | rv_frozen[Any, Any],
    dual: stats.rv_continuous | rv_frozen[Any, Any],
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    max_grid_mult: int,
) -> float:
    """Return the fixed-gap step of one law under the grid cap.

    The range is that of the projection: the law and its reflected dual for an upper
    bound, the law alone for a lower bound.
    """
    require_positive_real(value=loss_discretization, name="loss_discretization")
    x_min, x_max = joint_source_dual_bounds(
        dist=law,
        dual_dist=dual if bound_type == BoundType.DOMINATES else None,
        tail_truncation=tail_truncation,
    )
    return grid_step_for_point_cap(
        x_min=x_min,
        x_max=x_max,
        requested_step=loss_discretization,
        max_grid_points=max_grid_mult,
    )


def _dyadic_loss_points(
    *,
    law: stats.rv_continuous | rv_frozen[Any, Any],
    dual: stats.rv_continuous | rv_frozen[Any, Any],
    cells_per_band: int,
    tail_truncation: float,
) -> NDArray[np.float64]:
    """Return the logarithms of the dyadic points of the range that the tail budget allows.

    The range is the smallest ``[2**-r, 2**r]`` outside which each tail of ``law`` and
    of the reflected ``dual`` is at most ``tail_truncation / 4``.
    """
    # The joint range halves its budget between the two laws; each has two tails.
    x_min, x_max = joint_source_dual_bounds(
        dist=law, dual_dist=dual, tail_truncation=tail_truncation / 2
    )
    reach = max(1, math.ceil(max(-x_min, x_max) / math.log(2.0)))
    if reach >= MAX_DYADIC_REACH:
        raise ValueError(
            f"tail_truncation={tail_truncation:.3e} per factor needs likelihood ratios beyond "
            f"2**{MAX_DYADIC_REACH}, outside the float range; raise tail_truncation"
        )
    return np.log(dyadic_range_points(reach=reach, cells_per_band=cells_per_band))
