"""Projections selected by a bound or by a projection method.

Each function calls the connect-the-dots or the stochastic-domination
implementation of its operation; the implementations live in the module of their
kind.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    DiscreteDistBase,
    Domain,
    SparseDiscreteDist,
)
from PLD_accounting.projections_ctd import (
    halve_lattice_ctd,
    project_continuous_dist_ctd,
    project_continuous_dist_ctd_on_points,
    project_discrete_dist_ctd,
    project_discrete_dist_ctd_on_points,
    project_grid_offsets_ctd,
    truncate_edges_ctd,
    truncate_tails_ctd,
)
from PLD_accounting.projections_stoch_dom import (
    halve_lattice_stoch_dom,
    truncate_tails_stoch_dom,
    project_continuous_dist_stoch_dom,
    project_continuous_dist_stoch_dom_on_points,
    project_discrete_dist_stoch_dom,
    project_discrete_dist_stoch_dom_on_points,
    project_grid_offsets_stoch_dom,
    truncate_edges_stoch_dom,
)
from PLD_accounting.types import BoundType, ProjectionMethod, require_projection_method
from PLD_accounting.validation import require_nonnegative_real

# =============================================================================
# Selected by the Bound
# =============================================================================


def project_continuous_dist(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    dual_dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    step: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Project a continuous privacy-loss law onto a fixed-gap grid.

    Dominating outputs use CtD, which reads ``dual_dist``, the exact PLD dual of
    ``dist``. Dominated outputs use directional stochastic domination of ``dist``
    alone.
    """
    if bound_type == BoundType.DOMINATES:
        return project_continuous_dist_ctd(
            dist=dist,
            dual_dist=dual_dist,
            tail_truncation=tail_truncation,
            step=step,
        )
    if bound_type == BoundType.IS_DOMINATED:
        return project_continuous_dist_stoch_dom(
            dist=dist,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            step=step,
        )
    raise ValueError(f"Unknown BoundType: {bound_type}")


def project_discrete_dist(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    loss_discretization: float,
    bound_type: BoundType,
    max_grid_points: int = -1,
) -> DenseDiscreteDist:
    """Project a real-loss distribution onto a fixed-gap grid by its bound semantics.

    Dominating fixed-gap real-loss outputs use CtD. Dominated outputs use
    directional stochastic domination. This is structural routing by the
    mathematical bound direction, not a configurable projection method.
    Semantic PLD checks do not prove domination of an external mechanism; the
    caller remains responsible for that proof. A positive ``max_grid_points``
    coarsens the step so that the grid over the support left after truncation has
    at most that many points; the returned grid carries the step used.
    The source must use ``Domain.REALS`` and ``tail_truncation`` must be finite
    and nonnegative.

    A ``DOMINATES`` source must be a valid PLD: ``p_min`` exactly zero, finite
    strictly increasing support, and ``E[exp(-L)] <= 1`` within
    ``REALIZATION_MOMENT_TOL``; anything else raises ``ValueError``. The result is
    then a ``PLDRealization`` on a zero-anchored grid. An ``IS_DOMINATED`` result is
    a ``DenseDiscreteDist`` that may carry mass at ``-inf``.
    """
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    if dist.domain != Domain.REALS:
        raise ValueError("project_discrete_dist requires a real-domain source")
    if bound_type == BoundType.DOMINATES:
        return project_discrete_dist_ctd(
            dist=dist,
            tail_truncation=tail_truncation,
            loss_discretization=loss_discretization,
            max_grid_points=max_grid_points,
        )
    if bound_type == BoundType.IS_DOMINATED:
        return project_discrete_dist_stoch_dom(
            dist=dist,
            tail_truncation=tail_truncation,
            loss_discretization=loss_discretization,
            bound_type=bound_type,
            max_grid_points=max_grid_points,
        )
    raise ValueError(f"Unknown BoundType: {bound_type}")


def project_continuous_dist_on_points(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    dual_dist: stats.rv_continuous | rv_frozen[Any, Any],
    loss_points: NDArray[np.float64],
    bound_type: BoundType,
) -> SparseDiscreteDist:
    """Project a continuous privacy-loss law onto the given loss points.

    Dominating outputs use CtD, which reads ``dual_dist``, the exact PLD dual of
    ``dist``. Dominated outputs use directional stochastic domination of ``dist``
    alone. ``loss_points`` must be strictly increasing.
    """
    if bound_type == BoundType.DOMINATES:
        return project_continuous_dist_ctd_on_points(
            dist=dist, dual_dist=dual_dist, loss_points=loss_points
        )
    if bound_type == BoundType.IS_DOMINATED:
        return project_continuous_dist_stoch_dom_on_points(
            dist=dist, loss_points=loss_points, bound_type=bound_type
        )
    raise ValueError(f"Unknown BoundType: {bound_type}")


def project_discrete_dist_on_points(
    *,
    dist: DiscreteDistBase,
    loss_points: NDArray[np.float64],
    bound_type: BoundType,
) -> SparseDiscreteDist:
    """Project a real-loss distribution onto the given loss points by its bound semantics.

    Dominating outputs use CtD. Dominated outputs use directional stochastic
    domination, which first lowers the mass at ``+inf`` to the last atom.
    ``loss_points`` must be strictly increasing.
    """
    if bound_type == BoundType.DOMINATES:
        return project_discrete_dist_ctd_on_points(dist=dist, loss_points=loss_points)
    if bound_type == BoundType.IS_DOMINATED:
        return project_discrete_dist_stoch_dom_on_points(
            dist=dist, loss_points=loss_points, bound_type=bound_type
        )
    raise ValueError(f"Unknown BoundType: {bound_type}")


# =============================================================================
# Selected by the Projection Method
# =============================================================================


def truncate_edges(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> DiscreteDistBase:
    """Trim the tails of a real-loss distribution with the truncation of a projection.

    ``STOCH_DOM`` moves the trimmed mass to the boundary the bound requires. ``CTD``
    keeps the reciprocal moment of the high-loss tail and requires
    ``BoundType.DOMINATES``.
    """
    require_projection_method(value=projection_method, bound_type=bound_type)
    if projection_method == ProjectionMethod.CTD:
        return truncate_edges_ctd(dist=dist, tail_truncation=tail_truncation)
    if projection_method == ProjectionMethod.STOCH_DOM:
        return truncate_edges_stoch_dom(
            dist=dist, tail_truncation=tail_truncation, bound_type=bound_type
        )
    raise ValueError(f"Unknown ProjectionMethod: {projection_method}")


def project_grid_offsets(
    *,
    grid_offsets: NDArray[np.float64],
    log_step: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Return each position's grid point and the share of mass sent to the next point.

    ``grid_offsets`` are positions in units of grid points. ``CTD`` returns the
    point below with the share that keeps the mean on a grid of log spacing
    ``log_step``; ``STOCH_DOM`` returns the point the bound requires with the
    share zero.
    """
    require_projection_method(value=projection_method, bound_type=bound_type)
    if projection_method == ProjectionMethod.CTD:
        return project_grid_offsets_ctd(grid_offsets=grid_offsets, log_step=log_step)
    if projection_method == ProjectionMethod.STOCH_DOM:
        rounded = project_grid_offsets_stoch_dom(grid_offsets=grid_offsets, bound_type=bound_type)
        return rounded, np.zeros(grid_offsets.shape, dtype=np.float64)
    raise ValueError(f"Unknown ProjectionMethod: {projection_method}")


def halve_lattice(
    *,
    values: NDArray[np.float64],
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> NDArray[np.float64]:
    """Project masses at lattice points ``0..2m`` onto the even points, along the last axis.

    ``CTD`` splits an odd point equally between its two neighbours; ``STOCH_DOM`` moves
    it to the neighbour the bound requires.
    """
    require_projection_method(value=projection_method, bound_type=bound_type)
    if projection_method == ProjectionMethod.CTD:
        return halve_lattice_ctd(values)
    if projection_method == ProjectionMethod.STOCH_DOM:
        return halve_lattice_stoch_dom(values=values, bound_type=bound_type)
    raise ValueError(f"Unknown ProjectionMethod: {projection_method}")


def truncate_tails(
    *,
    values: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    p_min: float,
    p_max: float,
    moment_scale: float,
    tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> tuple[NDArray[np.float64], float, float, int]:
    """Trim both tails of a positive-valued distribution given by its arrays.

    ``STOCH_DOM`` moves the trimmed mass to the boundary the bound requires.
    ``CTD`` keeps the first moment of the low tail, measured over
    ``moment_scale``. Returns the kept masses, the two boundary masses and the
    index of the first kept point.
    """
    require_projection_method(value=projection_method, bound_type=bound_type)
    if projection_method == ProjectionMethod.CTD:
        first_kept, kept, p_min = truncate_tails_ctd(
            values=values,
            prob_arr=prob_arr,
            p_min=p_min,
            moment_scale=moment_scale,
            tail_truncation=tail_truncation,
        )
        return kept, p_min, p_max, first_kept
    if projection_method == ProjectionMethod.STOCH_DOM:
        kept, p_min, p_max, first_kept, _ = truncate_tails_stoch_dom(
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
        return kept, p_min, p_max, first_kept
    raise ValueError(f"Unknown ProjectionMethod: {projection_method}")
