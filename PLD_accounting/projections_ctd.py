"""Connect-the-dots rounding: a dominating bound that keeps the mean of ``exp(-L)``.

On a fixed-gap loss grid each cell's endpoint masses are built from local source
and reflected-dual measures. In a merge on a geometric grid each value is split
between its two neighbouring points, with positions given in units of grid points.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from PLD_accounting.discrete_dist import (
    REALIZATION_MOMENT_TOL,
    DiscreteDistBase,
    Domain,
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_utils import (
    MAX_SAFE_EXP_ARG,
    ORACLE_TOL,
    classify_residual,
    compensated_segmented_sum,
    exact_sum,
    exp_moment_terms,
    joint_source_dual_bounds,
    partition_masses,
    signed_unit_residual,
    trim_mass_to_moment_target,
)
from PLD_accounting.types import SpacingType
from PLD_accounting.utils import aligned_grid_params, grid_step_for_point_cap
from PLD_accounting.validation import require_finite_array, require_increasing_array

# Safety factor on the cellwise reciprocal-moment residual bound.
_CTD_MOMENT_REPAIR_FACTOR = 8.0
# Continuous interval oracle only: below three interior cells a cell spans O(support) and
# float64 CDF/SF differences fail the producer residual bound. Discrete CtD bins atoms exactly.
_MIN_CONTINUOUS_INTERVAL_KNOTS = 4
# The smallest normal float64. A cell mass below it is subnormal: it has too few significant
# bits to split.
_SMALLEST_NORMAL = float(np.finfo(np.float64).tiny)

# =============================================================================
# Real-Loss Projection
# =============================================================================


def project_continuous_dist_ctd(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    dual_dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    step: float,
) -> PLDRealization:
    """Construct a dominating fixed-gap real-loss PLD with connect-the-dots.

    ``dual_dist`` must be the exact PLD dual of ``dist``. The finite grid covers
    both ``dist`` and the reflected dual, and each cell's mass and reciprocal
    moment are read from the two laws locally: the resulting endpoint split
    preserves the hockey-stick values at the knots and dominates between them.

    A ``step`` that leaves fewer than four knots over that joint support is
    rejected rather than clamped: float64 interval measures are not valid there.
    """
    x_min, x_max = joint_source_dual_bounds(
        dist=dist,
        dual_dist=dual_dist,
        tail_truncation=tail_truncation,
    )
    grid = aligned_grid_params(x_min=x_min, x_max=x_max, discretization=step)
    if grid.n < _MIN_CONTINUOUS_INTERVAL_KNOTS:
        raise ValueError(
            f"loss_discretization={step:.6g} is coarser than the CtD source/dual support "
            f"[{x_min:.6g}, {x_max:.6g}] ({grid.n} knots); float64 interval measures are "
            "not valid on a grid this coarse. Reduce loss_discretization."
        )
    loss = grid.materialize()
    pld_pmf, pld_dual_pmf = _continuous_ctd_cell_measures(
        pld_in=dist,
        dual_pld_in=dual_dist,
        loss=loss,
    )
    prob, p_max = _ctd_masses_from_cell_measures(
        pld_pmf=pld_pmf,
        pld_dual_pmf=pld_dual_pmf,
        p_max_in=0.0,
        loss=loss,
        width=np.full(loss.size - 1, grid.step, dtype=np.float64),
    )
    return PLDRealization(grid=grid, prob_arr=prob, p_min=0.0, p_max=p_max)


def project_continuous_dist_ctd_on_points(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    dual_dist: stats.rv_continuous | rv_frozen[Any, Any],
    loss_points: NDArray[np.float64],
) -> SparseDiscreteDist:
    """Construct a dominating real-loss PLD on the given loss points with connect-the-dots.

    ``dual_dist`` must be the exact PLD dual of ``dist`` and ``loss_points`` strictly
    increasing. Each cell between two points is split between them as on a fixed-gap
    grid; the laws outside the points follow the same boundary rule.
    """
    loss = np.asarray(loss_points, dtype=np.float64)
    require_increasing_array(values=loss, name="loss_points")
    pld_pmf, pld_dual_pmf = _continuous_ctd_cell_measures(
        pld_in=dist,
        dual_pld_in=dual_dist,
        loss=loss,
    )
    prob, p_max = _ctd_masses_from_cell_measures(
        pld_pmf=pld_pmf,
        pld_dual_pmf=pld_dual_pmf,
        p_max_in=0.0,
        loss=loss,
        width=np.diff(loss),
    )
    return SparseDiscreteDist(x_array=loss, prob_arr=prob, p_min=0.0, p_max=p_max)


def project_discrete_dist_ctd(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    loss_discretization: float,
    max_grid_points: int = -1,
) -> PLDRealization:
    """Project a real-loss PLD onto a fixed-gap grid of the requested step with CtD.

    The semantic PLD contract is checked before truncation so the tail budget
    cannot hide an invalid ``-inf`` atom or reciprocal-moment excess. A positive
    ``max_grid_points`` coarsens the step so that the grid over the truncated
    support has at most that many points; the returned grid carries the step used.
    """
    _require_ctd_source(dist)
    trunc_dist = truncate_edges_ctd(dist=dist, tail_truncation=tail_truncation / 2)
    x_min = float(trunc_dist.x_array[0])
    x_max = float(trunc_dist.x_array[-1])
    if x_max == x_min:
        raise ValueError(
            "project_discrete_dist_ctd requires at least two distinct finite "
            f"support points after truncation, got x_min=x_max={x_min}"
        )
    step = grid_step_for_point_cap(
        x_min=x_min,
        x_max=x_max,
        requested_step=loss_discretization,
        max_grid_points=max_grid_points,
    )
    grid_out = aligned_grid_params(x_min=x_min, x_max=x_max, discretization=step)
    return project_dist_onto_grid_ctd(dist=trunc_dist, grid=grid_out)


def project_dist_onto_grid_ctd(
    *,
    dist: DiscreteDistBase,
    grid: GridSpec,
) -> PLDRealization:
    """CtD-project a valid real-loss PLD source onto a fixed-gap linear grid.

    CtD accepts any discrete source that is itself a semantically valid PLD
    realization.  This validates the source object only; callers remain
    responsible for ensuring that it dominates the external mechanism being
    accounted for.
    """
    if grid.spacing_type != SpacingType.LINEAR:
        raise ValueError("CtD projection requires a fixed-gap linear grid")
    _require_ctd_source(dist)
    loss = grid.materialize()
    pld_pmf, pld_dual_pmf = _discrete_ctd_cell_measures(dist_in=dist, loss_out=loss)
    prob, p_max = _ctd_masses_from_cell_measures(
        pld_pmf=pld_pmf,
        pld_dual_pmf=pld_dual_pmf,
        p_max_in=float(dist.p_max),
        loss=loss,
        width=np.full(loss.size - 1, grid.step, dtype=np.float64),
    )
    return PLDRealization(grid=grid, prob_arr=prob, p_min=0.0, p_max=p_max)


def project_discrete_dist_ctd_on_points(
    *,
    dist: DiscreteDistBase,
    loss_points: NDArray[np.float64],
) -> SparseDiscreteDist:
    """CtD-project a valid real-loss PLD source onto the given loss points.

    The source contract is that of ``project_dist_onto_grid_ctd``; ``loss_points``
    must be strictly increasing.
    """
    loss = np.asarray(loss_points, dtype=np.float64)
    require_increasing_array(values=loss, name="loss_points")
    _require_ctd_source(dist)
    pld_pmf, pld_dual_pmf = _discrete_ctd_cell_measures(dist_in=dist, loss_out=loss)
    prob, p_max = _ctd_masses_from_cell_measures(
        pld_pmf=pld_pmf,
        pld_dual_pmf=pld_dual_pmf,
        p_max_in=float(dist.p_max),
        loss=loss,
        width=np.diff(loss),
    )
    return SparseDiscreteDist(x_array=loss, prob_arr=prob, p_min=0.0, p_max=p_max)


# =============================================================================
# Edge Truncation
# =============================================================================


def truncate_edges_ctd(*, dist: DiscreteDistBase, tail_truncation: float) -> DiscreteDistBase:
    """Trim both tails of a dominating real-loss PLD, keeping ``E[exp(-L)]`` where it can be.

    Above the last kept loss the reciprocal moment of the trimmed region stays at
    that loss and the rest of its mass moves to ``+inf``. Below the first kept
    loss the mass of the trimmed region moves to that loss. The mass of a region
    and its reciprocal moment are each at most ``tail_truncation``.
    """
    if dist.domain != Domain.REALS:
        raise ValueError(f"CtD truncation requires a real-domain distribution, got {dist.domain}")
    loss = dist.x_array
    if max(float(loss[-1]), -float(loss[0])) > MAX_SAFE_EXP_ARG:
        raise ValueError(
            f"CtD support [{float(loss[0]):.6e}, {float(loss[-1]):.6e}] leaves the range where "
            "exp(x) is representable; the tails' moment factors cannot be formed"
        )
    # exp(-L) grows as the loss falls, so the reversed support is the increasing one
    # and the atom at zero is the mass at loss +inf.
    first_kept, kept, p_max = truncate_tails_ctd(
        values=np.exp(-loss[::-1]),
        prob_arr=dist.prob_arr[::-1],
        p_min=dist.p_max,
        moment_scale=1.0,
        tail_truncation=tail_truncation,
    )
    min_ind = loss.size - first_kept - kept.size
    return dist.rebuild_on_range(
        new_prob_arr=kept[::-1].copy(),
        new_p_min=dist.p_min,
        new_p_max=p_max,
        min_ind=min_ind,
        max_ind=min_ind + kept.size - 1,
    )


def truncate_tails_ctd(
    *,
    values: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    p_min: float,
    moment_scale: float,
    tail_truncation: float,
) -> tuple[int, NDArray[np.float64], float]:
    """Trim both tails of a positive-valued distribution, keeping the mean where it can be.

    Below the first kept point the first moment of the trimmed region stays at that
    point and the rest of its mass moves to the atom at zero. Above the last kept
    point the mass of the trimmed region moves to that point. The mass of a region
    and its first moment over ``moment_scale`` are each at most ``tail_truncation``.

    Returns:
        The index of the first kept point, the masses of the kept points, and the
        mass at zero.
    """
    if not np.any(prob_arr):
        raise ValueError("Cannot trim a connect-the-dots sum with zero finite mass")
    moment_share = prob_arr * values / moment_scale
    lower_count = _edge_trim_count(
        masses=prob_arr, moment_share=moment_share, tail_truncation=tail_truncation
    )
    upper_count = min(
        _edge_trim_count(
            masses=prob_arr[::-1],
            moment_share=moment_share[::-1],
            tail_truncation=tail_truncation,
        ),
        prob_arr.size - 1 - lower_count,
    )
    stop = prob_arr.size - upper_count
    kept = prob_arr[lower_count:stop].copy()
    if lower_count:
        stays = prob_arr[:lower_count] * (values[:lower_count] / values[lower_count])
        kept[0] += exact_sum(stays)
        p_min += exact_sum(prob_arr[:lower_count] - stays)
    if upper_count:
        kept[-1] += exact_sum(prob_arr[stop:])
    return lower_count, kept, p_min


# =============================================================================
# Projection of a Geometric-Grid Merge
# =============================================================================


def project_grid_offsets_ctd(
    *,
    grid_offsets: NDArray[np.float64],
    log_step: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Split values between their two neighbouring points of a geometric grid.

    ``grid_offsets`` are positions in units of grid points on a grid whose points
    grow by the factor ``exp(log_step)``. Returns the index of each value's lower
    neighbour and the share of its mass that the upper neighbour takes to keep the
    mean.
    """
    lower = np.floor(grid_offsets)
    shares = np.expm1((grid_offsets - lower) * log_step) / np.expm1(log_step)
    return lower.astype(np.int64), np.minimum(shares, 1.0)


# =============================================================================
# Projection of a Dyadic-Grid Merge
# =============================================================================


def halve_lattice_ctd(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Project masses at lattice points ``0..2m`` onto the even points, keeping the mean.

    The last axis runs over the lattice. An odd point is split equally between its two
    even neighbours, which keeps mass and first moment.
    """
    result = np.array(values[..., ::2], copy=True)
    half_odd = values[..., 1::2] * 0.5
    result[..., :-1] += half_odd
    result[..., 1:] += half_odd
    return result


# =============================================================================
# Helper Functions
# =============================================================================


def _continuous_ctd_cell_measures(
    *,
    pld_in: stats.rv_continuous | rv_frozen[Any, Any],
    dual_pld_in: stats.rv_continuous | rv_frozen[Any, Any],
    loss: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate every CtD cell's mass and reciprocal moment from a PLD and its dual.

    Uses reflected left limits so a privacy-loss atom at a knot stays in the correct cell.
    """
    require_finite_array(values=loss, name="CtD grid")
    if loss.size < 2:
        raise ValueError("CtD projection requires at least two finite grid knots")
    mass, mass_below, mass_above = partition_masses(dist=pld_in, points=loss, label="CtD source")
    reflected, reflected_above, reflected_below = partition_masses(
        dist=dual_pld_in,
        points=np.nextafter(-loss[::-1], -np.inf),
        label="CtD reflected-dual",
    )
    return (
        np.concatenate(([mass_below], mass, [mass_above])),
        np.concatenate(([reflected_below], reflected[::-1], [reflected_above])),
    )


def _ctd_masses_from_cell_measures(
    *,
    pld_pmf: NDArray[np.float64],
    pld_dual_pmf: NDArray[np.float64],
    p_max_in: float,
    loss: NDArray[np.float64],
    width: NDArray[np.float64],
) -> tuple[NDArray[np.float64], float]:
    """Return the CtD masses at the knots ``loss`` and the mass at ``+inf``.

    The per-cell source and dual measures cover the ``n + 1`` cells that ``n`` knots
    delimit, and ``width`` holds the widths of the ``n - 1`` interior cells.
    """
    if max(float(loss[-1]), -float(loss[0])) > MAX_SAFE_EXP_ARG:
        raise ValueError(
            f"CtD grid [{float(loss[0]):.6e}, {float(loss[-1]):.6e}] leaves the range where "
            "exp(x) is representable; the cells' moment factors cannot be formed"
        )
    contribution_left, contribution_right = _ctd_cell_endpoint_masses(
        pld_pmf=pld_pmf[1:-1],
        pld_dual_pmf=pld_dual_pmf[1:-1],
        left=loss[:-1],
        width=width,
    )
    prob = np.zeros(loss.size, dtype=np.float64)
    prob[:-1] += contribution_left
    prob[1:] += contribution_right
    lower_at_knot, upper_at_knot, p_max_out, lower_semantic_dual_mass = _ctd_exterior_policy(
        lower_mass=float(pld_pmf[0]),
        lower_reflected_mass=float(pld_dual_pmf[0]),
        upper_mass=float(pld_pmf[-1]),
        upper_reflected_mass=float(pld_dual_pmf[-1]),
        first_knot=float(loss[0]),
        last_knot=float(loss[-1]),
        p_max_in=p_max_in,
    )
    prob[0] += lower_at_knot
    prob[-1] += upper_at_knot
    # The endpoint split preserves each cell's mass exactly but only its reciprocal
    # moment up to rounding, so that is the one invariant repaired here.
    prob, p_max_out = _repair_ctd_reciprocal_moment(
        prob=prob,
        loss=loss,
        p_max=p_max_out,
        step=float(width.min()),
    )
    moment_terms = exp_moment_terms(prob_arr=prob, x_vals=loss)
    residual = signed_unit_residual(values=moment_terms, lower_term=0.0, upper_term=0.0)
    if residual < 0.0:
        # The repair above drains to a zero target, so this is a post-repair
        # inconsistency in the moment ledger, not an admissible excess.
        raise ValueError(
            "CtD reciprocal-moment repair left E[exp(-L)] above one by "
            f"{-residual:.3e}; the realization is invalid"
        )
    if lower_semantic_dual_mass > 0.0 and residual < lower_semantic_dual_mass:
        prob, p_max_out = trim_mass_to_moment_target(
            prob_arr=prob,
            loss=loss,
            p_max=p_max_out,
            target_residual=lower_semantic_dual_mass,
            context="CtD lower semantic dual mass repair",
        )
    return prob, p_max_out


def _require_ctd_source(dist: DiscreteDistBase) -> None:
    """Require the semantic PLD-realization contract CtD depends on."""
    if dist.domain != Domain.REALS:
        raise ValueError("CtD projection requires a real-domain source")
    if dist.p_min != 0.0:
        raise ValueError(f"CtD projection requires p_min = 0 exactly, got {dist.p_min:.2e}")
    support = np.asarray(dist.x_array, dtype=np.float64)
    require_finite_array(values=support, name="CtD source support")
    if support.size == 0 or np.any(np.diff(support) <= 0.0):
        raise ValueError("CtD projection requires finite, strictly increasing support")
    moment_terms = exp_moment_terms(prob_arr=dist.prob_arr, x_vals=support)
    if np.any(~np.isfinite(moment_terms)):
        raise ValueError("CtD source reciprocal moment must be finite")
    # Neither boundary atom carries reciprocal moment: p_min is 0 by the check above, and
    # p_max sits at L = +inf where exp(-L) is 0, so the finite terms are the whole moment.
    moment_residual = signed_unit_residual(values=moment_terms, lower_term=0.0, upper_term=0.0)
    if moment_residual < -REALIZATION_MOMENT_TOL:
        raise ValueError(
            "CtD source reciprocal-moment violates E[exp(-L)] <= 1 under the "
            "PLD invariant tolerance"
        )


def _discrete_ctd_cell_measures(
    *,
    dist_in: DiscreteDistBase,
    loss_out: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Aggregate atoms into per-cell mass and reciprocal moment via right-closed bins.

    Cell 0 is ``(-inf, loss[0]]``, interior cell ``j`` is ``(loss[j-1], loss[j]]``, and
    the final cell is ``(loss[-1], +inf)``, matching ``np.searchsorted(..., side="left")``.
    """
    loss = np.asarray(loss_out, dtype=np.float64)
    if loss.size < 2:
        raise ValueError("CtD projection requires at least two finite grid knots")
    masses = np.asarray(dist_in.prob_arr, dtype=np.float64)
    cell = np.searchsorted(loss, np.asarray(dist_in.x_array, dtype=np.float64), side="left")
    moments = exp_moment_terms(prob_arr=masses, x_vals=dist_in.x_array)
    return (
        compensated_segmented_sum(
            bin_index=cell,
            weights=masses,
            num_bins=loss.size + 1,
        ),
        compensated_segmented_sum(
            bin_index=cell,
            weights=moments,
            num_bins=loss.size + 1,
        ),
    )


def _edge_trim_count(
    *,
    masses: NDArray[np.float64],
    moment_share: NDArray[np.float64],
    tail_truncation: float,
) -> int:
    """Return how many leading points fit ``tail_truncation`` under both charges.

    The last point is never counted. The prefix sums only choose the count; the mass
    that moves is summed exactly by the caller.
    """
    mass_count = np.searchsorted(np.cumsum(masses[:-1]), tail_truncation, side="right")
    moment_count = np.searchsorted(np.cumsum(moment_share[:-1]), tail_truncation, side="right")
    return min(int(mass_count), int(moment_count))


def _ctd_cell_endpoint_masses(
    *,
    pld_pmf: NDArray[np.float64],
    pld_dual_pmf: NDArray[np.float64],
    left: NDArray[np.float64],
    width: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Split each cell's mass between its two endpoint knots.

    For a cell ``(a, b]`` with mass ``M``, rescaled reciprocal moment ``S = exp(a) R``,
    ``r = exp(a - b)`` and ``d = 1 - r``, CtD places ``(S - r M)/d`` at ``a`` and
    ``(M - S)/d`` at ``b``. ``r M <= S <= M`` holds exactly inside the cell, so the
    right fraction ``(M - S)/(d M)`` lies in ``[0, 1]``. A computed fraction outside it is
    clipped, which moves at most ``M`` between the knots and changes the cell's reciprocal
    moment by at most ``M d exp(-a)`` per unit of excess. On a narrow tail cell the
    masses are differences of nearly equal CDF values, so that excess is rounding noise
    of any size, while the mass it can move is negligible. The split raises only when the
    total reciprocal-moment change of the clipping exceeds ``ORACLE_TOL``. A cell with a
    source mass below the smallest normal float is not split: its source mass goes to
    ``b``, which dominates, and its dual mass counts as moment change.
    """
    pld_pmf = np.asarray(pld_pmf, dtype=np.float64)
    pld_dual_pmf = np.asarray(pld_dual_pmf, dtype=np.float64)
    if pld_pmf.shape != pld_dual_pmf.shape:
        raise ValueError("CtD source and reflected-dual cell arrays must have the same shape")
    if np.any(~np.isfinite(pld_pmf)) or np.any(~np.isfinite(pld_dual_pmf)):
        raise ValueError("CtD source and reflected-dual cell measures must be finite")
    if np.any(pld_pmf < 0.0) or np.any(pld_dual_pmf < 0.0):
        raise ValueError("CtD source and reflected-dual cell measures must be nonnegative")

    resolved = pld_pmf >= _SMALLEST_NORMAL
    denominator = -np.expm1(-width)
    fraction = np.ones_like(pld_pmf)
    fraction[resolved] = (
        (pld_pmf[resolved] - np.exp(left[resolved]) * pld_dual_pmf[resolved])
        / denominator[resolved]
        / pld_pmf[resolved]
    )
    excess = np.maximum(np.maximum(-fraction, fraction - 1.0), 0.0)
    moment_change = np.where(
        resolved,
        pld_pmf * np.minimum(excess, 1.0) * denominator * np.exp(-left),
        pld_dual_pmf,
    )
    if exact_sum(moment_change) > ORACLE_TOL:
        raise ValueError(
            "CtD source/dual interval measures violate r*M <= exp(a)*R <= M: clipping the "
            f"endpoint fractions {float(fraction.min()):.3e}..{float(fraction.max()):.3e} "
            f"into [0, 1] changes the reciprocal moment by {exact_sum(moment_change):.3e}, "
            f"above {ORACLE_TOL:.3e}"
        )
    fraction = np.clip(fraction, 0.0, 1.0)
    weight_right = np.where(
        fraction <= 0.5, pld_pmf * fraction, pld_pmf - pld_pmf * (1.0 - fraction)
    )
    return pld_pmf - weight_right, weight_right


def _ctd_exterior_policy(
    *,
    lower_mass: float,
    lower_reflected_mass: float,
    upper_mass: float,
    upper_reflected_mass: float,
    first_knot: float,
    last_knot: float,
    p_max_in: float,
) -> tuple[float, float, float, float]:
    """Return finite-boundary and ``+inf`` masses for the CtD truncation policy.

    The lower tail collapses onto the first knot. The upper tail is split between the
    last knot and ``+inf`` so as to preserve reflected-dual mass. Returns first-knot
    mass, last-knot mass, ``+inf`` mass, and ``eta = R_minus - exp(-x_0) M_minus``.
    """
    upper_at_knot = math.exp(last_knot) * upper_reflected_mass
    if upper_at_knot > upper_mass * (1.0 + ORACLE_TOL):
        raise ValueError(
            "CtD upper exterior cell violates exp(x_N) R_plus <= M_plus: "
            f"{upper_at_knot:.3e} > {upper_mass:.3e}"
        )
    upper_at_knot = min(upper_at_knot, upper_mass)

    retained_lower_dual = math.exp(-first_knot) * lower_mass
    lower_semantic_dual_mass = lower_reflected_mass - retained_lower_dual
    if lower_semantic_dual_mass < -ORACLE_TOL * max(lower_reflected_mass, 1.0):
        raise ValueError(
            "CtD lower exterior cell violates R_minus >= exp(-x_0) M_minus: "
            f"{lower_reflected_mass:.3e} < {retained_lower_dual:.3e}"
        )
    return (
        lower_mass,
        upper_at_knot,
        p_max_in + upper_mass - upper_at_knot,
        max(0.0, lower_semantic_dual_mass),
    )


def _repair_ctd_reciprocal_moment(
    *,
    prob: NDArray[np.float64],
    loss: NDArray[np.float64],
    p_max: float,
    step: float,
) -> tuple[NDArray[np.float64], float]:
    """Conservatively repair a bounded arithmetic moment excess by moving mass to +inf."""
    contributions = exp_moment_terms(prob_arr=prob, x_vals=loss)
    residual = signed_unit_residual(values=contributions, lower_term=0.0, upper_term=0.0)
    if residual >= 0.0:
        return prob, p_max

    repair_tol = _ctd_moment_repair_tol(max_abs_loss=float(np.max(np.abs(loss))), step=step)
    classify_residual(
        residual=-residual,
        drift_tol=min(REALIZATION_MOMENT_TOL, repair_tol),
        repair_tol=repair_tol,
        context="CtD reciprocal-moment repair",
        repair="moving the cheapest mass to +inf",
    )
    return trim_mass_to_moment_target(
        prob_arr=prob,
        loss=loss,
        p_max=p_max,
        target_residual=0.0,
        context="CtD reciprocal-moment repair",
    )


def _ctd_moment_repair_tol(*, max_abs_loss: float, step: float) -> float:
    """Producer bound on a CtD reciprocal-moment residual: about ``eps |a| / (1-exp(-step))``."""
    denominator = -math.expm1(-abs(step)) if step != 0.0 else 1.0
    amplification = max(1.0, abs(max_abs_loss)) / max(denominator, _SMALLEST_NORMAL)
    return _CTD_MOMENT_REPAIR_FACTOR * amplification * float(np.finfo(np.float64).eps)
