"""Stochastic-domination rounding: mass moves in the direction the bound requires.

On a fixed-gap grid interval mass and atoms move to the upper knot for
``DOMINATES`` and the lower knot for ``IS_DOMINATED``. In a merge on a grid each
value is rounded to the neighbouring point the bound requires, with positions
given in units of grid points.
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
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_utils import (
    drain_mass_from_edge,
    enforce_mass_conservation,
    exact_sum,
    partition_masses,
)
from PLD_accounting.types import BoundType, has_numba, optional_njit, require_bound_type
from PLD_accounting.utils import aligned_grid_params, grid_step_for_point_cap
from PLD_accounting.validation import require_increasing_array

# =============================================================================
# Loss Projection
# =============================================================================


def project_continuous_dist_stoch_dom(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    bound_type: BoundType,
    step: float,
) -> DenseDiscreteDist:
    """Project a continuous law onto a fixed-gap grid with directional stochastic domination.

    Quantile bounds define the finite grid. Interval mass then moves to the
    upper knot for ``DOMINATES`` and the lower knot for ``IS_DOMINATED``;
    intervals of mass below ``tail_truncation`` are batched.
    """
    grid = aligned_grid_params(
        x_min=float(dist.ppf(tail_truncation)),
        x_max=float(dist.isf(tail_truncation)),
        discretization=step,
    )
    prob_arr, p_min, p_max = _continuous_stoch_dom_masses(
        dist=dist,
        x_array=grid.materialize(),
        bound_type=bound_type,
        pmf_min_increment=tail_truncation,
    )
    return DenseDiscreteDist(
        grid=grid,
        prob_arr=prob_arr,
        p_min=p_min,
        p_max=p_max,
    )


def project_continuous_dist_stoch_dom_on_points(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    loss_points: NDArray[np.float64],
    bound_type: BoundType,
) -> SparseDiscreteDist:
    """Project a continuous law onto the given points with directional stochastic domination.

    ``loss_points`` must be strictly increasing. The mass of each interval moves to
    its upper point for ``DOMINATES`` and to its lower point for ``IS_DOMINATED``.
    The tail on the side of the rounding goes to the boundary and the other tail to
    the nearest point.
    """
    x_array = np.asarray(loss_points, dtype=np.float64)
    require_increasing_array(values=x_array, name="loss_points")
    prob_arr, p_min, p_max = _continuous_stoch_dom_masses(
        dist=dist,
        x_array=x_array,
        bound_type=bound_type,
        pmf_min_increment=0.0,
    )
    return SparseDiscreteDist(x_array=x_array, prob_arr=prob_arr, p_min=p_min, p_max=p_max)


def project_discrete_dist_stoch_dom(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    loss_discretization: float,
    bound_type: BoundType,
    max_grid_points: int = -1,
) -> DenseDiscreteDist:
    """Project a distribution onto a linear grid with directional stochastic domination.

    Finite atoms round upward for ``DOMINATES`` and downward for
    ``IS_DOMINATED``; conservatively movable real-domain boundary mass is first
    absorbed into the corresponding finite edge. A positive ``max_grid_points``
    coarsens the step so that the grid over the truncated support has at most that
    many points; the returned grid carries the step used.

    Algorithm 6 (`disc-dist`), in Appendix C of https://arxiv.org/abs/2602.17284.
    """
    working_dist = _absorb_movable_boundary(dist=dist, bound_type=bound_type)

    # Route truncated tail mass according to the requested stochastic bound.
    trunc_dist = truncate_edges_stoch_dom(
        dist=working_dist, tail_truncation=tail_truncation / 2, bound_type=bound_type
    )

    x_array = trunc_dist.x_array
    step = grid_step_for_point_cap(
        x_min=float(x_array[0]),
        x_max=float(x_array[-1]),
        requested_step=loss_discretization,
        max_grid_points=max_grid_points,
    )
    grid_out = aligned_grid_params(
        x_min=float(x_array[0]), x_max=float(x_array[-1]), discretization=step
    )

    return project_dist_onto_grid_stoch_dom(
        dist=trunc_dist,
        grid=grid_out,
        bound_type=bound_type,
    )


def project_dist_onto_grid_stoch_dom(
    *,
    dist: DiscreteDistBase,
    grid: GridSpec,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Project ``dist`` onto ``grid`` while preserving its boundary atoms.

    Finite atoms are rounded in the requested direction. Directional overflow
    that cannot be represented on ``grid`` is added to the corresponding
    conservative boundary.
    """
    prob_arr_out, p_min, p_max = _stoch_dom_masses_on_points(
        dist=dist, x_array_out=grid.materialize(), bound_type=bound_type
    )
    return DenseDiscreteDist(
        grid=grid,
        prob_arr=prob_arr_out,
        p_min=p_min,
        p_max=p_max,
        domain=dist.domain,
    )


def project_discrete_dist_stoch_dom_on_points(
    *,
    dist: DiscreteDistBase,
    loss_points: NDArray[np.float64],
    bound_type: BoundType,
) -> SparseDiscreteDist:
    """Project a distribution onto the given points with directional stochastic domination.

    Finite atoms round upward for ``DOMINATES`` and downward for ``IS_DOMINATED``;
    conservatively movable real-domain boundary mass is first absorbed into the
    corresponding finite edge, as by ``project_discrete_dist_stoch_dom``.
    ``loss_points`` must be strictly increasing.
    """
    x_array_out = np.asarray(loss_points, dtype=np.float64)
    require_increasing_array(values=x_array_out, name="loss_points")
    working_dist = _absorb_movable_boundary(dist=dist, bound_type=bound_type)
    prob_arr_out, p_min, p_max = _stoch_dom_masses_on_points(
        dist=working_dist, x_array_out=x_array_out, bound_type=bound_type
    )
    return SparseDiscreteDist(
        x_array=x_array_out,
        prob_arr=prob_arr_out,
        p_min=p_min,
        p_max=p_max,
        domain=working_dist.domain,
    )


def project_prob(
    *,
    x_array: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    x_array_out: NDArray[np.float64],
    dominates: bool,
) -> NDArray[np.float64]:
    """Dispatch PMF remapping to numba when available, else NumPy."""
    if has_numba():
        return _numba_project_prob(
            x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=dominates
        )
    return _numpy_project_prob(
        x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=dominates
    )


# =============================================================================
# Edge Truncation
# =============================================================================


def truncate_edges_stoch_dom(
    *, dist: DiscreteDistBase, tail_truncation: float, bound_type: BoundType
) -> DiscreteDistBase:
    """Return a copy with up to ``tail_truncation`` mass removed from each edge.

    Removed mass moves to whichever boundary the bound direction makes
    conservative, so the result still bounds the same quantity. ``IS_DOMINATED``
    truncation can move mass into ``p_min``, which a PLD realization cannot
    carry, so a realization is returned as a plain dense distribution.
    """
    if bound_type == BoundType.IS_DOMINATED and isinstance(dist, PLDRealization):
        prob_arr = dist.prob_arr.copy()
        dist = DenseDiscreteDist(
            grid=dist.grid,
            prob_arr=prob_arr,
            p_min=dist.p_min,
            p_max=dist.p_max,
        )
    new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = truncate_tails_stoch_dom(
        prob_arr=dist.prob_arr,
        p_min=dist.p_min,
        p_max=dist.p_max,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    return dist.rebuild_on_range(
        new_prob_arr=new_prob_arr,
        new_p_min=new_p_min,
        new_p_max=new_p_max,
        min_ind=min_ind,
        max_ind=max_ind,
    )


def truncate_tails_stoch_dom(
    *,
    prob_arr: NDArray[np.float64],
    p_min: float,
    p_max: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], float, float, int, int]:
    """Return truncated masses and their surviving range in ``prob_arr``.

    For ``DOMINATES``, removed left mass folds into the first retained value and
    removed right mass moves to ``p_max``; ``IS_DOMINATED`` applies the mirror
    policy. Zero edges are stripped once at the end -- including any the
    truncation itself created -- so the returned indices span the surviving
    nonzero range of the original array.
    """
    if tail_truncation == 0.0:
        prob_arr_out, p_min_out, p_max_out = prob_arr, p_min, p_max
    elif bound_type == BoundType.DOMINATES:
        prob_arr_out, p_min_out, p_max_out = _truncate_dominating_edges(
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            tail_truncation=tail_truncation,
        )
    elif bound_type == BoundType.IS_DOMINATED:
        prob_arr_out, p_min_out, p_max_out = _truncate_dominated_edges(
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            tail_truncation=tail_truncation,
        )
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    # One strip, after whatever the routing did. Stripping first would be redundant:
    # leading zeros contribute nothing to the cumulative scan, so they cannot move the
    # pivot, and the mass folds into the first *nonzero* bin either way.
    nonzero = np.nonzero(prob_arr_out)[0]
    if nonzero.size == 0:
        raise ValueError("Cannot truncate distribution with zero finite mass")
    inner_min, inner_max = int(nonzero[0]), int(nonzero[-1])
    return (
        prob_arr_out[slice(inner_min, inner_max + 1)].copy(),
        p_min_out,
        p_max_out,
        inner_min,
        inner_max,
    )


# =============================================================================
# Projection of a Grid Merge
# =============================================================================


def project_grid_offsets_stoch_dom(
    *,
    grid_offsets: NDArray[np.float64],
    bound_type: BoundType,
) -> NDArray[np.int64]:
    """Round values to the neighbouring grid point that the bound requires.

    ``grid_offsets`` are positions in units of grid points. Returns the index of the
    point above each value for ``DOMINATES`` and below it for ``IS_DOMINATED``.
    """
    if bound_type == BoundType.DOMINATES:
        return np.ceil(grid_offsets).astype(np.int64)
    if bound_type == BoundType.IS_DOMINATED:
        return np.floor(grid_offsets).astype(np.int64)
    raise ValueError(f"Unknown BoundType: {bound_type}")


def halve_lattice_stoch_dom(
    *, values: NDArray[np.float64], bound_type: BoundType
) -> NDArray[np.float64]:
    """Project masses at lattice points ``0..2m`` onto the even points by rounding.

    The last axis runs over the lattice. An odd point moves to the next even point for
    ``DOMINATES`` and to the previous one for ``IS_DOMINATED``.
    """
    result = np.array(values[..., ::2], copy=True)
    odd = values[..., 1::2]
    if bound_type == BoundType.DOMINATES:
        result[..., 1:] += odd
    elif bound_type == BoundType.IS_DOMINATED:
        result[..., :-1] += odd
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")
    return result


# =============================================================================
# Helper Functions
# =============================================================================


def _continuous_stoch_dom_masses(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    x_array: NDArray[np.float64],
    bound_type: BoundType,
    pmf_min_increment: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Return the rounded masses of a continuous law at ``x_array`` and its boundary masses."""
    # Compute finite interval probabilities; the exterior tails remain separate.
    bin_probs, p_left, p_right = _compute_discrete_prob(
        dist=dist, x_array=x_array, bound_type=bound_type, pmf_min_increment=pmf_min_increment
    )
    prob_arr = np.zeros(x_array.size)

    if bound_type == BoundType.DOMINATES:
        # Shift mass right: left tail (-inf, x0] -> x0,
        # each interval (x_i, x_{i+1}] -> x_{i+1}, right tail (x_n, inf) -> +inf.
        prob_arr[0] = p_left
        prob_arr[1:] = bin_probs
        return prob_arr, 0.0, p_right
    if bound_type == BoundType.IS_DOMINATED:
        # Shift mass left: left tail (-inf, x0] -> -inf,
        # each interval (x_i, x_{i+1}] -> x_i, right tail (x_n, inf) -> x_n. The cells
        # are right-closed, so an atom on a knot x_{i+1} moves down to x_i.
        prob_arr[:-1] = bin_probs
        prob_arr[-1] = p_right
        return prob_arr, p_left, 0.0
    raise ValueError(f"Unknown BoundType: {bound_type}")


def _absorb_movable_boundary(*, dist: DiscreteDistBase, bound_type: BoundType) -> DiscreteDistBase:
    """Move the real-domain boundary mass that the bound allows onto the finite edge.

    An upper bound may raise the mass at ``-inf`` to the first atom, and a lower
    bound may lower the mass at ``+inf`` to the last atom.
    """
    working_dist = dist
    if bound_type == BoundType.DOMINATES:
        if dist.domain == Domain.REALS and dist.p_min > 0.0:
            prob_arr = dist.prob_arr.copy()
            prob_arr[0] += dist.p_min
            working_dist = dist.with_probabilities(
                prob_arr=prob_arr,
                p_min=0.0,
                p_max=dist.p_max,
            )
    elif bound_type == BoundType.IS_DOMINATED:
        # A lower-bound projection is no longer an exact realization.
        if isinstance(dist, PLDRealization):
            working_dist = DenseDiscreteDist(
                grid=dist.grid,
                prob_arr=dist.prob_arr,
                p_min=dist.p_min,
                p_max=dist.p_max,
                domain=dist.domain,
            )
        if working_dist.domain == Domain.REALS and working_dist.p_max > 0.0:
            prob_arr = working_dist.prob_arr.copy()
            prob_arr[-1] += working_dist.p_max
            working_dist = working_dist.with_probabilities(
                prob_arr=prob_arr,
                p_min=working_dist.p_min,
                p_max=0.0,
            )
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")
    return working_dist


def _stoch_dom_masses_on_points(
    *,
    dist: DiscreteDistBase,
    x_array_out: NDArray[np.float64],
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], float, float]:
    """Return the rounded masses of ``dist`` at ``x_array_out`` and its boundary masses."""
    expected_p_min = dist.p_min
    expected_p_max = dist.p_max
    # Round in-range atoms conservatively; the kernel omits directional overflow.
    prob_arr_out = project_prob(
        x_array=dist.x_array,
        prob_arr=dist.prob_arr,
        x_array_out=x_array_out,
        dominates=(bound_type == BoundType.DOMINATES),
    )

    if bound_type == BoundType.DOMINATES:
        # Right overflow becomes semantic upper-boundary mass.
        omitted = dist.prob_arr[dist.x_array > x_array_out[-1]]
        expected_p_max += exact_sum(omitted)
    else:
        # Left underflow becomes semantic lower-boundary mass.
        omitted = dist.prob_arr[dist.x_array < x_array_out[0]]
        expected_p_min += exact_sum(omitted)

    # Repair only numerical mass drift after semantic overflow is accounted for.
    prob_arr_out, p_min, p_max = enforce_mass_conservation(
        prob_arr=prob_arr_out,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )
    return prob_arr_out, p_min, p_max


@optional_njit()
def _numba_project_prob(
    *,
    x_array: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    x_array_out: NDArray[np.float64],
    dominates: bool,
) -> NDArray[np.float64]:
    """Remap PMF onto a new grid with domination-aware rounding.

    Monotone supports permit one forward output pointer. ``dominates`` selects
    ceiling rather than floor placement; directional overflow is omitted for
    the caller to route to a boundary. Each output bin uses Kahan accumulation.
    """
    n_out = x_array_out.size
    prob_arr_out = np.zeros(n_out)
    compensations = np.zeros(n_out)

    # One forward pointer suffices because both supports are strictly increasing.
    j = 0

    if dominates:
        # Ceil to the first output knot at or above each input atom.
        for i in range(x_array.size):
            z = x_array[i]
            mass = prob_arr[i]
            # Skip exactly empty bins without discarding small positive masses.
            if mass <= 0:
                continue

            while j < n_out and x_array_out[j] < z:
                j += 1

            if j >= n_out:
                # The caller transfers right overflow to p_max.
                continue
            # Values below the first knot belong in that knot under ceiling.
            y = mass - compensations[j]
            t = prob_arr_out[j] + y
            compensations[j] = (t - prob_arr_out[j]) - y
            prob_arr_out[j] = t

    else:
        # Floor to the last output knot at or below each input atom.
        for i in range(x_array.size):
            z = x_array[i]
            mass = prob_arr[i]
            # Skip exactly empty bins without discarding small positive masses.
            if mass <= 0:
                continue

            while j < n_out and x_array_out[j] <= z:
                j += 1

            idx = j - 1
            if idx < 0:
                # The caller transfers left underflow to p_min.
                continue
            # Values above the last knot belong there under floor rounding.
            y = mass - compensations[idx]
            t = prob_arr_out[idx] + y
            compensations[idx] = (t - prob_arr_out[idx]) - y
            prob_arr_out[idx] = t

    return prob_arr_out


def _numpy_project_prob(
    *,
    x_array: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    x_array_out: NDArray[np.float64],
    dominates: bool,
) -> NDArray[np.float64]:
    """Numpy fallback for remapping PMF onto a new grid."""
    prob_arr_out = np.zeros(x_array_out.size, dtype=np.float64)
    positive = prob_arr > 0.0
    values = x_array[positive]
    if dominates:
        indices = np.searchsorted(x_array_out, values, side="left").astype(np.intp, copy=False)
        valid = indices < x_array_out.size
    else:
        indices = np.searchsorted(x_array_out, values, side="right").astype(np.intp, copy=False) - 1
        valid = indices >= 0
    np.add.at(prob_arr_out, indices[valid], prob_arr[positive][valid])
    return prob_arr_out


def _truncate_dominating_edges(
    *,
    prob_arr: NDArray[np.float64],
    p_min: float,
    p_max: float,
    tail_truncation: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Truncate both edges of a dominating distribution.

    Fold removed left-tail mass into the first retained value and route removed
    right-tail mass to ``p_max``.
    """
    extended_prob = np.concatenate([[p_min], prob_arr])
    before = extended_prob.copy()
    extended_prob = drain_mass_from_edge(
        values=extended_prob, mass=tail_truncation, from_left=True, exact=False
    )
    extended_prob[np.nonzero(extended_prob)[0][0]] += _drained_entry_sum(
        before=before, after=extended_prob
    )
    p_min_out = extended_prob[0]
    before = extended_prob.copy()
    extended_prob = drain_mass_from_edge(
        values=extended_prob, mass=tail_truncation, from_left=False, exact=False
    )
    return (
        extended_prob[1:],
        p_min_out,
        p_max + _drained_entry_sum(before=before, after=extended_prob),
    )


def _truncate_dominated_edges(
    *,
    prob_arr: NDArray[np.float64],
    p_min: float,
    p_max: float,
    tail_truncation: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Truncate both edges of a dominated distribution.

    Fold removed right-tail mass into the last retained value and route removed
    left-tail mass to ``p_min``.
    """
    extended_prob = np.concatenate((prob_arr, [p_max]))
    before = extended_prob.copy()
    extended_prob = drain_mass_from_edge(
        values=extended_prob, mass=tail_truncation, from_left=False, exact=False
    )
    extended_prob[np.nonzero(extended_prob)[0][-1]] += _drained_entry_sum(
        before=before, after=extended_prob
    )
    p_max_out = extended_prob[-1]
    before = extended_prob.copy()
    extended_prob = drain_mass_from_edge(
        values=extended_prob, mass=tail_truncation, from_left=True, exact=False
    )
    return (
        extended_prob[:-1],
        p_min + _drained_entry_sum(before=before, after=extended_prob),
        p_max_out,
    )


def _compute_discrete_prob(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    x_array: NDArray[np.float64],
    bound_type: BoundType,
    pmf_min_increment: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute bin probabilities from cancellation-free interval masses.

    ``pmf_min_increment`` is the minimum interval mass that becomes a bin of its own.
    """
    require_bound_type(value=bound_type)
    cell_masses, p_left, p_right = partition_masses(
        dist=dist, points=x_array, label="directional source"
    )
    # Direction only picks the scan: a dominating bin lands on each interval's upper knot,
    # a dominated one on the lower. Both orientations bin the same cell masses.
    bin_probs = _adaptive_bins_from_masses(
        masses=cell_masses,
        tail_truncation=max(0.0, pmf_min_increment),
        from_left=bound_type == BoundType.DOMINATES,
    )
    return bin_probs, p_left, p_right


def _drained_entry_sum(*, before: NDArray[np.float64], after: NDArray[np.float64]) -> float:
    """Mass a drain removed, summed entry by entry.

    Subtracting the two near-unit totals leaves an ulp of one, which at a large
    negative loss pushes ``E[e^{-L}]`` above one.
    """
    return exact_sum(before - after)


@optional_njit()
def _adaptive_bins_from_masses(
    *,
    masses: NDArray[np.float64],
    tail_truncation: float,
    from_left: bool,
) -> NDArray[np.float64]:
    """Batch already-formed cell masses until each bin reaches ``tail_truncation``.

    ``from_left`` accumulates upward so mass lands on each interval's upper knot;
    otherwise the scan runs downward and mass lands on the lower knot. Reversing the
    ends rather than the loop keeps the scan contiguous. The sub-threshold remainder
    stays in the last bin the scan reaches, so no mass is lost.
    """
    ordered = masses if from_left else masses[::-1]
    bin_probs = np.zeros(ordered.size, dtype=np.float64)
    accumulated_mass = 0.0
    for i in range(ordered.size):
        accumulated_mass += ordered[i]
        if accumulated_mass >= tail_truncation:
            bin_probs[i] = accumulated_mass
            accumulated_mass = 0.0
    if accumulated_mass > 0.0:
        bin_probs[ordered.size - 1] += accumulated_mass
    return bin_probs if from_left else bin_probs[::-1]
