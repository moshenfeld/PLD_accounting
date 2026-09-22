"""Continuous and discrete PMF construction with bound-preserving rounding.

Connect-the-dots builds each cell's endpoint masses from local source and
reflected-dual measures; stochastic discretization moves interval mass in the
requested direction.
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
    DenseDiscreteDist,
    DiscreteDistBase,
    Domain,
    GridSpec,
    PLDRealization,
)
from PLD_accounting.distribution_utils import (
    MAX_SAFE_EXP_ARG,
    classify_residual,
    compensated_segmented_sum,
    enforce_mass_conservation,
    exp_moment_terms,
    signed_unit_residual,
    trim_mass_to_moment_target,
)
from PLD_accounting.types import (
    BoundType,
    SpacingType,
    has_numba,
    optional_njit,
    require_bound_type,
)
from PLD_accounting.validation import (
    require_finite_array,
    require_open_unit_interval,
    require_positive_real,
)

# Slack allowed on the exact interval, cell and exterior inequalities before an
# inconsistency is treated as an oracle failure rather than as rounding of the inputs.
_ORACLE_TOL = 64.0 * float(np.finfo(np.float64).eps)
# Safety factor on the cellwise reciprocal-moment residual bound.
_CTD_MOMENT_REPAIR_FACTOR = 8.0
# Continuous interval oracle only: below three interior cells a cell spans O(support) and
# float64 CDF/SF differences fail the producer residual bound. Discrete CtD bins atoms exactly.
_MIN_CONTINUOUS_INTERVAL_KNOTS = 4

# =============================================================================
# Public API: Fixed-Gap Real-Loss Rediscretization
# =============================================================================


def rediscretize_dist_by_bound(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    loss_discretization: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Rediscretize a real-loss distribution using its fixed bound semantics.

    Dominating fixed-gap real-loss outputs use CtD. Dominated outputs use
    directional stochastic domination. This is structural routing by the
    mathematical bound direction, not a configurable discretization method.
    Semantic PLD checks do not prove domination of an external mechanism; the
    caller remains responsible for that proof.
    """
    if bound_type == BoundType.DOMINATES:
        return rediscretize_dist_ctd(
            dist=dist,
            tail_truncation=tail_truncation,
            loss_discretization=loss_discretization,
        )
    if bound_type == BoundType.IS_DOMINATED:
        return rediscretize_dist_stoch_dom(
            dist=dist,
            tail_truncation=tail_truncation,
            loss_discretization=loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
    raise ValueError(f"Unknown BoundType: {bound_type}")


# =============================================================================
# Discretization Engines
# =============================================================================


def discretize_continuous_ctd(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    dual_dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    step: float,
    align_to_multiples: bool,
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
    grid = aligned_grid_params(
        x_min=x_min,
        x_max=x_max,
        spacing_type=SpacingType.LINEAR,
        align_to_multiples=align_to_multiples,
        discretization=step,
    )
    if grid.n < _MIN_CONTINUOUS_INTERVAL_KNOTS:
        raise ValueError(
            f"loss_discretization={step:.6g} is coarser than the CtD source/dual support "
            f"[{x_min:.6g}, {x_max:.6g}] ({grid.n} knots); float64 interval measures are "
            "not valid on a grid this coarse. Reduce loss_discretization."
        )
    pld_pmf, pld_dual_pmf = _continuous_ctd_cell_measures(
        pld_in=dist,
        dual_pld_in=dual_dist,
        grid=grid,
    )
    return _ctd_realization_from_cell_measures(
        pld_pmf=pld_pmf,
        pld_dual_pmf=pld_dual_pmf,
        p_max_in=0.0,
        grid=grid,
    )


def discretize_continuous_stoch_dom(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    bound_type: BoundType,
    step: float,
    align_to_multiples: bool,
    domain: Domain = Domain.REALS,
) -> DenseDiscreteDist:
    """Discretize a continuous law with directional stochastic domination.

    Quantile bounds define the finite grid. Each interval then moves to its
    upper or lower knot according to ``bound_type``.
    """
    grid = aligned_grid_params(
        x_min=float(dist.ppf(tail_truncation)),
        x_max=float(dist.isf(tail_truncation)),
        spacing_type=SpacingType.LINEAR,
        align_to_multiples=align_to_multiples,
        discretization=step,
    )
    if grid.x_0 <= 0 and domain == Domain.POSITIVES:
        dist_name = getattr(dist, "name", type(dist).__name__)
        raise ValueError(f"Cannot discretize {dist_name} to a positive range, got x_0={grid.x_0}")
    return discretize_continuous_stoch_dom_on_grid(
        dist=dist,
        grid=grid,
        bound_type=bound_type,
        pmf_min_increment=tail_truncation,
        domain=domain,
    )


def discretize_continuous_stoch_dom_on_grid(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    grid: GridSpec,
    bound_type: BoundType,
    pmf_min_increment: float,
    domain: Domain = Domain.REALS,
) -> DenseDiscreteDist:
    """Discretize a continuous law onto a caller-supplied grid.

    Interval mass moves to the upper knot for ``DOMINATES`` and the lower knot
    for ``IS_DOMINATED``. ``pmf_min_increment`` batches smaller intervals.
    """
    x_array = grid.materialize()
    # Compute finite interval probabilities; the exterior tails remain separate.
    bin_probs, p_left, p_right = _compute_discrete_prob(
        dist=dist, x_array=x_array, bound_type=bound_type, pmf_min_increment=pmf_min_increment
    )
    prob_arr = np.zeros(grid.n)

    if bound_type == BoundType.DOMINATES:
        # Shift mass right: left tail (-inf, x0) -> x0,
        # each interval [x_i, x_{i+1}) -> x_{i+1}, right tail (x_n, inf) -> +inf.
        prob_arr[0] = p_left
        prob_arr[1:] = bin_probs
        p_min = 0.0
        p_max = p_right

    elif bound_type == BoundType.IS_DOMINATED:
        # Shift mass left: left tail (-inf, x0) -> -inf,
        # each interval [x_i, x_{i+1}) -> x_i, right tail (x_n, inf) -> x_n.
        prob_arr[:-1] = bin_probs
        prob_arr[-1] = p_right
        p_min = p_left
        p_max = 0.0

    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")
    return DenseDiscreteDist(
        grid=grid,
        prob_arr=prob_arr,
        p_min=p_min,
        p_max=p_max,
        domain=domain,
    )


def rediscretize_dist_ctd(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    loss_discretization: float,
) -> PLDRealization:
    """Rediscretize a real-loss PLD onto a fixed-gap grid with CtD.

    The semantic PLD contract is checked before truncation so the tail budget
    cannot hide an invalid ``-inf`` atom or reciprocal-moment excess.
    """
    _require_ctd_source(dist)
    # Account for truncated tails according to upper-bound semantics.
    trunc_dist = dist.truncate_edges(
        tail_truncation=tail_truncation / 2,
        bound_type=BoundType.DOMINATES,
    )
    x_min = float(trunc_dist.x_array[0])
    x_max = float(trunc_dist.x_array[-1])
    if x_max == x_min:
        raise ValueError(
            "rediscretize_dist_ctd requires at least two distinct finite "
            f"support points after truncation, got x_min=x_max={x_min}"
        )
    grid_out = aligned_grid_params(
        x_min=x_min,
        x_max=x_max,
        spacing_type=SpacingType.LINEAR,
        align_to_multiples=True,
        discretization=loss_discretization,
    )
    return project_dist_onto_grid_ctd(dist=trunc_dist, grid=grid_out)


def rediscretize_dist_stoch_dom(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    loss_discretization: float,
    spacing_type: SpacingType,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Rediscretize a distribution with directional stochastic domination.

    Finite atoms round upward for ``DOMINATES`` and downward for
    ``IS_DOMINATED``; conservatively movable real-domain boundary mass is first
    absorbed into the corresponding finite edge.

    Algorithm 6 (`disc-dist`), in Appendix C of https://arxiv.org/abs/2602.17284.
    """
    # On the real line, absorb the boundary that can be moved conservatively
    # onto the finite grid for the requested stochastic bound.
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
        # A lower-bound discretization is no longer an exact realization.
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

    # Route truncated tail mass according to the requested stochastic bound.
    trunc_dist = working_dist.truncate_edges(
        tail_truncation=tail_truncation / 2, bound_type=bound_type
    )

    x_array = trunc_dist.x_array
    grid_out = aligned_grid_params(
        x_min=x_array[0],
        x_max=x_array[-1],
        spacing_type=spacing_type,
        align_to_multiples=True,
        discretization=loss_discretization,
    )

    return project_dist_onto_grid_stoch_dom(
        dist=trunc_dist,
        grid=grid_out,
        bound_type=bound_type,
    )


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
    pld_pmf, pld_dual_pmf = _discrete_ctd_cell_measures(
        dist_in=dist,
        loss_out=grid.materialize(),
    )
    return _ctd_realization_from_cell_measures(
        pld_pmf=pld_pmf,
        pld_dual_pmf=pld_dual_pmf,
        p_max_in=float(dist.p_max),
        grid=grid,
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
    x_array_out = grid.materialize()
    expected_p_min = dist.p_min
    expected_p_max = dist.p_max
    # Round in-range atoms conservatively; the kernel omits directional overflow.
    prob_arr_out = rediscretize_prob(
        x_array=dist.x_array,
        prob_arr=dist.prob_arr,
        x_array_out=x_array_out,
        dominates=(bound_type == BoundType.DOMINATES),
    )

    if bound_type == BoundType.DOMINATES:
        # Right overflow becomes semantic upper-boundary mass.
        omitted = dist.prob_arr[dist.x_array > x_array_out[-1]]
        expected_p_max += math.fsum(map(float, omitted))
    else:
        # Left underflow becomes semantic lower-boundary mass.
        omitted = dist.prob_arr[dist.x_array < x_array_out[0]]
        expected_p_min += math.fsum(map(float, omitted))

    # Repair only numerical mass drift after semantic overflow is accounted for.
    prob_arr_out, p_min, p_max = enforce_mass_conservation(
        prob_arr=prob_arr_out,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )

    domain = Domain.POSITIVES if grid.spacing_type == SpacingType.GEOMETRIC else Domain.REALS
    return DenseDiscreteDist(
        grid=grid,
        prob_arr=prob_arr_out,
        p_min=p_min,
        p_max=p_max,
        domain=domain,
    )


def joint_source_dual_bounds(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    dual_dist: stats.rv_continuous | rv_frozen[Any, Any] | None,
    tail_truncation: float,
) -> tuple[float, float]:
    """Return finite bounds covering the source and reflected-dual laws.

    With a dual, each range query uses half of ``tail_truncation``. Pass
    ``dual_dist=None`` to retain the source-only range used by directional
    discretization.
    """
    require_open_unit_interval(value=tail_truncation, name="tail_truncation")
    range_tail = tail_truncation / 2.0 if dual_dist is not None else tail_truncation
    candidates = [float(dist.ppf(range_tail)), float(dist.isf(range_tail))]
    if dual_dist is not None:
        candidates += [
            -float(dual_dist.isf(range_tail)),
            -float(dual_dist.ppf(range_tail)),
        ]
    bounds = np.asarray(candidates, dtype=np.float64)
    require_finite_array(values=bounds, name="joint source/dual bounds")
    x_min, x_max = float(bounds.min()), float(bounds.max())
    if x_max <= x_min:
        raise ValueError(
            "Joint source/dual bounds must span a nonempty range, "
            f"got x_min={x_min}, x_max={x_max}"
        )
    return x_min, x_max


def aligned_grid_params(
    *,
    x_min: float,
    x_max: float,
    spacing_type: SpacingType,
    align_to_multiples: bool,
    discretization: float,
) -> GridSpec:
    """Return a ``GridSpec`` covering ``[x_min, x_max]``.

    ``discretization`` is the linear bin width or geometric log-ratio.
    ``align_to_multiples`` aligns the native coordinate to integer multiples.
    """
    if spacing_type not in (SpacingType.GEOMETRIC, SpacingType.LINEAR):
        raise ValueError(f"Unsupported spacing_type: {spacing_type}")
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got x_min={x_min}, x_max={x_max}")
    if spacing_type == SpacingType.GEOMETRIC and x_min <= 0:
        raise ValueError(
            f"Geometric spacing requires positive values, got x_min={x_min}, x_max={x_max}"
        )
    require_positive_real(value=discretization, name="discretization")

    d = float(discretization)
    if spacing_type == SpacingType.LINEAR:
        lower, upper, unaligned_anchor = x_min, x_max, x_min
    else:
        lower, upper, unaligned_anchor = math.log(x_min), math.log(x_max), x_min
    return _covering_grid(
        lower=lower,
        upper=upper,
        step=d,
        align_to_multiples=align_to_multiples,
        spacing_type=spacing_type,
        unaligned_anchor=unaligned_anchor,
        cover_max=x_max,
    )


def rediscretize_prob(
    *,
    x_array: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    x_array_out: NDArray[np.float64],
    dominates: bool,
) -> NDArray[np.float64]:
    """Dispatch PMF remapping to numba when available, else NumPy."""
    if has_numba():
        return _numba_rediscretize_prob(x_array, prob_arr, x_array_out, dominates)
    return _numpy_rediscretize_prob(x_array, prob_arr, x_array_out, dominates)


# =============================================================================
# Helper Functions
# =============================================================================


@optional_njit()
def _numba_rediscretize_prob(
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

            # Advance to the first output knot that bounds z from above.
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

            # Advance just past the last output knot that does not exceed z.
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


def _numpy_rediscretize_prob(
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


def _continuous_ctd_cell_measures(
    *,
    pld_in: stats.rv_continuous | rv_frozen[Any, Any],
    dual_pld_in: stats.rv_continuous | rv_frozen[Any, Any],
    grid: GridSpec,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate every CtD cell's mass and reciprocal moment from a PLD and its dual.

    Uses reflected left limits so a privacy-loss atom at a knot stays in the correct cell.
    """
    loss = grid.materialize()
    require_finite_array(values=loss, name="CtD grid")
    if loss.size < 2:
        raise ValueError("CtD projection requires at least two finite grid knots")
    mass, mass_below, mass_above = _partition_masses(dist=pld_in, points=loss, label="CtD source")
    reflected, reflected_above, reflected_below = _partition_masses(
        dist=dual_pld_in,
        points=np.nextafter(-loss[::-1], -np.inf),
        label="CtD reflected-dual",
    )
    return (
        np.concatenate(([mass_below], mass, [mass_above])),
        np.concatenate(([reflected_below], reflected[::-1], [reflected_above])),
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


def _ctd_realization_from_cell_measures(
    *,
    pld_pmf: NDArray[np.float64],
    pld_dual_pmf: NDArray[np.float64],
    p_max_in: float,
    grid: GridSpec,
) -> PLDRealization:
    """Build the CtD realization from per-cell source and dual measures (n+1 cells)."""
    if grid.spacing_type != SpacingType.LINEAR:
        raise ValueError("CtD output grid must be linear")
    loss = grid.materialize()
    if max(float(loss[-1]), -float(loss[0])) > MAX_SAFE_EXP_ARG:
        raise ValueError(
            f"CtD grid [{float(loss[0]):.6e}, {float(loss[-1]):.6e}] leaves the range where "
            "exp(x) is representable; the cells' moment factors cannot be formed"
        )
    contribution_left, contribution_right = _ctd_cell_endpoint_masses(
        pld_pmf=pld_pmf[1:-1],
        pld_dual_pmf=pld_dual_pmf[1:-1],
        left=loss[:-1],
        width=np.full(loss.size - 1, grid.step, dtype=np.float64),
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
        step=grid.step,
    )
    residual = signed_unit_residual(
        values=exp_moment_terms(prob_arr=prob, x_vals=loss), lower_term=0.0, upper_term=0.0
    )
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
    return PLDRealization(
        grid=grid,
        prob_arr=prob,
        p_min=0.0,
        p_max=p_max_out,
    )


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
    if upper_at_knot > upper_mass * (1.0 + _ORACLE_TOL):
        raise ValueError(
            "CtD upper exterior cell violates exp(x_N) R_plus <= M_plus: "
            f"{upper_at_knot:.3e} > {upper_mass:.3e}"
        )
    upper_at_knot = min(upper_at_knot, upper_mass)

    retained_lower_dual = math.exp(-first_knot) * lower_mass
    lower_semantic_dual_mass = lower_reflected_mass - retained_lower_dual
    if lower_semantic_dual_mass < -_ORACLE_TOL * max(lower_reflected_mass, 1.0):
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


def _partition_masses(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    points: NDArray[np.float64],
    label: str,
) -> tuple[NDArray[np.float64], float, float]:
    """Split unit mass over increasing ``points`` into cells and the two outer tails.

    Returns ``mass[i] = Pr[points[i] < X <= points[i+1]]`` plus ``Pr[X <= points[0]]`` and
    ``Pr[X > points[-1]]``; see ``_stable_cell_interval_masses`` for how the cells stay
    accurate in the tails.
    """
    cdf, sf = _stable_cdf_and_sf(dist=dist, x_array=points)
    if not np.all(np.isfinite(cdf)) or not np.all(np.isfinite(sf)):
        raise ValueError(f"{label} CDF and survival evaluations must be finite")
    mass = _stable_cell_interval_masses(cdf=cdf, sf=sf, label=label)
    return mass, float(cdf[0]), float(sf[-1])


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
    ``(M - S)/d`` at ``b``. ``r M <= S <= M`` holds exactly inside the cell.
    """
    pld_pmf = np.asarray(pld_pmf, dtype=np.float64)
    pld_dual_pmf = np.asarray(pld_dual_pmf, dtype=np.float64)
    if pld_pmf.shape != pld_dual_pmf.shape:
        raise ValueError("CtD source and reflected-dual cell arrays must have the same shape")
    if np.any(~np.isfinite(pld_pmf)) or np.any(~np.isfinite(pld_dual_pmf)):
        raise ValueError("CtD source and reflected-dual cell measures must be finite")
    if np.any(pld_pmf < 0.0) or np.any(pld_dual_pmf < 0.0):
        raise ValueError("CtD source and reflected-dual cell measures must be nonnegative")

    source_zero = pld_pmf == 0.0
    dual_zero = pld_dual_pmf == 0.0
    if np.any(source_zero != dual_zero):
        raise ValueError(
            "CtD source/dual interval measures are inconsistent: source and reflected-dual "
            "cell masses must either both be zero or both be positive"
        )
    active = ~source_zero
    denominator = -np.expm1(-width)
    fraction = np.zeros_like(pld_pmf)
    fraction[active] = (
        (pld_pmf[active] - np.exp(left[active]) * pld_dual_pmf[active])
        / denominator[active]
        / pld_pmf[active]
    )
    if np.any(np.abs(fraction - 0.5) > 0.5 + _ORACLE_TOL / denominator):
        raise ValueError(
            "CtD source/dual interval measures violate r*M <= exp(a)*R <= M: endpoint "
            f"fraction range [{float(fraction.min()):.3e}, {float(fraction.max()):.3e}] "
            "outside [0, 1]"
        )
    fraction = np.clip(fraction, 0.0, 1.0)
    weight_right = np.where(
        fraction <= 0.5, pld_pmf * fraction, pld_pmf - pld_pmf * (1.0 - fraction)
    )
    return pld_pmf - weight_right, weight_right


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

    repair_tol = _ctd_moment_repair_tol(
        max_abs_loss=float(np.max(np.abs(loss))),
        step=step,
    )
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
    amplification = max(1.0, abs(max_abs_loss)) / max(denominator, float(np.finfo(np.float64).tiny))
    return _CTD_MOMENT_REPAIR_FACTOR * amplification * float(np.finfo(np.float64).eps)


def _covering_int_range(*, lower: float, upper: float, step: float) -> tuple[int, int]:
    """Return integer indices whose multiples of ``step`` cover ``[lower, upper]``."""
    k_lo = int(np.floor(lower / step))
    k_hi = int(np.ceil(upper / step))
    if step * k_lo > lower:
        k_lo -= 1
    if step * k_hi < upper:
        k_hi += 1
    return k_lo, k_hi


def _covering_grid(
    *,
    lower: float,
    upper: float,
    step: float,
    align_to_multiples: bool,
    spacing_type: SpacingType,
    unaligned_anchor: float,
    cover_max: float,
) -> GridSpec:
    """Cover ``[lower, upper]`` in the lattice's native coordinate, then grow to ``cover_max``."""
    if align_to_multiples:
        k_lo, k_hi = _covering_int_range(lower=lower, upper=upper, step=step)
        grid = GridSpec(
            step=step,
            n=k_hi - k_lo + 1,
            spacing_type=spacing_type,
            anchor=1.0 if spacing_type == SpacingType.GEOMETRIC else 0.0,
            index_0=k_lo,
        )
    else:
        grid = GridSpec(
            step=step,
            n=int(np.ceil((upper - lower) / step)) + 1,
            spacing_type=spacing_type,
            anchor=unaligned_anchor,
        )
    return _cover_x_max(grid=grid, x_max=cover_max)


def _cover_x_max(*, grid: GridSpec, x_max: float) -> GridSpec:
    """Grow ``n`` until the materialized endpoint covers ``x_max`` after float rounding."""
    candidate = grid
    while candidate.last_point < x_max:
        candidate = candidate.with_n(candidate.n + 1)
    return candidate


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


def _stable_cell_interval_masses(
    *,
    cdf: NDArray[np.float64],
    sf: NDArray[np.float64],
    label: str,
) -> NDArray[np.float64]:
    """Return nonnegative ``Pr[x_i < X <= x_{i+1}]`` without catastrophic cancellation.

    ``cdf[i+1] - cdf[i]`` zeroes far upper-tail cells once both values round to nearly 1,
    silently weakening an upper bound. Each side of the median crossing therefore reads
    whichever cumulative is the small quantity there.

    A cumulative that is monotone in exact arithmetic need not be monotone in binary64, so
    a cell may come back a few ULP negative. That much is clipped; more is an oracle
    failure, because clipping it would manufacture mass in a bound.
    """
    pivot = min(max(int(np.searchsorted(cdf, 0.5)), 1), cdf.size - 1)
    mass = np.empty(cdf.size - 1, dtype=np.float64)
    mass[:pivot] = np.diff(cdf[: pivot + 1])
    mass[pivot:] = -np.diff(sf[pivot:])
    if np.any(mass < -_ORACLE_TOL):
        raise ValueError(
            f"{label} interval oracle returned a negative probability: "
            f"most negative cell {float(mass.min()):.3e}"
        )
    return np.maximum(mass, 0.0)


def _stable_cdf_and_sf(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    x_array: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate CDF and survival function without catastrophic cancellation.

    Below the median the CDF is the small quantity and the SF is recovered from
    it; above the median the roles swap. Both are computed through the log
    variants so the small side keeps full relative precision instead of being
    formed as ``1 - (nearly 1)``.
    """
    missing = [name for name in ("logcdf", "logsf", "median") if not hasattr(dist, name)]
    if missing:
        raise TypeError(
            f"{getattr(dist, 'name', type(dist).__name__)} lacks {', '.join(missing)}; "
            "interval measures require log primitives to stay accurate in the tails"
        )
    median = dist.median()
    cdf = np.empty_like(x_array, dtype=np.float64)
    sf = np.empty_like(x_array, dtype=np.float64)

    mask_left = x_array < median
    if np.any(mask_left):
        logcdf_vals = dist.logcdf(x_array[mask_left])
        cdf[mask_left] = np.exp(logcdf_vals)
        sf[mask_left] = -np.expm1(logcdf_vals)

    mask_right = ~mask_left
    if np.any(mask_right):
        logsf_vals = dist.logsf(x_array[mask_right])
        sf[mask_right] = np.exp(logsf_vals)
        cdf[mask_right] = -np.expm1(logsf_vals)

    cdf = np.clip(cdf, 0.0, 1.0)
    sf = np.clip(sf, 0.0, 1.0)
    return cdf, sf


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
    bound_type = require_bound_type(value=bound_type)
    cdf, sf = _stable_cdf_and_sf(dist=dist, x_array=x_array)
    cell_masses = _stable_cell_interval_masses(cdf=cdf, sf=sf, label="directional source")
    # Direction only picks the scan: a dominating bin lands on each interval's upper knot,
    # a dominated one on the lower. Both orientations bin the same cell masses.
    bin_probs = _adaptive_bins_from_masses(
        masses=cell_masses,
        tail_truncation=max(0.0, pmf_min_increment),
        from_left=bound_type == BoundType.DOMINATES,
    )
    return bin_probs, float(cdf[0]), float(sf[-1])
