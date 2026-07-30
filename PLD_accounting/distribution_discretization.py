"""Distribution discretization utilities for PMF construction."""

from __future__ import annotations

import math
from dataclasses import replace
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
    PMF_MASS_TOL,
    enforce_mass_conservation,
    exp_moment_terms,
    kahan_reverse_exclusive_cumsum,
)
from PLD_accounting.types import (
    BoundType,
    SpacingType,
    has_numba,
    optional_njit,
)
from PLD_accounting.validation import validate_finite_array

# =============================================================================
# Public API: Continuous Distribution Discretization
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

    Args:
        dist: Continuous privacy-loss distribution.
        dual_dist: Exact dual privacy-loss distribution.
        tail_truncation: Tail probability used to define finite grid bounds.
        step: Linear bin width.
        align_to_multiples: Whether to align the quantile-derived bounds to integer step multiples.
    """
    grid = aligned_grid_params(
        x_min=float(dist.ppf(tail_truncation)),
        x_max=float(dist.isf(tail_truncation)),
        spacing_type=SpacingType.LINEAR,
        align_to_multiples=align_to_multiples,
        discretization=step,
    )
    privacy_profile = _continuous_real_privacy_profile(
        pld_in=dist,
        dual_pld_in=dual_dist,
        eps_out=grid.materialize(),
    )
    return _pld_from_privacy_profile_ctd(
        privacy_profile=privacy_profile,
        x_0_out=grid.x_0,
        step_out=grid.step,
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
    """Discretize a continuous law with directional stochastic domination."""
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
    """Discretize a continuous law onto a caller-supplied grid with stochastic domination.

    Interval mass is assigned to the upper knot for ``DOMINATES`` and the
    lower knot for ``IS_DOMINATED``; the opposite tail remains a boundary mass.

    Args:
        dist: Continuous law to discretize.
        grid: Exact output grid; no quantile-derived range is computed here.
        bound_type: Rounding direction for interval mass.
        pmf_min_increment: Minimum CDF/SF increment that becomes a bin mass.
        domain: Support-domain semantics of the result.

    Returns:
        The discretized distribution on ``grid``.
    """
    x_array = grid.materialize()
    # Compute the finite interval probabilities; tails remain separate boundary masses.
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
        x_0=grid.x_0,
        step=grid.step,
        prob_arr=prob_arr,
        p_min=p_min,
        p_max=p_max,
        domain=domain,
    )


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


def rediscretize_dist_ctd(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    loss_discretization: float,
) -> PLDRealization:
    """Rediscretize a real-loss PLD onto a fixed-gap grid with CtD."""
    # Validate before truncation so a small, invalid -inf atom cannot be
    # consumed by the tail budget and thereby hidden from CtD validation.
    _validate_ctd_source(dist)
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

    Remaps PMF onto a new grid with the requested spacing and discretization.
    Implementation trims zero/tail regions, computes new grid size, then remaps
    using domination-aware rounding (e.g., linear grids for dp_accounting output).

    Algorithm 6 (`disc-dist`), in Appendix C
    of https://arxiv.org/abs/2602.17284.
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
                x_0=dist.x_0,
                step=dist.step,
                prob_arr=dist.prob_arr,
                p_min=dist.p_min,
                p_max=dist.p_max,
                spacing_type=dist.spacing_type,
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

    # Move truncated tail mass according to the requested bound.
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
    """CtD-project a valid real-loss PLD source onto a fixed-gap grid.

    CtD accepts any discrete source that is itself a semantically valid PLD
    realization.  This validates the source object only; callers remain
    responsible for ensuring that it dominates the external mechanism being
    accounted for.
    """
    if grid.spacing_type != SpacingType.LINEAR:
        raise ValueError("CtD projection requires a fixed-gap linear grid")
    _validate_ctd_source(dist)
    privacy_profile = _discrete_dist_privacy_profile(
        dist_in=dist,
        eps_out=grid.materialize(),
    )
    return _pld_from_privacy_profile_ctd(
        privacy_profile=privacy_profile,
        x_0_out=grid.x_0,
        step_out=grid.step,
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

    if grid.spacing_type == SpacingType.LINEAR:
        return DenseDiscreteDist(
            x_0=grid.x_0,
            step=grid.step,
            prob_arr=prob_arr_out,
            p_min=p_min,
            p_max=p_max,
        )

    if grid.spacing_type == SpacingType.GEOMETRIC:
        return DenseDiscreteDist(
            x_0=grid.x_0,
            step=grid.step,
            prob_arr=prob_arr_out,
            p_min=p_min,
            p_max=p_max,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

    raise ValueError(f"Invalid spacing_type: {grid.spacing_type}")


def aligned_grid_params(
    *,
    x_min: float,
    x_max: float,
    spacing_type: SpacingType,
    align_to_multiples: bool,
    discretization: float,
) -> GridSpec:
    """Return a :class:`GridSpec` covering [x_min, x_max].

    The returned spec is the single source of truth for a uniform grid and is
    meant to be passed straight to grid consumers, avoiding any re-derivation
    of the spacing from a materialized array.

    Args:
        x_min: Minimum value of the range.
        x_max: Maximum value of the range.
        spacing_type: Type of spacing (LINEAR or GEOMETRIC).
        align_to_multiples: If True, align range to whole multiples of discretization.
                           If False, use x_min and x_max directly without alignment.
        discretization: Grid spacing parameter (step size for LINEAR, log ratio for GEOMETRIC).

    Returns:
        A ``GridSpec`` whose ``step`` is the additive bin width (LINEAR) or
        multiplicative ratio (GEOMETRIC).
    """
    if spacing_type not in (SpacingType.GEOMETRIC, SpacingType.LINEAR):
        raise ValueError(f"Unsupported spacing_type: {spacing_type}")
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got x_min={x_min}, x_max={x_max}")
    if spacing_type == SpacingType.GEOMETRIC and x_min <= 0:
        raise ValueError(
            f"Geometric spacing requires positive values, got x_min={x_min}, x_max={x_max}"
        )
    if discretization <= 0:
        raise ValueError("discretization must be positive")

    d = float(discretization)
    if spacing_type == SpacingType.LINEAR:
        return _linear_grid_params(x_min, x_max, d, align_to_multiples)
    return _geometric_grid_params(x_min, x_max, d, align_to_multiples)


def rediscretize_prob(
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

    Maps each probability mass to output grid position based on domination semantics.
    Implementation: dominates=True uses ceil (pessimistic), False uses floor (optimistic).
    Uses Kahan summation for numerical accuracy.
    """
    n_out = x_array_out.size
    prob_arr_out = np.zeros(n_out)
    compensations = np.zeros(n_out)

    # single pointer into x_array_out since x_array is strictly increasing
    j = 0

    if dominates:
        # ceil: bin = first index with x_array_out[j] >= z; overflow right -> p_max
        for i in range(x_array.size):
            z = x_array[i]
            mass = prob_arr[i]
            # Skip only zero-mass bins, not small-mass bins
            if mass <= 0:
                continue

            # advance while x_array_out[j] < z
            while j < n_out and x_array_out[j] < z:
                j += 1

            if j >= n_out:
                # Overflow is omitted here and accounted explicitly by the caller.
                continue
            # include values below x_array_out[0] in the first bin (ceil behavior)
            y = mass - compensations[j]
            t = prob_arr_out[j] + y
            compensations[j] = (t - prob_arr_out[j]) - y
            prob_arr_out[j] = t

    else:
        # floor: bin = last index with x_array_out[j] <= z; underflow left -> p_min
        for i in range(x_array.size):
            z = x_array[i]
            mass = prob_arr[i]
            # Skip only zero-mass bins, not small-mass bins
            if mass <= 0:
                continue

            # advance while x_array_out[j] <= z
            while j < n_out and x_array_out[j] <= z:
                j += 1

            idx = j - 1
            if idx < 0:
                # Underflow is omitted here and accounted explicitly by the caller.
                continue
            # include values above x_array_out[-1] in the last bin (floor behavior)
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


def _continuous_real_privacy_profile(
    *,
    pld_in: stats.rv_continuous | rv_frozen[Any, Any],
    dual_pld_in: stats.rv_continuous | rv_frozen[Any, Any],
    eps_out: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate ``delta(eps)`` from a PLD and its dual.

    Atoms at ``L = eps`` contribute zero to the hockey-stick divergence. Use
    the strict identity
    ``delta(eps) = Pr[L > eps] - exp(eps) Pr[D(L) < -eps]``. The dual's strict
    CDF is evaluated at the representable point immediately below ``-eps``.
    """
    eps = np.asarray(eps_out, dtype=np.float64)
    validate_finite_array(eps, "privacy-profile epsilon")
    pld_sf = np.asarray(pld_in.sf(eps), dtype=np.float64)
    dual_left_limit = np.nextafter(-eps, -np.inf)
    dual_log_cdf = np.asarray(dual_pld_in.logcdf(dual_left_limit), dtype=np.float64)
    if np.any(~np.isfinite(pld_sf)) or np.any(np.isnan(dual_log_cdf)):
        raise ValueError("PLD and dual CDF evaluations must not be NaN")
    dual_cdf_eps = _safe_exp(eps + dual_log_cdf)
    return _validate_real_privacy_profile(eps=eps, profile=pld_sf - dual_cdf_eps)


def _discrete_dist_privacy_profile(
    *,
    dist_in: DiscreteDistBase,
    eps_out: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Evaluate a PLD realization's hockey-stick profile."""
    eps = np.asarray(eps_out, dtype=np.float64)
    validate_finite_array(eps, "privacy-profile epsilon")
    # Include the positive-infinity boundary atom locally so the tail sums use
    # one representation for both finite and boundary mass.
    losses = np.concatenate((dist_in.x_array, np.array([np.inf])))
    masses = np.concatenate((dist_in.prob_arr, np.array([dist_in.p_max])))
    reciprocal_moment = exp_moment_terms(prob_arr=masses, x_vals=losses)
    reciprocal_moment_total = math.fsum(map(float, reciprocal_moment))

    tail_mass = kahan_reverse_exclusive_cumsum(np.concatenate(([0.0], masses)))
    tail_moment = kahan_reverse_exclusive_cumsum(np.concatenate(([0.0], reciprocal_moment)))
    indices = np.asarray(np.searchsorted(losses, eps, side="right"), dtype=np.intp)

    selected_moment = tail_moment[indices]
    log_selected_moment = np.full_like(selected_moment, -np.inf)
    positive = selected_moment > 0.0
    log_selected_moment[positive] = np.log(selected_moment[positive])
    profile = tail_mass[indices] - _safe_exp(eps + log_selected_moment)
    return _validate_real_privacy_profile(
        eps=eps,
        profile=profile,
        reciprocal_moment=reciprocal_moment_total,
    )


def _pld_from_privacy_profile_ctd(
    *,
    privacy_profile: NDArray[np.float64],
    x_0_out: float,
    step_out: float,
) -> PLDRealization:
    """PLD-validating inversion on output knots ``x_0_out + k * step_out``."""
    delta = np.asarray(privacy_profile, dtype=np.float64)
    if step_out <= 0.0:
        raise ValueError("CtD output step must be positive")
    if delta.size < 2:
        raise ValueError("CtD profile inversion requires at least two values")
    validate_finite_array(delta, "CtD privacy profile")
    if np.any(delta < 0.0) or np.any(delta > 1.0):
        raise ValueError("CtD privacy profile must be finite and lie in [0, 1]")
    # Privacy profiles are non-increasing in epsilon. Evaluators can violate
    # this by a few ULPs in extreme tails; remove only that numerical noise
    # before the fixed-gap CtD inversion, and reject material violations.
    increases = np.diff(delta)
    if np.any(increases > PMF_MASS_TOL):
        raise ValueError(
            "privacy profile is not convex/monotone enough for fixed-gap CtD inversion"
        )
    delta = np.minimum.accumulate(delta)

    exp_step = math.exp(-step_out)
    denominator = -math.expm1(-step_out)
    diff = np.diff(delta)
    prob = np.empty_like(delta)
    prob[0] = 1.0 - delta[0] + exp_step * diff[0] / denominator
    prob[1:-1] = (exp_step * diff[1:] - diff[:-1]) / denominator
    prob[-1] = -diff[-1] / denominator
    # The division by ``denominator`` amplifies cancellation noise in the
    # delta differences by 1/(1 - exp(-step)), so test convexity in delta
    # space (numerator scale) rather than on the amplified probabilities.
    if np.min(prob) < -PMF_MASS_TOL / denominator:
        raise ValueError("privacy profile is not convex enough for fixed-gap CtD inversion")
    # Remove negative inversion noise before restoring total numerical mass.
    prob = np.maximum(prob, 0.0)
    # delta[-1] is the CtD profile's semantic +inf atom.
    prob, p_min, p_max = enforce_mass_conservation(
        prob_arr=prob,
        expected_p_min=0.0,
        expected_p_max=float(delta[-1]),
        bound_type=BoundType.DOMINATES,
    )
    return PLDRealization(
        x_0=float(x_0_out),
        step=float(step_out),
        prob_arr=prob,
        p_min=p_min,
        p_max=p_max,
    )


def _validate_real_privacy_profile(
    *,
    eps: NDArray[np.float64],
    profile: NDArray[np.float64],
    reciprocal_moment: float | None = None,
) -> NDArray[np.float64]:
    """Validate universal range and negative-epsilon PLD constraints."""
    profile = np.asarray(profile, dtype=np.float64)
    validate_finite_array(profile, "privacy profile")
    if profile.shape != eps.shape:
        raise ValueError("privacy profile shape must match epsilon shape")
    if np.any(profile < -PMF_MASS_TOL) or np.any(profile > 1.0 + PMF_MASS_TOL):
        raise ValueError("privacy profile must lie in [0, 1]")
    profile = np.clip(profile, 0.0, 1.0)
    negative = eps < 0.0
    moment_for_floor = (
        1.0 + REALIZATION_MOMENT_TOL
        if reciprocal_moment is None
        else min(reciprocal_moment, 1.0 + REALIZATION_MOMENT_TOL)
    )
    pld_floor = 1.0 - np.exp(eps[negative]) * moment_for_floor
    if np.any(profile[negative] < pld_floor - PMF_MASS_TOL):
        raise ValueError("privacy profile violates the PLD lower bound for negative epsilon")
    return profile


def _validate_ctd_source(dist: DiscreteDistBase) -> None:
    """Validate the semantic PLD-realization contract required by CtD."""
    if dist.domain != Domain.REALS:
        raise ValueError("CtD projection requires a real-domain source")
    if dist.p_min != 0.0:
        raise ValueError(f"CtD projection requires p_min = 0 exactly, got {dist.p_min:.2e}")
    support = np.asarray(dist.x_array, dtype=np.float64)
    validate_finite_array(support, "CtD source support")
    if support.size == 0 or np.any(np.diff(support) <= 0.0):
        raise ValueError("CtD projection requires finite, strictly increasing support")
    moment_terms = exp_moment_terms(prob_arr=dist.prob_arr, x_vals=support)
    if np.any(~np.isfinite(moment_terms)):
        raise ValueError("CtD source reciprocal moment must be finite")
    reciprocal_moment = math.fsum(map(float, moment_terms))
    if reciprocal_moment > 1.0 + REALIZATION_MOMENT_TOL:
        raise ValueError(
            "CtD source reciprocal-moment violates E[exp(-L)] <= 1 under the "
            "PLD invariant tolerance"
        )


def _safe_exp(log_values_in: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exponentiate log-values, flushing underflow to 0 and clamping overflow.

    Privacy-profile terms routinely underflow in deep tails, where the exact
    value is indistinguishable from zero; clamping the upper end keeps a single
    ``inf`` from poisoning an otherwise finite profile.
    """
    out = np.zeros_like(log_values_in, dtype=np.float64)
    active = log_values_in > math.log(np.finfo(float).tiny)
    out[active] = np.exp(np.minimum(log_values_in[active], math.log(np.finfo(float).max)))
    return out


def _linear_grid_params(x_min: float, x_max: float, d: float, align_to_multiples: bool) -> GridSpec:
    """Return a ``GridSpec`` for a uniformly-spaced linear grid covering [x_min, x_max]."""
    if align_to_multiples:
        k_lo = int(np.floor(x_min / d))
        k_hi = int(np.ceil(x_max / d))
        # It is possible that `ceil(x/d)*d < x` in float64 due to floating numerics
        if d * k_lo > x_min:
            k_lo -= 1
        if d * k_hi < x_max:
            k_hi += 1
        x0 = d * k_lo
        n = k_hi - k_lo + 1
    else:
        x0 = x_min
        n = int(np.ceil((x_max - x_min) / d)) + 1
    return _cover_x_max(GridSpec(x_0=x0, step=d, n=n, spacing_type=SpacingType.LINEAR), x_max)


def _geometric_grid_params(
    x_min: float, x_max: float, d: float, align_to_multiples: bool
) -> GridSpec:
    """Return a geometric ``GridSpec`` covering [x_min, x_max].

    ``d`` is the log-ratio per step.
    """
    step = float(np.exp(d))
    if align_to_multiples:
        k_lo = int(np.floor(np.log(x_min) / d))
        k_hi = int(np.ceil(np.log(x_max) / d))
        # It is possible that `ceil(x/d)*d < x` in float64 due to floating numerics
        if np.exp(d * k_lo) > x_min:
            k_lo -= 1
        if np.exp(d * k_hi) < x_max:
            k_hi += 1
        x0 = float(np.exp(d * k_lo))
        n = k_hi - k_lo + 1
    else:
        x0 = x_min
        n = int(np.ceil(np.log(x_max / x_min) / d)) + 1
    return _cover_x_max(GridSpec(x_0=x0, step=step, n=n, spacing_type=SpacingType.GEOMETRIC), x_max)


def _cover_x_max(grid: GridSpec, x_max: float) -> GridSpec:
    """Grow ``n`` until the materialized endpoint covers ``x_max`` after float rounding."""
    n = grid.n
    candidate = grid
    while candidate.last_point() < x_max:
        n += 1
        candidate = replace(grid, n=n)
    return candidate


@optional_njit()
def _adaptive_bins_from_cdf(
    *,
    cdf: NDArray[np.float64],
    tail_truncation: float,
) -> NDArray[np.float64]:
    """Adaptive binning from CDF with mass accumulation.

    Accumulates mass from CDF increments until threshold is reached, then assigns
    accumulated mass to current bin. All mass is conserved - no mass is discarded.
    """
    n = cdf.size
    bin_probs = np.zeros(n - 1, dtype=np.float64)
    accumulated_mass = 0.0

    for i in range(n - 1):
        # Current increment in CDF
        current_increment = cdf[i + 1] - cdf[i]
        accumulated_mass += current_increment

        if accumulated_mass >= tail_truncation:
            # Assign accumulated mass to this bin
            bin_probs[i] = accumulated_mass
            accumulated_mass = 0.0

    # Assign any remaining accumulated mass to the last bin
    if accumulated_mass > 0.0:
        bin_probs[n - 2] += accumulated_mass

    return bin_probs


@optional_njit()
def _adaptive_bins_from_sf(
    *,
    sf: NDArray[np.float64],
    tail_truncation: float,
) -> NDArray[np.float64]:
    """Adaptive binning from survival function with mass accumulation.

    Accumulates mass from SF increments until threshold is reached, then assigns
    accumulated mass to current bin. All mass is conserved - no mass is discarded.
    Processes from right to left (high to low x values).
    """
    n = sf.size
    bin_probs = np.zeros(n - 1, dtype=np.float64)
    accumulated_mass = 0.0

    for i in range(n - 2, -1, -1):
        # Current increment in SF (going backwards)
        current_increment = sf[i] - sf[i + 1]
        accumulated_mass += current_increment

        if accumulated_mass >= tail_truncation:
            # Assign accumulated mass to this bin
            bin_probs[i] = accumulated_mass
            accumulated_mass = 0.0

    # Assign any remaining accumulated mass to the first bin
    if accumulated_mass > 0.0:
        bin_probs[0] += accumulated_mass

    return bin_probs


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
    """Compute bin probabilities using adaptive CDF/SF increments with logcdf/logsf stability.

    pmf_min_increment controls the minimum CDF/SF increment that becomes a bin mass.

    """
    cdf, sf = _stable_cdf_and_sf(
        dist=dist,
        x_array=x_array,
    )
    p_left = cdf[0]
    p_right = sf[-1]
    pmf_min_increment = max(0.0, pmf_min_increment)

    if bound_type == BoundType.DOMINATES:
        # A dominating bin mass accumulates upward from the CDF, so mass lands on
        # the upper knot of each interval.
        bin_probs = _adaptive_bins_from_cdf(
            cdf=cdf,
            tail_truncation=pmf_min_increment,
        )
    elif bound_type == BoundType.IS_DOMINATED:
        # A dominated bin mass accumulates downward from the survival function,
        # so mass lands on the lower knot of each interval.
        bin_probs = _adaptive_bins_from_sf(
            sf=sf,
            tail_truncation=pmf_min_increment,
        )
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    return bin_probs, p_left, p_right
