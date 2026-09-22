"""Mass, moment and summation primitives shared by every distribution operation.

Mass conservation and residual policy, reciprocal-moment repair, compensated
accumulation, and edge truncation. All work on bare arrays, so they can be used
before a distribution object exists.

Repairs take their admission band from the caller rather than choosing one: the
operation that owns the budget decides what counts as tolerable drift.
"""

from __future__ import annotations

import math
import warnings
from itertools import chain

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.types import BoundType, optional_njit
from PLD_accounting.validation import (
    require_closed_unit_interval,
    require_nonnegative_masses,
)

# Drift is repaired silently; larger repairs are directional and warn.
# Units: probability mass.
# These bands assume a compensated producer (``math.fsum`` /
# ``compensated_segmented_sum``). An uncompensated producer such as FFT must pass an
# explicit size-scaled pair from ``_fft_mass_tolerances``; inheriting this default
# without scaling raises inside the repair helper.
PMF_MASS_DRIFT_TOL: float = float(4 * np.finfo(float).eps)
PMF_TOLERATED_MASS_TOL: float = float(10 * np.finfo(float).eps)
MIN_GRID_SIZE = 100
MAX_SAFE_EXP_ARG = math.log(np.finfo(np.float64).max)


# =============================================================================
# Mass Conservation and Residual Policy
# =============================================================================


def enforce_mass_conservation(
    *,
    prob_arr: NDArray[np.float64],
    expected_p_min: float,
    expected_p_max: float,
    bound_type: BoundType,
    drift_tol: float = PMF_MASS_DRIFT_TOL,
    repair_tol: float = PMF_TOLERATED_MASS_TOL,
    context: str = "enforce_mass_conservation",
) -> tuple[NDArray[np.float64], float, float]:
    """Enforce total mass, holding the bound-type-selected boundary fixed.

    ``DOMINATES`` fixes ``p_max`` and repairs from the low-loss side of
    ``[p_min, *prob_arr]``; ``IS_DOMINATED`` fixes ``p_min`` and repairs from
    the high-loss side of ``[*prob_arr, p_max]``. Callers must include genuine
    omitted support in the expected boundaries. Residuals below ``drift_tol``
    are repaired silently; larger admitted repairs warn.
    """
    if not 0.0 <= drift_tol <= repair_tol:
        raise ValueError(
            f"require 0 <= drift_tol <= repair_tol, got drift_tol={drift_tol:.3e}, "
            f"repair_tol={repair_tol:.3e}"
        )
    prob_arr = np.asarray(prob_arr, dtype=np.float64).copy()
    # Clamp boundary overshoot first so a float that landed just outside [0, 1] is
    # admitted before the nonnegative check sees it. Callers then use the coerced
    # values; neither branch below range-checks or clamps again.
    expected_p_min, expected_p_max = require_closed_unit_interval(
        value=[expected_p_min, expected_p_max],
        name=["expected_p_min", "expected_p_max"],
        atol=PMF_TOLERATED_MASS_TOL,
    )
    require_nonnegative_masses(prob_arr=prob_arr, p_min=expected_p_min, p_max=expected_p_max)
    total_mass = 1.0 - signed_unit_residual(
        values=prob_arr, lower_term=expected_p_min, upper_term=expected_p_max
    )
    if total_mass <= 0.0:
        raise ValueError("Cannot enforce mass conservation with zero total mass")

    if bound_type == BoundType.DOMINATES:
        # Hold semantic p_max fixed; every other mass is giveable from the left.
        fixed_boundary = expected_p_max
        extended = np.concatenate(([expected_p_min], prob_arr))
        from_left, deficit_edge = True, -1
    elif bound_type == BoundType.IS_DOMINATED:
        # Hold semantic p_min fixed; every other mass is giveable from the right.
        fixed_boundary = expected_p_min
        extended = np.concatenate((prob_arr, [expected_p_max]))
        from_left, deficit_edge = False, 0
    else:
        raise ValueError(
            f"Invalid bound_type: {bound_type}. "
            "Must be BoundType.DOMINATES or BoundType.IS_DOMINATED."
        )

    # One residual, repaired in whichever direction it points: negative is surplus mass,
    # positive a shortfall.
    residual = signed_unit_residual(values=extended, lower_term=0.0, upper_term=fixed_boundary)
    if residual != 0.0:
        # The bands judge what the caller handed in, so they are applied once, here.
        classify_residual(
            residual=abs(residual),
            drift_tol=drift_tol,
            repair_tol=repair_tol,
            context=context,
            repair=(
                "trimming it from the giveable edge"
                if residual < 0.0
                else "assigning it to the conservative edge"
            ),
        )

    if residual < 0.0:
        # Surplus: give it back from the edge this bound type is free to move.
        extended = _drain_mass_from_edge(
            values=extended, mass=-residual, from_left=from_left, exact=True
        )
        residual = signed_unit_residual(values=extended, lower_term=0.0, upper_term=fixed_boundary)

    if residual > 0.0:
        # Shortfall, either the caller's or the split remainder above.
        extended[deficit_edge] += residual

    if bound_type == BoundType.DOMINATES:
        return extended[1:].copy(), float(extended[0]), fixed_boundary
    return extended[:-1].copy(), fixed_boundary, float(extended[-1])


def trim_mass_from_edge(
    *,
    prob_arr: NDArray[np.float64],
    mass: float,
    from_left: bool,
) -> NDArray[np.float64]:
    """Return a copy with exactly ``mass`` removed from one support edge."""
    return _drain_mass_from_edge(
        values=np.asarray(prob_arr, dtype=np.float64).copy(),
        mass=mass,
        from_left=from_left,
        exact=True,
    )


def signed_unit_residual(
    *,
    values: NDArray[np.float64],
    lower_term: float,
    upper_term: float,
) -> float:
    """Return ``1 - (lower_term + sum(values) + upper_term)`` using ``math.fsum``."""
    return math.fsum(
        chain(
            (1.0, -float(lower_term), -float(upper_term)),
            (-float(value) for value in np.asarray(values, dtype=np.float64)),
        )
    )


def classify_residual(
    *,
    residual: float,
    drift_tol: float,
    repair_tol: float,
    context: str,
    repair: str,
) -> None:
    """Admit drift silently, warn for repair, and raise at the ceiling."""
    if not 0.0 <= drift_tol <= repair_tol:
        raise ValueError(
            f"require 0 <= drift_tol <= repair_tol, got drift_tol={drift_tol:.3e}, "
            f"repair_tol={repair_tol:.3e}"
        )
    if residual >= repair_tol:
        raise ValueError(
            f"{context}: residual {residual:.6e} exceeds the repair tolerance "
            f"{repair_tol:.3e}. A residual this large is outside the declared numerical "
            "repair policy."
        )
    if residual < drift_tol:
        return
    warnings.warn(
        f"{context}: residual {residual:.3e} exceeds the drift tolerance "
        f"{drift_tol:.3e}; {repair}.",
        RuntimeWarning,
        stacklevel=4,
    )


# =============================================================================
# Reciprocal-Moment Repair
# =============================================================================


def trim_mass_to_moment_target(
    *,
    prob_arr: NDArray[np.float64],
    loss: NDArray[np.float64],
    p_max: float,
    target_residual: float = 0.0,
    max_removed: float | None = None,
    context: str = "moment repair",
) -> tuple[NDArray[np.float64], float]:
    """Trim low-loss mass until the reciprocal residual reaches its target.

    Removed mass is banked at ``p_max``. A target of zero restores the PLD
    invariant; a positive target surrenders additional moment required by the
    caller's semantics. Pass ``p_max=0`` to read the movement from the return value.
    """
    original = np.asarray(prob_arr, dtype=np.float64)
    loss = np.asarray(loss, dtype=np.float64)
    if original.shape != loss.shape:
        raise ValueError("prob_arr and loss must have the same shape")
    target_residual = require_closed_unit_interval(value=target_residual, name="target_residual")

    repaired = original.copy()
    moment_terms = exp_moment_terms(prob_arr=repaired, x_vals=loss)
    residual = signed_unit_residual(values=moment_terms, lower_term=0.0, upper_term=0.0)
    if residual >= target_residual:
        return repaired, p_max

    _drain_moment_from_left(
        prob_arr=repaired,
        loss=loss,
        contributions=moment_terms,
        moment=target_residual - residual,
    )
    residual = signed_unit_residual(
        values=exp_moment_terms(prob_arr=repaired, x_vals=loss),
        lower_term=0.0,
        upper_term=0.0,
    )
    if residual < target_residual:
        raise ValueError(
            f"{context} did not reach residual {target_residual:.3e}: still at {residual:.3e}"
        )

    moved = math.fsum(map(float, original - repaired))
    if max_removed is not None and moved > max_removed:
        raise ValueError(f"{context} moved {moved:.3e} of mass, above the {max_removed:.3e} cap")
    return repaired, p_max + moved


def exp_moment_terms(
    *,
    prob_arr: NDArray[np.float64],
    x_vals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return per-bin contributions ``p * exp(-x)`` without avoidable overflow.

    When ``exp(-x)`` alone overflows but the product is representable, evaluate
    it as ``exp(log(p) - x)``. A combined term beyond float64 range remains
    ``inf`` so callers can reject the invalid moment rather than silently clip it.
    """
    prob_arr = np.asarray(prob_arr, dtype=np.float64)
    x_vals = np.asarray(x_vals, dtype=np.float64)
    if prob_arr.shape != x_vals.shape:
        raise ValueError("prob_arr and x_vals must have the same shape")

    terms = np.zeros_like(prob_arr, dtype=np.float64)
    positive_mask = prob_arr > 0.0
    safe_mask = positive_mask & (x_vals >= -MAX_SAFE_EXP_ARG)
    if np.any(safe_mask):
        terms[safe_mask] = prob_arr[safe_mask] * np.exp(-x_vals[safe_mask])

    extreme_mask = positive_mask & (x_vals < -MAX_SAFE_EXP_ARG)
    if np.any(extreme_mask):
        log_terms = np.log(prob_arr[extreme_mask]) - x_vals[extreme_mask]
        terms_extreme = np.exp(np.minimum(log_terms, MAX_SAFE_EXP_ARG))
        terms_extreme[log_terms > MAX_SAFE_EXP_ARG] = np.inf
        terms[extreme_mask] = terms_extreme

    return terms


# =============================================================================
# Compensated Accumulation
# =============================================================================


@optional_njit()
def compensated_segmented_sum(
    *,
    bin_index: NDArray[np.intp],
    weights: NDArray[np.float64],
    num_bins: int,
) -> NDArray[np.float64]:
    """Sum ``weights`` into bins with per-bin Kahan accumulation, in ``O(n + num_bins)``."""
    weights = np.asarray(weights, dtype=np.float64)
    bin_index = np.asarray(bin_index, dtype=np.intp)
    if bin_index.shape != weights.shape:
        raise ValueError("bin_index and weights must have the same shape")
    if num_bins < 0:
        raise ValueError(f"num_bins must be non-negative, got {num_bins}")
    totals = np.zeros(num_bins, dtype=np.float64)
    if weights.size == 0:
        return totals
    if bin_index.min() < 0 or bin_index.max() >= num_bins:
        raise ValueError(
            f"bin_index out of range for num_bins={num_bins}: "
            f"min={int(bin_index.min())}, max={int(bin_index.max())}"
        )

    compensations = np.zeros(num_bins, dtype=np.float64)
    for i in range(weights.size):
        bin_id = bin_index[i]
        y = weights[i] - compensations[bin_id]
        updated = totals[bin_id] + y
        compensations[bin_id] = (updated - totals[bin_id]) - y
        totals[bin_id] = updated
    return totals


@optional_njit()
def kahan_reverse_exclusive_cumsum(
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute ``out[i] = sum(values[i + 1:])`` with Kahan summation."""
    n = len(values)
    ccdf = np.zeros(n, dtype=np.float64)
    running_sum = 0.0
    compensation = 0.0
    for i in range(n - 1, -1, -1):
        ccdf[i] = running_sum
        y = values[i] - compensation
        updated = running_sum + y
        compensation = (updated - running_sum) - y
        running_sum = updated
    return ccdf


# =============================================================================
# Distribution Edge Truncation
# =============================================================================


def compute_truncation(
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
# Internal Helper Functions
# =============================================================================


def _drain_mass_from_edge(
    *,
    values: NDArray[np.float64],
    mass: float,
    from_left: bool,
    exact: bool,
) -> NDArray[np.float64]:
    """Remove mass inward from one edge of ``values``, in place.

    ``exact`` also part-debits the bin the target falls inside, so precisely ``mass``
    comes off; otherwise that bin is left whole.
    """
    if mass <= 0.0:
        return values
    total_mass = math.fsum(map(float, values))
    if mass >= total_mass:
        raise ValueError(
            "mass must be smaller than total array mass, "
            f"got mass={mass:.12g}, total={total_mass:.12g}"
        )

    # Mirror a right-edge removal so the scan always runs left to right.
    if not from_left:
        values = values[::-1]

    cumsum = np.cumsum(values, dtype=np.float64)
    pivot = int(np.searchsorted(cumsum, mass, side="left" if exact else "right"))
    pivot = min(pivot, values.size - 1)
    removed_before = float(cumsum[pivot - 1]) if pivot > 0 else 0.0
    if pivot > 0:
        values[:pivot] = 0.0
    if exact:
        values[pivot] = max(0.0, values[pivot] - (mass - removed_before))

    if not from_left:
        values = values[::-1]
    return values


def _drain_moment_from_left(
    *,
    prob_arr: NDArray[np.float64],
    loss: NDArray[np.float64],
    contributions: NDArray[np.float64],
    moment: float,
) -> None:
    """Remove ``moment`` from the low-loss edge of ``prob_arr``, in place.

    ``contributions`` is ``exp_moment_terms`` for the current ``prob_arr``, which the
    caller has already built to measure the residual. A request the law cannot supply
    raises rather than draining everything: surrendering the whole law would satisfy
    any target while leaving a vacuous result.
    """
    if moment >= math.fsum(map(float, contributions)):
        raise ValueError(f"Reciprocal-moment shortfall {moment:.3e} exceeds the law's total moment")

    cumulative = np.cumsum(contributions, dtype=np.float64)
    # cumsum is uncompensated, so it can land just short of a target fsum admits above.
    pivot = min(int(np.searchsorted(cumulative, moment, side="left")), prob_arr.size - 1)
    remainder = max(0.0, moment - math.fsum(map(float, contributions[:pivot])))
    prob_arr[:pivot] = 0.0
    if remainder > 0.0:
        loss_at_pivot = float(loss[pivot])
        # Mass per unit of moment. The direct product is ~10x more accurate here than
        # ``exp(log(remainder) + loss)``, which exponentiates the error of the sum.
        scale = math.exp(loss_at_pivot)
        if scale == 0.0 or not math.isfinite(scale):
            log_mass = math.log(remainder) + loss_at_pivot
            pivot_mass = math.exp(log_mass) if log_mass <= MAX_SAFE_EXP_ARG else math.inf
        else:
            # Overshoot by one ulp of whichever quantity the residual rebuild is
            # coarsest in. Removing extra moment only raises the residual, so this is
            # free insurance against that rebuild landing a fraction short.
            moment_ulp = max(
                float(np.spacing(max(float(contributions[pivot]), moment))),
                float(np.spacing(1.0)),
            )
            pivot_mass = remainder * scale + max(
                float(np.spacing(float(prob_arr[pivot]))),
                moment_ulp * scale,
            )
        prob_arr[pivot] = max(0.0, float(prob_arr[pivot]) - pivot_mass)


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
    original_mass = math.fsum(map(float, extended_prob))
    extended_prob = _drain_mass_from_edge(
        values=extended_prob, mass=tail_truncation, from_left=True, exact=False
    )
    shifted_mass = original_mass - math.fsum(map(float, extended_prob))
    extended_prob[np.nonzero(extended_prob)[0][0]] += shifted_mass
    p_min_out = extended_prob[0]
    extended_prob = _drain_mass_from_edge(
        values=extended_prob, mass=tail_truncation, from_left=False, exact=False
    )
    shifted_mass = original_mass - math.fsum(map(float, extended_prob))
    return extended_prob[1:], p_min_out, p_max + shifted_mass


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
    original_mass = math.fsum(map(float, extended_prob))
    extended_prob = _drain_mass_from_edge(
        values=extended_prob, mass=tail_truncation, from_left=False, exact=False
    )
    shifted_mass = original_mass - math.fsum(map(float, extended_prob))
    extended_prob[np.nonzero(extended_prob)[0][-1]] += shifted_mass
    p_max_out = extended_prob[-1]
    extended_prob = _drain_mass_from_edge(
        values=extended_prob, mass=tail_truncation, from_left=True, exact=False
    )
    shifted_mass = original_mass - math.fsum(map(float, extended_prob))
    return extended_prob[:-1], p_min + shifted_mass, p_max_out
