"""Mass, moment and summation primitives shared by every distribution operation.

Mass conservation and residual policy, reciprocal-moment repair, compensated
accumulation, and the ranges and interval masses of a continuous law. All work
on bare arrays, so they can be used before a distribution object exists.

Repairs take their admission band from the caller rather than choosing one: the
operation that owns the budget decides what counts as tolerable drift.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from PLD_accounting.types import BoundType, has_numba, optional_njit
from PLD_accounting.validation import (
    require_closed_unit_interval,
    require_finite_array,
    require_nonnegative_masses,
    require_open_unit_interval,
    require_positive_int,
)

# Drift is repaired silently; larger repairs are directional and warn.
# Units: probability mass.
# These bands assume a compensated producer (``math.fsum`` /
# ``compensated_segmented_sum``). An uncompensated producer such as FFT must pass an
# explicit size-scaled pair from ``transform_mass_tolerances``; inheriting this default
# without scaling raises inside the repair helper.
PMF_MASS_DRIFT_TOL: float = float(4 * np.finfo(float).eps)
PMF_TOLERATED_MASS_TOL: float = float(10 * np.finfo(float).eps)
MAX_SAFE_EXP_ARG = math.log(np.finfo(np.float64).max)
# Slack allowed on the exact interval, cell and exterior inequalities before an
# inconsistency is treated as an oracle failure rather than as rounding of the inputs.
ORACLE_TOL = 64.0 * float(np.finfo(np.float64).eps)
# Measured corrections the reciprocal-moment trim may spend before it spends the bin.
_TRIM_CORRECTIONS = 4
# Policy multipliers on the size-scaled mass residual of an uncompensated transform.
_TRANSFORM_MASS_DRIFT_FACTOR = 16.0
_TRANSFORM_MASS_REPAIR_FACTOR = 64.0
# Initial capacity of the exact sum's partials; the buffer doubles when a sum needs more.
_EXACT_SUM_PARTIALS = 32


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
    require_closed_unit_interval(
        value=[expected_p_min, expected_p_max],
        name=["expected_p_min", "expected_p_max"],
    )
    require_nonnegative_masses(prob_arr=prob_arr, p_min=expected_p_min, p_max=expected_p_max)
    # One residual, repaired in whichever direction it points: negative is surplus mass,
    # positive a shortfall. Both layouts below hold the same terms, so it is theirs too.
    residual = signed_unit_residual(
        values=prob_arr, lower_term=expected_p_min, upper_term=expected_p_max
    )
    if 1.0 - residual <= 0.0:
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
        extended = drain_mass_from_edge(
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
    return drain_mass_from_edge(
        values=np.asarray(prob_arr, dtype=np.float64).copy(),
        mass=mass,
        from_left=from_left,
        exact=True,
    )


def drain_mass_from_edge(
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
    total_mass = exact_sum(values)
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


def signed_unit_residual(
    *,
    values: NDArray[np.float64],
    lower_term: float,
    upper_term: float,
) -> float:
    """Return ``1 - (lower_term + sum(values) + upper_term)``, correctly rounded."""
    terms = np.concatenate(
        (
            (1.0, -float(lower_term), -float(upper_term)),
            -np.asarray(values, dtype=np.float64).ravel(),
        )
    )
    return exact_sum(terms)


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


def transform_mass_tolerances(*, num_bins: int, num_convolutions: int) -> tuple[float, float]:
    """Return ``(drift_tol, repair_tol)`` for the masses an FFT of ``num_bins`` points produces.

    ``num_convolutions`` is how many times the transform's residual enters the result.
    """
    require_positive_int(value=[num_bins, num_convolutions], name=["num_bins", "num_convolutions"])
    # Transform error grows logarithmically with size and linearly with the power.
    scale = float(num_convolutions) * math.log2(max(num_bins, 2)) * float(np.finfo(float).eps)
    return _TRANSFORM_MASS_DRIFT_FACTOR * scale, _TRANSFORM_MASS_REPAIR_FACTOR * scale


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
    require_closed_unit_interval(value=target_residual, name="target_residual")

    repaired = original.copy()
    moment_terms = exp_moment_terms(prob_arr=repaired, x_vals=loss)
    residual = signed_unit_residual(values=moment_terms, lower_term=0.0, upper_term=0.0)
    if residual >= target_residual:
        return repaired, p_max

    _drain_moment_from_left(
        prob_arr=repaired,
        loss=loss,
        contributions=moment_terms,
        target_residual=target_residual,
    )
    moment_terms = exp_moment_terms(prob_arr=repaired, x_vals=loss)
    residual = signed_unit_residual(values=moment_terms, lower_term=0.0, upper_term=0.0)
    if residual < target_residual:
        raise ValueError(
            f"{context} did not reach residual {target_residual:.3e}: still at {residual:.3e}"
        )

    moved = exact_sum(original - repaired)
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


def exact_sum(values: NDArray[np.float64]) -> float:
    """Return the correctly rounded sum of ``values``, the value ``math.fsum`` returns.

    Numba runs Shewchuk's partials, the algorithm behind ``math.fsum``; without Numba
    ``math.fsum`` itself runs. A value that is not finite raises ``ValueError``, and
    a partial sum that overflows raises ``OverflowError``.
    """
    values = np.ascontiguousarray(values, dtype=np.float64).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError("exact_sum requires finite values")
    if has_numba():
        return float(_numba_exact_sum(values))
    return _numpy_exact_sum(values)


# =============================================================================
# Ranges and Interval Masses of a Continuous Law
# =============================================================================


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


def partition_masses(
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


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _drain_moment_from_left(
    *,
    prob_arr: NDArray[np.float64],
    loss: NDArray[np.float64],
    contributions: NDArray[np.float64],
    target_residual: float,
) -> None:
    """Trim the low-loss edge of ``prob_arr``, in place, until its residual reaches the target.

    ``contributions`` is ``exp_moment_terms`` for the current ``prob_arr``, which the
    caller has already built to measure the residual. Whole bins are zeroed up to the
    pivot bin, which keeps the largest mass found to meet the target. A target the law
    cannot supply raises rather than draining everything: surrendering the whole law
    would satisfy any target while leaving a vacuous result.
    """
    moment = target_residual - signed_unit_residual(
        values=contributions, lower_term=0.0, upper_term=0.0
    )
    if moment >= exact_sum(contributions):
        raise ValueError(f"Reciprocal-moment shortfall {moment:.3e} exceeds the law's total moment")

    cumulative = np.cumsum(contributions, dtype=np.float64)
    # cumsum is uncompensated, so it can land just short of a target the exact sum admits above.
    pivot = min(int(np.searchsorted(cumulative, moment, side="left")), prob_arr.size - 1)
    remainder = max(0.0, moment - exact_sum(contributions[:pivot]))
    prob_arr[:pivot] = 0.0

    loss_at_pivot = float(loss[pivot])
    # Mass per unit of moment. The direct product is ~10x more accurate here than
    # ``exp(log(remainder) + loss)``, which exponentiates the error of the sum.
    scale = math.exp(loss_at_pivot)
    retained = float(prob_arr[pivot])
    if remainder > 0.0:
        if scale == 0.0 or not math.isfinite(scale):
            log_mass = math.log(remainder) + loss_at_pivot
            pivot_mass = math.exp(log_mass) if log_mass <= MAX_SAFE_EXP_ARG else math.inf
        else:
            pivot_mass = remainder * scale
        retained = max(0.0, retained - pivot_mass)

    # The residual the caller will check is measured with the bin at this mass, over the bins
    # from the pivot on: the emptied bins contribute exactly zero. A shortfall is converted to
    # mass at this bin and subtracted; one under half an ulp of the retained mass would round
    # back to it, so each correction steps at least one ulp down. A target still unmet after
    # ``_TRIM_CORRECTIONS`` spends the bin.
    for _ in range(_TRIM_CORRECTIONS):
        prob_arr[pivot] = retained
        moment_terms = exp_moment_terms(prob_arr=prob_arr[pivot:], x_vals=loss[pivot:])
        measured = signed_unit_residual(values=moment_terms, lower_term=0.0, upper_term=0.0)
        if measured >= target_residual:
            return
        retained = min(
            retained - (target_residual - measured) * scale,
            float(np.nextafter(retained, 0.0)),
        )
        retained = max(0.0, retained)
    warnings.warn(
        f"Reciprocal-moment trim spent a whole bin of mass {retained:.3e} at loss "
        f"{loss_at_pivot:.3e}: {_TRIM_CORRECTIONS} corrections left the residual at "
        f"{measured:.3e}, below the target {target_residual:.3e}. The mass moves to +inf; "
        "the bound stays valid and is looser.",
        RuntimeWarning,
        stacklevel=2,
    )
    prob_arr[pivot] = 0.0


@optional_njit()
def _numba_exact_sum(values: NDArray[np.float64]) -> float:
    """``math.fsum``'s partials and final rounding, for finite ``values``."""
    partials = np.empty(_EXACT_SUM_PARTIALS, dtype=np.float64)
    count = 0
    for index in range(values.size):
        x = values[index]
        kept = 0
        for position in range(count):
            y = partials[position]
            if abs(x) < abs(y):
                x, y = y, x
            high = x + y
            low = y - (high - x)
            if low != 0.0:
                partials[kept] = low
                kept += 1
            x = high
        count = kept
        if x != 0.0:
            if not math.isfinite(x):
                raise OverflowError("intermediate overflow in exact_sum")
            if count == partials.size:
                grown = np.empty(2 * partials.size, dtype=np.float64)
                grown[:count] = partials
                partials = grown
            partials[count] = x
            count += 1
    high = 0.0
    if count > 0:
        count -= 1
        high = partials[count]
        low = 0.0
        while count > 0:
            x = high
            count -= 1
            y = partials[count]
            high = x + y
            low = y - (high - x)
            if low != 0.0:
                break
        # Round half to even across partials, as math.fsum does.
        if count > 0 and (
            (low < 0.0 and partials[count - 1] < 0.0) or (low > 0.0 and partials[count - 1] > 0.0)
        ):
            y = low * 2.0
            x = high + y
            if y == x - high:
                high = x
    return high


def _numpy_exact_sum(values: NDArray[np.float64]) -> float:
    """Return ``math.fsum`` of ``values``: the NumPy twin of ``_numba_exact_sum``."""
    return math.fsum(values.tolist())


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
    if np.any(mass < -ORACLE_TOL):
        raise ValueError(
            f"{label} interval oracle returned a negative probability: "
            f"most negative cell {float(mass.min()):.3e}"
        )
    return np.maximum(mass, 0.0)
