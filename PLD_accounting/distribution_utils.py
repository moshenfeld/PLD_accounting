"""Utility functions for distribution combination and regridding."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.types import BoundType
from PLD_accounting.validation import validate_discrete_pmf_and_boundaries

PMF_MASS_TOL = 10 * np.finfo(float).eps  # total-mass tolerance (10× machine epsilon)
SPACING_ATOL = 1e-12
SPACING_RTOL = 1e-6
MIN_GRID_SIZE = 100  # Minimum number of points in a  discretization grid.
MAX_SAFE_EXP_ARG = math.log(np.finfo(np.float64).max)

# =============================================================================
# Public Utility Functions
# =============================================================================


def enforce_mass_conservation(
    *,
    prob_arr: NDArray[np.float64],
    expected_p_min: float,
    expected_p_max: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], float, float]:
    """Enforce total mass with one bound-type-selected boundary held fixed.

    - ``DOMINATES`` enforces ``expected_p_max``.
    - ``IS_DOMINATED`` enforces ``expected_p_min``.

    Excess mass is removed directionally over an extended array that includes the
    opposite boundary, matching the truncation logic:
    - ``DOMINATES`` trims from the left over ``[p_min, *prob_arr]``.
    - ``IS_DOMINATED`` trims from the right over ``[*prob_arr, p_max]``.

    Any remaining slack is assigned to the enforced boundary.
    """
    prob_arr = np.asarray(prob_arr, dtype=np.float64).copy()
    validate_discrete_pmf_and_boundaries(
        prob_arr,
        expected_p_min,
        expected_p_max,
    )

    total_mass = math.fsum(map(float, prob_arr)) + expected_p_min + expected_p_max
    if total_mass <= 0.0:
        raise ValueError("Cannot enforce mass conservation with zero total mass")

    if bound_type == BoundType.DOMINATES:
        if expected_p_max > 1.0:
            raise ValueError("Expected p_max cannot exceed 1")
        extended = np.concatenate(([expected_p_min], prob_arr))
        target_mass = 1.0 - expected_p_max
        current_mass = math.fsum(map(float, extended))
        excess = current_mass - target_mass
        extended = _zero_mass(values=extended, mass=excess, from_left=True, exact=True)
        current_mass = math.fsum(map(float, extended))
        return (
            extended[1:].copy(),
            float(extended[0]),
            expected_p_max + max(0.0, target_mass - current_mass),
        )

    if bound_type == BoundType.IS_DOMINATED:
        if expected_p_min > 1.0:
            raise ValueError("Expected p_min cannot exceed 1")
        extended = np.concatenate((prob_arr, [expected_p_max]))
        target_mass = 1.0 - expected_p_min
        current_mass = math.fsum(map(float, extended))
        excess = current_mass - target_mass
        extended = _zero_mass(values=extended, mass=excess, from_left=False, exact=True)
        current_mass = math.fsum(map(float, extended))
        return (
            extended[:-1].copy(),
            expected_p_min + max(0.0, target_mass - current_mass),
            float(extended[-1]),
        )

    raise ValueError(
        f"Invalid bound_type: {bound_type}. Must be BoundType.DOMINATES or BoundType.IS_DOMINATED."
    )


def compute_bin_ratio_two_arrays(
    *, x_array_1: NDArray[np.float64], x_array_2: NDArray[np.float64]
) -> float:
    """Compute geometric spacing ratio for two grids and return their average."""
    r1 = compute_bin_ratio(x_array_1)
    r2 = compute_bin_ratio(x_array_2)
    if not stable_isclose(a=r1, b=r2):
        raise ValueError(f"Grid ratios must match: ratio_1={r1:.12g}, ratio_2={r2:.12g}")
    return (r1 + r2) / 2


def compute_bin_width_two_arrays(
    *, x_array_1: NDArray[np.float64], x_array_2: NDArray[np.float64]
) -> float:
    """Compute linear spacing width for two grids and return their average."""
    w1 = compute_bin_width(x_array_1)
    w2 = compute_bin_width(x_array_2)
    if not stable_isclose(a=w1, b=w2):
        raise ValueError(f"Grid spacing must match: w1={w1:.12g} vs w2={w2:.12g}")
    return (w1 + w2) / 2


# =============================================================================
# Grid Spacing Utilities
# =============================================================================


def compute_bin_ratio(x_array: NDArray[np.float64]) -> float:
    """Compute geometric spacing ratio for a grid."""
    if x_array.size < 2:
        raise ValueError("Cannot compute geometric bin ratio with less than 2 bins")
    if np.any(x_array <= 0):
        raise ValueError("Cannot compute geometric bin ratio for non-positive values")
    log_ratios = np.log(x_array[1:] / x_array[:-1])
    med_log_ratio = np.median(log_ratios)
    if not np.allclose(med_log_ratio, log_ratios, rtol=SPACING_RTOL, atol=SPACING_ATOL):
        max_diff = np.max(np.abs(med_log_ratio - log_ratios))
        raise ValueError(
            "Distribution has non-uniform bin widths: "
            f"median_ratio={np.median(log_ratios)}, max_diff={max_diff}"
        )
    return np.exp(med_log_ratio)


def compute_bin_width(x_array: NDArray[np.float64]) -> float:
    """Compute linear spacing width for a grid."""
    if x_array.size < 2:
        raise ValueError("Cannot compute width with less than 2 bins")
    diffs = np.diff(x_array)
    median_diff = np.median(diffs)
    if not np.allclose(median_diff, diffs, rtol=SPACING_RTOL, atol=SPACING_ATOL):
        max_diff = np.max(np.abs(median_diff - diffs))
        raise ValueError(
            "Distribution has non-uniform bin widths: "
            f"median_diff={median_diff}, max diff={max_diff}"
        )
    return float(median_diff)


def stable_isclose(*, a: float, b: float) -> bool:
    """Consistent closeness check using shared spacing tolerances."""
    return bool(np.isclose(a, b, rtol=SPACING_RTOL, atol=SPACING_ATOL))


def stable_array_equal(*, a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """Consistent array closeness check using shared spacing tolerances."""
    return a.shape == b.shape and np.allclose(a, b, rtol=SPACING_RTOL, atol=SPACING_ATOL)


def exp_moment_terms(
    *,
    prob_arr: NDArray[np.float64],
    x_vals: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return per-bin contributions to ``E[exp(-X)]``.

    For very negative ``x_vals`` the naive product ``p * exp(-x)`` can overflow
    even when the combined term is representable. In that regime we evaluate the
    contribution as ``exp(log(p) - x)`` instead.

    Terms that still exceed float64 range are returned as ``inf``.
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
# Distribution Edge Truncation
# =============================================================================


def compute_truncation(
    prob_arr: NDArray[np.float64],
    p_min: float,
    p_max: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> tuple[NDArray[np.float64], float, float, int, int]:
    """Compute truncated distribution parameters without creating objects.

    Algorithm:
      A. Remove leading/trailing zeros from PMF (always done).
      B. If tail_truncation > 0:
         - Compute how much to consume from each side (up to tail_truncation).
         - For DOMINATES:    Operate over the [p_min, *prob_arr] range.
                             Left tail folds into first remaining element;
                             right tail goes to p_max.
         - For IS_DOMINATED: Operate over the [*prob_arr, p_max] range.
                             Right tail folds into last remaining element;
                             left tail goes to p_min;
      C. Apply step A again to remove any newly created leading/trailing zeros.

    Returns:
        (new_PMF, new_p_min, new_p_max, min_ind, max_ind) where min_ind and
        max_ind are indices into the original prob_arr.
    """
    # Remove zero probability tails to reduce unnecessary computations
    inner_min, inner_max = _strip_zero_edges(prob_arr)
    trimmed_prob_arr = prob_arr[slice(inner_min, inner_max + 1)].copy()

    if tail_truncation == 0.0:
        return trimmed_prob_arr.copy(), p_min, p_max, inner_min, inner_max

    if bound_type == BoundType.DOMINATES:
        extended_prob = np.concatenate([[p_min], trimmed_prob_arr])
        original_mass = math.fsum(map(float, extended_prob))
        # Truncate left tail and add its mass to the next finite bin
        extended_prob = _zero_mass(
            values=extended_prob, mass=tail_truncation, from_left=True, exact=False
        )
        shifted_mass = original_mass - math.fsum(map(float, extended_prob))
        extended_prob[np.nonzero(extended_prob)[0][0]] += shifted_mass
        p_min_out = extended_prob[0]
        # Truncate right tail and add its mass to to p_max
        extended_prob = _zero_mass(
            values=extended_prob, mass=tail_truncation, from_left=False, exact=False
        )
        shifted_mass = original_mass - math.fsum(map(float, extended_prob))
        p_max_out = p_max + shifted_mass
        prob_arr_out = extended_prob[1:]
    elif bound_type == BoundType.IS_DOMINATED:
        extended_prob = np.concatenate((trimmed_prob_arr, [p_max]))
        original_mass = math.fsum(map(float, extended_prob))
        # Truncate right tail and add its mass to the next finite bin
        extended_prob = _zero_mass(
            values=extended_prob, mass=tail_truncation, from_left=False, exact=False
        )
        shifted_mass = original_mass - math.fsum(map(float, extended_prob))
        extended_prob[np.nonzero(extended_prob)[0][-1]] += shifted_mass
        p_max_out = extended_prob[-1]
        # Truncate left tail and add its mass to to p_min
        extended_prob = _zero_mass(
            values=extended_prob, mass=tail_truncation, from_left=True, exact=False
        )
        shifted_mass = original_mass - math.fsum(map(float, extended_prob))
        p_min_out = p_min + shifted_mass
        prob_arr_out = extended_prob[:-1]
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    # Remove zero probability tails to reduce unnecessary computations
    inner_min_new, inner_max_new = _strip_zero_edges(prob_arr_out)
    min_ind_new = inner_min + inner_min_new
    max_ind_new = inner_min + inner_max_new
    return (
        prob_arr_out[slice(inner_min_new, inner_max_new + 1)].copy(),
        p_min_out,
        p_max_out,
        min_ind_new,
        max_ind_new,
    )


def _strip_zero_edges(prob_arr: NDArray[np.float64]) -> tuple[int, int]:
    """Return (min_ind, max_ind) of the nonzero range in prob_arr.

    Raises ValueError if all mass is zero.
    """
    nonzero_indices = np.nonzero(prob_arr)[0]
    if nonzero_indices.size == 0:
        raise ValueError("Cannot truncate distribution with zero finite mass")
    return int(nonzero_indices[0]), int(nonzero_indices[-1])


def _zero_mass(
    *,
    values: NDArray[np.float64],
    mass: float,
    from_left: bool,
    exact: bool,
) -> NDArray[np.float64]:
    """Remove mass probability from values from one of the side, based on ``from_left``.

    If ``exact`` is true, partially consume the pivot bin so that exactly
    ``mass`` is removed. Otherwise, consume only complete bins whose cumulative
    mass does not exceed ``mass`` and leave the pivot bin unchanged.
    """
    if mass <= 0.0:
        return values
    total_mass = math.fsum(map(float, values))
    if mass >= total_mass:
        raise ValueError(
            "mass must be smaller than total array mass, "
            f"got mass={mass:.12g}, total={total_mass:.12g}"
        )

    # When removing from the right, we just flip the array before and after the caculation
    if not from_left:
        values = values[::-1]

    # Find the pivot index
    cumsum = np.cumsum(values, dtype=np.float64)
    if exact:
        pivot = int(np.searchsorted(cumsum, mass, side="left"))
    else:
        pivot = int(np.searchsorted(cumsum, mass, side="right"))

    # Remove the probability mass below the pivot
    removed_before = float(cumsum[pivot - 1]) if pivot > 0 else 0.0
    if pivot > 0:
        values[:pivot] = 0.0
    # Remove the additional probability mass from the pivot if needed
    if exact:
        values[pivot] = max(0.0, values[pivot] - (mass - removed_before))

    if not from_left:
        values = values[::-1]
    return values
