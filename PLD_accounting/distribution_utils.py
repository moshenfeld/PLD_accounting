import math

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.types import BoundType

PMF_MASS_TOL = 10 * np.finfo(float).eps  # total-mass tolerance (10× machine epsilon)
SPACING_ATOL = 1e-12
SPACING_RTOL = 1e-6

# =============================================================================
# Public Utility Functions
# =============================================================================

def enforce_mass_conservation(*,
    PMF_array: NDArray[np.float64],
    expected_neg_inf: float,
    expected_pos_inf: float,
    bound_type: BoundType
) -> tuple[NDArray[np.float64], float, float]:
    """Enforce mass conservation while honoring expected infinity mass.

    Uses expected_inf as a lower bound on infinity mass, then renormalizes
    finite mass so total mass sums to 1.0.

    Sets either p_pos_inf or p_neg_inf to ensure total mass equals 1.0:
    - BoundType.DOMINATES: Sets p_pos_inf = 1 - pmf_sum, p_neg_inf = 0
    - BoundType.IS_DOMINATED: Sets p_neg_inf = 1 - pmf_sum, p_pos_inf = 0
    """
    pmf_sum = math.fsum(map(float, PMF_array))
    if pmf_sum <= 0.0:
        raise ValueError("Cannot enforce mass conservation with zero finite mass")
    expected_inf = float(expected_pos_inf if bound_type == BoundType.DOMINATES else expected_neg_inf)
    finite_target = 1.0 - expected_inf
    if finite_target < 0.0:
        raise ValueError("Expected infinity mass cannot exceed 1")

    # If finite mass is overfull, trim from the boundary that preserves semantics:
    # DOMINATES -> remove leftmost finite mass, IS_DOMINATED -> remove rightmost finite mass.
    if pmf_sum > finite_target:
        excess = min(pmf_sum - finite_target, pmf_sum)
        if bound_type == BoundType.DOMINATES:
            cumsum = np.cumsum(PMF_array, dtype=np.float64)
            pivot = int(np.searchsorted(cumsum, excess, side="left"))
            removed_before = float(cumsum[pivot - 1]) if pivot > 0 else 0.0
            if pivot > 0:
                PMF_array[:pivot] = 0.0
            PMF_array[pivot] = max(0.0, PMF_array[pivot] - (excess - removed_before))
        else:
            cumsum_rev = np.cumsum(PMF_array[::-1], dtype=np.float64)
            rev_pivot = int(np.searchsorted(cumsum_rev, excess, side="left"))
            removed_before = float(cumsum_rev[rev_pivot - 1]) if rev_pivot > 0 else 0.0
            pivot = PMF_array.size - 1 - rev_pivot
            if rev_pivot > 0:
                PMF_array[pivot + 1:] = 0.0
            PMF_array[pivot] = max(0.0, PMF_array[pivot] - (excess - removed_before))
        pmf_sum = math.fsum(map(float, PMF_array))
    remaining_mass = max(0.0, 1.0 - pmf_sum)
    output_inf = max(expected_inf, remaining_mass)
    if output_inf > remaining_mass:
        PMF_array = PMF_array * ((1.0 - expected_inf) / pmf_sum)
    if bound_type == BoundType.DOMINATES:
        return PMF_array, 0.0, output_inf
    if bound_type == BoundType.IS_DOMINATED:
        return PMF_array, output_inf, 0.0
    raise ValueError(
        "Invalid bound_type: "
        f"{bound_type}. Must be BoundType.DOMINATES or BoundType.IS_DOMINATED."
    )

def compute_bin_ratio_two_arrays(*, x_array_1: NDArray[np.float64], x_array_2: NDArray[np.float64]) -> float:
    """
    Compute geometric spacing ratio for two grids and return their average.
    """
    r1 = compute_bin_ratio(x_array_1)
    r2 = compute_bin_ratio(x_array_2)
    if not stable_isclose(a=r1, b=r2):
        raise ValueError(f"Grid ratios must match: ratio_1={r1:.12g}, ratio_2={r2:.12g}")
    return (r1 + r2) / 2

def compute_bin_width_two_arrays(*, x_array_1: NDArray[np.float64], x_array_2: NDArray[np.float64]) -> float:
    """
    Compute linear spacing width for two grids and return their average.
    """
    w1 = compute_bin_width(x_array_1)
    w2 = compute_bin_width(x_array_2)
    if not stable_isclose(a=w1, b=w2):
        raise ValueError(f"Grid spacing must match: w1={w1:.12g} vs w2={w2:.12g}")
    return (w1 + w2) / 2

# =============================================================================
# Grid Spacing Utilities
# =============================================================================

def compute_bin_ratio(x_array: NDArray[np.float64]) -> float:
    """
    Compute geometric spacing ratio for a grid.
    """
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
    """
    Compute linear spacing width for a grid.
    """
    if x_array.size < 2:
        raise ValueError("Cannot compute width with less than 2 bins")
    diffs = np.diff(x_array)
    median_diff = np.median(diffs)
    if not np.allclose(median_diff, diffs, rtol=SPACING_RTOL, atol=SPACING_ATOL):
        max_diff = np.max(np.abs(median_diff - diffs))
        raise ValueError(
            f"Distribution has non-uniform bin widths: median_diff={median_diff}, max diff={max_diff}"
        )
    return median_diff

def stable_isclose(*, a: float, b: float) -> bool:
    """
    Consistent closeness check using shared spacing tolerances.
    """
    return bool(np.isclose(a, b, rtol=SPACING_RTOL, atol=SPACING_ATOL))

def stable_array_equal(*, a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """
    Consistent array closeness check using shared spacing tolerances.
    """
    return a.shape == b.shape and np.allclose(a, b, rtol=SPACING_RTOL, atol=SPACING_ATOL)
