"""Distribution discretization utilities for PMF construction."""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from PLD_accounting.discrete_dist import DenseDiscreteDist, DiscreteDistBase, Domain
from PLD_accounting.distribution_utils import enforce_mass_conservation
from PLD_accounting.types import BoundType, SpacingType

# =============================================================================
# Public API: Continuous Distribution Discretization
# =============================================================================


def discretize_continuous_distribution(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    bound_type: BoundType,
    spacing_type: SpacingType,
    step: float,
    align_to_multiples: bool,
    domain: Domain = Domain.REALS,
) -> DenseDiscreteDist:
    """Discretize a continuous distribution to a typed structured representation.

    Args:
        dist: Continuous distribution to discretize.
        tail_truncation: Tail mass budget used to define quantile bounds and bin increment floor.
        bound_type: Tie-breaking direction for interval mass assignment.
        spacing_type: Output grid spacing family (linear or geometric).
        step: Linear bin width or geometric multiplicative ratio.
        align_to_multiples: Whether to align the quantile-derived bounds to integer step multiples.
        domain: Domain semantics for boundary masses in the output discrete distribution.

    Returns:
        Discretized distribution on a structured dense grid.
    """

    x_array = _discretize_continuous_to_grid(
        dist=dist,
        tail_truncation=tail_truncation,
        spacing_type=spacing_type,
        step=step,
        align_to_multiples=align_to_multiples,
    )
    if x_array[0] <= 0 and domain == Domain.POSITIVES:
        dist_name = getattr(dist, "name", type(dist).__name__)
        raise ValueError(
            f"Cannot discretize {dist_name} to a positive range, got x_min={x_array[0]}"
        )

    # 2. Map density to PMF with semantics.
    return discretize_continuous_dist(
        dist=dist,
        x_array=x_array,
        bound_type=bound_type,
        PMF_min_increment=tail_truncation,
        spacing_type=spacing_type,
        domain=domain,
    )


def discretize_continuous_dist(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    x_array: NDArray[np.float64],
    bound_type: BoundType,
    PMF_min_increment: float,
    spacing_type: SpacingType,
    domain: Domain = Domain.REALS,
) -> DenseDiscreteDist:
    """Convert continuous distribution to discrete PMF with bounding semantics."""
    # Compute raw probabilities for intervals [x_i, x_{i+1}) using PMF_min_increment.
    bin_probs, p_left, p_right = _compute_discrete_prob(
        dist=dist, x_array=x_array, bound_type=bound_type, PMF_min_increment=PMF_min_increment
    )

    n = x_array.size
    prob_arr = np.zeros(n)

    if bound_type == BoundType.DOMINATES:
        # Shift mass right: left tail (-inf, x0) -> x0,
        # each interval [x_i, x_{i+1}) -> x_{i+1}, right tail (x_n, inf) -> inf,
        prob_arr[0] = p_left
        prob_arr[1:] = bin_probs
        p_min = 0.0
        p_max = p_right

    elif bound_type == BoundType.IS_DOMINATED:
        # Shift mass left: left tail (-inf, x0) -> -inf,
        # each interval [x_i, x_{i+1}) -> x_i, right tail (x_n, inf) -> x_n,
        prob_arr[:-1] = bin_probs
        prob_arr[-1] = p_right
        p_min = p_left
        p_max = 0.0
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    if spacing_type == SpacingType.LINEAR:
        return DenseDiscreteDist.from_x_array(
            x_array=x_array,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            domain=domain,
        )

    if spacing_type == SpacingType.GEOMETRIC:
        return DenseDiscreteDist.from_x_array(
            x_array=x_array,
            prob_arr=prob_arr,
            p_min=p_min,
            p_max=p_max,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

    raise ValueError(f"Invalid spacing_type: {spacing_type}")


def rediscretize_dist(
    *,
    dist: DiscreteDistBase,
    tail_truncation: float,
    loss_discretization: float,
    spacing_type: SpacingType,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Rediscretize a distribution onto a requested grid spacing.

    Remaps PMF onto a new grid with the requested spacing and discretization.
    Implementation trims zero/tail regions, computes new grid size, then remaps
    using domination-aware rounding (e.g., linear grids for dp_accounting output).

    Algorithm 6 (`disc-dist`) in Appendix C.
    """

    # Support for rediscritizing a dominating distribution into a dominated one and vice versa
    working_dist = dist.copy()
    if bound_type == BoundType.IS_DOMINATED and working_dist.p_max > 0.0:
        working_dist.prob_arr[-1] += working_dist.p_max
        working_dist.p_max = 0.0
    elif (
        bound_type == BoundType.DOMINATES
        and spacing_type == SpacingType.LINEAR
        and working_dist.p_min > 0.0
    ):
        working_dist.prob_arr[0] += working_dist.p_min
        working_dist.p_min = 0.0

    # Quantile-truncation
    trunc_dist = working_dist.truncate_edges(
        tail_truncation=tail_truncation / 2, bound_type=bound_type
    )

    x_array = trunc_dist.x_array
    x_min = x_array[0]
    x_max = x_array[-1]

    x_array_out = discretize_aligned_range(
        x_min=x_min,
        x_max=x_max,
        spacing_type=spacing_type,
        align_to_multiples=True,
        discretization=loss_discretization,
    )

    prob_arr_out = rediscretize_prob(
        x_array=x_array,
        prob_arr=trunc_dist.prob_arr,
        x_array_out=x_array_out,
        dominates=(bound_type == BoundType.DOMINATES),
    )

    prob_arr_out, p_min, p_max = enforce_mass_conservation(
        prob_arr=prob_arr_out,
        expected_p_min=working_dist.p_min,
        expected_p_max=working_dist.p_max,
        bound_type=bound_type,
    )

    if spacing_type == SpacingType.LINEAR:
        return DenseDiscreteDist.from_x_array(
            x_array=x_array_out,
            prob_arr=prob_arr_out,
            p_min=p_min,
            p_max=p_max,
        )

    if spacing_type == SpacingType.GEOMETRIC:
        return DenseDiscreteDist.from_x_array(
            x_array=x_array_out,
            prob_arr=prob_arr_out,
            p_min=p_min,
            p_max=p_max,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

    raise ValueError(f"Invalid spacing_type: {spacing_type}")


def discretize_aligned_range(
    *,
    x_min: float,
    x_max: float,
    spacing_type: SpacingType,
    align_to_multiples: bool,
    discretization: float,
) -> NDArray[np.float64]:
    """Return a grid covering [x_min, x_max].

    Args:
        x_min: Minimum value of the range.
        x_max: Maximum value of the range.
        spacing_type: Type of spacing (LINEAR or GEOMETRIC).
        align_to_multiples: If True, align range to whole multiples of discretization.
                           If False, use x_min and x_max directly without alignment.
        discretization: Grid spacing parameter (step size for LINEAR, log ratio for GEOMETRIC).

    """
    # Validate inputs
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
        if align_to_multiples:
            k_lo = int(np.floor(x_min / d))
            k_hi = int(np.ceil(x_max / d))
            # It is possible that `ceil(x/d)*d < x` in float64 due to floating numerics
            if d * k_lo > x_min:
                k_lo -= 1
            if d * k_hi < x_max:
                k_hi += 1
            return d * np.arange(k_lo, k_hi + 1, dtype=np.float64)
        span = x_max - x_min
        n = int(np.ceil(span / d)) + 1
        x = x_min + d * np.arange(n, dtype=np.float64)
        if x[-1] < x_max:
            n += 1
            x = x_min + d * np.arange(n, dtype=np.float64)
        return x

    # GEOMETRIC: discretization is log-ratio per step; grid x = exp(d * k).
    if align_to_multiples:
        k_lo = int(np.floor(np.log(x_min) / d))
        k_hi = int(np.ceil(np.log(x_max) / d))
        # It is possible that `ceil(x/d)*d < x` in float64 due to floating numerics
        if np.exp(d * k_lo) > x_min:
            k_lo -= 1
        if np.exp(d * k_hi) < x_max:
            k_hi += 1
        return np.exp(d * np.arange(k_lo, k_hi + 1, dtype=np.float64))
    n = int(np.ceil(np.log(x_max / x_min) / d)) + 1
    x = x_min * np.exp(d * np.arange(n, dtype=np.float64))
    if x[-1] < x_max:
        n += 1
        x = x_min * np.exp(d * np.arange(n, dtype=np.float64))
    return x


@njit(cache=True)
def rediscretize_prob(
    x_array: NDArray[np.float64],
    prob_arr: NDArray[np.float64],
    x_array_out: NDArray[np.float64],
    dominates: bool,
) -> NDArray[np.float64]:
    """Remap PMF onto new grid with domination-aware rounding.

    Maps each probability mass to output grid position based on domination semantics.
    Implementation: dominates=True uses ceil (pessimistic), False uses floor (optimistic).
    Uses Kahan summation for numerical accuracy.
    Returns: remapped PMF array.

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
                # overflow to the right: discard mass (goes to p_max via enforce_mass_conservation)
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
                # underflow to the left: discard mass (goes to p_min via enforce_mass_conservation)
                continue
            # include values above x_array_out[-1] in the last bin (floor behavior)
            y = mass - compensations[idx]
            t = prob_arr_out[idx] + y
            compensations[idx] = (t - prob_arr_out[idx]) - y
            prob_arr_out[idx] = t

    return prob_arr_out


# =============================================================================
# Helper Functions
# =============================================================================


@njit(cache=True)
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


@njit(cache=True)
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
    PMF_min_increment: float,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute bin probabilities using adaptive CDF/SF increments with logcdf/logsf stability.

    PMF_min_increment controls the minimum CDF/SF increment that becomes a bin mass.

    """
    cdf, sf = _stable_cdf_and_sf(
        dist=dist,
        x_array=x_array,
    )
    p_left = cdf[0]
    p_right = sf[-1]
    PMF_min_increment = max(0.0, PMF_min_increment)

    if bound_type == BoundType.DOMINATES:
        # Suppress intermediate debug logging.
        bin_probs = _adaptive_bins_from_cdf(
            cdf=cdf,
            tail_truncation=PMF_min_increment,
        )
    elif bound_type == BoundType.IS_DOMINATED:
        # Suppress intermediate debug logging.
        bin_probs = _adaptive_bins_from_sf(
            sf=sf,
            tail_truncation=PMF_min_increment,
        )
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    return bin_probs, p_left, p_right


def _discretize_continuous_to_grid(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    spacing_type: SpacingType,
    step: float,
    align_to_multiples: bool,
) -> NDArray[np.float64]:
    """Generate grid covering the quantile range defined by tail_truncation."""
    # Determine support bounds via quantiles
    x_min = float(dist.ppf(tail_truncation))
    x_max = float(dist.isf(tail_truncation))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError(f"Quantiles not finite: x_min={x_min}, x_max={x_max}")

    if spacing_type == SpacingType.GEOMETRIC:
        if step <= 1.0:
            raise ValueError(f"Geometric step must be > 1, got {step}")
        discretization = float(np.log(step))
    else:
        if step <= 0.0:
            raise ValueError(f"Linear step must be positive, got {step}")
        discretization = float(step)

    x_array = discretize_aligned_range(
        x_min=x_min,
        x_max=x_max,
        spacing_type=spacing_type,
        align_to_multiples=align_to_multiples,
        discretization=discretization,
    )
    return x_array
