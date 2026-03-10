import numpy as np
from numba import njit
from numpy.typing import NDArray
from scipy import stats

from PLD_accounting.core_utils import enforce_mass_conservation
from PLD_accounting.discrete_dist import *
from PLD_accounting.utils import (
    exp_linear_to_geometric,
    log_geometric_to_linear,
)
from PLD_accounting.types import BoundType, SpacingType

# =============================================================================
# Continuous distribution discretization
# =============================================================================
TAIL_SWITCH = 1e-10
MIN_GRID_SIZE = 100
def discretize_continuous_distribution(
    dist: stats.rv_continuous,
    tail_truncation: float,
    bound_type: BoundType,
    spacing_type: SpacingType,
    n_grid: int,
    align_to_multiples: bool,
) -> LinearDiscreteDist | GeometricDiscreteDist:
    """
    Discretize a continuous distribution to a typed structured representation.
    """
    # 1. Generate grid
    if n_grid == 0:
        raise ValueError("n_grid must be positive")
    x_array = _discretize_continuous_to_grid(
        dist=dist,
        tail_truncation=tail_truncation,
        spacing_type=spacing_type,
        n_grid=n_grid,
        align_to_multiples=align_to_multiples,
    )
    # 2. Map density to PMF with semantics.
    return discretize_continuous_to_pmf(
        dist=dist,
        x_array=x_array,
        bound_type=bound_type,
        PMF_min_increment=tail_truncation,
        spacing_type=spacing_type,
    )

def _discretize_continuous_to_grid(
    dist: stats.rv_continuous,
    tail_truncation: float,
    spacing_type: SpacingType,
    n_grid: int,
    align_to_multiples: bool,
) -> NDArray[np.float64]:
    """
    Generate grid covering the quantile range defined by tail_truncation.
    """
    if n_grid <= 0:
        raise ValueError(
            "n_grid must be positive"
        )

    # Determine support bounds via quantiles
    x_min = dist.ppf(tail_truncation)
    x_max = dist.isf(tail_truncation)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError(f"Quantiles not finite: x_min={x_min}, x_max={x_max}")

    x_array = discretize_aligned_range(
        x_min=x_min,
        x_max=x_max,
        spacing_type=spacing_type,
        align_to_multiples=align_to_multiples,
        n_grid=n_grid,
    )
    # Truncate grid to stay strictly within the distribution's support
    support_min, support_max = dist.support()
    if np.isfinite(support_min):
        x_array = x_array[x_array > support_min]
    if np.isfinite(support_max):
        x_array = x_array[x_array < support_max]
    return x_array


def discretize_continuous_to_pmf(
    dist: stats.rv_continuous,
    x_array: NDArray[np.float64],
    bound_type: BoundType,
    PMF_min_increment: float,
    spacing_type: SpacingType,
) -> LinearDiscreteDist | GeometricDiscreteDist:
    """
    Convert continuous distribution to discrete PMF with bounding semantics.
    """
    # Compute raw probabilities for intervals [x_i, x_{i+1}) using PMF_min_increment.
    bin_probs, p_left, p_right = _compute_discrete_PMF(
        dist=dist,
        x_array=x_array,
        bound_type=bound_type,
        PMF_min_increment=PMF_min_increment
    )

    n = x_array.size
    PMF_array = np.zeros(n)

    if bound_type == BoundType.DOMINATES:
        # Shift mass right: left tail (-inf, x0) -> x0,
        # each interval [x_i, x_{i+1}) -> x_{i+1}, right tail (x_n, inf) -> inf,
        PMF_array[0] = p_left
        PMF_array[1:] = bin_probs
        p_neg_inf = 0.0
        p_pos_inf = p_right

    elif bound_type == BoundType.IS_DOMINATED:
        # Shift mass left: left tail (-inf, x0) -> -inf,
        # each interval [x_i, x_{i+1}) -> x_i, right tail (x_n, inf) -> x_n,
        PMF_array[:-1] = bin_probs
        PMF_array[-1] = p_right
        p_neg_inf = p_left
        p_pos_inf = 0.0
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    if spacing_type == SpacingType.LINEAR:
        return LinearDiscreteDist.from_x_array(
            x_array=x_array,
            PMF_array=PMF_array,
            p_neg_inf=p_neg_inf,
            p_pos_inf=p_pos_inf,
        ).validate_mass_conservation(bound_type)

    if spacing_type == SpacingType.GEOMETRIC:
        return GeometricDiscreteDist.from_x_array(
            x_array=x_array,
            PMF_array=PMF_array,
            p_neg_inf=p_neg_inf,
            p_pos_inf=p_pos_inf,
        ).validate_mass_conservation(bound_type)

    raise ValueError(f"Invalid spacing_type: {spacing_type}")

def discretize_aligned_range(
    x_min: float,
    x_max: float,
    spacing_type: SpacingType,
    align_to_multiples: bool,
    discretization: float | None = None,
    n_grid: int | None = None,
) -> NDArray[np.float64]:
    """Return a grid covering [x_min, x_max].

    Args:
        align_to_multiples: If True, align range to whole multiples of discretization.
                           If False, use x_min and x_max directly without alignment.
    """
    # Validate inputs
    if spacing_type not in (SpacingType.GEOMETRIC, SpacingType.LINEAR):
        raise ValueError(f"Unsupported spacing_type: {spacing_type}")
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got x_min={x_min}, x_max={x_max}")
    if spacing_type == SpacingType.GEOMETRIC and x_min <= 0:
        raise ValueError(f"Geometric spacing requires positive values, got x_min={x_min}, x_max={x_max}")
    
    # Compute missing parameter
    if n_grid is not None:
        if n_grid < MIN_GRID_SIZE:
            raise ValueError(f"n_grid must be >= {MIN_GRID_SIZE}, got {n_grid}")
        if discretization is not None:
            raise ValueError("Provide exactly one of discretization or n_grid")
        if spacing_type == SpacingType.GEOMETRIC:
            discretization = np.log(x_max / x_min) / (n_grid - 1)
        else:
            discretization = (x_max - x_min) / (n_grid - 1)
    else:
        if discretization is None:
            raise ValueError("Provide exactly one of discretization or n_grid")
        if discretization <= 0:
            raise ValueError("discretization must be positive")
        if spacing_type == SpacingType.GEOMETRIC:
            n_grid = max(int(np.ceil(np.log(x_max / x_min) / discretization)) + 1, MIN_GRID_SIZE)
        else:
            n_grid = max(int(np.ceil((x_max - x_min) / discretization)) + 1, MIN_GRID_SIZE)

    # From here, both discretization and n_grid are available
    if spacing_type == SpacingType.GEOMETRIC:
        if align_to_multiples:
            x_min = np.exp(np.floor(np.log(x_min) / discretization) * discretization)
            x_max = np.exp(np.ceil(np.log(x_max) / discretization) * discretization)
            n_grid = int(np.ceil(np.log(x_max / x_min) / discretization)) + 1
        return np.geomspace(x_min, x_max, n_grid)
    else:  # LINEAR
        if align_to_multiples:
            x_min = np.floor(x_min / discretization) * discretization
            x_max = np.ceil(x_max / discretization) * discretization
            n_grid = int(np.ceil((x_max - x_min) / discretization)) + 1
        return x_min + discretization * np.arange(n_grid, dtype=np.float64)

@njit(cache=True)
def _adaptive_bins_from_cdf(cdf: NDArray[np.float64], tail_truncation: float) -> NDArray[np.float64]:
    """Adaptive binning from CDF with mass accumulation.

    Accumulates mass from CDF increments until threshold is reached, then assigns
    accumulated mass to current bin. All mass is conserved - no mass is discarded.
    """
    n = cdf.size
    bin_probs = np.zeros(n - 1, dtype=np.float64)
    accumulated_mass = 0.0
    last_assignment_cdf = cdf[0]

    for i in range(n - 1):
        # Current increment in CDF
        current_increment = cdf[i + 1] - cdf[i]
        accumulated_mass += current_increment

        if accumulated_mass >= tail_truncation:
            # Assign accumulated mass to this bin
            bin_probs[i] = accumulated_mass
            accumulated_mass = 0.0
            last_assignment_cdf = cdf[i + 1]

    # Assign any remaining accumulated mass to the last bin
    if accumulated_mass > 0.0:
        bin_probs[n - 2] += accumulated_mass

    return bin_probs


@njit(cache=True)
def _adaptive_bins_from_sf(sf: NDArray[np.float64], tail_truncation: float) -> NDArray[np.float64]:
    """Adaptive binning from survival function with mass accumulation.

    Accumulates mass from SF increments until threshold is reached, then assigns
    accumulated mass to current bin. All mass is conserved - no mass is discarded.
    Processes from right to left (high to low x values).
    """
    n = sf.size
    bin_probs = np.zeros(n - 1, dtype=np.float64)
    accumulated_mass = 0.0
    last_assignment_sf = sf[-1]

    for i in range(n - 2, -1, -1):
        # Current increment in SF (going backwards)
        current_increment = sf[i] - sf[i + 1]
        accumulated_mass += current_increment

        if accumulated_mass >= tail_truncation:
            # Assign accumulated mass to this bin
            bin_probs[i] = accumulated_mass
            accumulated_mass = 0.0
            last_assignment_sf = sf[i]

    # Assign any remaining accumulated mass to the first bin
    if accumulated_mass > 0.0:
        bin_probs[0] += accumulated_mass

    return bin_probs

def _stable_cdf_and_sf(dist: stats.rv_continuous,
                       x_array: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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


def _compute_discrete_PMF(
    dist: stats.rv_continuous,
    x_array: NDArray[np.float64],
    bound_type: BoundType,
    PMF_min_increment: float
) -> tuple[NDArray[np.float64], float, float]:
    """
    Compute bin probabilities using adaptive CDF/SF increments with logcdf/logsf stability.
    PMF_min_increment controls the minimum CDF/SF increment that becomes a bin mass.
    """
    cdf, sf = _stable_cdf_and_sf(dist, x_array)
    p_left = cdf[0]
    p_right = sf[-1]
    PMF_min_increment = max(0.0, PMF_min_increment)

    if bound_type == BoundType.DOMINATES:
        # Suppress intermediate debug logging.
        bin_probs = _adaptive_bins_from_cdf(cdf, PMF_min_increment)
    elif bound_type == BoundType.IS_DOMINATED:
        # Suppress intermediate debug logging.
        bin_probs = _adaptive_bins_from_sf(sf, PMF_min_increment)
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    return bin_probs, p_left, p_right

# =============================================================================
# Change spacing type of discrete distributions
# =============================================================================

def change_spacing_type(dist: DiscreteDistBase,
                        tail_truncation: float,
                        loss_discretization: float,
                        spacing_type: SpacingType,
                        bound_type: BoundType) -> LinearDiscreteDist | GeometricDiscreteDist:
    """Convert distribution to different grid spacing (linear ↔ geometric).

    Remaps PMF onto a new grid with the requested spacing and discretization.
    Implementation trims zero/tail regions, computes new grid size, then remaps
    using domination-aware rounding (e.g., linear grids for dp_accounting output).

    Returns:
        Dense linear distribution if spacing_type is LINEAR,
        Dense geometric distribution if spacing_type is GEOMETRIC.
    """
    trunc_dist = dist.copy().truncate_edges(
        tail_truncation=tail_truncation / 2,
        bound_type=bound_type
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
    PMF_out = rediscritize_PMF(
        x_array=x_array,
        PMF_array=trunc_dist.PMF_array,
        x_array_out=x_array_out,
        dominates=(bound_type == BoundType.DOMINATES))
    PMF_out, p_neg_inf, p_pos_inf = enforce_mass_conservation(
        PMF_array=PMF_out,
        expected_neg_inf=dist.p_neg_inf,
        expected_pos_inf=dist.p_pos_inf,
        bound_type=bound_type,
    )

    if spacing_type == SpacingType.LINEAR:
        return LinearDiscreteDist.from_x_array(
            x_array=x_array_out,
            PMF_array=PMF_out,
            p_neg_inf=p_neg_inf,
            p_pos_inf=p_pos_inf,
        ).validate_mass_conservation(bound_type)

    if spacing_type == SpacingType.GEOMETRIC:
        return GeometricDiscreteDist.from_x_array(
            x_array=x_array_out,
            PMF_array=PMF_out,
            p_neg_inf=p_neg_inf,
            p_pos_inf=p_pos_inf,
        ).validate_mass_conservation(bound_type)

    raise ValueError(f"Invalid spacing_type: {spacing_type}")
@njit(cache=True)
def rediscritize_PMF(x_array: NDArray[np.float64],
                  PMF_array: NDArray[np.float64],
                  x_array_out: NDArray[np.float64],
                  dominates: bool,
                  tail_truncation: float = 0.0
                  ) -> NDArray[np.float64]:
    """Remap PMF onto new grid with domination-aware rounding.

    Maps each probability mass to output grid position based on domination semantics.
    Implementation: dominates=True uses ceil (pessimistic), False uses floor (optimistic).
    Uses Kahan summation for numerical accuracy.
    Returns: remapped PMF array.
    """
    n_out = x_array_out.size
    PMF_out = np.zeros(n_out)
    compensations = np.zeros(n_out)

    # single pointer into x_array_out since x_array is strictly increasing
    j = 0

    if dominates:
        # ceil: bin = first index with x_array_out[j] >= z; overflow right -> +inf bucket
        for i in range(x_array.size):
            z = x_array[i]
            mass = PMF_array[i]
            # Skip only zero-mass bins, not small-mass bins
            if mass <= 0:
                continue

            # advance while x_array_out[j] < z
            while j < n_out and x_array_out[j] < z:
                j += 1

            if j >= n_out:
                # overflow to the right: discard mass (goes to p_pos_inf via enforce_mass_conservation)
                continue
            else:
                # include values below x_array_out[0] in the first bin (ceil behavior)
                y = mass - compensations[j]
                t = PMF_out[j] + y
                compensations[j] = (t - PMF_out[j]) - y
                PMF_out[j] = t

    else:
        # floor: bin = last index with x_array_out[j] <= z; underflow left -> -inf bucket
        for i in range(x_array.size):
            z = x_array[i]
            mass = PMF_array[i]
            # Skip only zero-mass bins, not small-mass bins
            if mass <= 0:
                continue

            # advance while x_array_out[j] <= z
            while j < n_out and x_array_out[j] <= z:
                j += 1

            idx = j - 1
            if idx < 0:
                # underflow to the left: discard mass (goes to p_neg_inf via enforce_mass_conservation)
                continue
            else:
                # include values above x_array_out[-1] in the last bin (floor behavior)
                y = mass - compensations[idx]
                t = PMF_out[idx] + y
                compensations[idx] = (t - PMF_out[idx]) - y
                PMF_out[idx] = t

    return PMF_out
