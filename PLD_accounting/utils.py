from typing import Callable, Any
import numpy as np
from numpy.typing import NDArray
from numba import njit

from PLD_accounting.distribution_utils import stable_array_equal, enforce_mass_conservation
from PLD_accounting.types import BoundType
from PLD_accounting.discrete_dist import *

# =============================================================================
# Infinite-Mass Utilities
# =============================================================================

def convolve_infinite_masses(*,
    p_neg_inf_1: float,
    p_pos_inf_1: float,
    p_neg_inf_2: float,
    p_pos_inf_2: float
) -> tuple[float, float]:
    """
    Combine infinity atoms for two independent distributions.
    """
    p_neg_inf = float(np.clip(-np.expm1(np.log1p(-p_neg_inf_1) + np.log1p(-p_neg_inf_2)), 0.0, 1.0))
    p_pos_inf = float(np.clip(-np.expm1(np.log1p(-p_pos_inf_1) + np.log1p(-p_pos_inf_2)), 0.0, 1.0))
    return p_neg_inf, p_pos_inf


def self_convolve_infinite_mass(*, p_neg_inf: float, p_pos_inf: float, T: int) -> tuple[float, float]:
    """
    Compute infinity atoms for T-fold self-convolution.
    """
    p_neg_inf = float(np.clip(-np.expm1(T * np.log1p(-p_neg_inf)), 0.0, 1.0))
    p_pos_inf = float(np.clip(-np.expm1(T * np.log1p(-p_pos_inf)), 0.0, 1.0))
    return p_neg_inf, p_pos_inf


# =============================================================================
# Public Convolution Utilities
# =============================================================================

def binary_self_convolve(*,
    dist: DiscreteDistBase,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
    convolve: Callable[..., DiscreteDistBase]
) -> DiscreteDistBase:
    """
    Exponentiation by squaring-based self-convolution using a provided convolve function.

    Algorithm 3 (`self-conv`) in Appendix C.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if T == 1:
        return dist

    base_dist = dist
    acc_dist = None
    tail_truncation /= 4
    while T > 0:
        if T & 1:
            if acc_dist is None:
                acc_dist = base_dist
            else:
                acc_dist = convolve(
                    dist_1=acc_dist,
                    dist_2=base_dist,
                    tail_truncation=tail_truncation/T,
                    bound_type=bound_type
                )
        T >>= 1
        if T > 0:
            base_dist = convolve(
                dist_1=base_dist,
                dist_2=base_dist,
                tail_truncation=tail_truncation/T,
                bound_type=bound_type
            )
    # If T is a power of two, acc_dist is never set; return the final squared base_dist.
    return acc_dist if acc_dist is not None else base_dist

def combine_distributions(*,
    dist_1: DiscreteDistBase,
    dist_2: DiscreteDistBase,
    bound_type: BoundType
) -> GeneralDiscreteDist:
    """
    Combine two distributions by tightening bounds via CCDF min/max.

    For DOMINATES: returns tighter dominating distribution using pointwise min CCDF.
    For IS_DOMINATED: returns tighter dominated distribution using pointwise max CCDF.
    """
    ccdf_op: Any
    if bound_type == BoundType.DOMINATES:
        ccdf_op = np.minimum
    elif bound_type == BoundType.IS_DOMINATED:
        ccdf_op = np.maximum
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    if stable_array_equal(a=dist_1.x_array, b=dist_2.x_array):
        dist_1_aligned, dist_2_aligned = dist_1, dist_2
    else:
        dist_1_aligned, dist_2_aligned = _align_distributions_to_union_grid(
            dist_1=dist_1,
            dist_2=dist_2,
        )

    x_array = dist_1_aligned.x_array
    ccdf_1 = _CCDF_from_PMF(dist_1_aligned)
    ccdf_2 = _CCDF_from_PMF(dist_2_aligned)
    combined_ccdf = ccdf_op(ccdf_1, ccdf_2)
    PMF_array = combined_ccdf[:-2] - combined_ccdf[1:-1]

    PMF_array, p_neg_inf, p_pos_inf = enforce_mass_conservation(
        PMF_array=PMF_array,
        expected_neg_inf=max(dist_1_aligned.p_neg_inf, dist_2_aligned.p_neg_inf),
        expected_pos_inf=max(dist_1_aligned.p_pos_inf, dist_2_aligned.p_pos_inf),
        bound_type=bound_type,
    )

    return GeneralDiscreteDist(
        x_array=x_array,
        PMF_array=PMF_array,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf
    )

# =============================================================================
# Distribution Transform Utilities
# =============================================================================

def exp_linear_to_geometric(dist: LinearDiscreteDist) -> GeometricDiscreteDist:
    """
    Apply exp(.) to a linear-grid distribution, producing a geometric-grid distribution.
    """
    x_min_exp = float(np.exp(dist.x_min))
    ratio_exp = float(np.exp(dist.x_gap))

    return GeometricDiscreteDist(
        x_min=x_min_exp,
        ratio=ratio_exp,
        PMF_array=dist.PMF_array.copy(),
        p_neg_inf=dist.p_neg_inf,
        p_pos_inf=dist.p_pos_inf,
    )


def log_geometric_to_linear(dist: GeometricDiscreteDist) -> LinearDiscreteDist:
    """
    Apply log(.) to a geometric-grid distribution, producing a linear-grid distribution.
    """
    x_min_log = float(np.log(dist.x_min))
    x_gap_log = float(np.log(dist.ratio))

    return LinearDiscreteDist(
        x_min=x_min_log,
        x_gap=x_gap_log,
        PMF_array=dist.PMF_array.copy(),
        p_neg_inf=dist.p_neg_inf,
        p_pos_inf=dist.p_pos_inf,
    )


def negate_reverse_linear_distribution(
    dist: LinearDiscreteDist,
) -> LinearDiscreteDist:
    """
    Map X -> -X, reverse PMF order, and swap infinity atoms.
    """
    n = dist.PMF_array.size
    return LinearDiscreteDist(
        x_min=-(dist.x_min + dist.x_gap * (n - 1)),
        x_gap=dist.x_gap,
        PMF_array=np.flip(dist.PMF_array),
        p_neg_inf=dist.p_pos_inf,
        p_pos_inf=dist.p_neg_inf,
    )


def calc_pld_dual(realization: PLDRealization) -> PLDRealization:
    """
    Compute the paper PLD dual ``D(L)`` (Definition 3.1).

    Algorithm 7 (`PLD-dual`) in Appendix C.

    For a PLD realization ``L`` with support ``l`` and mass ``f_L(l)``, the dual has:
    - finite mass ``f_D(-l) = f_L(l) * exp(-l)``,
    - support reflected to ``-l``,
    - residual mass at ``+inf``.
    """
    if not isinstance(realization, PLDRealization):
        raise TypeError(f"calc_pld_dual requires PLDRealization, got {type(realization)}")

    dual_probs_aligned = np.zeros_like(realization.PMF_array)
    mask = realization.PMF_array > 0
    dual_probs_aligned[mask] = np.exp(np.log(realization.PMF_array[mask]) - realization.x_array[mask])
    dual_probs = np.flip(dual_probs_aligned)

    sum_prob = float(np.sum(dual_probs, dtype=np.float64))
    if sum_prob > 1.0:
        dual_probs *= 1.0 / sum_prob
        sum_prob = 1.0

    return PLDRealization(
        x_min=-(realization.x_min + realization.x_gap * (realization.PMF_array.size - 1)),
        x_gap=realization.x_gap,
        PMF_array=dual_probs,
        p_loss_inf=max(0.0, 1.0 - sum_prob),
        p_loss_neg_inf=0.0,
    )


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _align_distributions_to_union_grid(*,
    dist_1: DiscreteDistBase,
    dist_2: DiscreteDistBase,
) -> tuple[GeneralDiscreteDist, GeneralDiscreteDist]:
    """
    Return distributions on a shared grid by inserting zero-mass points.
    """
    x_union = np.unique(np.concatenate((dist_1.x_array, dist_2.x_array)))
    return (
        _expand_to_grid(
            dist=dist_1,
            grid=x_union,
        ),
        _expand_to_grid(
            dist=dist_2,
            grid=x_union,
        ),
    )


def _expand_to_grid(*,
    dist: DiscreteDistBase,
    grid: NDArray[np.float64],
) -> GeneralDiscreteDist:
    """
    Insert zero-mass points for missing support values.
    """
    x = dist.x_array
    pmf = dist.PMF_array
    expanded_pmf = np.zeros_like(grid, dtype=np.float64)
    indices = np.searchsorted(grid, x)
    if not np.all(grid[indices] == x):
        raise ValueError("Target grid must contain all original support points")
    expanded_pmf[indices] = pmf
    return GeneralDiscreteDist(
        x_array=grid,
        PMF_array=expanded_pmf,
        p_neg_inf=dist.p_neg_inf,
        p_pos_inf=dist.p_pos_inf
    )

def _CCDF_from_PMF(dist: DiscreteDistBase) -> NDArray[np.float64]:
    """
    Convert GeneralDiscreteDist PMF to padded complementary CDF.

    Returns [finite_mass + p_pos_inf, finite_mass - p_0 + p_pos_inf, ..., 0]
    for losses [−∞, l_0, l_1, ..., +∞].
    Implementation pads with zero and infinity mass, then computes reverse cumsum using Kahan summation.
    Used for stable CCDF operations when tightening bounds.
    """
    padded_probs = np.concatenate(([0.0], dist.PMF_array, [dist.p_pos_inf]))
    return _kahan_reverse_exclusive_cumsum(
        padded_probs=padded_probs,
    )

@njit(cache=True)
def _kahan_reverse_exclusive_cumsum(*,
    padded_probs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute exclusive CCDF using Kahan summation for numerical stability.

    Computes exclusive reverse cumulative sum: CCDF[i] = sum(padded_probs[i+1:]).
    Uses Kahan compensated summation to minimize floating-point rounding errors.
    """
    n = len(padded_probs)
    ccdf = np.zeros(n, dtype=np.float64)

    # Start from the right (highest index) and accumulate backwards
    running_sum = 0.0
    compensation = 0.0

    for i in range(n - 1, -1, -1):
        # Store the running sum BEFORE adding current element (exclusive)
        ccdf[i] = running_sum

        # Kahan summation: compensated addition of current element
        y = padded_probs[i] - compensation
        t = running_sum + y
        compensation = (t - running_sum) - y
        running_sum = t

    return ccdf
