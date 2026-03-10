import numpy as np
from numpy.typing import NDArray
import math

from dp_accounting.pld.privacy_loss_distribution import PrivacyLossDistribution

from PLD_accounting.dp_accounting_support import dp_accounting_pmf_to_discrete_dist, discrete_dist_to_dp_accounting_pmf
from PLD_accounting.types import Direction, BoundType, SpacingType
from PLD_accounting.discrete_dist import *
from PLD_accounting.core_utils import stable_array_equal, enforce_mass_conservation, compute_bin_width
from PLD_accounting.distribution_discretization import rediscritize_PMF, discretize_aligned_range


def subsample_PLD(
    pld: PrivacyLossDistribution,
    sampling_probability: float,
    bound_type: BoundType,
) -> PrivacyLossDistribution:
    """Apply PLD-dual based subsampling to a dp_accounting PLD.

    Args:
        pld: Privacy loss distribution to subsample
        sampling_probability: Probability of sampling each element
        bound_type: Whether to compute upper bounds (DOMINATES) or lower bounds (IS_DOMINATED)

    Returns:
        Subsampled privacy loss distribution
    """

    if sampling_probability <= 0 or sampling_probability > 1:
        raise ValueError("sampling_probability must be in (0, 1]")

    if sampling_probability == 1.0:
        return pld

    # Use pessimistic_estimate=True for upper bounds, False for lower bounds
    pessimistic = (bound_type == BoundType.DOMINATES)

    # Convert REMOVE direction
    remove_dist = dp_accounting_pmf_to_discrete_dist(pld._pmf_remove)
    subsampled_remove = subsample_PMF(
        base_pld=remove_dist,
        sampling_prob=sampling_probability,
        direction=Direction.REMOVE,
        bound_type=bound_type,
    )
    new_pmf_remove = discrete_dist_to_dp_accounting_pmf(
        subsampled_remove,
        pessimistic_estimate=pessimistic
    )

    # Handle ADD direction if present
    if pld._pmf_add is None:
        return PrivacyLossDistribution(
            pmf_remove=new_pmf_remove
        )

    add_dist = dp_accounting_pmf_to_discrete_dist(pld._pmf_add)
    subsampled_add = subsample_PMF(
        base_pld=add_dist,
        sampling_prob=sampling_probability,
        direction=Direction.ADD,
        bound_type=bound_type,
    )
    new_pmf_add = discrete_dist_to_dp_accounting_pmf(
        subsampled_add,
        pessimistic_estimate=pessimistic
    )

    return PrivacyLossDistribution(
        pmf_remove=new_pmf_remove,
        pmf_add=new_pmf_add
    )


def subsample_PMF(
    base_pld: LinearDiscreteDist,
    sampling_prob: float,
    direction: Direction,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """Apply subsampling amplification using the PLD-dual method on a discrete PMF.

    Args:
        base_pld: Base privacy loss distribution (must be LinearDiscreteDist)
        sampling_prob: Sampling probability in (0, 1]
        direction: Direction (REMOVE or ADD)
        bound_type: Whether to compute upper bounds (DOMINATES) or lower bounds (IS_DOMINATED)

    Returns:
        Subsampled privacy loss distribution.
        For sampling_prob < 1, returns LinearDiscreteDist on the computed target grid.
        For sampling_prob == 1, returns base_pld unchanged.
    """
    if direction not in (Direction.REMOVE, Direction.ADD):
        raise ValueError("Direction BOTH is invalid for subsampling")

    if sampling_prob <= 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in (0, 1]")

    if sampling_prob == 1.0:
        return base_pld

    if direction == Direction.REMOVE:
        width = compute_bin_width(base_pld.x_array)
        target_x_array = calc_subsampled_grid(
            lower_loss=base_pld.x_array[0],
            discretization=width,
            num_buckets=int(base_pld.x_array.size),
            grid_size=sampling_prob,
            direction=direction,
        )
        lower_pld = _calc_PLD_dual(base_pld)
        return _subsample_dist_mix(
            base_pld=base_pld,
            ref_pld=lower_pld,
            sampling_prob=sampling_prob,
            direction=direction,
            bound_type=bound_type,
            target_x_array=target_x_array,
        )

    return _subsample_dist(
        base_pld=base_pld,
        sampling_prob=sampling_prob,
        direction=direction,
        bound_type=bound_type,
    )


def calc_subsampled_grid(
    lower_loss: float,
    discretization: float,
    num_buckets: int,
    grid_size: float,
    direction: Direction,
) -> NDArray[np.float64]:
    """Compute a linear target grid using the subsampling-transformed endpoints."""
    sampling_prob = grid_size
    if sampling_prob <= 0 or sampling_prob > 1:
        raise ValueError("grid_size must be in (0, 1]")
    if num_buckets < 2:
        raise ValueError("num_buckets must be >= 2")

    base_lower = lower_loss
    base_upper = base_lower + num_buckets * discretization

    endpoints = np.array([base_lower, base_upper], dtype=np.float64)
    transformed_endpoints = _stable_subsampling_transformation(
        x_array=endpoints,
        sampling_prob=sampling_prob,
        direction=direction,
    )
    new_lower, new_upper = transformed_endpoints[0], transformed_endpoints[1]
    if not np.isfinite(new_lower) or not np.isfinite(new_upper) or new_upper <= new_lower:
        raise ValueError(
            "Subsampling transform produced invalid bounds: "
            f"new_lower={new_lower:.6g}, new_upper={new_upper:.6g}"
        )

    new_width = (new_upper - new_lower) / num_buckets
    return discretize_aligned_range(
        x_min=new_lower,
        x_max=new_upper,
        spacing_type=SpacingType.LINEAR,
        discretization=new_width,
        align_to_multiples=True,
    )


def _subsample_dist(base_pld: DiscreteDistBase,
                    sampling_prob: float,
                    direction: Direction,
                    bound_type: BoundType,
                    target_x_array: NDArray[np.float64] | None = None
                    ) -> LinearDiscreteDist:
    """Subsample a single distribution onto a linear target grid with domination semantics."""
    if target_x_array is None:
        width = compute_bin_width(base_pld.x_array)
        target_x_array = calc_subsampled_grid(
            lower_loss=base_pld.x_array[0],
            discretization=width,
            num_buckets=base_pld.x_array.size,
            grid_size=sampling_prob,
            direction=direction,
        )

    # transform the losses based on the direction
    transformed_x_array = _stable_subsampling_transformation(
        x_array=base_pld.x_array,
        sampling_prob=sampling_prob,
        direction=direction
    )

    # Rediscritize the random variable (transformed_x_array, PMF_array) to the target_x_array
    # using the domination direction and accounting for new inf / -inf mass.
    PMF_out = rediscritize_PMF(
        x_array=transformed_x_array,
        PMF_array=base_pld.PMF_array,
        x_array_out=target_x_array,
        dominates=(bound_type == BoundType.DOMINATES),
    )

    # Map infinite mass to the finite boundary of the subsampled loss range.
    expected_neg_inf = base_pld.p_neg_inf
    expected_pos_inf = base_pld.p_pos_inf
    if direction == Direction.REMOVE and base_pld.p_neg_inf > 0.0:
        min_loss = math.log1p(-sampling_prob)
        idx = int(np.searchsorted(target_x_array, min_loss, side="left"))
        idx = min(max(idx, 0), target_x_array.size - 1)
        PMF_out[idx] += base_pld.p_neg_inf
        expected_neg_inf = 0.0
    if direction == Direction.ADD and base_pld.p_pos_inf > 0.0:
        max_loss = -math.log1p(-sampling_prob)
        idx = int(np.searchsorted(target_x_array, max_loss, side="right")) - 1
        idx = min(max(idx, 0), target_x_array.size - 1)
        PMF_out[idx] += base_pld.p_pos_inf
        expected_pos_inf = 0.0

    PMF_out, p_neg_inf, p_pos_inf = enforce_mass_conservation(
        PMF_array=PMF_out,
        expected_neg_inf=expected_neg_inf,
        expected_pos_inf=expected_pos_inf,
        bound_type=bound_type,
    )
    return LinearDiscreteDist.from_x_array(
        x_array=target_x_array,
        PMF_array=PMF_out,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
    ).validate_mass_conservation(bound_type)


def _subsample_dist_mix(
    base_pld: DiscreteDistBase,
    ref_pld: DiscreteDistBase,
    sampling_prob: float,
    direction: Direction,
    bound_type: BoundType,
    target_x_array: NDArray[np.float64] | None = None
) -> LinearDiscreteDist:
    """Subsample and mix two bounds on a shared linear target grid."""
    if target_x_array is None:
        base_width = compute_bin_width(base_pld.x_array)
        target_x_array = calc_subsampled_grid(
            lower_loss=base_pld.x_array[0],
            discretization=base_width,
            num_buckets=int(base_pld.x_array.size),
            grid_size=sampling_prob,
            direction=direction,
        )
        target_x_array = _extend_target_grid_for_reference(
            target_x_array=target_x_array,
            ref_pld=ref_pld,
            sampling_prob=sampling_prob,
            direction=direction,
        )
    assert target_x_array is not None

    # Compute the transformed distribution of the two bounding random variables separately
    subsampled_base = _subsample_dist(
        base_pld=base_pld,
        target_x_array=target_x_array,
        sampling_prob=sampling_prob,
        direction=direction,
        bound_type=bound_type
    )
    subsampled_lower = _subsample_dist(
        base_pld=ref_pld,
        target_x_array=target_x_array,
        sampling_prob=sampling_prob,
        direction=direction,
        bound_type=bound_type
    )
    # The result is the mixture of the probabilities
    mixed = _mix_distributions(
        dist_1=subsampled_base,
        dist_2=subsampled_lower,
        weight_first=sampling_prob,
        bound_type=bound_type
    )
    return mixed


def _extend_target_grid_for_reference(
    target_x_array: NDArray[np.float64],
    ref_pld: DiscreteDistBase,
    sampling_prob: float,
    direction: Direction,
) -> NDArray[np.float64]:
    """Extend a target grid so it fully covers the transformed reference support."""
    ref_endpoints = _stable_subsampling_transformation(
        x_array=np.array([ref_pld.x_array[0], ref_pld.x_array[-1]], dtype=np.float64),
        sampling_prob=sampling_prob,
        direction=direction,
    )
    min_ref = np.min(ref_endpoints)
    max_ref = np.max(ref_endpoints)
    step = target_x_array[1] - target_x_array[0]

    if min_ref < target_x_array[0]:
        extra_bins = int(np.ceil((target_x_array[0] - min_ref) / step)) + 1
        extension = target_x_array[0] - step * np.arange(extra_bins, 0, -1, dtype=np.float64)
        target_x_array = np.concatenate([extension, target_x_array])

    if max_ref > target_x_array[-1]:
        extra_bins = int(np.ceil((max_ref - target_x_array[-1]) / step)) + 1
        extension = target_x_array[-1] + step * np.arange(1, extra_bins + 1, dtype=np.float64)
        target_x_array = np.concatenate([target_x_array, extension])

    return target_x_array


def _mix_distributions(dist_1: DiscreteDistBase, dist_2: DiscreteDistBase, weight_first: float, bound_type: BoundType) -> LinearDiscreteDist:
    """Mix two distributions on the same grid with weight_first for dist_1."""
    if not stable_array_equal(dist_1.x_array, dist_2.x_array):
        raise ValueError("Distributions must share the same loss grid for mixing")
    if weight_first < 0 or weight_first > 1:
        raise ValueError("weight_first must be in [0, 1]")

    mixed_probs = weight_first * dist_1.PMF_array + (1.0 - weight_first) * dist_2.PMF_array
    mixed_probs, mixed_p_neg_inf, mixed_p_pos_inf = enforce_mass_conservation(
        PMF_array=mixed_probs,
        expected_neg_inf=weight_first * dist_1.p_neg_inf + (1.0 - weight_first) * dist_2.p_neg_inf,
        expected_pos_inf=weight_first * dist_1.p_pos_inf + (1.0 - weight_first) * dist_2.p_pos_inf,
        bound_type=bound_type,
    )
    return LinearDiscreteDist.from_x_array(
        x_array=dist_1.x_array,
        PMF_array=mixed_probs,
        p_neg_inf=mixed_p_neg_inf,
        p_pos_inf=mixed_p_pos_inf,
    ).validate_mass_conservation(bound_type)


def _stable_subsampling_transformation(
    x_array: NDArray[np.float64],
    sampling_prob: float,
    direction: Direction,
) -> NDArray[np.float64]:
    """Transform privacy losses for subsampling in a stable manner.

    Remove direction: l' = log(1 + q * (exp(l) - 1))
    Add direction:    l' = -log(1 + q * (exp(-l) - 1))
    """
    if sampling_prob <= 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in (0, 1]")
    if sampling_prob == 1:
        return x_array
    if direction == Direction.ADD:
        x_array = -x_array

    new_x_array = np.zeros_like(x_array)
    pos = x_array > 0
    # Stable for large positive losses.
    new_x_array[pos] = x_array[pos] + np.log(
        sampling_prob + (1.0 - sampling_prob) * np.exp(-x_array[pos])
    )
    new_x_array[~pos] = np.log1p(sampling_prob * np.expm1(x_array[~pos]))

    return new_x_array if direction == Direction.REMOVE else -new_x_array


def _calc_PLD_dual(upper: DiscreteDistBase) -> GeneralDiscreteDist:
    """Compute the PLD dual Q(l)=P(l)*e^{-l} from an upper-bound distribution."""
    if upper.p_neg_inf > 1e-9:
        raise ValueError(
            f"Input is not a valid upper bound PLD (p_neg_inf={upper.p_neg_inf} > 0)"
        )

    losses = upper.x_array
    probs = upper.PMF_array

    # Q(l) = P(l) * e^{-l}, computed in log space for numerical stability
    lower_probs = np.zeros_like(probs)
    mask = probs > 0
    lower_probs[mask] = np.exp(np.log(probs[mask]) - losses[mask])

    sum_prob = math.fsum(map(float, lower_probs))
    p_neg_inf = max(0.0, 1.0 - sum_prob)
    # For the lower bound: p_pos_inf = 0, p_neg_inf chosen to make total mass = 1
    return GeneralDiscreteDist(
        x_array=losses,
        PMF_array=lower_probs,
        p_neg_inf=p_neg_inf,
        p_pos_inf=0.0
    )
