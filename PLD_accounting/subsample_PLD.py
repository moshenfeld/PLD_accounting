import numpy as np
from numpy.typing import NDArray
import math

from dp_accounting.pld.privacy_loss_distribution import PrivacyLossDistribution

from PLD_accounting.types import Direction, BoundType, SpacingType
from PLD_accounting.discrete_dist import DiscreteDistBase, LinearDiscreteDist, PLDRealization
from PLD_accounting.distribution_utils import stable_isclose, enforce_mass_conservation, compute_bin_width
from PLD_accounting.utils import calc_pld_dual, negate_reverse_linear_distribution
from PLD_accounting.dp_accounting_support import dp_accounting_pmf_to_pld_realization, linear_dist_to_dp_accounting_pmf
from PLD_accounting.distribution_discretization import rediscritize_PMF, discretize_aligned_range

# =============================================================================
# Public Subsampling API
# =============================================================================

def subsample_PLD(
    pld: PrivacyLossDistribution,
    sampling_probability: float,
) -> PrivacyLossDistribution:
    """Apply PLD-dual based subsampling to a dp_accounting PLD.

    Args:
        pld: Privacy loss distribution to subsample
        sampling_probability: Probability of sampling each element

    Returns:
        Subsampled privacy loss distribution
    """

    if sampling_probability <= 0 or sampling_probability > 1:
        raise ValueError("sampling_probability must be in (0, 1]")

    if sampling_probability == 1.0:
        return pld

    # Convert REMOVE direction
    remove_dist = dp_accounting_pmf_to_pld_realization(pld._pmf_remove)
    subsampled_remove = subsample_PMF(
        base_pld=remove_dist,
        sampling_prob=sampling_probability,
        direction=Direction.REMOVE,
    )
    subsampled_remove_pmf = linear_dist_to_dp_accounting_pmf(
        dist=subsampled_remove,
        pessimistic_estimate=True,
    )

    # Handle ADD direction if present
    if pld._pmf_add is None:
        return PrivacyLossDistribution(
            pmf_remove=subsampled_remove_pmf
        )

    add_dist = dp_accounting_pmf_to_pld_realization(pld._pmf_add)
    subsampled_add = subsample_PMF(
        base_pld=add_dist,
        sampling_prob=sampling_probability,
        direction=Direction.ADD,
    )
    subsampled_add_pmf = linear_dist_to_dp_accounting_pmf(
        dist=subsampled_add,
        pessimistic_estimate=True,
    )

    return PrivacyLossDistribution(
        pmf_remove=subsampled_remove_pmf,
        pmf_add=subsampled_add_pmf
    )


def subsample_PMF(
    base_pld: PLDRealization,
    sampling_prob: float,
    direction: Direction,
) -> PLDRealization:
    """Apply subsampling amplification using the PLD-dual method on a discrete PMF.

    Algorithms 8 and 9 (`PLD-subsam-remove/add`) in Appendix C.

    Args:
        base_pld: Base privacy-loss realization on a linear loss grid.
        sampling_prob: Sampling probability in (0, 1]
        direction: Direction (REMOVE or ADD)

    Returns:
        Subsampled loss-space dominating (upper) bound as a ``PLDRealization``.

    Notes:
        Supports only the DOMINATES bound type.
    """
    if direction not in (Direction.REMOVE, Direction.ADD):
        raise ValueError("Direction BOTH is invalid for subsampling")
    if not isinstance(base_pld, PLDRealization):
        raise TypeError(f"subsample_PMF requires PLDRealization, got {type(base_pld)}")
    if sampling_prob <= 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in (0, 1]")
    if sampling_prob == 1.0:
        return base_pld

    if direction == Direction.REMOVE:
        # Set target grid
        width = compute_bin_width(base_pld.x_array)
        target_x_array = _calc_subsampled_grid(
            lower_loss=base_pld.x_array[0],
            discretization=width,
            num_buckets=int(base_pld.x_array.size),
            grid_size=sampling_prob,
            direction=direction,
        )
        # Compute negative dual
        dual_pld = calc_pld_dual(base_pld)
        neg_dual_pld = negate_reverse_linear_distribution(dual_pld)
        # Transform and re-discretize each distribution and mix.
        out = _subsample_dist_mix(
            base_pld=base_pld,
            neg_dual_pld=neg_dual_pld,
            sampling_prob=sampling_prob,
            direction=direction,
            target_x_array=target_x_array,
        )
        return PLDRealization.from_linear_dist(out)
    elif direction == Direction.ADD:
        out = _subsample_dist(
            base_pld=base_pld,
            sampling_prob=sampling_prob,
            direction=direction,
        )
        return PLDRealization.from_linear_dist(out)
    raise RuntimeError("unreachable direction branch")

# =============================================================================
# Internal Subsampling Helpers
# =============================================================================

def _subsample_dist(*,
    base_pld: DiscreteDistBase,
    sampling_prob: float,
    direction: Direction,
    target_x_array: NDArray[np.float64] | None = None,
) -> LinearDiscreteDist:
    """Subsample a single distribution onto a linear target grid in DOMINATES mode.

    Paper mapping: Appendix C Algorithm 9 finite-support transform and PMF
    transfer. The implementation keeps these same components while making the
    re-binning and infinite-mass placement explicit.
    """

    if target_x_array is None:
        # Algorithm 9, support update: build transformed target grid.
        width = compute_bin_width(base_pld.x_array)
        include_right = None
        if direction == Direction.ADD and base_pld.p_pos_inf > 0.0:
            include_right = -math.log1p(-sampling_prob)
        target_x_array = _calc_subsampled_grid(
            lower_loss=base_pld.x_array[0],
            discretization=width,
            num_buckets=base_pld.x_array.size,
            grid_size=sampling_prob,
            direction=direction,
            include_right=include_right,
        )
    elif direction == Direction.ADD and base_pld.p_pos_inf > 0.0:
        max_loss = -math.log1p(-sampling_prob)
        if max_loss > target_x_array[-1]:
            raise ValueError(
                "target_x_array must include add-direction max loss "
                f"-log(1-q)={max_loss:.15g}, got right endpoint={target_x_array[-1]:.15g}"
            )

    # Algorithm 8/9 core transform: l -> l_lambda using stable inverse-phi.
    transformed_x_array = _stable_subsampling_transformation(
        x_array=base_pld.x_array,
        sampling_prob=sampling_prob,
        direction=direction
    )

    # Pseudocode PMF transfer on transformed support, implemented as
    # domination-preserving re-discretization onto a shared linear grid.
    PMF_out = rediscritize_PMF(
        x_array=transformed_x_array,
        PMF_array=base_pld.PMF_array,
        x_array_out=target_x_array,
        dominates=True,
    )

    # Boundary handling for infinite atoms: map to the extremal finite losses
    # implied by subsampling (log(1-lambda) for REMOVE, -log(1-lambda) for ADD).
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
        idx = int(np.searchsorted(target_x_array, max_loss, side="left"))
        idx = min(max(idx, 0), target_x_array.size - 1)
        PMF_out[idx] += base_pld.p_pos_inf
        expected_pos_inf = 0.0

    # Final normalization under DOMINATES semantics.
    PMF_out, p_neg_inf, p_pos_inf = enforce_mass_conservation(
        PMF_array=PMF_out,
        expected_neg_inf=expected_neg_inf,
        expected_pos_inf=expected_pos_inf,
        bound_type=BoundType.DOMINATES,
    )
    return LinearDiscreteDist.from_x_array(
        x_array=target_x_array,
        PMF_array=PMF_out,
        p_neg_inf=p_neg_inf,
        p_pos_inf=p_pos_inf,
    ).validate_mass_conservation(BoundType.DOMINATES)


def _subsample_dist_mix(*,
    base_pld: DiscreteDistBase,
    neg_dual_pld: DiscreteDistBase,
    sampling_prob: float,
    direction: Direction,
    target_x_array: NDArray[np.float64] | None = None,
) -> LinearDiscreteDist:
    """Subsample and mix base and negative-dual distributions on a shared linear grid.

    Paper mapping: Appendix C Algorithm 8 (`PLD-subsam-remove`) mixture line
    ``lambda * f_L + (1-lambda) * f_D``. The implementation computes each
    transformed branch on the same grid before applying that convex mixture.
    """

    if target_x_array is None:
        # Algorithm 8 support update for the base branch.
        base_width = compute_bin_width(base_pld.x_array)
        target_x_array = _calc_subsampled_grid(
            lower_loss=base_pld.x_array[0],
            discretization=base_width,
            num_buckets=int(base_pld.x_array.size),
            grid_size=sampling_prob,
            direction=direction,
        )
        # Ensure the -D(L) branch is covered by the same target lattice.
        target_x_array = _extend_target_grid_for_reference(
            target_x_array=target_x_array,
            neg_dual_pld=neg_dual_pld,
            sampling_prob=sampling_prob,
            direction=direction,
        )
    assert target_x_array is not None

    # Algorithm 8 branch 1: transform/re-discretize L.
    subsampled_base = _subsample_dist(
        base_pld=base_pld,
        target_x_array=target_x_array,
        sampling_prob=sampling_prob,
        direction=direction,
    )
    # Algorithm 8 branch 2: transform/re-discretize -D(L).
    subsampled_lower = _subsample_dist(
        base_pld=neg_dual_pld,
        target_x_array=target_x_array,
        sampling_prob=sampling_prob,
        direction=direction,
    )
    # Algorithm 8 mixture step.
    mixed = _mix_distributions(
        dist_1=subsampled_base,
        dist_2=subsampled_lower,
        weight_first=sampling_prob,
    )
    return mixed


def _calc_subsampled_grid(*,
    lower_loss: float,
    discretization: float,
    num_buckets: int,
    grid_size: float,
    direction: Direction,
    include_right: float | None = None,
) -> NDArray[np.float64]:
    """Compute the transformed linear target grid used by Algorithms 8/9."""
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

    if include_right is not None:
        new_upper = max(new_upper, include_right)

    new_width = (new_upper - new_lower) / num_buckets
    return discretize_aligned_range(
        x_min=new_lower,
        x_max=new_upper,
        spacing_type=SpacingType.LINEAR,
        discretization=new_width,
        align_to_multiples=True,
    )

def _extend_target_grid_for_reference(*,
    target_x_array: NDArray[np.float64],
    neg_dual_pld: DiscreteDistBase,
    sampling_prob: float,
    direction: Direction,
) -> NDArray[np.float64]:
    """Extend the target grid so transformed ``-D(L)`` support is fully covered.

    This is the implementation-level support completion used before the
    Algorithm 8 convex mixture.
    """
    ref_endpoints = _stable_subsampling_transformation(
        x_array=np.array([neg_dual_pld.x_array[0], neg_dual_pld.x_array[-1]], dtype=np.float64),
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


def _mix_distributions(*,
    dist_1: LinearDiscreteDist,
    dist_2: LinearDiscreteDist,
    weight_first: float,
) -> LinearDiscreteDist:
    """Mix two same-grid distributions with weight ``weight_first`` for ``dist_1``.

    Paper mapping: Algorithm 8 line
    ``f_{L_lambda} = lambda f_L + (1-lambda) f_D`` after both operands are
    represented on the common transformed grid.
    """
    if not isinstance(dist_1, LinearDiscreteDist) or not isinstance(dist_2, LinearDiscreteDist):
        raise TypeError(
            "_mix_distributions requires LinearDiscreteDist inputs, "
            f"got {type(dist_1)} and {type(dist_2)}"
        )
    if dist_1.PMF_array.size != dist_2.PMF_array.size:
        raise ValueError("Distributions must have the same number of bins for mixing")
    if not stable_isclose(a=dist_1.x_min, b=dist_2.x_min):
        raise ValueError("Distributions must share the same x_min for mixing")
    if not stable_isclose(a=dist_1.x_gap, b=dist_2.x_gap):
        raise ValueError("Distributions must share the same x_gap for mixing")
    if weight_first < 0 or weight_first > 1:
        raise ValueError("weight_first must be in [0, 1]")

    mixed_probs = weight_first * dist_1.PMF_array + (1.0 - weight_first) * dist_2.PMF_array
    mixed_probs, mixed_p_neg_inf, mixed_p_pos_inf = enforce_mass_conservation(
        PMF_array=mixed_probs,
        expected_neg_inf=weight_first * dist_1.p_neg_inf + (1.0 - weight_first) * dist_2.p_neg_inf,
        expected_pos_inf=weight_first * dist_1.p_pos_inf + (1.0 - weight_first) * dist_2.p_pos_inf,
        bound_type=BoundType.DOMINATES,
    )
    return LinearDiscreteDist(
        x_min=dist_1.x_min,
        x_gap=dist_1.x_gap,
        PMF_array=mixed_probs,
        p_neg_inf=mixed_p_neg_inf,
        p_pos_inf=mixed_p_pos_inf,
    ).validate_mass_conservation(BoundType.DOMINATES)
def _stable_subsampling_transformation(*,
    x_array: NDArray[np.float64],
    sampling_prob: float,
    direction: Direction,
) -> NDArray[np.float64]:
    """Transform privacy losses for subsampling in a stable manner.

    Remove direction: l' = log(1 + q * (exp(l) - 1))
    Add direction:    l' = -log(1 + q * (exp(-l) - 1))

    Paper mapping: Appendix C Algorithms 8-9 inverse ``phi_lambda`` transform.
    For positive losses we use a log-sum form to avoid overflow; for non-positive
    losses we use ``log1p(expm1(.))`` for cancellation stability.
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
