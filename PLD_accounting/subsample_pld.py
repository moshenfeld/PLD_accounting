"""Subsampling utilities for privacy loss distributions."""

from __future__ import annotations

import math

import numpy as np
from dp_accounting.pld.privacy_loss_distribution import PrivacyLossDistribution
from numpy.typing import NDArray

from PLD_accounting.discrete_dist import (
    DiscreteDistBase,
    Domain,
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_discretization import (
    aligned_grid_params,
    project_dist_onto_grid_ctd,
)
from PLD_accounting.distribution_utils import (
    compute_bin_width,
)
from PLD_accounting.dp_accounting_support import (
    dp_accounting_pmf_to_pld_realization,
    linear_dist_to_dp_accounting_pmf,
)
from PLD_accounting.types import (
    BoundType,
    Direction,
    SpacingType,
)
from PLD_accounting.utils import calc_pld_dual, negate_reverse_linear_distribution

# =============================================================================
# Public Subsampling API
# =============================================================================


def subsample_pld(
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
    subsampled_remove = subsample_pld_realization(
        base_pld=remove_dist,
        sampling_prob=sampling_probability,
        direction=Direction.REMOVE,
    )
    subsampled_remove_pmf = linear_dist_to_dp_accounting_pmf(
        dist=subsampled_remove,
        bound_type=BoundType.DOMINATES,
    )

    # Handle ADD direction if present
    if pld._pmf_add is None:
        return PrivacyLossDistribution(pmf_remove=subsampled_remove_pmf)

    add_dist = dp_accounting_pmf_to_pld_realization(pld._pmf_add)
    subsampled_add = subsample_pld_realization(
        base_pld=add_dist,
        sampling_prob=sampling_probability,
        direction=Direction.ADD,
    )
    subsampled_add_pmf = linear_dist_to_dp_accounting_pmf(
        dist=subsampled_add,
        bound_type=BoundType.DOMINATES,
    )

    return PrivacyLossDistribution(pmf_remove=subsampled_remove_pmf, pmf_add=subsampled_add_pmf)


def subsample_pld_realization(
    base_pld: PLDRealization,
    sampling_prob: float,
    direction: Direction,
) -> PLDRealization:
    """Apply subsampling amplification to a PLD realization using the PLD-dual method.

    Algorithms 8 and 9 (`PLDsubsam-remove` / `PLDsubsam-add`), using Algorithm
    10 (`subsam-core`) for the shared transform and the final projection
    described in the "Connect-the-dots rediscretization" remark following
    Algorithm 6, in Appendix C of https://arxiv.org/abs/2602.17284.

    Args:
        base_pld: Base privacy-loss realization on a linear loss grid.
        sampling_prob: Sampling probability in (0, 1]
        direction: Direction (REMOVE or ADD)

    Returns:
        Subsampled loss-space dominating (upper) bound as a ``PLDRealization``.

    The transformed source is projected with hockey-stick-preserving CtD.

    """
    if not isinstance(base_pld, PLDRealization):
        raise TypeError(f"subsample_pld_realization requires PLDRealization, got {type(base_pld)}")
    if sampling_prob <= 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in (0, 1]")
    if sampling_prob == 1.0:
        return base_pld

    if direction not in (Direction.REMOVE, Direction.ADD):
        raise ValueError("Direction BOTH is invalid for subsampling")

    if direction == Direction.REMOVE:
        target_x_array = _calc_subsampled_grid(
            min_loss=base_pld.x_array[0],
            discretization=compute_bin_width(base_pld.x_array),
            num_buckets=int(base_pld.x_array.size),
            sampling_prob=sampling_prob,
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
        return out
    out = _subsample_dist(
        base_pld=base_pld,
        sampling_prob=sampling_prob,
        direction=direction,
    )
    return out


# =============================================================================
# Internal Subsampling Helpers
# =============================================================================


def _subsample_dist_mix(
    *,
    base_pld: DiscreteDistBase,
    neg_dual_pld: DiscreteDistBase,
    sampling_prob: float,
    direction: Direction,
    target_x_array: NDArray[np.float64] | None = None,
) -> PLDRealization:
    """Subsample and mix base and negative-dual distributions on a shared linear grid.

    Paper mapping: Algorithm 8 (`PLDsubsam-remove`), in Appendix C of
    https://arxiv.org/abs/2602.17284, mixture line ``lambda * f_L +
    (1-lambda) * f_D``. The implementation computes each transformed branch on
    the same grid before applying that convex mixture.
    """
    if target_x_array is None:
        # Algorithm 8 support update for the base branch.
        base_width = compute_bin_width(base_pld.x_array)
        target_x_array = _calc_subsampled_grid(
            min_loss=base_pld.x_array[0],
            discretization=base_width,
            num_buckets=int(base_pld.x_array.size),
            sampling_prob=sampling_prob,
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

    base_source = _subsample_transformed_source(
        base_pld=base_pld, sampling_prob=sampling_prob, direction=direction
    )
    dual_source = _subsample_transformed_source(
        base_pld=neg_dual_pld, sampling_prob=sampling_prob, direction=direction
    )
    # Algorithm 8 projects each branch before mixing.  Projection onto a fixed
    # grid is linear in the atomic masses, so mix first and project only once.
    mixed_source = _atomic_source(
        losses=np.concatenate((base_source.x_array, dual_source.x_array)),
        masses=np.concatenate(
            (
                sampling_prob * base_source.prob_arr,
                (1.0 - sampling_prob) * dual_source.prob_arr,
            )
        ),
        p_min=(sampling_prob * base_source.p_min + (1.0 - sampling_prob) * dual_source.p_min),
        p_max=(sampling_prob * base_source.p_max + (1.0 - sampling_prob) * dual_source.p_max),
    )
    grid = GridSpec(
        x_0=float(target_x_array[0]),
        step=compute_bin_width(target_x_array),
        n=int(target_x_array.size),
        spacing_type=SpacingType.LINEAR,
    )
    return project_dist_onto_grid_ctd(
        dist=mixed_source,
        grid=grid,
    )


def _subsample_dist(
    *,
    base_pld: DiscreteDistBase,
    sampling_prob: float,
    direction: Direction,
    target_x_array: NDArray[np.float64] | None = None,
) -> PLDRealization:
    """Subsample a single distribution onto a linear target grid in DOMINATES mode.

    Paper mapping: Algorithm 10 (`subsam-core`), with the Algorithm 9 sign convention when
    ``direction`` is ADD. The implementation keeps these same components while
    making the re-binning and infinite-mass placement explicit.
    """
    if target_x_array is None:
        # Algorithm 10 support update: build transformed target grid.
        width = compute_bin_width(base_pld.x_array)
        include_right = None
        if direction == Direction.ADD and base_pld.p_max > 0.0:
            include_right = -math.log1p(-sampling_prob)
        target_x_array = _calc_subsampled_grid(
            min_loss=base_pld.x_array[0],
            discretization=width,
            num_buckets=base_pld.x_array.size,
            sampling_prob=sampling_prob,
            direction=direction,
            include_right=include_right,
        )
    elif direction == Direction.ADD and base_pld.p_max > 0.0:
        max_loss = -math.log1p(-sampling_prob)
        if max_loss > target_x_array[-1]:
            raise ValueError(
                "target_x_array must include add-direction max loss "
                f"-log(1-q)={max_loss:.15g}, got right endpoint={target_x_array[-1]:.15g}"
            )

    source = _subsample_transformed_source(
        base_pld=base_pld,
        sampling_prob=sampling_prob,
        direction=direction,
    )
    grid = GridSpec(
        x_0=float(target_x_array[0]),
        step=compute_bin_width(target_x_array),
        n=int(target_x_array.size),
        spacing_type=SpacingType.LINEAR,
    )
    return project_dist_onto_grid_ctd(
        dist=source,
        grid=grid,
    )


def _calc_subsampled_grid(
    *,
    min_loss: float,
    discretization: float,
    num_buckets: int,
    sampling_prob: float,
    direction: Direction,
    include_right: float | None = None,
) -> NDArray[np.float64]:
    """Compute the transformed linear target grid used by Algorithms 8-10."""
    if sampling_prob <= 0 or sampling_prob > 1:
        raise ValueError("sampling_prob must be in (0, 1]")
    if num_buckets < 2:
        raise ValueError("num_buckets must be >= 2")
    max_loss = min_loss + num_buckets * discretization

    endpoints = np.array([min_loss, max_loss], dtype=np.float64)
    transformed_endpoints = _stable_subsampling_transformation(
        x_array=endpoints,
        sampling_prob=sampling_prob,
        direction=direction,
    )
    new_min, new_max = transformed_endpoints[0], transformed_endpoints[1]
    if not np.isfinite(new_min) or not np.isfinite(new_max) or new_max <= new_min:
        raise ValueError(
            "Subsampling transform produced invalid bounds: "
            f"new_min={new_min:.6g}, new_max={new_max:.6g}"
        )

    if include_right is not None:
        new_max = max(new_max, include_right)

    new_width = (new_max - new_min) / num_buckets
    return aligned_grid_params(
        x_min=new_min,
        x_max=new_max,
        spacing_type=SpacingType.LINEAR,
        discretization=new_width,
        align_to_multiples=True,
    ).materialize()


def _extend_target_grid_for_reference(
    *,
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


def _subsample_transformed_source(
    *,
    base_pld: DiscreteDistBase,
    sampling_prob: float,
    direction: Direction,
) -> SparseDiscreteDist:
    """Transform one subsampling branch and fold finite boundary images."""
    losses = _stable_subsampling_transformation(
        x_array=base_pld.x_array, sampling_prob=sampling_prob, direction=direction
    )
    masses = np.asarray(base_pld.prob_arr, dtype=np.float64)
    p_min = float(base_pld.p_min)
    p_max = float(base_pld.p_max)
    if direction == Direction.REMOVE and p_min > 0.0:
        losses = np.append(losses, math.log1p(-sampling_prob))
        masses = np.append(masses, p_min)
        p_min = 0.0
    if direction == Direction.ADD and p_max > 0.0:
        losses = np.append(losses, -math.log1p(-sampling_prob))
        masses = np.append(masses, p_max)
        p_max = 0.0
    return _atomic_source(losses=losses, masses=masses, p_min=p_min, p_max=p_max)


def _stable_subsampling_transformation(
    *,
    x_array: NDArray[np.float64],
    sampling_prob: float,
    direction: Direction,
) -> NDArray[np.float64]:
    """Transform privacy losses for subsampling in a stable manner.

    Remove direction: l' = log(1 + q * (exp(l) - 1))
    Add direction:    l' = -log(1 + q * (exp(-l) - 1))

    Paper mapping: Algorithm 10 (`subsam-core`), in
    Appendix C of https://arxiv.org/abs/2602.17284, with Algorithm 9
    (`PLDsubsam-add`), using this transform on ``-L`` and negating back. For
    positive losses we use a log-sum form to avoid overflow; for non-positive
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


def _atomic_source(
    *,
    losses: NDArray[np.float64],
    masses: NDArray[np.float64],
    p_min: float,
    p_max: float,
) -> SparseDiscreteDist:
    """Coalesce repeated atoms to satisfy the strictly increasing sparse support."""
    losses = np.asarray(losses, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    unique_losses, inverse = np.unique(losses, return_inverse=True)
    unique_masses = np.asarray(
        np.bincount(inverse, weights=masses, minlength=unique_losses.size), dtype=np.float64
    )
    return SparseDiscreteDist(
        x_array=unique_losses,
        prob_arr=unique_masses,
        p_min=p_min,
        p_max=p_max,
        domain=Domain.REALS,
    )
