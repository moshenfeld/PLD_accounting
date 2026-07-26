"""Realization-specific random-allocation accounting."""

from __future__ import annotations

from PLD_accounting.discrete_dist import DenseDiscreteDist, PLDRealization
from PLD_accounting.distribution_discretization import (
    rediscretize_dist_by_bound,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import calc_pld_dual, negate_reverse_linear_distribution


def realization_remove_base_distributions(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    max_grid_mult: int = -1,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Prepare remove-direction factors from a loss-space realization.

    Algorithm 1 (`rand-alloc-rem`), with the dominating-path refinement in the
    "Connect-the-dots rediscretization" remark following Algorithm 6, in
    Appendix C of https://arxiv.org/abs/2602.17284.

    Args:
        realization: REMOVE-direction realization in linear loss space.
        loss_discretization: Target linear-grid spacing.
        tail_truncation: Tail truncation budget for regridding.
        bound_type: Bound direction.
        max_grid_mult: Upper bound on the number of base-factor grid points
            (``<= 0`` disables the cap); coarsens the effective step when the
            target spacing would exceed it.

    Returns:
        Tuple ``(base, dual_base)`` with the requested linear-grid spacing.

    """
    # Since dual can be derived only from a PLD realization, discretization can
    # come first for DOMINATES, but dual derivation must come first for IS_DOMINATED.
    # As described in the paper's CtD implementation remark, the DOMINATES path
    # discretizes first and takes the exact dual of the discretized base, avoiding
    # a second discretization of the dual while preserving a valid upper bound.
    effective_disc = _realization_geom_loss_discretization(
        realization=realization,
        loss_discretization=loss_discretization,
        max_grid_mult=max_grid_mult,
    )

    if bound_type == BoundType.DOMINATES:
        base_realization = rediscretize_dist_by_bound(
            dist=realization,
            tail_truncation=tail_truncation,
            loss_discretization=effective_disc,
            bound_type=bound_type,
        )
        if not isinstance(base_realization, PLDRealization):
            raise TypeError("Dominating linear rediscretization must return PLDRealization")
        neg_dual_dist = negate_reverse_linear_distribution(calc_pld_dual(base_realization))
        return base_realization, neg_dual_dist

    # Lower-bound truncation can move left-tail mass into p_min and must consume
    # any +inf mass before exp-space composition, so keep the lower path on the
    # plain DenseDiscreteDist rediscretization route unconditionally.
    dual_realization = calc_pld_dual(realization)
    neg_dual_linear = negate_reverse_linear_distribution(dual_realization)
    lower_realization_input = DenseDiscreteDist(
        x_0=realization.x_0,
        step=realization.step,
        prob_arr=realization.prob_arr.copy(),
        p_min=realization.p_min,
        p_max=realization.p_max,
    )
    lower_base_dist = rediscretize_dist_by_bound(
        dist=lower_realization_input,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        bound_type=bound_type,
    )
    neg_dual_dist = rediscretize_dist_by_bound(
        dist=neg_dual_linear,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        bound_type=bound_type,
    )
    if not (
        isinstance(neg_dual_dist, DenseDiscreteDist)
        and neg_dual_dist.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(neg_dual_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(neg_dual_dist).__name__} with spacing {_st}"
        )
    return lower_base_dist, neg_dual_dist


def realization_add_base_distribution(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    max_grid_mult: int = -1,
) -> DenseDiscreteDist:
    """Prepare add-direction factors from a loss-space realization.

    Algorithm 2 (`rand-alloc-add`), in Appendix C of https://arxiv.org/abs/2602.17284.

    Args:
        realization: ADD-direction realization in linear loss space.
        loss_discretization: Target linear-grid spacing.
        tail_truncation: Tail truncation budget for regridding.
        bound_type: Bound direction.
        max_grid_mult: Upper bound on the number of base-factor grid points
            (``<= 0`` disables the cap); coarsens the effective step when the
            target spacing would exceed it.

    Returns:
        One ADD loss factor aligned to the requested linear grid.

    """
    # Avoid inflating the grid when the target is finer than the original one.
    effective_disc = _realization_geom_loss_discretization(
        realization=realization,
        loss_discretization=loss_discretization,
        max_grid_mult=max_grid_mult,
    )
    coarsened = rediscretize_dist_by_bound(
        dist=realization,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        bound_type=bound_type,
    )
    if not (
        isinstance(coarsened, DenseDiscreteDist) and coarsened.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(coarsened, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(coarsened).__name__} with spacing {_st}"
        )
    return coarsened


def _realization_geom_loss_discretization(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    max_grid_mult: int,
) -> float:
    """Return the realization GEOM loss step for the configured grid cap."""
    grid_size = realization.prob_arr.size
    # Reserve two points for outward alignment when a finite cap is enabled.
    target_grid_size = min(max_grid_mult - 2, grid_size) if max_grid_mult > 0 else grid_size
    if target_grid_size <= 1:
        raise ValueError(
            "realization and max_grid_mult must leave at least two finite grid points, "
            f"got grid_size={grid_size}, max_grid_mult={max_grid_mult}"
        )
    # N support points span N-1 intervals.
    grid_size_rescaling = (grid_size - 1) / (target_grid_size - 1)
    return float(max(loss_discretization, realization.step * grid_size_rescaling))
