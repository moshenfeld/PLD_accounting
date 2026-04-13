"""Realization-specific random-allocation accounting."""

from __future__ import annotations

from PLD_accounting.discrete_dist import DenseDiscreteDist, PLDRealization
from PLD_accounting.distribution_discretization import rediscritize_dist
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import calc_pld_dual, negate_reverse_linear_distribution


def realization_remove_base_distributions(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Prepare remove-direction factors from a loss-space realization.

    Algorithm 1 (`rand-alloc-rem`) in Appendix C.

    Args:
        realization: REMOVE-direction realization in linear loss space.
        loss_discretization: Target linear-grid spacing.
        tail_truncation: Tail truncation budget for regridding.
        bound_type: Bound direction.

    Returns:
        Tuple ``(base, dual_base)`` aligned to the requested linear grid.

    """
    # Since dual can be derived only from a PLD realization, discretization can
    # come first for DOMINATES, but dual derivation must come first for IS_DOMINATED.
    if bound_type == BoundType.DOMINATES:
        # Avoid inflating the grid when the target is finer than the original one.
        effective_disc = max(realization.step, loss_discretization)
        coarsened_base = rediscritize_dist(
            dist=realization,
            tail_truncation=tail_truncation,
            loss_discretization=effective_disc,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
        if not (
            isinstance(coarsened_base, DenseDiscreteDist)
            and coarsened_base.spacing_type == SpacingType.LINEAR
        ):
            _st = getattr(coarsened_base, "spacing_type", "?")
            raise TypeError(
                "Expected DenseDiscreteDist with LINEAR spacing, "
                f"got {type(coarsened_base).__name__} with spacing {_st}"
            )
        base_realization = PLDRealization.from_linear_dist(coarsened_base)
        neg_dual_dist = negate_reverse_linear_distribution(calc_pld_dual(base_realization))
        return base_realization, neg_dual_dist

    # Lower-bound truncation can move left-tail mass into p_min and must consume
    # any +inf mass before exp-space composition, so keep the lower path on the
    # plain DenseDiscreteDist rediscretization route unconditionally.
    dual_realization = calc_pld_dual(realization)
    neg_dual_linear = negate_reverse_linear_distribution(dual_realization)
    # Avoid inflating the grid when the target is finer than the original one.
    effective_disc = max(realization.step, loss_discretization)
    lower_realization_input = DenseDiscreteDist(
        x_min=realization.x_min,
        step=realization.step,
        prob_arr=realization.prob_arr.copy(),
        p_min=realization.p_min,
        p_max=realization.p_max,
    )
    lower_base_dist = rediscritize_dist(
        dist=lower_realization_input,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    if not (
        isinstance(lower_base_dist, DenseDiscreteDist)
        and lower_base_dist.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(lower_base_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(lower_base_dist).__name__} with spacing {_st}"
        )
    neg_dual_dist = rediscritize_dist(
        dist=neg_dual_linear,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        spacing_type=SpacingType.LINEAR,
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
) -> DenseDiscreteDist:
    """Prepare add-direction factors from a loss-space realization.

    Algorithm 2 (`rand-alloc-add`) in Appendix C.

    Args:
        realization: ADD-direction realization in linear loss space.
        loss_discretization: Target linear-grid spacing.
        tail_truncation: Tail truncation budget for regridding.
        bound_type: Bound direction.

    Returns:
        One ADD loss factor aligned to the requested linear grid.

    """
    exp_bound_type = (
        BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    )
    neg_realization = negate_reverse_linear_distribution(realization)
    # Avoid inflating the grid when the target is finer than the original one.
    effective_disc = max(neg_realization.step, loss_discretization)
    neg_coarsened = rediscritize_dist(
        dist=neg_realization,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        spacing_type=SpacingType.LINEAR,
        bound_type=exp_bound_type,
    )
    if not (
        isinstance(neg_coarsened, DenseDiscreteDist)
        and neg_coarsened.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(neg_coarsened, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(neg_coarsened).__name__} with spacing {_st}"
        )
    return neg_coarsened
