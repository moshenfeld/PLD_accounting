"""Shared random-allocation composition helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.discrete_dist import DenseDiscreteDist
from PLD_accounting.distribution_discretization import rediscritize_dist
from PLD_accounting.dp_accounting_support import linear_dist_to_dp_accounting_pmf
from PLD_accounting.FFT_convolution import FFT_convolve, FFT_self_convolve
from PLD_accounting.geometric_convolution import (
    geometric_convolve,
    geometric_self_convolve,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import (
    exp_linear_to_geometric,
    log_geometric_to_linear,
    negate_reverse_linear_distribution,
)
from PLD_accounting.validation import (
    validate_allocation_params,
    validate_bound_type,
    validate_discretization_params,
)

# =============================================================================
# Public API
# =============================================================================


def allocation_full_PLD(
    *,
    compute_base_pld_remove: Callable[..., DenseDiscreteDist],
    compute_base_pld_add: Callable[..., DenseDiscreteDist],
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Orchestrate full allocation PLD construction for both directions.

    This function builds REMOVE and ADD directional PLDs via
    ``allocation_directional_PLD(...)`` and then converts them to the final
    ``dp_accounting`` PLD object.
    """
    # Input validation
    validate_allocation_params(num_steps, num_selected, num_epochs)
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)

    remove_dist = allocation_directional_PLD(
        compute_base_pld=compute_base_pld_remove,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    add_dist = allocation_directional_PLD(
        compute_base_pld=compute_base_pld_add,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    return _compose_full_PLD(
        remove_dist=remove_dist,
        add_dist=add_dist,
        bound_type=bound_type,
    )


def allocation_directional_PLD(
    *,
    compute_base_pld: Callable[..., DenseDiscreteDist],
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Build one-direction allocation PLD with adaptive floor/ceil decomposition.

    For divisible ``num_steps / num_selected``, this builds one component. For
    non-divisible cases, it builds floor and ceil components via
    ``_allocation_directional_PLD_core(...)`` and combines them with one final
    ``FFT_convolve(...)``.
    """
    # Input validation
    validate_allocation_params(num_steps, num_selected, num_epochs)
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)
    new_num_steps_floor = int(num_steps // num_selected)
    if new_num_steps_floor < 1:
        raise ValueError("num_steps must be >= num_selected")
    num_epochs_remainder = num_steps - num_selected * new_num_steps_floor
    new_num_steps_ceil = new_num_steps_floor + 1
    new_num_epochs_floor = (num_selected - num_epochs_remainder) * num_epochs
    new_num_epochs_ceil = num_epochs_remainder * num_epochs
    # Tail budget is applied twice (floor and ceil distributions).
    tail_truncation /= 2

    dist_floor = None
    dist_ceil = None
    if new_num_epochs_floor > 0:
        dist_floor = _allocation_directional_PLD_core(
            compute_base_pld=compute_base_pld,
            num_steps=new_num_steps_floor,
            num_epochs=new_num_epochs_floor,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    if new_num_epochs_ceil > 0:
        dist_ceil = _allocation_directional_PLD_core(
            compute_base_pld=compute_base_pld,
            num_steps=new_num_steps_ceil,
            num_epochs=new_num_epochs_ceil,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )

    if dist_floor is None:
        if dist_ceil is None:
            raise RuntimeError(
                "allocation_directional_PLD failed to build either floor or ceil component"
            )
        return dist_ceil
    if dist_ceil is None:
        return dist_floor
    return FFT_convolve(
        dist_1=dist_floor,
        dist_2=dist_ceil,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )


def geometric_allocation_PLD_base_remove(
    *,
    base_distributions_creation: Callable[..., tuple[DenseDiscreteDist, DenseDiscreteDist]],
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Build the REMOVE component PLD via exp-space geometric composition.

    The callback ``base_distributions_creation`` provides one-step
    ``(base, neg_dual_base)`` factors, which are shifted and composed.
    """
    # Input validation
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)
    # Loss grid is refined across ~2*ceil(log2(num_steps)) geometric-convolution stages;
    # split discretization across them and the initial discretization.
    loss_discretization /= int(2 * np.ceil(np.log2(num_steps)) + 1)
    # Tail mass is charged at base construction, self-convolution, and the final geometric convolve.
    tail_truncation /= 3
    base_factor_tail_truncation = tail_truncation / num_steps

    base, neg_dual_base = base_distributions_creation(
        loss_discretization=loss_discretization,
        tail_truncation=base_factor_tail_truncation,
        bound_type=bound_type,
    )

    # Subtract the average loss
    log_num_steps = float(np.log(num_steps))
    centered_neg_dual = DenseDiscreteDist(
        x_min=neg_dual_base.x_min - log_num_steps,
        step=neg_dual_base.step,
        prob_arr=neg_dual_base.prob_arr.copy(),
        p_min=neg_dual_base.p_min,
        p_max=neg_dual_base.p_max,
    )
    centered_base = DenseDiscreteDist(
        x_min=base.x_min - log_num_steps,
        step=base.step,
        prob_arr=base.prob_arr.copy(),
        p_min=base.p_min,
        p_max=base.p_max,
    )

    # Factor preparation in exp-space.
    exp_neg_dual = exp_linear_to_geometric(centered_neg_dual)
    exp_base = exp_linear_to_geometric(centered_base)

    if num_steps == 1:
        exp_convolved = exp_base
    else:
        # V_{t-1} <- self-conv(V1, t-1, ...).
        exp_convolved_dual = geometric_self_convolve(
            dist=exp_neg_dual,
            T=num_steps - 1,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
        # U_t <- conv(V_{t-1}, U1, ...).
        exp_convolved = geometric_convolve(
            dist_1=exp_convolved_dual,
            dist_2=exp_base,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    # L_t <- log(U_t).
    return log_geometric_to_linear(exp_convolved)


def geometric_allocation_PLD_base_add(
    *,
    base_distributions_creation: Callable[..., DenseDiscreteDist],
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Build the ADD component PLD via exp-space geometric self-composition.

    The callback ``base_distributions_creation`` provides the one-step ADD
    factor, which is shifted and composed before mapping back to linear loss.
    """
    # Input validation
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")

    # Loss grid is refined across ~2*ceil(log2(num_steps)) geometric-convolution stages;
    # split discretization across them and the initial discretization.
    loss_discretization /= int(2 * np.ceil(np.log2(num_steps)) + 1)
    # Tail mass is charged at base construction and self-convolution.
    tail_truncation /= 2
    base_factor_tail_truncation = tail_truncation / num_steps

    base = base_distributions_creation(
        loss_discretization=loss_discretization,
        tail_truncation=base_factor_tail_truncation,
        bound_type=bound_type,
    )

    log_num_steps = float(np.log(num_steps))

    centered_base = DenseDiscreteDist(
        x_min=base.x_min - log_num_steps,
        step=base.step,
        prob_arr=base.prob_arr.copy(),
        p_min=base.p_min,
        p_max=base.p_max,
    )

    # Factor preparation in exp-space.
    exp_base = exp_linear_to_geometric(centered_base)
    exp_bound_type = (
        BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    )
    if num_steps == 1:
        exp_convolved = exp_base
    else:
        # U_t <- self-conv(U, t, lower).
        # Self-convolution applies tail along a binary tree; halve again for the inner
        # combine vs outer structure.
        exp_convolved = geometric_self_convolve(
            dist=exp_base,
            T=num_steps,
            tail_truncation=tail_truncation / 2,
            bound_type=exp_bound_type,
        )
    # L_t <- -log(U_t).
    log_dist = log_geometric_to_linear(exp_convolved)
    return negate_reverse_linear_distribution(log_dist)


# =============================================================================
# Helper Functions
# =============================================================================


def _allocation_directional_PLD_core(
    *,
    compute_base_pld: Callable[..., DenseDiscreteDist],
    num_steps: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Build and finalize one floor/ceil decomposition component.

    This function derives component-level budgets, calls
    ``compute_base_pld(...)``, regrids to linear spacing, composes across
    epochs, and aligns to output discretization.
    """
    # Incoming tail and loss budgets are each split across three phases: base PLD,
    # optional epoch self-conv, final regrid.
    output_tail_truncation = tail_truncation / 3
    # Base tail is split across the base computation and potential rediscretization,
    # and divided by the number of compositions.
    base_tail_truncation = output_tail_truncation / (2 * num_epochs)
    output_loss_discretization = loss_discretization / 3
    # Per-epoch loss grid tightens like 1/sqrt(epochs) when composing IID epoch blocks.
    base_loss_discretization = output_loss_discretization / np.sqrt(num_epochs)

    base_dist = compute_base_pld(
        num_steps=num_steps,
        loss_discretization=base_loss_discretization,
        tail_truncation=base_tail_truncation,
        bound_type=bound_type,
    )
    rediscritized_dist = rediscritize_dist(
        dist=base_dist,
        tail_truncation=base_tail_truncation,
        loss_discretization=base_loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    if not (
        isinstance(rediscritized_dist, DenseDiscreteDist)
        and rediscritized_dist.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(rediscritized_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(rediscritized_dist).__name__} with spacing {_st}"
        )

    if num_epochs == 1:
        composed_dist = rediscritized_dist
    else:
        composed_dist = FFT_self_convolve(
            dist=rediscritized_dist,
            T=num_epochs,
            tail_truncation=output_tail_truncation,
            bound_type=bound_type,
            use_direct=True,
        )
    # Avoid inflating the grid when the target is finer than the original one.
    effective_disc = max(composed_dist.step, output_loss_discretization)
    final_dist = rediscritize_dist(
        dist=composed_dist,
        tail_truncation=output_tail_truncation,
        loss_discretization=effective_disc,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    if not (
        isinstance(final_dist, DenseDiscreteDist) and final_dist.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(final_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(final_dist).__name__} with spacing {_st}"
        )
    return final_dist


def _compose_full_PLD(
    *,
    remove_dist: DenseDiscreteDist | None,
    add_dist: DenseDiscreteDist | None,
    bound_type: BoundType,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Convert remove/add directional PLDs into a ``dp_accounting`` PLD.

    Args:
        remove_dist: REMOVE-direction linear PLD.
        add_dist: Optional ADD-direction linear PLD.
        bound_type: Bound direction used for pessimistic conversion.

    Returns:
        A ``dp_accounting`` privacy loss distribution.

    """
    if remove_dist is None:
        raise ValueError(
            "PLD construction requires remove-direction distribution. "
            "Provide remove_realization or use both directions."
        )
    pessimistic_estimate = bound_type == BoundType.DOMINATES
    pmf_remove = linear_dist_to_dp_accounting_pmf(
        dist=remove_dist,
        pessimistic_estimate=pessimistic_estimate,
    )
    if add_dist is None:
        return privacy_loss_distribution.PrivacyLossDistribution(
            pmf_remove=pmf_remove,
        )
    pmf_add = linear_dist_to_dp_accounting_pmf(
        dist=add_dist,
        pessimistic_estimate=pessimistic_estimate,
    )
    return privacy_loss_distribution.PrivacyLossDistribution(
        pmf_remove=pmf_remove,
        pmf_add=pmf_add,
    )
