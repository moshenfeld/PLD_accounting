"""Shared random-allocation composition helpers."""

from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.discrete_dist import DenseDiscreteDist
from PLD_accounting.distribution_discretization import rediscretize_dist
from PLD_accounting.dp_accounting_support import linear_dist_to_dp_accounting_pmf
from PLD_accounting.fft_convolution import (
    MAX_FFT_BYTES,
    fft_convolve,
    fft_self_convolve,
)
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

# Minimum number of bins to keep after per-epoch base PMF size capping.
_MIN_BASE_PMF_BINS = 4
# Estimated bytes consumed per input PMF bin in the direct epoch-compose cap.
_DIRECT_COMPOSE_BYTES_PER_BASE_BIN = 16

# =============================================================================
# Public API
# =============================================================================


def allocation_full_pld(
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
    ``allocation_directional_pld(...)`` and then converts them to the final
    ``dp_accounting`` PLD object.
    """
    # Input validation
    validate_allocation_params(num_steps, num_selected, num_epochs)
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)

    remove_dist = allocation_directional_pld(
        compute_base_pld=compute_base_pld_remove,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    add_dist = allocation_directional_pld(
        compute_base_pld=compute_base_pld_add,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )
    return _compose_full_pld(
        remove_dist=remove_dist,
        add_dist=add_dist,
        bound_type=bound_type,
    )


def allocation_directional_pld(
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
    ``_allocation_directional_pld_core(...)`` and combines them with one final
    ``fft_convolve(...)``.
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
    # Loss: fft_convolve preserves the grid step, so only the component builds add
    # rounding error.  Each component receives loss_discretization / component_count,
    # and all components together account for the full budget:
    #   component_count * (loss_discretization / component_count) = loss_discretization.
    # Tail: active tail-consuming ops = one _allocation_directional_pld_core per component
    # plus one fft_convolve when both components are active, giving 2*component_count - 1
    # ops total (1 for component_count=1, 3 for component_count=2).  After
    # tail_truncation /= (2*component_count - 1), each op consumes at most the rescaled
    # budget, and all ops together sum to <= (2*component_count - 1) * rescaled = tail_truncation.
    component_count = int(new_num_epochs_floor > 0) + int(new_num_epochs_ceil > 0)
    tail_truncation /= 2 * component_count - 1
    loss_discretization_component = loss_discretization / component_count
    # _allocation_directional_pld_core produces
    # output discretization = loss_discretization / (2*num_epochs).
    # When both floor and ceil are active their num_epochs differ, so scale each component's
    # loss budget proportionally so both land on the same output step (required for fft_convolve).
    if new_num_epochs_floor > 0 and new_num_epochs_ceil > 0:
        _max_epochs = max(new_num_epochs_floor, new_num_epochs_ceil)
        loss_disc_floor = loss_discretization_component * new_num_epochs_floor / _max_epochs
        loss_disc_ceil = loss_discretization_component * new_num_epochs_ceil / _max_epochs
    else:
        loss_disc_floor = loss_discretization_component
        loss_disc_ceil = loss_discretization_component

    dist_floor = None
    dist_ceil = None
    if new_num_epochs_floor > 0:
        dist_floor = _allocation_directional_pld_core(
            compute_base_pld=compute_base_pld,
            num_steps=new_num_steps_floor,
            num_epochs=new_num_epochs_floor,
            loss_discretization=loss_disc_floor,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    if new_num_epochs_ceil > 0:
        dist_ceil = _allocation_directional_pld_core(
            compute_base_pld=compute_base_pld,
            num_steps=new_num_steps_ceil,
            num_epochs=new_num_epochs_ceil,
            loss_discretization=loss_disc_ceil,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )

    if dist_floor is None:
        if dist_ceil is None:
            raise RuntimeError(
                "allocation_directional_pld failed to build either floor or ceil component"
            )
        return dist_ceil
    if dist_ceil is None:
        return dist_floor
    # Fallback: max_grid coarsening inside compute_base_pld can still produce
    # unequal steps even after pre-scaling.  Align to the coarser of the two.
    if dist_floor.step != dist_ceil.step:
        floor_step_before = dist_floor.step
        ceil_step_before = dist_ceil.step
        floor_size_before = dist_floor.prob_arr.size
        ceil_size_before = dist_ceil.prob_arr.size
        target_step = max(dist_floor.step, dist_ceil.step)
        if dist_floor.step < target_step:
            dist_floor = rediscretize_dist(
                dist=dist_floor,
                tail_truncation=tail_truncation,
                loss_discretization=target_step,
                spacing_type=SpacingType.LINEAR,
                bound_type=bound_type,
            )
        else:
            dist_ceil = rediscretize_dist(
                dist=dist_ceil,
                tail_truncation=tail_truncation,
                loss_discretization=target_step,
                spacing_type=SpacingType.LINEAR,
                bound_type=bound_type,
            )
        warnings.warn(
            "allocation_directional_pld: aligning mismatched floor/ceil grids. "
            f"floor size {floor_size_before}->{dist_floor.prob_arr.size}, "
            f"step {floor_step_before:.6e}->{dist_floor.step:.6e}; "
            f"ceil size {ceil_size_before}->{dist_ceil.prob_arr.size}, "
            f"step {ceil_step_before:.6e}->{dist_ceil.step:.6e}; "
            f"target_step={target_step:.6e}"
        )
    return fft_convolve(
        dist_1=dist_floor,
        dist_2=dist_ceil,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )


def geometric_allocation_pld_base_remove(
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
    # For num_steps > 1 there are active convolution stages beyond base construction.
    # For num_steps == 1 neither convolution stages nor Phases 2/3 execute, so no
    # division is needed for either budget.
    if num_steps > 1:
        # Loss: divide by the total stage count so each stage contributes at most
        # loss_discretization / num_stages rounding error, and all stages together sum to:
        #   remove_loss_stages * (loss_discretization / remove_loss_stages) = loss_discretization.
        # Stages: 2 for base + neg-dual factor creation
        #         + _binary_self_convolution_call_count(num_steps - 1) for geometric_self_convolve
        #         + 1 for the final geometric_convolve(neg_dual_convolved, base).
        remove_loss_stages = 3 + _binary_self_convolution_call_count(num_steps - 1)
        loss_discretization /= remove_loss_stages
        # Tail: three phases each receive tail_truncation / 3 after division.  Contributions
        # to output p_max (letting T = num_steps, budget/3 = rescaled per-phase share):
        #   Phase 1 (base_distributions_creation): base_factor_tail_truncation = budget/(3T).
        #            neg_dual is self-convolved T-1 times, base convolved once; total amplified
        #            contribution = ((T-1) + 1) * budget/(3T) = budget/3.
        #   Phase 2 (geometric_self_convolve, T-1): called with budget/3 -> <= budget/3.
        #   Phase 3 (geometric_convolve): called with budget/3 -> <= budget/3.
        # Sum <= budget/3 + budget/3 + budget/3 = budget
        tail_truncation /= 3
    # Each base factor is one of num_steps terms in the final product; its individual
    # tail error is amplified by num_steps through self-convolution, so scale its budget
    # down by num_steps so the amplified contribution stays within the phase budget.
    base_factor_tail_truncation = tail_truncation / num_steps

    base, neg_dual_base = base_distributions_creation(
        loss_discretization=loss_discretization,
        tail_truncation=base_factor_tail_truncation,
        bound_type=bound_type,
    )

    # For num_steps == 1 the centering shift is log(1) = 0 and the exp/log round-trip
    # is an identity, so base is already the final result.
    if num_steps == 1:
        return base

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


def geometric_allocation_pld_base_add(
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

    # For num_steps > 1 there are active convolution stages beyond base construction.
    # For num_steps == 1 neither convolution stages nor Phase 2 execute, so no
    # division is needed for either budget.
    if num_steps > 1:
        # Loss: divide by the total stage count so each stage contributes at most
        # loss_discretization / num_stages rounding error, and all stages together sum to:
        #   add_loss_stages * (loss_discretization / add_loss_stages) = loss_discretization.
        # Stages: 1 for base-factor creation + _binary_self_convolution_call_count(num_steps)
        #         squarings/multiplies in the self-convolution.
        add_loss_stages = 1 + _binary_self_convolution_call_count(num_steps)
        loss_discretization /= add_loss_stages
        # Tail: after dividing by 2, each of the two phases receives tail_truncation:
        #   Phase 1 (base creation): tail_truncation / num_steps per call; num_steps-fold
        #            self-convolution amplifies the contribution back to tail_truncation.
        #   Phase 2 (geometric_self_convolve, num_steps): tail_truncation directly.
        # Sum: tail_truncation + tail_truncation = 2 * tail_truncation = original ✓
        tail_truncation /= 2
    # Each base factor's tail error is amplified by num_steps through self-convolution,
    # so scale its budget down by num_steps so the amplified contribution stays within the
    # phase budget.
    base_factor_tail_truncation = tail_truncation / num_steps

    base = base_distributions_creation(
        loss_discretization=loss_discretization,
        tail_truncation=base_factor_tail_truncation,
        bound_type=bound_type,
    )

    # For num_steps == 1 the centering shift is log(1) = 0 and the exp/log round-trip
    # is an identity, so base is already the final result.
    if num_steps == 1:
        return base

    log_num_steps = float(np.log(num_steps))

    neg_base = negate_reverse_linear_distribution(base)
    centered_neg_base = DenseDiscreteDist(
        x_min=neg_base.x_min - log_num_steps,
        step=neg_base.step,
        prob_arr=neg_base.prob_arr.copy(),
        p_min=neg_base.p_min,
        p_max=neg_base.p_max,
    )

    # Factor preparation in exp-space.
    exp_base = exp_linear_to_geometric(centered_neg_base)
    exp_bound_type = (
        BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    )
    # U_t <- self-conv(U, t, lower).
    exp_convolved = geometric_self_convolve(
        dist=exp_base,
        T=num_steps,
        tail_truncation=tail_truncation,
        bound_type=exp_bound_type,
    )
    # L_t <- -log(U_t).
    log_dist = log_geometric_to_linear(exp_convolved)
    return negate_reverse_linear_distribution(log_dist)


# =============================================================================
# Helper Functions
# =============================================================================


def _binary_self_convolution_call_count(num_convolutions: int) -> int:
    """Return exact pairwise-convolution calls used by binary self-compose."""
    if num_convolutions <= 1:
        return 0
    return int(np.floor(np.log2(num_convolutions)) + int(num_convolutions).bit_count() - 1)


def _allocation_directional_pld_core(
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
    ``compute_base_pld(...)``, trims tails before and after epoch
    composition, and returns the resulting linear distribution.
    """
    # Tail: divide by 3; each of the three phases receives tail_truncation (the divided value):
    #   Phase 1 (base creation, amplified by num_epochs): up to 3 sub-ops per epoch
    #            (compute_base_pld, truncate_edges, optional rediscretize_dist), each with
    #            base_tail_truncation = tail_truncation / (3 * num_epochs).  Amplified total:
    #            num_epochs * 3 * tail_truncation / (3 * num_epochs) = tail_truncation.
    #   Phase 2 (fft_self_convolve, T=num_epochs): tail_truncation directly
    #            (skipped if num_epochs=1).
    #   Phase 3 (final truncate_edges): tail_truncation directly.
    # Sum: tail_truncation + tail_truncation + tail_truncation = 3 * tail_truncation = original ✓
    tail_truncation /= 3
    base_tail_truncation = tail_truncation / (3 * num_epochs)
    # Loss: FFT self-convolution preserves the grid step, so the output step equals the base
    # step.  Dividing by num_epochs ensures the output step <= loss_discretization; the extra
    # factor of 2 absorbs Chernoff window boundary rounding in the fft_self_convolve direct path.
    base_loss_discretization = loss_discretization / (2 * num_epochs)

    base_dist = compute_base_pld(
        num_steps=num_steps,
        loss_discretization=base_loss_discretization,
        tail_truncation=base_tail_truncation,
        bound_type=bound_type,
    )
    prepared_base_dist = base_dist.truncate_edges(
        tail_truncation=base_tail_truncation,
        bound_type=bound_type,
    )
    if not (
        isinstance(prepared_base_dist, DenseDiscreteDist)
        and prepared_base_dist.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(prepared_base_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(prepared_base_dist).__name__} with spacing {_st}"
        )

    if num_epochs == 1:
        composed_dist = prepared_base_dist
    else:
        # Cap base distribution so fft_self_convolve(T=num_epochs) stays within
        # MAX_FFT_BYTES.  For the direct method the FFT size is T * pmf_size;
        # for the binary method the worst-case FFT size is also T * pmf_size
        # (before truncation shrinks intermediate steps).
        max_base_pmf = max(
            _MIN_BASE_PMF_BINS,
            MAX_FFT_BYTES // (num_epochs * _DIRECT_COMPOSE_BYTES_PER_BASE_BIN),
        )
        if prepared_base_dist.prob_arr.size > max_base_pmf:
            capped_step = prepared_base_dist.step * math.ceil(
                prepared_base_dist.prob_arr.size / max_base_pmf
            )
            prepared_base_dist = rediscretize_dist(
                dist=prepared_base_dist,
                tail_truncation=base_tail_truncation,
                loss_discretization=capped_step,
                spacing_type=SpacingType.LINEAR,
                bound_type=bound_type,
            )
        composed_dist = fft_self_convolve(
            dist=prepared_base_dist,
            T=num_epochs,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            use_direct=True,
        )
    final_dist = composed_dist.truncate_edges(
        tail_truncation=tail_truncation,
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


def _compose_full_pld(
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
