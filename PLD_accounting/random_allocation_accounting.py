"""Shared random-allocation composition helpers."""

from __future__ import annotations

import math
import warnings
from typing import Callable

import numpy as np
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.discrete_dist import DenseDiscreteDist
from PLD_accounting.distribution_discretization import rediscretize_dist
from PLD_accounting.distribution_utils import stable_isclose
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


def allocation_directional_pld(
    *,
    compute_base_pld: Callable[..., DenseDiscreteDist],
    base_loss_discretization_count: Callable[[int], int],
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
    # Floor/ceil decomposition: non-divisible num_steps / num_selected splits into
    # rounds of floor(num_steps / num_selected) steps and rounds of one more step,
    # with per-round multiplicities scaled by num_epochs.
    new_num_steps_floor = int(num_steps // num_selected)
    if new_num_steps_floor < 1:
        raise ValueError("num_steps must be >= num_selected")
    num_epochs_remainder = num_steps - num_selected * new_num_steps_floor
    new_num_steps_ceil = new_num_steps_floor + 1
    new_num_epochs_floor = (num_selected - num_epochs_remainder) * num_epochs
    new_num_epochs_ceil = num_epochs_remainder * num_epochs
    # Tail: active tail-consuming ops = one _allocation_directional_pld_core per component
    # plus one fft_convolve when both components are active, giving 2*component_count - 1
    # ops total (1 for component_count=1, 3 for component_count=2).  After
    # tail_truncation /= (2*component_count - 1), each op consumes at most the rescaled
    # budget, and all ops together sum to <= (2*component_count - 1) * rescaled = tail_truncation.
    component_count = int(new_num_epochs_floor > 0) + int(new_num_epochs_ceil > 0)
    tail_truncation /= 2 * component_count - 1
    # Loss: when both components are active, split the budget proportional to each
    # component's effective discretization count (num_epochs * base count).  This is
    # intentional: each component's final step is its loss budget divided by its
    # discretization count, so a proportional split makes the floor/ceil output steps
    # exactly equal -- loss_discretization / (floor_count + ceil_count) for both --
    # even when the base counts differ (GEOM counts depend on num_steps), which the
    # final fft_convolve requires.  The components' budgets still sum to
    # loss_discretization, so the total rounding error stays within budget.
    if new_num_epochs_floor > 0 and new_num_epochs_ceil > 0:
        floor_count = new_num_epochs_floor * base_loss_discretization_count(new_num_steps_floor)
        ceil_count = new_num_epochs_ceil * base_loss_discretization_count(new_num_steps_ceil)
        loss_disc_floor = loss_discretization * floor_count / (floor_count + ceil_count)
        loss_disc_ceil = loss_discretization * ceil_count / (floor_count + ceil_count)
    else:
        loss_disc_floor = loss_discretization
        loss_disc_ceil = loss_discretization

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
    dist_floor, dist_ceil = _align_component_grids(
        dist_floor=dist_floor,
        dist_ceil=dist_ceil,
        bound_type=bound_type,
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
    # For num_steps == 1 neither convolution stages nor Phases 2/3 execute, so no
    # division is needed for either budget.
    if num_steps > 1:
        # Loss: divide by the discretizing stage count (see the count function) so all
        # stages together stay within loss_discretization.
        loss_discretization /= remove_geometric_loss_discretization_count(num_steps)
        # Tail: three phases each receive tail_truncation / 3:
        #   Phase 1 (base_distributions_creation): per-factor budget /(3T); the dual is
        #            self-convolved T-1 times and the base convolved once, amplifying the
        #            total contribution back to budget/3.
        #   Phase 2 (geometric_self_convolve, T-1) and Phase 3 (geometric_convolve):
        #            budget/3 each.
        tail_truncation /= 3
    # Each base factor's tail error is amplified by num_steps through self-convolution,
    # so scale its budget down by num_steps to stay within the phase budget.
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

    # Normalize each factor by num_steps before moving to exp-space.
    log_num_steps = float(np.log(num_steps))
    centered_neg_dual = DenseDiscreteDist(
        x_0=neg_dual_base.x_0 - log_num_steps,
        step=neg_dual_base.step,
        prob_arr=neg_dual_base.prob_arr.copy(),
        p_min=neg_dual_base.p_min,
        p_max=neg_dual_base.p_max,
    )
    centered_base = DenseDiscreteDist(
        x_0=base.x_0 - log_num_steps,
        step=base.step,
        prob_arr=base.prob_arr.copy(),
        p_min=base.p_min,
        p_max=base.p_max,
    )

    # Factor preparation in exp-space.
    exp_neg_dual = exp_linear_to_geometric(centered_neg_dual)
    exp_base = exp_linear_to_geometric(centered_base)
    factor_anchor = 1.0 / num_steps

    # V_{t-1} <- self-conv(V1, t-1, ...).
    exp_convolved_dual = geometric_self_convolve(
        dist=exp_neg_dual,
        T=num_steps - 1,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        lattice_anchor=factor_anchor,
    )
    # U_t <- conv(V_{t-1}, U1, ...). The T normalized factors sum to anchor 1.
    exp_convolved = geometric_convolve(
        dist_1=exp_convolved_dual,
        dist_2=exp_base,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        target_anchor=1.0,
    )
    # The composed anchor is one, so the log-grid is aligned to zero loss.
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
    # For num_steps == 1 neither convolution stages nor Phase 2 execute, so no
    # division is needed for either budget.
    if num_steps > 1:
        # Loss: divide by the discretizing stage count (see the count function) so all
        # stages together stay within loss_discretization.
        loss_discretization /= add_geometric_loss_discretization_count(num_steps)
        # Tail: two phases each receive tail_truncation / 2:
        #   Phase 1 (base creation): per-factor budget /(2*num_steps); num_steps-fold
        #            self-convolution amplifies the contribution back to budget/2.
        #   Phase 2 (geometric_self_convolve, num_steps): budget/2 directly.
        tail_truncation /= 2
    # Each base factor's tail error is amplified by num_steps through self-convolution,
    # so scale its budget down by num_steps to stay within the phase budget.
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

    neg_base = negate_reverse_linear_distribution(base)
    log_num_steps = float(np.log(num_steps))
    centered_neg_base = DenseDiscreteDist(
        x_0=neg_base.x_0 - log_num_steps,
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
        lattice_anchor=1.0 / num_steps,
    )
    # The composed anchor is one, so the log-grid is aligned to zero loss.
    log_dist = log_geometric_to_linear(exp_convolved)
    return negate_reverse_linear_distribution(log_dist)


def compose_full_pld(
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


def remove_geometric_loss_discretization_count(num_steps: int) -> int:
    """Return the effective GEOM REMOVE loss-discretization count.

    One base discretization, (num_steps-1) convolutions of the dual via
    exponentiation by squaring, and a final convolution with the base.
    """
    if num_steps <= 1:
        return 1
    return 2 + _binary_self_convolution_call_count(num_steps - 1)


def add_geometric_loss_discretization_count(num_steps: int) -> int:
    """Return the effective GEOM ADD loss-discretization count.

    One base discretization and num_steps convolutions via exponentiation by squaring.
    """
    return 1 + _binary_self_convolution_call_count(num_steps)


# =============================================================================
# Helper Functions
# =============================================================================


def _binary_self_convolution_call_count(num_convolutions: int) -> int:
    """Return exact pairwise-convolution calls used by binary self-compose."""
    if num_convolutions <= 1:
        return 0
    return int(np.floor(np.log2(num_convolutions)) + int(num_convolutions).bit_count() - 1)


def _align_component_grids(
    *,
    dist_floor: DenseDiscreteDist,
    dist_ceil: DenseDiscreteDist,
    bound_type: BoundType,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Fallback-align floor/ceil grids before the final ``fft_convolve``.

    Proportional budget pre-scaling normally yields equal steps, but max_grid
    coarsening inside ``compute_base_pld`` can still produce unequal ones.
    Align to the coarser of the two and warn.
    """
    if stable_isclose(a=dist_floor.step, b=dist_ceil.step):
        return dist_floor, dist_ceil
    floor_step_before = dist_floor.step
    ceil_step_before = dist_ceil.step
    floor_size_before = dist_floor.prob_arr.size
    ceil_size_before = dist_ceil.prob_arr.size
    target_step = max(dist_floor.step, dist_ceil.step)
    if dist_floor.step < target_step:
        dist_floor = rediscretize_dist(
            dist=dist_floor,
            # Alignment is a pure directional projection; tail mass was already
            # budgeted by the component builders and must not be spent again.
            tail_truncation=0.0,
            loss_discretization=target_step,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
    else:
        dist_ceil = rediscretize_dist(
            dist=dist_ceil,
            tail_truncation=0.0,
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
    return dist_floor, dist_ceil


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
    if num_epochs > 1:
        # Tail: divide by 3; each of the three phases receives tail_truncation (the divided value):
        #   Phase 1 (base creation, amplified by num_epochs): up to 3 sub-ops per epoch
        #            (compute_base_pld, truncate_edges, optional rediscretize_dist), each with
        #            base_tail_truncation = tail_truncation / (3 * num_epochs).  Amplified total:
        #            num_epochs * 3 * tail_truncation / (3 * num_epochs) = tail_truncation.
        #   Phase 2 (fft_self_convolve, T=num_epochs): tail_truncation directly.
        #   Phase 3 (final truncate_edges): tail_truncation directly.
        tail_truncation /= 3
        # Loss: T-fold convolution of a base distribution with step s accumulates quantization
        # error of at most num_epochs * s, so divide by num_epochs.  FFT self-convolution is
        # algebraically exact and introduces no additional rounding.
        base_loss_discretization = loss_discretization / num_epochs
    else:
        # num_epochs == 1: fft_self_convolve is SKIPPED, so only Phases 1 and 3 are active;
        # divide the tail budget by 2 instead, and there is no per-epoch loss accumulation.
        tail_truncation /= 2
        base_loss_discretization = loss_discretization
    base_tail_truncation = tail_truncation / (3 * num_epochs)

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
