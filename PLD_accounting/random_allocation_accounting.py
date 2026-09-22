"""Shared random-allocation decomposition and composition helpers.

The module owns the floor/ceil allocation split and the loss/tail budget ledger
across base construction, exp-space or FFT composition, and final combination.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Callable

import numpy as np
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.discrete_dist import DenseDiscreteDist
from PLD_accounting.distribution_discretization import (
    rediscretize_dist_by_bound,
)
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
from PLD_accounting.types import BoundType, SpacingType, require_bound_type
from PLD_accounting.utils import (
    exp_linear_to_geometric,
    log_geometric_to_linear,
    negate_reverse_linear_distribution,
)
from PLD_accounting.validation import (
    require_allocation_counts,
    require_nonnegative_real,
    require_positive_int,
    require_positive_real,
)

# Minimum number of bins to keep after per-epoch base PMF size capping.
_MIN_BASE_PMF_BINS = 4
# Estimated bytes consumed per input PMF bin in the direct epoch-compose cap.
_DIRECT_COMPOSE_BYTES_PER_BASE_BIN = 16
# Distinguishes last-bit step disagreement after "same common_step divided by
# stage counts" from "the grid cap actually coarsened a component". Producer: the
# floor/ceil budget split; units: relative step difference.
_COMPONENT_STEP_ULP_BAND = 64.0 * float(np.finfo(np.float64).eps)

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
    require_allocation_counts(num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs)
    require_positive_real(value=loss_discretization, name="loss_discretization")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_bound_type(value=bound_type)
    # README Parameter Mapping: floor/ceil remainder split of inner steps and
    # outer epoch multiplicities. remainder is in [0, num_selected), so
    # floor_epochs is always positive; ceil_epochs is zero iff remainder is 0.
    floor_steps = num_steps // num_selected
    remainder = num_steps - num_selected * floor_steps
    ceil_steps = floor_steps + 1
    floor_epochs = (num_selected - remainder) * num_epochs
    ceil_epochs = remainder * num_epochs
    # Tail: active tail-consuming ops = one _allocation_directional_pld_core per component
    # plus one fft_convolve when both components are active, giving 2*component_count - 1
    # ops total (1 for component_count=1, 3 for component_count=2).  After
    # tail_truncation /= (2*component_count - 1), each op consumes at most the rescaled
    # budget, and all ops together sum to <= (2*component_count - 1) * rescaled = tail_truncation.
    component_count = int(floor_epochs > 0) + int(ceil_epochs > 0)
    tail_truncation /= 2 * component_count - 1
    # Loss: when both components are active, split the budget proportional to each
    # component's effective discretization count (num_epochs * base count).  This is
    # intentional: each component's final step is its loss budget divided by its
    # discretization count, so a proportional split makes the floor/ceil output steps
    # exactly equal -- loss_discretization / (floor_count + ceil_count) for both --
    # even when the base counts differ (GEOM counts depend on num_steps), which the
    # final fft_convolve requires.  The components' budgets still sum to
    # loss_discretization, so the total rounding error stays within budget.
    if floor_epochs > 0 and ceil_epochs > 0:
        floor_count = floor_epochs * base_loss_discretization_count(floor_steps)
        ceil_count = ceil_epochs * base_loss_discretization_count(ceil_steps)
        loss_disc_floor = loss_discretization * floor_count / (floor_count + ceil_count)
        loss_disc_ceil = loss_discretization * ceil_count / (floor_count + ceil_count)
    else:
        loss_disc_floor = loss_discretization
        loss_disc_ceil = loss_discretization

    dist_floor = None
    dist_ceil = None
    if floor_epochs > 0:
        dist_floor = _allocation_directional_pld_core(
            compute_base_pld=compute_base_pld,
            num_steps=floor_steps,
            num_epochs=floor_epochs,
            loss_discretization=loss_disc_floor,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    if ceil_epochs > 0:
        dist_ceil = _allocation_directional_pld_core(
            compute_base_pld=compute_base_pld,
            num_steps=ceil_steps,
            num_epochs=ceil_epochs,
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
    ``(base, neg_dual_base)`` factors. They are exponentiated, summed according
    to the paper's REMOVE construction, divided by ``num_steps``, and mapped
    back to loss space.
    """
    require_positive_int(value=num_steps, name="num_steps")
    require_positive_real(value=loss_discretization, name="loss_discretization")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_bound_type(value=bound_type)
    # For num_steps == 1 neither convolution stages nor Phases 2/3 execute, so no
    # division is needed for either budget.
    if num_steps > 1:
        # Loss: divide by the discretizing stage count (see the count function) so all
        # stages together stay within loss_discretization.
        loss_discretization /= remove_geometric_loss_discretization_count(num_steps)
        # Tail: three phases each receive tail_truncation / 3:
        #   Phase 1 (base_distributions_creation): per-factor budget /(3 * num_steps); the
        #            dual is self-convolved num_steps-1 times and the base convolved once,
        #            amplifying the total contribution back to budget/3.
        #   Phase 2 (geometric_self_convolve, num_steps-1) and Phase 3
        #            (geometric_convolve): budget/3 each.
        tail_truncation /= 3
    # Each base factor's tail error is amplified by num_steps through self-convolution,
    # so scale its budget down by num_steps to stay within the phase budget.
    base_factor_tail_truncation = tail_truncation / num_steps

    base, neg_dual_base = base_distributions_creation(
        loss_discretization=loss_discretization,
        tail_truncation=base_factor_tail_truncation,
        bound_type=bound_type,
    )
    # A one-factor average is already ``base``; avoid an unnecessary exp/log round-trip.
    if num_steps == 1:
        return base

    # Zero-anchored loss grids exp to anchor 1, so the composed anchor is the exact
    # integer num_steps and the closing average is an exact division.
    exp_neg_dual = exp_linear_to_geometric(neg_dual_base)
    exp_base = exp_linear_to_geometric(base)

    # V_{t-1} <- self-conv(V1, t-1, ...).
    exp_convolved_dual = geometric_self_convolve(
        dist=exp_neg_dual,
        num_convolutions=num_steps - 1,
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
    return log_geometric_to_linear(_averaged_exp_factor(dist=exp_convolved, num_steps=num_steps))


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
    factor. Its reflected loss is exponentiated, averaged after composition,
    then mapped and reflected back to ADD loss.
    """
    require_positive_real(value=loss_discretization, name="loss_discretization")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_bound_type(value=bound_type)
    require_positive_int(value=num_steps, name="num_steps")
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
    # A one-factor average is already ``base``; avoid an unnecessary exp/log round-trip.
    if num_steps == 1:
        return base

    neg_base = negate_reverse_linear_distribution(base)

    # Zero-anchored loss grids exp to anchor 1; see the REMOVE route.
    exp_base = exp_linear_to_geometric(neg_base)
    exp_bound_type = (
        BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    )
    # U_t <- self-conv(U, t, lower).
    exp_convolved = geometric_self_convolve(
        dist=exp_base,
        num_convolutions=num_steps,
        tail_truncation=tail_truncation,
        bound_type=exp_bound_type,
    )
    log_dist = log_geometric_to_linear(
        _averaged_exp_factor(dist=exp_convolved, num_steps=num_steps)
    )
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
    require_bound_type(value=bound_type)
    pmf_remove = linear_dist_to_dp_accounting_pmf(
        dist=remove_dist,
        bound_type=bound_type,
    )
    if add_dist is None:
        return privacy_loss_distribution.PrivacyLossDistribution(
            pmf_remove=pmf_remove,
        )
    pmf_add = linear_dist_to_dp_accounting_pmf(
        dist=add_dist,
        bound_type=bound_type,
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


def _averaged_exp_factor(*, dist: DenseDiscreteDist, num_steps: int) -> DenseDiscreteDist:
    """Divide a composed exp-space sum by ``num_steps`` to make it the average.

    Composition sums the factors' unit anchors, so the anchor here is the exact integer
    ``num_steps`` and ``x / x`` is exactly one: the averaged lattice is ``1 * r**k``, which
    ``log`` maps to a zero-anchored loss grid.
    """
    anchor = dist.grid.anchor
    if anchor != float(num_steps):
        raise ValueError(
            f"composed exp-space anchor {anchor!r} is not the unit-anchored factor count "
            f"{float(num_steps)!r}; the loss factors were not zero-anchored"
        )
    return DenseDiscreteDist(
        grid=replace(dist.grid, anchor=anchor / num_steps),
        prob_arr=dist.prob_arr,
        p_min=dist.p_min,
        p_max=dist.p_max,
        domain=dist.domain,
    )


def _align_component_grids(
    *,
    dist_floor: DenseDiscreteDist,
    dist_ceil: DenseDiscreteDist,
    bound_type: BoundType,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Align floor/ceil grids before the final ``fft_convolve``.

    Both components descend from one shared ``common_step``, but each divides it by
    its own stage counts, so the two final steps can land a few ULP apart. An exact
    match returns immediately; a last-bit difference is reconciled by unifying the
    declared step without reprojection. A materially different step means the grid
    cap coarsened one component, which is reported.
    """
    if dist_floor.step == dist_ceil.step:
        return dist_floor, dist_ceil

    finer_step = min(dist_floor.step, dist_ceil.step)
    coarser_step = max(dist_floor.step, dist_ceil.step)
    if coarser_step <= finer_step * (1.0 + _COMPONENT_STEP_ULP_BAND):
        return (
            _with_declared_component_step(dist=dist_floor, target_step=coarser_step),
            _with_declared_component_step(dist=dist_ceil, target_step=coarser_step),
        )

    target_step = coarser_step
    finer = "floor" if dist_floor.step < target_step else "ceil"
    source = dist_floor if finer == "floor" else dist_ceil
    step_before = source.step
    size_before = source.prob_arr.size
    coarsened = rediscretize_dist_by_bound(
        dist=source,
        tail_truncation=0.0,  # Alignment must not spend the tail budget a second time.
        loss_discretization=target_step,
        bound_type=bound_type,
    )
    if finer == "floor":
        dist_floor = coarsened
    else:
        dist_ceil = coarsened

    if target_step > step_before * (1.0 + _COMPONENT_STEP_ULP_BAND):
        warnings.warn(
            "allocation_directional_pld: aligning mismatched floor/ceil grids. "
            f"{finer} size {size_before}->{coarsened.prob_arr.size}, "
            f"step {step_before:.6e}->{coarsened.step:.6e}; "
            f"requested target_step={target_step:.6e}"
        )
    return dist_floor, dist_ceil


def _with_declared_component_step(
    *,
    dist: DenseDiscreteDist,
    target_step: float,
) -> DenseDiscreteDist:
    """Unify a last-bit step disagreement without reprojecting the PMF.

    The caller has already established that both steps descend from one common
    budget and differ only inside ``_COMPONENT_STEP_ULP_BAND``. Retaining the
    integer indices avoids an additional directional rounding stage.
    """
    if dist.step == target_step:
        return dist
    return DenseDiscreteDist(
        grid=replace(dist.grid, step=target_step),
        prob_arr=dist.prob_arr,
        p_min=dist.p_min,
        p_max=dist.p_max,
        domain=dist.domain,
    )


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
        #   Phase 2 (fft_self_convolve, num_convolutions=num_epochs): tail_truncation directly.
        #   Phase 3 (final truncate_edges): tail_truncation directly.
        tail_truncation /= 3
        # Loss: composing a base distribution with step s num_epochs times accumulates
        # quantization error of at most num_epochs * s, so divide by num_epochs.  FFT
        # self-convolution is algebraically exact and introduces no additional rounding.
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
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(prepared_base_dist).__name__} with spacing "
            f"{getattr(prepared_base_dist, 'spacing_type', '?')}"
        )

    if num_epochs == 1:
        composed_dist = prepared_base_dist
    else:
        # Cap base distribution so fft_self_convolve(num_convolutions=num_epochs) stays within
        # MAX_FFT_BYTES.  For the direct method the FFT size is num_convolutions * pmf_size;
        # for the binary method the worst-case FFT size is also num_convolutions * pmf_size
        # (before truncation shrinks intermediate steps).
        max_base_pmf = max(
            _MIN_BASE_PMF_BINS,
            MAX_FFT_BYTES // (num_epochs * _DIRECT_COMPOSE_BYTES_PER_BASE_BIN),
        )
        if prepared_base_dist.prob_arr.size > max_base_pmf:
            capped_step = prepared_base_dist.step * math.ceil(
                prepared_base_dist.prob_arr.size / max_base_pmf
            )
            prepared_base_dist = rediscretize_dist_by_bound(
                dist=prepared_base_dist,
                tail_truncation=base_tail_truncation,
                loss_discretization=capped_step,
                bound_type=bound_type,
            )
        composed_dist = fft_self_convolve(
            dist=prepared_base_dist,
            num_convolutions=num_epochs,
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
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(final_dist).__name__} with spacing "
            f"{getattr(final_dist, 'spacing_type', '?')}"
        )
    return final_dist
