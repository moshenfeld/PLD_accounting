"""Random-allocation composition that no grid owns.

The module owns the floor/ceil allocation split with its loss and tail budgets,
the epoch composition, and the final ``dp_accounting`` conversion. One allocation
is built by ``random_allocation_base.py`` using the selected grid's operations.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Callable

import numpy as np
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.discrete_dist import DenseDiscreteDist
from PLD_accounting.dp_accounting_support import linear_dist_to_dp_accounting_pmf
from PLD_accounting.fft_convolution import MAX_FFT_BYTES, fft_convolve, fft_self_convolve
from PLD_accounting.projections import project_discrete_dist, truncate_edges
from PLD_accounting.types import (
    BoundType,
    ProjectionMethod,
    SpacingType,
    require_bound_type,
    require_projection_method,
)
from PLD_accounting.validation import (
    require_allocation_counts,
    require_nonnegative_real,
    require_positive_real,
)

# Minimum number of bins to keep after per-epoch base PMF size capping.
_MIN_BASE_PMF_BINS = 4
# Estimated bytes consumed per input PMF bin in the direct epoch-compose cap.
_DIRECT_COMPOSE_BYTES_PER_BASE_BIN = 16
# Distinguishes last-bit step disagreement after "one loss budget split in proportion
# to the stage counts and divided by them" from "the grid cap actually coarsened a
# component". Producer: the floor/ceil budget split; units: relative step difference.
_COMPONENT_STEP_ULP_BAND = 64.0 * float(np.finfo(np.float64).eps)

# =============================================================================
# Public API
# =============================================================================


def allocation_plds(
    *,
    compute_base_plds: Callable[..., tuple[DenseDiscreteDist, DenseDiscreteDist]],
    base_loss_discretization_counts: Callable[..., tuple[int, int]],
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Build the allocation PLDs ``(remove, add)`` with adaptive floor/ceil decomposition.

    For divisible ``num_steps / num_selected``, this builds one component. For
    non-divisible cases, it builds floor and ceil components via
    ``_allocation_plds_core(...)`` and combines them with one final
    ``fft_convolve(...)`` per direction. ``base_loss_discretization_counts`` is
    called as ``base_loss_discretization_counts(num_steps=...)`` and returns the
    number of loss-discretizing stages ``(remove, add)`` of one base build.
    With several selections, bounds refer to this floor/ceil composition; only
    its upper bound necessarily bounds uniform k-out-of-t allocation.
    ``projection_method`` selects the truncation of the tails of every stage after
    the base build.
    """
    require_allocation_counts(num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs)
    require_positive_real(value=loss_discretization, name="loss_discretization")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_projection_method(value=projection_method, bound_type=bound_type)
    # README Parameter Mapping: floor/ceil remainder split of inner steps and
    # outer epoch multiplicities. remainder is in [0, num_selected), so
    # floor_epochs is always positive; ceil_epochs is zero iff remainder is 0.
    floor_steps = num_steps // num_selected
    remainder = num_steps - num_selected * floor_steps
    ceil_steps = floor_steps + 1
    floor_epochs = (num_selected - remainder) * num_epochs
    ceil_epochs = remainder * num_epochs
    # Tail: active tail-consuming ops of a direction = one _allocation_plds_core per
    # component plus one fft_convolve when both components are active, giving
    # 2*component_count - 1 ops total (1 for component_count=1, 3 for component_count=2).
    # After tail_truncation /= (2*component_count - 1), each op consumes at most the rescaled
    # budget, and all ops together sum to <= (2*component_count - 1) * rescaled = tail_truncation.
    component_count = 1 + int(ceil_epochs > 0)
    tail_truncation /= 2 * component_count - 1
    if ceil_epochs == 0:
        return _allocation_plds_core(
            compute_base_plds=compute_base_plds,
            num_steps=floor_steps,
            num_epochs=floor_epochs,
            loss_discretizations=(loss_discretization, loss_discretization),
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            projection_method=projection_method,
        )
    # Loss: with both components active, split each direction's budget proportional to
    # each component's effective discretization count (num_epochs * base count).  This is
    # intentional: each component's final step is its loss budget divided by its
    # discretization count, so a proportional split makes the floor/ceil output steps
    # exactly equal -- loss_discretization / (floor_count + ceil_count) for both --
    # even when the base counts differ (GEOM counts depend on num_steps), which the
    # final fft_convolve requires.  The components' budgets still sum to
    # loss_discretization, so the total rounding error stays within budget.  The counts
    # of the two directions can differ, so the directions need not share a step.
    remove_floor_count, add_floor_count = (
        floor_epochs * count for count in base_loss_discretization_counts(num_steps=floor_steps)
    )
    remove_ceil_count, add_ceil_count = (
        ceil_epochs * count for count in base_loss_discretization_counts(num_steps=ceil_steps)
    )
    remove_floor, add_floor = _allocation_plds_core(
        compute_base_plds=compute_base_plds,
        num_steps=floor_steps,
        num_epochs=floor_epochs,
        loss_discretizations=(
            loss_discretization * remove_floor_count / (remove_floor_count + remove_ceil_count),
            loss_discretization * add_floor_count / (add_floor_count + add_ceil_count),
        ),
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        projection_method=projection_method,
    )
    remove_ceil, add_ceil = _allocation_plds_core(
        compute_base_plds=compute_base_plds,
        num_steps=ceil_steps,
        num_epochs=ceil_epochs,
        loss_discretizations=(
            loss_discretization * remove_ceil_count / (remove_floor_count + remove_ceil_count),
            loss_discretization * add_ceil_count / (add_floor_count + add_ceil_count),
        ),
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        projection_method=projection_method,
    )
    return (
        _combine_components(
            dist_floor=remove_floor,
            dist_ceil=remove_ceil,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            projection_method=projection_method,
        ),
        _combine_components(
            dist_floor=add_floor,
            dist_ceil=add_ceil,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            projection_method=projection_method,
        ),
    )


def compose_full_pld(
    *,
    remove_dist: DenseDiscreteDist,
    add_dist: DenseDiscreteDist | None,
    bound_type: BoundType,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Convert remove/add directional PLDs into a ``dp_accounting`` PLD.

    Both inputs are real-domain distributions on zero-anchored linear grids. With
    ``add_dist=None`` the result is symmetric: ``dp_accounting`` uses the remove PMF
    for both directions, which bounds the add direction only when remove dominates it.

    Args:
        remove_dist: REMOVE-direction linear PLD.
        add_dist: ADD-direction linear PLD, or ``None`` for a symmetric PLD.
        bound_type: ``DOMINATES`` exports pessimistic PMFs and requires ``p_min`` of
            zero; ``IS_DOMINATED`` exports optimistic PMFs and omits ``p_min``, whose
            loss ``-inf`` adds nothing to delta.

    Returns:
        A ``dp_accounting`` privacy loss distribution.

    """
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


# =============================================================================
# Helper Functions
# =============================================================================


def _allocation_plds_core(
    *,
    compute_base_plds: Callable[..., tuple[DenseDiscreteDist, DenseDiscreteDist]],
    num_steps: int,
    num_epochs: int,
    loss_discretizations: tuple[float, float],
    tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Build and finalize one floor/ceil decomposition component for both directions.

    This function derives component-level budgets, calls ``compute_base_plds(...)``
    once, and composes each base PLD over the epochs.
    """
    if num_epochs > 1:
        # Tail: divide by 3; each of the three phases receives tail_truncation (the divided value):
        #   Phase 1 (base creation, amplified by num_epochs): up to 3 sub-ops per epoch
        #            (compute_base_plds, truncate_edges, optional project_discrete_dist), each with
        #            base_tail_truncation = tail_truncation / (3 * num_epochs).  Amplified total:
        #            num_epochs * 3 * tail_truncation / (3 * num_epochs) = tail_truncation.
        #   Phase 2 (fft_self_convolve, num_convolutions=num_epochs): tail_truncation directly.
        #   Phase 3 (final truncate_edges): tail_truncation directly.
        tail_truncation /= 3
    else:
        # num_epochs == 1: fft_self_convolve is SKIPPED, so only Phases 1 and 3 are active;
        # divide the tail budget by 2 instead.
        tail_truncation /= 2
    base_tail_truncation = tail_truncation / (3 * num_epochs)
    # Loss: composing a base distribution with step s num_epochs times accumulates
    # quantization error of at most num_epochs * s, so divide by num_epochs.  FFT
    # self-convolution is algebraically exact and introduces no additional rounding.
    remove_loss_discretization, add_loss_discretization = loss_discretizations

    base_remove, base_add = compute_base_plds(
        num_steps=num_steps,
        loss_discretizations=(
            remove_loss_discretization / num_epochs,
            add_loss_discretization / num_epochs,
        ),
        tail_truncation=base_tail_truncation,
        bound_type=bound_type,
    )
    return (
        _compose_base_pld_over_epochs(
            base_dist=base_remove,
            num_epochs=num_epochs,
            tail_truncation=tail_truncation,
            base_tail_truncation=base_tail_truncation,
            bound_type=bound_type,
            projection_method=projection_method,
        ),
        _compose_base_pld_over_epochs(
            base_dist=base_add,
            num_epochs=num_epochs,
            tail_truncation=tail_truncation,
            base_tail_truncation=base_tail_truncation,
            bound_type=bound_type,
            projection_method=projection_method,
        ),
    )


def _combine_components(
    *,
    dist_floor: DenseDiscreteDist,
    dist_ceil: DenseDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> DenseDiscreteDist:
    """Align the floor and ceil components of one direction and convolve them."""
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
        projection_method=projection_method,
    )


def _compose_base_pld_over_epochs(
    *,
    base_dist: DenseDiscreteDist,
    num_epochs: int,
    tail_truncation: float,
    base_tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> DenseDiscreteDist:
    """Trim one base PLD, compose it over the epochs, and trim the result.

    The output step is the one the base builder produced, unless the FFT size cap
    coarsens it.
    """
    prepared_base_dist = truncate_edges(
        dist=base_dist,
        tail_truncation=base_tail_truncation,
        bound_type=bound_type,
        projection_method=projection_method,
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
        # MAX_FFT_BYTES. The full num_epochs-fold support is num_convolutions * pmf_size, the
        # worst case of the direct transform's window and of its binary fallback.
        max_base_pmf = max(
            _MIN_BASE_PMF_BINS,
            MAX_FFT_BYTES // (num_epochs * _DIRECT_COMPOSE_BYTES_PER_BASE_BIN),
        )
        if prepared_base_dist.prob_arr.size > max_base_pmf:
            capped_step = prepared_base_dist.step * math.ceil(
                prepared_base_dist.prob_arr.size / max_base_pmf
            )
            step_before = prepared_base_dist.step
            prepared_base_dist = project_discrete_dist(
                dist=prepared_base_dist,
                tail_truncation=base_tail_truncation,
                loss_discretization=capped_step,
                bound_type=bound_type,
            )
            warnings.warn(
                f"epoch composition: MAX_FFT_BYTES={MAX_FFT_BYTES} limits the base PLD of "
                f"{num_epochs} epochs to {max_base_pmf} points, which coarsens its step from "
                f"{step_before:.6e} to {prepared_base_dist.step:.6e}. The bound stays valid "
                "and is looser.",
                stacklevel=5,
            )
        composed_dist = fft_self_convolve(
            dist=prepared_base_dist,
            num_convolutions=num_epochs,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            projection_method=projection_method,
            use_direct=True,
        )
    final_dist = truncate_edges(
        dist=composed_dist,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        projection_method=projection_method,
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


def _align_component_grids(
    *,
    dist_floor: DenseDiscreteDist,
    dist_ceil: DenseDiscreteDist,
    bound_type: BoundType,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Align floor/ceil grids before the final ``fft_convolve``.

    Each component receives a share of the loss budget proportional to its stage
    count and divides it by that count, so in exact arithmetic both final steps are
    equal and in floating point they can land a few ULP apart. An exact match
    returns immediately; a last-bit difference is reconciled by unifying the
    declared step without reprojection. A materially different step means a grid
    cap coarsened one component; the finer one is projected onto the coarser step,
    which is reported.
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
    coarsened = project_discrete_dist(
        dist=source,
        tail_truncation=0.0,  # Alignment must not spend the tail budget a second time.
        loss_discretization=target_step,
        bound_type=bound_type,
    )
    if finer == "floor":
        dist_floor = coarsened
    else:
        dist_ceil = coarsened

    # This branch is reached only beyond the last-bit band, so the step changed materially.
    warnings.warn(
        "allocation_plds: aligning mismatched floor/ceil grids. "
        f"{finer} size {size_before}->{coarsened.prob_arr.size}, "
        f"step {step_before:.6e}->{coarsened.step:.6e}; "
        f"requested target_step={target_step:.6e}",
        UserWarning,
        stacklevel=5,
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
    target_grid = replace(dist.grid, step=target_step)
    return DenseDiscreteDist(
        grid=target_grid,
        prob_arr=dist.prob_arr,
        p_min=dist.p_min,
        p_max=dist.p_max,
        domain=dist.domain,
    )
