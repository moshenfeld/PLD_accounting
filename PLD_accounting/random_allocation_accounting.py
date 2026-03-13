"""
Shared random-allocation composition helpers.
"""

from __future__ import annotations

import numpy as np
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.types import AllocationSchemeConfig, BoundType, Direction, SpacingType
from PLD_accounting.discrete_dist import DiscreteDistBase, LinearDiscreteDist, PLDRealization
from PLD_accounting.utils import calc_pld_dual, exp_linear_to_geometric, log_geometric_to_linear, negate_reverse_linear_distribution
from PLD_accounting.dp_accounting_support import linear_dist_to_dp_accounting_pmf
from PLD_accounting.distribution_discretization import change_spacing_type
from PLD_accounting.distribution_utils import stable_isclose
from PLD_accounting.geometric_convolution import geometric_convolve, geometric_self_convolve
from PLD_accounting.FFT_convolution import FFT_self_convolve

# =============================================================================
# Public API
# =============================================================================

def allocation_PMF_from_realization(*,
    realization: PLDRealization,
    direction: Direction,
    num_steps_per_round: int,
    num_rounds: int,
    config: AllocationSchemeConfig,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Compute an allocation PMF from an explicit PLD realization.

    Algorithms 1 and 2 (`rand-alloc-rem/add`) in Apendix C.

    Args:
        realization: Explicit PLD realization describing the privacy loss distribution.
        direction: Privacy direction (REMOVE or ADD, not BOTH).
        num_steps_per_round: Number of times to compose the base realization for a
            single round. In ``general_allocation_PLD`` this corresponds to
            ``floor(num_steps / num_selected)``.
        num_rounds: Number of outer round/epoch compositions applied after
            building the per-round PMF.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated discretized bound.

    Returns:
        Composed allocation PMF as a linear discrete distribution.
    """
    if not isinstance(realization, PLDRealization):
        raise TypeError(
            f"allocation_PMF_from_realization requires PLDRealization, got {type(realization)}"
        )
    
    if direction == Direction.REMOVE:
        base_dist = _allocation_PMF_remove_from_realization(
            realization=realization,
            num_steps_per_round=num_steps_per_round,
            config=config,
            bound_type=bound_type,
        )
    elif direction == Direction.ADD:
        base_dist = _allocation_PMF_add_from_realization(
            realization=realization,
            num_steps_per_round=num_steps_per_round,
            config=config,
            bound_type=bound_type,
        )
    else:
        raise ValueError("allocation_PMF_from_realization requires REMOVE or ADD, not BOTH")
    return finalize_allocation_composition(
        round_dist=base_dist,
        num_rounds=num_rounds,
        pre_composition_loss_discretization=config.loss_discretization,
        pre_composition_tail_truncation=config.tail_truncation,
        output_loss_discretization=config.loss_discretization,
        output_tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )


def decompose_allocation_compositions(*,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
) -> tuple[int, int]:
    """
    Map API parameters to inner/outer composition counts.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if num_selected < 1:
        raise ValueError(f"num_selected must be >= 1, got {num_selected}")
    if num_epochs < 1:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")

    num_steps_per_round = int(num_steps // num_selected)
    if num_steps_per_round < 1:
        raise ValueError("num_steps must be >= num_selected to form at least one step per round")

    num_rounds = num_selected * num_epochs
    return num_steps_per_round, num_rounds


def compose_allocation_pmfs(
    *,
    round_dist: LinearDiscreteDist,
    num_rounds: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Compose per-round allocation PMF across rounds/epochs in loss-space.
    """
    if num_rounds < 1:
        raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")
    if num_rounds == 1:
        return round_dist
    return FFT_self_convolve(
        dist=round_dist,
        T=num_rounds,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        use_direct=True,
    )


# =============================================================================
# Helper Functions
# =============================================================================

def log_mean_exp_remove(*,
    lower_loss_factor: LinearDiscreteDist,
    upper_loss_factor: LinearDiscreteDist,
    num_steps: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Compose remove-direction loss-space factors via exp-space geometric convolution.

    Internal part of Algorithm 1 (`rand-alloc-rem`) in Apendix C.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    
    # Subtract the average loss
    log_num_steps = float(np.log(num_steps))
    scaled_lower_loss = LinearDiscreteDist(
        x_min=lower_loss_factor.x_min - log_num_steps,
        x_gap=lower_loss_factor.x_gap,
        PMF_array=lower_loss_factor.PMF_array.copy(),
        p_neg_inf=lower_loss_factor.p_neg_inf,
        p_pos_inf=lower_loss_factor.p_pos_inf,
    )
    scaled_upper_loss = LinearDiscreteDist(
        x_min=upper_loss_factor.x_min - log_num_steps,
        x_gap=upper_loss_factor.x_gap,
        PMF_array=upper_loss_factor.PMF_array.copy(),
        p_neg_inf=upper_loss_factor.p_neg_inf,
        p_pos_inf=upper_loss_factor.p_pos_inf,
    )

    # Factor preparation in exp-space.
    lower_exp_factor = exp_linear_to_geometric(scaled_lower_loss)
    upper_exp_factor = exp_linear_to_geometric(scaled_upper_loss)

    if num_steps == 1:
        composed_exp_dist = upper_exp_factor
    else:
        # V_{t-1} <- self-conv(V1, t-1, ...).
        convolved_lower = geometric_self_convolve(
            dist=lower_exp_factor,
            T=num_steps - 1,
            tail_truncation=tail_truncation / 3,
            bound_type=bound_type,
        )
        # U_t <- conv(V_{t-1}, U1, ...).
        composed_exp_dist = geometric_convolve(
            dist_1=convolved_lower,
            dist_2=upper_exp_factor,
            tail_truncation=tail_truncation / 3,
            bound_type=bound_type,
        )
    # L_t <- log(U_t).
    return log_geometric_to_linear(composed_exp_dist)


def log_mean_exp_add(*,
    add_loss_factor: LinearDiscreteDist,
    num_steps: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """Compose add-direction loss-space factor via exp-space geometric convolution.

    Internal part of Algorithm 2 (`rand-alloc-add`) in Apendix C.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    log_num_steps = float(np.log(num_steps))

    scaled_add_loss = LinearDiscreteDist(
        x_min=add_loss_factor.x_min - log_num_steps,
        x_gap=add_loss_factor.x_gap,
        PMF_array=add_loss_factor.PMF_array.copy(),
        p_neg_inf=add_loss_factor.p_neg_inf,
        p_pos_inf=add_loss_factor.p_pos_inf,
    )

    # Factor preparation in exp-space.
    exp_factor = exp_linear_to_geometric(scaled_add_loss)
    exp_bound_type = BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    if num_steps == 1:
        conv_dist = exp_factor
    else:
        # U_t <- self-conv(U, t, lower).
        conv_dist = geometric_self_convolve(
            dist=exp_factor,
            T=num_steps,
            tail_truncation=tail_truncation / 2,
            bound_type=exp_bound_type,
        )
    # L_t <- -log(U_t).
    log_dist = log_geometric_to_linear(conv_dist)
    return negate_reverse_linear_distribution(log_dist)


def compose_pld_from_pmfs(*,
    remove_dist: LinearDiscreteDist | None,
    add_dist: LinearDiscreteDist | None,
    pessimistic_estimate: bool,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """
    Convert remove/add PMFs into a dp_accounting PLD.
    """
    if remove_dist is None:
        raise ValueError(
            "PLD construction requires remove-direction PMF. "
            "Provide remove_realization or use both directions."
        )
    pmf_remove = linear_dist_to_dp_accounting_pmf(dist=remove_dist, pessimistic_estimate=pessimistic_estimate)
    if add_dist is None:
        return privacy_loss_distribution.PrivacyLossDistribution(
            pmf_remove=pmf_remove,
        )
    pmf_add = linear_dist_to_dp_accounting_pmf(dist=add_dist, pessimistic_estimate=pessimistic_estimate)
    return privacy_loss_distribution.PrivacyLossDistribution(
        pmf_remove=pmf_remove,
        pmf_add=pmf_add,
    )


def finalize_allocation_composition(*,
    round_dist: DiscreteDistBase,
    num_rounds: int,
    pre_composition_loss_discretization: float,
    pre_composition_tail_truncation: float,
    output_loss_discretization: float,
    output_tail_truncation: float,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Finalize allocation composition with pre/post regridding.

    Steps:
    1. Ensure the round PMF is on the expected pre-composition linear grid.
    2. Compose across rounds.
    3. Regrid to the output linear discretization.
    """
    if not (
        isinstance(round_dist, LinearDiscreteDist)
        and stable_isclose(a=round_dist.x_gap, b=pre_composition_loss_discretization)
    ):
        round_dist = change_spacing_type(
            dist=round_dist,
            tail_truncation=pre_composition_tail_truncation,
            loss_discretization=pre_composition_loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
    assert isinstance(round_dist, LinearDiscreteDist)

    composed_dist = compose_allocation_pmfs(
        round_dist=round_dist,
        num_rounds=num_rounds,
        tail_truncation=output_tail_truncation,
        bound_type=bound_type,
    )

    final_dist = change_spacing_type(
        dist=composed_dist,
        tail_truncation=output_tail_truncation,
        loss_discretization=output_loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    assert isinstance(final_dist, LinearDiscreteDist)
    return final_dist


# =============================================================================
# Internal Functions
# =============================================================================


def _allocation_PMF_remove_from_realization(*,
    realization: PLDRealization,
    num_steps_per_round: int,
    config: AllocationSchemeConfig,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Remove-direction random allocation from a loss-space realization.

    Algorithm 1 (`rand-alloc-rem`) in Apendix C.
    """
    if num_steps_per_round < 1:
        raise ValueError(f"num_steps_per_round must be >= 1, got {num_steps_per_round}")

    # Compute negative dual PLD
    dual_realization = calc_pld_dual(realization)
    neg_dual_linear = negate_reverse_linear_distribution(dual_realization)

    # Rediscritize the PLD if needed.
    upper_linear: LinearDiscreteDist = realization
    if realization.x_gap < config.loss_discretization:
        upper_coarsened = change_spacing_type(
            dist=realization,
            tail_truncation=config.tail_truncation,
            loss_discretization=config.loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
        if not isinstance(upper_coarsened, LinearDiscreteDist):
            raise TypeError(f"Expected LinearDiscreteDist, got {type(upper_coarsened)}")
        upper_linear = upper_coarsened

    # Rediscritize the PLD dual if needed.
    dual_linear: LinearDiscreteDist = neg_dual_linear
    if dual_linear.x_gap < config.loss_discretization:
        dual_coarsened = change_spacing_type(
            dist=dual_linear,
            tail_truncation=config.tail_truncation,
            loss_discretization=config.loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=bound_type,
        )
        if not isinstance(dual_coarsened, LinearDiscreteDist):
            raise TypeError(f"Expected LinearDiscreteDist, got {type(dual_coarsened)}")
        dual_linear = dual_coarsened

    # Composition core.
    composed_dist = log_mean_exp_remove(
        lower_loss_factor=dual_linear,
        upper_loss_factor=upper_linear,
        num_steps=num_steps_per_round,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )
    # Final lattice alignment for downstream accounting.
    aligned_dist = change_spacing_type(
        dist=composed_dist,
        tail_truncation=config.tail_truncation,
        loss_discretization=config.loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    if not isinstance(aligned_dist, LinearDiscreteDist):
        raise TypeError(f"Expected LinearDiscreteDist, got {type(aligned_dist)}")
    return aligned_dist


def _allocation_PMF_add_from_realization(*,
    realization: PLDRealization,
    num_steps_per_round: int,
    config: AllocationSchemeConfig,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    """
    Add-direction random allocation from a loss-space realization.

    Algorithm 2 (`rand-alloc-add`) in Apendix C.
    """
    if num_steps_per_round < 1:
        raise ValueError(f"num_steps_per_round must be >= 1, got {num_steps_per_round}")

    exp_bound_type = (
        BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    )
    # Negate the PLD
    add_realization = negate_reverse_linear_distribution(realization)
    # Rediscritize the negative PLD if needed.
    add_linear: LinearDiscreteDist = add_realization
    if add_realization.x_gap < config.loss_discretization:
        add_coarsened = change_spacing_type(
            dist=add_realization,
            tail_truncation=config.tail_truncation,
            loss_discretization=config.loss_discretization,
            spacing_type=SpacingType.LINEAR,
            bound_type=exp_bound_type,
        )
        if not isinstance(add_coarsened, LinearDiscreteDist):
            raise TypeError(f"Expected LinearDiscreteDist, got {type(add_coarsened)}")
        add_linear = add_coarsened

    # Composition core.
    composed_dist = log_mean_exp_add(
        add_loss_factor=add_linear,
        num_steps=num_steps_per_round,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )
    # Final lattice alignment for downstream accounting.
    aligned_dist = change_spacing_type(
        dist=composed_dist,
        tail_truncation=config.tail_truncation,
        loss_discretization=config.loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    if not isinstance(aligned_dist, LinearDiscreteDist):
        raise TypeError(f"Expected LinearDiscreteDist, got {type(aligned_dist)}")
    return aligned_dist
