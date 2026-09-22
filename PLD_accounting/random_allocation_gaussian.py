"""Gaussian-specific random-allocation accounting."""

from __future__ import annotations

import math
from functools import partial
from typing import Callable

import numpy as np
from scipy import stats

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
)
from PLD_accounting.distribution_discretization import (
    discretize_continuous_ctd,
    discretize_continuous_stoch_dom,
    joint_source_dual_bounds,
    rediscretize_dist_stoch_dom,
)
from PLD_accounting.distribution_utils import (
    MIN_GRID_SIZE,
)
from PLD_accounting.fft_convolution import fft_convolve, fft_self_convolve
from PLD_accounting.random_allocation_accounting import (
    add_geometric_loss_discretization_count,
    geometric_allocation_pld_base_add,
    geometric_allocation_pld_base_remove,
    remove_geometric_loss_discretization_count,
)
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    SpacingType,
    require_allocation_config,
    require_direction,
)
from PLD_accounting.utils import (
    calc_pld_dual,
    log_geometric_to_linear,
    negate_reverse_linear_distribution,
)
from PLD_accounting.validation import require_integer, require_positive_real

# Keeps per-step SciPy quantile requests away from a zero tail probability.
_TAIL_EPS_FLOOR = float(np.finfo(float).eps * 1e-10)


# =============================================================================
# Public Entry Point
# =============================================================================


def gaussian_allocation_pld_core_and_count(
    *,
    direction: Direction,
    sigma: float,
    config: AllocationSchemeConfig,
) -> tuple[Callable[..., DenseDiscreteDist], Callable[[int], int]]:
    """Return the Gaussian base PLD builder and its loss-discretization count.

    This is the Gaussian-side orchestrator used by the shared allocation core.
    """
    require_positive_real(value=sigma, name="sigma")
    require_direction(value=direction)
    require_allocation_config(value=config)

    convolution_method = config.convolution_method
    if convolution_method == ConvolutionMethod.COMBINED:
        if direction == Direction.ADD:
            convolution_method = ConvolutionMethod.GEOM
        elif direction == Direction.REMOVE:
            convolution_method = ConvolutionMethod.FFT
        else:
            raise ValueError(f"Invalid direction: {direction}")

    if convolution_method == ConvolutionMethod.GEOM:
        compute_base_pld = partial(
            _gaussian_allocation_geom,
            direction=direction,
            sigma=sigma,
            config=config,
        )
        loss_discretization_count = (
            add_geometric_loss_discretization_count
            if direction == Direction.ADD
            else remove_geometric_loss_discretization_count
        )
        return compute_base_pld, loss_discretization_count

    if convolution_method == ConvolutionMethod.FFT:
        compute_base_pld = partial(
            _gaussian_allocation_fft,
            direction=direction,
            sigma=sigma,
            config=config,
        )
        # FFT base construction discretizes the loss exactly once.
        return compute_base_pld, (lambda _num_steps: 1)

    # BEST_OF_TWO and COMBINED are resolved per pure route
    # in random_allocation_api and never reach this function.
    raise ValueError(f"Invalid convolution_method: {convolution_method}")


# =============================================================================
# Internal GEOM Route
# =============================================================================


def _gaussian_allocation_geom(
    *,
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    direction: Direction,
    sigma: float,
    config: AllocationSchemeConfig,
) -> DenseDiscreteDist:
    """GEOM path intentionally mirrors the realization path after base creation.

    Both call ``geometric_allocation_pld_base_add`` / ``_remove`` with identical
    wiring; only the one-step factor construction differs.
    """
    if direction == Direction.ADD:
        return geometric_allocation_pld_base_add(
            base_distributions_creation=partial(
                _gaussian_add_geom_loss_factor,
                sigma=sigma,
                config=config,
            ),
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    if direction == Direction.REMOVE:
        return geometric_allocation_pld_base_remove(
            base_distributions_creation=partial(
                _gaussian_remove_geom_loss_factors,
                sigma=sigma,
                config=config,
            ),
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )
    raise ValueError(f"Invalid direction: {direction}")


def _gaussian_remove_geom_loss_factors(
    *,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    sigma: float,
    config: AllocationSchemeConfig,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Build REMOVE GEOM one-step factors as ``(base, negative_dual)``."""
    sigma_inv = 1.0 / sigma
    base_mean = sigma_inv**2 / 2
    base_loss_dist = stats.norm(loc=base_mean, scale=sigma_inv)
    factor_tail_truncation = (
        tail_truncation if bound_type == BoundType.DOMINATES else tail_truncation / 2
    )
    x_min, x_max = joint_source_dual_bounds(
        dist=base_loss_dist,
        dual_dist=base_loss_dist if bound_type == BoundType.DOMINATES else None,
        tail_truncation=factor_tail_truncation,
    )
    shared_step = _grid_step_for_point_cap(
        x_min=x_min,
        x_max=x_max,
        requested_step=loss_discretization,
        max_grid_points=config.max_grid_mult,
    )

    if bound_type == BoundType.DOMINATES:
        base_realization = discretize_continuous_ctd(
            dist=base_loss_dist,
            dual_dist=base_loss_dist,
            tail_truncation=tail_truncation,
            step=shared_step,
            align_to_multiples=True,
        )

        neg_dual_realization = negate_reverse_linear_distribution(calc_pld_dual(base_realization))
        return base_realization, neg_dual_realization

    # Lower-bound truncation can create negative-infinity mass, so preserve the
    # dual-first path and discretize the two continuous factors separately.
    dual_loss_dist = stats.norm(loc=-(sigma_inv**2) / 2, scale=sigma_inv)
    dual_loss_factor = discretize_continuous_stoch_dom(
        dist=dual_loss_dist,
        tail_truncation=factor_tail_truncation,
        bound_type=bound_type,
        step=shared_step,
        align_to_multiples=True,
    )
    base_loss_factor = discretize_continuous_stoch_dom(
        dist=base_loss_dist,
        tail_truncation=factor_tail_truncation,
        bound_type=bound_type,
        step=shared_step,
        align_to_multiples=True,
    )
    return base_loss_factor, dual_loss_factor


def _gaussian_add_geom_loss_factor(
    *,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    sigma: float,
    config: AllocationSchemeConfig,
) -> DenseDiscreteDist:
    """Build ADD GEOM one-step linear PLD factor."""
    sigma_inv = 1.0 / sigma
    loss_dist = stats.norm(loc=sigma_inv**2 / 2, scale=sigma_inv)
    x_min, x_max = joint_source_dual_bounds(
        dist=loss_dist,
        dual_dist=loss_dist if bound_type == BoundType.DOMINATES else None,
        tail_truncation=tail_truncation,
    )
    effective_step = _grid_step_for_point_cap(
        x_min=x_min,
        x_max=x_max,
        requested_step=loss_discretization,
        max_grid_points=config.max_grid_mult,
    )
    if bound_type == BoundType.DOMINATES:
        return discretize_continuous_ctd(
            dist=loss_dist,
            dual_dist=loss_dist,
            tail_truncation=tail_truncation,
            step=effective_step,
            align_to_multiples=True,
        )
    return discretize_continuous_stoch_dom(
        dist=loss_dist,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        step=effective_step,
        align_to_multiples=True,
    )


# =============================================================================
# Internal FFT Route
# =============================================================================


def _gaussian_allocation_fft(
    *,
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    direction: Direction,
    sigma: float,
    config: AllocationSchemeConfig,
) -> DenseDiscreteDist:
    """Build one FFT-based Gaussian PMF component for REMOVE or ADD."""
    sigma_inv = 1.0 / sigma
    single_step_tail_truncation = max(float(tail_truncation / num_steps), _TAIL_EPS_FLOOR)
    single_step_n_grid = max(int(np.ceil(config.max_grid_fft / num_steps)), MIN_GRID_SIZE)

    if direction == Direction.ADD:
        return _gaussian_allocation_fft_add(
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            sigma_inv=sigma_inv,
            single_step_tail_truncation=single_step_tail_truncation,
            single_step_n_grid=single_step_n_grid,
        )

    if direction == Direction.REMOVE:
        return _gaussian_allocation_fft_remove(
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            sigma_inv=sigma_inv,
            single_step_tail_truncation=single_step_tail_truncation,
            single_step_n_grid=single_step_n_grid,
        )

    raise ValueError(f"Invalid direction: {direction}")


def _gaussian_allocation_fft_add(
    *,
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    sigma_inv: float,
    single_step_tail_truncation: float,
    single_step_n_grid: int,
) -> DenseDiscreteDist:
    """Build the ADD FFT component.

    FFT exposes a final dominating bound. ADD ends with the order-reversing
    ``-log(sum)`` transform, so its positive exp-space sum uses the flipped
    (dominated) bound.
    """
    exp_bound_type = (
        BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
    )
    fft_base = stats.lognorm(s=sigma_inv, scale=np.exp(-(sigma_inv**2) / 2 - np.log(num_steps)))
    x_min, x_max = joint_source_dual_bounds(
        dist=fft_base,
        dual_dist=None,
        tail_truncation=single_step_tail_truncation,
    )
    linear_step = _grid_step_for_point_cap(
        x_min=x_min,
        x_max=x_max,
        requested_step=loss_discretization,
        max_grid_points=single_step_n_grid,
    )
    base_dist = discretize_continuous_stoch_dom(
        dist=fft_base,
        tail_truncation=single_step_tail_truncation,
        bound_type=exp_bound_type,
        step=linear_step,
        align_to_multiples=False,
        domain=Domain.POSITIVES,
    )
    real_base_dist = _embed_positive_boundary_on_nonpositive_real_cell(base_dist)

    real_conv_dist = fft_self_convolve(
        dist=real_base_dist,
        num_convolutions=num_steps,
        tail_truncation=tail_truncation,
        bound_type=exp_bound_type,
        use_direct=True,
    )
    conv_dist = _fold_nonpositive_real_mass_to_positive_boundary(real_conv_dist)
    exp_geom = rediscretize_dist_stoch_dom(
        dist=conv_dist,
        tail_truncation=0.0,
        loss_discretization=loss_discretization,
        spacing_type=SpacingType.GEOMETRIC,
        bound_type=exp_bound_type,
    )
    if not (
        isinstance(exp_geom, DenseDiscreteDist) and exp_geom.spacing_type == SpacingType.GEOMETRIC
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist with GEOMETRIC spacing, "
            f"got {type(exp_geom).__name__} with spacing {getattr(exp_geom, 'spacing_type', '?')}"
        )
    log_dist = log_geometric_to_linear(exp_geom)
    return negate_reverse_linear_distribution(log_dist)


def _gaussian_allocation_fft_remove(
    *,
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    sigma_inv: float,
    single_step_tail_truncation: float,
    single_step_n_grid: int,
) -> DenseDiscreteDist:
    """Build the REMOVE FFT component.

    FFT exposes only a dominating bound. REMOVE ends with the increasing
    ``log(base + sum(exp(-dual)))`` transform, so both the base and
    exponentiated negative-dual factors use dominating bounds. Their lower
    tails therefore enter the first finite bin (``p_min = 0``), allowing the
    distinct factors to use REALS-domain pairwise FFT convolution safely.
    """
    if num_steps < 2:
        raise ValueError("REMOVE direction requires at least two steps per round")
    if bound_type != BoundType.DOMINATES:
        raise ValueError(f"FFT REMOVE route only supports BoundType.DOMINATES, got {bound_type}")

    factor_tail = single_step_tail_truncation / 2
    core_tail = tail_truncation / 2
    dual_norm_mean = -(sigma_inv**2) / 2 - np.log(num_steps)
    base_norm_mean = sigma_inv**2 / 2 - np.log(num_steps)

    exp_neg_dual = stats.lognorm(s=sigma_inv, scale=np.exp(dual_norm_mean))
    exp_base = stats.lognorm(s=sigma_inv, scale=np.exp(base_norm_mean))
    neg_dual_min, neg_dual_max = joint_source_dual_bounds(
        dist=exp_neg_dual,
        dual_dist=None,
        tail_truncation=factor_tail,
    )
    neg_dual_linear_step = _grid_step_for_point_cap(
        x_min=neg_dual_min,
        x_max=neg_dual_max,
        requested_step=loss_discretization,
        max_grid_points=single_step_n_grid,
    )
    base_min, base_max = joint_source_dual_bounds(
        dist=exp_base,
        dual_dist=None,
        tail_truncation=factor_tail,
    )
    base_linear_step = _grid_step_for_point_cap(
        x_min=base_min,
        x_max=base_max,
        requested_step=loss_discretization,
        max_grid_points=single_step_n_grid,
    )
    shared_linear_step = max(neg_dual_linear_step, base_linear_step)

    neg_dual_dist = discretize_continuous_stoch_dom(
        dist=exp_neg_dual,
        tail_truncation=factor_tail,
        bound_type=bound_type,
        step=shared_linear_step,
        align_to_multiples=True,
        domain=Domain.REALS,
    )
    base_dist = discretize_continuous_stoch_dom(
        dist=exp_base,
        tail_truncation=factor_tail,
        bound_type=bound_type,
        step=shared_linear_step,
        align_to_multiples=True,
        domain=Domain.REALS,
    )

    neg_dual_convolved_dist = fft_self_convolve(
        dist=neg_dual_dist,
        num_convolutions=num_steps - 1,
        tail_truncation=core_tail,
        bound_type=bound_type,
        use_direct=True,
    )

    conv_dist_raw = fft_convolve(
        dist_1=neg_dual_convolved_dist,
        dist_2=base_dist,
        tail_truncation=core_tail,
        bound_type=bound_type,
    )
    exp_geom = rediscretize_dist_stoch_dom(
        dist=conv_dist_raw,
        tail_truncation=0.0,
        loss_discretization=loss_discretization,
        spacing_type=SpacingType.GEOMETRIC,
        bound_type=bound_type,
    )
    if not (
        isinstance(exp_geom, DenseDiscreteDist) and exp_geom.spacing_type == SpacingType.GEOMETRIC
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist with GEOMETRIC spacing, "
            f"got {type(exp_geom).__name__} with spacing {getattr(exp_geom, 'spacing_type', '?')}"
        )
    return log_geometric_to_linear(exp_geom)


def _embed_positive_boundary_on_nonpositive_real_cell(
    dist: DenseDiscreteDist,
) -> DenseDiscreteDist:
    """Put a POSITIVES zero atom on a nonpositive REALS lattice cell.

    This makes the boundary mass participate in ordinary FFT arithmetic without
    moving it above any value in the true nonnegative tail it represents.
    """
    if dist.domain != Domain.POSITIVES or dist.spacing_type != SpacingType.LINEAR:
        raise ValueError("Expected a POSITIVES-domain linear distribution")
    if dist.x_0 <= 0.0:
        raise ValueError(f"Expected positive finite-grid origin, got x_0={dist.x_0}")

    # GridSpec.pad extends the lattice without recomputing retained coordinates,
    # so a cancellation-prone origin still occupies a representable nonpositive
    # cell instead of folding p_min into an existing positive knot.
    num_prepended = int(math.ceil(dist.x_0 / dist.step))
    padded_grid = dist.grid.pad(left=num_prepended, right=0)
    if padded_grid.x_0 > 0.0:
        # ceil under-counts when fl(k * step) rounds low; one cell always clears the
        # residue, which a step exceeds by construction.
        num_prepended += 1
        padded_grid = dist.grid.pad(left=num_prepended, right=0)

    prob_arr = np.zeros(dist.prob_arr.size + num_prepended, dtype=np.float64)
    prob_arr[0] = dist.p_min
    prob_arr[num_prepended:] = dist.prob_arr
    return DenseDiscreteDist(
        grid=padded_grid,
        prob_arr=prob_arr,
        p_min=0.0,
        p_max=dist.p_max,
        domain=Domain.REALS,
    )


def _fold_nonpositive_real_mass_to_positive_boundary(
    dist: DenseDiscreteDist,
) -> DenseDiscreteDist:
    """Move a nonnegative sum's nonpositive REALS cells to its zero boundary.

    The REALS carrier grid is only an FFT device; after composition, values at
    or below zero all represent the POSITIVES-domain zero atom and must not
    remain finite support points.
    """
    if dist.domain != Domain.REALS or dist.spacing_type != SpacingType.LINEAR:
        raise ValueError("Expected a REALS-domain linear distribution")

    first_positive = int(np.searchsorted(dist.x_array, 0.0, side="right"))
    if first_positive >= dist.prob_arr.size:
        raise ValueError("Convolved distribution has no positive finite support")
    p_min = dist.p_min + math.fsum(map(float, dist.prob_arr[:first_positive]))
    return DenseDiscreteDist(
        grid=dist.grid.slice(start=first_positive, n=dist.prob_arr.size - first_positive),
        prob_arr=dist.prob_arr[first_positive:].copy(),
        p_min=p_min,
        p_max=dist.p_max,
        domain=Domain.POSITIVES,
    )


# =============================================================================
# Internal helpers (grid bounds, coarsening, exp-space semantics)
# =============================================================================


def _grid_step_for_point_cap(
    *,
    x_min: float,
    x_max: float,
    requested_step: float,
    max_grid_points: int,
) -> float:
    """Increase a grid step when needed to fit a point cap."""
    requested_step = require_positive_real(value=requested_step, name="requested_step")
    max_grid_points = require_integer(value=max_grid_points, name="max_grid_points")
    if max_grid_points <= 0:
        return requested_step
    if max_grid_points < 2:
        raise ValueError(f"max_grid_points must be >= 2, got {max_grid_points!r}")

    # Reserve one fencepost plus up to two bins for outward grid alignment.
    usable_intervals = max(max_grid_points - 3, 1)
    return max(requested_step, (x_max - x_min) / usable_intervals)
