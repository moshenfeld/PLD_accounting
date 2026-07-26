"""Gaussian-specific random-allocation accounting."""

from __future__ import annotations

import warnings
from dataclasses import replace
from functools import partial
from typing import Any, Callable

import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
)
from PLD_accounting.distribution_discretization import (
    discretize_continuous_ctd,
    discretize_continuous_stoch_dom,
    discretize_continuous_stoch_dom_on_grid,
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
)
from PLD_accounting.utils import (
    calc_pld_dual,
    log_geometric_to_linear,
    negate_reverse_linear_distribution,
)

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
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if direction not in (Direction.ADD, Direction.REMOVE):
        raise ValueError(f"Invalid direction: {direction}")
    if not isinstance(config, AllocationSchemeConfig):
        raise TypeError(f"config must be AllocationSchemeConfig, got {type(config)}")

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
    """Build REMOVE GEOM one-step PLD factors as ``(base, dual_base)``."""
    sigma_inv = 1.0 / sigma
    base_mean = sigma_inv**2 / 2
    base_loss_dist = stats.norm(loc=base_mean, scale=sigma_inv)
    factor_tail_truncation = (
        tail_truncation if bound_type == BoundType.DOMINATES else tail_truncation / 2
    )
    shared_step = _loss_discretization_for_grid_cap(
        dist=base_loss_dist,
        tail_truncation=factor_tail_truncation,
        loss_discretization=loss_discretization,
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

        grid_min = float(base_realization.x_array[0])
        grid_max = float(base_realization.x_array[-1])
        if not grid_min + sigma_inv <= -base_mean <= grid_max - sigma_inv:
            warnings.warn(
                "Gaussian REMOVE negative-dual mean is not at least one standard "
                "deviation inside the finite grid inherited from the discretized base: "
                f"negative_dual_mean={-base_mean:.6e}, "
                f"negative_dual_std={sigma_inv:.6e}, "
                f"grid=[{grid_min:.6e}, {grid_max:.6e}]. "
                "Deriving the negative dual from this grid can move most of its "
                "probability to a boundary atom.",
                RuntimeWarning,
                stacklevel=2,
            )
        dual_realization = calc_pld_dual(base_realization)
        neg_dual_realization = negate_reverse_linear_distribution(dual_realization)
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
    effective_step = _loss_discretization_for_grid_cap(
        dist=loss_dist,
        tail_truncation=tail_truncation,
        loss_discretization=loss_discretization,
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
    """Build ADD-direction FFT component and convert back to linear loss space."""
    exp_bound_type = _flip_bound_type(bound_type)
    fft_base = stats.lognorm(s=sigma_inv, scale=np.exp(-(sigma_inv**2) / 2 - np.log(num_steps)))
    linear_step = _loss_discretization_for_grid_cap(
        dist=fft_base,
        tail_truncation=single_step_tail_truncation,
        loss_discretization=loss_discretization,
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
    if not (
        isinstance(base_dist, DenseDiscreteDist) and base_dist.spacing_type == SpacingType.LINEAR
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(base_dist).__name__} with spacing {getattr(base_dist, 'spacing_type', '?')}"
        )
    # Fold zero-atom into leftmost finite bin before FFT convolution.
    base_prob_arr = base_dist.prob_arr.copy()
    base_prob_arr[0] += base_dist.p_min
    base_dist = DenseDiscreteDist(
        x_0=base_dist.x_0,
        step=base_dist.step,
        prob_arr=base_prob_arr,
        p_min=0.0,
        p_max=base_dist.p_max,
        domain=base_dist.domain,
    )

    conv_dist = fft_self_convolve(
        dist=base_dist,
        num_convolutions=num_steps,
        tail_truncation=tail_truncation,
        bound_type=exp_bound_type,
        use_direct=True,
    )
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
    """Build REMOVE-direction FFT component and convert back to linear loss space.

    Only BoundType.DOMINATES is supported.
    """
    if num_steps < 2:
        raise ValueError("REMOVE direction requires at least two steps per round")
    if bound_type != BoundType.DOMINATES:
        raise ValueError(f"FFT REMOVE route only supports BoundType.DOMINATES, got {bound_type}")

    factor_tail = single_step_tail_truncation / 2
    core_tail = tail_truncation / 2
    dual_norm_mean = -(sigma_inv**2) / 2 - np.log(num_steps)
    base_norm_mean = sigma_inv**2 / 2 - np.log(num_steps)
    dual_shift = np.exp(dual_norm_mean + sigma_inv**2 / 2)
    base_shift = np.exp(base_norm_mean + sigma_inv**2 / 2)

    dual_lognorm = stats.lognorm(s=sigma_inv, scale=np.exp(dual_norm_mean))
    dual_linear_step = _loss_discretization_for_grid_cap(
        dist=dual_lognorm,
        tail_truncation=factor_tail,
        loss_discretization=loss_discretization,
        max_grid_points=single_step_n_grid,
    )
    dual_dist = discretize_continuous_stoch_dom(
        dist=dual_lognorm,
        tail_truncation=factor_tail,
        bound_type=bound_type,
        step=dual_linear_step,
        align_to_multiples=False,
        domain=Domain.POSITIVES,
    )
    if not (
        isinstance(dual_dist, DenseDiscreteDist) and dual_dist.spacing_type == SpacingType.LINEAR
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(dual_dist).__name__} with spacing {getattr(dual_dist, 'spacing_type', '?')}"
        )
    # Fold zero-atom into leftmost finite bin before FFT convolution.
    dual_prob_arr = dual_dist.prob_arr.copy()
    dual_prob_arr[0] += dual_dist.p_min
    dual_dist = DenseDiscreteDist(
        x_0=dual_dist.x_0 - dual_shift,
        step=dual_dist.step,
        prob_arr=dual_prob_arr,
        p_min=0.0,
        p_max=dual_dist.p_max,
        domain=dual_dist.domain,
    )

    dual_convolved_dist = fft_self_convolve(
        dist=dual_dist,
        num_convolutions=num_steps - 1,
        tail_truncation=core_tail,
        bound_type=bound_type,
        use_direct=True,
    )

    exp_base = stats.lognorm(s=sigma_inv, scale=np.exp(base_norm_mean))
    # Fresh grid copy shifted into base space; the convolved dist is reused below.
    src_grid = dual_convolved_dist.grid
    base_grid = _extend_base_grid_for_fft_remove(
        grid=GridSpec(
            x_0=src_grid.x_0 + base_shift,
            step=src_grid.step,
            n=src_grid.n,
            spacing_type=src_grid.spacing_type,
        ),
        dist=exp_base,
        factor_tail_truncation=factor_tail,
        tail_truncation=tail_truncation,
    )
    base_dist = discretize_continuous_stoch_dom_on_grid(
        dist=exp_base,
        grid=base_grid,
        bound_type=bound_type,
        pmf_min_increment=factor_tail,
        domain=Domain.POSITIVES,
    )
    if not (
        isinstance(base_dist, DenseDiscreteDist) and base_dist.spacing_type == SpacingType.LINEAR
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(base_dist).__name__} with spacing {getattr(base_dist, 'spacing_type', '?')}"
        )
    base_dist = DenseDiscreteDist(
        x_0=base_dist.x_0 - base_shift,
        step=base_dist.step,
        prob_arr=base_dist.prob_arr,
        p_min=base_dist.p_min,
        p_max=base_dist.p_max,
        domain=base_dist.domain,
    )

    conv_dist_raw = fft_convolve(
        dist_1=dual_convolved_dist,
        dist_2=base_dist,
        tail_truncation=core_tail,
        bound_type=bound_type,
    )
    conv_dist_raw = DenseDiscreteDist(
        x_0=conv_dist_raw.x_0 + (num_steps - 1) * dual_shift + base_shift,
        step=conv_dist_raw.step,
        prob_arr=conv_dist_raw.prob_arr,
        p_min=conv_dist_raw.p_min,
        p_max=conv_dist_raw.p_max,
        domain=conv_dist_raw.domain,
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


def _extend_base_grid_for_fft_remove(
    *,
    grid: GridSpec,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    factor_tail_truncation: float,
    tail_truncation: float,
) -> GridSpec:
    """Extend a REMOVE base-factor linear ``grid`` when right-tail support is truncated.

    ``grid.step`` is taken from the source distribution (the convolved dual factor),
    so the extended grid keeps the exact spacing rather than re-deriving it.
    """
    if grid.n <= 1:
        return grid

    x_last = grid.last_point()
    x_max_target = dist.isf(factor_tail_truncation)
    if not np.isfinite(x_max_target) or x_last >= x_max_target:
        return grid
    if dist.sf(x_last) <= tail_truncation / 10:
        return grid

    n_extra = int(np.ceil((x_max_target - x_last) / grid.step))
    if n_extra <= 0:
        return grid
    return replace(grid, n=grid.n + n_extra)


# =============================================================================
# Internal helpers (grid bounds, coarsening, exp-space semantics)
# =============================================================================


def _loss_discretization_for_grid_cap(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    loss_discretization: float,
    max_grid_points: int,
) -> float:
    """Coarsen a quantile-range grid to fit within the configured point cap."""
    if loss_discretization <= 0.0:
        raise ValueError(f"loss_discretization must be positive, got {loss_discretization}")
    if max_grid_points <= 0:
        return float(loss_discretization)
    if max_grid_points < 2:
        raise ValueError(f"max_grid_points must be >= 2, got {max_grid_points}")

    x_min = float(dist.ppf(tail_truncation))
    x_max = float(dist.isf(tail_truncation))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max < x_min:
        raise ValueError(
            f"Invalid quantile range for tail_truncation={tail_truncation}: "
            f"x_min={x_min}, x_max={x_max}"
        )

    # Reserve one fencepost plus up to two bins for outward grid alignment.
    usable_intervals = max(max_grid_points - 3, 1)
    return float(max(loss_discretization, (x_max - x_min) / usable_intervals))


def _flip_bound_type(bound_type: BoundType) -> BoundType:
    """Swap DOMINATES <-> IS_DOMINATED for exp-space transforms."""
    return BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
