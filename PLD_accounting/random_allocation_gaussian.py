"""Gaussian-specific random-allocation accounting."""

from __future__ import annotations

from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain
from PLD_accounting.distribution_discretization import (
    discretize_continuous_dist,
    discretize_continuous_distribution,
    rediscritize_dist,
)
from PLD_accounting.distribution_utils import MIN_GRID_SIZE, compute_bin_width
from PLD_accounting.FFT_convolution import FFT_convolve, FFT_self_convolve
from PLD_accounting.random_allocation_accounting import (
    geometric_allocation_PLD_base_add,
    geometric_allocation_PLD_base_remove,
)
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    SpacingType,
)
from PLD_accounting.utils import (
    combine_distributions,
    log_geometric_to_linear,
    negate_reverse_linear_distribution,
)
from PLD_accounting.validation import (
    validate_bound_type,
    validate_discretization_params,
)

_TAIL_EPS_FLOOR = float(np.finfo(float).eps * 1e-10)


# =============================================================================
# Public Entry Point
# =============================================================================


def gaussian_allocation_PLD_core(
    *,
    num_steps: int,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    direction: Direction,
    sigma: float,
    config: AllocationSchemeConfig,
) -> DenseDiscreteDist:
    """Route one Gaussian component through GEOM/FFT/BEST backend selection.

    This is the Gaussian-side orchestrator used by the shared allocation core.
    """
    # Input validation
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    validate_discretization_params(loss_discretization, tail_truncation)
    validate_bound_type(bound_type)
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
        return _gaussian_allocation_geom(
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            direction=direction,
            sigma=sigma,
            config=config,
        )

    if convolution_method == ConvolutionMethod.FFT:
        return _gaussian_allocation_fft(
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            direction=direction,
            sigma=sigma,
            config=config,
        )

    if convolution_method == ConvolutionMethod.BEST_OF_TWO:
        fft_dist = _gaussian_allocation_fft(
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            direction=direction,
            sigma=sigma,
            config=config,
        )
        geom_dist = _gaussian_allocation_geom(
            num_steps=num_steps,
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            direction=direction,
            sigma=sigma,
            config=config,
        )
        return _combine_best_of_two(
            fft_dist=fft_dist,
            geom_dist=geom_dist,
            tail_truncation=tail_truncation,
            loss_discretization=loss_discretization,
            bound_type=bound_type,
        )

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
    """GEOM path intentionally mirrors realization path after base creation.

    Both call geometric_allocation_PLD_base_* with identical wiring.

    """
    if direction == Direction.ADD:
        return geometric_allocation_PLD_base_add(
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
        return geometric_allocation_PLD_base_remove(
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
    factor_tail_truncation = tail_truncation / 2
    n_grid_geom = _geom_grid_size(
        sigma_inv=sigma_inv,
        loss_discretization=loss_discretization,
        tail_probability=factor_tail_truncation / 2,
        config=config,
    )

    dual_norm_mean = -(sigma_inv**2) / 2
    base_norm_mean = sigma_inv**2 / 2
    exp_dual = stats.lognorm(s=sigma_inv, scale=np.exp(dual_norm_mean))
    exp_base = stats.lognorm(s=sigma_inv, scale=np.exp(base_norm_mean))

    dual_x_min = exp_dual.ppf(factor_tail_truncation)
    dual_x_max = exp_dual.isf(factor_tail_truncation)
    base_x_min = exp_base.ppf(factor_tail_truncation)
    base_x_max = exp_base.isf(factor_tail_truncation)
    log_span = max(np.log(dual_x_max / dual_x_min), np.log(base_x_max / base_x_min))
    shared_log_step = log_span / (n_grid_geom - 1)

    dual_factor_dist = discretize_continuous_dist(
        dist=exp_dual,
        x_array=_build_shared_geometric_grid(
            dist=exp_dual,
            tail_truncation=factor_tail_truncation,
            log_step=shared_log_step,
        ),
        bound_type=bound_type,
        PMF_min_increment=factor_tail_truncation,
        spacing_type=SpacingType.GEOMETRIC,
    )
    if not (
        isinstance(dual_factor_dist, DenseDiscreteDist)
        and dual_factor_dist.spacing_type == SpacingType.GEOMETRIC
    ):
        _st = getattr(dual_factor_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with GEOMETRIC spacing, "
            f"got {type(dual_factor_dist).__name__} with spacing {_st}"
        )

    base_factor_dist = discretize_continuous_dist(
        dist=exp_base,
        x_array=_build_shared_geometric_grid(
            dist=exp_base,
            tail_truncation=factor_tail_truncation,
            log_step=shared_log_step,
        ),
        bound_type=bound_type,
        PMF_min_increment=factor_tail_truncation,
        spacing_type=SpacingType.GEOMETRIC,
    )
    if not (
        isinstance(base_factor_dist, DenseDiscreteDist)
        and base_factor_dist.spacing_type == SpacingType.GEOMETRIC
    ):
        _st = getattr(base_factor_dist, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with GEOMETRIC spacing, "
            f"got {type(base_factor_dist).__name__} with spacing {_st}"
        )

    dual_loss_factor = log_geometric_to_linear(dual_factor_dist)
    base_loss_factor = log_geometric_to_linear(base_factor_dist)
    # geometric_allocation_PLD_base_remove expects (base, dual_base).
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
    n_grid_geom = _geom_grid_size(
        sigma_inv=sigma_inv,
        loss_discretization=loss_discretization,
        tail_probability=tail_truncation,
        config=config,
    )
    exp_bound_type = _flip_bound_type(bound_type)

    base_dist = discretize_continuous_distribution(
        dist=stats.lognorm(s=sigma_inv, scale=np.exp(-(sigma_inv**2) / 2)),
        tail_truncation=tail_truncation,
        bound_type=exp_bound_type,
        spacing_type=SpacingType.GEOMETRIC,
        n_grid=n_grid_geom,
        align_to_multiples=True,
    )
    if not (
        isinstance(base_dist, DenseDiscreteDist) and base_dist.spacing_type == SpacingType.GEOMETRIC
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist with GEOMETRIC spacing, "
            f"got {type(base_dist).__name__} with spacing {getattr(base_dist, 'spacing_type', '?')}"
        )
    return log_geometric_to_linear(base_dist)


def _geom_grid_size(
    *,
    sigma_inv: float,
    loss_discretization: float,
    tail_probability: float,
    config: AllocationSchemeConfig,
) -> int:
    """Compute GEOM grid size from tail probability and log-loss span."""
    if tail_probability <= 0.0:
        grid_size = MIN_GRID_SIZE
    else:
        log_range = -stats.norm.ppf(tail_probability) * sigma_inv
        if np.isfinite(log_range) and log_range > 0.0:
            grid_size = max(int(2 * log_range / loss_discretization), MIN_GRID_SIZE)
        else:
            grid_size = MIN_GRID_SIZE

    if config.max_grid_mult > 0:
        grid_size = min(grid_size, config.max_grid_mult)
    return grid_size


def _build_shared_geometric_grid(
    *,
    dist: stats.rv_continuous,
    tail_truncation: float,
    log_step: float,
) -> NDArray[np.float64]:
    """Build a geometric grid snapped to a shared log-space lattice."""
    x_min = dist.ppf(tail_truncation)
    x_max = dist.isf(tail_truncation)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError(f"Quantiles not finite: x_min={x_min}, x_max={x_max}")
    if x_min <= 0.0:
        raise ValueError(
            f"Geometric spacing requires positive values, got x_min={x_min}, x_max={x_max}"
        )
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got x_min={x_min}, x_max={x_max}")

    start_idx = int(np.floor(np.log(x_min) / log_step))
    stop_idx = int(np.ceil(np.log(x_max) / log_step))
    x_array = np.exp(log_step * np.arange(start_idx, stop_idx + 1, dtype=np.float64))

    support_min, support_max = dist.support()
    if np.isfinite(support_min):
        x_array = x_array[x_array > support_min]
    if np.isfinite(support_max):
        x_array = x_array[x_array < support_max]
    return x_array


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
    single_step_n_grid = max(int(np.ceil(config.max_grid_FFT / num_steps)), MIN_GRID_SIZE)

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
    base_dist = discretize_continuous_distribution(
        dist=stats.lognorm(s=sigma_inv, scale=np.exp(-(sigma_inv**2) / 2 - np.log(num_steps))),
        tail_truncation=single_step_tail_truncation,
        bound_type=exp_bound_type,
        spacing_type=SpacingType.LINEAR,
        n_grid=single_step_n_grid,
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
    base_dist.prob_arr[0] += base_dist.p_min
    base_dist.p_min = 0

    conv_dist = FFT_self_convolve(
        dist=base_dist,
        T=num_steps,
        tail_truncation=tail_truncation,
        bound_type=exp_bound_type,
        use_direct=True,
    )
    exp_geom = rediscritize_dist(
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

    dual_dist = discretize_continuous_distribution(
        dist=stats.lognorm(s=sigma_inv, scale=np.exp(dual_norm_mean)),
        tail_truncation=factor_tail,
        bound_type=bound_type,
        spacing_type=SpacingType.LINEAR,
        n_grid=single_step_n_grid,
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
    dual_dist.prob_arr[0] += dual_dist.p_min
    dual_dist.p_min = 0

    dual_dist.x_min -= dual_shift

    dual_convolved_dist = FFT_self_convolve(
        dist=dual_dist,
        T=num_steps - 1,
        tail_truncation=core_tail,
        bound_type=bound_type,
        use_direct=True,
    )

    exp_base = stats.lognorm(s=sigma_inv, scale=np.exp(base_norm_mean))
    base_grid = _extend_base_grid_for_fft_remove(
        x_array=dual_convolved_dist.x_array + base_shift,
        dist=exp_base,
        factor_tail_truncation=factor_tail,
        tail_truncation=tail_truncation,
    )
    base_dist = discretize_continuous_dist(
        dist=exp_base,
        x_array=base_grid,
        bound_type=bound_type,
        PMF_min_increment=factor_tail,
        spacing_type=SpacingType.LINEAR,
        domain=Domain.POSITIVES,
    )
    if not (
        isinstance(base_dist, DenseDiscreteDist) and base_dist.spacing_type == SpacingType.LINEAR
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(base_dist).__name__} with spacing {getattr(base_dist, 'spacing_type', '?')}"
        )
    base_dist.x_min -= base_shift

    conv_dist_raw = FFT_convolve(
        dist_1=dual_convolved_dist,
        dist_2=base_dist,
        tail_truncation=core_tail,
        bound_type=bound_type,
    )
    conv_dist_raw.x_min += (num_steps - 1) * dual_shift + base_shift
    exp_geom = rediscritize_dist(
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
    x_array: np.ndarray,
    dist: stats.rv_continuous,
    factor_tail_truncation: float,
    tail_truncation: float,
) -> NDArray[np.float64]:
    """Extend REMOVE base-factor linear grid when right-tail support is truncated."""
    if x_array.size <= 1:
        return x_array

    x_max_target = dist.isf(factor_tail_truncation)
    if not np.isfinite(x_max_target) or x_array[-1] >= x_max_target:
        return x_array
    if dist.sf(x_array[-1]) <= tail_truncation / 10:
        return x_array

    step = compute_bin_width(x_array)
    n_extra = int(np.ceil((x_max_target - x_array[-1]) / step))
    if n_extra <= 0:
        return x_array
    return np.concatenate([x_array, x_array[-1] + step * np.arange(1, n_extra + 1)])


# =============================================================================
# Shared Internal Utilities
# =============================================================================


def _combine_best_of_two(
    *,
    fft_dist: DenseDiscreteDist,
    geom_dist: DenseDiscreteDist,
    tail_truncation: float,
    loss_discretization: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Combine FFT and GEOM candidates and regrid to linear output spacing."""
    combined_dist = combine_distributions(
        dist_1=fft_dist,
        dist_2=geom_dist,
        bound_type=bound_type,
    )
    combined_linear = rediscritize_dist(
        dist=combined_dist,
        tail_truncation=tail_truncation,
        loss_discretization=loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    if not (
        isinstance(combined_linear, DenseDiscreteDist)
        and combined_linear.spacing_type == SpacingType.LINEAR
    ):
        _st = getattr(combined_linear, "spacing_type", "?")
        raise TypeError(
            "Expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(combined_linear).__name__} with spacing {_st}"
        )
    return combined_linear


def _flip_bound_type(bound_type: BoundType) -> BoundType:
    """Swap DOMINATES <-> IS_DOMINATED for exp-space transforms."""
    return BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
