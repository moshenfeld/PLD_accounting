"""Gaussian-specific random-allocation accounting."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain
from PLD_accounting.distribution_discretization import (
    discretize_aligned_range,
    discretize_continuous_dist,
    discretize_continuous_distribution,
    rediscretize_dist,
)
from PLD_accounting.distribution_utils import MIN_GRID_SIZE, compute_bin_width
from PLD_accounting.fft_convolution import fft_convolve, fft_self_convolve
from PLD_accounting.random_allocation_accounting import (
    geometric_allocation_pld_base_add,
    geometric_allocation_pld_base_remove,
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


def gaussian_allocation_pld_core(
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

    # Match ADD GEOM: cap aligned lattice length at ``n_grid_geom`` (``max_grid_mult``) by
    # coarsening log-step from ``loss_discretization``; share one lattice for both factors.
    dual_step, dual_x_min, dual_x_max = _coarsen_discretization_for_tail_quantile_range(
        dist=exp_dual,
        tail_truncation=factor_tail_truncation,
        target_discretization=float(loss_discretization),
        max_points=n_grid_geom,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
    )
    base_step, base_x_min, base_x_max = _coarsen_discretization_for_tail_quantile_range(
        dist=exp_base,
        tail_truncation=factor_tail_truncation,
        target_discretization=float(loss_discretization),
        max_points=n_grid_geom,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
    )
    shared_log_step = max(dual_step, base_step)
    dual_x_array = discretize_aligned_range(
        x_min=dual_x_min,
        x_max=dual_x_max,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
        discretization=shared_log_step,
    )
    base_x_array = discretize_aligned_range(
        x_min=base_x_min,
        x_max=base_x_max,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
        discretization=shared_log_step,
    )

    dual_factor_dist = discretize_continuous_dist(
        dist=exp_dual,
        x_array=dual_x_array,
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
        x_array=base_x_array,
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
    # geometric_allocation_pld_base_remove expects (base, dual_base).
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

    base_lognorm = stats.lognorm(s=sigma_inv, scale=np.exp(+(sigma_inv**2) / 2))
    eff_log_step, _, _ = _coarsen_discretization_for_tail_quantile_range(
        dist=base_lognorm,
        tail_truncation=tail_truncation,
        target_discretization=float(loss_discretization),
        max_points=n_grid_geom,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
    )

    base_dist = discretize_continuous_distribution(
        dist=base_lognorm,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        spacing_type=SpacingType.GEOMETRIC,
        step=float(np.exp(eff_log_step)),
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
    linear_step, _, _ = _coarsen_discretization_for_tail_quantile_range(
        dist=fft_base,
        tail_truncation=single_step_tail_truncation,
        target_discretization=float(loss_discretization),
        max_points=single_step_n_grid,
        spacing_type=SpacingType.LINEAR,
        align_to_multiples=False,
    )
    base_dist = discretize_continuous_distribution(
        dist=fft_base,
        tail_truncation=single_step_tail_truncation,
        bound_type=exp_bound_type,
        spacing_type=SpacingType.LINEAR,
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
    base_dist.prob_arr[0] += base_dist.p_min
    base_dist.p_min = 0

    conv_dist = fft_self_convolve(
        dist=base_dist,
        T=num_steps,
        tail_truncation=tail_truncation,
        bound_type=exp_bound_type,
        use_direct=True,
    )
    exp_geom = rediscretize_dist(
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
    dual_linear_step, _, _ = _coarsen_discretization_for_tail_quantile_range(
        dist=dual_lognorm,
        tail_truncation=factor_tail,
        target_discretization=float(loss_discretization),
        max_points=single_step_n_grid,
        spacing_type=SpacingType.LINEAR,
        align_to_multiples=False,
    )
    dual_dist = discretize_continuous_distribution(
        dist=dual_lognorm,
        tail_truncation=factor_tail,
        bound_type=bound_type,
        spacing_type=SpacingType.LINEAR,
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
    dual_dist.prob_arr[0] += dual_dist.p_min
    dual_dist.p_min = 0

    dual_dist.x_min -= dual_shift

    dual_convolved_dist = fft_self_convolve(
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

    conv_dist_raw = fft_convolve(
        dist_1=dual_convolved_dist,
        dist_2=base_dist,
        tail_truncation=core_tail,
        bound_type=bound_type,
    )
    conv_dist_raw.x_min += (num_steps - 1) * dual_shift + base_shift
    exp_geom = rediscretize_dist(
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
    dist: stats.rv_continuous | rv_frozen[Any, Any],
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
# Best-of-two (FFT vs GEOM)
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
    combined_linear = rediscretize_dist(
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


# =============================================================================
# Internal helpers (grid bounds, coarsening, exp-space semantics)
# =============================================================================


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
            grid_size = max(int(np.ceil(2 * log_range / loss_discretization)) + 1, MIN_GRID_SIZE)
        else:
            grid_size = MIN_GRID_SIZE

    if config.max_grid_mult > 0:
        grid_size = min(grid_size, config.max_grid_mult)
    return grid_size


def _coarsen_discretization_for_tail_quantile_range(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    tail_truncation: float,
    target_discretization: float,
    max_points: int,
    spacing_type: SpacingType,
    align_to_multiples: bool,
) -> tuple[float, float, float]:
    """Return ``max(target_discretization, d_induced)`` with tail quantile bounds.

    ``d_induced`` is span divided by ``max(n_nominal - 1, 1)`` for
    ``n_nominal = max(max_points - 2, 2)``: linear span ``x_max - x_min``, or log-span
    ``log(x_max / x_min)`` for geometric aligned grids. Quantiles come from
    ``dist.ppf(tail_truncation)`` and ``dist.isf(tail_truncation)``.
    """
    if spacing_type not in (SpacingType.LINEAR, SpacingType.GEOMETRIC):
        raise ValueError(f"Unsupported spacing_type: {spacing_type}")
    if max_points < 2:
        raise ValueError(f"max_points must be >= 2, got {max_points}")
    x_min = float(dist.ppf(tail_truncation))
    x_max = float(dist.isf(tail_truncation))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError(
            f"Quantiles not finite for tail_truncation={tail_truncation}: "
            f"x_min={x_min}, x_max={x_max}"
        )
    if x_max <= x_min:
        raise ValueError(
            f"Invalid quantile range for tail_truncation={tail_truncation}: "
            f"x_min={x_min}, x_max={x_max}"
        )
    d = float(target_discretization)
    if d <= 0.0:
        raise ValueError(f"target_discretization must be positive, got {target_discretization}")

    n_nominal = max(max_points - 2, 2)
    denom = max(n_nominal - 1, 1)
    if spacing_type == SpacingType.LINEAR and not align_to_multiples:
        d_induced = float(x_max - x_min) / denom
    elif spacing_type == SpacingType.GEOMETRIC and align_to_multiples:
        d_induced = float(np.log(x_max / x_min)) / denom
    else:
        raise ValueError(
            "Coarsening is only implemented for LINEAR/align_to_multiples=False "
            f"or GEOMETRIC/align_to_multiples=True, got {spacing_type}, {align_to_multiples}"
        )

    return float(max(d, d_induced)), x_min, x_max


def _flip_bound_type(bound_type: BoundType) -> BoundType:
    """Swap DOMINATES <-> IS_DOMINATED for exp-space transforms."""
    return BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
