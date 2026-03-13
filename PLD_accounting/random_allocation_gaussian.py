"""
Gaussian-specific random-allocation accounting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from PLD_accounting.types import AllocationSchemeConfig, BoundType, ConvolutionMethod, Direction, PrivacyParams, SpacingType
from PLD_accounting.discrete_dist import DiscreteDistBase, GeometricDiscreteDist, LinearDiscreteDist
from PLD_accounting.distribution_utils import compute_bin_width
from PLD_accounting.utils import combine_distributions, log_geometric_to_linear, negate_reverse_linear_distribution
from PLD_accounting.distribution_discretization import MIN_GRID_SIZE, change_spacing_type, discretize_continuous_distribution, discretize_continuous_to_pmf
from PLD_accounting.random_allocation_accounting import *
from PLD_accounting.FFT_convolution import FFT_convolve, FFT_self_convolve


@dataclass
class ConvParams:
    num_steps_per_round: int
    num_rounds: int
    sigma: float
    output_tail_truncation: float
    pre_composition_tail_truncation: float
    discretization_tail_truncation: float
    n_grid_FFT: int
    n_grid_geom: int
    output_loss_discretization: float
    pre_composition_loss_discretization: float
    discretization_loss_discretization: float
    max_grid_FFT: int


def compute_conv_params(*,
    params: PrivacyParams,
    config: AllocationSchemeConfig,
) -> ConvParams:
    """
    Compute convolution parameters from privacy params and scheme config.
    """
    if config.max_grid_FFT <= 0:
        raise ValueError("max_grid_FFT must be positive for FFT convolution")

    num_steps_per_round, num_rounds = decompose_allocation_compositions(
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        num_epochs=params.num_epochs,
    )
    sigma_inv = 1.0 / params.sigma

    output_tail_truncation = config.tail_truncation / 3
    pre_composition_tail_truncation = output_tail_truncation / num_rounds
    discretization_tail_truncation = pre_composition_tail_truncation / num_steps_per_round
    discretization_tail_truncation = max(
        float(discretization_tail_truncation),
        float(np.finfo(float).eps * 1e-10),
    )

    output_loss_discretization = config.loss_discretization / 3
    pre_composition_loss_discretization = output_loss_discretization / np.sqrt(num_rounds)
    discretization_loss_discretization = pre_composition_loss_discretization / (2 * np.ceil(np.log2(num_steps_per_round)) + 1)

    log_range = -stats.norm.ppf(discretization_tail_truncation / 2) * sigma_inv
    n_grid_FFT = int(np.ceil(config.max_grid_FFT / num_steps_per_round))
    n_grid_geom = max(int(2 * log_range / discretization_loss_discretization), MIN_GRID_SIZE)
    if config.max_grid_mult > 0 and n_grid_geom > config.max_grid_mult:
        n_grid_geom = config.max_grid_mult

    return ConvParams(
        num_steps_per_round=num_steps_per_round,
        num_rounds=num_rounds,
        sigma=sigma_inv,
        output_tail_truncation=output_tail_truncation,
        pre_composition_tail_truncation=pre_composition_tail_truncation,
        discretization_tail_truncation=discretization_tail_truncation,
        n_grid_FFT=n_grid_FFT,
        n_grid_geom=n_grid_geom,
        output_loss_discretization=output_loss_discretization,
        pre_composition_loss_discretization=pre_composition_loss_discretization,
        discretization_loss_discretization=discretization_loss_discretization,
        max_grid_FFT=config.max_grid_FFT,
    )


def allocation_PMF_from_gaussian(
    conv_params: ConvParams,
    direction: Direction,
    bound_type: BoundType,
    convolution_method: ConvolutionMethod,
) -> LinearDiscreteDist:
    # Normalize COMBINED to preferred method based on direction
    if convolution_method == ConvolutionMethod.COMBINED:
        if direction == Direction.ADD:
            convolution_method = ConvolutionMethod.GEOM
        elif direction == Direction.REMOVE:
            convolution_method = ConvolutionMethod.FFT
        else:
            raise ValueError(f"Invalid direction: {direction}")

    # Determine which functions to use based on direction
    if direction == Direction.ADD:
        fft_func = _allocation_PMF_add_fft
        geom_func = _allocation_PMF_add_geom
    elif direction == Direction.REMOVE:
        fft_func = _allocation_PMF_remove_fft
        geom_func = _allocation_PMF_remove_geom
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # Compute distribution based on convolution method
    dist: DiscreteDistBase
    if convolution_method == ConvolutionMethod.FFT:
        dist = fft_func(conv_params=conv_params, bound_type=bound_type)
    elif convolution_method == ConvolutionMethod.GEOM:
        dist = geom_func(conv_params=conv_params, bound_type=bound_type)
    elif convolution_method == ConvolutionMethod.BEST_OF_TWO:
        fft_dist = fft_func(conv_params=conv_params, bound_type=bound_type)
        geom_dist = geom_func(conv_params=conv_params, bound_type=bound_type)
        dist = combine_distributions(dist_1=fft_dist, dist_2=geom_dist, bound_type=bound_type)
    else:
        raise ValueError(f"Invalid convolution_method: {convolution_method}")

    return finalize_allocation_composition(
        round_dist=dist,
        num_rounds=conv_params.num_rounds,
        pre_composition_loss_discretization=conv_params.pre_composition_loss_discretization,
        pre_composition_tail_truncation=conv_params.pre_composition_tail_truncation,
        output_loss_discretization=conv_params.output_loss_discretization,
        output_tail_truncation=conv_params.output_tail_truncation,
        bound_type=bound_type,
    )


# =============================================================================
# Internal Functions
# =============================================================================


def _allocation_PMF_remove_fft(*,
    conv_params: ConvParams,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    num_steps = conv_params.num_steps_per_round
    if num_steps < 2:
        raise ValueError("REMOVE direction requires at least two steps per round")

    sigma = conv_params.sigma
    discretization_tail_truncation = conv_params.discretization_tail_truncation / 2
    pre_composition_tail_truncation = conv_params.pre_composition_tail_truncation / 3

    lower_norm_mean = -sigma**2 / 2 - np.log(num_steps)
    upper_norm_mean = sigma**2 / 2 - np.log(num_steps)
    lower_shift = np.exp(lower_norm_mean + sigma**2 / 2)
    upper_shift = np.exp(upper_norm_mean + sigma**2 / 2)

    exp_L_QP_neg = stats.lognorm(s=sigma, scale=np.exp(lower_norm_mean))
    base_dist_lower = discretize_continuous_distribution(
        dist=exp_L_QP_neg,
        tail_truncation=discretization_tail_truncation,
        bound_type=bound_type,
        spacing_type=SpacingType.LINEAR,
        n_grid=conv_params.n_grid_FFT,
        align_to_multiples=False,
    )
    assert isinstance(base_dist_lower, LinearDiscreteDist)
    base_dist_lower.x_min -= lower_shift
    conv_dist_lower = FFT_self_convolve(
        dist=base_dist_lower,
        T=num_steps - 1,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=bound_type,
        use_direct=True,
    )

    exp_L_PQ = stats.lognorm(s=sigma, scale=np.exp(upper_norm_mean))
    upper_grid = conv_dist_lower.x_array + upper_shift
    x_max_target = exp_L_PQ.isf(discretization_tail_truncation)
    p_right = exp_L_PQ.sf(upper_grid[-1])
    p_right_threshold = conv_params.output_tail_truncation / 10
    if np.isfinite(x_max_target) and upper_grid[-1] < x_max_target and p_right > p_right_threshold:
        if upper_grid.size > 1:
            step = compute_bin_width(upper_grid)
            n_extra = int(np.ceil((x_max_target - upper_grid[-1]) / step))
            if n_extra > 0:
                upper_grid = np.concatenate(
                    [upper_grid, upper_grid[-1] + step * np.arange(1, n_extra + 1)]
                )

    base_dist_upper = discretize_continuous_to_pmf(
        dist=exp_L_PQ,
        x_array=upper_grid,
        bound_type=bound_type,
        PMF_min_increment=discretization_tail_truncation,
        spacing_type=SpacingType.LINEAR,
    )
    assert isinstance(base_dist_upper, LinearDiscreteDist)
    base_dist_upper.x_min -= upper_shift

    conv_dist_raw = FFT_convolve(
        dist_1=conv_dist_lower,
        dist_2=base_dist_upper,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=bound_type,
    )
    conv_dist_raw.x_min += (num_steps - 1) * lower_shift + upper_shift
    exp_geom = change_spacing_type(
        dist=conv_dist_raw,
        tail_truncation=0.0,
        loss_discretization=conv_params.pre_composition_loss_discretization,
        spacing_type=SpacingType.GEOMETRIC,
        bound_type=bound_type,
    )
    assert isinstance(exp_geom, GeometricDiscreteDist)
    return log_geometric_to_linear(exp_geom)


def _allocation_PMF_remove_geom(*,
    conv_params: ConvParams,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    num_steps = conv_params.num_steps_per_round
    if num_steps < 2:
        raise ValueError("REMOVE direction requires at least two steps per round")

    sigma = conv_params.sigma
    discretization_tail_truncation = conv_params.discretization_tail_truncation / 2
    pre_composition_tail_truncation = conv_params.pre_composition_tail_truncation / 3

    lower_norm_mean = -sigma**2 / 2
    upper_norm_mean = sigma**2 / 2
    exp_L_QP_neg = stats.lognorm(s=sigma, scale=np.exp(lower_norm_mean))
    exp_L_PQ = stats.lognorm(s=sigma, scale=np.exp(upper_norm_mean))

    lower_x_min = exp_L_QP_neg.ppf(discretization_tail_truncation)
    lower_x_max = exp_L_QP_neg.isf(discretization_tail_truncation)
    upper_x_min = exp_L_PQ.ppf(discretization_tail_truncation)
    upper_x_max = exp_L_PQ.isf(discretization_tail_truncation)
    log_span = max(np.log(lower_x_max / lower_x_min), np.log(upper_x_max / upper_x_min))
    shared_log_step = log_span / (conv_params.n_grid_geom - 1)
    base_dist_lower = discretize_continuous_to_pmf(
        dist=exp_L_QP_neg,
        x_array=_build_shared_geometric_grid(
            dist=exp_L_QP_neg,
            tail_truncation=discretization_tail_truncation,
            log_step=shared_log_step,
        ),
        bound_type=bound_type,
        PMF_min_increment=discretization_tail_truncation,
        spacing_type=SpacingType.GEOMETRIC,
    )
    assert isinstance(base_dist_lower, GeometricDiscreteDist)
    base_dist_upper = discretize_continuous_to_pmf(
        dist=exp_L_PQ,
        x_array=_build_shared_geometric_grid(
            dist=exp_L_PQ,
            tail_truncation=discretization_tail_truncation,
            log_step=shared_log_step,
        ),
        bound_type=bound_type,
        PMF_min_increment=discretization_tail_truncation,
        spacing_type=SpacingType.GEOMETRIC,
    )
    assert isinstance(base_dist_upper, GeometricDiscreteDist)
    
    lower_loss_factor = log_geometric_to_linear(base_dist_lower)
    upper_loss_factor = log_geometric_to_linear(base_dist_upper)
    return log_mean_exp_remove(
        lower_loss_factor=lower_loss_factor,
        upper_loss_factor=upper_loss_factor,
        num_steps=num_steps,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=bound_type,
    )


def _allocation_PMF_add_fft(*,
    conv_params: ConvParams,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    num_steps = conv_params.num_steps_per_round
    sigma = conv_params.sigma
    discretization_tail_truncation = conv_params.discretization_tail_truncation
    pre_composition_tail_truncation = conv_params.pre_composition_tail_truncation / 2
    exp_bound_type = BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES

    norm_mean = -sigma**2 / 2 - np.log(num_steps)
    base_dist = discretize_continuous_distribution(
        dist=stats.lognorm(s=sigma, scale=np.exp(norm_mean)),
        tail_truncation=discretization_tail_truncation,
        bound_type=exp_bound_type,
        spacing_type=SpacingType.LINEAR,
        n_grid=conv_params.n_grid_FFT,
        align_to_multiples=False,
    )
    assert isinstance(base_dist, LinearDiscreteDist)
    conv_dist = FFT_self_convolve(
        dist=base_dist,
        T=num_steps,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=exp_bound_type,
        use_direct=True,
    )
    exp_geom = change_spacing_type(
        dist=conv_dist,
        tail_truncation=0.0,
        loss_discretization=conv_params.pre_composition_loss_discretization,
        spacing_type=SpacingType.GEOMETRIC,
        bound_type=exp_bound_type,
    )
    assert isinstance(exp_geom, GeometricDiscreteDist)
    log_dist = log_geometric_to_linear(exp_geom)
    return negate_reverse_linear_distribution(log_dist)


def _allocation_PMF_add_geom(*,
    conv_params: ConvParams,
    bound_type: BoundType,
) -> LinearDiscreteDist:
    num_steps = conv_params.num_steps_per_round
    sigma = conv_params.sigma
    discretization_tail_truncation = conv_params.discretization_tail_truncation
    pre_composition_tail_truncation = conv_params.pre_composition_tail_truncation / 2
    exp_bound_type = BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES

    norm_mean = -sigma**2 / 2
    base_dist = discretize_continuous_distribution(
        dist=stats.lognorm(s=sigma, scale=np.exp(norm_mean)),
        tail_truncation=discretization_tail_truncation,
        bound_type=exp_bound_type,
        spacing_type=SpacingType.GEOMETRIC,
        n_grid=conv_params.n_grid_geom,
        align_to_multiples=True,
    )
    assert isinstance(base_dist, GeometricDiscreteDist)
    add_loss_factor = log_geometric_to_linear(base_dist)
    return log_mean_exp_add(
        add_loss_factor=add_loss_factor,
        num_steps=num_steps,
        tail_truncation=pre_composition_tail_truncation,
        bound_type=bound_type,
    )


def _build_shared_geometric_grid(*,
    dist: stats.rv_continuous,
    tail_truncation: float,
    log_step: float,
) -> np.ndarray:
    """
    Build a geometric grid snapped to a shared log-space lattice.
    """
    x_min = dist.ppf(tail_truncation)
    x_max = dist.isf(tail_truncation)
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError(f"Quantiles not finite: x_min={x_min}, x_max={x_max}")
    if x_min <= 0.0:
        raise ValueError(f"Geometric spacing requires positive values, got x_min={x_min}, x_max={x_max}")
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
