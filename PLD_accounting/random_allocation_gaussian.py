"""Gaussian-specific random-allocation accounting."""

from __future__ import annotations

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
    PLDRealization,
)
from PLD_accounting.distribution_discretization import (
    aligned_grid_params,
    discretize_continuous_dist,
    discretize_continuous_distribution,
    rediscretize_dist,
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

    if bound_type == BoundType.DOMINATES:
        loss_dist = stats.norm(loc=sigma_inv**2 / 2, scale=sigma_inv)
        effective_step = float(loss_discretization)
        if config.max_grid_mult > 0:
            effective_step, _, _ = _coarsen_discretization_for_tail_quantile_range(
                dist=loss_dist,
                tail_truncation=tail_truncation,
                target_discretization=effective_step,
                max_points=config.max_grid_mult,
                spacing_type=SpacingType.LINEAR,
                align_to_multiples=True,
            )
        base_dist = discretize_continuous_distribution(
            dist=loss_dist,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            spacing_type=SpacingType.LINEAR,
            step=effective_step,
            align_to_multiples=True,
        )
        base_realization = PLDRealization.from_linear_dist(base_dist)
        dual_realization = calc_pld_dual(base_realization)
        return base_realization, negate_reverse_linear_distribution(dual_realization)

    # Lower-bound truncation can create negative-infinity mass, so preserve the
    # dual-first path and discretize the two continuous factors separately.
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
    dual_grid = aligned_grid_params(
        x_min=dual_x_min,
        x_max=dual_x_max,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
        discretization=shared_log_step,
    )
    base_grid = aligned_grid_params(
        x_min=base_x_min,
        x_max=base_x_max,
        spacing_type=SpacingType.GEOMETRIC,
        align_to_multiples=True,
        discretization=shared_log_step,
    )

    dual_factor_dist = discretize_continuous_dist(
        dist=exp_dual,
        grid=dual_grid,
        bound_type=bound_type,
        pmf_min_increment=factor_tail_truncation,
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
        grid=base_grid,
        bound_type=bound_type,
        pmf_min_increment=factor_tail_truncation,
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
        T=num_steps - 1,
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
    base_dist = discretize_continuous_dist(
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
            # An aligned grid needs up to 3 points beyond the span/step interval count
            # (see _coarsen_discretization_for_tail_quantile_range); without them an
            # uncapped grid would be coarsened slightly past loss_discretization.
            grid_size = max(
                int(np.ceil(2 * log_range / loss_discretization)) + 3,
                MIN_GRID_SIZE,
            )
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

    ``d_induced`` is the coarsest step keeping an aligned grid within ``max_points``
    points: the quantile span (``x_max - x_min`` linear, ``log(x_max / x_min)``
    geometric) divided by ``max_points - 3`` (clamped to >= 1).  The 3 reserved
    points cover the fencepost (N points span N-1 steps) plus the up to 2 points
    ``discretize_aligned_range`` adds when rounding both edges outward to whole
    step multiples.  Quantile bounds are ``dist.ppf/isf(tail_truncation)``.
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
    if spacing_type == SpacingType.LINEAR:
        d_induced = float(x_max - x_min) / denom
    elif spacing_type == SpacingType.GEOMETRIC and align_to_multiples:
        d_induced = float(np.log(x_max / x_min)) / denom
    else:
        raise ValueError(
            "Coarsening is only implemented for LINEAR "
            f"or GEOMETRIC/align_to_multiples=True, got {spacing_type}, {align_to_multiples}"
        )

    return float(max(d, d_induced)), x_min, x_max


def _flip_bound_type(bound_type: BoundType) -> BoundType:
    """Swap DOMINATES <-> IS_DOMINATED for exp-space transforms."""
    return BoundType.IS_DOMINATED if bound_type == BoundType.DOMINATES else BoundType.DOMINATES
