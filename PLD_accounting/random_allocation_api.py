"""Public API surface for random-allocation accounting."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.adaptive_random_allocation import (
    optimize_allocation_epsilon_range,
)
from PLD_accounting.discrete_dist import DenseDiscreteDist, PLDRealization
from PLD_accounting.random_allocation_accounting import (
    add_geometric_loss_discretization_count,
    allocation_directional_pld,
    compose_full_pld,
    geometric_allocation_pld_base_add,
    geometric_allocation_pld_base_remove,
    remove_geometric_loss_discretization_count,
)
from PLD_accounting.random_allocation_gaussian import (
    gaussian_allocation_pld_core_and_count,
)
from PLD_accounting.random_allocation_realization import (
    realization_add_base_distribution,
    realization_remove_base_distributions,
)
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
    require_allocation_config,
    require_bound_type,
    require_direction,
    require_privacy_params,
)
from PLD_accounting.utils import combine_best_of_two_plds
from PLD_accounting.validation import (
    require_allocation_counts,
    require_finite_real,
    require_open_unit_interval,
    require_positive_real,
    require_type,
)

# =============================================================================
# Gaussian-Based Random Allocation API
# =============================================================================


def gaussian_allocation_epsilon_range(
    *,
    delta: float,
    sigma: float,
    num_steps: int,
    num_selected: int = 1,
    num_epochs: int = 1,
    epsilon_accuracy: float = -1.0,
) -> tuple[float, float]:
    """Compute epsilon bounds for Gaussian random-allocation (adaptive refinement).

    Args:
        delta: Target delta for the epsilon query.
        sigma: Gaussian noise scale.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        epsilon_accuracy: Nonnegative value is an absolute gap. A negative value
            stops when ``upper / lower <= 1 + DEFAULT_RELATIVE_ACCURACY`` and is
            reported back verbatim rather than resolved to a target.

    Returns:
        A tuple ``(upper_bound, lower_bound)``.

    """
    require_finite_real(value=epsilon_accuracy, name="epsilon_accuracy")
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        delta=delta,
    )
    params.require_delta()

    result = optimize_allocation_epsilon_range(
        params=params,
        target_accuracy=epsilon_accuracy,
        pld_builder=gaussian_allocation_pld,
    )
    return result.upper_bound, result.lower_bound


def gaussian_allocation_epsilon_configurable(
    *,
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """Compute epsilon for Gaussian random-allocation with configurable accuracy.

    Args:
        params: Privacy parameters with ``params.delta`` set.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated bound.
            ``BoundType.IS_DOMINATED`` is supported only with
            ``ConvolutionMethod.GEOM``; the FFT-based methods round losses up
            and cannot produce a valid lower bound, so they raise ``ValueError``.

    Returns:
        The epsilon value corresponding to ``params.delta``.

    """
    require_privacy_params(value=params)
    params.require_delta()
    require_allocation_config(value=config)
    require_bound_type(value=bound_type)

    full_pld = gaussian_allocation_pld(
        params=params,
        config=config,
        bound_type=bound_type,
    )
    return float(full_pld.get_epsilon_for_delta(params.delta))


def gaussian_allocation_delta_configurable(
    *,
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """Compute delta for Gaussian random-allocation with configurable accuracy.

    Args:
        params: Privacy parameters with ``params.epsilon`` set.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated bound.
            ``BoundType.IS_DOMINATED`` is supported only with
            ``ConvolutionMethod.GEOM``; the FFT-based methods round losses up
            and cannot produce a valid lower bound, so they raise ``ValueError``.

    Returns:
        The delta value corresponding to ``params.epsilon``.

    """
    require_privacy_params(value=params)
    params.require_epsilon()
    require_allocation_config(value=config)
    require_bound_type(value=bound_type)

    full_pld = gaussian_allocation_pld(
        params=params,
        config=config,
        bound_type=bound_type,
    )
    return float(full_pld.get_delta_for_epsilon(params.epsilon))


def gaussian_allocation_directional_pld(
    *,
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    direction: Direction,
    bound_type: BoundType = BoundType.DOMINATES,
) -> DenseDiscreteDist:
    """Compute one directional PLD for Gaussian random-allocation.

    ``BoundType.IS_DOMINATED`` is supported only by the GEOM convolution
    method, for both directions.
    """
    require_privacy_params(value=params)
    require_allocation_config(value=config)
    require_bound_type(value=bound_type)
    require_direction(value=direction)

    if bound_type == BoundType.IS_DOMINATED and config.convolution_method != ConvolutionMethod.GEOM:
        raise ValueError(
            "BoundType.IS_DOMINATED is supported only with "
            f"ConvolutionMethod.GEOM, got {config.convolution_method}"
        )

    if config.convolution_method == ConvolutionMethod.BEST_OF_TWO:
        geom_dist = gaussian_allocation_directional_pld(
            params=params,
            config=replace(config, convolution_method=ConvolutionMethod.GEOM),
            direction=direction,
            bound_type=bound_type,
        )
        fft_dist = gaussian_allocation_directional_pld(
            params=params,
            config=replace(config, convolution_method=ConvolutionMethod.FFT),
            direction=direction,
            bound_type=bound_type,
        )
        return combine_best_of_two_plds(
            dist_1=geom_dist,
            dist_2=fft_dist,
            bound_type=bound_type,
        )

    compute_base_pld, loss_discretization_count = gaussian_allocation_pld_core_and_count(
        direction=direction,
        sigma=params.sigma,
        config=config,
    )
    return allocation_directional_pld(
        compute_base_pld=compute_base_pld,
        base_loss_discretization_count=loss_discretization_count,
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        num_epochs=params.num_epochs,
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )


def gaussian_allocation_pld(
    *,
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Compute upper / lower PLD for random-allocation with the Gaussian mechanism.

    Args:
        params: Privacy parameters describing noise scale, number of steps,
            and optional delta/epsilon query target.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated discretized bound.
            ``BoundType.IS_DOMINATED`` is supported only with
            ``ConvolutionMethod.GEOM``; the FFT-based methods round losses up
            and cannot produce a valid lower bound, so they raise ``ValueError``.

    Returns:
        A ``dp_accounting`` ``PrivacyLossDistribution`` for both privacy directions.

    """
    require_privacy_params(value=params)
    require_allocation_config(value=config)
    require_bound_type(value=bound_type)
    remove_dist = gaussian_allocation_directional_pld(
        params=params,
        config=config,
        direction=Direction.REMOVE,
        bound_type=bound_type,
    )
    add_dist = gaussian_allocation_directional_pld(
        params=params,
        config=config,
        direction=Direction.ADD,
        bound_type=bound_type,
    )
    return compose_full_pld(
        remove_dist=remove_dist,
        add_dist=add_dist,
        bound_type=bound_type,
    )


# =============================================================================
# PLD Realization-Based Random Allocation API
# =============================================================================


def general_allocation_epsilon(
    *,
    delta: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    remove_realization: PLDRealization,
    add_realization: PLDRealization,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """Compute epsilon from explicit PLD realizations.

    Args:
        delta: Target delta for the epsilon query.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Explicit remove-direction PLD realization.
        add_realization: Explicit add-direction PLD realization.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated bound.

    Returns:
        The epsilon value corresponding to the given delta.

    Notes:
        Supports only the GEOM convolution method.

    """
    require_open_unit_interval(value=delta, name="delta")
    require_allocation_counts(num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs)
    require_type(value=remove_realization, expected_type=PLDRealization, name="remove_realization")
    require_type(value=add_realization, expected_type=PLDRealization, name="add_realization")
    require_allocation_config(value=config)
    require_bound_type(value=bound_type)

    pld = general_allocation_pld(
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        remove_realization=remove_realization,
        add_realization=add_realization,
        config=config,
        bound_type=bound_type,
    )
    return float(pld.get_epsilon_for_delta(delta))


def general_allocation_delta(
    *,
    epsilon: float,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    remove_realization: PLDRealization,
    add_realization: PLDRealization,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """Compute delta from explicit PLD realizations.

    Args:
        epsilon: Target epsilon for the delta query.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Explicit remove-direction PLD realization.
        add_realization: Explicit add-direction PLD realization.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated bound.

    Returns:
        The delta value corresponding to the given epsilon.

    Notes:
        Supports only the GEOM convolution method.

    """
    require_positive_real(value=epsilon, name="epsilon")
    require_allocation_counts(num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs)
    require_type(value=remove_realization, expected_type=PLDRealization, name="remove_realization")
    require_type(value=add_realization, expected_type=PLDRealization, name="add_realization")
    require_allocation_config(value=config)
    require_bound_type(value=bound_type)

    pld = general_allocation_pld(
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        remove_realization=remove_realization,
        add_realization=add_realization,
        config=config,
        bound_type=bound_type,
    )
    return float(pld.get_delta_for_epsilon(epsilon))


def general_allocation_pld(
    *,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
    remove_realization: PLDRealization,
    add_realization: PLDRealization,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> privacy_loss_distribution.PrivacyLossDistribution:
    """Build a random-allocation PLD from explicit PLD realizations.

    Args:
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Explicit remove-direction PLD realization.
        add_realization: Explicit add-direction PLD realization.
        config: Discretization and convolution configuration.
        bound_type: Whether to compute a dominating or dominated discretized bound.

    Returns:
        A ``dp_accounting`` ``PrivacyLossDistribution`` for the composed realization.

    Notes:
        Supports only the GEOM convolution method.

    """
    require_allocation_counts(num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs)
    require_type(value=remove_realization, expected_type=PLDRealization, name="remove_realization")
    require_type(value=add_realization, expected_type=PLDRealization, name="add_realization")
    require_allocation_config(value=config)
    require_bound_type(value=bound_type)
    if config.convolution_method != ConvolutionMethod.GEOM:
        raise ValueError(
            "PLD realization-based allocation requires geometric convolution. "
            f"Got convolution_method={config.convolution_method}. "
            "Use ConvolutionMethod.GEOM."
        )

    compute_base_pld_remove = partial(
        geometric_allocation_pld_base_remove,
        base_distributions_creation=partial(
            realization_remove_base_distributions,
            realization=remove_realization,
            max_grid_mult=config.max_grid_mult,
        ),
    )
    remove_dist = allocation_directional_pld(
        compute_base_pld=compute_base_pld_remove,
        base_loss_discretization_count=remove_geometric_loss_discretization_count,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )

    compute_base_pld_add = partial(
        geometric_allocation_pld_base_add,
        base_distributions_creation=partial(
            realization_add_base_distribution,
            realization=add_realization,
            max_grid_mult=config.max_grid_mult,
        ),
    )
    add_dist = allocation_directional_pld(
        compute_base_pld=compute_base_pld_add,
        base_loss_discretization_count=add_geometric_loss_discretization_count,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
    )
    return compose_full_pld(
        remove_dist=remove_dist,
        add_dist=add_dist,
        bound_type=bound_type,
    )
