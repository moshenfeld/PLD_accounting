"""Public API surface for random-allocation accounting."""

from __future__ import annotations

from functools import partial

from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.adaptive_random_allocation import (
    optimize_allocation_epsilon_range,
    require_target_accuracy,
)
from PLD_accounting.discrete_dist import PLDRealization, require_pld_realization
from PLD_accounting.mechanisms import gaussian_loss_law
from PLD_accounting.random_allocation_accounting import (
    allocation_plds,
    compose_full_pld,
)
from PLD_accounting.random_allocation_base import (
    allocation_plds_base,
    loss_discretization_counts,
)
from PLD_accounting.random_allocation_continuous import continuous_base_distributions
from PLD_accounting.random_allocation_realization import realization_base_distributions
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    PrivacyParams,
    require_bound_type,
    resolve_projection_method,
)
from PLD_accounting.validation import (
    require_allocation_counts,
    require_open_unit_interval,
    require_positive_real,
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

    The interval brackets the floor/ceil composition. With several selections,
    its lower endpoint need not lower-bound uniform k-out-of-t allocation. This
    adaptive helper uses ``GEOM`` with rounding for both bounds.

    Each pass builds an upper and a lower PLD with one configuration and keeps the
    best bounds seen so far, then halves ``loss_discretization`` and divides
    ``tail_truncation`` by ten, for at most ``MAX_ITERATIONS`` passes. The first
    ``tail_truncation`` is ``delta / 10``, within ``[1e-20, 1e-4]``. A run that does
    not meet the target warns
    with a ``RuntimeWarning`` and returns its best bounds.

    Args:
        delta: Target delta for the epsilon query.
        sigma: Noise standard deviation of the Gaussian mechanism (L2 sensitivity 1).
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        epsilon_accuracy: A positive value is the absolute gap ``upper - lower`` to
            reach, and the first ``loss_discretization`` is half of it, within
            ``[1e-6, 0.1]``, so a small gap is expensive. A negative value stops
            when ``upper / lower <= 1 + DEFAULT_RELATIVE_ACCURACY`` (``0.1``) and
            seeds the first step from a Poisson-subsampling estimate. Zero, a gap no
            pass can reach, is rejected.

    Returns:
        A tuple ``(upper_bound, lower_bound)``.

    """
    require_target_accuracy(value=epsilon_accuracy, name="epsilon_accuracy")
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        delta=delta,
    )
    params.get_delta()

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
        config: Discretization and convolution configuration. Its
            ``projection_method`` defaults to rounding with
            ``ConvolutionMethod.GEOM`` and to connect-the-dots (CtD) with
            ``ConvolutionMethod.MS_FFT``. Rounding supports both bounds; CtD
            supports ``BoundType.DOMINATES`` only, so a lower bound with
            ``MS_FFT`` needs ``ProjectionMethod.STOCH_DOM`` set explicitly.
        bound_type: Numerical bound on the floor/ceil composition. With several
            selections, its lower bound need not lower-bound uniform allocation.

    Returns:
        The epsilon value corresponding to ``params.delta``.

    """
    delta = params.get_delta()
    require_bound_type(value=bound_type)

    full_pld = gaussian_allocation_pld(
        params=params,
        config=config,
        bound_type=bound_type,
    )
    return float(full_pld.get_epsilon_for_delta(delta))


def gaussian_allocation_delta_configurable(
    *,
    params: PrivacyParams,
    config: AllocationSchemeConfig,
    bound_type: BoundType = BoundType.DOMINATES,
) -> float:
    """Compute delta for Gaussian random-allocation with configurable accuracy.

    Args:
        params: Privacy parameters with ``params.epsilon`` set.
        config: Discretization and convolution configuration. Its
            ``projection_method`` defaults to rounding with
            ``ConvolutionMethod.GEOM`` and to connect-the-dots (CtD) with
            ``ConvolutionMethod.MS_FFT``. Rounding supports both bounds; CtD
            supports ``BoundType.DOMINATES`` only, so a lower bound with
            ``MS_FFT`` needs ``ProjectionMethod.STOCH_DOM`` set explicitly.
        bound_type: Numerical bound on the floor/ceil composition. With several
            selections, its lower bound need not lower-bound uniform allocation.

    Returns:
        The delta value corresponding to ``params.epsilon``.

    """
    epsilon = params.get_epsilon()
    require_bound_type(value=bound_type)

    full_pld = gaussian_allocation_pld(
        params=params,
        config=config,
        bound_type=bound_type,
    )
    return float(full_pld.get_delta_for_epsilon(epsilon))


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
        config: Discretization and convolution configuration. Its
            ``projection_method`` defaults to rounding with
            ``ConvolutionMethod.GEOM`` and to connect-the-dots (CtD) with
            ``ConvolutionMethod.MS_FFT``. Rounding supports both bounds; CtD
            supports ``BoundType.DOMINATES`` only, so a lower bound with
            ``MS_FFT`` needs ``ProjectionMethod.STOCH_DOM`` set explicitly.
        bound_type: Numerical bound on the floor/ceil composition. With several
            selections, its lower bound need not lower-bound uniform allocation.

    Returns:
        A ``dp_accounting`` ``PrivacyLossDistribution`` for both privacy directions.

    """
    require_bound_type(value=bound_type)
    projection_method = resolve_projection_method(config=config, bound_type=bound_type)
    law = gaussian_loss_law(params.sigma)
    base_distributions_creation = partial(
        continuous_base_distributions,
        law=law,
        dual=law,
        convolution_method=config.convolution_method,
        max_grid_mult=config.max_grid_mult,
        max_cells_per_band=config.max_cells_per_band,
    )
    compute_base_plds = partial(
        allocation_plds_base,
        base_distributions_creation=base_distributions_creation,
        projection_method=projection_method,
        convolution_method=config.convolution_method,
        max_closing_grid=config.max_closing_grid,
    )
    base_loss_discretization_counts = partial(
        loss_discretization_counts,
        projection_method=projection_method,
        convolution_method=config.convolution_method,
    )
    remove_dist, add_dist = allocation_plds(
        compute_base_plds=compute_base_plds,
        base_loss_discretization_counts=base_loss_discretization_counts,
        num_steps=params.num_steps,
        num_selected=params.num_selected,
        num_epochs=params.num_epochs,
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
        projection_method=projection_method,
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

    Bounds refer to the supplied PLDs and their floor/ceil composition. A lower
    bound need not lower-bound a mechanism dominated by those inputs, or uniform
    k-out-of-t allocation when there are several selections.

    Args:
        delta: Target delta for the epsilon query.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Remove-direction PLD realization. A ``PLDRealization``
            is required; a lower-bound mechanism discretization is rejected. On
            ``GEOM`` its grid step is a floor on the loss step of its one-step
            factors, and a realization coarser than a stage's share of
            ``loss_discretization`` warns.
        add_realization: Add-direction PLD realization, with the same requirements.
        config: Discretization and convolution configuration. Its
            ``projection_method`` defaults to rounding with
            ``ConvolutionMethod.GEOM`` and to connect-the-dots (CtD) with
            ``ConvolutionMethod.MS_FFT``. Rounding supports both bounds; CtD
            supports ``BoundType.DOMINATES`` only, so a lower bound with
            ``MS_FFT`` needs ``ProjectionMethod.STOCH_DOM`` set explicitly.
        bound_type: Whether to compute a dominating or dominated bound.

    Returns:
        The epsilon value corresponding to the given delta.

    """
    require_open_unit_interval(value=delta, name="delta")
    require_allocation_counts(num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs)
    require_bound_type(value=bound_type)
    require_pld_realization(dist=remove_realization, name="remove_realization")
    require_pld_realization(dist=add_realization, name="add_realization")

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

    Bounds refer to the supplied PLDs and their floor/ceil composition. A lower
    bound need not lower-bound a mechanism dominated by those inputs, or uniform
    k-out-of-t allocation when there are several selections.

    Args:
        epsilon: Target epsilon for the delta query.
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Remove-direction PLD realization. A ``PLDRealization``
            is required; a lower-bound mechanism discretization is rejected. On
            ``GEOM`` its grid step is a floor on the loss step of its one-step
            factors, and a realization coarser than a stage's share of
            ``loss_discretization`` warns.
        add_realization: Add-direction PLD realization, with the same requirements.
        config: Discretization and convolution configuration. Its
            ``projection_method`` defaults to rounding with
            ``ConvolutionMethod.GEOM`` and to connect-the-dots (CtD) with
            ``ConvolutionMethod.MS_FFT``. Rounding supports both bounds; CtD
            supports ``BoundType.DOMINATES`` only, so a lower bound with
            ``MS_FFT`` needs ``ProjectionMethod.STOCH_DOM`` set explicitly.
        bound_type: Whether to compute a dominating or dominated bound.

    Returns:
        The delta value corresponding to the given epsilon.

    """
    require_positive_real(value=epsilon, name="epsilon")
    require_allocation_counts(num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs)
    require_bound_type(value=bound_type)
    require_pld_realization(dist=remove_realization, name="remove_realization")
    require_pld_realization(dist=add_realization, name="add_realization")

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

    Both realizations are validated and prepared. With CtD, a component with more
    than one step sums only the prepared ADD factors and takes REMOVE as the PLD
    dual of the ADD result, so ``remove_realization`` is not read there. That
    REMOVE bound holds when ``add_realization`` dominates the mechanism's ADD PLD at
    every epsilon, negative ones included, as a dominating discretization does.
    Rounding sums each direction's own factors, and a one-step component keeps
    each direction's own prepared input.

    Bounds refer to the supplied PLDs and their floor/ceil composition. A lower
    bound need not lower-bound a mechanism dominated by those inputs, or uniform
    k-out-of-t allocation when there are several selections.

    Args:
        num_steps: Total number of random-allocation steps.
        num_selected: Number of selections per epoch.
        num_epochs: Number of epochs.
        remove_realization: Remove-direction PLD realization. A ``PLDRealization``
            is required; a lower-bound mechanism discretization is rejected. On
            ``GEOM`` its grid step is a floor on the loss step of its one-step
            factors, and a realization coarser than a stage's share of
            ``loss_discretization`` warns.
        add_realization: Add-direction PLD realization, with the same requirements.
        config: Discretization and convolution configuration. Its
            ``projection_method`` defaults to rounding with
            ``ConvolutionMethod.GEOM`` and to connect-the-dots (CtD) with
            ``ConvolutionMethod.MS_FFT``. Rounding supports both bounds; CtD
            supports ``BoundType.DOMINATES`` only, so a lower bound with
            ``MS_FFT`` needs ``ProjectionMethod.STOCH_DOM`` set explicitly.
        bound_type: Whether to compute a dominating or dominated discretized bound.

    Returns:
        A ``dp_accounting`` ``PrivacyLossDistribution`` for the composed realization.

    """
    require_allocation_counts(num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs)
    require_bound_type(value=bound_type)
    require_pld_realization(dist=remove_realization, name="remove_realization")
    require_pld_realization(dist=add_realization, name="add_realization")
    projection_method = resolve_projection_method(config=config, bound_type=bound_type)
    base_distributions_creation = partial(
        realization_base_distributions,
        remove_realization=remove_realization,
        add_realization=add_realization,
        convolution_method=config.convolution_method,
        max_grid_mult=config.max_grid_mult,
        max_cells_per_band=config.max_cells_per_band,
    )
    compute_base_plds = partial(
        allocation_plds_base,
        base_distributions_creation=base_distributions_creation,
        projection_method=projection_method,
        convolution_method=config.convolution_method,
        max_closing_grid=config.max_closing_grid,
    )
    base_loss_discretization_counts = partial(
        loss_discretization_counts,
        projection_method=projection_method,
        convolution_method=config.convolution_method,
    )
    remove_dist, add_dist = allocation_plds(
        compute_base_plds=compute_base_plds,
        base_loss_discretization_counts=base_loss_discretization_counts,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=bound_type,
        projection_method=projection_method,
    )
    return compose_full_pld(
        remove_dist=remove_dist,
        add_dist=add_dist,
        bound_type=bound_type,
    )
