"""
Integration tests for PLD realization-based allocation (Phase 3).

Tests cover:
1. Definition-3.1 validation and linear-grid structure
2. Loss/exp-space transforms
3. Gaussian reference comparisons
4. Non-Gaussian custom realizations
5. Composition and error handling
"""

import numpy as np
import pytest
from scipy import stats

from PLD_accounting import (
    AllocationSchemeConfig,
    BoundType,
    Direction,
    PrivacyParams,
    gaussian_allocation_PLD,
    general_allocation_PLD,
)
from PLD_accounting.adaptive_random_allocation import optimize_allocation_epsilon_range
from PLD_accounting.discrete_dist import LinearDiscreteDist, PLDRealization
from PLD_accounting.distribution_discretization import change_spacing_type
from PLD_accounting.types import ConvolutionMethod, SpacingType
from PLD_accounting.utils import (
    exp_linear_to_geometric,
    log_geometric_to_linear,
    negate_reverse_linear_distribution,
)


def _realization_from_loss_values(
    loss_values: np.ndarray,
    probabilities: np.ndarray,
    p_loss_inf: float = 0.0,
    p_loss_neg_inf: float = 0.0,
) -> PLDRealization:
    dist = LinearDiscreteDist.from_x_array(
        x_array=loss_values,
        PMF_array=probabilities,
        p_neg_inf=p_loss_neg_inf,
        p_pos_inf=p_loss_inf,
    )
    return PLDRealization(
        x_min=dist.x_min,
        x_gap=dist.x_gap,
        PMF_array=dist.PMF_array,
        p_loss_inf=dist.p_pos_inf,
        p_loss_neg_inf=dist.p_neg_inf,
    )


def _gaussian_realization(
    sigma: float,
    direction: Direction,
    num_points: int = 4000,
    num_std: float = 8.0,
) -> PLDRealization:
    """Build a single-step Gaussian PLD realization on a wide linear grid."""
    if direction not in (Direction.REMOVE, Direction.ADD):
        raise ValueError(f"Unsupported direction {direction}")

    sigma_loss = 1.0 / sigma
    mean = 1.0 / (2.0 * sigma * sigma)
    loss_values = np.linspace(
        mean - num_std * sigma_loss,
        mean + num_std * sigma_loss,
        num_points,
    )
    loss_gap = loss_values[1] - loss_values[0]
    probabilities = stats.norm.pdf(loss_values, loc=mean, scale=sigma_loss) * loss_gap
    probabilities = probabilities / probabilities.sum()
    return _realization_from_loss_values(loss_values=loss_values, probabilities=probabilities)


def _simple_valid_realization() -> PLDRealization:
    return PLDRealization(
        x_min=0.0,
        x_gap=1.0,
        PMF_array=np.array([0.7, 0.2, 0.1]),
    )


class TestRealizationValidation:
    def test_pld_realization_is_linear_discrete_dist(self):
        realization = _simple_valid_realization()
        assert isinstance(realization, LinearDiscreteDist)
        assert np.isclose(realization.x_gap, 1.0)

    def test_valid_simple_realization(self):
        _simple_valid_realization().validate_pld_realization()

    def test_from_linear_dist_constructor(self):
        linear = LinearDiscreteDist(
            x_min=0.0,
            x_gap=1.0,
            PMF_array=np.array([0.7, 0.2, 0.1]),
        )
        realization = PLDRealization.from_linear_dist(linear)
        assert isinstance(realization, PLDRealization)
        np.testing.assert_allclose(realization.x_array, linear.x_array, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(realization.PMF_array, linear.PMF_array, rtol=1e-12, atol=1e-12)

    def test_non_linear_grid_is_rejected(self):
        with pytest.raises(ValueError, match="non-uniform bin widths"):
            _realization_from_loss_values(
                loss_values=np.array([0.0, 0.5, 2.0]),
                probabilities=np.array([0.7, 0.2, 0.1]),
            )

    def test_mass_conservation_violation(self):
        with pytest.raises(ValueError, match="MASS CONSERVATION ERROR"):
            _realization_from_loss_values(
                loss_values=np.array([0.0, 1.0, 2.0]),
                probabilities=np.array([0.5, 0.3, 0.1]),
            )

    def test_negative_probability(self):
        with pytest.raises(ValueError, match="PMF must be nonnegative"):
            _realization_from_loss_values(
                loss_values=np.array([0.0, 1.0, 2.0]),
                probabilities=np.array([0.6, 0.5, -0.1]),
            )

    def test_negative_infinity_mass_is_rejected(self):
        with pytest.raises(ValueError, match="DOMINATES bound_type requires p_neg_inf=0"):
            _realization_from_loss_values(
                loss_values=np.array([0.0, 1.0]),
                probabilities=np.array([0.6, 0.3]),
                p_loss_neg_inf=0.1,
            )

    def test_exp_moment_bound_violation(self):
        with pytest.raises(ValueError, match="E\\[exp\\(-L\\)\\]"):
            _realization_from_loss_values(
                loss_values=np.array([-10.0, -5.0, 0.0]),
                probabilities=np.array([0.5, 0.3, 0.2]),
            )


class TestExpMomentComputation:
    def test_exp_moment_simple(self):
        realization = _simple_valid_realization()
        moment = sum(
            float(p) * float(np.exp(-loss))
            for p, loss in zip(realization.probabilities, realization.loss_values)
        )
        expected = 0.7 + 0.2 * np.exp(-1.0) + 0.1 * np.exp(-2.0)
        assert abs(moment - expected) < 1e-10

    def test_exp_moment_with_positive_infinity_mass(self):
        realization = _realization_from_loss_values(
            loss_values=np.array([0.0, 1.0]),
            probabilities=np.array([0.6, 0.3]),
            p_loss_inf=0.1,
        )
        moment = sum(
            float(p) * float(np.exp(-loss))
            for p, loss in zip(realization.probabilities, realization.loss_values)
        )
        expected = 0.6 + 0.3 * np.exp(-1.0)
        assert abs(moment - expected) < 1e-10

    def test_exp_moment_with_negative_infinity_mass_is_infinite(self):
        with pytest.raises(ValueError, match="DOMINATES bound_type requires p_neg_inf=0"):
            PLDRealization(
                x_min=0.0,
                x_gap=1.0,
                PMF_array=np.array([0.45, 0.45]),
                p_loss_inf=0.0,
                p_loss_neg_inf=0.1,
            )


class TestLossToExpTransform:
    def test_remove_direction_transform_uses_exp_l(self):
        realization = _simple_valid_realization()
        exp_dist = exp_linear_to_geometric(realization)
        np.testing.assert_allclose(
            exp_dist.get_x_array(),
            np.array([1.0, np.exp(1.0), np.exp(2.0)]),
            rtol=1e-10,
        )
        np.testing.assert_allclose(exp_dist.PMF_array, realization.probabilities, rtol=1e-10)

    def test_add_direction_transform_uses_exp_minus_l(self):
        realization = _simple_valid_realization()
        negated = negate_reverse_linear_distribution(realization)
        exp_dist = exp_linear_to_geometric(negated)
        np.testing.assert_allclose(
            exp_dist.get_x_array(),
            np.array([np.exp(-2.0), np.exp(-1.0), 1.0]),
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            exp_dist.PMF_array,
            np.array([0.1, 0.2, 0.7]),
            rtol=1e-10,
        )

    def test_infinity_mass_handling_add(self):
        realization = _realization_from_loss_values(
            loss_values=np.array([0.0, 1.0]),
            probabilities=np.array([0.85, 0.1]),
            p_loss_inf=0.05,
        )
        negated = negate_reverse_linear_distribution(realization)
        exp_dist = exp_linear_to_geometric(negated)
        assert abs(exp_dist.p_neg_inf - 0.05) < 1e-10
        assert abs(exp_dist.p_pos_inf) < 1e-10


class TestRoundTripTransform:
    def test_round_trip_remove(self):
        original = _realization_from_loss_values(
            loss_values=np.array([0.5, 1.5, 2.5, 3.5]),
            probabilities=np.array([0.4, 0.3, 0.2, 0.1]),
        )
        exp_dist = exp_linear_to_geometric(original)
        linear_dist = log_geometric_to_linear(exp_dist)
        reconstructed = LinearDiscreteDist.from_x_array(
            x_array=linear_dist.x_array,
            PMF_array=linear_dist.PMF_array,
            p_neg_inf=original.p_loss_neg_inf,
            p_pos_inf=original.p_loss_inf,
        )
        np.testing.assert_allclose(reconstructed.x_array, original.loss_values, rtol=1e-10)
        np.testing.assert_allclose(reconstructed.PMF_array, original.probabilities, rtol=1e-10)

    def test_round_trip_add(self):
        original = _realization_from_loss_values(
            loss_values=np.array([0.5, 1.5, 2.5, 3.5]),
            probabilities=np.array([0.4, 0.3, 0.2, 0.1]),
        )
        negated = negate_reverse_linear_distribution(original)
        exp_dist = exp_linear_to_geometric(negated)
        linear_dist = log_geometric_to_linear(exp_dist)
        reconstructed = negate_reverse_linear_distribution(linear_dist)
        np.testing.assert_allclose(reconstructed.x_array, original.loss_values, rtol=1e-10)
        np.testing.assert_allclose(reconstructed.PMF_array, original.probabilities, rtol=1e-10)


class TestGaussianReferenceComparison:
    def test_gaussian_remove_matches_direct_path(self):
        sigma = 2.0
        num_steps = 2
        delta = 1e-5
        config = AllocationSchemeConfig(loss_discretization=1e-3, tail_truncation=1e-12)

        direct_pld = gaussian_allocation_PLD(
            PrivacyParams(sigma=sigma, num_steps=num_steps, delta=delta),
            config,
        )
        realization_pld = general_allocation_PLD(
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            remove_realization=_gaussian_realization(sigma, Direction.REMOVE),
            add_realization=_gaussian_realization(sigma, Direction.ADD),
            config=config,
        )

        epsilon_direct = direct_pld.get_epsilon_for_delta(delta)
        epsilon_realization = realization_pld.get_epsilon_for_delta(delta)
        assert abs(epsilon_direct - epsilon_realization) < 5e-3

    def test_gaussian_both_directions_match_direct_path(self):
        sigma = 3.0
        num_steps = 3
        delta = 1e-6
        config = AllocationSchemeConfig(loss_discretization=1e-3, tail_truncation=1e-12)

        direct_pld = gaussian_allocation_PLD(
            PrivacyParams(sigma=sigma, num_steps=num_steps, delta=delta),
            config=config,
        )
        realization_pld = general_allocation_PLD(
            num_steps=num_steps,
            num_selected=1,
            num_epochs=1,
            remove_realization=_gaussian_realization(sigma, Direction.REMOVE),
            add_realization=_gaussian_realization(sigma, Direction.ADD),
            config=config,
        )

        epsilon_direct = direct_pld.get_epsilon_for_delta(delta)
        epsilon_realization = realization_pld.get_epsilon_for_delta(delta)
        assert abs(epsilon_direct - epsilon_realization) < 7e-3


class TestCustomRealizations:
    def test_uniform_realization(self):
        loss_values = np.linspace(0.0, 0.5, 101)
        probabilities = np.ones(101) / 101
        realization = _realization_from_loss_values(loss_values=loss_values, probabilities=probabilities)
        realization.validate_pld_realization()
        pld = general_allocation_PLD(
            num_steps=3,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=AllocationSchemeConfig(),
        )
        assert pld is not None

    def test_exponential_realization(self):
        loss_values = np.linspace(0.0, 5.0, 201)
        pdf_values = 2.0 * np.exp(-2.0 * loss_values)
        probabilities = pdf_values / pdf_values.sum()
        realization = _realization_from_loss_values(loss_values=loss_values, probabilities=probabilities)
        realization.validate_pld_realization()
        pld = general_allocation_PLD(
            num_steps=2,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=AllocationSchemeConfig(),
        )
        assert pld.get_epsilon_for_delta(1e-5) > 0


class TestCompositionCorrectness:
    def test_composition_produces_valid_results(self):
        realization = _realization_from_loss_values(
            loss_values=np.array([0.0, 0.5, 1.0, 1.5]),
            probabilities=np.array([0.5, 0.3, 0.15, 0.05]),
        )
        delta = 1e-5
        config = AllocationSchemeConfig(loss_discretization=1e-2)
        pld = general_allocation_PLD(
            num_steps=2,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=config,
        )
        epsilon = pld.get_epsilon_for_delta(delta)
        delta_result = pld.get_delta_for_epsilon(1.0)
        assert epsilon > 0
        assert 0.0 <= delta_result <= 1.0

    def test_both_directions(self):
        realization = _simple_valid_realization()
        pld = general_allocation_PLD(
            num_steps=3,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=AllocationSchemeConfig(),
        )
        assert pld.get_epsilon_for_delta(1e-5) > 0

    def test_large_two_stage_composition_remains_queryable(self):
        realization = _realization_from_loss_values(
            loss_values=np.linspace(0.0, 0.2, 65),
            probabilities=np.full(65, 1.0 / 65.0),
        )
        pld = general_allocation_PLD(
            num_steps=100,
            num_selected=1,
            num_epochs=1,
            remove_realization=realization,
            add_realization=realization,
            config=AllocationSchemeConfig(loss_discretization=2e-2, tail_truncation=1e-12),
        )
        epsilon = pld.get_epsilon_for_delta(1e-4)
        delta = pld.get_delta_for_epsilon(5.0)
        assert epsilon >= 0.0
        assert 0.0 <= delta <= 1.0

    def test_adaptive_wrapper_supports_realization_builder(self):
        remove_realization = _gaussian_realization(2.0, Direction.REMOVE, num_points=1200)
        add_realization = _gaussian_realization(2.0, Direction.ADD, num_points=1200)

        def realization_builder(
            *,
            params: PrivacyParams,
            config: AllocationSchemeConfig,
            bound_type: BoundType,
        ):
            return general_allocation_PLD(
                num_steps=params.num_steps,
                num_selected=params.num_selected,
                num_epochs=params.num_epochs,
                remove_realization=remove_realization,
                add_realization=add_realization,
                config=config,
                bound_type=bound_type,
            )

        result = optimize_allocation_epsilon_range(
            params=PrivacyParams(sigma=2.0, num_steps=3, delta=1e-5),
            target_accuracy=0.2,
            pld_builder=realization_builder,
            initial_discretization=0.1,
            initial_tail_truncation=1e-6,
        )
        assert np.isfinite(result.upper_bound)
        assert np.isfinite(result.lower_bound)
        assert result.upper_bound >= result.lower_bound


class TestErrorHandling:
    def test_no_realizations_provided(self):
        with pytest.raises(TypeError):
            general_allocation_PLD()

    def test_invalid_num_steps(self):
        realization = _simple_valid_realization()
        with pytest.raises(ValueError, match="num_steps must be >= 1"):
            general_allocation_PLD(
                num_steps=0,
                num_selected=1,
                num_epochs=1,
                remove_realization=realization,
                add_realization=realization,
                config=AllocationSchemeConfig(),
            )

    def test_bound_type_both_rejected(self):
        realization = _simple_valid_realization()
        with pytest.raises(ValueError, match="build separate DOMINATES and IS_DOMINATED PLDs instead"):
            general_allocation_PLD(
                num_steps=1,
                num_selected=1,
                num_epochs=1,
                remove_realization=realization,
                add_realization=realization,
                config=AllocationSchemeConfig(),
                bound_type=BoundType.BOTH,
            )

    @pytest.mark.parametrize("method", [ConvolutionMethod.FFT, ConvolutionMethod.BEST_OF_TWO, ConvolutionMethod.COMBINED])
    def test_general_allocation_rejects_non_geom_convolution_methods(self, method: ConvolutionMethod):
        realization = _simple_valid_realization()
        with pytest.raises(ValueError, match="requires geometric convolution"):
            general_allocation_PLD(
                num_steps=1,
                num_selected=1,
                num_epochs=1,
                remove_realization=realization,
                add_realization=realization,
                config=AllocationSchemeConfig(convolution_method=method),
                bound_type=BoundType.DOMINATES,
            )

    def test_coarsen_then_exp_preserves_geometric_structure(self):
        linear_dist = change_spacing_type(
            dist=_simple_valid_realization(),
            tail_truncation=1e-12,
            loss_discretization=0.25,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.DOMINATES,
        )
        exp_dist = exp_linear_to_geometric(linear_dist)
        assert linear_dist.x_gap == 0.25
        assert exp_dist.PMF_array.size >= 2
        assert exp_dist.ratio > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
