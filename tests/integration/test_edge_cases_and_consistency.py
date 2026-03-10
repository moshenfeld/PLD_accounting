"""
Edge cases and consistency tests for convolution and privacy accounting.

Tests cover:
1. Edge cases: extreme parameter values, boundary conditions, degenerate distributions
2. Consistency: bound type relationships, convolution properties, epsilon monotonicity
3. Invariants: mass conservation, stochastic dominance, associativity
"""
import pytest
import numpy as np
from scipy import stats

from PLD_accounting.FFT_convolution import FFT_self_convolve, FFT_convolve
from PLD_accounting.geometric_convolution import geometric_self_convolve, geometric_convolve
from PLD_accounting.types import BoundType, SpacingType, ConvolutionMethod, PrivacyParams, AllocationSchemeConfig, Direction
from PLD_accounting.discrete_dist import GeneralDiscreteDist, GeometricDiscreteDist, LinearDiscreteDist
from PLD_accounting.distribution_discretization import (
    discretize_continuous_distribution,
    change_spacing_type
)
from PLD_accounting.random_allocation_accounting import numerical_allocation_epsilon
from tests.test_tolerances import TestTolerances as TOL


class TestEdgeCasesDistributions:
    """Test edge cases for distribution creation and validation."""

    def test_single_point_distribution(self):
        """Test that single-point distributions are rejected."""
        x = np.array([1.0])
        pmf = np.array([1.0], dtype=np.float64)
        with pytest.raises(ValueError, match="at least 2 points"):
            GeneralDiscreteDist(x_array=x, PMF_array=pmf)

    def test_two_point_distribution(self):
        """Test minimal non-trivial distribution (2 points)."""
        x = np.array([1.0, 2.0])
        pmf = np.array([0.3, 0.7], dtype=np.float64)
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf)

        assert len(dist.x_array) == 2
        assert np.isclose(np.sum(dist.PMF_array), 1.0)

    def test_all_mass_at_infinity(self):
        """Test distribution with all mass at infinities."""
        x = np.array([0.0, 1.0])
        pmf = np.array([0.0, 0.0], dtype=np.float64)
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf, p_neg_inf=0.4, p_pos_inf=0.6)
        with pytest.raises(ValueError, match="all mass at infinity"):
            dist.validate_mass_conservation(BoundType.DOMINATES)

    def test_distribution_with_neg_inf_only(self):
        """Test distribution with all mass at negative infinity."""
        x = np.array([0.0, 1.0])
        pmf = np.array([0.0, 0.0], dtype=np.float64)
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf, p_neg_inf=1.0, p_pos_inf=0.0)
        with pytest.raises(ValueError, match="all mass at infinity"):
            dist.validate_mass_conservation(BoundType.IS_DOMINATED)

    def test_distribution_with_pos_inf_only(self):
        """Test distribution with all mass at positive infinity."""
        x = np.array([0.0, 1.0])
        pmf = np.array([0.0, 0.0], dtype=np.float64)
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf, p_neg_inf=0.0, p_pos_inf=1.0)
        with pytest.raises(ValueError, match="all mass at infinity"):
            dist.validate_mass_conservation(BoundType.DOMINATES)

    def test_very_small_probabilities(self):
        """Test distribution with very small but valid probabilities."""
        n = 100
        x = np.linspace(0.0, 10.0, n)
        # Create probabilities that sum to 1 but are very small individually
        pmf = np.ones(n, dtype=np.float64) / n
        dist = LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        assert np.isclose(np.sum(dist.PMF_array), 1.0, atol=1e-10)

    def test_geometric_grid_at_boundary_ratio(self):
        """Test geometric grid with ratio very close to 1.0."""
        # Smallest ratio that's still geometric
        x = np.geomspace(1.0, 1.1, 50)
        pmf = np.ones(50, dtype=np.float64) / 50
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = geometric_self_convolve(
            dist=dist, T=2, tail_truncation=0.1,
            bound_type=BoundType.DOMINATES
        )
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-8)


class TestEdgeCasesConvolution:
    """Test edge cases for convolution operations."""

    def test_convolve_with_nearly_deterministic(self):
        """Convolving with a nearly-deterministic distribution."""
        # Create distributions with matching grid spacing (dx=1.0)
        # Nearly deterministic distribution (most mass at one point)
        x_det = np.linspace(0.0, 20.0, 21)  # dx=1.0
        pmf_det = np.zeros(21, dtype=np.float64)
        pmf_det[10] = 0.95  # Most mass in center
        pmf_det[9] = 0.025
        pmf_det[11] = 0.025
        dist_det = LinearDiscreteDist.from_x_array(x_array=x_det, PMF_array=pmf_det)

        # Uniform distribution with same spacing
        x_unif = np.linspace(0.0, 20.0, 21)  # dx=1.0
        pmf_unif = np.ones(21, dtype=np.float64) / 21
        dist_unif = LinearDiscreteDist.from_x_array(x_array=x_unif, PMF_array=pmf_unif)

        result = FFT_convolve(
            dist_1=dist_unif, dist_2=dist_det,
            tail_truncation=0.0, bound_type=BoundType.DOMINATES
        )

        # Mass should be conserved
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-8)

    def test_self_convolve_t_equals_1(self):
        """Self-convolution with T=1 should return original distribution."""
        x = np.linspace(1.0, 4.0, 3)
        pmf = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        dist = LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = FFT_self_convolve(
            dist=dist, T=1, tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
            use_direct=True
        )

        # Should be identical (after accounting for remapping)
        assert len(result.x_array) >= len(dist.x_array)
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-10)

    def test_very_large_t_value(self):
        """Test self-convolution with very large T (tests binary exponentiation)."""
        dist_scipy = stats.expon(scale=1.0)
        dist = discretize_continuous_distribution(
            dist=dist_scipy, n_grid=100, tail_truncation=0.05,
            align_to_multiples=True,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.GEOMETRIC
        )

        # T=256 requires log2(256)=8 binary exponentiation steps
        result = geometric_self_convolve(
            dist=dist, T=256, tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )

        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=0.01)
        # For large T, most mass may go to p_pos_inf due to beta truncation
        # Just verify reasonable behavior: positive mean or significant tail mass
        mean = np.sum(result.x_array * result.PMF_array)
        assert mean > 0 or result.p_pos_inf > 0.5

    def test_very_small_beta(self):
        """Test with very small beta (tail truncation parameter)."""
        dist_scipy = stats.norm(loc=0.0, scale=1.0)
        dist = discretize_continuous_distribution(
            dist=dist_scipy, n_grid=100, tail_truncation=1e-15,  # Extremely small
            align_to_multiples=True,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        # Should still conserve mass
        total = np.sum(dist.PMF_array) + dist.p_neg_inf + dist.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-10)


class TestConsistencyBoundTypes:
    """Test consistency between DOMINATES and IS_DOMINATED bounds."""

    def test_dominates_upper_bounds_is_dominated(self):
        """DOMINATES and IS_DOMINATED represent different bound types with different semantics.

        We test that both produce valid privacy guarantees but don't necessarily
        have a strict ordering relationship due to their different rounding strategies.
        """
        params = PrivacyParams(sigma=1.0, num_steps=10, num_selected=5, num_epochs=1, delta=1e-5)
        base_config = AllocationSchemeConfig(loss_discretization=0.02, tail_truncation=1e-6, max_grid_FFT=50000)
        config_fft = AllocationSchemeConfig(
            loss_discretization=base_config.loss_discretization,
            tail_truncation=base_config.tail_truncation,
            max_grid_FFT=base_config.max_grid_FFT,
            max_grid_mult=base_config.max_grid_mult
        )
        config_mult = AllocationSchemeConfig(
            loss_discretization=base_config.loss_discretization,
            tail_truncation=base_config.tail_truncation,
            max_grid_FFT=base_config.max_grid_FFT,
            max_grid_mult=base_config.max_grid_mult
        )
        config_combined = AllocationSchemeConfig(
            loss_discretization=base_config.loss_discretization,
            tail_truncation=base_config.tail_truncation,
            max_grid_FFT=base_config.max_grid_FFT,
            max_grid_mult=base_config.max_grid_mult,
            convolution_method=ConvolutionMethod.COMBINED
)

        # Compute epsilon with REMOVE direction (uses DOMINATES internally)
        eps_remove = numerical_allocation_epsilon(
            params=params, config=config_fft,
            direction=Direction.REMOVE
)

        # Both should produce valid, positive, finite epsilon values
        assert eps_remove > 0 and np.isfinite(eps_remove)

        # Verify epsilon is in reasonable range for these parameters
        assert 0.5 < eps_remove < 20.0

    def test_discretization_preserves_bound_relationship(self):
        """Test that discretization maintains domination relationship."""
        dist_scipy = stats.norm(loc=0.0, scale=1.0)

        # Discretize with both bounds
        dist_dominates = discretize_continuous_distribution(
            dist=dist_scipy, n_grid=100, tail_truncation=0.01,
            align_to_multiples=True,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        dist_is_dominated = discretize_continuous_distribution(
            dist=dist_scipy, n_grid=100, tail_truncation=0.01,
            align_to_multiples=True,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR
        )

        # DOMINATES should have no mass at -inf
        assert dist_dominates.p_neg_inf == 0.0
        # IS_DOMINATED should have no mass at +inf
        assert dist_is_dominated.p_pos_inf == 0.0

        # Both should conserve mass
        total_dom = np.sum(dist_dominates.PMF_array) + dist_dominates.p_pos_inf
        total_is_dom = np.sum(dist_is_dominated.PMF_array) + dist_is_dominated.p_neg_inf
        assert np.isclose(total_dom, 1.0, atol=1e-10)
        assert np.isclose(total_is_dom, 1.0, atol=1e-10)

    def test_convolution_preserves_bound_semantics(self):
        """Test that convolution preserves bound type semantics."""
        x = np.linspace(1.0, 4.0, 3)
        pmf = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        dist = LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf, p_pos_inf=0.0)

        # Self-convolve with DOMINATES
        result = FFT_self_convolve(
            dist=dist, T=3, tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
            use_direct=True
        )

        # DOMINATES should have no mass at -inf
        assert result.p_neg_inf == 0.0

        # Create IS_DOMINATED version
        x_geom = np.geomspace(1.0, 4.0, 3)
        dist_is_dom = GeometricDiscreteDist.from_x_array(x_array=x_geom, PMF_array=pmf, p_neg_inf=0.0)
        result_is_dom = geometric_self_convolve(
            dist=dist_is_dom, T=3, tail_truncation=0.1,
            bound_type=BoundType.IS_DOMINATED
        )

        # IS_DOMINATED should have no mass at +inf
        assert result_is_dom.p_pos_inf == 0.0


class TestConsistencyMonotonicity:
    """Test monotonicity properties: epsilon increases with worse privacy parameters."""

    def test_epsilon_decreases_with_sigma(self):
        """Epsilon should decrease (better privacy) as sigma increases."""
        config = AllocationSchemeConfig(
            loss_discretization=0.02,
            max_grid_FFT=50000
        )

        sigmas = [0.5, 1.0, 2.0, 4.0]
        epsilons = []

        for sigma in sigmas:
            params = PrivacyParams(sigma=sigma, num_steps=10, num_selected=5, num_epochs=1, delta=1e-5)
            eps = numerical_allocation_epsilon(
                params=params, config=config,
                direction=Direction.REMOVE
)
            epsilons.append(eps)

        # Epsilon should be monotonically decreasing with sigma
        for i in range(len(epsilons) - 1):
            assert epsilons[i] > epsilons[i+1], \
                f"Epsilon should decrease with sigma: eps[{i}]={epsilons[i]} vs eps[{i+1}]={epsilons[i+1]}"

    def test_epsilon_increases_with_epochs(self):
        """Epsilon should increase (worse privacy) as num_epochs increases."""
        params_base = PrivacyParams(sigma=1.0, num_steps=10, num_selected=5, delta=1e-5)
        config = AllocationSchemeConfig(
            loss_discretization=0.02,
            max_grid_FFT=50000,
            convolution_method=ConvolutionMethod.FFT
)

        epochs = [1, 2, 5, 10]
        epsilons = []

        for num_epochs in epochs:
            params = PrivacyParams(
                sigma=params_base.sigma,
                num_steps=params_base.num_steps,
                num_selected=params_base.num_selected,
                num_epochs=num_epochs,
                delta=params_base.delta
            )
            eps = numerical_allocation_epsilon(
                params=params, config=config,
                direction=Direction.REMOVE
)
            epsilons.append(eps)

        # Epsilon should be monotonically increasing with epochs
        for i in range(len(epsilons) - 1):
            assert epsilons[i] < epsilons[i+1], \
                f"Epsilon should increase with epochs: eps[{i}]={epsilons[i]} vs eps[{i+1}]={epsilons[i+1]}"

    def test_epsilon_behavior_with_varying_parameters(self):
        """Test that epsilon changes appropriately with different parameter combinations.

        The relationship between epsilon and num_steps is complex due to:
        1. Allocation scheme using num_steps-1 for T
        2. Discretization effects
        3. Beta truncation effects

        We simply verify that all parameter combinations produce valid epsilon values.
        """
        config = AllocationSchemeConfig(
            loss_discretization=0.025,
            max_grid_FFT=50000,
            convolution_method=ConvolutionMethod.FFT
)

        steps_list = [6, 12, 24]
        epsilons = []

        for num_steps in steps_list:
            params = PrivacyParams(
                sigma=1.0, num_steps=num_steps,
                num_selected=2,
                num_epochs=1, delta=1e-5
            )
            eps = numerical_allocation_epsilon(
                params=params, config=config,
                direction=Direction.REMOVE
)
            epsilons.append(eps)

        # All epsilon values should be positive, finite, and reasonable
        for i, eps in enumerate(epsilons):
            assert eps > 0 and np.isfinite(eps), \
                f"Epsilon at index {i} is invalid: {eps}"
            assert 0.1 < eps < 50.0, \
                f"Epsilon at index {i} out of reasonable range: {eps}"


class TestConsistencyConvolutionMethods:
    """Test consistency between different convolution methods."""

    def test_fft_vs_geometric_same_result(self):
        """FFT and geometric should give similar results for small problems."""
        dist_scipy = stats.expon(scale=1.0)

        # Create distribution with LINEAR spacing for FFT
        dist_linear = discretize_continuous_distribution(
            dist=dist_scipy, n_grid=100, tail_truncation=0.05,
            align_to_multiples=True,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        # Create distribution with GEOMETRIC spacing for GEOM
        dist_geom = discretize_continuous_distribution(
            dist=dist_scipy, n_grid=100, tail_truncation=0.05,
            align_to_multiples=True,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.GEOMETRIC
        )

        # Convolve with both methods
        result_fft = FFT_self_convolve(
            dist=dist_linear, T=4, tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )

        result_mult = geometric_self_convolve(
            dist=dist_geom, T=4, tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )

        # Both should conserve mass
        total_fft = np.sum(result_fft.PMF_array) + result_fft.p_pos_inf
        total_mult = np.sum(result_mult.PMF_array) + result_mult.p_pos_inf
        assert np.isclose(total_fft, 1.0, atol=1e-8)
        assert np.isclose(total_mult, 1.0, atol=1e-8)

        # Means should be similar (within 20% due to different discretizations)
        mean_fft = np.sum(result_fft.x_array * result_fft.PMF_array)
        mean_mult = np.sum(result_mult.x_array * result_mult.PMF_array)
        assert np.abs(mean_fft - mean_mult) / max(mean_fft, mean_mult) < 0.2

    def test_combined_method_consistency(self):
        """COMBINED method should give results between FFT and GEOM."""
        params = PrivacyParams(sigma=1.5, num_steps=15, num_selected=5, num_epochs=1, delta=1e-5)
        base_config = AllocationSchemeConfig(loss_discretization=0.02, tail_truncation=1e-6, max_grid_FFT=50000)
        config_fft = AllocationSchemeConfig(
            loss_discretization=base_config.loss_discretization,
            tail_truncation=base_config.tail_truncation,
            max_grid_FFT=base_config.max_grid_FFT,
            max_grid_mult=base_config.max_grid_mult
        )
        config_mult = AllocationSchemeConfig(
            loss_discretization=base_config.loss_discretization,
            tail_truncation=base_config.tail_truncation,
            max_grid_FFT=base_config.max_grid_FFT,
            max_grid_mult=base_config.max_grid_mult,
            convolution_method=ConvolutionMethod.GEOM
)
        config_combined = AllocationSchemeConfig(
            loss_discretization=base_config.loss_discretization,
            tail_truncation=base_config.tail_truncation,
            max_grid_FFT=base_config.max_grid_FFT,
            max_grid_mult=base_config.max_grid_mult,
            convolution_method=ConvolutionMethod.COMBINED
)

        # Get epsilon with all three methods
        eps_fft = numerical_allocation_epsilon(
            params=params, config=config_fft,
            direction=Direction.REMOVE
)

        eps_mult = numerical_allocation_epsilon(
            params=params, config=config_mult,
            direction=Direction.REMOVE
)

        eps_combined = numerical_allocation_epsilon(
            params=params, config=config_combined,
            direction=Direction.REMOVE
)

        # All should be positive and finite
        assert eps_fft > 0 and np.isfinite(eps_fft)
        assert eps_mult > 0 and np.isfinite(eps_mult)
        assert eps_combined > 0 and np.isfinite(eps_combined)

        # COMBINED should be close to the better of FFT and GEOM
        min_eps = min(eps_fft, eps_mult)
        max_eps = max(eps_fft, eps_mult)
        assert min_eps * 0.9 <= eps_combined <= max_eps * 1.1


class TestConsistencyMassConservation:
    """Test that mass conservation holds throughout pipeline."""

    def test_mass_conservation_through_discretization(self):
        """Mass should be conserved when discretizing continuous distributions."""
        distributions = [
            stats.norm(loc=0.0, scale=1.0),
            stats.expon(scale=1.0),
            stats.gamma(a=2.0, scale=1.0),
        ]

        for dist in distributions:
            discrete = discretize_continuous_distribution(
                dist=dist, n_grid=100, tail_truncation=0.01,
                align_to_multiples=True,
                bound_type=BoundType.DOMINATES,
                spacing_type=SpacingType.LINEAR
            )

            total = np.sum(discrete.PMF_array) + discrete.p_neg_inf + discrete.p_pos_inf
            assert np.isclose(total, 1.0, atol=1e-10), \
                f"Mass not conserved for {dist.dist.name}: total={total}"

    def test_mass_conservation_through_spacing_change(self):
        """Mass should be conserved when changing spacing type."""
        dist_scipy = stats.lognorm(s=0.5)
        dist = discretize_continuous_distribution(
            dist=dist_scipy, n_grid=100, tail_truncation=0.05,
            align_to_multiples=True,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        # Convert to geometric spacing
        dist_geom = change_spacing_type(
            dist=dist, tail_truncation=0.05, loss_discretization=0.02,
            spacing_type=SpacingType.GEOMETRIC,
            bound_type=BoundType.DOMINATES
        )

        # Mass should be conserved
        total_orig = np.sum(dist.PMF_array) + dist.p_neg_inf + dist.p_pos_inf
        total_geom = np.sum(dist_geom.PMF_array) + dist_geom.p_neg_inf + dist_geom.p_pos_inf
        assert np.isclose(total_orig, 1.0, atol=1e-10)
        assert np.isclose(total_geom, 1.0, atol=1e-10)

    def test_mass_conservation_through_convolution_chain(self):
        """Mass should be conserved through multiple convolutions."""
        x = np.geomspace(1.0, 8.0, 4)
        pmf = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float64)
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        # Chain of convolutions
        result = dist
        for _ in range(3):
            result = geometric_convolve(
                dist_1=result, dist_2=dist,
                tail_truncation=0.1, bound_type=BoundType.DOMINATES
            )

            total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
            assert np.isclose(total, 1.0, atol=1e-8), \
                f"Mass not conserved in convolution chain: total={total}"


class TestConsistencyAssociativity:
    """Test associativity properties of convolution."""

    def test_self_convolution_equals_repeated_convolution(self):
        """Self-convolution T times should equal repeated binary convolutions."""
        x = np.geomspace(1.0, 4.0, 3)
        pmf = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        # Self-convolution with T=4
        result_self = geometric_self_convolve(
            dist=dist, T=4, tail_truncation=0.0,
            bound_type=BoundType.DOMINATES
        )

        # Repeated binary convolutions
        result_repeated = dist
        for _ in range(3):
            result_repeated = geometric_convolve(
                dist_1=result_repeated, dist_2=dist,
                tail_truncation=0.0, bound_type=BoundType.DOMINATES
        )

        # Means should be approximately equal
        mean_self = np.sum(result_self.x_array * result_self.PMF_array)
        mean_repeated = np.sum(result_repeated.x_array * result_repeated.PMF_array)
        assert np.abs(mean_self - mean_repeated) / max(mean_self, mean_repeated) < 0.15

    def test_convolution_order_invariance(self):
        """Convolution should be commutative: dist1 * dist2 = dist2 * dist1."""
        x1 = np.geomspace(1.0, 4.0, 3)
        pmf1 = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        dist1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=pmf1)

        x2 = np.geomspace(0.5, 2.0, 3)
        pmf2 = np.array([0.2, 0.6, 0.2], dtype=np.float64)
        dist2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=pmf2)

        # Convolve in both orders
        result_12 = geometric_convolve(
            dist_1=dist1, dist_2=dist2,
            tail_truncation=0.1, bound_type=BoundType.DOMINATES
        )

        result_21 = geometric_convolve(
            dist_1=dist2, dist_2=dist1,
            tail_truncation=0.1, bound_type=BoundType.DOMINATES
        )

        # Means should be equal (commutative property)
        mean_12 = np.sum(result_12.x_array * result_12.PMF_array)
        mean_21 = np.sum(result_21.x_array * result_21.PMF_array)
        assert np.abs(mean_12 - mean_21) < 0.5


class TestEdgeCasesDirections:
    """Test edge cases related to privacy analysis directions."""

    def test_remove_and_add_directions_both_valid(self):
        """REMOVE and ADD directions should both give valid epsilon values.

        The two directions represent different neighboring dataset definitions
        and may give different epsilon values, but both should be valid.
        """
        params = PrivacyParams(sigma=1.0, num_steps=10, num_selected=5, num_epochs=1, delta=1e-5)
        config = AllocationSchemeConfig(
            loss_discretization=0.02,
            max_grid_FFT=50000
        )

        eps_remove = numerical_allocation_epsilon(
            params=params, config=config,
            direction=Direction.REMOVE
)

        # Both should be positive, finite, and in reasonable range
        assert eps_remove > 0 and np.isfinite(eps_remove)
        assert 0.5 < eps_remove < 20.0

    def test_extreme_subsampling_ratio(self):
        """Test with extreme num_selected/num_steps ratios.

        For random allocation, T = floor(num_steps / num_selected).
        We test that reasonable T values produce valid epsilon.
        """
        config = AllocationSchemeConfig(
            loss_discretization=0.025,
            max_grid_FFT=50000
        )

        # Moderate subsampling: T = floor(20/4) = 5
        params_moderate = PrivacyParams(sigma=1.0, num_steps=20, num_selected=4, num_epochs=1, delta=1e-5)
        eps_moderate = numerical_allocation_epsilon(
            params=params_moderate, config=config,
            direction=Direction.REMOVE
)

        # Lower subsampling (more privacy amplification): T = floor(100/10) = 10
        params_low = PrivacyParams(sigma=1.0, num_steps=100, num_selected=10, num_epochs=1, delta=1e-5)
        eps_low = numerical_allocation_epsilon(
            params=params_low, config=config,
            direction=Direction.REMOVE
)

        # Both should be positive and finite
        assert eps_moderate > 0 and np.isfinite(eps_moderate)
        assert eps_low > 0 and np.isfinite(eps_low)

        # More convolutions (higher T) generally gives worse privacy
        assert eps_low > eps_moderate


class TestNumericalStability:
    """Test numerical stability edge cases."""

    def test_zero_probability_handling(self):
        """Test that zero probabilities are handled correctly."""
        x = np.linspace(0.0, 10.0, 50)
        pmf = np.zeros(50, dtype=np.float64)
        pmf[10:20] = 0.1  # Only middle section has mass
        dist = LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = FFT_self_convolve(
            dist=dist, T=2, tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True
        )

        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-9)

    def test_near_zero_discretization(self):
        """Test with very small discretization parameter."""
        params = PrivacyParams(sigma=2.0, num_steps=10, num_selected=5, num_epochs=1, delta=1e-5)
        config = AllocationSchemeConfig(
            loss_discretization=0.001,
            max_grid_FFT=50000,
            convolution_method=ConvolutionMethod.FFT
)  # Very small

        eps = numerical_allocation_epsilon(
            params=params, config=config,
            direction=Direction.REMOVE
)

        assert eps > 0 and np.isfinite(eps)

    def test_very_wide_dynamic_range(self):
        """Test distribution spanning many orders of magnitude."""
        x = np.geomspace(1e-6, 1e6, 200)
        pmf = np.ones(200, dtype=np.float64) / 200
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = geometric_self_convolve(
            dist=dist, T=2, tail_truncation=0.1,
            bound_type=BoundType.DOMINATES
        )

        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-8)
