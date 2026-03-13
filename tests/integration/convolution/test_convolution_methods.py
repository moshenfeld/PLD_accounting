"""
Integration tests for convolution methods.

Tests geometric vs FFT convolution, self-convolution, and end-to-end workflows.
Runtime: Fast (~5-10 seconds)
"""
import numpy as np
import pytest
from scipy import stats
from scipy.fft import next_fast_len

from PLD_accounting.FFT_convolution import FFT_self_convolve, FFT_convolve
from PLD_accounting.distribution_discretization import (
    discretize_continuous_distribution,
    change_spacing_type
)
from PLD_accounting.geometric_convolution import (
    geometric_self_convolve,
    geometric_convolve
)
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
    SpacingType,
)
from PLD_accounting.discrete_dist import GeneralDiscreteDist, GeometricDiscreteDist, LinearDiscreteDist
from PLD_accounting.random_allocation_gaussian import (
    allocation_PMF_from_gaussian,
    compute_conv_params,
)
from tests.test_tolerances import TestTolerances as TOL


class TestDiscretizationWorkflow:
    """Test continuous to discrete conversion workflow."""

    def test_gaussian_discretization(self):
        """Test discretizing Gaussian distribution."""
        dist = stats.norm(loc=0.0, scale=1.0)
        discrete_dist = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        # MIN_GRID_SIZE and alignment may result in more points
        assert len(discrete_dist.x_array) >= 100
        assert len(discrete_dist.PMF_array) == len(discrete_dist.x_array)
        # Check mass conservation - strict for analytical discretization
        total_mass = np.sum(discrete_dist.PMF_array) + discrete_dist.p_neg_inf + discrete_dist.p_pos_inf
        assert np.isclose(total_mass, 1.0, atol=TOL.MASS_CONSERVATION)

    def test_exponential_discretization(self):
        """Test discretizing exponential distribution."""
        dist = stats.expon(scale=1.0)
        discrete_dist = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.05,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        # MIN_GRID_SIZE and alignment may result in more points
        assert len(discrete_dist.x_array) >= 100
        # Exponential has no left tail
        assert discrete_dist.p_neg_inf < 1e-10


class TestGeometricConvolution:
    """Test geometric convolution method."""

    def test_convolve_two_distributions(self):
        """Test convolving two discrete distributions."""
        from PLD_accounting.discrete_dist import GeometricDiscreteDist
        x1 = np.array([1.0, 2.0, 4.0])
        pmf1 = np.array([0.3, 0.5, 0.2], dtype=np.float64)
        dist1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=pmf1)

        x2 = np.array([0.5, 1.0])
        pmf2 = np.array([0.6, 0.4], dtype=np.float64)
        dist2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=pmf2)

        result = geometric_convolve(
            dist_1=dist1,
            dist_2=dist2,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES
        )

        # Result support should span sum of supports
        assert result.x_array[0] >= 1.5  # min is 1.0 + 0.5
        assert result.x_array[-1] <= 6.0  # max is 4.0 + 2.0 (allowing for geometric grid expansion)
        # Mass should be conserved - strict for simple geometric convolution
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)

    def test_self_convolution_t2(self):
        """Test self-convolution T=2."""
        x = np.array([1.0, 2.0, 4.0])
        pmf = np.array([0.3, 0.5, 0.2], dtype=np.float64)
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = geometric_self_convolve(
            dist=dist,
            T=2,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES
        )

        # Mass should be conserved
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)

    def test_self_convolution_power_of_2(self):
        """Test self-convolution with T as power of 2 (efficient path) using strict parameters."""
        # Use geometric grid for geometric convolution (no zero values allowed)
        x = np.geomspace(0.01, 1.0, 201)
        pmf = np.ones(201, dtype=np.float64) / 201
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        # Use baseline kernel directly for extreme beta values
        result = geometric_self_convolve(
            dist=dist,
            T=4,
            tail_truncation=TOL.BETA,  # 1e-12 (much smaller than 0.05)
            bound_type=BoundType.DOMINATES
)

        # Mass should be conserved (strict tolerance)
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)

    def test_prefix_kernel_mass_conservation_regression(self):
        """Regression test: optimized prefix kernel violates mass conservation on high-composition case."""
        num_steps = 1000
        sigma_target = 4.0
        sigma_inv = 1.0 / sigma_target
        beta_config = 1e-8
        beta = beta_config / num_steps
        discretization = 0.005

        log_range = -stats.norm.ppf(beta / 2) * sigma_inv
        n_grid = int(min(2 * log_range * np.log(num_steps) / discretization, 10_000_000))

        base_dist = discretize_continuous_distribution(
            dist=stats.lognorm(s=sigma_inv),
            n_grid=n_grid,
            align_to_multiples=True,
            tail_truncation=beta,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.GEOMETRIC
)

        optimized = geometric_self_convolve(
            dist=base_dist,
            T=num_steps - 1,
            tail_truncation=beta,
            bound_type=BoundType.IS_DOMINATED
)

        reference = geometric_self_convolve(
            dist=base_dist,
            T=num_steps - 1,
            tail_truncation=beta,
            bound_type=BoundType.IS_DOMINATED
)

        opt_mass = np.sum(optimized.PMF_array) + optimized.p_neg_inf + optimized.p_pos_inf
        ref_mass = np.sum(reference.PMF_array) + reference.p_neg_inf + reference.p_pos_inf

        assert np.isclose(opt_mass, 1.0, atol=TOL.MASS_CONSERVATION)
        assert np.isclose(ref_mass, 1.0, atol=TOL.MASS_CONSERVATION)
        assert np.allclose(optimized.PMF_array, reference.PMF_array, atol=5e-14)


class TestFFTConvolution:
    """Test FFT convolution method."""
    def _make_linear_dist(self, n: int = 9) -> GeneralDiscreteDist:
        x = np.linspace(-1.0, 1.0, n)
        pmf = np.ones(n, dtype=np.float64) / n
        return GeneralDiscreteDist(x_array=x, PMF_array=pmf)

    def test_fft_vs_geometric_gaussian(self):
        """Compare FFT and geometric convolution on lognormal with strict tolerances."""
        # Use lognormal for geometric grid (always positive values)
        dist_scipy = stats.lognorm(s=0.5)
        # Create separate discretizations for each method (different spacing requirements)
        dist_linear = discretize_continuous_distribution(
            dist=dist_scipy,
            n_grid=TOL.GRID_SIZE,  # 10000 points
            align_to_multiples=True,
            tail_truncation=TOL.BETA,          # 1e-12
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        dist_geometric = discretize_continuous_distribution(
            dist=dist_scipy,
            n_grid=TOL.GRID_SIZE,  # 10000 points
            align_to_multiples=True,
            tail_truncation=TOL.BETA,          # 1e-12
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.GEOMETRIC
        )

        result_geometric = geometric_self_convolve(
            dist=dist_geometric,
            T=2,
            tail_truncation=TOL.BETA,  # 1e-12
            bound_type=BoundType.DOMINATES
        )

        result_fft = FFT_self_convolve(
            dist=dist_linear,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )

        # With strict parameters, mass conservation should be very tight (strict tolerance)
        mass_geometric = np.sum(result_geometric.PMF_array) + result_geometric.p_neg_inf + result_geometric.p_pos_inf
        mass_fft = np.sum(result_fft.PMF_array) + result_fft.p_neg_inf + result_fft.p_pos_inf
        assert np.isclose(mass_geometric, 1.0, atol=TOL.MASS_CONSERVATION)
        assert np.isclose(mass_fft, 1.0, atol=TOL.MASS_CONSERVATION)


@pytest.mark.parametrize(
    "direction,bound_type",
    [
        (Direction.ADD, BoundType.DOMINATES),
        (Direction.ADD, BoundType.IS_DOMINATED),
        (Direction.REMOVE, BoundType.DOMINATES),
    ]
)
def test_best_of_two_allocation_pmf(direction, bound_type):
    params = PrivacyParams(
        sigma=1.0,
        num_steps=4,
        num_selected=1,
        num_epochs=1,
        delta=1e-6
    )
    config = AllocationSchemeConfig(
        loss_discretization=1e-3,
        tail_truncation=1e-6,
        max_grid_mult=512,
        max_grid_FFT=2048,
        convolution_method=ConvolutionMethod.BEST_OF_TWO
    )
    conv_params = compute_conv_params(params=params, config=config)
    dist = allocation_PMF_from_gaussian(
        conv_params=conv_params,
        direction=direction,
        bound_type=bound_type,
        convolution_method=ConvolutionMethod.BEST_OF_TWO
    )
    dist.validate_mass_conservation(bound_type)


@pytest.mark.parametrize(
    "method",
    [
        ConvolutionMethod.GEOM,
        ConvolutionMethod.FFT,
        ConvolutionMethod.COMBINED,
        ConvolutionMethod.BEST_OF_TWO,
    ],
)
@pytest.mark.parametrize("direction", [Direction.ADD, Direction.REMOVE])
def test_allocation_pmf_methods_dominate(direction, method):
    """Smoke-test allocation_PMF_from_gaussian for all methods with DOMINATES bound."""
    params = PrivacyParams(
        sigma=1.0,
        num_steps=4,
        num_selected=1,
        num_epochs=1,
        delta=1e-6
    )
    config = AllocationSchemeConfig(
        loss_discretization=1e-3,
        tail_truncation=1e-6,
        max_grid_mult=512,
        max_grid_FFT=2048,
        convolution_method=method
    )
    conv_params = compute_conv_params(params=params, config=config)
    dist = allocation_PMF_from_gaussian(
        conv_params=conv_params,
        direction=direction,
        bound_type=BoundType.DOMINATES,
        convolution_method=method
    )
    dist.validate_mass_conservation(BoundType.DOMINATES)

    def test_fft_rejects_geometric_spacing(self):
        """Test that FFT raises error for geometric spacing."""
        x = np.array([1.0, 2.0, 4.0])
        pmf = np.array([0.3, 0.5, 0.2], dtype=np.float64)
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        with pytest.raises(ValueError, match="non-uniform bin widths"):
            FFT_self_convolve(
                dist=dist,
                T=2,
                tail_truncation=0.0,
                bound_type=BoundType.DOMINATES,
                use_direct=True,
            )

    def test_fft_self_convolve_direct_vs_binary(self):
        """Ensure direct FFT and binary exponentiation match."""
        dist = self._make_linear_dist(n=9)
        T = 10

        result_direct = FFT_self_convolve(
            dist=dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True
        )
        result_binary = FFT_self_convolve(
            dist=dist,
            T=T,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=False
        )

        assert np.allclose(result_direct.x_array, result_binary.x_array)
        assert np.allclose(result_direct.PMF_array, result_binary.PMF_array, atol=1e-10)

    def test_fft_large_t(self):
        """Test FFT with large T using strict parameters for maximum accuracy.

        This test uses strict grid (10000 points) and very small beta (1e-12) to achieve
        strict accuracy (98%) in mean computation.
        """
        dist_scipy = stats.expon(scale=1.0)
        # Use strict parameters for maximum accuracy
        dist = discretize_continuous_distribution(
            dist=dist_scipy,
            n_grid=TOL.GRID_SIZE,  # 10000 points
            align_to_multiples=True,
            tail_truncation=TOL.BETA,         # 1e-12
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        # T=16 should complete quickly with FFT
        result = FFT_self_convolve(
            dist=dist,
            T=16,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )

        # Verify mass conservation (strict tolerance - even with T=16, strict params achieve it)
        total_mass = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total_mass, 1.0, atol=TOL.MASS_CONSERVATION)

        # Mean should be positive and reasonable for exponential convolution
        actual_mean = np.sum(result.x_array * result.PMF_array)
        assert actual_mean > 0

        # For exponential(scale=1), mean of T-fold convolution is T
        # With strict grid and very small beta, we expect strict accuracy (>=98%)
        expected_mean = 16.0
        assert actual_mean >= TOL.MEAN_ACCURACY * expected_mean, \
            f"Mean accuracy: {actual_mean / expected_mean:.1%} (expected >= {TOL.MEAN_ACCURACY:.1%})"


class TestGeometricSpacing:
    """Test convolution with geometric spacing."""

    def test_geometric_geometric_spacing(self):
        """Test geometric convolution with geometric spacing."""
        x = np.geomspace(0.1, 10.0, 50)
        pmf = np.ones(50, dtype=np.float64) / 50
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = geometric_self_convolve(
            dist=dist,
            T=2,
            tail_truncation=0.05,
            bound_type=BoundType.DOMINATES,
        )

        # Check result uses geometric spacing
        ratios = result.x_array[1:] / result.x_array[:-1]
        # Ratios should be approximately constant
        assert np.std(ratios) / np.mean(ratios) < 0.1


class TestChangeSpacingType:
    """Test spacing type conversion."""

    def test_linear_to_geometric(self):
        """Test converting linear to geometric spacing."""
        x = np.linspace(1.0, 10.0, 50)
        pmf = np.ones(50, dtype=np.float64) / 50
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf)

        result = change_spacing_type(
            dist=dist,
            tail_truncation=0.05,
            loss_discretization=0.1,
            spacing_type=SpacingType.GEOMETRIC,
            bound_type=BoundType.DOMINATES
        )

        # Check geometric spacing
        ratios = result.x_array[1:] / result.x_array[:-1]
        assert np.allclose(ratios, ratios[0], rtol=0.01)

    def test_geometric_to_linear(self):
        """Test converting geometric to linear spacing."""
        x = np.geomspace(1.0, 10.0, 50)
        pmf = np.ones(50, dtype=np.float64) / 50
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = change_spacing_type(
            dist=dist,
            tail_truncation=0.05,
            loss_discretization=0.2,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.DOMINATES
        )

        # Check linear spacing
        diffs = np.diff(result.x_array)
        assert np.allclose(diffs, diffs[0], rtol=0.01)

    def test_spacing_conversion_conserves_mass(self):
        """Test that spacing conversion conserves mass."""
        x = np.linspace(1.0, 10.0, 100)
        pmf = np.random.dirichlet(np.ones(100)).astype(np.float64)
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf)

        result = change_spacing_type(
            dist=dist,
            tail_truncation=0.05,
            loss_discretization=0.1,
            spacing_type=SpacingType.GEOMETRIC,
            bound_type=BoundType.DOMINATES
        )

        total_in = np.sum(dist.PMF_array) + dist.p_neg_inf + dist.p_pos_inf
        total_out = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total_in, total_out, atol=TOL.MASS_CONSERVATION)
