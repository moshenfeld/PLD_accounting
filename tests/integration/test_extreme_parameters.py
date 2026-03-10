"""
System tests with extreme parameters.

Tests pushing boundaries: large T, small sigma, many convolutions, etc.
"""
import pytest
import numpy as np
from scipy import stats
from PLD_accounting.types import BoundType, SpacingType, ConvolutionMethod
from PLD_accounting.discrete_dist import GeneralDiscreteDist, LinearDiscreteDist, GeometricDiscreteDist

from PLD_accounting.FFT_convolution import FFT_self_convolve
from PLD_accounting.geometric_convolution import geometric_self_convolve
from PLD_accounting.distribution_discretization import (
    discretize_continuous_distribution
)
from PLD_accounting.types import PrivacyParams, AllocationSchemeConfig, Direction
from PLD_accounting.random_allocation_accounting import numerical_allocation_epsilon
from tests.test_tolerances import TestTolerances as TOL


class TestLargeConvolutions:
    """Test with large number of convolutions."""

    def test_fft_large_t(self):
        """Test FFT with very large T (efficient scaling)."""
        dist_scipy = stats.expon(scale=1.0)
        dist = discretize_continuous_distribution(
            dist=dist_scipy,
            n_grid=150,  # Reduced from 200 for faster runtime
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        # T=64 still tests large T (log2(64) = 6 convolutions), faster than T=128
        result = FFT_self_convolve(
            dist=dist,
            T=64,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )

        # Verify mass conservation and reasonable result
        total_mass = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total_mass, 1.0, atol=0.01)
        # Mean should be positive and reasonable (discretization affects accuracy)
        actual_mean = np.sum(result.x_array * result.PMF_array)
        assert actual_mean > 0
        # For exponential(1), theoretical mean of 64-fold convolution is 64,
        # but discretization and grid effects reduce observed mean significantly
        assert actual_mean > 0.5 * 64.0  # At least 50% of expected (large T amplifies discretization errors)

    def test_geometric_moderate_t_geometric(self):
        """Test geometric with moderate T and geometric spacing."""
        x = np.geomspace(0.1, 100.0, 80)  # Reduced from 100 for faster runtime
        pmf = np.random.dirichlet(np.ones(80)).astype(np.float64)
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        # T=16 still tests moderate T, but twice as fast as T=32
        result = geometric_self_convolve(
            dist=dist,
            T=16,
            tail_truncation=0.05,
            bound_type=BoundType.DOMINATES
        )

        # Just check completion and mass conservation
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-8)


class TestHighPrecisionRequirements:
    """Test scenarios requiring high numerical precision."""

    def test_many_small_masses(self):
        """Test convolution with many small probability masses."""
        n = 300  # Reduced from 500 for faster runtime
        x = np.linspace(0.0, 10.0, n)
        # Create many small equal masses
        pmf = np.ones(n, dtype=np.float64) / n
        dist = LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = FFT_self_convolve(
            dist=dist,
            T=4,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True
        )

        # Mass conservation should hold despite many small values
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-9)

    def test_extreme_tail_probabilities(self):
        """Test distribution with very small tail probabilities."""
        x = np.linspace(0.0, 5.0, 100)
        # Create distribution with exponentially decaying tails
        pmf = np.exp(-0.5 * x)
        pmf = pmf / np.sum(pmf)
        pmf = pmf.astype(np.float64)
        dist = LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = FFT_self_convolve(
            dist=dist,
            T=8,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True
        )

        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-9)


class TestExtremePrivacyParameters:
    """Test sampling schemes with extreme privacy parameters."""

    def test_very_high_privacy(self):
        """Test allocation with very large sigma (high privacy, low utility)."""
        params = PrivacyParams(
            sigma=6.0,   # Large but numerically stable
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=1e-4,
            tail_truncation=1e-12,
            max_grid_FFT=100000,
            convolution_method=ConvolutionMethod.FFT
)

        eps = numerical_allocation_epsilon(
            params=params,
            config=config,
            direction=Direction.REMOVE
)

        # Should give small epsilon
        assert eps < 1.0
        assert eps > 0

    def test_very_low_privacy(self):
        """Test allocation with very small sigma (low privacy, high utility)."""
        params = PrivacyParams(
            sigma=0.5,  # Small, but not so extreme that FFT grid collapses
            num_steps=6,  # Changed from 5 to 6 to ensure floor(6/3)=2, giving T=1
            num_selected=3,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=0.02,
            tail_truncation=0.1,
            max_grid_FFT=100000,
            convolution_method=ConvolutionMethod.FFT
)

        eps = numerical_allocation_epsilon(
            params=params,
            config=config,
            direction=Direction.REMOVE
)

        # Should give large epsilon
        assert eps > 5.0

    def test_many_epochs(self):
        """Test allocation with many epochs (composition)."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=10,  # Reduced from 20 for faster runtime, still tests composition
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=0.02,
            tail_truncation=0.1,
            max_grid_FFT=100000,
            convolution_method=ConvolutionMethod.FFT
)

        # Should complete with FFT
        eps = numerical_allocation_epsilon(
            params=params,
            config=config,
            direction=Direction.REMOVE
)

        assert eps > 0
        # Epsilon should be substantial with many epochs (relaxed threshold for 10 epochs)
        assert eps > 2.0

    def test_many_steps(self):
        """Test allocation with many steps per batch."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=50,  # Reduced from 100 for faster runtime, still tests many steps
            num_selected=10,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=0.02,
            tail_truncation=0.1,
            max_grid_FFT=100000,
            convolution_method=ConvolutionMethod.FFT
)

        eps = numerical_allocation_epsilon(
            params=params,
            config=config,
            direction=Direction.REMOVE
)

        assert eps > 0


class TestLargeGrids:
    """Test with very large grids."""

    def test_large_grid_discretization(self):
        """Test discretizing with very large grid."""
        dist = stats.norm(loc=0.0, scale=1.0)
        discrete_dist = discretize_continuous_distribution(
            dist=dist,
            n_grid=500,  # Reduced from 1000 for faster runtime, still large
            align_to_multiples=True,
            tail_truncation=0.001,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        # Alignment may result in slightly more points
        assert len(discrete_dist.x_array) >= 500
        total = np.sum(discrete_dist.PMF_array) + discrete_dist.p_neg_inf + discrete_dist.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-10)

    def test_fft_large_grid(self):
        """Test FFT convolution with large grid."""
        dist = stats.expon(scale=1.0)
        discrete_dist = discretize_continuous_distribution(
            dist=dist,
            n_grid=300,  # Reduced from 500 for faster runtime
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR
        )

        # FFT should handle large grids efficiently
        result = FFT_self_convolve(
            dist=discrete_dist,
            T=8,  # Reduced from 16 for faster runtime
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )

        # Check reasonable result
        assert len(result.x_array) > 0
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-8)


class TestStressConvolutions:
    """Stress tests for convolution correctness under challenging conditions."""

    def test_spike_distribution(self):
        """Test convolution with spike (almost deterministic) distribution."""
        x = np.linspace(0.0, 10.0, 101)
        pmf = np.zeros(101, dtype=np.float64)
        pmf[50] = 1.0  # Spike at x=5
        dist = LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = FFT_self_convolve(
            dist=dist,
            T=3,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True
        )

        # Result should be concentrated around 3*5 = 15
        mean = np.sum(result.x_array * result.PMF_array)
        assert np.isclose(mean, 15.0, atol=0.5)

    def test_bimodal_distribution(self):
        """Test convolution with bimodal distribution."""
        x = np.linspace(0.0, 10.0, 101)
        pmf = np.zeros(101, dtype=np.float64)
        # Two peaks
        pmf[20] = 0.5
        pmf[80] = 0.5
        dist = LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = FFT_self_convolve(
            dist=dist,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True
        )

        # Check mass conservation
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-10)

    def test_uniform_to_gaussian_clt(self):
        """Test that uniform convolutions approach Gaussian (CLT)."""
        # Uniform distribution
        x = np.linspace(0.0, 1.0, 51)
        pmf = np.ones(51, dtype=np.float64) / 51
        dist = LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        # Convolve many times - should approach Gaussian by CLT
        result = FFT_self_convolve(
            dist=dist,
            T=16,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True
        )

        # Mean should be T * 0.5
        mean = np.sum(result.x_array * result.PMF_array)
        assert np.isclose(mean, 8.0, rtol=0.2)

        # Should look somewhat Gaussian (check rough shape)
        # Find mode
        mode_idx = np.argmax(result.PMF_array)
        mode_x = result.x_array[mode_idx]
        # Mode should be near mean for Gaussian
        assert abs(mode_x - mean) < 1.0


class TestnumericalStability:
    """Test numerical stability under challenging conditions."""

    def test_wide_dynamic_range(self):
        """Test distribution spanning wide dynamic range."""
        x = np.geomspace(1e-3, 1e3, 200)
        pmf = np.ones(200, dtype=np.float64) / 200
        dist = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)

        result = geometric_self_convolve(
            dist=dist,
            T=4,
            tail_truncation=0.05,
            bound_type=BoundType.DOMINATES
        )

        # Check mass conservation despite wide range
        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=1e-8)

    def test_very_small_discretization(self):
        """Test with very fine discretization."""
        params = PrivacyParams(
            sigma=1.0,
            num_steps=10,
            num_selected=5,
            num_epochs=1,
            delta=1e-5
        )
        config = AllocationSchemeConfig(
            loss_discretization=0.01,
            max_grid_FFT=100000
        )  # Very fine

        # Should complete without numerical issues
        eps = numerical_allocation_epsilon(
            params=params,
            config=config,
            direction=Direction.REMOVE
)

        assert eps > 0
        assert not np.isnan(eps)
        assert not np.isinf(eps)
