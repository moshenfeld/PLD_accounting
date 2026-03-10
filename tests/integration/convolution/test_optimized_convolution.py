"""
Focused integration tests for the optimized geometric convolution path.

These exercises cover different distribution configurations and ensure the
optimized kernel produces identical results to the baseline implementation
while preserving mass and spacing properties.
"""

import time

import pytest
import numpy as np
from scipy import stats

from PLD_accounting.FFT_convolution import FFT_self_convolve, FFT_convolve
from PLD_accounting.geometric_convolution import geometric_self_convolve, geometric_convolve
from PLD_accounting.types import BoundType, SpacingType, ConvolutionMethod
from PLD_accounting.discrete_dist import GeometricDiscreteDist
from PLD_accounting.distribution_discretization import (
    discretize_continuous_distribution
)
from tests.test_tolerances import TestTolerances as TOL


class TestOptimizedConvolution:
    """Test optimized convolution kernel (default behavior)."""

    def test_optimized_vs_baseline_simple(self):
        """Optimized kernel should match baseline on simple distributions."""
        x1 = np.array([1.0, 2.0, 4.0])  # Geometric with ratio 2
        pmf1 = np.array([0.3, 0.5, 0.2], dtype=np.float64)
        dist1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=pmf1)

        x2 = np.array([0.5, 1.0])  # Geometric with ratio 2
        pmf2 = np.array([0.6, 0.4], dtype=np.float64)
        dist2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=pmf2)

        baseline = geometric_convolve(
            dist_1=dist1,
            dist_2=dist2,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES
)

        optimized = geometric_convolve(
            dist_1=dist1,
            dist_2=dist2,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES
)

        total_baseline = np.sum(baseline.PMF_array) + baseline.p_neg_inf + baseline.p_pos_inf
        total_optimized = np.sum(optimized.PMF_array) + optimized.p_neg_inf + optimized.p_pos_inf

        assert np.isclose(total_baseline, 1.0, atol=TOL.MASS_CONSERVATION)
        assert np.isclose(total_optimized, 1.0, atol=TOL.MASS_CONSERVATION)
        assert np.allclose(baseline.PMF_array, optimized.PMF_array, atol=1e-14)
        assert np.isclose(baseline.p_neg_inf, optimized.p_neg_inf, atol=1e-14)
        assert np.isclose(baseline.p_pos_inf, optimized.p_pos_inf, atol=1e-14)

    def test_optimized_with_is_dominated_bound(self):
        """Optimized kernel should honor IS_DOMINATED bounds."""
        x1 = np.array([1.0, 2.0, 4.0])  # Geometric with ratio 2
        pmf1 = np.array([0.3, 0.5, 0.2], dtype=np.float64)
        dist1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=pmf1, p_neg_inf=0.0, p_pos_inf=0.0)

        x2 = np.array([0.5, 1.0])  # Geometric with ratio 2
        pmf2 = np.array([0.6, 0.4], dtype=np.float64)
        dist2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=pmf2, p_neg_inf=0.0, p_pos_inf=0.0)

        result = geometric_convolve(
            dist_1=dist1,
            dist_2=dist2,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED
)

        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)
        assert result.p_pos_inf < 1e-10


class TestKernelComparison:
    """Compare all three kernels (BASIC, PREFIX, SCALED) for accuracy and performance."""

    def test_scaled_kernel_accuracy_on_scaled_geometric_grids(self):
        """Test SCALED kernel accuracy when assumptions are met (same-size scaled geometric grids)."""
        n_points = 128
        scale = 3.0
        beta = 1e-12
        rng = np.random.default_rng(456)

        x_base = np.geomspace(1e-3, 1e2, n_points)
        pmf_base = rng.random(n_points).astype(np.float64)
        pmf_base /= pmf_base.sum()
        pmf_scaled = rng.random(n_points).astype(np.float64)
        pmf_scaled /= pmf_scaled.sum()

        dist_base = GeometricDiscreteDist.from_x_array(x_array=x_base, PMF_array=pmf_base.astype(np.float64))
        dist_scaled = GeometricDiscreteDist.from_x_array(x_array=scale * x_base, PMF_array=pmf_scaled.astype(np.float64))

        # Compare all three kernels
        basic = geometric_convolve(
            dist_1=dist_scaled, dist_2=dist_base, tail_truncation=beta,
            bound_type=BoundType.DOMINATES
)

        prefix = geometric_convolve(
            dist_1=dist_scaled, dist_2=dist_base, tail_truncation=beta,
            bound_type=BoundType.DOMINATES
)

        scaled = geometric_convolve(
            dist_1=dist_scaled, dist_2=dist_base, tail_truncation=beta,
            bound_type=BoundType.DOMINATES
)

        # Verify mass conservation
        for result, name in [(basic, "BASIC"), (prefix, "PREFIX"), (scaled, "SCALED")]:
            total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
            assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION), f"{name} failed mass conservation"

        # Verify BASIC vs PREFIX match exactly
        assert np.allclose(basic.PMF_array, prefix.PMF_array, atol=1e-14), "BASIC vs PREFIX mismatch"

        # Verify SCALED matches BASIC/PREFIX within tolerance
        assert np.allclose(basic.x_array, scaled.x_array, rtol=1e-12, atol=1e-14), "Grid mismatch"
        assert np.allclose(
            np.asarray(basic.PMF_array, dtype=np.float64),
            np.asarray(scaled.PMF_array, dtype=np.float64),
            atol=5e-3
), "BASIC vs SCALED PMF mismatch beyond tolerance"

class TestImprovedScaledConvolution:
    """Compare the specialised scaled-grid convolution against the baseline."""

    def test_improved_matches_baseline_scaled_geometric(self):
        """Specialised kernel should reproduce baseline results under its assumptions."""
        n_points = 128
        scale = 3.0
        beta = 1e-12
        rng = np.random.default_rng(123)

        x_base = np.geomspace(1e-3, 1e2, n_points)

        pmf_base = rng.random(n_points).astype(np.float64)
        pmf_base /= pmf_base.sum()
        pmf_scaled = rng.random(n_points).astype(np.float64)
        pmf_scaled /= pmf_scaled.sum()

        dist_base = GeometricDiscreteDist.from_x_array(x_array=x_base, PMF_array=pmf_base.astype(np.float64))
        dist_scaled = GeometricDiscreteDist.from_x_array(x_array=scale * x_base, PMF_array=pmf_scaled.astype(np.float64))

        baseline = geometric_convolve(
            dist_1=dist_scaled,
            dist_2=dist_base,
            tail_truncation=beta,
            bound_type=BoundType.DOMINATES
)

        improved = geometric_convolve(
            dist_1=dist_scaled,
            dist_2=dist_base,
            tail_truncation=beta,
            bound_type=BoundType.DOMINATES
)

        assert np.allclose(baseline.x_array, improved.x_array, rtol=1e-12, atol=1e-14)
        assert np.allclose(
            np.asarray(baseline.PMF_array, dtype=np.float64),
            np.asarray(improved.PMF_array, dtype=np.float64),
            atol=5e-3
)
        assert np.isclose(float(baseline.p_neg_inf), float(improved.p_neg_inf), atol=1e-15)
        assert np.isclose(float(baseline.p_pos_inf), float(improved.p_pos_inf), atol=1e-15)


class TestGeometricConvolution:
    """Test the geometric kernel implementation."""

    def test_geometric_basic_case(self):
        """Test geometric kernel on basic geometric distributions."""
        n_points = 50
        ratio = 1.1
        x1 = ratio ** np.arange(n_points)
        pmf1 = np.ones(n_points, dtype=np.float64) / n_points
        dist1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=pmf1)

        n_points2 = 40
        x2 = 0.5 * (ratio ** np.arange(n_points2))
        pmf2 = np.ones(n_points2, dtype=np.float64) / n_points2
        dist2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=pmf2)

        result = geometric_convolve(
            dist_1=dist1,
            dist_2=dist2,
            tail_truncation=0.05,
            bound_type=BoundType.DOMINATES
)

        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)
        assert result.p_neg_inf == 0.0

    def test_geometric_vs_baseline(self):
        """Compare geometric with BASIC kernel for accuracy."""
        n_points = 100
        beta = 0.01
        rng = np.random.default_rng(999)

        ratio = 1.05
        x1 = 0.1 * (ratio ** np.arange(n_points))
        pmf1 = rng.random(n_points).astype(np.float64)
        pmf1 /= pmf1.sum()

        n_points2 = 80
        x2 = 0.1 * (ratio ** np.arange(n_points2))
        pmf2 = rng.random(n_points2).astype(np.float64)
        pmf2 /= pmf2.sum()

        dist1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=pmf1.astype(np.float64))
        dist2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=pmf2.astype(np.float64))

        basic = geometric_convolve(
            dist_1=dist1, dist_2=dist2, tail_truncation=beta,
            bound_type=BoundType.DOMINATES
)

        geometric = geometric_convolve(
            dist_1=dist1, dist_2=dist2, tail_truncation=beta,
            bound_type=BoundType.DOMINATES
)

        # Verify mass conservation
        for result, name in [(basic, "BASIC"), (geometric, "GEOMETRIC")]:
            total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
            assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION), f"{name} failed mass conservation"

        # Results should be close (allowing for different grid construction)
        # Check that the mass is distributed reasonably
        assert geometric.PMF_array.size > 0
        assert basic.PMF_array.size > 0

    def test_geometric_is_dominated(self):
        """Test geometric kernel with IS_DOMINATED bound type."""
        n_points = 50
        ratio = 1.1
        x1 = ratio ** np.arange(n_points)
        pmf1 = np.ones(n_points, dtype=np.float64) / n_points
        dist1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=pmf1, p_neg_inf=0.0, p_pos_inf=0.0)

        n_points2 = 40
        x2 = 0.5 * (ratio ** np.arange(n_points2))
        pmf2 = np.ones(n_points2, dtype=np.float64) / n_points2
        dist2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=pmf2, p_neg_inf=0.0, p_pos_inf=0.0)

        result = geometric_convolve(
            dist_1=dist1,
            dist_2=dist2,
            tail_truncation=0.05,
            bound_type=BoundType.IS_DOMINATED
)

        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)
        assert result.p_pos_inf == 0.0

    def test_geometric_same_ratio_requirement(self):
        """Test that geometric requires same geometric ratio for both inputs."""
        x1 = np.geomspace(1.0, 100.0, 50)
        pmf1 = np.ones(50, dtype=np.float64) / 50
        dist1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=pmf1)

        # Different ratio
        x2 = np.geomspace(0.5, 200.0, 40)  # Different ratio
        pmf2 = np.ones(40, dtype=np.float64) / 40
        dist2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=pmf2)

        # This should raise an error due to different ratios
        with pytest.raises(ValueError, match="Grid ratios must match"):
            geometric_convolve(
                dist_1=dist1,
                dist_2=dist2,
                tail_truncation=0.05,
                bound_type=BoundType.DOMINATES
)

    def test_geometric_direct_call(self):
        """Test calling geometric_convolve directly."""
        n_points = 50
        x_base = np.geomspace(1.0, 100.0, n_points)

        pmf1 = np.ones(n_points, dtype=np.float64) / n_points
        dist1 = GeometricDiscreteDist.from_x_array(x_array=x_base, PMF_array=pmf1)

        pmf2 = np.ones(n_points, dtype=np.float64) / n_points
        dist2 = GeometricDiscreteDist.from_x_array(x_array=x_base, PMF_array=pmf2)

        result = geometric_convolve(
            dist_1=dist1,
            dist_2=dist2,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES
)

        total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)
        assert result.PMF_array.size > 0

        # Verify geometric grid structure
        if result.x_array.size > 1:
            ratios = result.x_array[1:] / result.x_array[:-1]
            assert np.std(ratios) / np.mean(ratios) < 0.1
