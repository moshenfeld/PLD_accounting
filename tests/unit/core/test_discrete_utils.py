"""
Unit tests for PLD_accounting.distribution_discretization module.

Tests grid generation, discretization, and PMF operations.
"""
import pytest
import numpy as np
import warnings
from scipy import stats
from PLD_accounting.core_utils import compute_bin_width, compute_bin_ratio, enforce_mass_conservation
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.discrete_dist import GeneralDiscreteDist
from PLD_accounting.distribution_discretization import (
    discretize_aligned_range,
    _compute_discrete_PMF as compute_discrete_PMF,
    rediscritize_PMF as pmf_remap_to_grid_kernel
)
from PLD_accounting.utils import _CCDF_from_PMF
from tests.test_tolerances import TestTolerances as TOL


class TestDiscritizeRange:
    """Test discretize_aligned_range function."""

    def test_linear_spacing(self):
        """Test linear spacing generation."""
        x = discretize_aligned_range(x_min=0.0, x_max=10.0, spacing_type=SpacingType.LINEAR, align_to_multiples=True, n_grid=100)
        # Should have at least MIN_GRID_SIZE points
        assert len(x) >= 100
        # Range should cover requested bounds (may extend due to alignment)
        assert x[0] <= 0.0
        assert x[-1] >= 10.0
        # Check uniform spacing
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])

    def test_geometric_spacing(self):
        """Test geometric spacing generation."""
        x = discretize_aligned_range(x_min=1.0, x_max=100.0, spacing_type=SpacingType.GEOMETRIC, align_to_multiples=True, n_grid=100)
        # Should have at least MIN_GRID_SIZE points
        assert len(x) >= 100
        # Range should cover requested bounds (may extend due to alignment)
        assert x[0] <= 1.0
        assert x[-1] >= 100.0
        # Check uniform ratio
        ratios = x[1:] / x[:-1]
        assert np.allclose(ratios, ratios[0])

    def test_single_point(self):
        """Test edge case with single point - should fail with n_grid < MIN_GRID_SIZE."""
        with pytest.raises(ValueError, match="n_grid must be >= 100"):
            discretize_aligned_range(x_min=0.0, x_max=10.0, spacing_type=SpacingType.LINEAR, align_to_multiples=True, n_grid=1)

    def test_two_points_linear(self):
        """Test linear grid."""
        x = discretize_aligned_range(x_min=1.0, x_max=3.0, spacing_type=SpacingType.LINEAR, align_to_multiples=True, n_grid=100)
        # Should have at least MIN_GRID_SIZE points
        assert len(x) >= 100
        # Range should cover requested bounds (may extend due to alignment)
        assert x[0] <= 1.0
        assert x[-1] >= 3.0
        # Check uniform spacing
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])


class TestComputeBinWidth:
    """Test compute_bin_width function."""

    def test_uniform_grid(self):
        """Test bin width computation for uniform grid."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        width = compute_bin_width(x)
        assert np.isclose(width, 1.0)

    def test_nonuniform_grid_raises(self):
        """Test bin width for non-uniform grid raises error."""
        x = np.array([1.0, 2.0, 3.5, 6.0])
        # Should raise ValueError for non-uniform grid
        with pytest.raises(ValueError, match="non-uniform bin widths"):
            compute_bin_width(x)

    def test_single_point_raises(self):
        """Test that single point raises an error."""
        x = np.array([1.0])
        with pytest.raises(ValueError, match="less than 2 bins"):
            compute_bin_width(x)


class TestComputeBinRatio:
    """Test compute_bin_ratio function."""

    def test_geometric_grid(self):
        """Test ratio computation for geometric grid."""
        x = np.array([1.0, 2.0, 4.0, 8.0])
        ratio = compute_bin_ratio(x)
        assert np.isclose(ratio, 2.0)

    def test_nonuniform_grid_raises(self):
        """Test ratio for non-uniform grid raises error."""
        x = np.array([1.0, 3.0, 6.0, 18.0])
        # Should raise ValueError for non-uniform grid
        with pytest.raises(ValueError, match="non-uniform bin widths"):
            compute_bin_ratio(x)


class TestComputeDiscretePMF:
    """Test compute_discrete_PMF function."""

    def test_uniform_distribution(self):
        """Test discretization of uniform distribution."""
        dist = stats.uniform(loc=0.0, scale=1.0)
        x_array = np.linspace(0.0, 1.0, 11)
        bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist,
            x_array=x_array,
            bound_type=BoundType.DOMINATES,
            PMF_min_increment=0.0
        )

        assert len(bin_prob) == 10  # n-1 bins
        assert np.all(bin_prob >= 0)
        # For uniform, bins should have roughly equal probability
        assert np.allclose(bin_prob, 0.1, atol=0.01)
        # Tails should be near zero
        assert p_left < 0.01
        assert p_right < 0.01

    def test_normal_distribution(self):
        """Test discretization of normal distribution."""
        dist = stats.norm(loc=0.0, scale=1.0)
        # Use more points for strict accuracy
        x_array = np.linspace(-3.0, 3.0, 1001)
        bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist,
            x_array=x_array,
            bound_type=BoundType.DOMINATES,
            PMF_min_increment=0.0
        )

        # Check that probabilities sum with tails to near 1 (strict tolerance)
        total = np.sum(bin_prob) + p_left + p_right
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)

    def test_exponential_distribution(self):
        """Test discretization of exponential distribution."""
        dist = stats.expon(scale=1.0)
        x_array = np.linspace(0.0, 5.0, 51)
        bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist,
            x_array=x_array,
            bound_type=BoundType.DOMINATES,
            PMF_min_increment=0.0
        )

        # Exponential should have near-zero left tail
        assert p_left < 0.01
        # Right tail should be significant
        assert p_right > 0.0


class TestPMFRemapToGrid:
    """Test rediscritize_PMF function."""

    def test_exact_alignment(self):
        """Test remapping when grids are aligned."""
        x_in = np.array([1.0, 2.0, 3.0])
        pmf_in = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        x_out = x_in.copy()

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=True)
        assert np.allclose(pmf_out, pmf_in)

    def test_dominates_rounding(self):
        """Test dominates (pessimistic) rounding."""
        x_in = np.array([1.0, 2.5, 4.0])
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0, 4.0])

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=True)
        # 2.5 should round up to 3.0
        assert pmf_out[2] >= 0.4

    def test_is_dominated_rounding(self):
        """Test is_dominated (optimistic) rounding."""
        x_in = np.array([1.0, 2.5, 4.0])
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0, 4.0])

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=False)
        # 2.5 should round down to 2.0
        assert pmf_out[1] >= 0.4

    def test_overflow_to_infinity(self):
        """Test overflow handling."""
        x_in = np.array([1.0, 2.0, 5.0])
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0])  # 5.0 is beyond output grid

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=True)
        _, _, ppos = enforce_mass_conservation(
            pmf_out, expected_neg_inf=0.0, expected_pos_inf=0.0, bound_type=BoundType.DOMINATES
        )
        assert ppos >= 0.3

    def test_mass_conservation_in_remap(self):
        """Test that remapping conserves total mass."""
        x_in = np.array([0.5, 1.5, 2.5, 3.5])
        pmf_in = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0])

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=True)
        total_in = np.sum(pmf_in)
        pmf_out, pneg, ppos = enforce_mass_conservation(
            pmf_out, expected_neg_inf=0.0, expected_pos_inf=0.0, bound_type=BoundType.DOMINATES
        )
        total_out = np.sum(pmf_out) + pneg + ppos
        assert np.isclose(total_in, total_out, atol=TOL.MASS_CONSERVATION)


class TestCCDFComputation:
    """Test CCDF computation from GeneralDiscreteDist."""

    def test_ccdf_from_pmf_padded(self):
        dist = GeneralDiscreteDist(
            x_array=np.array([0.0, 1.0]),
            PMF_array=np.array([0.25, 0.5]),
            p_neg_inf=0.0,
            p_pos_inf=0.25
        )
        ccdf = _CCDF_from_PMF(dist)
        assert ccdf.shape == (4,)
        assert np.allclose(ccdf, np.array([1.0, 0.75, 0.25, 0.0]))
