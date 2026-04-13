"""
Unit tests for PLD_accounting.distribution_discretization module.

Tests grid generation, discretization, and PMF operations.
"""

import math

import numpy as np
import pytest
from scipy import stats

from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain, SparseDiscreteDist
from PLD_accounting.distribution_discretization import (
    _compute_discrete_prob as compute_discrete_PMF,
)
from PLD_accounting.distribution_discretization import discretize_aligned_range
from PLD_accounting.distribution_discretization import (
    rediscritize_dist,
    rediscritize_prob as pmf_remap_to_grid_kernel,
)
from PLD_accounting.distribution_utils import (
    _zero_mass,
    compute_bin_ratio,
    compute_bin_width,
    compute_truncation,
    enforce_mass_conservation,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import _CCDF_from_PMF

from tests.test_tolerances import TestTolerances as TOL


class TestDiscritizeRange:
    """Test discretize_aligned_range function."""

    def test_linear_spacing(self):
        """Test linear spacing generation."""
        x = discretize_aligned_range(
            x_min=0.0,
            x_max=10.0,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            n_grid=100,
        )
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
        x = discretize_aligned_range(
            x_min=1.0,
            x_max=100.0,
            spacing_type=SpacingType.GEOMETRIC,
            align_to_multiples=True,
            n_grid=100,
        )
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
            discretize_aligned_range(
                x_min=0.0,
                x_max=10.0,
                spacing_type=SpacingType.LINEAR,
                align_to_multiples=True,
                n_grid=1,
            )

    def test_two_points_linear(self):
        """Test linear grid."""
        x = discretize_aligned_range(
            x_min=1.0,
            x_max=3.0,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            n_grid=100,
        )
        # Should have at least MIN_GRID_SIZE points
        assert len(x) >= 100
        # Range should cover requested bounds (may extend due to alignment)
        assert x[0] <= 1.0
        assert x[-1] >= 3.0
        # Check uniform spacing
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])

    def test_linear_spacing_covers_endpoint_after_alignment_rounding(self):
        """Aligned linear grids should still cover the requested right endpoint."""
        x_min = -1.411426541779732
        x_max = 1.4160541697856062
        discretization = 0.041648652052517825

        x = discretize_aligned_range(
            x_min=x_min,
            x_max=x_max,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            discretization=discretization,
        )

        assert x[0] <= x_min
        assert x[-1] >= x_max
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
        step = compute_bin_ratio(x)
        assert np.isclose(step, 2.0)

    def test_nonuniform_grid_raises(self):
        """Test ratio for non-uniform grid raises error."""
        x = np.array([1.0, 3.0, 6.0, 18.0])
        # Should raise ValueError for non-uniform grid
        with pytest.raises(ValueError, match="non-uniform bin widths"):
            compute_bin_ratio(x)

    def test_single_point_raises(self):
        """Test that a single-point geometric grid raises an explicit error."""
        x = np.array([1.0])
        with pytest.raises(ValueError, match="less than 2 bins"):
            compute_bin_ratio(x)


class TestComputeDiscretePMF:
    """Test compute_discrete_PMF function."""

    def test_uniform_distribution(self):
        """Test discretization of uniform distribution."""
        dist = stats.uniform(loc=0.0, scale=1.0)
        x_array = np.linspace(0.0, 1.0, 11)
        bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist, x_array=x_array, bound_type=BoundType.DOMINATES, PMF_min_increment=0.0
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
            dist=dist, x_array=x_array, bound_type=BoundType.DOMINATES, PMF_min_increment=0.0
        )

        # Check that probabilities sum with tails to near 1 (strict tolerance)
        total = math.fsum([*map(float, bin_prob), p_left, p_right])
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)

    def test_exponential_distribution(self):
        """Test discretization of exponential distribution."""
        dist = stats.expon(scale=1.0)
        x_array = np.linspace(0.0, 5.0, 51)
        _bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist, x_array=x_array, bound_type=BoundType.DOMINATES, PMF_min_increment=0.0
        )

        # Exponential should have near-zero left tail
        assert p_left < 0.01
        # Right tail should be significant
        assert p_right > 0.0


class TestPMFRemapToGrid:
    """Test rediscritize_prob function."""

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
            prob_arr=pmf_out, expected_p_min=0.0, expected_p_max=0.0, bound_type=BoundType.DOMINATES
        )
        assert ppos >= 0.3

    def test_mass_conservation_in_remap(self):
        """Test that remapping conserves total mass."""
        x_in = np.array([0.5, 1.5, 2.5, 3.5])
        pmf_in = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0])

        pmf_out = pmf_remap_to_grid_kernel(x_in, pmf_in, x_out, dominates=True)
        total_in = math.fsum(map(float, pmf_in))
        pmf_out, pneg, ppos = enforce_mass_conservation(
            prob_arr=pmf_out, expected_p_min=0.0, expected_p_max=0.0, bound_type=BoundType.DOMINATES
        )
        total_out = math.fsum([*map(float, pmf_out), pneg, ppos])
        assert np.isclose(total_in, total_out, atol=TOL.MASS_CONSERVATION)


class TestCCDFComputation:
    """Test CCDF computation from SparseDiscreteDist."""

    def test_ccdf_from_pmf_padded(self):
        dist = SparseDiscreteDist(
            x_array=np.array([0.0, 1.0]), prob_arr=np.array([0.25, 0.5]), p_min=0.0, p_max=0.25
        )
        ccdf = _CCDF_from_PMF(dist)
        assert ccdf.shape == (4,)
        assert np.allclose(ccdf, np.array([1.0, 0.75, 0.25, 0.0]))


class TestEnforceMassConservation:
    """Test directional boundary enforcement semantics."""

    def test_dominates_can_consume_soft_p_min(self):
        """DOMINATES holds p_max fixed and trims from the left including p_min."""
        prob_arr = np.array([0.4, 0.1], dtype=np.float64)
        prob_out, p_min, p_max = enforce_mass_conservation(
            prob_arr=prob_arr,
            expected_p_min=0.3,
            expected_p_max=0.4,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(prob_out, np.array([0.4, 0.1]))
        assert np.isclose(p_min, 0.1)
        assert np.isclose(p_max, 0.4)
        assert np.isclose(math.fsum([*map(float, prob_out), p_min, p_max]), 1.0)

    def test_is_dominated_can_consume_soft_p_max(self):
        """IS_DOMINATED holds p_min fixed and trims from the right including p_max."""
        prob_arr = np.array([0.1, 0.4], dtype=np.float64)
        prob_out, p_min, p_max = enforce_mass_conservation(
            prob_arr=prob_arr,
            expected_p_min=0.4,
            expected_p_max=0.3,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert np.allclose(prob_out, np.array([0.1, 0.4]))
        assert np.isclose(p_min, 0.4)
        assert np.isclose(p_max, 0.1)
        assert np.isclose(math.fsum([*map(float, prob_out), p_min, p_max]), 1.0)


class TestComputeTruncation:
    """Test zero-edge stripping and index bookkeeping in truncation."""

    def test_strips_zero_edges_before_tail_truncation(self):
        new_pmf, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.0, 0.8], dtype=np.float64),
            p_min=0.0,
            p_max=0.2,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(new_pmf, np.array([0.8], dtype=np.float64))
        assert np.isclose(new_p_min, 0.0)
        assert np.isclose(new_p_max, 0.2)
        assert (min_ind, max_ind) == (1, 1)

    def test_keeps_boundary_when_it_is_the_first_remaining_element(self):
        new_pmf, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.0, 0.2, 0.5], dtype=np.float64),
            p_min=0.3,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(new_pmf, np.array([0.2, 0.5], dtype=np.float64))
        assert np.isclose(new_p_min, 0.3)
        assert np.isclose(new_p_max, 0.0)
        assert (min_ind, max_ind) == (1, 2)

    def test_truncation_folds_consumed_boundary_into_first_finite_bin(self):
        new_pmf, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.2, 0.75], dtype=np.float64),
            p_min=0.05,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(new_pmf, np.array([0.25, 0.75], dtype=np.float64))
        assert np.isclose(new_p_min, 0.0)
        assert np.isclose(new_p_max, 0.0)
        assert (min_ind, max_ind) == (0, 1)

    def test_strips_zero_edges_for_is_dominated_right_tail(self):
        new_pmf, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.8, 0.0], dtype=np.float64),
            p_min=0.2,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert np.allclose(new_pmf, np.array([0.8], dtype=np.float64))
        assert np.isclose(new_p_min, 0.2)
        assert np.isclose(new_p_max, 0.0)
        assert (min_ind, max_ind) == (0, 0)

    def test_dense_truncate_edges_updates_x_min_after_zero_edge_removal(self):
        dist = DenseDiscreteDist(
            x_min=0.0,
            step=1.0,
            prob_arr=np.array([0.0, 0.8], dtype=np.float64),
            p_max=0.2,
        )

        result = dist.truncate_edges(0.1, BoundType.DOMINATES)

        assert np.allclose(result.x_array, np.array([1.0], dtype=np.float64))
        assert np.allclose(result.prob_arr, np.array([0.8], dtype=np.float64))
        assert np.isclose(result.p_min, 0.0)
        assert np.isclose(result.p_max, 0.2)

    def test_sparse_truncate_edges_updates_support_after_tail_zero_removal(self):
        dist = SparseDiscreteDist(
            x_array=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            prob_arr=np.array([0.8, 0.1, 0.1], dtype=np.float64),
        )

        result = dist.truncate_edges(0.15, BoundType.DOMINATES)

        assert np.allclose(result.x_array, np.array([1.0, 2.0], dtype=np.float64))
        assert np.allclose(result.prob_arr, np.array([0.8, 0.1], dtype=np.float64))
        assert np.isclose(result.p_min, 0.0)
        assert np.isclose(result.p_max, 0.1)


class TestZeroMass:
    """Test directional zero-mass helper edge cases."""

    def test_raises_when_mass_is_at_least_total(self):
        with pytest.raises(ValueError, match="mass must be smaller than total array mass"):
            _zero_mass(
                values=np.array([0.2, 0.8], dtype=np.float64),
                mass=1.0,
                from_left=True,
                exact=True,
            )


class TestRediscretizeBoundaryFolding:
    """Test rediscretization of soft boundary masses into edge bins."""

    def test_rediscretize_near_point_mass_distribution(self):
        # prob_arr has two nonzero bins so a valid grid range exists after truncation.
        # Both bins have mass >> tail_truncation so neither is consumed.
        dist = DenseDiscreteDist(
            x_min=0.5,
            step=0.5,
            prob_arr=np.array([1.0 - 1e-6, 1e-6], dtype=np.float64),
        )

        result = rediscritize_dist(
            dist=dist,
            tail_truncation=1e-8,
            loss_discretization=1e-2,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.DOMINATES,
        )

        assert np.isclose(result.step, 1e-2)
        assert np.isclose(result.x_array[0], 0.5)
        assert np.isclose(result.x_array[-1], 1.0)
        total = math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max])
        assert np.isclose(total, 1.0)

    def test_is_dominated_moves_p_max_into_last_finite_cell(self):
        dist = DenseDiscreteDist.from_x_array(
            x_array=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_max=0.1,
        )

        result = rediscritize_dist(
            dist=dist,
            tail_truncation=0.0,
            loss_discretization=1.0,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert np.isclose(result.p_max, 0.0)
        assert np.isclose(result.prob_arr[-1], 0.5)
        assert np.isclose(
            math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max]), 1.0
        )

    def test_dominates_linear_moves_p_min_into_first_finite_cell(self):
        dist = DenseDiscreteDist.from_x_array(
            x_array=np.array([0.0, 1.0, 2.0], dtype=np.float64),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_min=0.1,
        )

        result = rediscritize_dist(
            dist=dist,
            tail_truncation=0.0,
            loss_discretization=1.0,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.DOMINATES,
        )

        assert np.isclose(result.p_min, 0.0)
        assert np.isclose(result.prob_arr[0], 0.3)
        assert np.isclose(
            math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max]), 1.0
        )

    def test_dominates_geometric_keeps_zero_atom(self):
        dist = DenseDiscreteDist.from_x_array(
            x_array=np.array([1.0, 2.0, 4.0], dtype=np.float64),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_min=0.1,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

        result = rediscritize_dist(
            dist=dist,
            tail_truncation=0.0,
            loss_discretization=np.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            bound_type=BoundType.DOMINATES,
        )

        assert np.isclose(result.p_min, 0.1)
        assert np.isclose(
            math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max]), 1.0
        )
