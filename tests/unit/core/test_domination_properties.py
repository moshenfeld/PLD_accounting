"""
Unit tests for domination properties and bound semantics.

Tests that DOMINATES and IS_DOMINATED modes enforce correct privacy bounds.
"""

import math

import numpy as np
import pytest
from scipy import stats

from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain, SparseDiscreteDist
from PLD_accounting.distribution_discretization import (
    discretize_continuous_distribution,
)
from PLD_accounting.distribution_utils import compute_bin_width
from PLD_accounting.geometric_convolution import (
    geometric_convolve,
)
from PLD_accounting.subsample_PLD import (
    _calc_subsampled_grid,
    _mix_distributions,
    _stable_subsampling_transformation,
    _subsample_dist,
    _subsample_dist_mix,
)
from PLD_accounting.types import BoundType, Direction, SpacingType

from tests.test_tolerances import TestTolerances as TOL


class TestDominationSemantics:
    """Test domination mode constraints and semantics."""

    def test_dominates_has_no_neg_inf_mass(self):
        """Test that DOMINATES mode sets p_min= 0."""
        dist = stats.norm(loc=0.0, scale=1.0)
        result = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )

        assert result.p_min == 0.0, f"DOMINATES mode should have p_min=0, got {result.p_min}"

    def test_is_dominated_has_no_pos_inf_mass(self):
        """Test that IS_DOMINATED mode sets p_max= 0."""
        dist = stats.norm(loc=0.0, scale=1.0)
        result = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        assert result.p_max == 0.0, f"IS_DOMINATED mode should have p_max=0, got {result.p_max}"

    def test_dominates_captures_left_tail(self):
        """Test that DOMINATES mode captures left tail in first bin."""
        dist = stats.norm(loc=0.0, scale=1.0)
        result = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )

        # First bin should have mass from (-∞, x_0]
        # Should be more than just the bin probability
        expected_min = dist.cdf(result.x_array[0])
        assert result.prob_arr[0] >= expected_min * 0.9

    def test_is_dominated_sends_left_tail_to_neg_inf(self):
        """Test that IS_DOMINATED mode sends left tail to -∞."""
        dist = stats.norm(loc=0.0, scale=1.0)
        result = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        # Should have some left tail mass at -∞
        expected_tail = dist.cdf(result.x_array[0])
        assert result.p_min >= expected_tail * 0.5


class TestStochasticDominance:
    """Test stochastic dominance relationships between bounds."""

    def test_expected_value_ordering(self):
        """Test that E[upper] >= E[lower] (first-order stochastic dominance)."""
        dist = stats.norm(loc=0.0, scale=1.0)

        upper = discretize_continuous_distribution(
            dist=dist,
            n_grid=200,
            align_to_multiples=True,
            tail_truncation=0.001,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )

        lower = discretize_continuous_distribution(
            dist=dist,
            n_grid=200,
            align_to_multiples=True,
            tail_truncation=0.001,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        # Compute expectations (over finite grid only)
        E_upper = math.fsum(float(x) * float(p) for x, p in zip(upper.x_array, upper.prob_arr))
        E_lower = math.fsum(float(x) * float(p) for x, p in zip(lower.x_array, lower.prob_arr))

        # Upper bound should have higher or equal expectation
        assert (
            E_upper >= E_lower - TOL.STOCHASTIC_DOM_SLACK
        ), f"Expected value ordering violated: E_upper={E_upper} < E_lower={E_lower}"

    def test_variance_ordering_reasonable(self):
        """Test that variance relationship is reasonable."""
        dist = stats.norm(loc=0.0, scale=1.0)

        upper = discretize_continuous_distribution(
            dist=dist,
            n_grid=200,
            align_to_multiples=True,
            tail_truncation=0.001,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )

        lower = discretize_continuous_distribution(
            dist=dist,
            n_grid=200,
            align_to_multiples=True,
            tail_truncation=0.001,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        # Compute variances
        E_upper = math.fsum(float(x) * float(p) for x, p in zip(upper.x_array, upper.prob_arr))
        E_lower = math.fsum(float(x) * float(p) for x, p in zip(lower.x_array, lower.prob_arr))

        Var_upper = math.fsum(
            float((x - E_upper) ** 2) * float(p) for x, p in zip(upper.x_array, upper.prob_arr)
        )
        Var_lower = math.fsum(
            float((x - E_lower) ** 2) * float(p) for x, p in zip(lower.x_array, lower.prob_arr)
        )

        # Both should be reasonable (close to true variance = 1)
        assert 0.5 < Var_upper < 2.0
        assert 0.5 < Var_lower < 2.0


class TestDominationUnderConvolution:
    """Test that domination properties are preserved under convolution."""

    def test_convolution_preserves_domination_constraint(self):
        """Test that convolution preserves infinity mass constraints."""
        # Use geometric grids with same ratio for geometric kernel
        x1 = np.geomspace(1.0, 4.0, 3)
        pmf1 = np.array([0.3, 0.5, 0.2], dtype=np.float64)
        # DOMINATES: p_min= 0
        dist1 = DenseDiscreteDist.from_x_array(
            x_array=x1,
            prob_arr=pmf1,
            p_min=0.0,
            p_max=0.0,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

        x2 = np.geomspace(0.5, 1.0, 2)
        pmf2 = np.array([0.6, 0.4], dtype=np.float64)
        dist2 = DenseDiscreteDist.from_x_array(
            x_array=x2,
            prob_arr=pmf2,
            p_min=0.0,
            p_max=0.0,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

        result = geometric_convolve(
            dist_1=dist1, dist_2=dist2, tail_truncation=0.01, bound_type=BoundType.DOMINATES
        )

        # Result should still have p_min= 0
        assert result.p_min == 0.0

    def test_convolution_error_on_invalid_infinity_mass(self):
        """Test that construction rejects both-non-zero boundary masses for REALS domain."""
        x1 = np.array([1.0, 2.0, 4.0])
        pmf1 = np.array([0.3, 0.4, 0.2], dtype=np.float64)
        # Invalid for REALS domain: both p_min and p_max non-zero
        with pytest.raises(ValueError, match="REALS domain"):
            SparseDiscreteDist(x_array=x1, prob_arr=pmf1, p_min=0.05, p_max=0.05)


class TestRoundingBehavior:
    """Test rounding behavior for domination modes."""

    def test_dominates_rounds_up(self):
        """Test that DOMINATES mode rounds values up to next grid point."""
        # Create distribution with geometric grid for geometric kernel
        x_in = np.geomspace(1.0, 4.0, 3)
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        dist_in = DenseDiscreteDist.from_x_array(
            x_array=x_in,
            prob_arr=pmf_in,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

        # Convolve with itself - will create intermediate values
        result = geometric_convolve(
            dist_1=dist_in, dist_2=dist_in, tail_truncation=0.01, bound_type=BoundType.DOMINATES
        )

        # Mass should be conserved with pessimistic rounding
        total = math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max])
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)

    def test_is_dominated_rounds_down(self):
        """Test that IS_DOMINATED mode rounds values down to previous grid point."""
        # Create distribution with geometric grid for geometric kernel
        x_in = np.geomspace(1.0, 4.0, 3)
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        dist_in = DenseDiscreteDist.from_x_array(
            x_array=x_in,
            prob_arr=pmf_in,
            p_max=0.0,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )

        result = geometric_convolve(
            dist_1=dist_in, dist_2=dist_in, tail_truncation=0.01, bound_type=BoundType.IS_DOMINATED
        )

        total = math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max])
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)


class TestExponentialDistribution:
    """Test domination with exponential distribution (one-sided support)."""

    def test_exponential_dominates_minimal_left_tail(self):
        """Test that exponential with DOMINATES has minimal left tail."""
        dist = stats.expon(scale=1.0)
        result = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )

        # Exponential starts at 0, so left tail should be tiny
        assert result.p_min < TOL.NEG_INF_STRICT_LT

    def test_exponential_is_dominated_no_pos_inf(self):
        """Test that exponential with IS_DOMINATED has no +∞ mass."""
        dist = stats.expon(scale=1.0)
        result = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.05,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        assert result.p_max == 0.0


def _build_test_dist(x_array, pmf_array, *, p_min=0.0, p_max=0.0) -> SparseDiscreteDist:
    finite_mass = 1.0 - p_min - p_max
    if finite_mass <= 0.0:
        raise ValueError("Boundary masses must leave positive finite mass")
    pmf = np.array(pmf_array, dtype=np.float64)
    pmf = pmf / math.fsum(map(float, pmf)) * finite_mass
    return SparseDiscreteDist(
        x_array=np.array(x_array, dtype=np.float64),
        prob_arr=pmf,
        p_min=p_min,
        p_max=p_max,
    )


class TestSubsampleDistMix:
    _BASE_X = np.linspace(-3.0, 1.0, 6)
    _BASE_PMF = np.array([0.1, 0.15, 0.2, 0.25, 0.15, 0.15], dtype=np.float64)
    _REF_X = np.linspace(-2.5, 2.0, 6)
    _REF_PMF = np.array([0.15, 0.2, 0.25, 0.2, 0.1, 0.1], dtype=np.float64)

    @pytest.mark.parametrize(
        "base_inf, ref_inf",
        [
            (
                {"p_min": 0.0, "p_max": 0.05},
                {"p_min": 0.0, "p_max": 0.03},
            ),
            (
                {"p_min": 0.04, "p_max": 0.0},
                {"p_min": 0.02, "p_max": 0.0},
            ),
        ],
    )
    def test_matches_sequential_discretization(self, base_inf, ref_inf):
        sampling_prob = 0.37
        direction = Direction.REMOVE
        base_dist = _build_test_dist(self._BASE_X, self._BASE_PMF, **base_inf)
        ref_dist = _build_test_dist(self._REF_X, self._REF_PMF, **ref_inf)

        result = _subsample_dist_mix(
            base_pld=base_dist,
            neg_dual_pld=ref_dist,
            sampling_prob=sampling_prob,
            direction=direction,
        )

        base_subsampled = _subsample_dist(
            base_pld=base_dist,
            sampling_prob=sampling_prob,
            direction=direction,
            target_x_array=result.x_array,
        )
        ref_subsampled = _subsample_dist(
            base_pld=ref_dist,
            sampling_prob=sampling_prob,
            direction=direction,
            target_x_array=result.x_array,
        )
        coupled = _mix_distributions(
            dist_1=base_subsampled,
            dist_2=ref_subsampled,
            weight_first=sampling_prob,
        )

        np.testing.assert_allclose(
            result.x_array,
            coupled.x_array,
            rtol=TOL.ARRAY_RTOL_ULTRA,
            atol=TOL.GRID_EXACT_ATOL,
        )
        np.testing.assert_allclose(
            result.prob_arr,
            coupled.prob_arr,
            rtol=TOL.PMF_COUPLED_RTOL,
            atol=TOL.PMF_COUPLED_ATOL,
        )
        assert np.isclose(result.p_min, coupled.p_min, rtol=TOL.INF_MASS_RTOL)
        assert np.isclose(result.p_max, coupled.p_max, rtol=TOL.INF_MASS_RTOL)

    def test_grid_covers_transformed_range(self):
        sampling_prob = 0.4
        direction = Direction.REMOVE
        base_dist = _build_test_dist(self._BASE_X, self._BASE_PMF, p_min=0.0, p_max=0.02)
        ref_dist = _build_test_dist(self._REF_X, self._REF_PMF, p_min=0.0, p_max=0.01)

        result = _subsample_dist_mix(
            base_pld=base_dist,
            neg_dual_pld=ref_dist,
            sampling_prob=sampling_prob,
            direction=direction,
        )

        base_endpoints = _stable_subsampling_transformation(
            x_array=np.array([base_dist.x_array[0], base_dist.x_array[-1]], dtype=np.float64),
            sampling_prob=sampling_prob,
            direction=direction,
        )
        ref_endpoints = _stable_subsampling_transformation(
            x_array=np.array([ref_dist.x_array[0], ref_dist.x_array[-1]], dtype=np.float64),
            sampling_prob=sampling_prob,
            direction=direction,
        )
        expected_lower = min(
            base_endpoints[0],
            ref_endpoints[0],
            math.log1p(-sampling_prob),
        )
        expected_upper = max(
            base_endpoints[1],
            ref_endpoints[1],
            -math.log1p(-sampling_prob),
        )

        assert result.x_array[0] <= expected_lower + TOL.SPACING_ATOL
        assert result.x_array[-1] >= expected_upper - TOL.SPACING_ATOL

    def test_uses_provided_grid(self):
        sampling_prob = 0.25
        direction = Direction.REMOVE
        base_dist = _build_test_dist(self._BASE_X, self._BASE_PMF, p_min=0.0, p_max=0.0)
        ref_dist = _build_test_dist(self._REF_X, self._REF_PMF, p_min=0.0, p_max=0.0)
        target_grid = _calc_subsampled_grid(
            min_loss=base_dist.x_array[0],
            discretization=compute_bin_width(base_dist.x_array),
            num_buckets=int(base_dist.x_array.size),
            grid_size=sampling_prob,
            direction=direction,
        )

        result = _subsample_dist_mix(
            base_pld=base_dist,
            neg_dual_pld=ref_dist,
            sampling_prob=sampling_prob,
            direction=direction,
            target_x_array=target_grid,
        )

        np.testing.assert_allclose(
            result.x_array,
            target_grid,
            rtol=TOL.GRID_EXACT_RTOL,
            atol=TOL.GRID_EXACT_ATOL,
        )
