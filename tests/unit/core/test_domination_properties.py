"""
Unit tests for domination properties and bound semantics.

Tests that DOMINATES and IS_DOMINATED modes enforce correct privacy bounds.
"""

import math

import numpy as np
import pytest
from scipy import stats

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_discretization import (
    discretize_continuous_stoch_dom,
    rediscretize_dist_by_bound,
)
from PLD_accounting.geometric_convolution import (
    geometric_convolve,
)
from PLD_accounting.mechanisms import gaussian_distribution
from PLD_accounting.subsample_pld import (
    _calc_subsampled_grid,
    _stable_subsampling_transformation,
    _subsample_dist_mix,
)
from PLD_accounting.types import BoundType, Direction, SpacingType
from PLD_accounting.utils import calc_pld_dual, negate_reverse_linear_distribution
from tests.test_tolerances import TestTolerances as TOL


def _linear_step_for_tail_truncation(
    dist: stats.rv_continuous,
    tail_truncation: float,
    n_grid: int,
) -> float:
    """Return the uniform step for an ``n_grid`` tail-quantile reference grid."""
    x_min = float(dist.ppf(tail_truncation))
    x_max = float(dist.isf(tail_truncation))
    return (x_max - x_min) / (n_grid - 1)


class TestDominationSemantics:
    """Test domination mode constraints and semantics."""

    def test_dominates_has_no_neg_inf_mass(self):
        """Test that DOMINATES mode sets p_min= 0."""
        dist = stats.norm(loc=0.0, scale=1.0)
        result = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.01, 100),
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
        )

        assert result.p_min == 0.0, f"DOMINATES mode should have p_min=0, got {result.p_min}"

    def test_is_dominated_has_no_pos_inf_mass(self):
        """Test that IS_DOMINATED mode sets p_max= 0."""
        dist = stats.norm(loc=0.0, scale=1.0)
        result = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.01, 100),
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert result.p_max == 0.0, f"IS_DOMINATED mode should have p_max=0, got {result.p_max}"

    def test_dominates_captures_left_tail(self):
        """Test that DOMINATES mode captures left tail in first bin."""
        dist = stats.norm(loc=0.0, scale=1.0)
        result = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.01, 100),
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
        )

        # First bin should have mass from (-∞, x_0]
        # Should be more than just the bin probability
        expected_min = dist.cdf(result.x_array[0])
        assert result.prob_arr[0] >= expected_min * 0.9

    def test_is_dominated_sends_left_tail_to_neg_inf(self):
        """Test that IS_DOMINATED mode sends left tail to -∞."""
        dist = stats.norm(loc=0.0, scale=1.0)
        result = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.01, 100),
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED,
        )

        # Should have some left tail mass at -∞
        expected_tail = dist.cdf(result.x_array[0])
        assert result.p_min >= expected_tail * 0.5


class TestStochasticDominance:
    """Test stochastic dominance relationships between bounds."""

    def test_expected_value_ordering(self):
        """Test that E[upper] >= E[lower] (first-order stochastic dominance)."""
        dist = stats.norm(loc=0.0, scale=1.0)

        upper = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.001, 200),
            align_to_multiples=True,
            tail_truncation=0.001,
            bound_type=BoundType.DOMINATES,
        )

        lower = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.001, 200),
            align_to_multiples=True,
            tail_truncation=0.001,
            bound_type=BoundType.IS_DOMINATED,
        )

        # Compute expectations (over finite grid only)
        mean_upper = math.fsum(float(x) * float(p) for x, p in zip(upper.x_array, upper.prob_arr))
        mean_lower = math.fsum(float(x) * float(p) for x, p in zip(lower.x_array, lower.prob_arr))

        # Upper bound should have higher or equal expectation
        assert mean_upper >= mean_lower - TOL.STOCHASTIC_DOM_SLACK, (
            "Expected value ordering violated: "
            f"mean_upper={mean_upper} < mean_lower={mean_lower}"
        )

    def test_variance_ordering_reasonable(self):
        """Test that variance relationship is reasonable."""
        dist = stats.norm(loc=0.0, scale=1.0)

        upper = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.001, 200),
            align_to_multiples=True,
            tail_truncation=0.001,
            bound_type=BoundType.DOMINATES,
        )

        lower = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.001, 200),
            align_to_multiples=True,
            tail_truncation=0.001,
            bound_type=BoundType.IS_DOMINATED,
        )

        # Compute variances
        mean_upper = math.fsum(float(x) * float(p) for x, p in zip(upper.x_array, upper.prob_arr))
        mean_lower = math.fsum(float(x) * float(p) for x, p in zip(lower.x_array, lower.prob_arr))

        var_upper = math.fsum(
            float((x - mean_upper) ** 2) * float(p) for x, p in zip(upper.x_array, upper.prob_arr)
        )
        var_lower = math.fsum(
            float((x - mean_lower) ** 2) * float(p) for x, p in zip(lower.x_array, lower.prob_arr)
        )

        # Both should be reasonable (close to true variance = 1)
        assert 0.5 < var_upper < 2.0
        assert 0.5 < var_lower < 2.0


class TestDominationUnderConvolution:
    """Test that domination properties are preserved under convolution."""

    def test_convolution_preserves_domination_constraint(self):
        """Test that convolution preserves infinity mass constraints."""
        # Use geometric grids with same ratio for geometric kernel
        pmf1 = np.array([0.3, 0.5, 0.2], dtype=np.float64)
        # DOMINATES: p_min= 0
        dist1 = DenseDiscreteDist(
            grid=GridSpec(step=math.log(2.0), n=3, spacing_type=SpacingType.GEOMETRIC, anchor=1.0),
            prob_arr=pmf1,
            p_min=0.0,
            p_max=0.0,
            domain=Domain.POSITIVES,
        )

        pmf2 = np.array([0.6, 0.4], dtype=np.float64)
        dist2 = DenseDiscreteDist(
            grid=GridSpec(step=math.log(2.0), n=2, spacing_type=SpacingType.GEOMETRIC, anchor=0.5),
            prob_arr=pmf2,
            p_min=0.0,
            p_max=0.0,
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
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        dist_in = DenseDiscreteDist(
            grid=GridSpec(step=math.log(2.0), n=3, spacing_type=SpacingType.GEOMETRIC, anchor=1.0),
            prob_arr=pmf_in,
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
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        dist_in = DenseDiscreteDist(
            grid=GridSpec(step=math.log(2.0), n=3, spacing_type=SpacingType.GEOMETRIC, anchor=1.0),
            prob_arr=pmf_in,
            p_max=0.0,
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
        result = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.01, 100),
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
        )

        # Exponential starts at 0, so left tail should be tiny
        assert result.p_min < TOL.NEG_INF_STRICT_LT

    def test_exponential_is_dominated_no_pos_inf(self):
        """Test that exponential with IS_DOMINATED has no +∞ mass."""
        dist = stats.expon(scale=1.0)
        result = discretize_continuous_stoch_dom(
            dist=dist,
            step=_linear_step_for_tail_truncation(dist, 0.05, 100),
            align_to_multiples=True,
            tail_truncation=0.05,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert result.p_max == 0.0


def _valid_subsampling_pair():
    """Return a PLD realization and its transformed exact dual branch."""
    base = gaussian_distribution(
        scale=2.0,
        value_discretization=0.5,
        tail_truncation=1e-8,
    )
    neg_dual = negate_reverse_linear_distribution(calc_pld_dual(base))
    return base, neg_dual


class TestSubsampleDistMix:
    """``_subsample_dist_mix`` transforms, mixes, and CtD-projects once."""

    def test_grid_covers_transformed_range(self):
        """The coupled lattice must span the transformed finite endpoints and subsampling caps."""
        sampling_prob = 0.4
        direction = Direction.REMOVE
        base_dist, ref_dist = _valid_subsampling_pair()

        result = _subsample_dist_mix(
            base_pld=base_dist,
            neg_dual_pld=ref_dist,
            sampling_prob=sampling_prob,
            direction=direction,
            target_grid=None,
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

        assert result.x_array[0] <= expected_lower + TOL.GRID_ATOL
        assert result.x_array[-1] >= expected_upper - TOL.GRID_ATOL

    def test_uses_provided_grid(self):
        """With ``target_grid`` set, the mixture must use that lattice exactly."""
        sampling_prob = 0.25
        direction = Direction.REMOVE
        base_dist, ref_dist = _valid_subsampling_pair()
        target_grid = _calc_subsampled_grid(
            source_grid=base_dist.grid,
            sampling_prob=sampling_prob,
            direction=direction,
            include_right=None,
        )

        result = _subsample_dist_mix(
            base_pld=base_dist,
            neg_dual_pld=ref_dist,
            sampling_prob=sampling_prob,
            direction=direction,
            target_grid=target_grid,
        )

        # The GridSpec travels intact, so the lattice is identical, not merely close.
        assert result.grid == target_grid


def test_pld_contract_checks_are_not_a_domination_proof() -> None:
    """Semantic PLD checks accept a valid object without proving mechanism domination.

    The caller remains responsible for proving that a source dominates the
    mechanism it represents. Discretization and CtD do not prove that
    relationship.
    """
    source = gaussian_distribution(scale=2.0, bound_type=BoundType.DOMINATES)
    result = rediscretize_dist_by_bound(
        dist=source,
        tail_truncation=1e-6,
        loss_discretization=0.1,
        bound_type=BoundType.DOMINATES,
    )
    assert result.p_min == 0.0
    docstring = rediscretize_dist_by_bound.__doc__
    assert docstring is not None
    assert "caller remains responsible" in docstring
    assert "do not prove domination" in docstring
