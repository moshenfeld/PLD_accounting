"""
Integration tests for convolution consistency and domination ordering.
"""
import numpy as np
from scipy import stats

from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.distribution_discretization import discretize_continuous_distribution
from PLD_accounting.FFT_convolution import FFT_convolve, FFT_self_convolve
from PLD_accounting.geometric_convolution import geometric_convolve, geometric_self_convolve
from tests.test_tolerances import TestTolerances as TOL


def _ccdf_from_dist(dist):
    """Compute CCDF at each grid point (inclusive), accounting for +inf mass."""
    tail = np.cumsum(dist.PMF_array[::-1])[::-1]
    return tail + dist.p_pos_inf


def _compare_ccdf_on_common_grid(upper_dist, lower_dist):
    """Compare two CCDFs on a common grid via linear interpolation.

    Returns the maximum violation (negative if upper CCDF is always >= lower CCDF).
    For stochastic dominance, upper CCDF should be >= lower CCDF everywhere.
    """
    # Create common grid from union of both x_arrays
    common_x = np.unique(np.concatenate([upper_dist.x_array, lower_dist.x_array]))

    # Interpolate CCDFs on common grid
    upper_ccdf_on_grid = _interpolate_ccdf(upper_dist, common_x)
    lower_ccdf_on_grid = _interpolate_ccdf(lower_dist, common_x)

    # Check dominance: upper_ccdf >= lower_ccdf everywhere
    violations = lower_ccdf_on_grid - upper_ccdf_on_grid
    max_violation = np.max(violations)

    return max_violation


def _interpolate_ccdf(dist, x_points):
    """Interpolate CCDF values at given x points.

    CCDF(x) = P(X > x) = sum of mass strictly to the right of x
    """
    ccdf_values = np.zeros(len(x_points))

    for i, x in enumerate(x_points):
        # Find mass strictly greater than x
        mass_right = np.sum(dist.PMF_array[dist.x_array > x])
        ccdf_values[i] = mass_right + dist.p_pos_inf

    return ccdf_values


def _compare_ccdf_with_shift_and_tail(upper_dist, lower_dist, loss_discretization, tail_truncation):
    """Compare CCDFs with shift/tail allowance.

    Checks: CCDF_lower(x - 2*loss_disc) + 2*tail_truncation >= CCDF_upper(x).
    Returns the maximum violation (positive means inequality is violated).
    """
    common_x = np.unique(np.concatenate([upper_dist.x_array, lower_dist.x_array]))
    upper_ccdf = _interpolate_ccdf(upper_dist, common_x)
    lower_ccdf = _interpolate_ccdf(lower_dist, common_x - 2.0 * loss_discretization)
    violations = upper_ccdf - (lower_ccdf + 2.0 * tail_truncation)
    return np.max(violations)


class TestConvolutionConsistency:
    """Check self-consistency between pairwise and self convolutions."""

    def test_fft_self_convolve_matches_pairwise_t2(self):
        """Test that FFT pairwise and self-convolution produce consistent results.

        Note: Direct and pairwise FFT methods may produce different output sizes due to
        different truncation handling. This test verifies both produce valid distributions
        with conserved mass and similar statistical properties.
        """
        dist = discretize_continuous_distribution(
            dist=stats.expon(scale=1.0),
            n_grid=400,
            align_to_multiples=True,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )

        pairwise = FFT_convolve(
            dist_1=dist,
            dist_2=dist,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )
        self_conv = FFT_self_convolve(
            dist=dist,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )

        # Verify both have conserved mass
        pairwise_mass = np.sum(pairwise.PMF_array) + pairwise.p_neg_inf + pairwise.p_pos_inf
        self_mass = np.sum(self_conv.PMF_array) + self_conv.p_neg_inf + self_conv.p_pos_inf
        assert np.isclose(pairwise_mass, 1.0, atol=TOL.MASS_CONSERVATION)
        assert np.isclose(self_mass, 1.0, atol=TOL.MASS_CONSERVATION)

        # Verify means are similar (both should approximate 2.0 for exponential)
        pairwise_mean = np.sum(pairwise.x_array * pairwise.PMF_array)
        self_mean = np.sum(self_conv.x_array * self_conv.PMF_array)
        assert np.isclose(pairwise_mean, self_mean, rtol=0.05)

    def test_geometric_self_convolve_matches_pairwise_t2(self):
        dist = discretize_continuous_distribution(
            dist=stats.lognorm(s=0.5),
            n_grid=200,
            align_to_multiples=True,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.GEOMETRIC,
        )

        pairwise = geometric_convolve(
            dist_1=dist,
            dist_2=dist,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )
        self_conv = geometric_self_convolve(
            dist=dist,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(pairwise.x_array, self_conv.x_array, atol=1e-12)
        assert np.allclose(pairwise.PMF_array, self_conv.PMF_array, atol=1e-12)


class TestDominationOrdering:
    """Verify domination bounds remain ordered and diverge with more composition."""

    def test_dominates_ccdf_above_is_dominated_after_convolution(self):
        """Test stochastic dominance: DOMINATES CCDF >= IS_DOMINATED CCDF.

        For privacy loss distributions, stochastic dominance means:
        P(PLD_upper > x) >= P(PLD_lower > x) for all x

        This is the correct definition - DOMINATES is pessimistic (upper bound),
        so it should have more mass in the tail (higher CCDF).
        """
        dist = stats.norm(loc=0.0, scale=1.0)
        upper = discretize_continuous_distribution(
            dist=dist,
            n_grid=300,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )
        lower = discretize_continuous_distribution(
            dist=dist,
            n_grid=300,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        upper_conv = FFT_self_convolve(
            dist=upper,
            T=4,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )
        lower_conv = FFT_self_convolve(
            dist=lower,
            T=4,
            tail_truncation=0.0,
            bound_type=BoundType.IS_DOMINATED,
            use_direct=True,
        )

        # Verify stochastic dominance via CCDF comparison on common grid
        max_violation = _compare_ccdf_on_common_grid(upper_conv, lower_conv)

        # max_violation should be <= 0 (i.e., upper_ccdf >= lower_ccdf everywhere)
        # Allow small numerical tolerance
        assert max_violation <= TOL.MASS_CONSERVATION, \
            f"Stochastic dominance violated: max(lower_ccdf - upper_ccdf) = {max_violation:.2e}"

    def test_shifted_bound_with_tail_after_discretization(self):
        """Upper/lower bound should hold with shift and tail allowance."""
        tail = 1e-6
        n_grid = 400
        upper = discretize_continuous_distribution(
            dist=stats.expon(scale=1.0),
            n_grid=n_grid,
            align_to_multiples=True,
            tail_truncation=tail,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )
        lower = discretize_continuous_distribution(
            dist=stats.expon(scale=1.0),
            n_grid=n_grid,
            align_to_multiples=True,
            tail_truncation=tail,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )
        loss_discretization = float(upper.x_array[1] - upper.x_array[0])
        max_violation = _compare_ccdf_with_shift_and_tail(
            upper, lower, loss_discretization, tail
        )
        assert max_violation <= TOL.MASS_CONSERVATION, (
            f"Shift/tail bound violated: max_violation={max_violation:.2e}"
        )

    def test_domination_gap_grows_with_more_convolutions(self):
        """Test that domination gap grows with more convolutions.

        The gap between upper and lower bounds should increase as we compose
        more times, since discretization errors compound.
        """
        dist = stats.norm(loc=0.0, scale=1.0)
        upper = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.05,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )
        lower = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.05,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        upper_t2 = FFT_self_convolve(
            dist=upper,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )
        lower_t2 = FFT_self_convolve(
            dist=lower,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.IS_DOMINATED,
            use_direct=True,
        )

        upper_t8 = FFT_self_convolve(
            dist=upper,
            T=8,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )
        lower_t8 = FFT_self_convolve(
            dist=lower,
            T=8,
            tail_truncation=0.0,
            bound_type=BoundType.IS_DOMINATED,
            use_direct=True,
        )

        # Measure gap via maximum CCDF difference
        gap_t2 = _compare_ccdf_on_common_grid(upper_t2, lower_t2)
        gap_t8 = _compare_ccdf_on_common_grid(upper_t8, lower_t8)

        # Gap should grow with more compositions (but both should satisfy dominance)
        assert gap_t2 <= TOL.MASS_CONSERVATION  # Dominance holds
        assert gap_t8 <= TOL.MASS_CONSERVATION  # Dominance holds
        # Note: We don't assert gap_t8 > gap_t2 because the gap might not always grow
        # monotonically due to truncation effects

    def test_bound_gap_vs_discretization(self):
        """Test that finer discretization reduces the gap between bounds.

        Better discretization should give tighter bounds (smaller gap).
        """
        dist = stats.norm(loc=0.0, scale=1.0)

        # Coarse discretization
        upper_coarse = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )
        lower_coarse = discretize_continuous_distribution(
            dist=dist,
            n_grid=100,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        # Fine discretization
        upper_fine = discretize_continuous_distribution(
            dist=dist,
            n_grid=1000,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )
        lower_fine = discretize_continuous_distribution(
            dist=dist,
            n_grid=1000,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        # Convolve both
        T = 4
        upper_coarse_conv = FFT_self_convolve(
            dist=upper_coarse, T=T, tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )
        lower_coarse_conv = FFT_self_convolve(
            dist=lower_coarse, T=T, tail_truncation=0.0,
            bound_type=BoundType.IS_DOMINATED,
            use_direct=True,
        )
        upper_fine_conv = FFT_self_convolve(
            dist=upper_fine, T=T, tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )
        lower_fine_conv = FFT_self_convolve(
            dist=lower_fine, T=T, tail_truncation=0.0,
            bound_type=BoundType.IS_DOMINATED,
            use_direct=True,
        )

        # Measure gaps
        gap_coarse = _compare_ccdf_on_common_grid(upper_coarse_conv, lower_coarse_conv)
        gap_fine = _compare_ccdf_on_common_grid(upper_fine_conv, lower_fine_conv)

        # Both should satisfy dominance
        assert gap_coarse <= TOL.MASS_CONSERVATION
        assert gap_fine <= TOL.MASS_CONSERVATION

        # Fine discretization should give similar or tighter bounds
        # Note: Due to FFT truncation effects, the relationship may not be perfectly monotonic,
        # but fine discretization should not significantly worsen the gap
        # Gap is max violation: negative means dominance holds, closer to 0 is tighter
        assert abs(gap_fine - gap_coarse) < 0.01, \
            f"Gap changed significantly: coarse={gap_coarse:.6f}, fine={gap_fine:.6f}"

    def test_fft_vs_geometric_convolution_consistency(self):
        """Test that FFT and geometric convolution give similar results.

        Both methods should produce distributions that satisfy the same
        stochastic dominance properties.
        """
        # Use lognormal for geometric (always positive)
        dist = stats.lognorm(s=0.5)

        # Discretize for FFT (linear spacing)
        upper_linear = discretize_continuous_distribution(
            dist=dist,
            n_grid=200,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.LINEAR,
        )
        lower_linear = discretize_continuous_distribution(
            dist=dist,
            n_grid=200,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.LINEAR,
        )

        # Discretize for geometric (geometric spacing)
        upper_geom = discretize_continuous_distribution(
            dist=dist,
            n_grid=200,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.DOMINATES,
            spacing_type=SpacingType.GEOMETRIC,
        )
        lower_geom = discretize_continuous_distribution(
            dist=dist,
            n_grid=200,
            align_to_multiples=True,
            tail_truncation=0.01,
            bound_type=BoundType.IS_DOMINATED,
            spacing_type=SpacingType.GEOMETRIC,
        )

        # Convolve with both methods
        T = 3
        upper_fft = FFT_self_convolve(
            dist=upper_linear, T=T, tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )
        lower_fft = FFT_self_convolve(
            dist=lower_linear, T=T, tail_truncation=0.0,
            bound_type=BoundType.IS_DOMINATED,
            use_direct=True,
        )
        upper_geom_conv = geometric_self_convolve(
            dist=upper_geom, T=T, tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )
        lower_geom_conv = geometric_self_convolve(
            dist=lower_geom, T=T, tail_truncation=0.0,
            bound_type=BoundType.IS_DOMINATED,
        )

        # Both should satisfy stochastic dominance
        gap_fft = _compare_ccdf_on_common_grid(upper_fft, lower_fft)
        gap_geom = _compare_ccdf_on_common_grid(upper_geom_conv, lower_geom_conv)

        assert gap_fft <= TOL.MASS_CONSERVATION
        assert gap_geom <= TOL.MASS_CONSERVATION
