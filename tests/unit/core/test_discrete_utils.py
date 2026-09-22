"""
Unit tests for ``PLD_accounting.distribution_discretization``.

Tests grid generation, discretization, and PMF operations.
"""

import math

import numpy as np
import pytest
from scipy import stats

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_discretization import (
    _compute_discrete_prob as compute_discrete_PMF,
)
from PLD_accounting.distribution_discretization import (
    aligned_grid_params,
    discretize_continuous_ctd,
    discretize_continuous_stoch_dom,
    project_dist_onto_grid_ctd,
    project_dist_onto_grid_stoch_dom,
    rediscretize_dist_ctd,
    rediscretize_dist_stoch_dom,
)
from PLD_accounting.distribution_discretization import (
    rediscretize_prob as pmf_remap_to_grid_kernel,
)
from PLD_accounting.distribution_utils import (
    enforce_mass_conservation,
    signed_unit_residual,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import _ccdf_from_pmf, exp_linear_to_geometric
from tests.test_tolerances import TestTolerances as TOL


class TestDiscretizeRange:
    """Test aligned grid generation."""

    def test_linear_spacing(self):
        """Test linear spacing generation."""
        n_grid = 100
        x = aligned_grid_params(
            x_min=0.0,
            x_max=10.0,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            discretization=(10.0 - 0.0) / (n_grid - 1),
        ).materialize()
        assert len(x) >= n_grid
        # Range should cover requested bounds (may extend due to alignment)
        assert x[0] <= 0.0
        assert x[-1] >= 10.0
        # Check uniform spacing
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])

    def test_geometric_spacing(self):
        """Test geometric spacing generation."""
        n_grid = 100
        x = aligned_grid_params(
            x_min=1.0,
            x_max=100.0,
            spacing_type=SpacingType.GEOMETRIC,
            align_to_multiples=True,
            discretization=np.log(100.0 / 1.0) / (n_grid - 1),
        ).materialize()
        assert len(x) >= n_grid
        # Range should cover requested bounds (may extend due to alignment)
        assert x[0] <= 1.0
        assert x[-1] >= 100.0
        # Check uniform ratio
        ratios = x[1:] / x[:-1]
        assert np.allclose(ratios, ratios[0])

    def test_nonpositive_discretization_rejected(self):
        """Discretization must be positive."""
        with pytest.raises(ValueError, match="discretization must be positive"):
            aligned_grid_params(
                x_min=0.0,
                x_max=10.0,
                spacing_type=SpacingType.LINEAR,
                align_to_multiples=True,
                discretization=0.0,
            )

    def test_two_points_linear(self):
        """Test linear grid."""
        n_grid = 100
        x = aligned_grid_params(
            x_min=1.0,
            x_max=3.0,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            discretization=(3.0 - 1.0) / (n_grid - 1),
        ).materialize()
        assert len(x) >= n_grid
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

        x = aligned_grid_params(
            x_min=x_min,
            x_max=x_max,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            discretization=discretization,
        ).materialize()

        assert x[0] <= x_min
        assert x[-1] >= x_max
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])

    def test_linear_aligned_spacing_matches_requested_step(self):
        """Aligned linear grids use the requested discretization as bin width."""
        discretization = 0.25
        grid = aligned_grid_params(
            x_min=-1.12,
            x_max=2.18,
            spacing_type=SpacingType.LINEAR,
            align_to_multiples=True,
            discretization=discretization,
        )
        x = grid.materialize()

        assert grid.step == discretization
        assert np.allclose(x / discretization, np.round(x / discretization))

    def test_continuous_discretization_uses_requested_linear_step(self):
        """Continuous discretization should preserve the requested linear step."""
        result = discretize_continuous_stoch_dom(
            dist=stats.norm(loc=0.0, scale=1.0),
            tail_truncation=1e-3,
            bound_type=BoundType.DOMINATES,
            step=0.1,
            align_to_multiples=True,
        )

        assert result.grid.step == 0.1

    def test_exponentiating_continuous_discretization_uses_requested_geometric_ratio(self):
        """Exponentiating a linear discretization produces the requested ratio."""
        linear_dist = discretize_continuous_stoch_dom(
            dist=stats.norm(loc=0.0, scale=0.5),
            tail_truncation=1e-3,
            bound_type=BoundType.DOMINATES,
            step=np.log(1.05),
            align_to_multiples=True,
        )
        result = exp_linear_to_geometric(linear_dist)

        assert np.isclose(math.exp(result.grid.step), 1.05)

    def test_continuous_ctd_requires_pld_dual(self):
        """CtD needs both laws in the closed-form privacy-profile identity."""
        with pytest.raises(TypeError, match="dual_dist"):
            discretize_continuous_ctd(  # pylint: disable=missing-kwoa
                dist=stats.logistic(),
                tail_truncation=1e-5,
                step=0.2,
                align_to_multiples=True,
            )

    def test_continuous_ctd_accepts_general_pld_and_dual(self):
        """CtD is not restricted to named Gaussian and Laplace families."""
        log_two = math.log(2.0)
        pld = stats.expon(loc=-log_two, scale=1.0)
        dual_pld = stats.weibull_max(c=1.0, loc=log_two, scale=0.5)

        result = discretize_continuous_ctd(
            dist=pld,
            dual_dist=dual_pld,
            tail_truncation=1e-5,
            step=0.2,
            align_to_multiples=True,
        )

        assert isinstance(result, PLDRealization)
        assert result.p_min == 0.0


class TestComputeDiscretePMF:
    """Test compute_discrete_PMF function."""

    def test_uniform_distribution(self):
        """Test discretization of uniform distribution."""
        dist = stats.uniform(loc=0.0, scale=1.0)
        x_array = np.linspace(0.0, 1.0, 11)
        bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist, x_array=x_array, bound_type=BoundType.DOMINATES, pmf_min_increment=0.0
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
            dist=dist, x_array=x_array, bound_type=BoundType.DOMINATES, pmf_min_increment=0.0
        )

        # Check that probabilities sum with tails to near 1 (strict tolerance)
        total = math.fsum([*map(float, bin_prob), p_left, p_right])
        assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)

    def test_exponential_distribution(self):
        """Test discretization of exponential distribution."""
        dist = stats.expon(scale=1.0)
        x_array = np.linspace(0.0, 5.0, 51)
        _bin_prob, p_left, p_right = compute_discrete_PMF(
            dist=dist, x_array=x_array, bound_type=BoundType.DOMINATES, pmf_min_increment=0.0
        )

        # Exponential should have near-zero left tail
        assert p_left < 0.01
        # Right tail should be significant
        assert p_right > 0.0


class TestPMFRemapToGrid:
    """Test rediscretize_prob function."""

    def test_exact_alignment(self):
        """Test remapping when grids are aligned."""
        x_in = np.array([1.0, 2.0, 3.0])
        pmf_in = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        x_out = x_in.copy()

        pmf_out = pmf_remap_to_grid_kernel(
            x_array=x_in, prob_arr=pmf_in, x_array_out=x_out, dominates=True
        )
        assert np.allclose(pmf_out, pmf_in)

    def test_dominates_rounding(self):
        """Test dominates (pessimistic) rounding."""
        x_in = np.array([1.0, 2.5, 4.0])
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0, 4.0])

        pmf_out = pmf_remap_to_grid_kernel(
            x_array=x_in, prob_arr=pmf_in, x_array_out=x_out, dominates=True
        )
        # 2.5 should round up to 3.0
        assert pmf_out[2] >= 0.4

    def test_is_dominated_rounding(self):
        """Test is_dominated (optimistic) rounding."""
        x_in = np.array([1.0, 2.5, 4.0])
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0, 4.0])

        pmf_out = pmf_remap_to_grid_kernel(
            x_array=x_in, prob_arr=pmf_in, x_array_out=x_out, dominates=False
        )
        # 2.5 should round down to 2.0
        assert pmf_out[1] >= 0.4

    def test_overflow_to_infinity(self):
        """Test overflow handling."""
        x_in = np.array([1.0, 2.0, 5.0])
        pmf_in = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0])  # 5.0 is beyond output grid

        pmf_out = pmf_remap_to_grid_kernel(
            x_array=x_in, prob_arr=pmf_in, x_array_out=x_out, dominates=True
        )
        _, _, ppos = enforce_mass_conservation(
            prob_arr=pmf_out,
            expected_p_min=0.0,
            expected_p_max=0.3,
            bound_type=BoundType.DOMINATES,
        )
        assert ppos >= 0.3

    def test_mass_conservation_in_remap(self):
        """Test that remapping conserves total mass."""
        x_in = np.array([0.5, 1.5, 2.5, 3.5])
        pmf_in = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float64)
        x_out = np.array([1.0, 2.0, 3.0])

        pmf_out = pmf_remap_to_grid_kernel(
            x_array=x_in, prob_arr=pmf_in, x_array_out=x_out, dominates=True
        )
        total_in = math.fsum(map(float, pmf_in))
        pmf_out, pneg, ppos = enforce_mass_conservation(
            prob_arr=pmf_out,
            expected_p_min=0.0,
            expected_p_max=0.2,
            bound_type=BoundType.DOMINATES,
        )
        total_out = math.fsum([*map(float, pmf_out), pneg, ppos])
        assert np.isclose(total_in, total_out, atol=TOL.MASS_CONSERVATION)


class TestStochasticProjectionBoundaries:
    """The source distribution is the sole authority for boundary mass."""

    def test_dominating_projection_preserves_p_max_and_adds_right_overflow(self):
        """A dominating projection retains source and overflow upper-boundary mass."""
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=1.0,
                n=3,
                anchor=0.0,
            ),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_max=0.1,
        )

        result = project_dist_onto_grid_stoch_dom(
            dist=dist,
            grid=GridSpec(anchor=0.0, step=1.0, n=2),
            bound_type=BoundType.DOMINATES,
        )

        np.testing.assert_array_equal(result.prob_arr, np.array([0.2, 0.3]))
        assert result.p_min == 0.0
        assert result.p_max == pytest.approx(0.5)

    def test_dominated_projection_preserves_p_min_and_adds_left_underflow(self):
        """A dominated projection retains source and underflow lower-boundary mass."""
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=1.0,
                n=3,
                anchor=0.0,
            ),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_min=0.1,
        )

        result = project_dist_onto_grid_stoch_dom(
            dist=dist,
            grid=GridSpec(anchor=1.0, step=1.0, n=2),
            bound_type=BoundType.IS_DOMINATED,
        )

        # expected_p_min is 0.1 + 0.2 = 0.30000000000000004, so the projection carries a
        # 5.55e-17 excess. The directional trim actually removes it from the giveable
        # edge; the previous proportional rescale computed a factor of exactly 1.0 and
        # silently left the excess in place.
        np.testing.assert_allclose(result.prob_arr, np.array([0.3, 0.4]), rtol=1e-15)
        assert result.p_min == pytest.approx(0.3)
        assert result.p_max == 0.0
        assert (
            abs(
                signed_unit_residual(
                    values=result.prob_arr, lower_term=result.p_min, upper_term=result.p_max
                )
            )
            <= 2.0**-53
        )

    @pytest.mark.parametrize(
        ("bound_type", "source_boundary", "grid_x_0", "result_boundary"),
        [
            (BoundType.DOMINATES, "p_max", -1.0, "p_max"),
            (BoundType.IS_DOMINATED, "p_min", 10.0, "p_min"),
        ],
    )
    def test_all_finite_mass_overflow_tolerates_one_ulp_source_excess(
        self,
        bound_type: BoundType,
        source_boundary: str,
        grid_x_0: float,
        result_boundary: str,
    ) -> None:
        """Explicit overflow may reach one plus source-level rounding noise."""
        boundaries = {"p_min": 0.0, "p_max": 0.0}
        boundaries[source_boundary] = 0.1
        dist = DenseDiscreteDist(
            grid=GridSpec(step=1.0, n=2, anchor=1.0),
            prob_arr=np.array([0.3, 0.6000000000000001], dtype=np.float64),
            **boundaries,
        )

        result = project_dist_onto_grid_stoch_dom(
            dist=dist,
            grid=GridSpec(anchor=grid_x_0, step=1.0, n=1),
            bound_type=bound_type,
        )

        np.testing.assert_array_equal(result.prob_arr, np.array([0.0]))
        assert getattr(result, result_boundary) == 1.0


def test_ccdf_from_pmf_padded():
    """Ccdf from pmf padded."""
    dist = SparseDiscreteDist(
        x_array=np.array([0.0, 1.0]), prob_arr=np.array([0.25, 0.5]), p_min=0.0, p_max=0.25
    )
    ccdf = _ccdf_from_pmf(dist)
    assert ccdf.shape == (4,)
    assert np.allclose(ccdf, np.array([1.0, 0.75, 0.25, 0.0]))


class TestRediscretizeBoundarySemantics:
    """Test explicit boundary contracts during rediscretization."""

    def test_ctd_rejects_single_point_range(self):
        """CtD rediscretization requires distinct truncated support bounds."""
        dist = PLDRealization(
            grid=GridSpec(
                step=1.0,
                n=1,
                anchor=0.0,
            ),
            prob_arr=np.array([1.0], dtype=np.float64),
        )

        with pytest.raises(ValueError, match="at least two distinct finite support points"):
            rediscretize_dist_ctd(
                dist=dist,
                tail_truncation=0.0,
                loss_discretization=0.1,
            )

    def test_rediscretize_near_point_mass_distribution(self):
        # prob_arr has two nonzero bins so a valid grid range exists after truncation.
        # Both bins have mass >> tail_truncation so neither is consumed.
        """Rediscretize near point mass distribution."""
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=0.5,
                n=2,
                anchor=0.5,
            ),
            prob_arr=np.array([1.0 - 1e-6, 1e-6], dtype=np.float64),
        )

        result = rediscretize_dist_stoch_dom(
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
        """A lower bound deliberately relaxes +inf to the last finite cell."""
        dist = DenseDiscreteDist(
            grid=GridSpec(step=1.0, n=3, anchor=0.0),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_max=0.1,
        )

        result = rediscretize_dist_stoch_dom(
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

    def test_is_dominated_boundary_fold_downgrades_pld_realization(self):
        """Relaxing positive-infinity mass downgrades an exact realization."""
        dist = PLDRealization(
            grid=GridSpec(
                step=0.1,
                n=2,
                anchor=math.log(0.9),
            ),
            prob_arr=np.array([0.8, 0.1], dtype=np.float64),
            p_max=0.1,
        )

        result = rediscretize_dist_stoch_dom(
            dist=dist,
            tail_truncation=0.0,
            loss_discretization=0.1,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert isinstance(result, DenseDiscreteDist)
        assert not isinstance(result, PLDRealization)
        assert result.p_max == 0.0
        assert math.fsum([*map(float, result.prob_arr), result.p_min]) == pytest.approx(1.0)

    def test_is_dominated_relaxes_p_max_before_left_tail_truncation(self):
        """Lower canonicalization prevents simultaneous real-domain boundaries."""
        dist = DenseDiscreteDist(
            grid=GridSpec(step=1.0, n=3, anchor=0.0),
            prob_arr=np.array([0.05, 0.4, 0.45], dtype=np.float64),
            p_max=0.1,
        )

        result = rediscretize_dist_stoch_dom(
            dist=dist,
            tail_truncation=0.2,
            loss_discretization=1.0,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert result.p_min == pytest.approx(0.05)
        assert result.p_max == 0.0
        assert math.fsum([result.p_min, *map(float, result.prob_arr)]) == pytest.approx(1.0)

    def test_dominates_moves_real_p_min_into_first_finite_cell(self):
        """An upper real bound absorbs -inf mass into its first finite cell."""
        dist = DenseDiscreteDist(
            grid=GridSpec(step=1.0, n=3, anchor=0.0),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_min=0.1,
        )

        result = rediscretize_dist_stoch_dom(
            dist=dist,
            tail_truncation=0.0,
            loss_discretization=1.0,
            spacing_type=SpacingType.LINEAR,
            bound_type=BoundType.DOMINATES,
        )

        assert result.p_min == 0.0
        assert result.prob_arr[0] == pytest.approx(0.3)
        assert math.fsum([*map(float, result.prob_arr), result.p_max]) == pytest.approx(1.0)

    def test_dominates_geometric_keeps_zero_atom(self):
        """Dominates geometric keeps zero atom."""
        dist = DenseDiscreteDist(
            grid=GridSpec(step=math.log(2.0), n=3, spacing_type=SpacingType.GEOMETRIC, anchor=1.0),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_min=0.1,
            domain=Domain.POSITIVES,
        )

        result = rediscretize_dist_stoch_dom(
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

    def test_ctd_real_linear_rediscretization_preserves_mass(self):
        """CtD rejects a real-domain distribution that is not a PLD realization."""
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=0.5,
                n=4,
                anchor=-1.0,
            ),
            prob_arr=np.array([0.15, 0.2, 0.25, 0.4], dtype=np.float64),
        )

        with pytest.raises(ValueError, match="reciprocal-moment"):
            rediscretize_dist_ctd(
                dist=dist,
                tail_truncation=0.0,
                loss_discretization=1.0,
            )

    def test_ctd_rejects_p_min_before_tail_truncation(self):
        """A truncation budget cannot hide an invalid negative-infinity atom."""
        p_min = 1e-6
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=1.0,
                n=2,
                anchor=0.0,
            ),
            prob_arr=np.array([0.5, 0.5 - p_min], dtype=np.float64),
            p_min=p_min,
        )

        with pytest.raises(ValueError, match="requires p_min = 0 exactly"):
            rediscretize_dist_ctd(
                dist=dist,
                tail_truncation=1e-3,
                loss_discretization=0.5,
            )

    def test_ctd_positive_geometric_rediscretization_preserves_zero_atom(self):
        """CtD rejects positive/geometric grids rather than silently rounding."""
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=math.log(2.0),
                spacing_type=SpacingType.GEOMETRIC,
                n=3,
                anchor=1.0,
            ),
            prob_arr=np.array([0.2, 0.3, 0.4], dtype=np.float64),
            p_min=0.1,
            domain=Domain.POSITIVES,
        )

        with pytest.raises(ValueError, match="fixed-gap linear grid"):
            project_dist_onto_grid_ctd(
                dist=dist,
                grid=GridSpec(
                    anchor=1.0,
                    step=2.0,
                    n=3,
                    spacing_type=SpacingType.GEOMETRIC,
                ),
            )

    def test_ctd_rejects_positive_domain_linear_rediscretization(self):
        """CtD must not reinterpret positive values as real privacy losses."""
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=1.0,
                n=2,
                anchor=1.0,
            ),
            prob_arr=np.array([0.4, 0.6], dtype=np.float64),
            domain=Domain.POSITIVES,
        )

        with pytest.raises(ValueError, match="real-domain"):
            rediscretize_dist_ctd(
                dist=dist,
                tail_truncation=0.0,
                loss_discretization=0.5,
            )
