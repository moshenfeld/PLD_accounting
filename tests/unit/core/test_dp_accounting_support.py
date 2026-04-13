"""Unit tests for dp_accounting_support."""

import math

import numpy as np
import pytest
from dp_accounting.pld import privacy_loss_distribution as dp_pld
from dp_accounting.pld.pld_pmf import DensePLDPmf
from PLD_accounting.discrete_dist import (
    REALIZATION_MOMENT_TOL,
    DenseDiscreteDist,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_discretization import rediscritize_dist
from PLD_accounting.distribution_utils import MAX_SAFE_EXP_ARG, exp_moment_terms
from PLD_accounting.dp_accounting_support import (
    dp_accounting_pmf_to_pld_realization,
    linear_dist_to_dp_accounting_pmf,
)
from PLD_accounting.mechanisms import gaussian_distribution, laplace_distribution
from PLD_accounting.random_allocation_realization import (
    realization_remove_base_distributions,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import calc_pld_dual, negate_reverse_linear_distribution

from tests.test_tolerances import TestTolerances as TOL


def _make_realization() -> PLDRealization:
    return PLDRealization(
        x_min=-0.5,
        step=0.5,
        prob_arr=np.array([0.2, 0.3, 0.25, 0.15], dtype=np.float64),
        p_max=0.1,
    )


def test_dp_accounting_roundtrip_preserves_mass_and_grid_shape():
    original = _make_realization()
    pmf = linear_dist_to_dp_accounting_pmf(dist=original, pessimistic_estimate=True)
    restored = dp_accounting_pmf_to_pld_realization(pmf)

    assert restored.x_array.shape == original.x_array.shape
    total_mass = math.fsum([*map(float, restored.prob_arr), restored.p_min, restored.p_max])
    assert np.isclose(total_mass, 1.0, atol=TOL.SPACING_ATOL)
    assert np.isclose(restored.p_max, original.p_max, atol=TOL.SPACING_ATOL)


def test_linear_dist_to_dp_accounting_handles_zero_finite_mass():
    """Test that distributions with all mass at infinity are handled correctly."""
    realization = DenseDiscreteDist(
        x_min=0.0,
        step=1.0,
        prob_arr=np.array([0.0, 0.0], dtype=np.float64),
        p_max=1.0,
    )
    pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
    assert pmf._infinity_mass == 1.0
    assert np.allclose(pmf._probs, np.array([0.0, 0.0]))


def test_dp_accounting_composed_gaussian_add_pmf_converts_with_repair():
    """Composed Gaussian add-direction PMF converts successfully with the always-on repair.

    dp_accounting's Gaussian discretization produces a systematic exp-moment
    violation that grows with composition (~33 after 20 rounds of subsampled
    Gaussian σ=0.5, q=1/20).  The repair is applied automatically and must
    produce a valid realization with E[exp(-L)] <= 1.
    """
    pld = dp_pld.from_gaussian_mechanism(
        standard_deviation=0.5,
        value_discretization_interval=1e-4,
        pessimistic_estimate=True,
        sampling_prob=1.0 / 20.0,
        use_connect_dots=True,
    ).self_compose(20)

    realization = dp_accounting_pmf_to_pld_realization(pld._pmf_add)
    total_mass = math.fsum(
        [*map(float, realization.prob_arr), realization.p_min, realization.p_max]
    )
    exp_moment_val = math.fsum(
        map(float, exp_moment_terms(prob_arr=realization.prob_arr, x_vals=realization.x_array))
    )

    assert np.isclose(total_mass, 1.0, atol=TOL.SPACING_ATOL)
    assert exp_moment_val <= 1.0 + REALIZATION_MOMENT_TOL


def test_exp_moment_handles_very_negative_losses_with_tiny_mass():
    """Tiny mass at very negative loss should be handled without clipping artifacts."""
    tiny_prob = np.exp(-(MAX_SAFE_EXP_ARG + 21.0))
    realization = PLDRealization(
        x_min=-(MAX_SAFE_EXP_ARG + 20.0),
        step=MAX_SAFE_EXP_ARG + 21.0,
        prob_arr=np.array([tiny_prob, 1.0 - tiny_prob], dtype=np.float64),
    )

    expected = np.exp(np.log(realization.prob_arr[0]) - realization.x_array[0])
    expected += realization.prob_arr[1] * np.exp(-realization.x_array[1])
    assert np.isclose(
        math.fsum(
            map(float, exp_moment_terms(prob_arr=realization.prob_arr, x_vals=realization.x_array))
        ),
        expected,
        rtol=1e-12,
        atol=1e-15,
    )


def test_realization_remove_base_distributions_derives_dual_from_coarsened_base():
    """The remove path should coarsen once, then build the negated dual from that base."""
    stage1_poisson = dp_pld.from_gaussian_mechanism(
        standard_deviation=1.0,
        value_discretization_interval=1e-4,
        pessimistic_estimate=True,
        sampling_prob=1.0 / 100.0,
        use_connect_dots=True,
    ).self_compose(1000)
    remove_realization = dp_accounting_pmf_to_pld_realization(stage1_poisson._pmf_remove)

    base_dist, neg_dual_dist = realization_remove_base_distributions(
        realization=remove_realization,
        loss_discretization=0.01 / int(2 * np.ceil(np.log2(100)) + 1),
        tail_truncation=(1e-8 * 0.01) / 3 / 100,
        bound_type=BoundType.DOMINATES,
    )
    expected_neg_dual = negate_reverse_linear_distribution(calc_pld_dual(base_dist))

    assert base_dist.p_min == pytest.approx(0.0, abs=TOL.SPACING_ATOL)
    assert isinstance(base_dist, PLDRealization)
    np.testing.assert_allclose(neg_dual_dist.x_array, expected_neg_dual.x_array)
    np.testing.assert_allclose(neg_dual_dist.prob_arr, expected_neg_dual.prob_arr)
    assert neg_dual_dist.p_min == pytest.approx(expected_neg_dual.p_min, abs=TOL.SPACING_ATOL)
    assert neg_dual_dist.p_max == pytest.approx(0.0, abs=TOL.SPACING_ATOL)


def test_realization_remove_base_distributions_handles_is_dominated_coarsening():
    """Lower path should dualize first, then coarsen base and negated dual separately."""
    remove_realization = laplace_distribution(
        2.0 / np.sqrt(2.0),
        value_discretization=1e-4,
    )
    loss_discretization = 0.005 / int(2 * np.ceil(np.log2(10)) + 1)
    tail_truncation = 1e-12 / 3 / (2 * 1) / (2 * 1) / 10

    base_dist, neg_dual_dist = realization_remove_base_distributions(
        realization=remove_realization,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=BoundType.IS_DOMINATED,
    )
    exact_neg_dual = negate_reverse_linear_distribution(calc_pld_dual(remove_realization))
    expected_base = rediscritize_dist(
        dist=remove_realization,
        tail_truncation=tail_truncation,
        loss_discretization=loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=BoundType.IS_DOMINATED,
    )
    expected_neg_dual = rediscritize_dist(
        dist=exact_neg_dual,
        tail_truncation=tail_truncation,
        loss_discretization=loss_discretization,
        spacing_type=SpacingType.LINEAR,
        bound_type=BoundType.IS_DOMINATED,
    )

    assert isinstance(base_dist, DenseDiscreteDist)
    assert not isinstance(base_dist, PLDRealization)
    np.testing.assert_allclose(base_dist.x_array, expected_base.x_array)
    np.testing.assert_allclose(base_dist.prob_arr, expected_base.prob_arr)
    assert base_dist.p_min == pytest.approx(expected_base.p_min, abs=TOL.SPACING_ATOL)
    assert base_dist.p_max == pytest.approx(expected_base.p_max, abs=TOL.SPACING_ATOL)
    np.testing.assert_allclose(neg_dual_dist.x_array, expected_neg_dual.x_array)
    np.testing.assert_allclose(neg_dual_dist.prob_arr, expected_neg_dual.prob_arr)
    assert neg_dual_dist.p_min == pytest.approx(expected_neg_dual.p_min, abs=TOL.SPACING_ATOL)
    assert neg_dual_dist.p_max == pytest.approx(0.0, abs=TOL.SPACING_ATOL)
    total_mass = math.fsum(
        [*map(float, neg_dual_dist.prob_arr), neg_dual_dist.p_min, neg_dual_dist.p_max]
    )
    assert np.isclose(total_mass, 1.0, atol=TOL.SPACING_ATOL)


def test_realization_remove_base_distributions_allows_left_tail_truncation_of_realization():
    """Lower coarsening should treat the truncated base as a plain dense distribution."""
    remove_realization = gaussian_distribution(
        0.2,
        value_discretization=1e-4,
    )

    base_dist, neg_dual_dist = realization_remove_base_distributions(
        realization=remove_realization,
        loss_discretization=0.1,
        tail_truncation=1.6666666666666668e-9,
        bound_type=BoundType.IS_DOMINATED,
    )

    assert isinstance(base_dist, DenseDiscreteDist)
    assert not isinstance(base_dist, PLDRealization)
    assert base_dist.p_min > 0.0
    assert np.isclose(
        math.fsum([*map(float, base_dist.prob_arr), base_dist.p_min, base_dist.p_max]),
        1.0,
        atol=TOL.SPACING_ATOL,
    )
    assert isinstance(neg_dual_dist, DenseDiscreteDist)


def test_realization_remove_base_distributions_is_dominated_clamps_when_refining():
    """Lower path should clamp to realization.step when the requested step is finer.

    Rediscretizing to a finer target inflates the grid with interior zeros,
    making all subsequent O(N²) convolutions unnecessarily slow.
    The effective discretization is max(realization.step, loss_discretization).
    """
    remove_realization = gaussian_distribution(
        2.0,
        value_discretization=1e-4,
    )
    loss_discretization = remove_realization.step / 2.0  # finer than the realization
    tail_truncation = 2.777777777777778e-16

    base_dist, neg_dual_dist = realization_remove_base_distributions(
        realization=remove_realization,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=BoundType.IS_DOMINATED,
    )

    # Effective discretization is clamped to realization.step, not the finer target.
    effective_disc = max(remove_realization.step, loss_discretization)
    exact_neg_dual = negate_reverse_linear_distribution(calc_pld_dual(remove_realization))
    expected_base = rediscritize_dist(
        dist=DenseDiscreteDist(
            x_min=remove_realization.x_min,
            step=remove_realization.step,
            prob_arr=remove_realization.prob_arr.copy(),
            p_min=remove_realization.p_min,
            p_max=remove_realization.p_max,
        ),
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        spacing_type=SpacingType.LINEAR,
        bound_type=BoundType.IS_DOMINATED,
    )
    expected_neg_dual = rediscritize_dist(
        dist=exact_neg_dual,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        spacing_type=SpacingType.LINEAR,
        bound_type=BoundType.IS_DOMINATED,
    )

    np.testing.assert_allclose(base_dist.x_array, expected_base.x_array)
    np.testing.assert_allclose(base_dist.prob_arr, expected_base.prob_arr)
    assert base_dist.p_min == pytest.approx(expected_base.p_min, abs=TOL.SPACING_ATOL)
    assert base_dist.p_max == pytest.approx(0.0, abs=TOL.SPACING_ATOL)
    np.testing.assert_allclose(neg_dual_dist.x_array, expected_neg_dual.x_array)
    np.testing.assert_allclose(neg_dual_dist.prob_arr, expected_neg_dual.prob_arr)
    assert neg_dual_dist.p_min == pytest.approx(expected_neg_dual.p_min, abs=TOL.SPACING_ATOL)
    assert neg_dual_dist.p_max == pytest.approx(expected_neg_dual.p_max, abs=TOL.SPACING_ATOL)


class TestRealizationAdapter:
    """Test conversion between PLDRealization and DensePLDPmf."""

    def test_dense_linear_to_dense_pmf(self):
        """Test that PLDRealization converts to DensePLDPmf."""
        realization = PLDRealization(
            x_min=0.0,
            step=0.5,
            prob_arr=np.array([0.2, 0.3, 0.4, 0.1]),
        )

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert np.allclose(pmf._probs, realization.prob_arr)

    def test_dense_linear_with_nonzero_base(self):
        """Test PLDRealization with non-zero x_min."""
        realization = PLDRealization(
            x_min=1.0,
            step=0.5,
            prob_arr=np.array([0.3, 0.4, 0.3]),
        )

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert pmf._lower_loss == 2  # 1.0 / 0.5 = 2

    def test_dense_linear_with_infinity_mass(self):
        """Test PLDRealization with p_max."""
        realization = PLDRealization(
            x_min=0.0,
            step=0.25,
            prob_arr=np.array([0.2, 0.5, 0.2]),
            p_max=0.1,
        )

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._infinity_mass == 0.1

    def test_dense_linear_roundtrip(self):
        """Test PLDRealization -> PMF -> PLDRealization roundtrip."""
        realization = PLDRealization(
            x_min=0.5,
            step=0.25,
            prob_arr=np.array([0.2, 0.3, 0.4, 0.1]),
        )

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
        restored = dp_accounting_pmf_to_pld_realization(pmf)

        # Check values match
        assert isinstance(restored, PLDRealization)
        assert np.allclose(realization.x_array, restored.x_array)
        assert np.allclose(realization.prob_arr, restored.prob_arr)
        assert np.isclose(realization.p_max, restored.p_max)

    def test_linear_dist_to_dp_accounting_rejects_non_linear_dist(self):
        dist = SparseDiscreteDist(
            x_array=np.array([0.0, 0.5]),
            prob_arr=np.array([0.5, 0.5]),
        )
        with pytest.raises(TypeError, match="requires DenseDiscreteDist"):
            linear_dist_to_dp_accounting_pmf(dist=dist, pessimistic_estimate=True)
