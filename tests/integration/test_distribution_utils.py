"""
Integration tests for distribution_utils.
"""
import numpy as np
import pytest

from PLD_accounting.distribution_utils import (
    SPACING_RTOL,
    SPACING_ATOL,
    compute_bin_ratio,
    compute_bin_width,
    enforce_mass_conservation,
    PMF_MASS_TOL,
)
from PLD_accounting.types import BoundType
import math


class TestCoreUtils:
    """Integration tests for distribution_utils behavior."""

    def test_compute_bin_ratio_geometric(self):
        ratio = 1.5
        x = 2.0 * ratio ** np.arange(6, dtype=np.float64)
        assert np.isclose(compute_bin_ratio(x), ratio, rtol=SPACING_RTOL, atol=SPACING_ATOL)

    def test_compute_bin_ratio_noisy_geometric(self):
        ratio = 1.7
        deltas = np.array([0.0, SPACING_RTOL / 10, -SPACING_RTOL / 10, SPACING_RTOL / 20, 0.0])
        ratios = ratio * (1.0 + deltas)
        x = 3.0 * np.cumprod(np.concatenate(([1.0], ratios))).astype(np.float64)
        assert np.isclose(compute_bin_ratio(x), ratio, rtol=SPACING_RTOL, atol=SPACING_ATOL)

    def test_compute_bin_ratio_non_positive(self):
        x = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        with pytest.raises(ValueError, match="bin ratio"):
            compute_bin_ratio(x)

    def test_compute_bin_width_linear(self):
        x = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        assert np.isclose(compute_bin_width(x), 0.5, rtol=SPACING_RTOL, atol=SPACING_ATOL)

    def test_compute_bin_width_noisy_linear(self):
        width = 0.25
        deltas = np.array([0.0, SPACING_RTOL / 10, -SPACING_RTOL / 10, 0.0])
        diffs = width * (1.0 + deltas)
        x = np.cumsum(np.concatenate(([0.0], diffs))).astype(np.float64)
        assert np.isclose(compute_bin_width(x), width, rtol=SPACING_RTOL, atol=SPACING_ATOL)

    def test_enforce_mass_conservation_dominates(self):
        pmf = np.array([0.2, 0.3], dtype=np.float64)
        pmf_out, p_neg_inf, p_pos_inf = enforce_mass_conservation(
            PMF_array=pmf, expected_neg_inf=0.0, expected_pos_inf=0.0, bound_type=BoundType.DOMINATES
        )
        assert p_neg_inf == 0.0
        assert np.isclose(p_pos_inf, 0.5)
        assert np.isclose(np.sum(pmf_out) + p_pos_inf, 1.0)

    def test_enforce_mass_conservation_is_dominated(self):
        pmf = np.array([0.2, 0.3], dtype=np.float64)
        pmf_out, p_neg_inf, p_pos_inf = enforce_mass_conservation(
            PMF_array=pmf, expected_neg_inf=0.0, expected_pos_inf=0.0, bound_type=BoundType.IS_DOMINATED
        )
        assert p_pos_inf == 0.0
        assert np.isclose(p_neg_inf, 0.5)
        assert np.isclose(np.sum(pmf_out) + p_neg_inf, 1.0)

    def test_enforce_mass_conservation_overfull(self):
        pmf = np.array([0.6, 0.6], dtype=np.float64)
        pmf_sum = math.fsum(map(float, pmf))
        assert pmf_sum > 1.0 + PMF_MASS_TOL, "Test setup error: PMF should sum to > 1.0"

        pmf_out, p_neg_inf, p_pos_inf = enforce_mass_conservation(
            PMF_array=pmf.copy(), expected_neg_inf=0.0, expected_pos_inf=0.0, bound_type=BoundType.DOMINATES
        )
        total = math.fsum(map(float, pmf_out)) + p_neg_inf + p_pos_inf
        assert total <= 1.0 + PMF_MASS_TOL
        assert np.isclose(total, 1.0, atol=PMF_MASS_TOL)

    def test_enforce_mass_conservation_overfull_trims_left_for_dominates(self):
        excess = 1e-12
        pmf = np.array([0.2, 0.3, 0.5 + excess], dtype=np.float64)
        pmf_out, p_neg_inf, p_pos_inf = enforce_mass_conservation(
            PMF_array=pmf.copy(), expected_neg_inf=0.0, expected_pos_inf=0.0, bound_type=BoundType.DOMINATES
        )
        assert p_neg_inf == 0.0
        assert p_pos_inf >= 0.0
        assert np.isclose(pmf_out[0], 0.2 - excess, atol=1e-15)
        assert np.isclose(pmf_out[1], 0.3, atol=1e-15)
        assert np.isclose(pmf_out[2], 0.5 + excess, atol=1e-15)
        total = math.fsum(map(float, pmf_out)) + p_pos_inf
        assert np.isclose(total, 1.0, atol=PMF_MASS_TOL)

    def test_enforce_mass_conservation_overfull_trims_right_for_is_dominated(self):
        excess = 1e-12
        pmf = np.array([0.2 + excess, 0.3, 0.5], dtype=np.float64)
        pmf_out, p_neg_inf, p_pos_inf = enforce_mass_conservation(
            PMF_array=pmf.copy(), expected_neg_inf=0.0, expected_pos_inf=0.0, bound_type=BoundType.IS_DOMINATED
        )
        assert p_pos_inf == 0.0
        assert p_neg_inf >= 0.0
        assert np.isclose(pmf_out[0], 0.2 + excess, atol=1e-15)
        assert np.isclose(pmf_out[1], 0.3, atol=1e-15)
        assert np.isclose(pmf_out[2], 0.5 - excess, atol=1e-15)
        total = math.fsum(map(float, pmf_out)) + p_neg_inf
        assert np.isclose(total, 1.0, atol=PMF_MASS_TOL)
