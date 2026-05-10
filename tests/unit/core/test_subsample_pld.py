"""Tests for the subsample_pld() wrapper function."""

import numpy as np
import pytest
from dp_accounting.pld import privacy_loss_distribution as dp_pld
from PLD_accounting.discrete_dist import PLDRealization
from PLD_accounting.dp_accounting_support import linear_dist_to_dp_accounting_pmf
from PLD_accounting.subsample_pld import subsample_pld


def _make_pld_remove_only() -> dp_pld.PrivacyLossDistribution:
    """Build a PLD with only the REMOVE direction."""
    dist = PLDRealization(
        x_min=0.0,
        step=0.5,
        prob_arr=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
    )
    pmf = linear_dist_to_dp_accounting_pmf(dist=dist, pessimistic_estimate=True)
    return dp_pld.PrivacyLossDistribution(pmf_remove=pmf)


def _make_pld_both_directions() -> dp_pld.PrivacyLossDistribution:
    """Build a PLD with both REMOVE and ADD directions."""
    remove_dist = PLDRealization(
        x_min=0.0,
        step=0.5,
        prob_arr=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
    )
    add_dist = PLDRealization(
        x_min=0.0,
        step=0.25,
        prob_arr=np.array([0.24, 0.2, 0.18, 0.16, 0.14], dtype=np.float64),
        p_max=0.08,
    )
    remove_pmf = linear_dist_to_dp_accounting_pmf(dist=remove_dist, pessimistic_estimate=True)
    add_pmf = linear_dist_to_dp_accounting_pmf(dist=add_dist, pessimistic_estimate=True)
    return dp_pld.PrivacyLossDistribution(pmf_remove=remove_pmf, pmf_add=add_pmf)


def test_subsample_pld_rejects_zero_sampling_probability():
    """Subsample pld rejects zero sampling probability."""
    pld = _make_pld_remove_only()
    with pytest.raises(ValueError, match="sampling_probability must be in"):
        subsample_pld(pld=pld, sampling_probability=0.0)


def test_subsample_pld_rejects_sampling_probability_above_one():
    """Subsample pld rejects sampling probability above one."""
    pld = _make_pld_remove_only()
    with pytest.raises(ValueError, match="sampling_probability must be in"):
        subsample_pld(pld=pld, sampling_probability=1.1)


def test_subsample_pld_q1_returns_same_object():
    """Subsample pld q1 returns same object."""
    pld = _make_pld_remove_only()
    result = subsample_pld(pld=pld, sampling_probability=1.0)
    assert result is pld


def test_subsample_pld_remove_only_returns_pld():
    """Subsample pld remove only returns pld."""
    pld = _make_pld_remove_only()
    result = subsample_pld(pld=pld, sampling_probability=0.3)
    assert isinstance(result, dp_pld.PrivacyLossDistribution)
    assert result._pmf_remove is not None


def test_subsample_pld_both_directions_returns_pld_with_both():
    """Subsample pld both directions returns pld with both."""
    pld = _make_pld_both_directions()
    result = subsample_pld(pld=pld, sampling_probability=0.3)
    assert isinstance(result, dp_pld.PrivacyLossDistribution)
    assert result._pmf_remove is not None
    assert result._pmf_add is not None


def test_subsample_pld_result_has_valid_epsilon():
    """Subsampled PLD must produce a finite, positive epsilon."""
    pld = _make_pld_remove_only()
    result = subsample_pld(pld=pld, sampling_probability=0.3)
    eps = result.get_epsilon_for_delta(1e-5)
    assert np.isfinite(eps)
    assert eps > 0


def test_subsample_pld_smaller_q_gives_smaller_epsilon():
    """Stronger subsampling must produce a smaller or equal epsilon."""
    pld = _make_pld_remove_only()
    delta = 1e-5
    eps_high_q = subsample_pld(pld=pld, sampling_probability=0.5).get_epsilon_for_delta(delta)
    eps_low_q = subsample_pld(pld=pld, sampling_probability=0.1).get_epsilon_for_delta(delta)
    assert eps_low_q <= eps_high_q + 1e-9
