"""Tests for the subsample_pld() wrapper function."""

import warnings

import numpy as np
import pytest
from dp_accounting.pld import privacy_loss_distribution as dp_pld
from dp_accounting.pld.pld_pmf import DensePLDPmf

from PLD_accounting.discrete_dist import GridSpec, PLDRealization
from PLD_accounting.dp_accounting_support import (
    DP_ACCOUNTING_MASS_REPAIR_TOL,
    dp_accounting_pmf_to_pld_realization,
    linear_dist_to_dp_accounting_pmf,
)
from PLD_accounting.subsample_pld import subsample_pld
from PLD_accounting.types import BoundType


def _make_pld_remove_only() -> dp_pld.PrivacyLossDistribution:
    """Build a PLD with only the REMOVE direction."""
    dist = PLDRealization(
        grid=GridSpec(
            step=0.5,
            n=4,
            anchor=0.0,
        ),
        prob_arr=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
    )
    pmf = linear_dist_to_dp_accounting_pmf(dist=dist, bound_type=BoundType.DOMINATES)
    return dp_pld.PrivacyLossDistribution(pmf_remove=pmf)


def _make_pld_both_directions() -> dp_pld.PrivacyLossDistribution:
    """Build a PLD with both REMOVE and ADD directions."""
    remove_dist = PLDRealization(
        grid=GridSpec(
            step=0.5,
            n=4,
            anchor=0.0,
        ),
        prob_arr=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
    )
    add_dist = PLDRealization(
        grid=GridSpec(
            step=0.25,
            n=5,
            anchor=0.0,
        ),
        prob_arr=np.array([0.24, 0.2, 0.18, 0.16, 0.14], dtype=np.float64),
        p_max=0.08,
    )
    remove_pmf = linear_dist_to_dp_accounting_pmf(dist=remove_dist, bound_type=BoundType.DOMINATES)
    add_pmf = linear_dist_to_dp_accounting_pmf(dist=add_dist, bound_type=BoundType.DOMINATES)
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


def test_subsample_pld_rejects_optimistic_remove_pmf() -> None:
    """Optimistic REMOVE PMFs are rejected for both the shortcut and the full path."""
    dist = PLDRealization(
        grid=GridSpec(step=0.5, n=4, anchor=0.0),
        prob_arr=np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
    )
    pmf = linear_dist_to_dp_accounting_pmf(dist=dist, bound_type=BoundType.IS_DOMINATED)
    pld = dp_pld.PrivacyLossDistribution(pmf_remove=pmf)
    with pytest.raises(ValueError, match="optimistic dp_accounting PMF does not map"):
        subsample_pld(pld=pld, sampling_probability=1.0)
    with pytest.raises(ValueError, match="optimistic dp_accounting PMF does not map"):
        subsample_pld(pld=pld, sampling_probability=0.3)


def test_subsample_pld_rejects_optimistic_add_pmf_on_q1_shortcut() -> None:
    """The q==1 identity return still validates an optional ADD PMF."""
    pld = _make_pld_both_directions()
    add_dist = PLDRealization(
        grid=GridSpec(step=0.25, n=5, anchor=0.0),
        prob_arr=np.array([0.24, 0.2, 0.18, 0.16, 0.14], dtype=np.float64),
        p_max=0.08,
    )
    optimistic_add = linear_dist_to_dp_accounting_pmf(
        dist=add_dist, bound_type=BoundType.IS_DOMINATED
    )
    mixed = dp_pld.PrivacyLossDistribution(pmf_remove=pld._pmf_remove, pmf_add=optimistic_add)
    with pytest.raises(ValueError, match="optimistic dp_accounting PMF does not map"):
        subsample_pld(pld=mixed, sampling_probability=1.0)
    with pytest.raises(ValueError, match="optimistic dp_accounting PMF does not map"):
        subsample_pld(pld=mixed, sampling_probability=0.3)


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


def test_subsample_pld_composed_source_warns_at_library_defaults() -> None:
    """subsample_pld imports at the adapter defaults, so a composed source says so."""
    pld = dp_pld.from_gaussian_mechanism(
        standard_deviation=1.0,
        value_discretization_interval=1e-4,
        pessimistic_estimate=True,
        sampling_prob=0.01,
        use_connect_dots=True,
    ).self_compose(100)
    with pytest.warns(RuntimeWarning, match="dp_accounting import mass"):
        subsampled = subsample_pld(
            pld=pld,
            sampling_probability=0.1,
        )
    eps = subsampled.get_epsilon_for_delta(1e-5)
    assert np.isfinite(eps)
    assert eps > 0


def test_subsample_pld_documented_signature_emits_no_warning() -> None:
    """The documented two-argument call must not warn about a retracted budget."""
    pld = _make_pld_remove_only()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        subsample_pld(pld=pld, sampling_probability=0.3)


def _pmf_with_mass_residual(residual: float) -> DensePLDPmf:
    """Dense PMF whose finite mass falls short of one by exactly ``residual``."""
    probs = np.array([0.25, 0.25, 0.25, 0.25 - residual], dtype=np.float64)
    return DensePLDPmf(
        discretization=0.5,
        lower_loss=0,
        probs=probs,
        infinity_mass=0.0,
        pessimistic_estimate=True,
    )


def test_import_at_noise_scale_is_normalized_silently() -> None:
    """Only a residual at arithmetic-noise scale is repaired without a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        realization = dp_accounting_pmf_to_pld_realization(pmf=_pmf_with_mass_residual(2.0**-53))
    assert realization.prob_arr.size == 4


def test_import_above_noise_scale_warns_even_though_it_is_ordinary() -> None:
    """A residual typical of dp_accounting is repaired, but never silently.

    dp_accounting drift runs many orders above float noise, so every real import repair
    is directional and is reported. The ceiling decides admission, not silence.
    """
    with pytest.warns(RuntimeWarning, match="dp_accounting import mass"):
        dp_accounting_pmf_to_pld_realization(pmf=_pmf_with_mass_residual(1.0e-8))


def test_import_below_the_ceiling_repairs_with_a_warning() -> None:
    """A residual under the ceiling is repaired directionally, and says so."""
    with pytest.warns(RuntimeWarning, match="dp_accounting import mass"):
        dp_accounting_pmf_to_pld_realization(pmf=_pmf_with_mass_residual(5.0e-6))


def test_import_at_repair_ceiling_rejects() -> None:
    """A residual no uncomposed Gaussian produces is rejected rather than repaired."""
    with pytest.raises(ValueError, match="dp_accounting import mass"):
        dp_accounting_pmf_to_pld_realization(pmf=_pmf_with_mass_residual(1.0e-5))


def test_composed_pmf_imports_when_the_caller_raises_the_ceiling() -> None:
    """Raising the ceiling admits a composed source; it does not silence the repair."""
    depth = 1_000
    with pytest.warns(RuntimeWarning, match="dp_accounting import mass"):
        realization = dp_accounting_pmf_to_pld_realization(
            pmf=_pmf_with_mass_residual(1.0e-5),
            mass_repair_tol=DP_ACCOUNTING_MASS_REPAIR_TOL * depth,
        )
    assert realization.prob_arr.size == 4
