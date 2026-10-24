"""Smoke and integration tests for the published ``PLD_accounting`` API.

Each export from :mod:`PLD_accounting` is exercised with deliberately coarse
numerical settings so the suite stays fast while still checking type contracts,
mass conservation, and a few monotonicity or consistency relationships.
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    DenseDiscreteDist,
    Direction,
    GridSpec,
    PLDRealization,
    PrivacyParams,
    ProjectionMethod,
    discrete_distribution,
    gaussian_allocation_delta_configurable,
    gaussian_allocation_epsilon_configurable,
    gaussian_allocation_epsilon_range,
    gaussian_allocation_pld,
    gaussian_distribution,
    general_allocation_delta,
    general_allocation_epsilon,
    general_allocation_pld,
    laplace_distribution,
    project_discrete_dist,
    subsample_pld,
    subsample_pld_realization,
)
from PLD_accounting.projections_stoch_dom import truncate_edges_stoch_dom
from PLD_accounting.types import resolve_projection_method

# =============================================================================
# Shared constants — deliberately coarse for speed
# =============================================================================

SIGMA = 2.0
NUM_STEPS = 5
NUM_SELECTED = 1
NUM_EPOCHS = 1
DELTA = 1e-3
EPSILON = 1.0
MASS_TOL = 1e-6  # relaxed for coarse grids

COARSE_CONFIG = AllocationSchemeConfig(
    loss_discretization=0.05,
    tail_truncation=1e-6,
    convolution_method=ConvolutionMethod.GEOM,
)
CONVOLUTION_METHODS = list(ConvolutionMethod)


def _total_mass(dist):
    """Sum of prob_arr + boundary masses."""
    return float(np.sum(dist.prob_arr)) + dist.p_min + dist.p_max


def _make_gaussian_realizations(sigma: float):
    """Return (remove_pld, add_pld) PLDRealizations from a Gaussian mechanism."""
    remove = gaussian_distribution(scale=sigma, bound_type=BoundType.DOMINATES)
    add = gaussian_distribution(scale=sigma, bound_type=BoundType.DOMINATES)
    return remove, add


# ===================================================================
# 1. Mechanism distributions
# ===================================================================


class TestMechanismDistributions:
    """Mechanism helpers should return the expected discrete object kind per bound type."""

    def test_gaussian_dominates_returns_pld_realization(self):
        """Upper-bound Gaussian discretization must yield a validated :class:`PLDRealization`."""
        d = gaussian_distribution(scale=1.0, bound_type=BoundType.DOMINATES)
        assert isinstance(d, PLDRealization)

    def test_gaussian_is_dominated_returns_dense_dist(self):
        """Lower-bound Gaussian path returns a dense grid without realization metadata."""
        d = gaussian_distribution(scale=1.0, bound_type=BoundType.IS_DOMINATED)
        assert isinstance(d, DenseDiscreteDist)
        assert not isinstance(d, PLDRealization)

    def test_laplace_dominates_returns_pld_realization(self):
        """Upper-bound Laplace discretization must yield a validated :class:`PLDRealization`."""
        d = laplace_distribution(scale=1.0, bound_type=BoundType.DOMINATES)
        assert isinstance(d, PLDRealization)

    def test_laplace_is_dominated_returns_dense_dist(self):
        """Lower-bound Laplace path returns a dense grid without realization metadata."""
        d = laplace_distribution(scale=1.0, bound_type=BoundType.IS_DOMINATED)
        assert isinstance(d, DenseDiscreteDist)
        assert not isinstance(d, PLDRealization)

    def test_count_noise_returns_directional_realizations(self):
        """The public discrete-noise helper returns valid REMOVE and ADD PLDs."""
        grid = GridSpec(step=1.0, n=5, anchor=-2.0)
        noise_dist = DenseDiscreteDist(grid=grid, prob_arr=np.array([0.1, 0.2, 0.4, 0.2, 0.1]))
        remove, add = discrete_distribution(
            noise_dist=noise_dist,
            loss_discretization=0.05,
            tail_truncation=0.0,
        )
        assert isinstance(remove, PLDRealization)
        assert isinstance(add, PLDRealization)
        assert _total_mass(remove) == pytest.approx(1.0)
        assert _total_mass(add) == pytest.approx(1.0)

    @pytest.mark.parametrize("mech", [gaussian_distribution, laplace_distribution])
    @pytest.mark.parametrize("bt", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_mass_conservation(self, mech, bt):
        """Finite plus boundary masses must sum to one for either bound type."""
        d = mech(scale=1.0, bound_type=bt)
        assert abs(_total_mass(d) - 1.0) < MASS_TOL

    @pytest.mark.parametrize("mech", [gaussian_distribution, laplace_distribution])
    def test_dominates_has_zero_p_min(self, mech):
        """Dominating (upper-bound) discretizations must not allocate mass at −∞."""
        d = mech(scale=1.0, bound_type=BoundType.DOMINATES)
        assert d.p_min == 0.0

    @pytest.mark.parametrize("mech", [gaussian_distribution, laplace_distribution])
    @pytest.mark.parametrize("bad_scale", [0.0, -1.0])
    def test_invalid_scale_raises(self, mech, bad_scale):
        """Non-positive noise scales are rejected before any discretization work."""
        with pytest.raises(ValueError):
            mech(scale=bad_scale)

    @pytest.mark.parametrize("mech", [gaussian_distribution, laplace_distribution])
    def test_rejects_bound_type_both(self, mech):
        """``BoundType.BOTH`` is unsupported for one-sided mechanism discretization."""
        with pytest.raises(ValueError):
            mech(scale=1.0, bound_type=BoundType.BOTH)


# ===================================================================
# 2. PLDRealization type
# ===================================================================


class TestPLDRealizationType:
    """``PLDRealization`` construction, invariants, copying, and truncation helpers."""

    def test_construction_with_valid_data(self):
        """Accept valid grids on non-negative losses with ``E[exp(-L)] < 1``."""
        prob_arr = np.array([0.3, 0.5, 0.2])
        grid = GridSpec(step=0.1, n=prob_arr.size, anchor=0.1)
        r = PLDRealization(grid=grid, prob_arr=prob_arr)
        assert r.step == 0.1
        np.testing.assert_array_equal(r.prob_arr, prob_arr)

    def test_rejects_nonzero_p_min(self):
        """Reject mass at negative infinity while claiming a PLD realization."""
        grid = GridSpec(step=0.1, n=1, anchor=0.1)
        prob_arr = np.array([0.5])
        with pytest.raises(ValueError):
            PLDRealization(grid=grid, prob_arr=prob_arr, p_min=0.5)

    def test_rejects_sub_tolerance_nonzero_p_min(self):
        """The semantic p_min invariant is exact, not a floating-point estimate."""
        p_min = np.finfo(float).eps
        grid = GridSpec(step=0.1, n=1, anchor=0.1)
        prob_arr = np.array([1.0 - p_min])
        with pytest.raises(ValueError, match="requires p_min = 0"):
            PLDRealization(grid=grid, prob_arr=prob_arr, p_min=p_min)

    def test_deepcopy_is_independent(self):
        """``copy.deepcopy`` must detach arrays and preserve immutability."""
        r = gaussian_distribution(scale=1.0)
        c = copy.deepcopy(r)
        assert c is not r
        assert c.prob_arr is not r.prob_arr
        with pytest.raises(ValueError, match="read-only"):
            c.prob_arr[0] = -999.0

    def test_realization_from_dominating_gaussian_grid(self):
        """Build a ``PLDRealization`` on a dominating Gaussian grid without losing mass."""
        d = gaussian_distribution(scale=1.0, bound_type=BoundType.DOMINATES)
        r = PLDRealization(grid=d.grid, prob_arr=d.prob_arr, p_min=d.p_min, p_max=d.p_max)
        assert isinstance(r, PLDRealization)
        assert abs(_total_mass(r) - 1.0) < MASS_TOL

    def test_truncate_edges(self):
        """Tail truncation should shrink support, keep the realization type, and conserve mass."""
        d = gaussian_distribution(scale=1.0)
        orig_len = len(d.prob_arr)
        t = truncate_edges_stoch_dom(dist=d, tail_truncation=0.05, bound_type=BoundType.DOMINATES)
        assert isinstance(t, PLDRealization)
        assert len(t.prob_arr) <= orig_len
        assert abs(_total_mass(t) - 1.0) < MASS_TOL

    @pytest.mark.parametrize(
        ("bound_type", "expected_type"),
        [
            (BoundType.DOMINATES, PLDRealization),
            (BoundType.IS_DOMINATED, DenseDiscreteDist),
        ],
    )
    def test_public_linear_rediscretization_routes_by_bound(self, bound_type, expected_type):
        """The public router fixes CtD/stochastic selection from bound semantics."""
        source = gaussian_distribution(scale=2.0, value_discretization=0.05)
        result = project_discrete_dist(
            dist=source,
            tail_truncation=0.0,
            loss_discretization=0.1,
            bound_type=bound_type,
        )

        assert isinstance(result, expected_type)
        if bound_type == BoundType.IS_DOMINATED:
            assert not isinstance(result, PLDRealization)


# ===================================================================
# 3. Gaussian allocation API
# ===================================================================


class TestGaussianAllocationAPI:
    """Gaussian random-allocation helpers should return finite, ordered privacy metrics."""

    def test_epsilon_range_returns_valid_bounds(self):
        """Bracketing helpers must return ordered, finite ε values for a fixed δ query."""
        upper, lower = gaussian_allocation_epsilon_range(
            delta=DELTA,
            sigma=SIGMA,
            num_steps=NUM_STEPS,
        )
        assert np.isfinite(upper) and upper > 0
        assert np.isfinite(lower) and lower > 0
        assert upper >= lower

    def test_epsilon_configurable_returns_positive_float(self):
        """``gaussian_allocation_epsilon_configurable`` should return a finite positive float ε."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, delta=DELTA)
        eps = gaussian_allocation_epsilon_configurable(params=params, config=COARSE_CONFIG)
        assert isinstance(eps, float)
        assert np.isfinite(eps) and eps > 0

    def test_delta_configurable_returns_valid_probability(self):
        """``gaussian_allocation_delta_configurable`` must return δ strictly inside (0, 1)."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, epsilon=EPSILON)
        d = gaussian_allocation_delta_configurable(params=params, config=COARSE_CONFIG)
        assert isinstance(d, float)
        assert 0 < d < 1

    def test_pld_returns_privacy_loss_distribution(self):
        """``gaussian_allocation_pld`` should emit a standard ``dp_accounting`` PLD object."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS)
        pld = gaussian_allocation_pld(params=params, config=COARSE_CONFIG)
        assert isinstance(pld, privacy_loss_distribution.PrivacyLossDistribution)

    def test_pld_epsilon_query(self):
        """The composed PLD must answer ``get_epsilon_for_delta`` with a finite positive ε."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS)
        pld = gaussian_allocation_pld(params=params, config=COARSE_CONFIG)
        eps = pld.get_epsilon_for_delta(DELTA)
        assert np.isfinite(eps) and eps > 0

    def test_pld_delta_query(self):
        """The composed PLD must answer ``get_delta_for_epsilon`` with a valid δ in [0, 1)."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS)
        pld = gaussian_allocation_pld(params=params, config=COARSE_CONFIG)
        d = pld.get_delta_for_epsilon(EPSILON)
        assert 0 <= d < 1

    def test_epsilon_delta_round_trip(self):
        """ε→δ conversion should stay consistent with the original δ up to discretization slack."""
        params_eps = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, delta=DELTA)
        eps = gaussian_allocation_epsilon_configurable(params=params_eps, config=COARSE_CONFIG)
        params_del = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, epsilon=eps)
        d = gaussian_allocation_delta_configurable(params=params_del, config=COARSE_CONFIG)
        assert d <= DELTA * 1.5  # allow slack for discretization

    def test_dominates_geq_is_dominated(self):
        """Upper-bound (DOMINATES) ε should be no smaller than optimistic (IS_DOMINATED) ε."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, delta=DELTA)
        eps_dom = gaussian_allocation_epsilon_configurable(
            params=params,
            config=COARSE_CONFIG,
            bound_type=BoundType.DOMINATES,
        )
        eps_sub = gaussian_allocation_epsilon_configurable(
            params=params,
            config=COARSE_CONFIG,
            bound_type=BoundType.IS_DOMINATED,
        )
        assert eps_dom >= eps_sub

    def test_more_steps_increases_epsilon(self):
        """With full participation, doubling composed steps should not shrink the reported ε."""
        p5 = PrivacyParams(sigma=SIGMA, num_steps=5, num_selected=5, delta=DELTA)
        p10 = PrivacyParams(sigma=SIGMA, num_steps=10, num_selected=10, delta=DELTA)
        eps5 = gaussian_allocation_epsilon_configurable(params=p5, config=COARSE_CONFIG)
        eps10 = gaussian_allocation_epsilon_configurable(params=p10, config=COARSE_CONFIG)
        assert eps10 >= eps5

    def test_larger_sigma_decreases_epsilon(self):
        """Increasing Gaussian noise scale should (weakly) reduce ε for an identical δ target."""
        p_lo = PrivacyParams(sigma=1.0, num_steps=NUM_STEPS, delta=DELTA)
        p_hi = PrivacyParams(sigma=3.0, num_steps=NUM_STEPS, delta=DELTA)
        eps_lo = gaussian_allocation_epsilon_configurable(params=p_lo, config=COARSE_CONFIG)
        eps_hi = gaussian_allocation_epsilon_configurable(params=p_hi, config=COARSE_CONFIG)
        assert eps_lo >= eps_hi


# ===================================================================
# 4. Convolution methods
# ===================================================================


class TestConvolutionMethods:
    """Convolution backends selected via ``AllocationSchemeConfig`` should stay numerically sane."""

    @pytest.mark.parametrize("method", CONVOLUTION_METHODS)
    def test_all_methods_produce_finite_epsilon(self, method):
        """Every convolution method enum should still yield a finite ε under coarse settings."""
        cfg = AllocationSchemeConfig(
            loss_discretization=0.05,
            tail_truncation=1e-6,
            convolution_method=method,
        )
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, delta=DELTA)
        eps = gaussian_allocation_epsilon_configurable(
            params=params,
            config=cfg,
            bound_type=BoundType.DOMINATES,
        )
        assert np.isfinite(eps) and eps > 0

    def test_geom_supports_is_dominated(self):
        """Geometric convolution must remain stable when requesting optimistic accounting."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, delta=DELTA)
        eps = gaussian_allocation_epsilon_configurable(
            params=params,
            config=COARSE_CONFIG,
            bound_type=BoundType.IS_DOMINATED,
        )
        assert np.isfinite(eps) and eps > 0


class TestProjectionMethods:
    """The projection at the merges, selected via ``AllocationSchemeConfig``."""

    # Every combination of grid, configured projection and bound: the projection that
    # runs, or None where the combination is not supported.
    _GEOM, _MS_FFT = ConvolutionMethod.GEOM, ConvolutionMethod.MS_FFT
    _SUPPORTED = [
        (_GEOM, None, BoundType.DOMINATES, ProjectionMethod.STOCH_DOM),
        (_GEOM, None, BoundType.IS_DOMINATED, ProjectionMethod.STOCH_DOM),
        (_GEOM, ProjectionMethod.STOCH_DOM, BoundType.DOMINATES, ProjectionMethod.STOCH_DOM),
        (_GEOM, ProjectionMethod.STOCH_DOM, BoundType.IS_DOMINATED, ProjectionMethod.STOCH_DOM),
        (_GEOM, ProjectionMethod.CTD, BoundType.DOMINATES, ProjectionMethod.CTD),
        (_GEOM, ProjectionMethod.CTD, BoundType.IS_DOMINATED, None),
        (_MS_FFT, None, BoundType.DOMINATES, ProjectionMethod.CTD),
        (_MS_FFT, None, BoundType.IS_DOMINATED, None),
        (_MS_FFT, ProjectionMethod.STOCH_DOM, BoundType.DOMINATES, ProjectionMethod.STOCH_DOM),
        (_MS_FFT, ProjectionMethod.STOCH_DOM, BoundType.IS_DOMINATED, ProjectionMethod.STOCH_DOM),
        (_MS_FFT, ProjectionMethod.CTD, BoundType.DOMINATES, ProjectionMethod.CTD),
        (_MS_FFT, ProjectionMethod.CTD, BoundType.IS_DOMINATED, None),
    ]

    @pytest.mark.parametrize("grid, configured, bound_type, resolved", _SUPPORTED)
    def test_resolved_projection_for_every_configuration_and_bound(
        self, grid, configured, bound_type, resolved
    ):
        """Each grid has its own default; rounding gives both bounds, connect-the-dots upper."""
        config = AllocationSchemeConfig(convolution_method=grid, projection_method=configured)
        if resolved is None:
            with pytest.raises(ValueError, match="DOMINATES only"):
                resolve_projection_method(config=config, bound_type=bound_type)
        else:
            assert resolve_projection_method(config=config, bound_type=bound_type) == resolved

    @pytest.mark.parametrize("grid, configured, bound_type, resolved", _SUPPORTED)
    def test_allocation_supports_exactly_the_resolved_combinations(
        self, grid, configured, bound_type, resolved
    ):
        """The allocation API computes every supported combination and rejects the others."""
        config = AllocationSchemeConfig(
            loss_discretization=0.05,
            tail_truncation=1e-6,
            convolution_method=grid,
            projection_method=configured,
        )
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, delta=DELTA)
        if resolved is None:
            with pytest.raises(ValueError, match="DOMINATES only"):
                gaussian_allocation_epsilon_configurable(
                    params=params, config=config, bound_type=bound_type
                )
        else:
            eps = gaussian_allocation_epsilon_configurable(
                params=params, config=config, bound_type=bound_type
            )
            assert np.isfinite(eps) and eps > 0

    def test_resolver_rejects_bound_type_both(self):
        """The projection is resolved for one bound orientation."""
        config = AllocationSchemeConfig()
        with pytest.raises(ValueError):
            resolve_projection_method(config=config, bound_type=BoundType.BOTH)


# ===================================================================
# 5. General (realization) allocation API
# ===================================================================


class TestGeneralAllocationAPI:
    """Realization-driven allocation should mirror the Gaussian API shape and validation rules."""

    def test_epsilon_returns_positive_float(self):
        """``general_allocation_epsilon`` must return a finite positive ε for valid PLD inputs."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        eps = general_allocation_epsilon(
            delta=DELTA,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=NUM_EPOCHS,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
        )
        assert isinstance(eps, float) and np.isfinite(eps) and eps > 0

    def test_delta_returns_valid_probability(self):
        """``general_allocation_delta`` must return δ strictly between zero and one."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        d = general_allocation_delta(
            epsilon=EPSILON,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=NUM_EPOCHS,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
        )
        assert isinstance(d, float) and 0 < d < 1

    def test_pld_returns_privacy_loss_distribution(self):
        """``general_allocation_pld`` should emit a reusable ``PrivacyLossDistribution`` object."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        pld = general_allocation_pld(
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=NUM_EPOCHS,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
        )
        assert isinstance(pld, privacy_loss_distribution.PrivacyLossDistribution)

    def test_epsilon_delta_round_trip(self):
        """ε→δ round trip for the realization API should stay near the requested δ budget."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        eps = general_allocation_epsilon(
            delta=DELTA,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=NUM_EPOCHS,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
        )
        d = general_allocation_delta(
            epsilon=eps,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=NUM_EPOCHS,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
        )
        assert d <= DELTA * 1.5

    def test_dominates_geq_is_dominated(self):
        """Dominating compositions should report ε no smaller than optimistic compositions."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        eps_dom = general_allocation_epsilon(
            delta=DELTA,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=NUM_EPOCHS,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
            bound_type=BoundType.DOMINATES,
        )
        eps_sub = general_allocation_epsilon(
            delta=DELTA,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=NUM_EPOCHS,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
            bound_type=BoundType.IS_DOMINATED,
        )
        assert eps_dom >= eps_sub

    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    @pytest.mark.parametrize("slot", ["remove_realization", "add_realization"])
    @pytest.mark.parametrize(
        "api, query",
        [
            (general_allocation_pld, {}),
            (general_allocation_epsilon, {"delta": DELTA}),
            (general_allocation_delta, {"epsilon": EPSILON}),
        ],
        ids=["pld", "epsilon", "delta"],
    )
    def test_rejects_an_input_that_is_not_a_pld_realization(self, api, query, slot, bound_type):
        """A lower-bound mechanism discretization is not a PLD and is rejected up front."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        lower = gaussian_distribution(scale=SIGMA, bound_type=BoundType.IS_DOMINATED)
        assert isinstance(lower, DenseDiscreteDist)
        assert not isinstance(lower, PLDRealization)
        kwargs = {
            "num_steps": NUM_STEPS,
            "num_selected": NUM_SELECTED,
            "num_epochs": NUM_EPOCHS,
            "remove_realization": rm,
            "add_realization": ad,
            "config": COARSE_CONFIG,
            "bound_type": bound_type,
            **query,
        }
        kwargs[slot] = lower
        with pytest.raises(TypeError, match=f"{slot} must be PLDRealization"):
            api(**kwargs)

    def test_warns_when_a_realization_is_coarser_than_the_loss_budget(self):
        """On GEOM the realization's step floors the one-step factor's, and that is reported."""
        coarse = gaussian_distribution(scale=SIGMA, value_discretization=0.05)
        with pytest.warns(UserWarning, match="realization's grid step 5.000000e-02 is coarser"):
            general_allocation_pld(
                num_steps=NUM_STEPS,
                num_selected=NUM_SELECTED,
                num_epochs=NUM_EPOCHS,
                remove_realization=coarse,
                add_realization=coarse,
                config=COARSE_CONFIG,
            )

    @pytest.mark.parametrize(
        "convolution_method, value_discretization",
        [(ConvolutionMethod.GEOM, 1e-3), (ConvolutionMethod.MS_FFT, 0.05)],
        ids=["geom-fine-realization", "ms-fft-coarse-realization"],
    )
    def test_no_step_floor_warning_when_the_grid_is_not_floored(
        self, convolution_method, value_discretization
    ):
        """A fine realization on GEOM, or any realization on MS_FFT, sets no floor."""
        realization = gaussian_distribution(scale=SIGMA, value_discretization=value_discretization)
        config = AllocationSchemeConfig(
            loss_discretization=COARSE_CONFIG.loss_discretization,
            tail_truncation=COARSE_CONFIG.tail_truncation,
            convolution_method=convolution_method,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            general_allocation_pld(
                num_steps=NUM_STEPS,
                num_selected=NUM_SELECTED,
                num_epochs=NUM_EPOCHS,
                remove_realization=realization,
                add_realization=realization,
                config=config,
            )
        assert not [w for w in caught if "realization's grid step" in str(w.message)]

    def test_more_epochs_increases_epsilon(self):
        """Repeating the allocation schedule across more epochs cannot shrink the ε estimate."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        eps1 = general_allocation_epsilon(
            delta=DELTA,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=1,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
        )
        eps2 = general_allocation_epsilon(
            delta=DELTA,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=2,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
        )
        assert eps2 >= eps1


# ===================================================================
# 6. Cross-path consistency
# ===================================================================


class TestCrossPathConsistency:
    """Gaussian shortcuts and realization-based APIs should stay on the same numerical page."""

    def test_gaussian_and_general_epsilon_agree(self):
        """Specialized Gaussian accounting should match the general PLD pipeline within slack."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, delta=DELTA)
        eps_gauss = gaussian_allocation_epsilon_configurable(
            params=params,
            config=COARSE_CONFIG,
            bound_type=BoundType.DOMINATES,
        )
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        eps_general = general_allocation_epsilon(
            delta=DELTA,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=NUM_EPOCHS,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
            bound_type=BoundType.DOMINATES,
        )
        assert (
            abs(eps_gauss - eps_general) < 0.3
        ), f"Gaussian={eps_gauss:.6f}, General={eps_general:.6f}"

    def test_gaussian_and_general_delta_agree(self):
        """Gaussian vs realization δ queries should agree in order of magnitude."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, epsilon=EPSILON)
        d_gauss = gaussian_allocation_delta_configurable(
            params=params,
            config=COARSE_CONFIG,
            bound_type=BoundType.DOMINATES,
        )
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        d_general = general_allocation_delta(
            epsilon=EPSILON,
            num_steps=NUM_STEPS,
            num_selected=NUM_SELECTED,
            num_epochs=NUM_EPOCHS,
            remove_realization=rm,
            add_realization=ad,
            config=COARSE_CONFIG,
            bound_type=BoundType.DOMINATES,
        )
        assert d_gauss > 0 and d_general > 0
        ratio = max(d_gauss, d_general) / min(d_gauss, d_general)
        assert ratio < 10, f"Gaussian={d_gauss:.2e}, General={d_general:.2e}"


# ===================================================================
# 7. Subsampling
# ===================================================================


class TestSubsampling:
    """Subsampling helpers should preserve mass, honor sampling probabilities, and shrink ε."""

    def test_realization_remove_returns_pld(self):
        """``subsample_pld_realization`` on REMOVE must return another valid ``PLDRealization``."""
        rm, _ = _make_gaussian_realizations(sigma=SIGMA)
        out = subsample_pld_realization(base_pld=rm, sampling_prob=0.5, direction=Direction.REMOVE)
        assert isinstance(out, PLDRealization)
        assert abs(_total_mass(out) - 1.0) < MASS_TOL

    def test_realization_add_returns_pld(self):
        """``subsample_pld_realization`` on ADD must return another valid ``PLDRealization``."""
        _, ad = _make_gaussian_realizations(sigma=SIGMA)
        out = subsample_pld_realization(base_pld=ad, sampling_prob=0.5, direction=Direction.ADD)
        assert isinstance(out, PLDRealization)
        assert abs(_total_mass(out) - 1.0) < MASS_TOL

    def test_realization_q1_returns_same(self):
        """Sampling probability ``1.0`` should short-circuit to the original realization object."""
        rm, _ = _make_gaussian_realizations(sigma=SIGMA)
        out = subsample_pld_realization(base_pld=rm, sampling_prob=1.0, direction=Direction.REMOVE)
        assert out is rm

    def test_realization_rejects_direction_both(self):
        """Direction ``BOTH`` is invalid for the realization-level subsampling helper."""
        rm, _ = _make_gaussian_realizations(sigma=SIGMA)
        with pytest.raises(ValueError):
            subsample_pld_realization(base_pld=rm, sampling_prob=0.5, direction=Direction.BOTH)

    @pytest.mark.parametrize("bad_q", [0.0, -0.1, 1.5])
    def test_realization_rejects_invalid_prob(self, bad_q):
        """Sampling probabilities must lie strictly between zero and one."""
        rm, _ = _make_gaussian_realizations(sigma=SIGMA)
        with pytest.raises(ValueError):
            subsample_pld_realization(base_pld=rm, sampling_prob=bad_q, direction=Direction.REMOVE)

    def test_subsample_pld_returns_pld_type(self):
        """``subsample_pld`` must return another ``PrivacyLossDistribution`` for downstream use."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS)
        pld = gaussian_allocation_pld(params=params, config=COARSE_CONFIG)
        out = subsample_pld(pld=pld, sampling_probability=0.5)
        assert isinstance(out, privacy_loss_distribution.PrivacyLossDistribution)

    def test_subsample_pld_reduces_epsilon(self):
        """Subsampling with ``q<1`` should not increase ε at a fixed δ (up to numeric slack)."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS)
        pld = gaussian_allocation_pld(params=params, config=COARSE_CONFIG)
        subsampled = subsample_pld(pld=pld, sampling_probability=0.5)
        eps_orig = pld.get_epsilon_for_delta(DELTA)
        eps_sub = subsampled.get_epsilon_for_delta(DELTA)
        assert eps_sub <= eps_orig + 0.01  # small slack for numerics

    def test_subsample_pld_q1_returns_same(self):
        """``sampling_probability=1`` should return the identical dp_accounting PLD object."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS)
        pld = gaussian_allocation_pld(params=params, config=COARSE_CONFIG)
        out = subsample_pld(pld=pld, sampling_probability=1.0)
        assert out is pld

    @pytest.mark.parametrize("bad_q", [0.0, -0.5, 1.1])
    def test_subsample_pld_rejects_invalid_prob(self, bad_q):
        """Invalid ``sampling_probability`` values must raise before touching the PLD."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS)
        pld = gaussian_allocation_pld(params=params, config=COARSE_CONFIG)
        with pytest.raises(ValueError):
            subsample_pld(pld=pld, sampling_probability=bad_q)


# ===================================================================
# 8. Input validation
# ===================================================================


class TestInputValidation:
    """Public helpers should fail fast on nonsensical privacy parameters."""

    @pytest.mark.parametrize("bad_sigma", [0, -1.0])
    def test_gaussian_rejects_bad_sigma(self, bad_sigma):
        """Gaussian accounting must reject non-positive noise scales."""
        with pytest.raises(ValueError):
            PrivacyParams(sigma=bad_sigma, num_steps=NUM_STEPS, delta=DELTA)

    def test_gaussian_rejects_zero_num_steps(self):
        """Composition depth ``num_steps`` must be strictly positive."""
        with pytest.raises(ValueError):
            PrivacyParams(sigma=SIGMA, num_steps=0, delta=DELTA)

    @pytest.mark.parametrize("bad_sigma", [np.nan, np.inf])
    def test_gaussian_rejects_nonfinite_sigma(self, bad_sigma):
        """Non-finite noise scales are rejected explicitly."""
        with pytest.raises(ValueError, match="finite"):
            PrivacyParams(sigma=bad_sigma, num_steps=NUM_STEPS, delta=DELTA)

    def test_gaussian_rejects_boolean_integer_parameters(self):
        """Booleans are not accepted as composition counts."""
        with pytest.raises(TypeError, match="num_steps must be an integer"):
            PrivacyParams(sigma=SIGMA, num_steps=True, delta=DELTA)

    def test_rejects_invalid_convolution_method_type(self):
        """Configuration enum fields must contain the declared enum type."""
        with pytest.raises(TypeError, match="convolution_method"):
            AllocationSchemeConfig(convolution_method="fft")  # type: ignore[arg-type]

    def test_rejects_invalid_projection_method_type(self):
        """The projection field holds a ``ProjectionMethod`` or ``None``."""
        with pytest.raises(TypeError, match="projection_method"):
            AllocationSchemeConfig(projection_method="ctd")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_cells", [0, -4])
    def test_rejects_cells_per_band_cap_that_is_not_positive(self, bad_cells):
        """The cap on the points per octave is a positive integer."""
        with pytest.raises(ValueError, match="max_cells_per_band"):
            AllocationSchemeConfig(max_cells_per_band=bad_cells)

    @pytest.mark.parametrize("bad_grid", [0, 1, 2, 3, -1])
    def test_rejects_closing_grid_cap_below_the_smallest_aligned_grid(self, bad_grid):
        """An aligned grid over one interval of its step can take four points."""
        with pytest.raises(ValueError, match="max_closing_grid"):
            AllocationSchemeConfig(max_closing_grid=bad_grid)

    @pytest.mark.parametrize("field", ["max_cells_per_band", "max_closing_grid"])
    def test_rejects_cap_that_is_not_an_integer(self, field):
        """The caps are integers."""
        with pytest.raises(TypeError, match=field):
            AllocationSchemeConfig(**{field: 1024.0})

    @pytest.mark.parametrize("bad_delta", [0.0, 1.0, -0.1])
    def test_gaussian_rejects_bad_delta(self, bad_delta):
        """δ targets must be valid probabilities strictly between zero and one."""
        with pytest.raises(ValueError):
            PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, delta=bad_delta)

    @pytest.mark.parametrize("bad_eps", [0.0, -1.0])
    def test_gaussian_rejects_bad_epsilon(self, bad_eps):
        """ε targets must be strictly positive when requesting δ via the configurable API."""
        with pytest.raises(ValueError):
            PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, epsilon=bad_eps)

    def test_num_selected_exceeds_num_steps(self):
        """``num_selected`` cannot exceed ``num_steps`` for random-allocation accounting."""
        with pytest.raises(ValueError, match="num_selected"):
            PrivacyParams(
                sigma=SIGMA,
                num_steps=5,
                num_selected=10,
                delta=DELTA,
            )

    def test_bound_type_both_rejected_for_gaussian_epsilon(self):
        """``BoundType.BOTH`` is not implemented for the Gaussian ε helper."""
        params = PrivacyParams(sigma=SIGMA, num_steps=NUM_STEPS, delta=DELTA)
        with pytest.raises(ValueError):
            gaussian_allocation_epsilon_configurable(
                params=params,
                config=COARSE_CONFIG,
                bound_type=BoundType.BOTH,
            )

    def test_bound_type_both_rejected_for_general_epsilon(self):
        """``BoundType.BOTH`` is not implemented for the realization-based ε helper."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        with pytest.raises(ValueError):
            general_allocation_epsilon(
                delta=DELTA,
                num_steps=NUM_STEPS,
                num_selected=NUM_SELECTED,
                num_epochs=NUM_EPOCHS,
                remove_realization=rm,
                add_realization=ad,
                config=COARSE_CONFIG,
                bound_type=BoundType.BOTH,
            )

    def test_general_rejects_bad_delta(self):
        """``general_allocation_epsilon`` must reject δ outside (0, 1)."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        with pytest.raises(ValueError):
            general_allocation_epsilon(
                delta=0.0,
                num_steps=NUM_STEPS,
                num_selected=NUM_SELECTED,
                num_epochs=NUM_EPOCHS,
                remove_realization=rm,
                add_realization=ad,
                config=COARSE_CONFIG,
            )

    def test_general_rejects_bad_epsilon(self):
        """``general_allocation_delta`` must reject non-positive ε targets."""
        rm, ad = _make_gaussian_realizations(sigma=SIGMA)
        with pytest.raises(ValueError):
            general_allocation_delta(
                epsilon=0.0,
                num_steps=NUM_STEPS,
                num_selected=NUM_SELECTED,
                num_epochs=NUM_EPOCHS,
                remove_realization=rm,
                add_realization=ad,
                config=COARSE_CONFIG,
            )
