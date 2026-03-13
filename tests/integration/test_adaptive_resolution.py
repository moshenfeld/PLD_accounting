"""Integration tests for adaptive random-allocation queries."""

import warnings

import numpy as np
import pytest

import PLD_accounting.random_allocation_api as random_allocation_api_module
from PLD_accounting import (
    AllocationSchemeConfig,
    PrivacyParams,
    gaussian_allocation_delta_extended,
    gaussian_allocation_delta_range,
    gaussian_allocation_epsilon_extended,
    gaussian_allocation_epsilon_range,
)
from PLD_accounting.adaptive_random_allocation import (
    AdaptiveResult,
    optimize_allocation_delta_range,
    optimize_allocation_epsilon_range,
    estimate_poisson_query,
)
from PLD_accounting.random_allocation_api import (
    gaussian_allocation_PLD as allocation_pld_api,
    gaussian_allocation_delta_extended as gaussian_allocation_delta_extended_api,
    gaussian_allocation_delta_range as gaussian_allocation_delta_range_api,
    gaussian_allocation_epsilon_extended as gaussian_allocation_epsilon_extended_api,
    gaussian_allocation_epsilon_range as gaussian_allocation_epsilon_range_api,
)
from PLD_accounting.random_allocation_gaussian import allocation_PMF_from_gaussian, compute_conv_params
from PLD_accounting.types import BoundType, ConvolutionMethod, Direction


def _stub_result(target_accuracy: float, value: float = 0.5, converged: bool = True) -> AdaptiveResult:
    lower_bound = value - max(target_accuracy / 2.0, 1e-12)
    return AdaptiveResult(
        upper_bound=value,
        lower_bound=lower_bound,
        absolute_gap=value - lower_bound,
        converged=converged,
        iterations=2,
        initial_discretization=0.1,
        discretization=0.05,
        initial_tail_truncation=1e-6,
        tail_truncation=5e-7,
        target_accuracy=target_accuracy,
    )


def _run_real_epsilon_adaptive(
    *,
    sigma: float,
    num_steps: int,
    delta: float,
    epsilon_accuracy: float,
    num_selected: int = 1,
    num_epochs: int = 1,
) -> AdaptiveResult:
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        delta=delta,
    )

    return optimize_allocation_epsilon_range(
        params=params,
        target_accuracy=epsilon_accuracy,
        pld_builder=allocation_pld_api,
    )


def _run_real_delta_adaptive(
    *,
    sigma: float,
    num_steps: int,
    epsilon: float,
    delta_accuracy: float,
    num_selected: int = 1,
    num_epochs: int = 1,
) -> AdaptiveResult:
    params = PrivacyParams(
        sigma=sigma,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        epsilon=epsilon,
    )

    return optimize_allocation_delta_range(
        params=params,
        target_accuracy=delta_accuracy,
        pld_builder=allocation_pld_api,
    )


class TestAdaptivePublicApi:
    def test_epsilon_range_returns_tuple(self):
        bounds = gaussian_allocation_epsilon_range(
            sigma=2.0,
            num_steps=20,
            delta=1e-6,
            epsilon_accuracy=1e-2,
        )

        assert isinstance(bounds, tuple)
        assert len(bounds) == 2
        assert np.isfinite(bounds[0])
        assert np.isfinite(bounds[1])
        assert bounds[0] >= bounds[1]

    def test_delta_range_returns_tuple(self):
        bounds = gaussian_allocation_delta_range(
            sigma=2.0,
            num_steps=20,
            epsilon=1.0,
            delta_accuracy=1e-6,
        )

        assert isinstance(bounds, tuple)
        assert len(bounds) == 2
        assert np.isfinite(bounds[0])
        assert np.isfinite(bounds[1])
        assert bounds[0] >= bounds[1]

    def test_epsilon_range_consistent_with_fixed_resolution(self):
        sigma = 1.5
        num_steps = 5
        delta = 1e-5
        epsilon_accuracy = 5e-2

        upper_bound, lower_bound = gaussian_allocation_epsilon_range(
            sigma=sigma,
            num_steps=num_steps,
            delta=delta,
            epsilon_accuracy=epsilon_accuracy,
        )

        params = PrivacyParams(sigma=sigma, num_steps=num_steps, delta=delta)
        config_fine = AllocationSchemeConfig(
            loss_discretization=1e-2,
            tail_truncation=1e-8,
        )
        epsilon_fine = gaussian_allocation_epsilon_extended(params=params, config=config_fine)

        assert upper_bound >= epsilon_fine >= lower_bound
        assert upper_bound - lower_bound < epsilon_accuracy

    def test_delta_range_consistent_with_fixed_resolution(self):
        sigma = 1.5
        num_steps = 5
        epsilon = 1.0
        delta_accuracy = 1e-4

        upper_bound, lower_bound = gaussian_allocation_delta_range(
            sigma=sigma,
            num_steps=num_steps,
            epsilon=epsilon,
            delta_accuracy=delta_accuracy,
        )

        params = PrivacyParams(sigma=sigma, num_steps=num_steps, epsilon=epsilon)
        config_fine = AllocationSchemeConfig(
            loss_discretization=1e-2,
            tail_truncation=1e-8,
        )
        delta_fine_upper = gaussian_allocation_delta_extended(params=params, config=config_fine)
        assert np.isfinite(delta_fine_upper)
        assert upper_bound >= lower_bound >= 0.0
        assert upper_bound - lower_bound < delta_accuracy

    @pytest.mark.parametrize(
        ("sigma", "num_steps", "loss_discretization", "tail_truncation"),
        [
            (2.0, 20, 0.5, 1e-6),
            (1.5, 5, 0.25, 1e-4),
        ],
    )
    def test_remove_geom_uses_shared_ratio_for_delta_regressions(
        self,
        sigma: float,
        num_steps: int,
        loss_discretization: float,
        tail_truncation: float,
    ):
        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            epsilon=1.0,
        )
        config = AllocationSchemeConfig(
            loss_discretization=loss_discretization,
            tail_truncation=tail_truncation,
            convolution_method=ConvolutionMethod.GEOM,
        )

        dist = allocation_PMF_from_gaussian(
            conv_params=compute_conv_params(params=params, config=config),
            direction=Direction.REMOVE,
            bound_type=BoundType.DOMINATES,
            convolution_method=ConvolutionMethod.GEOM,
        )

        assert np.all(np.isfinite(dist.x_array))
        assert dist.PMF_array.size > 1

    def test_default_target_accuracy_is_forwarded_for_internal_initialization(self, monkeypatch):
        captured: dict[str, float] = {}

        def fake_optimize_allocation_epsilon_range(
            *,
            params,
            target_accuracy,
            pld_builder,
        ):
            captured["target_accuracy"] = target_accuracy
            return _stub_result(target_accuracy=0.07, value=0.7)

        monkeypatch.setattr(random_allocation_api_module, "optimize_allocation_epsilon_range", fake_optimize_allocation_epsilon_range)

        value = gaussian_allocation_epsilon_range(
            sigma=5.0,
            num_steps=10,
            delta=1e-6,
        )

        assert np.isclose(captured["target_accuracy"], -1.0)
        assert np.allclose(value, (0.7, 0.665))

    def test_default_target_accuracy_is_forwarded_for_internal_delta_initialization(self, monkeypatch):
        captured: dict[str, float] = {}

        def fake_optimize_allocation_delta_range(
            *,
            params,
            target_accuracy,
            pld_builder,
        ):
            captured["target_accuracy"] = target_accuracy
            return _stub_result(target_accuracy=4e-2, value=0.4)

        monkeypatch.setattr(random_allocation_api_module, "optimize_allocation_delta_range", fake_optimize_allocation_delta_range)

        value = gaussian_allocation_delta_range(
            sigma=5.0,
            num_steps=10,
            epsilon=1.0,
        )

        assert np.isclose(captured["target_accuracy"], -1.0)
        assert value == (0.4, 0.38)

    def test_epsilon_range_uses_tail_shifted_lower_bound_formula(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def get_epsilon_for_delta(self, delta_value: float) -> float:
                if np.isclose(delta_value, params.delta):
                    return 1.2
                if np.isclose(delta_value, params.delta - 1e-7):
                    return 0.9
                raise AssertionError(f"unexpected delta: {delta_value}")

        def fake_builder(*, params, config, bound_type):
            return FakePLD()

        result = optimize_allocation_epsilon_range(
            params=params,
            target_accuracy=0.5,
            initial_discretization=0.2,
            initial_tail_truncation=1e-7,
            pld_builder=fake_builder,
        )

        assert np.isclose(result.upper_bound, 1.2)
        assert np.isclose(result.lower_bound, 0.9)

    def test_delta_range_uses_discretization_shifted_lower_bound_formula(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, epsilon=1.0)

        class FakePLD:
            def get_delta_for_epsilon(self, epsilon_value: float) -> float:
                if np.isclose(epsilon_value, params.epsilon):
                    return 0.3
                if np.isclose(epsilon_value, params.epsilon - 0.1):
                    return 0.2
                raise AssertionError(f"unexpected epsilon: {epsilon_value}")

        def fake_builder(*, params, config, bound_type):
            return FakePLD()

        result = optimize_allocation_delta_range(
            params=params,
            target_accuracy=0.2,
            initial_discretization=0.1,
            initial_tail_truncation=1e-6,
            pld_builder=fake_builder,
        )

        assert np.isclose(result.upper_bound, 0.3)
        assert np.isclose(result.lower_bound, 0.2)

    def test_num_selected_and_epochs_are_forwarded(self, monkeypatch):
        captured: dict[str, object] = {}

        def fake_optimize_allocation_epsilon_range(
            *,
            params,
            target_accuracy,
            pld_builder,
        ):
            captured["params"] = params
            return _stub_result(target_accuracy=target_accuracy)

        monkeypatch.setattr(random_allocation_api_module, "optimize_allocation_epsilon_range", fake_optimize_allocation_epsilon_range)

        gaussian_allocation_epsilon_range(
            sigma=2.0,
            num_steps=20,
            delta=1e-6,
            num_selected=5,
            num_epochs=2,
            epsilon_accuracy=1e-2,
        )

        assert captured["params"].num_selected == 5
        assert captured["params"].num_epochs == 2

    def test_api_module_exports_main_functions(self):
        assert gaussian_allocation_epsilon_range_api is gaussian_allocation_epsilon_range
        assert gaussian_allocation_delta_range_api is gaussian_allocation_delta_range
        assert gaussian_allocation_epsilon_extended_api is gaussian_allocation_epsilon_extended
        assert gaussian_allocation_delta_extended_api is gaussian_allocation_delta_extended
        assert callable(allocation_pld_api)


class TestAdaptiveIntegrationConvergence:
    @pytest.mark.parametrize(
        ("sigma", "num_steps", "num_selected", "num_epochs", "delta", "epsilon_accuracy"),
        [
            (1.2, 12, 1, 1, 1e-5, 2e-2),
            (2.0, 20, 1, 1, 1e-6, 1e-2),
        ],
    )
    def test_epsilon_range_converges_within_requested_gap_and_brackets_fine_value(
        self,
        sigma: float,
        num_steps: int,
        num_selected: int,
        num_epochs: int,
        delta: float,
        epsilon_accuracy: float,
    ):
        result = _run_real_epsilon_adaptive(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            delta=delta,
            epsilon_accuracy=epsilon_accuracy,
        )

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            delta=delta,
        )
        epsilon_fine = gaussian_allocation_epsilon_extended(
            params=params,
            config=AllocationSchemeConfig(
                loss_discretization=1e-2,
                tail_truncation=1e-8,
            ),
        )

        assert result.converged
        assert result.absolute_gap < result.target_accuracy
        assert np.isfinite(epsilon_fine)
        assert result.upper_bound >= result.lower_bound >= 0.0
        assert result.iterations >= 1

    @pytest.mark.parametrize(
        ("sigma", "num_steps", "num_selected", "num_epochs", "epsilon", "delta_accuracy"),
        [
            (1.5, 5, 1, 1, 1.0, 1e-4),
            (2.0, 20, 1, 1, 1.0, 1e-6),
        ],
    )
    def test_delta_range_converges_within_requested_gap_and_brackets_fine_value(
        self,
        sigma: float,
        num_steps: int,
        num_selected: int,
        num_epochs: int,
        epsilon: float,
        delta_accuracy: float,
    ):
        result = _run_real_delta_adaptive(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            epsilon=epsilon,
            delta_accuracy=delta_accuracy,
        )

        params = PrivacyParams(
            sigma=sigma,
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
            epsilon=epsilon,
        )
        delta_fine_upper = gaussian_allocation_delta_extended(
            params=params,
            config=AllocationSchemeConfig(
                loss_discretization=1e-2,
                tail_truncation=1e-8,
            ),
        )
        config_fine = AllocationSchemeConfig(
            loss_discretization=1e-2,
            tail_truncation=1e-8,
        )
        assert result.converged
        assert result.absolute_gap < result.target_accuracy
        assert np.isfinite(delta_fine_upper)
        assert result.upper_bound >= result.lower_bound >= 0.0
        assert result.iterations >= 1

    def test_public_epsilon_range_matches_direct_adaptive_result(self):
        direct = _run_real_epsilon_adaptive(
            sigma=1.5,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
            epsilon_accuracy=2e-2,
        )
        public = gaussian_allocation_epsilon_range(
            sigma=1.5,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            delta=1e-5,
            epsilon_accuracy=2e-2,
        )

        assert direct.converged
        assert np.allclose(public, (direct.upper_bound, direct.lower_bound))

    def test_public_delta_range_matches_direct_adaptive_result(self):
        direct = _run_real_delta_adaptive(
            sigma=1.7,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            epsilon=0.9,
            delta_accuracy=5e-5,
        )
        public = gaussian_allocation_delta_range(
            sigma=1.7,
            num_steps=10,
            num_selected=1,
            num_epochs=1,
            epsilon=0.9,
            delta_accuracy=5e-5,
        )

        assert direct.converged
        assert np.allclose(public, (direct.upper_bound, direct.lower_bound))


class TestAdaptiveConvergence:
    def test_returns_bounds_and_metadata(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

            def get_epsilon_for_delta(self, delta_value: float) -> float:
                return self.value

        def fake_builder(*, params, config, bound_type):
            if bound_type == BoundType.DOMINATES:
                return FakePLD(0.55)
            return FakePLD(0.545)

        result = optimize_allocation_epsilon_range(
            params=params,
            target_accuracy=1e-2,
            initial_discretization=0.1,
            initial_tail_truncation=1e-6,
            pld_builder=fake_builder,
        )

        assert isinstance(result, AdaptiveResult)
        assert np.isclose(result.upper_bound, 0.55)
        assert np.isclose(result.lower_bound, 0.545)
        assert result.converged
        assert result.absolute_gap < result.target_accuracy

    def test_negative_target_accuracy_relaxes_from_lower_bound(self, monkeypatch):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

            def get_epsilon_for_delta(self, delta_value: float) -> float:
                return self.value

        call_count = {"dominates": 0, "dominated": 0}

        def fake_builder(*, params, config, bound_type):
            if bound_type == BoundType.DOMINATES:
                call_count["dominates"] += 1
                return FakePLD(1.0 if call_count["dominates"] == 1 else 0.54)
            call_count["dominated"] += 1
            return FakePLD(0.5 if call_count["dominated"] == 1 else 0.51)

        monkeypatch.setattr(
            "PLD_accounting.adaptive_random_allocation.estimate_poisson_query",
            lambda **_: 0.01,
        )

        result = optimize_allocation_epsilon_range(
            params=params,
            target_accuracy=-1.0,
            pld_builder=fake_builder,
        )

        assert result.converged
        assert np.isclose(result.target_accuracy, 0.051)

    def test_propagates_builder_errors(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        def fake_builder(*, params, config, bound_type):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            optimize_allocation_epsilon_range(
                params=params,
                target_accuracy=1e-2,
                initial_discretization=0.1,
                initial_tail_truncation=1e-6,
                pld_builder=fake_builder,
            )

    def test_tracks_best_bounds_across_non_monotonic_refinement(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)
        call_count = {"dominates": 0, "dominated": 0}

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

            def get_epsilon_for_delta(self, delta_value: float) -> float:
                return self.value

        def fake_builder(*, params, config, bound_type):
            if bound_type == BoundType.DOMINATES:
                call_count["dominates"] += 1
                if call_count["dominates"] == 1:
                    return FakePLD(0.55)
                return FakePLD(0.56)

            call_count["dominated"] += 1
            if call_count["dominated"] == 1:
                return FakePLD(0.54)
            return FakePLD(0.541)

        result = optimize_allocation_epsilon_range(
            params=params,
            target_accuracy=1e-2,
            initial_discretization=0.1,
            initial_tail_truncation=1e-6,
            pld_builder=fake_builder,
        )

        assert np.isclose(result.upper_bound, 0.55)
        assert np.isclose(result.lower_bound, 0.541)
        assert result.converged

    def test_clamps_lower_bound_to_upper_bound(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

            def get_epsilon_for_delta(self, delta_value: float) -> float:
                return self.value

        def fake_builder(*, params, config, bound_type):
            if bound_type == BoundType.DOMINATES:
                return FakePLD(0.53)
            return FakePLD(0.54)

        result = optimize_allocation_epsilon_range(
            params=params,
            target_accuracy=1e-6,
            initial_discretization=0.1,
            initial_tail_truncation=1e-6,
            pld_builder=fake_builder,
        )

        assert np.isclose(result.upper_bound, 0.53)
        assert np.isclose(result.lower_bound, 0.53)

    def test_warns_when_not_converged(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

            def get_epsilon_for_delta(self, delta_value: float) -> float:
                return self.value

        def fake_builder(*, params, config, bound_type):
            if bound_type == BoundType.DOMINATES:
                return FakePLD(1.0)
            return FakePLD(0.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = optimize_allocation_epsilon_range(
                params=params,
                target_accuracy=1e-12,
                initial_discretization=0.1,
                initial_tail_truncation=1e-6,
                pld_builder=fake_builder,
            )

        assert not result.converged
        assert any("did not converge" in str(w.message) for w in caught)

    def test_reports_effective_initial_values_after_clamping(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

            def get_epsilon_for_delta(self, delta_value: float) -> float:
                return self.value

        def fake_builder(*, params, config, bound_type):
            if bound_type == BoundType.DOMINATES:
                return FakePLD(0.55)
            return FakePLD(0.545)

        result = optimize_allocation_epsilon_range(
            params=params,
            target_accuracy=1e-2,
            initial_discretization=0.2,
            initial_tail_truncation=1e-3,
            pld_builder=fake_builder,
        )

        assert np.isclose(result.initial_discretization, 0.1)
        assert np.isclose(result.initial_tail_truncation, 1e-4)

    def test_second_refinement_step_uses_tail_after_initial_discretization_step(self):
        params = PrivacyParams(sigma=2.0, num_steps=20, delta=1e-6)
        configs: list[AllocationSchemeConfig] = []

        class FakePLD:
            def __init__(self, value: float):
                self.value = value

            def get_epsilon_for_delta(self, delta_value: float) -> float:
                return self.value

        def fake_builder(*, params, config, bound_type):
            configs.append(config)
            if len(configs) <= 2:
                return FakePLD(1.0 if bound_type == BoundType.DOMINATES else 0.0)
            if len(configs) <= 4:
                return FakePLD(0.9 if bound_type == BoundType.DOMINATES else 0.0)
            return FakePLD(0.8 if bound_type == BoundType.DOMINATES else 0.0)

        result = optimize_allocation_epsilon_range(
            params=params,
            target_accuracy=1e-12,
            initial_discretization=0.2,
            initial_tail_truncation=1e-3,
            pld_builder=fake_builder,
        )

        assert not result.converged
        assert np.isclose(configs[0].loss_discretization, 0.1)
        assert np.isclose(configs[0].tail_truncation, 1e-4)
        assert np.isclose(configs[2].loss_discretization, 0.05)
        assert np.isclose(configs[2].tail_truncation, 1e-5)
        assert np.isclose(configs[4].loss_discretization, 0.025)
        assert np.isclose(configs[4].tail_truncation, 1e-6)


class TestPoissonAdaptiveEstimate:
    def test_poisson_estimate_uses_requested_sampling_probability_and_composition(self, monkeypatch):
        captured: dict[str, float | int] = {}

        class FakePLD:
            def self_compose(self, steps: int):
                captured["num_rounds"] = steps
                return self

        def fake_from_gaussian_mechanism(**kwargs):
            captured["sampling_prob"] = kwargs["sampling_prob"]
            captured["standard_deviation"] = kwargs["standard_deviation"]
            return FakePLD()

        monkeypatch.setattr(
            "PLD_accounting.adaptive_random_allocation.privacy_loss_distribution.from_gaussian_mechanism",
            fake_from_gaussian_mechanism,
        )

        params = PrivacyParams(
            sigma=3.0,
            num_steps=200,
            num_selected=10,
            num_epochs=4,
            delta=1e-6,
        )

        estimate = estimate_poisson_query(
            params=params,
            query_func=lambda pld: 0.25,
        )

        assert np.isclose(estimate, 0.25)
        assert np.isclose(captured["standard_deviation"], 3.0)
        assert np.isclose(captured["sampling_prob"], 10 / 200)
        assert captured["num_rounds"] == 40
