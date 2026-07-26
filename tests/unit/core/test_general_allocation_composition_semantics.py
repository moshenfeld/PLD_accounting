"""Unit tests for random-allocation composition wiring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import PLD_accounting.random_allocation_accounting as random_allocation_accounting_module
import PLD_accounting.random_allocation_api as random_allocation_api_module
import PLD_accounting.random_allocation_gaussian as random_allocation_gaussian_module
import PLD_accounting.random_allocation_realization as random_allocation_realization_module
from PLD_accounting.discrete_dist import DenseDiscreteDist, PLDRealization
from PLD_accounting.random_allocation_accounting import (
    _allocation_directional_pld_core as allocation_directional_pld_core,
)
from PLD_accounting.random_allocation_api import (
    gaussian_allocation_pld,
    general_allocation_pld,
)
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
    SpacingType,
)
from tests.test_tolerances import TestTolerances as TOL


def _simple_realization() -> PLDRealization:
    return PLDRealization(
        x_0=0.0,
        step=0.5,
        prob_arr=np.array([0.6, 0.3, 0.1]),
    )


def _stub_linear_dist() -> DenseDiscreteDist:
    return DenseDiscreteDist(
        x_0=0.0,
        step=0.5,
        prob_arr=np.array([0.5, 0.3, 0.2]),
    )


def _aligned_base_dist(step: float, origin_index: int) -> DenseDiscreteDist:
    """Build a small loss distribution on exact integer multiples of ``step``."""
    return DenseDiscreteDist(
        x_0=origin_index * step,
        step=step,
        prob_arr=np.array([1e-8, 0.2, 0.5, 0.29999999]),
    )


class TestGeneralAllocationWiring:
    """Tests that general (geometric-base) allocation delegates to shared helpers."""

    def test_general_allocation_uses_directional_plds(self, monkeypatch: pytest.MonkeyPatch):
        """General allocation builds one directional PLD per direction and composes them."""
        calls: list[dict[str, Any]] = []
        sentinel_pld = object()

        def fake_allocation_directional_pld(**kwargs: Any) -> DenseDiscreteDist:
            calls.append(kwargs)
            return _stub_linear_dist()

        def fake_compose_full_pld(*, remove_dist, add_dist, bound_type):
            del remove_dist, add_dist, bound_type
            return sentinel_pld

        monkeypatch.setattr(
            random_allocation_api_module,
            "allocation_directional_pld",
            fake_allocation_directional_pld,
        )
        monkeypatch.setattr(random_allocation_api_module, "compose_full_pld", fake_compose_full_pld)

        config = AllocationSchemeConfig(convolution_method=ConvolutionMethod.GEOM)
        remove_realization = _simple_realization()
        add_realization = _simple_realization()
        result = general_allocation_pld(
            num_steps=23,
            num_selected=5,
            num_epochs=4,
            remove_realization=remove_realization,
            add_realization=add_realization,
            config=config,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert result is sentinel_pld
        assert len(calls) == 2
        for call in calls:
            assert call["num_steps"] == 23
            assert call["num_selected"] == 5
            assert call["num_epochs"] == 4
            assert call["loss_discretization"] == config.loss_discretization
            assert call["tail_truncation"] == config.tail_truncation
            assert call["bound_type"] == BoundType.IS_DOMINATED

        remove_builder = calls[0]["compute_base_pld"]
        add_builder = calls[1]["compute_base_pld"]
        assert callable(remove_builder)
        assert callable(add_builder)
        assert (
            remove_builder.func is random_allocation_api_module.geometric_allocation_pld_base_remove
        )
        assert add_builder.func is random_allocation_api_module.geometric_allocation_pld_base_add
        remove_base_creation = remove_builder.keywords["base_distributions_creation"]
        add_base_creation = add_builder.keywords["base_distributions_creation"]
        assert (
            remove_base_creation.func
            is random_allocation_api_module.realization_remove_base_distributions
        )
        assert (
            add_base_creation.func is random_allocation_api_module.realization_add_base_distribution
        )
        assert remove_base_creation.keywords == {
            "realization": remove_realization,
            "max_grid_mult": config.max_grid_mult,
        }
        assert add_base_creation.keywords == {
            "realization": add_realization,
            "max_grid_mult": config.max_grid_mult,
        }
        assert (
            calls[0]["base_loss_discretization_count"]
            is random_allocation_api_module.remove_geometric_loss_discretization_count
        )
        assert (
            calls[1]["base_loss_discretization_count"]
            is random_allocation_api_module.add_geometric_loss_discretization_count
        )

    @pytest.mark.parametrize(
        "bound_type",
        [BoundType.DOMINATES, BoundType.IS_DOMINATED],
    )
    def test_realization_geometric_factors_honor_max_grid_mult(self, bound_type: BoundType):
        """Realization REMOVE and ADD factors use the shared geometric grid cap."""
        max_grid_mult = 100
        realization = PLDRealization(
            x_0=0.0,
            step=1e-3,
            prob_arr=np.full(1_001, 1.0 / 1_001),
        )

        remove_base, remove_dual = (
            random_allocation_realization_module.realization_remove_base_distributions(
                realization=realization,
                loss_discretization=1e-4,
                tail_truncation=1e-10,
                bound_type=bound_type,
                max_grid_mult=max_grid_mult,
            )
        )
        add_base = random_allocation_realization_module.realization_add_base_distribution(
            realization=realization,
            loss_discretization=1e-4,
            tail_truncation=1e-10,
            bound_type=bound_type,
            max_grid_mult=max_grid_mult,
        )

        assert remove_base.prob_arr.size <= max_grid_mult
        assert remove_dual.prob_arr.size <= max_grid_mult
        assert add_base.prob_arr.size <= max_grid_mult

    @pytest.mark.parametrize(
        "builder",
        [
            random_allocation_realization_module.realization_remove_base_distributions,
            random_allocation_realization_module.realization_add_base_distribution,
        ],
    )
    def test_realization_geometric_factors_require_two_grid_points(self, builder):
        """Realization factor builders reject a grid with no finite interval."""
        realization = PLDRealization(
            x_0=0.0,
            step=0.1,
            prob_arr=np.array([1.0]),
        )

        with pytest.raises(ValueError, match="at least two finite grid points"):
            builder(
                realization=realization,
                loss_discretization=0.1,
                tail_truncation=1e-10,
                bound_type=BoundType.DOMINATES,
            )

    def test_general_allocation_rejects_num_steps_less_than_num_selected(self):
        """General allocation rejects num steps less than num selected."""
        with pytest.raises(ValueError, match="num_selected .* cannot exceed num_steps"):
            general_allocation_pld(
                num_steps=3,
                num_selected=4,
                num_epochs=1,
                remove_realization=_simple_realization(),
                add_realization=_simple_realization(),
                config=AllocationSchemeConfig(convolution_method=ConvolutionMethod.GEOM),
            )


def test_gaussian_allocation_wires_directional_plds(
    monkeypatch: pytest.MonkeyPatch,
):
    """Gaussian allocation builds one directional PLD per direction and composes them."""
    calls: list[dict[str, Any]] = []
    sentinel_pld = object()

    def fake_allocation_directional_pld(**kwargs: Any) -> DenseDiscreteDist:
        calls.append(kwargs)
        return _stub_linear_dist()

    def fake_compose_full_pld(*, remove_dist, add_dist, bound_type):
        del remove_dist, add_dist, bound_type
        return sentinel_pld

    monkeypatch.setattr(
        random_allocation_api_module,
        "allocation_directional_pld",
        fake_allocation_directional_pld,
    )
    monkeypatch.setattr(random_allocation_api_module, "compose_full_pld", fake_compose_full_pld)

    config = AllocationSchemeConfig(convolution_method=ConvolutionMethod.FFT)
    params = PrivacyParams(
        sigma=1.75,
        num_steps=19,
        num_selected=4,
        num_epochs=3,
    )
    result = gaussian_allocation_pld(
        params=params,
        config=config,
        bound_type=BoundType.DOMINATES,
    )

    assert result is sentinel_pld
    assert len(calls) == 2
    for call, direction in zip(calls, (Direction.REMOVE, Direction.ADD)):
        assert call["num_steps"] == 19
        assert call["num_selected"] == 4
        assert call["num_epochs"] == 3
        assert call["loss_discretization"] == config.loss_discretization
        assert call["tail_truncation"] == config.tail_truncation
        assert call["bound_type"] == BoundType.DOMINATES
        builder = call["compute_base_pld"]
        # FFT config: gaussian_allocation_directional_pld wires the FFT route directly.
        assert builder.func is random_allocation_gaussian_module._gaussian_allocation_fft
        assert builder.keywords == {
            "direction": direction,
            "sigma": params.sigma,
            "config": config,
        }
        assert call["base_loss_discretization_count"](7) == 1


def test_gaussian_allocation_best_of_two_combines_full_pipelines(
    monkeypatch: pytest.MonkeyPatch,
):
    """BEST_OF_TWO runs full GEOM and FFT directional pipelines and combines at the end."""
    calls: list[dict[str, Any]] = []

    def fake_allocation_directional_pld(**kwargs: Any) -> DenseDiscreteDist:
        compute_base_pld = kwargs["compute_base_pld"]
        method = (
            ConvolutionMethod.GEOM
            if compute_base_pld.func is random_allocation_gaussian_module._gaussian_allocation_geom
            else ConvolutionMethod.FFT
        )
        calls.append(
            {
                "method": method,
                "direction": compute_base_pld.keywords["direction"],
                "count_fn": kwargs["base_loss_discretization_count"],
                "num_steps": kwargs["num_steps"],
                "num_selected": kwargs["num_selected"],
                "num_epochs": kwargs["num_epochs"],
                "loss_discretization": kwargs["loss_discretization"],
                "tail_truncation": kwargs["tail_truncation"],
                "bound_type": kwargs["bound_type"],
            }
        )
        # GEOM pipelines emit a finer grid than FFT ones.
        step = 0.25 if method == ConvolutionMethod.GEOM else 0.5
        return DenseDiscreteDist(
            x_0=0.0,
            step=step,
            prob_arr=np.array([0.5, 0.3, 0.2]),
        )

    monkeypatch.setattr(
        random_allocation_api_module,
        "allocation_directional_pld",
        fake_allocation_directional_pld,
    )

    config = AllocationSchemeConfig(convolution_method=ConvolutionMethod.BEST_OF_TWO)
    params = PrivacyParams(
        sigma=1.75,
        num_steps=19,
        num_selected=4,
        num_epochs=3,
    )
    result = gaussian_allocation_pld(
        params=params,
        config=config,
        bound_type=BoundType.DOMINATES,
    )

    # Four full pipeline runs: each pure method per direction, never BEST_OF_TWO itself.
    assert [(c["method"], c["direction"]) for c in calls] == [
        (ConvolutionMethod.GEOM, Direction.REMOVE),
        (ConvolutionMethod.FFT, Direction.REMOVE),
        (ConvolutionMethod.GEOM, Direction.ADD),
        (ConvolutionMethod.FFT, Direction.ADD),
    ]
    for call in calls:
        # Each run is an alternative over the full budget, not a composed stage.
        assert call["loss_discretization"] == config.loss_discretization
        assert call["tail_truncation"] == config.tail_truncation
        assert call["num_steps"] == 19
        assert call["num_selected"] == 4
        assert call["num_epochs"] == 3
        assert call["bound_type"] == BoundType.DOMINATES
        # Each pipeline carries its own method-appropriate scaling contract.
        if call["method"] == ConvolutionMethod.GEOM:
            if call["direction"] == Direction.REMOVE:
                expected = random_allocation_api_module.remove_geometric_loss_discretization_count
            else:
                expected = random_allocation_api_module.add_geometric_loss_discretization_count
            assert call["count_fn"] is expected
        else:
            assert call["count_fn"](7) == 1
    # The real end-level combine + compose path runs and yields a dp_accounting PLD.
    assert result.get_delta_for_epsilon(1.0) >= 0.0


def test_allocation_directional_pld_core_truncates_without_regridding(
    monkeypatch: pytest.MonkeyPatch,
):
    """Allocation directional pld core keeps the base step through composition."""
    captured: dict[str, float] = {}

    def fake_fft_self_convolve(
        *,
        dist: DenseDiscreteDist,
        num_convolutions: int,
        tail_truncation: float,
        bound_type: BoundType,
        use_direct: bool,
    ) -> DenseDiscreteDist:
        del tail_truncation, bound_type, use_direct
        captured["base_gap_at_compose"] = dist.step
        captured["num_epochs"] = float(num_convolutions)
        return dist

    def fake_compute_base_pld(
        *,
        num_steps: int,
        loss_discretization: float,
        tail_truncation: float,
        bound_type: BoundType,
    ) -> DenseDiscreteDist:
        del tail_truncation, bound_type
        captured["num_steps"] = float(num_steps)
        captured["core_loss_discretization"] = loss_discretization
        return _stub_linear_dist()

    monkeypatch.setattr(
        random_allocation_accounting_module,
        "fft_self_convolve",
        fake_fft_self_convolve,
    )

    config = AllocationSchemeConfig(
        loss_discretization=0.3,
        tail_truncation=1e-8,
        convolution_method=ConvolutionMethod.FFT,
    )
    result = allocation_directional_pld_core(
        num_steps=7,
        num_epochs=5,
        compute_base_pld=fake_compute_base_pld,
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=BoundType.DOMINATES,
    )

    expected_core_loss = config.loss_discretization / 5
    expected_step = _stub_linear_dist().step

    assert captured["num_steps"] == 7.0
    assert captured["num_epochs"] == 5.0
    assert np.isclose(
        captured["core_loss_discretization"],
        expected_core_loss,
        atol=TOL.SPACING_ATOL,
    )
    assert np.isclose(
        captured["base_gap_at_compose"],
        expected_step,
        atol=TOL.SPACING_ATOL,
    )
    assert np.isclose(result.step, expected_step, atol=TOL.SPACING_ATOL)


def test_allocation_directional_pld_warns_when_fallback_regrids(
    monkeypatch: pytest.MonkeyPatch,
):
    """Allocation directional pld warns when it must align floor/ceil grids."""

    def fake_core(**kwargs: Any) -> DenseDiscreteDist:
        step = 0.1 if kwargs["num_steps"] == 3 else 0.2
        return DenseDiscreteDist(
            x_0=0.0,
            step=step,
            prob_arr=np.array([0.5, 0.5]),
        )

    def fake_rediscretize_dist(**kwargs: Any) -> DenseDiscreteDist:
        dist = kwargs["dist"]
        return DenseDiscreteDist(
            x_0=dist.x_0,
            step=kwargs["loss_discretization"],
            prob_arr=dist.prob_arr,
        )

    def fake_fft_convolve(
        *,
        dist_1: DenseDiscreteDist,
        dist_2: DenseDiscreteDist,
        tail_truncation: float,
        bound_type: BoundType,
    ) -> DenseDiscreteDist:
        del tail_truncation, bound_type
        assert np.isclose(dist_1.step, dist_2.step)
        return dist_1

    monkeypatch.setattr(
        random_allocation_accounting_module,
        "_allocation_directional_pld_core",
        fake_core,
    )
    monkeypatch.setattr(
        random_allocation_accounting_module,
        "rediscretize_dist_by_bound",
        fake_rediscretize_dist,
    )
    monkeypatch.setattr(
        random_allocation_accounting_module,
        "fft_convolve",
        fake_fft_convolve,
    )

    with pytest.warns(
        UserWarning,
        match="allocation_directional_pld: aligning mismatched floor/ceil grids",
    ):
        random_allocation_accounting_module.allocation_directional_pld(
            compute_base_pld=lambda **_kwargs: _stub_linear_dist(),
            base_loss_discretization_count=lambda _num_steps: 1,
            num_steps=7,
            num_selected=2,
            num_epochs=2,
            loss_discretization=0.4,
            tail_truncation=1e-8,
            bound_type=BoundType.DOMINATES,
        )


def test_gaussian_allocation_warns_for_capped_grid_regrid():
    """A deliberately capped GEOM run should surface the fallback re-grid warning."""
    params = PrivacyParams(
        sigma=1.0,
        num_steps=5,
        num_selected=2,
        num_epochs=1,
        delta=1e-5,
    )
    config = AllocationSchemeConfig(
        loss_discretization=0.01,
        tail_truncation=0.1,
        max_grid_mult=100,
        convolution_method=ConvolutionMethod.GEOM,
    )

    with pytest.warns(
        UserWarning,
        match="allocation_directional_pld: aligning mismatched floor/ceil grids",
    ):
        pld = gaussian_allocation_pld(
            params=params,
            config=config,
            bound_type=BoundType.DOMINATES,
        )

    assert pld.get_epsilon_for_delta(params.delta) > 0.0


def test_gaussian_allocation_uncapped_nondivisible_avoids_regrid_warning():
    """Uncapped GEOM floor/ceil components should align without fallback re-grid."""
    params = PrivacyParams(
        sigma=2.0,
        num_steps=5,
        num_selected=3,
        delta=1e-3,
    )
    config = AllocationSchemeConfig(
        loss_discretization=0.1,
        tail_truncation=1e-4,
        convolution_method=ConvolutionMethod.GEOM,
    )

    pld = gaussian_allocation_pld(
        params=params,
        config=config,
        bound_type=BoundType.DOMINATES,
    )

    assert pld.get_epsilon_for_delta(params.delta) > 0.0


def test_geom_is_dominated_path_handles_tiny_nonpositive_exp_tail():
    # The FFT REMOVE route only supports BoundType.DOMINATES; IS_DOMINATED
    # uses the GEOM route. This regression guards against the case where
    # IS_DOMINATED produces a non-finite or non-positive epsilon.
    """Geom is dominated path handles tiny nonpositive exp tail."""
    params = PrivacyParams(
        sigma=2.0,
        num_steps=5,
        num_selected=1,
        num_epochs=1,
        delta=1e-5,
    )
    config = AllocationSchemeConfig(
        loss_discretization=5e-3,
        tail_truncation=1e-10,
        convolution_method=ConvolutionMethod.GEOM,
    )

    pld = gaussian_allocation_pld(
        params=params,
        config=config,
        bound_type=BoundType.IS_DOMINATED,
    )

    epsilon = float(pld.get_epsilon_for_delta(params.delta))
    assert np.isfinite(epsilon)
    assert epsilon > 0.0


def test_gaussian_remove_geom_dominates_discretizes_only_primary(monkeypatch):
    """Gaussian upper REMOVE derives its transformed dual after one discretization."""
    calls = []
    actual_discretize = random_allocation_gaussian_module.discretize_continuous_ctd

    def recording_discretize(**kwargs):
        calls.append(kwargs)
        return actual_discretize(**kwargs)

    monkeypatch.setattr(
        random_allocation_gaussian_module,
        "discretize_continuous_ctd",
        recording_discretize,
    )

    base, neg_dual = random_allocation_gaussian_module._gaussian_remove_geom_loss_factors(
        loss_discretization=0.1,
        tail_truncation=1e-6,
        bound_type=BoundType.DOMINATES,
        sigma=2.0,
        config=AllocationSchemeConfig(
            loss_discretization=0.1,
            tail_truncation=1e-6,
            convolution_method=ConvolutionMethod.GEOM,
        ),
    )
    expected = random_allocation_gaussian_module.negate_reverse_linear_distribution(
        random_allocation_gaussian_module.calc_pld_dual(base)
    )

    assert len(calls) == 1
    assert "dual_dist" in calls[0]
    assert isinstance(base, PLDRealization)
    assert np.isclose(base.step, neg_dual.step, atol=TOL.SPACING_ATOL)
    np.testing.assert_array_equal(neg_dual.x_array, expected.x_array)
    np.testing.assert_array_equal(neg_dual.prob_arr, expected.prob_arr)
    assert neg_dual.p_min == expected.p_min
    assert neg_dual.p_max == expected.p_max


@pytest.mark.parametrize("sigma", [0.05, 0.15])
def test_gaussian_remove_geom_warns_when_negative_dual_mean_lacks_grid_margin(sigma: float):
    """Gaussian REMOVE requires one standard deviation around its negative-dual mean."""
    config = AllocationSchemeConfig()

    with pytest.warns(
        RuntimeWarning,
        match="negative-dual mean is not at least one standard deviation inside",
    ):
        base, _ = random_allocation_gaussian_module._gaussian_remove_geom_loss_factors(
            loss_discretization=config.loss_discretization,
            tail_truncation=config.tail_truncation,
            bound_type=BoundType.DOMINATES,
            sigma=sigma,
            config=config,
        )

    negative_dual_mean = -0.5 / sigma**2
    negative_dual_std = 1.0 / sigma
    assert negative_dual_mean < base.x_array[0] + negative_dual_std


def test_gaussian_remove_geom_dominates_honors_max_grid_mult():
    """Gaussian upper REMOVE coarsens its single primary grid to the configured cap."""
    requested_step = 1e-4
    max_grid_mult = 100

    base, neg_dual = random_allocation_gaussian_module._gaussian_remove_geom_loss_factors(
        loss_discretization=requested_step,
        tail_truncation=1e-10,
        bound_type=BoundType.DOMINATES,
        sigma=1.0,
        config=AllocationSchemeConfig(
            loss_discretization=requested_step,
            tail_truncation=1e-10,
            max_grid_mult=max_grid_mult,
            convolution_method=ConvolutionMethod.GEOM,
        ),
    )

    assert base.prob_arr.size <= max_grid_mult
    assert base.step > requested_step
    assert neg_dual.prob_arr.size == base.prob_arr.size
    assert neg_dual.step == base.step


def test_gaussian_remove_geom_is_dominated_keeps_two_discretizations(monkeypatch):
    """Gaussian lower REMOVE retains the continuous dual-first construction."""
    calls = 0
    actual_discretize = random_allocation_gaussian_module.discretize_continuous_stoch_dom

    def recording_discretize(**kwargs):
        nonlocal calls
        calls += 1
        return actual_discretize(**kwargs)

    monkeypatch.setattr(
        random_allocation_gaussian_module,
        "discretize_continuous_stoch_dom",
        recording_discretize,
    )

    random_allocation_gaussian_module._gaussian_remove_geom_loss_factors(
        loss_discretization=0.1,
        tail_truncation=1e-6,
        bound_type=BoundType.IS_DOMINATED,
        sigma=2.0,
        config=AllocationSchemeConfig(
            loss_discretization=0.1,
            tail_truncation=1e-6,
            convolution_method=ConvolutionMethod.GEOM,
        ),
    )

    assert calls == 2


@pytest.mark.parametrize(
    ("bound_type", "function_name"),
    [
        (BoundType.DOMINATES, "discretize_continuous_ctd"),
        (BoundType.IS_DOMINATED, "discretize_continuous_stoch_dom"),
    ],
)
def test_gaussian_add_geom_discretizes_in_linear_loss_space(
    monkeypatch, bound_type: BoundType, function_name: str
):
    """Gaussian ADD chooses its fixed engine from the requested bound."""
    calls = []
    actual_discretize = getattr(random_allocation_gaussian_module, function_name)

    def recording_discretize(**kwargs):
        calls.append(kwargs)
        return actual_discretize(**kwargs)

    monkeypatch.setattr(
        random_allocation_gaussian_module,
        function_name,
        recording_discretize,
    )
    result = random_allocation_gaussian_module._gaussian_add_geom_loss_factor(
        loss_discretization=0.1,
        tail_truncation=1e-6,
        bound_type=bound_type,
        sigma=2.0,
        config=AllocationSchemeConfig(
            loss_discretization=0.1,
            tail_truncation=1e-6,
            convolution_method=ConvolutionMethod.GEOM,
        ),
    )

    assert len(calls) == 1
    assert ("dual_dist" in calls[0]) == (bound_type == BoundType.DOMINATES)
    assert result.spacing_type == SpacingType.LINEAR


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_gaussian_geom_ctd_handles_upper_and_lower_paths(bound_type: BoundType):
    """GEOM uses CtD for upper factors and stochastic projection for lower factors."""
    params = PrivacyParams(
        sigma=2.0,
        num_steps=5,
        num_selected=3,
        num_epochs=1,
        delta=1e-3,
    )
    config = AllocationSchemeConfig(
        loss_discretization=0.1,
        tail_truncation=1e-4,
        convolution_method=ConvolutionMethod.GEOM,
    )

    pld = gaussian_allocation_pld(params=params, config=config, bound_type=bound_type)
    epsilon = float(pld.get_epsilon_for_delta(params.delta))

    assert np.isfinite(epsilon)
    assert epsilon > 0.0


class TestGeometricBaseTailScaling:
    """Tests how tail truncation is threaded into geometric base construction."""

    def test_remove_base_factor_tail_scales_with_num_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Remove base factor tail scales with num steps."""
        captured_tails: list[float] = []
        sentinel = object()

        def fake_base_distributions_creation(
            *,
            loss_discretization: float,
            tail_truncation: float,
            bound_type: BoundType,
        ) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
            del loss_discretization, bound_type
            captured_tails.append(tail_truncation)
            return _stub_linear_dist(), _stub_linear_dist()

        monkeypatch.setattr(
            random_allocation_accounting_module, "exp_linear_to_geometric", lambda _dist: sentinel
        )
        monkeypatch.setattr(
            random_allocation_accounting_module,
            "geometric_self_convolve",
            lambda **_kwargs: sentinel,
        )
        monkeypatch.setattr(
            random_allocation_accounting_module, "geometric_convolve", lambda **_kwargs: sentinel
        )
        monkeypatch.setattr(
            random_allocation_accounting_module,
            "log_geometric_to_linear",
            lambda _dist: _stub_linear_dist(),
        )

        for num_steps in (5, 10):
            random_allocation_accounting_module.geometric_allocation_pld_base_remove(
                base_distributions_creation=fake_base_distributions_creation,
                num_steps=num_steps,
                loss_discretization=0.1,
                tail_truncation=1e-6,
                bound_type=BoundType.DOMINATES,
            )

        assert len(captured_tails) == 2
        assert np.isclose(
            captured_tails[0] * 5, captured_tails[1] * 10, atol=TOL.TAIL_LINEAR_RELATION_ATOL
        )

    def test_add_base_factor_tail_scales_with_num_steps(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Add base factor tail scales with num steps."""
        captured_tails: list[float] = []
        sentinel = object()

        def fake_base_distributions_creation(
            *,
            loss_discretization: float,
            tail_truncation: float,
            bound_type: BoundType,
        ) -> DenseDiscreteDist:
            del loss_discretization, bound_type
            captured_tails.append(tail_truncation)
            return _stub_linear_dist()

        monkeypatch.setattr(
            random_allocation_accounting_module, "exp_linear_to_geometric", lambda _dist: sentinel
        )
        monkeypatch.setattr(
            random_allocation_accounting_module,
            "geometric_self_convolve",
            lambda **_kwargs: sentinel,
        )
        monkeypatch.setattr(
            random_allocation_accounting_module,
            "log_geometric_to_linear",
            lambda _dist: _stub_linear_dist(),
        )
        monkeypatch.setattr(
            random_allocation_accounting_module,
            "negate_reverse_linear_distribution",
            lambda _dist: _stub_linear_dist(),
        )

        for num_steps in (5, 10):
            random_allocation_accounting_module.geometric_allocation_pld_base_add(
                base_distributions_creation=fake_base_distributions_creation,
                num_steps=num_steps,
                loss_discretization=0.1,
                tail_truncation=1e-6,
                bound_type=BoundType.DOMINATES,
            )

        assert len(captured_tails) == 2
        assert np.isclose(
            captured_tails[0] * 5, captured_tails[1] * 10, atol=TOL.TAIL_LINEAR_RELATION_ATOL
        )


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_geometric_allocation_preserves_integer_loss_lattice(bound_type: BoundType):
    """Anchored ADD and REMOVE composition return grids aligned to zero loss."""
    num_steps = 7

    remove = random_allocation_accounting_module.geometric_allocation_pld_base_remove(
        base_distributions_creation=lambda loss_discretization, **_kwargs: (
            _aligned_base_dist(loss_discretization, -2),
            _aligned_base_dist(loss_discretization, -1),
        ),
        num_steps=num_steps,
        loss_discretization=0.1,
        tail_truncation=1e-6,
        bound_type=bound_type,
    )
    add = random_allocation_accounting_module.geometric_allocation_pld_base_add(
        base_distributions_creation=lambda loss_discretization, **_kwargs: _aligned_base_dist(
            loss_discretization, -2
        ),
        num_steps=num_steps,
        loss_discretization=0.1,
        tail_truncation=1e-6,
        bound_type=bound_type,
    )

    for dist in (remove, add):
        lower_index = round(dist.x_0 / dist.step)
        assert np.isclose(
            dist.x_0,
            lower_index * dist.step,
            atol=TOL.SPACING_ATOL,
        )
