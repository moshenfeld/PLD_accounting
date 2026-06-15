"""Unit tests for random-allocation composition wiring."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import PLD_accounting.random_allocation_accounting as random_allocation_accounting_module
import PLD_accounting.random_allocation_api as random_allocation_api_module
import PLD_accounting.random_allocation_gaussian as random_allocation_gaussian_module
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


class TestGeneralAllocationWiring:
    """Tests that general (geometric-base) allocation delegates to shared helpers."""

    def test_general_allocation_uses_directional_plds(self, monkeypatch: pytest.MonkeyPatch):
        """General allocation builds one directional PLD per direction and composes them."""
        calls: list[dict[str, Any]] = []
        sentinel_pld = object()

        def fake_allocation_directional_pld(
            *,
            compute_base_pld,
            base_loss_discretization_count,
            num_steps: int,
            num_selected: int,
            num_epochs: int,
            loss_discretization: float,
            tail_truncation: float,
            bound_type: BoundType,
        ) -> DenseDiscreteDist:
            calls.append(
                {
                    "compute_base_pld": compute_base_pld,
                    "base_loss_discretization_count": base_loss_discretization_count,
                    "num_steps": num_steps,
                    "num_selected": num_selected,
                    "num_epochs": num_epochs,
                    "loss_discretization": loss_discretization,
                    "tail_truncation": tail_truncation,
                    "bound_type": bound_type,
                }
            )
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
        assert remove_base_creation.keywords == {"realization": remove_realization}
        assert add_base_creation.keywords == {"realization": add_realization}
        assert (
            calls[0]["base_loss_discretization_count"]
            is random_allocation_api_module.remove_geometric_loss_discretization_count
        )
        assert (
            calls[1]["base_loss_discretization_count"]
            is random_allocation_api_module.add_geometric_loss_discretization_count
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

    def fake_allocation_directional_pld(
        *,
        compute_base_pld,
        base_loss_discretization_count,
        num_steps: int,
        num_selected: int,
        num_epochs: int,
        loss_discretization: float,
        tail_truncation: float,
        bound_type: BoundType,
    ) -> DenseDiscreteDist:
        calls.append(
            {
                "compute_base_pld": compute_base_pld,
                "base_loss_discretization_count": base_loss_discretization_count,
                "num_steps": num_steps,
                "num_selected": num_selected,
                "num_epochs": num_epochs,
                "loss_discretization": loss_discretization,
                "tail_truncation": tail_truncation,
                "bound_type": bound_type,
            }
        )
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

    def fake_allocation_directional_pld(
        *,
        compute_base_pld,
        base_loss_discretization_count,
        num_steps: int,
        num_selected: int,
        num_epochs: int,
        loss_discretization: float,
        tail_truncation: float,
        bound_type: BoundType,
    ) -> DenseDiscreteDist:
        method = (
            ConvolutionMethod.GEOM
            if compute_base_pld.func is random_allocation_gaussian_module._gaussian_allocation_geom
            else ConvolutionMethod.FFT
        )
        calls.append(
            {
                "method": method,
                "direction": compute_base_pld.keywords["direction"],
                "count_fn": base_loss_discretization_count,
                "num_steps": num_steps,
                "num_selected": num_selected,
                "num_epochs": num_epochs,
                "loss_discretization": loss_discretization,
                "tail_truncation": tail_truncation,
                "bound_type": bound_type,
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
        T: int,
        tail_truncation: float,
        bound_type: BoundType,
        use_direct: bool,
    ) -> DenseDiscreteDist:
        del tail_truncation, bound_type, use_direct
        captured["base_gap_at_compose"] = dist.step
        captured["num_epochs"] = float(T)
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

    def fake_rediscretize_dist(
        *,
        dist: DenseDiscreteDist,
        tail_truncation: float,
        loss_discretization: float,
        spacing_type,
        bound_type: BoundType,
    ) -> DenseDiscreteDist:
        del tail_truncation, spacing_type, bound_type
        return DenseDiscreteDist(
            x_0=dist.x_0,
            step=loss_discretization,
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
        "rediscretize_dist",
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
