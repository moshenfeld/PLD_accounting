"""Unit tests for realization-path composition semantics."""

from __future__ import annotations

import numpy as np
import pytest

import PLD_accounting.random_allocation_api as random_allocation_api_module
import PLD_accounting.random_allocation_accounting as random_allocation_accounting_module
from PLD_accounting.discrete_dist import LinearDiscreteDist, PLDRealization
from PLD_accounting.random_allocation_accounting import (
    decompose_allocation_compositions,
    finalize_allocation_composition,
)
from PLD_accounting.random_allocation_api import general_allocation_PLD
from PLD_accounting.types import AllocationSchemeConfig, BoundType, ConvolutionMethod, Direction


def _simple_realization() -> PLDRealization:
    return PLDRealization(
        x_min=0.0,
        x_gap=0.5,
        PMF_array=np.array([0.6, 0.3, 0.1]),
    )


def _stub_linear_dist() -> LinearDiscreteDist:
    return LinearDiscreteDist(
        x_min=0.0,
        x_gap=0.5,
        PMF_array=np.array([0.5, 0.3, 0.2]),
    )


class TestGeneralAllocationCompositionDecomposition:
    @pytest.mark.parametrize(
        ("num_steps", "num_selected", "num_epochs", "expected_inner", "expected_outer"),
        [
            (10, 1, 1, 10, 1),
            (10, 5, 3, 2, 15),
            (9, 2, 4, 4, 8),
            (17, 4, 2, 4, 8),
        ],
    )
    def test_decompose_general_allocation_compositions(
        self,
        num_steps: int,
        num_selected: int,
        num_epochs: int,
        expected_inner: int,
        expected_outer: int,
    ):
        inner, outer = decompose_allocation_compositions(
            num_steps=num_steps,
            num_selected=num_selected,
            num_epochs=num_epochs,
        )
        assert inner == expected_inner
        assert outer == expected_outer

    def test_decompose_rejects_num_steps_less_than_num_selected(self):
        with pytest.raises(ValueError, match="num_steps must be >= num_selected"):
            decompose_allocation_compositions(
                num_steps=3,
                num_selected=4,
                num_epochs=1,
            )


class TestGeneralAllocationPLDWiring:
    def test_general_allocation_uses_two_stage_composition(self, monkeypatch: pytest.MonkeyPatch):
        inner_calls: list[tuple[Direction, int, int]] = []
        captured: dict[str, object] = {}

        remove_round_dist = _stub_linear_dist()
        add_round_dist = _stub_linear_dist()

        def fake_allocation_PMF_from_realization(
            *,
            realization: PLDRealization,
            direction: Direction,
            num_steps_per_round: int,
            num_rounds: int,
            config: AllocationSchemeConfig,
            bound_type: BoundType,
        ) -> LinearDiscreteDist:
            del realization, config, bound_type
            inner_calls.append((direction, num_steps_per_round, num_rounds))
            return remove_round_dist if direction == Direction.REMOVE else add_round_dist

        sentinel_pld = object()

        def fake_compose_pld_from_pmfs(
            *,
            remove_dist: LinearDiscreteDist,
            add_dist: LinearDiscreteDist,
            pessimistic_estimate: bool,
        ) -> object:
            captured["final_remove"] = remove_dist
            captured["final_add"] = add_dist
            captured["pessimistic"] = pessimistic_estimate
            return sentinel_pld

        monkeypatch.setattr(
            random_allocation_api_module,
            "allocation_PMF_from_realization",
            fake_allocation_PMF_from_realization,
        )
        monkeypatch.setattr(
            random_allocation_api_module,
            "compose_pld_from_pmfs",
            fake_compose_pld_from_pmfs,
        )

        config = AllocationSchemeConfig(convolution_method=ConvolutionMethod.GEOM)
        result = general_allocation_PLD(
            num_steps=23,
            num_selected=5,
            num_epochs=4,
            remove_realization=_simple_realization(),
            add_realization=_simple_realization(),
            config=config,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert result is sentinel_pld
        assert inner_calls == [
            (Direction.REMOVE, 4, 20),
            (Direction.ADD, 4, 20),
        ]

        assert captured["final_remove"] is remove_round_dist
        assert captured["final_add"] is add_round_dist
        assert captured["pessimistic"] is False

    def test_general_allocation_rejects_num_steps_less_than_num_selected(self):
        with pytest.raises(ValueError, match="num_steps must be >= num_selected"):
            general_allocation_PLD(
                num_steps=3,
                num_selected=4,
                num_epochs=1,
                remove_realization=_simple_realization(),
                add_realization=_simple_realization(),
                config=AllocationSchemeConfig(convolution_method=ConvolutionMethod.GEOM),
            )


class TestAllocationFinalization:
    def test_finalize_allocation_composition_regrids_before_and_after_compose(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ):
        captured: dict[str, float] = {}

        def fake_compose_allocation_pmfs(
            *,
            round_dist: LinearDiscreteDist,
            num_rounds: int,
            tail_truncation: float,
            bound_type: BoundType,
        ) -> LinearDiscreteDist:
            del num_rounds, tail_truncation, bound_type
            captured["round_gap_at_compose"] = round_dist.x_gap
            return round_dist

        monkeypatch.setattr(
            random_allocation_accounting_module,
            "compose_allocation_pmfs",
            fake_compose_allocation_pmfs,
        )

        result = finalize_allocation_composition(
            round_dist=_stub_linear_dist(),
            num_rounds=5,
            pre_composition_loss_discretization=0.1,
            pre_composition_tail_truncation=1e-8,
            output_loss_discretization=0.2,
            output_tail_truncation=1e-8,
            bound_type=BoundType.DOMINATES,
        )

        assert np.isclose(captured["round_gap_at_compose"], 0.1, atol=1e-12)
        assert np.isclose(result.x_gap, 0.2, atol=1e-12)
