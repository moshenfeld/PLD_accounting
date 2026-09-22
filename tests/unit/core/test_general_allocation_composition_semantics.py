"""Unit tests for random-allocation composition wiring."""

# pylint: disable=too-many-lines

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import PLD_accounting.random_allocation_accounting as random_allocation_accounting_module
import PLD_accounting.random_allocation_api as random_allocation_api_module
import PLD_accounting.random_allocation_gaussian as random_allocation_gaussian_module
import PLD_accounting.random_allocation_realization as random_allocation_realization_module
from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    PLDRealization,
)
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


def _core_composition_calls(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
) -> list[tuple[int, int]]:
    """Return ``(num_steps, num_epochs)`` passed to each core component build."""
    calls: list[tuple[int, int]] = []

    def fake_core(**kwargs: Any) -> DenseDiscreteDist:
        calls.append((kwargs["num_steps"], kwargs["num_epochs"]))
        return _stub_linear_dist()

    monkeypatch.setattr(
        random_allocation_accounting_module,
        "_allocation_directional_pld_core",
        fake_core,
    )
    random_allocation_accounting_module.allocation_directional_pld(
        compute_base_pld=lambda **_kwargs: _stub_linear_dist(),
        base_loss_discretization_count=lambda _n: 1,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        loss_discretization=0.1,
        tail_truncation=1e-10,
        bound_type=BoundType.DOMINATES,
    )
    return calls


class TestAllocationCompositionCounts:
    """Integer lock tests for the README Parameter Mapping split."""

    def test_divisible_split(self, monkeypatch: pytest.MonkeyPatch):
        """Divisible ``num_steps`` uses only the floor branch."""
        assert _core_composition_calls(monkeypatch, num_steps=10, num_selected=2, num_epochs=3) == [
            (5, 6)
        ]

    def test_remainder_split(self, monkeypatch: pytest.MonkeyPatch):
        """A nonzero remainder splits epochs across floor and ceil branches."""
        assert _core_composition_calls(monkeypatch, num_steps=11, num_selected=2, num_epochs=3) == [
            (5, 3),
            (6, 3),
        ]

    def test_num_selected_one(self, monkeypatch: pytest.MonkeyPatch):
        """``num_selected=1`` is a single ceil-empty floor composition."""
        assert _core_composition_calls(monkeypatch, num_steps=17, num_selected=1, num_epochs=4) == [
            (17, 4)
        ]


def _simple_realization() -> PLDRealization:
    return PLDRealization(
        grid=GridSpec(step=0.5, n=3, anchor=0.0), prob_arr=np.array([0.6, 0.3, 0.1])
    )


def _summed_anchor_stub(**kwargs: Any) -> DenseDiscreteDist:
    """Identity convolution that still sums the two multiplicative anchors."""
    dist, other = kwargs["dist_1"], kwargs["dist_2"]
    return DenseDiscreteDist(
        grid=replace(dist.grid, anchor=dist.grid.anchor + other.grid.anchor),
        prob_arr=dist.prob_arr,
        p_min=dist.p_min,
        p_max=dist.p_max,
        domain=dist.domain,
    )


def _self_convolved_stub(**kwargs: Any) -> DenseDiscreteDist:
    """Identity convolution that still accumulates the multiplicative anchor."""
    dist = kwargs["dist"]
    return DenseDiscreteDist(
        grid=replace(dist.grid, anchor=dist.grid.anchor * kwargs["num_convolutions"]),
        prob_arr=dist.prob_arr,
        p_min=dist.p_min,
        p_max=dist.p_max,
        domain=dist.domain,
    )


def _stub_linear_dist() -> DenseDiscreteDist:
    return DenseDiscreteDist(
        grid=GridSpec(step=0.5, n=3, anchor=0.0), prob_arr=np.array([0.5, 0.3, 0.2])
    )


def _aligned_base_dist(step: float, origin_index: int) -> DenseDiscreteDist:
    """Build a small loss distribution on exact integer multiples of ``step``."""
    return DenseDiscreteDist(
        grid=GridSpec(step=step, n=4, index_0=origin_index),
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
            grid=GridSpec(step=1e-3, n=np.full(1_001, 1.0 / 1_001).size, anchor=0.0),
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
            grid=GridSpec(step=0.1, n=1, anchor=0.0), prob_arr=np.array([1.0])
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
            grid=GridSpec(step=step, n=3, anchor=0.0), prob_arr=np.array([0.5, 0.3, 0.2])
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


def test_gaussian_fft_add_folds_zero_atom_before_actual_self_convolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADD includes zero-plus-finite cross terms in the FFT array convolution."""
    source = DenseDiscreteDist(
        grid=GridSpec(step=1.0, n=2, anchor=1.0),
        prob_arr=np.array([0.5, 0.3]),
        p_min=0.2,
        domain=Domain.POSITIVES,
    )
    captured: dict[str, DenseDiscreteDist] = {}

    def fake_discretize(**kwargs: Any) -> DenseDiscreteDist:
        assert kwargs["bound_type"] == BoundType.IS_DOMINATED
        assert kwargs["align_to_multiples"] is False
        assert kwargs["domain"] == Domain.POSITIVES
        return source

    def capture_rediscretize(**kwargs: Any) -> DenseDiscreteDist:
        conv_dist = kwargs["dist"]
        captured["conv_dist"] = conv_dist
        return DenseDiscreteDist(
            grid=GridSpec.geometric(ratio=np.e, n=conv_dist.prob_arr.size, anchor=1.0),
            prob_arr=conv_dist.prob_arr,
            p_min=conv_dist.p_min,
            p_max=conv_dist.p_max,
            domain=Domain.POSITIVES,
        )

    monkeypatch.setattr(
        random_allocation_gaussian_module,
        "discretize_continuous_stoch_dom",
        fake_discretize,
    )
    monkeypatch.setattr(
        random_allocation_gaussian_module,
        "rediscretize_dist_stoch_dom",
        capture_rediscretize,
    )

    random_allocation_gaussian_module._gaussian_allocation_fft_add(
        num_steps=2,
        loss_discretization=0.1,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        sigma_inv=0.5,
        single_step_tail_truncation=1e-6,
        single_step_n_grid=128,
    )

    conv_dist = captured["conv_dist"]
    assert conv_dist.domain == Domain.POSITIVES
    assert conv_dist.x_0 == 1.0
    assert conv_dist.p_min == pytest.approx(0.04)
    np.testing.assert_allclose(conv_dist.prob_arr, np.array([0.20, 0.37, 0.30, 0.09]))


def test_gaussian_fft_add_embeds_boundary_at_nonpositive_cell_for_wide_offset() -> None:
    """The prepended-cell count puts the zero atom at or below zero."""
    source = DenseDiscreteDist(
        grid=GridSpec(step=1.0, n=1, anchor=2.5),
        prob_arr=np.array([0.7]),
        p_min=0.3,
        domain=Domain.POSITIVES,
    )

    result = random_allocation_gaussian_module._embed_positive_boundary_on_nonpositive_real_cell(
        source
    )

    assert result.domain == Domain.REALS
    assert result.x_0 == -0.5
    assert result.p_min == 0.0
    np.testing.assert_array_equal(result.prob_arr, np.array([0.3, 0.0, 0.0, 0.7]))
    np.testing.assert_array_equal(result.x_array[3:], source.x_array)


def test_gaussian_fft_add_embed_pads_when_x_0_is_swallowed_by_step() -> None:
    """A cancellation-prone origin still occupies a new nonpositive REALS cell."""
    p_min = 1e-13
    prob_arr = np.array([0.999999, 7.0e-07, 3.0e-07])
    prob_arr *= (1.0 - p_min) / prob_arr.sum()
    source = DenseDiscreteDist(
        grid=GridSpec(step=11820.764697407296, n=prob_arr.size, anchor=1.2644809843750233e-14),
        prob_arr=prob_arr,
        p_min=p_min,
        domain=Domain.POSITIVES,
    )

    result = random_allocation_gaussian_module._embed_positive_boundary_on_nonpositive_real_cell(
        source
    )

    n_pad = result.prob_arr.size - source.prob_arr.size
    assert n_pad >= 1
    assert result.domain == Domain.REALS
    assert result.x_0 <= 0.0
    assert result.p_min == 0.0
    np.testing.assert_array_equal(result.x_array[n_pad:], source.x_array)
    np.testing.assert_array_equal(result.prob_arr[n_pad:], source.prob_arr)
    assert result.prob_arr[0] == source.p_min


def test_gaussian_fft_add_embed_corrects_an_under_counted_ceil() -> None:
    """When fl(k * step) rounds low, ceil leaves a positive origin and one more cell is taken."""
    # ceil(x_0 / step) is 5 here, but pad(left=5) lands at +9.09e-13 rather than at or
    # below zero, because 5 * step rounds below the real product.
    step = 1309.374640277622
    source = DenseDiscreteDist(
        grid=GridSpec(step=step, n=3, anchor=6546.87320138811, index_0=0),
        prob_arr=np.array([0.5, 0.3, 0.1]),
        p_min=0.1,
        domain=Domain.POSITIVES,
    )
    assert source.grid.pad(left=5, right=0).x_0 > 0.0

    result = random_allocation_gaussian_module._embed_positive_boundary_on_nonpositive_real_cell(
        source
    )

    n_pad = result.prob_arr.size - source.prob_arr.size
    assert n_pad == 6
    assert result.x_0 <= 0.0
    assert result.p_min == 0.0
    assert result.prob_arr[0] == source.p_min
    np.testing.assert_array_equal(result.x_array[n_pad:], source.x_array)


def test_gaussian_fft_add_folds_all_nonpositive_cells_to_zero_boundary() -> None:
    """Post-processing preserves mass while tightening nonpositive artifacts to zero."""
    source = DenseDiscreteDist(
        grid=GridSpec(step=1.0, n=4, anchor=-1.0),
        prob_arr=np.array([0.1, 0.2, 0.3, 0.35]),
        p_min=0.05,
        domain=Domain.REALS,
    )

    result = random_allocation_gaussian_module._fold_nonpositive_real_mass_to_positive_boundary(
        source
    )

    assert result.domain == Domain.POSITIVES
    assert result.x_0 == 1.0
    assert result.p_min == pytest.approx(0.35)
    np.testing.assert_array_equal(result.prob_arr, np.array([0.3, 0.35]))


def test_gaussian_fft_add_unaligned_positive_grid_respects_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The unaligned positive ADD grid stays within its configured cap."""
    captured: dict[str, DenseDiscreteDist] = {}

    def fake_fft_self_convolve(**kwargs: Any) -> DenseDiscreteDist:
        captured["dist"] = kwargs["dist"]
        return kwargs["dist"]

    monkeypatch.setattr(
        random_allocation_gaussian_module,
        "fft_self_convolve",
        fake_fft_self_convolve,
    )

    random_allocation_gaussian_module._gaussian_allocation_fft_add(
        num_steps=5,
        loss_discretization=1e-3,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        sigma_inv=0.5,
        single_step_tail_truncation=1e-8,
        single_step_n_grid=128,
    )

    fft_input = captured["dist"]
    assert fft_input.domain == Domain.REALS
    assert fft_input.x_0 <= 0.0
    assert fft_input.p_min == 0.0
    assert fft_input.prob_arr[0] == pytest.approx(1e-8)
    assert fft_input.prob_arr.size <= 129


def test_gaussian_fft_add_real_convolution_keeps_zero_boundary_within_budget() -> None:
    """The restored positive zero boundary remains within the tail budget."""
    tail_truncation = 2.2222222222222222e-14
    result = random_allocation_gaussian_module._gaussian_allocation_fft_add(
        num_steps=2,
        loss_discretization=0.004,
        tail_truncation=tail_truncation,
        bound_type=BoundType.DOMINATES,
        sigma_inv=1.0,
        single_step_tail_truncation=tail_truncation / 2,
        single_step_n_grid=25_000,
    )

    # The single-step lower tail participates in FFT composition on a real cell.
    assert result.p_max < 1e-12


def test_gaussian_fft_remove_uses_aligned_real_factors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REMOVE convolves base and exp(negative dual) on one real-domain lattice."""
    actual_discretize = random_allocation_gaussian_module.discretize_continuous_stoch_dom
    calls: list[tuple[dict[str, Any], DenseDiscreteDist]] = []

    def recording_discretize(**kwargs: Any) -> DenseDiscreteDist:
        result = actual_discretize(**kwargs)
        calls.append((kwargs, result))
        return result

    monkeypatch.setattr(
        random_allocation_gaussian_module,
        "discretize_continuous_stoch_dom",
        recording_discretize,
    )

    random_allocation_gaussian_module._gaussian_allocation_fft_remove(
        num_steps=5,
        loss_discretization=1e-3,
        tail_truncation=1e-6,
        bound_type=BoundType.DOMINATES,
        sigma_inv=0.5,
        single_step_tail_truncation=1e-7,
        single_step_n_grid=128,
    )

    assert len(calls) == 2
    steps = {kwargs["step"] for kwargs, _ in calls}
    assert len(steps) == 1
    for kwargs, dist in calls:
        assert kwargs["align_to_multiples"] is True
        assert kwargs["domain"] == Domain.REALS
        assert dist.domain == Domain.REALS
        assert dist.x_0 >= 0.0
        assert dist.x_0 / dist.step == pytest.approx(round(dist.x_0 / dist.step))


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
        atol=TOL.GRID_ATOL,
    )
    assert np.isclose(
        captured["base_gap_at_compose"],
        expected_step,
        atol=TOL.GRID_ATOL,
    )
    assert np.isclose(result.step, expected_step, atol=TOL.GRID_ATOL)


def test_allocation_directional_pld_warns_when_fallback_regrids(
    monkeypatch: pytest.MonkeyPatch,
):
    """Allocation directional pld warns when it must align floor/ceil grids."""

    def fake_core(**kwargs: Any) -> DenseDiscreteDist:
        step = 0.1 if kwargs["num_steps"] == 3 else 0.2
        return DenseDiscreteDist(
            grid=GridSpec(step=step, n=2, anchor=0.0), prob_arr=np.array([0.5, 0.5])
        )

    def fake_rediscretize_dist(**kwargs: Any) -> DenseDiscreteDist:
        dist = kwargs["dist"]
        return DenseDiscreteDist(
            grid=GridSpec(
                step=kwargs["loss_discretization"], n=dist.prob_arr.size, anchor=dist.x_0
            ),
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


def test_gaussian_remove_geom_dominates_retains_discretize_then_dual(monkeypatch):
    """Gaussian upper REMOVE retains the original discretize-then-dual flow."""
    calls = []
    realizations = []
    actual_discretize = random_allocation_gaussian_module.discretize_continuous_ctd

    def recording_discretize(**kwargs):
        calls.append(kwargs)
        realization = actual_discretize(**kwargs)
        realizations.append(realization)
        return realization

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
        random_allocation_gaussian_module.calc_pld_dual(realizations[0])
    )

    assert len(calls) == 1
    assert "dual_dist" in calls[0]
    assert isinstance(base, PLDRealization)
    assert np.isclose(base.step, neg_dual.step, atol=TOL.GRID_ATOL)
    np.testing.assert_array_equal(neg_dual.x_array, expected.x_array)
    np.testing.assert_array_equal(neg_dual.prob_arr, expected.prob_arr)
    assert neg_dual.p_min == expected.p_min
    assert neg_dual.p_max == expected.p_max


@pytest.mark.parametrize("sigma", [0.05, 0.15])
def test_gaussian_remove_geom_joint_grid_covers_negative_dual_mean(sigma: float):
    """Joint CtD support covers the reflected-dual law even for small sigma."""
    config = AllocationSchemeConfig()

    base, _ = random_allocation_gaussian_module._gaussian_remove_geom_loss_factors(
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=BoundType.DOMINATES,
        sigma=sigma,
        config=config,
    )
    sigma_inv = 1.0 / sigma
    negative_dual_mean = -(sigma_inv**2) / 2.0
    assert base.x_array[0] + sigma_inv <= negative_dual_mean
    assert negative_dual_mean <= base.x_array[-1] - sigma_inv


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
            random_allocation_accounting_module,
            "exp_linear_to_geometric",
            lambda dist: DenseDiscreteDist(
                grid=dist.grid.exp(),
                prob_arr=dist.prob_arr,
                p_min=dist.p_min,
                p_max=dist.p_max,
                domain=Domain.POSITIVES,
            ),
        )
        monkeypatch.setattr(
            random_allocation_accounting_module,
            "geometric_self_convolve",
            _self_convolved_stub,
        )
        monkeypatch.setattr(
            random_allocation_accounting_module,
            "geometric_convolve",
            # The closing average divides by the factor count, so the stub must sum
            # anchors like the real convolution or that check correctly rejects it.
            _summed_anchor_stub,
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
            random_allocation_accounting_module,
            "exp_linear_to_geometric",
            lambda dist: DenseDiscreteDist(
                grid=dist.grid.exp(),
                prob_arr=dist.prob_arr,
                p_min=dist.p_min,
                p_max=dist.p_max,
                domain=Domain.POSITIVES,
            ),
        )
        monkeypatch.setattr(
            random_allocation_accounting_module,
            "geometric_self_convolve",
            _self_convolved_stub,
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
            atol=TOL.GRID_ATOL,
        )
