"""Unit tests for random-allocation composition wiring."""

# pylint: disable=too-many-lines

from __future__ import annotations

import warnings
from functools import partial
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from scipy import stats

import PLD_accounting.projections as projections_module
import PLD_accounting.random_allocation_accounting as random_allocation_accounting_module
import PLD_accounting.random_allocation_base as random_allocation_base_module
import PLD_accounting.random_allocation_api as random_allocation_api_module
import PLD_accounting.random_allocation_realization as random_allocation_realization_module
from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.mechanisms import gaussian_loss_law
from PLD_accounting.random_allocation_accounting import (
    _allocation_plds_core as allocation_plds_core,
)
from PLD_accounting.random_allocation_api import gaussian_allocation_pld, general_allocation_pld
from PLD_accounting.random_allocation_base import (
    allocation_plds_base,
    loss_discretization_counts,
)
from PLD_accounting.projections_stoch_dom import project_continuous_dist_stoch_dom
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    PrivacyParams,
    ProjectionMethod,
    SpacingType,
)
from PLD_accounting.utils import calc_pld_dual, negate_reverse_distribution
from tests.test_tolerances import TestTolerances as TOL
from tests.unit.allocation_oracles import (
    continuous_add_base_distribution,
    continuous_remove_base_distributions,
    realization_add_base_distribution,
    realization_remove_base_distributions,
)


def _core_composition_calls(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_steps: int,
    num_selected: int,
    num_epochs: int,
) -> list[tuple[int, int]]:
    """Return ``(num_steps, num_epochs)`` passed to each core component build.

    One core call builds both directions of a component.
    """
    calls: list[tuple[int, int]] = []

    def fake_core(**kwargs: Any) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
        calls.append((kwargs["num_steps"], kwargs["num_epochs"]))
        return _stub_linear_dist(), _stub_linear_dist()

    base_plds = (_stub_linear_dist(), _stub_linear_dist())
    compute_base_plds = Mock(return_value=base_plds)
    single_stage_counts = Mock(return_value=(1, 1))
    monkeypatch.setattr(
        random_allocation_accounting_module,
        "_allocation_plds_core",
        fake_core,
    )
    random_allocation_accounting_module.allocation_plds(
        compute_base_plds=compute_base_plds,
        base_loss_discretization_counts=single_stage_counts,
        projection_method=ProjectionMethod.STOCH_DOM,
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
    grid = GridSpec(step=0.5, n=3, anchor=0.0)
    return PLDRealization(grid=grid, prob_arr=np.array([0.6, 0.3, 0.1]))


def _stub_linear_dist() -> DenseDiscreteDist:
    grid = GridSpec(step=0.5, n=3, anchor=0.0)
    return DenseDiscreteDist(grid=grid, prob_arr=np.array([0.5, 0.3, 0.2]))


def _aligned_base_dist(step: float, origin_index: int) -> DenseDiscreteDist:
    """Build a small loss distribution on exact integer multiples of ``step``."""
    grid = GridSpec(step=step, n=4, index_0=origin_index)
    return DenseDiscreteDist(grid=grid, prob_arr=np.array([1e-8, 0.2, 0.5, 0.29999999]))


@pytest.mark.parametrize("direction", ["remove", "add"])
def test_one_step_geometric_factors_require_dense_distributions(direction: str) -> None:
    """One-step GEOM rejects either sparse factor and identifies the failing direction."""
    dense = _stub_linear_dist()
    sparse = SparseDiscreteDist(x_array=dense.x_array, prob_arr=dense.prob_arr)
    remove = sparse if direction == "remove" else dense
    add = sparse if direction == "add" else dense
    create_factors = Mock(return_value=((remove, dense), add))

    with pytest.raises(
        TypeError,
        match=f"GEOM {direction} input factor must be DenseDiscreteDist, got SparseDiscreteDist",
    ):
        allocation_plds_base(
            base_distributions_creation=create_factors,
            num_steps=1,
            loss_discretizations=(0.1, 0.1),
            tail_truncation=1e-10,
            bound_type=BoundType.DOMINATES,
            projection_method=ProjectionMethod.STOCH_DOM,
            convolution_method=ConvolutionMethod.GEOM,
            max_closing_grid=64,
        )


class TestGeneralAllocationWiring:
    """Tests that general (geometric-base) allocation delegates to shared helpers."""

    def test_general_allocation_builds_both_directions_in_one_call(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """General allocation makes one paired build from both realizations and composes it."""
        calls: list[dict[str, Any]] = []
        sentinel_pld = object()

        def fake_allocation_plds(**kwargs: Any) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
            calls.append(kwargs)
            return _stub_linear_dist(), _stub_linear_dist()

        def fake_compose_full_pld(*, remove_dist, add_dist, bound_type):
            del remove_dist, add_dist, bound_type
            return sentinel_pld

        monkeypatch.setattr(random_allocation_api_module, "allocation_plds", fake_allocation_plds)
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
        assert len(calls) == 1
        call = calls[0]
        assert call["num_steps"] == 23
        assert call["num_selected"] == 5
        assert call["num_epochs"] == 4
        assert call["loss_discretization"] == config.loss_discretization
        assert call["tail_truncation"] == config.tail_truncation
        assert call["bound_type"] == BoundType.IS_DOMINATED

        base_creation = _assert_route_wiring(call=call, config=config)
        assert base_creation.func is random_allocation_api_module.realization_base_distributions
        assert base_creation.keywords == {
            "remove_realization": remove_realization,
            "add_realization": add_realization,
            "convolution_method": config.convolution_method,
            "max_grid_mult": config.max_grid_mult,
            "max_cells_per_band": config.max_cells_per_band,
        }

    @pytest.mark.parametrize(
        "bound_type",
        [BoundType.DOMINATES, BoundType.IS_DOMINATED],
    )
    def test_realization_geometric_factors_honor_max_grid_mult(self, bound_type: BoundType):
        """Realization REMOVE and ADD factors use the shared geometric grid cap."""
        max_grid_mult = 100
        prob_arr = np.full(1_001, 1.0 / 1_001)
        grid = GridSpec(step=1e-3, n=prob_arr.size, anchor=0.0)
        realization = PLDRealization(grid=grid, prob_arr=prob_arr)

        remove_base, remove_dual = realization_remove_base_distributions(
            realization=realization,
            loss_discretization=1e-4,
            tail_truncation=1e-10,
            bound_type=bound_type,
            max_grid_mult=max_grid_mult,
        )
        add_base = realization_add_base_distribution(
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
            realization_remove_base_distributions,
            realization_add_base_distribution,
        ],
    )
    def test_realization_geometric_factors_require_two_grid_points(self, builder):
        """Realization factor builders reject a grid with no finite interval."""
        grid = GridSpec(step=0.1, n=1, anchor=0.0)
        realization = PLDRealization(grid=grid, prob_arr=np.array([1.0]))

        with pytest.raises(ValueError, match="at least two finite grid points"):
            builder(
                realization=realization,
                loss_discretization=0.1,
                tail_truncation=1e-10,
                bound_type=BoundType.DOMINATES,
                max_grid_mult=-1,
            )

    def test_general_allocation_rejects_num_steps_less_than_num_selected(self):
        """General allocation rejects num steps less than num selected."""
        remove_realization = _simple_realization()
        add_realization = _simple_realization()
        config = AllocationSchemeConfig(convolution_method=ConvolutionMethod.GEOM)
        with pytest.raises(ValueError, match="num_selected .* cannot exceed num_steps"):
            general_allocation_pld(
                num_steps=3,
                num_selected=4,
                num_epochs=1,
                remove_realization=remove_realization,
                add_realization=add_realization,
                config=config,
            )


def test_gaussian_allocation_builds_both_directions_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
):
    """Gaussian allocation makes one paired build from the Gaussian law and composes it."""
    calls: list[dict[str, Any]] = []
    sentinel_pld = object()

    def fake_allocation_plds(**kwargs: Any) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
        calls.append(kwargs)
        return _stub_linear_dist(), _stub_linear_dist()

    def fake_compose_full_pld(*, remove_dist, add_dist, bound_type):
        del remove_dist, add_dist, bound_type
        return sentinel_pld

    monkeypatch.setattr(random_allocation_api_module, "allocation_plds", fake_allocation_plds)
    monkeypatch.setattr(random_allocation_api_module, "compose_full_pld", fake_compose_full_pld)

    config = AllocationSchemeConfig(max_grid_mult=4_000)
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
    assert len(calls) == 1
    call = calls[0]
    assert call["num_steps"] == 19
    assert call["num_selected"] == 4
    assert call["num_epochs"] == 3
    assert call["loss_discretization"] == config.loss_discretization
    assert call["tail_truncation"] == config.tail_truncation
    assert call["bound_type"] == BoundType.DOMINATES

    base_creation = _assert_route_wiring(call=call, config=config)
    assert base_creation.func is random_allocation_api_module.continuous_base_distributions
    assert set(base_creation.keywords) == {
        "law",
        "dual",
        "convolution_method",
        "max_grid_mult",
        "max_cells_per_band",
    }
    assert base_creation.keywords["convolution_method"] == config.convolution_method
    assert base_creation.keywords["max_grid_mult"] == config.max_grid_mult
    assert base_creation.keywords["max_cells_per_band"] == config.max_cells_per_band
    # The Gaussian law is its own dual, and the mean is the one of ``gaussian_loss_law``.
    expected_law = gaussian_loss_law(params.sigma)
    assert base_creation.keywords["law"] is base_creation.keywords["dual"]
    assert base_creation.keywords["law"].mean() == expected_law.mean()
    assert base_creation.keywords["law"].std() == expected_law.std()


def _assert_route_wiring(*, call: dict[str, Any], config: AllocationSchemeConfig) -> Any:
    """Check the base builder and counts of a paired call; return its factor builder.

    Both are bound to the convolution method of the configuration and to rounding,
    the projection of a default configuration on the geometric grid.
    """
    api = random_allocation_api_module
    route = {
        "projection_method": ProjectionMethod.STOCH_DOM,
        "convolution_method": config.convolution_method,
    }
    builder = call["compute_base_plds"]
    assert builder.func is api.allocation_plds_base
    base_creation = builder.keywords["base_distributions_creation"]
    assert builder.keywords == {
        "base_distributions_creation": base_creation,
        "max_closing_grid": config.max_closing_grid,
        **route,
    }
    counts = call["base_loss_discretization_counts"]
    assert counts.func is api.loss_discretization_counts
    assert counts.keywords == route
    return base_creation


def test_allocation_plds_core_truncates_without_regridding(
    monkeypatch: pytest.MonkeyPatch,
):
    """The allocation core keeps the base step of each direction through composition."""
    captured: dict[str, float] = {}

    def fake_fft_self_convolve(
        *,
        dist: DenseDiscreteDist,
        num_convolutions: int,
        **_kwargs: Any,
    ) -> DenseDiscreteDist:
        captured["base_gap_at_compose"] = dist.step
        captured["num_epochs"] = float(num_convolutions)
        return dist

    def fake_compute_base_plds(
        *,
        num_steps: int,
        loss_discretizations: tuple[float, float],
        tail_truncation: float,
        bound_type: BoundType,
    ) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
        del tail_truncation, bound_type
        captured["num_steps"] = float(num_steps)
        captured["core_loss_discretization"], captured["add_loss_discretization"] = (
            loss_discretizations
        )
        return _stub_linear_dist(), _stub_linear_dist()

    monkeypatch.setattr(
        random_allocation_accounting_module,
        "fft_self_convolve",
        fake_fft_self_convolve,
    )

    config = AllocationSchemeConfig(
        loss_discretization=0.3,
        tail_truncation=1e-8,
    )
    result, add_result = allocation_plds_core(
        num_steps=7,
        num_epochs=5,
        compute_base_plds=fake_compute_base_plds,
        projection_method=ProjectionMethod.STOCH_DOM,
        loss_discretizations=(config.loss_discretization, 2 * config.loss_discretization),
        tail_truncation=config.tail_truncation,
        bound_type=BoundType.DOMINATES,
    )
    assert captured["add_loss_discretization"] == pytest.approx(
        2 * config.loss_discretization / 5, abs=TOL.GRID_ATOL
    )
    expected_step = _stub_linear_dist().step
    assert np.isclose(add_result.step, expected_step, atol=TOL.GRID_ATOL)

    expected_core_loss = config.loss_discretization / 5

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


def test_allocation_plds_warns_when_fallback_regrids(
    monkeypatch: pytest.MonkeyPatch,
):
    """The allocation warns when it must align the floor/ceil grids of a direction."""

    def fake_core(**kwargs: Any) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
        step = 0.1 if kwargs["num_steps"] == 3 else 0.2
        grid = GridSpec(step=step, n=2, anchor=0.0)
        dist = DenseDiscreteDist(grid=grid, prob_arr=np.array([0.5, 0.5]))
        return dist, dist

    def fake_rediscretize_dist(**kwargs: Any) -> DenseDiscreteDist:
        dist = kwargs["dist"]
        grid = GridSpec(step=kwargs["loss_discretization"], n=dist.prob_arr.size, anchor=dist.x_0)
        return DenseDiscreteDist(grid=grid, prob_arr=dist.prob_arr)

    def fake_fft_convolve(
        *,
        dist_1: DenseDiscreteDist,
        dist_2: DenseDiscreteDist,
        tail_truncation: float,
        bound_type: BoundType,
        projection_method: ProjectionMethod,
    ) -> DenseDiscreteDist:
        del tail_truncation, bound_type, projection_method
        assert np.isclose(dist_1.step, dist_2.step)
        return dist_1

    base_plds = (_stub_linear_dist(), _stub_linear_dist())
    compute_base_plds = Mock(return_value=base_plds)
    single_stage_counts = Mock(return_value=(1, 1))
    monkeypatch.setattr(
        random_allocation_accounting_module,
        "_allocation_plds_core",
        fake_core,
    )
    monkeypatch.setattr(
        random_allocation_accounting_module,
        "project_discrete_dist",
        fake_rediscretize_dist,
    )
    monkeypatch.setattr(
        random_allocation_accounting_module,
        "fft_convolve",
        fake_fft_convolve,
    )

    with pytest.warns(
        UserWarning,
        match="allocation_plds: aligning mismatched floor/ceil grids",
    ):
        random_allocation_accounting_module.allocation_plds(
            compute_base_plds=compute_base_plds,
            base_loss_discretization_counts=single_stage_counts,
            projection_method=ProjectionMethod.STOCH_DOM,
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
        match="allocation_plds: aligning mismatched floor/ceil grids",
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


def test_continuous_remove_dominates_discretizes_once_then_takes_the_dual(monkeypatch):
    """An upper bound discretizes each law once and takes the exact dual of the remove factor."""
    calls = []
    realizations = []
    actual_discretize = projections_module.project_continuous_dist_ctd

    def recording_discretize(**kwargs):
        calls.append(kwargs)
        realization = actual_discretize(**kwargs)
        realizations.append(realization)
        return realization

    monkeypatch.setattr(projections_module, "project_continuous_dist_ctd", recording_discretize)

    law = gaussian_loss_law(2.0)
    base, neg_dual = continuous_remove_base_distributions(
        law=law,
        dual=law,
        loss_discretization=0.1,
        tail_truncation=1e-6,
        bound_type=BoundType.DOMINATES,
        max_grid_mult=-1,
    )
    expected = negate_reverse_distribution(calc_pld_dual(realizations[0]))

    assert len(calls) == 2
    assert "dual_dist" in calls[0]
    assert isinstance(base, PLDRealization)
    assert np.isclose(base.step, neg_dual.step, atol=TOL.GRID_ATOL)
    np.testing.assert_array_equal(neg_dual.x_array, expected.x_array)
    np.testing.assert_array_equal(neg_dual.prob_arr, expected.prob_arr)
    assert neg_dual.p_min == expected.p_min
    assert neg_dual.p_max == expected.p_max


@pytest.mark.parametrize("sigma", [0.05, 0.15])
def test_continuous_remove_joint_grid_covers_negative_dual_mean(sigma: float):
    """Joint CtD support covers the reflected-dual law even for small sigma."""
    config = AllocationSchemeConfig()
    law = gaussian_loss_law(sigma)

    base, _ = continuous_remove_base_distributions(
        law=law,
        dual=law,
        loss_discretization=config.loss_discretization,
        tail_truncation=config.tail_truncation,
        bound_type=BoundType.DOMINATES,
        max_grid_mult=-1,
    )
    sigma_inv = 1.0 / sigma
    negative_dual_mean = -(sigma_inv**2) / 2.0
    assert base.x_array[0] + sigma_inv <= negative_dual_mean
    assert negative_dual_mean <= base.x_array[-1] - sigma_inv


def test_continuous_remove_dominates_honors_max_grid_mult():
    """Upper REMOVE coarsens its single primary grid to the point cap."""
    requested_step = 1e-4
    max_grid_mult = 100
    law = gaussian_loss_law(1.0)

    base, neg_dual = continuous_remove_base_distributions(
        law=law,
        dual=law,
        loss_discretization=requested_step,
        tail_truncation=1e-10,
        bound_type=BoundType.DOMINATES,
        max_grid_mult=max_grid_mult,
    )

    assert base.prob_arr.size <= max_grid_mult
    assert base.step > requested_step
    assert neg_dual.prob_arr.size == base.prob_arr.size
    assert neg_dual.step == base.step


def test_continuous_remove_is_dominated_rounds_each_law_on_its_own():
    """Lower REMOVE rounds ``law`` down and ``dual`` up, each read for its own factor."""
    law = stats.norm(loc=0.125, scale=0.5)
    dual = stats.norm(loc=0.4, scale=0.7)
    tail_truncation = 1e-6

    loss_factor, neg_dual_factor = continuous_remove_base_distributions(
        law=law,
        dual=dual,
        loss_discretization=0.1,
        tail_truncation=tail_truncation,
        bound_type=BoundType.IS_DOMINATED,
        max_grid_mult=-1,
    )

    expected_loss = project_continuous_dist_stoch_dom(
        dist=law,
        tail_truncation=tail_truncation / 2,
        bound_type=BoundType.IS_DOMINATED,
        step=0.1,
    )
    expected_neg_dual = negate_reverse_distribution(
        project_continuous_dist_stoch_dom(
            dist=dual,
            tail_truncation=tail_truncation / 2,
            bound_type=BoundType.DOMINATES,
            step=0.1,
        )
    )
    for actual, expected in (
        (loss_factor, expected_loss),
        (neg_dual_factor, expected_neg_dual),
    ):
        assert actual.grid == expected.grid
        np.testing.assert_array_equal(actual.prob_arr, expected.prob_arr)
        assert actual.p_min == expected.p_min
        assert actual.p_max == expected.p_max
    # Rounding the dual up and reflecting is a lower bound of the negated dual.
    assert neg_dual_factor.p_max == 0.0
    assert neg_dual_factor.p_min > 0.0


@pytest.mark.parametrize(
    "builder",
    [
        continuous_remove_base_distributions,
        continuous_add_base_distribution,
    ],
)
def test_continuous_dominating_builders_read_the_dual_argument(builder):
    """An equal but distinct dual object gives the factors of one object passed twice."""
    keywords = {
        "loss_discretization": 0.1,
        "tail_truncation": 1e-6,
        "bound_type": BoundType.DOMINATES,
    }
    shared = gaussian_loss_law(2.0)
    one_object = builder(law=shared, dual=shared, **keywords, max_grid_mult=-1)
    law = gaussian_loss_law(2.0)
    equal_dual = gaussian_loss_law(2.0)
    two_objects = builder(law=law, dual=equal_dual, **keywords, max_grid_mult=-1)
    # A dual that is not the law's dual breaks the CtD cell inequalities.
    mismatched_dual = stats.norm(loc=0.0, scale=0.5)
    with pytest.raises(ValueError):
        builder(law=shared, dual=mismatched_dual, **keywords, max_grid_mult=-1)

    if isinstance(one_object, DenseDiscreteDist):
        one_object, two_objects = (one_object,), (two_objects,)
    for actual, expected in zip(two_objects, one_object):
        assert actual.grid == expected.grid
        np.testing.assert_array_equal(actual.prob_arr, expected.prob_arr)
        assert actual.p_min == expected.p_min
        assert actual.p_max == expected.p_max


@pytest.mark.parametrize(
    ("bound_type", "function_name"),
    [
        (BoundType.DOMINATES, "project_continuous_dist_ctd"),
        (BoundType.IS_DOMINATED, "project_continuous_dist_stoch_dom"),
    ],
)
def test_continuous_add_discretizes_in_linear_loss_space(
    monkeypatch, bound_type: BoundType, function_name: str
):
    """Each direction chooses its fixed engine from the requested bound, once."""
    calls = []
    actual_discretize = getattr(projections_module, function_name)

    def recording_discretize(**kwargs):
        calls.append(kwargs)
        return actual_discretize(**kwargs)

    monkeypatch.setattr(projections_module, function_name, recording_discretize)
    law = gaussian_loss_law(2.0)
    result = continuous_add_base_distribution(
        law=law,
        dual=law,
        loss_discretization=0.1,
        tail_truncation=1e-6,
        bound_type=bound_type,
        max_grid_mult=-1,
    )

    assert len(calls) == 2
    assert ("dual_dist" in calls[1]) == (bound_type == BoundType.DOMINATES)
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


@pytest.mark.parametrize(
    "num_steps, depth_of_all, depth_of_all_but_one",
    [(1, 0, 0), (2, 1, 0), (3, 2, 1), (8, 3, 3), (9, 4, 3), (100, 7, 7)],
)
def test_geometric_loss_stage_counts(num_steps: int, depth_of_all: int, depth_of_all_but_one: int):
    """The stages are the input and the merges above one copy; one step is the input alone.

    Binary self-composition of ``n`` copies has depth ``ceil(log2 n)``, below its
    ``floor(log2 n) + popcount(n) - 1`` merges when ``n`` is not a power of two (8
    copies: depth 3 and 3 merges; 100 copies: depth 7 and 8 merges). Rounding the
    remove direction sums ``num_steps - 1`` copies and adds the closing merge; every
    other layout sums ``num_steps`` copies.
    """
    counts = partial(loss_discretization_counts, convolution_method=ConvolutionMethod.GEOM)
    rounding, ctd = ProjectionMethod.STOCH_DOM, ProjectionMethod.CTD
    sums_all = 1 + depth_of_all
    closes_with_p = 1 if num_steps == 1 else 2 + depth_of_all_but_one

    # The counts are ordered (remove, add).
    assert counts(num_steps=num_steps, projection_method=ctd) == (sums_all, sums_all)
    assert counts(num_steps=num_steps, projection_method=rounding) == (closes_with_p, sums_all)


def test_floor_and_ceil_steps_agree_within_each_direction_of_unequal_stage_counts():
    """Each direction splits its loss budget by its own counts, so its two components align.

    With ``num_steps=17`` and ``num_selected=2`` the components have 8 and 9 steps, where
    rounding counts ``(5, 4)`` and ``(5, 5)`` stages for ``(remove, add)``. The remove and
    add steps then differ, and neither needs the other's.
    """
    num_epochs = 3
    loss_discretization = 0.1
    base_calls: list[tuple[int, tuple[float, float]]] = []

    def counts(num_steps: int) -> tuple[int, int]:
        return {8: (5, 4), 9: (5, 5)}[num_steps]

    def compute_base_plds(*, num_steps, loss_discretizations, tail_truncation, bound_type):
        del tail_truncation, bound_type
        base_calls.append((num_steps, loss_discretizations))
        stage_counts = counts(num_steps)
        base_dists = []
        for step, count in zip(loss_discretizations, stage_counts):
            grid = GridSpec(step=step / count, n=3, anchor=0.0)
            base_dists.append(DenseDiscreteDist(grid=grid, prob_arr=np.array([0.5, 0.3, 0.2])))
        return tuple(base_dists)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        remove, add = random_allocation_accounting_module.allocation_plds(
            compute_base_plds=compute_base_plds,
            base_loss_discretization_counts=counts,
            projection_method=ProjectionMethod.STOCH_DOM,
            num_steps=17,
            num_selected=2,
            num_epochs=num_epochs,
            loss_discretization=loss_discretization,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )

    assert [num_steps for num_steps, _ in base_calls] == [8, 9]
    assert remove.step == pytest.approx(loss_discretization / (num_epochs * (5 + 5)), rel=1e-12)
    assert add.step == pytest.approx(loss_discretization / (num_epochs * (4 + 5)), rel=1e-12)
    for direction in (0, 1):
        component_budgets = [budgets[direction] for _, budgets in base_calls]
        assert sum(component_budgets) * num_epochs == pytest.approx(loss_discretization, rel=1e-12)


@pytest.mark.parametrize("num_steps", [5, 6, 10])
@pytest.mark.parametrize("projection_method", [ProjectionMethod.STOCH_DOM, ProjectionMethod.CTD])
def test_geometric_base_splits_the_tail_between_the_input_and_the_merges(
    monkeypatch: pytest.MonkeyPatch, projection_method: ProjectionMethod, num_steps: int
):
    """Half of the tail goes to the input, over the factors, and half to each sum's merges.

    The rounding remove sum gives ``(num_steps - 1) / num_steps`` of its half to the
    copies under ``Q`` and the rest to the closing merge with the factor under ``P``.
    """
    tail_truncation = 1e-6
    input_tails: list[float] = []
    self_merge_tails: dict[int, float] = {}
    closing_tails: list[float] = []
    base = random_allocation_base_module
    actual_self_convolve = base.self_convolve
    actual_convolve = base.convolve

    def base_distributions_creation(**kwargs):
        input_tails.append(kwargs["tail_truncation"])
        remove_realization = _simple_realization()
        add_realization = _simple_realization()
        return random_allocation_realization_module.realization_base_distributions(
            convolution_method=ConvolutionMethod.GEOM,
            max_grid_mult=-1,
            max_cells_per_band=1 << 14,
            remove_realization=remove_realization,
            add_realization=add_realization,
            **kwargs,
        )

    def recording_self_convolve(**kwargs):
        self_merge_tails[kwargs["num_convolutions"]] = kwargs["tail_truncation"]
        return actual_self_convolve(**kwargs)

    def recording_convolve(**kwargs):
        closing_tails.append(kwargs["tail_truncation"])
        return actual_convolve(**kwargs)

    monkeypatch.setattr(base, "self_convolve", recording_self_convolve)
    monkeypatch.setattr(base, "convolve", recording_convolve)
    allocation_plds_base(
        convolution_method=ConvolutionMethod.GEOM,
        max_closing_grid=1_000_000,
        base_distributions_creation=base_distributions_creation,
        num_steps=num_steps,
        loss_discretizations=(0.1, 0.1),
        tail_truncation=tail_truncation,
        bound_type=BoundType.DOMINATES,
        projection_method=projection_method,
    )

    half = tail_truncation / 2
    assert input_tails == [pytest.approx(half / num_steps, rel=1e-12)]
    assert self_merge_tails[num_steps] == pytest.approx(half, rel=1e-12)
    if projection_method == ProjectionMethod.CTD:
        assert set(self_merge_tails) == {num_steps} and not closing_tails
    else:
        assert self_merge_tails[num_steps - 1] == pytest.approx(
            half * (num_steps - 1) / num_steps, rel=1e-12
        )
        assert closing_tails == [pytest.approx(half / num_steps, rel=1e-12)]


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_geometric_allocation_preserves_integer_loss_lattice(bound_type: BoundType):
    """Anchored ADD and REMOVE composition return grids aligned to zero loss."""
    num_steps = 7

    def base_distributions_creation(*, loss_discretizations, **_kwargs):
        remove_step, add_step = loss_discretizations
        return (
            (_aligned_base_dist(remove_step, -2), _aligned_base_dist(remove_step, -1)),
            _aligned_base_dist(add_step, -2),
        )

    remove, add = allocation_plds_base(
        convolution_method=ConvolutionMethod.GEOM,
        max_closing_grid=1_000_000,
        base_distributions_creation=base_distributions_creation,
        num_steps=num_steps,
        loss_discretizations=(0.1, 0.1),
        tail_truncation=1e-6,
        bound_type=bound_type,
        projection_method=ProjectionMethod.STOCH_DOM,
    )

    for dist in (remove, add):
        lower_index = round(dist.x_0 / dist.step)
        assert np.isclose(
            dist.x_0,
            lower_index * dist.step,
            atol=TOL.GRID_ATOL,
        )


def test_fft_memory_cap_on_the_base_pld_warns(monkeypatch):
    """Coarsening a base PLD to fit the FFT memory limit is reported, like every other cap."""
    num_epochs = 2
    # Room for 64 base points: MAX_FFT_BYTES // (num_epochs * 16 bytes per point).
    monkeypatch.setattr(random_allocation_accounting_module, "MAX_FFT_BYTES", 64 * num_epochs * 16)
    params = PrivacyParams(sigma=1.0, num_steps=4, num_epochs=num_epochs, delta=1e-5)
    config = AllocationSchemeConfig(loss_discretization=0.05, tail_truncation=1e-8)
    with pytest.warns(UserWarning, match="epoch composition: MAX_FFT_BYTES=.* coarsens its step"):
        gaussian_allocation_pld(params=params, config=config, bound_type=BoundType.DOMINATES)
