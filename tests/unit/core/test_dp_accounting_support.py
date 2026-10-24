"""Unit tests for dp_accounting_support."""

import math

import numpy as np
import pytest
from dp_accounting.pld import privacy_loss_distribution as dp_pld
from dp_accounting.pld.pld_pmf import DensePLDPmf, SparsePLDPmf

from PLD_accounting.discrete_dist import (
    REALIZATION_MOMENT_TOL,
    DenseDiscreteDist,
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_utils import (
    MAX_SAFE_EXP_ARG,
    PMF_MASS_DRIFT_TOL,
    PMF_TOLERATED_MASS_TOL,
    exp_moment_terms,
    signed_unit_residual,
)
from PLD_accounting.dp_accounting_support import (
    DP_ACCOUNTING_MASS_REPAIR_TOL,
    DP_ACCOUNTING_NEGATIVE_REPAIR_TOL,
    dp_accounting_pmf_to_pld_realization,
    linear_dist_to_dp_accounting_pmf,
)
from PLD_accounting.mechanisms import gaussian_distribution, laplace_distribution
from PLD_accounting.projections import project_discrete_dist
from PLD_accounting.projections_stoch_dom import project_discrete_dist_stoch_dom
from PLD_accounting.types import BoundType
from PLD_accounting.utils import calc_pld_dual, negate_reverse_distribution
from tests.test_tolerances import TestTolerances as TOL
from tests.unit.allocation_oracles import realization_remove_base_distributions


def test_realization_remove_dominates_uses_existing_ctd_then_dual(monkeypatch):
    """An upper bound projects each direction once and takes the dual of the remove base."""
    remove_grid = GridSpec(step=0.5, n=3, anchor=0.0)
    remove_realization = PLDRealization(
        grid=remove_grid, prob_arr=np.array([0.4, 0.35, 0.25], dtype=np.float64)
    )
    requested_spacing = 0.25
    calls = []
    realizations = []
    actual_rediscretize = project_discrete_dist

    def recording_rediscretize(**kwargs):
        calls.append(kwargs)
        rediscretized = actual_rediscretize(**kwargs)
        realizations.append(rediscretized)
        return rediscretized

    monkeypatch.setattr(
        "PLD_accounting.random_allocation_realization.project_discrete_dist",
        recording_rediscretize,
    )

    base_dist, neg_dual_dist = realization_remove_base_distributions(
        realization=remove_realization,
        loss_discretization=requested_spacing,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        max_grid_mult=-1,
    )
    expected_neg_dual = negate_reverse_distribution(calc_pld_dual(realizations[0]))
    effective_spacing = remove_realization.step

    assert len(calls) == 2
    assert calls[0]["dist"] is remove_realization
    assert calls[0]["loss_discretization"] == effective_spacing
    assert calls[0]["bound_type"] == BoundType.DOMINATES
    assert base_dist.step == effective_spacing
    assert neg_dual_dist.step == effective_spacing
    assert isinstance(base_dist, PLDRealization)
    np.testing.assert_array_equal(neg_dual_dist.x_array, expected_neg_dual.x_array)
    np.testing.assert_array_equal(neg_dual_dist.prob_arr, expected_neg_dual.prob_arr)
    assert neg_dual_dist.p_min == expected_neg_dual.p_min
    assert neg_dual_dist.p_max == expected_neg_dual.p_max


def test_calc_pld_dual_of_an_add_pld_weights_by_the_ratio_and_reflects_the_support() -> None:
    """The dual of ``-ln Z`` has mass ``z f(z)`` at ``ln z`` and ``1 - E[Z]`` at ``+inf``."""
    # Ratios 2, 1 and 1/2 at add losses -ln 2, 0 and ln 2; the atom at ratio zero is the
    # add mass at +inf. The mean of the ratio is 0.9.
    step = math.log(2.0)
    q_masses = np.array([0.2, 0.35, 0.3])
    ratios = np.array([2.0, 1.0, 0.5])
    add_grid = GridSpec(step=step, n=3, index_0=-1)
    add_pld = PLDRealization(grid=add_grid, prob_arr=q_masses, p_max=0.15)

    remove_pld = calc_pld_dual(add_pld)

    assert remove_pld.grid == add_pld.grid.reflect()
    float_eps = np.finfo(float).eps
    np.testing.assert_allclose(remove_pld.prob_arr, (q_masses * ratios)[::-1], rtol=4 * float_eps)
    assert remove_pld.p_min == 0.0
    assert remove_pld.p_max == pytest.approx(0.1, abs=4 * float_eps)
    assert (
        abs(
            signed_unit_residual(
                values=remove_pld.prob_arr, lower_term=0.0, upper_term=remove_pld.p_max
            )
        )
        <= PMF_MASS_DRIFT_TOL
    )


def test_calc_pld_dual_of_an_add_pld_with_unit_mean_has_no_mass_at_infinity() -> None:
    """A mean of the ratio equal to one leaves nothing at ``+inf``, whatever the add atom there."""
    step = math.log(2.0)
    add_grid = GridSpec(step=step, n=2, index_0=-1)
    add_pld = PLDRealization(grid=add_grid, prob_arr=np.array([0.25, 0.5]), p_max=0.25)

    remove_pld = calc_pld_dual(add_pld)

    np.testing.assert_array_equal(remove_pld.prob_arr, [0.5, 0.5])
    assert (remove_pld.p_min, remove_pld.p_max) == (0.0, 0.0)


def test_calc_pld_dual_on_explicit_points_is_the_dual_of_the_realization() -> None:
    """A distribution on the points of a grid has the masses of the realization's dual."""
    add_grid = GridSpec(step=math.log(2.0), n=3, index_0=-1)
    q_masses = np.array([0.2, 0.35, 0.3])
    realization = PLDRealization(grid=add_grid, prob_arr=q_masses, p_max=0.15)
    on_points = SparseDiscreteDist(x_array=realization.x_array, prob_arr=q_masses, p_max=0.15)

    dual = calc_pld_dual(on_points)
    expected = calc_pld_dual(realization)

    assert isinstance(dual, SparseDiscreteDist)
    np.testing.assert_array_equal(dual.x_array, -on_points.x_array[::-1])
    np.testing.assert_array_equal(dual.prob_arr, expected.prob_arr)
    assert (dual.p_min, dual.p_max) == (0.0, expected.p_max)
    reflected = negate_reverse_distribution(dual)
    np.testing.assert_array_equal(reflected.x_array, on_points.x_array)
    assert (reflected.p_min, reflected.p_max) == (expected.p_max, 0.0)


def test_calc_pld_dual_on_explicit_points_rejects_mass_at_minus_infinity() -> None:
    """The dual is of a privacy loss, which has no mass at ``-inf``."""
    dist = SparseDiscreteDist(
        x_array=np.array([-1.0, 1.0]), prob_arr=np.array([0.4, 0.4]), p_min=0.2
    )
    with pytest.raises(ValueError, match="no mass at -inf"):
        calc_pld_dual(dist)


def test_calc_pld_dual_repairs_a_drift_scale_moment_excess_directionally() -> None:
    """Even sub-ULP excess is trimmed at one edge rather than globally rescaled."""
    subnormal_step = float(np.nextafter(0.0, 1.0))
    grid = GridSpec(step=subnormal_step, n=np.full(10, 0.1).size, anchor=0.0)
    realization = PLDRealization(grid=grid, prob_arr=np.full(10, 0.1))
    expected_finite = exp_moment_terms(prob_arr=realization.prob_arr, x_vals=realization.x_array)[
        ::-1
    ]
    excess = -signed_unit_residual(values=expected_finite, lower_term=0.0, upper_term=0.0)
    assert 0.0 < excess < PMF_MASS_DRIFT_TOL

    dual = calc_pld_dual(realization)

    assert dual.prob_arr[0] == expected_finite[0] - excess
    np.testing.assert_array_equal(dual.prob_arr[1:], expected_finite[1:])
    assert (
        abs(
            signed_unit_residual(values=dual.prob_arr, lower_term=dual.p_min, upper_term=dual.p_max)
        )
        <= PMF_TOLERATED_MASS_TOL
    )


def test_calc_pld_dual_trims_a_material_moment_excess_from_the_lowest_loss_edge() -> None:
    """An admitted arithmetic excess is removed directionally without global rescaling."""
    grid = GridSpec(step=1e-16, n=np.full(10, 0.1).size, anchor=-2e-15)
    realization = PLDRealization(grid=grid, prob_arr=np.full(10, 0.1))
    expected_finite = exp_moment_terms(prob_arr=realization.prob_arr, x_vals=realization.x_array)[
        ::-1
    ]
    excess = -signed_unit_residual(values=expected_finite, lower_term=0.0, upper_term=0.0)
    assert PMF_MASS_DRIFT_TOL <= excess < REALIZATION_MOMENT_TOL

    dual = calc_pld_dual(realization)

    assert dual.prob_arr[0] == expected_finite[0] - excess
    np.testing.assert_array_equal(dual.prob_arr[1:], expected_finite[1:])
    assert dual.p_max == 0.0


def test_dp_accounting_roundtrip_preserves_mass_and_grid_shape():
    """Dp accounting roundtrip preserves mass and grid shape."""
    original = _make_realization()
    pmf = linear_dist_to_dp_accounting_pmf(dist=original, bound_type=BoundType.DOMINATES)
    restored = dp_accounting_pmf_to_pld_realization(pmf=pmf)

    assert restored.x_array.shape == original.x_array.shape
    total_mass = math.fsum([*map(float, restored.prob_arr), restored.p_min, restored.p_max])
    assert np.isclose(total_mass, 1.0, atol=TOL.PROBABILITY_ATOL)
    assert np.isclose(restored.p_max, original.p_max, atol=TOL.PROBABILITY_ATOL)


def test_linear_dist_to_dp_accounting_handles_zero_finite_mass():
    """Test that distributions with all mass at infinity are handled correctly."""
    grid = GridSpec(step=1.0, n=2, anchor=0.0)
    realization = DenseDiscreteDist(
        grid=grid, prob_arr=np.array([0.0, 0.0], dtype=np.float64), p_max=1.0
    )
    pmf = linear_dist_to_dp_accounting_pmf(dist=realization, bound_type=BoundType.DOMINATES)
    assert pmf._infinity_mass == 1.0
    assert np.allclose(pmf._probs, np.array([0.0, 0.0]))


@pytest.mark.parametrize("pessimistic", [True, False])
@pytest.mark.parametrize("index_0", [0, -3, 7])
def test_linear_adapter_exports_a_zero_anchored_index_exactly(index_0: int, pessimistic: bool):
    """A zero-anchored grid exports its own integer index, with no tolerance involved.

    Internal producers carry the integer index structurally, so the export is exact
    and needs no band that snaps a near-zero origin such as ``x_0 = 2.2e-16`` to ``0.0``.
    """
    grid = GridSpec(step=1.0, n=1, index_0=index_0)
    dist = DenseDiscreteDist(grid=grid, prob_arr=np.array([1.0]))

    bound_type = BoundType.DOMINATES if pessimistic else BoundType.IS_DOMINATED
    pmf = linear_dist_to_dp_accounting_pmf(dist=dist, bound_type=bound_type)

    assert pmf._lower_loss == index_0


SUBNORMAL_ANCHOR = float(np.nextafter(0.0, 1.0))


@pytest.mark.parametrize("pessimistic", [True, False])
@pytest.mark.parametrize(
    "anchor",
    [SUBNORMAL_ANCHOR, 0.3, 1.0],
    ids=["subnormal", "off_lattice", "whole_multiple"],
)
def test_linear_adapter_rejects_a_non_zero_anchor(anchor: float, pessimistic: bool):
    """Export requires the k*step lattice; the offset belongs in ``index_0``.

    A whole multiple is rejected with the rest: it is the same lattice written into the
    wrong field, and accepting it would reinstate the quotient correction this replaced.
    """
    grid = GridSpec(step=1.0, n=1, anchor=anchor)
    dist = DenseDiscreteDist(grid=grid, prob_arr=np.array([1.0]))

    bound_type = BoundType.DOMINATES if pessimistic else BoundType.IS_DOMINATED
    with pytest.raises(ValueError, match="whole multiples of its step"):
        linear_dist_to_dp_accounting_pmf(dist=dist, bound_type=bound_type)


@pytest.mark.parametrize("index_0", [1, 2, -3])
@pytest.mark.parametrize("pessimistic", [True, False])
def test_linear_adapter_preserves_the_bound_between_lattice_knots(index_0: int, pessimistic: bool):
    """The converted PMF bounds the source hockey-stick curve at every epsilon.

    Regression test for nearest-lattice rounding, which shifted losses by up to half a
    step in whichever direction happened to be closer and so could report a delta below
    the truth for DOMINATES (an invalid upper bound) or above it for IS_DOMINATED. The
    epsilon sweep deliberately falls between lattice knots. Zero-anchored export makes
    that rounding structurally impossible, so this now guards the exactness claim.
    """
    grid = GridSpec(step=0.1, n=5, index_0=index_0)
    dist = DenseDiscreteDist(
        grid=grid, prob_arr=np.array([0.15, 0.25, 0.3, 0.2, 0.05], dtype=np.float64), p_max=0.05
    )
    epsilons = np.linspace(-0.6, 0.6, 601)
    true_deltas = np.array(
        [
            dist.p_max
            + math.fsum(
                float(prob) * -math.expm1(float(epsilon - loss))
                for loss, prob in zip(dist.x_array, dist.prob_arr, strict=True)
                if loss > epsilon
            )
            for epsilon in epsilons
        ]
    )

    bound_type = BoundType.DOMINATES if pessimistic else BoundType.IS_DOMINATED
    pmf = linear_dist_to_dp_accounting_pmf(dist=dist, bound_type=bound_type)
    deltas = np.asarray(pmf.get_delta_for_epsilon(list(epsilons)), dtype=np.float64)

    if pessimistic:
        assert np.all(deltas >= true_deltas - TOL.MASS_CONSERVATION)
    else:
        assert np.all(deltas <= true_deltas + TOL.MASS_CONSERVATION)


def test_dp_accounting_composed_gaussian_add_pmf_converts_with_repair():
    """Composed Gaussian add-direction PMF converts successfully with the always-on repair.

    dp_accounting's Gaussian discretization produces a systematic exp-moment
    violation that grows with composition (~33 after 20 rounds of subsampled
    Gaussian σ=0.5, q=1/20).  The repair is applied automatically and must
    produce a valid realization with E[exp(-L)] <= 1.
    """
    pld = dp_pld.from_gaussian_mechanism(
        standard_deviation=0.5,
        value_discretization_interval=1e-4,
        pessimistic_estimate=True,
        sampling_prob=1.0 / 20.0,
        use_connect_dots=True,
    ).self_compose(20)

    realization = dp_accounting_pmf_to_pld_realization(pmf=pld._pmf_add)
    total_mass = math.fsum(
        [*map(float, realization.prob_arr), realization.p_min, realization.p_max]
    )
    exp_moment_val = math.fsum(
        map(float, exp_moment_terms(prob_arr=realization.prob_arr, x_vals=realization.x_array))
    )

    assert np.isclose(total_mass, 1.0, atol=TOL.PROBABILITY_ATOL)
    assert exp_moment_val <= 1.0 + REALIZATION_MOMENT_TOL


def test_dp_accounting_import_rejects_excessive_moment_repair() -> None:
    """The import boundary rejects a law requiring material mass movement."""
    pmf = DensePLDPmf(
        discretization=25.0,
        lower_loss=-1,
        probs=np.array([0.5, 0.5], dtype=np.float64),
        infinity_mass=0.0,
        pessimistic_estimate=True,
    )
    with pytest.raises(ValueError, match="dp_accounting import reciprocal moment"):
        dp_accounting_pmf_to_pld_realization(pmf=pmf)


def test_dp_accounting_import_validates_moment_bands_without_a_repair() -> None:
    """Invalid moment bands fail even when the input moment already satisfies the invariant."""
    pmf = DensePLDPmf(
        discretization=1.0,
        lower_loss=0,
        probs=np.array([1.0], dtype=np.float64),
        infinity_mass=0.0,
        pessimistic_estimate=True,
    )
    with pytest.raises(ValueError, match="0 <= drift_tol <= repair_tol"):
        dp_accounting_pmf_to_pld_realization(
            pmf=pmf,
            moment_drift_tol=2.0,
            moment_repair_tol=1.0,
        )


def test_dp_accounting_composed_gaussian_imports_when_depth_is_declared() -> None:
    """Compose-100 drift is admitted once the ceiling is raised, and still reported."""
    depth = 100
    pld = dp_pld.from_gaussian_mechanism(
        standard_deviation=1.0,
        value_discretization_interval=1e-4,
    ).self_compose(depth)
    with pytest.warns(RuntimeWarning, match="dp_accounting import mass"):
        realization = dp_accounting_pmf_to_pld_realization(
            pmf=pld._pmf_remove,
            mass_repair_tol=DP_ACCOUNTING_MASS_REPAIR_TOL * depth,
        )
    assert realization.p_min == 0.0


def test_dp_accounting_import_rejects_optimistic_pmf() -> None:
    """An optimistic dp_accounting PMF does not map to either package BoundType."""
    original = _make_realization()
    pmf = linear_dist_to_dp_accounting_pmf(dist=original, bound_type=BoundType.IS_DOMINATED)
    assert pmf._pessimistic_estimate is False
    with pytest.raises(ValueError, match="optimistic dp_accounting PMF does not map"):
        dp_accounting_pmf_to_pld_realization(pmf=pmf)


def test_dp_accounting_import_rejects_unsupported_pmf_type() -> None:
    """Unsupported PMF representations fail with TypeError, not AttributeError."""
    with pytest.raises(TypeError, match="Unrecognized PMF format"):
        dp_accounting_pmf_to_pld_realization(pmf=object())  # type: ignore[arg-type]


def test_dp_accounting_import_rejects_negative_infinity_mass() -> None:
    """Negative infinity mass is classified with the finite negative bins."""
    pmf = DensePLDPmf(
        discretization=1.0,
        lower_loss=0,
        probs=np.array([1.0], dtype=np.float64),
        infinity_mass=-0.05,
        pessimistic_estimate=True,
    )
    with pytest.raises(ValueError, match="dp_accounting import negative mass"):
        dp_accounting_pmf_to_pld_realization(pmf=pmf)


def test_dp_accounting_import_does_not_cancel_overshoot_with_negative_infinity() -> None:
    """Finite mass 1.1 plus infinity mass -0.1 must not become unit mass after clipping.

    The negative-mass ceiling is widened so clipping is admitted; an upper clip
    to ``[0, 1]`` would then hide the 1.1 overshoot by cancelling it with the
    removed negative infinity mass.
    """
    pmf = DensePLDPmf(
        discretization=1.0,
        lower_loss=0,
        probs=np.array([1.1], dtype=np.float64),
        infinity_mass=-0.1,
        pessimistic_estimate=True,
    )
    with pytest.raises(ValueError, match="dp_accounting import mass"):
        dp_accounting_pmf_to_pld_realization(
            pmf=pmf,
            negative_drift_tol=0.2,
            negative_repair_tol=0.2,
        )


def test_dp_accounting_import_rejects_finite_mass_overshoot() -> None:
    """Upper-clipping is not applied, so a finite overshoot reaches mass validation."""
    pmf = DensePLDPmf(
        discretization=1.0,
        lower_loss=0,
        probs=np.array([1.1], dtype=np.float64),
        infinity_mass=0.0,
        pessimistic_estimate=True,
    )
    with pytest.raises(ValueError, match="dp_accounting import mass"):
        dp_accounting_pmf_to_pld_realization(pmf=pmf)


def test_dp_accounting_import_rejects_clipped_negative_mass() -> None:
    """A large negative bin is illegal even when the net residual is near zero."""
    pmf = DensePLDPmf(
        discretization=1.0,
        lower_loss=0,
        probs=np.array([0.5, -0.3, 0.8], dtype=np.float64),
        infinity_mass=0.0,
        pessimistic_estimate=True,
    )
    assert 0.3 > DP_ACCOUNTING_NEGATIVE_REPAIR_TOL
    with pytest.raises(ValueError, match="dp_accounting import negative mass"):
        dp_accounting_pmf_to_pld_realization(pmf=pmf)


def test_dp_accounting_import_warns_on_sub_ceiling_negative_mass() -> None:
    """Negative mass above noise but under the ceiling is clipped, and reported."""
    tiny = DP_ACCOUNTING_NEGATIVE_REPAIR_TOL / 2.0
    pmf = DensePLDPmf(
        discretization=1.0,
        lower_loss=0,
        probs=np.array([0.5, -tiny, 0.5 + tiny], dtype=np.float64),
        infinity_mass=0.0,
        pessimistic_estimate=True,
    )
    with pytest.warns(RuntimeWarning, match="dp_accounting import negative mass"):
        dp_accounting_pmf_to_pld_realization(pmf=pmf)


def test_exp_moment_handles_very_negative_losses_with_tiny_mass():
    """Tiny mass at very negative loss should be handled without clipping artifacts."""
    tiny_prob = np.exp(-(MAX_SAFE_EXP_ARG + 21.0))
    grid = GridSpec(step=MAX_SAFE_EXP_ARG + 21.0, n=2, anchor=-(MAX_SAFE_EXP_ARG + 20.0))
    realization = PLDRealization(
        grid=grid, prob_arr=np.array([tiny_prob, 1.0 - tiny_prob], dtype=np.float64)
    )

    expected = np.exp(np.log(realization.prob_arr[0]) - realization.x_array[0])
    expected += realization.prob_arr[1] * np.exp(-realization.x_array[1])
    assert np.isclose(
        math.fsum(
            map(float, exp_moment_terms(prob_arr=realization.prob_arr, x_vals=realization.x_array))
        ),
        expected,
        rtol=1e-12,
        atol=1e-15,
    )


def test_realization_remove_base_distributions_derives_dual_from_coarsened_base():
    """The remove path should coarsen once, then build the negated dual from that base."""
    stage1_poisson = dp_pld.from_gaussian_mechanism(
        standard_deviation=1.0,
        value_discretization_interval=1e-4,
        pessimistic_estimate=True,
        sampling_prob=1.0 / 100.0,
        use_connect_dots=True,
    ).self_compose(1000)
    remove_realization = dp_accounting_pmf_to_pld_realization(pmf=stage1_poisson._pmf_remove)

    loss_discretization = 0.01 / int(2 * np.ceil(np.log2(100)) + 1)
    base_dist, neg_dual_dist = realization_remove_base_distributions(
        realization=remove_realization,
        loss_discretization=loss_discretization,
        tail_truncation=(1e-8 * 0.01) / 3 / 100,
        bound_type=BoundType.DOMINATES,
        max_grid_mult=-1,
    )
    expected_neg_dual = negate_reverse_distribution(calc_pld_dual(base_dist))

    assert base_dist.p_min == pytest.approx(0.0, abs=TOL.PROBABILITY_ATOL)
    assert isinstance(base_dist, PLDRealization)
    np.testing.assert_allclose(neg_dual_dist.x_array, expected_neg_dual.x_array)
    np.testing.assert_allclose(neg_dual_dist.prob_arr, expected_neg_dual.prob_arr)
    assert neg_dual_dist.p_min == pytest.approx(expected_neg_dual.p_min, abs=TOL.PROBABILITY_ATOL)
    assert neg_dual_dist.p_max == pytest.approx(0.0, abs=TOL.PROBABILITY_ATOL)


def test_realization_remove_base_distributions_handles_is_dominated_coarsening():
    """Lower path should dualize first, then coarsen base and negated dual separately."""
    laplace_scale = 2.0 / np.sqrt(2.0)
    remove_realization = laplace_distribution(scale=laplace_scale, value_discretization=1e-4)
    loss_discretization = 0.005 / int(2 * np.ceil(np.log2(10)) + 1)
    tail_truncation = 1e-12 / 3 / (2 * 1) / (2 * 1) / 10

    base_dist, neg_dual_dist = realization_remove_base_distributions(
        realization=remove_realization,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=BoundType.IS_DOMINATED,
        max_grid_mult=-1,
    )
    exact_neg_dual = negate_reverse_distribution(calc_pld_dual(remove_realization))
    expected_base = project_discrete_dist_stoch_dom(
        dist=remove_realization,
        tail_truncation=tail_truncation,
        loss_discretization=loss_discretization,
        bound_type=BoundType.IS_DOMINATED,
    )
    expected_neg_dual = project_discrete_dist_stoch_dom(
        dist=exact_neg_dual,
        tail_truncation=tail_truncation,
        loss_discretization=loss_discretization,
        bound_type=BoundType.IS_DOMINATED,
    )

    assert isinstance(base_dist, DenseDiscreteDist)
    assert not isinstance(base_dist, PLDRealization)
    np.testing.assert_allclose(base_dist.x_array, expected_base.x_array)
    np.testing.assert_allclose(base_dist.prob_arr, expected_base.prob_arr)
    assert base_dist.p_min == pytest.approx(expected_base.p_min, abs=TOL.PROBABILITY_ATOL)
    assert base_dist.p_max == pytest.approx(expected_base.p_max, abs=TOL.PROBABILITY_ATOL)
    np.testing.assert_allclose(neg_dual_dist.x_array, expected_neg_dual.x_array)
    np.testing.assert_allclose(neg_dual_dist.prob_arr, expected_neg_dual.prob_arr)
    assert neg_dual_dist.p_min == pytest.approx(expected_neg_dual.p_min, abs=TOL.PROBABILITY_ATOL)
    assert neg_dual_dist.p_max == pytest.approx(0.0, abs=TOL.PROBABILITY_ATOL)
    total_mass = math.fsum(
        [*map(float, neg_dual_dist.prob_arr), neg_dual_dist.p_min, neg_dual_dist.p_max]
    )
    assert np.isclose(total_mass, 1.0, atol=TOL.PROBABILITY_ATOL)


def test_realization_remove_base_distributions_allows_left_tail_truncation_of_realization():
    """Lower coarsening should treat the truncated base as a plain dense distribution."""
    remove_realization = gaussian_distribution(
        scale=0.2,
        value_discretization=1e-4,
    )

    base_dist, neg_dual_dist = realization_remove_base_distributions(
        realization=remove_realization,
        loss_discretization=0.1,
        tail_truncation=1.6666666666666668e-9,
        bound_type=BoundType.IS_DOMINATED,
        max_grid_mult=-1,
    )

    assert isinstance(base_dist, DenseDiscreteDist)
    assert not isinstance(base_dist, PLDRealization)
    assert base_dist.p_min > 0.0
    assert np.isclose(
        math.fsum([*map(float, base_dist.prob_arr), base_dist.p_min, base_dist.p_max]),
        1.0,
        atol=TOL.PROBABILITY_ATOL,
    )
    assert isinstance(neg_dual_dist, DenseDiscreteDist)


def test_realization_remove_base_distributions_is_dominated_clamps_when_refining():
    """Lower path should clamp to realization.step when the requested step is finer.

    Rediscretizing to a finer target inflates the grid with interior zeros,
    making all subsequent O(N²) convolutions unnecessarily slow.
    The effective discretization is max(realization.step, loss_discretization).
    """
    remove_realization = gaussian_distribution(
        scale=2.0,
        value_discretization=1e-4,
    )
    loss_discretization = remove_realization.step / 2.0  # finer than the realization
    tail_truncation = 2.777777777777778e-16

    base_dist, neg_dual_dist = realization_remove_base_distributions(
        realization=remove_realization,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
        bound_type=BoundType.IS_DOMINATED,
        max_grid_mult=-1,
    )

    # Effective discretization is clamped to realization.step, not the finer target.
    effective_disc = max(remove_realization.step, loss_discretization)
    exact_neg_dual = negate_reverse_distribution(calc_pld_dual(remove_realization))
    # The lower path downgrades the realization's type but keeps its GridSpec, so the
    # reference must bin against the same lattice rather than an affine rebuild of it.
    plain_base = DenseDiscreteDist(
        grid=remove_realization.grid,
        prob_arr=remove_realization.prob_arr.copy(),
        p_min=remove_realization.p_min,
        p_max=remove_realization.p_max,
    )
    expected_base = project_discrete_dist_stoch_dom(
        dist=plain_base,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        bound_type=BoundType.IS_DOMINATED,
    )
    expected_neg_dual = project_discrete_dist_stoch_dom(
        dist=exact_neg_dual,
        tail_truncation=tail_truncation,
        loss_discretization=effective_disc,
        bound_type=BoundType.IS_DOMINATED,
    )

    np.testing.assert_allclose(base_dist.x_array, expected_base.x_array)
    np.testing.assert_allclose(base_dist.prob_arr, expected_base.prob_arr)
    assert base_dist.p_min == pytest.approx(expected_base.p_min, abs=TOL.PROBABILITY_ATOL)
    assert base_dist.p_max == pytest.approx(0.0, abs=TOL.PROBABILITY_ATOL)
    np.testing.assert_allclose(neg_dual_dist.x_array, expected_neg_dual.x_array)
    np.testing.assert_allclose(neg_dual_dist.prob_arr, expected_neg_dual.prob_arr)
    assert neg_dual_dist.p_min == pytest.approx(expected_neg_dual.p_min, abs=TOL.PROBABILITY_ATOL)
    assert neg_dual_dist.p_max == pytest.approx(expected_neg_dual.p_max, abs=TOL.PROBABILITY_ATOL)


class TestRealizationAdapter:
    """Test conversion between PLDRealization and DensePLDPmf."""

    def test_dense_linear_to_dense_pmf(self):
        """Test that PLDRealization converts to DensePLDPmf."""
        grid = GridSpec(step=0.5, n=4, anchor=0.0)
        realization = PLDRealization(grid=grid, prob_arr=np.array([0.2, 0.3, 0.4, 0.1]))

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, bound_type=BoundType.DOMINATES)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert np.allclose(pmf._probs, realization.prob_arr)

    def test_dense_linear_with_nonzero_base(self):
        """Test PLDRealization with non-zero x_min."""
        grid = GridSpec(step=0.5, n=3, index_0=2)
        realization = PLDRealization(grid=grid, prob_arr=np.array([0.3, 0.4, 0.3]))

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, bound_type=BoundType.DOMINATES)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert pmf._lower_loss == 2  # index_0 exports unchanged

    def test_dense_linear_with_infinity_mass(self):
        """Test PLDRealization with p_max."""
        grid = GridSpec(step=0.25, n=3, anchor=0.0)
        realization = PLDRealization(grid=grid, prob_arr=np.array([0.2, 0.5, 0.2]), p_max=0.1)

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, bound_type=BoundType.DOMINATES)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._infinity_mass == 0.1

    def test_dense_linear_roundtrip(self):
        """Test PLDRealization -> PMF -> PLDRealization roundtrip."""
        grid = GridSpec(step=0.25, n=4, index_0=2)
        realization = PLDRealization(grid=grid, prob_arr=np.array([0.2, 0.3, 0.4, 0.1]))

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, bound_type=BoundType.DOMINATES)
        restored = dp_accounting_pmf_to_pld_realization(pmf=pmf)

        assert isinstance(restored, PLDRealization)
        assert np.allclose(realization.x_array, restored.x_array)
        assert np.allclose(realization.prob_arr, restored.prob_arr)
        assert np.isclose(realization.p_max, restored.p_max)

    def test_dominating_conversion_rejects_negative_infinity_mass(self):
        """A dominating dp_accounting PMF cannot represent source mass at -infinity."""
        grid = GridSpec(step=1.0, n=1, anchor=0.3)
        dist = DenseDiscreteDist(grid=grid, prob_arr=np.array([0.9]), p_min=0.1)

        with pytest.raises(ValueError, match="requires p_min = 0"):
            linear_dist_to_dp_accounting_pmf(dist=dist, bound_type=BoundType.DOMINATES)

    def test_linear_dist_to_dp_accounting_rejects_non_linear_dist(self):
        """Linear dist to dp accounting rejects non linear dist."""
        dist = SparseDiscreteDist(x_array=np.array([0.0, 0.5]), prob_arr=np.array([0.5, 0.5]))
        with pytest.raises(TypeError, match="must be DenseDiscreteDist with LINEAR spacing"):
            linear_dist_to_dp_accounting_pmf(dist=dist, bound_type=BoundType.DOMINATES)


def _make_realization() -> PLDRealization:
    grid = GridSpec(step=0.5, n=4, index_0=-1)
    return PLDRealization(
        grid=grid, prob_arr=np.array([0.2, 0.3, 0.25, 0.15], dtype=np.float64), p_max=0.1
    )


def test_dp_accounting_still_exposes_the_private_fields_the_adapter_reads():
    """The adapter reads private dp_accounting fields; a release that renames one fails here."""
    pld = dp_pld.from_gaussian_mechanism(
        standard_deviation=1.0, sampling_prob=0.5, value_discretization_interval=1e-2
    )
    for name in ("_pmf_remove", "_pmf_add"):
        assert hasattr(pld, name), name
    dense = pld._pmf_remove
    assert isinstance(dense, DensePLDPmf)
    for name in ("_discretization", "_lower_loss", "_probs", "_infinity_mass"):
        assert hasattr(dense, name), name
    assert getattr(dense, "_pessimistic_estimate") is True
    sparse = SparsePLDPmf(
        loss_probs={0: 1.0}, discretization=0.1, infinity_mass=0.0, pessimistic_estimate=True
    )
    for name in ("_discretization", "_loss_probs", "_infinity_mass", "_pessimistic_estimate"):
        assert hasattr(sparse, name), name
