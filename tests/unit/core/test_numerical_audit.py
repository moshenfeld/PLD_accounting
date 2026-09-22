"""Regression tests for the numerical-audit producer fixes."""

from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest

from PLD_accounting.discrete_dist import (
    REALIZATION_MOMENT_TOL,
    DenseDiscreteDist,
    Domain,
    GridSpec,
    PLDRealization,
)
from PLD_accounting.distribution_discretization import _ctd_moment_repair_tol
from PLD_accounting.distribution_utils import (
    PMF_MASS_DRIFT_TOL,
    PMF_TOLERATED_MASS_TOL,
    _drain_mass_from_edge,
    classify_residual,
    compensated_segmented_sum,
    exp_moment_terms,
    signed_unit_residual,
    trim_mass_to_moment_target,
)
from PLD_accounting.dp_accounting_support import (
    DP_ACCOUNTING_MASS_REPAIR_TOL,
    linear_dist_to_dp_accounting_pmf,
)
from PLD_accounting.geometric_convolution import (
    _add_single_zero_atom_cross_term,
    _numba_geometric_kernel,
    _numpy_geometric_kernel,
    geometric_convolve,
    geometric_self_convolve,
)
from PLD_accounting.mechanisms import gaussian_distribution
from PLD_accounting.random_allocation_accounting import (
    _align_component_grids,
    _averaged_exp_factor,
)
from PLD_accounting.random_allocation_api import (
    gaussian_allocation_directional_pld,
    gaussian_allocation_pld,
)
from PLD_accounting.subsample_pld import (
    _atomic_source,
    _calc_subsampled_grid,
    _extend_target_grid_for_reference,
    _stable_subsampling_transformation,
    subsample_pld_realization,
)
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
    SpacingType,
)
from PLD_accounting.utils import (
    calc_pld_dual,
    combine_best_of_two_plds,
    negate_reverse_linear_distribution,
)


def _reference_segmented_sum(
    bin_index: np.ndarray,
    weights: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    totals = np.zeros(num_bins, dtype=np.float64)
    order = np.argsort(bin_index, kind="mergesort")
    sorted_bins = bin_index[order]
    sorted_weights = weights[order]
    start = 0
    for end in np.concatenate((np.flatnonzero(np.diff(sorted_bins)) + 1, [sorted_weights.size])):
        bin_id = int(sorted_bins[start])
        totals[bin_id] = math.fsum(map(float, sorted_weights[start:end]))
        start = int(end)
    return totals


def test_compensated_segmented_sum_matches_fsum_reference() -> None:
    """Coalescing helper must match per-bin math.fsum on adversarial weights."""
    weights = np.array([1.0, 1e-16] * 5000 + [1.0], dtype=np.float64)
    bin_index = np.zeros(weights.size, dtype=np.intp)
    result = compensated_segmented_sum(bin_index=bin_index, weights=weights, num_bins=1)
    expected = _reference_segmented_sum(bin_index, weights, 1)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-12)
    raw = np.bincount(bin_index, weights=weights, minlength=1)
    assert raw[0] != expected[0]


def test_compensated_segmented_sum_rejects_out_of_range_indices() -> None:
    """Bin indices outside the declared range are a caller error, not clamped."""
    weights = np.array([0.5, 0.5], dtype=np.float64)
    bin_index = np.array([0, 2], dtype=np.intp)
    with pytest.raises(ValueError, match="bin_index out of range"):
        compensated_segmented_sum(bin_index=bin_index, weights=weights, num_bins=2)


def test_atomic_source_bincount_residual_within_tolerance() -> None:
    """Producer-stage coalescing must land inside the mass contract without the adapter."""
    losses = np.array([0.0, 0.0, 0.1, 0.1, 0.2], dtype=np.float64)
    masses = np.array([0.4, 0.4, 1e-16, 1.0 - 0.8 - 1e-16, 0.0], dtype=np.float64)
    source = _atomic_source(losses=losses, masses=masses, p_min=0.0, p_max=0.0)
    residual = signed_unit_residual(
        values=source.prob_arr,
        lower_term=source.p_min,
        upper_term=source.p_max,
    )
    assert abs(residual) <= PMF_TOLERATED_MASS_TOL


def test_truncation_direct_adversarial_leading_zeros_and_near_boundary_moment() -> None:
    """Truncation repair must be driven by measured moment, not constructor exceptions."""
    step = 1e-3
    prob_arr = np.concatenate(
        [np.zeros(100, dtype=np.float64), np.array([0.35, 0.65], dtype=np.float64)]
    )
    realization = PLDRealization(
        grid=GridSpec(
            step=step,
            n=prob_arr.size,
            anchor=0.0,
        ),
        prob_arr=prob_arr,
    )
    truncated = realization.truncate_edges(tail_truncation=0.0, bound_type=BoundType.DOMINATES)
    assert truncated.prob_arr.size == 2
    assert truncated.x_0 == pytest.approx(0.1)
    assert truncated.p_min == 0.0


def test_moment_trim_reports_actual_removal() -> None:
    """Returned removal must equal the stored finite mass difference."""
    loss = np.array([-19.132, -1.0, 0.0], dtype=np.float64)
    prob = np.array([1e-12, 0.2, 0.8 - 1e-12], dtype=np.float64)
    repaired, removed = trim_mass_to_moment_target(prob_arr=prob, loss=loss, p_max=0.0)
    assert removed == pytest.approx(math.fsum(map(float, prob - repaired)))


def test_moment_trim_rejects_a_removal_beyond_the_cap() -> None:
    """A moved-mass cap must abort the repair, not report it after the fact."""
    loss = np.array([-19.132, -1.0, 0.0], dtype=np.float64)
    prob = np.array([1e-12, 0.2, 0.8 - 1e-12], dtype=np.float64)
    with pytest.raises(ValueError, match="above the .* cap"):
        trim_mass_to_moment_target(
            prob_arr=prob, loss=loss, p_max=0.0, max_removed=1e-30, context="probe"
        )


def test_moment_trim_rejects_an_unsatisfiable_target_instead_of_emptying_the_law() -> None:
    """A target requiring every finite atom must raise instead of returning delta one."""
    prob = np.array([0.4, 0.6], dtype=np.float64)
    loss = np.array([0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="exceeds the law's total moment"):
        trim_mass_to_moment_target(
            prob_arr=prob,
            loss=loss,
            p_max=0.0,
            target_residual=1.0,
        )


def test_moment_trim_reaches_the_target_in_a_single_pass() -> None:
    """The trim has no retry loop, so one pass must land on or past the target."""
    base = gaussian_distribution(scale=0.8, value_discretization=1e-2, tail_truncation=1e-10)
    mass, loss = base.prob_arr.copy(), base.x_array.copy()
    moment = 1.0 - signed_unit_residual(
        values=exp_moment_terms(prob_arr=mass, x_vals=loss), lower_term=0.0, upper_term=0.0
    )
    for eta in (1e-16, 1e-15, 1e-13, 1e-11, 1e-9, 1e-7):
        nudged = mass * ((1.0 + eta) / moment)
        repaired, _p_max = trim_mass_to_moment_target(prob_arr=nudged, loss=loss, p_max=0.0)
        residual = signed_unit_residual(
            values=exp_moment_terms(prob_arr=repaired, x_vals=loss),
            lower_term=0.0,
            upper_term=0.0,
        )
        assert residual >= 0.0, f"single pass fell short for eta={eta:g}"


def test_align_component_grids_reconciles_last_bit_steps_quietly() -> None:
    """Dividing one common_step by different stage counts is not worth a warning."""
    step_a = 0.01
    step_b = float(np.nextafter(step_a, 2.0))
    dist_floor = DenseDiscreteDist(
        grid=GridSpec(
            step=step_a,
            n=2,
            anchor=0.0,
        ),
        prob_arr=np.array([0.5, 0.5]),
    )
    dist_ceil = DenseDiscreteDist(
        grid=GridSpec(
            step=step_b,
            n=2,
            anchor=0.0,
        ),
        prob_arr=np.array([0.25, 0.75]),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        floor_out, ceil_out = _align_component_grids(
            dist_floor=dist_floor,
            dist_ceil=dist_ceil,
            bound_type=BoundType.DOMINATES,
        )

    assert floor_out.step == ceil_out.step


def test_align_component_grids_reports_a_material_step_mismatch() -> None:
    """A cap-coarsened component is a different grid, not float noise, so it is reported."""
    dist_floor = DenseDiscreteDist(
        grid=GridSpec(
            step=0.01,
            n=2,
            anchor=0.0,
        ),
        prob_arr=np.array([0.5, 0.5]),
    )
    dist_ceil = DenseDiscreteDist(
        grid=GridSpec(
            step=0.05,
            n=2,
            anchor=0.0,
        ),
        prob_arr=np.array([0.25, 0.75]),
    )

    with pytest.warns(UserWarning, match="aligning mismatched floor/ceil grids"):
        floor_out, ceil_out = _align_component_grids(
            dist_floor=dist_floor,
            dist_ceil=dist_ceil,
            bound_type=BoundType.DOMINATES,
        )

    assert floor_out.step == ceil_out.step


def test_geometric_convolve_reads_anchors_from_the_input_grids() -> None:
    """A caller cannot declare a lattice the data does not have.

    The predecessor accepted ``source_anchor``/``target_anchor`` certificates beside the
    distributions and had to detect the ones that lied. The anchor is now a field of the
    input grid, so the output anchor is the sum of what the inputs actually carry.
    """
    dist = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            n=2,
            anchor=0.6,
        ),
        prob_arr=np.array([0.5, 0.5], dtype=np.float64),
        domain=Domain.POSITIVES,
    )
    assert dist.grid.anchor == 0.6
    result = geometric_convolve(
        dist_1=dist,
        dist_2=dist,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
    )
    assert result.grid.anchor == 1.2
    assert result.x_0 == pytest.approx(1.2)


def test_geometric_convolve_keeps_a_deep_index_hit_exact() -> None:
    """A same-index sum stays exact at a lattice index a log recovery would miss."""
    anchor = 1e-5
    ratio = 1.0001167359166174
    lattice_index = 500
    dist = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(ratio),
            n=4,
            spacing_type=SpacingType.GEOMETRIC,
            anchor=anchor,
            index_0=lattice_index,
        ),
        prob_arr=np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
        domain=Domain.POSITIVES,
    )
    result = geometric_convolve(
        dist_1=dist,
        dist_2=dist,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
    )
    assert result.grid.index_0 == lattice_index
    assert result.grid.anchor == pytest.approx(2.0 * anchor)
    assert result.x_0 == pytest.approx(2.0 * dist.x_0)


def test_geometric_self_convolve_doubles_the_input_anchor() -> None:
    """Self-convolution derives its lattice; there is no certificate to disbelieve."""
    dist = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            n=2,
            anchor=0.6,
        ),
        prob_arr=np.array([0.5, 0.5], dtype=np.float64),
        domain=Domain.POSITIVES,
    )
    result = geometric_self_convolve(
        dist=dist,
        num_convolutions=2,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
    )
    assert result.grid.anchor == 1.2


def test_combine_best_of_two_projects_when_anchor_embedding_shifts() -> None:
    """A one-ULP left extension must take the projection branch, not silent drift."""
    anchor = DenseDiscreteDist(
        grid=GridSpec(
            step=1.0,
            n=2,
            anchor=float(np.nextafter(0.0, 1.0)),
        ),
        prob_arr=np.array([0.4, 0.6], dtype=np.float64),
    )
    other = DenseDiscreteDist(
        grid=GridSpec(
            step=1.0,
            n=2,
            anchor=0.0,
        ),
        prob_arr=np.array([0.3, 0.7], dtype=np.float64),
    )
    result = combine_best_of_two_plds(
        dist_1=anchor,
        dist_2=other,
        bound_type=BoundType.DOMINATES,
    )
    assert result.prob_arr.size >= anchor.prob_arr.size


@pytest.mark.parametrize("num_steps", [2, 10, 17])
@pytest.mark.parametrize("direction", [Direction.REMOVE, Direction.ADD])
def test_composed_loss_grid_is_exactly_zero_anchored(num_steps: int, direction: Direction) -> None:
    """The composed loss lattice must be zero-anchored for every composition count."""
    config = AllocationSchemeConfig(
        loss_discretization=1e-2,
        tail_truncation=1e-10,
        max_grid_mult=20_000,
        convolution_method=ConvolutionMethod.GEOM,
    )
    params = PrivacyParams(sigma=1.0, num_steps=num_steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = gaussian_allocation_directional_pld(
            params=params,
            config=config,
            direction=direction,
            bound_type=BoundType.DOMINATES,
        )
    assert (
        result.grid.anchor == 0.0
    ), f"composed loss grid must be zero-anchored, got {result.grid.anchor!r}"


def test_averaged_exp_factor_rejects_a_non_unit_anchored_composition() -> None:
    """The averaging guard names the zero-anchor precondition it depends on."""
    dist = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            n=2,
            spacing_type=SpacingType.GEOMETRIC,
            anchor=1.5,
        ),
        prob_arr=np.array([0.5, 0.5], dtype=np.float64),
        domain=Domain.POSITIVES,
    )
    with pytest.raises(ValueError, match="were not zero-anchored"):
        _averaged_exp_factor(dist=dist, num_steps=2)


def test_dp_accounting_import_mass_band_admits_observed_residual() -> None:
    """The repair ceiling must clear the residuals an uncomposed Gaussian import makes.

    Measured over 288 uncomposed PMFs: p90 4.1e-7, max 4.8e-6. The ceiling sits above
    the max so none is rejected; the drift floor stays at noise so every real repair
    is reported rather than normalized away.
    """
    assert DP_ACCOUNTING_MASS_REPAIR_TOL >= 4.9e-6
    assert DP_ACCOUNTING_MASS_REPAIR_TOL > PMF_MASS_DRIFT_TOL


def test_zero_anchored_materialization_matches_global_index_formula() -> None:
    """Zero-anchored linear grids must not drift from ``(index_0 + i) * step``."""
    grid = GridSpec(step=1e-6, n=3, index_0=-1_000_000)
    materialized = grid.materialize()
    assert materialized[0] == grid.index_0 * grid.step
    assert materialized[1] == (grid.index_0 + 1) * grid.step
    sliced = grid.slice(start=1, n=2)
    assert sliced.index_0 == grid.index_0 + 1
    np.testing.assert_array_equal(sliced.materialize(), materialized[1:3])


def test_geometric_truncation_preserves_bitwise_points() -> None:
    """Geometric truncation must update integer exponents, not re-derive origins."""
    grid = GridSpec(step=1.01, n=8, spacing_type=SpacingType.GEOMETRIC, anchor=1.0, index_0=-5)
    materialized = grid.materialize()
    sliced = grid.slice(start=2, n=3)
    assert sliced.index_0 == grid.index_0 + 2
    np.testing.assert_array_equal(sliced.materialize(), materialized[2:5])


def test_nextafter_anchor_never_collapses_onto_the_zero_anchored_lattice() -> None:
    """An anchor one ULP off ``k * step`` stays its own lattice."""
    step = 1.0
    grid = GridSpec(step=step, n=2, anchor=float(np.nextafter(0.0, 1.0)))
    assert grid.anchor != 0.0
    assert grid != GridSpec(step=step, n=2)
    # Equality is decided on the five fields, without allocating either coordinate array.
    assert grid.slice(start=1, n=1) != GridSpec(step=step, n=1, index_0=1)


def test_dp_accounting_export_rejects_anchors_the_quotient_cannot_see() -> None:
    """The two anchors that defeated the old quotient correction are now refused outright.

    Regression: the export index was derived as ``x_0 / step``. An anchor below half an
    ULP of ``index_0 * step`` vanishes in that sum, and an anchor one ULP above a lattice
    point leaves the quotient a whole number -- both made a dominating export return the
    *unshifted* index, a non-conservative bound. Requiring ``anchor == 0`` removes the
    derivation entirely, so neither fixture can reach the adapter.
    """
    step = 1e-4
    swallowed = GridSpec(
        step=step,
        n=3,
        anchor=float(np.spacing(100_000 * step) / 4),  # > 0, invisible in x_0
        index_0=100_000,
    )
    whole_step = 0.0005640745848883568
    whole_quotient = GridSpec(
        step=whole_step,
        n=3,
        anchor=float(np.nextafter(368 * whole_step, np.inf)),  # anchor/step == 368.0
        index_0=0,
    )
    assert whole_quotient.anchor / whole_step == 368.0

    for grid in (swallowed, whole_quotient):
        dist = PLDRealization(grid=grid, prob_arr=np.array([0.0, 0.0, 1.0]))
        for bound_type in (BoundType.DOMINATES, BoundType.IS_DOMINATED):
            with pytest.raises(ValueError, match="whole multiples of its step"):
                linear_dist_to_dp_accounting_pmf(dist=dist, bound_type=bound_type)


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_every_geometric_diagonal_sum_is_placed_conservatively(bound_type: BoundType) -> None:
    """Each same-index sum must be bounded in its own direction, not just the first."""
    step = 3.9915989732929734e-05
    index_0 = -9017
    n = 8
    grid_1 = GridSpec(
        step=step,
        n=n,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=0.004066111046665461,
        index_0=index_0,
    )
    grid_2 = GridSpec(
        step=step,
        n=n,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=0.0108584022381824,
        index_0=index_0,
    )

    for i in range(n):
        mass = np.zeros(n)
        mass[i] = 1.0
        result = geometric_convolve(
            dist_1=DenseDiscreteDist(grid=grid_1, prob_arr=mass, domain=Domain.POSITIVES),
            dist_2=DenseDiscreteDist(grid=grid_2, prob_arr=mass, domain=Domain.POSITIVES),
            tail_truncation=0.0,
            bound_type=bound_type,
        )
        placed = float(result.x_array[np.flatnonzero(result.prob_arr)][0])
        true_sum = grid_1.point(i=i) + grid_2.point(i=i)
        if bound_type == BoundType.DOMINATES:
            assert placed >= true_sum, f"diagonal {i} placed below its sum"
        else:
            assert placed <= true_sum, f"diagonal {i} placed above its sum"


def test_geometric_convolve_compares_stored_log_steps() -> None:
    """Distinct log steps that exponentiate to one ratio are still distinct lattices."""
    step = 1e-8
    other = float(np.nextafter(step, 1.0))
    assert math.exp(step) == math.exp(other)  # indistinguishable as ratios

    def make(spacing: float) -> DenseDiscreteDist:
        return DenseDiscreteDist(
            grid=GridSpec(step=spacing, n=2, spacing_type=SpacingType.GEOMETRIC, anchor=1.0),
            prob_arr=np.array([0.5, 0.5]),
            domain=Domain.POSITIVES,
        )

    with pytest.raises(ValueError, match="must share one exact log spacing"):
        geometric_convolve(
            dist_1=make(step),
            dist_2=make(other),
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )


@pytest.mark.parametrize(
    ("prob", "loss", "eta"),
    [
        ([1.0], [0.0], 1e-20),
        ([1.0], [0.0], 1e-30),
        ([0.5, 0.5], [0.0, 1e-12], 1e-20),
    ],
)
def test_lower_semantic_dual_mass_postcondition_holds(
    prob: list[float], loss: list[float], eta: float
) -> None:
    """The produced realization must actually carry eta, not merely try to."""
    prob_arr = np.array(prob, dtype=np.float64)
    loss_arr = np.array(loss, dtype=np.float64)

    adjusted, removed = trim_mass_to_moment_target(
        prob_arr=prob_arr,
        loss=loss_arr,
        p_max=0.0,
        target_residual=eta,
    )

    residual = signed_unit_residual(
        values=exp_moment_terms(prob_arr=adjusted, x_vals=loss_arr), lower_term=0.0, upper_term=0.0
    )
    assert residual >= eta
    assert removed == pytest.approx(math.fsum(map(float, prob_arr - adjusted)))


def test_lower_semantic_dual_mass_continues_past_an_exhausted_subnormal() -> None:
    """Surrender must keep walking after a subnormal atom is driven to zero."""
    subnormal = float(np.nextafter(0.0, 1.0))
    # Start from a valid residual so the walk is the thing under test, not the
    # "already exceeds one" guard. The subnormal cannot supply eta by itself.
    prob_arr = np.array([subnormal, 0.5], dtype=np.float64)
    loss_arr = np.array([0.0, 0.0], dtype=np.float64)
    eta = 0.5 + 6.0e-17

    adjusted, removed = trim_mass_to_moment_target(
        prob_arr=prob_arr,
        loss=loss_arr,
        p_max=0.0,
        target_residual=eta,
    )

    residual = signed_unit_residual(
        values=exp_moment_terms(prob_arr=adjusted, x_vals=loss_arr), lower_term=0.0, upper_term=0.0
    )
    assert residual >= eta
    assert adjusted[0] == 0.0
    assert removed > 0.0


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_diagonal_placement_is_per_element(bound_type: BoundType) -> None:
    """Elements that land off their knot get their own bin, not the whole diagonal."""
    step = 3.9915989732929734e-05
    index_0 = -9017
    n = 8
    grid_1 = GridSpec(
        step=step,
        n=n,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=0.004066111046665461,
        index_0=index_0,
    )
    grid_2 = GridSpec(
        step=step,
        n=n,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=0.0108584022381824,
        index_0=index_0,
    )
    result = geometric_convolve(
        dist_1=DenseDiscreteDist(
            grid=grid_1, prob_arr=np.full(n, 1.0 / n), domain=Domain.POSITIVES
        ),
        dist_2=DenseDiscreteDist(
            grid=grid_2, prob_arr=np.full(n, 1.0 / n), domain=Domain.POSITIVES
        ),
        tail_truncation=0.0,
        bound_type=bound_type,
    )

    # The anchor is the untouched exact sum: correctness came from placement, not scaling.
    assert result.grid.anchor == grid_1.anchor + grid_2.anchor
    assert result.p_max == 0.0
    assert result.prob_arr.size >= n
    for i in range(n):
        mass = np.zeros(n)
        mass[i] = 1.0
        placed = geometric_convolve(
            dist_1=DenseDiscreteDist(grid=grid_1, prob_arr=mass, domain=Domain.POSITIVES),
            dist_2=DenseDiscreteDist(grid=grid_2, prob_arr=mass, domain=Domain.POSITIVES),
            tail_truncation=0.0,
            bound_type=bound_type,
        )
        knot = float(placed.x_array[np.flatnonzero(placed.prob_arr)][0])
        true_sum = grid_1.point(i=i) + grid_2.point(i=i)
        if bound_type == BoundType.DOMINATES:
            assert knot >= true_sum
        else:
            assert knot <= true_sum
        # The uniform-PMF result must keep that knot on a finite bin, not overflow it.
        assert np.any(np.isclose(result.x_array, knot, rtol=0.0, atol=0.0))


def test_classify_residual_rejects_inverted_tolerance_order() -> None:
    """Drift tolerance must not exceed repair tolerance."""
    with pytest.raises(ValueError, match="require 0 <= drift_tol <= repair_tol"):
        classify_residual(
            residual=2e-15,
            drift_tol=REALIZATION_MOMENT_TOL,
            repair_tol=REALIZATION_MOMENT_TOL / 2.0,
            context="test",
            repair="noop",
        )


def test_drain_mass_from_edge_handles_fsum_cumsum_pivot_disagreement() -> None:
    """Exact trim must not index past the last element when totals disagree."""
    values = np.array([1.0, *([1e-16] * 100)], dtype=np.float64)
    values[0] = float(np.nextafter(values[0], 2.0))
    total_fsum = math.fsum(map(float, values))
    total_cumsum = float(np.cumsum(values)[-1])
    assert total_fsum > total_cumsum
    mass = float(np.nextafter(total_cumsum, total_fsum))
    assert total_cumsum <= mass < total_fsum
    trimmed = _drain_mass_from_edge(values=values.copy(), mass=mass, from_left=True, exact=True)
    assert math.fsum(map(float, trimmed)) == pytest.approx(total_fsum - mass, abs=1e-14)


def test_zero_atom_cross_term_uses_stored_log_step() -> None:
    """A lattice hit must stay on its knot when log spacing is used directly."""
    log_step = math.log(2.0)
    output_grid = GridSpec(
        step=log_step,
        n=5,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=1.0,
        index_0=0,
    )
    x_on_lattice = float(output_grid.point(i=2))
    pmf = np.zeros(5)
    result, below, above = _add_single_zero_atom_cross_term(
        pmf_conv=pmf,
        x_arr=np.array([x_on_lattice]),
        prob_arr=np.array([1.0]),
        zero_prob=0.25,
        output_grid=output_grid,
        bound_type=BoundType.DOMINATES,
    )
    assert below == 0.0
    assert above == 0.0
    assert result[2] == pytest.approx(0.25)


def test_geometric_kernel_numba_and_numpy_paths_agree_within_machine_eps() -> None:
    """The Numba and NumPy geometric kernels must stay observationally equivalent."""
    pmf = np.array([0.2, 0.3, 0.25, 0.25], dtype=np.float64)
    delta_lohi = np.array([0, 1, 2, 3], dtype=np.int64)
    delta_hilo = np.array([0, 1, 2, 3], dtype=np.int64)
    diagonal_bins = np.arange(pmf.size, dtype=np.int64)
    numba_out = _numba_geometric_kernel(
        pmf_base=pmf,
        pmf_scaled=pmf,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        diagonal_bins=diagonal_bins,
        output_size=8,
    )
    numpy_out = _numpy_geometric_kernel(
        pmf_base=pmf,
        pmf_scaled=pmf,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        diagonal_bins=diagonal_bins,
        output_size=8,
    )
    assert np.max(np.abs(numba_out - numpy_out)) <= 2e-17


def test_remove_subsample_extends_target_grid_for_dual_branch() -> None:
    """The REMOVE path must pad the lattice before mixing transformed branches."""
    base = gaussian_distribution(scale=2.0, value_discretization=0.5, tail_truncation=1e-8)
    dual = negate_reverse_linear_distribution(calc_pld_dual(base))
    sampling_prob = 0.4
    base_only = _calc_subsampled_grid(
        source_grid=base.grid,
        sampling_prob=sampling_prob,
        direction=Direction.REMOVE,
        include_right=None,
    )
    extended = _extend_target_grid_for_reference(
        target_grid=base_only,
        neg_dual_pld=dual,
        sampling_prob=sampling_prob,
        direction=Direction.REMOVE,
    )
    assert extended.n >= base_only.n
    assert extended.anchor == 0.0, "padding is an index operation, never an anchor shift"
    ref_endpoints = _stable_subsampling_transformation(
        x_array=np.array([dual.x_array[0], dual.x_array[-1]], dtype=np.float64),
        sampling_prob=sampling_prob,
        direction=Direction.REMOVE,
    )
    assert extended.x_0 <= float(np.min(ref_endpoints))
    assert extended.last_point >= float(np.max(ref_endpoints))

    result = subsample_pld_realization(
        base_pld=base,
        sampling_prob=sampling_prob,
        direction=Direction.REMOVE,
    )
    assert result.grid.n >= extended.n - 1
    assert result.x_array[0] <= float(np.min(ref_endpoints)) + 1e-12
    assert result.x_array[-1] >= float(np.max(ref_endpoints)) - 1e-12


_PRODUCTION_PARAMS = PrivacyParams(
    sigma=1.0,
    num_steps=100,
    num_selected=1,
    num_epochs=5,
    delta=1e-10,
)
_PRODUCTION_CONFIG = AllocationSchemeConfig(
    loss_discretization=1e-3,
    tail_truncation=1e-10,
    max_grid_mult=20_000,
    convolution_method=ConvolutionMethod.GEOM,
)


def test_gaussian_internal_routes_emit_no_repair_warnings() -> None:
    """Normal Gaussian allocation paths must not lean on generic repair fallbacks."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for method in (
            ConvolutionMethod.GEOM,
            ConvolutionMethod.FFT,
            ConvolutionMethod.BEST_OF_TWO,
        ):
            config = replace(_PRODUCTION_CONFIG, convolution_method=method)
            for direction in (Direction.REMOVE, Direction.ADD):
                dist = gaussian_allocation_directional_pld(
                    params=_PRODUCTION_PARAMS,
                    config=config,
                    direction=direction,
                )
                dist.truncate_edges(
                    tail_truncation=config.tail_truncation,
                    bound_type=BoundType.DOMINATES,
                )
        base = gaussian_distribution(scale=1.0, value_discretization=1e-3, tail_truncation=1e-10)
        _ = calc_pld_dual(base)


def test_gaussian_floor_ceil_combination_emits_no_repair_warnings() -> None:
    """Combining the two allocation components must not lean on a mass-repair fallback.

    ``num_selected > 1`` with a remainder is the only route that builds both a floor and a
    ceil component and combines them. It does fire one declared fallback: the components
    divide one ``common_step`` by different stage counts (33 versus 34 here), so
    ``_align_component_grids`` coarsens the finer one and says so. That alignment is
    pinned here rather than tolerated silently; no probability repair may fire alongside it.
    """
    params = replace(_PRODUCTION_PARAMS, num_steps=101, num_selected=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for direction in (Direction.REMOVE, Direction.ADD):
            with pytest.warns(UserWarning, match="aligning mismatched floor/ceil grids"):
                dist = gaussian_allocation_directional_pld(
                    params=params,
                    config=_PRODUCTION_CONFIG,
                    direction=direction,
                )
            dist.truncate_edges(
                tail_truncation=_PRODUCTION_CONFIG.tail_truncation,
                bound_type=BoundType.DOMINATES,
            )


def test_dp_accounting_round_trip_emits_no_repair_warnings() -> None:
    """Exporting and re-importing a production PLD must stay inside the mass contract."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        pld = gaussian_allocation_pld(params=_PRODUCTION_PARAMS, config=_PRODUCTION_CONFIG)
        epsilon = float(pld.get_epsilon_for_delta(1e-10))
        assert math.isfinite(epsilon)


def test_ctd_moment_repair_respects_coarse_grid_tolerance_order() -> None:
    """Coarse-grid repair tolerance must cap drift tolerance, not invert it."""
    repair_tol = _ctd_moment_repair_tol(max_abs_loss=1.0, step=10.0)
    assert repair_tol < REALIZATION_MOMENT_TOL
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        classify_residual(
            residual=repair_tol / 2.0,
            drift_tol=min(REALIZATION_MOMENT_TOL, repair_tol),
            repair_tol=repair_tol,
            context="CtD reciprocal-moment repair",
            repair="moving the cheapest mass to +inf",
        )
