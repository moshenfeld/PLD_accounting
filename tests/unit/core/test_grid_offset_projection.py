"""Projection of off-grid positions onto neighbouring grid points.

The two kinds of rounding, on positions given in units of grid points:
connect-the-dots splits a value between both neighbours with its mean kept, and
stochastic domination moves it to the neighbour that the bound requires.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from PLD_accounting.discrete_dist import DiscreteDistBase
from PLD_accounting.distribution_utils import exact_sum, exp_moment_terms
from PLD_accounting.mechanisms import gaussian_loss_law
from PLD_accounting.projections import project_grid_offsets, truncate_edges
from PLD_accounting.projections_ctd import (
    project_continuous_dist_ctd,
    project_grid_offsets_ctd,
    truncate_edges_ctd,
    truncate_tails_ctd,
)
from PLD_accounting.projections_stoch_dom import (
    project_grid_offsets_stoch_dom,
    truncate_edges_stoch_dom,
)
from PLD_accounting.types import BoundType, ProjectionMethod

_OFFSETS = np.array([-2.75, -1.0, -0.25, 0.0, 0.5, 3.0, 7.999])


@pytest.mark.parametrize("log_step", [1e-3, 0.3, 2.0])
def test_connect_the_dots_split_keeps_the_value(log_step: float):
    """The two shares of a value, at its neighbouring geometric grid points, average to it."""
    lower, shares = project_grid_offsets_ctd(grid_offsets=_OFFSETS, log_step=log_step)

    np.testing.assert_array_equal(lower, np.floor(_OFFSETS))
    assert np.all((shares >= 0.0) & (shares <= 1.0))
    np.testing.assert_array_equal(shares[_OFFSETS == np.floor(_OFFSETS)], 0.0)
    split_mean = (1.0 - shares) * np.exp(lower * log_step) + shares * np.exp((lower + 1) * log_step)
    np.testing.assert_allclose(split_mean, np.exp(_OFFSETS * log_step), rtol=1e-14)


def test_stochastic_domination_rounds_to_the_side_of_the_bound():
    """An upper bound takes the grid point above a value and a lower bound the one below."""
    upper = project_grid_offsets_stoch_dom(grid_offsets=_OFFSETS, bound_type=BoundType.DOMINATES)
    lower = project_grid_offsets_stoch_dom(grid_offsets=_OFFSETS, bound_type=BoundType.IS_DOMINATED)

    np.testing.assert_array_equal(upper, [-2, -1, 0, 0, 1, 3, 8])
    np.testing.assert_array_equal(lower, [-3, -1, -1, 0, 0, 3, 7])


def test_connect_the_dots_trim_keeps_mass_and_keeps_the_mean_below():
    """The lower trim leaves the first moment at the first kept point; the upper one drops it."""
    values = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    prob_arr = np.array([1e-4, 0.5, 0.4996, 2e-4, 1e-4])

    first_kept, kept, p_min = truncate_tails_ctd(
        values=values, prob_arr=prob_arr, p_min=0.0, moment_scale=4.0, tail_truncation=3e-4
    )

    # The two upper points together exceed the charge on the first moment; one fits.
    assert first_kept == 1 and kept.size == 3
    assert math.fsum(kept.tolist()) + p_min == pytest.approx(1.0, abs=1e-15)
    assert p_min == pytest.approx(5e-5, rel=1e-12)
    kept_mean = math.fsum((kept * values[1:4]).tolist())
    full_mean = math.fsum((prob_arr * values).tolist())
    assert kept_mean == pytest.approx(full_mean - 1e-4 * (8.0 - 4.0), rel=1e-12)


def _hockey_stick(*, dist: DiscreteDistBase, epsilon: float) -> float:
    """``delta(epsilon)`` of a real-loss distribution."""
    return (
        exact_sum(dist.prob_arr * np.maximum((-np.expm1(epsilon - dist.x_array)), 0.0)) + dist.p_max
    )


def _reciprocal_moment(dist: DiscreteDistBase) -> float:
    return exact_sum(exp_moment_terms(prob_arr=dist.prob_arr, x_vals=dist.x_array))


@pytest.mark.parametrize("tail_truncation", [1e-6, 1e-3, 0.05])
def test_connect_the_dots_edge_trim_dominates_and_is_no_looser_than_rounding(
    tail_truncation: float,
):
    """The trim raises every hockey-stick value, by no more than the rounding trim does."""
    law = gaussian_loss_law(1.0)
    source = project_continuous_dist_ctd(dist=law, dual_dist=law, tail_truncation=1e-12, step=0.05)

    trimmed = truncate_edges_ctd(dist=source, tail_truncation=tail_truncation)
    rounded = truncate_edges_stoch_dom(
        dist=source, tail_truncation=tail_truncation, bound_type=BoundType.DOMINATES
    )

    assert trimmed.prob_arr.size < source.prob_arr.size
    assert trimmed.p_min == 0.0
    assert exact_sum(trimmed.prob_arr) + trimmed.p_max == pytest.approx(1.0, abs=1e-15)
    assert _reciprocal_moment(trimmed) <= _reciprocal_moment(source) + 1e-15
    for epsilon in (-1.0, 0.0, 0.5, 2.0):
        exact = _hockey_stick(dist=source, epsilon=epsilon)
        assert exact - 1e-15 <= _hockey_stick(dist=trimmed, epsilon=epsilon)
        assert _hockey_stick(dist=trimmed, epsilon=epsilon) <= (
            _hockey_stick(dist=rounded, epsilon=epsilon) + 1e-15
        )


def test_connect_the_dots_edge_trim_keeps_the_reciprocal_moment_of_the_high_loss_tail():
    """Mass trimmed above the last kept loss leaves its reciprocal moment at that loss."""
    law = gaussian_loss_law(1.0)
    source = project_continuous_dist_ctd(dist=law, dual_dist=law, tail_truncation=1e-12, step=0.05)
    trimmed = truncate_edges_ctd(dist=source, tail_truncation=1e-3)

    last_kept = trimmed.x_array[-1]
    above = source.x_array >= last_kept
    tail_moment = exact_sum(
        exp_moment_terms(prob_arr=source.prob_arr[above], x_vals=source.x_array[above])
    )
    moment_at_last_kept = trimmed.prob_arr[-1] * math.exp(-last_kept)
    np.testing.assert_allclose(tail_moment, moment_at_last_kept, rtol=1e-12)
    assert trimmed.p_max > source.p_max


def test_joint_functions_select_the_projection_of_the_method():
    """Each joint function returns what the function of the selected kind returns."""
    law = gaussian_loss_law(1.0)
    source = project_continuous_dist_ctd(dist=law, dual_dist=law, tail_truncation=1e-12, step=0.05)

    by_ctd = truncate_edges(
        dist=source,
        tail_truncation=1e-3,
        bound_type=BoundType.DOMINATES,
        projection_method=ProjectionMethod.CTD,
    )
    expected_ctd = truncate_edges_ctd(dist=source, tail_truncation=1e-3)
    np.testing.assert_array_equal(by_ctd.prob_arr, expected_ctd.prob_arr)
    for bound_type in (BoundType.DOMINATES, BoundType.IS_DOMINATED):
        by_rounding = truncate_edges(
            dist=source,
            tail_truncation=1e-3,
            bound_type=bound_type,
            projection_method=ProjectionMethod.STOCH_DOM,
        )
        expected = truncate_edges_stoch_dom(
            dist=source, tail_truncation=1e-3, bound_type=bound_type
        )
        np.testing.assert_array_equal(by_rounding.prob_arr, expected.prob_arr)
        bins, shares = project_grid_offsets(
            grid_offsets=_OFFSETS,
            log_step=0.3,
            bound_type=bound_type,
            projection_method=ProjectionMethod.STOCH_DOM,
        )
        expected_bins = project_grid_offsets_stoch_dom(grid_offsets=_OFFSETS, bound_type=bound_type)
        np.testing.assert_array_equal(bins, expected_bins)
        np.testing.assert_array_equal(shares, 0.0)
    with pytest.raises(ValueError, match="dominating bounds only"):
        truncate_edges(
            dist=source,
            tail_truncation=1e-3,
            bound_type=BoundType.IS_DOMINATED,
            projection_method=ProjectionMethod.CTD,
        )
