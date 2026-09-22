"""Tests for binary self-convolution on linear and geometric grids."""

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.fft import next_fast_len

import PLD_accounting.fft_convolution as fft_convolution_module
from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain, GridSpec
from PLD_accounting.distribution_utils import (
    PMF_TOLERATED_MASS_TOL,
    signed_unit_residual,
)
from PLD_accounting.fft_convolution import (
    _fft_mass_tolerances,
    fft_convolve,
    fft_self_convolve,
)
from PLD_accounting.geometric_convolution import (
    _compute_geometric_convolution,
    geometric_convolve,
    geometric_self_convolve,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import binary_self_convolve
from tests.test_tolerances import TestTolerances as TOL


def _unit_geometric(*, anchor: float, index_0: int, ratio: float) -> DenseDiscreteDist:
    """A single-atom geometric distribution from multiplicative anchor and ratio."""
    return DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(ratio),
            n=1,
            spacing_type=SpacingType.GEOMETRIC,
            anchor=anchor,
            index_0=index_0,
        ),
        prob_arr=np.array([1.0]),
        domain=Domain.POSITIVES,
    )


def _linear_dist(n: int = 5) -> DenseDiscreteDist:
    pmf = np.ones(n, dtype=np.float64) / n
    return DenseDiscreteDist(grid=GridSpec(step=1.0 / (n - 1), n=n, anchor=0.0), prob_arr=pmf)


def _geometric_dist(n: int = 6) -> DenseDiscreteDist:
    """A geometric distribution on an explicit log-coordinate lattice.

    Built structurally rather than through ``from_x_array``: ``np.geomspace`` output is
    only approximately regular, and inferring a lattice from it would relocate atoms.
    """
    return DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(1.0 / 0.1) / (n - 1),
            n=n,
            spacing_type=SpacingType.GEOMETRIC,
            anchor=0.1,
        ),
        prob_arr=np.ones(n, dtype=np.float64) / n,
        domain=Domain.POSITIVES,
    )


def test_fft_rejects_numerically_close_but_distinct_lattices() -> None:
    """Approximate equality must not move a dominating atom down to another lattice."""
    dist_1 = DenseDiscreteDist(
        grid=GridSpec(
            step=1.0,
            n=1,
            anchor=1.0,
        ),
        prob_arr=np.array([1.0]),
    )
    dist_2 = DenseDiscreteDist(
        grid=GridSpec(
            step=1.0000005,
            n=2,
            anchor=0.0,
        ),
        prob_arr=np.array([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="Grid spacing must match"):
        fft_convolve(
            dist_1=dist_1,
            dist_2=dist_2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )


def test_geometric_convolution_rejects_numerically_close_distinct_ratios() -> None:
    """Distinct ratios cannot be treated as one semantic geometric lattice."""
    dist_1 = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            n=2,
            anchor=1.0,
        ),
        prob_arr=np.array([0.0, 1.0]),
        domain=Domain.POSITIVES,
    )
    dist_2 = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.000001),
            spacing_type=SpacingType.GEOMETRIC,
            n=2,
            anchor=1.0,
        ),
        prob_arr=np.array([0.0, 1.0]),
        domain=Domain.POSITIVES,
    )
    with pytest.raises(ValueError, match="must share one exact log spacing"):
        geometric_convolve(
            dist_1=dist_1,
            dist_2=dist_2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )


def test_binary_self_convolve_rejects_invalid_t():
    """Binary self convolve rejects invalid t."""
    dist = _linear_dist()
    with pytest.raises(ValueError, match="num_convolutions must be >= 1"):
        binary_self_convolve(
            dist=dist,
            num_convolutions=0,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            convolve=fft_convolve,
        )


def test_binary_self_convolve_t1_identity():
    """Binary self convolve t1 identity."""
    dist = _linear_dist()
    result = binary_self_convolve(
        dist=dist,
        num_convolutions=1,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        convolve=fft_convolve,
    )
    assert np.allclose(result.x_array, dist.x_array)
    assert np.allclose(result.prob_arr, dist.prob_arr)


def test_binary_self_convolve_matches_direct_fft_t2():
    """Binary self convolve matches direct fft t2."""
    dist = _linear_dist()
    result = binary_self_convolve(
        dist=dist,
        num_convolutions=2,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        convolve=fft_convolve,
    )
    direct = fft_convolve(
        dist_1=dist, dist_2=dist, tail_truncation=0.0, bound_type=BoundType.DOMINATES
    )
    assert np.allclose(result.x_array, direct.x_array)
    assert np.allclose(result.prob_arr, direct.prob_arr, atol=TOL.PROBABILITY_ATOL)


def test_binary_self_convolve_matches_repeated_geometric():
    """Binary self convolve matches repeated geometric."""
    dist = _geometric_dist()
    result = binary_self_convolve(
        dist=dist,
        num_convolutions=3,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        convolve=geometric_convolve,
    )
    repeated = geometric_convolve(
        dist_1=dist, dist_2=dist, tail_truncation=0.0, bound_type=BoundType.DOMINATES
    )
    repeated = geometric_convolve(
        dist_1=repeated, dist_2=dist, tail_truncation=0.0, bound_type=BoundType.DOMINATES
    )
    assert np.allclose(result.x_array, repeated.x_array)
    assert np.allclose(result.prob_arr, repeated.prob_arr, atol=TOL.PROBABILITY_ATOL)


def test_binary_self_convolve_accumulates_anchor():
    """Self-composition sums the lattice anchors carried by the grids themselves."""
    anchors = []

    def convolve(*, dist_1, dist_2, **_kwargs):
        summed = dist_1.grid.anchor + dist_2.grid.anchor
        anchors.append(summed)
        return DenseDiscreteDist(
            grid=replace(dist_1.grid, anchor=summed),
            prob_arr=dist_1.prob_arr,
            p_min=dist_1.p_min,
            p_max=dist_1.p_max,
            domain=dist_1.domain,
        )

    dist = _geometric_dist()
    result = binary_self_convolve(
        dist=dist,
        num_convolutions=3,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        convolve=convolve,
    )

    base = dist.grid.anchor
    assert anchors == pytest.approx([2 * base, 3 * base])
    assert result.grid.anchor == pytest.approx(3 * base)


def test_geometric_self_convolve_keeps_anchored_lattice():
    """Self-convolution returns a ``T * r**k`` grid without shifting its points."""
    num_summands = 7
    result = geometric_self_convolve(
        dist=_geometric_dist(),
        num_convolutions=num_summands,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
    )

    # The composed anchor is the exact sum of the summand anchors, and the origin is
    # that anchor times an integer power of the ratio -- no logarithmic recovery.
    assert result.grid.anchor == pytest.approx(num_summands * _geometric_dist().grid.anchor)
    assert result.x_0 == result.grid.point(i=0)


@pytest.mark.parametrize("use_numba", [True, False])
def test_zero_tail_geometric_self_convolution_does_not_create_infinity_mass(
    monkeypatch: pytest.MonkeyPatch,
    use_numba: bool,
) -> None:
    """Upper-bound roundoff never becomes an absorbing infinity atom."""
    monkeypatch.setattr(
        "PLD_accounting.geometric_convolution.has_numba",
        lambda: use_numba,
    )
    dist = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(1.01),
            spacing_type=SpacingType.GEOMETRIC,
            n=(np.ones(3, dtype=np.float64) / 3.0).size,
            anchor=0.1,
        ),
        prob_arr=np.ones(3, dtype=np.float64) / 3.0,
        p_min=0.0,
        p_max=0.0,
        domain=Domain.POSITIVES,
    )

    result = geometric_self_convolve(
        dist=dist,
        num_convolutions=100_000,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
    )

    assert result.p_max == 0.0
    assert math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max]) == pytest.approx(
        1.0,
        abs=TOL.MASS_CONSERVATION,
    )


def test_dominated_geometric_convolution_preserves_tiny_zero_cross_underflow() -> None:
    """Semantic zero-plus-finite underflow is explicitly retained at p_min."""
    underflow_mass = PMF_TOLERATED_MASS_TOL / 2
    dist_with_zero = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            n=1,
            anchor=1.0,
        ),
        prob_arr=np.array([1.0 - underflow_mass]),
        p_min=underflow_mass,
        domain=Domain.POSITIVES,
    )
    finite_dist = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            n=1,
            anchor=0.5,
        ),
        prob_arr=np.array([1.0]),
        domain=Domain.POSITIVES,
    )

    result = geometric_convolve(
        dist_1=dist_with_zero,
        dist_2=finite_dist,
        tail_truncation=0.0,
        bound_type=BoundType.IS_DOMINATED,
    )

    assert result.p_min == pytest.approx(underflow_mass)
    assert result.p_max == 0.0
    np.testing.assert_allclose(result.x_array, np.array([1.5]))
    np.testing.assert_allclose(result.prob_arr, np.array([1.0 - underflow_mass]))


def test_dominating_geometric_convolution_rounds_zero_cross_underflow_up() -> None:
    """A dominating bound moves a below-grid zero cross-term to its first cell."""
    underflow_mass = 0.1
    dist_with_zero = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            n=1,
            anchor=1.0,
        ),
        prob_arr=np.array([1.0 - underflow_mass]),
        p_min=underflow_mass,
        domain=Domain.POSITIVES,
    )
    finite_dist = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            n=1,
            anchor=0.5,
        ),
        prob_arr=np.array([1.0]),
        domain=Domain.POSITIVES,
    )

    result = geometric_convolve(
        dist_1=dist_with_zero,
        dist_2=finite_dist,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
    )

    assert result.p_min == 0.0
    assert result.p_max == 0.0
    np.testing.assert_allclose(result.x_array, np.array([1.5]))
    np.testing.assert_allclose(result.prob_arr, np.array([1.0]))


@pytest.mark.parametrize(
    ("bound_type", "expected_x"),
    [(BoundType.DOMINATES, 4.0), (BoundType.IS_DOMINATED, 2.0)],
)
def test_off_lattice_sum_rounds_onto_the_summed_anchor_lattice(
    bound_type: BoundType, expected_x: float
):
    """A sum off the shared lattice rounds onto ``(a_1 + a_2) * r**k``."""
    ratio = 2.0
    # Different global exponents, so 1 + 2 = 3 is not on the anchor-2.0 lattice.
    dist_1 = _unit_geometric(anchor=1.0, index_0=0, ratio=ratio)
    dist_2 = _unit_geometric(anchor=1.0, index_0=1, ratio=ratio)

    result = geometric_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        tail_truncation=0.0,
        bound_type=bound_type,
    )

    np.testing.assert_allclose(result.x_array, np.array([expected_x]))
    np.testing.assert_allclose(result.prob_arr, np.array([1.0]))


def test_convolution_output_grid_is_bound_independent():
    """The output grid covers the sum range independently of mass-rounding direction."""
    ratio = 2.0
    grid_1 = GridSpec(step=ratio, n=1, spacing_type=SpacingType.GEOMETRIC, anchor=1.0, index_0=0)
    grid_2 = GridSpec(step=ratio, n=1, spacing_type=SpacingType.GEOMETRIC, anchor=1.0, index_0=1)
    grids_and_pmfs = [
        _compute_geometric_convolution(
            grid_1=grid_1,
            pmf_1=np.array([1.0]),
            grid_2=grid_2,
            pmf_2=np.array([1.0]),
            bound_type=bound_type,
            output_anchor=None,
        )
        for bound_type in (BoundType.DOMINATES, BoundType.IS_DOMINATED)
    ]

    assert grids_and_pmfs[0][0].x_0 == 2.0
    assert grids_and_pmfs[1][0].x_0 == 2.0
    np.testing.assert_array_equal(grids_and_pmfs[0][1], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(grids_and_pmfs[1][1], np.array([1.0, 0.0]))


@pytest.mark.parametrize("step", [1e-3, 1e-4, 1e-5])
@pytest.mark.parametrize("loss_origin", [-45.0, -5.0])
def test_diagonal_exact_hits_stay_in_place_at_pld_magnitudes(step: float, loss_origin: float):
    """Same-index diagonal sums stay exactly where they belong.

    In a self-squaring convolution every diagonal pair (i, i) sums to exactly
    ``2 * anchor * r**i``, a point of the summed-anchor lattice. At realistic PLD
    magnitudes (tiny exp-space origins, fine ratios) a logarithmic index recovery
    carries fp noise far above machine epsilon; reading the integer exponent off the
    input grids removes that failure mode entirely.
    """
    anchor = 1.0 / 7.0
    origin_index = math.floor(loss_origin / step)
    grid = GridSpec(
        step=step,
        n=4,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=anchor,
        index_0=origin_index,
    )
    pmf = np.full(4, 0.25)

    out_grid, pmf_out = _compute_geometric_convolution(
        grid_1=grid,
        pmf_1=pmf,
        grid_2=grid,
        pmf_2=pmf,
        bound_type=BoundType.DOMINATES,
        output_anchor=None,
    )

    # Bin 0 holds exactly the (0, 0) diagonal mass, at exactly twice the origin.
    assert pmf_out[0] == pytest.approx(0.0625, abs=0.0)
    assert out_grid.anchor == pytest.approx(2.0 * anchor)
    assert out_grid.index_0 == origin_index
    assert out_grid.x_0 == pytest.approx(2.0 * grid.x_0)


def test_near_lattice_value_is_not_snapped_against_domination() -> None:
    """An off-lattice sum must round up even when it is near an integer index."""
    ratio = float(np.exp(1e-5))
    index = 1_000_000
    # Two grids one exponent apart: their smallest sum is genuinely between lattice
    # points, and lies a hair above the point below it.
    grid_1 = GridSpec(
        step=math.log(ratio), n=1, spacing_type=SpacingType.GEOMETRIC, anchor=1.0, index_0=index
    )
    grid_2 = GridSpec(
        step=math.log(ratio), n=1, spacing_type=SpacingType.GEOMETRIC, anchor=1.0, index_0=index + 1
    )
    smallest_sum = grid_1.x_0 + grid_2.x_0

    out_grid, pmf = _compute_geometric_convolution(
        grid_1=grid_1,
        pmf_1=np.array([1.0]),
        grid_2=grid_2,
        pmf_2=np.array([1.0]),
        bound_type=BoundType.DOMINATES,
        output_anchor=None,
    )

    occupied = int(np.flatnonzero(pmf)[0])
    assert out_grid.point(i=occupied) >= smallest_sum


def test_same_index_hit_does_not_shift_a_full_bin() -> None:
    """Reading the exponent off the input grids makes an exact hit exact."""
    ratio = float(np.exp(1.0))
    source_anchor = 0.2
    grid = GridSpec(
        step=math.log(ratio),
        n=1,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=source_anchor,
        index_0=-1,
    )

    out_grid, pmf = _compute_geometric_convolution(
        grid_1=grid,
        pmf_1=np.array([1.0]),
        grid_2=grid,
        pmf_2=np.array([1.0]),
        bound_type=BoundType.DOMINATES,
        output_anchor=None,
    )

    assert out_grid.x_0 == pytest.approx(2.0 * source_anchor * ratio**-1)
    np.testing.assert_array_equal(pmf, np.array([1.0]))


def test_binary_self_convolve_preserves_mass_fft():
    """Binary self convolve preserves mass fft."""
    dist = _linear_dist()
    result = binary_self_convolve(
        dist=dist,
        num_convolutions=5,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        convolve=fft_convolve,
    )
    total = math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max])
    assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)


def test_fft_self_convolve_direct_vs_binary():
    """Test direct vs binary FFT self-convolution.

    Note: Direct and binary methods may produce different output sizes due to
    different truncation handling (direct uses tail_truncation/2 accounting for
    double truncation). This test verifies that both methods produce valid results
    with conserved mass.
    """
    dist = _linear_dist(n=9)
    direct = fft_self_convolve(
        dist=dist,
        num_convolutions=7,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        use_direct=True,
    )
    binary = fft_self_convolve(
        dist=dist,
        num_convolutions=7,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        use_direct=False,
    )

    # Verify both results have conserved mass
    direct_mass = math.fsum([*map(float, direct.prob_arr), direct.p_min, direct.p_max])
    binary_mass = math.fsum([*map(float, binary.prob_arr), binary.p_min, binary.p_max])
    assert np.isclose(direct_mass, 1.0, atol=TOL.MASS_CONSERVATION)
    assert np.isclose(binary_mass, 1.0, atol=TOL.MASS_CONSERVATION)

    # Both should produce valid distributions
    assert direct.x_array.size >= 2
    assert binary.x_array.size >= 2


def test_fft_tolerances_bound_the_measured_worst_case() -> None:
    """The bands must cover the worst residuals measured on the FFT routes."""
    # 1.81 * log2(n) * eps pairwise; 2.37 * num_convolutions * log2(n) * eps direct.
    eps = float(np.finfo(np.float64).eps)
    for num_bins, num_convolutions in ((311, 100000), (1871, 1668), (15707, 100)):
        drift_tol, repair_tol = _fft_mass_tolerances(
            num_bins=num_bins, num_convolutions=num_convolutions
        )
        worst = 2.37 * num_convolutions * math.log2(num_bins) * eps
        assert drift_tol > worst
        assert repair_tol > drift_tol

    pairwise_drift, _ = _fft_mass_tolerances(num_bins=10017, num_convolutions=1)
    assert pairwise_drift > 1.81 * math.log2(10017) * eps


def test_fft_pairwise_tolerance_uses_live_transform_size_and_bounds_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise a live FFT and audit its pre-repair residual against the band."""

    def normalized_pmf(size: int) -> np.ndarray:
        weights = np.linspace(1.0, 2.0, size, dtype=np.float64)
        probabilities = weights / math.fsum(map(float, weights))
        probabilities[-1] += signed_unit_residual(
            values=probabilities,
            lower_term=0.0,
            upper_term=0.0,
        )
        return probabilities

    dist_1 = DenseDiscreteDist(
        grid=GridSpec(
            step=0.01,
            n=(normalized_pmf(311)).size,
            anchor=-1.0,
        ),
        prob_arr=normalized_pmf(311),
    )
    dist_2 = DenseDiscreteDist(
        grid=GridSpec(
            step=0.01,
            n=(normalized_pmf(907)).size,
            anchor=-0.5,
        ),
        prob_arr=normalized_pmf(907),
    )
    captured: dict[str, float] = {}
    original_enforce = fft_convolution_module.enforce_mass_conservation

    def capture_enforce(**kwargs):
        captured["residual"] = abs(
            signed_unit_residual(
                values=kwargs["prob_arr"],
                lower_term=kwargs["expected_p_min"],
                upper_term=kwargs["expected_p_max"],
            )
        )
        captured["drift_tol"] = kwargs["drift_tol"]
        return original_enforce(**kwargs)

    monkeypatch.setattr(fft_convolution_module, "enforce_mass_conservation", capture_enforce)
    fft_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
    )

    fft_size = next_fast_len(dist_1.prob_arr.size + dist_2.prob_arr.size - 1)
    expected_drift, _ = _fft_mass_tolerances(num_bins=fft_size, num_convolutions=1)
    assert captured["drift_tol"] == expected_drift
    assert captured["residual"] < captured["drift_tol"]
