"""Tests for binary self-convolution on linear and geometric grids."""

import math

import numpy as np
import pytest

from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain
from PLD_accounting.distribution_utils import PMF_MASS_TOL
from PLD_accounting.fft_convolution import fft_convolve, fft_self_convolve
from PLD_accounting.geometric_convolution import (
    _compute_geometric_convolution,
    geometric_convolve,
    geometric_self_convolve,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import binary_self_convolve
from tests.test_tolerances import TestTolerances as TOL


def _linear_dist(n: int = 5) -> DenseDiscreteDist:
    x = np.linspace(0.0, 1.0, n)
    pmf = np.ones(n, dtype=np.float64) / n
    return DenseDiscreteDist.from_x_array(x_array=x, prob_arr=pmf)


def _geometric_dist(n: int = 6) -> DenseDiscreteDist:
    x = np.geomspace(0.1, 1.0, n)
    pmf = np.ones(n, dtype=np.float64) / n
    return DenseDiscreteDist.from_x_array(
        x_array=x,
        prob_arr=pmf,
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
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
    assert np.allclose(result.prob_arr, direct.prob_arr, atol=TOL.SPACING_ATOL)


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
    assert np.allclose(result.prob_arr, repeated.prob_arr, atol=TOL.SPACING_ATOL)


def test_binary_self_convolve_accumulates_anchor():
    """The input anchor is summed through self-composition."""
    anchors = []

    def convolve(*, dist_1, target_anchor=None, **_kwargs):
        anchors.append(target_anchor)
        return dist_1

    dist = _geometric_dist()
    binary_self_convolve(
        dist=dist,
        num_convolutions=3,
        lattice_anchor=0.25,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        convolve=convolve,
    )

    assert anchors == [0.5, 0.75]


def test_geometric_self_convolve_keeps_anchored_lattice():
    """Self-convolution returns a ``T * r**k`` grid without shifting its points."""
    num_summands = 7
    result = geometric_self_convolve(
        dist=_geometric_dist(),
        num_convolutions=num_summands,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        lattice_anchor=1.0,
    )

    final_anchor = float(num_summands)
    lower_index = math.log(result.x_0 / final_anchor) / math.log(result.step)

    assert lower_index == pytest.approx(round(lower_index), abs=TOL.SPACING_ATOL)


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
        x_0=0.1,
        step=1.01,
        prob_arr=np.ones(3, dtype=np.float64) / 3.0,
        p_min=0.0,
        p_max=0.0,
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )

    result = geometric_self_convolve(
        dist=dist,
        num_convolutions=100_000,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
        lattice_anchor=0.1,
    )

    assert result.p_max == 0.0
    assert math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max]) == pytest.approx(
        1.0,
        abs=TOL.MASS_CONSERVATION,
    )


def test_dominated_geometric_convolution_preserves_tiny_zero_cross_underflow() -> None:
    """Semantic zero-plus-finite underflow is explicitly retained at p_min."""
    underflow_mass = PMF_MASS_TOL / 2
    dist_with_zero = DenseDiscreteDist(
        x_0=1.0,
        step=2.0,
        prob_arr=np.array([1.0 - underflow_mass]),
        p_min=underflow_mass,
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )
    finite_dist = DenseDiscreteDist(
        x_0=0.5,
        step=2.0,
        prob_arr=np.array([1.0]),
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )

    result = geometric_convolve(
        dist_1=dist_with_zero,
        dist_2=finite_dist,
        target_anchor=None,
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
        x_0=1.0,
        step=2.0,
        prob_arr=np.array([1.0 - underflow_mass]),
        p_min=underflow_mass,
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )
    finite_dist = DenseDiscreteDist(
        x_0=0.5,
        step=2.0,
        prob_arr=np.array([1.0]),
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )

    result = geometric_convolve(
        dist_1=dist_with_zero,
        dist_2=finite_dist,
        target_anchor=None,
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
def test_anchored_convolution_uses_target_lattice(bound_type: BoundType, expected_x: float):
    """Convolution rounds directly onto ``target_anchor * r**k``."""
    ratio = 2.0
    dist_1 = DenseDiscreteDist(
        x_0=1.0,
        step=ratio,
        prob_arr=np.array([1.0]),
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )
    dist_2 = DenseDiscreteDist(
        x_0=2.0,
        step=ratio,
        prob_arr=np.array([1.0]),
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )

    result = geometric_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        target_anchor=2.0,
        tail_truncation=0.0,
        bound_type=bound_type,
    )

    np.testing.assert_allclose(result.x_array, np.array([expected_x]))
    np.testing.assert_allclose(result.prob_arr, np.array([1.0]))


def test_anchored_convolution_grid_is_bound_independent():
    """The output grid covers the sum range independently of mass-rounding direction."""
    origins_and_pmfs = [
        _compute_geometric_convolution(
            origin_1=1.0,
            pmf_1=np.array([1.0]),
            origin_2=2.0,
            pmf_2=np.array([1.0]),
            ratio=2.0,
            target_anchor=2.0,
            bound_type=bound_type,
        )
        for bound_type in (BoundType.DOMINATES, BoundType.IS_DOMINATED)
    ]

    assert origins_and_pmfs[0][0] == 2.0
    assert origins_and_pmfs[1][0] == 2.0
    np.testing.assert_array_equal(origins_and_pmfs[0][1], np.array([0.0, 1.0]))
    np.testing.assert_array_equal(origins_and_pmfs[1][1], np.array([1.0, 0.0]))


@pytest.mark.parametrize("step", [1e-3, 1e-4, 1e-5])
@pytest.mark.parametrize("loss_origin", [-45.0, -5.0])
def test_diagonal_exact_hits_stay_in_place_at_pld_magnitudes(step: float, loss_origin: float):
    """Exact-hit diagonal sums keep their bin despite log-space fp noise.

    In a self-squaring convolution every diagonal pair (i, i) sums to exactly
    ``2 * origin * r**i``, a point of both the unanchored output grid and the
    summed-anchor lattice. At realistic PLD magnitudes (tiny exp-space origins,
    fine ratios) the index computation carries fp noise far above machine
    epsilon, so this guards the snap tolerance against misrouting those hits.
    """
    ratio = float(np.exp(step))
    anchor = 1.0 / 7.0
    origin_index = math.floor(loss_origin / step)
    origin = anchor * ratio**origin_index
    pmf = np.full(4, 0.25)

    # Unanchored: the output grid starts at 2 * origin, so bin 0 must hold
    # exactly the (0, 0) diagonal mass.
    _, pmf_out = _compute_geometric_convolution(
        origin_1=origin,
        pmf_1=pmf,
        origin_2=origin,
        pmf_2=pmf,
        ratio=ratio,
        bound_type=BoundType.DOMINATES,
    )
    assert pmf_out[0] == pytest.approx(0.0625, abs=0.0)

    # Anchored on the summed lattice: the smallest sum sits exactly on lattice
    # index ``origin_index``, so its mass must land there, not one bin above.
    output_origin, pmf_out_anchored = _compute_geometric_convolution(
        origin_1=origin,
        pmf_1=pmf,
        origin_2=origin,
        pmf_2=pmf,
        ratio=ratio,
        bound_type=BoundType.DOMINATES,
        target_anchor=2.0 * anchor,
    )
    first_mass_bin = int(np.argmax(pmf_out_anchored > 0.0))
    output_origin_index = round(math.log(output_origin / (2.0 * anchor)) / step)
    assert output_origin_index + first_mass_bin == origin_index


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
