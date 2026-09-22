"""Tests for optional-numba fallback paths."""

import math

import numpy as np
import pytest

from PLD_accounting import types
from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    PLDRealization,
)
from PLD_accounting.distribution_discretization import (
    _numba_rediscretize_prob,
    _numpy_rediscretize_prob,
    rediscretize_dist_by_bound,
    rediscretize_prob,
)
from PLD_accounting.geometric_convolution import (
    _add_single_zero_atom_cross_term,
    _geometric_kernel,
    _numba_geometric_kernel,
    _numpy_geometric_kernel,
    geometric_convolve,
)
from PLD_accounting.mechanisms import gaussian_distribution
from PLD_accounting.random_allocation_api import gaussian_allocation_directional_pld
from PLD_accounting.subsample_pld import subsample_pld_realization
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
    SpacingType,
)


def test_rediscretize_numpy_matches_numba_for_both_bounds():
    """The NumPy rediscretization kernel agrees with the numba one for both bounds."""
    x_array = np.array([-1.2, -0.1, 0.0, 0.8, 2.2])
    prob_arr = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    x_array_out = np.array([-1.0, 0.0, 1.0, 2.0])

    for dominates in (True, False):
        expected = _numba_rediscretize_prob(x_array, prob_arr, x_array_out, dominates)
        actual = _numpy_rediscretize_prob(x_array, prob_arr, x_array_out, dominates)
        np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_rediscretize_dispatch_uses_numpy_when_numba_unavailable(monkeypatch):
    """Rediscretization falls back to the NumPy kernel when numba is absent."""
    x_array = np.array([-1.2, -0.1, 0.0, 0.8, 2.2])
    prob_arr = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    x_array_out = np.array([-1.0, 0.0, 1.0, 2.0])
    monkeypatch.setattr(types, "has_numba", lambda: False)
    monkeypatch.setattr("PLD_accounting.distribution_discretization.has_numba", lambda: False)

    expected = _numpy_rediscretize_prob(x_array, prob_arr, x_array_out, True)
    actual = rediscretize_prob(
        x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=True
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_rediscretize_exact_knots_and_nextafter_are_directional():
    """Exact knots stay put while adjacent floats round conservatively."""
    x_array_out = np.array([0.0, 0.1, 0.2, 0.3])
    x_array = np.array([0.1, np.nextafter(0.1, np.inf), np.nextafter(0.2, -np.inf)])
    prob_arr = np.array([0.2, 0.3, 0.5])

    for remapper in (_numpy_rediscretize_prob, _numba_rediscretize_prob):
        dominating = remapper(x_array, prob_arr, x_array_out, True)
        dominated = remapper(x_array, prob_arr, x_array_out, False)
        np.testing.assert_allclose(dominating, [0.0, 0.2, 0.8, 0.0], atol=1e-15)
        np.testing.assert_allclose(dominated, [0.0, 1.0, 0.0, 0.0], atol=1e-15)

    for dominates in (True, False):
        np.testing.assert_array_equal(
            _numpy_rediscretize_prob(x_array, prob_arr, x_array_out, dominates),
            _numba_rediscretize_prob(x_array, prob_arr, x_array_out, dominates),
        )


def test_geometric_numpy_matches_numba_kernel():
    """The NumPy geometric-convolution kernel agrees with the numba one."""
    pmf_base = np.array([0.2, 0.3, 0.5])
    pmf_scaled = np.array([0.4, 0.1, 0.5])
    delta_lohi = np.array([0, 1, 2], dtype=np.int64)
    delta_hilo = np.array([0, 1, 2], dtype=np.int64)
    # Per-element diagonal placement: the shared offset unless an element needs its own.
    diagonal_bins = np.full(3, delta_lohi[0], dtype=np.int64)

    expected = _numba_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        diagonal_bins=diagonal_bins,
        output_size=pmf_base.size,
    )
    actual = _numpy_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        diagonal_bins=diagonal_bins,
        output_size=pmf_base.size,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_geometric_dispatch_uses_numpy_when_numba_unavailable(monkeypatch):
    """Geometric convolution falls back to the NumPy kernel when numba is absent."""
    pmf_base = np.array([0.2, 0.3, 0.5])
    pmf_scaled = np.array([0.4, 0.1, 0.5])
    delta_lohi = np.array([0, 1, 2], dtype=np.int64)
    delta_hilo = np.array([0, 1, 2], dtype=np.int64)
    # Per-element diagonal placement: the shared offset unless an element needs its own.
    diagonal_bins = np.full(3, delta_lohi[0], dtype=np.int64)
    monkeypatch.setattr("PLD_accounting.geometric_convolution.has_numba", lambda: False)

    expected = _numpy_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        diagonal_bins=diagonal_bins,
        output_size=pmf_base.size,
    )
    actual = _geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        diagonal_bins=diagonal_bins,
        output_size=pmf_base.size,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_zero_atom_cross_term_vectorized_rounding():
    """The vectorized zero-atom cross term rounds onto the geometric grid."""
    pmf = np.zeros(3)
    x_arr = np.array([0.5, 1.0, 2.0, 4.0])
    prob_arr = np.array([0.1, 0.2, 0.3, 0.4])
    output_grid = GridSpec(step=math.log(2.0), n=3, spacing_type=SpacingType.GEOMETRIC, anchor=1.0)

    dominates, dominates_below, dominates_above = _add_single_zero_atom_cross_term(
        pmf_conv=pmf.copy(),
        x_arr=x_arr,
        prob_arr=prob_arr,
        zero_prob=0.5,
        output_grid=output_grid,
        bound_type=BoundType.DOMINATES,
    )
    np.testing.assert_allclose(dominates, np.array([0.1, 0.15, 0.2]))
    assert dominates_below == pytest.approx(0.05)
    assert dominates_above == 0.0

    is_dominated, is_dominated_below, is_dominated_above = _add_single_zero_atom_cross_term(
        pmf_conv=pmf.copy(),
        x_arr=x_arr,
        prob_arr=prob_arr,
        zero_prob=0.5,
        output_grid=output_grid,
        bound_type=BoundType.IS_DOMINATED,
    )
    np.testing.assert_allclose(is_dominated, np.array([0.1, 0.15, 0.2]))
    assert is_dominated_below == pytest.approx(0.05)
    assert is_dominated_above == 0.0


def test_zero_atom_cross_term_returns_below_and_above_mass_separately():
    """Out-of-grid zero cross-terms retain their direction."""
    output_grid = GridSpec(step=math.log(2.0), n=3, spacing_type=SpacingType.GEOMETRIC, anchor=1.0)
    pmf, omitted_below, omitted_above = _add_single_zero_atom_cross_term(
        pmf_conv=np.zeros(3),
        x_arr=np.array([0.5, 1.0, 8.0]),
        prob_arr=np.array([0.4, 0.0, 0.6]),
        zero_prob=0.5,
        output_grid=output_grid,
        bound_type=BoundType.DOMINATES,
    )

    np.testing.assert_array_equal(pmf, np.zeros(3))
    assert omitted_below == pytest.approx(0.2)
    assert omitted_above == pytest.approx(0.3)


def test_zero_atom_cross_term_rejects_nonpositive_support():
    """Geometric zero cross-terms require strictly positive finite support."""
    with pytest.raises(ValueError, match="strictly positive"):
        _add_single_zero_atom_cross_term(
            pmf_conv=np.zeros(3),
            x_arr=np.array([0.0, 1.0]),
            prob_arr=np.array([0.5, 0.5]),
            zero_prob=0.5,
            output_grid=GridSpec(
                step=math.log(2.0), n=3, spacing_type=SpacingType.GEOMETRIC, anchor=1.0
            ),
            bound_type=BoundType.IS_DOMINATED,
        )


# The geometric kernels replay the same per-bin Kahan updates in the same order, so they
# must agree bitwise. The rediscretization kernels still accumulate differently -- Kahan
# inside the Numba loop, ``np.add.at`` in the NumPy one. A bin receiving ``m``
# contributions accumulates at most ``(m - 1) * eps`` of unordered-sum error against a
# compensated one, and no bin receives more than ``2 * n`` contributions, so this is the
# band inside which "correct without Numba" is a testable claim for them.
_EPS = float(np.finfo(np.float64).eps)


def _summation_band(num_atoms: int) -> float:
    """Declared agreement band between a compensated and an unordered per-bin sum."""
    return 2.0 * num_atoms * _EPS


def _scattered_geometric_inputs(num_bins: int, seed: int) -> dict:
    """A kernel fixture large enough for per-bin accumulation order to matter."""
    rng = np.random.default_rng(seed)
    pmf_base = rng.random(num_bins)
    pmf_base /= pmf_base.sum()
    pmf_scaled = rng.random(num_bins)
    pmf_scaled /= pmf_scaled.sum()
    delta_lohi = np.sort(rng.integers(0, num_bins, size=num_bins)).astype(np.int64)
    delta_hilo = np.sort(rng.integers(0, num_bins, size=num_bins)).astype(np.int64)
    return {
        "pmf_base": pmf_base,
        "pmf_scaled": pmf_scaled,
        "delta_lohi": delta_lohi,
        "delta_hilo": delta_hilo,
        "diagonal_bins": np.clip(
            delta_lohi[0] + rng.integers(-1, 2, size=num_bins), 0, num_bins - 1
        ).astype(np.int64),
        "output_size": num_bins,
    }


@pytest.mark.parametrize("num_bins", [256, 2048])
def test_geometric_kernel_variants_agree_bitwise(num_bins: int) -> None:
    """The NumPy variant must be compensated like the Numba one, not merely close to it.

    The mass repair after the kernel admits only a few ULPs, so an uncompensated
    fallback breaks allocation without Numba even when it is close element by element.
    """
    inputs = _scattered_geometric_inputs(num_bins, seed=num_bins)
    np.testing.assert_array_equal(
        _numpy_geometric_kernel(**inputs), _numba_geometric_kernel(**inputs)
    )


@pytest.mark.parametrize("num_atoms", [1000, 20000])
@pytest.mark.parametrize("dominates", [True, False])
def test_rediscretize_variants_agree_within_their_summation_band(
    num_atoms: int, dominates: bool
) -> None:
    """The same claim for the remapping kernel, at a scale where bins collide heavily."""
    rng = np.random.default_rng(num_atoms)
    x_array = np.sort(rng.normal(size=num_atoms))
    prob_arr = rng.random(num_atoms)
    prob_arr /= prob_arr.sum()
    x_array_out = np.linspace(x_array[0] - 0.1, x_array[-1] + 0.1, num_atoms // 4)

    from_numba = _numba_rediscretize_prob(x_array, prob_arr, x_array_out, dominates)
    from_numpy = _numpy_rediscretize_prob(x_array, prob_arr, x_array_out, dominates)

    band = _summation_band(num_atoms)
    assert np.max(np.abs(from_numba - from_numpy)) <= band
    assert math.fsum(map(float, from_numba)) == pytest.approx(
        math.fsum(map(float, from_numpy)), abs=band
    )


def _disable_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every dispatch point onto its NumPy variant for the duration of a test."""
    monkeypatch.setattr(types, "has_numba", lambda: False)
    for module in (
        "PLD_accounting.geometric_convolution",
        "PLD_accounting.distribution_discretization",
    ):
        monkeypatch.setattr(f"{module}.has_numba", lambda: False)


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_geometric_convolution_agrees_with_and_without_numba(
    monkeypatch: pytest.MonkeyPatch, bound_type: BoundType
) -> None:
    """End to end, not kernel to kernel: the public result must not depend on the backend."""
    num_bins = 512
    rng = np.random.default_rng(3)
    probs = rng.random(num_bins)
    probs /= probs.sum()
    dist = DenseDiscreteDist(
        grid=GridSpec(
            step=1e-3,
            n=num_bins,
            spacing_type=SpacingType.GEOMETRIC,
            anchor=0.5,
            index_0=-num_bins // 2,
        ),
        prob_arr=probs,
        domain=Domain.POSITIVES,
    )
    with_numba = geometric_convolve(
        dist_1=dist, dist_2=dist, tail_truncation=0.0, bound_type=bound_type
    )
    _disable_numba(monkeypatch)
    without_numba = geometric_convolve(
        dist_1=dist, dist_2=dist, tail_truncation=0.0, bound_type=bound_type
    )

    assert with_numba.grid == without_numba.grid
    np.testing.assert_array_equal(without_numba.prob_arr, with_numba.prob_arr)
    assert without_numba.p_max == with_numba.p_max


@pytest.mark.parametrize("direction", [Direction.REMOVE, Direction.ADD])
def test_geom_allocation_agrees_with_and_without_numba(
    monkeypatch: pytest.MonkeyPatch, direction: Direction
) -> None:
    """Regression: the GEOM Gaussian route raised a mass-repair error without Numba.

    The uncompensated NumPy kernel left a 46 eps mass residual on the first REMOVE
    self-convolution here, beyond the 10 eps repair band.
    """
    params = PrivacyParams(sigma=1.0, num_steps=100, num_selected=1, num_epochs=5, delta=1e-10)
    config = AllocationSchemeConfig(
        loss_discretization=1e-3,
        tail_truncation=1e-10,
        max_grid_mult=20_000,
        convolution_method=ConvolutionMethod.GEOM,
    )
    with_numba = gaussian_allocation_directional_pld(
        params=params, config=config, direction=direction
    )
    _disable_numba(monkeypatch)
    without_numba = gaussian_allocation_directional_pld(
        params=params, config=config, direction=direction
    )

    assert with_numba.grid == without_numba.grid
    np.testing.assert_array_equal(without_numba.prob_arr, with_numba.prob_arr)
    assert (without_numba.p_min, without_numba.p_max) == (with_numba.p_min, with_numba.p_max)


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_ctd_and_subsampling_agree_with_and_without_numba(
    monkeypatch: pytest.MonkeyPatch, bound_type: BoundType
) -> None:
    """CtD rediscretization and the subsampling projection must survive the same swap."""
    source = gaussian_distribution(scale=1.0, value_discretization=1e-2, tail_truncation=1e-12)

    def run() -> tuple[PLDRealization, PLDRealization]:
        rediscretized = rediscretize_dist_by_bound(
            dist=source, loss_discretization=3e-2, tail_truncation=1e-12, bound_type=bound_type
        )
        subsampled = subsample_pld_realization(
            base_pld=source, sampling_prob=0.5, direction=Direction.REMOVE
        )
        return rediscretized, subsampled

    ctd_with, subsampled_with = run()
    _disable_numba(monkeypatch)
    ctd_without, subsampled_without = run()

    for with_numba, without_numba in (
        (ctd_with, ctd_without),
        (subsampled_with, subsampled_without),
    ):
        assert with_numba.grid == without_numba.grid
        band = _summation_band(with_numba.prob_arr.size)
        np.testing.assert_allclose(without_numba.prob_arr, with_numba.prob_arr, atol=band, rtol=0.0)
        assert without_numba.p_max == pytest.approx(with_numba.p_max, abs=band)
