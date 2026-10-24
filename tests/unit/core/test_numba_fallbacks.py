"""Tests for optional-numba fallback paths."""

import math

import numpy as np
import pytest

from PLD_accounting import types
from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain, GridSpec, PLDRealization
from PLD_accounting.distribution_utils import _numba_exact_sum, _numpy_exact_sum, exact_sum
from PLD_accounting.geometric_convolution import (
    _add_zero_atom_cross_term_stoch_dom,
    _geometric_kernel,
    _numba_geometric_kernel,
    _numpy_geometric_kernel,
    geometric_convolve,
)
from PLD_accounting.mechanisms import gaussian_distribution
from PLD_accounting.projections import project_discrete_dist
from PLD_accounting.random_allocation_api import gaussian_allocation_pld, general_allocation_pld
from PLD_accounting.projections_stoch_dom import (
    _numba_project_prob,
    _numpy_project_prob,
    project_prob,
)
from PLD_accounting.subsample_pld import subsample_pld_realization
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
    ProjectionMethod,
    SpacingType,
)
from tests.unit.allocation_dists import allocation_dists


def _numba_unavailable() -> bool:
    """Replacement for ``has_numba`` that reports numba as absent."""
    return False


def test_rediscretize_numpy_matches_numba_for_both_bounds():
    """The NumPy rediscretization kernel agrees with the numba one for both bounds."""
    x_array = np.array([-1.2, -0.1, 0.0, 0.8, 2.2])
    prob_arr = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    x_array_out = np.array([-1.0, 0.0, 1.0, 2.0])

    for dominates in (True, False):
        expected = _numba_project_prob(
            x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=dominates
        )
        actual = _numpy_project_prob(
            x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=dominates
        )
        np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_rediscretize_dispatch_uses_numpy_when_numba_unavailable(monkeypatch):
    """Rediscretization falls back to the NumPy kernel when numba is absent."""
    x_array = np.array([-1.2, -0.1, 0.0, 0.8, 2.2])
    prob_arr = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
    x_array_out = np.array([-1.0, 0.0, 1.0, 2.0])
    monkeypatch.setattr(types, "has_numba", _numba_unavailable)
    monkeypatch.setattr("PLD_accounting.projections_stoch_dom.has_numba", _numba_unavailable)

    expected = _numpy_project_prob(
        x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=True
    )
    actual = project_prob(
        x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=True
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def _exact_sum_cases() -> list[np.ndarray]:
    """Finite inputs whose correctly rounded sum is hard to reach by plain accumulation.

    Random magnitudes over sixty binary orders, cancelling pairs, subnormals, the ties
    that round half to even, more non-overlapping partials than the initial buffer
    holds, and the largest finite values cancelling.
    """
    rng = np.random.default_rng(20260928)
    cases = [np.array([])]
    for trial in range(300):
        size = int(rng.integers(0, 400))
        values = rng.standard_normal(size) * np.exp2(rng.integers(-30, 31, size=size))
        if trial % 3 == 0:
            values = np.abs(values)
        if trial % 5 == 0:
            values = np.concatenate((values, -values[: size // 2]))
        cases.append(values)
    cases.append(rng.random(200) * np.exp2(rng.integers(-1070, 1, size=200).astype(float)))
    cases.append(np.array([5e-324] * 7 + [1e-310, -1e-310]))
    cases.append(np.array([1e-16, 1.0, 1e16]))
    cases.append(np.array([2.0**53, 1.0, 2.0**-60]))
    cases.append(np.array([1.0, 2.0**-53]))
    cases.append(np.array([1.0, 2.0**-53, 2.0**-110]))
    cases.append(np.array([1.0, 2.0**-53, -(2.0**-110)]))
    cases.append(np.array([1.0, 1e100, 1.0, -1e100] * 50))
    spread = np.exp2(60.0 * np.arange(-17, 17))
    cases.append(spread)
    cases.append(np.concatenate((spread, spread, -spread)))
    largest = float(np.finfo(np.float64).max)
    cases.append(np.array([largest, 1.0, -largest]))
    return cases


def test_exact_sum_numba_and_numpy_twins_agree_bit_for_bit():
    """Both twins are correctly rounded, so they return the same float for every input."""
    for values in _exact_sum_cases():
        total = _numba_exact_sum(np.ascontiguousarray(values))
        assert total == _numpy_exact_sum(values) == math.fsum(values.tolist())
        assert exact_sum(values) == total


def test_exact_sum_rejects_a_value_that_is_not_finite():
    """An infinite or NaN value raises instead of producing a sum that is not finite."""
    for values in ([1.0, np.inf], [np.nan, 1.0], [np.inf, -np.inf], [0.0, -np.inf]):
        with pytest.raises(ValueError, match="finite values"):
            exact_sum(np.array(values))


@pytest.mark.parametrize("twin", [_numba_exact_sum, _numpy_exact_sum, exact_sum])
def test_exact_sum_rejects_a_partial_sum_that_overflows(twin):
    """Finite values whose partial sum overflows raise in both twins."""
    with pytest.raises(OverflowError):
        twin(np.array([1e308, 1e308, -1e308]))


def test_exact_sum_dispatch_uses_numpy_when_numba_unavailable(monkeypatch):
    """Without numba the NumPy twin runs and returns the same value."""
    values = _exact_sum_cases()[2]
    monkeypatch.setattr("PLD_accounting.distribution_utils.has_numba", _numba_unavailable)
    assert exact_sum(values) == math.fsum(values.tolist())


def test_rediscretize_exact_knots_and_nextafter_are_directional():
    """Exact knots stay put while adjacent floats round conservatively."""
    x_array_out = np.array([0.0, 0.1, 0.2, 0.3])
    x_array = np.array([0.1, np.nextafter(0.1, np.inf), np.nextafter(0.2, -np.inf)])
    prob_arr = np.array([0.2, 0.3, 0.5])

    for remapper in (_numpy_project_prob, _numba_project_prob):
        dominating = remapper(
            x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=True
        )
        dominated = remapper(
            x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=False
        )
        np.testing.assert_allclose(dominating, [0.0, 0.2, 0.8, 0.0], atol=1e-15)
        np.testing.assert_allclose(dominated, [0.0, 1.0, 0.0, 0.0], atol=1e-15)

    for dominates in (True, False):
        from_numpy = _numpy_project_prob(
            x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=dominates
        )
        from_numba = _numba_project_prob(
            x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=dominates
        )
        np.testing.assert_array_equal(from_numpy, from_numba)


def test_geometric_numpy_matches_numba_kernel():
    """The NumPy geometric-convolution kernel agrees with the numba one."""
    pmf_base = np.array([0.2, 0.3, 0.5])
    pmf_scaled = np.array([0.4, 0.1, 0.5])
    bin_offsets = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    # Per-element diagonal placement: the shared offset unless an element needs its own.
    diagonal_bins = np.full(3, bin_offsets[0, 0], dtype=np.int64)

    expected = _numba_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        bin_offsets=bin_offsets,
        weights=np.zeros(bin_offsets.shape),
        diagonal_bins=diagonal_bins,
        output_size=pmf_base.size,
    )
    actual = _numpy_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        bin_offsets=bin_offsets,
        weights=np.zeros(bin_offsets.shape),
        diagonal_bins=diagonal_bins,
        output_size=pmf_base.size,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_geometric_dispatch_uses_numpy_when_numba_unavailable(monkeypatch):
    """Geometric convolution falls back to the NumPy kernel when numba is absent."""
    pmf_base = np.array([0.2, 0.3, 0.5])
    pmf_scaled = np.array([0.4, 0.1, 0.5])
    bin_offsets = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
    # Per-element diagonal placement: the shared offset unless an element needs its own.
    diagonal_bins = np.full(3, bin_offsets[0, 0], dtype=np.int64)
    monkeypatch.setattr("PLD_accounting.geometric_convolution.has_numba", _numba_unavailable)

    expected = _numpy_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        bin_offsets=bin_offsets,
        weights=np.zeros(bin_offsets.shape),
        diagonal_bins=diagonal_bins,
        output_size=pmf_base.size,
    )
    actual = _geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        bin_offsets=bin_offsets,
        weights=np.zeros(bin_offsets.shape),
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

    pmf_conv = pmf.copy()
    dominates, dominates_below, dominates_above = _add_zero_atom_cross_term_stoch_dom(
        pmf_conv=pmf_conv,
        x_arr=x_arr,
        prob_arr=prob_arr,
        zero_prob=0.5,
        output_grid=output_grid,
        bound_type=BoundType.DOMINATES,
    )
    expected_pmf = np.array([0.1, 0.15, 0.2])
    np.testing.assert_allclose(dominates, expected_pmf)
    assert dominates_below == pytest.approx(0.05)
    assert dominates_above == 0.0

    pmf_conv = pmf.copy()
    is_dominated, is_dominated_below, is_dominated_above = _add_zero_atom_cross_term_stoch_dom(
        pmf_conv=pmf_conv,
        x_arr=x_arr,
        prob_arr=prob_arr,
        zero_prob=0.5,
        output_grid=output_grid,
        bound_type=BoundType.IS_DOMINATED,
    )
    np.testing.assert_allclose(is_dominated, expected_pmf)
    assert is_dominated_below == pytest.approx(0.05)
    assert is_dominated_above == 0.0


def test_zero_atom_cross_term_returns_below_and_above_mass_separately():
    """Out-of-grid zero cross-terms retain their direction."""
    output_grid = GridSpec(step=math.log(2.0), n=3, spacing_type=SpacingType.GEOMETRIC, anchor=1.0)
    pmf, omitted_below, omitted_above = _add_zero_atom_cross_term_stoch_dom(
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
    pmf_conv = np.zeros(3)
    x_arr = np.array([0.0, 1.0])
    prob_arr = np.array([0.5, 0.5])
    log_two = math.log(2.0)
    with pytest.raises(ValueError, match="strictly positive"):
        output_grid = GridSpec(step=log_two, n=3, spacing_type=SpacingType.GEOMETRIC, anchor=1.0)
        _add_zero_atom_cross_term_stoch_dom(
            pmf_conv=pmf_conv,
            x_arr=x_arr,
            prob_arr=prob_arr,
            zero_prob=0.5,
            output_grid=output_grid,
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


def _scattered_geometric_inputs(num_bins: int, seed: int, split: bool) -> dict:
    """A kernel fixture large enough for per-bin accumulation order to matter.

    ``split`` gives every index difference a positive share for the next bin, the
    connect-the-dots placement; otherwise the shares are zero, the rounding placement.
    """
    rng = np.random.default_rng(seed)
    pmf_base = rng.random(num_bins)
    pmf_base /= pmf_base.sum()
    pmf_scaled = rng.random(num_bins)
    pmf_scaled /= pmf_scaled.sum()
    bin_offsets = np.stack(
        [np.sort(rng.integers(0, num_bins, size=num_bins)) for _ in range(2)]
    ).astype(np.int64)
    weights = rng.random((2, num_bins)) if split else np.zeros((2, num_bins))
    return {
        "pmf_base": pmf_base,
        "pmf_scaled": pmf_scaled,
        "bin_offsets": bin_offsets,
        "weights": weights,
        "diagonal_bins": np.clip(
            bin_offsets[0, 0] + rng.integers(-1, 2, size=num_bins), 0, num_bins - 1
        ).astype(np.int64),
        "output_size": num_bins,
    }


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("num_bins", [256, 2048])
def test_geometric_kernel_variants_agree_bitwise(num_bins: int, split: bool) -> None:
    """The NumPy variant must be compensated like the Numba one, not merely close to it.

    The mass repair after the kernel admits only a few ULPs, so an uncompensated
    fallback breaks allocation without Numba even when it is close element by element.
    This holds for the rounding placement and for the connect-the-dots split.
    """
    inputs = _scattered_geometric_inputs(num_bins, seed=num_bins, split=split)
    from_numpy = _numpy_geometric_kernel(**inputs)
    from_numba = _numba_geometric_kernel(**inputs)
    np.testing.assert_array_equal(from_numpy, from_numba)


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

    from_numba = _numba_project_prob(
        x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=dominates
    )
    from_numpy = _numpy_project_prob(
        x_array=x_array, prob_arr=prob_arr, x_array_out=x_array_out, dominates=dominates
    )

    band = _summation_band(num_atoms)
    assert np.max(np.abs(from_numba - from_numpy)) <= band
    assert math.fsum(map(float, from_numba)) == pytest.approx(
        math.fsum(map(float, from_numpy)), abs=band
    )


def _disable_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every dispatch point onto its NumPy variant for the duration of a test."""
    monkeypatch.setattr(types, "has_numba", _numba_unavailable)
    for module in (
        "PLD_accounting.geometric_convolution",
        "PLD_accounting.projections_stoch_dom",
    ):
        monkeypatch.setattr(f"{module}.has_numba", _numba_unavailable)


@pytest.mark.parametrize(
    "bound_type, projection_method",
    [
        (BoundType.DOMINATES, ProjectionMethod.STOCH_DOM),
        (BoundType.IS_DOMINATED, ProjectionMethod.STOCH_DOM),
        (BoundType.DOMINATES, ProjectionMethod.CTD),
    ],
)
def test_geometric_convolution_agrees_with_and_without_numba(
    monkeypatch: pytest.MonkeyPatch,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> None:
    """End to end, not kernel to kernel: the public result must not depend on the backend."""
    num_bins = 512
    rng = np.random.default_rng(3)
    probs = rng.random(num_bins)
    probs /= probs.sum()
    grid = GridSpec(
        step=1e-3,
        n=num_bins,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=0.5,
        index_0=-num_bins // 2,
    )
    dist = DenseDiscreteDist(grid=grid, prob_arr=probs, domain=Domain.POSITIVES)
    with_numba = geometric_convolve(
        dist_1=dist,
        dist_2=dist,
        tail_truncation=0.0,
        bound_type=bound_type,
        projection_method=projection_method,
    )
    _disable_numba(monkeypatch)
    without_numba = geometric_convolve(
        dist_1=dist,
        dist_2=dist,
        tail_truncation=0.0,
        bound_type=bound_type,
        projection_method=projection_method,
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
    with_numba = allocation_dists(
        pld_builder=gaussian_allocation_pld, params=params, config=config
    )[direction]
    _disable_numba(monkeypatch)
    without_numba = allocation_dists(
        pld_builder=gaussian_allocation_pld, params=params, config=config
    )[direction]

    assert with_numba.grid == without_numba.grid
    np.testing.assert_array_equal(without_numba.prob_arr, with_numba.prob_arr)
    assert (without_numba.p_min, without_numba.p_max) == (with_numba.p_min, with_numba.p_max)


@pytest.mark.parametrize(
    "bound_type, projection_method",
    [
        (BoundType.DOMINATES, ProjectionMethod.CTD),
        (BoundType.DOMINATES, ProjectionMethod.STOCH_DOM),
        (BoundType.IS_DOMINATED, ProjectionMethod.STOCH_DOM),
    ],
)
@pytest.mark.parametrize("direction", [Direction.REMOVE, Direction.ADD])
def test_multiscale_allocation_agrees_with_and_without_numba(
    monkeypatch: pytest.MonkeyPatch,
    direction: Direction,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
) -> None:
    """The multiscale sum and its projections do not depend on the backend.

    A realization input reaches the rounding kernel that has a compiled twin.
    """
    realization = gaussian_distribution(scale=1.0, value_discretization=1e-2, tail_truncation=1e-10)

    def run() -> DenseDiscreteDist:
        config = AllocationSchemeConfig(
            loss_discretization=5e-2,
            tail_truncation=1e-10,
            convolution_method=ConvolutionMethod.MS_FFT,
            projection_method=projection_method,
        )
        return allocation_dists(
            pld_builder=general_allocation_pld,
            num_steps=37,
            num_selected=3,
            num_epochs=2,
            remove_realization=realization,
            add_realization=realization,
            config=config,
            bound_type=bound_type,
        )[direction]

    with_numba = run()
    _disable_numba(monkeypatch)
    without_numba = run()

    assert with_numba.grid == without_numba.grid
    band = _summation_band(with_numba.prob_arr.size)
    np.testing.assert_allclose(without_numba.prob_arr, with_numba.prob_arr, atol=band, rtol=0.0)
    assert without_numba.p_min == pytest.approx(with_numba.p_min, abs=band)
    assert without_numba.p_max == pytest.approx(with_numba.p_max, abs=band)


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_ctd_and_subsampling_agree_with_and_without_numba(
    monkeypatch: pytest.MonkeyPatch, bound_type: BoundType
) -> None:
    """CtD rediscretization and the subsampling projection must survive the same swap."""
    source = gaussian_distribution(scale=1.0, value_discretization=1e-2, tail_truncation=1e-12)

    def run() -> tuple[PLDRealization, PLDRealization]:
        rediscretized = project_discrete_dist(
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
