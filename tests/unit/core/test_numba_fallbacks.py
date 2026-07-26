"""Tests for optional-numba fallback paths."""

import numpy as np

from PLD_accounting import types
from PLD_accounting.distribution_discretization import (
    _numba_rediscretize_prob,
    _numpy_rediscretize_prob,
    rediscretize_prob,
)
from PLD_accounting.geometric_convolution import (
    _add_single_zero_atom_cross_term,
    _geometric_kernel,
    _numba_geometric_kernel,
    _numpy_geometric_kernel,
)
from PLD_accounting.types import BoundType


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
    actual = rediscretize_prob(x_array, prob_arr, x_array_out, True)
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

    expected = _numba_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        output_size=pmf_base.size,
    )
    actual = _numpy_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        output_size=pmf_base.size,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_geometric_dispatch_uses_numpy_when_numba_unavailable(monkeypatch):
    """Geometric convolution falls back to the NumPy kernel when numba is absent."""
    pmf_base = np.array([0.2, 0.3, 0.5])
    pmf_scaled = np.array([0.4, 0.1, 0.5])
    delta_lohi = np.array([0, 1, 2], dtype=np.int64)
    delta_hilo = np.array([0, 1, 2], dtype=np.int64)
    monkeypatch.setattr("PLD_accounting.geometric_convolution.has_numba", lambda: False)

    expected = _numpy_geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        output_size=pmf_base.size,
    )
    actual = _geometric_kernel(
        pmf_base=pmf_base,
        pmf_scaled=pmf_scaled,
        delta_lohi=delta_lohi,
        delta_hilo=delta_hilo,
        output_size=pmf_base.size,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-15)


def test_zero_atom_cross_term_vectorized_rounding():
    """The vectorized zero-atom cross term rounds onto the geometric grid."""
    pmf = np.zeros(3)
    x_arr = np.array([0.5, 1.0, 2.0, 4.0])
    prob_arr = np.array([0.1, 0.2, 0.3, 0.4])

    dominates = _add_single_zero_atom_cross_term(
        pmf_conv=pmf.copy(),
        x_arr=x_arr,
        prob_arr=prob_arr,
        zero_prob=0.5,
        x_out_0=1.0,
        ratio=2.0,
        bound_type=BoundType.DOMINATES,
    )
    np.testing.assert_allclose(dominates, np.array([0.15, 0.15, 0.2]))

    is_dominated = _add_single_zero_atom_cross_term(
        pmf_conv=pmf.copy(),
        x_arr=x_arr,
        prob_arr=prob_arr,
        zero_prob=0.5,
        x_out_0=1.0,
        ratio=2.0,
        bound_type=BoundType.IS_DOMINATED,
    )
    np.testing.assert_allclose(is_dominated, np.array([0.1, 0.15, 0.2]))
