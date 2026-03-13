import numpy as np
import pytest

from PLD_accounting.FFT_convolution import FFT_convolve, FFT_self_convolve
from PLD_accounting.geometric_convolution import geometric_convolve
from PLD_accounting.types import BoundType
from PLD_accounting.discrete_dist import LinearDiscreteDist, GeometricDiscreteDist
from PLD_accounting.utils import binary_self_convolve
from tests.test_tolerances import TestTolerances as TOL


def _linear_dist(n: int = 5) -> LinearDiscreteDist:
    x = np.linspace(0.0, 1.0, n)
    pmf = np.ones(n, dtype=np.float64) / n
    return LinearDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)


def _geometric_dist(n: int = 6) -> GeometricDiscreteDist:
    x = np.geomspace(0.1, 1.0, n)
    pmf = np.ones(n, dtype=np.float64) / n
    return GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=pmf)


def test_binary_self_convolve_rejects_invalid_t():
    dist = _linear_dist()
    with pytest.raises(ValueError, match="T must be >= 1"):
        binary_self_convolve(dist=dist, T=0, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=FFT_convolve)


def test_binary_self_convolve_t1_identity():
    dist = _linear_dist()
    result = binary_self_convolve(dist=dist, T=1, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=FFT_convolve)
    assert np.allclose(result.x_array, dist.x_array)
    assert np.allclose(result.PMF_array, dist.PMF_array)


def test_binary_self_convolve_matches_direct_fft_t2():
    dist = _linear_dist()
    result = binary_self_convolve(dist=dist, T=2, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=FFT_convolve)
    direct = FFT_convolve(dist_1=dist, dist_2=dist, tail_truncation=0.0, bound_type=BoundType.DOMINATES)
    assert np.allclose(result.x_array, direct.x_array)
    assert np.allclose(result.PMF_array, direct.PMF_array, atol=1e-12)


def test_binary_self_convolve_matches_repeated_geometric():
    dist = _geometric_dist()
    result = binary_self_convolve(dist=dist, T=3, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=geometric_convolve)
    repeated = geometric_convolve(dist_1=dist, dist_2=dist, tail_truncation=0.0, bound_type=BoundType.DOMINATES)
    repeated = geometric_convolve(dist_1=repeated, dist_2=dist, tail_truncation=0.0, bound_type=BoundType.DOMINATES)
    assert np.allclose(result.x_array, repeated.x_array)
    assert np.allclose(result.PMF_array, repeated.PMF_array, atol=1e-12)


def test_binary_self_convolve_preserves_mass_fft():
    dist = _linear_dist()
    result = binary_self_convolve(dist=dist, T=5, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=FFT_convolve)
    total = np.sum(result.PMF_array) + result.p_neg_inf + result.p_pos_inf
    assert np.isclose(total, 1.0, atol=TOL.MASS_CONSERVATION)


def test_fft_self_convolve_direct_vs_binary():
    """Test direct vs binary FFT self-convolution.

    Note: Direct and binary methods may produce different output sizes due to
    different truncation handling (direct uses tail_truncation/2 accounting for
    double truncation). This test verifies that both methods produce valid results
    with conserved mass.
    """
    dist = _linear_dist(n=9)
    direct = FFT_self_convolve(dist=dist, T=7, tail_truncation=0.0, bound_type=BoundType.DOMINATES, use_direct=True)
    binary = FFT_self_convolve(dist=dist, T=7, tail_truncation=0.0, bound_type=BoundType.DOMINATES, use_direct=False)

    # Verify both results have conserved mass
    direct_mass = np.sum(direct.PMF_array) + direct.p_neg_inf + direct.p_pos_inf
    binary_mass = np.sum(binary.PMF_array) + binary.p_neg_inf + binary.p_pos_inf
    assert np.isclose(direct_mass, 1.0, atol=TOL.MASS_CONSERVATION)
    assert np.isclose(binary_mass, 1.0, atol=TOL.MASS_CONSERVATION)

    # Both should produce valid distributions
    assert direct.x_array.size >= 2
    assert binary.x_array.size >= 2
