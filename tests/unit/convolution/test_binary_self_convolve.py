"""Tests for binary self-convolution on linear and geometric grids."""

import math

import numpy as np
import pytest
from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain
from PLD_accounting.fft_convolution import fft_convolve, fft_self_convolve
from PLD_accounting.geometric_convolution import geometric_convolve
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
    with pytest.raises(ValueError, match="T must be >= 1"):
        binary_self_convolve(
            dist=dist,
            T=0,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            convolve=fft_convolve,
        )


def test_binary_self_convolve_t1_identity():
    """Binary self convolve t1 identity."""
    dist = _linear_dist()
    result = binary_self_convolve(
        dist=dist, T=1, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=fft_convolve
    )
    assert np.allclose(result.x_array, dist.x_array)
    assert np.allclose(result.prob_arr, dist.prob_arr)


def test_binary_self_convolve_matches_direct_fft_t2():
    """Binary self convolve matches direct fft t2."""
    dist = _linear_dist()
    result = binary_self_convolve(
        dist=dist, T=2, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=fft_convolve
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
        T=3,
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


def test_binary_self_convolve_preserves_mass_fft():
    """Binary self convolve preserves mass fft."""
    dist = _linear_dist()
    result = binary_self_convolve(
        dist=dist, T=5, tail_truncation=0.0, bound_type=BoundType.DOMINATES, convolve=fft_convolve
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
        dist=dist, T=7, tail_truncation=0.0, bound_type=BoundType.DOMINATES, use_direct=True
    )
    binary = fft_self_convolve(
        dist=dist, T=7, tail_truncation=0.0, bound_type=BoundType.DOMINATES, use_direct=False
    )

    # Verify both results have conserved mass
    direct_mass = math.fsum([*map(float, direct.prob_arr), direct.p_min, direct.p_max])
    binary_mass = math.fsum([*map(float, binary.prob_arr), binary.p_min, binary.p_max])
    assert np.isclose(direct_mass, 1.0, atol=TOL.MASS_CONSERVATION)
    assert np.isclose(binary_mass, 1.0, atol=TOL.MASS_CONSERVATION)

    # Both should produce valid distributions
    assert direct.x_array.size >= 2
    assert binary.x_array.size >= 2
