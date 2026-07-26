"""Unit tests for convolution validations."""

import numpy as np
import pytest

from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain
from PLD_accounting.fft_convolution import fft_convolve, fft_self_convolve
from PLD_accounting.types import BoundType, SpacingType


def _linear_dist() -> DenseDiscreteDist:
    return DenseDiscreteDist(
        x_0=0.0,
        step=0.5,
        prob_arr=np.array([0.2, 0.5, 0.3], dtype=np.float64),
        p_max=0.0,
    )


def _geometric_dist() -> DenseDiscreteDist:
    return DenseDiscreteDist(
        x_0=1.0,
        step=2.0,
        prob_arr=np.array([0.3, 0.4, 0.3], dtype=np.float64),
        p_max=0.0,
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )


def test_fft_requires_linear_spacing():
    """Test that FFT convolution rejects geometric distributions."""
    geometric = _geometric_dist()
    with pytest.raises(TypeError, match="DenseDiscreteDist"):
        fft_self_convolve(
            dist=geometric,
            num_convolutions=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )


def test_fft_accepts_different_origins_and_support_lengths():
    """FFT inputs need equal spacing and domains, not identical support metadata."""
    dist_1 = DenseDiscreteDist(
        x_0=-1.0,
        step=0.5,
        prob_arr=np.array([0.25, 0.75], dtype=np.float64),
    )
    dist_2 = DenseDiscreteDist(
        x_0=0.25,
        step=0.5,
        prob_arr=np.array([0.2, 0.3, 0.5], dtype=np.float64),
    )

    result = fft_convolve(
        dist_1=dist_1,
        dist_2=dist_2,
        tail_truncation=0.0,
        bound_type=BoundType.DOMINATES,
    )

    assert result.x_0 == -0.75
    assert result.step == 0.5
    np.testing.assert_allclose(result.prob_arr, np.convolve(dist_1.prob_arr, dist_2.prob_arr))
