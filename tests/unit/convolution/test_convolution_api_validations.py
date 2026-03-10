"""Unit tests for convolution validations."""
import numpy as np
import pytest

from PLD_accounting.discrete_dist import GeometricDiscreteDist, LinearDiscreteDist
from PLD_accounting.FFT_convolution import FFT_self_convolve
from PLD_accounting.types import BoundType


def _linear_dist() -> LinearDiscreteDist:
    return LinearDiscreteDist(
        x_min=0.0,
        x_gap=0.5,
        PMF_array=np.array([0.2, 0.5, 0.3], dtype=np.float64),
        p_pos_inf=0.0,
    )


def _geometric_dist() -> GeometricDiscreteDist:
    return GeometricDiscreteDist(
        x_min=1.0,
        ratio=2.0,
        PMF_array=np.array([0.3, 0.4, 0.3], dtype=np.float64),
        p_pos_inf=0.0,
    )


def test_fft_requires_linear_spacing():
    """Test that FFT convolution rejects geometric distributions."""
    geometric = _geometric_dist()
    with pytest.raises(TypeError, match="LinearDiscreteDist"):
        FFT_self_convolve(
            dist=geometric,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )
