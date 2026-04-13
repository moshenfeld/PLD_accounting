"""Unit tests for convolution validations."""

import numpy as np
import pytest
from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain
from PLD_accounting.FFT_convolution import FFT_self_convolve
from PLD_accounting.types import BoundType, SpacingType


def _linear_dist() -> DenseDiscreteDist:
    return DenseDiscreteDist(
        x_min=0.0,
        step=0.5,
        prob_arr=np.array([0.2, 0.5, 0.3], dtype=np.float64),
        p_max=0.0,
    )


def _geometric_dist() -> DenseDiscreteDist:
    return DenseDiscreteDist(
        x_min=1.0,
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
        FFT_self_convolve(
            dist=geometric,
            T=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )
