"""Unit tests for convolution validations."""

import math
import warnings

import numpy as np
import pytest

from PLD_accounting.discrete_dist import DenseDiscreteDist, Domain, GridSpec
from PLD_accounting.fft_convolution import (
    fft_convolve,
    fft_self_convolve,
)
from PLD_accounting.types import BoundType, SpacingType


def _linear_dist() -> DenseDiscreteDist:
    return DenseDiscreteDist(
        grid=GridSpec(
            step=0.5,
            n=3,
            anchor=0.0,
        ),
        prob_arr=np.array([0.2, 0.5, 0.3], dtype=np.float64),
        p_max=0.0,
    )


def _geometric_dist() -> DenseDiscreteDist:
    return DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            spacing_type=SpacingType.GEOMETRIC,
            n=3,
            anchor=1.0,
        ),
        prob_arr=np.array([0.3, 0.4, 0.3], dtype=np.float64),
        p_max=0.0,
        domain=Domain.POSITIVES,
    )


def _linear_positive_dist(*, p_min: float = 0.0) -> DenseDiscreteDist:
    return DenseDiscreteDist(
        grid=GridSpec(
            step=1.0,
            n=2,
            anchor=1.0,
        ),
        prob_arr=np.array([0.4, 0.6 - p_min], dtype=np.float64),
        p_min=p_min,
        domain=Domain.POSITIVES,
    )


def _linear_real_boundary_dist(*, p_min: float = 0.0, p_max: float = 0.0) -> DenseDiscreteDist:
    return DenseDiscreteDist(
        grid=GridSpec(
            step=1.0,
            n=1,
            anchor=0.0,
        ),
        prob_arr=np.array([1.0 - p_min - p_max], dtype=np.float64),
        p_min=p_min,
        p_max=p_max,
        domain=Domain.REALS,
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
        grid=GridSpec(
            step=0.5,
            n=2,
            anchor=-1.0,
        ),
        prob_arr=np.array([0.25, 0.75], dtype=np.float64),
    )
    dist_2 = DenseDiscreteDist(
        grid=GridSpec(
            step=0.5,
            n=3,
            anchor=0.25,
        ),
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
    assert result.domain == Domain.REALS
    np.testing.assert_allclose(result.prob_arr, np.convolve(dist_1.prob_arr, dist_2.prob_arr))


@pytest.mark.parametrize("positive_operand", [0, 1])
@pytest.mark.parametrize("p_min", [0.0, 0.2])
def test_fft_pair_requires_real_domain(positive_operand: int, p_min: float) -> None:
    """Pairwise FFT rejects positive-domain inputs regardless of boundary mass."""
    dists = [_linear_positive_dist(), _linear_positive_dist()]
    dists[1 - positive_operand] = _linear_dist()
    dists[positive_operand] = _linear_positive_dist(p_min=p_min)

    with pytest.raises(ValueError, match=r"must use Domain\.REALS"):
        fft_convolve(
            dist_1=dists[0],
            dist_2=dists[1],
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )


@pytest.mark.parametrize("use_direct", [False, True])
def test_fft_self_rejects_boundary_only_input(use_direct: bool) -> None:
    """Boundary-only inputs fail before FFT work, without warnings or NaNs."""
    dist = DenseDiscreteDist(
        grid=GridSpec(step=1.0, n=2, anchor=0.0),
        prob_arr=np.array([0.0, 0.0], dtype=np.float64),
        p_max=1.0,
        domain=Domain.REALS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="strictly positive finite-support mass"):
            fft_self_convolve(
                dist=dist,
                num_convolutions=2,
                tail_truncation=0.0,
                bound_type=BoundType.DOMINATES,
                use_direct=use_direct,
            )


@pytest.mark.parametrize("use_direct", [False, True])
@pytest.mark.parametrize("p_min", [0.0, 0.2])
def test_fft_self_requires_real_domain(use_direct: bool, p_min: float) -> None:
    """FFT self-convolution receives REALS intermediates from both allocation paths."""
    with pytest.raises(ValueError, match=r"must use Domain\.REALS"):
        fft_self_convolve(
            dist=_linear_positive_dist(p_min=p_min),
            num_convolutions=2,
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
            use_direct=use_direct,
        )


@pytest.mark.parametrize("reverse_inputs", [False, True])
def test_fft_pair_rejects_opposing_real_boundary_atoms(reverse_inputs: bool) -> None:
    """The sum -inf + +inf has no distribution-independent interpretation."""
    dists = [
        _linear_real_boundary_dist(p_min=0.2),
        _linear_real_boundary_dist(p_max=0.3),
    ]
    if reverse_inputs:
        dists.reverse()

    with pytest.raises(ValueError, match=r"undefined.*-inf.*\+inf"):
        fft_convolve(
            dist_1=dists[0],
            dist_2=dists[1],
            tail_truncation=0.0,
            bound_type=BoundType.DOMINATES,
        )
