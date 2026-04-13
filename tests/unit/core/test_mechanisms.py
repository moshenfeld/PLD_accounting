"""Tests for Gaussian / Laplace mechanism discretization APIs."""

import numpy as np
import pytest
from PLD_accounting.discrete_dist import DenseDiscreteDist, PLDRealization
from PLD_accounting.FFT_convolution import FFT_convolve
from PLD_accounting.mechanisms import gaussian_distribution, laplace_distribution
from PLD_accounting.types import DEFAULT_TAIL_TRUNCATION, BoundType, SpacingType
from PLD_accounting.utils import binary_self_convolve


def test_gaussian_distribution_dominates_returns_pld_realization():
    d = gaussian_distribution(1.0, tail_truncation=DEFAULT_TAIL_TRUNCATION)
    assert isinstance(d, PLDRealization)


def test_gaussian_distribution_is_dominated_returns_linear_only():
    d = gaussian_distribution(
        1.0,
        tail_truncation=DEFAULT_TAIL_TRUNCATION,
        bound_type=BoundType.IS_DOMINATED,
    )
    assert isinstance(d, DenseDiscreteDist) and d.spacing_type == SpacingType.LINEAR
    assert not isinstance(d, PLDRealization)


def test_laplace_distribution_dominates_returns_pld_realization():
    d = laplace_distribution(1.0, tail_truncation=DEFAULT_TAIL_TRUNCATION)
    assert isinstance(d, PLDRealization)


def test_laplace_distribution_is_dominated_returns_linear_only():
    d = laplace_distribution(
        1.0,
        tail_truncation=DEFAULT_TAIL_TRUNCATION,
        bound_type=BoundType.IS_DOMINATED,
    )
    assert isinstance(d, DenseDiscreteDist) and d.spacing_type == SpacingType.LINEAR
    assert not isinstance(d, PLDRealization)


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_laplace_distribution_has_no_infinite_mass(scale):
    """Laplace PLD is bounded in [-lam, lam]; p_max must be 0 or negligible."""
    d = laplace_distribution(scale, tail_truncation=DEFAULT_TAIL_TRUNCATION)
    assert d.p_max < 1e-6, (
        f"laplace_distribution(scale={scale}) produced p_max={d.p_max:.3e}; "
        "the 0.5 atom at +lam must land in a finite bin, not at infinity"
    )
    total = float(np.sum(d.prob_arr)) + d.p_max + d.p_min
    assert abs(total - 1.0) < 1e-6


def test_laplace_distribution_self_convolve_t100():
    """Regression: laplace_distribution must survive 100 self-convolutions (T=100 epochs).

    Previously, the 0.5 atom at +lam was misrouted to p_max=0.5; after ~5
    binary squarings the finite mass fell below the truncation budget and raised
    ValueError inside truncate_edges.
    """
    tail_truncation = 1e-8
    d = laplace_distribution(
        scale=0.7071,
        value_discretization=0.01,
        tail_truncation=DEFAULT_TAIL_TRUNCATION,
        bound_type=BoundType.IS_DOMINATED,
    )
    # Should not raise
    composed = binary_self_convolve(
        dist=d,
        T=100,
        tail_truncation=tail_truncation,
        bound_type=BoundType.IS_DOMINATED,
        convolve=FFT_convolve,
    )
    total = float(np.sum(composed.prob_arr)) + composed.p_min + composed.p_max
    assert abs(total - 1.0) < 1e-4
