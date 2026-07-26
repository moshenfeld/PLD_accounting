"""Tests for Gaussian, Laplace, and discrete-noise mechanism PLDs."""

import inspect
import math

import numpy as np
import pytest

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_utils import PMF_MASS_TOL
from PLD_accounting.fft_convolution import fft_convolve
from PLD_accounting.mechanisms import (
    _LaplacePLD,
    discrete_distribution,
    gaussian_distribution,
    laplace_distribution,
)
from PLD_accounting.types import (
    DEFAULT_TAIL_TRUNCATION,
    BoundType,
    SpacingType,
)
from PLD_accounting.utils import binary_self_convolve


def _noise_distribution(
    pmf: np.ndarray,
    *,
    x_0: float = 0.0,
    step: float = 1.0,
    boundaries: tuple[float, float] = (0.0, 0.0),
    spacing_type: SpacingType = SpacingType.LINEAR,
) -> DenseDiscreteDist:
    """Build a count-noise distribution for mechanism tests."""
    domain = Domain.POSITIVES if spacing_type == SpacingType.GEOMETRIC else Domain.REALS
    return DenseDiscreteDist(
        x_0=x_0,
        step=step,
        prob_arr=pmf,
        p_min=boundaries[0],
        p_max=boundaries[1],
        spacing_type=spacing_type,
        domain=domain,
    )


def atomic_hockey_stick(
    losses: np.ndarray,
    masses: np.ndarray,
    epsilons: np.ndarray,
    p_max: float = 0.0,
) -> np.ndarray:
    """Evaluate an atomic PLD's hockey-stick profile."""
    return np.array(
        [
            p_max
            + math.fsum(
                float(mass) * max(0.0, -math.expm1(float(epsilon - loss)))
                for loss, mass in zip(losses, masses, strict=True)
            )
            for epsilon in epsilons
        ]
    )


def _exact_count_noise_atoms(
    pmf: np.ndarray, sensitivity: int, denominator_offset: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return exact finite loss atoms and infinity mass for one shift direction."""
    denominator = np.zeros_like(pmf)
    if denominator_offset > 0:
        denominator[: pmf.size - sensitivity] = pmf[sensitivity:]
    else:
        denominator[sensitivity:] = pmf[: pmf.size - sensitivity]
    finite = (pmf > 0.0) & (denominator > 0.0)
    losses = np.log(pmf[finite]) - np.log(denominator[finite])
    probs = pmf[finite]
    return losses, probs, 1.0 - math.fsum(map(float, probs))


def test_gaussian_distribution_dominates_returns_pld_realization():
    """Gaussian distribution dominates returns pld realization."""
    d = gaussian_distribution(1.0, tail_truncation=DEFAULT_TAIL_TRUNCATION)
    assert isinstance(d, PLDRealization)


def test_gaussian_distribution_ctd_returns_pld_realization():
    """Gaussian CtD upper discretization returns a valid PLD realization."""
    d = gaussian_distribution(
        1.0,
        value_discretization=0.05,
        tail_truncation=1e-10,
    )
    assert isinstance(d, PLDRealization)
    assert d.p_min == pytest.approx(0.0, abs=1e-14)
    assert d.p_max >= 0.0


def test_gaussian_distribution_is_dominated_returns_linear_only():
    """Gaussian distribution is dominated returns linear only."""
    d = gaussian_distribution(
        1.0,
        tail_truncation=DEFAULT_TAIL_TRUNCATION,
        bound_type=BoundType.IS_DOMINATED,
    )
    assert isinstance(d, DenseDiscreteDist) and d.spacing_type == SpacingType.LINEAR
    assert not isinstance(d, PLDRealization)


def test_laplace_distribution_dominates_returns_pld_realization():
    """Laplace distribution dominates returns pld realization."""
    d = laplace_distribution(1.0, tail_truncation=DEFAULT_TAIL_TRUNCATION)
    assert isinstance(d, PLDRealization)


def test_laplace_distribution_ctd_returns_pld_realization():
    """Laplace CtD upper discretization accounts for the mixed atoms."""
    d = laplace_distribution(
        1.0,
        value_discretization=0.02,
        tail_truncation=1e-12,
    )
    assert isinstance(d, PLDRealization)
    total = float(np.sum(d.prob_arr)) + d.p_max + d.p_min
    assert total == pytest.approx(1.0, abs=1e-10)
    assert d.p_max < 1e-10


def test_laplace_pld_tail_functions_include_boundary_atoms():
    """The mixed-law tail functions must retain atoms at support endpoints."""
    dist = _LaplacePLD(sigma=1.0)
    left_mass = 0.5 * np.exp(-dist.lam)

    assert dist.cdf(-dist.lam) == pytest.approx(left_mass)
    assert np.exp(dist.logcdf(-dist.lam)) == pytest.approx(left_mass)
    assert dist.sf(-dist.lam) == pytest.approx(1.0 - left_mass)
    assert dist.cdf(dist.lam) == 1.0
    assert dist.logcdf(dist.lam) == 0.0
    assert dist.sf(dist.lam) == 0.0


def test_laplace_distribution_is_dominated_returns_linear_only():
    """Laplace distribution is dominated returns linear only."""
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
    """Regression: laplace_distribution must survive 100 self-convolutions (100 epochs).

    Previously, the 0.5 atom at +lam was misrouted to p_max=0.5; after ~5
    binary squarings the finite mass fell below the truncation budget and raised
    ValueError inside truncate_edges.
    """
    # Use a looser truncation budget for the 100-step composition to avoid
    # re-triggering the truncation error; the initial distribution uses the
    # default (tight) budget.
    convolve_tail_truncation = 1e-8
    d = laplace_distribution(
        scale=0.7071,
        value_discretization=0.01,
        tail_truncation=DEFAULT_TAIL_TRUNCATION,
        bound_type=BoundType.IS_DOMINATED,
    )
    # Should not raise
    composed = binary_self_convolve(
        dist=d,
        num_convolutions=100,
        tail_truncation=convolve_tail_truncation,
        bound_type=BoundType.IS_DOMINATED,
        convolve=fft_convolve,
    )
    total = float(np.sum(composed.prob_arr)) + composed.p_min + composed.p_max
    assert abs(total - 1.0) < 1e-4


def test_count_noise_coalesces_duplicate_loss_atoms():
    """Repeated shifted-PMF ratios are combined before sparse CtD construction."""
    pmf = np.array([0.1, 0.2, 0.1, 0.2, 0.4])
    step = 0.05
    remove, _ = discrete_distribution(
        noise_dist=_noise_distribution(pmf),
        loss_discretization=step,
        tail_truncation=0.0,
    )
    losses, probs, p_max = _exact_count_noise_atoms(pmf, 1, 1)

    assert np.unique(losses).size < losses.size
    np.testing.assert_allclose(
        atomic_hockey_stick(remove.x_array, remove.prob_arr, remove.x_array, remove.p_max),
        atomic_hockey_stick(losses, probs, remove.x_array, p_max),
        atol=5 * PMF_MASS_TOL,
    )


@pytest.mark.parametrize("sensitivity", [1, 2])
def test_discrete_distribution_match_exact_shift_profiles(sensitivity: int):
    """Both directional count-noise PLDs CtD-dominate their exact shifted PMFs."""
    pmf = np.array([0.05, 0.15, 0.40, 0.30, 0.10])
    step = 0.025
    remove, add = discrete_distribution(
        noise_dist=_noise_distribution(pmf),
        sensitivity=sensitivity,
        loss_discretization=step,
        tail_truncation=0.0,
    )

    for result, denominator_offset in ((remove, sensitivity), (add, -sensitivity)):
        losses, probs, p_max = _exact_count_noise_atoms(pmf, sensitivity, denominator_offset)
        epsilons = np.linspace(result.x_array[0] - step, result.x_array[-1] + step, 1601)
        exact_profile = atomic_hockey_stick(losses, probs, epsilons, p_max)
        result_profile = atomic_hockey_stick(
            result.x_array, result.prob_arr, epsilons, result.p_max
        )
        assert np.all(result_profile >= exact_profile - 5 * PMF_MASS_TOL)
        np.testing.assert_allclose(
            atomic_hockey_stick(result.x_array, result.prob_arr, result.x_array, result.p_max),
            atomic_hockey_stick(losses, probs, result.x_array, p_max),
            atol=5 * PMF_MASS_TOL,
        )
        assert result.p_max == pytest.approx(p_max, abs=5 * PMF_MASS_TOL)


def test_count_noise_support_offset_does_not_change_pld():
    """Translating the integer support leaves privacy-loss ratios unchanged."""
    pmf = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    at_negative_offset = discrete_distribution(
        noise_dist=_noise_distribution(pmf, x_0=-17.0),
        loss_discretization=0.01,
        tail_truncation=0.0,
    )
    at_positive_offset = discrete_distribution(
        noise_dist=_noise_distribution(pmf, x_0=23.0),
        loss_discretization=0.01,
        tail_truncation=0.0,
    )

    for negative, positive in zip(at_negative_offset, at_positive_offset, strict=True):
        np.testing.assert_array_equal(negative.x_array, positive.x_array)
        np.testing.assert_array_equal(negative.prob_arr, positive.prob_arr)
        assert negative.p_max == positive.p_max


@pytest.mark.parametrize(
    ("boundaries", "boundary_mass"),
    [((0.007, 0.0), 0.007), ((0.0, 0.013), 0.013)],
)
def test_count_noise_each_boundary_mass_is_routed_to_infinity(boundaries, boundary_mass):
    """The distinct lower and upper noise tails each become infinite loss."""
    interior = (1.0 - boundary_mass) * np.array([0.05, 0.15, 0.40, 0.30, 0.10])
    remove, add = discrete_distribution(
        noise_dist=_noise_distribution(
            interior,
            x_0=-2.0,
            boundaries=boundaries,
        ),
        loss_discretization=0.05,
        tail_truncation=0.0,
    )

    assert remove.p_max == pytest.approx(interior[-1] + boundary_mass, abs=5 * PMF_MASS_TOL)
    assert add.p_max == pytest.approx(interior[0] + boundary_mass, abs=5 * PMF_MASS_TOL)


def test_count_noise_keeps_every_positive_loss_atom():
    """The exact sparse loss law retains positive atoms without thresholding."""
    tiny = 1e-12
    pmf = np.array([tiny, 0.25, 0.50, 0.25 - tiny])
    remove, add = discrete_distribution(
        noise_dist=_noise_distribution(pmf),
        loss_discretization=0.01,
        tail_truncation=0.0,
    )

    assert remove.p_max == pytest.approx(0.25 - tiny, abs=5 * PMF_MASS_TOL)
    assert add.p_max == pytest.approx(tiny, abs=5 * PMF_MASS_TOL)


def test_discrete_distribution_rediscretizes_sparse_loss_laws(monkeypatch):
    """Both exact directional laws are delegated to the shared rediscretizer."""
    calls = []

    def fake_rediscretize_dist_by_bound(**kwargs):
        calls.append(kwargs)
        return PLDRealization(
            x_0=0.0,
            step=kwargs["loss_discretization"],
            prob_arr=np.array([1.0]),
        )

    monkeypatch.setattr(
        "PLD_accounting.mechanisms.rediscretize_dist_by_bound",
        fake_rediscretize_dist_by_bound,
    )

    discrete_distribution(
        noise_dist=_noise_distribution(np.array([0.1, 0.2, 0.4, 0.2, 0.1])),
        loss_discretization=0.125,
        tail_truncation=0.025,
    )

    assert len(calls) == 2
    for call in calls:
        assert isinstance(call["dist"], SparseDiscreteDist)
        assert call["loss_discretization"] == 0.125
        assert call["tail_truncation"] == 0.025
        assert call["bound_type"] == BoundType.DOMINATES


def test_count_noise_with_no_overlapping_support_is_all_infinity():
    """A PMF with no positive shifted overlap produces the valid all-infinity PLD."""
    remove, add = discrete_distribution(
        noise_dist=_noise_distribution(np.array([0.5, 0.0, 0.5])),
        loss_discretization=0.1,
        tail_truncation=0.0,
        sensitivity=1,
    )

    for result in (remove, add):
        np.testing.assert_array_equal(result.prob_arr, np.zeros(2))
        assert result.p_max == 1.0
        assert result.p_min == 0.0


def test_count_noise_rejects_single_finite_loss_atom():
    """A one-atom finite overlap cannot define distinct CtD grid bounds."""
    with pytest.raises(ValueError, match="at least two distinct finite support points"):
        discrete_distribution(
            noise_dist=_noise_distribution(np.array([0.4, 0.6])),
            loss_discretization=0.1,
            tail_truncation=0.0,
        )


def test_discrete_distribution_requires_numerical_parameters():
    """Loss discretization and tail truncation must be chosen explicitly."""
    parameters = inspect.signature(discrete_distribution).parameters
    assert parameters["loss_discretization"].default is inspect.Parameter.empty
    assert parameters["tail_truncation"].default is inspect.Parameter.empty


@pytest.mark.parametrize(
    ("kwargs", "error_type", "match"),
    [
        ({"sensitivity": 1.5}, TypeError, "sensitivity must be an integer"),
        ({"sensitivity": True}, TypeError, "sensitivity must be an integer"),
    ],
)
def test_count_noise_rejects_invalid_parameters(kwargs, error_type, match):
    """Scalar count-noise parameters receive explicit type and range validation."""
    params = {"loss_discretization": 0.1, "tail_truncation": 0.0}
    params.update(kwargs)
    with pytest.raises(error_type, match=match):
        discrete_distribution(noise_dist=_noise_distribution(np.array([0.4, 0.6])), **params)


@pytest.mark.parametrize(
    ("noise_dist", "error_type", "match"),
    [
        (np.array([0.4, 0.6]), TypeError, "must be a DenseDiscreteDist"),
        (
            PLDRealization(x_0=0.0, step=1.0, prob_arr=np.array([1.0])),
            TypeError,
            "not a PLDRealization",
        ),
        (
            _noise_distribution(np.array([0.4, 0.6]), step=0.5),
            ValueError,
            "unit spacing",
        ),
        (
            _noise_distribution(np.array([0.4, 0.6]), x_0=0.5),
            ValueError,
            "integer grid origin",
        ),
        (
            _noise_distribution(
                np.array([0.4, 0.6]),
                x_0=1.0,
                step=2.0,
                spacing_type=SpacingType.GEOMETRIC,
            ),
            ValueError,
            "real-domain linear grid",
        ),
    ],
)
def test_count_noise_rejects_invalid_distributions(noise_dist, error_type, match):
    """Count noise requires a unit-spaced real-domain noise distribution."""
    with pytest.raises(error_type, match=match):
        discrete_distribution(
            noise_dist=noise_dist,
            loss_discretization=0.1,
            tail_truncation=0.0,
        )


@pytest.mark.parametrize("sensitivity", [2, 3])
def test_count_noise_without_possible_shift_overlap_is_all_infinity(sensitivity: int):
    """Sensitivity at least as large as the support yields mutually singular laws."""
    remove, add = discrete_distribution(
        noise_dist=_noise_distribution(np.array([0.4, 0.6])),
        loss_discretization=0.1,
        tail_truncation=0.0,
        sensitivity=sensitivity,
    )

    for result in (remove, add):
        np.testing.assert_array_equal(result.prob_arr, np.zeros(2))
        assert result.p_max == 1.0
        assert result.p_min == 0.0
