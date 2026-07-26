"""Tests for fixed-gap, NumPy-only real-loss CtD."""

import math

import numpy as np
import pytest
from scipy import stats

from PLD_accounting.discrete_dist import (
    REALIZATION_MOMENT_TOL,
    Domain,
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_discretization import (
    _continuous_real_privacy_profile,
    _discrete_dist_privacy_profile,
    _pld_from_privacy_profile_ctd,
    project_dist_onto_grid_ctd,
    rediscretize_dist_by_bound,
)
from PLD_accounting.distribution_utils import PMF_MASS_TOL
from PLD_accounting.types import BoundType


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


def test_continuous_profile_uses_general_pld_dual_identity() -> None:
    """The continuous evaluator depends on the two laws, not mechanism names."""
    log_two = math.log(2.0)
    pld = stats.expon(loc=-log_two, scale=1.0)
    dual_pld = stats.weibull_max(c=1.0, loc=log_two, scale=0.5)
    eps = np.array([-1.0, -log_two, 0.0, 1.0])
    expected = np.where(eps < -log_two, -np.expm1(eps), 0.25 * np.exp(-eps))

    actual = _continuous_real_privacy_profile(
        pld_in=pld,
        dual_pld_in=dual_pld,
        eps_out=eps,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-15)


def test_continuous_profile_rejects_non_pld_pair() -> None:
    """The profile evaluator validates the negative-epsilon PLD invariant."""
    eps = np.array([-1.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="privacy profile"):
        _continuous_real_privacy_profile(
            pld_in=stats.norm(),
            dual_pld_in=stats.norm(),
            eps_out=eps,
        )


def test_continuous_profile_handles_generic_atomic_law_with_standard_cdf() -> None:
    """The strict/strict identity handles atoms with standard CDF hooks."""
    loss = 0.7
    p_high = math.exp(loss) / (1.0 + math.exp(loss))

    class TwoAtomLaw:
        """Minimal self-dual two-atom law with standard probability hooks."""

        def sf(self, values: np.ndarray) -> np.ndarray:
            """Return Pr[L > x]."""
            values = np.asarray(values)
            return p_high * (loss > values) + (1.0 - p_high) * (-loss > values)

        def logcdf(self, values: np.ndarray) -> np.ndarray:
            """Return log Pr[L <= x]."""
            values = np.asarray(values)
            cdf = (1.0 - p_high) * (-loss <= values) + p_high * (loss <= values)
            with np.errstate(divide="ignore"):
                return np.log(cdf)

    eps = np.array([0.0, loss])
    actual = _continuous_real_privacy_profile(
        pld_in=TwoAtomLaw(), dual_pld_in=TwoAtomLaw(), eps_out=eps
    )
    np.testing.assert_allclose(
        actual,
        [(math.exp(loss) - 1) / (math.exp(loss) + 1), 0.0],
        atol=PMF_MASS_TOL,
    )


def test_discrete_profile_matches_atomic_hockey_stick() -> None:
    """The discrete evaluator includes finite atoms and positive-infinity mass."""
    dist = PLDRealization(
        x_0=0.0,
        step=0.3,
        prob_arr=np.array([0.2, 0.4, 0.3]),
        p_max=0.1,
    )
    eps = np.array([-0.5, 0.0, 0.8, 2.0])

    actual = _discrete_dist_privacy_profile(dist_in=dist, eps_out=eps)
    expected = atomic_hockey_stick(dist.x_array, dist.prob_arr, eps, dist.p_max)

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=PMF_MASS_TOL)


def test_k_randomized_response_rediscretization_dominates_exact_profile() -> None:
    """CtD preserves k-RR at knots and dominates between them."""
    num_buckets = 7
    noise_probability = 0.35
    step = 0.05
    p_other = noise_probability / num_buckets
    p_correct = 1.0 - noise_probability + p_other
    max_loss = math.log(p_correct / p_other)
    losses = np.array([max_loss, 0.0, -max_loss])
    probs = np.array([p_correct, (num_buckets - 2) * p_other, p_other])
    order = np.argsort(losses)

    result = rediscretize_dist_by_bound(
        dist=SparseDiscreteDist(x_array=losses[order], prob_arr=probs[order]),
        tail_truncation=0.0,
        loss_discretization=step,
        bound_type=BoundType.DOMINATES,
    )

    np.testing.assert_allclose(
        atomic_hockey_stick(result.x_array, result.prob_arr, result.x_array, result.p_max),
        atomic_hockey_stick(losses, probs, result.x_array),
        atol=5 * PMF_MASS_TOL,
    )
    epsilons = np.linspace(result.x_array[0] - step, result.x_array[-1] + step, 2001)
    exact_profile = atomic_hockey_stick(losses, probs, epsilons)
    result_profile = atomic_hockey_stick(result.x_array, result.prob_arr, epsilons, result.p_max)
    assert np.all(result_profile >= exact_profile - 5 * PMF_MASS_TOL)
    assert result.p_min == 0.0
    assert math.fsum(map(float, result.prob_arr)) + result.p_max == pytest.approx(1.0)
    reciprocal_moment = math.fsum(
        float(prob) * math.exp(-float(loss))
        for loss, prob in zip(result.x_array, result.prob_arr, strict=True)
    )
    assert reciprocal_moment <= 1.0 + REALIZATION_MOMENT_TOL


def test_grid_atoms_are_unchanged() -> None:
    """CtD preserves source atoms that already lie on target grid knots."""
    result = project_dist_onto_grid_ctd(
        dist=PLDRealization(x_0=-1.0, step=1.0, prob_arr=np.array([0.1, 0.3, 0.6])),
        grid=GridSpec(x_0=-1.0, step=1.0, n=3),
    )
    assert isinstance(result, PLDRealization)
    assert np.allclose(result.prob_arr, [0.1, 0.3, 0.6])


def test_grid_atom_with_independently_formed_coordinate_lands_on_its_knot() -> None:
    """Fine grids may form the same knot through different float operations."""
    step = 1.0 / 1200.0
    x_0 = 2.9
    # Profile evaluation uses the materialized target knots directly, so this
    # independently formed coordinate lands on the same intended knot.
    source_loss = 2.9 + 3 * step
    result = project_dist_onto_grid_ctd(
        dist=PLDRealization(x_0=source_loss, step=step, prob_arr=np.array([1.0])),
        grid=GridSpec(x_0=x_0, step=step, n=5),
    )
    assert result.prob_arr[3] == pytest.approx(1.0)


def test_between_knots_splits_and_preserves_hockey_values() -> None:
    """CtD splits an off-grid atom while preserving knot hockey values."""
    loss, mass = 0.3, 1.0
    result = project_dist_onto_grid_ctd(
        dist=PLDRealization(x_0=loss, step=0.1, prob_arr=np.array([mass])),
        grid=GridSpec(x_0=0.0, step=1.0, n=2),
    )
    high = -math.expm1(-loss) / -math.expm1(-1.0)
    assert result.prob_arr[1] == pytest.approx(high)
    assert result.prob_arr[0] == pytest.approx(1.0 - high)
    assert np.allclose(
        atomic_hockey_stick(np.array([loss]), np.array([mass]), np.array([0.0, 1.0])),
        atomic_hockey_stick(result.x_array, result.prob_arr, result.x_array, result.p_max),
    )


def test_irregular_support_and_boundaries() -> None:
    """CtD handles irregular support and moves overflow to boundary mass."""
    result = project_dist_onto_grid_ctd(
        dist=PLDRealization(
            x_0=-0.1,
            step=400.05,
            prob_arr=np.array([0.1, 0.8, 0.1]),
        ),
        grid=GridSpec(x_0=0.0, step=1.0, n=2),
    )
    assert result.prob_arr[0] == pytest.approx(0.1, abs=PMF_MASS_TOL)
    assert result.p_max == pytest.approx(0.9, abs=1e-14)
    assert math.fsum(map(float, result.prob_arr)) + result.p_max == pytest.approx(1.0)


def test_adapter_accepts_semantically_valid_sparse_source() -> None:
    """CtD accepts irregular sparse sources satisfying the PLD contract."""
    dist = SparseDiscreteDist(np.array([0.0, 1.0]), np.array([0.5, 0.5]), domain=Domain.REALS)
    result = project_dist_onto_grid_ctd(
        dist=dist,
        grid=GridSpec(x_0=0.0, step=0.25, n=5),
    )
    eps = np.linspace(0.0, 1.0, 101)
    source_profile = atomic_hockey_stick(dist.x_array, dist.prob_arr, eps)
    output_profile = atomic_hockey_stick(result.x_array, result.prob_arr, eps, result.p_max)
    assert np.all(output_profile >= source_profile - 2 * PMF_MASS_TOL)
    np.testing.assert_allclose(output_profile[::25], source_profile[::25], atol=2 * PMF_MASS_TOL)


def test_rediscretization_rejects_single_finite_atom() -> None:
    """CtD rejects a degenerate finite range with no distinct grid bounds."""
    source = SparseDiscreteDist(
        x_array=np.array([0.2]),
        prob_arr=np.array([0.7]),
        p_max=0.3,
    )

    with pytest.raises(ValueError, match="at least two distinct finite support points"):
        rediscretize_dist_by_bound(
            dist=source,
            tail_truncation=0.0,
            loss_discretization=0.1,
            bound_type=BoundType.DOMINATES,
        )


def test_nonuniform_sparse_source_dominates_densely_between_knots() -> None:
    """CtD of a sparse PLD with unequal gaps dominates it between target knots."""
    losses = np.array([0.0, 0.2, 1.3])
    masses = np.array([0.2, 0.3, 0.5])
    dist = SparseDiscreteDist(losses, masses, domain=Domain.REALS)
    result = project_dist_onto_grid_ctd(
        dist=dist,
        grid=GridSpec(x_0=0.0, step=0.25, n=7),
    )

    assert result.p_min == 0.0
    total_mass = math.fsum(map(float, result.prob_arr)) + result.p_max
    assert total_mass == pytest.approx(1.0, abs=2 * PMF_MASS_TOL)
    reciprocal_moment = math.fsum(
        float(p) * math.exp(-float(x)) for x, p in zip(result.x_array, result.prob_arr)
    )
    assert reciprocal_moment <= 1.0 + REALIZATION_MOMENT_TOL

    eps = np.linspace(-0.25, 1.5, 351)
    source_profile = atomic_hockey_stick(losses, masses, eps)
    output_profile = atomic_hockey_stick(result.x_array, result.prob_arr, eps, result.p_max)
    assert np.all(output_profile >= source_profile - 2 * PMF_MASS_TOL)
    np.testing.assert_allclose(
        atomic_hockey_stick(result.x_array, result.prob_arr, result.x_array, result.p_max),
        atomic_hockey_stick(losses, masses, result.x_array),
        atol=2 * PMF_MASS_TOL,
    )


def test_adapter_accepts_moment_excess_within_shared_policy() -> None:
    """Profile validation uses the same reciprocal-moment slack as realizations."""
    moment = 1.0 + REALIZATION_MOMENT_TOL / 2.0
    dist = SparseDiscreteDist(np.array([-math.log(moment)]), np.array([1.0]))
    result = project_dist_onto_grid_ctd(
        dist=dist,
        grid=GridSpec(x_0=-0.1, step=0.1, n=3),
    )
    assert isinstance(result, PLDRealization)


def test_profile_inversion_and_pld_validation() -> None:
    """Profile inversion reconstructs the source PMF and a valid PLD realization."""
    losses, masses = np.array([0.0, 1.0]), np.array([0.4, 0.6])
    profile = atomic_hockey_stick(losses, masses, losses)
    result = _pld_from_privacy_profile_ctd(
        privacy_profile=profile,
        x_0_out=0.0,
        step_out=1.0,
    )
    assert np.allclose(result.prob_arr, masses)
    assert isinstance(result, PLDRealization)


def test_pld_profile_inversion_does_not_repair_invalid_profile() -> None:
    """The PLD wrapper validates rather than raising a negative-epsilon profile."""
    with pytest.raises(ValueError, match="not a valid PLD"):
        _pld_from_privacy_profile_ctd(
            privacy_profile=np.array([0.1, 0.0]),
            x_0_out=-1.0,
            step_out=1.0,
        )


def test_profile_inversion_removes_tiny_non_monotonic_tail_noise() -> None:
    """Profile inversion removes floating-point monotonicity noise in the tail."""
    # The second value is a few ULPs above the first.  CtD profile evaluators
    # can produce this in extreme tails, where it must not change a grid atom.
    profile = np.array([0.6, 0.4, 0.4 + PMF_MASS_TOL / 4.0])
    result = _pld_from_privacy_profile_ctd(
        privacy_profile=profile,
        x_0_out=0.0,
        step_out=1.0,
    )
    assert np.all(result.prob_arr >= 0.0)


def test_inconsistent_profiles_and_inputs_rejected() -> None:
    """CtD rejects non-convex profiles and sources violating its contract."""
    with pytest.raises(ValueError, match="convex"):
        _pld_from_privacy_profile_ctd(
            privacy_profile=np.array([0.1, 0.2]),
            x_0_out=0.0,
            step_out=1.0,
        )
    with pytest.raises(ValueError, match="convex"):
        _pld_from_privacy_profile_ctd(
            privacy_profile=np.array([0.8, 0.1, 0.0]),
            x_0_out=0.0,
            step_out=1.0,
        )
    with pytest.raises(ValueError, match="moment"):
        project_dist_onto_grid_ctd(
            dist=SparseDiscreteDist(np.array([-1.0]), np.array([1.0])),
            grid=GridSpec(x_0=-1.0, step=1.0, n=2),
        )
