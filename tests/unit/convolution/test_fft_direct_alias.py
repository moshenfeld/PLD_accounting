"""Directional safety of direct FFT self-convolution under circular aliasing.

The direct route keeps a bounded FFT window, so a long enough composition wraps the
exact support around the transform period. The tests below pin the Bernoulli law that
exposed the missing ledger (``q=0.001``, ``m=300``, ``tail_truncation=0.01``) against an
exact binomial oracle, and cover the silent / warned / rejected bands of the shared
mass-correction policy.
"""

import math
import warnings

import numpy as np
import pytest

import PLD_accounting.fft_convolution as fft_convolution_module
from PLD_accounting.discrete_dist import DenseDiscreteDist, GridSpec
from PLD_accounting.fft_convolution import (
    _declared_alias_shift,
    fft_self_convolve,
)
from PLD_accounting.types import BoundType
from tests.test_tolerances import TestTolerances as TOL

BERNOULLI_Q = 0.001
COMPOSITION_DEPTH = 300
TAIL_TRUNCATION = 0.01


def _bernoulli_pld() -> DenseDiscreteDist:
    """Loss 0 with probability ``1 - q`` and loss 1 with probability ``q``."""
    return DenseDiscreteDist(
        grid=GridSpec(step=1.0, n=2, anchor=0.0, index_0=0),
        prob_arr=np.array([1.0 - BERNOULLI_Q, BERNOULLI_Q]),
    )


def _hockey_stick(dist: DenseDiscreteDist, epsilon: float) -> float:
    """``delta(epsilon)`` of a realization: ``+inf`` mass counts fully, ``-inf`` not at all."""
    terms = [float(dist.p_max)]
    for loss, prob in zip(dist.x_array, dist.prob_arr):
        if prob <= 0.0:
            continue
        weight = -math.expm1(epsilon - float(loss))
        if weight > 0.0:
            terms.append(float(prob) * weight)
    return math.fsum(terms)


def _exact_hockey_stick(epsilon: float) -> float:
    """``delta(epsilon)`` of the exact ``Binomial(m, q)`` loss, summed in log space."""
    terms = []
    for successes in range(COMPOSITION_DEPTH + 1):
        weight = -math.expm1(epsilon - successes)
        if weight <= 0.0:
            continue
        log_prob = (
            math.lgamma(COMPOSITION_DEPTH + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(COMPOSITION_DEPTH - successes + 1)
            + successes * math.log(BERNOULLI_Q)
            + (COMPOSITION_DEPTH - successes) * math.log1p(-BERNOULLI_Q)
        )
        terms.append(math.exp(log_prob) * weight)
    return math.fsum(terms)


def _direct(bound_type: BoundType) -> DenseDiscreteDist:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fft_self_convolve(
            dist=_bernoulli_pld(),
            num_convolutions=COMPOSITION_DEPTH,
            tail_truncation=TAIL_TRUNCATION,
            bound_type=bound_type,
            use_direct=True,
        )


def _epsilon_knots() -> list[float]:
    """Every integer loss the exact support can take, plus the midpoints between them."""
    knots = [float(k) for k in range(COMPOSITION_DEPTH + 1)]
    return knots + [k + 0.5 for k in knots[:-1]]


def test_declared_alias_shift_is_zero_when_the_period_covers_the_support() -> None:
    """An FFT at least as long as the exact support is not circular, so nothing is owed."""
    assert (
        _declared_alias_shift(
            input_size=2,
            num_convolutions=300,
            fft_size=301,
            finite_mass=1.0,
            window_tail_truncation=0.0025,
        )
        == 0.0
    )


def test_declared_alias_shift_is_the_window_allowance_in_output_mass_units() -> None:
    """The allowance is the window's normalized budget scaled by the output's own mass."""
    shift = _declared_alias_shift(
        input_size=2,
        num_convolutions=4,
        fft_size=4,
        finite_mass=0.5,
        window_tail_truncation=0.02,
    )
    assert shift == pytest.approx(0.02 * 0.5**4, rel=1e-15)


@pytest.mark.parametrize(
    "bound_type,is_conservative",
    [
        (BoundType.DOMINATES, lambda produced, exact: produced >= exact),
        (BoundType.IS_DOMINATED, lambda produced, exact: produced <= exact),
    ],
)
def test_direct_self_convolve_respects_its_direction_against_an_exact_oracle(
    bound_type: BoundType, is_conservative
) -> None:
    """Aliased direct output must still bound the exact binomial profile everywhere."""
    result = _direct(bound_type)
    violations = [
        (epsilon, produced, exact)
        for epsilon in _epsilon_knots()
        for produced, exact in [(_hockey_stick(result, epsilon), _exact_hockey_stick(epsilon))]
        if not is_conservative(produced, exact)
    ]
    assert not violations, f"{bound_type} violated at {violations[:5]}"


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_direct_self_convolve_keeps_total_mass_one_while_shifting(
    bound_type: BoundType,
) -> None:
    """The declared shift is funded from the giveable edge, not added on top of the mass."""
    result = _direct(bound_type)
    total = math.fsum([*map(float, result.prob_arr), result.p_min, result.p_max])
    assert total == pytest.approx(1.0, abs=TOL.MASS_CONSERVATION)


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
@pytest.mark.parametrize("tail_truncation", [0.0, TAIL_TRUNCATION])
def test_direct_self_convolve_applies_the_declared_shift_silently(
    bound_type: BoundType, tail_truncation: float
) -> None:
    """A budgeted allocation is ledger work, not drift, so it must not warn.

    A bounded window is shorter than the exact m-fold support by construction, so this
    allowance is owed on essentially every direct call; warning about it would fire on
    every normal FFT composition.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        fft_self_convolve(
            dist=_bernoulli_pld(),
            num_convolutions=COMPOSITION_DEPTH,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            use_direct=True,
        )


def _with_surplus_boundary_mass(monkeypatch: pytest.MonkeyPatch, surplus: float) -> None:
    """Inject unexplained mass above the declared allowance into the composed boundary."""
    original = fft_convolution_module.self_convolve_boundary_masses

    def inflated(*, dist, num_convolutions):
        p_min, p_max = original(dist=dist, num_convolutions=num_convolutions)
        return p_min, p_max + surplus

    monkeypatch.setattr(fft_convolution_module, "self_convolve_boundary_masses", inflated)


def _run_direct_with_fixed_bands(
    monkeypatch: pytest.MonkeyPatch, *, drift_tol: float, repair_tol: float
) -> None:
    """Pin the numerical bands so the band offset, not the FFT size, is under test."""
    monkeypatch.setattr(
        fft_convolution_module,
        "_fft_mass_tolerances",
        lambda **_kwargs: (drift_tol, repair_tol),
    )
    fft_self_convolve(
        dist=_bernoulli_pld(),
        num_convolutions=COMPOSITION_DEPTH,
        tail_truncation=TAIL_TRUNCATION,
        bound_type=BoundType.DOMINATES,
        use_direct=True,
    )


def test_direct_self_convolve_warns_above_the_declared_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Residual the allocation does not explain still runs the unchanged warning band."""
    _with_surplus_boundary_mass(monkeypatch, 1e-8)
    with pytest.warns(RuntimeWarning, match="enforce_mass_conservation"):
        _run_direct_with_fixed_bands(monkeypatch, drift_tol=1e-9, repair_tol=1e-6)


def test_direct_self_convolve_rejects_a_residual_beyond_the_declared_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Excess above the alias allowance plus the FFT band is outside the repair policy."""
    _with_surplus_boundary_mass(monkeypatch, 1e-3)
    with pytest.raises(ValueError, match="exceeds the repair tolerance"):
        _run_direct_with_fixed_bands(monkeypatch, drift_tol=1e-9, repair_tol=1e-6)


def test_direct_self_convolve_rejects_an_unfundable_declared_shift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared shift the finite ledger cannot cover must fail rather than renormalize."""
    monkeypatch.setattr(fft_convolution_module, "_declared_alias_shift", lambda **_kwargs: 1.5)
    with pytest.raises(ValueError):
        fft_self_convolve(
            dist=_bernoulli_pld(),
            num_convolutions=COMPOSITION_DEPTH,
            tail_truncation=TAIL_TRUNCATION,
            bound_type=BoundType.DOMINATES,
            use_direct=True,
        )
