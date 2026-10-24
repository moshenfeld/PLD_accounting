"""The reciprocal-moment repair reaches its target, and overshoots it by a bounded amount.

Separate from ``test_numerical_audit`` because the reference here is deliberately independent of the
code under test: it searches the retained mass of the straddling bin over float64 bit patterns
against the full evaluator, so it cannot inherit the implementation's arithmetic. The repair
converts the moment it must still remove into mass at that bin and corrects the kept mass by the
measured shortfall; this pins both ends of that -- the target is met, and the mass surrendered
beyond what the target requires is bounded by the resolution of the bin's mass and of the
requested moment. All surrendered mass is banked at ``+inf``, so the overshoot is conservative,
and bounding it is what keeps a 100-fold composition from turning it into a lost ``delta``.
"""

from __future__ import annotations

import math

import numpy as np

from PLD_accounting.distribution_utils import (
    exp_moment_terms,
    signed_unit_residual,
    trim_mass_to_moment_target,
)


def test_moment_trim_meets_its_target_within_a_bounded_overshoot() -> None:
    # pylint: disable=too-many-locals  # each compared quantity is named
    """The repair lands on the safe side of the boundary, within the resolution of its inputs.

    The kept mass is the bin's mass minus the converted moment, lowered by the measured
    corrections: four ulps of the bin cover those and the two roundings. The moment to remove
    is one float, so its own last bit, converted to mass at the bin's loss, is the other part
    of the bound.
    """
    # Geometry chosen to make the repair bind rather than to look plausible: mass rising steeply
    # with loss, so the straddling bin's own mass is small next to the surviving tail's moment,
    # the regime where spending a whole bin would be most expensive.
    loss = np.linspace(-1.0, 25.0, 4000)
    prob = np.exp(2.0 * (loss - loss[-1]))
    prob /= prob.sum()
    moment_terms = exp_moment_terms(prob_arr=prob, x_vals=loss)
    moment = 1.0 - signed_unit_residual(values=moment_terms, lower_term=0.0, upper_term=0.0)
    nudged = prob * ((1.0 + 5e-16) / moment)
    nudged_terms = exp_moment_terms(prob_arr=nudged, x_vals=loss)
    excess = -signed_unit_residual(values=nudged_terms, lower_term=0.0, upper_term=0.0)
    assert excess > 0.0

    repaired, _p_max = trim_mass_to_moment_target(
        prob_arr=nudged, loss=loss, p_max=0.0, context="bounded overshoot"
    )
    removed = math.fsum(map(float, nudged - repaired))
    assert removed > 0.0, "the fixture must make the repair bind"

    repaired_terms = exp_moment_terms(prob_arr=repaired, x_vals=loss)
    residual = signed_unit_residual(values=repaired_terms, lower_term=0.0, upper_term=0.0)
    assert residual >= 0.0, f"the repair must reach its target, residual {residual:.3e}"

    edge, boundary = _largest_retained_mass_reaching_zero_residual(prob=nudged, loss=loss)
    # Compared at the straddling bin, not on the total removed: the total is 5e-02 here, whose own
    # last bit is coarser than the quantity being bounded, so a comparison there cannot see it.
    kept = float(repaired[edge])
    overshoot = boundary - kept
    allowance = 4.0 * float(np.spacing(float(nudged[edge]))) + float(np.spacing(excess)) * math.exp(
        float(loss[edge])
    )
    assert 0.0 <= overshoot <= allowance, (
        f"the repair kept {kept:.17e} where {boundary:.17e} is the boundary: overshoot"
        f" {overshoot:.3e} is outside the allowance {allowance:.3e}"
    )


def test_moment_trim_closes_a_shortfall_below_half_an_ulp_of_the_straddling_mass() -> None:
    """A shortfall too small to move the retained mass by subtraction costs ulps, not the bin.

    The first kept mass leaves the moment above one by less than half an ulp of that bin's
    mass, converted at its loss. The retained tail is two bins at loss zero, whose masses are
    their moments exactly.
    """
    loss = np.array([float.fromhex("-0x1.cad97a89a0047p-10"), 0.0, 0.0])
    prob = np.array(
        [
            float.fromhex("0x1.c443aad68b6c0p-26"),
            float.fromhex("0x1.ffffff1d78bfbp-1"),
            float.fromhex("0x1.904ac0ffa8271p-55"),
        ]
    )

    repaired, _p_max = trim_mass_to_moment_target(
        prob_arr=prob, loss=loss, p_max=0.0, context="sub-ulp shortfall"
    )
    repaired_terms = exp_moment_terms(prob_arr=repaired, x_vals=loss)
    residual = signed_unit_residual(values=repaired_terms, lower_term=0.0, upper_term=0.0)
    assert residual >= 0.0, f"the repair must reach its target, residual {residual:.3e}"

    edge, boundary = _largest_retained_mass_reaching_zero_residual(prob=prob, loss=loss)
    assert edge == 0
    kept = float(repaired[edge])
    overshoot = boundary - kept
    allowance = 2.0 * float(np.spacing(float(prob[edge])))
    assert 0.0 <= overshoot <= allowance, (
        f"the repair kept {kept:.17e} where {boundary:.17e} is the boundary: overshoot"
        f" {overshoot:.3e} is outside the two-ulp allowance {allowance:.3e}"
    )


def _largest_retained_mass_reaching_zero_residual(
    *, prob: np.ndarray, loss: np.ndarray
) -> tuple[int, float]:
    """The straddling bin, and the most mass it may keep at a zero reciprocal residual.

    Deliberately not shared with the implementation: it locates the bin by bisection on exact
    suffix residuals and then searches the retained mass over float64 bit patterns against the
    full evaluator, so it cannot inherit the implementation's own conversion from moment to mass.
    """
    terms = exp_moment_terms(prob_arr=prob, x_vals=loss)

    def residual_from(index: int) -> float:
        return signed_unit_residual(values=terms[index:], lower_term=0.0, upper_term=0.0)

    low, high = 1, prob.size
    while low < high:
        middle = (low + high) // 2
        if residual_from(middle) >= 0.0:
            high = middle
        else:
            low = middle + 1
    edge = low - 1

    def accepts(mass: float) -> bool:
        trial = prob[edge:].copy()
        trial[0] = mass
        trial_terms = exp_moment_terms(prob_arr=trial, x_vals=loss[edge:])
        return signed_unit_residual(values=trial_terms, lower_term=0.0, upper_term=0.0) >= 0.0

    # Non-negative float64 values are ordered by their bit patterns, so bisecting those converges
    # in 64 probes however many orders of magnitude separate the ends.
    low_bits = int(np.float64(0.0).view(np.int64))
    high_bits = int(np.float64(float(prob[edge])).view(np.int64))
    while high_bits - low_bits > 1:
        middle_bits = (low_bits + high_bits) // 2
        if accepts(float(np.int64(middle_bits).view(np.float64))):
            low_bits = middle_bits
        else:
            high_bits = middle_bits
    return edge, float(np.int64(low_bits).view(np.float64))
