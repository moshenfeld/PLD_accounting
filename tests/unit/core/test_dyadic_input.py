"""One-step loss factors on the logarithms of the dyadic grid of likelihood ratios.

Both input kinds, a continuous law with its dual and a realization per direction, give
the remove loss factors ``(base, neg_dual_base)`` and the add loss factor ``add_base``:
``base`` is the remove loss under ``P``, ``neg_dual_base`` the same loss under ``Q``, and
``add_base`` the add loss.
"""

from __future__ import annotations

import math
from functools import partial

import numpy as np
import pytest

from PLD_accounting.discrete_dist import DyadicDist, GridSpec, PLDRealization, SparseDiscreteDist
from PLD_accounting.mechanisms import gaussian_distribution, gaussian_loss_law
from PLD_accounting.multiscale_fft_convolution import dyadic_dist_on_range
from PLD_accounting.random_allocation_continuous import continuous_base_distributions
from PLD_accounting.random_allocation_realization import realization_base_distributions
from PLD_accounting.types import BoundType, ConvolutionMethod
from PLD_accounting.utils import negate_reverse_distribution
from tests.unit.allocation_oracles import (
    add_realization,
    exact_add_delta,
    exact_remove_delta,
    gaussian_delta,
    lattice_pair,
    remove_realization,
)

_UPPER, _LOWER = BoundType.DOMINATES, BoundType.IS_DOMINATED
_BOUNDS = [pytest.param(_UPPER, id="upper"), pytest.param(_LOWER, id="lower")]
_SIGMA = 1.5
_CELLS = 35
# A loss budget that 35 points per octave meet and 34 do not.
_LOSS_STEP = 0.02
_MAX_CELLS = 1 << 14
_TAIL = 1e-10
# The realization is a discretization of the Gaussian law, so its factors match the
# Gaussian profile to its own step and tail rather than to rounding.
_REALIZATION_STEP = 1e-4
_GAUSSIAN_REALIZATION = gaussian_distribution(
    scale=_SIGMA, value_discretization=_REALIZATION_STEP, tail_truncation=1e-13
)
_GAUSSIAN_LAW = gaussian_loss_law(_SIGMA)
_GAUSSIAN_DUAL = gaussian_loss_law(_SIGMA)
_law_builder = partial(
    continuous_base_distributions,
    convolution_method=ConvolutionMethod.MS_FFT,
    max_grid_mult=-1,
    law=_GAUSSIAN_LAW,
    dual=_GAUSSIAN_DUAL,
)
_realization_builder = partial(
    realization_base_distributions,
    convolution_method=ConvolutionMethod.MS_FFT,
    max_grid_mult=-1,
    remove_realization=_GAUSSIAN_REALIZATION,
    add_realization=_GAUSSIAN_REALIZATION,
)
_INPUTS = [
    pytest.param(_law_builder, 1e-9, id="law"),
    pytest.param(_realization_builder, 2 * _REALIZATION_STEP, id="realization"),
]


def _factors(builder, *, bound_type: BoundType, tail_truncation: float):
    return builder(
        loss_discretizations=(_LOSS_STEP, _LOSS_STEP),
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        max_cells_per_band=_MAX_CELLS,
    )


def _delta(*, factor: SparseDiscreteDist, epsilon: float) -> float:
    """``delta(epsilon)`` of a loss factor of its own direction."""
    signed_excess = -np.expm1(epsilon - factor.x_array)
    excess = np.maximum(signed_excess, 0.0)
    return math.fsum((factor.prob_arr * excess).tolist()) + factor.p_max


def _reciprocal_moment(factor: SparseDiscreteDist) -> float:
    return math.fsum((factor.prob_arr * np.exp(-factor.x_array)).tolist())


def _ratio_law(factor: SparseDiscreteDist, *, negated: bool) -> DyadicDist:
    """The law of ``exp(L)``, or ``exp(-L)``; a factor off the dyadic loss points raises."""
    loss_dist = negate_reverse_distribution(factor) if negated else factor
    return dyadic_dist_on_range(loss_dist)


def _occupied_losses(factor: SparseDiscreteDist) -> np.ndarray:
    return factor.x_array[np.flatnonzero(factor.prob_arr)]


@pytest.mark.parametrize("bound_type", _BOUNDS)
@pytest.mark.parametrize("builder, profile_rtol", _INPUTS)
def test_every_factor_is_on_the_dyadic_loss_points_and_its_ratio_law_has_unit_mass(
    builder, profile_rtol: float, bound_type: BoundType
):
    """The boundary atom of a ratio law is the exact rest of unit mass."""
    del profile_rtol
    (base, neg_dual_base), add_base = _factors(
        builder, bound_type=bound_type, tail_truncation=_TAIL
    )

    for law in (
        _ratio_law(base, negated=False),
        _ratio_law(neg_dual_base, negated=False),
        _ratio_law(add_base, negated=True),
    ):
        assert law.cells_per_band == _CELLS
        assert abs(law.total_mass() - 1.0) <= np.finfo(float).eps


@pytest.mark.parametrize("builder, profile_rtol", _INPUTS)
def test_upper_bound_negated_dual_is_the_dual_of_the_loss_under_p(builder, profile_rtol: float):
    """The mean of the ratio under ``Q`` is the mass of ``P`` that is not at infinity."""
    del profile_rtol
    (base, neg_dual_base), add_base = _factors(builder, bound_type=_UPPER, tail_truncation=_TAIL)

    ratio_weighted_masses = neg_dual_base.prob_arr * np.exp(neg_dual_base.x_array)
    np.testing.assert_allclose(ratio_weighted_masses, base.prob_arr, rtol=1e-12, atol=0.0)
    mean_ratio = math.fsum(ratio_weighted_masses.tolist())
    assert mean_ratio == pytest.approx(1.0 - base.p_max, abs=1e-14)
    assert (neg_dual_base.p_max, add_base.p_min, base.p_min) == (0.0, 0.0, 0.0)
    assert _reciprocal_moment(add_base) <= 1.0


@pytest.mark.parametrize("builder, profile_rtol", _INPUTS)
def test_upper_bound_factors_keep_the_profile_at_the_grid_points_and_dominate_between(
    builder, profile_rtol: float
):
    """Connect-the-dots keeps ``delta`` at every grid point, up to the tail outside the range."""
    (base, _), add_base = _factors(builder, bound_type=_UPPER, tail_truncation=_TAIL)

    for factor in (base, add_base):
        losses = _occupied_losses(factor)[::37]
        for epsilon in losses:
            exact = gaussian_delta(sigma=_SIGMA, epsilon=epsilon)
            computed = _delta(factor=factor, epsilon=epsilon)
            assert computed == pytest.approx(exact, rel=profile_rtol, abs=_TAIL)
        for epsilon in (losses[:-1] + losses[1:]) / 2:
            exact = gaussian_delta(sigma=_SIGMA, epsilon=epsilon)
            assert _delta(factor=factor, epsilon=epsilon) >= exact * (1.0 - profile_rtol) - _TAIL


@pytest.mark.parametrize("builder, profile_rtol", _INPUTS)
def test_lower_bound_factors_are_below_the_profile(builder, profile_rtol: float):
    """Each factor is rounded to the side that lowers its direction's ``delta``."""
    (base, neg_dual_base), add_base = _factors(builder, bound_type=_LOWER, tail_truncation=_TAIL)

    for epsilon in np.linspace(-2.0, 3.0, 41):
        exact = gaussian_delta(sigma=_SIGMA, epsilon=epsilon)
        slack = profile_rtol * exact + 1e-15
        assert _delta(factor=base, epsilon=epsilon) <= exact + slack
        assert _delta(factor=add_base, epsilon=epsilon) <= exact + slack
    # Rounding the ratio down under Q leaves its mean at most the mass of P.
    assert _reciprocal_moment(negate_reverse_distribution(neg_dual_base)) <= 1.0
    assert (base.p_max, neg_dual_base.p_max, add_base.p_max) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("bound_type", _BOUNDS)
def test_range_is_the_smallest_whose_four_tails_fit_the_budget(bound_type: BoundType):
    """Outside ``[2**-r, 2**r]`` each tail of the law and of its dual is a quarter of the budget."""
    law = gaussian_loss_law(_SIGMA)
    builder = partial(
        continuous_base_distributions,
        convolution_method=ConvolutionMethod.MS_FFT,
        max_grid_mult=-1,
        law=law,
        dual=law,
    )

    def largest_tail(reach: int) -> float:
        bound = reach * math.log(2.0)
        lower_tail = float(law.cdf(-bound))
        upper_tail = float(law.sf(bound))
        return max(lower_tail, upper_tail)

    reaches = []
    for tail_truncation in (1e-4, 1e-8, 1e-12):
        (base, neg_dual_base), add_base = _factors(
            builder, bound_type=bound_type, tail_truncation=tail_truncation
        )
        reach = -_ratio_law(base, negated=False).first_band
        assert _ratio_law(neg_dual_base, negated=False).first_band == -reach
        assert _ratio_law(add_base, negated=True).first_band == -reach
        assert base.x_array[-1] == pytest.approx(reach * math.log(2.0), rel=1e-15)
        assert largest_tail(reach) <= tail_truncation / 4 < largest_tail(reach - 1)
        reaches.append(reach)

    assert reaches[0] < reaches[1] < reaches[2]


@pytest.mark.parametrize("bound_type", _BOUNDS)
def test_realization_range_is_the_smallest_whose_four_tails_fit_the_budget(
    bound_type: BoundType,
):
    """The bisection over the range finds the same smallest range from exact tail sums."""
    realization = _GAUSSIAN_REALIZATION
    builder = partial(
        realization_base_distributions,
        convolution_method=ConvolutionMethod.MS_FFT,
        max_grid_mult=-1,
        remove_realization=realization,
        add_realization=realization,
    )
    losses, masses = realization.x_array, realization.prob_arr
    dual_masses = masses * np.exp(-losses)

    def largest_tail(reach: int) -> float:
        bound = reach * math.log(2.0)
        below, above = losses < -bound, losses > bound
        return max(
            math.fsum(tail.tolist())
            for tail in (masses[below], dual_masses[below], masses[above], dual_masses[above])
        )

    tail_truncation = 1e-6
    (base, _), _ = _factors(builder, bound_type=bound_type, tail_truncation=tail_truncation)
    reach = -_ratio_law(base, negated=False).first_band

    assert largest_tail(reach) <= tail_truncation / 4 < largest_tail(reach - 1)


def _realization_with_a_loss_beyond_the_float_range(mass: float) -> PLDRealization:
    """Losses 0 and 800: the ratio ``exp(800)`` is not a float."""
    grid = GridSpec(step=800.0, n=2)
    prob_arr = np.array([1.0 - mass, mass])
    return PLDRealization(grid=grid, prob_arr=prob_arr)


def test_realization_mass_beyond_the_float_range_within_the_budget_goes_to_infinity():
    """Such an atom is in the upper tail of every range; its mass under ``P`` is at ``+inf``."""
    mass = 1e-13
    realization = _realization_with_a_loss_beyond_the_float_range(mass)

    (base, neg_dual_base), _ = realization_base_distributions(
        convolution_method=ConvolutionMethod.MS_FFT,
        max_grid_mult=-1,
        remove_realization=realization,
        add_realization=realization,
        loss_discretizations=(_LOSS_STEP, _LOSS_STEP),
        tail_truncation=1e-10,
        bound_type=_UPPER,
        max_cells_per_band=_MAX_CELLS,
    )

    assert base.p_max == pytest.approx(mass, rel=1e-3)
    assert _ratio_law(base, negated=False).first_band == -1
    assert _ratio_law(neg_dual_base, negated=False).first_moment() == pytest.approx(
        1.0 - mass, abs=1e-15
    )


def test_realization_mass_beyond_the_float_range_above_the_budget_is_rejected():
    """No range inside the float exponent leaves that tail within the budget."""
    realization = _realization_with_a_loss_beyond_the_float_range(1e-3)

    with pytest.raises(ValueError, match="tail_truncation"):
        realization_base_distributions(
            convolution_method=ConvolutionMethod.MS_FFT,
            max_grid_mult=-1,
            remove_realization=realization,
            add_realization=realization,
            loss_discretizations=(_LOSS_STEP, _LOSS_STEP),
            tail_truncation=1e-10,
            bound_type=_UPPER,
            max_cells_per_band=_MAX_CELLS,
        )


def test_continuous_range_beyond_the_float_range_is_rejected():
    """Noise this small puts the loss quantiles beyond ``ln(2) * 1000``."""
    law = gaussian_loss_law(0.02)

    with pytest.raises(ValueError, match="tail_truncation"):
        continuous_base_distributions(
            convolution_method=ConvolutionMethod.MS_FFT,
            max_grid_mult=-1,
            law=law,
            dual=law,
            loss_discretizations=(_LOSS_STEP, _LOSS_STEP),
            tail_truncation=1e-10,
            bound_type=_UPPER,
            max_cells_per_band=_MAX_CELLS,
        )


def test_each_direction_is_built_from_its_own_realization_and_loss_budget():
    """Two mechanisms: the remove factors come from one, the add factor from the other.

    The loss zero is a grid point, so connect-the-dots keeps ``delta(0)`` of each input.
    """
    step = 0.25
    remove_indices = np.array([-3, -1, 0, 2])
    remove_p_masses = np.array([0.05, 0.25, 0.4, 0.25])
    remove_pair = lattice_pair(step=step, indices=remove_indices, p_masses=remove_p_masses)
    add_indices = np.array([-2, 0, 1])
    add_p_masses = np.array([0.15, 0.5, 0.3])
    add_pair = lattice_pair(step=step, indices=add_indices, p_masses=add_p_masses)
    remove_input = remove_realization(pair=remove_pair, step=step)
    add_input = add_realization(pair=add_pair, step=step)

    (base, neg_dual_base), add_base = realization_base_distributions(
        convolution_method=ConvolutionMethod.MS_FFT,
        max_grid_mult=-1,
        remove_realization=remove_input,
        add_realization=add_input,
        loss_discretizations=(0.061, 0.04),
        tail_truncation=1e-12,
        bound_type=_UPPER,
        max_cells_per_band=_MAX_CELLS,
    )

    assert _ratio_law(base, negated=False).cells_per_band == 12
    assert _ratio_law(neg_dual_base, negated=False).cells_per_band == 12
    assert _ratio_law(add_base, negated=True).cells_per_band == 18
    exact_remove = exact_remove_delta(pair=remove_pair, num_steps=1, epsilon=0.0)
    exact_add = exact_add_delta(pair=add_pair, num_steps=1, epsilon=0.0)
    assert _delta(factor=base, epsilon=0.0) == pytest.approx(exact_remove, rel=1e-12)
    assert _delta(factor=add_base, epsilon=0.0) == pytest.approx(exact_add, rel=1e-12)
    exact_remove_of_add_pair = exact_remove_delta(pair=add_pair, num_steps=1, epsilon=0.0)
    assert exact_remove != pytest.approx(exact_remove_of_add_pair, rel=1e-3)
    assert base.p_max == pytest.approx(remove_pair.p_mass_at_infinity, rel=1e-12)
    assert add_base.p_max == pytest.approx(add_pair.q_mass_at_zero, rel=1e-12)


def test_points_per_octave_are_the_fewest_that_meet_the_loss_budget_up_to_the_cap():
    """The mean spacing ``log(2) / m`` is at most the budget; a cap that prevents it warns."""
    law = gaussian_loss_law(_SIGMA)
    builder = partial(
        continuous_base_distributions,
        convolution_method=ConvolutionMethod.MS_FFT,
        max_grid_mult=-1,
        law=law,
        dual=law,
    )

    (base, _), add_base = builder(
        loss_discretizations=(math.log(2) / 127.5, math.log(2) / 129),
        tail_truncation=_TAIL,
        bound_type=_UPPER,
        max_cells_per_band=_MAX_CELLS,
    )
    with pytest.warns(UserWarning, match="max_cells_per_band=32"):
        (capped, _), _ = builder(
            loss_discretizations=(_LOSS_STEP, _LOSS_STEP),
            tail_truncation=_TAIL,
            bound_type=_UPPER,
            max_cells_per_band=32,
        )

    cells = [
        _ratio_law(base, negated=False).cells_per_band,
        _ratio_law(add_base, negated=True).cells_per_band,
        _ratio_law(capped, negated=False).cells_per_band,
    ]
    assert cells == [128, 129, 32]


def test_cap_on_the_points_per_octave_need_not_be_a_power_of_two():
    """A cap below the derived count is used as it is."""
    law = gaussian_loss_law(_SIGMA)
    with pytest.warns(UserWarning, match="max_cells_per_band=48"):
        (base, _), _ = continuous_base_distributions(
            convolution_method=ConvolutionMethod.MS_FFT,
            max_grid_mult=-1,
            law=law,
            dual=law,
            loss_discretizations=(_LOSS_STEP / 2, _LOSS_STEP / 2),
            tail_truncation=_TAIL,
            bound_type=_UPPER,
            max_cells_per_band=48,
        )

    law = _ratio_law(base, negated=False)
    assert law.cells_per_band == 48
    assert abs(law.total_mass() - 1.0) <= np.finfo(float).eps
