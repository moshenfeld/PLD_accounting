"""Validity of the allocation routes, tested through the public API.

A route is a grid of likelihood ratios with a projection at the merges. The grids are
the geometric grid of direct convolution and the dyadic grid of multiscale FFT. On
either grid rounding gives upper and lower bounds and connect-the-dots gives upper
bounds.
"""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Callable

import numpy as np
import pytest

from PLD_accounting import (
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    DenseDiscreteDist,
    Direction,
    ProjectionMethod,
    PrivacyParams,
    compose_full_pld,
    gaussian_allocation_epsilon_configurable,
    gaussian_distribution,
    general_allocation_epsilon,
    general_allocation_pld,
)
from PLD_accounting.mechanisms import gaussian_loss_law
from PLD_accounting.random_allocation_api import gaussian_allocation_pld
from PLD_accounting.random_allocation_continuous import continuous_base_distributions
from PLD_accounting.random_allocation_base import allocation_plds_base
from PLD_accounting.random_allocation_realization import realization_base_distributions
from PLD_accounting.utils import calc_pld_dual
from tests.unit.allocation_dists import allocation_dists
from tests.unit.allocation_oracles import (
    add_realization,
    exact_add_delta,
    exact_remove_delta,
    gaussian_delta,
    lattice_pair,
    remove_realization,
)

_UPPER, _LOWER = BoundType.DOMINATES, BoundType.IS_DOMINATED
_ROUNDING, _CTD = ProjectionMethod.STOCH_DOM, ProjectionMethod.CTD
_GEOM, _MS_FFT = ConvolutionMethod.GEOM, ConvolutionMethod.MS_FFT
_GRIDS = [pytest.param(_GEOM, id="geom"), pytest.param(_MS_FFT, id="ms-fft")]
# Every supported combination of grid, projection and bound.
_PROJECTION_BOUNDS = ((_ROUNDING, _UPPER), (_ROUNDING, _LOWER), (_CTD, _UPPER))
_ROUTE_BOUNDS = [
    pytest.param(grid, method, bound_type, id=f"{grid.value}-{method.name}-{bound_type.name}")
    for grid in (_GEOM, _MS_FFT)
    for method, bound_type in _PROJECTION_BOUNDS
]
_UPPER_ROUTES = [pytest.param(_ROUNDING, id="rounding"), pytest.param(_CTD, id="ctd")]
_DIRECTIONS = [Direction.REMOVE, Direction.ADD]
# Slack for comparing a float64 delta with one computed along another path.
_DELTA_RTOL = 1e-9
_DELTA_ATOL = 1e-15
_REGRID_WARNING = "allocation_plds: aligning mismatched floor/ceil grids"

# A mechanism with five outcomes: losses ln(P/Q) on multiples of 0.25, one outcome
# only P produces (5% of P) and one only Q produces.
_PAIR_STEP = 0.25
_PAIR = lattice_pair(
    step=_PAIR_STEP, indices=np.array([-3, -1, 0, 2]), p_masses=np.array([0.05, 0.25, 0.4, 0.25])
)
_EXACT_DELTA = {Direction.REMOVE: exact_remove_delta, Direction.ADD: exact_add_delta}
_PAIR_REALIZATION = {
    Direction.REMOVE: remove_realization(pair=_PAIR, step=_PAIR_STEP),
    Direction.ADD: add_realization(pair=_PAIR, step=_PAIR_STEP),
}


def _config(
    *,
    projection_method: ProjectionMethod,
    loss_discretization: float,
    max_grid_mult: int,
    convolution_method: ConvolutionMethod,
) -> AllocationSchemeConfig:
    return AllocationSchemeConfig(
        loss_discretization=loss_discretization,
        tail_truncation=1e-10,
        max_grid_mult=max_grid_mult,
        convolution_method=convolution_method,
        projection_method=projection_method,
    )


def _delta(*, dist: DenseDiscreteDist, bound_type: BoundType, epsilon: float) -> float:
    """``delta(epsilon)`` of one directional loss distribution."""
    pld = compose_full_pld(remove_dist=dist, add_dist=None, bound_type=bound_type)
    return float(pld.get_delta_for_epsilon(epsilon))


def _assert_bounds(*, delta: float, exact: float, bound_type: BoundType) -> None:
    slack = _DELTA_RTOL * exact + _DELTA_ATOL
    if bound_type == _UPPER:
        assert delta >= exact - slack
    else:
        assert delta <= exact + slack


@pytest.mark.parametrize("num_steps", [1, 2, 3, 5])
@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("convolution_method, projection_method, bound_type", _ROUTE_BOUNDS)
def test_exact_profile_lies_between_the_bounds(
    convolution_method: ConvolutionMethod,
    projection_method: ProjectionMethod,
    bound_type: BoundType,
    direction: Direction,
    num_steps: int,
):
    """The enumerated privacy profile is above every lower bound and below every upper bound."""
    config = _config(
        max_grid_mult=-1,
        convolution_method=convolution_method,
        projection_method=projection_method,
        loss_discretization=0.05,
    )
    dist = allocation_dists(
        pld_builder=general_allocation_pld,
        num_steps=num_steps,
        num_selected=1,
        num_epochs=1,
        remove_realization=_PAIR_REALIZATION[Direction.REMOVE],
        add_realization=_PAIR_REALIZATION[Direction.ADD],
        config=config,
        bound_type=bound_type,
    )[direction]

    for epsilon in (0.0, 0.05, 0.2, 0.45):
        exact = _EXACT_DELTA[direction](pair=_PAIR, num_steps=num_steps, epsilon=epsilon)
        delta = _delta(dist=dist, bound_type=bound_type, epsilon=epsilon)
        _assert_bounds(delta=delta, exact=exact, bound_type=bound_type)


@pytest.mark.parametrize("direction", _DIRECTIONS)
@pytest.mark.parametrize("convolution_method", _GRIDS)
def test_one_step_returns_the_bound_of_the_input_mechanism(
    convolution_method: ConvolutionMethod, direction: Direction
):
    """With one step there is no merge: each route returns the Gaussian mechanism's bound."""
    sigma = 1.0
    params = PrivacyParams(sigma=sigma, num_steps=1)
    dists = {}
    for method, bound_type in _PROJECTION_BOUNDS:
        config = _config(
            max_grid_mult=-1,
            convolution_method=convolution_method,
            projection_method=method,
            loss_discretization=0.01,
        )
        dists[(method, bound_type)] = allocation_dists(
            pld_builder=gaussian_allocation_pld,
            params=params,
            config=config,
            bound_type=bound_type,
        )[direction]

    # Both upper bounds discretize the input the same way. The projection sets the trim
    # of its tails, and the connect-the-dots trim keeps the reciprocal moment of the
    # high-loss tail, so its bound is no looser.
    for epsilon in (0.25, 1.0, 2.5):
        exact = gaussian_delta(sigma=sigma, epsilon=epsilon)
        upper = _delta(dist=dists[(_ROUNDING, _UPPER)], bound_type=_UPPER, epsilon=epsilon)
        ctd_upper = _delta(dist=dists[(_CTD, _UPPER)], bound_type=_UPPER, epsilon=epsilon)
        lower = _delta(dist=dists[(_ROUNDING, _LOWER)], bound_type=_LOWER, epsilon=epsilon)
        assert lower <= exact <= upper
        _assert_bounds(delta=ctd_upper, exact=exact, bound_type=_UPPER)
        assert ctd_upper <= upper + _DELTA_RTOL * exact + _DELTA_ATOL
        assert upper - lower <= 0.05 * exact


@pytest.mark.parametrize("convolution_method, projection_method, bound_type", _ROUTE_BOUNDS)
def test_gaussian_input_and_a_fine_gaussian_realization_agree(
    convolution_method: ConvolutionMethod,
    projection_method: ProjectionMethod,
    bound_type: BoundType,
):
    """The two input kinds describe one mechanism, so every route gives close epsilons."""
    sigma, num_steps, delta = 2.0, 10, 1e-5
    config = _config(
        max_grid_mult=-1,
        convolution_method=convolution_method,
        projection_method=projection_method,
        loss_discretization=0.02,
    )
    realization = gaussian_distribution(
        scale=sigma, value_discretization=1e-3, tail_truncation=1e-12
    )

    params = PrivacyParams(sigma=sigma, num_steps=num_steps, delta=delta)
    from_law = gaussian_allocation_epsilon_configurable(
        params=params, config=config, bound_type=bound_type
    )
    from_realization = general_allocation_epsilon(
        delta=delta,
        num_steps=num_steps,
        num_selected=1,
        num_epochs=1,
        remove_realization=realization,
        add_realization=realization,
        config=config,
        bound_type=bound_type,
    )

    assert from_law == pytest.approx(from_realization, abs=2e-2)


def _epsilon(
    *,
    params: PrivacyParams,
    projection_method: ProjectionMethod,
    bound_type: BoundType,
    max_grid_mult: int,
    regrids: bool,
) -> float:
    """Epsilon of a Gaussian allocation; ``regrids`` states that the grid cap must bind."""
    config = _config(
        projection_method=projection_method,
        loss_discretization=0.05,
        max_grid_mult=max_grid_mult,
        convolution_method=_GEOM,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        epsilon = gaussian_allocation_epsilon_configurable(
            params=params, config=config, bound_type=bound_type
        )
    assert any(_REGRID_WARNING in str(item.message) for item in caught) == regrids
    return epsilon


_REGRID_QUERIES = [
    pytest.param(12, 3, 2, -1, False, id="divisible"),
    pytest.param(13, 3, 2, -1, False, id="non-divisible"),
    pytest.param(13, 3, 2, 300, True, id="non-divisible-binding-grid-cap"),
    pytest.param(23, 5, 3, 200, True, id="five-selected-binding-grid-cap"),
]


@pytest.mark.parametrize(
    "num_steps, num_selected, num_epochs, max_grid_mult, regrids", _REGRID_QUERIES
)
def test_lower_bound_is_at_most_every_upper_bound(
    num_steps: int, num_selected: int, num_epochs: int, max_grid_mult: int, regrids: bool
):
    """Across routes, with several selections and epochs, no lower bound exceeds an upper one."""
    params = PrivacyParams(
        sigma=1.5,
        num_steps=num_steps,
        num_selected=num_selected,
        num_epochs=num_epochs,
        delta=1e-5,
    )
    lower, rounding_upper, ctd_upper = (
        _epsilon(
            params=params,
            projection_method=method,
            bound_type=bound_type,
            max_grid_mult=max_grid_mult,
            regrids=regrids,
        )
        for method, bound_type in ((_ROUNDING, _LOWER), (_ROUNDING, _UPPER), (_CTD, _UPPER))
    )

    assert lower <= ctd_upper
    assert lower <= rounding_upper


@pytest.mark.parametrize(
    "sigma, num_steps",
    [(1.0, 10), (2.0, 30), (3.0, 100)],
)
def test_connect_the_dots_upper_bound_is_at_most_the_rounding_one(sigma: float, num_steps: int):
    """On these settings, keeping the mean at the merges tightens the upper bound."""
    params = PrivacyParams(sigma=sigma, num_steps=num_steps, delta=1e-6)
    rounding_upper, ctd_upper = (
        _epsilon(
            params=params,
            projection_method=method,
            bound_type=_UPPER,
            max_grid_mult=-1,
            regrids=False,
        )
        for method in (_ROUNDING, _CTD)
    )

    assert ctd_upper <= rounding_upper


_GRID_PROJECTIONS = [
    pytest.param(_GEOM, _ROUNDING, id="geom"),
    pytest.param(_MS_FFT, _CTD, id="ms-fft"),
]


@pytest.mark.parametrize("projection_method", _UPPER_ROUTES)
@pytest.mark.parametrize("convolution_method, grid_projection", _GRID_PROJECTIONS)
def test_default_configuration_is_the_projection_of_the_grid(
    convolution_method: ConvolutionMethod,
    grid_projection: ProjectionMethod,
    projection_method: ProjectionMethod,
):
    """Without a projection a grid uses its own: rounding on one, connect-the-dots on the other."""
    params = PrivacyParams(sigma=2.0, num_steps=6, delta=1e-4)
    default_config = AllocationSchemeConfig(
        loss_discretization=0.05,
        tail_truncation=1e-10,
        convolution_method=convolution_method,
    )
    default = gaussian_allocation_epsilon_configurable(
        params=params, config=default_config, bound_type=_UPPER
    )
    explicit_config = _config(
        max_grid_mult=-1,
        convolution_method=convolution_method,
        projection_method=projection_method,
        loss_discretization=0.05,
    )
    explicit = gaussian_allocation_epsilon_configurable(
        params=params, config=explicit_config, bound_type=_UPPER
    )

    assert (default == explicit) == (projection_method == grid_projection)


def _connect_the_dots_bases(
    *,
    base_distributions_creation: Callable[
        ..., tuple[tuple[DenseDiscreteDist, DenseDiscreteDist], DenseDiscreteDist]
    ],
    num_steps: int,
    tail_truncation: float,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """The ``(remove, add)`` base PLDs of one connect-the-dots allocation."""
    return allocation_plds_base(
        convolution_method=ConvolutionMethod.GEOM,
        max_closing_grid=1_000_000,
        base_distributions_creation=base_distributions_creation,
        num_steps=num_steps,
        loss_discretizations=(0.05, 0.05),
        tail_truncation=tail_truncation,
        bound_type=_UPPER,
        projection_method=_CTD,
    )


def _assert_same_distribution(*, actual: DenseDiscreteDist, expected: DenseDiscreteDist) -> None:
    assert actual.grid == expected.grid
    np.testing.assert_array_equal(actual.prob_arr, expected.prob_arr)
    assert (actual.p_min, actual.p_max) == (expected.p_min, expected.p_max)


_GAUSSIAN_LAW = gaussian_loss_law(1.5)
_GAUSSIAN_DUAL_LAW = gaussian_loss_law(1.5)
_law_bases = partial(
    continuous_base_distributions,
    convolution_method=ConvolutionMethod.GEOM,
    max_grid_mult=-1,
    max_cells_per_band=1 << 14,
    law=_GAUSSIAN_LAW,
    dual=_GAUSSIAN_DUAL_LAW,
)
_realization_bases = partial(
    realization_base_distributions,
    convolution_method=ConvolutionMethod.GEOM,
    max_grid_mult=-1,
    max_cells_per_band=1 << 14,
    remove_realization=_PAIR_REALIZATION[Direction.REMOVE],
    add_realization=_PAIR_REALIZATION[Direction.ADD],
)
_BASE_INPUTS = [
    pytest.param(_law_bases, id="law"),
    pytest.param(_realization_bases, id="realizations"),
]


@pytest.mark.parametrize("num_steps", [2, 3, 5])
@pytest.mark.parametrize("base_distributions_creation", _BASE_INPUTS)
def test_connect_the_dots_remove_base_is_the_dual_of_the_add_base(
    base_distributions_creation, num_steps: int
):
    """One sum under Q gives the add PLD, and the remove PLD is its dual."""
    remove_base, add_base = _connect_the_dots_bases(
        base_distributions_creation=base_distributions_creation,
        num_steps=num_steps,
        tail_truncation=1e-10,
    )

    dual_of_add_base = calc_pld_dual(add_base)
    _assert_same_distribution(actual=remove_base, expected=dual_of_add_base)


@pytest.mark.parametrize("num_steps", [2, 3, 5])
def test_connect_the_dots_base_keeps_the_boundary_masses_of_the_exact_sum(num_steps: int):
    """The add mass at ``+inf`` is the sum's atom at zero; the remove one is the mean deficit."""
    base_distributions_creation = partial(
        realization_base_distributions,
        convolution_method=ConvolutionMethod.GEOM,
        max_grid_mult=-1,
        max_cells_per_band=1 << 14,
        remove_realization=_PAIR_REALIZATION[Direction.REMOVE],
        add_realization=_PAIR_REALIZATION[Direction.ADD],
    )
    remove_base, add_base = _connect_the_dots_bases(
        base_distributions_creation=base_distributions_creation,
        num_steps=num_steps,
        tail_truncation=0.0,
    )

    assert add_base.p_min == 0.0 and remove_base.p_min == 0.0
    assert add_base.p_max == pytest.approx(_PAIR.q_mass_at_zero**num_steps, rel=1e-12)
    assert remove_base.p_max == pytest.approx(_PAIR.p_mass_at_infinity, rel=1e-12)


# A second mechanism, used as an add input that is not the dual of ``_PAIR``'s remove PLD.
_OTHER_PAIR = lattice_pair(
    step=_PAIR_STEP, indices=np.array([-2, 0, 1]), p_masses=np.array([0.15, 0.5, 0.3])
)


@pytest.mark.parametrize("projection_method", _UPPER_ROUTES)
@pytest.mark.parametrize("convolution_method", _GRIDS)
def test_which_realization_each_direction_is_built_from(
    convolution_method: ConvolutionMethod,
    projection_method: ProjectionMethod,
):
    """Rounding builds each direction from its own realization; connect-the-dots from add.

    The add realization here is of another mechanism than the remove one. With
    connect-the-dots and more than one step the remove PLD is the dual of the add
    PLD, so it bounds the remove profile of the add realization's mechanism.
    """
    num_steps = 3
    other_add_realization = add_realization(pair=_OTHER_PAIR, step=_PAIR_STEP)
    config = _config(
        max_grid_mult=-1,
        convolution_method=convolution_method,
        projection_method=projection_method,
        loss_discretization=0.05,
    )
    pld = general_allocation_pld(
        num_steps=num_steps,
        num_selected=1,
        num_epochs=1,
        remove_realization=_PAIR_REALIZATION[Direction.REMOVE],
        add_realization=other_add_realization,
        config=config,
        bound_type=_UPPER,
    )
    remove_pair = _OTHER_PAIR if projection_method == _CTD else _PAIR

    for epsilon in (0.0, 0.05, 0.2, 0.45):
        remove_delta = float(pld._pmf_remove.get_delta_for_epsilon(epsilon))
        exact_remove = exact_remove_delta(pair=remove_pair, num_steps=num_steps, epsilon=epsilon)
        _assert_bounds(delta=remove_delta, exact=exact_remove, bound_type=_UPPER)
        add_delta = float(pld._pmf_add.get_delta_for_epsilon(epsilon))
        exact_add = exact_add_delta(pair=_OTHER_PAIR, num_steps=num_steps, epsilon=epsilon)
        _assert_bounds(delta=add_delta, exact=exact_add, bound_type=_UPPER)


_COMPONENT_QUERIES = [
    pytest.param(10, 1, 1, -1, id="one-component"),
    pytest.param(13, 3, 2, -1, id="floor-and-ceil-epochs"),
    pytest.param(3, 2, 2, -1, id="one-step-floor-two-step-ceil"),
    pytest.param(13, 3, 2, 300, id="binding-grid-cap"),
]


@pytest.mark.parametrize("num_steps, num_selected, num_epochs, max_grid_mult", _COMPONENT_QUERIES)
def test_gaussian_connect_the_dots_directions_bound_the_rounding_lower_bounds(
    num_steps: int, num_selected: int, num_epochs: int, max_grid_mult: int
):
    """Each direction of a connect-the-dots query is above that direction's lower bound.

    The remove direction is the dual of the add sum, taken per floor/ceil component
    before the epochs are composed; it is compared on its own so that the larger
    direction cannot hide it.
    """
    params = PrivacyParams(
        sigma=1.5, num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_REGRID_WARNING)
        plds = []
        for method, bound_type in ((_CTD, _UPPER), (_ROUNDING, _LOWER)):
            config = _config(
                projection_method=method,
                loss_discretization=0.05,
                max_grid_mult=max_grid_mult,
                convolution_method=_GEOM,
            )
            pld = gaussian_allocation_pld(params=params, config=config, bound_type=bound_type)
            plds.append(pld)
        upper, lower = plds[0], plds[1]

    for epsilon in (0.0, 0.1, 0.5, 1.0, 2.0):
        for name in ("_pmf_remove", "_pmf_add"):
            assert float(getattr(lower, name).get_delta_for_epsilon(epsilon)) <= float(
                getattr(upper, name).get_delta_for_epsilon(epsilon)
            )


@pytest.mark.parametrize("convolution_method", _GRIDS)
def test_connect_the_dots_gives_no_lower_bound(convolution_method: ConvolutionMethod):
    """Both input kinds reject connect-the-dots with a dominated bound, on either grid."""
    config = _config(
        max_grid_mult=-1,
        convolution_method=convolution_method,
        projection_method=_CTD,
        loss_discretization=0.05,
    )
    params = PrivacyParams(sigma=1.0, num_steps=4)
    with pytest.raises(ValueError, match="DOMINATES only"):
        gaussian_allocation_pld(params=params, config=config, bound_type=_LOWER)
    with pytest.raises(ValueError, match="DOMINATES only"):
        general_allocation_pld(
            num_steps=4,
            num_selected=1,
            num_epochs=1,
            remove_realization=_PAIR_REALIZATION[Direction.REMOVE],
            add_realization=_PAIR_REALIZATION[Direction.ADD],
            config=config,
            bound_type=_LOWER,
        )


# =============================================================================
# Multiscale FFT
# =============================================================================

_MS_FFT_UPPER_ROUTES = [
    pytest.param(_ROUNDING, id="rounding"),
    pytest.param(_CTD, id="ctd"),
]
_CELLS_CAP_WARNING = "max_cells_per_band"
_CLOSING_CAP_WARNING = "max_closing_grid"


def _multiscale_plds(
    *,
    params: PrivacyParams,
    loss_discretization: float,
    bounds: tuple[tuple[ProjectionMethod, BoundType], ...],
    **caps: int,
) -> tuple[list, list[str]]:
    """Return the multiscale PLDs of the given projections and bounds, and the warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plds = []
        for method, bound_type in bounds:
            config = AllocationSchemeConfig(
                loss_discretization=loss_discretization,
                tail_truncation=1e-10,
                convolution_method=_MS_FFT,
                projection_method=method,
                **caps,
            )
            pld = gaussian_allocation_pld(params=params, config=config, bound_type=bound_type)
            plds.append(pld)
    return plds, [str(item.message) for item in caught]


def _assert_directions_ordered(*, lower, upper) -> None:
    """Each direction of ``lower`` is at most that direction of ``upper``, at several epsilons."""
    for epsilon in (0.0, 0.1, 0.5, 1.0, 2.0):
        for name in ("_pmf_remove", "_pmf_add"):
            assert float(getattr(lower, name).get_delta_for_epsilon(epsilon)) <= float(
                getattr(upper, name).get_delta_for_epsilon(epsilon)
            )


_TREE_DEPTHS = [
    pytest.param(1, _CTD, Direction.ADD, 0, id="one-step"),
    pytest.param(8, _CTD, Direction.REMOVE, 3, id="ctd-eight"),
    pytest.param(9, _CTD, Direction.ADD, 4, id="ctd-nine"),
    pytest.param(9, _ROUNDING, Direction.ADD, 4, id="rounding-add-nine"),
    pytest.param(9, _ROUNDING, Direction.REMOVE, 4, id="rounding-remove-nine"),
    pytest.param(10, _ROUNDING, Direction.REMOVE, 5, id="rounding-remove-ten"),
]


@pytest.mark.parametrize("num_steps, projection_method, direction, tree_depth", _TREE_DEPTHS)
def test_multiscale_grid_step_is_the_loss_discretization_over_the_stages(
    num_steps: int, projection_method: ProjectionMethod, direction: Direction, tree_depth: int
):
    """The stages are the input, the levels of the merge tree and the closing projection.

    Rounding in the remove direction sums one copy less and merges the factor under
    ``P`` last, so its tree is one level deeper when ``num_steps - 1`` is a power of two.
    """
    loss_discretization = 0.06
    params = PrivacyParams(sigma=2.0, num_steps=num_steps)
    config = _config(
        max_grid_mult=-1,
        convolution_method=_MS_FFT,
        projection_method=projection_method,
        loss_discretization=loss_discretization,
    )
    dist = allocation_dists(
        pld_builder=gaussian_allocation_pld, params=params, config=config, bound_type=_UPPER
    )[direction]

    assert dist.step == loss_discretization / (tree_depth + 2)


@pytest.mark.parametrize("projection_method", _MS_FFT_UPPER_ROUTES)
def test_binding_cells_cap_warns_with_the_achieved_spacing(projection_method: ProjectionMethod):
    """A cap on the points per octave coarsens the grid to ``log(2) / m`` and says so."""
    params = PrivacyParams(sigma=2.0, num_steps=8)
    cap = 64
    (capped, lower), messages = _multiscale_plds(
        params=params,
        loss_discretization=0.01,
        bounds=((projection_method, _UPPER), (_ROUNDING, _LOWER)),
        max_cells_per_band=cap,
    )
    (free,), free_messages = _multiscale_plds(
        params=params, loss_discretization=0.01, bounds=((projection_method, _UPPER),)
    )

    achieved = math.log(2.0) / cap
    assert any(_CELLS_CAP_WARNING in text and f"{achieved:.6e}" in text for text in messages)
    assert not any(_CELLS_CAP_WARNING in text for text in free_messages)
    # The returned grid has the achieved spacing as its step, and the bound stays valid.
    assert capped._pmf_remove._discretization == pytest.approx(achieved, rel=1e-12)
    assert free._pmf_remove._discretization < achieved
    _assert_directions_ordered(lower=lower, upper=capped)


@pytest.mark.parametrize("projection_method", _MS_FFT_UPPER_ROUTES)
def test_binding_closing_grid_cap_warns_and_coarsens_the_returned_grid(
    projection_method: ProjectionMethod,
):
    """A cap on the returned grid coarsens its step; the bound stays above the lower bound."""
    params = PrivacyParams(sigma=2.0, num_steps=8)
    cap = 200
    (capped, lower), messages = _multiscale_plds(
        params=params,
        loss_discretization=0.01,
        bounds=((projection_method, _UPPER), (_ROUNDING, _LOWER)),
        max_closing_grid=cap,
    )

    assert any(_CLOSING_CAP_WARNING in text for text in messages)
    assert capped._pmf_remove._discretization > 0.01 / 5
    assert capped._pmf_remove._probs.size <= cap
    _assert_directions_ordered(lower=lower, upper=capped)


@pytest.mark.parametrize("cap", [4, 5, 7, 16])
@pytest.mark.parametrize("projection_method", _MS_FFT_UPPER_ROUTES)
def test_returned_grids_have_at_most_the_closing_cap_of_points_down_to_the_smallest_cap(
    projection_method: ProjectionMethod, cap: int
):
    """The losses of two steps span zero, where the aligned grid needs a point on each side."""
    params = PrivacyParams(sigma=2.0, num_steps=2)
    (capped,), _ = _multiscale_plds(
        params=params,
        loss_discretization=0.01,
        bounds=((projection_method, _UPPER),),
        max_closing_grid=cap,
    )

    assert capped._pmf_remove._probs.size <= cap
    assert capped._pmf_add._probs.size <= cap


@pytest.mark.parametrize("projection_method", _MS_FFT_UPPER_ROUTES)
def test_caps_that_give_the_components_different_steps_keep_the_bound_valid(
    projection_method: ProjectionMethod,
):
    """A non-divisible query whose floor and ceil components end on different steps.

    The cap on the returned grid binds for the wider component only, so the two
    components are aligned after their epochs are composed.
    """
    params = PrivacyParams(sigma=1.5, num_steps=13, num_selected=3, num_epochs=2)
    (capped, lower), messages = _multiscale_plds(
        params=params,
        loss_discretization=0.05,
        bounds=((projection_method, _UPPER), (_ROUNDING, _LOWER)),
        max_closing_grid=150,
    )

    assert any(_REGRID_WARNING in text for text in messages)
    _assert_directions_ordered(lower=lower, upper=capped)


_SELECTION_QUERIES = [
    pytest.param(12, 3, 2, id="divisible"),
    pytest.param(13, 3, 2, id="non-divisible"),
    pytest.param(3, 2, 2, id="one-step-floor-two-step-ceil"),
]


@pytest.mark.parametrize("num_steps, num_selected, num_epochs", _SELECTION_QUERIES)
def test_each_direction_of_every_upper_bound_is_above_every_lower_bound(
    num_steps: int, num_selected: int, num_epochs: int
):
    """Over both grids and both projections, direction by direction."""
    params = PrivacyParams(
        sigma=1.5, num_steps=num_steps, num_selected=num_selected, num_epochs=num_epochs
    )
    plds = {}
    for grid in (_GEOM, _MS_FFT):
        for method, bound_type in _PROJECTION_BOUNDS:
            config = _config(
                max_grid_mult=-1,
                convolution_method=grid,
                projection_method=method,
                loss_discretization=0.05,
            )
            plds[(grid, method, bound_type)] = gaussian_allocation_pld(
                params=params, config=config, bound_type=bound_type
            )

    for lower_key, lower in plds.items():
        for upper_key, upper in plds.items():
            if lower_key[2] == _LOWER and upper_key[2] == _UPPER:
                _assert_directions_ordered(lower=lower, upper=upper)


@pytest.mark.parametrize("num_steps", [2, 3, 5])
def test_multiscale_connect_the_dots_base_keeps_the_boundary_masses_of_the_exact_sum(
    num_steps: int,
):
    """The add mass at ``+inf`` is the sum's atom at zero; the remove one is the mean deficit.

    The remove PLD is the dual of the add PLD on its linear grid, so its mass at
    ``+inf`` is what the mean of the sum is short of one.
    """
    tail_truncation = 1e-15
    base_distributions_creation = partial(
        realization_base_distributions,
        convolution_method=ConvolutionMethod.MS_FFT,
        max_grid_mult=-1,
        remove_realization=_PAIR_REALIZATION[Direction.REMOVE],
        add_realization=_PAIR_REALIZATION[Direction.ADD],
        max_cells_per_band=1 << 14,
    )
    remove_base, add_base = allocation_plds_base(
        convolution_method=ConvolutionMethod.MS_FFT,
        base_distributions_creation=base_distributions_creation,
        num_steps=num_steps,
        loss_discretizations=(0.05, 0.05),
        tail_truncation=tail_truncation,
        bound_type=_UPPER,
        projection_method=_CTD,
        max_closing_grid=1_000_000,
    )

    dual_of_add_base = calc_pld_dual(add_base)
    _assert_same_distribution(actual=remove_base, expected=dual_of_add_base)
    assert add_base.p_min == 0.0 and remove_base.p_min == 0.0
    assert add_base.p_max == pytest.approx(_PAIR.q_mass_at_zero**num_steps, rel=1e-9)
    assert remove_base.p_max == pytest.approx(_PAIR.p_mass_at_infinity, rel=1e-9)


@pytest.mark.parametrize(
    "sigma, num_steps",
    [(0.25, 5), (0.25, 10), (0.3, 2)],
)
def test_multiscale_remove_mass_at_infinity_stays_within_the_tail_budget(
    sigma: float, num_steps: int
):
    """The mass of the sum is repaired by a common factor, which moves no first moment.

    The Gaussian mechanism has no mass at ``+inf``. Small noise puts tiny masses at
    likelihood ratios up to ``2**50``; a repair of the sum's rounding error at those
    points would cost the dual their first moment and show up here.
    """
    tail_truncation = 1e-12
    params = PrivacyParams(sigma=sigma, num_steps=num_steps)
    config = AllocationSchemeConfig(
        loss_discretization=0.05,
        tail_truncation=tail_truncation,
        convolution_method=_MS_FFT,
    )
    dist = allocation_dists(
        pld_builder=gaussian_allocation_pld, params=params, config=config, bound_type=_UPPER
    )[Direction.REMOVE]

    assert dist.p_max <= tail_truncation
