"""Acceptance tests for the structural grid contract.

Every internally regular grid is five fields and two formulas::

    LINEAR:    x[i] = anchor + (index_0 + i) * step
    GEOMETRIC: x[i] = anchor * exp((index_0 + i) * step)

The contract these tests pin down is that structure-preserving operations are exact
integer arithmetic on those fields, so a retained coordinate comes back *bitwise*
identical rather than within a tolerance. That is what stops a downstream ``ceil`` /
``floor`` / ``searchsorted`` from moving an atom a whole bin.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    PLDRealization,
)
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import (
    calc_pld_dual,
    exp_linear_to_geometric,
    log_geometric_to_linear,
    negate_reverse_linear_distribution,
)


def _zero_anchored(step: float, n: int, index_0: int) -> GridSpec:
    return GridSpec(step=step, n=n, index_0=index_0)


def _geometric(ratio: float, n: int, anchor: float, index_0: int) -> GridSpec:
    """Build a geometric grid: multiplicative anchor, log-ratio spacing."""
    return GridSpec(
        step=math.log(ratio),
        spacing_type=SpacingType.GEOMETRIC,
        n=n,
        anchor=anchor,
        index_0=index_0,
    )


# =============================================================================
# Structure-preserving operations are bitwise exact
# =============================================================================


@pytest.mark.parametrize("index_0", [-1_000_000, -7, 0, 3, 999_999])
@pytest.mark.parametrize("step", [1e-6, 2e-4, 1.0])
def test_zero_anchored_slice_and_pad_retain_points_bitwise(step: float, index_0: int) -> None:
    """Slicing and padding shift an integer index; nothing is re-derived."""
    grid = _zero_anchored(step, 40, index_0)
    points = grid.materialize()

    sliced = grid.slice(start=9, n=20)
    np.testing.assert_array_equal(sliced.materialize(), points[9:29])
    assert sliced.index_0 == index_0 + 9

    padded = grid.pad(left=5, right=3)
    np.testing.assert_array_equal(padded.materialize()[5:45], points)
    assert padded.index_0 == index_0 - 5
    assert padded.n == 48

    # A slice of a pad returns to exactly the original grid, fields and all.
    assert padded.slice(start=5, n=grid.n) == grid


@pytest.mark.parametrize("anchor", [1e-300, 1e-14, 0.5, 12345.678])
def test_affine_slice_retains_points_bitwise(anchor: float) -> None:
    """The prototype counterexample: an affine left slice must move no point.

    ``anchor=1e-300, step=1e-6, left slice 10`` moved a retained coordinate by one ULP
    when the child origin was reconstructed as a float.
    """
    grid = GridSpec(step=1e-6, n=64, anchor=anchor)
    points = grid.materialize()

    sliced = grid.slice(start=10, n=54)
    np.testing.assert_array_equal(sliced.materialize(), points[10:64])
    assert sliced.anchor == anchor


def test_double_reflection_is_the_identity() -> None:
    """Reflecting twice returns the same lattice, not an approximation of it."""
    for grid in (
        _zero_anchored(2e-4, 33, -17),
        GridSpec(step=0.25, n=9, anchor=1e-300),
        GridSpec(step=1.0, n=4, anchor=float(np.nextafter(0.0, 1.0))),
    ):
        assert grid.reflect().reflect() == grid
        np.testing.assert_array_equal(grid.reflect().materialize(), -grid.materialize()[::-1])


@pytest.mark.parametrize("step", [2e-4, 1e-3, 1e-5, 0.05, 0.5])
def test_exp_log_round_trip_is_exact_in_step_and_coordinates(step: float) -> None:
    """exp/log reinterpret the stored numbers, so a round trip changes nothing.

    Storing the ratio instead of the log spacing loses the step on the way out and back.
    """
    grid = GridSpec(step=step, n=64, index_0=-20)
    exp_grid = grid.exp()
    assert exp_grid.spacing_type == SpacingType.GEOMETRIC
    assert exp_grid.index_0 == grid.index_0

    back = exp_grid.log()
    assert back == grid  # structural identity, not just anchor and index
    assert back.step == grid.step
    np.testing.assert_array_equal(back.materialize(), grid.materialize())


def test_exp_grid_materializes_as_the_exponential_of_the_linear_grid() -> None:
    """The geometric support is exactly ``exp`` of the loss support, bit for bit."""
    grid = GridSpec(step=2e-4, n=1000, index_0=-500)
    np.testing.assert_array_equal(grid.exp().materialize(), np.exp(grid.materialize()))


def test_affine_exp_keeps_the_step_but_not_coordinate_identity() -> None:
    """The documented guarantee is the step, not bitwise coordinates, on affine grids."""
    grid = GridSpec(step=2e-4, n=64, anchor=-3.25, index_0=-20)
    exp_grid = grid.exp()
    assert exp_grid.step == grid.step  # the spacing always survives
    assert not np.array_equal(exp_grid.materialize(), np.exp(grid.materialize()))


@pytest.mark.parametrize("index_0", [-30, -1, 0, 5])
def test_geometric_slice_keeps_exponents_not_origins(index_0: int) -> None:
    """The legacy geometric truncation regression: [2, 6, 18, 54][1:3] is [6, 18]."""
    grid = _geometric(3.0, 4, 2.0, index_0)
    points = grid.materialize()
    sliced = grid.slice(start=1, n=2)
    np.testing.assert_array_equal(sliced.materialize(), points[1:3])
    assert sliced.index_0 == index_0 + 1


def test_geometric_slice_regression_exact_values() -> None:
    """Spelled out: slicing [2, 6, 18, 54] at index 1 never yields [5, 15]."""
    grid = _geometric(3.0, 4, 2.0, 0)
    np.testing.assert_allclose(grid.materialize(), [2.0, 6.0, 18.0, 54.0], rtol=1e-14)
    np.testing.assert_array_equal(grid.slice(start=1, n=2).materialize(), grid.materialize()[1:3])


def test_convolution_indices_add_algebraically() -> None:
    """The sum lattice is derived, never searched for in materialized coordinates."""
    grid_1 = _zero_anchored(0.01, 7, -3)
    grid_2 = _zero_anchored(0.01, 5, 11)
    out = grid_1.convolve(other=grid_2)
    assert out.index_0 == -3 + 11
    assert out.n == 7 + 5 - 1
    assert out.anchor == 0.0
    assert out.x_0 == grid_1.x_0 + grid_2.x_0

    assert grid_1.self_convolve(num_convolutions=4).index_0 == -12
    assert grid_1.self_convolve(num_convolutions=4).n == 4 * (7 - 1) + 1


def test_convolution_requires_one_exact_step() -> None:
    """Numerically close but distinct steps are different lattices."""
    grid = _zero_anchored(0.01, 4, 0)
    other = _zero_anchored(float(np.nextafter(0.01, 1.0)), 4, 0)
    with pytest.raises(ValueError, match="one exact step"):
        grid.convolve(other=other)


# =============================================================================
# Equality is structural, not coordinate-wise
# =============================================================================


def test_equality_distinguishes_grids_whose_coordinates_coincide() -> None:
    """A nextafter anchor is its own lattice even where the doubles agree."""
    zero_anchored = GridSpec(step=1.0, n=3)
    affine = GridSpec(step=1.0, n=3, anchor=float(np.nextafter(0.0, 1.0)))
    assert zero_anchored != affine
    # The materialized arrays are *not* identical here, but equality never consults
    # them: it is decided on the five fields alone.
    assert zero_anchored == GridSpec(step=1.0, n=3, anchor=0.0, index_0=0)


def test_equality_does_not_materialize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Comparing million-point grids must not allocate their coordinate arrays."""
    grid_1 = _zero_anchored(1e-6, 1_000_000, -500_000)
    grid_2 = _zero_anchored(1e-6, 1_000_000, -500_000)

    def explode(self: GridSpec) -> np.ndarray:
        raise AssertionError("equality must not materialize coordinates")

    monkeypatch.setattr(GridSpec, "materialize", explode)
    assert grid_1 == grid_2
    assert grid_1 != grid_2.slice(start=1, n=999_999)


# =============================================================================
# The contract survives the public route
# =============================================================================


def test_truncation_through_the_public_route_is_an_index_slice() -> None:
    """``truncate_edges`` must slice the lattice, not rebuild an origin."""
    grid = _zero_anchored(1e-3, 200, -50)
    prob = np.zeros(200)
    prob[40:160] = 1.0 / 120
    realization = PLDRealization(grid=grid, prob_arr=prob)

    truncated = realization.truncate_edges(tail_truncation=0.0, bound_type=BoundType.DOMINATES)

    assert truncated.grid == grid.slice(start=40, n=120)
    np.testing.assert_array_equal(truncated.x_array, grid.materialize()[40:160])


def test_dual_and_reflection_round_trip_preserves_the_loss_grid() -> None:
    """Dualizing then reflecting returns the source lattice exactly."""
    # Nonnegative losses keep E[exp(-L)] <= 1, so this is a valid realization.
    grid = _zero_anchored(5e-4, 101, 0)
    prob = np.full(101, 1.0 / 101)
    realization = PLDRealization(grid=grid, prob_arr=prob)

    reflected_dual = negate_reverse_linear_distribution(calc_pld_dual(realization))

    assert reflected_dual.grid == grid
    np.testing.assert_array_equal(reflected_dual.x_array, grid.materialize())


def test_geometric_transform_round_trip_through_distributions() -> None:
    """exp then log returns a distribution on the original loss lattice."""
    grid = _zero_anchored(1e-3, 50, -20)
    dist = DenseDiscreteDist(grid=grid, prob_arr=np.full(50, 1.0 / 50))

    back = log_geometric_to_linear(exp_linear_to_geometric(dist))

    assert back.grid.anchor == 0.0
    assert back.grid.index_0 == grid.index_0
    assert back.domain == Domain.REALS


# =============================================================================
# Structural validity: a GridSpec must not be constructible in a broken state
# =============================================================================


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"step": 1.0, "n": 3, "index_0": 0.5}, "must be an integer"),
        ({"step": 1.0, "n": True}, "must be an integer"),
        ({"step": 1.0, "n": 3, "index_0": True}, "must be an integer"),
    ],
)
def test_fractional_and_boolean_indices_are_rejected(kwargs: dict, match: str) -> None:
    """A lattice index that is not an integer silently produces an off-lattice grid."""
    with pytest.raises(TypeError, match=match):
        GridSpec(**kwargs)


def test_index_range_leaving_int64_is_rejected() -> None:
    """Index arithmetic is int64; an overflowing range wraps into a descending grid."""
    with pytest.raises(ValueError, match="leaves int64"):
        GridSpec(step=1.0, n=2, index_0=2**63 - 1)


def test_non_increasing_materialization_is_rejected() -> None:
    """A grid whose points do not increase is not a grid, however it arose."""
    with pytest.raises(ValueError, match="strictly increasing"):
        # Below the resolution of its own origin, so every point collapses to one value.
        GridSpec(step=5e-324, n=4, anchor=1e300)


def test_dense_support_is_exactly_the_grid_it_was_built_from() -> None:
    """A dense support is its ``GridSpec``, never an array a lattice was inferred from.

    Inference is what let ``[0, 1, 2.0000001]`` come back as ``[0, 1.00000005, 2.0000001]``,
    moving an atom by 5e-8. Taking the grid as given removes the failure mode.
    """
    grid = GridSpec(step=0.25, n=5, index_0=-2)
    dist = DenseDiscreteDist(grid=grid, prob_arr=np.full(5, 0.2))
    np.testing.assert_array_equal(dist.x_array, grid.materialize())
    assert dist.grid == grid


def test_singleton_grid_validates_only_its_own_point() -> None:
    """A one-point lattice has no cell, so a nonexistent neighbor must not be evaluated.

    Both fixtures below have a perfectly representable single point whose ``point(1)``
    overflows; rejecting them would refuse a lattice that is entirely well formed.
    """
    geometric = GridSpec(
        step=10.0,
        n=1,
        spacing_type=SpacingType.GEOMETRIC,
        anchor=1e307,
        index_0=0,
    )
    assert geometric.x_0 == 1e307
    assert math.isinf(geometric.anchor * math.exp(geometric.step))

    linear = GridSpec(step=1e308, n=1, anchor=1.7e308, index_0=0)
    assert linear.x_0 == 1.7e308
    assert math.isinf(linear.anchor + linear.step)

    # Two points still validate the cell they actually have.
    with pytest.raises(ValueError):
        GridSpec(step=1e308, n=2, anchor=1.7e308, index_0=0)
