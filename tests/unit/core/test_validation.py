"""Tests for constraint primitives and dataclass construction contracts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    SparseDiscreteDist,
    cast_dense_dist,
    cast_geometric_positives_dist,
    cast_linear_reals_dist,
    require_geometric_positives_dist,
    require_linear_reals_dist,
)
from PLD_accounting.mechanisms import discrete_distribution
from PLD_accounting.projections import project_discrete_dist
from PLD_accounting.projections_stoch_dom import project_dist_onto_grid_stoch_dom
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    Direction,
    PrivacyParams,
    SpacingType,
    require_bound_type,
    require_direction,
)
from PLD_accounting.validation import (
    require_allocation_counts,
    require_closed_unit_interval,
    require_enum,
    require_finite_real,
    require_integer,
    require_increasing_array,
    require_nonnegative_masses,
    require_open_unit_interval,
    require_positive_int,
    require_positive_real,
    cast_type,
    require_unit_interval_left_open,
)


def test_require_finite_real_rejects_bool_nan_and_inf() -> None:
    """Boolean, NaN, and infinite values are not finite reals."""
    with pytest.raises(TypeError, match="sigma must be a real number"):
        require_finite_real(value=True, name="sigma")
    with pytest.raises(ValueError, match="sigma must be finite"):
        require_finite_real(value=math.nan, name="sigma")
    with pytest.raises(ValueError, match="sigma must be finite"):
        require_finite_real(value=math.inf, name="sigma")


def test_require_positive_real_rejects_nonpositive() -> None:
    """Positive reals exclude zero and negatives."""
    with pytest.raises(ValueError, match="scale must be positive"):
        require_positive_real(value=0.0, name="scale")
    with pytest.raises(ValueError, match="scale must be positive"):
        require_positive_real(value=-1.0, name="scale")


def test_require_positive_real_accepts_parallel_sequences() -> None:
    """Scalar helpers accept parallel value and name sequences."""
    assert (
        require_positive_real(
            value=[1.5, 0.25, 2.0],
            name=["sigma", "loss_discretization", "tail_truncation"],
        )
        is None
    )
    with pytest.raises(ValueError, match="loss_discretization must be positive"):
        require_positive_real(value=[1.5, 0.0], name=["sigma", "loss_discretization"])
    with pytest.raises(ValueError, match="values and names must have equal length"):
        require_positive_real(value=[1.5], name=["sigma", "loss_discretization"])
    with pytest.raises(TypeError, match="values must be a sequence"):
        require_positive_real(value=1.5, name=["sigma"])


def test_require_integer_rejects_bool() -> None:
    """``bool`` is a subclass of ``int`` and must still be rejected."""
    with pytest.raises(TypeError, match="num_steps must be an integer"):
        require_integer(value=True, name="num_steps")


def test_require_positive_int_rejects_zero() -> None:
    """Positive integers start at 1."""
    with pytest.raises(ValueError, match="num_convolutions must be >= 1"):
        require_positive_int(value=0, name="num_convolutions")


def test_require_open_unit_interval_is_exclusive() -> None:
    """DP ``delta`` lives in ``(0, 1)``."""
    with pytest.raises(ValueError, match=r"delta must be in \(0, 1\)"):
        require_open_unit_interval(value=0.0, name="delta")
    with pytest.raises(ValueError, match=r"delta must be in \(0, 1\)"):
        require_open_unit_interval(value=1.0, name="delta")
    assert require_open_unit_interval(value=0.5, name="delta") is None


def test_require_unit_interval_left_open_allows_one() -> None:
    """Sampling probability lives in ``(0, 1]``."""
    with pytest.raises(ValueError, match=r"sampling_probability must be in \(0, 1\]"):
        require_unit_interval_left_open(value=0.0, name="sampling_probability")
    assert require_unit_interval_left_open(value=1.0, name="sampling_probability") is None


def test_require_closed_unit_interval_is_exact() -> None:
    """The closed unit interval admits its endpoints and rejects the smallest overshoot."""
    assert require_closed_unit_interval(value=0.0, name="p_min") is None
    assert (
        require_closed_unit_interval(value=[0.0, 1.0], name=["expected_p_min", "expected_p_max"])
        is None
    )
    with pytest.raises(ValueError, match="p_min must be in \\[0, 1\\]"):
        require_closed_unit_interval(value=-1e-300, name="p_min")
    with pytest.raises(ValueError, match="p_max must be in \\[0, 1\\]"):
        require_closed_unit_interval(value=float(np.nextafter(1.0, 2.0)), name="p_max")


def test_require_enum_allows_both_unless_restricted() -> None:
    """``BOTH`` is a real member; APIs that reject it pass an ``allowed`` subset."""
    assert require_enum(value=Direction.BOTH, enum_cls=Direction, name="direction") is None
    with pytest.raises(ValueError, match="direction must be one of"):
        require_enum(
            value=Direction.BOTH,
            enum_cls=Direction,
            name="direction",
            allowed=(Direction.ADD, Direction.REMOVE),
        )
    assert require_enum(value=BoundType.BOTH, enum_cls=BoundType, name="bound_type") is None
    with pytest.raises(ValueError, match="bound_type must be one of"):
        require_enum(
            value=BoundType.BOTH,
            enum_cls=BoundType,
            name="bound_type",
            allowed=(BoundType.DOMINATES, BoundType.IS_DOMINATED),
        )


def test_require_bound_type_rejects_both() -> None:
    """The repeated one-sided bound-type check lives in one wrapper."""
    assert require_bound_type(value=BoundType.DOMINATES) is None
    assert require_bound_type(value=BoundType.IS_DOMINATED) is None
    with pytest.raises(ValueError, match="bound_type must be one of"):
        require_bound_type(value=BoundType.BOTH)


def test_require_direction_rejects_both() -> None:
    """The repeated one-sided direction check lives in one wrapper."""
    assert require_direction(value=Direction.ADD) is None
    assert require_direction(value=Direction.REMOVE) is None
    with pytest.raises(ValueError, match="direction must be one of"):
        require_direction(value=Direction.BOTH)


def test_cast_type_names_the_argument() -> None:
    """Checked casting preserves identity and names invalid arguments."""
    config = AllocationSchemeConfig()
    assert cast_type(value=config, expected_type=AllocationSchemeConfig, name="config") is config
    with pytest.raises(TypeError, match="config must be AllocationSchemeConfig"):
        cast_type(value="fft", expected_type=AllocationSchemeConfig, name="config")


def test_require_allocation_counts_enforces_order() -> None:
    """Selection count cannot exceed the number of steps."""
    require_allocation_counts(num_steps=4, num_selected=2, num_epochs=1)
    with pytest.raises(ValueError, match="num_selected"):
        require_allocation_counts(num_steps=2, num_selected=4, num_epochs=1)


def test_require_nonnegative_masses_names_boundary_fields() -> None:
    """Boundary-mass errors use ``p_min`` / ``p_max``, not ``min`` / ``max``."""
    prob_arr = np.array([1.0])
    with pytest.raises(ValueError, match="p_min must be nonnegative"):
        require_nonnegative_masses(prob_arr=prob_arr, p_min=-0.1, p_max=0.0)
    with pytest.raises(ValueError, match="p_max must be nonnegative"):
        require_nonnegative_masses(prob_arr=prob_arr, p_min=0.0, p_max=-0.1)


def test_cast_dense_dist_checks_type_spacing_and_domain() -> None:
    """Dense-dist checks live next to the type, not in ``validation.py``."""
    sparse = SparseDiscreteDist(x_array=np.array([0.0, 1.0]), prob_arr=np.array([0.5, 0.5]))
    with pytest.raises(TypeError, match="dist must be DenseDiscreteDist with LINEAR spacing"):
        cast_dense_dist(dist=sparse, name="dist", spacing=SpacingType.LINEAR)

    geometric_grid = GridSpec(
        step=math.log(2.0), n=2, spacing_type=SpacingType.GEOMETRIC, anchor=1.0
    )
    geometric = DenseDiscreteDist(
        grid=geometric_grid, prob_arr=np.array([0.4, 0.6]), domain=Domain.POSITIVES
    )
    with pytest.raises(TypeError, match="expected DenseDiscreteDist with LINEAR spacing"):
        cast_dense_dist(dist=geometric, name="dist", spacing=SpacingType.LINEAR)
    with pytest.raises(ValueError, match=r"must use Domain\.REALS"):
        cast_dense_dist(
            dist=geometric, name="dist", spacing=SpacingType.GEOMETRIC, domain=Domain.REALS
        )
    linear_grid = GridSpec(step=1.0, n=2, anchor=0.0)
    linear = DenseDiscreteDist(grid=linear_grid, prob_arr=np.array([0.4, 0.6]))
    assert cast_dense_dist(dist=linear, name="dist") is linear
    assert cast_linear_reals_dist(dist=linear, name="dist") is linear
    assert require_linear_reals_dist(dist=linear, name="dist") is None
    with pytest.raises(TypeError, match="expected DenseDiscreteDist with LINEAR spacing"):
        cast_linear_reals_dist(dist=geometric, name="dist")
    assert cast_geometric_positives_dist(dist=geometric, name="dist") is geometric
    assert require_geometric_positives_dist(dist=geometric, name="dist") is None
    with pytest.raises(TypeError, match="expected DenseDiscreteDist with GEOMETRIC spacing"):
        cast_geometric_positives_dist(dist=linear, name="dist")


def test_privacy_params_validate_on_construction() -> None:
    """Invalid privacy fields are unconstructable."""
    PrivacyParams(sigma=1.0, num_steps=4)
    with pytest.raises(ValueError, match="sigma must be positive"):
        PrivacyParams(sigma=0.0, num_steps=4)
    with pytest.raises(TypeError, match="num_steps must be an integer"):
        PrivacyParams(sigma=1.0, num_steps=True)
    with pytest.raises(ValueError, match=r"delta must be in \(0, 1\)"):
        PrivacyParams(sigma=1.0, num_steps=4, delta=1.0)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        PrivacyParams(sigma=1.0, num_steps=4, epsilon=0.0)


def test_privacy_params_get_delta_and_epsilon() -> None:
    """Query helpers demand the optional field that the query uses."""
    params = PrivacyParams(sigma=1.0, num_steps=4)
    with pytest.raises(ValueError, match="delta must be in"):
        params.get_delta()
    with pytest.raises(ValueError, match="epsilon must be positive"):
        params.get_epsilon()
    delta = np.float64(1e-5)
    assert PrivacyParams(sigma=1.0, num_steps=4, delta=delta).get_delta() is delta
    epsilon = np.float64(1.0)
    assert PrivacyParams(sigma=1.0, num_steps=4, epsilon=epsilon).get_epsilon() is epsilon


def test_allocation_scheme_config_validates_on_construction() -> None:
    """Config tail budget is strictly positive; ``max_grid_mult <= 0`` stays uncapped."""
    AllocationSchemeConfig(max_grid_mult=-1)
    with pytest.raises(ValueError, match="tail_truncation must be positive"):
        AllocationSchemeConfig(tail_truncation=0.0)
    with pytest.raises(ValueError, match="loss_discretization must be positive"):
        AllocationSchemeConfig(loss_discretization=0.0)
    with pytest.raises(TypeError, match="convolution_method must be ConvolutionMethod"):
        AllocationSchemeConfig(convolution_method="fft")  # type: ignore[arg-type]


@pytest.mark.parametrize("cap", [1, 2, 3])
def test_allocation_scheme_config_rejects_positive_caps_below_four(cap: int) -> None:
    """A positive geometric cap must leave enough points for outward grid alignment."""
    with pytest.raises(ValueError, match="max_grid_mult must be <= 0 or >= 4"):
        AllocationSchemeConfig(max_grid_mult=cap)


@pytest.mark.parametrize("cap", [-1, 0, 4])
def test_allocation_scheme_config_accepts_uncapped_or_usable_grids(cap: int) -> None:
    """Zero and negative disable the cap; four is the smallest supported cap."""
    assert AllocationSchemeConfig(max_grid_mult=cap).max_grid_mult == cap


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_real_loss_projection_rejects_positive_domain(bound_type: BoundType, sparse: bool) -> None:
    """A real-loss projection must not reinterpret an atom at zero as negative infinity."""
    grid = GridSpec(step=1.0, n=2, index_0=1)
    kwargs = {"prob_arr": np.array([0.4, 0.4]), "p_min": 0.2, "domain": Domain.POSITIVES}
    source = (
        SparseDiscreteDist(x_array=grid.materialize(), **kwargs)
        if sparse
        else DenseDiscreteDist(grid=grid, **kwargs)
    )
    with pytest.raises(ValueError, match="requires a real-domain source"):
        project_discrete_dist(
            dist=source,
            tail_truncation=0.0,
            loss_discretization=1.0,
            bound_type=bound_type,
        )


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
def test_generic_grid_projection_preserves_positive_boundary(bound_type: BoundType) -> None:
    """The generic rounding projector keeps zero-boundary semantics on a positive grid."""
    grid = GridSpec(step=1.0, n=2, index_0=1)
    source = DenseDiscreteDist(
        grid=grid, prob_arr=np.array([0.25, 0.25]), p_min=0.5, domain=Domain.POSITIVES
    )
    result = project_dist_onto_grid_stoch_dom(dist=source, grid=grid, bound_type=bound_type)

    assert result.domain == Domain.POSITIVES
    assert result.p_min == 0.5
    np.testing.assert_array_equal(result.prob_arr, source.prob_arr)


@pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
@pytest.mark.parametrize("tail", [-1.0, math.nan, math.inf, -math.inf])
def test_real_loss_projection_rejects_invalid_tail_budget(
    bound_type: BoundType, tail: float
) -> None:
    """Invalid tail budgets fail before either discretization engine runs."""
    source = SparseDiscreteDist(x_array=np.array([0.0, 1.0]), prob_arr=np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="tail_truncation must be (nonnegative|finite)"):
        project_discrete_dist(
            dist=source,
            tail_truncation=tail,
            loss_discretization=0.1,
            bound_type=bound_type,
        )


def test_algorithm_tail_truncation_zero_remains_valid() -> None:
    """Convolution/discretization ``tail_truncation=0`` is an algorithm arg, not a config field."""
    grid = GridSpec(step=1.0, n=3, anchor=0.0)
    noise = DenseDiscreteDist(grid=grid, prob_arr=np.array([0.5, 0.0, 0.5]))
    remove, add = discrete_distribution(
        noise_dist=noise,
        loss_discretization=0.1,
        tail_truncation=0.0,
    )
    assert remove.p_min == 0.0
    assert add.p_min == 0.0


def test_require_increasing_array_preserves_readonly_input() -> None:
    """Validation neither returns a converted array nor writes to the input."""
    values = np.array([-1.0, 0.0, 2.0])
    values.flags.writeable = False
    assert require_increasing_array(values=values, name="loss_points") is None
    np.testing.assert_array_equal(values, [-1.0, 0.0, 2.0])
    assert not values.flags.writeable


def test_require_numeric_sequences_preserve_numpy_scalar_types() -> None:
    """Batch validation leaves the supplied values and their scalar types intact."""
    values = [np.int64(2), np.int64(5)]
    assert require_positive_int(value=values, name=["num_steps", "num_epochs"]) is None
    assert all(isinstance(value, np.int64) for value in values)
    assert values == [2, 5]
