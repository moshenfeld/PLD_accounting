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
    require_dense_dist,
    require_geometric_positives_dist,
    require_linear_reals_dist,
)
from PLD_accounting.mechanisms import discrete_distribution
from PLD_accounting.types import (
    AllocationSchemeConfig,
    BoundType,
    Direction,
    PrivacyParams,
    SpacingType,
    require_allocation_config,
    require_bound_type,
    require_direction,
    require_privacy_params,
)
from PLD_accounting.validation import (
    require_allocation_counts,
    require_closed_unit_interval,
    require_enum,
    require_finite_real,
    require_integer,
    require_nonnegative_masses,
    require_open_unit_interval,
    require_positive_int,
    require_positive_real,
    require_type,
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
    assert require_positive_real(
        value=[1.5, 0.25, 2.0],
        name=["sigma", "loss_discretization", "tail_truncation"],
    ) == [1.5, 0.25, 2.0]
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
    assert require_open_unit_interval(value=0.5, name="delta") == 0.5


def test_require_unit_interval_left_open_allows_one() -> None:
    """Sampling probability lives in ``(0, 1]``."""
    with pytest.raises(ValueError, match=r"sampling_probability must be in \(0, 1\]"):
        require_unit_interval_left_open(value=0.0, name="sampling_probability")
    assert require_unit_interval_left_open(value=1.0, name="sampling_probability") == 1.0


def test_require_closed_unit_interval_clamps_overshoot_within_atol() -> None:
    """Float overshoot inside ``atol`` is a representation artifact and is clamped."""
    atol = 1e-10
    assert require_closed_unit_interval(value=-atol / 2, name="p_min", atol=atol) == 0.0
    assert require_closed_unit_interval(value=1.0 + atol / 2, name="p_max", atol=atol) == 1.0
    assert require_closed_unit_interval(
        value=[-atol / 2, 1.0 + atol / 2],
        name=["expected_p_min", "expected_p_max"],
        atol=atol,
    ) == [0.0, 1.0]


def test_require_closed_unit_interval_rejects_values_beyond_atol() -> None:
    """Overshoot larger than ``atol`` is a real out-of-range value."""
    atol = 1e-10
    with pytest.raises(ValueError, match="p_min must be in \\[0, 1\\]"):
        require_closed_unit_interval(value=-2 * atol, name="p_min", atol=atol)
    with pytest.raises(ValueError, match="p_max must be in \\[0, 1\\]"):
        require_closed_unit_interval(value=1.0 + 2 * atol, name="p_max", atol=atol)
    with pytest.raises(ValueError, match="atol must be nonnegative"):
        require_closed_unit_interval(value=0.5, name="p_min", atol=-1.0)


def test_require_enum_allows_both_unless_restricted() -> None:
    """``BOTH`` is a real member; APIs that reject it pass an ``allowed`` subset."""
    assert (
        require_enum(value=Direction.BOTH, enum_cls=Direction, name="direction") is Direction.BOTH
    )
    with pytest.raises(ValueError, match="direction must be one of"):
        require_enum(
            value=Direction.BOTH,
            enum_cls=Direction,
            name="direction",
            allowed=(Direction.ADD, Direction.REMOVE),
        )
    assert (
        require_enum(value=BoundType.BOTH, enum_cls=BoundType, name="bound_type") is BoundType.BOTH
    )
    with pytest.raises(ValueError, match="bound_type must be one of"):
        require_enum(
            value=BoundType.BOTH,
            enum_cls=BoundType,
            name="bound_type",
            allowed=(BoundType.DOMINATES, BoundType.IS_DOMINATED),
        )


def test_require_bound_type_rejects_both() -> None:
    """The repeated one-sided bound-type check lives in one wrapper."""
    assert require_bound_type(value=BoundType.DOMINATES) is BoundType.DOMINATES
    assert require_bound_type(value=BoundType.IS_DOMINATED) is BoundType.IS_DOMINATED
    with pytest.raises(ValueError, match="bound_type must be one of"):
        require_bound_type(value=BoundType.BOTH)


def test_require_direction_rejects_both() -> None:
    """The repeated one-sided direction check lives in one wrapper."""
    assert require_direction(value=Direction.ADD) is Direction.ADD
    assert require_direction(value=Direction.REMOVE) is Direction.REMOVE
    with pytest.raises(ValueError, match="direction must be one of"):
        require_direction(value=Direction.BOTH)


def test_require_privacy_params_and_allocation_config() -> None:
    """Dataclass instance checks share one wrapper per type."""
    params = PrivacyParams(sigma=1.0, num_steps=4)
    config = AllocationSchemeConfig()
    assert require_privacy_params(value=params) is params
    assert require_allocation_config(value=config) is config
    with pytest.raises(TypeError, match="params must be PrivacyParams"):
        require_privacy_params(value="params")
    with pytest.raises(TypeError, match="config must be AllocationSchemeConfig"):
        require_allocation_config(value="fft")


def test_require_type_names_the_argument() -> None:
    """Type errors name the argument and the expected class."""
    with pytest.raises(TypeError, match="config must be AllocationSchemeConfig"):
        require_type(value="fft", expected_type=AllocationSchemeConfig, name="config")


def test_require_allocation_counts_enforces_order() -> None:
    """Selection count cannot exceed the number of steps."""
    require_allocation_counts(num_steps=4, num_selected=2, num_epochs=1)
    with pytest.raises(ValueError, match="num_selected"):
        require_allocation_counts(num_steps=2, num_selected=4, num_epochs=1)


def test_require_nonnegative_masses_names_boundary_fields() -> None:
    """Boundary-mass errors use ``p_min`` / ``p_max``, not ``min`` / ``max``."""
    with pytest.raises(ValueError, match="p_min must be nonnegative"):
        require_nonnegative_masses(prob_arr=np.array([1.0]), p_min=-0.1, p_max=0.0)
    with pytest.raises(ValueError, match="p_max must be nonnegative"):
        require_nonnegative_masses(prob_arr=np.array([1.0]), p_min=0.0, p_max=-0.1)


def test_require_dense_dist_checks_type_spacing_and_domain() -> None:
    """Dense-dist checks live next to the type, not in ``validation.py``."""
    sparse = SparseDiscreteDist(
        x_array=np.array([0.0, 1.0]),
        prob_arr=np.array([0.5, 0.5]),
    )
    with pytest.raises(TypeError, match="dist must be DenseDiscreteDist with LINEAR spacing"):
        require_dense_dist(dist=sparse, name="dist", spacing=SpacingType.LINEAR)

    geometric = DenseDiscreteDist(
        grid=GridSpec(
            step=math.log(2.0),
            n=2,
            spacing_type=SpacingType.GEOMETRIC,
            anchor=1.0,
        ),
        prob_arr=np.array([0.4, 0.6]),
        domain=Domain.POSITIVES,
    )
    with pytest.raises(TypeError, match="expected DenseDiscreteDist with LINEAR spacing"):
        require_dense_dist(dist=geometric, name="dist", spacing=SpacingType.LINEAR)
    with pytest.raises(ValueError, match=r"must use Domain\.REALS"):
        require_dense_dist(
            dist=geometric, name="dist", spacing=SpacingType.GEOMETRIC, domain=Domain.REALS
        )
    linear = DenseDiscreteDist(
        grid=GridSpec(step=1.0, n=2, anchor=0.0),
        prob_arr=np.array([0.4, 0.6]),
    )
    assert require_linear_reals_dist(dist=linear, name="dist") is linear
    with pytest.raises(TypeError, match="expected DenseDiscreteDist with LINEAR spacing"):
        require_linear_reals_dist(dist=geometric, name="dist")
    assert require_geometric_positives_dist(dist=geometric, name="dist") is geometric
    with pytest.raises(TypeError, match="expected DenseDiscreteDist with GEOMETRIC spacing"):
        require_geometric_positives_dist(dist=linear, name="dist")


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


def test_privacy_params_require_delta_and_epsilon() -> None:
    """Query helpers demand the optional field that the query uses."""
    params = PrivacyParams(sigma=1.0, num_steps=4)
    with pytest.raises(ValueError, match="delta must be in"):
        params.require_delta()
    with pytest.raises(ValueError, match="epsilon must be positive"):
        params.require_epsilon()
    assert PrivacyParams(sigma=1.0, num_steps=4, delta=1e-5).require_delta() == 1e-5
    assert PrivacyParams(sigma=1.0, num_steps=4, epsilon=1.0).require_epsilon() == 1.0


def test_allocation_scheme_config_validates_on_construction() -> None:
    """Config tail budget is strictly positive; ``max_grid_mult <= 0`` stays uncapped."""
    AllocationSchemeConfig(max_grid_mult=-1)
    with pytest.raises(ValueError, match="tail_truncation must be positive"):
        AllocationSchemeConfig(tail_truncation=0.0)
    with pytest.raises(ValueError, match="loss_discretization must be positive"):
        AllocationSchemeConfig(loss_discretization=0.0)
    with pytest.raises(ValueError, match="max_grid_fft must be >= 1"):
        AllocationSchemeConfig(max_grid_fft=0)
    with pytest.raises(TypeError, match="convolution_method must be ConvolutionMethod"):
        AllocationSchemeConfig(convolution_method="fft")  # type: ignore[arg-type]


def test_algorithm_tail_truncation_zero_remains_valid() -> None:
    """Convolution/discretization ``tail_truncation=0`` is an algorithm arg, not a config field."""
    noise = DenseDiscreteDist(
        grid=GridSpec(step=1.0, n=3, anchor=0.0),
        prob_arr=np.array([0.5, 0.0, 0.5]),
    )
    remove, add = discrete_distribution(
        noise_dist=noise,
        loss_discretization=0.1,
        tail_truncation=0.0,
    )
    assert remove.p_min == 0.0
    assert add.p_min == 0.0
