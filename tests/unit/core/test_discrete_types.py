"""
Unit tests for discrete_dist module.

Tests all distribution types: General, Linear (Dense/Sparse), Geometric (Dense/Sparse),
and transform functions between linear and geometric grids.
"""

import copy
import math

import numpy as np
import pytest

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_utils import PMF_TOLERATED_MASS_TOL
from PLD_accounting.types import BoundType, ConvolutionMethod, SpacingType
from PLD_accounting.utils import (
    exp_linear_to_geometric,
    log_geometric_to_linear,
)


class TestGeneralDiscreteDist:
    """Test SparseDiscreteDist dataclass validation."""

    def test_valid_distribution(self):
        """Test that valid distribution is accepted."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        dist = SparseDiscreteDist(x_array=x, prob_arr=pmf)
        assert np.allclose(dist.x_array, x)
        assert np.allclose(dist.prob_arr, pmf)
        assert dist.p_min == 0.0
        assert dist.p_max == 0.0

    def test_constructor_detaches_and_exposes_read_only_arrays(self):
        """Validated support and probability arrays cannot be mutated by callers."""
        x = np.array([1.0, 2.0])
        pmf = np.array([0.4, 0.6])
        dist = SparseDiscreteDist(x_array=x, prob_arr=pmf)
        x[0] = -10.0
        pmf[0] = 0.0

        np.testing.assert_array_equal(dist.x_array, np.array([1.0, 2.0]))
        np.testing.assert_array_equal(dist.prob_arr, np.array([0.4, 0.6]))
        with pytest.raises(ValueError, match="read-only"):
            dist.x_array[0] = -10.0
        with pytest.raises(ValueError, match="read-only"):
            dist.prob_arr[0] = 0.0

    def test_deepcopy_preserves_read_only_arrays(self):
        """Deep copies remain immutable rather than silently weakening invariants."""
        dist = SparseDiscreteDist(
            x_array=np.array([1.0, 2.0]),
            prob_arr=np.array([0.4, 0.6]),
        )
        copied = copy.deepcopy(dist)

        with pytest.raises(ValueError, match="read-only"):
            copied.prob_arr[0] = 0.0

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_support_and_probability_values(self, bad: float):
        """NaN and infinity cannot bypass distribution invariants."""
        with pytest.raises(ValueError, match="finite"):
            SparseDiscreteDist(
                x_array=np.array([1.0, 2.0]),
                prob_arr=np.array([bad, 1.0]),
            )
        with pytest.raises(ValueError, match="finite"):
            SparseDiscreteDist(
                x_array=np.array([bad, 2.0]),
                prob_arr=np.array([0.5, 0.5]),
            )

    @pytest.mark.parametrize("boundary", ["p_min", "p_max"])
    @pytest.mark.parametrize("bad", [True, False, "0.0"])
    def test_rejects_non_real_boundary_masses(self, boundary: str, bad: object):
        """Boundary masses must be real scalars and must reject boolean coercion."""
        kwargs = {boundary: bad}
        with pytest.raises(TypeError, match=rf"{boundary} must be a real number"):
            SparseDiscreteDist(
                x_array=np.array([1.0]),
                prob_arr=np.array([1.0]),
                **kwargs,
            )

    @pytest.mark.parametrize("boundary", ["p_min", "p_max"])
    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_boundary_masses(self, boundary: str, bad: float):
        """Boundary masses use the shared finite-real validation contract."""
        kwargs = {boundary: bad}
        with pytest.raises(ValueError, match=rf"{boundary} must be finite"):
            SparseDiscreteDist(
                x_array=np.array([1.0]),
                prob_arr=np.array([1.0]),
                **kwargs,
            )

    def test_with_boundary_mass(self):
        """Test distribution with mass at boundaries (POSITIVES domain allows both)."""
        x = np.array([1.0, 2.0])
        pmf = np.array([0.3, 0.5], dtype=np.float64)
        dist = SparseDiscreteDist(
            x_array=x, prob_arr=pmf, p_min=0.1, p_max=0.1, domain=Domain.POSITIVES
        )
        assert dist.p_min == 0.1
        assert dist.p_max == 0.1

    def test_non_increasing_x_raises(self):
        """Test that non-increasing x array raises error."""
        x = np.array([1.0, 3.0, 2.0])
        pmf = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        with pytest.raises(ValueError, match="strictly increasing"):
            SparseDiscreteDist(x_array=x, prob_arr=pmf)

    def test_negative_pmf_raises(self):
        """Test that negative PMF values raise error."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.3, -0.1, 0.8], dtype=np.float64)
        with pytest.raises(ValueError, match="nonnegative"):
            SparseDiscreteDist(x_array=x, prob_arr=pmf)

    def test_mismatched_shapes_raises(self):
        """Test that mismatched array shapes raise error."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.5, 0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="equal length"):
            SparseDiscreteDist(x_array=x, prob_arr=pmf)

    def test_mass_not_conserved_raises(self):
        """Test that non-unit total mass raises error at construction."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.2, 0.3, 0.4], dtype=np.float64)  # sums to 0.9
        with pytest.raises(ValueError, match="PMF mass does not total 1"):
            SparseDiscreteDist(x_array=x, prob_arr=pmf)

    def test_mass_within_tolerance_accepted(self):
        """Test that mass within tolerance is accepted at construction."""
        x = np.array([1.0, 2.0])
        pmf = np.array([0.5, 0.5 + PMF_TOLERATED_MASS_TOL / 2], dtype=np.float64)
        dist = SparseDiscreteDist(x_array=x, prob_arr=pmf)
        assert dist is not None


class TestMassConservationValidation:
    """Test that mass conservation is enforced at construction."""

    def test_exact_conservation(self):
        """Test exact mass conservation passes."""
        dist = SparseDiscreteDist(
            x_array=np.array([1.0, 2.0, 3.0]),
            prob_arr=np.array([0.25, 0.5, 0.25], dtype=np.float64),
        )
        assert dist is not None

    def test_with_infinite_mass_both_nonzero_raises(self):
        """Test that both boundaries non-zero raises for REALS domain at construction."""
        with pytest.raises(ValueError, match="REALS domain"):
            SparseDiscreteDist(
                x_array=np.array([1.0, 2.0]),
                prob_arr=np.array([0.2, 0.3], dtype=np.float64),
                p_min=0.1,
                p_max=0.4,
            )

    def test_violation_raises(self):
        """Test that mass violation raises detailed error at construction."""
        with pytest.raises(ValueError, match="PMF mass does not total 1") as exc_info:
            SparseDiscreteDist(
                x_array=np.array([1.0, 2.0]), prob_arr=np.array([0.3, 0.3], dtype=np.float64)
            )
        assert "tolerance=" in str(exc_info.value)
        assert "PMF sum=" in str(exc_info.value)

    def test_high_precision_summation(self):
        """Test that high-precision summation is used at construction."""
        n = 1000
        pmf = np.array([1.0 / n] * n, dtype=np.float64)
        dist = SparseDiscreteDist(x_array=np.arange(n, dtype=np.float64), prob_arr=pmf)
        assert dist is not None


class TestEnums:
    """Test enum definitions."""

    def test_bound_type_values(self):
        """Test BoundType enum values."""
        assert BoundType.DOMINATES.value == "DOMINATES"
        assert BoundType.IS_DOMINATED.value == "IS_DOMINATED"

    def test_spacing_type_values(self):
        """Test SpacingType enum values."""
        assert SpacingType.LINEAR.value == "linear"
        assert SpacingType.GEOMETRIC.value == "geometric"

    def test_convolution_method_values(self):
        """Test ConvolutionMethod enum values."""
        assert ConvolutionMethod.GEOM.value == "geometric"
        assert ConvolutionMethod.FFT.value == "fft"


class TestDenseDiscreteDistLinear:
    """Test DenseDiscreteDist validation and properties for linear spacing."""

    def test_valid_dense_linear(self):
        """Test creating valid dense linear distribution."""
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=0.5,
                n=3,
                anchor=0.0,
            ),
            prob_arr=np.array([0.2, 0.5, 0.3]),
        )
        expected_x = np.array([0.0, 0.5, 1.0])
        assert np.allclose(dist.x_array, expected_x)

    def test_skip_must_be_positive(self):
        """Test that negative step raises error."""
        with pytest.raises(ValueError, match="step must be positive"):
            DenseDiscreteDist(
                grid=GridSpec(
                    step=-0.1,
                    n=2,
                    anchor=0.0,
                ),
                prob_arr=np.array([0.5, 0.5]),
            )

    def test_zero_skip_raises(self):
        """Test that zero step raises error."""
        with pytest.raises(ValueError, match="step must be positive"):
            DenseDiscreteDist(
                grid=GridSpec(
                    step=0.0,
                    n=2,
                    anchor=0.0,
                ),
                prob_arr=np.array([0.5, 0.5]),
            )


class TestDenseDiscreteDistGeometric:
    """Test DenseDiscreteDist validation and properties for geometric spacing."""

    def test_valid_dense_geometric(self):
        """Test creating valid dense geometric distribution."""
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=math.log(2.0),
                spacing_type=SpacingType.GEOMETRIC,
                n=3,
                anchor=1.0,
            ),
            prob_arr=np.array([0.2, 0.5, 0.3]),
            domain=Domain.POSITIVES,
        )
        expected_x = np.array([1.0, 2.0, 4.0])  # x_min * ratio^i
        assert np.allclose(dist.x_array, expected_x)

    def test_x_0_must_be_positive(self):
        """Non-positive geometric anchors are rejected."""
        with pytest.raises(ValueError, match="must be positive"):
            DenseDiscreteDist(
                grid=GridSpec(
                    step=math.log(2.0),
                    spacing_type=SpacingType.GEOMETRIC,
                    n=2,
                    anchor=0.0,
                ),
                prob_arr=np.array([0.5, 0.5]),
                domain=Domain.POSITIVES,
            )

    def test_ratio_must_exceed_one(self):
        """A geometric ratio of one is a zero log-step, which is not a lattice."""
        with pytest.raises(ValueError, match="step must be positive"):
            DenseDiscreteDist(
                grid=GridSpec(
                    step=math.log(1.0),
                    spacing_type=SpacingType.GEOMETRIC,
                    n=2,
                    anchor=1.0,
                ),
                prob_arr=np.array([0.5, 0.5]),
                domain=Domain.POSITIVES,
            )


class TestLinearGeometricTransforms:
    """Test exp_linear_to_geometric and log_geometric_to_linear transform functions."""

    def test_dense_linear_to_geometric_roundtrip(self):
        """Test dense linear -> geometric -> linear preserves structure.

        Uses a zero-anchored grid: the transform is defined only there, because that is
        the only anchor for which it agrees bitwise with ``np.exp`` of the support.
        """
        dist_linear = DenseDiscreteDist(
            grid=GridSpec(step=0.5, n=3, index_0=2),
            prob_arr=np.array([0.2, 0.5, 0.3]),
        )

        # Transform to geometric (exp)
        dist_geom = exp_linear_to_geometric(dist_linear)
        assert (
            isinstance(dist_geom, DenseDiscreteDist)
            and dist_geom.spacing_type == SpacingType.GEOMETRIC
        )

        # Transform back to linear (log)
        dist_linear_back = log_geometric_to_linear(dist_geom)
        assert (
            isinstance(dist_linear_back, DenseDiscreteDist)
            and dist_linear_back.spacing_type == SpacingType.LINEAR
        )

        # Structural round trip: exact, not approximate.
        assert dist_linear_back.grid == dist_linear.grid
        assert np.allclose(dist_linear.prob_arr, dist_linear_back.prob_arr)
        assert dist_linear.p_min == dist_linear_back.p_min
        assert dist_linear.p_max == dist_linear_back.p_max

    def test_dense_geometric_to_linear_roundtrip(self):
        """Test dense geometric -> linear -> geometric preserves structure."""
        dist_geom = DenseDiscreteDist(
            grid=GridSpec(
                step=math.log(1.5),
                n=3,
                spacing_type=SpacingType.GEOMETRIC,
                anchor=1.0,
                index_0=2,
            ),
            prob_arr=np.array([0.2, 0.5, 0.3]),
            domain=Domain.POSITIVES,
        )

        # Transform to linear
        dist_linear = log_geometric_to_linear(dist_geom)
        assert (
            isinstance(dist_linear, DenseDiscreteDist)
            and dist_linear.spacing_type == SpacingType.LINEAR
        )

        # Transform back to geometric
        dist_geom_back = exp_linear_to_geometric(dist_linear)
        assert (
            isinstance(dist_geom_back, DenseDiscreteDist)
            and dist_geom_back.spacing_type == SpacingType.GEOMETRIC
        )

        # Check roundtrip preserves values
        assert np.isclose(dist_geom.x_0, dist_geom_back.x_0)
        assert np.isclose(dist_geom.step, dist_geom_back.step)
        assert np.allclose(dist_geom.prob_arr, dist_geom_back.prob_arr)

    def test_exp_of_an_affine_grid_still_preserves_the_step(self):
        """Affine transforms keep the spacing exactly; only coordinates may shift a ULP."""
        affine = DenseDiscreteDist(
            grid=GridSpec(
                step=0.5,
                n=3,
                anchor=1.0,
            ),
            prob_arr=np.array([0.2, 0.5, 0.3]),
        )
        geom = exp_linear_to_geometric(affine)
        assert geom.grid.step == affine.grid.step
        assert log_geometric_to_linear(geom).grid.step == affine.grid.step

    def test_transform_preserves_boundary_masses(self):
        """Test that exp/log transforms preserve p_min and p_max.

        For REALS domain only one boundary is non-zero at a time.
        For POSITIVES domain both can coexist.
        """
        # REALS with p_min (mass at -inf) only
        dist_neg = DenseDiscreteDist(
            grid=GridSpec(step=0.5, n=2, index_0=2),
            prob_arr=np.array([0.7, 0.2]),
            p_min=0.1,
        )
        geom_neg = exp_linear_to_geometric(dist_neg)
        assert geom_neg.p_min == 0.1
        assert geom_neg.p_max == 0.0
        back_neg = log_geometric_to_linear(geom_neg)
        assert back_neg.p_min == 0.1
        assert back_neg.p_max == 0.0

        # REALS with p_max (mass at +inf) only
        dist_pos = DenseDiscreteDist(
            grid=GridSpec(step=0.5, n=2, index_0=2),
            prob_arr=np.array([0.6, 0.2]),
            p_max=0.2,
        )
        geom_pos = exp_linear_to_geometric(dist_pos)
        assert geom_pos.p_min == 0.0
        assert geom_pos.p_max == 0.2
        back_pos = log_geometric_to_linear(geom_pos)
        assert back_pos.p_min == 0.0
        assert back_pos.p_max == 0.2

        # POSITIVES with both non-zero is valid
        geom_both = DenseDiscreteDist(
            grid=GridSpec(
                step=math.log(np.exp(0.5)),
                spacing_type=SpacingType.GEOMETRIC,
                n=2,
                anchor=np.exp(1.0),
            ),
            prob_arr=np.array([0.3, 0.5]),
            p_min=0.1,
            p_max=0.1,
            domain=Domain.POSITIVES,
        )
        assert geom_both.p_min == 0.1
        assert geom_both.p_max == 0.1
