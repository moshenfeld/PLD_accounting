"""
Unit tests for discrete_dist module.

Tests all distribution types: General, Linear (Dense/Sparse), Geometric (Dense/Sparse),
and transform functions between linear and geometric grids.
"""

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_utils import PMF_MASS_TOL
from PLD_accounting.types import BoundType, ConvolutionMethod, SpacingType
from PLD_accounting.utils import (
    exp_linear_to_geometric,
    log_geometric_to_linear,
)

mypy_api: Optional[Any]
try:
    from mypy import api as _mypy_api

    mypy_api = _mypy_api
except ImportError:
    mypy_api = None


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

    def test_with_boundary_mass(self):
        """Test distribution with mass at boundaries (POSITIVES domain allows both)."""
        from PLD_accounting.discrete_dist import Domain

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
        with pytest.raises(ValueError, match="MASS CONSERVATION ERROR"):
            SparseDiscreteDist(x_array=x, prob_arr=pmf)

    def test_mass_within_tolerance_accepted(self):
        """Test that mass within tolerance is accepted at construction."""
        x = np.array([1.0, 2.0])
        pmf = np.array([0.5, 0.5 + PMF_MASS_TOL / 2], dtype=np.float64)
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
        with pytest.raises(ValueError, match="MASS CONSERVATION ERROR") as exc_info:
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
        dist = DenseDiscreteDist(x_min=0.0, step=0.5, prob_arr=np.array([0.2, 0.5, 0.3]))
        expected_x = np.array([0.0, 0.5, 1.0])
        assert np.allclose(dist.x_array, expected_x)

    def test_skip_must_be_positive(self):
        """Test that negative step raises error."""
        with pytest.raises(ValueError, match="step must be positive"):
            DenseDiscreteDist(x_min=0.0, step=-0.1, prob_arr=np.array([0.5, 0.5]))

    def test_zero_skip_raises(self):
        """Test that zero step raises error."""
        with pytest.raises(ValueError, match="step must be positive"):
            DenseDiscreteDist(x_min=0.0, step=0.0, prob_arr=np.array([0.5, 0.5]))


class TestDenseDiscreteDistGeometric:
    """Test DenseDiscreteDist validation and properties for geometric spacing."""

    def test_valid_dense_geometric(self):
        """Test creating valid dense geometric distribution."""
        from PLD_accounting.discrete_dist import Domain

        dist = DenseDiscreteDist(
            x_min=1.0,
            step=2.0,
            prob_arr=np.array([0.2, 0.5, 0.3]),
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )
        expected_x = np.array([1.0, 2.0, 4.0])  # x_min * ratio^i
        assert np.allclose(dist.x_array, expected_x)

    def test_x_min_must_be_positive(self):
        """Test that non-positive x_min raises error for geometric grid."""
        from PLD_accounting.discrete_dist import Domain

        with pytest.raises(ValueError, match="x_min must be positive"):
            DenseDiscreteDist(
                x_min=0.0,
                step=2.0,
                prob_arr=np.array([0.5, 0.5]),
                spacing_type=SpacingType.GEOMETRIC,
                domain=Domain.POSITIVES,
            )

    def test_skip_must_exceed_one(self):
        """Test that step <= 1 raises error for geometric grid."""
        from PLD_accounting.discrete_dist import Domain

        with pytest.raises(ValueError, match="step must be > 1"):
            DenseDiscreteDist(
                x_min=1.0,
                step=1.0,
                prob_arr=np.array([0.5, 0.5]),
                spacing_type=SpacingType.GEOMETRIC,
                domain=Domain.POSITIVES,
            )


class TestLinearGeometricTransforms:
    """Test exp_linear_to_geometric and log_geometric_to_linear transform functions."""

    def test_dense_linear_to_geometric_roundtrip(self):
        """Test dense linear -> geometric -> linear preserves structure."""
        dist_linear = DenseDiscreteDist(x_min=1.0, step=0.5, prob_arr=np.array([0.2, 0.5, 0.3]))

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

        # Check roundtrip preserves values
        assert np.isclose(dist_linear.x_min, dist_linear_back.x_min)
        assert np.isclose(dist_linear.step, dist_linear_back.step)
        assert np.allclose(dist_linear.prob_arr, dist_linear_back.prob_arr)
        assert dist_linear.p_min == dist_linear_back.p_min
        assert dist_linear.p_max == dist_linear_back.p_max

    def test_dense_geometric_to_linear_roundtrip(self):
        """Test dense geometric -> linear -> geometric preserves structure."""
        from PLD_accounting.discrete_dist import Domain

        dist_geom = DenseDiscreteDist(
            x_min=2.0,
            step=1.5,
            prob_arr=np.array([0.2, 0.5, 0.3]),
            spacing_type=SpacingType.GEOMETRIC,
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
        assert np.isclose(dist_geom.x_min, dist_geom_back.x_min)
        assert np.isclose(dist_geom.step, dist_geom_back.step)
        assert np.allclose(dist_geom.prob_arr, dist_geom_back.prob_arr)

    def test_transform_preserves_boundary_masses(self):
        """Test that exp/log transforms preserve p_min and p_max.

        For REALS domain only one boundary is non-zero at a time.
        For POSITIVES domain both can coexist.
        """
        # REALS with p_min (mass at -inf) only
        dist_neg = DenseDiscreteDist(
            x_min=1.0,
            step=0.5,
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
            x_min=1.0,
            step=0.5,
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
        from PLD_accounting.discrete_dist import Domain

        geom_both = DenseDiscreteDist(
            x_min=np.exp(1.0),
            step=np.exp(0.5),
            prob_arr=np.array([0.3, 0.5]),
            p_min=0.1,
            p_max=0.1,
            spacing_type=SpacingType.GEOMETRIC,
            domain=Domain.POSITIVES,
        )
        assert geom_both.p_min == 0.1
        assert geom_both.p_max == 0.1


@pytest.mark.unit
def test_project_type_hints_with_mypy_static_analysis():
    if mypy_api is None:
        raise AssertionError(
            "mypy is required for full-project static type checks. Install test dependencies."
        )

    repo_root = Path(__file__).resolve().parents[3]
    targets = [str(repo_root / "PLD_accounting"), str(repo_root / "tests")]
    stdout, stderr, exit_status = mypy_api.run(
        [
            "--config-file",
            str(repo_root / "pyproject.toml"),
            "--cache-dir",
            str(repo_root / ".mypy_cache"),
            *targets,
        ]
    )
    if exit_status != 0:
        output = "\n".join(part for part in (stdout, stderr) if part.strip()).strip()
        pytest.fail(f"Full-project mypy type check failed:\n{output}")
