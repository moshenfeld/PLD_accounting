"""
Unit tests for discrete_dist module.

Tests all distribution types: General, Linear (Dense/Sparse), Geometric (Dense/Sparse),
and transform functions between linear and geometric grids.
"""
from pathlib import Path

import pytest
import numpy as np
from PLD_accounting.core_utils import PMF_MASS_TOL
from PLD_accounting.types import BoundType, SpacingType, ConvolutionMethod
from PLD_accounting.discrete_dist import (
    GeneralDiscreteDist,
    LinearDiscreteDist,
    
    GeometricDiscreteDist,
    
)
from PLD_accounting.utils import (
    exp_linear_to_geometric,
    log_geometric_to_linear,
)


class TestGeneralDiscreteDist:
    """Test GeneralDiscreteDist dataclass validation."""

    def test_valid_distribution(self):
        """Test that valid distribution is accepted."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.2, 0.5, 0.3], dtype=np.float64)
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf)
        assert np.allclose(dist.x_array, x)
        assert np.allclose(dist.PMF_array, pmf)
        assert dist.p_neg_inf == 0.0
        assert dist.p_pos_inf == 0.0

    def test_with_infinite_mass(self):
        """Test distribution with mass at infinity."""
        x = np.array([1.0, 2.0])
        pmf = np.array([0.3, 0.5], dtype=np.float64)
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf, p_neg_inf=0.1, p_pos_inf=0.1)
        assert dist.p_neg_inf == 0.1
        assert dist.p_pos_inf == 0.1

    def test_non_increasing_x_raises(self):
        """Test that non-increasing x array raises error."""
        x = np.array([1.0, 3.0, 2.0])
        pmf = np.array([0.3, 0.4, 0.3], dtype=np.float64)
        with pytest.raises(ValueError, match="strictly increasing"):
            GeneralDiscreteDist(x_array=x, PMF_array=pmf)

    def test_negative_pmf_raises(self):
        """Test that negative PMF values raise error."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.3, -0.1, 0.8], dtype=np.float64)
        with pytest.raises(ValueError, match="nonnegative"):
            GeneralDiscreteDist(x_array=x, PMF_array=pmf)

    def test_mismatched_shapes_raises(self):
        """Test that mismatched array shapes raise error."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.5, 0.5], dtype=np.float64)
        with pytest.raises(ValueError, match="equal length"):
            GeneralDiscreteDist(x_array=x, PMF_array=pmf)

    def test_mass_not_conserved_raises(self):
        """Test that non-unit total mass raises error."""
        x = np.array([1.0, 2.0, 3.0])
        pmf = np.array([0.2, 0.3, 0.4], dtype=np.float64)  # sums to 0.9
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf)
        with pytest.raises(ValueError, match="MASS CONSERVATION ERROR"):
            dist.validate_mass_conservation(BoundType.DOMINATES)

    def test_mass_within_tolerance_accepted(self):
        """Test that mass within tolerance is accepted."""
        x = np.array([1.0, 2.0])
        # Create PMF that sums to 1.0 within tolerance
        pmf = np.array([0.5, 0.5 + PMF_MASS_TOL/2], dtype=np.float64)
        # Should not raise
        dist = GeneralDiscreteDist(x_array=x, PMF_array=pmf)
        dist.validate_mass_conservation(BoundType.DOMINATES)
        assert dist is not None


class TestEnsureMassConservation:
    """Test GeneralDiscreteDist.validate_mass_conservation."""

    def test_exact_conservation(self):
        """Test exact mass conservation passes."""
        dist = GeneralDiscreteDist(x_array=np.array([1.0, 2.0, 3.0]),
                            PMF_array=np.array([0.25, 0.5, 0.25], dtype=np.float64))
        dist.validate_mass_conservation(BoundType.DOMINATES)

    def test_with_infinite_mass(self):
        """Test conservation with infinite mass."""
        dist = GeneralDiscreteDist(x_array=np.array([1.0, 2.0]),
                            PMF_array=np.array([0.2, 0.3], dtype=np.float64),
                            p_neg_inf=0.1,
                            p_pos_inf=0.4)
        # This test has both infinities, which violates both bound types
        # We'll use DOMINATES to check the validation
        with pytest.raises(ValueError, match="DOMINATES.*p_neg_inf"):
            dist.validate_mass_conservation(BoundType.DOMINATES)

    def test_violation_raises(self):
        """Test that violation raises detailed error."""
        dist = GeneralDiscreteDist(x_array=np.array([1.0, 2.0]),
                            PMF_array=np.array([0.3, 0.3], dtype=np.float64))
        with pytest.raises(ValueError, match="MASS CONSERVATION ERROR") as exc_info:
            dist.validate_mass_conservation(BoundType.DOMINATES)
        # Check detailed error message
        assert "tolerance=" in str(exc_info.value)
        assert "PMF sum=" in str(exc_info.value)

    def test_high_precision_summation(self):
        """Test that high-precision summation is used."""
        # Create many small values that could accumulate rounding error
        n = 1000
        pmf = np.array([1.0/n] * n, dtype=np.float64)
        dist = GeneralDiscreteDist(x_array=np.arange(n, dtype=np.float64), PMF_array=pmf)
        dist.validate_mass_conservation(BoundType.DOMINATES)  # Should pass with fsum


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


class TestLinearDiscreteDist:
    """Test LinearDiscreteDist validation and properties."""

    def test_valid_dense_linear(self):
        """Test creating valid dense linear distribution."""
        dist = LinearDiscreteDist(
            x_min=0.0,
            x_gap=0.5,
            PMF_array=np.array([0.2, 0.5, 0.3])
        )
        expected_x = np.array([0.0, 0.5, 1.0])
        assert np.allclose(dist.x_array, expected_x)
        assert dist.get_support_size() == 3

    def test_x_gap_must_be_positive(self):
        """Test that negative x_gap raises error."""
        with pytest.raises(ValueError, match="x_gap must be positive"):
            LinearDiscreteDist(
                x_min=0.0,
                x_gap=-0.1,
                PMF_array=np.array([0.5, 0.5])
            )

    def test_zero_x_gap_raises(self):
        """Test that zero x_gap raises error."""
        with pytest.raises(ValueError, match="x_gap must be positive"):
            LinearDiscreteDist(
                x_min=0.0,
                x_gap=0.0,
                PMF_array=np.array([0.5, 0.5])
            )


class TestGeometricDiscreteDist:
    """Test GeometricDiscreteDist validation and properties."""

    def test_valid_dense_geometric(self):
        """Test creating valid dense geometric distribution."""
        dist = GeometricDiscreteDist(
            x_min=1.0,
            ratio=2.0,
            PMF_array=np.array([0.2, 0.5, 0.3])
        )
        expected_x = np.array([1.0, 2.0, 4.0])  # x_min * ratio^i
        assert np.allclose(dist.x_array, expected_x)
        assert dist.get_support_size() == 3

    def test_x_min_must_be_positive(self):
        """Test that non-positive x_min raises error."""
        with pytest.raises(ValueError, match="x_min must be positive"):
            GeometricDiscreteDist(
                x_min=0.0,
                ratio=2.0,
                PMF_array=np.array([0.5, 0.5])
            )

    def test_ratio_must_exceed_one(self):
        """Test that ratio <= 1 raises error."""
        with pytest.raises(ValueError, match="ratio must be > 1"):
            GeometricDiscreteDist(
                x_min=1.0,
                ratio=1.0,
                PMF_array=np.array([0.5, 0.5])
            )


class TestLinearGeometricTransforms:
    """Test exp_linear_to_geometric and log_geometric_to_linear transform functions."""

    def test_dense_linear_to_geometric_roundtrip(self):
        """Test dense linear -> geometric -> linear preserves structure."""
        dist_linear = LinearDiscreteDist(
            x_min=1.0,
            x_gap=0.5,
            PMF_array=np.array([0.2, 0.5, 0.3])
        )

        # Transform to geometric (exp)
        dist_geom = exp_linear_to_geometric(dist_linear)
        assert isinstance(dist_geom, GeometricDiscreteDist)

        # Transform back to linear (log)
        dist_linear_back = log_geometric_to_linear(dist_geom)
        assert isinstance(dist_linear_back, LinearDiscreteDist)

        # Check roundtrip preserves values
        assert np.isclose(dist_linear.x_min, dist_linear_back.x_min)
        assert np.isclose(dist_linear.x_gap, dist_linear_back.x_gap)
        assert np.allclose(dist_linear.PMF_array, dist_linear_back.PMF_array)
        assert dist_linear.p_neg_inf == dist_linear_back.p_neg_inf
        assert dist_linear.p_pos_inf == dist_linear_back.p_pos_inf

    def test_dense_geometric_to_linear_roundtrip(self):
        """Test dense geometric -> linear -> geometric preserves structure."""
        dist_geom = GeometricDiscreteDist(
            x_min=2.0,
            ratio=1.5,
            PMF_array=np.array([0.2, 0.5, 0.3])
        )

        # Transform to linear
        dist_linear = log_geometric_to_linear(dist_geom)
        assert isinstance(dist_linear, LinearDiscreteDist)

        # Transform back to geometric
        dist_geom_back = exp_linear_to_geometric(dist_linear)
        assert isinstance(dist_geom_back, GeometricDiscreteDist)

        # Check roundtrip preserves values
        assert np.isclose(dist_geom.x_min, dist_geom_back.x_min)
        assert np.isclose(dist_geom.ratio, dist_geom_back.ratio)
        assert np.allclose(dist_geom.PMF_array, dist_geom_back.PMF_array)

    def test_transform_preserves_infinite_masses(self):
        """Test that transforms preserve p_neg_inf and p_pos_inf."""
        dist_linear = LinearDiscreteDist(
            x_min=1.0,
            x_gap=0.5,
            PMF_array=np.array([0.2, 0.5]),
            p_neg_inf=0.1,
            p_pos_inf=0.2
        )

        dist_geom = exp_linear_to_geometric(dist_linear)
        assert dist_geom.p_neg_inf == 0.1
        assert dist_geom.p_pos_inf == 0.2

        dist_linear_back = log_geometric_to_linear(dist_geom)
        assert dist_linear_back.p_neg_inf == 0.1
        assert dist_linear_back.p_pos_inf == 0.2
@pytest.mark.unit
def test_project_type_hints_with_mypy_static_analysis():
    try:
        from mypy import api as mypy_api
    except ModuleNotFoundError as exc:
        raise AssertionError(
            "mypy is required for full-project static type checks. Install test dependencies."
        ) from exc

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
