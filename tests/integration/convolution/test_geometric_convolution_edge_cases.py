"""
Integration tests for geometric convolution edge cases.
"""
import numpy as np

from PLD_accounting.discrete_dist import GeometricDiscreteDist
from PLD_accounting.geometric_convolution import geometric_convolve
from PLD_accounting.types import BoundType
from tests.test_tolerances import TestTolerances as TOL


class TestGeometricConvolutionEdgeCases:
    """Edge cases for geometric convolution behavior."""

    def test_grid_alignment_starts_at_sum(self):
        x1 = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        x2 = np.array([3.0, 6.0, 12.0], dtype=np.float64)
        p1 = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        p2 = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        d1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=p1)
        d2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=p2)

        result = geometric_convolve(d1, d2, tail_truncation=0.0, bound_type=BoundType.DOMINATES)
        assert np.isclose(result.x_array[0], x1[0] + x2[0])

    def test_symmetry(self):
        x1 = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        x2 = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float64)
        p1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        p2 = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        d1 = GeometricDiscreteDist.from_x_array(x_array=x1, PMF_array=p1)
        d2 = GeometricDiscreteDist.from_x_array(x_array=x2, PMF_array=p2)

        r12 = geometric_convolve(d1, d2, tail_truncation=0.0, bound_type=BoundType.DOMINATES)
        r21 = geometric_convolve(d2, d1, tail_truncation=0.0, bound_type=BoundType.DOMINATES)

        assert np.allclose(r12.x_array, r21.x_array)
        assert np.allclose(r12.PMF_array, r21.PMF_array)

    def test_tail_mass_conservation(self):
        x = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        p1 = np.array([0.0, 0.1, 0.0, 0.9], dtype=np.float64)
        p2 = np.array([0.0, 0.2, 0.0, 0.8], dtype=np.float64)
        d1 = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=p1)
        d2 = GeometricDiscreteDist.from_x_array(x_array=x, PMF_array=p2)

        result = geometric_convolve(d1, d2, tail_truncation=0.0, bound_type=BoundType.DOMINATES)
        assert result.p_neg_inf == 0.0
        assert result.p_pos_inf <= TOL.MASS_CONSERVATION
