import numpy as np
import pytest

from PLD_accounting.discrete_dist import (
    GeneralDiscreteDist,
    LinearDiscreteDist,
)
from PLD_accounting.dp_accounting_support import (
    discrete_dist_to_dp_accounting_pmf,
    dp_accounting_pmf_to_discrete_dist,
)
from dp_accounting.pld.pld_pmf import DensePLDPmf, SparsePLDPmf


def _make_linear_dist() -> LinearDiscreteDist:
    return LinearDiscreteDist.from_x_array(
        x_array=np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float64),
        PMF_array=np.array([0.2, 0.3, 0.25, 0.15], dtype=np.float64),
        p_neg_inf=0.0,
        p_pos_inf=0.1,
    )


def test_dp_accounting_roundtrip_preserves_mass_and_grid_shape():
    original = _make_linear_dist()
    pmf = discrete_dist_to_dp_accounting_pmf(original, pessimistic_estimate=True)
    restored = dp_accounting_pmf_to_discrete_dist(pmf)

    assert restored.x_array.shape == original.x_array.shape
    total_mass = np.sum(restored.PMF_array) + restored.p_neg_inf + restored.p_pos_inf
    assert np.isclose(total_mass, 1.0, atol=1e-12)
    assert np.isclose(restored.p_pos_inf, original.p_pos_inf, atol=1e-12)


def test_discrete_dist_to_dp_accounting_handles_zero_finite_mass():
    """Test that distributions with all mass at infinity are handled correctly."""
    dist = LinearDiscreteDist.from_x_array(
        x_array=np.array([0.0, 1.0], dtype=np.float64),
        PMF_array=np.array([0.0, 0.0], dtype=np.float64),
        p_neg_inf=0.0,
        p_pos_inf=1.0,
    )
    # dp_accounting allows all mass at infinity
    pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
    assert pmf._infinity_mass == 1.0
    assert np.allclose(pmf._probs, np.array([0.0, 0.0]))
class TestLinearDistAdapter:
    """Test conversion from LinearDiscreteDist to DensePLDPmf."""

    def test_dense_linear_to_dense_pmf(self):
        """Test that LinearDiscreteDist converts to DensePLDPmf."""
        dist = LinearDiscreteDist(
            x_min=0.0,
            x_gap=0.5,
            PMF_array=np.array([0.2, 0.3, 0.4, 0.1]),
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert np.allclose(pmf._probs, dist.PMF_array)

    def test_dense_linear_with_nonzero_base(self):
        """Test LinearDiscreteDist with non-zero x_min."""
        # x_min = 1.0, x_gap = 0.5 -> base_index = 2
        dist = LinearDiscreteDist(
            x_min=1.0,
            x_gap=0.5,
            PMF_array=np.array([0.3, 0.4, 0.3]),
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert pmf._lower_loss == 2  # 1.0 / 0.5 = 2

    def test_dense_linear_with_infinity_mass(self):
        """Test LinearDiscreteDist with p_pos_inf."""
        dist = LinearDiscreteDist(
            x_min=0.0,
            x_gap=0.25,
            PMF_array=np.array([0.2, 0.5, 0.2]),
            p_pos_inf=0.1
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._infinity_mass == 0.1

    def test_dense_linear_roundtrip(self):
        """Test DenseLinear -> PMF -> DiscreteDist roundtrip."""
        dist = LinearDiscreteDist(
            x_min=0.5,
            x_gap=0.25,
            PMF_array=np.array([0.2, 0.3, 0.4, 0.1]),
            p_pos_inf=0.0
        )

        pmf = discrete_dist_to_dp_accounting_pmf(dist, pessimistic_estimate=True)
        restored = dp_accounting_pmf_to_discrete_dist(pmf)

        # Check values match
        assert np.allclose(dist.x_array, restored.x_array)
        assert np.allclose(dist.PMF_array, restored.PMF_array)
        assert np.isclose(dist.p_pos_inf, restored.p_pos_inf)

