import numpy as np
import pytest

from PLD_accounting.discrete_dist import (
    LinearDiscreteDist,
    PLDRealization,
)
from PLD_accounting.dp_accounting_support import (
    dp_accounting_pmf_to_pld_realization,
    linear_dist_to_dp_accounting_pmf,
)
from dp_accounting.pld.pld_pmf import DensePLDPmf


def _make_realization() -> PLDRealization:
    return PLDRealization(
        x_min=-0.5,
        x_gap=0.5,
        PMF_array=np.array([0.2, 0.3, 0.25, 0.15], dtype=np.float64),
        p_loss_inf=0.1,
    )


def test_dp_accounting_roundtrip_preserves_mass_and_grid_shape():
    original = _make_realization()
    pmf = linear_dist_to_dp_accounting_pmf(dist=original, pessimistic_estimate=True)
    restored = dp_accounting_pmf_to_pld_realization(pmf)

    assert restored.x_array.shape == original.x_array.shape
    total_mass = np.sum(restored.PMF_array) + restored.p_neg_inf + restored.p_pos_inf
    assert np.isclose(total_mass, 1.0, atol=1e-12)
    assert np.isclose(restored.p_pos_inf, original.p_pos_inf, atol=1e-12)


def test_linear_dist_to_dp_accounting_handles_zero_finite_mass():
    """Test that distributions with all mass at infinity are handled correctly."""
    realization = LinearDiscreteDist(
        x_min=0.0,
        x_gap=1.0,
        PMF_array=np.array([0.0, 0.0], dtype=np.float64),
        p_pos_inf=1.0,
    )
    pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
    assert pmf._infinity_mass == 1.0
    assert np.allclose(pmf._probs, np.array([0.0, 0.0]))


class TestRealizationAdapter:
    """Test conversion between PLDRealization and DensePLDPmf."""

    def test_dense_linear_to_dense_pmf(self):
        """Test that PLDRealization converts to DensePLDPmf."""
        realization = PLDRealization(
            x_min=0.0,
            x_gap=0.5,
            PMF_array=np.array([0.2, 0.3, 0.4, 0.1]),
        )

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert np.allclose(pmf._probs, realization.PMF_array)

    def test_dense_linear_with_nonzero_base(self):
        """Test PLDRealization with non-zero x_min."""
        realization = PLDRealization(
            x_min=1.0,
            x_gap=0.5,
            PMF_array=np.array([0.3, 0.4, 0.3]),
        )

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._discretization == 0.5
        assert pmf._lower_loss == 2  # 1.0 / 0.5 = 2

    def test_dense_linear_with_infinity_mass(self):
        """Test PLDRealization with p_loss_inf."""
        realization = PLDRealization(
            x_min=0.0,
            x_gap=0.25,
            PMF_array=np.array([0.2, 0.5, 0.2]),
            p_loss_inf=0.1,
        )

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
        assert isinstance(pmf, DensePLDPmf)
        assert pmf._infinity_mass == 0.1

    def test_dense_linear_roundtrip(self):
        """Test PLDRealization -> PMF -> PLDRealization roundtrip."""
        realization = PLDRealization(
            x_min=0.5,
            x_gap=0.25,
            PMF_array=np.array([0.2, 0.3, 0.4, 0.1]),
        )

        pmf = linear_dist_to_dp_accounting_pmf(dist=realization, pessimistic_estimate=True)
        restored = dp_accounting_pmf_to_pld_realization(pmf)

        # Check values match
        assert isinstance(restored, PLDRealization)
        assert np.allclose(realization.x_array, restored.x_array)
        assert np.allclose(realization.PMF_array, restored.PMF_array)
        assert np.isclose(realization.p_pos_inf, restored.p_pos_inf)

    def test_linear_dist_to_dp_accounting_rejects_non_linear_dist(self):
        from PLD_accounting.discrete_dist import GeneralDiscreteDist

        dist = GeneralDiscreteDist(
            x_array=np.array([0.0, 0.5]),
            PMF_array=np.array([0.5, 0.5]),
        )
        with pytest.raises(TypeError, match="requires LinearDiscreteDist"):
            linear_dist_to_dp_accounting_pmf(dist=dist, pessimistic_estimate=True)
