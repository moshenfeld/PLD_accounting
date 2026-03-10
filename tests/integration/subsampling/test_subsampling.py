"""
Subsampling implementation tests.

Tests custom subsampling against analytical ground truth and dp_accounting.

IMPORTANT: Discretization should scale with q for accurate results.
Using a fixed coarser discretization will introduce errors, especially for small q.
"""
import pytest
import numpy as np
from scipy import stats
from dp_accounting.pld import privacy_loss_distribution

from PLD_accounting.subsample_PLD import (
    subsample_PMF,
    _calc_PLD_dual,
)
from PLD_accounting.discrete_dist import GeneralDiscreteDist
from PLD_accounting.types import Direction, BoundType
from tests.integration.subsampling.analytic_gaussian import gaussian_pld


def create_pld_and_extract_pmf(
    standard_deviation: float,
    sensitivity: float,
    sampling_prob: float,
    value_discretization_interval: float,
    remove_direction: bool = True
):
    """Create a PLD via dp-accounting and return the internal PMF for one direction.

    When `sampling_prob < 1`, `dp-accounting` constructs the amplified PLD directly. This
    helper returns either the remove-direction PMF (`_pmf_remove`) or the add-direction
    PMF (`_pmf_add`) from that PLD depending on `remove_direction`.
    """
    if sampling_prob < 1.0:
        pld = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=standard_deviation,
            sensitivity=sensitivity,
            sampling_prob=sampling_prob,
            value_discretization_interval=value_discretization_interval,
            pessimistic_estimate=True
)
    else:
        pld = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=standard_deviation,
            sensitivity=sensitivity,
            value_discretization_interval=value_discretization_interval,
            pessimistic_estimate=True
)
    return pld._pmf_remove if remove_direction else pld._pmf_add


def compute_analytical_subsampled_gaussian(
    sigma: float,
    sampling_prob: float,
    discretization: float,
    remove_direction: bool = True
):
    return gaussian_pld(sigma, sampling_prob, discretization, remove_direction)


def compute_analytical_base_gaussian(
    sigma: float,
    discretization: float,
    remove_direction: bool = True
):
    """
    Compute analytical base (unsubsampled) Gaussian PLD.
    """
    return compute_analytical_subsampled_gaussian(sigma, 1.0, discretization, remove_direction)


def _upper_to_lower(dist: GeneralDiscreteDist) -> GeneralDiscreteDist:
    if dist.p_neg_inf > 0:
        raise ValueError("Expected p_neg_inf=0 for upper PLD")
    losses = dist.x_array
    probs = dist.PMF_array
    lower_probs = np.zeros_like(probs)
    mask = probs > 0
    lower_probs[mask] = np.exp(np.log(probs[mask]) - losses[mask])
    sum_prob = float(np.sum(lower_probs))
    return GeneralDiscreteDist(
        x_array=losses,
        PMF_array=lower_probs,
        p_neg_inf=max(0.0, 1.0 - sum_prob),
        p_pos_inf=0.0,
    )


def _negate_distribution(dist: GeneralDiscreteDist) -> GeneralDiscreteDist:
    return GeneralDiscreteDist(
        x_array=-np.flip(dist.x_array),
        PMF_array=np.flip(dist.PMF_array),
        p_neg_inf=dist.p_pos_inf,
        p_pos_inf=dist.p_neg_inf,
    )


class TestPLDDualTransformation:
    """Tests for the PLD dual transformation Q(l) = P(l) * e^{-l}."""

    def test_upper_to_lower_basic(self):
        losses = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        probs = np.array([0.1, 0.2, 0.3, 0.25, 0.1])

        upper = GeneralDiscreteDist(
            x_array=losses,
            PMF_array=probs,
            p_neg_inf=0.0,
            p_pos_inf=0.05
        )

        lower = _calc_PLD_dual(upper)

        expected_lower = probs * np.exp(-losses)
        assert np.allclose(lower.PMF_array, expected_lower, rtol=1e-10)

        total_mass = np.sum(lower.PMF_array) + lower.p_neg_inf + lower.p_pos_inf
        assert abs(total_mass - 1.0) < 1e-9
        assert lower.p_pos_inf == 0.0

    def test_upper_to_lower_rejects_invalid_upper(self):
        invalid_upper = GeneralDiscreteDist(
            x_array=np.array([0.0, 1.0]),
            PMF_array=np.array([0.5, 0.4]),
            p_neg_inf=0.1,
            p_pos_inf=0.0
        )

        with pytest.raises(ValueError, match="not a valid upper bound"):
            _calc_PLD_dual(invalid_upper)

    def test_upper_to_lower_mass_conservation(self):
        test_cases = [
            (np.array([0.0, 1.0, 2.0]), np.array([0.3, 0.5, 0.15]), 0.05),
            (np.linspace(0, 3, 20), np.ones(20) / 21.0, 0.01),
            (np.array([0.0, 1.0]), np.array([0.4, 0.4]), 0.2),
        ]

        for losses, probs, p_pos_inf in test_cases:
            upper = GeneralDiscreteDist(
                x_array=losses,
                PMF_array=probs,
                p_neg_inf=0.0,
                p_pos_inf=p_pos_inf
            )

            lower = _calc_PLD_dual(upper)
            total_mass = np.sum(lower.PMF_array) + lower.p_neg_inf + lower.p_pos_inf
            assert abs(total_mass - 1.0) < 1e-9


class TestSubsampleDistDual:
    """Tests for PLD-dual based distribution subsampling."""

    def test_subsample_PMF_mass_conservation_remove(self):
        losses = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        probs = np.array([0.3, 0.25, 0.2, 0.15, 0.05])

        dual_sum = np.sum(probs * np.exp(-losses))
        assert dual_sum <= 1.0

        base_dist = GeneralDiscreteDist(
            x_array=losses,
            PMF_array=probs,
            p_neg_inf=0.0,
            p_pos_inf=0.05
        )

        subsampled = subsample_PMF(
            base_pld=base_dist,
            sampling_prob=0.5,
            direction=Direction.REMOVE,
            bound_type=BoundType.DOMINATES
        )

        total_mass = (
            np.sum(subsampled.PMF_array)
            + subsampled.p_neg_inf
            + subsampled.p_pos_inf
        )

        assert abs(total_mass - 1.0) < 1e-6

    def test_subsample_PMF_mass_conservation_add(self):
        losses = np.linspace(0.0, 3.0, 50)
        add_upper = GeneralDiscreteDist(
            x_array=-losses[::-1],
            PMF_array=np.ones(50)[::-1] / 52.0,
            p_neg_inf=0.02,
            p_pos_inf=0.0
        )

        subsampled = subsample_PMF(
            base_pld=add_upper,
            sampling_prob=0.4,
            direction=Direction.ADD,
            bound_type=BoundType.IS_DOMINATED
        )

        total_mass = (
            np.sum(subsampled.PMF_array)
            + subsampled.p_neg_inf
            + subsampled.p_pos_inf
        )

        assert abs(total_mass - 1.0) < 1e-6

    @pytest.mark.parametrize("sampling_prob", [0.01, 0.1, 0.5, 0.8])
    @pytest.mark.parametrize("direction", [Direction.REMOVE, Direction.ADD])
    def test_subsample_PMF_various_params(self, sampling_prob, direction):
        if direction == Direction.REMOVE:
            losses = np.linspace(0.0, 3.0, 30)
            dist = GeneralDiscreteDist(
                x_array=losses,
                PMF_array=np.ones(30) / 32.0,
                p_neg_inf=0.0,
                p_pos_inf=0.02
            )
            bound_type = BoundType.DOMINATES
        else:
            losses = np.linspace(0.0, 3.0, 30)
            dist = GeneralDiscreteDist(
                x_array=-losses[::-1],
                PMF_array=np.ones(30)[::-1] / 32.0,
                p_neg_inf=0.02,
                p_pos_inf=0.0
            )
            bound_type = BoundType.IS_DOMINATED

        subsampled = subsample_PMF(
            base_pld=dist,
            sampling_prob=sampling_prob,
            direction=direction,
            bound_type=bound_type,
        )

        total_mass = np.sum(subsampled.PMF_array) + subsampled.p_neg_inf + subsampled.p_pos_inf
        assert abs(total_mass - 1.0) < 1e-6

        assert np.sum(subsampled.PMF_array > 1e-10) > 0

