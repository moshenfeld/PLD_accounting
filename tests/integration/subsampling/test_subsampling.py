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
)
from PLD_accounting.discrete_dist import GeneralDiscreteDist, LinearDiscreteDist, PLDRealization
from PLD_accounting.utils import calc_pld_dual, negate_reverse_linear_distribution
from PLD_accounting.types import Direction
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


def _as_realization(dist: LinearDiscreteDist) -> PLDRealization:
    return PLDRealization(
        x_min=dist.x_min,
        x_gap=dist.x_gap,
        PMF_array=dist.PMF_array,
        p_loss_inf=dist.p_pos_inf,
        p_loss_neg_inf=dist.p_neg_inf,
    )


def _realization_from_loss_values(
    loss_values: np.ndarray,
    probabilities: np.ndarray,
    p_loss_inf: float = 0.0,
    p_loss_neg_inf: float = 0.0,
) -> PLDRealization:
    return _as_realization(
        LinearDiscreteDist.from_x_array(
            x_array=loss_values,
            PMF_array=probabilities,
            p_neg_inf=p_loss_neg_inf,
            p_pos_inf=p_loss_inf,
        )
    )


class TestPLDDualTransformation:
    """Tests for PLD dual extraction and explicit negation."""

    def test_calc_pld_dual_matches_paper_definition(self):
        losses = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        probs = np.array([0.1, 0.2, 0.3, 0.25, 0.1])

        upper = _realization_from_loss_values(
            loss_values=losses,
            probabilities=probs,
            p_loss_inf=0.05,
        )

        dual = calc_pld_dual(upper)

        # Paper Definition 3.1 defines D(L) on the negated support with a +inf atom.
        paper_dual_losses = -losses[::-1]
        paper_dual_probs = probs[::-1] * np.exp(-losses[::-1])
        paper_dual_p_pos_inf = 1.0 - float(np.sum(paper_dual_probs))

        assert np.allclose(dual.x_array, paper_dual_losses, rtol=1e-12, atol=1e-12)
        assert np.allclose(dual.PMF_array, paper_dual_probs, rtol=1e-12, atol=1e-12)
        assert dual.p_pos_inf == pytest.approx(paper_dual_p_pos_inf, abs=1e-12)
        assert dual.p_neg_inf == pytest.approx(0.0, abs=1e-12)

        # External remove-direction code explicitly negates to get aligned -D(L).
        dual_negated = negate_reverse_linear_distribution(dual)
        assert np.allclose(dual_negated.x_array, losses, rtol=1e-12, atol=1e-12)
        assert np.allclose(dual_negated.PMF_array, probs * np.exp(-losses), rtol=1e-12, atol=1e-12)
        assert dual_negated.p_neg_inf == pytest.approx(paper_dual_p_pos_inf, abs=1e-12)
        assert dual_negated.p_pos_inf == pytest.approx(0.0, abs=1e-12)

    def test_dual_mass_conservation_basic(self):
        losses = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        probs = np.array([0.1, 0.2, 0.3, 0.25, 0.1])
        upper = _realization_from_loss_values(
            loss_values=losses,
            probabilities=probs,
            p_loss_inf=0.05,
        )

        dual = calc_pld_dual(upper)
        total_mass = np.sum(dual.PMF_array) + dual.p_neg_inf + dual.p_pos_inf
        assert abs(total_mass - 1.0) < 1e-9
        assert dual.p_neg_inf == 0.0

    def test_calc_pld_dual_rejects_non_realization_input(self):
        invalid_upper = LinearDiscreteDist.from_x_array(
            x_array=np.array([0.0, 1.0]),
            PMF_array=np.array([0.5, 0.4]),
            p_neg_inf=0.1,
            p_pos_inf=0.0,
        )

        with pytest.raises(TypeError, match="requires PLDRealization"):
            calc_pld_dual(invalid_upper)

    def test_calc_pld_dual_mass_conservation(self):
        test_cases = [
            (np.array([0.0, 1.0, 2.0]), np.array([0.3, 0.5, 0.15]), 0.05),
            (np.linspace(0, 3, 20), np.ones(20) / 21.0, 0.01),
            (np.array([0.0, 1.0]), np.array([0.4, 0.4]), 0.2),
        ]

        for losses, probs, p_pos_inf in test_cases:
            finite_target = 1.0 - p_pos_inf
            probs = probs / np.sum(probs) * finite_target
            upper = _realization_from_loss_values(
                loss_values=losses,
                probabilities=probs,
                p_loss_inf=p_pos_inf,
            )

            dual = calc_pld_dual(upper)
            total_mass = np.sum(dual.PMF_array) + dual.p_neg_inf + dual.p_pos_inf
            assert abs(total_mass - 1.0) < 1e-9


class TestSubsampleDistDual:
    """Tests for PLD-dual based distribution subsampling."""

    def test_subsample_PMF_mass_conservation_remove(self):
        losses = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        probs = np.array([0.3, 0.25, 0.2, 0.15, 0.05])

        dual_sum = np.sum(probs * np.exp(-losses))
        assert dual_sum <= 1.0

        base_dist = _realization_from_loss_values(
            loss_values=losses,
            probabilities=probs,
            p_loss_inf=0.05,
        )

        subsampled = subsample_PMF(
            base_pld=base_dist,
            sampling_prob=0.5,
            direction=Direction.REMOVE,
        )

        total_mass = (
            np.sum(subsampled.PMF_array)
            + subsampled.p_neg_inf
            + subsampled.p_pos_inf
        )

        assert abs(total_mass - 1.0) < 1e-6

    def test_subsample_PMF_mass_conservation_add(self):
        losses = np.linspace(0.0, 3.0, 50)
        add_upper = _realization_from_loss_values(
            loss_values=losses,
            probabilities=np.full(50, 0.98 / 50.0),
            p_loss_inf=0.02,
        )

        subsampled = subsample_PMF(
            base_pld=add_upper,
            sampling_prob=0.4,
            direction=Direction.ADD,
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
        losses = np.linspace(0.0, 3.0, 30)
        dist = _realization_from_loss_values(
            loss_values=losses,
            probabilities=np.full(30, 0.98 / 30.0),
            p_loss_inf=0.02,
        )

        subsampled = subsample_PMF(
            base_pld=dist,
            sampling_prob=sampling_prob,
            direction=direction,
        )

        total_mass = np.sum(subsampled.PMF_array) + subsampled.p_neg_inf + subsampled.p_pos_inf
        assert abs(total_mass - 1.0) < 1e-6

        assert np.sum(subsampled.PMF_array > 1e-10) > 0
