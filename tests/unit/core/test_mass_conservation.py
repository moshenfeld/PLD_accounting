"""Unit tests for mass-conservation and truncation primitives.

Covers ``enforce_mass_conservation``, ``compute_truncation``, and mass-edge draining.
"""

import math

import numpy as np
import pytest

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    GridSpec,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_utils import (
    PMF_MASS_DRIFT_TOL,
    PMF_TOLERATED_MASS_TOL,
    _drain_mass_from_edge,
    compute_truncation,
    enforce_mass_conservation,
)
from PLD_accounting.types import BoundType
from tests.test_tolerances import TestTolerances as TOL


class TestEnforceMassConservation:
    """Test directional boundary enforcement semantics."""

    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_tiny_excess_is_trimmed_from_the_giveable_edge(self, bound_type: BoundType) -> None:
        """Even drift-sized excess is repaired directionally.

        The bands decide whether to warn; they do not change the bound-preserving
        repair direction.
        """
        prob_arr = np.array(
            [0.2, 0.3, 0.5 + PMF_MASS_DRIFT_TOL / 2],
            dtype=np.float64,
        )
        current_mass = math.fsum(map(float, prob_arr))
        excess = current_mass - 1.0
        assert 0.0 < excess < PMF_MASS_DRIFT_TOL

        prob_out, _, _ = enforce_mass_conservation(
            prob_arr=prob_arr,
            expected_p_min=0.0,
            expected_p_max=0.0,
            bound_type=bound_type,
        )

        giveable = 0 if bound_type == BoundType.DOMINATES else prob_arr.size - 1
        untouched = [i for i in range(prob_arr.size) if i != giveable]
        np.testing.assert_array_equal(prob_out[untouched], prob_arr[untouched])
        assert prob_out[giveable] < prob_arr[giveable]
        assert math.isclose(
            math.fsum(map(float, prob_out)), 1.0, rel_tol=0.0, abs_tol=TOL.MASS_CONSERVATION
        )

    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_rejects_excess_too_large_to_be_drift(self, bound_type: BoundType) -> None:
        """Undeclared surplus mass must fail rather than be trimmed away silently."""
        with pytest.raises(ValueError, match="exceeds the repair tolerance"):
            enforce_mass_conservation(
                prob_arr=np.array([0.4, 0.1], dtype=np.float64),
                expected_p_min=0.3,
                expected_p_max=0.4,
                bound_type=bound_type,
            )

    def test_dominates_trims_left_including_p_min_within_widened_bands(self):
        """Middle-band excess is trimmed from the left, consuming p_min, and warns."""
        prob_arr = np.array([0.4, 0.1], dtype=np.float64)
        with pytest.warns(RuntimeWarning, match="enforce_mass_conservation: residual"):
            prob_out, p_min, p_max = enforce_mass_conservation(
                prob_arr=prob_arr,
                expected_p_min=0.3,
                expected_p_max=0.4,
                bound_type=BoundType.DOMINATES,
                drift_tol=0.1,
                repair_tol=0.5,
            )

        assert np.allclose(prob_out, np.array([0.4, 0.1]))
        assert np.isclose(p_min, 0.1)
        assert np.isclose(p_max, 0.4)
        assert np.isclose(math.fsum([*map(float, prob_out), p_min, p_max]), 1.0)

    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_clamps_float_overshoot_on_expected_boundaries(self, bound_type: BoundType) -> None:
        """A slightly negative expected boundary is overshoot, not an illegal mass."""
        overshoot = -PMF_TOLERATED_MASS_TOL / 2
        prob_arr = np.array([0.4, 0.6], dtype=np.float64)
        _, p_min, p_max = enforce_mass_conservation(
            prob_arr=prob_arr,
            expected_p_min=overshoot,
            expected_p_max=0.0,
            bound_type=bound_type,
        )
        assert p_min >= 0.0
        assert p_max >= 0.0

    def test_is_dominated_trims_right_including_p_max_within_widened_bands(self):
        """Middle-band excess is trimmed from the right, consuming p_max, and warns."""
        prob_arr = np.array([0.1, 0.4], dtype=np.float64)
        with pytest.warns(RuntimeWarning, match="enforce_mass_conservation: residual"):
            prob_out, p_min, p_max = enforce_mass_conservation(
                prob_arr=prob_arr,
                expected_p_min=0.4,
                expected_p_max=0.3,
                bound_type=BoundType.IS_DOMINATED,
                drift_tol=0.1,
                repair_tol=0.5,
            )

        assert np.allclose(prob_out, np.array([0.1, 0.4]))
        assert np.isclose(p_min, 0.4)
        assert np.isclose(p_max, 0.1)
        assert np.isclose(math.fsum([*map(float, prob_out), p_min, p_max]), 1.0)


class TestMassEnforcementDeficit:
    """Test the three-band repair of mass residuals."""

    @pytest.mark.parametrize(
        "bound_type",
        [
            BoundType.DOMINATES,
            BoundType.IS_DOMINATED,
        ],
    )
    def test_drift_sized_deficit_loads_the_conservative_edge(
        self,
        bound_type: BoundType,
    ) -> None:
        """A drift-sized deficit goes to the conservative edge.

        A negligible bin at that edge distinguishes the directional repair from
        a proportional rescale.
        """
        tiny = 1e-300
        bulk = [0.6, 0.4 - 2.0**-53]
        prob_arr = np.array([*bulk, tiny] if bound_type == BoundType.DOMINATES else [tiny, *bulk])
        deficit = 1.0 - math.fsum(map(float, prob_arr))
        assert 0.0 < deficit < PMF_MASS_DRIFT_TOL
        edge_index = -1 if bound_type == BoundType.DOMINATES else 0

        prob_out, p_min_out, p_max_out = enforce_mass_conservation(
            prob_arr=prob_arr,
            expected_p_min=0.0,
            expected_p_max=0.0,
            bound_type=bound_type,
        )

        assert prob_out[edge_index] > 1e-299
        untouched = [i for i in range(prob_arr.size) if i != edge_index % prob_arr.size]
        np.testing.assert_array_equal(prob_out[untouched], prob_arr[untouched])
        assert math.isclose(
            math.fsum(map(float, prob_out)), 1.0, rel_tol=0.0, abs_tol=TOL.MASS_CONSERVATION
        )
        assert p_min_out == 0.0
        assert p_max_out == 0.0

    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_warns_and_uses_conservative_edge_between_the_thresholds(
        self, bound_type: BoundType
    ) -> None:
        """A deficit above drift scale is still repaired directionally, but visibly."""
        deficit = (PMF_MASS_DRIFT_TOL + PMF_TOLERATED_MASS_TOL) / 2
        prob_arr = np.array([0.5, 0.5 - deficit])
        edge_index = -1 if bound_type == BoundType.DOMINATES else 0

        with pytest.warns(RuntimeWarning, match="enforce_mass_conservation: residual"):
            prob_out, _, _ = enforce_mass_conservation(
                prob_arr=prob_arr,
                expected_p_min=0.0,
                expected_p_max=0.0,
                bound_type=bound_type,
            )

        assert prob_out[edge_index] > prob_arr[edge_index]
        assert math.isclose(
            math.fsum(map(float, prob_out)), 1.0, rel_tol=0.0, abs_tol=TOL.MASS_CONSERVATION
        )

    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_rejects_deficit_too_large_to_be_drift(self, bound_type: BoundType) -> None:
        """Undeclared omitted mass must fail rather than be hidden as conservatism."""
        with pytest.raises(ValueError, match="exceeds the repair tolerance"):
            enforce_mass_conservation(
                prob_arr=np.array([0.4, 0.4]),
                expected_p_min=0.0,
                expected_p_max=0.0,
                bound_type=bound_type,
            )

    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_caller_supplied_tolerances_widen_the_bands(self, bound_type: BoundType) -> None:
        """Widened bands suppress the warning without changing repair direction."""
        prob_arr = np.array([0.4, 0.4])
        edge_index = -1 if bound_type == BoundType.DOMINATES else 0
        prob_out, _, _ = enforce_mass_conservation(
            prob_arr=prob_arr,
            expected_p_min=0.0,
            expected_p_max=0.0,
            bound_type=bound_type,
            drift_tol=0.5,
            repair_tol=1.0,
        )

        expected = prob_arr.copy()
        expected[edge_index] += 0.2
        np.testing.assert_allclose(prob_out, expected)

    @pytest.mark.parametrize("bound_type", [BoundType.DOMINATES, BoundType.IS_DOMINATED])
    def test_repairs_deficit_when_bands_permit(self, bound_type: BoundType) -> None:
        """With wide enough bands the directional edge repair still conserves mass."""
        prob_arr = np.array([0.4, 0.4])
        expected_prob = prob_arr.copy()
        edge_index = -1 if bound_type == BoundType.DOMINATES else 0
        expected_prob[edge_index] += 0.2

        with pytest.warns(RuntimeWarning, match="enforce_mass_conservation: residual"):
            prob_out, p_min, p_max = enforce_mass_conservation(
                prob_arr=prob_arr,
                expected_p_min=0.0,
                expected_p_max=0.0,
                bound_type=bound_type,
                drift_tol=0.1,
                repair_tol=0.5,
            )

        np.testing.assert_allclose(prob_out, expected_prob)
        assert p_min == 0.0
        assert p_max == 0.0


class TestComputeTruncation:
    """Test zero-edge stripping and index bookkeeping in truncation."""

    def test_strips_zero_edges_before_tail_truncation(self):
        """Strips zero edges before tail truncation."""
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.0, 0.8], dtype=np.float64),
            p_min=0.0,
            p_max=0.2,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(new_prob_arr, np.array([0.8], dtype=np.float64))
        assert np.isclose(new_p_min, 0.0)
        assert np.isclose(new_p_max, 0.2)
        assert (min_ind, max_ind) == (1, 1)

    def test_keeps_boundary_when_it_is_the_first_remaining_element(self):
        """Keeps boundary when it is the first remaining element."""
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.0, 0.2, 0.5], dtype=np.float64),
            p_min=0.3,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(new_prob_arr, np.array([0.2, 0.5], dtype=np.float64))
        assert np.isclose(new_p_min, 0.3)
        assert np.isclose(new_p_max, 0.0)
        assert (min_ind, max_ind) == (1, 2)

    def test_rejects_a_pmf_whose_finite_mass_is_all_zero(self):
        """A law with no finite mass has no support to truncate; it must not slip through."""
        with pytest.raises(ValueError, match="zero finite mass"):
            compute_truncation(
                prob_arr=np.zeros(4, dtype=np.float64),
                p_min=0.0,
                p_max=0.0,
                tail_truncation=0.0,
                bound_type=BoundType.DOMINATES,
            )

    def test_rejects_all_zero_finite_mass_even_when_truncating(self):
        """The same holds on the truncating path, not just the tail_truncation == 0 one."""
        with pytest.raises(ValueError):
            compute_truncation(
                prob_arr=np.zeros(4, dtype=np.float64),
                p_min=1.0,
                p_max=0.0,
                tail_truncation=1e-3,
                bound_type=BoundType.DOMINATES,
            )

    def test_truncation_folds_consumed_boundary_into_first_finite_bin(self):
        """Truncation folds consumed boundary into first finite bin."""
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.2, 0.75], dtype=np.float64),
            p_min=0.05,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=BoundType.DOMINATES,
        )

        assert np.allclose(new_prob_arr, np.array([0.25, 0.75], dtype=np.float64))
        assert np.isclose(new_p_min, 0.0)
        assert np.isclose(new_p_max, 0.0)
        assert (min_ind, max_ind) == (0, 1)

    def test_strips_zero_edges_for_is_dominated_right_tail(self):
        """Strips zero edges for is dominated right tail."""
        new_prob_arr, new_p_min, new_p_max, min_ind, max_ind = compute_truncation(
            prob_arr=np.array([0.8, 0.0], dtype=np.float64),
            p_min=0.2,
            p_max=0.0,
            tail_truncation=0.1,
            bound_type=BoundType.IS_DOMINATED,
        )

        assert np.allclose(new_prob_arr, np.array([0.8], dtype=np.float64))
        assert np.isclose(new_p_min, 0.2)
        assert np.isclose(new_p_max, 0.0)
        assert (min_ind, max_ind) == (0, 0)

    def test_dense_truncate_edges_updates_x_min_after_zero_edge_removal(self):
        """Dense truncate edges updates x min after zero edge removal."""
        dist = DenseDiscreteDist(
            grid=GridSpec(
                step=1.0,
                n=2,
                anchor=0.0,
            ),
            prob_arr=np.array([0.0, 0.8], dtype=np.float64),
            p_max=0.2,
        )

        result = dist.truncate_edges(tail_truncation=0.1, bound_type=BoundType.DOMINATES)

        assert np.allclose(result.x_array, np.array([1.0], dtype=np.float64))
        assert np.allclose(result.prob_arr, np.array([0.8], dtype=np.float64))
        assert np.isclose(result.p_min, 0.0)
        assert np.isclose(result.p_max, 0.2)

    def test_sparse_truncate_edges_updates_support_after_tail_zero_removal(self):
        """Sparse truncate edges updates support after tail zero removal."""
        dist = SparseDiscreteDist(
            x_array=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            prob_arr=np.array([0.8, 0.1, 0.1], dtype=np.float64),
        )

        result = dist.truncate_edges(tail_truncation=0.15, bound_type=BoundType.DOMINATES)

        assert np.allclose(result.x_array, np.array([1.0, 2.0], dtype=np.float64))
        assert np.allclose(result.prob_arr, np.array([0.8, 0.1], dtype=np.float64))
        assert np.isclose(result.p_min, 0.0)
        assert np.isclose(result.p_max, 0.1)


def test_raises_when_mass_is_at_least_total():
    """Raises when mass is at least total."""
    with pytest.raises(ValueError, match="mass must be smaller than total array mass"):
        _drain_mass_from_edge(
            values=np.array([0.2, 0.8], dtype=np.float64),
            mass=1.0,
            from_left=True,
            exact=True,
        )
