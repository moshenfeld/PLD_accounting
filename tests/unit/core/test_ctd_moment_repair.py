"""The two CtD reciprocal-moment repairs, both settled by moving mass to ``+inf``.

One is an arithmetic excess left by the endpoint split; the other is the semantic dual
mass ``eta`` that the lower exterior cell cannot keep on a finite knot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from PLD_accounting.discrete_dist import GridSpec
from PLD_accounting.distribution_discretization import (
    _ctd_moment_repair_tol,
    _ctd_realization_from_cell_measures,
    _repair_ctd_reciprocal_moment,
)
from PLD_accounting.distribution_utils import exp_moment_terms, signed_unit_residual


def _residual(prob: np.ndarray, loss: np.ndarray) -> float:
    return signed_unit_residual(
        values=exp_moment_terms(prob_arr=prob, x_vals=loss), lower_term=0.0, upper_term=0.0
    )


def _flat_ctd_cell_measures(
    *, loss: np.ndarray, semantic_dual_mass: float
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-density interior cells plus a lower exterior carrying ``eta``."""
    interior_mass = np.full(loss.size - 1, 0.3, dtype=np.float64)
    interior_dual = np.array(
        [
            interior_mass[i]
            * (math.exp(-loss[i]) - math.exp(-loss[i + 1]))
            / (loss[i + 1] - loss[i])
            for i in range(loss.size - 1)
        ]
    )
    lower_mass = 1.0 - float(interior_mass.sum())
    lower_dual = math.exp(-float(loss[0])) * lower_mass + semantic_dual_mass
    return (
        np.concatenate(([lower_mass], interior_mass, [0.0])),
        np.concatenate(([lower_dual], interior_dual, [0.0])),
    )


def test_arithmetic_repair_restores_the_invariant_and_banks_the_move_at_p_max() -> None:
    """An excess inside the producer band is drained to ``+inf``, not clipped away."""
    # Symmetric about zero, so E[exp(-L)] exceeds one by 2 * p * (cosh(0.1) - 1), which
    # stays inside the producer band.
    loss = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    prob = np.array([1e-13, 1.0 - 2e-13, 1e-13], dtype=np.float64)
    excess = -_residual(prob, loss)
    assert 0.0 < excess < _ctd_moment_repair_tol(max_abs_loss=0.1, step=0.1)

    repaired, p_max = _repair_ctd_reciprocal_moment(prob=prob, loss=loss, p_max=0.0, step=0.1)

    assert _residual(repaired, loss) >= 0.0
    assert p_max == pytest.approx(math.fsum(map(float, prob - repaired)))
    assert p_max > 0.0


def test_arithmetic_repair_rejects_an_excess_above_the_producer_band() -> None:
    """An excess this large is not rounding, so it is refused rather than drained."""
    loss = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    prob = np.array([1e-3, 1.0 - 2e-3, 1e-3], dtype=np.float64)
    assert -_residual(prob, loss) > _ctd_moment_repair_tol(max_abs_loss=0.1, step=0.1)

    with pytest.raises(ValueError, match="CtD reciprocal-moment repair"):
        _repair_ctd_reciprocal_moment(prob=prob, loss=loss, p_max=0.0, step=0.1)


def test_semantic_repair_surrenders_only_the_moment_still_due() -> None:
    """The arithmetic repair goes first, so the semantic drain owes only the remainder."""
    grid = GridSpec(step=0.05, n=4, anchor=0.0, index_0=0)
    loss = grid.materialize()

    exact_pmf, exact_dual_pmf = _flat_ctd_cell_measures(loss=loss, semantic_dual_mass=0.0)
    baseline = _ctd_realization_from_cell_measures(
        pld_pmf=exact_pmf, pld_dual_pmf=exact_dual_pmf, p_max_in=0.0, grid=grid
    )
    residual = _residual(baseline.prob_arr, loss)
    assert residual > 0.0

    semantic_dual_mass = residual + 1e-16
    pld_pmf, pld_dual_pmf = _flat_ctd_cell_measures(
        loss=loss, semantic_dual_mass=semantic_dual_mass
    )
    result = _ctd_realization_from_cell_measures(
        pld_pmf=pld_pmf, pld_dual_pmf=pld_dual_pmf, p_max_in=0.0, grid=grid
    )

    still_due = semantic_dual_mass - residual
    assert result.p_max == pytest.approx(still_due, rel=1e-3)
    assert _residual(result.prob_arr, loss) >= semantic_dual_mass
