"""dp_accounting compatibility wrappers for subsampling implementation.

Provides translation between dp_accounting's PrivacyLossDistribution objects
and this project's structured discrete-distribution API.
"""

import math

import numpy as np
from dp_accounting.pld.pld_pmf import DensePLDPmf, PLDPmf, SparsePLDPmf

from PLD_accounting.discrete_dist import (
    REALIZATION_MOMENT_TOL,
    DenseDiscreteDist,
    PLDRealization,
)
from PLD_accounting.distribution_utils import (
    MAX_SAFE_EXP_ARG,
    SPACING_ATOL,
    SPACING_RTOL,
    exp_moment_terms,
)
from PLD_accounting.types import SpacingType

# ============================================================================
# Translation Functions: PLD realizations <-> dp_accounting
# ============================================================================


def linear_dist_to_dp_accounting_pmf(
    *,
    dist: DenseDiscreteDist,
    pessimistic_estimate: bool = True,
) -> DensePLDPmf:
    """Convert a linear-grid loss PMF to a dp_accounting PMF.

    Args:
        dist: Linear-grid loss distribution compatible with dp_accounting.
            Must be a linear DenseDiscreteDist with x_min aligned to step multiples.
        pessimistic_estimate: Whether to use pessimistic estimate in dp_accounting.

    Returns:
        dp_accounting DensePLDPmf with infinity mass taken from dist.p_max.
    """
    if not (isinstance(dist, DenseDiscreteDist) and dist.spacing_type == SpacingType.LINEAR):
        raise TypeError(
            f"linear_dist_to_dp_accounting_pmf requires DenseDiscreteDist, got {type(dist)}."
        )

    base_index = int(np.rint(dist.x_min / dist.step))
    if not np.isclose(base_index * dist.step, dist.x_min, atol=SPACING_ATOL, rtol=SPACING_RTOL):
        raise ValueError("PLDRealization x_min is not aligned to step multiples")
    return DensePLDPmf(
        discretization=dist.step,
        lower_loss=base_index,
        probs=dist.prob_arr.astype(np.float64),
        infinity_mass=dist.p_max,
        pessimistic_estimate=pessimistic_estimate,
    )


def dp_accounting_pmf_to_pld_realization(pmf: PLDPmf) -> PLDRealization:
    """Convert a dp_accounting PMF to a linear-grid PLD realization.

    Args:
        pmf: dp_accounting DensePLDPmf or SparsePLDPmf to convert.

    Returns:
        PLDRealization on a uniform linear grid with infinity mass in
        p_max and p_min set to 0.
    """
    x_min, discretization, probs_dense, x_values, inf_mass = _pmf_to_dense_components(pmf)
    probs_dense, inf_mass = _normalize_finite_mass(probs=probs_dense, inf_mass=inf_mass)
    probs_dense, inf_mass = _ensure_exp_moment_upper(
        probs=probs_dense,
        x_values=x_values,
        inf_mass=inf_mass,
    )
    return PLDRealization(
        x_min=x_min,
        step=discretization,
        prob_arr=probs_dense,
        p_max=inf_mass,
        p_min=0.0,
    )


def _pmf_to_dense_components(
    pmf: PLDPmf,
) -> tuple[float, float, np.ndarray, np.ndarray, float]:
    """Densify a dp_accounting PMF into a uniform loss grid."""
    if isinstance(pmf, DensePLDPmf):
        lower_index = int(pmf._lower_loss)
        probs_dense = np.asarray(pmf._probs, dtype=np.float64).copy()
    elif isinstance(pmf, SparsePLDPmf):
        loss_probs = pmf._loss_probs.copy()
        if len(loss_probs) == 0:
            raise ValueError("Empty dp_accounting PMF is not supported")

        loss_indices = np.array(sorted(loss_probs.keys()), dtype=np.int64)
        probs_sparse = np.array([loss_probs[int(idx)] for idx in loss_indices], dtype=np.float64)

        lower_index = int(loss_indices[0])
        upper_index = int(loss_indices[-1])
        probs_dense = np.zeros(upper_index - lower_index + 1, dtype=np.float64)
        for idx, prob in zip(loss_indices, probs_sparse):
            probs_dense[int(idx - lower_index)] = float(prob)
    else:
        raise AttributeError(
            f"Unrecognized PMF format: {type(pmf)}. Expected DensePLDPmf or SparsePLDPmf."
        )

    discretization = float(pmf._discretization)
    x_min = float(lower_index) * discretization
    x_values = x_min + discretization * np.arange(probs_dense.size, dtype=np.float64)
    return x_min, discretization, probs_dense, x_values, float(pmf._infinity_mass)


def _normalize_finite_mass(*, probs: np.ndarray, inf_mass: float) -> tuple[np.ndarray, float]:
    """Clip probabilities and adjust inf_mass so total mass equals exactly 1.

    Clipping negative entries to 0 can reduce the finite sum below ``1 - inf_mass``.
    Any such deficit is conservatively routed to ``inf_mass`` (i.e. ``p_max``),
    which is safe under DOMINATES semantics.  Excess finite mass is scaled down.
    """
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0.0, 1.0)
    inf_mass = float(np.clip(inf_mass, 0.0, 1.0))
    sum_probs = math.fsum(map(float, probs))
    finite_target = max(0.0, 1.0 - inf_mass)
    if sum_probs > finite_target:
        probs = probs * (finite_target / sum_probs)
    elif sum_probs < finite_target:
        # Deficit from clipped negatives: add to inf_mass to conserve total mass.
        inf_mass += finite_target - sum_probs
    return probs, inf_mass


def _ensure_exp_moment_upper(
    *,
    probs: np.ndarray,
    x_values: np.ndarray,
    inf_mass: float,
) -> tuple[np.ndarray, float]:
    """Enforce ``E[exp(-L)] <= 1`` by removing mass from the lowest-loss bins.

    Removed mass is routed to ``p_max``, which is conservative for DOMINATES
    semantics. Uses cumsum + ``searchsorted`` to locate the pivot bin in one pass.
    """
    if probs.size == 0:
        return probs, inf_mass

    contributions = exp_moment_terms(prob_arr=probs, x_vals=x_values)
    exp_moment_val = math.fsum(map(float, contributions))
    if exp_moment_val <= 1.0:
        return probs, inf_mass

    # Add a buffer of REALIZATION_MOMENT_TOL so floating-point rounding in the
    # repair itself cannot leave a residual that still fails validation.
    excess = exp_moment_val - 1.0 + REALIZATION_MOMENT_TOL
    cumsum = np.cumsum(contributions, dtype=np.float64)
    pivot = int(np.searchsorted(cumsum, excess, side="left"))
    prior = math.fsum(map(float, contributions[:pivot]))
    delta_contribution = max(0.0, excess - prior)

    if delta_contribution == 0.0:
        delta_mass = 0.0
    elif x_values[pivot] < -MAX_SAFE_EXP_ARG:
        delta_mass = math.exp(math.log(delta_contribution) + float(x_values[pivot]))
    else:
        delta_mass = delta_contribution / math.exp(-float(x_values[pivot]))
    delta_mass = min(max(0.0, delta_mass), float(probs[pivot]))

    new_probs = probs.copy()
    new_probs[:pivot] = 0.0
    new_probs[pivot] = max(0.0, float(probs[pivot]) - delta_mass)
    removed_mass = math.fsum(map(float, probs[:pivot])) + delta_mass
    new_inf_mass = inf_mass + removed_mass
    return new_probs, new_inf_mass
