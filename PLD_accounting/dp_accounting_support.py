"""Convert between ``dp_accounting`` PMFs and structured PLDs.

Imports admit only pessimistic PMFs, then classify negative mass, total-mass
drift, and reciprocal-moment repair under separate caller-visible bands.
"""

from __future__ import annotations

import math
from itertools import chain

import numpy as np
from dp_accounting.pld.pld_pmf import DensePLDPmf, PLDPmf, SparsePLDPmf

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    GridSpec,
    PLDRealization,
    require_linear_reals_dist,
    require_zero_anchor,
)
from PLD_accounting.distribution_utils import (
    PMF_MASS_DRIFT_TOL,
    PMF_TOLERATED_MASS_TOL,
    classify_residual,
    enforce_mass_conservation,
    trim_mass_to_moment_target,
)
from PLD_accounting.types import BoundType, require_bound_type

# Repair ceilings for an uncomposed dp_accounting PMF. The shared drift floor
# keeps larger repairs visible; composed inputs may require wider ceilings.

# Mass residual.
DP_ACCOUNTING_MASS_REPAIR_TOL: float = 1e-5

# Mass the reciprocal-moment repair moves to +inf.
DP_ACCOUNTING_MOMENT_REPAIR_TOL: float = 1e-8

# Negative mass clipped to zero before any repair.
DP_ACCOUNTING_NEGATIVE_REPAIR_TOL: float = 1e-12


def linear_dist_to_dp_accounting_pmf(
    *,
    dist: DenseDiscreteDist,
    bound_type: BoundType,
) -> DensePLDPmf:
    """Convert a linear-grid loss PMF to a dp_accounting PMF.

    The input must use a zero-anchored ``k * step`` grid. Its ``index_0`` becomes
    ``lower_loss`` and ``p_max`` becomes the infinity mass. ``p_min`` is omitted
    because negative-infinity loss contributes no hockey-stick divergence.

    Args:
        dist: Real-domain linear distribution on ``k * step``.
        bound_type: Bound recorded by the PMF. A dominating export requires
            ``p_min`` within ``PMF_TOLERATED_MASS_TOL`` of zero.

    Returns:
        dp_accounting DensePLDPmf with infinity mass taken from dist.p_max.
    """
    require_linear_reals_dist(dist=dist, name="dist")
    require_bound_type(value=bound_type)
    if bound_type == BoundType.DOMINATES and dist.p_min > PMF_TOLERATED_MASS_TOL:
        raise ValueError("Dominating PMF conversion requires p_min = 0")

    # Zero-anchored by contract, so the lattice index is the export index, exactly.
    grid = require_zero_anchor(grid=dist.grid, name="dist.grid")
    base_index = grid.index_0
    return DensePLDPmf(
        discretization=dist.step,
        lower_loss=base_index,
        probs=dist.prob_arr.astype(np.float64),
        infinity_mass=dist.p_max,
        pessimistic_estimate=bound_type == BoundType.DOMINATES,
    )


def dp_accounting_pmf_to_pld_realization(
    *,
    pmf: PLDPmf,
    mass_drift_tol: float = PMF_MASS_DRIFT_TOL,
    mass_repair_tol: float = DP_ACCOUNTING_MASS_REPAIR_TOL,
    moment_drift_tol: float = PMF_MASS_DRIFT_TOL,
    moment_repair_tol: float = DP_ACCOUNTING_MOMENT_REPAIR_TOL,
    negative_drift_tol: float = PMF_MASS_DRIFT_TOL,
    negative_repair_tol: float = DP_ACCOUNTING_NEGATIVE_REPAIR_TOL,
) -> PLDRealization:
    """Convert a pessimistic dp_accounting PMF to a linear-grid PLD realization.

    Negative finite and infinity mass are clipped to zero without an upper clip,
    total mass is repaired directionally, and reciprocal-moment excess is moved
    to ``+inf``. Repairs above their drift band warn; repairs at or above their
    ceiling raise. Defaults cover an uncomposed PMF, so composed inputs may need
    wider ceilings.

    Args:
        pmf: Dense or sparse pessimistic dp_accounting PMF.
        mass_drift_tol: Mass residual normalized away as noise.
        mass_repair_tol: Mass residual that rejects the input.
        moment_drift_tol: Moment-repair mass movement treated as noise.
        moment_repair_tol: Moment-repair mass movement that rejects the input.
        negative_drift_tol: Negative mass clipped to zero as noise.
        negative_repair_tol: Negative mass that rejects the input.

    Returns:
        A zero-anchored ``PLDRealization`` satisfying the package invariants.
    """
    require_importable_dp_accounting_pmf(pmf)
    grid, probs_dense, inf_mass = _pmf_to_dense_components(pmf)
    # The moment repair and the realization both take their losses from this one grid,
    # so they cannot disagree about a coordinate.
    x_values = grid.materialize()

    classify_residual(
        residual=math.fsum(
            chain(
                (float(-value) for value in probs_dense[probs_dense < 0.0]),
                (float(-inf_mass),) if inf_mass < 0.0 else (),
            )
        ),
        drift_tol=negative_drift_tol,
        repair_tol=negative_repair_tol,
        context="dp_accounting import negative mass",
        repair="clipping it to zero",
    )
    probs_dense = np.maximum(probs_dense, 0.0)
    inf_mass = max(inf_mass, 0.0)

    # Repair total mass before the moment, because finite mass contributes to both.
    probs_dense, _p_min, inf_mass = enforce_mass_conservation(
        prob_arr=probs_dense,
        expected_p_min=0.0,
        expected_p_max=inf_mass,
        bound_type=BoundType.DOMINATES,
        drift_tol=mass_drift_tol,
        repair_tol=mass_repair_tol,
        context="dp_accounting import mass",
    )
    # Restore the moment invariant, then classify the mass this repair moved.
    repaired_probs, repaired_inf_mass = trim_mass_to_moment_target(
        prob_arr=probs_dense,
        loss=x_values,
        p_max=inf_mass,
        context="dp_accounting import reciprocal moment",
    )
    moved_mass = repaired_inf_mass - inf_mass
    classify_residual(
        residual=moved_mass,
        drift_tol=moment_drift_tol,
        repair_tol=moment_repair_tol,
        context="dp_accounting import reciprocal moment",
        repair="moving finite mass to +inf",
    )
    probs_dense, inf_mass = repaired_probs, repaired_inf_mass

    # The constructor validates unit mass against PMF_TOLERATED_MASS_TOL and the
    # reciprocal moment against REALIZATION_MOMENT_TOL, so the repairs above are
    # checked by construction; a separate postcondition would restate it.
    return PLDRealization(
        grid=grid,
        prob_arr=probs_dense,
        p_max=inf_mass,
        p_min=0.0,
    )


def require_importable_dp_accounting_pmf(pmf: object) -> PLDPmf:
    """Reject an unsupported representation or an optimistic dp_accounting PMF."""
    if not isinstance(pmf, (DensePLDPmf, SparsePLDPmf)):
        raise TypeError(
            f"Unrecognized PMF format: {type(pmf)}. Expected DensePLDPmf or SparsePLDPmf."
        )
    if getattr(pmf, "_pessimistic_estimate") is not True:
        raise ValueError(
            "An optimistic dp_accounting PMF does not map to either package BoundType. "
            "Only pessimistic PMFs can be imported."
        )
    return pmf


# ============================================================================
# Conversion helpers
# ============================================================================


def _pmf_to_dense_components(pmf: PLDPmf) -> tuple[GridSpec, np.ndarray, float]:
    """Densify a dp_accounting PMF on its zero-anchored integer lattice."""
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
        raise TypeError(
            f"Unrecognized PMF format: {type(pmf)}. Expected DensePLDPmf or SparsePLDPmf."
        )

    inf_mass = float(pmf._infinity_mass)
    grid = GridSpec(
        step=float(pmf._discretization),
        n=probs_dense.size,
        index_0=lower_index,
    )
    return grid, probs_dense, inf_mass
