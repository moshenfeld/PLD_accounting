"""Distribution-level operations shared by the convolution and allocation routes.

Boundary-mass algebra for sums, binary self-composition, combination of two bounds
of one quantity, the exp/log and reflection transforms, and the canonical PLD dual.

Every operation preserves or explicitly replaces its inputs' ``GridSpec``.
``calc_pld_dual`` is the single dualization boundary, and the place where a
reciprocal-moment deficit becomes mass at ``+inf``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    DiscreteDistBase,
    Domain,
    GridSpec,
    PLDRealization,
    require_dense_dist,
    require_geometric_positives_dist,
    require_linear_reals_dist,
)
from PLD_accounting.distribution_discretization import (
    project_dist_onto_grid_ctd,
    project_dist_onto_grid_stoch_dom,
)
from PLD_accounting.distribution_utils import (
    enforce_mass_conservation,
    exp_moment_terms,
    kahan_reverse_exclusive_cumsum,
    signed_unit_residual,
    trim_mass_from_edge,
)
from PLD_accounting.types import BoundType, require_bound_type
from PLD_accounting.validation import (
    require_nonnegative_real,
    require_positive_int,
    require_type,
)

# =============================================================================
# Boundary-Mass Convolution Utilities
# =============================================================================


def convolve_boundary_masses(
    *,
    p_min_1: float,
    p_max_1: float,
    p_min_2: float,
    p_max_2: float,
    domain: Domain,
) -> tuple[float, float]:
    """Compute boundary masses (p_min, p_max) for the convolution Z = X + Y.

    Domain semantics differ for the lower boundary:
    - REALS    (−∞ absorbing): P(Z=−∞) = 1 − (1−p_min_1)(1−p_min_2)
    - POSITIVES (0 neutral):   P(Z=0)  = p_min_1 · p_min_2

    The upper boundary (+∞ absorbing) uses the same formula for both domains:
      P(Z=+∞) = 1 − (1−p_max_1)(1−p_max_2)

    Both inputs must share the same domain.
    """

    # p_max: +∞ is always absorbing
    p_max = float(np.clip(-np.expm1(np.log1p(-p_max_1) + np.log1p(-p_max_2)), 0.0, 1.0))

    # p_min: depends on domain
    if domain == Domain.POSITIVES:
        # 0 is neutral: Z=0 only when both X=0 and Y=0
        p_min = p_min_1 * p_min_2
    else:
        # −∞ is absorbing: Z=−∞ when either X=−∞ or Y=−∞
        p_min = float(np.clip(-np.expm1(np.log1p(-p_min_1) + np.log1p(-p_min_2)), 0.0, 1.0))

    return p_min, p_max


def self_convolve_boundary_masses(
    *,
    dist: DiscreteDistBase,
    num_convolutions: int,
) -> tuple[float, float]:
    """Compute boundary masses after ``num_convolutions`` self-convolutions."""
    # p_max: absorbing in both domains
    p_max = float(np.clip(-np.expm1(num_convolutions * np.log1p(-dist.p_max)), 0.0, 1.0))

    if dist.domain == Domain.POSITIVES:
        # 0 is neutral: Z=0 only when all num_convolutions factors are 0
        p_min = dist.p_min**num_convolutions
    else:
        # −∞ is absorbing: Z=−∞ when any copy is −∞
        p_min = float(np.clip(-np.expm1(num_convolutions * np.log1p(-dist.p_min)), 0.0, 1.0))

    return p_min, p_max


# =============================================================================
# Public Convolution Utilities
# =============================================================================


def binary_self_convolve(
    *,
    dist: DenseDiscreteDist,
    num_convolutions: int,
    tail_truncation: float,
    bound_type: BoundType,
    convolve: Callable[..., DenseDiscreteDist],
) -> DenseDiscreteDist:
    """Exponentiation by squaring-based self-convolution using a provided convolve function.

    Algorithm 3 (`self-conv`), in Appendix C of
    https://arxiv.org/abs/2602.17284.

    Each intermediate distribution carries its own ``GridSpec``, so ``convolve``
    derives the output lattice from its two inputs; no lattice metadata travels
    alongside the distributions.
    """
    require_dense_dist(dist=dist, name="dist")
    require_positive_int(value=num_convolutions, name="num_convolutions")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_bound_type(value=bound_type)
    if num_convolutions == 1:
        return dist

    base_dist = dist
    acc_dist = None
    # Tail budget. Every convolve below is charged tail_truncation / num_convolutions
    # using the *current* counter, which halves each pass, so the per-call charge
    # doubles each pass and peaks at tail_truncation on the final pass. A doubling
    # series sums to under twice its last term, so the squaring calls together spend
    # under 2 * tail_truncation, and the accumulating calls likewise. Pre-dividing by
    # those two families' combined factor of 4 keeps the total within the caller's budget.
    tail_truncation /= 4
    while num_convolutions > 0:
        if num_convolutions & 1:
            if acc_dist is None:
                acc_dist = base_dist
            else:
                acc_dist = convolve(
                    dist_1=acc_dist,
                    dist_2=base_dist,
                    tail_truncation=tail_truncation / num_convolutions,
                    bound_type=bound_type,
                )
        num_convolutions >>= 1
        if num_convolutions > 0:
            base_dist = convolve(
                dist_1=base_dist,
                dist_2=base_dist,
                tail_truncation=tail_truncation / num_convolutions,
                bound_type=bound_type,
            )
    # For a power-of-two count acc_dist is never set; return the final squared base_dist.
    return acc_dist if acc_dist is not None else base_dist


def combine_distributions(
    *,
    dist_1: DenseDiscreteDist,
    dist_2: DenseDiscreteDist,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Combine two same-grid distributions by tightening bounds via CCDF min/max.

    For DOMINATES: returns tighter dominating distribution using pointwise min CCDF.
    For IS_DOMINATED: returns tighter dominated distribution using pointwise max CCDF.
    Both inputs must use the same domain and spacing type and already share an
    identical dense support grid; callers are responsible for projecting one
    distribution onto the other's grid first.
    """
    require_dense_dist(dist=dist_1, name="dist_1")
    require_dense_dist(dist=dist_2, name="dist_2")
    require_bound_type(value=bound_type)
    if dist_1.domain != dist_2.domain:
        raise ValueError("combine_distributions requires matching domains")
    if dist_1.spacing_type != dist_2.spacing_type:
        raise ValueError("combine_distributions requires matching spacing types")
    ccdf_op: Any
    if bound_type == BoundType.DOMINATES:
        ccdf_op = np.minimum
    elif bound_type == BoundType.IS_DOMINATED:
        ccdf_op = np.maximum
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    if dist_1.grid != dist_2.grid:
        raise ValueError(
            f"combine_distributions requires identical support grids, got "
            f"{dist_1.grid} and {dist_2.grid}"
        )

    ccdf_1 = _ccdf_from_pmf(dist_1)
    ccdf_2 = _ccdf_from_pmf(dist_2)
    combined_ccdf = ccdf_op(ccdf_1, ccdf_2)
    prob_arr = combined_ccdf[:-2] - combined_ccdf[1:-1]

    # The boundary atoms are the CCDF limits, so the pointwise CCDF operation
    # carries through to them: min CCDF gives max p_min and min p_max, max CCDF
    # the reverse.
    if bound_type == BoundType.DOMINATES:
        expected_p_min = max(dist_1.p_min, dist_2.p_min)
        expected_p_max = min(dist_1.p_max, dist_2.p_max)
    else:
        expected_p_min = min(dist_1.p_min, dist_2.p_min)
        expected_p_max = max(dist_1.p_max, dist_2.p_max)
    # Repair numerical drift after the CCDF operation fixes both boundary atoms.
    prob_arr, p_min, p_max = enforce_mass_conservation(
        prob_arr=prob_arr,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        grid=dist_1.grid,
        prob_arr=prob_arr,
        p_min=p_min,
        p_max=p_max,
        domain=dist_1.domain,
    )


def combine_best_of_two_plds(
    *,
    dist_1: DenseDiscreteDist,
    dist_2: DenseDiscreteDist,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Combine two same-quantity directional PLDs on the finer of their grids.

    Both inputs are complete directional PLDs of the same quantity (e.g. the
    FFT and GEOM candidates of a BEST_OF_TWO build), each constructed with its
    own correct budget scaling.  The coarser candidate is projected onto the
    finer candidate's exact lattice with domination-aware rounding, and the
    pointwise CCDF combination runs on that shared grid.  The anchored (finer)
    candidate's values are untouched, so the result is never looser than it,
    and never looser than the other candidate by more than one fine grid step.
    On equal steps ``dist_1`` is the anchor.
    """
    require_linear_reals_dist(dist=dist_1, name="dist_1")
    require_linear_reals_dist(dist=dist_2, name="dist_2")
    require_bound_type(value=bound_type)
    if bound_type == BoundType.DOMINATES:
        boundary, label = "p_min", "dominating"
    else:
        boundary, label = "p_max", "dominated"
    for name, dist in (("dist_1", dist_1), ("dist_2", dist_2)):
        value = getattr(dist, boundary)
        if value != 0.0:
            raise ValueError(
                f"combine_best_of_two_plds requires canonical {label} inputs with "
                f"{boundary} = 0 exactly; {name}.{boundary}={value:.2e}"
            )

    if dist_1.step <= dist_2.step:
        anchor_dist, other_dist = dist_1, dist_2
    else:
        anchor_dist, other_dist = dist_2, dist_1

    anchor_grid = anchor_dist.grid
    other_grid = other_dist.grid
    step = anchor_grid.step

    if anchor_grid.step == other_grid.step and anchor_grid.anchor == other_grid.anchor:
        # Same lattice, different windows: the union is an integer index range and
        # both candidates embed into it exactly, so neither needs reprojection.
        out_index_0 = min(anchor_grid.index_0, other_grid.index_0)
        out_n = (
            max(anchor_grid.index_0 + anchor_grid.n, other_grid.index_0 + other_grid.n)
            - out_index_0
        )
        out_grid = replace(anchor_grid, index_0=out_index_0, n=out_n)
        return combine_distributions(
            dist_1=_embed_on_grid(
                dist=anchor_dist, grid=out_grid, offset=anchor_grid.index_0 - out_index_0
            ),
            dist_2=_embed_on_grid(
                dist=other_dist, grid=out_grid, offset=other_grid.index_0 - out_index_0
            ),
            bound_type=bound_type,
        )

    # Genuinely different lattices: keep the finer candidate's own lattice, widen it by
    # whole steps to cover the other, and reproject only the other candidate.
    n_left = max(0, int(np.ceil((anchor_grid.x_0 - other_grid.x_0) / step)))
    n_right = max(0, int(np.ceil((other_grid.last_point - anchor_grid.last_point) / step)))
    out_grid = anchor_grid.pad(left=n_left, right=n_right)

    other_on_grid: DenseDiscreteDist
    if bound_type == BoundType.DOMINATES:
        other_on_grid = project_dist_onto_grid_ctd(dist=other_dist, grid=out_grid)
    else:
        other_on_grid = project_dist_onto_grid_stoch_dom(
            dist=other_dist,
            grid=out_grid,
            bound_type=bound_type,
        )

    return combine_distributions(
        dist_1=_embed_on_grid(dist=anchor_dist, grid=out_grid, offset=n_left),
        dist_2=other_on_grid,
        bound_type=bound_type,
    )


# =============================================================================
# Distribution Transform Utilities
# =============================================================================


def exp_linear_to_geometric(dist: DenseDiscreteDist) -> DenseDiscreteDist:
    """Apply exp(.) to a linear-grid distribution, producing a geometric-grid distribution.

    Maps REALS domain → POSITIVES domain.
    The −∞ atom (p_min in REALS) maps to the 0 atom (p_min in POSITIVES).
    """
    require_linear_reals_dist(dist=dist, name="dist")
    return DenseDiscreteDist(
        grid=dist.grid.exp(),
        prob_arr=dist.prob_arr.copy(),
        p_min=dist.p_min,
        p_max=dist.p_max,
        domain=Domain.POSITIVES,
    )


def log_geometric_to_linear(dist: DenseDiscreteDist) -> DenseDiscreteDist:
    """Apply log(.) to a geometric-grid distribution, producing a linear-grid distribution.

    Maps POSITIVES domain → REALS domain.
    The 0 atom (p_min in POSITIVES) maps to the −∞ atom (p_min in REALS).
    """
    require_geometric_positives_dist(dist=dist, name="dist")
    return DenseDiscreteDist(
        grid=dist.grid.log(),
        prob_arr=dist.prob_arr.copy(),
        p_min=dist.p_min,
        p_max=dist.p_max,
        domain=Domain.REALS,
    )


def negate_reverse_linear_distribution(
    dist: DenseDiscreteDist,
) -> DenseDiscreteDist:
    """Map X -> -X, reverse PMF order, and swap boundary atoms."""
    require_linear_reals_dist(dist=dist, name="dist")
    return DenseDiscreteDist(
        grid=dist.grid.reflect(),
        prob_arr=np.flip(dist.prob_arr),
        p_min=dist.p_max,
        p_max=dist.p_min,
    )


def calc_pld_dual(realization: PLDRealization) -> PLDRealization:
    """Compute the paper PLD dual ``D(L)`` (Definition 3.1).

    Algorithm 7 (`PLD-dual`), in Appendix C of
    https://arxiv.org/abs/2602.17284.

    For a PLD realization ``L`` with support ``l`` and mass ``f_L(l)``, the dual has:
    - finite mass ``f_D(-l) = f_L(l) * exp(-l)``,
    - support reflected to ``-l``,
    - residual mass at ``+inf``.
    """
    require_type(value=realization, expected_type=PLDRealization, name="realization")

    dual_grid = realization.grid.reflect()
    dual_loss = dual_grid.materialize()
    dual_probs = exp_moment_terms(
        prob_arr=np.flip(realization.prob_arr),
        x_vals=-dual_loss,
    )

    dual_residual = signed_unit_residual(values=dual_probs, lower_term=0.0, upper_term=0.0)
    if dual_residual < 0.0:
        # Input may sit in (1, 1 + REALIZATION_MOMENT_TOL]; trim cheapest mass to +inf.
        # Do not classify_residual: its middle band would warn on that admitted excess.
        dual_probs = trim_mass_from_edge(
            prob_arr=dual_probs,
            mass=-dual_residual,
            from_left=True,
        )
        dual_residual = signed_unit_residual(values=dual_probs, lower_term=0.0, upper_term=0.0)

    return PLDRealization(
        grid=dual_grid,
        prob_arr=dual_probs,
        p_max=max(0.0, dual_residual),
        p_min=0.0,
    )


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _embed_on_grid(
    *,
    dist: DenseDiscreteDist,
    grid: GridSpec,
    offset: int,
) -> DenseDiscreteDist:
    """Zero-pad ``dist`` onto a wider grid it already sits on at integer ``offset``.

    ``grid.slice(start=offset, n=dist.grid.n)`` reproduces the source grid exactly, so this
    moves no mass and rounds no coordinate.
    """
    if grid.slice(start=offset, n=dist.grid.n) != dist.grid:
        raise ValueError(
            f"Cannot embed {dist.grid} into {grid} at offset {offset}: not the same lattice"
        )
    prob_arr = np.zeros(grid.n, dtype=np.float64)
    end = offset + dist.grid.n
    prob_arr[offset:end] = dist.prob_arr
    return DenseDiscreteDist(
        grid=grid,
        prob_arr=prob_arr,
        p_min=dist.p_min,
        p_max=dist.p_max,
        domain=dist.domain,
    )


def _ccdf_from_pmf(dist: DiscreteDistBase) -> NDArray[np.float64]:
    """Convert distribution PMF to padded complementary CDF.

    Returns CCDF over [−∞/0, l_0, l_1, ..., +∞]:
      CCDF[i] = P(X > position[i])

    Includes both boundary atoms (p_min at the left, p_max at the right).
    """
    padded_probs = np.concatenate(([dist.p_min], dist.prob_arr, [dist.p_max]))
    return kahan_reverse_exclusive_cumsum(values=padded_probs)
