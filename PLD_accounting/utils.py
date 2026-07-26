"""Utility functions for distribution operations and numerical stability."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    DiscreteDistBase,
    Domain,
    GridSpec,
    PLDRealization,
)
from PLD_accounting.distribution_discretization import (
    fold_absorbable_boundary_atom,
    project_dist_onto_grid_ctd,
    project_dist_onto_grid_stoch_dom,
)
from PLD_accounting.distribution_utils import (
    enforce_mass_conservation,
    kahan_reverse_exclusive_cumsum,
    stable_array_equal,
)
from PLD_accounting.types import BoundType, SpacingType

# =============================================================================
# Boundary-Mass Convolution Utilities
# =============================================================================


def convolve_boundary_masses(
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
    lattice_anchor: float | None = None,
) -> DenseDiscreteDist:
    """Exponentiation by squaring-based self-convolution using a provided convolve function.

    Algorithm 3 (`self-conv`), in Appendix C of
    https://arxiv.org/abs/2602.17284.

    If ``lattice_anchor`` is provided, it is carried alongside each intermediate
    distribution and their sum is supplied to ``convolve`` as ``target_anchor``.
    """
    if num_convolutions < 1:
        raise ValueError(f"num_convolutions must be >= 1, got {num_convolutions}")
    if num_convolutions == 1:
        return dist

    def convolve_with_anchor(
        dist_1: DenseDiscreteDist,
        anchor_1: float | None,
        dist_2: DenseDiscreteDist,
        anchor_2: float | None,
        truncation: float,
    ) -> tuple[DenseDiscreteDist, float | None]:
        """Convolve two distributions and sum their optional lattice anchors."""
        # Anchors all descend from the one lattice_anchor, so they are either
        # both set or both None; only anchored convolves accept target_anchor.
        if anchor_1 is None or anchor_2 is None:
            target_anchor = None
            anchor_kwargs: dict[str, float] = {}
        else:
            target_anchor = anchor_1 + anchor_2
            anchor_kwargs = {"target_anchor": target_anchor}
        return (
            convolve(
                dist_1=dist_1,
                dist_2=dist_2,
                tail_truncation=truncation,
                bound_type=bound_type,
                **anchor_kwargs,
            ),
            target_anchor,
        )

    base_dist = dist
    base_anchor = lattice_anchor
    acc_dist = None
    acc_anchor = lattice_anchor
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
                acc_anchor = base_anchor
            else:
                acc_dist, acc_anchor = convolve_with_anchor(
                    acc_dist,
                    acc_anchor,
                    base_dist,
                    base_anchor,
                    tail_truncation / num_convolutions,
                )
        num_convolutions >>= 1
        if num_convolutions > 0:
            base_dist, base_anchor = convolve_with_anchor(
                base_dist,
                base_anchor,
                base_dist,
                base_anchor,
                tail_truncation / num_convolutions,
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

    if not stable_array_equal(value_1=dist_1.x_array, value_2=dist_2.x_array):
        raise ValueError(
            "combine_distributions requires identical support grids, got sizes "
            f"{dist_1.x_array.size} and {dist_2.x_array.size} with ranges "
            f"[{dist_1.x_array[0]}, {dist_1.x_array[-1]}] and "
            f"[{dist_2.x_array[0]}, {dist_2.x_array[-1]}]"
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
    prob_arr, p_min, p_max = enforce_mass_conservation(
        prob_arr=prob_arr,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_0=dist_1.x_0,
        step=dist_1.step,
        prob_arr=prob_arr,
        p_min=p_min,
        p_max=p_max,
        spacing_type=dist_1.spacing_type,
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
    if not (isinstance(dist_1, DenseDiscreteDist) and dist_1.spacing_type == SpacingType.LINEAR):
        raise TypeError(
            "dist_1: expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(dist_1).__name__} with spacing {getattr(dist_1, 'spacing_type', '?')}"
        )
    if not (isinstance(dist_2, DenseDiscreteDist) and dist_2.spacing_type == SpacingType.LINEAR):
        raise TypeError(
            "dist_2: expected DenseDiscreteDist with LINEAR spacing, "
            f"got {type(dist_2).__name__} with spacing {getattr(dist_2, 'spacing_type', '?')}"
        )

    if dist_1.step <= dist_2.step:
        anchor_dist, other_dist = dist_1, dist_2
    else:
        anchor_dist, other_dist = dist_2, dist_1

    # Extend the anchor lattice (preserving its offset) to cover the other support.
    step = anchor_dist.step
    other_x_max = other_dist.x_0 + (other_dist.prob_arr.size - 1) * other_dist.step
    anchor_x_max = anchor_dist.x_0 + (anchor_dist.prob_arr.size - 1) * step
    n_left = max(0, int(np.ceil((anchor_dist.x_0 - other_dist.x_0) / step)))
    n_right = max(0, int(np.ceil((other_x_max - anchor_x_max) / step)))
    out_grid = GridSpec(
        x_0=anchor_dist.x_0 - n_left * step,
        step=step,
        n=n_left + anchor_dist.prob_arr.size + n_right,
        spacing_type=SpacingType.LINEAR,
    )

    # Project the coarser candidate onto the shared lattice with
    # domination-aware rounding.
    other_working = fold_absorbable_boundary_atom(
        dist=other_dist,
        spacing_type=SpacingType.LINEAR,
        bound_type=bound_type,
    )
    other_on_grid: DenseDiscreteDist
    if bound_type == BoundType.DOMINATES:
        other_on_grid = project_dist_onto_grid_ctd(
            dist=other_working,
            grid=out_grid,
        )
    else:
        other_on_grid = project_dist_onto_grid_stoch_dom(
            dist=other_working,
            grid=out_grid,
            expected_p_min=other_working.p_min,
            expected_p_max=other_working.p_max,
            bound_type=bound_type,
        )

    # Embed the anchor candidate exactly (zero padding only; no rounding).
    anchor_prob_out = np.zeros(out_grid.n, dtype=np.float64)
    anchor_prob_out[slice(n_left, n_left + anchor_dist.prob_arr.size)] = anchor_dist.prob_arr
    return combine_distributions(
        dist_1=DenseDiscreteDist(
            x_0=out_grid.x_0,
            step=out_grid.step,
            prob_arr=anchor_prob_out,
            p_min=anchor_dist.p_min,
            p_max=anchor_dist.p_max,
        ),
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
    if dist.spacing_type != SpacingType.LINEAR:
        raise ValueError(
            f"exp_linear_to_geometric requires LINEAR spacing input, got {dist.spacing_type}"
        )
    x_min_exp = float(np.exp(dist.x_0))
    ratio_exp = float(np.exp(dist.step))

    return DenseDiscreteDist(
        x_0=x_min_exp,
        step=ratio_exp,
        prob_arr=dist.prob_arr.copy(),
        p_min=dist.p_min,  # −∞ atom → 0 atom (p_min identity preserved)
        p_max=dist.p_max,  # +∞ atom unchanged
        spacing_type=SpacingType.GEOMETRIC,
        domain=Domain.POSITIVES,
    )


def log_geometric_to_linear(dist: DenseDiscreteDist) -> DenseDiscreteDist:
    """Apply log(.) to a geometric-grid distribution, producing a linear-grid distribution.

    Maps POSITIVES domain → REALS domain.
    The 0 atom (p_min in POSITIVES) maps to the −∞ atom (p_min in REALS).
    """
    if dist.spacing_type != SpacingType.GEOMETRIC:
        raise ValueError(
            f"log_geometric_to_linear requires GEOMETRIC spacing input, got {dist.spacing_type}"
        )
    x_min_log = float(np.log(dist.x_0))
    step_log = float(np.log(dist.step))

    return DenseDiscreteDist(
        x_0=x_min_log,
        step=step_log,
        prob_arr=dist.prob_arr.copy(),
        p_min=dist.p_min,  # 0 atom → −∞ atom (p_min identity preserved)
        p_max=dist.p_max,
        spacing_type=SpacingType.LINEAR,
        domain=Domain.REALS,
    )


def negate_reverse_linear_distribution(
    dist: DenseDiscreteDist,
) -> DenseDiscreteDist:
    """Map X -> -X, reverse PMF order, and swap boundary atoms."""
    n = dist.prob_arr.size
    return DenseDiscreteDist(
        x_0=-(dist.x_0 + dist.step * (n - 1)),
        step=dist.step,
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
    if not isinstance(realization, PLDRealization):
        raise TypeError(f"calc_pld_dual requires PLDRealization, got {type(realization)}")

    dual_probs_aligned = np.zeros_like(realization.prob_arr)
    mask = realization.prob_arr > 0
    dual_probs_aligned[mask] = np.exp(
        np.log(realization.prob_arr[mask]) - realization.x_array[mask]
    )
    dual_probs = np.flip(dual_probs_aligned)

    sum_prob = math.fsum(map(float, dual_probs))
    if sum_prob > 1.0:
        dual_probs *= 1.0 / sum_prob
        sum_prob = 1.0

    return PLDRealization(
        x_0=-(realization.x_0 + realization.step * (realization.prob_arr.size - 1)),
        step=realization.step,
        prob_arr=dual_probs,
        p_max=max(0.0, 1.0 - sum_prob),
        p_min=0.0,
    )


# =============================================================================
# Internal Helper Functions
# =============================================================================


def _ccdf_from_pmf(dist: DiscreteDistBase) -> NDArray[np.float64]:
    """Convert distribution PMF to padded complementary CDF.

    Returns CCDF over [−∞/0, l_0, l_1, ..., +∞]:
      CCDF[i] = P(X > position[i])

    Includes both boundary atoms (p_min at the left, p_max at the right).
    """
    padded_probs = np.concatenate(([dist.p_min], dist.prob_arr, [dist.p_max]))
    return kahan_reverse_exclusive_cumsum(values=padded_probs)
