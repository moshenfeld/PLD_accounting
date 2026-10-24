"""Distribution-level operations shared by the convolution and allocation routes.

The aligned linear grid of a range, boundary-mass algebra for sums, binary
self-composition, the exp/log and reflection transforms, and the canonical PLD
dual.

Representation metadata stays with the distributions throughout these operations.
``calc_pld_dual`` is the dualization boundary: the one place where a
reciprocal-moment deficit becomes mass at ``+inf``.
"""

from __future__ import annotations

from typing import Callable, TypeVar, overload

import numpy as np

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    DiscreteDistBase,
    Domain,
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
    require_geometric_positives_dist,
    require_linear_reals_dist,
)
from PLD_accounting.distribution_utils import (
    exp_moment_terms,
    signed_unit_residual,
    trim_mass_from_edge,
)
from PLD_accounting.types import MIN_GRID_POINT_CAP, BoundType, require_bound_type
from PLD_accounting.validation import (
    require_integer,
    require_nonnegative_real,
    require_positive_int,
    require_positive_real,
)

_Dist = TypeVar("_Dist")

# =============================================================================
# Grid Construction
# =============================================================================


def aligned_grid_params(
    *,
    x_min: float,
    x_max: float,
    discretization: float,
) -> GridSpec:
    """Return a linear grid on multiples of ``discretization`` that covers ``[x_min, x_max]``."""
    if x_max <= x_min:
        raise ValueError(f"x_max must be greater than x_min, got x_min={x_min}, x_max={x_max}")
    require_positive_real(value=discretization, name="discretization")

    k_lo = int(np.floor(x_min / discretization))
    k_hi = int(np.ceil(x_max / discretization))
    if discretization * k_lo > x_min:
        k_lo -= 1
    if discretization * k_hi < x_max:
        k_hi += 1

    grid = GridSpec(step=discretization, n=k_hi - k_lo + 1, anchor=0.0, index_0=k_lo)
    candidate = grid
    while candidate.last_point < x_max:
        candidate = candidate.with_n(candidate.n + 1)
    return candidate


def grid_step_for_point_cap(
    *,
    x_min: float,
    x_max: float,
    requested_step: float,
    max_grid_points: int,
) -> float:
    """Return ``requested_step``, increased when needed for an aligned grid to fit a point cap.

    A non-positive ``max_grid_points`` means no cap. A cap is at least
    ``MIN_GRID_POINT_CAP``: the grid of ``aligned_grid_params`` over one interval of
    the step can take that many points.
    """
    require_positive_real(value=requested_step, name="requested_step")
    require_integer(value=max_grid_points, name="max_grid_points")
    if max_grid_points <= 0:
        return requested_step
    if max_grid_points < MIN_GRID_POINT_CAP:
        raise ValueError(
            f"max_grid_points must be >= {MIN_GRID_POINT_CAP}, got {max_grid_points!r}"
        )

    # Reserve one fencepost plus up to two bins for outward grid alignment.
    usable_intervals = max_grid_points - 3
    return max(requested_step, (x_max - x_min) / usable_intervals)


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
    dist: _Dist,
    num_convolutions: int,
    tail_truncation: float,
    bound_type: BoundType,
    convolve: Callable[..., _Dist],
) -> _Dist:
    """Sum copies by repeated squaring with a representation-preserving merge.

    Algorithm 3 (`self-conv`), in Appendix C of
    https://arxiv.org/abs/2602.17284.

    ``convolve`` takes ``dist_1``, ``dist_2``, ``tail_truncation`` and
    ``bound_type`` as keywords and validates the representation of its operands.
    The caller validates ``dist`` before this loop; one copy returns it unchanged
    without invoking the merge.
    Powers of two are accumulated from smallest to largest. The result retains
    the representation of ``dist``.

    ``tail_truncation`` is divided over the merges and passed to each as a
    keyword. Use zero for untrimmed merges. The supplied merge owns projection,
    truncation and any per-merge admission. This loop performs no admission of
    the completed sum.
    """
    require_positive_int(value=num_convolutions, name="num_convolutions")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_bound_type(value=bound_type)
    if num_convolutions == 1:
        return dist

    base_dist = dist
    acc_dist = None
    # Squaring and accumulation each spend less than twice their largest charge.
    # Divide by four so both families together stay within the tail budget.
    tail_truncation /= 4
    while num_convolutions > 0:
        if num_convolutions & 1:
            if acc_dist is None:
                acc_dist = base_dist
            else:
                acc_dist = convolve(
                    dist_1=acc_dist,
                    dist_2=base_dist,
                    bound_type=bound_type,
                    tail_truncation=tail_truncation / num_convolutions,
                )
        num_convolutions >>= 1
        if num_convolutions > 0:
            base_dist = convolve(
                dist_1=base_dist,
                dist_2=base_dist,
                bound_type=bound_type,
                tail_truncation=tail_truncation / num_convolutions,
            )
    return acc_dist if acc_dist is not None else base_dist


def binary_self_convolve_depth(num_convolutions: int) -> int:
    """Return the largest number of merges above one copy in ``binary_self_convolve``.

    Squaring gives the sum of ``2**b`` copies the depth ``b``, and an accumulation is
    one merge above the deeper of its operands, which is the power just added. The
    sum of ``num_convolutions`` copies thus has the depth ``ceil(log2(num_convolutions))``,
    against up to ``2 * ceil(log2(num_convolutions))`` merges. A merge that projects onto
    a grid of consecutive ratio at most ``r`` scales a sum by at most ``r``, and adding
    values preserves a common factor, so the errors of the merges accumulate along
    this depth and not over all of the merges.
    """
    require_positive_int(value=num_convolutions, name="num_convolutions")
    return (num_convolutions - 1).bit_length()


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


@overload
def negate_reverse_distribution(dist: SparseDiscreteDist) -> SparseDiscreteDist:
    ...


@overload
def negate_reverse_distribution(dist: DenseDiscreteDist) -> DenseDiscreteDist:
    ...


def negate_reverse_distribution(
    dist: DenseDiscreteDist | SparseDiscreteDist,
) -> DenseDiscreteDist | SparseDiscreteDist:
    """Map X -> -X, reverse PMF order, and swap boundary atoms.

    ``dist`` is on a linear grid or on explicit points, over the reals; the result
    has the representation of ``dist``.
    """
    if isinstance(dist, SparseDiscreteDist):
        if dist.domain != Domain.REALS:
            raise ValueError(f"dist must be over the reals, got {dist.domain}")
        return SparseDiscreteDist(
            x_array=-dist.x_array[::-1],
            prob_arr=dist.prob_arr[::-1],
            p_min=dist.p_max,
            p_max=dist.p_min,
        )
    require_linear_reals_dist(dist=dist, name="dist")
    return DenseDiscreteDist(
        grid=dist.grid.reflect(),
        prob_arr=np.flip(dist.prob_arr),
        p_min=dist.p_max,
        p_max=dist.p_min,
    )


@overload
def calc_pld_dual(dist: PLDRealization) -> PLDRealization:
    ...


@overload
def calc_pld_dual(dist: SparseDiscreteDist) -> SparseDiscreteDist:
    ...


def calc_pld_dual(
    dist: PLDRealization | SparseDiscreteDist,
) -> PLDRealization | SparseDiscreteDist:
    """Compute the paper PLD dual ``D(L)`` (Definition 3.1).

    Algorithm 7 (`PLD-dual`), in Appendix C of
    https://arxiv.org/abs/2602.17284.

    For a PLD ``L`` with support ``l`` and mass ``f_L(l)``, the dual has:
    - finite mass ``f_D(-l) = f_L(l) * exp(-l)``,
    - support reflected to ``-l``,
    - residual mass at ``+inf``.

    ``dist`` is a realization, or a loss distribution on explicit points with no
    mass at ``-inf``; the dual has the representation of ``dist``.
    """
    if isinstance(dist, SparseDiscreteDist):
        if dist.domain != Domain.REALS or dist.p_min != 0.0:
            raise ValueError(
                "dist must be over the reals with no mass at -inf, "
                f"got {dist.domain} and p_min={dist.p_min:.12g}"
            )
        dual_loss = -dist.x_array[::-1]
    else:
        dual_grid = dist.grid.reflect()
        dual_loss = dual_grid.materialize()
    dual_probs = exp_moment_terms(
        prob_arr=np.flip(dist.prob_arr),
        x_vals=-dual_loss,
    )

    dual_residual = signed_unit_residual(values=dual_probs, lower_term=0.0, upper_term=0.0)
    if dual_residual < 0.0:
        # Input may sit in (1, 1 + REALIZATION_MOMENT_TOL]; drop that surplus from the
        # low-loss edge of the dual, where it is cheapest, so the dual has unit mass.
        # Do not classify_residual: its middle band would warn on that admitted excess.
        dual_probs = trim_mass_from_edge(
            prob_arr=dual_probs,
            mass=-dual_residual,
            from_left=True,
        )
        dual_residual = signed_unit_residual(values=dual_probs, lower_term=0.0, upper_term=0.0)

    dual_p_max = max(0.0, dual_residual)
    if isinstance(dist, SparseDiscreteDist):
        return SparseDiscreteDist(
            x_array=dual_loss, prob_arr=dual_probs, p_min=0.0, p_max=dual_p_max
        )
    return PLDRealization(grid=dual_grid, prob_arr=dual_probs, p_max=dual_p_max, p_min=0.0)
