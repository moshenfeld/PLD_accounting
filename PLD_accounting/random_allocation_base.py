"""One allocation of ``num_steps`` steps, on the grid of a convolution method.

The one-step loss factors become laws of likelihood ratios, the laws are summed with
the requested projection, and the sum becomes the loss of the average ratio on a
linear grid. The flow is written once; ``convolutions.py`` selects the grid of the
laws and of their merges.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from typing import Callable

import numpy as np

from PLD_accounting.convolutions import (
    RatioDist,
    convolve,
    loss_stage_count,
    loss_to_ratio_dist,
    self_convolve,
)
from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    DyadicDist,
    LossFactor,
    PLDRealization,
    SparseDiscreteDist,
    cast_dense_dist,
)
from PLD_accounting.projections import project_discrete_dist
from PLD_accounting.types import (
    BoundType,
    ConvolutionMethod,
    ProjectionMethod,
    require_projection_method,
)
from PLD_accounting.utils import (
    calc_pld_dual,
    log_geometric_to_linear,
    negate_reverse_distribution,
)
from PLD_accounting.validation import (
    require_enum,
    require_nonnegative_real,
    require_positive_int,
    require_positive_real,
)

# =============================================================================
# Public API
# =============================================================================


def allocation_plds_base(
    *,
    base_distributions_creation: Callable[..., tuple[tuple[LossFactor, LossFactor], LossFactor]],
    num_steps: int,
    loss_discretizations: tuple[float, float],
    tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
    convolution_method: ConvolutionMethod,
    max_closing_grid: int,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Build the PLDs ``(remove, add)`` of one allocation of ``num_steps`` steps.

    The callback ``base_distributions_creation`` provides the one-step loss factors
    on the loss points of ``convolution_method``: ``(base, neg_dual_base)`` for remove
    and ``base`` for add. The add PLD comes from the sum of ``num_steps`` copies of the
    reflected add base. Rounding sums ``num_steps - 1`` copies of ``neg_dual_base`` and
    merges the remove base last. Connect-the-dots requires ``BoundType.DOMINATES`` and
    returns the dual of the add PLD on its linear grid as the remove PLD, without
    reading the remove factors.

    Each loss discretization is divided equally among the stages that
    ``loss_discretization_counts`` counts. Half of ``tail_truncation`` bounds the
    one-step input and half is spent after it: by the merges on ``GEOM``, and by the
    projection onto the linear grid on ``MS_FFT``, whose returned grids have at most
    ``max_closing_grid`` points. One step on ``GEOM`` has no later stage: the input
    takes the whole budget and its factors are returned as they are.
    """
    require_positive_int(value=num_steps, name="num_steps")
    require_positive_real(
        value=list(loss_discretizations),
        name=["remove loss_discretization", "add loss_discretization"],
    )
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_projection_method(value=projection_method, bound_type=bound_type)
    connect_the_dots = projection_method == ProjectionMethod.CTD
    counts = loss_discretization_counts(
        num_steps=num_steps,
        projection_method=projection_method,
        convolution_method=convolution_method,
    )
    remove_step, add_step = (
        loss_discretizations[0] / counts[0],
        loss_discretizations[1] / counts[1],
    )
    input_only = num_steps == 1 and convolution_method == ConvolutionMethod.GEOM
    if not input_only:
        tail_truncation /= 2
    (remove_loss, neg_dual_loss), add_loss = base_distributions_creation(
        loss_discretizations=(remove_step, add_step),
        # Each base factor's tail error is amplified by num_steps through the sum.
        tail_truncation=tail_truncation / num_steps,
        bound_type=bound_type,
    )
    if input_only:
        return (
            cast_dense_dist(dist=remove_loss, name="GEOM remove input factor"),
            cast_dense_dist(dist=add_loss, name="GEOM add input factor"),
        )

    neg_add_loss = negate_reverse_distribution(add_loss)
    neg_base = loss_to_ratio_dist(loss_dist=neg_add_loss, convolution_method=convolution_method)
    # A one-factor average is the base. Otherwise the add loss decreases with the
    # sum, so rounding rounds the sum to the other side.
    add_sum = neg_base
    if num_steps > 1:
        add_bound_type = (
            BoundType.DOMINATES
            if connect_the_dots or bound_type == BoundType.IS_DOMINATED
            else BoundType.IS_DOMINATED
        )
        add_sum = self_convolve(
            dist=neg_base,
            num_convolutions=num_steps,
            tail_truncation=tail_truncation,
            bound_type=add_bound_type,
            projection_method=projection_method,
            convolution_method=convolution_method,
        )
    add = _loss_dist_of_average(
        summed=add_sum,
        num_steps=num_steps,
        negated=True,
        loss_discretization=add_step,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        max_closing_grid=max_closing_grid,
    )
    if connect_the_dots and num_steps > 1:
        # The dual is taken on the linear grid, before any composition of a direction.
        add_pld = PLDRealization(
            grid=add.grid, prob_arr=add.prob_arr, p_min=add.p_min, p_max=add.p_max
        )
        return calc_pld_dual(add_pld), add_pld

    remove_sum = loss_to_ratio_dist(loss_dist=remove_loss, convolution_method=convolution_method)
    if num_steps > 1:
        neg_dual_base = loss_to_ratio_dist(
            loss_dist=neg_dual_loss, convolution_method=convolution_method
        )
        # The copies under Q receive (num_steps - 1) / num_steps of the merge tail and
        # the closing merge with the base, the factor under P, the remaining 1 / num_steps.
        neg_dual_sum = self_convolve(
            dist=neg_dual_base,
            num_convolutions=num_steps - 1,
            tail_truncation=tail_truncation * (num_steps - 1) / num_steps,
            bound_type=bound_type,
            projection_method=projection_method,
            convolution_method=convolution_method,
        )
        remove_sum = convolve(
            dist_1=neg_dual_sum,
            dist_2=remove_sum,
            tail_truncation=tail_truncation / num_steps,
            bound_type=bound_type,
            projection_method=projection_method,
            convolution_method=convolution_method,
        )
    remove = _loss_dist_of_average(
        summed=remove_sum,
        num_steps=num_steps,
        negated=False,
        loss_discretization=remove_step,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        max_closing_grid=max_closing_grid,
    )
    return remove, add


def loss_discretization_counts(
    *,
    num_steps: int,
    projection_method: ProjectionMethod,
    convolution_method: ConvolutionMethod,
) -> tuple[int, int]:
    """Return the loss-discretizing stage counts ``(remove, add)`` of one allocation.

    The add direction sums ``num_steps`` copies. So does connect-the-dots for both
    directions. The rounding remove direction sums ``num_steps - 1`` copies and adds
    the merge with the remove base.
    """
    require_positive_int(value=num_steps, name="num_steps")
    require_enum(value=projection_method, enum_cls=ProjectionMethod, name="projection_method")
    add_count = loss_stage_count(num_convolutions=num_steps, convolution_method=convolution_method)
    if num_steps == 1 or projection_method == ProjectionMethod.CTD:
        return add_count, add_count
    copies_count = loss_stage_count(
        num_convolutions=num_steps - 1, convolution_method=convolution_method
    )
    return copies_count + 1, add_count


# =============================================================================
# Helper Functions
# =============================================================================


def _loss_dist_of_average(
    *,
    summed: RatioDist,
    num_steps: int,
    negated: bool,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    max_closing_grid: int,
) -> DenseDiscreteDist:
    """Return the loss of the average of a sum ``S`` of ``num_steps`` ratios, on a linear grid.

    The loss is ``ln(S / num_steps)``, or ``-ln(S / num_steps)`` when ``negated``. The
    logarithm of a geometric grid is linear. A sum on the dyadic grid is truncated by
    ``tail_truncation`` and projected, by ``bound_type``, onto a linear grid of step
    ``loss_discretization``, or of the mean spacing ``log(2) / m`` of the dyadic grid
    when that is coarser, with at most ``max_closing_grid`` points.
    """
    if isinstance(summed, DyadicDist):
        points_loss = _dyadic_loss_of_average(summed=summed, num_steps=num_steps)
        if negated:
            points_loss = negate_reverse_distribution(points_loss)
        step = max(loss_discretization, math.log(2.0) / summed.cells_per_band)
        return _project_onto_linear_grid(
            dist=points_loss,
            loss_discretization=step,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            max_closing_grid=max_closing_grid,
        )
    loss = _geometric_loss_of_average(summed=summed, num_steps=num_steps)
    return negate_reverse_distribution(loss) if negated else loss


def _dyadic_loss_of_average(*, summed: DyadicDist, num_steps: int) -> SparseDiscreteDist:
    """Return the loss ``ln(S / num_steps)`` of a sum of ratios ``S`` on the dyadic grid.

    The masses stay on their points; the atom at zero is the loss ``-inf`` and the
    absorbing mass is ``+inf``. The whole measure is divided by its total mass, which
    the transforms of admitted merges leave within their band from one. A common
    factor moves no first moment between points; the residual placed at one end of
    the support would be a floor on delta there, or would cost the dual of the
    distribution the first moment of that end.
    """
    total_mass = summed.total_mass()
    occupied = np.flatnonzero(summed.masses())
    return SparseDiscreteDist(
        x_array=np.log(summed.support()[occupied]) - math.log(num_steps),
        prob_arr=summed.masses()[occupied] / total_mass,
        p_min=float(summed.p_min) / total_mass,
        p_max=float(summed.p_max) / total_mass,
    )


def _project_onto_linear_grid(
    *,
    dist: SparseDiscreteDist,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    max_closing_grid: int,
) -> DenseDiscreteDist:
    """Truncate a loss distribution and project it onto a linear grid of the requested step."""
    closed = project_discrete_dist(
        dist=dist,
        tail_truncation=tail_truncation,
        loss_discretization=loss_discretization,
        bound_type=bound_type,
        max_grid_points=max_closing_grid,
    )
    if closed.step > loss_discretization:
        warnings.warn(
            f"multiscale allocation: max_closing_grid={max_closing_grid} coarsens the returned "
            f"grid to a step of {closed.step:.6e}, above the requested "
            f"{loss_discretization:.6e}. The bound stays valid and is looser.",
            stacklevel=4,
        )
    return closed


def _geometric_loss_of_average(*, summed: DenseDiscreteDist, num_steps: int) -> DenseDiscreteDist:
    """Return the loss ``ln(S / num_steps)`` of a sum of ratios ``S`` on a geometric grid.

    Composition sums the factors' unit anchors, so the anchor of the sum is the exact
    integer ``num_steps`` and ``x / x`` is exactly one: the averaged lattice is
    ``1 * r**k``, which ``log`` maps to a zero-anchored loss grid.
    """
    anchor = summed.grid.anchor
    if anchor != float(num_steps):
        raise ValueError(
            f"composed exp-space anchor {anchor!r} is not the unit-anchored factor count "
            f"{float(num_steps)!r}; the loss factors were not zero-anchored"
        )
    averaged_grid = replace(summed.grid, anchor=anchor / num_steps)
    averaged = DenseDiscreteDist(
        grid=averaged_grid,
        prob_arr=summed.prob_arr,
        p_min=summed.p_min,
        p_max=summed.p_max,
        domain=summed.domain,
    )
    return log_geometric_to_linear(averaged)
