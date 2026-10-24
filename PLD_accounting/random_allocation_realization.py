"""One-step random-allocation factors from the privacy-loss realization of each direction."""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from PLD_accounting.discrete_dist import LossFactor, PLDRealization, SparseDiscreteDist
from PLD_accounting.distribution_utils import exact_sum, exp_moment_terms
from PLD_accounting.multiscale_fft_convolution import (
    MAX_DYADIC_REACH,
    dyadic_cells_per_band,
    dyadic_range_points,
)
from PLD_accounting.projections import project_discrete_dist, project_discrete_dist_on_points
from PLD_accounting.types import BoundType, ConvolutionMethod, Direction, require_bound_type
from PLD_accounting.utils import (
    calc_pld_dual,
    grid_step_for_point_cap,
    negate_reverse_distribution,
)
from PLD_accounting.validation import (
    require_integer,
    require_nonnegative_real,
    require_positive_real,
)

_Projection = Callable[..., LossFactor]


# =============================================================================
# Public API
# =============================================================================


def realization_base_distributions(
    *,
    remove_realization: PLDRealization,
    add_realization: PLDRealization,
    loss_discretizations: tuple[float, float],
    tail_truncation: float,
    bound_type: BoundType,
    convolution_method: ConvolutionMethod,
    max_grid_mult: int,
    max_cells_per_band: int,
) -> tuple[tuple[LossFactor, LossFactor], LossFactor]:
    """Prepare the one-step factors of both directions from the realization of each direction.

    Algorithms 1 and 2 (`rand-alloc-rem`, `rand-alloc-add`), with the dominating-path
    refinement in the "Connect-the-dots rediscretization" remark following
    Algorithm 6, in Appendix C of https://arxiv.org/abs/2602.17284.

    Budgets are ordered ``(remove, add)``; ``tail_truncation`` is the budget of one
    projection. Returns the remove loss factors ``(base, neg_dual_base)`` and the add
    loss factor ``base`` on the loss points of ``convolution_method``. ``GEOM`` takes
    a fixed-gap grid of at most ``max_grid_mult`` points (``<= 0`` disables the cap),
    as ``DenseDiscreteDist``. ``MS_FFT`` takes the logarithms of the dyadic points of
    a range of likelihood ratios, with at most ``max_cells_per_band`` points per
    octave, negated for the add factor, as ``SparseDiscreteDist``.
    """
    require_bound_type(value=bound_type)
    remove_projection = _loss_projection(
        realization=remove_realization,
        loss_discretization=loss_discretizations[0],
        tail_truncation=tail_truncation,
        convolution_method=convolution_method,
        max_grid_mult=max_grid_mult,
        max_cells_per_band=max_cells_per_band,
        direction=Direction.REMOVE,
    )
    add_projection = _loss_projection(
        realization=add_realization,
        loss_discretization=loss_discretizations[1],
        tail_truncation=tail_truncation,
        convolution_method=convolution_method,
        max_grid_mult=max_grid_mult,
        max_cells_per_band=max_cells_per_band,
        direction=Direction.ADD,
    )

    # The dual can be derived only from a PLD realization. An upper bound discretizes
    # first and takes the exact dual of the discretized base, which avoids a second
    # discretization of the dual and keeps the bound valid. A lower-bound projection
    # is no longer a realization, so its dual is derived first and the two laws are
    # rounded down separately.
    if bound_type == BoundType.DOMINATES:
        base = remove_projection(dist=remove_realization, bound_type=bound_type)
        if not isinstance(base, (PLDRealization, SparseDiscreteDist)):
            raise TypeError(f"dominating projection must return a PLD, got {type(base).__name__}")
        neg_dual_base = negate_reverse_distribution(calc_pld_dual(base))
    else:
        neg_dual_realization = negate_reverse_distribution(calc_pld_dual(remove_realization))
        base = remove_projection(dist=remove_realization, bound_type=bound_type)
        neg_dual_base = remove_projection(dist=neg_dual_realization, bound_type=bound_type)
    add_base = add_projection(dist=add_realization, bound_type=bound_type)
    return (base, neg_dual_base), add_base


# =============================================================================
# Helper Functions
# =============================================================================


def _loss_projection(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    convolution_method: ConvolutionMethod,
    max_grid_mult: int,
    max_cells_per_band: int,
    direction: Direction,
) -> _Projection:
    """Return the projection of one direction onto the loss points of a convolution method.

    ``GEOM`` projects onto a fixed-gap grid. Its step is the loss budget, never finer
    than the grid of the realization (a coarser realization warns), and coarsened
    when that grid would exceed ``max_grid_mult`` points (``<= 0`` disables the cap).
    ``MS_FFT`` projects onto the logarithms of dyadic points, with the points per
    octave that the loss budget gives, up to ``max_cells_per_band``; the realization's
    grid sets no floor there. The add ratio is ``exp(-L)``, so the add
    factor is on the negated points. A lower bound on points lowers the mass at
    ``+inf`` to the last finite loss, so a sum of ratios carries no factor at infinity.
    """
    if convolution_method == ConvolutionMethod.GEOM:
        require_nonnegative_real(value=tail_truncation, name="tail_truncation")
        require_integer(value=max_grid_mult, name="max_grid_mult")
        step = _geometric_step(
            realization=realization,
            loss_discretization=loss_discretization,
            max_grid_mult=max_grid_mult,
            direction=direction,
        )
        return partial(
            project_discrete_dist, tail_truncation=tail_truncation, loss_discretization=step
        )
    if convolution_method == ConvolutionMethod.MS_FFT:
        cells_per_band = dyadic_cells_per_band(
            loss_step=loss_discretization, max_cells_per_band=max_cells_per_band
        )
        loss_points = _dyadic_loss_points(
            realization=realization, cells_per_band=cells_per_band, tail_truncation=tail_truncation
        )
        if direction == Direction.ADD:
            loss_points = -loss_points[::-1]
        return partial(project_discrete_dist_on_points, loss_points=loss_points)
    raise ValueError(f"Unknown ConvolutionMethod: {convolution_method}")


def _geometric_step(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    max_grid_mult: int,
    direction: Direction,
) -> float:
    """Return the fixed-gap step of one realization under the grid cap.

    The step is never finer than the realization's own grid. When that grid is
    coarser than ``loss_discretization``, its step is used and a warning reports it.
    The cap is applied over the realization's support as it is for a continuous law.
    """
    require_positive_real(value=loss_discretization, name="loss_discretization")
    if realization.prob_arr.size < 2:
        raise ValueError(
            "realization must have at least two finite grid points, "
            f"got grid_size={realization.prob_arr.size}"
        )
    if realization.step > loss_discretization:
        warnings.warn(
            f"general allocation: the {direction.value} realization's grid step "
            f"{realization.step:.6e} is coarser than the loss step {loss_discretization:.6e} "
            "requested for its one-step factor, so the realization's step is used. The bound "
            "stays valid and is looser than loss_discretization requests; discretize the "
            f"realization with a step of at most {loss_discretization:.6e} to meet it.",
            # The caller of general_allocation_pld.
            stacklevel=8,
        )
    return grid_step_for_point_cap(
        x_min=realization.x_0,
        x_max=realization.grid.last_point,
        requested_step=max(loss_discretization, realization.step),
        max_grid_points=max_grid_mult,
    )


def _dyadic_loss_points(
    *,
    realization: PLDRealization,
    cells_per_band: int,
    tail_truncation: float,
) -> NDArray[np.float64]:
    """Return the logarithms of the dyadic points of the range that the tail budget allows.

    The range is the smallest ``[2**-r, 2**r]`` outside which each tail of the
    realization and of its dual is at most ``tail_truncation / 4``.
    """
    require_positive_real(value=tail_truncation, name="tail_truncation")
    loss = realization.x_array
    dual_masses = exp_moment_terms(prob_arr=realization.prob_arr, x_vals=loss)
    # The tails shrink as the range grows, so the smallest range that fits is found by
    # bisection, each probe an exact sum of the atoms outside the range.
    low, high = 1, MAX_DYADIC_REACH - 1
    if not _tails_fit(
        realization=realization,
        dual_masses=dual_masses,
        reach=high,
        tail_truncation=tail_truncation,
    ):
        raise ValueError(
            f"tail_truncation={tail_truncation:.3e} per factor needs likelihood ratios beyond "
            f"2**{MAX_DYADIC_REACH}, outside the float range; raise tail_truncation"
        )
    while low < high:
        middle = (low + high) // 2
        if _tails_fit(
            realization=realization,
            dual_masses=dual_masses,
            reach=middle,
            tail_truncation=tail_truncation,
        ):
            high = middle
        else:
            low = middle + 1
    return np.log(dyadic_range_points(reach=low, cells_per_band=cells_per_band))


def _tails_fit(
    *,
    realization: PLDRealization,
    dual_masses: NDArray[np.float64],
    reach: int,
    tail_truncation: float,
) -> bool:
    """Whether the tails of the realization and of its dual beyond ``reach`` octaves fit."""
    loss = realization.x_array
    bound = reach * math.log(2.0)
    below = int(np.searchsorted(loss, -bound, side="left"))
    above = int(np.searchsorted(loss, bound, side="right"))
    largest_tail = max(
        exact_sum(realization.prob_arr[:below]),
        exact_sum(dual_masses[:below]),
        exact_sum(realization.prob_arr[above:]),
        exact_sum(dual_masses[above:]),
    )
    return largest_tail <= tail_truncation / 4
