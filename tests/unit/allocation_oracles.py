"""Reference computations for random-allocation tests.

Exact quantities that the tests of the allocation routes and of the merges compare
against: the enumerated sum of likelihood ratios of a small discrete mechanism with
its privacy profiles, the privacy profile of the Gaussian mechanism, the points of the
dyadic grid, and the projection of explicit atoms onto grid points. Also the
one-step factors of a single direction on the fixed-gap grid, read from the paired
builders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from PLD_accounting.discrete_dist import DenseDiscreteDist, GridSpec, PLDRealization
from PLD_accounting.random_allocation_continuous import continuous_base_distributions
from PLD_accounting.random_allocation_realization import realization_base_distributions
from PLD_accounting.types import BoundType, ConvolutionMethod

# A value this close to an end point is that point evaluated along another float path.
_END_POINT_RTOL = 1e-12


@dataclass(frozen=True, kw_only=True)
class DiscretePair:
    """One step of a discrete mechanism, as the likelihood ratio ``P / Q`` under ``Q``.

    ``ratios`` are the positive finite values of the ratio and ``q_masses`` their
    probabilities under ``Q``. The rest of ``Q`` sits where ``P`` is zero (ratio zero),
    and ``1 - E_Q[ratio]`` of ``P`` sits where ``Q`` is zero.
    """

    ratios: NDArray[np.float64]
    q_masses: NDArray[np.float64]

    @property
    def q_mass_at_zero(self) -> float:
        """Mass of ``Q`` where ``P`` is zero."""
        return 1.0 - math.fsum(self.q_masses.tolist())

    @property
    def p_mass_at_infinity(self) -> float:
        """Mass of ``P`` where ``Q`` is zero."""
        return 1.0 - math.fsum((self.ratios * self.q_masses).tolist())


def lattice_pair(
    *, step: float, indices: NDArray[np.int64], p_masses: NDArray[np.float64]
) -> DiscretePair:
    """Return a pair whose losses ``ln(P / Q)`` are ``indices * step`` with ``P``-masses given.

    The masses may total less than one; the rest of ``P`` is where ``Q`` is zero.
    """
    ratios = np.exp(np.asarray(indices, dtype=np.float64) * step)
    q_masses = np.asarray(p_masses, dtype=np.float64) / ratios
    if math.fsum(q_masses.tolist()) > 1.0:
        raise ValueError("the P-masses do not define a likelihood ratio under Q")
    return DiscretePair(ratios=ratios, q_masses=q_masses)


def remove_realization(*, pair: DiscretePair, step: float) -> PLDRealization:
    """Return the remove-direction PLD of ``pair``: the loss ``ln(P / Q)`` under ``P``."""
    return _lattice_realization(
        losses=np.log(pair.ratios),
        masses=pair.ratios * pair.q_masses,
        mass_at_infinity=pair.p_mass_at_infinity,
        step=step,
    )


def add_realization(*, pair: DiscretePair, step: float) -> PLDRealization:
    """Return the add-direction PLD of ``pair``: the loss ``ln(Q / P)`` under ``Q``."""
    losses = -np.log(pair.ratios)
    return _lattice_realization(
        losses=losses,
        masses=pair.q_masses,
        mass_at_infinity=pair.q_mass_at_zero,
        step=step,
    )


def enumerate_ratio_sum(
    *, pair: DiscretePair, num_steps: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return every value of the sum of ``num_steps`` ratios under ``Q`` with its mass.

    The atom at ratio zero takes part, so the masses total one. Values are not merged.
    """
    ratios = np.concatenate(([0.0], pair.ratios))
    masses = np.concatenate(([pair.q_mass_at_zero], pair.q_masses))
    values, weights = np.zeros(1), np.ones(1)
    for _ in range(num_steps):
        values = np.add.outer(values, ratios).ravel()
        weights = np.multiply.outer(weights, masses).ravel()
    return values, weights


def exact_remove_delta(*, pair: DiscretePair, num_steps: int, epsilon: float) -> float:
    """Return the remove-direction ``delta(epsilon)`` of ``num_steps`` steps of allocation.

    With ``Z`` the average ratio under ``Q``, this is ``E[(Z - exp(epsilon))_+]`` plus
    the mass of ``P`` where ``Q`` is zero.
    """
    values, weights = enumerate_ratio_sum(pair=pair, num_steps=num_steps)
    excess = np.maximum(values / num_steps - math.exp(epsilon), 0.0)
    return math.fsum((weights * excess).tolist()) + pair.p_mass_at_infinity


def exact_add_delta(*, pair: DiscretePair, num_steps: int, epsilon: float) -> float:
    """Return the add-direction ``delta(epsilon)``: ``E[(1 - exp(epsilon) Z)_+]`` under ``Q``."""
    values, weights = enumerate_ratio_sum(pair=pair, num_steps=num_steps)
    excess = np.maximum(1.0 - math.exp(epsilon) * values / num_steps, 0.0)
    return math.fsum((weights * excess).tolist())


def gaussian_delta(*, sigma: float, epsilon: float) -> float:
    """Return ``delta(epsilon)`` of the Gaussian mechanism with sensitivity one."""
    mean = 1.0 / (2.0 * sigma**2)
    upper = stats.norm.sf((epsilon - mean) * sigma)
    lower = stats.norm.cdf(-(epsilon + mean) * sigma)
    return float(upper - math.exp(epsilon) * lower)


def dyadic_points(*, first_band: int, num_bands: int, cells_per_band: int) -> NDArray[np.float64]:
    """Return the dyadic grid points of ``num_bands`` octaves from ``2**first_band``, and the next.

    Octave ``j`` holds ``2**j * (1 + i / cells_per_band)`` for ``i`` below
    ``cells_per_band``. The first point of the octave after the last closes the range.
    """
    octaves = 2.0 ** np.arange(first_band, first_band + num_bands, dtype=np.float64)
    cells = 1.0 + np.arange(cells_per_band, dtype=np.float64) / cells_per_band
    return np.append(np.multiply.outer(octaves, cells).ravel(), 2.0 ** (first_band + num_bands))


def project_onto_points(
    *,
    values: NDArray[np.float64],
    masses: NDArray[np.float64],
    points: NDArray[np.float64],
    mode: str,
) -> NDArray[np.float64]:
    """Project atoms onto increasing ``points`` and return the mass at each point.

    ``mode`` is ``"up"`` or ``"down"`` for the nearest point on that side, or
    ``"split"`` for the two neighbours with the weights that keep the mean. Every
    value must lie between the first and the last point; one a rounding error outside
    counts as the end point.
    """
    points = np.asarray(points, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if np.any(values < points[0] * (1.0 - _END_POINT_RTOL)) or np.any(
        values > points[-1] * (1.0 + _END_POINT_RTOL)
    ):
        raise ValueError("a value lies outside the points")
    values = np.clip(values, points[0], points[-1])
    out = np.zeros(points.size)
    below = np.clip(np.searchsorted(points, values, side="right") - 1, 0, points.size - 2)
    width = points[below + 1] - points[below]
    if mode == "split":
        share = (values - points[below]) / width
    elif mode == "up":
        share = (values > points[below]).astype(np.float64)
    elif mode == "down":
        share = (values >= points[below + 1]).astype(np.float64)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    np.add.at(out, below, masses * (1.0 - share))
    np.add.at(out, below + 1, masses * share)
    return out


def _lattice_realization(
    *,
    losses: NDArray[np.float64],
    masses: NDArray[np.float64],
    mass_at_infinity: float,
    step: float,
) -> PLDRealization:
    """Put atoms at integer multiples of ``step`` on the dense grid that spans them."""
    indices = np.rint(losses / step).astype(np.int64)
    if not np.allclose(indices * step, losses, rtol=0.0, atol=1e-12):
        raise ValueError("the losses are not multiples of the step")
    first = int(indices.min())
    prob_arr = np.zeros(int(indices.max()) - first + 1)
    np.add.at(prob_arr, indices - first, masses)
    residual = 1.0 - math.fsum(prob_arr.tolist()) - mass_at_infinity
    prob_arr[int(np.argmax(prob_arr))] += residual
    grid = GridSpec(step=step, n=prob_arr.size, index_0=first)
    return PLDRealization(grid=grid, prob_arr=prob_arr, p_max=max(0.0, mass_at_infinity))


def continuous_remove_base_distributions(  # pylint: disable=too-many-arguments
    *,
    law,
    dual,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    max_grid_mult: int,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Remove factors of a law and its dual on the fixed-gap grid."""
    return continuous_base_distributions(
        law=law,
        dual=dual,
        loss_discretizations=(loss_discretization, loss_discretization),
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolution_method=ConvolutionMethod.GEOM,
        max_grid_mult=max_grid_mult,
        max_cells_per_band=1,
    )[0]


def continuous_add_base_distribution(  # pylint: disable=too-many-arguments
    *,
    law,
    dual,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    max_grid_mult: int,
) -> DenseDiscreteDist:
    """Add factor of an add law and its dual on the fixed-gap grid."""
    return continuous_base_distributions(
        law=dual,
        dual=law,
        loss_discretizations=(loss_discretization, loss_discretization),
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolution_method=ConvolutionMethod.GEOM,
        max_grid_mult=max_grid_mult,
        max_cells_per_band=1,
    )[1]


def realization_remove_base_distributions(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    max_grid_mult: int,
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Remove factors of one realization on the fixed-gap grid."""
    return realization_base_distributions(
        remove_realization=realization,
        add_realization=realization,
        loss_discretizations=(loss_discretization, loss_discretization),
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolution_method=ConvolutionMethod.GEOM,
        max_grid_mult=max_grid_mult,
        max_cells_per_band=1,
    )[0]


def realization_add_base_distribution(
    *,
    realization: PLDRealization,
    loss_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
    max_grid_mult: int,
) -> DenseDiscreteDist:
    """Add factor of one realization on the fixed-gap grid."""
    return realization_base_distributions(
        remove_realization=realization,
        add_realization=realization,
        loss_discretizations=(loss_discretization, loss_discretization),
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolution_method=ConvolutionMethod.GEOM,
        max_grid_mult=max_grid_mult,
        max_cells_per_band=1,
    )[1]
