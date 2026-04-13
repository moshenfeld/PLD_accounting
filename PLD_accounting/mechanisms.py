"""Mechanism PLDs — Gaussian and Laplace privacy-loss discretizations.

Discretizes the REMOVE-direction (one record removed) continuous PLD onto a
linear grid. With ``bound_type=DOMINATES`` (default), returns a validated
:class:`PLDRealization`. With ``bound_type=IS_DOMINATED``, returns a plain
:class:`DenseDiscreteDist` (no PLD validation — the lower-bound grid can have
mass at negative infinity loss).

For Laplace with sampling probability 1, ADD and REMOVE PLDs coincide; callers
may use ``dist.copy()`` on a :class:`PLDRealization` for the ADD slot when a
second object is required.

Sensitivity is fixed at 1 (L2 for Gaussian, L1 for Laplace).

Gaussian privacy-loss parameterization (remove): if noise is
``N(0, sigma^2)`` and L2 sensitivity is 1, then the PLD is ``N(mu, sd^2)`` with
``mu = 1 / (2*sigma^2)``, ``sd = 1/sigma``.

Laplace (remove): noise ``Laplace(0, sigma)`` with L1 sensitivity 1 gives a
bounded mixed PLD on ``[-lam, lam]`` with ``lam = 1/sigma`` (see ``LaplacePLD``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from PLD_accounting.discrete_dist import DenseDiscreteDist, PLDRealization
from PLD_accounting.distribution_discretization import (
    discretize_continuous_distribution,
)
from PLD_accounting.distribution_utils import MIN_GRID_SIZE
from PLD_accounting.types import (
    DEFAULT_LOSS_DISCRETIZATION,
    DEFAULT_TAIL_TRUNCATION,
    BoundType,
    SpacingType,
)
from PLD_accounting.validation import validate_bound_type


def gaussian_distribution(
    scale: float,
    value_discretization: float = DEFAULT_LOSS_DISCRETIZATION,
    tail_truncation: float = DEFAULT_TAIL_TRUNCATION,
    bound_type: BoundType = BoundType.DOMINATES,
) -> PLDRealization | DenseDiscreteDist:
    """Discretized Gaussian mechanism PLD (L2 sensitivity 1) on a linear grid.

    Args:
        scale: Noise standard deviation of the Gaussian mechanism.
        value_discretization: Target step size for the linear grid.
        tail_truncation: Tail probability budget; quantiles ``ppf``/``isf`` at
            this level define the grid range.
        bound_type: Rounding semantics for mapping continuous mass to grid points.
            ``DOMINATES`` for upper bound and ``IS_DOMINATED`` for lower

    Returns:
        ``PLDRealization`` when ``bound_type`` is ``DOMINATES``,
        ``DenseDiscreteDist`` when it is ``IS_DOMINATED``.
    """
    validate_bound_type(bound_type)
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    scale_f = float(scale)
    mu = 1.0 / (2.0 * scale_f**2)
    sd = 1.0 / scale_f
    dist = stats.norm(loc=mu, scale=sd)
    return _continuous_mechanism_distribution(
        dist=dist,
        value_discretization=value_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )


def laplace_distribution(
    scale: float,
    value_discretization: float = DEFAULT_LOSS_DISCRETIZATION,
    tail_truncation: float = DEFAULT_TAIL_TRUNCATION,
    bound_type: BoundType = BoundType.DOMINATES,
) -> PLDRealization | DenseDiscreteDist:
    """Discretized Laplace mechanism PLD (L1 sensitivity 1) on a linear grid.

    Args:
        scale: Laplace noise scale
        value_discretization: Target step size for the linear grid.
        tail_truncation: Tail probability budget; quantiles ``ppf``/``isf`` at
            this level define the grid range.
        bound_type: Rounding semantics for mapping continuous mass to grid points.
            ``DOMINATES`` for upper bound and ``IS_DOMINATED`` for lower

    Returns:
        ``PLDRealization`` when ``bound_type`` is ``DOMINATES``,
        ``DenseDiscreteDist`` when it is ``IS_DOMINATED``.
    """
    validate_bound_type(bound_type)
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    dist = LaplacePLD(sigma=scale)
    return _continuous_mechanism_distribution(
        dist=dist,
        value_discretization=value_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )


class LaplacePLD(stats.rv_continuous):
    """Exact PLD for the Laplace mechanism (remove direction).

    ``M(D) = f(D) + Laplace(0, sigma)`` with L1 sensitivity 1;
    ``lam = 1/sigma`` is the maximum finite privacy loss.

    Mixed distribution with:
      - Atom at ``-lam`` with mass ``0.5 * exp(-lam)``
      - Continuous density ``0.25 * exp((x - lam) / 2)`` on ``(-lam, lam)``
      - Atom at ``+lam`` with mass ``0.5``

    Note: ``.pdf()`` returns only the continuous part; atoms are not included in
    the density. CDF, PPF, and RVS are exact for the full distribution.
    """

    def __init__(self, sigma: float, name: str = "laplace_pld") -> None:
        """Initialize the Laplace-mechanism PLD law with noise scale ``sigma``."""
        self.sigma = float(sigma)
        self.lam = 1.0 / self.sigma
        super().__init__(a=-self.lam, b=self.lam, name=name)

    @property
    def atoms(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(points, probabilities)`` for the two boundary atoms."""
        lam = self.lam
        points = np.array([-lam, lam])
        probs = np.array([0.5 * np.exp(-lam), 0.5])
        return points, probs

    def _pdf(self, x: NDArray[np.floating[Any]], *args: Any) -> NDArray[np.float64]:
        del args
        x_arr = np.asarray(x)
        lam = self.lam
        out = np.zeros_like(x_arr, dtype=float)
        mask = (-lam < x_arr) & (x_arr < lam)
        out[mask] = 0.25 * np.exp((x_arr[mask] - lam) / 2.0)
        return out

    def _cdf(self, x: NDArray[np.floating[Any]], *args: Any) -> NDArray[np.float64]:
        del args
        x_arr = np.asarray(x)
        lam = self.lam
        out = np.zeros_like(x_arr, dtype=float)
        mid = (-lam <= x_arr) & (x_arr < lam)
        out[mid] = 0.5 * np.exp((x_arr[mid] - lam) / 2.0)
        out[x_arr >= lam] = 1.0
        return out

    def _ppf(self, q: NDArray[np.floating[Any]], *args: Any) -> NDArray[np.float64]:
        del args
        q_arr = np.asarray(q)
        lam = self.lam
        q_left = 0.5 * np.exp(-lam)
        out = np.empty_like(q_arr, dtype=float)
        left = q_arr <= q_left
        mid = (q_arr > q_left) & (q_arr <= 0.5)
        right = q_arr > 0.5
        out[left] = -lam
        out[mid] = lam + 2.0 * np.log(2.0 * q_arr[mid])
        out[right] = lam
        return out

    def _rvs(self, size: Any = None, random_state: Any = None) -> Any:
        if random_state is None:
            raise ValueError("random_state is required for LaplacePLD._rvs")
        u = random_state.uniform(size=size)
        return self._ppf(u)

    def _stats(self, *args: Any, **kwds: Any) -> tuple[Any, Any, None, None]:
        del args, kwds
        lam = self.lam
        mean = lam - 1.0 + np.exp(-lam)
        var = 3.0 - (4.0 * lam + 2.0) * np.exp(-lam) - np.exp(-2.0 * lam)
        return mean, var, None, None


def _continuous_mechanism_distribution(
    *,
    dist: stats.rv_continuous,
    value_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> PLDRealization | DenseDiscreteDist:
    """Discretize a continuous privacy-loss law onto a uniform linear grid.

    Wraps in `PLDRealization` only when ``bound_type`` is ``DOMINATES``.
    """
    x_min = float(dist.ppf(tail_truncation))
    x_max = float(dist.isf(tail_truncation))
    n_grid = max(int(np.ceil((x_max - x_min) / value_discretization)) + 1, MIN_GRID_SIZE)
    linear_dist = discretize_continuous_distribution(
        dist=dist,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        spacing_type=SpacingType.LINEAR,
        n_grid=n_grid,
        align_to_multiples=True,
    )
    if not (
        isinstance(linear_dist, DenseDiscreteDist)
        and linear_dist.spacing_type == SpacingType.LINEAR
    ):
        raise TypeError(
            f"linear mechanism discretization expected DenseDiscreteDist, got {type(linear_dist)}"
        )
    if bound_type == BoundType.IS_DOMINATED:
        return linear_dist
    return PLDRealization.from_linear_dist(linear_dist)
