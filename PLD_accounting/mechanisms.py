"""Mechanism PLDs for Gaussian, Laplace, and integer-valued count noise.

Discretizes the REMOVE-direction (one record removed) continuous PLD onto a
linear grid. With ``bound_type=DOMINATES`` (default), returns a validated
:class:`PLDRealization`. With ``bound_type=IS_DOMINATED``, returns a plain
:class:`DenseDiscreteDist` (no PLD validation — the lower-bound grid can have
mass at negative infinity loss).

For Laplace with sampling probability 1, ADD and REMOVE PLDs coincide; callers
may use ``copy.deepcopy(dist)`` on a :class:`PLDRealization` for the ADD slot
when a second object is required.

The Gaussian and Laplace helpers use sensitivity 1 (L2 and L1, respectively).

Gaussian privacy-loss parameterization (remove): if noise is
``N(0, sigma^2)`` and L2 sensitivity is 1, then the PLD is ``N(mu, sd^2)`` with
``mu = 1 / (2*sigma^2)``, ``sd = 1/sigma``.

Laplace (remove): noise ``Laplace(0, sigma)`` with L1 sensitivity 1 gives a
bounded mixed PLD on ``[-lam, lam]`` with ``lam = 1/sigma``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_discretization import (
    discretize_continuous_ctd,
    discretize_continuous_stoch_dom,
    rediscretize_dist_by_bound,
)
from PLD_accounting.types import (
    DEFAULT_LOSS_DISCRETIZATION,
    DEFAULT_TAIL_TRUNCATION,
    BoundType,
    SpacingType,
    require_bound_type,
)
from PLD_accounting.validation import (
    require_nonnegative_real,
    require_open_unit_interval,
    require_positive_int,
    require_positive_real,
)

# =============================================================================
# Public mechanism factories
# =============================================================================


def gaussian_distribution(
    *,
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
            ``DOMINATES`` for an upper bound and ``IS_DOMINATED`` for a lower bound.

    Returns:
        ``PLDRealization`` when ``bound_type`` is ``DOMINATES``,
        ``DenseDiscreteDist`` when it is ``IS_DOMINATED``.
    """
    require_bound_type(value=bound_type)
    require_positive_real(
        value=[scale, value_discretization], name=["scale", "value_discretization"]
    )
    require_open_unit_interval(value=tail_truncation, name="tail_truncation")
    scale_f = float(scale)
    mu = 1.0 / (2.0 * scale_f**2)
    sd = 1.0 / scale_f
    dist = stats.norm(loc=mu, scale=sd)
    return _continuous_mechanism_distribution(
        dist=dist,
        dual_dist=dist,
        value_discretization=value_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )


def laplace_distribution(
    *,
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
            ``DOMINATES`` for an upper bound and ``IS_DOMINATED`` for a lower bound.

    Returns:
        ``PLDRealization`` when ``bound_type`` is ``DOMINATES``,
        ``DenseDiscreteDist`` when it is ``IS_DOMINATED``.
    """
    require_bound_type(value=bound_type)
    require_positive_real(
        value=[scale, value_discretization], name=["scale", "value_discretization"]
    )
    require_open_unit_interval(value=tail_truncation, name="tail_truncation")
    dist = _LaplacePLD(sigma=scale)
    return _continuous_mechanism_distribution(
        dist=dist,
        dual_dist=dist,
        value_discretization=value_discretization,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
    )


def discrete_distribution(
    *,
    noise_dist: DenseDiscreteDist,
    loss_discretization: float,
    tail_truncation: float,
    sensitivity: int = 1,
) -> tuple[PLDRealization, PLDRealization]:
    """Build dominating directional PLDs for additive integer count noise.

    Models releasing ``true_count + R`` for an integer-sensitivity-``s`` count query,
    where ``noise_dist`` describes the integer-valued additive noise ``R`` on a
    unit-spaced linear grid. Loss atoms are log-ratios of probabilities separated
    by ``sensitivity``. Finite noise atoms without shifted support are routed to
    positive-infinity privacy loss. The input ``p_min`` and ``p_max`` masses are
    unrelated omitted noise tails; neither has a known shifted counterpart, so
    both are also routed in full to positive-infinity privacy loss. The exact
    sparse loss distributions are then truncated and projected onto the
    requested linear loss grid by the shared rediscretization implementation.

    Args:
        noise_dist: Finite approximation of the additive-noise distribution. It
            must be a real-domain, unit-spaced ``DenseDiscreteDist`` with an
            integer grid origin. Its distinct ``p_min`` and ``p_max`` masses are
            each treated as unmatched tail mass and mapped to positive-infinity
            privacy loss.
        loss_discretization: Target fixed spacing for the output loss grid.
        tail_truncation: Tail probability budget passed to the shared
            rediscretization implementation.
        sensitivity: Positive integer query sensitivity.

    Returns:
        The ``(remove, add)`` PLDRealization pair.
    """
    require_positive_real(value=loss_discretization, name="loss_discretization")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_positive_int(value=sensitivity, name="sensitivity")
    if not isinstance(noise_dist, DenseDiscreteDist) or isinstance(noise_dist, PLDRealization):
        raise TypeError("noise_dist must be a DenseDiscreteDist, not a PLDRealization")
    if noise_dist.spacing_type != SpacingType.LINEAR or noise_dist.domain != Domain.REALS:
        raise ValueError("noise_dist must use a real-domain linear grid")
    if noise_dist.step != 1.0 or not float(noise_dist.x_0).is_integer():
        raise ValueError(
            "noise_dist must have unit spacing and an integer grid origin, got "
            f"x_0={noise_dist.x_0}, step={noise_dist.step}"
        )

    # REMOVE: atom o (mass C(o)) read against C(o+s).
    remove_dist = _count_noise_direction_pld(
        noise_dist=noise_dist,
        denominator_offset=sensitivity,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
    )

    # ADD: atom o (mass C(o)) read against C(o-s).
    add_dist = _count_noise_direction_pld(
        noise_dist=noise_dist,
        denominator_offset=-sensitivity,
        loss_discretization=loss_discretization,
        tail_truncation=tail_truncation,
    )
    return remove_dist, add_dist


# =============================================================================
# Internal continuous-mechanism adapters
# =============================================================================


class _LaplacePLD(stats.rv_continuous):
    """Exact PLD for the Laplace mechanism (remove direction).

    ``M(D) = f(D) + Laplace(0, sigma)`` with L1 sensitivity 1;
    ``lam = 1/sigma`` is the maximum finite privacy loss.

    Mixed distribution with:
      - Atom at ``-lam`` with mass ``0.5 * exp(-lam)``
      - Continuous density ``0.25 * exp((x - lam) / 2)`` on ``(-lam, lam)``
      - Atom at ``+lam`` with mass ``0.5``

    This is an internal adapter for continuous-discretization operations, not a
    general-purpose SciPy distribution.
    """

    def __init__(self, sigma: float) -> None:
        """Initialize the Laplace-mechanism PLD law with noise scale ``sigma``."""
        self.lam = 1.0 / float(sigma)
        super().__init__(a=-self.lam, b=self.lam, name="laplace_pld")

    def _cdf(self, x: Any, *args: Any) -> Any:
        del args
        x_arr = np.asarray(x)
        lam = self.lam
        out = np.zeros_like(x_arr, dtype=float)
        mid = (-lam <= x_arr) & (x_arr < lam)
        out[mid] = 0.5 * np.exp((x_arr[mid] - lam) / 2.0)
        out[x_arr >= lam] = 1.0
        return out

    def cdf(self, x: Any, *args: Any, **kwds: Any) -> Any:
        """Return the full mixed CDF, including atoms at equality."""
        del args, kwds
        return self._cdf(x)

    def sf(self, x: Any, *args: Any, **kwds: Any) -> Any:
        """Return ``Pr[L > x]``, excluding an atom at equality."""
        del args, kwds
        return 1.0 - self._cdf(x)

    def logcdf(self, x: Any, *args: Any, **kwds: Any) -> Any:
        """Return the logarithm of the full mixed CDF."""
        del args, kwds
        with np.errstate(divide="ignore"):
            return np.log(self._cdf(x))

    def logsf(self, x: Any, *args: Any, **kwds: Any) -> Any:
        """Return the stable logarithm of ``Pr[L > x]``."""
        del args, kwds
        with np.errstate(divide="ignore"):
            return np.log1p(-self._cdf(x))

    def _ppf(self, q: Any, *args: Any) -> Any:
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


def _continuous_mechanism_distribution(
    *,
    dist: stats.rv_continuous | rv_frozen[Any, Any],
    dual_dist: stats.rv_continuous | rv_frozen[Any, Any],
    value_discretization: float,
    tail_truncation: float,
    bound_type: BoundType,
) -> PLDRealization | DenseDiscreteDist:
    """Discretize a continuous privacy-loss law onto a uniform linear grid.

    Upper bounds use CtD and lower bounds use stochastic domination.
    """
    linear_dist: DenseDiscreteDist
    if bound_type == BoundType.DOMINATES:
        linear_dist = discretize_continuous_ctd(
            dist=dist,
            dual_dist=dual_dist,
            tail_truncation=tail_truncation,
            step=value_discretization,
            align_to_multiples=True,
        )
    else:
        linear_dist = discretize_continuous_stoch_dom(
            dist=dist,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            step=value_discretization,
            align_to_multiples=True,
        )
    if not (
        isinstance(linear_dist, DenseDiscreteDist)
        and linear_dist.spacing_type == SpacingType.LINEAR
    ):
        raise TypeError(
            f"linear mechanism discretization expected DenseDiscreteDist, got {type(linear_dist)}"
        )
    return linear_dist


# =============================================================================
# Internal discrete integer-noise adapters
# =============================================================================


def _count_noise_direction_pld(
    *,
    noise_dist: DenseDiscreteDist,
    denominator_offset: int,
    loss_discretization: float,
    tail_truncation: float,
) -> PLDRealization:
    """Build and rediscretize one directional additive-noise PLD.

    ``noise_dist.p_min`` and ``noise_dist.p_max`` represent separate unmatched
    tails of the input noise law. Regardless of direction, all mass from both
    tails is assigned to positive-infinity privacy loss. Finite numerator atoms
    whose shifted denominator is zero are assigned there as well.
    """
    probs = noise_dist.prob_arr
    shift = abs(denominator_offset)
    shifted = np.zeros_like(probs)
    if shift < probs.size:
        if denominator_offset > 0:
            shifted[: probs.size - shift] = probs[shift:]
        else:
            shifted[shift:] = probs[: probs.size - shift]
    finite = (probs > 0.0) & (shifted > 0.0)
    losses = np.log(probs[finite]) - np.log(shifted[finite])
    finite_probs = probs[finite]
    infinite_loss_mass = math.fsum(
        (
            noise_dist.p_min,
            noise_dist.p_max,
            math.fsum(map(float, probs[~finite])),
        )
    )
    if finite_probs.size == 0:
        return PLDRealization(
            grid=GridSpec(step=float(loss_discretization), n=2, anchor=0.0),
            prob_arr=np.zeros(2, dtype=np.float64),
            p_max=infinite_loss_mass,
        )

    losses, groups = np.unique(losses, return_inverse=True)
    probs_grouped = np.zeros_like(losses)
    np.add.at(probs_grouped, groups, finite_probs)
    sparse_dist = SparseDiscreteDist(
        x_array=losses,
        prob_arr=probs_grouped,
        p_max=infinite_loss_mass,
    )
    result = rediscretize_dist_by_bound(
        dist=sparse_dist,
        tail_truncation=tail_truncation,
        loss_discretization=loss_discretization,
        bound_type=BoundType.DOMINATES,
    )
    if not isinstance(result, PLDRealization):
        raise TypeError("dominating count-noise rediscretization must return PLDRealization")
    return result
