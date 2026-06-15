"""dp_accounting compatibility wrappers for subsampling implementation.

Provides translation between dp_accounting's PrivacyLossDistribution objects
and this project's structured discrete-distribution API.

Also provides :func:`safe_self_compose` as a memory-safe replacement for
``PrivacyLossDistribution.self_compose``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from dp_accounting.pld.common import compute_self_convolve_bounds
from dp_accounting.pld.pld_pmf import DensePLDPmf, PLDPmf, SparsePLDPmf
from dp_accounting.pld.privacy_loss_distribution import PrivacyLossDistribution
from scipy.fft import next_fast_len

from PLD_accounting.discrete_dist import (
    REALIZATION_MOMENT_TOL,
    DenseDiscreteDist,
    Domain,
    PLDRealization,
)
from PLD_accounting.distribution_utils import (
    MAX_SAFE_EXP_ARG,
    PMF_MASS_TOL,
    SPACING_ATOL,
    exp_moment_terms,
    stable_isclose,
)
from PLD_accounting.fft_convolution import MAX_FFT_BYTES, fft_convolve
from PLD_accounting.types import BoundType, SpacingType

# Maximum bytes for dp_accounting's self_compose FFT (default 8 GB)
_MAX_SELF_COMPOSE_BYTES = 8 * 1024**3
# Conservative estimate for peak memory per input bin during pairwise fft_convolve:
# two ~2N-element complex128 FFTs plus output and intermediate buffers ≈ 10 × 16 bytes/bin.
_PAIRWISE_FFT_BYTES_PER_BIN = 160
# Bytes per FFT element when estimating dp_accounting self-convolve memory
# (two complex128 working arrays × 16 bytes each).
_SELF_CONVOLVE_BYTES_PER_FFT_ELEMENT = 32
# Fraction of the caller's tail_mass_truncation budget consumed by the inner
# squaring loop in _compose_pmf_binary (1/6, leaving the majority for the caller's phases).
_BINARY_COMPOSE_TRUNC_FRACTION = 1.0 / 6.0
# Relative tolerance for warning when binary composition changes the discretization step.
_DISC_DRIFT_RTOL = 0.01

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
            Must be a linear DenseDiscreteDist. x_min is rounded to the nearest
            step multiple; offsets larger than 0.5 * step raise ValueError.
        pessimistic_estimate: Whether to use pessimistic estimate in dp_accounting.

    Returns:
        dp_accounting DensePLDPmf with infinity mass taken from dist.p_max.
    """
    if not (isinstance(dist, DenseDiscreteDist) and dist.spacing_type == SpacingType.LINEAR):
        raise TypeError(
            f"linear_dist_to_dp_accounting_pmf requires DenseDiscreteDist, got {type(dist)}."
        )

    # The geometric-route pipeline produces x_min values that are sums of
    # geometric-grid minimums, which land off the linear step grid by up to
    # ~0.5 * step.  Round to the nearest integer index; raise only if the
    # offset exceeds half a step, which would indicate a real bug upstream.
    base_index = int(np.rint(dist.x_0 / dist.step))
    offset = abs(base_index * dist.step - dist.x_0)
    if offset > 0.5 * dist.step + SPACING_ATOL:
        raise ValueError(
            f"x_0={dist.x_0!r} is more than 0.5 steps from the nearest grid point "
            f"(step={dist.step!r}, offset={offset:.3e})"
        )
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
        x_0=x_min,
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


# ============================================================================
# Memory-safe self-composition
# ============================================================================


def safe_self_compose(  # pylint: disable=too-many-locals
    pld: PrivacyLossDistribution,
    num_times: int,
    tail_mass_truncation: float = 1e-15,
) -> PrivacyLossDistribution:
    """Memory-safe replacement for PrivacyLossDistribution.self_compose.

    Estimates the memory dp_accounting's direct FFT approach would need.
    If within limits, delegates to dp_accounting (fastest, most accurate).
    Otherwise pre-coarsens the PMF by the minimum factor needed and retries
    dp_accounting's direct FFT path. As a last resort, falls back to binary
    self-convolution with proactive coarsening.

    Args:
        pld: The PrivacyLossDistribution to compose.
        num_times: Number of times to self-compose.
        tail_mass_truncation: Total tail mass budget for truncation.

    Returns:
        A new PrivacyLossDistribution representing the num_times-fold composition.
    """
    if not isinstance(pld, PrivacyLossDistribution):
        raise TypeError(f"pld must be PrivacyLossDistribution, got {type(pld).__name__}")
    if not isinstance(num_times, int) or isinstance(num_times, bool):
        raise TypeError(f"num_times must be an integer, got {type(num_times).__name__}")
    if num_times < 1:
        raise ValueError(f"num_times must be >= 1, got {num_times}")
    if isinstance(tail_mass_truncation, bool):
        raise TypeError("tail_mass_truncation must be a finite nonnegative float, got bool")
    try:
        tail_mass_truncation = float(tail_mass_truncation)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "tail_mass_truncation must be a finite nonnegative float, "
            f"got {type(tail_mass_truncation).__name__}"
        ) from exc
    if not math.isfinite(tail_mass_truncation) or tail_mass_truncation < 0.0:
        raise ValueError(
            "tail_mass_truncation must be finite and nonnegative, " f"got {tail_mass_truncation!r}"
        )

    dense_remove = pld._pmf_remove.to_dense_pmf()
    remove_bytes = _estimate_self_compose_memory(
        dense_remove._probs, num_times, tail_mass_truncation
    )

    add_bytes = 0
    if pld._pmf_add is not None:
        dense_add = pld._pmf_add.to_dense_pmf()
        add_bytes = _estimate_self_compose_memory(dense_add._probs, num_times, tail_mass_truncation)

    max_bytes = max(remove_bytes, add_bytes)

    if max_bytes <= _MAX_SELF_COMPOSE_BYTES:
        return pld.self_compose(num_times, tail_mass_truncation)

    coarsen_factor = math.ceil(max_bytes / _MAX_SELF_COMPOSE_BYTES)
    coarsened_remove = _coarsen_dense_pmf(dense_remove, coarsen_factor)
    coarsened_add = None
    if pld._pmf_add is not None:
        coarsened_add = _coarsen_dense_pmf(dense_add, coarsen_factor)

    coarsened_remove_bytes = _estimate_self_compose_memory(
        coarsened_remove._probs, num_times, tail_mass_truncation
    )
    coarsened_max_bytes = coarsened_remove_bytes
    if coarsened_add is not None:
        coarsened_add_bytes = _estimate_self_compose_memory(
            coarsened_add._probs, num_times, tail_mass_truncation
        )
        coarsened_max_bytes = max(coarsened_max_bytes, coarsened_add_bytes)

    if coarsened_max_bytes <= _MAX_SELF_COMPOSE_BYTES:
        orig_disc = dense_remove._discretization
        new_disc = coarsened_remove._discretization
        warnings.warn(
            f"safe_self_compose: pre-coarsened PMF by {coarsen_factor}x "
            f"(disc {orig_disc:.2e} -> {new_disc:.2e}) for direct FFT "
            f"composition ({coarsened_max_bytes / 1024**3:.1f} GB, "
            f"num_times={num_times:,})."
        )
        coarsened_pld = PrivacyLossDistribution(
            pmf_remove=coarsened_remove,
            pmf_add=coarsened_add,
        )
        return coarsened_pld.self_compose(num_times, tail_mass_truncation)

    warnings.warn(
        f"safe_self_compose: pre-coarsened by {coarsen_factor}x but still "
        f"needs {coarsened_max_bytes / 1024**3:.1f} GB. "
        f"Falling back to binary self-convolution for num_times={num_times}, "
        f"pmf_size={len(dense_remove._probs):,}."
    )

    composed_remove = _compose_pmf_binary(
        coarsened_remove,
        num_times,
        tail_mass_truncation,
    )
    composed_add = None
    if pld._pmf_add is not None:
        assert coarsened_add is not None  # set above whenever pld._pmf_add is not None
        composed_add = _compose_pmf_binary(coarsened_add, num_times, tail_mass_truncation)

    return PrivacyLossDistribution(pmf_remove=composed_remove, pmf_add=composed_add)


def _estimate_self_compose_memory(
    probs: np.ndarray, num_times: int, tail_mass_truncation: float
) -> int:
    """Estimate peak memory (bytes) for dp_accounting's self_convolve."""
    lower_idx, upper_idx = compute_self_convolve_bounds(probs, num_times, tail_mass_truncation)
    output_len = upper_idx - lower_idx + 1
    if not 0 < output_len < float("inf"):
        output_len = num_times * (len(probs) - 1) + 1
    fast_len = next_fast_len(max(int(output_len), len(probs)))
    return fast_len * _SELF_CONVOLVE_BYTES_PER_FFT_ELEMENT


def _coarsen_dense_pmf(dense: DensePLDPmf, factor: int) -> DensePLDPmf:
    """Re-bin a DensePLDPmf by combining groups of ``factor`` adjacent bins."""
    if factor <= 1:
        return dense
    # dp_accounting exposes no public API for PMF internals; protected access is unavoidable here.
    probs = dense._probs  # pylint: disable=protected-access
    n = len(probs)
    lower = dense._lower_loss  # pylint: disable=protected-access

    rem = lower % factor
    if rem != 0:
        pad_left = rem % factor
        probs = np.concatenate([np.zeros(pad_left), probs])
        lower -= pad_left
        n = len(probs)

    pad_right = (-n) % factor
    if pad_right > 0:
        probs = np.concatenate([probs, np.zeros(pad_right)])

    new_n = len(probs) // factor
    new_probs = probs.reshape(new_n, factor).sum(axis=1)
    new_lower = lower // factor

    return DensePLDPmf(
        discretization=dense._discretization * factor,  # pylint: disable=protected-access
        lower_loss=new_lower,
        probs=new_probs,
        infinity_mass=dense._infinity_mass,  # pylint: disable=protected-access
        pessimistic_estimate=dense._pessimistic_estimate,  # pylint: disable=protected-access
    )


def _compose_pmf_binary(
    pmf_obj: PLDPmf, num_times: int, tail_mass_truncation: float
) -> DensePLDPmf:
    """Compose a PLDPmf num_times using memory-bounded binary self-convolution."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks
    dense = pmf_obj.to_dense_pmf()
    # dp_accounting exposes no public API for PMF internals; protected access is unavoidable here.
    probs = dense._probs  # pylint: disable=protected-access
    disc = dense._discretization  # pylint: disable=protected-access
    lower_idx = dense._lower_loss  # pylint: disable=protected-access
    inf_mass = dense._infinity_mass  # pylint: disable=protected-access
    pessimistic = dense._pessimistic_estimate  # pylint: disable=protected-access
    bt = BoundType.DOMINATES if pessimistic else BoundType.IS_DOMINATED

    n = len(probs)
    x_min = float(lower_idx) * disc

    if bt == BoundType.DOMINATES:
        dist = DenseDiscreteDist(
            x_0=x_min,
            step=disc,
            prob_arr=probs,
            p_min=0.0,
            p_max=inf_mass,
            domain=Domain.REALS,
        )
    else:
        dist = DenseDiscreteDist(
            x_0=x_min,
            step=disc,
            prob_arr=probs,
            p_min=inf_mass,
            p_max=0.0,
            domain=Domain.REALS,
        )

    if num_times == 1:
        composed = dist
    else:
        max_bins = _max_pairwise_bins()
        orig_disc = disc

        if n > max_bins:
            dist = _coarsen_dist(dist, max_bins)
            warnings.warn(
                f"_compose_pmf_binary: coarsened PMF from {n:,} to "
                f"{dist.prob_arr.size:,} bins (disc {orig_disc:.2e} -> "
                f"{dist.step:.2e}) to fit pairwise FFT in memory."
            )

        trunc_budget = tail_mass_truncation * _BINARY_COMPOSE_TRUNC_FRACTION
        squaring_limit = max_bins // 2
        base = dist
        acc = None
        t = num_times

        while t > 0:
            step_trunc = trunc_budget / max(t, 1)
            if base.prob_arr.size > squaring_limit:
                base = _coarsen_dist(base, squaring_limit)
            if acc is not None and acc.prob_arr.size > squaring_limit:
                acc = _coarsen_dist(acc, squaring_limit)
            if t & 1:
                if acc is None:
                    acc = base
                else:
                    acc, base = _align_discretizations(acc, base)
                    while True:
                        try:
                            acc = fft_convolve(
                                dist_1=acc,
                                dist_2=base,
                                tail_truncation=step_trunc,
                                bound_type=bt,
                            )
                            break
                        except MemoryError:
                            assert acc is not None
                            acc = _coarsen_dist_by_factor(acc, 2)
                            base = _coarsen_dist_by_factor(base, 2)
            t >>= 1
            if t > 0:
                while True:
                    try:
                        base = fft_convolve(
                            dist_1=base,
                            dist_2=base,
                            tail_truncation=step_trunc,
                            bound_type=bt,
                        )
                        break
                    except MemoryError:
                        base = _coarsen_dist_by_factor(base, 2)
                        if acc is not None:
                            acc = _coarsen_dist_by_factor(acc, 2)

        composed = acc if acc is not None else base
        final_disc = composed.step
        if not np.isclose(final_disc, orig_disc, rtol=_DISC_DRIFT_RTOL):
            warnings.warn(
                f"_compose_pmf_binary: discretization changed from "
                f"{orig_disc:.2e} to {final_disc:.2e} after {num_times}-fold "
                f"composition due to memory-bounded coarsening."
            )

    total = float(np.sum(composed.prob_arr)) + composed.p_min + composed.p_max
    if total > 0 and abs(total - 1.0) > PMF_MASS_TOL:
        composed = DenseDiscreteDist(
            x_0=composed.x_0,
            step=composed.step,
            prob_arr=composed.prob_arr / total,
            p_min=composed.p_min / total,
            p_max=composed.p_max / total,
            domain=composed.domain,
        )

    final_step = composed.step
    lower_loss = int(np.round(composed.x_0 / final_step))
    inf_mass_out = composed.p_max if pessimistic else composed.p_min
    return DensePLDPmf(
        discretization=final_step,
        lower_loss=lower_loss,
        probs=composed.prob_arr.astype(np.float64),
        infinity_mass=inf_mass_out,
        pessimistic_estimate=pessimistic,
    )


def _max_pairwise_bins() -> int:
    """Maximum PMF bins per input for pairwise fft_convolve within memory limit."""
    return int(MAX_FFT_BYTES / _PAIRWISE_FFT_BYTES_PER_BIN)


def _coarsen_dist(dist: DenseDiscreteDist, max_bins: int) -> DenseDiscreteDist:
    """Re-bin a DenseDiscreteDist to at most ``max_bins`` bins."""
    n = dist.prob_arr.size
    if n <= max_bins:
        return dist
    return _coarsen_dist_by_factor(dist, math.ceil(n / max_bins))


def _align_discretizations(
    dist_1: DenseDiscreteDist, dist_2: DenseDiscreteDist
) -> tuple[DenseDiscreteDist, DenseDiscreteDist]:
    """Coarsen the finer distribution so both share the same bin width."""
    d1 = dist_1.step
    d2 = dist_2.step
    if stable_isclose(a=d1, b=d2):
        return dist_1, dist_2
    if d1 < d2:
        return _coarsen_dist_by_factor(dist_1, max(1, round(d2 / d1))), dist_2
    return dist_1, _coarsen_dist_by_factor(dist_2, max(1, round(d1 / d2)))


def _coarsen_dist_by_factor(dist: DenseDiscreteDist, factor: int) -> DenseDiscreteDist:
    """Re-bin a DenseDiscreteDist by combining groups of ``factor`` adjacent bins."""
    if factor <= 1:
        return dist
    n = dist.prob_arr.size
    new_n = math.ceil(n / factor)
    padded_n = new_n * factor
    padded_pmf = np.zeros(padded_n, dtype=np.float64)
    padded_pmf[:n] = dist.prob_arr
    new_prob_arr = padded_pmf.reshape(new_n, factor).sum(axis=1)
    return DenseDiscreteDist(
        x_0=dist.x_0,
        step=dist.step * factor,
        prob_arr=new_prob_arr,
        p_min=dist.p_min,
        p_max=dist.p_max,
        domain=dist.domain,
    )
