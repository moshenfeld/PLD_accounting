"""FFT-based convolution for linear-grid distributions."""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
from dp_accounting.pld.common import compute_self_convolve_bounds
from scipy.fft import irfft, next_fast_len, rfft

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    require_linear_reals_dist,
)
from PLD_accounting.distribution_utils import enforce_mass_conservation, trim_mass_from_edge
from PLD_accounting.types import BoundType, require_bound_type
from PLD_accounting.utils import (
    binary_self_convolve,
    convolve_boundary_masses,
    self_convolve_boundary_masses,
)
from PLD_accounting.validation import require_nonnegative_real, require_positive_int

# Maximum bytes for a single FFT allocation (default 8 GB, override via MAX_FFT_BYTES env var)
MAX_FFT_BYTES = int(os.environ.get("MAX_FFT_BYTES", 8 * 1024**3))
# Policy multipliers on the size-scaled FFT mass residual.
_FFT_MASS_DRIFT_FACTOR = 16.0
_FFT_MASS_REPAIR_FACTOR = 64.0


def fft_convolve(
    *,
    dist_1: DenseDiscreteDist,
    dist_2: DenseDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Convolve two real-domain linear-grid distributions via FFT.

    Inputs must share the exact structural step; their output grid is the
    lattice sum. Opposing infinite atoms are rejected because ``-inf + inf``
    has no defined boundary placement.
    """
    require_linear_reals_dist(dist=dist_1, name="dist_1")
    require_linear_reals_dist(dist=dist_2, name="dist_2")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_bound_type(value=bound_type)
    if (dist_1.p_min > 0.0 and dist_2.p_max > 0.0) or (dist_1.p_max > 0.0 and dist_2.p_min > 0.0):
        raise ValueError(
            "FFT convolution is undefined when one real-domain input has -inf mass "
            "and the other has +inf mass"
        )
    _require_positive_finite_support(dist=dist_1, context="FFT convolution")
    _require_positive_finite_support(dist=dist_2, context="FFT convolution")
    if dist_1.step != dist_2.step:
        raise ValueError(f"Grid spacing must match: w1={dist_1.step:.12g} vs w2={dist_2.step:.12g}")

    conv_grid = dist_1.grid.convolve(dist_2.grid)

    # --- Manual rfft/irfft with in-place multiply (saves one complex128 buffer) ---
    conv_full_len = dist_1.prob_arr.size + dist_2.prob_arr.size - 1
    fft_size = next_fast_len(conv_full_len)
    _check_fft_memory(fft_size=fft_size, label="fft_convolve")

    # Capture reachable-support bounds before FFT buffers are allocated.
    nz1 = np.nonzero(dist_1.prob_arr)[0]
    nz2 = np.nonzero(dist_2.prob_arr)[0]
    min_idx = int(nz1[0] + nz2[0])
    max_idx = int(nz1[-1] + nz2[-1])

    # Self-squaring optimization: if both inputs are the same object,
    # compute rfft once and square in-place (saves one complex buffer)
    is_self_convolve = dist_1 is dist_2 or dist_1.prob_arr is dist_2.prob_arr
    fft1 = rfft(dist_1.prob_arr, n=fft_size)
    if is_self_convolve:
        fft1 *= fft1  # in-place square
    else:
        fft2 = rfft(dist_2.prob_arr, n=fft_size)
        fft1 *= fft2  # in-place multiply
        del fft2  # free second complex buffer immediately
    conv_full = irfft(fft1, n=fft_size, overwrite_x=True)
    del fft1  # free complex buffer
    conv_pmf = conv_full[:conv_full_len].copy()  # copy needed portion
    del conv_full  # free full irfft output

    # Zero negative roundoff and ghost mass outside reachable support
    conv_pmf[conv_pmf < 0] = 0.0
    conv_pmf[:min_idx] = 0.0
    if max_idx + 1 < conv_pmf.size:
        first_unreachable = max_idx + 1
        conv_pmf[first_unreachable:] = 0.0

    if math.fsum(map(float, conv_pmf)) <= 0.0:
        raise ValueError("FFT convolution produced zero finite mass")

    # Account exactly for boundary-by-boundary convolution mass.
    expected_p_min, expected_p_max = convolve_boundary_masses(
        p_min_1=dist_1.p_min,
        p_max_1=dist_1.p_max,
        p_min_2=dist_2.p_min,
        p_max_2=dist_2.p_max,
        domain=Domain.REALS,
    )
    # Repair FFT/clipping drift; the uncompensated transform needs the FFT-sized bands.
    drift_tol, repair_tol = _fft_mass_tolerances(num_bins=fft_size, num_convolutions=1)
    conv_pmf, p_min, p_max = enforce_mass_conservation(
        prob_arr=conv_pmf,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
        drift_tol=drift_tol,
        repair_tol=repair_tol,
    )

    return DenseDiscreteDist(
        grid=conv_grid,
        prob_arr=conv_pmf,
        p_min=p_min,
        p_max=p_max,
        domain=Domain.REALS,
    ).truncate_edges(tail_truncation=tail_truncation, bound_type=bound_type)


def fft_self_convolve(
    *,
    dist: DenseDiscreteDist,
    num_convolutions: int,
    tail_truncation: float,
    bound_type: BoundType,
    use_direct: bool,
) -> DenseDiscreteDist:
    """Self-convolve a real-domain distribution via FFT.

    ``use_direct`` first tries one powered transform with a Chernoff-sized
    window and falls back to binary composition if that transform exceeds the
    configured memory limit.
    """
    require_linear_reals_dist(dist=dist, name="dist")
    require_positive_int(value=num_convolutions, name="num_convolutions")
    require_nonnegative_real(value=tail_truncation, name="tail_truncation")
    require_bound_type(value=bound_type)
    _require_positive_finite_support(dist=dist, context="FFT self-convolution")

    if use_direct:
        try:
            return _fft_self_convolve_direct(
                dist=dist,
                num_convolutions=num_convolutions,
                tail_truncation=tail_truncation,
                bound_type=bound_type,
            )
        except MemoryError:
            warnings.warn(
                f"fft_self_convolve: direct method exceeded {MAX_FFT_BYTES / 1024**3:.0f} GB "
                f"memory limit for num_convolutions={num_convolutions}, "
                f"pmf_size={dist.prob_arr.size:,}. "
                f"Falling back to binary self-convolution."
            )

    return binary_self_convolve(
        dist=dist,
        num_convolutions=num_convolutions,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=fft_convolve,
    )


def _require_positive_finite_support(*, dist: DenseDiscreteDist, context: str) -> None:
    """Reject a boundary-only input before FFT work or finite-mass normalization."""
    if math.fsum(map(float, dist.prob_arr)) <= 0.0:
        raise ValueError(
            f"{context} requires strictly positive finite-support mass; "
            "boundary-only inputs are not supported"
        )


def _fft_mass_tolerances(
    *,
    num_bins: int,
    num_convolutions: int,
) -> tuple[float, float]:
    """Return ``(drift_tol, repair_tol)`` for an FFT-produced PMF."""
    require_positive_int(value=[num_bins, num_convolutions], name=["num_bins", "num_convolutions"])
    # Transform error grows logarithmically with size and linearly with the power.
    scale = float(num_convolutions) * math.log2(max(num_bins, 2)) * float(np.finfo(float).eps)
    return _FFT_MASS_DRIFT_FACTOR * scale, _FFT_MASS_REPAIR_FACTOR * scale


def _fft_self_convolve_direct(
    *,
    dist: DenseDiscreteDist,
    num_convolutions: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Self-convolve in one shot by raising the PMF's DFT to the ``num_convolutions`` power.

    A Chernoff bound sizes the retained output window, so the single transform
    stays bounded instead of growing with the full ``num_convolutions``-fold
    support. Raises ``MemoryError`` when that window still exceeds the FFT
    safety limit, which the caller treats as a signal to fall back to binary
    self-convolution.
    """
    require_linear_reals_dist(dist=dist, name="dist")

    # Budget split: the input tail_truncation is divided into four equal quarters.
    #   _calc_fft_window_size: Chernoff-based window determines the one-sided tail
    #          cutoff (right-tail for DOMINATES, folded-back mass bound for IS_DOMINATED).
    #   circular alias reserve: spent only when the true support outruns the FFT period,
    #          where the same window allowance bounds a second, indistinguishable amount
    #          of mass that may have folded into a wrong bin (see _declared_alias_shift).
    #   explicit opposite-side trim: left_tail_ind for DOMINATES (zeroes left bins,
    #          pushes mass to p_max) / right_tail_ind for IS_DOMINATED (zeroes right bins,
    #          pushes mass to p_min).
    #   final truncate_edges: trims actual near-zero edge bins the Chernoff window
    #          conservatively included on the remaining untrimmed side, reducing output
    #          bin count without sacrificing accuracy.
    # Total: 4 * (tail_truncation / 4) = tail_truncation
    tail_truncation /= 4

    finite_mass = math.fsum(map(float, dist.prob_arr))
    # The Chernoff window calculation expects a normalized finite PMF, so the
    # tail target must be rescaled when some mass already sits at infinity.
    normalized_pmf = dist.prob_arr / finite_mass
    tail_truncation_rescaled = tail_truncation / finite_mass

    shift_left, window_size = _calc_fft_window_size(
        pmf=normalized_pmf,
        num_convolutions=num_convolutions,
        tail_truncation=tail_truncation_rescaled,
    )

    fft_size = next_fast_len(max(window_size, dist.prob_arr.size))
    _check_fft_memory(
        fft_size=fft_size,
        label=f"_fft_self_convolve_direct(num_convolutions={num_convolutions})",
    )
    fft_data = rfft(dist.prob_arr, n=fft_size)
    fft_data **= num_convolutions  # in-place power: avoids allocating a second complex buffer
    raw_conv = np.asarray(irfft(fft_data, n=fft_size, overwrite_x=True), dtype=np.float64)
    del fft_data  # free complex buffer
    raw_conv[raw_conv < 0] = 0.0
    # ``shift_left`` is the left edge of the retained convolution window. Rolling aligns
    # that window to index 0 so truncation logic can work in-place.
    rolled_conv = np.roll(raw_conv, -shift_left)

    alias_shift = _declared_alias_shift(
        input_size=dist.prob_arr.size,
        num_convolutions=num_convolutions,
        fft_size=fft_size,
        finite_mass=finite_mass,
        window_tail_truncation=tail_truncation_rescaled,
    )

    # Account exactly for boundary mass after repeated composition.
    conv_p_min, conv_p_max = self_convolve_boundary_masses(
        dist=dist, num_convolutions=num_convolutions
    )
    if bound_type == BoundType.DOMINATES:
        # For an upper bound, any dropped left-tail mass is pushed to +inf.
        cumsum = np.cumsum(rolled_conv)
        left_tail_ind = int(np.searchsorted(cumsum, tail_truncation, side="right"))
        shifted_mass = math.fsum(map(float, rolled_conv[:left_tail_ind]))
        rolled_conv[:left_tail_ind] = 0.0
        right_tail_mass = math.fsum(map(float, rolled_conv[window_size:]))
        conv_p_max += shifted_mass + right_tail_mass + alias_shift
    elif bound_type == BoundType.IS_DOMINATED:
        # For a lower bound, dropped right-tail mass moves to -inf, while any
        # overflow beyond the retained FFT window is folded onto the last kept
        # finite bin to preserve domination direction.
        cumsum = np.cumsum(rolled_conv[::-1])
        right_tail_ind = (
            rolled_conv.size - 1 - int(np.searchsorted(cumsum, tail_truncation, side="right"))
        )
        after_right_tail = right_tail_ind + 1
        shifted_mass = math.fsum(map(float, rolled_conv[after_right_tail:]))
        rolled_conv[after_right_tail:] = 0.0
        conv_p_min += shifted_mass + alias_shift

        right_tail_mass = math.fsum(map(float, rolled_conv[window_size:]))
        rolled_conv[min(window_size, right_tail_ind) - 1] += right_tail_mass
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    # The retained window is an integer slice of the exact m-fold self-sum lattice.
    out_grid = dist.grid.self_convolve(num_convolutions).slice(start=shift_left, n=window_size)
    pmf_conv = rolled_conv[:window_size]
    # ``alias_shift`` was banked at the conservative boundary above. Fund it here by
    # taking the same mass off the giveable edge, so the declared allocation is a move
    # and total mass is never off one. Those bins contribute least to the divergence,
    # whereas spreading the cost over every bin would take most of it straight back out
    # of the boundary. Mass conservation below then sees only genuine transform drift.
    if alias_shift > 0.0:
        pmf_conv = trim_mass_from_edge(
            prob_arr=pmf_conv,
            mass=alias_shift,
            from_left=bound_type == BoundType.DOMINATES,
        )
    # Repair only numerical drift after retained and discarded mass is explicit. The
    # in-place power above scales the residual with num_convolutions as well as n.
    drift_tol, repair_tol = _fft_mass_tolerances(
        num_bins=fft_size, num_convolutions=num_convolutions
    )
    pmf_conv, p_min_final, p_max_final = enforce_mass_conservation(
        prob_arr=pmf_conv,
        expected_p_min=conv_p_min,
        expected_p_max=conv_p_max,
        bound_type=bound_type,
        drift_tol=drift_tol,
        repair_tol=repair_tol,
    )

    return DenseDiscreteDist(
        grid=out_grid,
        prob_arr=pmf_conv,
        p_min=p_min_final,
        p_max=p_max_final,
        domain=dist.domain,
    ).truncate_edges(tail_truncation=tail_truncation, bound_type=bound_type)


def _declared_alias_shift(
    *,
    input_size: int,
    num_convolutions: int,
    fft_size: int,
    finite_mass: float,
    window_tail_truncation: float,
) -> float:
    """Return the probability that circular aliasing may have placed in a wrong bin.

    The transform has period ``fft_size``. When the exact ``num_convolutions``-fold
    support is longer than that, output mass at index ``j >= fft_size`` folds onto
    ``j mod fft_size``, and no sum over the circular array can say how much did: the fold
    is mass-preserving. Two window indices cannot collide with each other, so everything
    misplaced comes from outside the retained window, and the window was selected to leave
    at most ``window_tail_truncation`` of the *normalized* m-fold mass out there. Scaling
    by ``finite_mass ** num_convolutions`` restores that allowance to the output's own
    mass units, which is what the caller's ledger is denominated in.

    Returns 0.0 when the FFT period covers the true support, where the convolution is
    exact and no allowance is owed.
    """
    if num_convolutions * (input_size - 1) + 1 <= fft_size:
        return 0.0
    return window_tail_truncation * finite_mass**num_convolutions


def _calc_fft_window_size(
    *, pmf: np.ndarray, num_convolutions: int, tail_truncation: float
) -> tuple[int, int]:
    """Calculate FFT window bounds for ``num_convolutions`` self-convolutions with fallback."""
    # ``compute_self_convolve_bounds`` gives a Chernoff-style window [lower, upper] that
    # should contain all but ``tail_truncation`` mass after ``num_convolutions`` convolutions.
    lower_idx, upper_idx = compute_self_convolve_bounds(pmf, num_convolutions, tail_truncation)
    window_size = upper_idx - lower_idx + 1

    if not 0 < window_size < float("inf"):
        lower_idx = 0
        n = len(pmf)
        # Fallback to the exact full-support FFT length when the bound becomes
        # numerically unusable for extreme truncation parameters.
        window_size = num_convolutions * (n - 1) + 1
        warnings.warn(
            "calc_fft_window_size: Chernoff bounds failed "
            f"(tail_truncation={tail_truncation:.3e}, num_convolutions={num_convolutions}). "
            f"Using fallback lower_idx=0, window_size={window_size:,} (n={n})."
        )

    return int(lower_idx), int(window_size)


def _check_fft_memory(*, fft_size: int, label: str) -> None:
    """Raise ``MemoryError`` when an FFT exceeds the configured memory limit."""
    # One complex128 output plus one float64 input is about 24 bytes per point.
    estimated_bytes = 24 * fft_size
    if estimated_bytes > MAX_FFT_BYTES:
        raise MemoryError(
            f"{label}: estimated {estimated_bytes / 1024**3:.1f} GB for "
            f"fft_size={fft_size:,} exceeds safety limit of "
            f"{MAX_FFT_BYTES / 1024**3:.1f} GB. "
            f"Reduce grid size, increase loss_discretization, or raise MAX_FFT_BYTES."
        )
