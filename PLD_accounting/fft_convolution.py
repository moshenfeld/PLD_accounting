"""FFT-based convolution for linear-grid distributions."""

from __future__ import annotations

import math
import os
import warnings

import numpy as np
from dp_accounting.pld.common import compute_self_convolve_bounds
from scipy.fft import irfft, next_fast_len, rfft

from PLD_accounting.discrete_dist import DenseDiscreteDist
from PLD_accounting.distribution_utils import enforce_mass_conservation, stable_isclose
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import (
    binary_self_convolve,
    convolve_boundary_masses,
    self_convolve_boundary_masses,
)

# Maximum bytes for a single FFT allocation (default 8 GB, override via MAX_FFT_BYTES env var)
MAX_FFT_BYTES = int(os.environ.get("MAX_FFT_BYTES", 8 * 1024**3))


def fft_convolve(
    *,
    dist_1: DenseDiscreteDist,
    dist_2: DenseDiscreteDist,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    """Convolve two linear-grid distributions via FFT."""
    if not (
        isinstance(dist_1, DenseDiscreteDist) and dist_1.spacing_type == SpacingType.LINEAR
    ) or not (isinstance(dist_2, DenseDiscreteDist) and dist_2.spacing_type == SpacingType.LINEAR):
        raise TypeError(
            "fft_convolve requires linear DenseDiscreteDist inputs; "
            f"got dist_1={type(dist_1).__name__} (spacing={dist_1.spacing_type}), "
            f"dist_2={type(dist_2).__name__} (spacing={dist_2.spacing_type})"
        )
    if dist_1.domain != dist_2.domain:
        raise ValueError(f"Input domains must be identical, got {dist_1.domain} vs {dist_2.domain}")
    if not np.any(dist_1.prob_arr) or not np.any(dist_2.prob_arr):
        raise ValueError("FFT convolution requires nonzero finite mass in both inputs")
    if not stable_isclose(a=dist_1.step, b=dist_2.step):
        raise ValueError(f"Grid spacing must match: w1={dist_1.step:.12g} vs w2={dist_2.step:.12g}")

    width = dist_1.step
    conv_x_min = dist_1.x_min + dist_2.x_min

    # --- Manual rfft/irfft with in-place multiply (saves one complex128 buffer) ---
    conv_full_len = dist_1.prob_arr.size + dist_2.prob_arr.size - 1
    fft_size = next_fast_len(conv_full_len)
    _check_fft_memory(fft_size, label="fft_convolve")

    # Capture ghost-mass bounds and normalization factors before FFT buffers are allocated
    nz1 = np.nonzero(dist_1.prob_arr)[0]
    nz2 = np.nonzero(dist_2.prob_arr)[0]
    min_idx = int(nz1[0] + nz2[0])
    max_idx = int(nz1[-1] + nz2[-1])
    finite_prob_1 = math.fsum(map(float, dist_1.prob_arr))
    finite_prob_2 = math.fsum(map(float, dist_2.prob_arr))

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
    max_idx_plus_one = max_idx + 1
    if max_idx_plus_one < conv_pmf.size:
        conv_pmf[max_idx_plus_one:] = 0.0

    current_finite_mass = math.fsum(map(float, conv_pmf))
    if current_finite_mass <= 0.0:
        raise ValueError("FFT convolution produced zero finite mass")
    # Renormalize finite mass before reattaching the analytically computed
    # infinity masses. This corrects small drift from FFT arithmetic/clipping.
    conv_pmf *= finite_prob_1 * finite_prob_2 / current_finite_mass

    expected_p_min, expected_p_max = convolve_boundary_masses(
        dist_1.p_min, dist_1.p_max, dist_2.p_min, dist_2.p_max, dist_1.domain
    )
    conv_pmf, p_min, p_max = enforce_mass_conservation(
        prob_arr=conv_pmf,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_min=conv_x_min,
        step=width,
        prob_arr=conv_pmf,
        p_min=p_min,
        p_max=p_max,
        domain=dist_1.domain,
    ).truncate_edges(tail_truncation, bound_type)


def fft_self_convolve(
    *,
    dist: DenseDiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
    use_direct: bool,
) -> DenseDiscreteDist:
    """T-fold self-convolution via FFT with optional direct exponentiation path."""
    if not (isinstance(dist, DenseDiscreteDist) and dist.spacing_type == SpacingType.LINEAR):
        raise TypeError("fft_self_convolve requires DenseDiscreteDist input")

    if use_direct:
        try:
            return _fft_self_convolve_direct(
                dist=dist,
                T=T,
                tail_truncation=tail_truncation,
                bound_type=bound_type,
            )
        except MemoryError:
            warnings.warn(
                f"fft_self_convolve: direct method exceeded {MAX_FFT_BYTES / 1024**3:.0f} GB "
                f"memory limit for T={T}, pmf_size={dist.prob_arr.size:,}. "
                f"Falling back to binary self-convolution."
            )

    self_conv = binary_self_convolve(
        dist=dist,
        T=T,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=fft_convolve,
    )
    if not (
        isinstance(self_conv, DenseDiscreteDist) and self_conv.spacing_type == SpacingType.LINEAR
    ):
        raise TypeError(
            f"Expected DenseDiscreteDist from FFT self-convolution, got {type(self_conv)}"
        )
    return self_conv


def _fft_self_convolve_direct(
    *,
    dist: DenseDiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
) -> DenseDiscreteDist:
    # Budget split: the input tail_truncation is divided into three equal thirds.
    #   _calc_fft_window_size: Chernoff-based window determines the one-sided tail
    #          cutoff (right-tail for DOMINATES, folded-back mass bound for IS_DOMINATED).
    #   explicit opposite-side trim: left_tail_ind for DOMINATES (zeroes left bins,
    #          pushes mass to p_max) / right_tail_ind for IS_DOMINATED (zeroes right bins,
    #          pushes mass to p_min).
    #   final truncate_edges: trims actual near-zero edge bins the Chernoff window
    #          conservatively included on the remaining untrimmed side, reducing output
    #          bin count without sacrificing accuracy.
    # Total: 3 * (tail_truncation / 3) = tail_truncation
    tail_truncation /= 3

    finite_mass = math.fsum(map(float, dist.prob_arr))
    # The Chernoff window calculation expects a normalized finite PMF, so the
    # tail target must be rescaled when some mass already sits at infinity.
    normalized_pmf = dist.prob_arr / finite_mass
    tail_truncation_rescaled = tail_truncation / finite_mass

    shift_left, window_size = _calc_fft_window_size(
        pmf=normalized_pmf, num_convolutions=T, tail_truncation=tail_truncation_rescaled
    )

    fft_size = next_fast_len(max(window_size, dist.prob_arr.size))
    _check_fft_memory(fft_size, label=f"_fft_self_convolve_direct(T={T})")
    fft_data = rfft(dist.prob_arr, n=fft_size)
    fft_data **= T  # in-place power: avoids allocating a second complex buffer
    raw_conv = np.asarray(irfft(fft_data, n=fft_size, overwrite_x=True), dtype=np.float64)
    del fft_data  # free complex buffer
    raw_conv[raw_conv < 0] = 0.0
    # ``shift_left`` is the left edge of the retained convolution window. Rolling aligns
    # that window to index 0 so truncation logic can work in-place.
    rolled_conv = np.roll(raw_conv, -shift_left)

    conv_p_min, conv_p_max = self_convolve_boundary_masses(dist, num_convolutions=T)
    if bound_type == BoundType.DOMINATES:
        # For an upper bound, any dropped left-tail mass is pushed to +inf.
        cumsum = np.cumsum(rolled_conv)
        left_tail_ind = int(np.searchsorted(cumsum, tail_truncation, side="right"))
        shifted_mass = math.fsum(map(float, rolled_conv[:left_tail_ind]))
        rolled_conv[:left_tail_ind] = 0.0
        right_tail_mass = math.fsum(map(float, rolled_conv[window_size:]))
        conv_p_max += shifted_mass + right_tail_mass
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
        conv_p_min += shifted_mass

        right_tail_mass = math.fsum(map(float, rolled_conv[window_size:]))
        rolled_conv[min(window_size, right_tail_ind) - 1] += right_tail_mass
    else:
        raise ValueError(f"Unknown BoundType: {bound_type}")

    x_min = dist.x_min * T + shift_left * dist.step
    pmf_conv = rolled_conv[:window_size]
    pmf_conv, p_min_final, p_max_final = enforce_mass_conservation(
        prob_arr=pmf_conv,
        expected_p_min=conv_p_min,
        expected_p_max=conv_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_min=x_min,
        step=dist.step,
        prob_arr=pmf_conv,
        p_min=p_min_final,
        p_max=p_max_final,
        domain=dist.domain,
    ).truncate_edges(tail_truncation, bound_type)


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


def _check_fft_memory(fft_size: int, label: str = "FFT") -> None:
    """Raise MemoryError if an FFT of this size would exceed the safety limit.

    rfft produces complex128 output (~16 bytes per element) and the input is
    float64 (~8 bytes), so peak usage is roughly 24 * fft_size bytes.
    """
    estimated_bytes = 24 * fft_size
    if estimated_bytes > MAX_FFT_BYTES:
        raise MemoryError(
            f"{label}: estimated {estimated_bytes / 1024**3:.1f} GB for "
            f"fft_size={fft_size:,} exceeds safety limit of "
            f"{MAX_FFT_BYTES / 1024**3:.1f} GB. "
            f"Reduce grid size, increase loss_discretization, or raise MAX_FFT_BYTES."
        )
