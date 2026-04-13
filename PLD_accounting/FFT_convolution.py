"""FFT-based convolution for linear-grid distributions."""

from __future__ import annotations

import math
import warnings

import numpy as np
from dp_accounting.pld.common import compute_self_convolve_bounds
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import fftconvolve

from PLD_accounting.discrete_dist import DenseDiscreteDist
from PLD_accounting.distribution_utils import enforce_mass_conservation, stable_isclose
from PLD_accounting.types import BoundType, SpacingType
from PLD_accounting.utils import (
    binary_self_convolve,
    convolve_boundary_masses,
    self_convolve_boundary_masses,
)


def FFT_convolve(
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
            "FFT_convolve requires linear DenseDiscreteDist inputs; "
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
    conv_PMF = fftconvolve(dist_1.prob_arr, dist_2.prob_arr, mode="full")
    conv_PMF[conv_PMF < 0] = 0.0

    # Zero any mass outside the mathematically reachable support. FFT roundoff
    # can otherwise leave tiny "ghost" mass in bins that should be impossible.
    nz1 = np.nonzero(dist_1.prob_arr)[0]
    nz2 = np.nonzero(dist_2.prob_arr)[0]
    min_idx = int(nz1[0] + nz2[0])
    max_idx = int(nz1[-1] + nz2[-1])
    conv_PMF[:min_idx] = 0.0
    if max_idx + 1 < conv_PMF.size:
        tail_start = max_idx + 1
        conv_PMF[tail_start:] = 0.0

    finite_prob_1 = math.fsum(map(float, dist_1.prob_arr))
    finite_prob_2 = math.fsum(map(float, dist_2.prob_arr))
    current_finite_mass = math.fsum(map(float, conv_PMF))
    if current_finite_mass <= 0.0:
        raise ValueError("FFT convolution produced zero finite mass")
    # Renormalize finite mass before reattaching the analytically computed
    # infinity masses. This corrects small drift from FFT arithmetic/clipping.
    conv_PMF *= finite_prob_1 * finite_prob_2 / current_finite_mass

    expected_p_min, expected_p_max = convolve_boundary_masses(
        dist_1.p_min, dist_1.p_max, dist_2.p_min, dist_2.p_max, dist_1.domain
    )
    conv_PMF, p_min, p_max = enforce_mass_conservation(
        prob_arr=conv_PMF,
        expected_p_min=expected_p_min,
        expected_p_max=expected_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_min=conv_x_min,
        step=width,
        prob_arr=conv_PMF,
        p_min=p_min,
        p_max=p_max,
        domain=dist_1.domain,
    ).truncate_edges(tail_truncation, bound_type)


def FFT_self_convolve(
    *,
    dist: DenseDiscreteDist,
    T: int,
    tail_truncation: float,
    bound_type: BoundType,
    use_direct: bool,
) -> DenseDiscreteDist:
    """T-fold self-convolution via FFT with optional direct exponentiation path."""
    if not (isinstance(dist, DenseDiscreteDist) and dist.spacing_type == SpacingType.LINEAR):
        raise TypeError("FFT_self_convolve requires DenseDiscreteDist input")

    if use_direct:
        return _fft_self_convolve_direct(
            dist=dist,
            T=T,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
        )

    self_conv = binary_self_convolve(
        dist=dist,
        T=T,
        tail_truncation=tail_truncation,
        bound_type=bound_type,
        convolve=FFT_convolve,
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
    # Half the tail budget is spent inside the FFT windowing logic, and the
    # remainder is consumed by the final truncation step below.
    tail_truncation /= 2

    finite_mass = math.fsum(map(float, dist.prob_arr))
    # The Chernoff window calculation expects a normalized finite PMF, so the
    # tail target must be rescaled when some mass already sits at infinity.
    normalized_PMF = dist.prob_arr / finite_mass
    tail_truncation_rescaled = tail_truncation / finite_mass

    L, window_size = _calc_FFT_window_size(
        pmf=normalized_PMF, T=T, tail_truncation=tail_truncation_rescaled
    )

    fft_size = next_fast_len(max(window_size, dist.prob_arr.size))
    raw_conv = np.asarray(
        irfft(rfft(dist.prob_arr, n=fft_size) ** T, n=fft_size),
        dtype=np.float64,
    )
    raw_conv[raw_conv < 0] = 0.0
    # `L` is the left edge of the retained convolution window. Rolling aligns
    # that window to index 0 so truncation logic can work in-place.
    rolled_conv = np.roll(raw_conv, -L)

    conv_p_min, conv_p_max = self_convolve_boundary_masses(dist, T)
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

    x_min = dist.x_min * T + L * dist.step
    PMF_conv = rolled_conv[:window_size]
    PMF_conv, p_min_final, p_max_final = enforce_mass_conservation(
        prob_arr=PMF_conv,
        expected_p_min=conv_p_min,
        expected_p_max=conv_p_max,
        bound_type=bound_type,
    )

    return DenseDiscreteDist(
        x_min=x_min,
        step=dist.step,
        prob_arr=PMF_conv,
        p_min=p_min_final,
        p_max=p_max_final,
        domain=dist.domain,
    ).truncate_edges(0.0, bound_type)


def _calc_FFT_window_size(*, pmf: np.ndarray, T: int, tail_truncation: float) -> tuple[int, int]:
    """Calculate FFT window bounds for T-fold self-convolution with fallback."""
    # `compute_self_convolve_bounds` gives a Chernoff-style window [L, U] that
    # should contain all but `tail_truncation` mass of the T-fold convolution.
    L, U = compute_self_convolve_bounds(pmf, T, tail_truncation)
    window_size = U - L + 1

    if not 0 < window_size < float("inf"):
        L = 0
        n = len(pmf)
        # Fallback to the exact full-support FFT length when the bound becomes
        # numerically unusable for extreme truncation parameters.
        window_size = T * (n - 1) + 1
        warnings.warn(
            "calc_FFT_window_size: Chernoff bounds failed "
            f"(tail_truncation={tail_truncation:.3e}, T={T}). "
            f"Using fallback L=0, window_size={window_size:,} (n={n})."
        )

    return int(L), int(window_size)
