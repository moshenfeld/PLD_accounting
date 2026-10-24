"""Convolutions selected by a convolution method.

Each function calls the geometric or the multiscale implementation of its
operation; the implementations live in the module of their grid.
"""

from __future__ import annotations

from functools import partial

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    DiscreteDistBase,
    DyadicDist,
    SparseDiscreteDist,
    cast_dense_dist,
    cast_linear_reals_dist,
    cast_geometric_positives_dist,
)
from PLD_accounting.geometric_convolution import geometric_convolve
from PLD_accounting.multiscale_fft_convolution import (
    dyadic_dist_on_range,
    multiscale_convolve,
    multiscale_self_convolve,
)
from PLD_accounting.types import BoundType, ConvolutionMethod, ProjectionMethod
from PLD_accounting.utils import (
    binary_self_convolve,
    binary_self_convolve_depth,
    exp_linear_to_geometric,
)
from PLD_accounting.validation import cast_type

RatioDist = DenseDiscreteDist | DyadicDist


def loss_to_ratio_dist(
    *, loss_dist: DiscreteDistBase, convolution_method: ConvolutionMethod
) -> RatioDist:
    """Return the law of the likelihood ratio ``exp(L)`` of a loss distribution.

    The mass at ``-inf`` becomes the atom at zero and the mass at ``+inf`` stays at
    infinity. ``GEOM`` exponentiates a linear grid onto a geometric one. ``MS_FFT``
    places a distribution on the logarithms of dyadic points onto the dyadic grid.
    """
    if convolution_method == ConvolutionMethod.GEOM:
        linear_dist = cast_linear_reals_dist(dist=loss_dist, name="dist")
        return exp_linear_to_geometric(linear_dist)
    if convolution_method == ConvolutionMethod.MS_FFT:
        points_dist = cast_type(value=loss_dist, expected_type=SparseDiscreteDist, name="loss_dist")
        return dyadic_dist_on_range(points_dist)
    raise ValueError(f"Unknown ConvolutionMethod: {convolution_method}")


def convolve(
    *,
    dist_1: RatioDist,
    dist_2: RatioDist,
    tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
    convolution_method: ConvolutionMethod,
) -> RatioDist:
    """Return the sum of two laws of likelihood ratios, projected onto their grid.

    ``GEOM`` trims ``tail_truncation`` from the sum. ``MS_FFT`` does not trim.
    """
    if convolution_method == ConvolutionMethod.GEOM:
        geometric_1 = cast_geometric_positives_dist(dist=dist_1, name="dist_1")
        geometric_2 = cast_geometric_positives_dist(dist=dist_2, name="dist_2")
        return geometric_convolve(
            dist_1=geometric_1,
            dist_2=geometric_2,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            projection_method=projection_method,
        )
    if convolution_method == ConvolutionMethod.MS_FFT:
        dyadic_1 = cast_type(value=dist_1, expected_type=DyadicDist, name="dist_1")
        dyadic_2 = cast_type(value=dist_2, expected_type=DyadicDist, name="dist_2")
        return multiscale_convolve(
            dist_1=dyadic_1,
            dist_2=dyadic_2,
            tail_truncation=0.0,
            projection_method=projection_method,
            bound_type=bound_type,
        )
    raise ValueError(f"Unknown ConvolutionMethod: {convolution_method}")


def self_convolve(
    *,
    dist: RatioDist,
    num_convolutions: int,
    tail_truncation: float,
    bound_type: BoundType,
    projection_method: ProjectionMethod,
    convolution_method: ConvolutionMethod,
) -> RatioDist:
    """Return the sum of ``num_convolutions`` copies of a law of likelihood ratios.

    Both grids sum by squaring. ``GEOM`` uses ``geometric_convolve`` and spends
    ``tail_truncation`` over its merges. ``MS_FFT`` does not trim and admits the
    completed sum once against its copies.
    """
    if convolution_method == ConvolutionMethod.GEOM:
        geometric_dist = cast_dense_dist(dist=dist, name="dist")
        geometric_merge = partial(geometric_convolve, projection_method=projection_method)
        return binary_self_convolve(
            dist=geometric_dist,
            num_convolutions=num_convolutions,
            tail_truncation=tail_truncation,
            bound_type=bound_type,
            convolve=geometric_merge,
        )
    if convolution_method == ConvolutionMethod.MS_FFT:
        dyadic_dist = cast_type(value=dist, expected_type=DyadicDist, name="dist")
        return multiscale_self_convolve(
            dist=dyadic_dist,
            num_convolutions=num_convolutions,
            tail_truncation=0.0,
            projection_method=projection_method,
            bound_type=bound_type,
        )
    raise ValueError(f"Unknown ConvolutionMethod: {convolution_method}")


def loss_stage_count(*, num_convolutions: int, convolution_method: ConvolutionMethod) -> int:
    """Return the stages that discretize the loss of a sum of ``num_convolutions`` copies.

    Both grids count the projection of the one-step input and the merges above one
    copy, ``binary_self_convolve_depth(num_convolutions)``, along which the errors of
    the merges accumulate. ``MS_FFT`` adds the projection of the sum onto a linear
    grid; on ``GEOM`` the logarithm of the sum is already on one.
    """
    if convolution_method == ConvolutionMethod.GEOM:
        closing_stages = 0
    elif convolution_method == ConvolutionMethod.MS_FFT:
        closing_stages = 1
    else:
        raise ValueError(f"Unknown ConvolutionMethod: {convolution_method}")
    return 1 + binary_self_convolve_depth(num_convolutions) + closing_stages
