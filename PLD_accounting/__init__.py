"""Public entry points for random-allocation privacy accounting."""

from PLD_accounting.discrete_dist import PLDRealization
from PLD_accounting.mechanisms import (
    gaussian_distribution,
    laplace_distribution,
)
from PLD_accounting.random_allocation_accounting import compose_full_pld
from PLD_accounting.random_allocation_api import (
    gaussian_allocation_delta_configurable,
    gaussian_allocation_directional_pld,
    gaussian_allocation_epsilon_configurable,
    gaussian_allocation_epsilon_range,
    gaussian_allocation_pld,
    general_allocation_delta,
    general_allocation_epsilon,
    general_allocation_pld,
)
from PLD_accounting.subsample_pld import (
    subsample_pld,
    subsample_pld_realization,
)
from PLD_accounting.types import (
    DEFAULT_LOSS_DISCRETIZATION,
    DEFAULT_TAIL_TRUNCATION,
    AllocationSchemeConfig,
    BoundType,
    ConvolutionMethod,
    Direction,
    PrivacyParams,
    SpacingType,
    has_numba,
)

__all__ = [
    "PLDRealization",
    "AllocationSchemeConfig",
    "BoundType",
    "ConvolutionMethod",
    "DEFAULT_LOSS_DISCRETIZATION",
    "DEFAULT_TAIL_TRUNCATION",
    "Direction",
    "PrivacyParams",
    "SpacingType",
    "has_numba",
    "compose_full_pld",
    "gaussian_allocation_directional_pld",
    "gaussian_allocation_pld",
    "gaussian_allocation_delta_configurable",
    "gaussian_allocation_epsilon_configurable",
    "gaussian_allocation_epsilon_range",
    "gaussian_distribution",
    "general_allocation_pld",
    "general_allocation_delta",
    "general_allocation_epsilon",
    "laplace_distribution",
    "subsample_pld",
    "subsample_pld_realization",
]
