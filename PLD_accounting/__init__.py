"""Public entry points for random-allocation privacy accounting."""

from PLD_accounting.discrete_dist import (
    DenseDiscreteDist,
    Domain,
    GridSpec,
    PLDRealization,
)
from PLD_accounting.dp_accounting_support import dp_accounting_pmf_to_pld_realization
from PLD_accounting.mechanisms import (
    discrete_distribution,
    gaussian_distribution,
    laplace_distribution,
)
from PLD_accounting.projections import project_discrete_dist
from PLD_accounting.random_allocation_accounting import compose_full_pld
from PLD_accounting.random_allocation_api import (
    gaussian_allocation_delta_configurable,
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
    ProjectionMethod,
    PrivacyParams,
    SpacingType,
    has_numba,
)

__all__ = [
    "DenseDiscreteDist",
    "Domain",
    "GridSpec",
    "PLDRealization",
    "AllocationSchemeConfig",
    "BoundType",
    "ConvolutionMethod",
    "DEFAULT_LOSS_DISCRETIZATION",
    "DEFAULT_TAIL_TRUNCATION",
    "Direction",
    "ProjectionMethod",
    "PrivacyParams",
    "SpacingType",
    "has_numba",
    "compose_full_pld",
    "discrete_distribution",
    "dp_accounting_pmf_to_pld_realization",
    "gaussian_allocation_pld",
    "gaussian_allocation_delta_configurable",
    "gaussian_allocation_epsilon_configurable",
    "gaussian_allocation_epsilon_range",
    "gaussian_distribution",
    "general_allocation_pld",
    "general_allocation_delta",
    "general_allocation_epsilon",
    "laplace_distribution",
    "project_discrete_dist",
    "subsample_pld",
    "subsample_pld_realization",
]
