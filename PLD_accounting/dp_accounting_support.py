"""
dp_accounting compatibility wrappers for subsampling implementation.

Provides translation between dp_accounting's PrivacyLossDistribution objects
and this project's structured discrete-distribution API.
"""
import numpy as np

from dp_accounting.pld.pld_pmf import DensePLDPmf, PLDPmf, SparsePLDPmf

from PLD_accounting.discrete_dist import LinearDiscreteDist


# ============================================================================
# Translation Functions: discrete distributions <-> dp_accounting
# ============================================================================

def discrete_dist_to_dp_accounting_pmf(dist: LinearDiscreteDist, pessimistic_estimate: bool = True) -> DensePLDPmf:
    """Convert a dense linear discrete distribution to dp_accounting PMF.

    Args:
        dist: Must be LinearDiscreteDist
        pessimistic_estimate: Whether to use pessimistic estimate in dp_accounting

    Returns:
        DensePLDPmf object

    Raises:
        TypeError: If dist is not a LinearDiscreteDist

    Notes:
        - Infinity masses are handled via p_pos_inf
        - The dp_accounting library requires uniformly-spaced linear grids
        - x_min must be aligned to x_gap multiples
    """
    if not isinstance(dist, LinearDiscreteDist):
        raise TypeError(
            f"discrete_dist_to_dp_accounting_pmf requires LinearDiscreteDist, got {type(dist)}. "
            f"The dp_accounting library requires uniformly-spaced linear grids. "
            f"Use change_spacing_type() to convert to linear if needed."
        )

    base_index = int(np.rint(dist.x_min / dist.x_gap))
    if not np.isclose(base_index * dist.x_gap, dist.x_min, atol=1e-12, rtol=1e-8):
        raise ValueError("LinearDiscreteDist x_min is not aligned to x_gap multiples")
    return DensePLDPmf(
        discretization=dist.x_gap,
        lower_loss=base_index,
        probs=dist.PMF_array.astype(np.float64),
        infinity_mass=dist.p_pos_inf,
        pessimistic_estimate=pessimistic_estimate,
    )



def dp_accounting_pmf_to_discrete_dist(pmf: PLDPmf) -> LinearDiscreteDist:
    """Convert dp_accounting PMF to a dense linear distribution.

    Notes:
        - Infinity mass from dp_accounting becomes p_pos_inf in returned distribution
        - p_neg_inf is set to 0 (not used in privacy loss distributions)
        - SparsePLDPmf inputs are densified to LinearDiscreteDist
    """

    # Extract dense loss grid and probabilities from PMF
    if isinstance(pmf, DensePLDPmf):
        probs = np.clip(pmf._probs.copy(), 0.0, 1.0)
        finite_target = max(0.0, 1.0 - pmf._infinity_mass)
        sum_probs = float(np.sum(probs, dtype=np.float64))
        if sum_probs > 0.0:
            probs = probs * (finite_target / sum_probs)

        return LinearDiscreteDist(
            x_min=float(pmf._lower_loss) * pmf._discretization,
            x_gap=pmf._discretization,
            PMF_array=probs,
            p_neg_inf=0.0,
            p_pos_inf=pmf._infinity_mass,
        )
    elif isinstance(pmf, SparsePLDPmf):
        loss_probs = pmf._loss_probs.copy()
        if len(loss_probs) == 0:
            raise ValueError("Empty dp_accounting PMF is not supported")

        loss_indices = np.array(sorted(loss_probs.keys()), dtype=np.int64)
        probs_sparse = np.array([loss_probs[int(idx)] for idx in loss_indices], dtype=np.float64)
        probs_sparse = np.clip(probs_sparse, 0.0, 1.0)

        finite_target = max(0.0, 1.0 - pmf._infinity_mass)
        sum_probs = float(np.sum(probs_sparse, dtype=np.float64))
        if sum_probs > 0.0:
            probs_sparse = probs_sparse * (finite_target / sum_probs)

        min_index = int(loss_indices[0])
        max_index = int(loss_indices[-1])

        # Densify the sparse PMF
        dense_size = max_index - min_index + 1
        probs_dense = np.zeros(dense_size, dtype=np.float64)
        for idx, prob in zip(loss_indices, probs_sparse):
            probs_dense[int(idx - min_index)] = float(prob)

        return LinearDiscreteDist(
            x_min=float(min_index) * pmf._discretization,
            x_gap=pmf._discretization,
            PMF_array=probs_dense,
            p_neg_inf=0.0,
            p_pos_inf=pmf._infinity_mass,
        )
    else:
        raise AttributeError(f"Unrecognized PMF format: {type(pmf)}. Expected DensePLDPmf or SparsePLDPmf.")
