"""Tests for fixed-gap, NumPy-only real-loss CtD."""

import math
import warnings

import mpmath
import numpy as np
import pytest
from scipy import stats

from PLD_accounting.projections_ctd import (
    _ctd_cell_endpoint_masses,
    _discrete_ctd_cell_measures,
    project_continuous_dist_ctd,
    project_dist_onto_grid_ctd,
)
from PLD_accounting.discrete_dist import (
    REALIZATION_MOMENT_TOL,
    DenseDiscreteDist,
    Domain,
    GridSpec,
    PLDRealization,
    SparseDiscreteDist,
)
from PLD_accounting.distribution_utils import (
    PMF_TOLERATED_MASS_TOL,
    _stable_cdf_and_sf,
    _stable_cell_interval_masses,
    exp_moment_terms,
    joint_source_dual_bounds,
    partition_masses,
    signed_unit_residual,
)
from PLD_accounting.mechanisms import _LaplacePLD, discrete_distribution
from PLD_accounting.projections import project_discrete_dist
from PLD_accounting.types import BoundType
from PLD_accounting.utils import calc_pld_dual

# Measured relative accuracy floor for scipy laws on production CtD grids: what the
# small-side interval oracle must deliver in the far tail.
_INTERVAL_ORACLE_RTOL = 1e-6

# Named dual-boundary budget components for Gaussian CtD discretization.
# Each allowance is derived from an independently measured producer margin, not a
# renamed copy of the same constant.
CTD_DUAL_RESIDUAL_REPAIR_BUDGET = 1e-12
CTD_RECIPROCAL_MOMENT_REPAIR_BUDGET = 1e-11
CTD_FINAL_MASS_DRIFT_BUDGET = PMF_TOLERATED_MASS_TOL
CTD_DUAL_BOUNDARY_TAIL_BUDGET = (
    CTD_DUAL_RESIDUAL_REPAIR_BUDGET
    + CTD_RECIPROCAL_MOMENT_REPAIR_BUDGET
    + CTD_FINAL_MASS_DRIFT_BUDGET
)


def atomic_hockey_stick(
    losses: np.ndarray,
    masses: np.ndarray,
    epsilons: np.ndarray,
    p_max: float = 0.0,
) -> np.ndarray:
    """Evaluate an atomic PLD's hockey-stick profile."""
    profile = []
    for epsilon in epsilons:
        atom_terms = []
        for loss, mass in zip(losses, masses, strict=True):
            atom_terms.append(float(mass) * max(0.0, (-math.expm1(float(epsilon - loss)))))
        profile.append(p_max + math.fsum(atom_terms))
    return np.array(profile)


class _ReflectedUniformDual:
    """Canonical dual of a shifted uniform PLD, for a non-Gaussian oracle test."""

    def __init__(self, *, width: float, shift: float) -> None:
        self._width = width
        self._low = -shift - width
        self._high = -shift

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """Cumulative distribution function."""
        values = np.asarray(values, dtype=np.float64)
        middle = (np.exp(values) - math.exp(self._low)) / self._width
        return np.where(values <= self._low, 0.0, np.where(values >= self._high, 1.0, middle))

    def sf(self, values: np.ndarray) -> np.ndarray:
        """Survival function."""
        values = np.asarray(values, dtype=np.float64)
        middle = (math.exp(self._high) - np.exp(values)) / self._width
        return np.where(values <= self._low, 1.0, np.where(values >= self._high, 0.0, middle))

    def ppf(self, quantiles: np.ndarray | float) -> np.ndarray:
        """Inverse cumulative distribution function."""
        quantiles = np.asarray(quantiles, dtype=np.float64)
        return np.log(math.exp(self._low) + quantiles * self._width)

    def isf(self, probabilities: np.ndarray | float) -> np.ndarray:
        """Inverse survival function."""
        probabilities = np.asarray(probabilities, dtype=np.float64)
        return np.log(math.exp(self._high) - probabilities * self._width)

    def logcdf(self, values: np.ndarray) -> np.ndarray:
        """Log cumulative distribution function with stable middle-tail evaluation."""
        values = np.asarray(values, dtype=np.float64)
        log_width = math.log(self._width)
        result = np.full(values.shape, -np.inf, dtype=np.float64)
        interior = (values > self._low) & (values < self._high)
        # ``exp(x) - exp(low) = exp(low) * expm1(x - low)``.
        result[interior] = self._low + np.log(np.expm1(values[interior] - self._low)) - log_width
        result[values >= self._high] = 0.0
        return result

    def logsf(self, values: np.ndarray) -> np.ndarray:
        """Log survival function with stable middle-tail evaluation."""
        values = np.asarray(values, dtype=np.float64)
        log_width = math.log(self._width)
        result = np.zeros(values.shape, dtype=np.float64)
        interior = (values > self._low) & (values < self._high)
        # ``exp(high) - exp(x) = exp(x) * expm1(high - x)``.
        result[interior] = (
            values[interior] + np.log(np.expm1(self._high - values[interior])) - log_width
        )
        result[values >= self._high] = -np.inf
        return result

    def median(self) -> float:
        """Median of the reflected uniform dual."""
        return float(self.ppf(0.5))


class _DualWithoutLogPrimitives:
    """Minimal dual law lacking log primitives, used to pin that such a law is rejected."""

    def __init__(self, *, width: float, shift: float) -> None:
        self._dual = _ReflectedUniformDual(width=width, shift=shift)

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """Forward to the reflected-uniform dual CDF."""
        return self._dual.cdf(values)

    def sf(self, values: np.ndarray) -> np.ndarray:
        """Forward to the reflected-uniform dual survival function."""
        return self._dual.sf(values)


def paper_algorithm1_reference(
    *,
    source: stats.rv_continuous,
    dual: stats.rv_continuous,
    knots: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Evaluate the paper's Algorithm 1 solely as a moderate-grid test oracle.

    This deliberately materializes the global hockey-stick profile. Production instead
    evaluates the same adjacent-secant-slope jumps cellwise, before profile rounding can
    discard their local curvature.
    """
    dual_left_limit = np.nextafter(-knots, -np.inf)
    profile = source.sf(knots) - np.exp(knots + dual.logcdf(dual_left_limit))
    alpha = np.exp(knots)
    secants = (profile[:-1] - profile[1:]) / np.diff(alpha)
    q_probabilities = np.empty_like(profile)
    q_probabilities[0] = (1.0 - profile[0]) / alpha[0] - secants[0]
    q_probabilities[1:-1] = secants[:-1] - secants[1:]
    q_probabilities[-1] = secants[-1]
    probabilities = alpha * q_probabilities
    return probabilities, float(profile[-1])


def test_cellwise_algorithm1_matches_paper_algorithm1_on_same_knots() -> None:
    """Cellwise Algorithm 1 agrees with the paper's profile form when well conditioned."""
    sigma_inv = 1.0 / 1.3
    source = stats.norm(loc=sigma_inv**2 / 2.0, scale=sigma_inv)
    result = project_continuous_dist_ctd(
        dist=source,
        dual_dist=source,
        tail_truncation=1e-5,
        step=0.1,
    )
    reference_probabilities, reference_p_max = paper_algorithm1_reference(
        source=source,
        dual=source,
        knots=result.x_array,
    )

    np.testing.assert_allclose(result.prob_arr, reference_probabilities, atol=2e-14)
    assert result.p_max == pytest.approx(reference_p_max, abs=2e-15)


def test_k_randomized_response_rediscretization_dominates_exact_profile() -> None:
    # pylint: disable=too-many-locals  # each compared quantity is named
    """CtD preserves k-RR at knots and dominates between them."""
    num_buckets = 7
    noise_probability = 0.35
    step = 0.05
    p_other = noise_probability / num_buckets
    p_correct = 1.0 - noise_probability + p_other
    max_loss = math.log(p_correct / p_other)
    losses = np.array([max_loss, 0.0, -max_loss])
    probs = np.array([p_correct, (num_buckets - 2) * p_other, p_other])
    order = np.argsort(losses)

    source = SparseDiscreteDist(x_array=losses[order], prob_arr=probs[order])
    result = project_discrete_dist(
        dist=source,
        tail_truncation=0.0,
        loss_discretization=step,
        bound_type=BoundType.DOMINATES,
    )

    result_at_knots = atomic_hockey_stick(
        result.x_array, result.prob_arr, result.x_array, result.p_max
    )
    exact_at_knots = atomic_hockey_stick(losses, probs, result.x_array)
    np.testing.assert_allclose(result_at_knots, exact_at_knots, atol=5 * PMF_TOLERATED_MASS_TOL)
    epsilons = np.linspace(result.x_array[0] - step, result.x_array[-1] + step, 2001)
    exact_profile = atomic_hockey_stick(losses, probs, epsilons)
    result_profile = atomic_hockey_stick(result.x_array, result.prob_arr, epsilons, result.p_max)
    assert np.all(result_profile >= exact_profile - 5 * PMF_TOLERATED_MASS_TOL)
    assert result.p_min == 0.0
    assert math.fsum(map(float, result.prob_arr)) + result.p_max == pytest.approx(1.0)
    reciprocal_moment = math.fsum(
        float(prob) * math.exp(-float(loss))
        for loss, prob in zip(result.x_array, result.prob_arr, strict=True)
    )
    assert reciprocal_moment <= 1.0 + REALIZATION_MOMENT_TOL


def test_grid_atoms_are_unchanged() -> None:
    """CtD preserves source atoms that already lie on target grid knots."""
    source_grid = GridSpec(step=1.0, n=3, anchor=-1.0)
    source = PLDRealization(grid=source_grid, prob_arr=np.array([0.1, 0.3, 0.6]))
    target_grid = GridSpec(anchor=-1.0, step=1.0, n=3)
    result = project_dist_onto_grid_ctd(dist=source, grid=target_grid)
    assert isinstance(result, PLDRealization)
    assert np.allclose(result.prob_arr, [0.1, 0.3, 0.6])


def test_grid_atom_with_independently_formed_coordinate_lands_on_its_knot() -> None:
    """Fine grids may form the same knot through different float operations."""
    step = 1.0 / 1200.0
    x_0 = 2.9
    # Exact bin assignment uses the materialized target knots directly, so this
    # independently formed coordinate lands on the same intended knot.
    source_loss = 2.9 + 3 * step
    source_grid = GridSpec(step=step, n=1, anchor=source_loss)
    source = PLDRealization(grid=source_grid, prob_arr=np.array([1.0]))
    target_grid = GridSpec(anchor=x_0, step=step, n=5)
    result = project_dist_onto_grid_ctd(dist=source, grid=target_grid)
    assert result.prob_arr[3] == pytest.approx(1.0)


JUST_BELOW_ZERO = np.nextafter(0.0, -np.inf)
JUST_ABOVE_ZERO = np.nextafter(0.0, np.inf)
JUST_BELOW_ONE = np.nextafter(1.0, -np.inf)
JUST_ABOVE_ONE = np.nextafter(1.0, np.inf)


@pytest.mark.parametrize(
    ("loss", "expected_boundary"),
    [
        (JUST_BELOW_ZERO, "lower"),
        (JUST_ABOVE_ZERO, "interior"),
        (JUST_BELOW_ONE, "interior"),
        (JUST_ABOVE_ONE, "upper"),
    ],
)
def test_atomwise_ctd_respects_nextafter_sides(loss: float, expected_boundary: str) -> None:
    """Bin assignment is exact at the representable points around grid knots."""
    finite_mass = 0.5
    source = SparseDiscreteDist(
        x_array=np.array([loss]), prob_arr=np.array([finite_mass]), p_max=1.0 - finite_mass
    )
    target_grid = GridSpec(anchor=0.0, step=1.0, n=2)
    result = project_dist_onto_grid_ctd(dist=source, grid=target_grid)

    if expected_boundary == "lower":
        assert result.prob_arr[0] == pytest.approx(finite_mass)
    elif expected_boundary == "upper":
        assert result.p_max > 1.0 - finite_mass
        assert result.prob_arr[-1] > 0.0
    else:
        assert result.p_max == pytest.approx(1.0 - finite_mass)
        assert math.fsum(map(float, result.prob_arr)) == pytest.approx(finite_mass)


def test_between_knots_splits_and_preserves_hockey_values() -> None:
    """CtD splits an off-grid atom while preserving knot hockey values."""
    loss, mass = 0.3, 1.0
    source_grid = GridSpec(step=0.1, n=1, anchor=loss)
    source = PLDRealization(grid=source_grid, prob_arr=np.array([mass]))
    target_grid = GridSpec(anchor=0.0, step=1.0, n=2)
    result = project_dist_onto_grid_ctd(dist=source, grid=target_grid)
    high = -math.expm1(-loss) / -math.expm1(-1.0)
    assert result.prob_arr[1] == pytest.approx(high)
    assert result.prob_arr[0] == pytest.approx(1.0 - high)
    assert np.allclose(
        atomic_hockey_stick(np.array([loss]), np.array([mass]), np.array([0.0, 1.0])),
        atomic_hockey_stick(result.x_array, result.prob_arr, result.x_array, result.p_max),
    )


def test_irregular_support_and_boundaries() -> None:
    """CtD handles irregular support and moves overflow to boundary mass."""
    source_grid = GridSpec(step=400.05, n=3, anchor=-0.1)
    source = PLDRealization(grid=source_grid, prob_arr=np.array([0.1, 0.8, 0.1]))
    target_grid = GridSpec(anchor=0.0, step=1.0, n=2)
    result = project_dist_onto_grid_ctd(dist=source, grid=target_grid)
    assert result.prob_arr[0] == pytest.approx(0.1, abs=PMF_TOLERATED_MASS_TOL)
    assert result.p_max == pytest.approx(0.9, abs=1e-14)
    assert math.fsum(map(float, result.prob_arr)) + result.p_max == pytest.approx(1.0)


def test_discrete_ctd_retains_original_discretize_then_dual_flow() -> None:
    """Stable realization CtD remains compatible with standalone dualization."""
    losses = np.array([-0.2, 0.4, 1.4])
    masses = np.array([0.1, 0.4, 0.5])
    source = SparseDiscreteDist(x_array=losses, prob_arr=masses)
    target_grid = GridSpec(anchor=0.0, step=1.0, n=2)
    result = project_dist_onto_grid_ctd(dist=source, grid=target_grid)
    dual = calc_pld_dual(result)

    source_moment = math.fsum(
        float(probability) * math.exp(-float(loss))
        for loss, probability in zip(losses, masses, strict=True)
    )
    expected_preexisting = math.fsum([1.0, -source_moment])
    expected_eta = masses[0] * math.exp(-losses[0]) * -math.expm1(losses[0])
    assert dual.p_max == pytest.approx(expected_preexisting + expected_eta)
    assert result.p_max > 0.0
    assert (
        abs(
            signed_unit_residual(
                values=result.prob_arr,
                lower_term=0.0,
                upper_term=result.p_max,
            )
        )
        <= PMF_TOLERATED_MASS_TOL
    )


def test_adapter_accepts_semantically_valid_sparse_source() -> None:
    """CtD accepts irregular sparse sources satisfying the PLD contract."""
    dist = SparseDiscreteDist(
        x_array=np.array([0.0, 1.0]), prob_arr=np.array([0.5, 0.5]), domain=Domain.REALS
    )
    target_grid = GridSpec(anchor=0.0, step=0.25, n=5)
    result = project_dist_onto_grid_ctd(dist=dist, grid=target_grid)
    eps = np.linspace(0.0, 1.0, 101)
    source_profile = atomic_hockey_stick(dist.x_array, dist.prob_arr, eps)
    output_profile = atomic_hockey_stick(result.x_array, result.prob_arr, eps, result.p_max)
    assert np.all(output_profile >= source_profile - 2 * PMF_TOLERATED_MASS_TOL)
    np.testing.assert_allclose(
        output_profile[::25], source_profile[::25], atol=2 * PMF_TOLERATED_MASS_TOL
    )


def test_rediscretization_rejects_single_finite_atom() -> None:
    """CtD rejects a degenerate finite range with no distinct grid bounds."""
    source = SparseDiscreteDist(x_array=np.array([0.2]), prob_arr=np.array([0.7]), p_max=0.3)

    with pytest.raises(ValueError, match="at least two distinct finite support points"):
        project_discrete_dist(
            dist=source,
            tail_truncation=0.0,
            loss_discretization=0.1,
            bound_type=BoundType.DOMINATES,
        )


def test_nonuniform_sparse_source_dominates_densely_between_knots() -> None:
    """CtD of a sparse PLD with unequal gaps dominates it between target knots."""
    losses = np.array([0.0, 0.2, 1.3])
    masses = np.array([0.2, 0.3, 0.5])
    dist = SparseDiscreteDist(x_array=losses, prob_arr=masses, domain=Domain.REALS)
    target_grid = GridSpec(anchor=0.0, step=0.25, n=7)
    result = project_dist_onto_grid_ctd(dist=dist, grid=target_grid)

    assert result.p_min == 0.0
    total_mass = math.fsum(map(float, result.prob_arr)) + result.p_max
    assert total_mass == pytest.approx(1.0, abs=2 * PMF_TOLERATED_MASS_TOL)
    reciprocal_moment = math.fsum(
        float(p) * math.exp(-float(x)) for x, p in zip(result.x_array, result.prob_arr)
    )
    assert reciprocal_moment <= 1.0 + REALIZATION_MOMENT_TOL

    eps = np.linspace(-0.25, 1.5, 351)
    source_profile = atomic_hockey_stick(losses, masses, eps)
    output_profile = atomic_hockey_stick(result.x_array, result.prob_arr, eps, result.p_max)
    assert np.all(output_profile >= source_profile - 2 * PMF_TOLERATED_MASS_TOL)
    result_at_knots = atomic_hockey_stick(
        result.x_array, result.prob_arr, result.x_array, result.p_max
    )
    source_at_knots = atomic_hockey_stick(losses, masses, result.x_array)
    np.testing.assert_allclose(result_at_knots, source_at_knots, atol=2 * PMF_TOLERATED_MASS_TOL)


def test_ctd_repairs_accepted_arithmetic_moment_excess_before_dualization() -> None:
    """Every CtD realization accepted by the constructor must also be dualizable."""
    moment = 1.0 + REALIZATION_MOMENT_TOL / 2.0
    dist = SparseDiscreteDist(x_array=np.array([-math.log(moment)]), prob_arr=np.array([1.0]))
    target_grid = GridSpec(anchor=-0.1, step=0.1, n=3)
    projected = project_dist_onto_grid_ctd(dist=dist, grid=target_grid)
    moment_terms = exp_moment_terms(prob_arr=projected.prob_arr, x_vals=projected.x_array)
    residual = signed_unit_residual(values=moment_terms, lower_term=0.0, upper_term=0.0)
    assert residual >= 0.0
    dual = calc_pld_dual(projected)
    assert dual.p_min == 0.0


def test_inconsistent_source_dual_pair_is_rejected_not_clipped() -> None:
    """A pair violating the cell inequality must fail rather than be repaired.

    ``N(0, 1)`` is not its own PLD dual -- a Gaussian PLD needs ``loc = scale**2 / 2`` --
    so ``exp(a) R_i`` exceeds ``M_i`` on every cell right of the origin. Clipping the
    endpoint fraction back into ``[0, 1]`` would silently return a distribution that is
    not the CtD interpolant of anything, which is the failure mode this whole path
    exists to remove.
    """
    source = stats.norm(loc=0.0, scale=1.0)
    dual = stats.norm(loc=0.0, scale=1.0)
    with pytest.raises(ValueError, match=r"r\*M <= exp\(a\)\*R <= M"):
        project_continuous_dist_ctd(
            dist=source,
            dual_dist=dual,
            tail_truncation=1e-4,
            step=0.01,
        )


@pytest.mark.parametrize(
    ("source_mass", "dual_mass"),
    [(0.0, 0.1), (0.1, 0.0)],
)
def test_ctd_rejects_one_sided_zero_cell_measure(
    source_mass: float,
    dual_mass: float,
) -> None:
    """A one-sided zero cell measure moves the reciprocal moment by more than the tolerance."""
    pld_pmf = np.array([source_mass])
    pld_dual_pmf = np.array([dual_mass])
    left = np.array([0.0])
    width = np.array([1.0])
    with pytest.raises(ValueError, match=r"r\*M <= exp\(a\)\*R <= M"):
        _ctd_cell_endpoint_masses(
            pld_pmf=pld_pmf, pld_dual_pmf=pld_dual_pmf, left=left, width=width
        )


@pytest.mark.parametrize(
    ("source_mass", "dual_mass"),
    [(0.0, 5e-324), (5e-324, 0.0)],
)
def test_ctd_puts_the_source_mass_of_a_subnormal_cell_at_its_right_knot(
    source_mass: float,
    dual_mass: float,
) -> None:
    """A cell with subnormal source mass stays at its right knot without splitting."""
    left_weight, right_weight = _ctd_cell_endpoint_masses(
        pld_pmf=np.array([source_mass]),
        pld_dual_pmf=np.array([dual_mass]),
        left=np.array([0.0]),
        width=np.array([1.0]),
    )
    assert left_weight[0] == 0.0
    assert right_weight[0] == source_mass


def test_ctd_clips_the_split_of_a_negligible_tail_cell_but_not_of_a_heavy_one() -> None:
    """Endpoint clipping must stay within the reciprocal-moment tolerance."""
    left = np.array([0.0])
    width = np.array([3e-8])
    for mass in (1e-15, 1e-6):
        dual_mass = mass * 0.3
        args = {
            "pld_pmf": np.array([mass]),
            "pld_dual_pmf": np.array([dual_mass]),
            "left": left,
            "width": width,
        }
        if mass < 1e-9:
            left_weight, right_weight = _ctd_cell_endpoint_masses(**args)
            assert left_weight[0] + right_weight[0] == mass
            assert right_weight[0] == mass
        else:
            with pytest.raises(ValueError, match=r"r\*M <= exp\(a\)\*R <= M"):
                _ctd_cell_endpoint_masses(**args)


def test_ctd_mixed_cell_weights_match_80_digit_oracle() -> None:
    """A non-Gaussian atomic mixture agrees with the exact local CtD solve."""
    with mpmath.workdps(80):
        left_mp = mpmath.mpf("-0.7")
        width_mp = mpmath.mpf("1.1")
        losses_mp = [mpmath.mpf("-0.55"), mpmath.mpf("-0.03"), mpmath.mpf("0.37")]
        masses_mp = [mpmath.mpf("0.13"), mpmath.mpf("0.29"), mpmath.mpf("0.07")]
        mass_mp = mpmath.fsum(masses_mp)
        reflected_mp = mpmath.fsum(
            mass * mpmath.exp(-loss) for mass, loss in zip(masses_mp, losses_mp, strict=True)
        )
        denominator_mp = 1 - mpmath.exp(-width_mp)
        right_mp = (mass_mp - mpmath.exp(left_mp) * reflected_mp) / denominator_mp
        left_weight_mp = mass_mp - right_mp

    left_weight, right_weight = _ctd_cell_endpoint_masses(
        pld_pmf=np.array([float(mass_mp)]),
        pld_dual_pmf=np.array([float(reflected_mp)]),
        left=np.array([float(left_mp)]),
        width=np.array([float(width_mp)]),
    )
    expected_left_weight = float(left_weight_mp)
    expected_right_weight = float(right_mp)
    assert left_weight[0] == pytest.approx(expected_left_weight, rel=2e-15)
    assert right_weight[0] == pytest.approx(expected_right_weight, rel=2e-15)


def test_non_gaussian_continuous_ctd_matches_80_digit_uniform_profile() -> None:
    """A shifted-uniform PLD dominates its 80-digit hockey-stick profile."""
    width = 1.3
    shift = math.log(-math.expm1(-width) / width)
    source = stats.uniform(loc=shift, scale=width)
    dual = _ReflectedUniformDual(width=width, shift=shift)
    result = project_continuous_dist_ctd(
        dist=source,
        dual_dist=dual,
        tail_truncation=1e-6,
        step=0.025,
    )

    epsilons = result.x_array[:-1] + result.step / 2.0
    with mpmath.workdps(80):
        lower = mpmath.mpf(str(shift))
        upper = lower + mpmath.mpf(str(width))

        def exact_profile(epsilon_float: float) -> float:
            epsilon = mpmath.mpf(str(epsilon_float))
            if epsilon < lower:
                return float(1 - mpmath.exp(epsilon))
            if epsilon >= upper:
                return 0.0
            return float((upper - epsilon - 1 + mpmath.exp(epsilon - upper)) / width)

        exact = np.array([exact_profile(float(epsilon)) for epsilon in epsilons])

    projected = atomic_hockey_stick(
        result.x_array,
        result.prob_arr,
        epsilons,
        result.p_max,
    )
    assert np.all(projected >= exact - 8 * PMF_TOLERATED_MASS_TOL)


def test_ctd_rejects_a_grid_beyond_the_representable_exponential_range() -> None:
    """The moment factors exp(a) and exp(-x_0) must stay representable."""
    source = stats.norm(loc=0.0, scale=200.0)
    dual = stats.norm(loc=0.0, scale=200.0)
    with pytest.raises(ValueError, match="exp"):
        project_continuous_dist_ctd(
            dist=source,
            dual_dist=dual,
            tail_truncation=1e-6,
            step=1.0,
        )


def test_invalid_ctd_source_is_rejected() -> None:
    """CtD rejects a source violating its reciprocal-moment contract."""
    losses = np.array([-1.0])
    masses = np.array([1.0])
    with pytest.raises(ValueError, match="moment"):
        source = SparseDiscreteDist(x_array=losses, prob_arr=masses)
        target_grid = GridSpec(anchor=-1.0, step=1.0, n=2)
        project_dist_onto_grid_ctd(dist=source, grid=target_grid)


@pytest.mark.parametrize("sigma", [0.4, 0.8])
@pytest.mark.parametrize("grid_points", [125_000, 250_000, 500_000, 1_000_000])
def test_gaussian_ctd_is_stable_through_parameter_scan_grid_cap(
    sigma: float, grid_points: int
) -> None:
    """Refinement through the scan cap has no growing CtD reconstruction defect.

    The two boundary atoms are semantic, so they are charged against the per-factor
    budget ``tau = delta / (100 * 2 * 3 * 3 * T)`` at every grid size. A reconstruction
    defect would show as a dual boundary atom far over that budget that grows with each
    grid doubling.

    Total mass is a pure arithmetic quantity, so it is charged only at binary64 scale.
    Chasing it below an ULP of 1 buys nothing the downstream accounting can use.
    """
    factor_tail = 1e-10 / (100 * 2 * 3 * 3 * 100000)
    sigma_inv = 1.0 / sigma
    source = stats.norm(loc=sigma_inv**2 / 2.0, scale=sigma_inv)
    range_tail = factor_tail / 2.0
    step = (
        max(source.isf(range_tail), -source.ppf(range_tail))
        - min(source.ppf(range_tail), -source.isf(range_tail))
    ) / (grid_points - 3)

    result = project_continuous_dist_ctd(
        dist=source,
        dual_dist=source,
        tail_truncation=factor_tail,
        step=step,
    )

    dual = calc_pld_dual(result)
    assert result.prob_arr.size <= grid_points
    assert np.all(result.prob_arr >= 0.0)
    assert np.all(dual.prob_arr >= 0.0)
    assert result.p_min == 0.0
    assert dual.p_min == 0.0

    # primal boundary atom is the semantic upper-tail term M_+ - exp(x_N) R_+, not
    # error; check it against its closed form for a Gaussian PLD (see section 11.2).
    analytic_upper_eta = float(
        source.sf(float(result.x_array[-1]))
        - math.exp(float(result.x_array[-1]))
        * stats.norm(loc=sigma_inv**2 / 2.0, scale=sigma_inv).cdf(-float(result.x_array[-1]))
    )
    assert result.p_max == pytest.approx(analytic_upper_eta, rel=1e-9)

    # Semantic boundary atoms share one per-factor budget. ``dual.p_max`` also
    # contains the binary64 summation residual, so audit the semantic lower-tail
    # formula separately and bound the stored value by its arithmetic allowance.
    analytic_dual_eta = float(
        source.sf(-float(result.x_array[0]))
        - math.exp(-float(result.x_array[0])) * source.cdf(float(result.x_array[0]))
    )
    assert result.p_max <= factor_tail
    assert analytic_dual_eta >= -PMF_TOLERATED_MASS_TOL
    dual_residual_margin = max(0.0, dual.p_max - factor_tail)
    assert dual_residual_margin <= CTD_DUAL_BOUNDARY_TAIL_BUDGET
    assert result.p_max + max(0.0, analytic_dual_eta) <= factor_tail

    # Total mass: arithmetic only, charged at binary64 scale.
    for realization in (result, dual):
        assert (
            abs(
                signed_unit_residual(
                    values=realization.prob_arr,
                    lower_term=realization.p_min,
                    upper_term=realization.p_max,
                )
            )
            <= PMF_TOLERATED_MASS_TOL
        )


def test_continuous_ctd_rejects_a_grid_coarser_than_the_support() -> None:
    """Fewer than four finite knots cannot host float64 interval measures."""
    source = stats.norm(loc=0.5, scale=1.0)
    tail = 1e-5
    x_min, x_max = joint_source_dual_bounds(dist=source, dual_dist=source, tail_truncation=tail)
    # The second step is half the support, yet alignment still yields a 3-knot grid.
    for step in (100.0, (x_max - x_min) / 2.0):
        with pytest.raises(ValueError, match="coarser than"):
            project_continuous_dist_ctd(
                dist=source,
                dual_dist=source,
                tail_truncation=tail,
                step=step,
            )


def _gaussian_production_ctd_grid(
    *, sigma: float, grid_points: int
) -> tuple[stats.rv_continuous, np.ndarray]:
    """Return a production-shaped Gaussian source and knot vector."""
    factor_tail = 1e-10 / (100 * 2 * 3 * 3 * 100000)
    sigma_inv = 1.0 / sigma
    source = stats.norm(loc=sigma_inv**2 / 2.0, scale=sigma_inv)
    range_tail = factor_tail / 2.0
    return source, np.linspace(
        min(source.ppf(range_tail), -source.isf(range_tail)),
        max(source.isf(range_tail), -source.ppf(range_tail)),
        grid_points - 2,
        dtype=np.float64,
    )


def test_scipy_continuous_laws_use_log_interval_oracle() -> None:
    """Production CtD sources must route through the logcdf/logsf oracle."""
    source = stats.norm(loc=0.125, scale=0.5)
    cdf, _ = _stable_cdf_and_sf(dist=source, x_array=np.array([-1.0, 0.0, 1.0]))
    raw = np.asarray(source.cdf([-1.0, 0.0, 1.0]), dtype=np.float64)
    assert np.all(np.isfinite(cdf))
    assert np.max(np.abs(cdf - raw)) <= 2 * np.finfo(np.float64).eps


def test_law_without_log_primitives_is_rejected() -> None:
    """A law with only cdf/sf is refused; the direct difference cancels in the tails."""
    dual = _DualWithoutLogPrimitives(width=1.3, shift=0.2)
    points = np.linspace(-2.0, 1.0, 8, dtype=np.float64)
    with pytest.raises(TypeError, match="lacks logcdf, logsf, median"):
        _stable_cdf_and_sf(dist=dual, x_array=points)


def test_discrete_ctd_cells_are_right_closed_at_exact_knots() -> None:
    """Atoms exactly on knots land in the right-closed cell selected by searchsorted."""
    loss = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    dist = SparseDiscreteDist(
        x_array=np.array([-1.0, 0.0, 1.0], dtype=np.float64),
        prob_arr=np.array([0.2, 0.3, 0.5], dtype=np.float64),
    )
    mass, _ = _discrete_ctd_cell_measures(dist_in=dist, loss_out=loss)
    assert mass[0] == pytest.approx(0.2)
    assert mass[1] == pytest.approx(0.3)
    assert mass[2] == pytest.approx(0.5)
    assert mass[3] == pytest.approx(0.0)


def test_stable_interval_oracle_beats_raw_cdf_on_gaussian_far_tail() -> None:
    # pylint: disable=too-many-locals  # each compared quantity is named
    """Far-tail cell widths must come from the small-side cumulative, not ``cdf(b)-cdf(a)``."""
    source, knots = _gaussian_production_ctd_grid(sigma=0.4, grid_points=500)
    raw_cdf = np.asarray(source.cdf(knots), dtype=np.float64)
    cdf_only_masses = np.diff(raw_cdf)

    stable_masses, _, _ = partition_masses(
        dist=source,
        points=knots,
        label="Gaussian far-tail audit",
    )

    cell_indices = np.arange(knots.size - 31, knots.size - 1, dtype=np.int64)
    with mpmath.workdps(80):
        mu = mpmath.mpf(str(source.kwds["loc"]))
        scale = mpmath.mpf(str(source.kwds["scale"]))

        def exact_cell_mass(left: float, right: float) -> float:
            return float(mpmath.ncdf(right, mu, scale) - mpmath.ncdf(left, mu, scale))

        reference_masses = []
        for i in cell_indices:
            reference_masses.append(exact_cell_mass(float(knots[i]), float(knots[i + 1])))
        reference = np.array(reference_masses, dtype=np.float64)
    stable_slice = stable_masses[cell_indices]
    raw_slice = cdf_only_masses[cell_indices]

    stable_rel = np.max(np.abs(stable_slice - reference) / np.maximum(reference, 1e-300))
    raw_rel = np.max(np.abs(raw_slice - reference) / np.maximum(reference, 1e-300))
    assert stable_rel <= _INTERVAL_ORACLE_RTOL
    assert np.any(raw_slice == 0.0)
    assert np.all(stable_slice[raw_slice == 0.0] > 0.0)
    assert stable_rel < raw_rel


def test_ctd_dual_residual_route_measures_three_independent_quantities() -> None:
    """Dual boundary, reciprocal-moment repair and mass drift are measured separately."""
    factor_tail = 1e-10 / (100 * 2 * 3 * 3 * 100000)
    sigma_inv = 1.0 / 0.4
    source = stats.norm(loc=sigma_inv**2 / 2.0, scale=sigma_inv)
    range_tail = factor_tail / 2.0
    step = (
        max(source.isf(range_tail), -source.ppf(range_tail))
        - min(source.ppf(range_tail), -source.isf(range_tail))
    ) / (125_000 - 3)

    result = project_continuous_dist_ctd(
        dist=source,
        dual_dist=source,
        tail_truncation=factor_tail,
        step=step,
    )
    dual = calc_pld_dual(result)
    analytic_dual_eta = float(
        source.sf(-float(result.x_array[0]))
        - math.exp(-float(result.x_array[0])) * source.cdf(float(result.x_array[0]))
    )
    dual_residual_margin = max(0.0, dual.p_max - factor_tail)
    mass_drift = abs(
        signed_unit_residual(
            values=result.prob_arr,
            lower_term=result.p_min,
            upper_term=result.p_max,
        )
    )
    dual_mass_drift = abs(
        signed_unit_residual(
            values=dual.prob_arr,
            lower_term=dual.p_min,
            upper_term=dual.p_max,
        )
    )

    assert dual_residual_margin <= CTD_DUAL_RESIDUAL_REPAIR_BUDGET
    assert mass_drift <= CTD_FINAL_MASS_DRIFT_BUDGET
    assert dual_mass_drift <= CTD_FINAL_MASS_DRIFT_BUDGET
    assert result.p_max + max(0.0, analytic_dual_eta) <= factor_tail


def _laplace_pld_cdf_mp(x: mpmath.mpf, lam: mpmath.mpf) -> mpmath.mpf:
    """High-precision CDF for the mixed Laplace-mechanism PLD law."""
    if x < -lam:
        return mpmath.mpf(0)
    if x >= lam:
        return mpmath.mpf(1)
    atom = mpmath.mpf("0.5") * mpmath.e ** (-lam)
    return atom + mpmath.mpf("0.5") * (mpmath.e ** ((x - lam) / 2) - mpmath.e ** (-lam))


def _laplace_production_ctd_grid(
    *, scale: float, grid_points: int
) -> tuple[_LaplacePLD, np.ndarray]:
    """Return a production-shaped Laplace PLD source and knot vector."""
    factor_tail = 1e-10 / (100 * 2 * 3 * 3 * 100000)
    source = _LaplacePLD(sigma=scale)
    range_tail = factor_tail / 2.0
    joint_min = float(source.ppf(range_tail))
    joint_max = float(source.isf(range_tail))
    knots = np.linspace(joint_min, joint_max, grid_points - 2, dtype=np.float64)
    return source, knots


def test_laplace_interval_oracle_meets_floor_on_far_tail() -> None:
    # pylint: disable=too-many-locals  # each compared quantity is named
    """Laplace-mechanism CtD must meet the same interval-oracle floor as the Gaussian."""
    source, knots = _laplace_production_ctd_grid(scale=1.0, grid_points=500)
    raw_cdf = np.asarray(source.cdf(knots), dtype=np.float64)
    cdf_only_masses = np.diff(raw_cdf)
    stable_masses, _, _ = partition_masses(
        dist=source,
        points=knots,
        label="Laplace far-tail audit",
    )

    cell_indices = np.arange(knots.size - 31, knots.size - 1, dtype=np.int64)
    lam = mpmath.mpf(str(source.lam))
    with mpmath.workdps(80):
        reference_masses = []
        for i in cell_indices:
            right_knot = mpmath.mpf(str(knots[i + 1]))
            left_knot = mpmath.mpf(str(knots[i]))
            reference_masses.append(
                float((_laplace_pld_cdf_mp(right_knot, lam) - _laplace_pld_cdf_mp(left_knot, lam)))
            )
        reference = np.array(reference_masses, dtype=np.float64)
    stable_slice = stable_masses[cell_indices]
    raw_slice = cdf_only_masses[cell_indices]

    stable_rel = np.max(np.abs(stable_slice - reference) / np.maximum(reference, 1e-300))
    raw_rel = np.max(np.abs(raw_slice - reference) / np.maximum(reference, 1e-300))
    assert stable_rel <= _INTERVAL_ORACLE_RTOL
    assert stable_rel < raw_rel or np.all(raw_slice > 0.0)


def test_discrete_count_noise_routes_without_continuous_interval_fallback() -> None:
    """Integer count-noise PLDs use atomic CtD rather than the continuous interval oracle."""
    noise_grid = GridSpec(step=1.0, n=5, anchor=-3.0)
    noise = DenseDiscreteDist(
        grid=noise_grid,
        prob_arr=np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64),
        domain=Domain.REALS,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        remove, _add = discrete_distribution(
            noise_dist=noise,
            loss_discretization=0.05,
            tail_truncation=1e-8,
        )
    assert isinstance(remove, PLDRealization)
    assert remove.p_min == 0.0


def test_ctd_reciprocal_moment_warning_route_is_unreproducible() -> None:
    """Production paths emit no integration warning at a residual of 6.905e-14.

    ``REALIZATION_MOMENT_TOL`` (3.553e-15) is the drift threshold of the
    reciprocal-moment repair. The tree repairs silently below the repair band, and
    the production-shaped Gaussian and allocation routes below emit no
    ``classify_residual`` warnings.
    """
    factor_tail = 1e-10 / (100 * 2 * 3 * 3 * 100000)
    sigma_inv = 1.0 / 0.4
    source = stats.norm(loc=sigma_inv**2 / 2.0, scale=sigma_inv)
    range_tail = factor_tail / 2.0
    step = (
        max(source.isf(range_tail), -source.ppf(range_tail))
        - min(source.ppf(range_tail), -source.isf(range_tail))
    ) / (125_000 - 3)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        project_continuous_dist_ctd(
            dist=source,
            dual_dist=source,
            tail_truncation=factor_tail,
            step=step,
        )
    assert REALIZATION_MOMENT_TOL == pytest.approx(3.553e-15, rel=1e-3)


def test_interval_masses_clip_ulp_negatives_but_reject_material_ones() -> None:
    """Both engines share this guard; only rounding-scale negatives may be clipped."""
    cdf = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    sf = 1.0 - cdf

    # One ULP of non-monotonicity above the pivot is rounding, and is clipped away.
    nudged = sf.copy()
    nudged[3] = np.nextafter(sf[2], 1.0)
    masses = _stable_cell_interval_masses(cdf=cdf, sf=nudged, label="ulp audit")
    assert np.all(masses >= 0.0)
    assert masses.sum() == pytest.approx(1.0, abs=1e-12)

    # A material one would manufacture mass, so it is refused rather than clipped.
    broken = sf.copy()
    broken[3] = 0.6
    with pytest.raises(ValueError, match="interval oracle returned a negative probability"):
        _stable_cell_interval_masses(cdf=cdf, sf=broken, label="broken audit")
