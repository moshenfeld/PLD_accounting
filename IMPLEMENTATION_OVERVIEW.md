# Implementation Overview

This document describes the internal structure of `PLD_accounting` and how the implementation maps to the paper's random-allocation setting.

For user-facing examples, see [README.md](README.md) and
[PLD_accounting_tutorial.ipynb](PLD_accounting_tutorial.ipynb).

`numba` is an optional performance dependency. The implementation dispatches to
NumPy fallbacks when it is unavailable.

## Paper-Aligned Semantics

The package follows the `k`-out-of-`t` random-allocation language:

- In each epoch, a record participates in `k` selected steps out of `t` total steps.
- This is repeated for `num_epochs` epochs.

API parameter mapping:

- `num_steps = t`
- `num_selected = k`
- `num_epochs = number of epochs`

In code, this decomposition is implemented in
`allocation_directional_pld()` in `PLD_accounting/random_allocation_accounting.py`:

- Floor component:
  - `floor_steps = floor(num_steps / num_selected)`
  - `remainder = num_steps - num_selected * floor_steps`
  - `floor_epochs = (num_selected - remainder) * num_epochs`
- Ceil component (only when `remainder > 0`):
  - `ceil_steps = floor_steps + 1`
  - `ceil_epochs = remainder * num_epochs`

Both Gaussian and realization paths use the same floor/ceil decomposition and
compose both components when needed.

Input validation in `allocation_directional_pld()`:

- `num_steps`, `num_selected`, and `num_epochs` must be at least `1`.
- `num_steps` must be at least `num_selected` to ensure at least one
  per-selection step.

## High-Level Pipeline

1. Public API validates inputs and builds PMFs for REMOVE and ADD directions.
2. Per-round random-allocation PMFs are computed in loss-space via exp-space convolution helpers.
3. Floor/ceil PMF components are composed across their epoch counts and
   combined when both are present.
4. Final PMFs are converted to `dp_accounting.PrivacyLossDistribution`.
5. Epsilon/delta queries are answered on that PLD object.

Both input modes share this shape:

- Gaussian mode: starts from analytic log-normal factors.
- Realization mode: starts from user-provided `PLDRealization`.

## Canonical Distribution Model

The active runtime is built on:

- `DenseDiscreteDist`: regular-grid distribution with `x_0`, `step`,
  `spacing_type`, `prob_arr`, `p_min`, and `p_max`.
- `SparseDiscreteDist`: explicit-support distribution with `x_array`,
  `prob_arr`, `p_min`, and `p_max`.
- `PLDRealization`: linear-grid specialization for privacy-loss space.

Boundary semantics depend on `Domain`:

- `Domain.REALS`:
  - `p_min` is mass at `-inf`
  - `p_max` is mass at `+inf`
- `Domain.POSITIVES`:
  - `p_min` is mass at `0`
  - `p_max` is mass at `+inf`

CtD is the fixed engine for dominating fixed-gap real-loss construction. It
reconstructs a PLD from its hockey-stick profile; there is no public method
selector. Lower bounds, FFT exp-space factors, and geometric-grid
regridding use separate stochastic-domination functions because CtD is not
defined for those representations. A CtD target grid must be linear and
fixed-gap. A discrete source may be dense or sparse, but must use
`Domain.REALS`, have exact `p_min == 0`, finite strictly ordered support,
conserved mass, and reciprocal moment `E[exp(-L)] <= 1` under the shared
realization tolerance. These checks certify the object as a realizable PLD,
not its domination relationship to an external mechanism.

Linear real-loss callers route through the public
`rediscretize_dist_by_bound` helper. Bound direction alone selects the
engine: `DOMINATES` uses CtD and `IS_DOMINATED` uses stochastic domination.

CtD is used because repeatedly rounding every atom upward can accumulate an
`O(compositions * step)` loss-space shift. CtD instead evaluates the source's
hockey-stick profile on the target knots and inverts those values to a
fixed-gap PLD. Convex interpolation in `exp(epsilon)` preserves the upper-bound
profile without the systematic one-cell shift at every projection. This is a
structural domination argument, not a universal accuracy guarantee; numerical
accuracy must still be assessed against converged or independent references.

The CtD source check is intentionally local to
`distribution_discretization.py`: it validates the algorithm-specific semantic
contract after generic distribution constructors have already checked shape,
mass, and boundary nonnegativity. In particular, CtD cannot accept lower
infinite mass, non-real support, unordered support, or a reciprocal moment
above one. The reciprocal moment is `E[exp(-L)]`, the total mass of the implied
dual law on the source's finite support. It is also used when validating the
negative-epsilon profile floor `delta(epsilon) >= 1 - exp(epsilon) E[exp(-L)]`;
using the measured value gives the correct stronger floor when the dual law is
a sub-probability because of singular mass.

For continuous and mixed laws, an atom at `L = epsilon` contributes exactly
zero to the hockey-stick integrand. The implementation uses the strict identity
`delta(epsilon) = Pr[L > epsilon] - exp(epsilon) Pr[D < -epsilon]`, evaluating
the dual CDF immediately below `-epsilon` with `nextafter` so an atom at the
threshold is excluded while retaining the standard `sf`/`logcdf` interfaces.

## Parameter Budget Conventions

Shared composition budgets are derived inline in `allocation_directional_pld()`
and `_allocation_directional_pld_core()` in
`PLD_accounting/random_allocation_accounting.py`.

- `allocation_directional_pld()` divides the tail budget by the number of active
  tail-consuming ops, `2 * component_count - 1`: one component core call per
  floor/ceil component plus the final `fft_convolve` when both are active.
  When both components are active, the loss budget is split between them
  proportional to each component's effective discretization count
  (`num_epochs * base_loss_discretization_count(num_steps)`), so both components
  land on exactly the same output step while their budgets still sum to
  `loss_discretization`.
- `_allocation_directional_pld_core()` divides the component tail budget by 3
  (base creation, epoch composition, final truncation) when `num_epochs > 1`,
  or by 2 when the FFT phase is skipped (`num_epochs == 1`).
  `base_tail_truncation = tail_truncation / (3 * num_epochs)` covers up to 3
  tail-consuming sub-ops per epoch; `base_loss_discretization =
  loss_discretization / num_epochs` absorbs linear-in-epochs quantization
  growth (no division when `num_epochs == 1`).
- GEOM base builds (`geometric_allocation_pld_base_add/remove()`) divide their
  loss budget by the exact discretizing stage count
  (`add/remove_geometric_loss_discretization_count(num_steps)`) and their tail
  budget by the phase count (2 for ADD, 3 for REMOVE), with each one-step
  factor receiving a further `1 / num_steps` to absorb self-convolution
  amplification.

Gaussian FFT path needs additional one-step parameters for discretizing analytic
continuous factors. These are derived in
`_gaussian_allocation_fft()` in `PLD_accounting/random_allocation_gaussian.py`:

- `single_step_tail_truncation = tail_truncation / num_steps`
  with a numerical-stability floor (`eps * 1e-10`) chosen empirically as a
  reasonable value (no strict derivation).
- Per-factor FFT tail allocation:
  REMOVE uses `single_step_tail_truncation / 2` (lower + upper factors),
  ADD uses no extra split.

Gaussian GEOM path now mirrors realization wiring after factor creation:
- both routes call shared `geometric_allocation_pld_base_add/remove(...)`;
- only the base distribution creation differs (analytic Gaussian vs explicit realization).

Realization path uses the same depth factor for component-level loss
discretization before shared composition finalization.

## File Map

| File | Responsibility |
|---|---|
| `PLD_accounting/__init__.py` | Public exports. |
| `PLD_accounting/types.py` | Enums and configs (`PrivacyParams`, `AllocationSchemeConfig`, `BoundType`, etc.). |
| `PLD_accounting/random_allocation_api.py` | Public entry points for Gaussian and realization accounting. |
| `PLD_accounting/random_allocation_accounting.py` | Shared composition/finalization helpers used by both Gaussian and realization paths. |
| `PLD_accounting/random_allocation_gaussian.py` | Gaussian-specific factor construction and convolution method selection. |
| `PLD_accounting/random_allocation_realization.py` | Realization-specific factor construction from `PLDRealization` inputs. |
| `PLD_accounting/adaptive_random_allocation.py` | Adaptive upper/lower range refinement for epsilon/delta queries. |
| `PLD_accounting/mechanisms.py` | Mechanism PLD factory helpers (`gaussian_distribution`, `laplace_distribution`, `discrete_distribution`). |
| `PLD_accounting/validation.py` | Centralized input validation (`validate_privacy_params`, `validate_allocation_params`, etc.). |
| `PLD_accounting/discrete_dist.py` | Distribution classes (`DenseDiscreteDist`, `SparseDiscreteDist`, `PLDRealization`, `Domain`). |
| `PLD_accounting/distribution_discretization.py` | Continuous-to-discrete conversion and spacing changes (linear/geometric). |
| `PLD_accounting/fft_convolution.py` | FFT-based convolution and self-convolution on linear grids. |
| `PLD_accounting/geometric_convolution.py` | Convolution and self-convolution on geometric grids. |
| `PLD_accounting/utils.py` | PLD transforms (`exp`, `log`, dual, negate-reverse, composition helpers). |
| `PLD_accounting/distribution_utils.py` | Numerical utilities (mass conservation, spacing checks, stable comparisons). |
| `PLD_accounting/dp_accounting_support.py` | Conversion between internal probability representations and `dp_accounting` PMF/PLD types. |
| `PLD_accounting/subsample_pld.py` | PLD-level subsampling amplification helpers (DOMINATES-only path). |

## Public API Surface

Random allocation (defined in `PLD_accounting/random_allocation_api.py`):

- Gaussian path:
  - `gaussian_allocation_pld(...)`
  - `gaussian_allocation_epsilon_configurable(...)`
  - `gaussian_allocation_delta_configurable(...)`
  - `gaussian_allocation_epsilon_range(...)`
- Realization path:
  - `general_allocation_pld(...)`
  - `general_allocation_epsilon(...)`
  - `general_allocation_delta(...)`

Mechanism PLD helpers (defined in `PLD_accounting/mechanisms.py`):

- `gaussian_distribution(scale, value_discretization, tail_truncation, bound_type)`
- `laplace_distribution(scale, value_discretization, tail_truncation, bound_type)`
- `discrete_distribution(*, noise_dist, loss_discretization, tail_truncation, sensitivity=1)`
  - Maps both input noise-boundary masses (`p_min` and `p_max`) to positive-infinity privacy loss; they are separate unmatched tails and need not agree.

Subsampling (defined in `PLD_accounting/subsample_pld.py`):

- `subsample_pld(pld, sampling_probability)`
- `subsample_pld_realization(base_pld, sampling_prob, direction)`

Distribution type (defined in `PLD_accounting/discrete_dist.py`):

- `DenseDiscreteDist` — regular-grid distribution used to describe integer count noise.
- `PLDRealization` — linear-grid privacy-loss distribution used as input to realization APIs.

Notes:

- PLD builders reject `BoundType.BOTH`; users build separate DOMINATES and IS_DOMINATED PLDs.
- Realization-based allocation requires `ConvolutionMethod.GEOM`.

## Core Composition Modules

### `random_allocation_accounting.py`

This is the shared composition core used by both Gaussian and realization accounting.

Key functions:

- `_allocation_directional_pld_core(...)`:
  Calls a base-PLD callback, regrids to core resolution, composes across
  epochs, then regrids to output discretization.
- `geometric_allocation_pld_base_remove(...)`:
  Shared exp-space geometric composer for REMOVE. Accepts a callback that
  builds lower/upper loss factors.
- `geometric_allocation_pld_base_add(...)`:
  Shared exp-space geometric composer for ADD. Accepts a callback that builds
  the add loss factor.
- `allocation_directional_pld(...)`:
  Applies adaptive step decomposition and composes floor/ceil components.
- `compose_full_pld(...)`:
  Converts internal directional PLDs into a `dp_accounting` PLD object.

### `random_allocation_realization.py`

Realization-specific path that starts from explicit `PLDRealization` factors and
then reuses shared composition logic.

Key functions:

- `realization_remove_base_distributions(...)`: prepares REMOVE realization
  base/dual loss factors for shared geometric composition.
- `realization_add_base_distribution(...)`: prepares ADD realization base
  factor for shared geometric composition.

### `random_allocation_gaussian.py`

Gaussian-specific path that constructs factors analytically, then reuses shared composition logic.

Key functions:

- `gaussian_allocation_directional_pld(...)`: resolves the convolution route
  once for one direction and runs the full directional pipeline via
  `allocation_directional_pld(...)`:
  - FFT route uses `_gaussian_allocation_fft(...)` with compact ADD/REMOVE
    internals, paired with the constant loss-discretization count 1.
  - GEOM route uses shared add/remove geometric cores with Gaussian factor
    builders (matching realization route structure), paired with the
    geometric discretization counts.
  - BEST_OF_TWO recursively runs the directional pipeline once per pure route
    and combines same-direction results at the very end via
    `combine_best_of_two_plds(...)` (in `utils.py`).
- Internal builders:
  - `_gaussian_allocation_fft_remove(...)`: the final `log(sum)` is increasing,
    so FFT's final dominating bound uses dominating bounds for both the base
    and exponentiated negative-dual factors. Their lower tails are folded into
    finite mass (`p_min = 0`) before REALS-domain pairwise convolution.
  - `_gaussian_remove_geom_loss_factors(...)`
  - `_gaussian_allocation_fft_add(...)`: the final `-log(sum)` reverses order,
    so FFT's final dominating bound uses a dominated POSITIVES-domain
    exp-space sum. Its zero atom is temporarily embedded at a nonpositive point
    on a REALS lattice so FFT captures every cross-term; after convolution,
    all nonpositive mass is folded back to the POSITIVES zero boundary.
  - `_gaussian_add_geom_loss_factor(...)`

## Adaptive Refinement

`PLD_accounting/adaptive_random_allocation.py` computes upper/lower ranges by iteratively refining:

- `loss_discretization` (halved each step)
- `tail_truncation` (divided by 10 each step)

Entry point:

- `optimize_allocation_epsilon_range(...)`

The module tracks best upper/lower bounds across iterations and returns `AdaptiveResult`.

## Subsampling Integration

`PLD_accounting/subsample_pld.py` provides:

- `subsample_pld(pld, sampling_probability)`
- `subsample_pld_realization(base_pld, sampling_prob, direction)`

This module implements PLD-based subsampling amplification (Appendix C, Algorithms 8-10: `PLDsubsam-remove`, `PLDsubsam-add`, and `subsam-core`) and uses DOMINATES semantics.

## Numerical Invariants

Across the codebase:

- Boundary atoms are represented explicitly as `p_min` and `p_max`.
- `p_min` means `-inf` on `Domain.REALS` and `0` on `Domain.POSITIVES`.
- Mass conservation is enforced after discretization and convolution.
- Bound semantics are preserved during regridding/truncation:
  - `BoundType.DOMINATES` for upper bounds
  - `BoundType.IS_DOMINATED` for lower bounds
- Loss-space and exp-space transforms are explicit (`exp_linear_to_geometric`, `log_geometric_to_linear`).

## Practical Extension Points

- New mechanisms can be added by producing valid `PLDRealization` inputs and
  using `general_allocation_pld(...)`.
- Gaussian method tuning is controlled by `AllocationSchemeConfig` and
  `ConvolutionMethod`.
- Additional accounting workflows can compose returned `dp_accounting` PLDs directly.
