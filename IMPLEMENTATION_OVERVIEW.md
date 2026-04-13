# Implementation Overview

This document describes the internal structure of `PLD_accounting` and how the implementation maps to the paper's random-allocation setting.

For user-facing examples, see [README.md](README.md) and
[PLD_accounting_tutorial.ipynb](PLD_accounting_tutorial.ipynb).

## Paper-Aligned Semantics

The package follows the `k`-out-of-`t` random-allocation language:

- In each epoch, a record participates in `k` selected steps out of `t` total steps.
- This is repeated for `num_epochs` epochs.

API parameter mapping:

- `num_steps = t`
- `num_selected = k`
- `num_epochs = number of epochs`

In code, this decomposition is implemented in
`allocation_directional_PLD()` in `PLD_accounting/random_allocation_accounting.py`:

- Floor component:
  - `floor_steps = floor(num_steps / num_selected)`
  - `remainder = num_steps - num_selected * floor_steps`
  - `floor_epochs = (num_selected - remainder) * num_epochs`
- Ceil component (only when `remainder > 0`):
  - `ceil_steps = floor_steps + 1`
  - `ceil_epochs = remainder * num_epochs`

Both Gaussian and realization paths use the same floor/ceil decomposition and
compose both components when needed.

Input validation in `allocation_directional_PLD()`:

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

- `DenseDiscreteDist`: regular-grid distribution with `x_min`, `step`,
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

## Parameter Budget Conventions

Shared composition budgets are derived inside
`_allocation_directional_PLD_core()` in `PLD_accounting/random_allocation_accounting.py`.

- `output_tail_truncation = component_tail_truncation / 3`
- `base_tail_truncation = output_tail_truncation / (2 * component_num_epochs)`
- `output_loss_discretization = config.loss_discretization / 3`
- `base_loss_discretization = output_loss_discretization / sqrt(component_num_epochs)`

Interpretation used in code:

- `allocation_directional_PLD()` keeps the full top-level tail budget when only one
  component is present (divisible case), and splits evenly only when both
  floor and ceil components are present.
- `/3` is used in each component core because truncation is handled across
  multiple stages (base creation, composition, output alignment).
- `1 / (2 * num_epochs)` is used for base tail truncation in each component.
- `1 / sqrt(num_epochs)` is used for core loss discretization because
  discretization error scales approximately like the square root of the number
  of compositions.

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
- both routes call shared `geometric_allocation_PLD_base_add/remove(...)`;
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
| `PLD_accounting/mechanisms.py` | Mechanism PLD factory helpers (`gaussian_distribution`, `laplace_distribution`). |
| `PLD_accounting/validation.py` | Centralized input validation (`validate_privacy_params`, `validate_allocation_params`, etc.). |
| `PLD_accounting/discrete_dist.py` | Distribution classes (`DenseDiscreteDist`, `SparseDiscreteDist`, `PLDRealization`, `Domain`). |
| `PLD_accounting/distribution_discretization.py` | Continuous-to-discrete conversion and spacing changes (linear/geometric). |
| `PLD_accounting/FFT_convolution.py` | FFT-based convolution and self-convolution on linear grids. |
| `PLD_accounting/geometric_convolution.py` | Convolution and self-convolution on geometric grids. |
| `PLD_accounting/utils.py` | PLD transforms (`exp`, `log`, dual, negate-reverse, composition helpers). |
| `PLD_accounting/distribution_utils.py` | Numerical utilities (mass conservation, spacing checks, stable comparisons). |
| `PLD_accounting/dp_accounting_support.py` | Conversion between internal probability representations and `dp_accounting` PMF/PLD types. |
| `PLD_accounting/subsample_PLD.py` | PLD-level subsampling amplification helpers (DOMINATES-only path). |

## Public API Surface

Random allocation (defined in `PLD_accounting/random_allocation_api.py`):

- Gaussian path:
  - `gaussian_allocation_PLD(...)`
  - `gaussian_allocation_epsilon_configurable(...)`
  - `gaussian_allocation_delta_configurable(...)`
  - `gaussian_allocation_epsilon_range(...)`
- Realization path:
  - `general_allocation_PLD(...)`
  - `general_allocation_epsilon(...)`
  - `general_allocation_delta(...)`

Mechanism PLD helpers (defined in `PLD_accounting/mechanisms.py`):

- `gaussian_distribution(scale, value_discretization, tail_truncation, bound_type)`
- `laplace_distribution(scale, value_discretization, tail_truncation, bound_type)`

Subsampling (defined in `PLD_accounting/subsample_PLD.py`):

- `subsample_PLD(pld, sampling_probability)`
- `subsample_PLD_realization(base_pld, sampling_prob, direction)`

Distribution type (defined in `PLD_accounting/discrete_dist.py`):

- `PLDRealization` — linear-grid privacy-loss distribution used as input to realization APIs.

Notes:

- PLD builders reject `BoundType.BOTH`; users build separate DOMINATES and IS_DOMINATED PLDs.
- Realization-based allocation requires `ConvolutionMethod.GEOM`.

## Core Composition Modules

### `random_allocation_accounting.py`

This is the shared composition core used by both Gaussian and realization accounting.

Key functions:

- `allocation_full_PLD(...)`:
  Shared top-level orchestrator used by both API paths. Calls
  `allocation_directional_PLD(...)` for REMOVE and ADD, then combines with
  `_compose_full_PLD(...)`.
- `_allocation_directional_PLD_core(...)`:
  Calls a base-PLD callback, regrids to core resolution, composes across
  epochs, then regrids to output discretization.
- `geometric_allocation_PLD_base_remove(...)`:
  Shared exp-space geometric composer for REMOVE. Accepts a callback that
  builds lower/upper loss factors.
- `geometric_allocation_PLD_base_add(...)`:
  Shared exp-space geometric composer for ADD. Accepts a callback that builds
  the add loss factor.
- `allocation_directional_PLD(...)`:
  Applies adaptive step decomposition and composes floor/ceil components.
- `_compose_full_PLD(...)`:
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

- `gaussian_allocation_PLD_core(...)`: selects FFT/GEOM/BEST computation and
  returns the base directional PLD used by `_allocation_directional_PLD_core(...)`:
  - FFT callback uses `_gaussian_allocation_fft(...)` with compact ADD/REMOVE internals.
  - GEOM callback uses shared add/remove geometric cores with Gaussian factor
    builders, matching realization route structure.
  - BEST callback combines FFT and GEOM PMFs.
- Internal builders:
  - `_gaussian_allocation_fft_remove(...)`
  - `_gaussian_remove_geom_loss_factors(...)`
  - `_gaussian_allocation_fft_add(...)`
  - `_gaussian_add_geom_loss_factor(...)`

## Adaptive Refinement

`PLD_accounting/adaptive_random_allocation.py` computes upper/lower ranges by iteratively refining:

- `loss_discretization` (halved each step)
- `tail_truncation` (divided by 10 each step)

Entry point:

- `optimize_allocation_epsilon_range(...)`

The module tracks best upper/lower bounds across iterations and returns `AdaptiveResult`.

## Subsampling Integration

`PLD_accounting/subsample_PLD.py` provides:

- `subsample_PLD(pld, sampling_probability)`
- `subsample_PLD_realization(base_pld, sampling_prob, direction)`

This module implements PLD-based subsampling amplification (Appendix C mapping) and uses DOMINATES semantics.

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
  using `general_allocation_PLD(...)`.
- Gaussian method tuning is controlled by `AllocationSchemeConfig` and
  `ConvolutionMethod`.
- Additional accounting workflows can compose returned `dp_accounting` PLDs directly.
