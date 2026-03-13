# Implementation Overview

This document describes the internal structure of `PLD_accounting` and how the implementation maps to the paper's random-allocation setting.

For user-facing examples, see [README.md](README.md) and [usage_example.py](usage_example.py).

## Paper-Aligned Semantics

The package follows the `k`-out-of-`t` random-allocation language:

- In each epoch, a record participates in `k` selected steps out of `t` total steps.
- This is repeated for `num_epochs` epochs.

API parameter mapping:

- `num_steps = t`
- `num_selected = k`
- `num_epochs = number of epochs`

In code, this decomposition is implemented in
`decompose_allocation_compositions()` in
`PLD_accounting/random_allocation_accounting.py`:

- `num_steps_per_round = floor(num_steps / num_selected)`
- `num_rounds = num_selected * num_epochs`

This is the composition structure used by both Gaussian and realization paths.

## High-Level Pipeline

1. Public API validates inputs and builds PMFs for REMOVE and ADD directions.
2. Per-round random-allocation PMFs are computed in loss-space via exp-space convolution helpers.
3. Per-round PMFs are composed across `num_rounds`.
4. Final PMFs are converted to `dp_accounting.PrivacyLossDistribution`.
5. Epsilon/delta queries are answered on that PLD object.

Both input modes share this shape:

- Gaussian mode: starts from analytic log-normal factors.
- Realization mode: starts from user-provided `PLDRealization`.

## File Map

| File | Responsibility |
|---|---|
| `PLD_accounting/__init__.py` | Public exports. |
| `PLD_accounting/types.py` | Enums and configs (`PrivacyParams`, `AllocationSchemeConfig`, `BoundType`, etc.). |
| `PLD_accounting/random_allocation_api.py` | Public entry points for Gaussian and realization accounting. |
| `PLD_accounting/random_allocation_accounting.py` | Shared realization-based composition helpers and final composition logic. |
| `PLD_accounting/random_allocation_gaussian.py` | Gaussian-specific factor construction and convolution method selection. |
| `PLD_accounting/adaptive_random_allocation.py` | Adaptive upper/lower range refinement for epsilon/delta queries. |
| `PLD_accounting/discrete_dist.py` | Distribution classes (`LinearDiscreteDist`, `GeometricDiscreteDist`, `PLDRealization`, etc.). |
| `PLD_accounting/distribution_discretization.py` | Continuous-to-discrete conversion and spacing changes (linear/geometric). |
| `PLD_accounting/FFT_convolution.py` | FFT-based convolution and self-convolution on linear grids. |
| `PLD_accounting/geometric_convolution.py` | Convolution and self-convolution on geometric grids. |
| `PLD_accounting/utils.py` | PLD transforms (`exp`, `log`, dual, negate-reverse, composition helpers). |
| `PLD_accounting/distribution_utils.py` | Numerical utilities (mass conservation, spacing checks, stable comparisons). |
| `PLD_accounting/dp_accounting_support.py` | Conversion between internal PMFs and `dp_accounting` PMFs/PLDs. |
| `PLD_accounting/subsample_PLD.py` | PLD-level subsampling amplification helpers (DOMINATES-only path). |

## Public API Surface

Defined in `PLD_accounting/random_allocation_api.py`:

- Gaussian path:
  - `gaussian_allocation_PLD(...)`
  - `gaussian_allocation_epsilon_extended(...)`
  - `gaussian_allocation_delta_extended(...)`
  - `gaussian_allocation_epsilon_range(...)`
  - `gaussian_allocation_delta_range(...)`
- Realization path:
  - `general_allocation_PLD(...)`
  - `general_allocation_epsilon(...)`
  - `general_allocation_delta(...)`

Notes:

- PLD builders reject `BoundType.BOTH`; users build separate DOMINATES and IS_DOMINATED PLDs.
- Realization-based allocation requires `ConvolutionMethod.GEOM`.

## Core Composition Modules

### `random_allocation_accounting.py`

This is the shared composition core for realization-based accounting and shared finalize helpers.

Key functions:

- `decompose_allocation_compositions(...)`:
  Converts `(num_steps, num_selected, num_epochs)` into
  `(num_steps_per_round, num_rounds)`.
- `allocation_PMF_from_realization(...)`:
  Builds per-direction PMF from a `PLDRealization`.
- `log_mean_exp_remove(...)` and `log_mean_exp_add(...)`:
  Implement the exp-space convolution core used by Appendix C algorithms.
- `finalize_allocation_composition(...)`:
  Regrid, compose across `num_rounds`, and regrid to output discretization.
- `compose_pld_from_pmfs(...)`:
  Converts internal PMFs into a `dp_accounting` PLD object.

### `random_allocation_gaussian.py`

Gaussian-specific path that constructs factors analytically, then reuses shared composition logic.

Key functions:

- `compute_conv_params(...)`: derives grid sizes, truncation budgets, and
  `(num_steps_per_round, num_rounds)`.
- `allocation_PMF_from_gaussian(...)`: dispatches by direction and convolution method (`FFT`, `GEOM`, `BEST_OF_TWO`, `COMBINED`), then finalizes composition.
- Internal builders:
  - `_allocation_PMF_remove_fft(...)`
  - `_allocation_PMF_remove_geom(...)`
  - `_allocation_PMF_add_fft(...)`
  - `_allocation_PMF_add_geom(...)`

## Adaptive Refinement

`PLD_accounting/adaptive_random_allocation.py` computes upper/lower ranges by iteratively refining:

- `loss_discretization` (halved each step)
- `tail_truncation` (divided by 10 each step)

Entry points:

- `optimize_allocation_epsilon_range(...)`
- `optimize_allocation_delta_range(...)`

The module tracks best upper/lower bounds across iterations and returns `AdaptiveResult`.

## Subsampling Integration

`PLD_accounting/subsample_PLD.py` provides:

- `subsample_PLD(pld, sampling_probability)`
- `subsample_PMF(base_pld, sampling_prob, direction)`

This module implements PLD-based subsampling amplification (Appendix C mapping) and uses DOMINATES semantics.

## Numerical Invariants

Across the codebase:

- Infinity atoms (`p_neg_inf`, `p_pos_inf`) are represented explicitly.
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
