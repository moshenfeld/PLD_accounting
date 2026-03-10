# Implementation Overview

This document explains the architecture and numerical design of this repository, and how the implementation supports the intended privacy-bound guarantees.

## 1. Core Design Goals

- Keep upper-bound and lower-bound accounting explicit at every stage.
- Preserve mass and tail semantics under discretization, transforms, and composition.
- Support both accurate and practical runtime paths through interchangeable convolution backends.
- Make each stage reusable: discretize, transform, convolve, convert, and query.

## 2. Data Model

The central internal type is `DiscreteDist` (`PLD_accounting/discrete_dist.py`):

- `x_array`: strictly increasing grid.
- `PMF_array`: finite probability mass on that grid.
- `p_neg_inf` and `p_pos_inf`: explicit mass at infinite endpoints.

Bound semantics are represented by `BoundType` (`PLD_accounting/types.py`):

- `DOMINATES`: pessimistic/upper-side bound.
- `IS_DOMINATED`: optimistic/lower-side bound.

Mass conservation and endpoint semantics are enforced by:

- `DiscreteDist.validate_mass_conservation`.
- `core_utils.enforce_mass_conservation`.

## 3. Pipeline Stages

### 3.1 Parameter Derivation

`_compute_conv_params` in `PLD_accounting/random_allocation_accounting.py` converts user inputs into internal budgets:

- per-round composition steps,
- per-stage truncation budgets,
- per-stage discretization budgets,
- grid sizes for FFT and geometric paths.

### 3.2 Continuous-to-Discrete Construction

`PLD_accounting/distribution_discretization.py` builds discrete approximations from continuous random variables using:

- linear or geometric grids,
- domination-aware rounding rules,
- stable `logcdf`/`logsf`-based probability extraction,
- adaptive bin mass accumulation.

### 3.3 Composition

`PLD_accounting/convolution_API.py` dispatches to:

- `PLD_accounting/FFT_convolution.py` for linear-spacing convolution.
- `PLD_accounting/geometric_convolution.py` for geometric-spacing convolution.

Repeated composition uses binary exponentiation (`PLD_accounting/utils.py`) to keep composition depth logarithmic in the number of rounds.

### 3.4 Direction-Specific Accounting

`_allocation_PMF_remove` and `_allocation_PMF_add` compute direction-dependent transformed random variables, compose them, and map results back to privacy-loss space.

### 3.5 Final PLD Conversion

`allocation_PLD` converts internal `DiscreteDist` objects into `dp_accounting` PMFs via `PLD_accounting/dp_accounting_support.py`, enabling standard epsilon/delta queries.

## 4. Subsampling Design

`PLD_accounting/subsample_PLD.py` applies subsampling directly in PLD space by combining:

- a stable subsampling loss transform,
- dual-based construction for remove-direction mixture handling,
- grid remapping with bound-aware rounding.

This keeps subsampling compatible with the same composition and bound-tracking machinery used for base accounting.

## 5. Numerical Stability Strategy

The implementation uses multiple safeguards:

- compensated summation for cumulative operations,
- explicit clipping/cleanup for FFT artifacts,
- conservative endpoint/tail reassignment tied to bound semantics,
- explicit finite/infinite mass tracking at each stage,
- staged discretization and tail-budget scaling across composition layers.

## 6. Convolution Method Tradeoffs

- `GEOM`: better aligned with multiplicative grids and positive supports.
- `FFT`: efficient for large linear-grid compositions.
- `COMBINED` and `BEST_OF_TWO`: hybrid strategies to tighten bounds by direction/method behavior.

## 7. Relation to Theory

Theoretical guarantees rely on maintaining order relationships between upper/lower privacy-loss representations through transformation and composition. The code reflects this by:

- keeping bound direction explicit in public and internal APIs,
- using domination-aware rounding and truncation semantics,
- preserving tail behavior via explicit infinity-mass bookkeeping,
- composing with operators that keep these relationships coherent.

As a result, each numerical stage can be interpreted as a controlled approximation that remains consistent with the intended upper/lower guarantee semantics.

## 8. Testing Strategy

The test suite checks:

- validation and mass-conservation invariants,
- convolution behavior and backend consistency,
- subsampling and direction semantics,
- edge conditions and stress scenarios.

Typical command:

```bash
pytest -q
```
