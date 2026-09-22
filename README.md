# PLD_accounting

``PLD_accounting`` is a Python package for tight differential privacy accounting for random allocation and subsampling using Privacy Loss Distributions (PLDs), as described in:

> Vitaly Feldman and Moshe Shenfeld. *[Efficient Privacy Loss Accounting for Subsampling and Random Allocation](https://arxiv.org/abs/2602.17284).* International Conference on Machine Learning (ICML), 2026.

## Paper version

The [`PLD-paper`](https://github.com/moshenfeld/PLD_accounting/tree/PLD-paper)
branch preserves the code snapshot corresponding to the paper. Refer to that
branch for a stable paper-version reference; the default branch continues to
receive subsequent development changes.

## Purpose

- Compute tight upper/lower DP bounds for random allocation.
- Support both Gaussian mechanisms and explicit PLD realizations.
- Return `dp_accounting` PLDs for epsilon/delta queries and composition workflows.

## Random Allocation Model

The package accounts for the following sampling pattern:

- Per epoch: choose `k` steps out of `t` uniformly at random.
- Across training: repeat this for `num_epochs` epochs.

## Parameter Mapping

- `num_steps = t` (total candidate steps per epoch)
- `num_selected = k` (selected steps per epoch)
- `num_epochs` (number of repeated epochs)

Internal composition:

- `floor_steps = floor(num_steps / num_selected)`
- `remainder = num_steps - num_selected * floor_steps`
- `floor_epochs = (num_selected - remainder) * num_epochs`
- `ceil_steps = floor_steps + 1`
- `ceil_epochs = remainder * num_epochs`
- We compute 1-out-of-`floor_steps` then self compose it `floor_epochs` times, compute 1-out-of-`ceil_steps` then self compose it `ceil_epochs` times, and finally compose them with each other

## API Overview

### Random Allocation APIs

Gaussian path (most common):

- `gaussian_allocation_epsilon_range(*, delta, sigma, num_steps, num_selected=1, num_epochs=1, epsilon_accuracy=-1.0)`
  - Adaptive upper/lower bounds for epsilon.
- `gaussian_allocation_epsilon_configurable(*, params, config, bound_type=BoundType.DOMINATES)`
  - Single epsilon query with explicit discretization/convolution config.
- `gaussian_allocation_delta_configurable(*, params, config, bound_type=BoundType.DOMINATES)`
  - Single delta query with explicit discretization/convolution config.
- `gaussian_allocation_pld(*, params, config, bound_type=BoundType.DOMINATES)`
  - Build a reusable `dp_accounting.PrivacyLossDistribution`.
- `gaussian_allocation_directional_pld(*, params, config, direction, bound_type=BoundType.DOMINATES)`
  - One ADD or REMOVE directional distribution for the Gaussian path.

Realization path (advanced): these APIs require `ConvolutionMethod.GEOM`; other
methods raise `ValueError`.

- `general_allocation_pld(*, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Build PLD from explicit `PLDRealization` inputs.
- `general_allocation_epsilon(*, delta, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Epsilon query from explicit realizations.
- `general_allocation_delta(*, epsilon, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Delta query from explicit realizations.

Common notes:

- `BoundType.DOMINATES` gives an upper (pessimistic) bound.
- `BoundType.IS_DOMINATED` gives a lower (optimistic) bound.
- Builders do not accept `BoundType.BOTH`; build two PLDs if both bounds are needed.

#### Bound type and convolution method support

Not every `ConvolutionMethod` can produce both bounds. Only the geometric backend
builds the one-step factors in a way that rounds *down* throughout, which is what a
valid lower bound requires; the FFT routes round up. Passing an unsupported pair
raises `ValueError` rather than silently returning a bound that does not hold.
This applies to both `Direction.ADD` and `Direction.REMOVE`. Use
`ConvolutionMethod.GEOM` whenever you need a lower bound.

Dominating fixed-gap real-loss construction always uses connect-the-dots
(CtD). Lower bounds, FFT exp-space factors, and geometric-grid
regridding use the internal stochastic-domination engine required by those
representations. This routing is fixed by the operation; there is no public
discretization-method option.

CtD reads each cell's mass and reciprocal moment from the source law and its
dual, so a `loss_discretization` too coarse to leave at least four knots over
the joint source/dual support raises `ValueError` instead of being clamped to a
grid on which the float64 interval measures are not valid.

For fixed-gap real-loss regridding, use
`rediscretize_dist_by_bound(*, dist, tail_truncation, loss_discretization, bound_type)`.
This public entry point selects CtD for `DOMINATES` and
directional stochastic domination for `IS_DOMINATED`; callers do not choose a
discretization engine. The caller remains responsible for proving that the source
dominates the mechanism it represents.

### Mechanism PLD Helpers

Factory helpers for building `PLDRealization` inputs from specific mechanisms:

- `gaussian_distribution(*, scale, value_discretization=DEFAULT_LOSS_DISCRETIZATION, tail_truncation=DEFAULT_TAIL_TRUNCATION, bound_type=BoundType.DOMINATES)`
  - Discretizes the Gaussian mechanism PLD (L2 sensitivity 1) onto a linear grid.
  - Returns a `PLDRealization` for `DOMINATES` and a `DenseDiscreteDist` for `IS_DOMINATED`.
- `laplace_distribution(*, scale, value_discretization=DEFAULT_LOSS_DISCRETIZATION, tail_truncation=DEFAULT_TAIL_TRUNCATION, bound_type=BoundType.DOMINATES)`
  - Discretizes the Laplace mechanism PLD (L1 sensitivity 1) onto a linear grid.
  - Same return semantics as `gaussian_distribution`.
- `discrete_distribution(*, noise_dist, loss_discretization, tail_truncation, sensitivity=1)`
  - Builds the dominating `(remove_realization, add_realization)` pair for an integer count query with additive noise described by a unit-spaced `DenseDiscreteDist`.
  - Builds exact sparse directional loss distributions, then delegates truncation and fixed-gap projection to `rediscretize_dist_by_bound`.
  - Treats the input distribution's distinct `p_min` and `p_max` masses as unmatched noise tails and maps both entirely to positive-infinity privacy loss.

For Gaussian and Laplace, `scale` is the noise standard deviation or Laplace scale parameter,
respectively. Pass mechanism results directly as `remove_realization` and `add_realization` to
the general allocation APIs.

### Subsampling APIs

PLD-based subsampling helpers:

- `subsample_pld(*, pld, sampling_probability)`
  - Applies subsampling amplification to a `dp_accounting` PLD.
  - Uses the dp_accounting adapter's default import bands.
- `subsample_pld_realization(*, base_pld, sampling_prob, direction)`
  - Lower-level helper for `PLDRealization` inputs (REMOVE/ADD direction).

Subsampling helpers use DOMINATES semantics (upper-bound style).

### Supporting exports

- `compose_full_pld(*, remove_dist, add_dist, bound_type)`
  - Convert internal directional PLDs into a `dp_accounting` PLD.
- `rediscretize_dist_by_bound(*, dist, tail_truncation, loss_discretization, bound_type)`
  - Public fixed-gap real-loss regridding. See the bound-type notes above.
- `dp_accounting_pmf_to_pld_realization(*, pmf, mass_drift_tol=..., mass_repair_tol=..., moment_drift_tol=..., moment_repair_tol=..., negative_drift_tol=..., negative_repair_tol=...)`
  - Admit a pessimistic `dp_accounting` PMF as a `PLDRealization`.
- `has_numba()`
  - Whether optional Numba acceleration is active. Results are unaffected.

Published library functions with more than one input argument are keyword-only.
`dp_accounting_pmf_to_pld_realization` exposes keyword-only drift and repair
bands for callers importing a known-noisier PMF; `subsample_pld` uses the
defaults and does not forward them.

## Install

```bash
pip install PLD_accounting  # distribution name; then: import PLD_accounting
```

`numba` is optional. Install `PLD_accounting[performance]` to enable JIT
kernels; without it, NumPy fallbacks are used and an `ImportWarning` is emitted.

## Where To Start

- End-to-end tutorial notebook: [PLD_accounting_tutorial.ipynb](PLD_accounting_tutorial.ipynb)
- Implementation details: [IMPLEMENTATION_OVERVIEW.md](IMPLEMENTATION_OVERVIEW.md)
