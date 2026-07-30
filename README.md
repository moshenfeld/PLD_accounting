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

- `gaussian_allocation_epsilon_range(delta, sigma, num_steps, num_selected=1, num_epochs=1, epsilon_accuracy=-1.0)`
  - Adaptive upper/lower bounds for epsilon.
- `gaussian_allocation_epsilon_configurable(params, config, bound_type=BoundType.DOMINATES)`
  - Single epsilon query with explicit discretization/convolution config.
- `gaussian_allocation_delta_configurable(params, config, bound_type=BoundType.DOMINATES)`
  - Single delta query with explicit discretization/convolution config.
- `gaussian_allocation_pld(params, config, bound_type=BoundType.DOMINATES)`
  - Build a reusable `dp_accounting.PrivacyLossDistribution`.

Realization path (advanced):

- `general_allocation_pld(num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Build PLD from explicit `PLDRealization` inputs.
- `general_allocation_epsilon(delta, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Epsilon query from explicit realizations.
- `general_allocation_delta(epsilon, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
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

For fixed-gap real-loss regridding, use
`rediscretize_dist_by_bound(dist, tail_truncation, loss_discretization,
bound_type)`. This public entry point selects CtD for `DOMINATES` and
directional stochastic domination for `IS_DOMINATED`; callers do not choose a
discretization engine.

### Mechanism PLD Helpers

Factory helpers for building `PLDRealization` inputs from specific mechanisms:

- `gaussian_distribution(scale, value_discretization, tail_truncation, bound_type=BoundType.DOMINATES)`
  - Discretizes the Gaussian mechanism PLD (L2 sensitivity 1) onto a linear grid.
  - Returns a `PLDRealization` for `DOMINATES` and a `DenseDiscreteDist` for `IS_DOMINATED`.
- `laplace_distribution(scale, value_discretization, tail_truncation, bound_type=BoundType.DOMINATES)`
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

- `subsample_pld(pld, sampling_probability)`
  - Applies subsampling amplification to a `dp_accounting` PLD.
- `subsample_pld_realization(base_pld, sampling_prob, direction)`
  - Lower-level helper for `PLDRealization` inputs (REMOVE/ADD direction).

Subsampling helpers use DOMINATES semantics (upper-bound style).

## Install

```bash
pip install PLD_accounting  # distribution name; then: import PLD_accounting
```

`numba` is optional. Install `PLD_accounting[performance]` to enable JIT
kernels; without it, NumPy fallbacks are used and an `ImportWarning` is emitted.

## Where To Start

- End-to-end tutorial notebook: [PLD_accounting_tutorial.ipynb](PLD_accounting_tutorial.ipynb)
- Implementation details: [IMPLEMENTATION_OVERVIEW.md](IMPLEMENTATION_OVERVIEW.md)
