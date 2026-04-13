# PLD_accounting

`PLD_accounting` is a Python package for tight differential privacy accounting of random allocation and subsampling using Privacy Loss Distributions (PLDs) as described in: [Efficient privacy loss accounting for subsampling and random allocation](https://arxiv.org/pdf/2602.17284)

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
- `gaussian_allocation_PLD(params, config, bound_type=BoundType.DOMINATES)`
  - Build a reusable `dp_accounting.PrivacyLossDistribution`.

Realization path (advanced):

- `general_allocation_PLD(num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Build PLD from explicit `PLDRealization` inputs.
- `general_allocation_epsilon(delta, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Epsilon query from explicit realizations.
- `general_allocation_delta(epsilon, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Delta query from explicit realizations.

Common notes:

- `BoundType.DOMINATES` gives an upper (pessimistic) bound.
- `BoundType.IS_DOMINATED` gives a lower (optimistic) bound.
- Builders do not accept `BoundType.BOTH`; build two PLDs if both bounds are needed.

### Subsampling APIs

PLD-based subsampling helpers:

- `subsample_PLD(pld, sampling_probability)`
  - Applies subsampling amplification to a `dp_accounting` PLD.
- `subsample_PLD_realization(base_pld, sampling_prob, direction)`
  - Lower-level helper for `PLDRealization` inputs (REMOVE/ADD direction).

Subsampling helpers use DOMINATES semantics (upper-bound style).

## Install

```bash
pip install PLD_accounting
```

## Where To Start

- End-to-end tutorial notebook: [PLD_accounting_tutorial.ipynb](PLD_accounting_tutorial.ipynb)
- Implementation details: [IMPLEMENTATION_OVERVIEW.md](IMPLEMENTATION_OVERVIEW.md)
