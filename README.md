# PLD_accounting

``PLD_accounting`` is a Python package for tight differential privacy accounting for random allocation and subsampling using Privacy Loss Distributions (PLDs), as described in:

> Vitaly Feldman and Moshe Shenfeld. *[Fast Privacy Loss Accounting for Random Allocation via Multiscale FFT]().*

## Purpose

- Compute tight numerical upper/lower bounds for the PLD construction used to
  account for random allocation.
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

For `num_selected > 1`, this composition upper-bounds uniform `k`-out-of-`t`
allocation by Lemma 2.8 of the paper. The returned numerical upper and lower
bounds bracket that composition. Its upper bound is an upper bound on uniform
allocation; its lower bound need not be a lower bound on uniform allocation.
An adaptive interval measures numerical accuracy for this composition and does
not measure the looseness of the reduction.

For the general APIs, bounds are relative to the supplied PLD realizations.
If those realizations upper-bound an underlying mechanism, a numerical lower
bound on their allocation construction need not lower-bound that mechanism.
This distinction applies even when `num_selected = 1`.

## API Overview

### Random Allocation APIs

Gaussian path (most common):

- `gaussian_allocation_epsilon_range(*, delta, sigma, num_steps, num_selected=1, num_epochs=1, epsilon_accuracy=-1.0)`
  - Adaptive bounds for epsilon, returned as `(upper, lower)`.
  - A positive `epsilon_accuracy` is the absolute gap to reach; a negative one
    stops when `upper / lower <= 1.1`. Zero is rejected.
- `gaussian_allocation_epsilon_configurable(*, params, config, bound_type=BoundType.DOMINATES)`
  - Single epsilon query with explicit discretization/convolution config.
- `gaussian_allocation_delta_configurable(*, params, config, bound_type=BoundType.DOMINATES)`
  - Single delta query with explicit discretization/convolution config.
- `gaussian_allocation_pld(*, params, config, bound_type=BoundType.DOMINATES)`
  - Build a reusable `dp_accounting.PrivacyLossDistribution`.

Realization path (advanced):

- `general_allocation_pld(*, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Build PLD from explicit `PLDRealization` inputs. All three general APIs
    require both inputs to be `PLDRealization` objects and raise `TypeError`
    otherwise.
- `general_allocation_epsilon(*, delta, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Epsilon query from explicit realizations.
- `general_allocation_delta(*, epsilon, num_steps, num_selected, num_epochs, remove_realization, add_realization, config, bound_type=BoundType.DOMINATES)`
  - Delta query from explicit realizations.

Common notes:

- `BoundType.DOMINATES` gives an upper (pessimistic) bound.
- `BoundType.IS_DOMINATED` gives a lower (optimistic) bound on the PLD
  construction; see Parameter Mapping for what it does and does not bound.
- Builders do not accept `BoundType.BOTH`; build two PLDs if both bounds are needed.
- The adaptive Gaussian helper uses `GEOM`; select `MS_FFT` through the
  configurable APIs.

#### Configuration

`AllocationSchemeConfig` holds the numerical settings of one build:

- `loss_discretization` (default `1e-2`): the loss-resolution budget of the
  whole build. It is divided among the epochs and among the discretizing stages
  of one allocation, so the grid step of the returned PLD is much smaller than
  this value. It bounds grid displacement in the loss coordinate, not the error
  of an epsilon query.
- `tail_truncation` (default `1e-12`): the probability-mass budget for
  truncating the tails.
- `max_grid_mult` (default `-1`, no cap; `ConvolutionMethod.GEOM` only): the
  largest number of points of a one-step factor grid. A positive value must be
  at least 4. A binding cap coarsens the step without a warning.
- `convolution_method`, `projection_method`, `max_cells_per_band` and
  `max_closing_grid`: see below.

With realization inputs on `ConvolutionMethod.GEOM`, the grid step of a
realization is a floor on the step of its one-step factors. A realization
coarser than a stage's share of `loss_discretization` raises a `UserWarning`;
build it with a finer `value_discretization` to meet the budget.

#### Bound type, convolution method and projection

`AllocationSchemeConfig.convolution_method` selects the grid on which the
likelihood ratios of one allocation are summed, for `Direction.ADD` and
`Direction.REMOVE`:

- `ConvolutionMethod.GEOM`, the default, sums them on a geometric grid by direct
  convolution.
- `ConvolutionMethod.MS_FFT` (multiscale FFT) sums them on a dyadic grid, which
  is uniform inside each octave `[2^j, 2^(j+1))`, by FFTs inside each octave.

A sum of two grid points usually falls between two grid points.
`AllocationSchemeConfig.projection_method` selects the projection that puts
it back on the grid at every merge:

- `ProjectionMethod.STOCH_DOM` (rounding) moves the sum to the neighbouring
  grid point that the bound requires.
- `ProjectionMethod.CTD` (connect-the-dots, CtD) splits the sum between both
  neighbours so that its mean is kept.
- `None`, the default, selects the grid's own projection: rounding on the
  geometric grid and connect-the-dots on the dyadic grid.

| `convolution_method` | `projection_method` | `BoundType.DOMINATES` | `BoundType.IS_DOMINATED` |
|---|---|---|---|
| `ConvolutionMethod.GEOM` | `None` or `ProjectionMethod.STOCH_DOM` | rounding | rounding |
| `ConvolutionMethod.GEOM` | `ProjectionMethod.CTD` | connect-the-dots | `ValueError` |
| `ConvolutionMethod.MS_FFT` | `None` or `ProjectionMethod.CTD` | connect-the-dots | `ValueError` |
| `ConvolutionMethod.MS_FFT` | `ProjectionMethod.STOCH_DOM` | rounding | rounding |

Rounding gives both bounds and CtD gives upper bounds only, on either grid, so a
lower bound with `ConvolutionMethod.MS_FFT` needs
`projection_method=ProjectionMethod.STOCH_DOM` set explicitly. Both privacy profiles of random allocation are
expectations of convex functions of the average likelihood ratio, and a
projection that keeps the mean can only raise such an expectation, so it cannot
produce a lower bound. The CtD upper bound is usually tighter than the rounding
one, since rounding shifts every merge in one direction.

With CtD and more than one step per allocation, the remove direction is the dual
of the add direction. `general_allocation_pld` prepares and validates both
realizations, then sums only the prepared ADD factors and does not read
`remove_realization` there. The remove bound then holds when `add_realization`
dominates the mechanism's add PLD at every epsilon, negative ones included, as a
dominating discretization does. A component with one step uses each
direction's own prepared input.

The setting applies to the merges and to the truncation of the tails at every
stage after the one-step input. The one-step input follows the bound
type: dominating fixed-gap real-loss construction always uses CtD, and lower
bounds use the internal stochastic-domination engine. That routing is fixed by
the operation and has no option.

Fixed-gap CtD reads each cell's mass and reciprocal moment from the source law
and its dual, so a `loss_discretization` too coarse to leave at least four knots over
the joint source/dual support raises `ValueError` instead of being clamped to a
grid on which the float64 interval measures are not valid.

With `ConvolutionMethod.MS_FFT`, `loss_discretization` is divided equally among
the stages of one allocation: the projection of the one-step input onto the
dyadic grid, each level of the merge tree, and the projection of the result onto
a linear grid of losses. The number of grid points per octave is derived from
that share and is not set directly. Half of the tail budget of an allocation
bounds the range of the input grid, and half is spent by the projection onto the
linear grid. Two fields of `AllocationSchemeConfig` limit this route, as
`max_grid_mult` limits the one-step grid of `ConvolutionMethod.GEOM`:

- `max_cells_per_band` (default `2**18`, a positive integer): the largest
  number of grid points per octave. The derived number is the smallest integer
  whose mean loss spacing, `log(2)` over the points per octave, meets the
  share of `loss_discretization`; when the limit is smaller, the limit itself is
  used.
- `max_closing_grid` (default `1_000_000`, at least 4): the largest number of
  points of the linear loss grid that one allocation returns.

A limit that binds coarsens the grid and warns; the bound stays valid and is
looser. The returned grid carries the step that was used, which is then larger
than the requested one.

`loss_discretization` is a nominal resolution budget. On the dyadic grid it
controls mean loss spacing, while the widest cell has width
`log(1 + 1/m) <= h / log(2)` for a stage's share `h`, when the cap is inactive.
The resulting worst-case accumulated grid displacement is at most
`loss_discretization / log(2)`, rather than `loss_discretization`. This is a
loss-coordinate displacement bound, not a guarantee on epsilon-query error;
binding caps and prior approximation of an input realization require separate
accounting. The resolution rule itself is the same for both projections.

For fixed-gap real-loss regridding, use
`project_discrete_dist(*, dist, tail_truncation, loss_discretization, bound_type, max_grid_points=-1)`.
This public entry point selects CtD for `DOMINATES` and
directional stochastic domination for `IS_DOMINATED`; callers do not choose a
discretization engine. The caller remains responsible for proving that the source
dominates the mechanism it represents. The source must use `Domain.REALS`, and
`tail_truncation` must be finite and nonnegative. A positive `max_grid_points`
must be at least four and coarsens the returned grid when necessary.

### Mechanism PLD Helpers

Factory helpers for building `PLDRealization` inputs from specific mechanisms:

- `gaussian_distribution(*, scale, value_discretization=DEFAULT_LOSS_DISCRETIZATION, tail_truncation=DEFAULT_TAIL_TRUNCATION, bound_type=BoundType.DOMINATES)`
  - Discretizes the Gaussian mechanism PLD (L2 sensitivity 1) onto a linear grid.
  - Returns a `PLDRealization` for `DOMINATES` and a `DenseDiscreteDist` for
    `IS_DOMINATED`. The lower-bound object may carry mass at `-inf`; it is not a
    PLD realization and the general allocation APIs reject it.
- `laplace_distribution(*, scale, value_discretization=DEFAULT_LOSS_DISCRETIZATION, tail_truncation=DEFAULT_TAIL_TRUNCATION, bound_type=BoundType.DOMINATES)`
  - Discretizes the Laplace mechanism PLD (L1 sensitivity 1) onto a linear grid.
  - Same return semantics as `gaussian_distribution`.
- `discrete_distribution(*, noise_dist, loss_discretization, tail_truncation, sensitivity=1)`
  - Builds the dominating `(remove_realization, add_realization)` pair for an integer count query with additive noise described by a unit-spaced `DenseDiscreteDist`.
  - Builds exact sparse directional loss distributions, then delegates truncation and fixed-gap projection to `project_discrete_dist`.
  - Treats the input distribution's distinct `p_min` and `p_max` masses as unmatched noise tails and maps both entirely to positive-infinity privacy loss.

For Gaussian and Laplace, `scale` is the noise standard deviation or Laplace scale parameter,
respectively. Pass the `DOMINATES` results as `remove_realization` and
`add_realization` to the general allocation APIs. Both PLDs are symmetric, so one
realization serves both directions. The general APIs' lower bounds refer to the
supplied realizations, including their discretization, rather than necessarily
to the original continuous mechanism.

### Subsampling APIs

PLD-based subsampling helpers:

- `subsample_pld(*, pld, sampling_probability)`
  - Applies Poisson subsampling, under add/remove adjacency, to a pessimistic
    `dp_accounting` PLD.
  - Uses the dp_accounting adapter's default import bands.
  - An input with no add PMF gives an output with none, so `dp_accounting` uses
    the subsampled remove PMF for both directions.
- `subsample_pld_realization(*, base_pld, sampling_prob, direction)`
  - Lower-level helper for `PLDRealization` inputs (REMOVE/ADD direction).

Subsampling helpers use DOMINATES semantics (upper-bound style).

### Supporting exports

- `compose_full_pld(*, remove_dist, add_dist, bound_type)`
  - Convert internal directional PLDs into a `dp_accounting` PLD.
- `ConvolutionMethod`
  - The grid on which an allocation sums likelihood ratios; see the notes above.
- `ProjectionMethod`
  - The projection at the merges of an allocation; see the notes above.
- `project_discrete_dist(*, dist, tail_truncation, loss_discretization, bound_type, max_grid_points=-1)`
  - Public fixed-gap real-loss regridding. See the bound-type notes above.
- `dp_accounting_pmf_to_pld_realization(*, pmf, mass_drift_tol=..., mass_repair_tol=..., moment_drift_tol=..., moment_repair_tol=..., negative_drift_tol=..., negative_repair_tol=...)`
  - Admit a pessimistic `dp_accounting` PMF as a `PLDRealization`.
- `has_numba()`
  - Whether optional Numba acceleration is active. Results agree with the NumPy
    fallbacks up to the last bits of some sums.

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
Python hides `ImportWarning` by default; call `has_numba()` to check.

The environment variable `MAX_FFT_BYTES` (default 8 GiB) limits the memory of a
single FFT. Epoch composition coarsens a base PLD that would exceed it, with a
warning, and the error raised by an FFT that exceeds it names the variable.

## Where To Start

- End-to-end tutorial notebook: [PLD_accounting_tutorial.ipynb](PLD_accounting_tutorial.ipynb)
- Implementation details: [IMPLEMENTATION_OVERVIEW.md](IMPLEMENTATION_OVERVIEW.md)
