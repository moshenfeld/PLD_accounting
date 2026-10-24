# Implementation Overview

This document describes the internal structure of `PLD_accounting` and how the implementation maps to the paper's random-allocation setting.

For user-facing examples, see [README.md](README.md) and
[PLD_accounting_tutorial.ipynb](PLD_accounting_tutorial.ipynb).

`numba` is an optional performance dependency. The implementation dispatches to
NumPy fallbacks when it is unavailable. The geometric-convolution fallback replays
the numba kernel's per-bin Kahan updates in the same order, so both backends return
identical arrays; the mass-repair bands after the kernel assume that compensated
accumulation, and an uncompensated fallback exceeds them.

## Paper-Aligned Semantics

The package follows the `k`-out-of-`t` random-allocation language:

- In each epoch, a record participates in `k` selected steps out of `t` total steps.
- This is repeated for `num_epochs` epochs.

API parameter mapping:

- `num_steps = t`
- `num_selected = k`
- `num_epochs = number of epochs`

In code, this decomposition is implemented in
`allocation_plds()` in `PLD_accounting/random_allocation_accounting.py`:

- Floor component:
  - `floor_steps = floor(num_steps / num_selected)`
  - `remainder = num_steps - num_selected * floor_steps`
  - `floor_epochs = (num_selected - remainder) * num_epochs`
- Ceil component (only when `remainder > 0`):
  - `ceil_steps = floor_steps + 1`
  - `ceil_epochs = remainder * num_epochs`

Both Gaussian and realization paths use the same floor/ceil decomposition and
compose both components when needed.

For `num_selected > 1`, this composition is the upper-bound reduction of
Lemma 2.8 of the paper, not an identity of privacy profiles for uniform
`k`-out-of-`t` allocation. Numerical upper/lower bounds and adaptive intervals
refer to this composition. Only its upper bound transfers through that
reduction to uniform allocation. In the realization path the reference is
additionally the supplied PLDs: lower-bounding a dominating realization does
not necessarily lower-bound the underlying mechanism.

Allocation counts are checked by `require_allocation_counts()` in
`validation.py`, called from `allocation_plds()` and from
`PrivacyParams.__post_init__`:

- `num_steps`, `num_selected`, and `num_epochs` must be positive integers.
- `num_selected` may not exceed `num_steps`, so every round has at least one
  per-selection step.

Field invariants are enforced where the value is first constructed —
`PrivacyParams` and `AllocationSchemeConfig` validate in `__post_init__`, so a
malformed configuration cannot reach an accounting routine. Operations re-check
only what they additionally require, such as a compatible spacing or domain.

## High-Level Pipeline

1. Public API validates inputs and builds PMFs for REMOVE and ADD directions.
2. Per-round random-allocation PMFs are computed in loss-space via exp-space
   convolution helpers, on the grid that
   `AllocationSchemeConfig.convolution_method` selects and with the projection
   that `AllocationSchemeConfig.projection_method` selects at the merges.
3. Floor/ceil PMF components are composed across their epoch counts and
   combined when both are present.
4. Final PMFs are converted to `dp_accounting.PrivacyLossDistribution`.
5. Epsilon/delta queries are answered on that PLD object.

Both input modes share this shape:

- Gaussian mode: starts from the continuous Gaussian privacy-loss law, which is its
  own dual.
- Realization mode: starts from user-provided `PLDRealization`.

## Canonical Distribution Model

The active runtime is built on:

- `DenseDiscreteDist`: regular-grid distribution carrying an immutable
  `GridSpec` plus `prob_arr`, `p_min`, and `p_max`.
- `SparseDiscreteDist`: explicit-support distribution with `x_array`,
  `prob_arr`, `p_min`, and `p_max`.
- `PLDRealization`: linear-grid specialization for privacy-loss space.
- `DyadicDist`: nonnegative measure on the dyadic grid of likelihood ratios,
  stored as one row of `cells_per_band` masses per octave from `first_band` up,
  with `p_min` at zero and `p_max` at infinity. It is a measure, not a
  `DiscreteDistBase`: a multiscale sum misses unit mass by the rounding of its
  transforms. See `multiscale_fft_convolution.py` below.

`GridSpec` is `(step, n, spacing_type, anchor, index_0)`. A point is
`anchor + k * step` on a linear lattice and `anchor * exp(k * step)` on a
geometric one, with `k = index_0 + i`; geometric grids store the log spacing and
a multiplicative anchor. `x_0`, `step` and `x_array` are derived properties, not
stored state.

Spacing is stated one way only. `step` is the stored field for both families —
the additive bin width when linear, the log ratio when geometric — and nothing
converts to or from a multiplicative ratio internally. `GridSpec.geometric(*, ratio,
...)` takes a ratio and stores `math.log(ratio)` once at construction; there is no
`ratio` field, property, or accessor, because reasoning in ratios about a type that
stores logs is how an `exp(log(step))` round trip drifted by tens of thousands of
ULPs. See **Public API Surface** for the signature.

Two consequences matter to anyone touching this code:

- Grid equality is structural and allocates nothing. Two grids are the same
  lattice because their fields match, never because their materialized
  coordinates compare close.
- Slice, pad, reflect, `exp`/`log` and convolution are integer-index operations,
  so coordinates retained across them are bitwise identical. Internal code must
  not re-infer a step, anchor or index from `x_array`; producers pass the
  `GridSpec`. Approximate external arrays stay sparse or are projected onto an
  explicit target grid. The one exception is `dyadic_dist_on_range`: a loss
  factor on explicit dyadic points carries no lattice, so it reads the reach and
  the points per octave from the coordinates and accepts them only when the
  dyadic points they describe reproduce those coordinates bitwise.

Boundary semantics depend on `Domain`:

- `Domain.REALS`:
  - `p_min` is mass at `-inf`
  - `p_max` is mass at `+inf`
- `Domain.POSITIVES`:
  - `p_min` is mass at `0`
  - `p_max` is mass at `+inf`

CtD is the fixed engine for dominating fixed-gap real-loss construction; there
is no public method selector. Lower bounds use separate stochastic-domination
functions because CtD is not defined for them. The projection inside a
convolution of likelihood ratios is a separate matter, described under
`geometric_convolution.py` and `multiscale_fft_convolution.py` below. A CtD
target is a fixed-gap linear grid or, for
`project_continuous_dist_ctd_on_points` and `project_discrete_dist_ctd_on_points`,
explicit strictly increasing loss points. A discrete source may be dense or sparse, but must use
`Domain.REALS`, have exact `p_min == 0`, finite strictly ordered support,
conserved mass, and reciprocal moment `E[exp(-L)] <= 1` under the shared
realization tolerance. These checks certify the object as a realizable PLD,
not its domination relationship to an external mechanism.

Linear real-loss callers route through the public
`project_discrete_dist` helper. Bound direction alone selects the
engine: `DOMINATES` uses CtD and `IS_DOMINATED` uses stochastic domination.

CtD is used because repeatedly rounding every atom upward accumulates an
`O(compositions * step)` loss-space shift. CtD instead places, in each cell, the
two endpoint masses that preserve both the cell's probability and its reciprocal
moment, so the output hockey-stick curve is the piecewise-linear interpolant of
the source's through the grid knots. That is a structural domination argument,
not a universal accuracy guarantee; numerical accuracy must still be assessed
against converged or independent references.

**The profile is never materialized.** Each cell's contribution comes from local
source and reflected-dual measures `(M_i, R_i)` — `_continuous_ctd_cell_measures`
for continuous laws, `_discrete_ctd_cell_measures` for realizations, each
evaluated at the loss knots it is given — and the endpoint split, exterior policy
and finite-knot assembly follow from those in `_ctd_masses_from_cell_measures`,
which takes the knots and the widths of the cells between them and returns the
masses at the knots and the mass at `+inf`. A fixed-gap grid and explicit loss
points therefore share one cell computation. This
is algebraically the paper's Algorithm 1, factored so that no cell probability is
recovered from second differences of a rounded global cumulative curve. Sampling
the hockey-stick curve into binary64 first and inverting it is the approach this
implementation replaced: that inversion divided adjacent differences by
`1 - exp(-step)`, amplifying cancellation by `1/step`, and produced 11 523
negative masses at `sigma = 0.5, step = 1e-4` — 4% of knots — which it then
clipped under a threshold that grew as the grid was refined. Do not reintroduce
it.

**Do not revive:**

- dual residual mass on a finite loss knot;
- reassociated profile inversion (`expm1`/`log1p` after rounding the hockey-stick curve);
- clipping plus normalization as a validity repair;
- paired projection APIs that reintroduce profile inversion;
- endpoint split via `log(S_i/M_i)` instead of the cellwise `(M_i, R_i)` ledger.

Continuous CtD derives one finite grid from the union of the source quantiles
and the reflected-dual quantiles, with the caller's tail budget split between the
two range queries. The grid-cap calculation uses the same joint range, so
coarsening cannot satisfy its point cap by silently dropping the part of the
support needed by the dual; a source-only range leaves a semantic
reciprocal-moment loss orders of magnitude over budget. Source-only directional
discretization keeps its original source-quantile range. The functions on
explicit loss points take the knots, and with them the range, from the caller.
Exterior cells are
handled algebraically rather than by clipping: the lower tail collapses onto the
first knot, and the upper finite tail splits between the last knot and `+inf` so
that reflected-dual mass is preserved.

A realization can leave two kinds of reciprocal moment owing, and both are settled
by moving mass to `+inf`. The endpoint split preserves each cell's mass exactly but
its moment only up to rounding, so `_repair_ctd_reciprocal_moment` drains an
arithmetic excess that `classify_residual` has admitted as producer error; anything
larger is refused rather than repaired. The lower exterior cell then owes
`eta = R_minus - exp(-x_0) M_minus`, which is singular dual mass that must not sit on
a finite loss, and the semantic drain surrenders whatever the arithmetic repair did
not already give up. Draining only ever moves mass toward `+inf`, so it loosens the
bound and can never invalidate it.

The CtD source check is intentionally local to
`projections_ctd.py`: it validates the algorithm-specific semantic
contract after generic distribution constructors have already checked shape,
mass, and boundary nonnegativity. In particular, CtD cannot accept lower
infinite mass, non-real support, unordered support, or a reciprocal moment
above one. The reciprocal moment is `E[exp(-L)]`, the total mass of the implied
dual law on the source's finite support; `1 - E[exp(-L)]` is singular dual mass
and must stay at `+inf` rather than being moved onto a finite loss.

Interval endpoints use a right-closed convention on the loss axis: cell 0 is
`(-inf, loss[0]]`, interior cell `j` is `(loss[j-1], loss[j]]`, and the final
cell is `(loss[-1], +inf)`. An atom exactly on a knot therefore stays in one
cell. The reflected-dual measures are evaluated one ULP below the negated knots
(`nextafter`) to realize that convention on the dual side, which is what keeps a
privacy-loss atom at `L = epsilon` contributing exactly zero to the hockey-stick
integrand.

Epsilon and delta queries are not implemented here. `compose_full_pld(...)`
converts the directional PLDs to `dp_accounting` types and the queries are
answered by `dp_accounting`.

## Parameter Budget Conventions

Shared composition budgets are derived inline in `allocation_plds()`
and `_allocation_plds_core()` in
`PLD_accounting/random_allocation_accounting.py`. Both carry the two directions
together, as a pair ordered `(remove, add)`: the stage counts, the per-component
loss budgets and the distributions. `loss_discretization` and `tail_truncation`
are the budgets of each direction; building both directions does not halve them.

- `allocation_plds()` divides the tail budget by the number of active
  tail-consuming ops, `2 * component_count - 1`: one component core call per
  floor/ceil component plus the final `fft_convolve` when both are active. After
  that division each op consumes at most the rescaled budget, so all of them
  together stay within the caller's `tail_truncation`.

  It owns only the split *between* components; the per-epoch and per-count
  divisions belong to the two layers below it.

  - One active component: `loss_discretization` is passed through unchanged.
  - Both active: each direction's budget is split proportional to each
    component's effective discretization count in that direction —
    `floor_count = floor_epochs * count(floor_steps)` and the ceil analogue, with
    `count` the direction's member of `base_loss_discretization_counts`; note the
    component's own epoch multiplicity and step count, not `num_epochs` and
    `num_steps` — giving `loss_discretization * floor_count /
    (floor_count + ceil_count)` for the floor component and likewise for ceil.

  `_allocation_plds_core()` then divides its component's budgets by that
  component's own `num_epochs`. The base builder divides once more by the
  direction's stage count. The multiplications above and these divisions
  cancel, so within a direction both uncapped final steps reduce to the same
  `loss_discretization / (floor_count + ceil_count)` — which is what the final
  `fft_convolve` requires. Keeping the structure explicit makes each layer's budget
  ownership visible even though the real-number algebra cancels. The stage counts
  of the two directions can differ (rounding), so the remove and add steps need
  not be equal; nothing combines the two directions on one grid.

  The two final steps can still differ when the component-local grid cap binds.
  The floor and ceil components have different source ranges and factor budgets,
  so `grid_step_for_point_cap()` can independently raise their factor
  steps by different amounts. At `sigma=1`, `loss_discretization=1e-3`,
  `tail_truncation=1e-12`, `T=101`, `num_selected=3`, `num_epochs=5`, and
  `max_grid_mult=20000`, the REMOVE final steps of the floor and ceil components
  become `8.792390e-4` and `8.877958e-4`. With `loss_discretization=0.2` the cap
  stays inactive, and within each direction the two components share an exact
  step: `0.2 / 115` for REMOVE and `0.2 / 105` for ADD.

  `_align_component_grids()` reconciles a cap-bound mismatch directionally: a
  last-bit step difference is relabeled; a material cap-driven mismatch coarsens
  the finer component through `project_discrete_dist` and warns. That
  post-build reconciliation is the intended behavior.
- `_allocation_plds_core()` divides the component *tail* budget:
  by 3 (base creation, epoch composition, final truncation) when
  `num_epochs > 1`, or by 2 when the FFT phase is skipped (`num_epochs == 1`).
  `base_tail_truncation = tail_truncation / (3 * num_epochs)` covers up to 3
  tail-consuming sub-ops per epoch. The matching loss division is the per-epoch
  `/ num_epochs` already described above.
- On `GEOM`, the base build (`allocation_plds_base()`) divides each
  direction's loss budget by its exact discretizing stage count
  (`loss_discretization_counts()`) and splits the tail budget in two:
  half for the input and half for the merges, the same for both directions and
  both projections. Each one-step factor receives a further `1 / num_steps` of
  the input half to absorb self-convolution amplification. With
  `d(n) = ceil(log2 n)` (`binary_self_convolve_depth`) the largest number of
  merges above one copy in binary self-composition of `n` copies, which has up
  to `2 d(n)` merges in all; a merge scales a sum by at most the grid ratio and
  adding values keeps a common factor, so the errors accumulate along `d(n)`:

  | Projection | Direction | Loss stages (`num_steps > 1`) | Merge half of the tail |
  |---|---|---|---|
  | rounding | ADD | `1 + d(num_steps)` | the self-convolution of `num_steps` copies |
  | rounding | REMOVE | `2 + d(num_steps - 1)` | `(num_steps - 1) / num_steps` to the self-convolution of `num_steps - 1` copies, `1 / num_steps` to the closing merge with the factor under `P` |
  | connect-the-dots | either | `1 + d(num_steps)` | the self-convolution of `num_steps` copies |

  One step has one stage, the input, and no division of either budget.
- On `MS_FFT`, the base build divides each
  direction's loss budget equally among its stages
  (`loss_discretization_counts()`): the projection of the input onto
  the dyadic grid, one stage per level of the merge tree, and the closing
  projection onto the linear grid. With `d(n) = ceil(log2 n)` the depth of
  binary self-composition of `n` copies, as on `GEOM`:

  | Projection | Direction | Tree depth | Loss stages |
  |---|---|---|---|
  | connect-the-dots | either | `d(num_steps)` | `d(num_steps) + 2` |
  | rounding | ADD | `d(num_steps)` | `d(num_steps) + 2` |
  | rounding | REMOVE | `d(num_steps - 1) + 1`: `num_steps - 1` copies under `Q`, then the closing merge with the factor under `P` | `d(num_steps - 1) + 3` |

  One step has depth zero and two stages, the two projections.

  The share of one stage is `h = loss_discretization / stages`, which the base
  build hands to the factor builder as its loss budget. The factor builder
  derives the number of grid points per octave, `m`, with
  `dyadic_cells_per_band()`: the smallest integer with
  `log(2) / m <= h`, or `max_cells_per_band` when that is smaller. `log(2) / m`
  is the mean spacing of the logarithms of an octave's points, so the dyadic
  grid has as many points per octave as a linear loss grid of step `h`; the
  spacing runs from `log(1 + 1/m)`, at most `h / log(2)`, at the bottom of an
  octave to half of that at its top. Matching the mean spacing rather than the
  widest cell gives the projection of a one-step input onto the dyadic grid
  about the error of a projection onto a linear grid of step `h`. A merge with
  rounding moves less: a sum of two points of one octave lies on the half
  lattice of the next octave, so half of such sums are already grid points and
  the others move by half a cell. In our measurements on Gaussian inputs with
  `k = 1`, a merge with rounding raises the mean loss of its sum by
  `0.29 h` to `0.35 h` on the dyadic grid and `0.47 h` to `0.50 h` on the
  geometric grid. Where one
  summand of a sum dominates, a small other summand puts the sum just above a
  grid point, and a level moves it by nearly a whole cell on either grid. The
  bound on the error of a level is the widest cell, `log(1 + 1/m)`, on the
  dyadic grid and `h` on the geometric one. The grid and the merge
  accept every positive integer `m`; the transforms of a merge have the first
  fast length of at least `4m` (`next_fast_len`), whatever `m` is. `m` is never set, and
  the base build reads it from the factors it receives. The returned linear
  grid has the step `max(h, log(2) / m)`, which is `h` unless the cap on `m`
  binds, and is coarsened further only when `max_closing_grid` binds. Either
  cap warns when it binds. For `num_steps = 1000` and a loss budget of `0.02`
  with connect-the-dots: depth `10`, `12` stages, `h = 1.67e-3`, and
  `m = 416`, since `log(2) / h = 415.9`; the closing step is `h`. The rounding remove direction of the same
  allocation has depth `11` and `13` stages.

  The tail budget is split in two, the same for both directions and both
  projections: half bounds the range of the input and half is spent by the
  closing projection, `project_discrete_dist`. Each one-step factor receives
  `1 / num_steps` of the input half, and its grid spans the smallest range
  `[2^-r, 2^r]` outside which each of four tails, of the law and of its dual,
  below and above the range, is at most a quarter of that: `T / (8 num_steps)`
  for a base tail budget `T`. The merge tree spends no tail budget, since it
  does not trim. A range with `r >= MAX_DYADIC_REACH` raises.

  The caps act on each floor/ceil component separately, so a cap that binds can
  leave the two components on different steps, which `_align_component_grids()`
  reconciles as it does for the geometric cap.

`_fft_self_convolve_direct()` splits its own `tail_truncation` into four equal
quarters: the Chernoff window, the circular-alias reserve, the explicit
opposite-side trim, and the final truncation of the edges by
`projections.truncate_edges`. The alias reserve is the one
that is not always spent. The transform has period `fft_size`; when the exact
`num_convolutions`-fold support is longer than that, output mass at index
`j >= fft_size` folds onto `j mod fft_size`, and no sum over the circular array can
say how much did, because the fold preserves mass. Two window indices cannot
collide with each other, so everything misplaced comes from outside the retained
window, which the window was sized to hold to one quarter of the budget. That
allowance is banked at the conservative boundary and funded by trimming the same
mass from the giveable edge, so the ledger stays a move rather than a widening and
mass conservation still sees only transform drift. `_declared_alias_shift()`
returns `0.0` when the period covers the true support, where the convolution is
exact and nothing is owed.

The Gaussian and the realization paths share the wiring after factor creation:
- both call `allocation_plds_base(...)`, which runs one flow on either grid
  through the functions of `convolutions.py`;
- only the base distribution creation differs (a continuous law and its dual vs an
  explicit realization).

The realization path divides its loss budgets by the same stage counts. On
`GEOM` the grid step of a realization is a floor on the step of its one-step
factors, and a realization coarser than its stage's share warns; on `MS_FFT` the
realization sets no floor.

All `require_*` functions and methods are standalone checks returning `None`.
They never supply converted or narrowed values. Checked `cast_*` functions
validate the runtime type and any representation constraints, then return the
original object with a narrowed type, and are called only where the static type
narrows; a constraint on a value that keeps its type is a `require_*` call, and
no function re-checks an argument against its own annotated class, with one
exception: the `general_allocation_*` entry points call `require_pld_realization`
on both inputs, because a lower-bound mechanism discretization is a
`DenseDiscreteDist`, not a PLD, and the builder would otherwise take a dual of it.
A value that passed its check is used as it is, without a conversion to its
annotated type, and no function clamps a value into the range it checks.
`enforce_mass_conservation` requires its expected boundaries to lie in `[0, 1]`
exactly and repairs drift in the finite masses only. Representation casts
preserve object identity. `PrivacyParams.get_delta` and
`PrivacyParams.get_epsilon` check presence and return the existing field value.

## File Map

| File | Responsibility |
|---|---|
| `PLD_accounting/__init__.py` | Public exports. |
| `PLD_accounting/types.py` | Enums and configs (`PrivacyParams`, `AllocationSchemeConfig`, `BoundType`, etc.). `PrivacyParams` and `AllocationSchemeConfig` validate field invariants on construction. |
| `PLD_accounting/random_allocation_api.py` | Public entry points for Gaussian and realization accounting. Each documented function validates its own arguments. |
| `PLD_accounting/random_allocation_accounting.py` | Composition that no grid owns, for both directions together: the floor/ceil split with its budgets, epoch composition, and the `dp_accounting` conversion. |
| `PLD_accounting/random_allocation_base.py` | One allocation, both directions, on either grid: budgets and stage counts, the sum of likelihood ratios for each projection, the conversion of an averaged sum to a linear loss grid, and the choice of the remove computation (its own sum for rounding, the dual of the add PLD for connect-the-dots). Ratio conversion and merges are selected through `convolutions.py`; closing loss conversion selects by representation here. |
| `PLD_accounting/convolutions.py` | Selects the law of likelihood ratios of a loss factor, pairwise convolution, self-convolution, and the stage count by convolution method. Multiscale admission stays inside its backend; conversion of the averaged sum to losses stays in `random_allocation_base.py`. |
| `PLD_accounting/random_allocation_continuous.py` | One-step factors of both directions from a continuous privacy-loss law and its dual: one public builder, which holds the flow of both directions and selects the grid by the convolution method, and the projections onto the fixed-gap loss grid and onto the logarithms of the dyadic grid of likelihood ratios. |
| `PLD_accounting/random_allocation_realization.py` | One-step factors of both directions from `PLDRealization` inputs: one public builder, which holds the flow of both directions and selects the grid by the convolution method, and the projections onto the fixed-gap loss grid and onto the logarithms of the dyadic grid of likelihood ratios. |
| `PLD_accounting/adaptive_random_allocation.py` | Adaptive upper/lower range refinement for epsilon queries, using `GEOM`. |
| `PLD_accounting/mechanisms.py` | Mechanism PLD factory helpers (`gaussian_distribution`, `laplace_distribution`, `discrete_distribution`) and the Gaussian privacy-loss law (`gaussian_loss_law`). |
| `PLD_accounting/validation.py` | Constraint primitives (`require_positive_real`, `require_enum`, `require_allocation_counts`, ...), named by the fact they enforce rather than by caller. Scalar helpers accept one `(value, name)` pair or parallel sequences of values and names. Repeated enum constraints live next to those types (`require_bound_type`, `require_direction`). `cast_type` checks a runtime class and returns the object typed as that class, for a caller whose static type narrows; no function re-checks an argument against its own annotated class. Dense/linear/domain checks live on `discrete_dist.cast_dense_dist`, with `cast_linear_reals_dist` / `cast_geometric_positives_dist` and their `require_*` twins for the two usual lattices. |
| `PLD_accounting/discrete_dist.py` | `GridSpec`, distribution classes (`DenseDiscreteDist`, `SparseDiscreteDist`, `PLDRealization`, `Domain`) and the dyadic-grid measure `DyadicDist`. |
| `PLD_accounting/projections.py` | One function per projection operation that a caller selects: `project_continuous_dist`, `project_discrete_dist` and their counterparts on explicit points, `project_continuous_dist_on_points` and `project_discrete_dist_on_points`, by the bound, `truncate_edges`, `truncate_tails`, `project_grid_offsets` and `halve_lattice` by the projection method. |
| `PLD_accounting/projections_ctd.py` | Connect-the-dots rounding: continuous-to-discrete conversion and rediscretization on a fixed-gap loss grid or on explicit loss points, the projection and tail trim of a merge on a geometric grid, and the halving of a lattice at a merge on the dyadic grid. |
| `PLD_accounting/projections_stoch_dom.py` | Stochastic-domination rounding: continuous-to-discrete conversion and rediscretization on a fixed-gap grid or on explicit points, and the projection of a merge on either grid. |
| `PLD_accounting/fft_convolution.py` | FFT-based convolution and self-convolution on linear grids. |
| `PLD_accounting/geometric_convolution.py` | Convolution on geometric grids, with the projection of a pairwise sum (rounding or connect-the-dots) as a parameter. |
| `PLD_accounting/multiscale_fft_convolution.py` | Convolution and self-convolution of `DyadicDist` measures by FFTs inside each octave, with the same projection parameter and the admission of each result, and the grid of a one-step input: its points per octave (`dyadic_cells_per_band`), its points over a range and the law of a likelihood ratio on them from a loss distribution on their logarithms (`dyadic_range_points`, `dyadic_dist_on_range`). |
| `PLD_accounting/utils.py` | The aligned linear grid that covers a range (`aligned_grid_params`) and its step under a point cap (`grid_step_for_point_cap`), and PLD transforms (`exp`, `log`, the dual of a realization and of a loss distribution on explicit points, negate-reverse on a grid and on explicit points, composition helpers). |
| `PLD_accounting/distribution_utils.py` | Mass and reciprocal-moment repair policy: the drift/repair bands, `enforce_mass_conservation`, `classify_residual`, the mass tolerances of an FFT (`transform_mass_tolerances`), `trim_mass_to_moment_target`, compensated summation and the exact total `exact_sum`, and the range and interval masses of a continuous law (`joint_source_dual_bounds`, `partition_masses`). |
| `PLD_accounting/dp_accounting_support.py` | Conversion between internal probability representations and `dp_accounting` PMF/PLD types. Import repairs use caller-visible drift and repair bands; the defaults are calibrated for uncomposed PMFs. |
| `PLD_accounting/subsample_pld.py` | PLD-level subsampling amplification helpers (DOMINATES-only path). |

## Public API Surface

Random allocation (defined in `PLD_accounting/random_allocation_api.py`):

- Gaussian path:
  - `gaussian_allocation_pld(...)`: both directions. Builds the Gaussian law once
    with `gaussian_loss_law(...)`, resolves the projection once with
    `resolve_projection_method(...)`, binds three partials and makes one
    `allocation_plds(...)` call. It names the convolution method and does not
    branch on it:

    | Partial of | Bound arguments |
    |---|---|
    | `continuous_base_distributions` | the law and its dual, `convolution_method`, `max_grid_mult`, `max_cells_per_band` |
    | `allocation_plds_base` | the partial above, `projection_method`, `convolution_method`, `max_closing_grid` |
    | `loss_discretization_counts` | `projection_method`, `convolution_method` |

    The wiring is the same for both grids and both projections.
  - `gaussian_allocation_epsilon_configurable(...)`
  - `gaussian_allocation_delta_configurable(...)`
  - `gaussian_allocation_epsilon_range(...)`
- Realization path:
  - `general_allocation_pld(...)`: both directions, wired like the Gaussian entry
    with `realization_base_distributions` bound to the two realizations.
  - `general_allocation_epsilon(...)`
  - `general_allocation_delta(...)`

Composition conversion (defined in `PLD_accounting/random_allocation_accounting.py`):

- `compose_full_pld(...)`: converts internal directional PLDs into a
  `dp_accounting` PLD object. With `add_dist=None` the result is symmetric:
  `dp_accounting` reuses the remove PMF for the add direction.

Fixed-gap projection of a discrete distribution (defined in `PLD_accounting/projections.py`):

- `project_discrete_dist(*, dist, tail_truncation, loss_discretization, bound_type, max_grid_points=-1)`
  - Public fixed-gap real-loss entry. Bound direction selects CtD for
    `DOMINATES` and directional stochastic domination for `IS_DOMINATED`. A
    positive `max_grid_points` coarsens the step, through
    `grid_step_for_point_cap`, so that the grid over the support left after
    truncation has at most that many points; the returned grid carries the step
    used.

Mechanism PLD helpers (defined in `PLD_accounting/mechanisms.py`):

- `gaussian_distribution(*, scale, value_discretization, tail_truncation, bound_type)`
- `laplace_distribution(*, scale, value_discretization, tail_truncation, bound_type)`
- `discrete_distribution(*, noise_dist, loss_discretization, tail_truncation, sensitivity=1)`
  - Maps both input noise-boundary masses (`p_min` and `p_max`) to positive-infinity privacy loss; they are separate unmatched tails and need not agree.

Subsampling (defined in `PLD_accounting/subsample_pld.py`):

- `subsample_pld(*, pld, sampling_probability)`
- `subsample_pld_realization(*, base_pld, sampling_prob, direction)`

dp_accounting support (defined in `PLD_accounting/dp_accounting_support.py`):

- `dp_accounting_pmf_to_pld_realization(*, pmf, ...)` — admits a pessimistic
  `dp_accounting` PMF as a `PLDRealization`. Keyword-only drift and repair
  bands default to uncomposed-PMF calibration; callers widen them only when
  the input's provenance justifies it.

Distribution types (defined in `PLD_accounting/discrete_dist.py`):

- `GridSpec(step=..., n=..., spacing_type=..., anchor=..., index_0=...)` —
  immutable lattice. `GridSpec.geometric(*, ratio, n, anchor, index_0=0)` stores
  `step=math.log(ratio)`; direct `GridSpec(step=..., spacing_type=GEOMETRIC)`
  remains valid. There is no `ratio` field or property.
- `Domain` — `REALS` / `POSITIVES` boundary semantics.
- `DenseDiscreteDist(grid=..., prob_arr=...)` — general regular-grid
  distribution. Integer count noise is one use, via `discrete_distribution`.
  `x_0`, `step`, and `x_array` are derived from the grid; `step` is the stored
  spacing, which is the log ratio when geometric.
- `PLDRealization(grid=..., prob_arr=...)` — linear-grid privacy-loss
  distribution used as input to realization APIs.

Supporting exports (defined in `PLD_accounting/types.py`):

- `BoundType`, `Direction`, `SpacingType`, `ConvolutionMethod`,
  `ProjectionMethod` — the enums that select bound direction, adjacency
  direction, lattice family, backend and the projection at the merges.
  `ConvolutionMethod` has the members `GEOM` (geometric grid, direct
  convolution) and `MS_FFT` (dyadic grid, FFTs inside each octave).
  `resolve_projection_method(*, config, bound_type)` in `types.py` is the one
  place that maps a configuration to the projection that runs: `None` is
  rounding (`STOCH_DOM`) on the geometric grid and connect-the-dots (`CTD`) on
  the dyadic grid.
  `require_projection_method(*, value, bound_type)` checks its compatibility
  with the bound for the resolver, projection operations, geometric, multiscale
  and FFT convolution, and allocation builders. It rejects connect-the-dots with
  `IS_DOMINATED` before any identity return or numerical work.
- `PrivacyParams`, `AllocationSchemeConfig` — frozen configuration records that
  validate their own fields on construction. `max_grid_mult` applies to `GEOM`
  only; `max_cells_per_band` and `max_closing_grid` apply to `MS_FFT` only, with
  the defaults `DEFAULT_MAX_CELLS_PER_BAND` and `DEFAULT_MAX_CLOSING_GRID` of
  `types.py`. A cap on the points of a linear grid, `max_closing_grid` or a
  positive `max_grid_mult`, is at least `MIN_GRID_POINT_CAP`: the step that
  `grid_step_for_point_cap` returns covers the range in `cap - 3` intervals,
  which are `cap - 2` points, and the alignment of the grid to multiples of the
  step adds up to two more.
- `DEFAULT_LOSS_DISCRETIZATION`, `DEFAULT_TAIL_TRUNCATION` — the defaults
  `AllocationSchemeConfig` applies.
- `has_numba()` — whether the optional Numba acceleration is active. Results are
  unaffected; only speed and last-bit summation order differ.

Notes:

- PLD builders reject `BoundType.BOTH`; users build separate DOMINATES and IS_DOMINATED PLDs.

## Core Composition Modules

### `random_allocation_accounting.py`

This is the shared composition core used by both Gaussian and realization accounting.
It holds the floor/ceil split and epoch composition. Its base-PLD callback is
`allocation_plds_base`, bound to the input builder, grid and projection.

Key functions:

- `allocation_plds(...)`:
  Applies adaptive step decomposition once, splits each direction's loss budget,
  builds the floor and ceil components, and combines them within each direction
  (`_combine_components(...)`). Returns the pair `(remove, add)`.
- `_allocation_plds_core(...)`:
  Calls the base-PLD callback once for both directions of a component, then
  passes each base PLD to `_compose_base_pld_over_epochs(...)`.
- `_compose_base_pld_over_epochs(...)`:
  Trims the edges of one base PLD, caps its size when `fft_self_convolve` would
  exceed the FFT memory limit, composes across epochs, then trims the composed
  result. It does not regrid to the output discretization; the output step is
  whatever the base builder produced from the budget it was handed.
- `compose_full_pld(...)`:
  Converts internal directional PLDs into a `dp_accounting` PLD object.

#### Why connect-the-dots merges give an upper bound

Under `Q` the likelihood ratio of one allocation of `t` steps is the average
`Z = S / t` of `t` independent one-step ratios. Both privacy profiles are
expectations under `Q` of convex, decreasing functions of `Z`:

- ADD: `delta(eps) = E[(1 - exp(eps) Z)_+]`.
- REMOVE: `delta(eps) = E[(Z - exp(eps))_+] + (1 - E[Z]) = 1 - E[min(Z, exp(eps))]`,
  where `1 - E[Z]` is the mass of `P` on which `Q` is zero.

A merge that splits every pairwise sum between its two neighbouring grid points
with the weights that keep its mean replaces the sum by a mean-preserving spread
of it. That raises the expectation of every convex function, and the property
survives adding further independent copies, so every later merge and the final
profiles are bounded from above. Moving mass to a smaller value of the sum also
raises both profiles, since both functions decrease. These two moves are the only
ones a connect-the-dots sum makes, which is why its layout has no factor under
`P`: a mean-preserving spread of a law sampled under `P` raises no such
expectation. The same argument gives no lower bound, so a lower bound needs
rounding.

#### Grid, projection and bound

The grid and the projection are independent settings, and the bounds a
combination gives depend on the projection alone:

| Grid | Projection | Bounds |
|---|---|---|
| `GEOM` | `STOCH_DOM` (the grid's own) | upper and lower |
| `GEOM` | `CTD` | upper only |
| `MS_FFT` | `CTD` (the grid's own) | upper only |
| `MS_FFT` | `STOCH_DOM` | upper and lower |

### `convolutions.py`

The operations on laws of likelihood ratios that a caller selects by the
convolution method. The module is to `geometric_convolution.py` and
`multiscale_fft_convolution.py` what `projections.py` is to `projections_ctd.py`
and `projections_stoch_dom.py`.

| Operation | `GEOM` | `MS_FFT` |
|---|---|---|
| `loss_to_ratio_dist(...)` | `exp_linear_to_geometric` | `dyadic_dist_on_range` |
| `convolve(...)` | `geometric_convolve` | `multiscale_convolve` |
| `self_convolve(...)` | `binary_self_convolve` on `geometric_convolve` | `multiscale_self_convolve` |
| `loss_stage_count(...)` | the input and the merges above one copy | the input, the merges above one copy and the closing projection |

A function of the gate takes the arguments of both implementations and passes
each the ones it reads. Both merges take a numeric `tail_truncation`; the gate
passes zero on `MS_FFT`, whose merges do not trim. Both grids sum copies by
repeated squaring, building powers of two and accumulating the powers selected
by the binary expansion from smallest to largest. The maximum depth is
`ceil(log2 n)` and the number of distinct merges is
`floor(log2 n) + popcount(n) - 1`. Self-products reuse the same operand and halve
the dyadic transforms.

Both backends use `binary_self_convolve` in `utils.py`, which preserves the
operand type and calls the supplied merge with an explicit tail keyword. The
geometric tail schedule is retained; zero keeps every dyadic merge untrimmed.
The dyadic backend supplies its raw merge and admits the completed sum once.
The shared loop contains no representation-specific validation or admission.
The loop divides the tail budget by four, then each merge receives that share
divided by the current copy counter (the halved counter for a squaring).
Squaring and accumulation each spend less than twice their largest charge, so
both families together stay within the original budget.

Repeated squaring is the algorithm the papers analyse. An earlier version of
this package used a balanced recursion instead, which merges the sums of
`floor(n/2)` and `ceil(n/2)` copies and has the same depth. A one-off allocation
probe at `loss_discretization = 0.02` and `tail_truncation = 1e-9`, including
Gaussian and realization inputs, found a rounding upper epsilon of
`0.08493710434928026` with the balanced recursion and `0.08573443555405082` with
squaring at `sigma = 3`, `t = 100`, one selection and epoch, `delta = 1e-4`:
0.94% looser. CtD upper bounds differed by at most `1.254e-4` relative over the
probe. Neither figure is a general bound on the change in accuracy. A timing
sample at the same loss resolution measured squaring 0 to 15% faster on the
dyadic grid and the balanced recursion 10 to 45% slower on the geometric grid.
These timings are indicative because other jobs were running; equal depth does
not imply equal work or equal running time.

### `random_allocation_base.py`

One allocation of `num_steps` steps, written once for both grids. The shared
`LossFactor` alias in `discrete_dist.py` is `DenseDiscreteDist | SparseDiscreteDist`;
both input builders and the base builder's callback use the paired return type
`tuple[tuple[LossFactor, LossFactor], LossFactor]`.

- `allocation_plds_base(...)`: divides the budgets (see Parameter Budget
  Conventions), calls the paired one-step factor builder of the input kind once,
  turns the loss factors into laws of likelihood ratios with
  `loss_to_ratio_dist`, sums them with `self_convolve` and `convolve`, and
  turns each sum into the loss of the average ratio on a linear grid. It returns the pair
  `(remove, add)`. The add PLD is computed one way; the projection sets only the
  projection of its merges and decides how remove is computed:

  | Projection | Add base | Remove base |
  |---|---|---|
  | connect-the-dots | `t` copies under `Q`, then `-ln(S / t)` | `calc_pld_dual` of the add PLD of the same sum, on its linear grid |
  | rounding | `t` copies under `Q`, rounded to the other side, since the add loss decreases with the sum | `t - 1` copies under `Q`, then one merge with the factor under `P`, then `ln(S / t)` |

  The factor builder returns the remove loss factors `(base, neg_dual_base)` and
  the add `base`, on fixed-gap grids for `GEOM` and on the logarithms of dyadic
  points for `MS_FFT`. Several steps sum the reflected add base for the add PLD;
  the rounding remove sum reads `neg_dual_base` and the remove base, the factor
  under `P`, and a connect-the-dots build of several steps does not read the
  remove factors. One step makes no merge: on `GEOM` each returned factor passes
  `cast_dense_dist`, which checks the representation and returns the original
  factor. Its direction is named in a type error. On `MS_FFT`
  each is still projected onto its linear grid.
- `loss_discretization_counts(...)`: the stage counts `(remove, add)` of the
  tables in Parameter Budget Conventions. A direction that sums `n` copies has
  `loss_stage_count(n)` stages; the rounding remove direction sums
  `num_steps - 1` copies and adds the merge with the remove base.

The functions of `random_allocation_accounting.py` know only the paired
distributions, budgets and stage counts. They do not branch on the projection or
on the input kind.

#### From a geometric sum to a loss distribution

A sum `S` of unit-anchored factors has the exact integer anchor `t`, so dividing
the anchor by `t` gives the lattice of the average `Z = S / t` with anchor one.
`log_geometric_to_linear` maps it to a zero-anchored loss grid, and
`negate_reverse_distribution` reflects it where the loss is `-ln Z`. A
rounding sum ends there: the add loss is the reflection, and the remove loss is
the logarithm of the sum that carries the factor under `P`.

A connect-the-dots sum is a law under `Q`, and it is the add PLD after the same
two transforms: `A = -ln Z`, with the sum's atom at zero as its mass at `+inf`.
The builder wraps `A` in a `PLDRealization` on that final grid, which checks
`E[exp(-A)] = E[Z] <= 1` on the grid's own coordinates. The add member of the
pair is `A`. The remove member is `calc_pld_dual(A)`: the reflected support, the
finite mass `z f(z)` at `ln z` for each mass `f(z)` of the sum, and
`max(0, 1 - E[Z])` at `+inf`. The mass at `+inf` of the remove direction is
therefore derived by the dual from the sum; no caller computes or carries it.

#### The remove direction of a connect-the-dots build

Both privacy profiles of a connect-the-dots build are functions of one sum under
`Q`, so `allocation_plds_base` sums the reflected add base once and returns
`(calc_pld_dual(A), A)`. A query therefore makes one sum per floor/ceil
component, for either input kind. On both grids `A` is a `PLDRealization` on
its linear grid, and the pair is two exact duals on one grid.

The remove PLD of a mechanism is the dual of its add PLD, and the dual of an
upper bound of the add PLD is an upper bound of the remove PLD. The remove bound
is valid when the add input bounds the mechanism that the remove input
describes. A continuous law is supplied with its exact dual. Two realizations
passed to `general_allocation_pld` must describe one mechanism. Both realizations
are prepared and validated for every projection. With connect-the-dots and more
than one step per allocation, the prepared remove factors are not used in the
sum; the remove PLD is derived from the add PLD. Rounding uses each direction's
own factors, and a component of one step returns each direction's own base.

The dual is taken at the base PLD on its linear grid, before
`_compose_base_pld_over_epochs` truncates each direction and composes it over
the epochs. Those steps are directional approximations, so a remove PLD derived
after them would not be the dual of a valid add bound of the same budget; after
them the two members need not be duals of each other.

On the geometric grid the base PLD has had no truncation of a direction. On the
dyadic grid it is the add loss distribution after its projection onto the
linear grid, so one closing projection runs for the pair. The connect-the-dots
projection keeps the mass and the reciprocal moment of every cell, and those are
the two quantities the dual exchanges, so projecting and then dualizing puts the
same masses on the reflected points as dualizing and then projecting. The order
changes only how the truncation of the tails is charged: the mass that the trim
of the add PLD moves up at its low-loss end appears as mass at `+inf` of the
remove direction, bounded by the same tail budget. Measured across the
comparison settings, the order changes a connect-the-dots epsilon by at most
`3.4e-9` relative.

### `geometric_convolution.py`

`geometric_convolve(...)` enumerates every pair of grid points of its two inputs,
so it projects the exact sum. The module holds what is specific to the geometric
grid: the position of every pairwise sum on the output lattice, in units of grid
points; the kernel that accumulates the products of masses; and the output
lattice. The projection of a position is selected by
`projections.project_grid_offsets(...)` and lives in the module of its kind of
rounding, `projections_stoch_dom.py` and `projections_ctd.py`:

- `project_grid_offsets_stoch_dom(...)` returns the grid point above each
  position for `DOMINATES` and the one below it for `IS_DOMINATED`.
- `project_grid_offsets_ctd(...)` returns the grid point below each position and
  the share of mass the next point takes. For a sum at the fractional offset `a`
  (in units of the log step `s`) the share is `(exp(a s) - 1) / (exp(s) - 1)`,
  the weight that keeps its mean.

One kernel serves both: every pair contributes its product of masses to a lower
output bin and sends a share `w` of it to the next bin, and rounding is `w = 0`.
The cross-terms of the atom at zero, which is neutral under addition, are
projected by the same two functions, and under connect-the-dots the lattice is
extended where such a term lies outside the bins of the finite pairs. The numba
kernel and its NumPy twin apply the same per-bin Kahan updates in the same order.

After a connect-the-dots merge:

- The mass is repaired by `enforce_mass_conservation(...)` in the direction of a
  dominated rounding, which moves mass down: a surplus is removed from the largest
  sums and a shortfall goes to the smallest one. Moving mass to a smaller value is
  a move a connect-the-dots sum may make, so the repair keeps the bound; it is the
  repair the add sum receives under rounding. The gap is the rounding residual of
  the compensated kernel and is admitted inside the default bands of
  `distribution_utils.py`. A surplus lowers the mean by the residual times the
  largest sums, which `calc_pld_dual` turns into mass at loss `+inf` of the remove
  direction; on a lattice spanning `exp(-20)` to `exp(20)`, 64 merges lowered the
  mean by `9e-6` relative. Keeping the mean is preferred where the bound allows
  it, and a repair that keeps the mean of a surplus does not keep the bound.
- The tails are trimmed by `truncate_tails_ctd(...)` of `projections_ctd.py`, a rule that keeps the mean where it can be kept. Below
  the first kept grid point, the first moment of the trimmed region stays at that
  point and the rest of its mass moves to the atom at zero: a mean-preserving
  spread. Above the last kept grid point, the mass of the trimmed region moves to
  that point: mass moves to a smaller value and its excess first moment is
  dropped. Both moves raise both profiles, so one rule is valid for both
  directions, and the two directions use one summed distribution. The dropped
  moment lowers `E[Z]`, so `calc_pld_dual` turns it into mass at loss `+inf` of
  the remove direction.
- A region is trimmed only while its mass under `Q` and its mass under `P`, the
  first moment divided by the grid anchor, are both at most the tail share of the
  merge. On unit-anchored factors the anchor of a node is the number of copies it
  sums, so the second charge bounds what the upper trim adds to the remove
  direction's mass at `+inf`. At least one grid point is always kept.

With rounding, nothing after the kernel differs from a merge without the
projection parameter: boundary masses, `enforce_mass_conservation` and the
truncation of `truncate_tails_stoch_dom`.

The same trim applies to a real-loss PLD in the coordinate `exp(-L)`, which grows
as the loss falls: `truncate_edges_ctd(...)` keeps the reciprocal moment of the
region above the last kept loss at that loss and moves the rest of its mass to
`+inf`, and moves the mass of the region below the first kept loss to that loss.
`delta(epsilon)` is the expectation of a convex decreasing function of `exp(-L)`,
so both moves raise it, and the first moves less mass to `+inf` than rounding
does. It is the truncation of `project_discrete_dist_ctd(...)` and, with the
connect-the-dots projection, of the stages of an allocation after the base
build. The FFT window itself rounds in the direction of the bound for both
projections.

The mass repair of a merge, the moment check of the `PLDRealization` that holds
the add PLD, and the trim of a moment excess inside `calc_pld_dual` are admission
checks of the floating-point computation: each admits a residual of rounding
size. They are not error certificates. The certificate of a connect-the-dots
bound is the convex-order argument above, applied to exact arithmetic.

#### Laws of likelihood ratios on the dyadic grid

On `MS_FFT` the factor builder returns the loss factors on the logarithms of
dyadic points. `loss_to_ratio_dist` turns each one that is summed into a
`DyadicDist` law of a likelihood ratio with `dyadic_dist_on_range`, the add
factor after its reflection: the law of `exp(L)` keeps the masses on their
points and keeps the boundary atoms of the factor: its mass at `-inf` is the
atom at zero and its mass at `+inf` the mass at infinity. The projections that
built the factor placed those atoms by the bound, so the law under `P` of an
upper bound has its atom at infinity and that of a lower bound at zero; the
negated dual has it at zero, and the add law at zero for an upper bound and at
infinity for a lower bound.

#### From a multiscale sum to a loss distribution

A sum `S` keeps its masses on its grid points. The loss of the average ratio is
`ln S - ln t` on those points as a `SparseDiscreteDist`; the atom at zero is the
loss `-inf` and the mass at infinity is `+inf`. The add direction reflects it
with `negate_reverse_distribution`. `project_discrete_dist` then
truncates it and projects it onto the linear grid. The three steps are private helpers of
`random_allocation_base.py`. With connect-the-dots the add
member is that `PLDRealization` and the remove member is its `calc_pld_dual`,
which derives the mass at `+inf` from the reciprocal moment of the add PLD.

A `DyadicDist` sum misses unit mass by the rounding of its transforms.
The repair divides the whole measure, both boundary atoms included, by its
total mass. A common factor moves no first moment between points, whereas a
residual moved to one end of the support costs the first moment of that end.
Measured at `sigma = 0.3`, `t = 10`, `m = 2^14` and a tail budget of `1e-18`:
draining a surplus of `2e-16` from the low-loss end of the add distribution
removes `1.2e-7` of its reciprocal moment, because those points carry likelihood
ratios up to `2^50`, and the dual turns that into `1.2e-7` of mass at `+inf` of
the remove direction. With the common factor the remove mass at `+inf` stays
within the tail budget.

#### Accuracy of the dyadic route

Two statements, about different references:

- Relative to the discretized one-step input, the merge tree displaces every
  likelihood ratio by at most the factor `(1 + 1/m)^depth`, which is a loss of
  `depth * log(1 + 1/m) <= depth * h / log(2)`. A merge moves a sum to one of
  its two neighbouring grid points (rounding) or splits it between them with
  the mean kept (connect-the-dots), the ratio of neighbours is at most
  `1 + 1/m`, and a value passes through at most `depth` merges.
- The projection of the input onto the dyadic grid adds one more stage of at
  most `h / log(2)`, and the closing projection onto the linear grid one of at
  most `h`. The loss budget is divided by `depth + 2` so that each stage's mean
  spacing is `h`, as each stage's step is on the geometric grid; the widest
  cell of an octave is `1 / log(2)` times the mean, so the worst-case
  displacement is at most `loss_discretization / log(2)` while
  `max_cells_per_band` does not bind; a binding cap widens every dyadic cell to
  `log(1 + 1/m)` for the capped `m`. Rounding moves a
  value by a fraction of its cell, so the realized error is far below the
  worst case on both grids.

### `multiscale_fft_convolution.py`

The dyadic grid is `{0}` with the points `2^j (1 + i/m)` for `i = 0..m-1`: `m`
uniform points per octave `[2^j, 2^(j+1))`, with spacing `h_j = 2^j / m`, so two
neighbouring points have ratio at most `1 + 1/m`. `m` is any positive integer.
The lattices of the octaves are nested for every `m`: every grid point at or
above `2^j` is a multiple of `h_j`. The module also holds the grid of a one-step
input: `dyadic_cells_per_band`, `dyadic_range_points`, `dyadic_dist_on_range`
and `MAX_DYADIC_REACH`.

`multiscale_convolve(...)` returns the pairwise sum of two measures projected
back onto the grid. The operands need not have unit mass: the atom at zero of
the sum is the product of the two atoms, and its mass at infinity is each
operand's mass at infinity times the other's mass, counted once where both are
at infinity, so the mass of a merge is the product of the masses. It groups the pairs by the octave `j` of their larger
summand. That summand lies on the lattice `h_j Z`; the smaller one is first
projected onto `h_j Z`, and the sum, which lies in `[2^j, 2^(j+2))`, is then on
`h_j Z` too. Each group is therefore a convolution on a uniform lattice, and
one merge of `J` octaves costs `O(J m log m)`. In exact arithmetic the result is
the pairwise convolution followed by the projection.

The array layout, in `_merge_positive_pairs` and the kernels it calls:

- Each octave is a row of exactly `m` masses.
- The *projected prefix* of octave `j` is all mass below the octave projected
  onto `h_j Z`. It occupies `m + 1` points, the multiples `0..m`, whatever the
  number of lower octaves. `_projected_prefixes` builds it by
  `prefix[j + 1] = halve(prefix[j] + block[j])`, where
  `projections.halve_lattice(...)` projects the points `0..2m` onto the even
  ones; the lattices are nested, so the recurrence is exact for every
  projection.
- In units of `h_j` a block occupies the indices `m..2m-1` and a prefix `0..m`,
  so their products occupy `m..4m-2` and transforms of length `4m` do not wrap
  around. `_band_products` takes one inverse transform of `F G + F V + U G` for
  blocks `F`, `G` and prefixes `U`, `V`: block with block, and each block with
  the other operand's prefix. Prefix with prefix is absent, since such a pair
  belongs to a lower octave. A self-product is `F (F + 2U)` and needs half the
  forward transforms.
- `_route_band_products` reads the result. Indices `m..2m-1` are the points of
  octave `j`. Indices `2m..4m-2` lie in octave `j + 1`, whose points are the
  even ones, so the same halving applies: `halve_lattice_ctd` in
  `projections_ctd.py` splits an odd index equally between its two neighbours,
  which keeps mass and first moment, and `halve_lattice_stoch_dom` in
  `projections_stoch_dom.py` moves it to the upper neighbour for `DOMINATES`
  and to the lower one for `IS_DOMINATED`. Indices below `m`, the index `4m - 1` and negative values are
  transform noise and are set to zero.

The octaves are transformed in batches under a memory budget; the result does
not depend on the batch size. The atoms at zero and at infinity stay outside the
transforms: zero is neutral under addition, so its cross-terms are the other
operand's own octaves, and the mass at infinity is absorbing
(`convolve_boundary_masses`).

`multiscale_self_convolve(...)` sums `n` copies through the shared
`binary_self_convolve` loop with a zero tail budget. It builds powers of two and
merges the powers
selected by the binary expansion in ascending
order. The maximum depth is `ceil(log2 n)` and there are
`floor(log2 n) + popcount(n) - 1` distinct merges. Squaring passes the same object
as both operands and takes the self-product path. The loop calls the raw merge
and admits the completed sum once against its copies. It does not trim: a merge
adds at most one octave at the top, and nothing is removed from either end.

#### Admission of a multiscale sum

`multiscale_convolve(...)` checks its result against its two operands, and
`multiscale_self_convolve(...)` checks its sum once against its copies. Three
residuals of the floating-point computation are judged, each through
`classify_residual`:

- The mass on grid points that no sum of the leaves' values can reach after
  `depth` projections. It is judged and then set to zero.
- The total mass against the product of the leaf masses.
- The first moment of the finite part against its exact value, an exact sum over
  the leaves of each leaf's first moment times the finite masses of all other
  copies (`DyadicDist.finite_mass`, summed on its own). Connect-the-dots
  keeps the moment, so the admissible difference is zero. Rounding moves it to
  one side by at most the factor `(1 + 1/m)^depth`, so the admissible difference
  is the one-sided interval of that width.

The mass tolerances are built on `transform_mass_tolerances` with the transform
length `4m` and one charge per merge: a sum that is reused is charged in both
factors of its parent, so `t` copies are charged `t - 1` merges. The raise
threshold is linear in that count and the silent threshold follows its square
root, since residuals of either sign largely cancel. A moment residual is judged
by the mass tolerance times the size of the moment.

The sum is admitted once and not after every merge: an exact sum over the dyadic
grid costs about as much as the transforms of a merge, and an admission needs
three. Measured at `m = 2^14`, admitting every merge made a build 1.4 to 1.65
times slower. A `DyadicDist` copies its masses at construction and exposes only
read-only arrays, including after a deep copy. It computes each exact sum once,
so a merge that follows a sum reads the cached totals without repeating them.

These checks admit or reject a floating-point computation. They do not certify
a bound: the certificate is the convex-order argument for connect-the-dots, or
the one-sided rounding, applied to exact arithmetic.

### `random_allocation_realization.py`

Realization-specific path that starts from explicit `PLDRealization` factors and
then reuses shared composition logic.

Key functions:

- `realization_base_distributions(...)`: the paired builder, the one public
  function of the module. It holds the flow of both directions once and selects
  the grid by `convolution_method`, which supplies the projection of each
  direction onto its loss points. Both grids return the remove loss factors
  `(base, neg_dual_base)` and the add loss factor `base`: `GEOM` on fixed-gap
  loss grids, `MS_FFT` as `SparseDiscreteDist` on the logarithms of the dyadic
  points of a range, negated for the add factor.
  - `DOMINATES`: one projection of the remove realization, then the exact dual
    of the projected base, reflected.
  - `IS_DOMINATED`: a lower-bound projection is no longer a realization, so the
    dual is derived from the realization first and the two are rounded down
    separately.
- `_loss_projection(...)`: the projection of one direction onto the loss
  points of the convolution method. On `GEOM` it is `project_discrete_dist` bound to the step of
  the direction. The step is the loss budget, never finer than the grid of the
  realization (a coarser realization warns), and coarser when that grid would
  exceed `max_grid_mult` points; the cap uses `grid_step_for_point_cap` over the
  realization's support, as the continuous path does over its range.
  On `MS_FFT` it is the projection onto the
  logarithms of the dyadic points of the direction, which derives its points per octave
  from its loss budget with `dyadic_cells_per_band`, under the cap
  `max_cells_per_band`. The projections are described under One-step factors on
  the dyadic grid below, through `project_discrete_dist_on_points`, which
  selects `project_discrete_dist_ctd_on_points` or
  `project_discrete_dist_stoch_dom_on_points` by the bound. The range is found
  by bisection on `r`, each probe an exact sum of the atoms outside the range.
  For a lower bound the rounding first moves the mass at `+inf` of the
  projected distribution to its last finite loss, as the fixed-gap rounding
  does: lowering a loss lowers the profile, so the lower bound stays valid, and
  a sum of ratios carries no factor at infinity.

### `random_allocation_continuous.py`

Continuous-input path that starts from a continuous privacy-loss law and its exact
dual, then reuses shared composition logic. The Gaussian entry points pass
`gaussian_loss_law(sigma)` as both, since that law is its own dual.

Key functions:

- `continuous_base_distributions(...)`: the paired builder, the one public
  function of the module. It holds the flow of both directions once and selects
  the grid by `convolution_method`, which supplies the projections onto its loss
  points. Both grids return the remove loss factors `(base, neg_dual_base)` and
  the add loss factor `base`: `GEOM` on fixed-gap loss grids, `MS_FFT` as
  `SparseDiscreteDist` on the logarithms of the dyadic points of a range,
  negated for the add factor.
  - `DOMINATES`: one CtD discretization of the law, then the exact dual of the
    discretized factor, reflected. The dual is not discretized a second time.
  - `IS_DOMINATED`: truncation can create mass at `-inf`, which the exact dual of
    a discretized factor cannot carry, so the two laws are rounded separately:
    the law is rounded down, and the dual is rounded up and reflected, which
    rounds the negated dual down.
  - The add factor is projected by CtD for `DOMINATES` and rounded down for
    `IS_DOMINATED`.
- `_loss_projections(...)`: the projection of one direction onto the loss
  points of the convolution method, and the rounding of a law up on the
  reflection of those points, for the dual of a lower bound. On `GEOM` the step
  of a direction covers the range its tail budget allows (the joint range of the
  law and its reflected dual for an upper bound) and is raised by
  `grid_step_for_point_cap()` when the point cap binds. For a lower bound the
  two remove factors each take half of the tail budget and share one step.
  On `MS_FFT` the points are the logarithms of dyadic
  points. A direction derives its points per octave from its loss budget
  with `dyadic_cells_per_band`, under the cap `max_cells_per_band`, and uses
  `project_continuous_dist_on_points`, which selects
  `project_continuous_dist_ctd_on_points` or
  `project_continuous_dist_stoch_dom_on_points` by the bound. The range comes from
  `joint_source_dual_bounds`.

#### One-step factors on the dyadic grid

The dyadic projections of both input kinds evaluate the discretizations of the
fixed-gap route on other knots: the logarithms of the dyadic points of the range
`[2^-r, 2^r]` (`dyadic_range_points`). A loss point `l` holds the mass of the
ratio `exp(l)`. The paired builders return loss distributions on those points; the
base build places their masses on the dyadic grid with
`dyadic_dist_on_range`.

- `DOMINATES`: one CtD discretization of the law on those loss points. The law
  under `Q` is the exact dual of the discretized factor, as on the fixed-gap
  grid, and is not discretized a second time. The discretized factor is a
  `SparseDiscreteDist` on the dyadic loss points, and its dual has mass
  `f(l) exp(-l)` at `-l`. `calc_pld_dual` takes a distribution on explicit points
  as it takes a realization, and `negate_reverse_distribution` reflects the dual
  back onto the same points. The CtD projection leaves `E[exp(-L)] <= 1`, so the rest
  of unit mass is the dual's mass at `+inf`, the `-inf` of the negated dual.
- `IS_DOMINATED`: the two remove factors are rounded separately from the exact
  input: the law is rounded down, and the dual is rounded up and reflected.
- The add factor is the add loss on the negated points: CtD for `DOMINATES`,
  and the loss rounded down, which rounds the ratio `exp(-L)` up, for
  `IS_DOMINATED`.

## Adaptive Refinement

`PLD_accounting/adaptive_random_allocation.py` computes upper/lower ranges by iteratively refining:

- `loss_discretization` (halved each step)
- `tail_truncation` (divided by 10 each step)

Entry point:

- `optimize_allocation_epsilon_range(...)`

The module tracks best upper/lower bounds across iterations and returns `AdaptiveResult`.
The first pair is evaluated before the loop; later passes refine, then evaluate.
A no-change refinement exits without a further evaluation, so it is not counted:
`AdaptiveResult.iterations` and the non-convergence warning both report pairs actually
evaluated.

`target_accuracy` is the caller's request and is reported back unchanged. A positive value
is an absolute gap; zero is rejected, since convergence needs the gap strictly below the
target. A negative value is a sentinel selecting the relative rule
`upper_bound / lower_bound <= 1 + DEFAULT_RELATIVE_ACCURACY`, which is scale-free and so needs
no estimate of the answer to be meaningful. Nothing resolves the sentinel into a number, because
a resolved target would have to be ratcheted against the running lower bound, and a field
documenting the request must not change as the search proceeds. `estimate_poisson_query` is
therefore consulted only to seed `initial_discretization` when the caller supplied none.
Consequences worth knowing: `absolute_gap < target_accuracy` does not characterize convergence
in the relative mode, and a lower bound of exactly zero cannot satisfy a ratio, so it runs to
the iteration cap and warns.

## Subsampling Integration

`PLD_accounting/subsample_pld.py` provides:

- `subsample_pld(*, pld, sampling_probability)`
- `subsample_pld_realization(*, base_pld, sampling_prob, direction)`

This module implements PLD-based subsampling amplification (Appendix C,
Algorithms 8-10: `PLDsubsam-remove`, `PLDsubsam-add`, and `subsam-core`) and
uses DOMINATES semantics. The wrapper imports each dp_accounting PMF with the
adapter defaults and does not widen those bands itself.

The REMOVE direction mixes the transformed `L` under `P` with the transformed
`-D(L)` under `Q`. When `E[exp(-L)] < 1`, `-D(L)` has mass at `-inf`, which the
transform maps to `log(1 - q)`, below every transformed finite loss. The target
grid is padded to cover that point, as the ADD direction's grid covers
`-log(1 - q)`, the image of its atom at `+inf`. Left outside, the CtD exterior
rule would move that mass up to the first knot, which is valid but loose once
the PLD is composed.

### dp_accounting import admission bands

The import path is a numerical trust boundary. `dp_accounting_pmf_to_pld_realization`
admits only pessimistic PMFs: an optimistic dp_accounting PMF does not map to either
package `BoundType`. It then performs three distinct repairs — clipping negative finite
and infinity mass to zero without an upper clip, repairing total mass directionally,
and surrendering reciprocal-moment excess to `+inf`.

Each quantity gets a **drift/repair pair**, applied through the shared
`classify_residual` policy: below the drift tolerance the repair is silent,
between the two it warns, and at or above the repair tolerance the import
raises. The keyword arguments default to the shared drift floor and
dp_accounting-specific ceilings. A caller with a known-noisier source widens a
pair explicitly at the adapter call. `subsample_pld` does not expose the bands.

The bands are input-quality gates, not an error budget. Negative-mass clipping
raises those entries to zero. Total-mass repair holds the input infinity mass
fixed: excess is removed from the low-loss edge, while a deficit is added at the
largest finite loss. Only reciprocal-moment repair explicitly moves finite mass
to exported `infinity_mass`. These changes are reflected in the returned PLD,
but they do not all raise its delta floor. Admission of numerical drift does not
certify domination of an unknown source of missing mass.

**Sizing.** The defaults are measured on a single **uncomposed** call to the (optionally
subsampled) Gaussian mechanism, which is what the adapter is for — 288 PMFs over sigma in
[0.5, 10], `value_discretization_interval` in [1e-5, 1e-2], `sampling_prob` in
{none, 1e-3, 1e-2, 0.1, 0.5, 1}, both adjacency directions, connect-the-dots, pessimistic.

| quantity | median | p90 | max | drift | repair |
|---|---|---|---|---|---|
| mass residual | 3.0e-11 | 4.1e-7 | 4.8e-6 | `PMF_MASS_DRIFT_TOL` | 1e-5 |
| moment mass moved to `+inf` | 0.0 | 1.0e-15 | 2.0e-12 | `PMF_MASS_DRIFT_TOL` | 1e-8 |
| clipped negative mass | 0.0 | 0.0 | 0.0 | `PMF_MASS_DRIFT_TOL` | 1e-12 |

Composition is deliberately **outside** the calibration. A base residual composes to
roughly `depth` times itself, so a caller handing the adapter a `self_compose(n)` PMF is
past the defaults by construction and must say so, passing bands scaled by the `n` it
just composed to. That is a fact such a caller has in hand a few lines earlier, not an
unverifiable assertion about an opaque input, and it keeps the default from silently
pre-authorizing depths nothing in the repository runs.

**The mass band is symmetric although its repairs are not.** An excess residual is
trimmed from the low-loss edge, which is nearly free at the deltas of interest; a deficit
is added at the last finite loss while `p_max` stays fixed. Its profile
contribution is therefore loss-dependent, rather than a constant delta floor.
Excess dominates —
`pessimistic_estimate` rounds probabilities up — but deficits do occur: 36 of the 288
measured PMFs carry one, all at or below 4.9e-12. The symmetric band therefore
stands; these repairs are visible warnings under the shared drift floor.

A caller-owned error ledger would be legitimate for an `IS_DOMINATED` import, where
clipping and `+inf` trimming would push a lower bound past what the source supports. No
such path exists: `PLDRealization` requires `p_min = 0`, and every caller of
`dp_accounting_pmf_to_pld_realization` uses `DOMINATES`.

## Numerical Invariants

Across the codebase:

- Boundary atoms are represented explicitly as `p_min` and `p_max`.
- `p_min` means `-inf` on `Domain.REALS` and `0` on `Domain.POSITIVES`.
- Mass conservation is enforced after discretization and convolution. A
  multiscale sum is the exception: it is rescaled to unit mass once, where it
  becomes a loss distribution.
- A total of masses or of moment terms is the correctly rounded sum of its terms:
  `exact_sum` returns the value `math.fsum` returns, and `signed_unit_residual` is built on it.
- Edge truncation folds the sum of the drained entries, not the difference of the two near-unit masses.
- Bound semantics are preserved during regridding/truncation:
  - `BoundType.DOMINATES` for upper bounds
  - `BoundType.IS_DOMINATED` for lower bounds
- Loss-space and exp-space transforms are explicit (`exp_linear_to_geometric`, `log_geometric_to_linear`).

### Moment bands are not mass bands

Mass `m` at loss `L` carries `m * exp(-L)` of reciprocal moment, so a moment residual is a
different quantity from the mass residual that produced it and takes its own band.
`REALIZATION_MOMENT_TOL` is the PLD invariant. An admitted *repair* band is derived from the
producer rather than fixed: `_ctd_moment_repair_tol(max_abs_loss=, step=)` returns
`8 * eps * max(1, max_abs_loss) / (1 - exp(-step))`, the error the cellwise endpoint split
can leave behind on a grid of that width and reach. That is the same shape as
`transform_mass_tolerances` in `distribution_utils.py`, and it is why no fixed `n * eps`
moment constant exists: the quantity being bounded scales with the grid, so a constant would
be a calibration against one workload.

The ceiling takes its scale from that moment error and never from the producer's *mass*
band. Its silent threshold does not scale at all: it stays at `REALIZATION_MOMENT_TOL`
whatever the grid, because what a moment repair moves lands at `p_max` and raises delta
directly, so a repair above that floor warns even when the grid-dependent ceiling admits it.

`trim_mass_to_moment_target` restores `E[exp(-L)] <= 1 - target` by moving mass from the
low-loss edge to `+inf`. The moment to remove is the target minus the measured residual. The
steps:

1. A running sum of the moment terms locates the pivot bin, the first bin at which the
   moment removed so far reaches the request. Every bin below it is emptied, and the
   remainder still owed is the request minus the exact sum of the emptied terms.
2. The remainder times `exp(L)` is the mass the pivot bin gives up; the bin keeps the rest.
3. The residual is measured with the bin at that mass, by the exact sum the postcondition
   uses. If it is short of the target, the shortfall is converted to mass at this bin and
   subtracted, at least one unit in the last place per correction, at most
   `_TRIM_CORRECTIONS` times.
4. If the target is still unmet, the bin is emptied and a `RuntimeWarning` reports the mass,
   the loss and the residual.

All mass the trim removes goes to `+inf`, so the bound stays valid. The pivot bin keeps less
than the largest mass that meets the target by at most a few units in the last place of its
mass, plus the last bit of the requested moment converted at the bin's loss. The overshoot
matters because composition multiplies the mass at `+inf` by the number of compositions,
and that product is a floor on delta. A request that is not smaller than the law's total
moment raises.

## Practical Extension Points

- New mechanisms can be added by producing valid `PLDRealization` inputs and
  using `general_allocation_pld(...)`.
- Discretization, tail budget, the grid with its projection, and the caps of each grid are
  set by `AllocationSchemeConfig`.
- Additional accounting workflows can compose returned `dp_accounting` PLDs directly.
