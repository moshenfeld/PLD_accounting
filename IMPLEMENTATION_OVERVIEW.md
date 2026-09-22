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

Allocation counts are checked by `require_allocation_counts()` in
`validation.py`, called from `allocation_directional_pld()` and from
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

- `DenseDiscreteDist`: regular-grid distribution carrying an immutable
  `GridSpec` plus `prob_arr`, `p_min`, and `p_max`.
- `SparseDiscreteDist`: explicit-support distribution with `x_array`,
  `prob_arr`, `p_min`, and `p_max`.
- `PLDRealization`: linear-grid specialization for privacy-loss space.

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
  explicit target grid.

Boundary semantics depend on `Domain`:

- `Domain.REALS`:
  - `p_min` is mass at `-inf`
  - `p_max` is mass at `+inf`
- `Domain.POSITIVES`:
  - `p_min` is mass at `0`
  - `p_max` is mass at `+inf`

CtD is the fixed engine for dominating fixed-gap real-loss construction; there
is no public method selector. Lower bounds, FFT exp-space factors, and
geometric-grid regridding use separate stochastic-domination functions because
CtD is not defined for those representations. A CtD target grid must be linear
and fixed-gap. A discrete source may be dense or sparse, but must use
`Domain.REALS`, have exact `p_min == 0`, finite strictly ordered support,
conserved mass, and reciprocal moment `E[exp(-L)] <= 1` under the shared
realization tolerance. These checks certify the object as a realizable PLD,
not its domination relationship to an external mechanism.

Linear real-loss callers route through the public
`rediscretize_dist_by_bound` helper. Bound direction alone selects the
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
for continuous laws, `_discrete_ctd_cell_measures` for realizations — and the
endpoint split, exterior policy and finite-knot assembly follow from those. This
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
discretization keeps its original source-quantile range. Exterior cells are
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
`distribution_discretization.py`: it validates the algorithm-specific semantic
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

Shared composition budgets are derived inline in `allocation_directional_pld()`
and `_allocation_directional_pld_core()` in
`PLD_accounting/random_allocation_accounting.py`.

- `allocation_directional_pld()` divides the tail budget by the number of active
  tail-consuming ops, `2 * component_count - 1`: one component core call per
  floor/ceil component plus the final `fft_convolve` when both are active. After
  that division each op consumes at most the rescaled budget, so all of them
  together stay within the caller's `tail_truncation`.

  It owns only the split *between* components; the per-epoch and per-count
  divisions belong to the two layers below it.

  - One active component: `loss_discretization` is passed through unchanged.
  - Both active: the budget is split proportional to each component's effective
    discretization count — `floor_count = floor_epochs *
    base_loss_discretization_count(floor_steps)` and the ceil analogue, note the
    component's own epoch multiplicity and step count, not `num_epochs` and
    `num_steps` — giving `loss_disc_floor = loss_discretization * floor_count /
    (floor_count + ceil_count)` and likewise for ceil.

  `_allocation_directional_pld_core()` then divides its component's budget by that
  component's own `num_epochs` to get `base_loss_discretization`, skipping the
  division when `num_epochs == 1`. The base builder divides once more by
  `base_loss_discretization_count`. The multiplications above and these divisions
  cancel, so both uncapped final steps reduce to the same
  `loss_discretization / (floor_count + ceil_count)` — which is what the final
  `fft_convolve` requires. Keeping the structure explicit makes each layer's budget
  ownership visible even though the real-number algebra cancels.

  The two final steps can still differ when the component-local grid cap binds.
  The floor and ceil components have different source ranges and factor budgets,
  so `_grid_step_for_point_cap()` can independently raise their factor
  steps by different amounts. At `sigma=1`, `loss_discretization=1e-3`,
  `T=101`, `num_selected=3`, `num_epochs=5`, and `max_grid_mult=20000`, the
  REMOVE final steps become `8.277295e-4` and `8.368303e-4`; with
  `loss_discretization=0.1` the cap stays inactive and both directions retain an
  exact shared step.   `_align_component_grids()` reconciles a cap-bound mismatch directionally: a
  last-bit step difference is relabeled; a material cap-driven mismatch coarsens
  the finer component through `rediscretize_dist_by_bound` and warns. That
  post-build reconciliation is the intended behavior.
- `_allocation_directional_pld_core()` divides the component *tail* budget:
  by 3 (base creation, epoch composition, final truncation) when
  `num_epochs > 1`, or by 2 when the FFT phase is skipped (`num_epochs == 1`).
  `base_tail_truncation = tail_truncation / (3 * num_epochs)` covers up to 3
  tail-consuming sub-ops per epoch. The matching loss division is the per-epoch
  `/ num_epochs` already described above.
- GEOM base builds (`geometric_allocation_pld_base_add/remove()`) divide their
  loss budget by the exact discretizing stage count
  (`add/remove_geometric_loss_discretization_count(num_steps)`) and their tail
  budget by the phase count (2 for ADD, 3 for REMOVE), with each one-step
  factor receiving a further `1 / num_steps` to absorb self-convolution
  amplification.

`_fft_self_convolve_direct()` splits its own `tail_truncation` into four equal
quarters: the Chernoff window, the circular-alias reserve, the explicit
opposite-side trim, and the final `truncate_edges`. The alias reserve is the one
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

`BEST_OF_TWO` is not a composition and its budget does not divide. Each pure route
is built by a recursive call carrying the **full** `config.tail_truncation`, because
the two candidates are alternatives rather than factors: the result is a pointwise
min/max CCDF of two independently valid bounds, so their tail errors do not add.
`combine_best_of_two_plds()` then receives the same full budget. It spends it on at
most one CtD reprojection — only when the two candidates land on genuinely different
lattices; on a shared lattice both embed exactly and nothing is spent. That
reprojection acts on already-composed PLDs and nothing composes them again, so no
per-factor `1 / num_steps` divisor applies.

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
| `PLD_accounting/types.py` | Enums and configs (`PrivacyParams`, `AllocationSchemeConfig`, `BoundType`, etc.). `PrivacyParams` and `AllocationSchemeConfig` validate field invariants on construction. |
| `PLD_accounting/random_allocation_api.py` | Public entry points for Gaussian and realization accounting. Each documented function validates its own arguments. |
| `PLD_accounting/random_allocation_accounting.py` | Shared composition/finalization helpers used by both Gaussian and realization paths. |
| `PLD_accounting/random_allocation_gaussian.py` | Gaussian-specific factor construction and convolution method selection. |
| `PLD_accounting/random_allocation_realization.py` | Realization-specific factor construction from `PLDRealization` inputs. |
| `PLD_accounting/adaptive_random_allocation.py` | Adaptive upper/lower range refinement for epsilon/delta queries. |
| `PLD_accounting/mechanisms.py` | Mechanism PLD factory helpers (`gaussian_distribution`, `laplace_distribution`, `discrete_distribution`). |
| `PLD_accounting/validation.py` | Constraint primitives (`require_positive_real`, `require_enum`, `require_allocation_counts`, ...), named by the fact they enforce rather than by caller. Scalar helpers accept one `(value, name)` pair or parallel sequences of values and names. Repeated enum/type combinations live next to those types (`require_bound_type`, `require_direction`, `require_privacy_params`, `require_allocation_config`). Dense/linear/domain checks live on `discrete_dist.require_dense_dist`, with `require_linear_reals_dist` / `require_geometric_positives_dist` for the two usual lattices. |
| `PLD_accounting/discrete_dist.py` | `GridSpec` and distribution classes (`DenseDiscreteDist`, `SparseDiscreteDist`, `PLDRealization`, `Domain`). |
| `PLD_accounting/distribution_discretization.py` | Continuous-to-discrete conversion and spacing changes (linear/geometric). |
| `PLD_accounting/fft_convolution.py` | FFT-based convolution and self-convolution on linear grids. |
| `PLD_accounting/geometric_convolution.py` | Convolution and self-convolution on geometric grids. |
| `PLD_accounting/utils.py` | PLD transforms (`exp`, `log`, dual, negate-reverse, composition helpers). |
| `PLD_accounting/distribution_utils.py` | Mass and reciprocal-moment repair policy: the drift/repair bands, `enforce_mass_conservation`, `classify_residual`, compensated summation. |
| `PLD_accounting/dp_accounting_support.py` | Conversion between internal probability representations and `dp_accounting` PMF/PLD types. Import repairs use caller-visible drift and repair bands; the defaults are calibrated for uncomposed PMFs. |
| `PLD_accounting/subsample_pld.py` | PLD-level subsampling amplification helpers (DOMINATES-only path). |

## Public API Surface

Random allocation (defined in `PLD_accounting/random_allocation_api.py`):

- Gaussian path:
  - `gaussian_allocation_pld(...)`
  - `gaussian_allocation_directional_pld(...)`: public directional entry.
    Resolves the convolution route once for one direction and runs the full
    directional pipeline via `allocation_directional_pld(...)`:
    - FFT route uses `_gaussian_allocation_fft(...)` with compact ADD/REMOVE
      internals, paired with the constant loss-discretization count 1.
    - GEOM route uses shared add/remove geometric cores with Gaussian factor
      builders (matching realization route structure), paired with the
      geometric discretization counts.
    - BEST_OF_TWO recursively runs the directional pipeline once per pure route
      and combines same-direction results at the very end via
      `combine_best_of_two_plds(...)` (in `utils.py`).
    The Gaussian orchestrator is `gaussian_allocation_pld_core_and_count(...)`
    in `random_allocation_gaussian.py`.
  - `gaussian_allocation_epsilon_configurable(...)`
  - `gaussian_allocation_delta_configurable(...)`
  - `gaussian_allocation_epsilon_range(...)`
- Realization path:
  - `general_allocation_pld(...)`
  - `general_allocation_epsilon(...)`
  - `general_allocation_delta(...)`

Composition conversion (defined in `PLD_accounting/random_allocation_accounting.py`):

- `compose_full_pld(...)`: converts internal directional PLDs into a
  `dp_accounting` PLD object.

Fixed-gap rediscretization (defined in `PLD_accounting/distribution_discretization.py`):

- `rediscretize_dist_by_bound(*, dist, tail_truncation, loss_discretization, bound_type)`
  - Public fixed-gap real-loss entry. Bound direction selects CtD for
    `DOMINATES` and directional stochastic domination for `IS_DOMINATED`.

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

- `BoundType`, `Direction`, `SpacingType`, `ConvolutionMethod` — the enums that
  select bound direction, adjacency direction, lattice family and backend.
- `PrivacyParams`, `AllocationSchemeConfig` — frozen configuration records that
  validate their own fields on construction.
- `DEFAULT_LOSS_DISCRETIZATION`, `DEFAULT_TAIL_TRUNCATION` — the defaults
  `AllocationSchemeConfig` applies.
- `has_numba()` — whether the optional Numba acceleration is active. Results are
  unaffected; only speed and last-bit summation order differ.

Notes:

- PLD builders reject `BoundType.BOTH`; users build separate DOMINATES and IS_DOMINATED PLDs.
- Realization-based allocation requires `ConvolutionMethod.GEOM`.

## Core Composition Modules

### `random_allocation_accounting.py`

This is the shared composition core used by both Gaussian and realization accounting.

Key functions:

- `_allocation_directional_pld_core(...)`:
  Calls a base-PLD callback, trims its edges, caps the base PMF size when
  `fft_self_convolve` would exceed the FFT memory limit, composes across epochs,
  then trims the composed result. It does not regrid to the output
  discretization; the output step is whatever the base builder produced from the
  `base_loss_discretization` it was handed.
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

- `gaussian_allocation_pld_core_and_count(...)`: Gaussian orchestrator used by
  `gaussian_allocation_directional_pld(...)` in `random_allocation_api.py`.
  Selects the FFT or GEOM base builder for one direction (COMBINED currently
  chooses GEOM for ADD and FFT for REMOVE) and the matching
  loss-discretization count. BEST_OF_TWO is resolved in the public directional
  entry and does not reach this function.
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
    all nonpositive mass is folded back to the POSITIVES zero boundary. The pair
    is `_embed_positive_boundary_on_nonpositive_real_cell()` /
    `_fold_nonpositive_real_mass_to_positive_boundary()` in
    `random_allocation_gaussian.py`. See "Embedding the exp-space zero atom" below.
  - `_gaussian_add_geom_loss_factor(...)`

### Embedding the exp-space zero atom

`fft_convolve` and `fft_self_convolve` require `Domain.REALS`, and a POSITIVES
distribution cannot simply be relabelled: its lower atom is additively neutral
(`p_min = p1 * p2`), while the `-inf` atom on REALS is absorbing
(`p_min = 1 - (1-p1)(1-p2)`). Relabelling would silently convert one into the other.

The alternative is what the geometric kernel does — keep the atom as a boundary and
inject the zero-plus-finite cross terms pairwise during geometric convolution.
That works pairwise. It does not work for `_fft_self_convolve_direct()`, which raises
the DFT to the m-th power in one shot: injecting the j-zero / (m-j)-finite cross terms
afterwards would need `m` separate convolutions. Placing the atom on the lattice makes
one transform produce all of them.

The embedding cell must land at a coordinate `<= 0`. Placing it at `x_0 > 0` would
compute `x_0 + y` rather than `0 + y = y` and overstate the sum, which breaks the
flipped `IS_DOMINATED` bound this route runs at. A cancellation-prone origin is
handled by an extra `GridSpec.pad` cell so the atom occupies a representable
nonpositive knot while every original finite coordinate stays bitwise unchanged;
it is never folded into an existing positive cell.

The round trip is deliberately not an identity.
`_embed_positive_boundary_on_nonpositive_real_cell()` puts the atom in one cell
`<= 0`; `_fold_nonpositive_real_mass_to_positive_boundary()` collapses every cell
`<= 0`. So sums using `k` copies of the atom sit about `k` steps low. That is conservative for a
lower bound and bounded: mass with at least one zero copy is about `T * p_min ~ tail`,
inside the tail budget.

`ceil(x_0 / step)` evaluates to one prepended cell in every measured production
configuration (`sigma` in {0.4, 1, 2} crossed with `T` in {10^2, 10^4, 10^5}); the
lognormal lower quantile runs 4 to 12 orders of magnitude below one step. The general
form and its rounding guard are kept, but one cell is the normal case, not a hot path.

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

`target_accuracy` is the caller's request and is reported back unchanged. A nonnegative value
is an absolute gap. A negative value is a sentinel selecting the relative rule
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

The bands are input-quality gates, not an error budget. Every repair is conservative
under `DOMINATES` and lands in the exported `infinity_mass`, so the looseness it
introduces is already inside the epsilon the caller receives.

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
is routed to `p_max` and raises the delta floor directly. Excess dominates —
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
- Mass conservation is enforced after discretization and convolution.
- Bound semantics are preserved during regridding/truncation:
  - `BoundType.DOMINATES` for upper bounds
  - `BoundType.IS_DOMINATED` for lower bounds
- Loss-space and exp-space transforms are explicit (`exp_linear_to_geometric`, `log_geometric_to_linear`).

### Moment bands are not mass bands

Mass `m` at loss `L` carries `m * exp(-L)` of reciprocal moment, so a moment residual is a
different quantity from the mass residual that produced it and takes its own band.
`REALIZATION_MOMENT_TOL` is the PLD invariant. An admitted *repair* band is derived from the
producer rather than fixed: `_ctd_moment_repair_tol(max_abs_loss=, step=)` returns about
`eps * |a| / (1 - exp(-step))`, the error the cellwise endpoint split can leave behind on a
grid of that width and reach. That is the same shape as `_fft_mass_tolerances`, and it is
why no fixed `n * eps` moment constant exists: the quantity being bounded scales with the
grid, so a constant would be a calibration against one workload.

What a moment ceiling may **not** do is scale with the producer's *mass* band. What a
moment repair moves lands at `p_max`, so it is charged to delta, and delta has nothing to do
with grid size.

## Practical Extension Points

- New mechanisms can be added by producing valid `PLDRealization` inputs and
  using `general_allocation_pld(...)`.
- Gaussian method tuning is controlled by `AllocationSchemeConfig` and
  `ConvolutionMethod`.
- Additional accounting workflows can compose returned `dp_accounting` PLDs directly.
