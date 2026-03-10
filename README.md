# RA_PLD

Numerical privacy accounting for random allocation and subsampling using Privacy Loss Distributions (PLDs).

## What This Repository Provides

- End-to-end numerical accounting for random allocation privacy guarantees.
- Direction-aware accounting (`REMOVE`, `ADD`, or `BOTH`) with explicit upper/lower bound semantics.
- `FFT` convolution for linearly spaced grids.
- `GEOM` convolution for geometrically spaced positive grids.
- Subsampling amplification directly on PLD representations.
- Interop helpers for `dp_accounting` PMF objects.

## Repository Structure

- `PLD_accounting/`: core library code (types, discretization, convolution, accounting, subsampling).
- `tests/`: unit and integration tests.
- `usage_example.py`: executable usage examples for common workflows.

## Requirements

Dependency definitions live in `pyproject.toml` (single source of truth).

Install package (runtime):

```bash
pip install .
```

Install as a local package (recommended):

```bash
pip install -e .
```

Install package with test extras:

```bash
pip install -e ".[test]"
```

Install development dependencies:

```bash
pip install ".[dev]"
```

A compatibility wrapper is also provided:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from PLD_accounting import (
    PrivacyParams,
    AllocationSchemeConfig,
    Direction,
    BoundType,
    numerical_allocation_epsilon,
)

params = PrivacyParams(
    sigma=1.0,
    num_steps=1000,
    num_selected=10,
    num_epochs=1,
    delta=1e-6,
)

config = AllocationSchemeConfig(
    loss_discretization=0.02,
    tail_truncation=1e-8,
    max_grid_FFT=1_000_000,
)

eps = numerical_allocation_epsilon(
    params=params,
    config=config,
    direction=Direction.BOTH,
    bound_type=BoundType.DOMINATES,
)
print(eps)
```

## Main API

- `allocation_PLD(params, config, direction, bound_type)`: returns a `dp_accounting` `PrivacyLossDistribution`.
- `numerical_allocation_epsilon(params, config, direction, bound_type)`: computes `epsilon` for `params.delta`.
- `numerical_allocation_delta(params, config, direction, bound_type)`: computes `delta` for `params.epsilon`.
- `subsample_PLD(pld, sampling_probability, bound_type)`: applies subsampling amplification to an existing PLD.

## Important Parameter Notes

- `sigma` must be positive and finite.
- `tail_truncation` must be in `(0, 1)`.
- `loss_discretization` controls the accuracy/runtime tradeoff; smaller values are tighter but slower and larger-memory.
- `max_grid_FFT` and `max_grid_mult` cap grid sizes; too-small budgets can invalidate a run.
- `bound_type=DOMINATES` is pessimistic (upper bound), `IS_DOMINATED` is optimistic (lower bound).

## Running Tests

From repository root:

```bash
pytest -q
```

With coverage:

```bash
./tests/run_tests.sh --coverage
```

Build a wheel/sdist:

```bash
python -m build
```

## Examples

- `usage_example.py` includes direct epsilon queries.
- `usage_example.py` includes PLD construction and repeated epsilon lookups.
- `usage_example.py` includes a subsampling + composition workflow.
