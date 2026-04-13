# Tests

## Basic suite (public / default `pytest`)

The **basic** suite is **unit tests only** under `tests/unit/` inside the **`public/`** tree. That folder is mirrored to the public GitHub repository root.

From the **private monorepo** root (adds `public/` to `PYTHONPATH`):

```bash
pytest
```

From **`public/`** alone (same layout as after `git clone` of the public repo):

```bash
cd public && pytest
```

Fixtures live in `public/conftest.py`. The private repo also has a root `conftest.py` shim so the same fixtures apply when collecting `tests_extended/`.

## Extended suite (private repository only)

Integration, golden, matrix, differential, and property tests live under `tests_extended/` at the **monorepo** root. They are not under `public/` and are not synced.

```bash
bash tests_extended/run_suites.sh short
bash tests_extended/run_suites.sh medium
bash tests_extended/run_suites.sh long
```

These tiers are nested, and each one already begins by running the public/basic unit suite from `public/tests/unit/`. If you just ran one of the scripted tiers, rerunning root `pytest` is redundant unless you specifically want the public/basic suite output on its own.

Or run explicit paths, for example:

```bash
pytest tests_extended/integration/test_pld_realizations.py -q
```

## Legacy implementation tests

Tests for the frozen reference implementation under `legacy/PLD_accounting/` are in `legacy/tests/`. Use an explicit path and set `PYTHONPATH` so `PLD_accounting` resolves to `legacy/PLD_accounting`, the in-tree `tests` package wins over any site-packages `tests`, and (for a few integration tests) `utils` / `comparisons` resolve via `experiments`:

```bash
PYTHONPATH=legacy/tests:legacy:experiments \
  pytest legacy/tests -q --import-mode=importlib --confcutdir=legacy/tests
```

Install runtime and test dependencies with `pip install -r experiments/requirements.txt` (not a separate `legacy/requirements.txt`).

## Markers

Markers are defined in `public/pytest.ini` (on GitHub that file lives at the repository root). The private monorepo’s root `pytest.ini` repeats them for `public/tests` and `tests_extended` discovery. Keep expensive tests marked `slow` / `nightly` where appropriate.
