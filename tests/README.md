# Tests

## Basic suite (public / default `pytest`)

The default collection is all of `tests/` (`public/pytest.ini` sets
`testpaths = tests`), including `tests/unit/`, `tests/test_public_api.py`, and
`tests/test_tolerances.py`. That folder is mirrored to the public GitHub
repository root.

From the **private monorepo** root (adds `public/` to `PYTHONPATH`):

```bash
pytest public/tests
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

These tiers are nested, and each one already begins by running the full public suite from `public/tests/`. If you just ran one of the scripted tiers, rerunning root `pytest public/tests` is redundant unless you specifically want the public suite output on its own.

Or run explicit paths, for example:

```bash
pytest tests_extended/integration/test_pld_realizations.py -q
```

## Markers

Markers are defined in `public/pytest.ini` (on GitHub that file lives at the repository root). The private monorepo’s root `pytest.ini` repeats them for `public/tests` and `tests_extended` discovery. Keep expensive tests marked `slow` / `nightly` where appropriate.
