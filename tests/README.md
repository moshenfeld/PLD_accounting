# Tests

The suite is all of `tests/`: the unit tests under `tests/unit/`,
`tests/test_public_api.py`, `tests/test_package_exports.py` and
`tests/test_tolerances.py`. `pytest.ini` sets `testpaths = tests`, so from the
repository root:

```bash
pip install -e ".[test]"
pytest
```

Run it with and without `numba` installed: the NumPy fallbacks must give the
same results. Shared fixtures live in `conftest.py`.

## Markers

Markers are defined in `pytest.ini`, and `--strict-markers` rejects any other.
Mark expensive tests `slow` so that `pytest -m "not slow"` gives a quick run.
