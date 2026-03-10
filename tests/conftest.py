"""
Pytest configuration and fixtures for test suite.
"""
import os
from pathlib import Path

import pytest
import numpy as np

from PLD_accounting.types import PrivacyParams, AllocationSchemeConfig
from PLD_accounting.discrete_dist import GeneralDiscreteDist

# Numba expects a writable cache directory in some environments.
# Point numba to a workspace-local cache to avoid filesystem permission errors during tests.
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".numba_cache"
_DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Force cache settings for deterministic behavior across environments.
os.environ["NUMBA_CACHE_DIR"] = str(_DEFAULT_CACHE_DIR)
os.environ["NUMBA_DISABLE_FILE_SYSTEM_CACHE"] = "1"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "system: marks tests as system tests"
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def simple_uniform_dist():
    """Fixture providing simple uniform distribution for testing."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    pmf = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    return GeneralDiscreteDist(x_array=x, PMF_array=pmf)


@pytest.fixture
def simple_gaussian_params():
    """Fixture providing typical Gaussian mechanism parameters."""
    return PrivacyParams(
        sigma=1.0,
        num_steps=10,
        num_selected=5,
        num_epochs=1,
        epsilon=None,
        delta=1e-5
    )


@pytest.fixture
def default_allocation_config():
    """Fixture providing default allocation configuration."""
    return AllocationSchemeConfig(
        loss_discretization=0.1,
        tail_truncation=0.001
    )
