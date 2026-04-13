"""Pytest configuration and fixtures for the basic (unit) test suite."""

import os
from pathlib import Path

import numpy as np
import pytest
from PLD_accounting.discrete_dist import SparseDiscreteDist
from PLD_accounting.types import AllocationSchemeConfig, PrivacyParams

# Numba expects a writable cache directory in some environments.
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / ".numba_cache"
_DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["NUMBA_CACHE_DIR"] = str(_DEFAULT_CACHE_DIR)
os.environ["NUMBA_DISABLE_FILE_SYSTEM_CACHE"] = "1"


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
    return SparseDiscreteDist(x_array=x, prob_arr=pmf)


@pytest.fixture
def simple_gaussian_params():
    """Fixture providing typical Gaussian mechanism parameters."""
    return PrivacyParams(
        sigma=1.0,
        num_steps=10,
        num_selected=5,
        num_epochs=1,
        epsilon=None,
        delta=1e-5,
    )


@pytest.fixture
def default_allocation_config():
    """Fixture providing default allocation configuration."""
    return AllocationSchemeConfig(
        loss_discretization=0.1,
        tail_truncation=0.001,
    )
