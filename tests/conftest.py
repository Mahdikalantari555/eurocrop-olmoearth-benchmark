"""
Shared test fixtures.
"""

import pytest
import numpy as np
import tempfile
import os


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    N, T, C = 100, 12, 13
    X = np.random.rand(N, T, C).astype(np.float32) * 1000
    y = np.random.randint(0, 5, size=N)
    return X, y


@pytest.fixture
def small_data():
    """Small dataset for quick tests."""
    np.random.seed(42)
    N, T, C = 20, 6, 13
    X = np.random.rand(N, T, C).astype(np.float32) * 1000
    y = np.array([0] * 5 + [1] * 5 + [2] * 5 + [3] * 3 + [4] * 2)
    return X, y


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary."""
    return {
        "overall_accuracy": 0.85,
        "macro_f1": 0.78,
        "weighted_f1": 0.84,
        "kappa": 0.81,
    }
