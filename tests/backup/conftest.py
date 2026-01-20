"""
Pytest configuration for backup module tests.

Provides fixtures and cleanup hooks to ensure test isolation.
"""

import pytest

from aragora.resilience import reset_all_circuit_breakers
from aragora.storage.schema import DatabaseManager


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Ensure clean global state before and after each test.

    This prevents cross-test pollution from:
    - Circuit breaker states
    - Database connection pools
    - Cached singletons
    """
    # Pre-test cleanup
    reset_all_circuit_breakers()
    DatabaseManager.clear_instances()

    yield

    # Post-test cleanup
    reset_all_circuit_breakers()
    DatabaseManager.clear_instances()


@pytest.fixture(autouse=True)
def isolate_backup_tests(tmp_path):
    """Ensure each test uses isolated temporary directories.

    This is especially important for backup tests that create
    files on disk.
    """
    import os

    original_cwd = os.getcwd()

    yield tmp_path

    # Restore original working directory
    os.chdir(original_cwd)
