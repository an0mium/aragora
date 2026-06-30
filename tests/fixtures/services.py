"""Service and database-backed fixtures (pytest plugin)."""

from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

from tests.utils import managed_fixture

if TYPE_CHECKING:
    from aragora.memory.continuum import ContinuumMemory
    from aragora.ranking.elo import EloSystem


@pytest.fixture
def stop_auth_cleanup():
    """Fixture to stop auth cleanup threads after tests.

    Use this fixture for tests that create AuthConfig instances.
    It yields a function that stops the cleanup thread, and also
    cleans up on teardown.

    Usage:
        def test_auth_config(stop_auth_cleanup):
            from aragora.server.auth import AuthConfig
            config = AuthConfig()
            # ... test code ...
            stop_auth_cleanup(config)
    """
    configs = []

    def _stop(auth_config):
        configs.append(auth_config)
        if hasattr(auth_config, "stop_cleanup_thread"):
            auth_config.stop_cleanup_thread()

    yield _stop

    # Cleanup any remaining configs
    for config in configs:
        if hasattr(config, "stop_cleanup_thread"):
            try:
                config.stop_cleanup_thread()
            except Exception:
                pass


@pytest.fixture
def mock_auth_config():
    """Create a mock AuthConfig.

    Returns an AuthConfig configured for authentication testing.
    Cleanup thread is suppressed by _suppress_auth_cleanup_threads.
    """
    from aragora.server.auth import AuthConfig

    config = AuthConfig()
    config.api_token = "test_secret_key_12345"
    config.enabled = True
    config.rate_limit_per_minute = 60
    config.ip_rate_limit_per_minute = 120
    yield config
    config.stop_cleanup_thread()


@pytest.fixture
def handler_context(mock_storage, mock_elo_system, temp_nomic_dir) -> dict:
    """Create a complete handler context.

    Returns a dict with all common handler dependencies configured.
    """
    return {
        "storage": mock_storage,
        "elo_system": mock_elo_system,
        "nomic_dir": temp_nomic_dir,
        "debate_embeddings": None,
        "critique_store": None,
    }


@pytest.fixture
def event_loop_policy():
    """Configure event loop policy for async tests.

    This fixture ensures consistent async behavior across platforms.
    """
    import asyncio

    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
def elo_system(temp_db) -> Generator["EloSystem", None, None]:
    """Create a real EloSystem with a temporary database.

    Yields an EloSystem instance backed by a temp database.
    The database connection is properly closed after the test.
    """
    from aragora.ranking.elo import EloSystem

    system = EloSystem(db_path=temp_db)
    with managed_fixture(system, name="EloSystem"):
        yield system


@pytest.fixture
def continuum_memory(temp_db) -> Generator["ContinuumMemory", None, None]:
    """Create a real ContinuumMemory with a temporary database.

    Yields a ContinuumMemory instance backed by a temp database.
    The database connection is properly closed after the test.
    """
    from aragora.memory.continuum import ContinuumMemory

    memory = ContinuumMemory(db_path=temp_db)
    with managed_fixture(memory, name="ContinuumMemory"):
        yield memory
