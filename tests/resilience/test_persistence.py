"""
Tests for circuit breaker SQLite persistence.

Tests cover:
- init_circuit_breaker_persistence creates database
- persist_circuit_breaker stores state
- persist_all_circuit_breakers stores all registered
- load_circuit_breakers restores state
- cleanup_stale_persisted removes old entries
- Error handling for database failures
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from aragora.resilience.circuit_breaker import CircuitBreaker
from aragora.resilience.persistence import (
    _DB_PATH,
    cleanup_stale_persisted,
    init_circuit_breaker_persistence,
    load_circuit_breakers,
    persist_all_circuit_breakers,
    persist_circuit_breaker,
)
from aragora.resilience.registry import (
    _circuit_breakers,
    _circuit_breakers_lock,
    get_circuit_breaker,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Clean registry and reset DB path before/after each test."""
    import aragora.resilience.persistence as persistence_mod

    with _circuit_breakers_lock:
        _circuit_breakers.clear()

    old_db_path = persistence_mod._DB_PATH
    yield
    persistence_mod._DB_PATH = old_db_path
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database path and initialize persistence."""
    db_path = str(tmp_path / "test_cb.db")
    init_circuit_breaker_persistence(db_path)
    return db_path


# ============================================================================
# init_circuit_breaker_persistence Tests
# ============================================================================


class TestInitPersistence:
    """Tests for init_circuit_breaker_persistence."""

    def test_creates_database_file(self, tmp_path):
        """Test database file is created."""
        db_path = str(tmp_path / "circuit_breaker.db")
        init_circuit_breaker_persistence(db_path)
        assert Path(db_path).exists()

    def test_creates_parent_directory(self, tmp_path):
        """Test parent directories are created if missing."""
        db_path = str(tmp_path / "nested" / "dir" / "cb.db")
        init_circuit_breaker_persistence(db_path)
        assert Path(db_path).exists()

    def test_creates_table(self, tmp_path):
        """Test circuit_breakers table is created."""
        db_path = str(tmp_path / "test.db")
        init_circuit_breaker_persistence(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='circuit_breakers'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_idempotent(self, tmp_path):
        """Test calling init multiple times is safe."""
        db_path = str(tmp_path / "test.db")
        init_circuit_breaker_persistence(db_path)
        init_circuit_breaker_persistence(db_path)  # Should not raise


# ============================================================================
# persist_circuit_breaker Tests
# ============================================================================


class TestPersistCircuitBreaker:
    """Tests for persist_circuit_breaker."""

    def test_persist_basic(self, tmp_db):
        """Test basic circuit breaker persistence."""
        cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)
        persist_circuit_breaker("test-service", cb)

        # Verify in database
        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("SELECT name, failure_threshold FROM circuit_breakers")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test-service"
        assert row[1] == 5

    def test_persist_with_failures(self, tmp_db):
        """Test persisting circuit breaker with failure state."""
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        persist_circuit_breaker("failing-svc", cb)

        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute(
            "SELECT state_json FROM circuit_breakers WHERE name=?", ("failing-svc",)
        )
        row = cursor.fetchone()
        conn.close()

        state = json.loads(row[0])
        assert state["single_mode"]["failures"] == 2

    def test_persist_overwrites_existing(self, tmp_db):
        """Test persisting overwrites existing entry."""
        cb = CircuitBreaker(failure_threshold=3)
        persist_circuit_breaker("svc", cb)

        cb.record_failure()
        persist_circuit_breaker("svc", cb)

        conn = sqlite3.connect(tmp_db)
        cursor = conn.execute("SELECT state_json FROM circuit_breakers WHERE name=?", ("svc",))
        row = cursor.fetchone()
        conn.close()

        state = json.loads(row[0])
        assert state["single_mode"]["failures"] == 1

    def test_persist_no_op_when_not_initialized(self):
        """Test persist is a no-op when persistence not initialized."""
        import aragora.resilience.persistence as mod

        mod._DB_PATH = None
        cb = CircuitBreaker()
        # Should not raise
        persist_circuit_breaker("svc", cb)


# ============================================================================
# persist_all_circuit_breakers Tests
# ============================================================================


class TestPersistAll:
    """Tests for persist_all_circuit_breakers."""

    def test_persists_all_registered(self, tmp_db):
        """Test persists all circuit breakers from registry."""
        get_circuit_breaker("svc-1")
        get_circuit_breaker("svc-2")
        get_circuit_breaker("svc-3")

        count = persist_all_circuit_breakers()
        assert count == 3

    def test_returns_zero_when_not_initialized(self):
        """Test returns 0 when persistence not initialized."""
        import aragora.resilience.persistence as mod

        mod._DB_PATH = None
        count = persist_all_circuit_breakers()
        assert count == 0


# ============================================================================
# load_circuit_breakers Tests
# ============================================================================


class TestLoadCircuitBreakers:
    """Tests for load_circuit_breakers."""

    def test_load_persisted_state(self, tmp_db):
        """Test loading circuit breakers from database."""
        # Persist some state
        cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)
        cb.record_failure()
        persist_circuit_breaker("loaded-svc", cb)

        # Clear registry
        with _circuit_breakers_lock:
            _circuit_breakers.clear()

        # Load
        count = load_circuit_breakers()
        assert count == 1
        assert "loaded-svc" in _circuit_breakers

    def test_load_empty_database(self, tmp_db):
        """Test loading from empty database."""
        count = load_circuit_breakers()
        assert count == 0

    def test_load_returns_zero_when_not_initialized(self):
        """Test returns 0 when persistence not initialized."""
        import aragora.resilience.persistence as mod

        mod._DB_PATH = None
        count = load_circuit_breakers()
        assert count == 0


# ============================================================================
# cleanup_stale_persisted Tests
# ============================================================================


class TestCleanupStale:
    """Tests for cleanup_stale_persisted."""

    def test_removes_old_entries(self, tmp_db):
        """Test cleanup removes entries older than max_age_hours."""
        cb = CircuitBreaker()
        persist_circuit_breaker("old-svc", cb)

        # Manually backdate the entry
        conn = sqlite3.connect(tmp_db)
        conn.execute("UPDATE circuit_breakers SET updated_at = '2020-01-01T00:00:00'")
        conn.commit()
        conn.close()

        deleted = cleanup_stale_persisted(max_age_hours=1.0)
        assert deleted == 1

    def test_keeps_recent_entries(self, tmp_db):
        """Test cleanup keeps recent entries."""
        cb = CircuitBreaker()
        persist_circuit_breaker("recent-svc", cb)

        deleted = cleanup_stale_persisted(max_age_hours=72.0)
        assert deleted == 0

    def test_returns_zero_when_not_initialized(self):
        """Test returns 0 when persistence not initialized."""
        import aragora.resilience.persistence as mod

        mod._DB_PATH = None
        deleted = cleanup_stale_persisted()
        assert deleted == 0
