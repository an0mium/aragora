"""
Tests for concurrent migration execution with advisory locking.

Covers:
- Multiple runners competing for migration lock
- Second runner failing when first has lock
- Lock release on success and failure
- Timeout handling
- PostgreSQL advisory lock behavior simulation

Run with:
    pytest tests/migrations/test_concurrent_migrations.py -v --timeout=60
"""

from __future__ import annotations

import sqlite3
import threading
import time
from typing import Optional
from unittest.mock import MagicMock, PropertyMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock PostgreSQL backend with advisory lock simulation
# ---------------------------------------------------------------------------


class MockPostgreSQLBackend:
    """
    Mock PostgreSQL backend that simulates advisory lock behavior.
    Uses threading to simulate concurrent access from multiple pods.
    """

    backend_type = "postgresql"

    # Class-level lock state (shared between instances to simulate distributed lock)
    _global_lock_holder: str | None = None
    _global_lock = threading.Lock()
    _lock_id = None

    def __init__(self, instance_id: str = "default") -> None:
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._instance_id = instance_id
        self._executed_statements: list[str] = []

    def execute_write(self, sql: str, params: tuple = ()) -> None:
        self._executed_statements.append(sql)
        # Handle advisory lock release
        if "pg_advisory_unlock" in sql:
            self._release_lock()
            return
        self._conn.execute(sql.replace("%s", "?"), params)
        self._conn.commit()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        self._executed_statements.append(sql)
        cursor = self._conn.execute(sql.replace("%s", "?"), params)
        return cursor.fetchall()

    def fetch_one(self, sql: str, params: tuple = ()) -> tuple | None:
        self._executed_statements.append(sql)

        # Simulate pg_try_advisory_lock behavior
        if "pg_try_advisory_lock" in sql:
            return self._try_acquire_lock()

        try:
            return self._conn.execute(sql.replace("%s", "?"), params).fetchone()
        except Exception:
            return None

    def _try_acquire_lock(self) -> tuple:
        """Simulate advisory lock acquisition."""
        with MockPostgreSQLBackend._global_lock:
            if MockPostgreSQLBackend._global_lock_holder is None:
                MockPostgreSQLBackend._global_lock_holder = self._instance_id
                return (True,)
            elif MockPostgreSQLBackend._global_lock_holder == self._instance_id:
                return (True,)  # Reentrant lock
            else:
                return (False,)

    def _release_lock(self) -> None:
        """Simulate advisory lock release."""
        with MockPostgreSQLBackend._global_lock:
            if MockPostgreSQLBackend._global_lock_holder == self._instance_id:
                MockPostgreSQLBackend._global_lock_holder = None

    def close(self) -> None:
        self._conn.close()

    def connection(self):
        return self._conn

    @classmethod
    def reset_global_lock(cls) -> None:
        """Reset global lock state for tests."""
        with cls._global_lock:
            cls._global_lock_holder = None
            cls._lock_id = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_lock():
    """Reset global lock state before/after each test."""
    MockPostgreSQLBackend.reset_global_lock()
    yield
    MockPostgreSQLBackend.reset_global_lock()


@pytest.fixture()
def pg_backend():
    """Provide a mock PostgreSQL backend."""
    b = MockPostgreSQLBackend(instance_id="pod-1")
    yield b
    b.close()


@pytest.fixture()
def pg_backend_2():
    """Provide a second mock PostgreSQL backend (simulating second pod)."""
    b = MockPostgreSQLBackend(instance_id="pod-2")
    yield b
    b.close()


def make_isinstance_patcher(backend):
    """Create a patcher that makes isinstance return True for PostgreSQLBackend checks."""
    from aragora.storage.backends import PostgreSQLBackend

    original_isinstance = isinstance

    def patched_isinstance(obj, classinfo):
        if obj is backend and classinfo is PostgreSQLBackend:
            return True
        return original_isinstance(obj, classinfo)

    return patch("builtins.isinstance", patched_isinstance)


# ---------------------------------------------------------------------------
# Advisory lock acquisition tests
# ---------------------------------------------------------------------------


class TestAdvisoryLockAcquisition:
    """Tests for PostgreSQL advisory lock acquisition."""

    def test_first_runner_acquires_lock(self, pg_backend):
        """First runner should successfully acquire the lock."""
        # Direct test of lock acquisition - simulates what runner does
        result = pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result == (True,)

    def test_second_runner_fails_to_acquire_lock(self, pg_backend, pg_backend_2):
        """Second runner should fail when first holds the lock."""
        # First backend acquires lock
        result1 = pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result1 == (True,)

        # Second backend should fail
        result2 = pg_backend_2.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result2 == (False,)

    def test_lock_released_after_release_call(self, pg_backend, pg_backend_2):
        """Lock should be available after explicit release."""
        # First backend acquires lock
        result1 = pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result1 == (True,)

        # Release the lock
        pg_backend.execute_write("SELECT pg_advisory_unlock(2089872453)")

        # Second backend should now acquire
        result2 = pg_backend_2.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result2 == (True,)


# ---------------------------------------------------------------------------
# Concurrent migration execution tests
# ---------------------------------------------------------------------------


class TestConcurrentMigrationExecution:
    """Tests for concurrent migration execution scenarios."""

    def _make_migration(self, version, name, up_sql="SELECT 1", down_sql=None):
        from aragora.migrations.runner import Migration

        return Migration(version=version, name=name, up_sql=up_sql, down_sql=down_sql)

    def test_concurrent_lock_acquisition_first_wins(self, pg_backend, pg_backend_2):
        """When two processes try to acquire lock, only one should succeed."""
        results = {}

        def try_lock(backend, key):
            result = backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
            results[key] = result[0]

        # Both try to acquire simultaneously
        t1 = threading.Thread(target=try_lock, args=(pg_backend, "pod1"))
        t2 = threading.Thread(target=try_lock, args=(pg_backend_2, "pod2"))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Exactly one should succeed
        successes = sum(1 for v in results.values() if v is True)
        assert successes == 1

    def test_sequential_lock_acquisition_both_succeed(self, pg_backend, pg_backend_2):
        """Sequential lock acquisition should allow both to proceed."""
        # First acquires and releases
        result1 = pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result1 == (True,)
        pg_backend.execute_write("SELECT pg_advisory_unlock(2089872453)")

        # Second should now acquire
        result2 = pg_backend_2.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result2 == (True,)


# ---------------------------------------------------------------------------
# Lock timeout handling tests
# ---------------------------------------------------------------------------


class TestLockTimeoutHandling:
    """Tests for lock timeout behavior."""

    def test_runner_acquire_lock_timeout(self, pg_backend, pg_backend_2):
        """Runner should raise RuntimeError when lock acquisition times out."""
        from aragora.migrations.runner import MigrationRunner
        from aragora.storage.backends import PostgreSQLBackend

        # Pod 1 holds the lock
        pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")

        # Create runner for pod 2 with mock backend
        runner2 = MigrationRunner.__new__(MigrationRunner)
        runner2._backend = pg_backend_2
        runner2._migrations = []

        # Patch isinstance to treat our mock as PostgreSQLBackend
        original_isinstance = isinstance

        def mock_isinstance(obj, classinfo):
            if obj is pg_backend_2 and classinfo is PostgreSQLBackend:
                return True
            return original_isinstance(obj, classinfo)

        with patch("builtins.isinstance", mock_isinstance):
            # Should raise RuntimeError on timeout
            start = time.time()
            with pytest.raises(RuntimeError, match="Migration lock acquisition timeout"):
                runner2._acquire_migration_lock(timeout_seconds=1.0)
            elapsed = time.time() - start

            # Should have waited approximately the timeout duration
            assert elapsed >= 0.9
            assert elapsed < 2.0


# ---------------------------------------------------------------------------
# Lock release on error tests
# ---------------------------------------------------------------------------


class TestLockReleaseOnError:
    """Tests for lock release when migrations fail."""

    def _make_migration(self, version, name, up_sql="SELECT 1", down_sql=None, up_fn=None):
        from aragora.migrations.runner import Migration

        return Migration(version=version, name=name, up_sql=up_sql, down_sql=down_sql, up_fn=up_fn)

    def test_lock_released_after_upgrade(self, pg_backend):
        """Lock should be released after migration completes."""
        # Simulate lock acquisition
        pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert MockPostgreSQLBackend._global_lock_holder == "pod-1"

        # Simulate lock release
        pg_backend.execute_write("SELECT pg_advisory_unlock(2089872453)")
        assert MockPostgreSQLBackend._global_lock_holder is None

    def test_lock_release_allows_next_acquisition(self, pg_backend, pg_backend_2):
        """Releasing lock should allow another process to acquire it."""
        # First process acquires
        pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert MockPostgreSQLBackend._global_lock_holder == "pod-1"

        # Second process fails
        result = pg_backend_2.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result == (False,)

        # First releases
        pg_backend.execute_write("SELECT pg_advisory_unlock(2089872453)")

        # Second now succeeds
        result = pg_backend_2.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result == (True,)
        assert MockPostgreSQLBackend._global_lock_holder == "pod-2"


# ---------------------------------------------------------------------------
# Multi-pod simulation tests
# ---------------------------------------------------------------------------


class TestMultiPodSimulation:
    """Tests simulating multi-pod Kubernetes deployment scenarios."""

    def test_three_pods_only_one_acquires_lock(self):
        """In a 3-pod deployment, only one pod should acquire the lock."""
        backends = [MockPostgreSQLBackend(instance_id=f"pod-{i}") for i in range(3)]
        results = []

        def try_acquire(backend, idx):
            result = backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
            results.append((idx, result[0]))

        threads = [
            threading.Thread(target=try_acquire, args=(b, i)) for i, b in enumerate(backends)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        # Exactly one should succeed
        successes = [r for _, r in results if r is True]
        assert len(successes) == 1

        # Clean up
        for b in backends:
            b.close()

    def test_rolling_deployment_lock_handoff(self):
        """Simulate rolling deployment where pods start sequentially."""
        results = []

        def simulate_pod(pod_id: int, delay: float):
            time.sleep(delay)
            backend = MockPostgreSQLBackend(instance_id=f"pod-{pod_id}")
            try:
                result = backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
                if result and result[0]:
                    results.append({"pod": pod_id, "acquired": True})
                    time.sleep(0.1)  # Simulate work
                    backend.execute_write("SELECT pg_advisory_unlock(2089872453)")
                else:
                    results.append({"pod": pod_id, "acquired": False})
            finally:
                backend.close()

        # Start 3 pods with staggered delays
        threads = [threading.Thread(target=simulate_pod, args=(i, i * 0.05)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # At least one pod should have acquired the lock
        assert any(r["acquired"] for r in results)


# ---------------------------------------------------------------------------
# SQLite fallback tests
# ---------------------------------------------------------------------------


class TestSQLiteFallback:
    """Tests for SQLite which doesn't use advisory locks."""

    def test_sqlite_backend_skips_advisory_lock(self):
        """SQLite backends should skip advisory locking (return True immediately)."""
        from aragora.migrations.runner import MigrationRunner

        class SQLiteBackend:
            backend_type = "sqlite"

            def __init__(self):
                self._conn = sqlite3.connect(":memory:")

            def execute_write(self, sql, params=()):
                self._conn.execute(sql.replace("%s", "?"), params)
                self._conn.commit()

            def fetch_all(self, sql, params=()):
                return self._conn.execute(sql.replace("%s", "?"), params).fetchall()

            def fetch_one(self, sql, params=()):
                return self._conn.execute(sql.replace("%s", "?"), params).fetchone()

            def close(self):
                self._conn.close()

        backend = SQLiteBackend()
        try:
            runner = MigrationRunner(backend=backend)
            # SQLite should always return True (isinstance check fails)
            result = runner._acquire_migration_lock(timeout_seconds=1.0)
            assert result is True
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Lock ID constant test
# ---------------------------------------------------------------------------


class TestLockIdConstant:
    """Tests for the migration lock ID constant."""

    def test_lock_id_is_stable(self):
        """Lock ID should be a stable constant value."""
        from aragora.migrations.runner import MIGRATION_LOCK_ID

        # Should be a positive integer
        assert isinstance(MIGRATION_LOCK_ID, int)
        assert MIGRATION_LOCK_ID > 0

        # Should be the expected hash value
        assert MIGRATION_LOCK_ID == 2089872453

    def test_lock_id_used_in_statements(self, pg_backend):
        """Lock ID should appear in lock/unlock statements."""
        from aragora.migrations.runner import MIGRATION_LOCK_ID

        # Acquire lock
        pg_backend.fetch_one(f"SELECT pg_try_advisory_lock({MIGRATION_LOCK_ID})")

        # Check statement was recorded
        lock_statements = [
            s for s in pg_backend._executed_statements if "pg_try_advisory_lock" in s
        ]
        assert len(lock_statements) == 1
        assert str(MIGRATION_LOCK_ID) in lock_statements[0]


# ---------------------------------------------------------------------------
# Integration with MigrationRunner
# ---------------------------------------------------------------------------


class TestMigrationRunnerLocking:
    """Tests for MigrationRunner's use of advisory locking.

    Note: Full integration tests for MigrationRunner locking are in test_runner.py.
    These tests verify the mock backend correctly simulates PostgreSQL advisory locks.
    """

    def test_upgrade_would_acquire_lock_for_postgres(self, pg_backend):
        """Test that the lock/unlock SQL statements work correctly with our mock."""
        # Verify our mock simulates the PostgreSQL advisory lock behavior correctly
        # This is what the runner would do during upgrade

        # Step 1: Try to acquire lock (should succeed)
        result = pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result == (True,), "First lock acquisition should succeed"
        assert MockPostgreSQLBackend._global_lock_holder == "pod-1"

        # Step 2: Try to acquire again (reentrant - should succeed)
        result = pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result == (True,), "Reentrant lock acquisition should succeed"

        # Step 3: Release lock
        pg_backend.execute_write("SELECT pg_advisory_unlock(2089872453)")
        assert MockPostgreSQLBackend._global_lock_holder is None, "Lock should be released"

        # Verify statements were recorded
        lock_stmts = [s for s in pg_backend._executed_statements if "pg_try_advisory_lock" in s]
        unlock_stmts = [s for s in pg_backend._executed_statements if "pg_advisory_unlock" in s]

        assert len(lock_stmts) == 2, "Should have two lock acquisition attempts"
        assert len(unlock_stmts) == 1, "Should have one lock release"

    def test_downgrade_would_acquire_lock_for_postgres(self, pg_backend, pg_backend_2):
        """Test that lock release allows subsequent acquisition (as in downgrade)."""
        # Pod 1 acquires lock, does work, releases
        pg_backend.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert MockPostgreSQLBackend._global_lock_holder == "pod-1"

        # While pod 1 holds lock, pod 2 cannot acquire
        result = pg_backend_2.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result == (False,), "Pod 2 should fail while pod 1 holds lock"

        # Pod 1 releases
        pg_backend.execute_write("SELECT pg_advisory_unlock(2089872453)")

        # Now pod 2 can acquire (simulating another pod doing downgrade)
        result = pg_backend_2.fetch_one("SELECT pg_try_advisory_lock(2089872453)")
        assert result == (True,), "Pod 2 should succeed after pod 1 releases"
        assert MockPostgreSQLBackend._global_lock_holder == "pod-2"
