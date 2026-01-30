"""
Tests for SQLite database resilience module.

Tests cover:
- Transient error detection
- ResilientConnection with retry logic
- with_retry decorator
- atomic_transaction context manager
- ConnectionPool with health checks
"""

from __future__ import annotations

import pytest
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.storage.resilience import (
    TRANSIENT_ERRORS,
    is_transient_error,
    ResilientConnection,
    with_retry,
    atomic_transaction,
    ConnectionPool,
)


# =============================================================================
# Constants Tests
# =============================================================================


class TestTransientErrors:
    """Tests for TRANSIENT_ERRORS constant."""

    def test_contains_database_locked(self):
        """Should include 'database is locked'."""
        assert "database is locked" in TRANSIENT_ERRORS

    def test_contains_database_busy(self):
        """Should include 'database is busy'."""
        assert "database is busy" in TRANSIENT_ERRORS

    def test_contains_unable_to_open(self):
        """Should include 'unable to open database file'."""
        assert "unable to open database file" in TRANSIENT_ERRORS

    def test_contains_disk_io_error(self):
        """Should include 'disk i/o error'."""
        assert "disk i/o error" in TRANSIENT_ERRORS


# =============================================================================
# is_transient_error Tests
# =============================================================================


class TestIsTransientError:
    """Tests for is_transient_error function."""

    def test_detects_database_locked(self):
        """Should detect 'database is locked' as transient."""
        error = sqlite3.OperationalError("database is locked")
        assert is_transient_error(error) is True

    def test_detects_database_busy(self):
        """Should detect 'database is busy' as transient."""
        error = sqlite3.OperationalError("database is busy")
        assert is_transient_error(error) is True

    def test_detects_disk_io_error(self):
        """Should detect disk I/O errors as transient."""
        error = sqlite3.OperationalError("disk i/o error")
        assert is_transient_error(error) is True

    def test_case_insensitive(self):
        """Should match regardless of case."""
        error = sqlite3.OperationalError("DATABASE IS LOCKED")
        assert is_transient_error(error) is True

    def test_partial_match(self):
        """Should match if error message contains transient pattern."""
        error = sqlite3.OperationalError("Error: database is locked (errno=5)")
        assert is_transient_error(error) is True

    def test_non_transient_error(self):
        """Should return False for non-transient errors."""
        error = sqlite3.IntegrityError("UNIQUE constraint failed: users.email")
        assert is_transient_error(error) is False

    def test_syntax_error_not_transient(self):
        """Should return False for syntax errors."""
        error = sqlite3.OperationalError("near 'SELEKT': syntax error")
        assert is_transient_error(error) is False


# =============================================================================
# ResilientConnection Tests
# =============================================================================


class TestResilientConnectionInit:
    """Tests for ResilientConnection initialization."""

    def test_default_values(self, tmp_path: Path):
        """Should initialize with default values."""
        db_path = str(tmp_path / "test.db")
        conn = ResilientConnection(db_path)

        assert conn.db_path == db_path
        assert conn.max_retries == 3
        assert conn.base_delay == 0.1
        assert conn.max_delay == 2.0

    def test_custom_values(self, tmp_path: Path):
        """Should accept custom configuration."""
        db_path = str(tmp_path / "test.db")
        conn = ResilientConnection(
            db_path,
            max_retries=5,
            base_delay=0.5,
            max_delay=10.0,
            timeout=30.0,
        )

        assert conn.max_retries == 5
        assert conn.base_delay == 0.5
        assert conn.max_delay == 10.0
        assert conn.timeout == 30.0


class TestResilientConnectionDelayCalculation:
    """Tests for exponential backoff delay calculation."""

    def test_calculate_delay_attempt_0(self, tmp_path: Path):
        """First attempt should use base delay."""
        conn = ResilientConnection(str(tmp_path / "test.db"), base_delay=0.1)
        assert conn._calculate_delay(0) == 0.1

    def test_calculate_delay_exponential(self, tmp_path: Path):
        """Delay should increase exponentially."""
        conn = ResilientConnection(str(tmp_path / "test.db"), base_delay=0.1)
        assert conn._calculate_delay(0) == 0.1
        assert conn._calculate_delay(1) == 0.2
        assert conn._calculate_delay(2) == 0.4
        assert conn._calculate_delay(3) == 0.8

    def test_calculate_delay_capped_at_max(self, tmp_path: Path):
        """Delay should be capped at max_delay."""
        conn = ResilientConnection(
            str(tmp_path / "test.db"), base_delay=0.1, max_delay=0.5
        )
        assert conn._calculate_delay(10) == 0.5  # Would be 102.4 without cap


class TestResilientConnectionTransaction:
    """Tests for transaction context manager."""

    def test_successful_transaction(self, tmp_path: Path):
        """Should execute and commit on success."""
        db_path = str(tmp_path / "test.db")
        conn = ResilientConnection(db_path)

        with conn.transaction() as cursor:
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            cursor.execute("INSERT INTO test (value) VALUES (?)", ("hello",))

        # Verify commit happened
        with conn.transaction() as cursor:
            cursor.execute("SELECT value FROM test WHERE id = 1")
            row = cursor.fetchone()
            assert row["value"] == "hello"

    def test_rollback_on_error(self, tmp_path: Path):
        """Should rollback on non-transient error."""
        db_path = str(tmp_path / "test.db")
        conn = ResilientConnection(db_path)

        with conn.transaction() as cursor:
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")

        with pytest.raises(sqlite3.IntegrityError):
            with conn.transaction() as cursor:
                cursor.execute("INSERT INTO test (id, value) VALUES (1, 'first')")
                cursor.execute(
                    "INSERT INTO test (id, value) VALUES (1, 'duplicate')"
                )  # Will fail

    def test_retries_on_transient_error(self, tmp_path: Path):
        """Should retry on transient errors."""
        db_path = str(tmp_path / "test.db")
        conn = ResilientConnection(db_path, max_retries=2, base_delay=0.01)

        # Create the table first
        with conn.transaction() as cursor:
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")

        # Mock to fail once then succeed
        original_create = conn._create_connection
        call_count = [0]

        def mock_create():
            call_count[0] += 1
            if call_count[0] == 1:
                raise sqlite3.OperationalError("database is locked")
            return original_create()

        conn._create_connection = mock_create

        with conn.transaction() as cursor:
            cursor.execute("SELECT 1")

        assert call_count[0] >= 2  # Should have retried


class TestResilientConnectionExecute:
    """Tests for execute method."""

    def test_execute_insert(self, tmp_path: Path):
        """Execute should return lastrowid for inserts."""
        db_path = str(tmp_path / "test.db")
        conn = ResilientConnection(db_path)

        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        rowid = conn.execute("INSERT INTO test (value) VALUES (?)", ("hello",))

        assert rowid == 1

    def test_execute_with_fetch(self, tmp_path: Path):
        """Execute with fetch=True should return rows."""
        db_path = str(tmp_path / "test.db")
        conn = ResilientConnection(db_path)

        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.execute("INSERT INTO test (value) VALUES (?)", ("hello",))
        conn.execute("INSERT INTO test (value) VALUES (?)", ("world",))

        rows = conn.execute("SELECT * FROM test ORDER BY id", fetch=True)
        assert len(rows) == 2
        assert rows[0]["value"] == "hello"
        assert rows[1]["value"] == "world"


class TestResilientConnectionExecutemany:
    """Tests for executemany method."""

    def test_executemany_bulk_insert(self, tmp_path: Path):
        """Executemany should insert multiple rows."""
        db_path = str(tmp_path / "test.db")
        conn = ResilientConnection(db_path)

        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        params = [("a",), ("b",), ("c",)]
        rowcount = conn.executemany("INSERT INTO test (value) VALUES (?)", params)

        assert rowcount == 3

        rows = conn.execute("SELECT COUNT(*) as cnt FROM test", fetch=True)
        assert rows[0]["cnt"] == 3


# =============================================================================
# with_retry Decorator Tests
# =============================================================================


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    def test_returns_result_on_success(self):
        """Should return function result on success."""

        @with_retry(max_retries=3)
        def successful_func():
            return "success"

        assert successful_func() == "success"

    def test_retries_on_transient_error(self):
        """Should retry on transient SQLite errors."""
        call_count = [0]

        @with_retry(max_retries=3, base_delay=0.01)
        def failing_then_succeeding():
            call_count[0] += 1
            if call_count[0] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "success"

        result = failing_then_succeeding()
        assert result == "success"
        assert call_count[0] == 3

    def test_raises_after_max_retries(self):
        """Should raise after exhausting retries."""

        @with_retry(max_retries=2, base_delay=0.01)
        def always_fails():
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            always_fails()

    def test_no_retry_for_non_transient_errors(self):
        """Should not retry non-transient errors."""
        call_count = [0]

        @with_retry(max_retries=3, base_delay=0.01)
        def non_transient_error():
            call_count[0] += 1
            raise sqlite3.IntegrityError("UNIQUE constraint failed")

        with pytest.raises(sqlite3.IntegrityError):
            non_transient_error()

        assert call_count[0] == 1  # No retries

    def test_preserves_function_metadata(self):
        """Should preserve function name and docstring."""

        @with_retry(max_retries=3)
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert "My docstring" in my_function.__doc__


# =============================================================================
# atomic_transaction Tests
# =============================================================================


class TestAtomicTransaction:
    """Tests for atomic_transaction context manager."""

    def test_successful_atomic_transaction(self, tmp_path: Path):
        """Should commit on successful completion."""
        db_path = str(tmp_path / "test.db")

        # Create table first
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.close()

        with atomic_transaction(db_path) as conn:
            conn.execute("INSERT INTO test (value) VALUES (?)", ("atomic",))

        # Verify commit
        verify_conn = sqlite3.connect(db_path)
        cursor = verify_conn.execute("SELECT value FROM test")
        row = cursor.fetchone()
        assert row[0] == "atomic"
        verify_conn.close()

    def test_rollback_on_exception(self, tmp_path: Path):
        """Should rollback on exception."""
        db_path = str(tmp_path / "test.db")

        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.close()

        with pytest.raises(ValueError):
            with atomic_transaction(db_path) as conn:
                conn.execute("INSERT INTO test (value) VALUES (?)", ("should_rollback",))
                raise ValueError("Intentional error")

        # Verify rollback
        verify_conn = sqlite3.connect(db_path)
        cursor = verify_conn.execute("SELECT COUNT(*) FROM test")
        count = cursor.fetchone()[0]
        assert count == 0
        verify_conn.close()


# =============================================================================
# ConnectionPool Tests
# =============================================================================


class TestConnectionPoolInit:
    """Tests for ConnectionPool initialization."""

    def test_default_values(self, tmp_path: Path):
        """Should initialize with default values."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path)

        assert pool.db_path == db_path
        assert pool.max_connections == 5
        assert pool.enable_wal is True

    def test_custom_values(self, tmp_path: Path):
        """Should accept custom configuration."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(
            db_path,
            max_connections=10,
            timeout=60.0,
            enable_wal=False,
        )

        assert pool.max_connections == 10
        assert pool.timeout == 60.0
        assert pool.enable_wal is False


class TestConnectionPoolGetConnection:
    """Tests for get_connection context manager."""

    def test_creates_connection(self, tmp_path: Path):
        """Should create a new connection."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path)

        with pool.get_connection() as conn:
            cursor = conn.execute("SELECT 1")
            result = cursor.fetchone()[0]
            assert result == 1

    def test_reuses_connections(self, tmp_path: Path):
        """Should reuse connections from pool."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path)

        # First use creates connection
        with pool.get_connection() as conn1:
            pass

        # Second use should reuse
        with pool.get_connection() as conn2:
            pass

        stats = pool.get_stats()
        assert stats["connections_reused"] >= 1

    def test_returns_to_pool(self, tmp_path: Path):
        """Should return connection to pool after use."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path)

        with pool.get_connection():
            stats = pool.get_stats()
            assert stats["active"] == 1
            assert stats["idle"] == 0

        stats = pool.get_stats()
        assert stats["active"] == 0
        assert stats["idle"] == 1

    def test_respects_max_connections(self, tmp_path: Path):
        """Should not keep more than max_connections in pool."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path, max_connections=2)

        # Create and release 5 connections
        for _ in range(5):
            with pool.get_connection():
                pass

        stats = pool.get_stats()
        assert stats["idle"] <= 2


class TestConnectionPoolHealthCheck:
    """Tests for connection health checking."""

    def test_removes_unhealthy_connections(self, tmp_path: Path):
        """Should remove connections that fail health check."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path)

        # Get a connection and close it manually to make it unhealthy
        with pool.get_connection() as conn:
            pass

        # Close the pooled connection to make it unhealthy
        if pool._pool:
            pool._pool[0].close()

        # Next get_connection should detect unhealthy and create new
        with pool.get_connection():
            pass

        stats = pool.get_stats()
        assert stats["health_check_failures"] >= 1


class TestConnectionPoolCloseAll:
    """Tests for close_all method."""

    def test_closes_all_idle_connections(self, tmp_path: Path):
        """Should close all idle connections."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path)

        # Create some connections
        with pool.get_connection():
            pass
        with pool.get_connection():
            pass

        assert pool.get_stats()["idle"] > 0

        pool.close_all()

        stats = pool.get_stats()
        assert stats["idle"] == 0
        assert stats["active"] == 0


class TestConnectionPoolStats:
    """Tests for get_stats method."""

    def test_returns_all_metrics(self, tmp_path: Path):
        """Should return all expected metrics."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path, max_connections=5)

        with pool.get_connection():
            pass

        stats = pool.get_stats()

        assert "db_path" in stats
        assert "max_connections" in stats
        assert "active" in stats
        assert "idle" in stats
        assert "total" in stats
        assert "connections_created" in stats
        assert "connections_reused" in stats
        assert "connections_closed" in stats
        assert "health_check_failures" in stats
        assert "reuse_rate" in stats

    def test_reuse_rate_calculation(self, tmp_path: Path):
        """Should calculate reuse rate correctly."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path)

        # Create one connection (1 created, 0 reused)
        with pool.get_connection():
            pass

        # Reuse it (1 created, 1 reused)
        with pool.get_connection():
            pass

        # Reuse again (1 created, 2 reused)
        with pool.get_connection():
            pass

        stats = pool.get_stats()
        # reuse_rate = reused / (created + reused) = 2 / (1 + 2) = 0.666...
        assert stats["reuse_rate"] == pytest.approx(2 / 3, rel=0.01)


class TestConnectionPoolResetMetrics:
    """Tests for reset_metrics method."""

    def test_resets_all_metrics(self, tmp_path: Path):
        """Should reset all metric counters to zero."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path)

        # Generate some metrics
        with pool.get_connection():
            pass
        with pool.get_connection():
            pass

        assert pool.get_stats()["connections_created"] > 0

        pool.reset_metrics()

        stats = pool.get_stats()
        assert stats["connections_created"] == 0
        assert stats["connections_reused"] == 0
        assert stats["connections_closed"] == 0
        assert stats["health_check_failures"] == 0


class TestConnectionPoolThreadSafety:
    """Tests for thread-safety of ConnectionPool."""

    def test_concurrent_access(self, tmp_path: Path):
        """Should handle concurrent access from multiple threads."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path, max_connections=3)

        # Create table first
        with pool.get_connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, thread TEXT)")

        errors = []
        threads_completed = [0]
        lock = threading.Lock()

        def worker(thread_id: int):
            try:
                for _ in range(10):
                    with pool.get_connection() as conn:
                        conn.execute(
                            "INSERT INTO test (thread) VALUES (?)", (f"thread-{thread_id}",)
                        )
                        conn.commit()
                with lock:
                    threads_completed[0] += 1
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert threads_completed[0] == 5

        # Verify all inserts happened
        with pool.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM test")
            count = cursor.fetchone()[0]
            assert count == 50  # 5 threads x 10 inserts


class TestConnectionPoolWALMode:
    """Tests for WAL mode functionality."""

    def test_enables_wal_by_default(self, tmp_path: Path):
        """Should enable WAL mode by default."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path, enable_wal=True)

        with pool.get_connection() as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"

    def test_can_disable_wal(self, tmp_path: Path):
        """Should respect enable_wal=False."""
        db_path = str(tmp_path / "test.db")
        pool = ConnectionPool(db_path, enable_wal=False)

        with pool.get_connection() as conn:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            # Default SQLite mode is usually 'delete' when WAL not enabled
            assert mode.lower() != "wal"


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        from aragora.storage import resilience

        for name in resilience.__all__:
            assert hasattr(resilience, name), f"Missing export: {name}"

    def test_key_exports(self):
        """Key exports should be available."""
        from aragora.storage.resilience import (
            TRANSIENT_ERRORS,
            is_transient_error,
            ResilientConnection,
            with_retry,
            atomic_transaction,
            ConnectionPool,
        )

        assert TRANSIENT_ERRORS is not None
        assert callable(is_transient_error)
        assert ResilientConnection is not None
        assert callable(with_retry)
        assert callable(atomic_transaction)
        assert ConnectionPool is not None
