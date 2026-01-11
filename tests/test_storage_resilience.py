"""
Tests for database connection resilience with retry logic.

Tests cover:
- ResilientConnection retry behavior
- with_retry decorator
- ConnectionPool functionality
- Transient error detection
- Exponential backoff delays
"""

import sqlite3
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.resilience import (
    ConnectionPool,
    ResilientConnection,
    atomic_transaction,
    is_transient_error,
    with_retry,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        # Initialize schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
        conn.commit()
        conn.close()
        yield str(db_path)


class TestTransientErrorDetection:
    """Tests for transient error detection."""

    def test_database_is_locked_is_transient(self):
        """Test that 'database is locked' is detected as transient."""
        error = sqlite3.OperationalError("database is locked")
        assert is_transient_error(error) is True

    def test_database_is_busy_is_transient(self):
        """Test that 'database is busy' is detected as transient."""
        error = sqlite3.OperationalError("database is busy")
        assert is_transient_error(error) is True

    def test_disk_io_error_is_transient(self):
        """Test that disk I/O errors are detected as transient."""
        error = sqlite3.OperationalError("disk i/o error")
        assert is_transient_error(error) is True

    def test_unable_to_open_is_transient(self):
        """Test that file open errors are detected as transient."""
        error = sqlite3.OperationalError("unable to open database file")
        assert is_transient_error(error) is True

    def test_constraint_violation_not_transient(self):
        """Test that constraint violations are not transient."""
        error = sqlite3.IntegrityError("UNIQUE constraint failed")
        assert is_transient_error(error) is False

    def test_syntax_error_not_transient(self):
        """Test that syntax errors are not transient."""
        error = sqlite3.OperationalError("near 'SELCT': syntax error")
        assert is_transient_error(error) is False

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        error = sqlite3.OperationalError("DATABASE IS LOCKED")
        assert is_transient_error(error) is True


class TestResilientConnection:
    """Tests for ResilientConnection."""

    def test_basic_execute(self, temp_db):
        """Test basic query execution."""
        conn = ResilientConnection(temp_db)

        # Insert
        last_id = conn.execute(
            "INSERT INTO test (value) VALUES (?)",
            ("hello",)
        )
        assert last_id is not None

        # Select
        rows = conn.execute(
            "SELECT * FROM test WHERE value = ?",
            ("hello",),
            fetch=True
        )
        assert len(rows) == 1
        assert rows[0]["value"] == "hello"

    def test_executemany(self, temp_db):
        """Test bulk insert."""
        conn = ResilientConnection(temp_db)

        params = [("value1",), ("value2",), ("value3",)]
        affected = conn.executemany(
            "INSERT INTO test (value) VALUES (?)",
            params
        )
        assert affected == 3

    def test_transaction_commit(self, temp_db):
        """Test that transactions are committed."""
        conn = ResilientConnection(temp_db)

        with conn.transaction() as cursor:
            cursor.execute("INSERT INTO test (value) VALUES (?)", ("committed",))

        # Verify with new connection
        rows = conn.execute(
            "SELECT * FROM test WHERE value = ?",
            ("committed",),
            fetch=True
        )
        assert len(rows) == 1

    def test_transaction_rollback_on_error(self, temp_db):
        """Test that transactions are rolled back on error."""
        conn = ResilientConnection(temp_db)

        try:
            with conn.transaction() as cursor:
                cursor.execute("INSERT INTO test (value) VALUES (?)", ("rollback_test",))
                # Force an error
                cursor.execute("INVALID SQL")
        except sqlite3.Error:
            pass

        # Verify rollback
        rows = conn.execute(
            "SELECT * FROM test WHERE value = ?",
            ("rollback_test",),
            fetch=True
        )
        assert len(rows) == 0

    def test_retry_on_transient_error(self, temp_db):
        """Test that transient errors trigger retry."""
        conn = ResilientConnection(temp_db, max_retries=3, base_delay=0.01)

        call_count = 0
        original_create = conn._create_connection

        def mock_create():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            return original_create()

        conn._create_connection = mock_create

        # Should succeed after retries
        with conn.transaction() as cursor:
            cursor.execute("SELECT 1")

        assert call_count == 3

    def test_max_retries_exceeded(self, temp_db):
        """Test that error is raised after max retries."""
        conn = ResilientConnection(temp_db, max_retries=2, base_delay=0.01)

        def always_fail():
            raise sqlite3.OperationalError("database is locked")

        conn._create_connection = always_fail

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            with conn.transaction() as cursor:
                cursor.execute("SELECT 1")

    def test_non_transient_error_not_retried(self, temp_db):
        """Test that non-transient errors are not retried."""
        conn = ResilientConnection(temp_db, max_retries=3, base_delay=0.01)

        call_count = 0
        original_create = conn._create_connection

        def mock_create():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise sqlite3.IntegrityError("constraint failed")
            return original_create()

        conn._create_connection = mock_create

        with pytest.raises(sqlite3.IntegrityError):
            with conn.transaction() as cursor:
                cursor.execute("SELECT 1")

        # Should not have retried
        assert call_count == 1

    def test_exponential_backoff_delays(self, temp_db):
        """Test that exponential backoff is applied correctly."""
        conn = ResilientConnection(
            temp_db,
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0
        )

        # Check delay calculations
        assert conn._calculate_delay(0) == 0.1  # 0.1 * 2^0
        assert conn._calculate_delay(1) == 0.2  # 0.1 * 2^1
        assert conn._calculate_delay(2) == 0.4  # 0.1 * 2^2
        assert conn._calculate_delay(5) == 1.0  # Capped at max_delay


class TestWithRetryDecorator:
    """Tests for the with_retry decorator."""

    def test_successful_execution(self, temp_db):
        """Test decorator with successful function."""
        @with_retry(max_retries=3)
        def insert_record(db_path: str, value: str):
            conn = sqlite3.connect(db_path)
            conn.execute("INSERT INTO test (value) VALUES (?)", (value,))
            conn.commit()
            conn.close()

        insert_record(temp_db, "decorated")

        # Verify
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT * FROM test WHERE value = ?", ("decorated",))
        assert cursor.fetchone() is not None
        conn.close()

    def test_retry_on_failure(self, temp_db):
        """Test that decorator retries on transient error."""
        call_count = 0

        @with_retry(max_retries=3, base_delay=0.01)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is busy")
            return "success"

        result = flaky_function()

        assert result == "success"
        assert call_count == 3

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function name and docstring."""
        @with_retry()
        def documented_function():
            """This is the docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is the docstring."


class TestConnectionPool:
    """Tests for ConnectionPool."""

    def test_basic_pool_usage(self, temp_db):
        """Test basic connection pool usage."""
        pool = ConnectionPool(temp_db, max_connections=3)

        try:
            with pool.get_connection() as conn:
                cursor = conn.execute("SELECT 1")
                result = cursor.fetchone()
                assert result[0] == 1
        finally:
            pool.close_all()

    def test_connection_reuse(self, temp_db):
        """Test that connections are reused."""
        pool = ConnectionPool(temp_db, max_connections=3)

        try:
            # Get first connection
            with pool.get_connection() as conn1:
                id1 = id(conn1)

            # Pool should have 1 connection now
            assert len(pool._pool) == 1

            # Get another connection - should reuse
            with pool.get_connection() as conn2:
                id2 = id(conn2)

            # Same connection object should be reused
            assert id1 == id2
        finally:
            pool.close_all()

    def test_max_pool_size(self, temp_db):
        """Test that pool respects max size."""
        pool = ConnectionPool(temp_db, max_connections=2)

        try:
            # Acquire and release 5 connections
            for _ in range(5):
                with pool.get_connection():
                    pass

            # Pool should have at most max_connections
            assert len(pool._pool) <= 2
        finally:
            pool.close_all()

    def test_concurrent_pool_access(self, temp_db):
        """Test thread-safe concurrent access to pool."""
        pool = ConnectionPool(temp_db, max_connections=5)
        errors = []
        success_count = 0
        lock = threading.Lock()

        def worker(worker_id):
            nonlocal success_count
            try:
                with pool.get_connection() as conn:
                    # Simulate some work
                    conn.execute("SELECT 1")
                    time.sleep(0.01)
                with lock:
                    success_count += 1
            except Exception as e:
                with lock:
                    errors.append(str(e))

        try:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(worker, i) for i in range(20)]
                for f in futures:
                    f.result()

            assert len(errors) == 0, f"Errors: {errors}"
            assert success_count == 20
        finally:
            pool.close_all()

    def test_unhealthy_connection_removed(self, temp_db):
        """Test that unhealthy connections are removed from pool."""
        pool = ConnectionPool(temp_db, max_connections=3)

        try:
            # Get and return a connection
            with pool.get_connection() as conn:
                pass

            assert len(pool._pool) == 1

            # Corrupt the connection
            pool._pool[0].close()

            # Getting new connection should skip the closed one
            with pool.get_connection() as conn:
                cursor = conn.execute("SELECT 1")
                assert cursor.fetchone()[0] == 1

            # Pool should have 1 healthy connection now
            assert len(pool._pool) == 1
        finally:
            pool.close_all()

    def test_close_all_clears_pool(self, temp_db):
        """Test that close_all removes all connections."""
        pool = ConnectionPool(temp_db, max_connections=3)

        # Create some connections
        for _ in range(3):
            with pool.get_connection():
                pass

        assert len(pool._pool) > 0

        pool.close_all()

        assert len(pool._pool) == 0
        assert len(pool._in_use) == 0


class TestIntegration:
    """Integration tests for resilience features."""

    def test_concurrent_writes_with_resilience(self, temp_db):
        """Test concurrent writes with resilient connections."""
        success_count = 0
        errors = []
        lock = threading.Lock()

        def write_record(worker_id):
            nonlocal success_count
            conn = ResilientConnection(temp_db, max_retries=5, base_delay=0.01)
            try:
                conn.execute(
                    "INSERT INTO test (value) VALUES (?)",
                    (f"worker_{worker_id}",)
                )
                with lock:
                    success_count += 1
            except Exception as e:
                with lock:
                    errors.append(str(e))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_record, i) for i in range(50)]
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors: {errors}"
        assert success_count == 50

        # Verify all records were inserted
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT COUNT(*) FROM test")
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 50

    def test_pool_with_retry_decorator(self, temp_db):
        """Test combining pool with retry decorator."""
        pool = ConnectionPool(temp_db, max_connections=3)

        @with_retry(max_retries=3, base_delay=0.01)
        def pool_operation(value: str):
            with pool.get_connection() as conn:
                conn.execute("INSERT INTO test (value) VALUES (?)", (value,))
                conn.commit()

        try:
            for i in range(10):
                pool_operation(f"pool_retry_{i}")

            # Verify
            with pool.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM test WHERE value LIKE ?",
                    ("pool_retry_%",)
                )
                count = cursor.fetchone()[0]
                assert count == 10
        finally:
            pool.close_all()


class TestAtomicTransaction:
    """Tests for atomic_transaction context manager."""

    def test_atomic_transaction_commit(self, temp_db):
        """Test atomic transaction commits on success."""
        with atomic_transaction(temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO test (value) VALUES (?)", ("atomic1",))
            cursor.execute("INSERT INTO test (value) VALUES (?)", ("atomic2",))

        # Verify both inserts were committed
        verify_conn = sqlite3.connect(temp_db)
        cursor = verify_conn.execute("SELECT COUNT(*) FROM test WHERE value LIKE ?", ("atomic%",))
        count = cursor.fetchone()[0]
        verify_conn.close()
        assert count == 2

    def test_atomic_transaction_rollback_on_error(self, temp_db):
        """Test atomic transaction rolls back on error."""
        try:
            with atomic_transaction(temp_db) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO test (value) VALUES (?)", ("rollback_test",))
                # Force an error
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Verify insert was rolled back
        verify_conn = sqlite3.connect(temp_db)
        cursor = verify_conn.execute("SELECT COUNT(*) FROM test WHERE value = ?", ("rollback_test",))
        count = cursor.fetchone()[0]
        verify_conn.close()
        assert count == 0

    def test_atomic_transaction_uses_immediate(self, temp_db):
        """Test that atomic transaction uses BEGIN IMMEDIATE."""
        with patch("aragora.storage.schema.get_wal_connection") as mock_get_conn:
            mock_conn = MagicMock()
            mock_get_conn.return_value = mock_conn

            try:
                with atomic_transaction(temp_db) as conn:
                    pass
            except Exception:
                pass  # May fail due to mock, but we just want to check the call

            # Verify BEGIN IMMEDIATE was called
            mock_conn.execute.assert_any_call("BEGIN IMMEDIATE")

    def test_atomic_transaction_retry_on_lock(self, temp_db):
        """Test atomic transaction retries on database lock."""
        call_count = 0

        def mock_get_conn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            # Return real connection on third attempt
            conn = sqlite3.connect(temp_db)
            conn.row_factory = sqlite3.Row
            return conn

        with patch("aragora.storage.schema.get_wal_connection", side_effect=mock_get_conn):
            with atomic_transaction(temp_db, max_retries=3, base_delay=0.01) as conn:
                conn.execute("INSERT INTO test (value) VALUES (?)", ("retry_success",))

        assert call_count == 3

        # Verify the insert succeeded
        verify_conn = sqlite3.connect(temp_db)
        cursor = verify_conn.execute("SELECT COUNT(*) FROM test WHERE value = ?", ("retry_success",))
        count = cursor.fetchone()[0]
        verify_conn.close()
        assert count == 1

    def test_atomic_transaction_max_retries_exceeded(self, temp_db):
        """Test atomic transaction fails after max retries."""
        def always_locked(*args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

        with patch("aragora.storage.schema.get_wal_connection", side_effect=always_locked):
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                with atomic_transaction(temp_db, max_retries=2, base_delay=0.01) as conn:
                    pass

    def test_atomic_transaction_non_transient_error_not_retried(self, temp_db):
        """Test non-transient errors are not retried."""
        call_count = 0

        def fail_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("no such table: nonexistent")

        with patch("aragora.storage.schema.get_wal_connection", side_effect=fail_once):
            with pytest.raises(sqlite3.OperationalError, match="no such table"):
                with atomic_transaction(temp_db, max_retries=3, base_delay=0.01) as conn:
                    pass

        # Should not have retried
        assert call_count == 1

    def test_atomic_transaction_concurrent_writes(self, temp_db):
        """Test atomic transaction handles concurrent writes."""
        errors = []
        success_count = 0
        lock = threading.Lock()

        def writer(worker_id):
            nonlocal success_count
            try:
                with atomic_transaction(temp_db, max_retries=5, base_delay=0.01) as conn:
                    conn.execute(
                        "INSERT INTO test (value) VALUES (?)",
                        (f"concurrent_{worker_id}",)
                    )
                with lock:
                    success_count += 1
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert success_count == 10

        # Verify all inserts
        verify_conn = sqlite3.connect(temp_db)
        cursor = verify_conn.execute("SELECT COUNT(*) FROM test WHERE value LIKE ?", ("concurrent_%",))
        count = cursor.fetchone()[0]
        verify_conn.close()
        assert count == 10
