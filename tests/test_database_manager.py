"""Tests for the DatabaseManager class in aragora.storage.schema."""

import pytest
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from aragora.storage.schema import DatabaseManager, DB_TIMEOUT


class TestDatabaseManagerSingleton:
    """Tests for DatabaseManager singleton pattern."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_get_instance_creates_new(self, tmp_path):
        """get_instance creates a new manager for new path."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        assert manager is not None
        assert manager.db_path == str(db_path.resolve())

    def test_get_instance_returns_same(self, tmp_path):
        """get_instance returns same manager for same path."""
        db_path = tmp_path / "test.db"
        manager1 = DatabaseManager.get_instance(db_path)
        manager2 = DatabaseManager.get_instance(db_path)

        assert manager1 is manager2

    def test_get_instance_different_paths(self, tmp_path):
        """get_instance returns different managers for different paths."""
        db1 = tmp_path / "test1.db"
        db2 = tmp_path / "test2.db"

        manager1 = DatabaseManager.get_instance(db1)
        manager2 = DatabaseManager.get_instance(db2)

        assert manager1 is not manager2

    def test_get_instance_normalizes_path(self, tmp_path):
        """get_instance normalizes paths to resolve same manager."""
        db_path = tmp_path / "test.db"
        # Use string path that includes .. to test normalization
        parent = tmp_path.parent
        relative_path = parent / "nonexistent" / ".." / tmp_path.name / "test.db"

        manager1 = DatabaseManager.get_instance(db_path)
        manager2 = DatabaseManager.get_instance(relative_path)

        # Both should resolve to the same absolute path
        assert manager1 is manager2

    def test_clear_instances_closes_connections(self, tmp_path):
        """clear_instances closes all connections."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)
        # Force connection creation
        _ = manager.get_connection()

        DatabaseManager.clear_instances()

        # After clearing, new get_instance should create new manager
        manager2 = DatabaseManager.get_instance(db_path)
        assert manager2 is not manager


class TestDatabaseManagerConnection:
    """Tests for DatabaseManager connection management."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_get_connection_creates_connection(self, tmp_path):
        """get_connection creates a new connection."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        conn = manager.get_connection()

        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)

    def test_get_connection_reuses_connection(self, tmp_path):
        """get_connection reuses existing connection."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        conn1 = manager.get_connection()
        conn2 = manager.get_connection()

        assert conn1 is conn2

    def test_connection_is_wal_mode(self, tmp_path):
        """Connection uses WAL mode."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)
        conn = manager.get_connection()

        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"

    def test_connection_context_manager_commits(self, tmp_path):
        """connection() context manager commits on success."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO test (value) VALUES (?)", ("test_value",))

        # Verify data persisted
        result = manager.fetch_one("SELECT value FROM test WHERE id = 1")
        assert result is not None
        assert result[0] == "test_value"

    def test_connection_context_manager_rollback_on_error(self, tmp_path):
        """connection() context manager rolls back on exception."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")

        try:
            with manager.connection() as conn:
                conn.execute("INSERT INTO test (value) VALUES (?)", ("will_rollback",))
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Verify data was not persisted
        result = manager.fetch_one("SELECT COUNT(*) FROM test")
        assert result[0] == 0

    def test_close_closes_connection(self, tmp_path):
        """close() closes the underlying connection."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)
        conn = manager.get_connection()

        manager.close()

        # Connection should be closed
        assert manager._conn is None

    def test_close_is_idempotent(self, tmp_path):
        """close() can be called multiple times safely."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)
        manager.get_connection()

        manager.close()
        manager.close()  # Should not raise

        assert manager._conn is None


class TestDatabaseManagerTransaction:
    """Tests for DatabaseManager transaction handling."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_transaction_commits_on_success(self, tmp_path):
        """transaction() commits on successful completion."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")

        with manager.transaction() as conn:
            conn.execute("INSERT INTO test (value) VALUES (?)", ("txn_value",))

        result = manager.fetch_one("SELECT value FROM test WHERE id = 1")
        assert result is not None
        assert result[0] == "txn_value"

    def test_transaction_rollback_on_error(self, tmp_path):
        """transaction() rolls back on exception."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")

        try:
            with manager.transaction() as conn:
                conn.execute("INSERT INTO test (value) VALUES (?)", ("will_rollback",))
                raise ValueError("Test exception")
        except ValueError:
            pass

        result = manager.fetch_one("SELECT COUNT(*) FROM test")
        assert result[0] == 0


class TestDatabaseManagerExecute:
    """Tests for DatabaseManager execute methods."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_execute_returns_cursor(self, tmp_path):
        """execute() returns a cursor."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        cursor = manager.execute("SELECT 1 + 1")

        assert cursor is not None
        result = cursor.fetchone()
        assert result[0] == 2

    def test_execute_with_params(self, tmp_path):
        """execute() works with parameters."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")

        manager.execute("INSERT INTO test (id, name) VALUES (?, ?)", (1, "Alice"))

        result = manager.fetch_one("SELECT name FROM test WHERE id = ?", (1,))
        assert result[0] == "Alice"

    def test_executemany(self, tmp_path):
        """executemany() executes with multiple parameter sets."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")

        params = [(1, "Alice"), (2, "Bob"), (3, "Charlie")]
        manager.executemany("INSERT INTO test (id, name) VALUES (?, ?)", params)

        result = manager.fetch_all("SELECT name FROM test ORDER BY id")
        names = [row[0] for row in result]
        assert names == ["Alice", "Bob", "Charlie"]


class TestDatabaseManagerFetch:
    """Tests for DatabaseManager fetch methods."""

    def setup_method(self):
        """Clear singleton instances and create test table."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_fetch_one_returns_single_row(self, tmp_path):
        """fetch_one() returns a single row."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")

        result = manager.fetch_one("SELECT name FROM test WHERE id = ?", (1,))

        assert result is not None
        assert result[0] == "Alice"

    def test_fetch_one_returns_none_for_no_match(self, tmp_path):
        """fetch_one() returns None when no rows match."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")

        result = manager.fetch_one("SELECT name FROM test WHERE id = ?", (999,))

        assert result is None

    def test_fetch_all_returns_all_rows(self, tmp_path):
        """fetch_all() returns all matching rows."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie')")

        result = manager.fetch_all("SELECT name FROM test ORDER BY id")

        assert len(result) == 3
        names = [row[0] for row in result]
        assert names == ["Alice", "Bob", "Charlie"]

    def test_fetch_all_returns_empty_list(self, tmp_path):
        """fetch_all() returns empty list when no rows match."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")

        result = manager.fetch_all("SELECT name FROM test")

        assert result == []

    def test_fetch_many_limits_rows(self, tmp_path):
        """fetch_many() returns at most 'size' rows."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            for i in range(10):
                conn.execute("INSERT INTO test VALUES (?, ?)", (i, f"Name{i}"))

        result = manager.fetch_many("SELECT name FROM test ORDER BY id", size=3)

        assert len(result) == 3
        names = [row[0] for row in result]
        assert names == ["Name0", "Name1", "Name2"]

    def test_fetch_many_returns_all_when_fewer_than_size(self, tmp_path):
        """fetch_many() returns all rows when fewer than size exist."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")

        result = manager.fetch_many("SELECT name FROM test ORDER BY id", size=10)

        assert len(result) == 2


class TestDatabaseManagerThreadSafety:
    """Tests for DatabaseManager thread safety."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_get_instance_thread_safe(self, tmp_path):
        """get_instance is thread-safe."""
        db_path = tmp_path / "test.db"
        managers = []
        errors = []

        def get_manager():
            try:
                manager = DatabaseManager.get_instance(db_path)
                managers.append(manager)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_manager) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All should be the same instance
        assert all(m is managers[0] for m in managers)

    def test_connection_reuse_single_thread(self, tmp_path):
        """Connection is reused within the same thread."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, count INTEGER)")
            conn.execute("INSERT INTO test VALUES (1, 0)")

        # Multiple operations in same thread should work
        for _ in range(10):
            with manager.connection() as conn:
                current = conn.execute("SELECT count FROM test WHERE id = 1").fetchone()[0]
                conn.execute("UPDATE test SET count = ? WHERE id = 1", (current + 1,))

        result = manager.fetch_one("SELECT count FROM test WHERE id = 1")
        assert result[0] == 10

    def test_cross_thread_access_raises_error(self, tmp_path):
        """Using connection across threads raises ProgrammingError.

        Note: DatabaseManager uses check_same_thread=True by default, so
        connections cannot be shared across threads. For multi-threaded access,
        use per-operation connections (like EloDatabase pattern).
        """
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        # Create connection in main thread
        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")

        errors = []

        def access_from_thread():
            try:
                with manager.connection() as conn:
                    conn.execute("SELECT * FROM test")
            except sqlite3.ProgrammingError as e:
                errors.append(e)

        thread = threading.Thread(target=access_from_thread)
        thread.start()
        thread.join()

        # Should raise ProgrammingError about thread safety
        assert len(errors) == 1
        assert "thread" in str(errors[0]).lower()


class TestDatabaseManagerErrorHandling:
    """Tests for DatabaseManager error handling."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_execute_raises_on_invalid_sql(self, tmp_path):
        """execute() raises on invalid SQL."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with pytest.raises(sqlite3.OperationalError):
            manager.execute("INVALID SQL STATEMENT")

    def test_connection_context_manager_preserves_original_exception(self, tmp_path):
        """connection() context manager preserves the original exception type."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            with manager.connection():
                raise CustomError("Custom error")

    def test_fetch_one_raises_on_invalid_sql(self, tmp_path):
        """fetch_one() raises on invalid SQL."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with pytest.raises(sqlite3.OperationalError):
            manager.fetch_one("SELECT * FROM nonexistent_table")


class TestDatabaseManagerRepr:
    """Tests for DatabaseManager repr."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_repr(self, tmp_path):
        """__repr__ returns useful string."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        repr_str = repr(manager)

        assert "DatabaseManager" in repr_str
        assert "test.db" in repr_str


class TestDatabaseManagerTimeout:
    """Tests for DatabaseManager timeout handling."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_custom_timeout(self, tmp_path):
        """get_instance respects custom timeout."""
        db_path = tmp_path / "test.db"
        custom_timeout = 60.0

        manager = DatabaseManager.get_instance(db_path, timeout=custom_timeout)

        assert manager.timeout == custom_timeout


class TestDatabaseManagerFreshConnection:
    """Tests for DatabaseManager fresh_connection (per-operation)."""

    def setup_method(self):
        """Clear singleton instances before each test."""
        DatabaseManager.clear_instances()

    def teardown_method(self):
        """Clean up after each test."""
        DatabaseManager.clear_instances()

    def test_fresh_connection_closes_after_use(self, tmp_path):
        """fresh_connection closes connection after exiting context."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        conn_ref = None
        with manager.fresh_connection() as conn:
            conn.execute("SELECT 1")
            conn_ref = conn

        # Connection should be closed after context exits
        # Attempting to use it should raise an error
        with pytest.raises(sqlite3.ProgrammingError):
            conn_ref.execute("SELECT 1")

    def test_fresh_connection_commits_on_success(self, tmp_path):
        """fresh_connection commits on successful completion."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.fresh_connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice')")

        # Verify with new connection
        with manager.fresh_connection() as conn:
            result = conn.execute("SELECT name FROM test WHERE id = 1").fetchone()
            assert result[0] == "Alice"

    def test_fresh_connection_rollback_on_error(self, tmp_path):
        """fresh_connection rolls back on exception."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.fresh_connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")

        try:
            with manager.fresh_connection() as conn:
                conn.execute("INSERT INTO test VALUES (1, 'will_rollback')")
                raise ValueError("Test exception")
        except ValueError:
            pass

        with manager.fresh_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM test").fetchone()
            assert result[0] == 0

    def test_fresh_connection_thread_safe(self, tmp_path):
        """fresh_connection works correctly from multiple threads."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.fresh_connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, count INTEGER)")
            conn.execute("INSERT INTO test VALUES (1, 0)")

        errors = []
        increment_count = 20

        def increment():
            try:
                for _ in range(increment_count):
                    with manager.fresh_connection() as conn:
                        current = conn.execute("SELECT count FROM test WHERE id = 1").fetchone()[0]
                        conn.execute("UPDATE test SET count = ? WHERE id = 1", (current + 1,))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Primary assertion: no thread-safety errors (like ProgrammingError)
        assert len(errors) == 0
        # Count should have increased (though may be less than 100 due to race conditions)
        with manager.fresh_connection() as conn:
            result = conn.execute("SELECT count FROM test WHERE id = 1").fetchone()
            assert result[0] > 0
