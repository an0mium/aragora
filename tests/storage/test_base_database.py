"""
Tests for aragora.storage.base_database - BaseDatabase abstraction.

Tests cover:
- BaseDatabase initialization
- Connection management (connection(), transaction(), fresh_connection)
- Fetch methods (fetch_one, fetch_all)
- Write operations (execute_write, executemany)
- Thread-safety and concurrent access
- Transaction rollback behavior
- Connection pooling via DatabaseManager
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.base_database import BaseDatabase
from aragora.storage.schema import DatabaseManager


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_database_manager():
    """Clean up DatabaseManager instances after each test."""
    yield
    DatabaseManager.clear_instances()


@pytest.fixture
def tmp_db_path(tmp_path):
    """Provide a temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def db_with_table(tmp_db_path):
    """Create a database with a test table."""
    db = BaseDatabase(tmp_db_path)
    with db.connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_data (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                value INTEGER DEFAULT 0
            )
        """)
    return db


# ============================================================================
# Test BaseDatabase Initialization
# ============================================================================


class TestBaseDatabaseInit:
    """Tests for BaseDatabase initialization."""

    def test_creates_database_file(self, tmp_db_path):
        """Database file is created on first connection."""
        assert not tmp_db_path.exists()
        db = BaseDatabase(tmp_db_path)
        # File is created when connection is accessed
        with db.connection() as conn:
            conn.execute("SELECT 1")
        assert tmp_db_path.exists()

    def test_accepts_string_path(self, tmp_path):
        """Accepts string path."""
        db_path = str(tmp_path / "test.db")
        db = BaseDatabase(db_path)
        assert db.db_path == Path(db_path)

    def test_accepts_pathlib_path(self, tmp_db_path):
        """Accepts Path object."""
        db = BaseDatabase(tmp_db_path)
        assert db.db_path == tmp_db_path

    def test_custom_timeout(self, tmp_db_path):
        """Custom timeout is stored."""
        db = BaseDatabase(tmp_db_path, timeout=60.0)
        assert db._timeout == 60.0

    def test_uses_database_manager(self, tmp_db_path):
        """Uses DatabaseManager for connection management."""
        db = BaseDatabase(tmp_db_path)
        assert db._manager is not None
        assert isinstance(db._manager, DatabaseManager)


# ============================================================================
# Test Connection Context Manager
# ============================================================================


class TestConnectionContextManager:
    """Tests for connection() context manager."""

    def test_yields_connection(self, db_with_table):
        """connection() yields a sqlite3 connection."""
        with db_with_table.connection() as conn:
            assert isinstance(conn, sqlite3.Connection)
            cursor = conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1

    def test_auto_commits_on_success(self, db_with_table):
        """Changes are committed on successful exit."""
        with db_with_table.connection() as conn:
            conn.execute(
                "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                ("1", "Test", 100),
            )

        # Data should be persisted
        row = db_with_table.fetch_one("SELECT * FROM test_data WHERE id = ?", ("1",))
        assert row is not None
        assert row[1] == "Test"

    def test_rollback_on_exception(self, db_with_table):
        """Changes are rolled back on exception."""
        with pytest.raises(RuntimeError):
            with db_with_table.connection() as conn:
                conn.execute(
                    "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                    ("1", "Test", 100),
                )
                raise RuntimeError("Test error")

        # Data should not be persisted
        row = db_with_table.fetch_one("SELECT * FROM test_data WHERE id = ?", ("1",))
        assert row is None

    def test_rollback_on_sqlite_error(self, db_with_table):
        """Changes are rolled back on SQLite error."""
        with db_with_table.connection() as conn:
            conn.execute(
                "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                ("1", "First", 100),
            )

        with pytest.raises(sqlite3.IntegrityError):
            with db_with_table.connection() as conn:
                # This should fail due to primary key violation
                conn.execute(
                    "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                    ("1", "Duplicate", 200),
                )

        # Original data should still be there, not corrupted
        row = db_with_table.fetch_one("SELECT * FROM test_data WHERE id = ?", ("1",))
        assert row is not None
        assert row[1] == "First"


# ============================================================================
# Test Transaction Context Manager
# ============================================================================


class TestTransactionContextManager:
    """Tests for transaction() context manager."""

    def test_explicit_begin_commit(self, db_with_table):
        """transaction() uses explicit BEGIN/COMMIT."""
        with db_with_table.transaction() as conn:
            conn.execute(
                "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                ("1", "Test", 100),
            )

        # Data should be persisted
        row = db_with_table.fetch_one("SELECT * FROM test_data WHERE id = ?", ("1",))
        assert row is not None

    def test_rollback_on_exception(self, db_with_table):
        """transaction() rolls back on exception."""
        with pytest.raises(ValueError):
            with db_with_table.transaction() as conn:
                conn.execute(
                    "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                    ("1", "Test", 100),
                )
                raise ValueError("Validation failed")

        # Data should not be persisted
        row = db_with_table.fetch_one("SELECT * FROM test_data WHERE id = ?", ("1",))
        assert row is None

    def test_multiple_operations_atomic(self, db_with_table):
        """Multiple operations in transaction are atomic."""
        with pytest.raises(RuntimeError):
            with db_with_table.transaction() as conn:
                conn.execute(
                    "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                    ("1", "First", 100),
                )
                conn.execute(
                    "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                    ("2", "Second", 200),
                )
                raise RuntimeError("Error after inserts")

        # Both should be rolled back
        count = db_with_table.fetch_one("SELECT COUNT(*) FROM test_data")
        assert count[0] == 0


# ============================================================================
# Test Fetch Methods
# ============================================================================


class TestFetchMethods:
    """Tests for fetch_one and fetch_all methods."""

    @pytest.fixture
    def db_with_data(self, db_with_table):
        """Create database with test data."""
        with db_with_table.connection() as conn:
            conn.execute(
                "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                ("1", "Alpha", 100),
            )
            conn.execute(
                "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                ("2", "Beta", 200),
            )
            conn.execute(
                "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                ("3", "Gamma", 300),
            )
        return db_with_table

    def test_fetch_one_returns_row(self, db_with_data):
        """fetch_one returns a single row."""
        row = db_with_data.fetch_one("SELECT * FROM test_data WHERE id = ?", ("1",))
        assert row is not None
        assert row[0] == "1"
        assert row[1] == "Alpha"
        assert row[2] == 100

    def test_fetch_one_returns_none(self, db_with_data):
        """fetch_one returns None when no match."""
        row = db_with_data.fetch_one("SELECT * FROM test_data WHERE id = ?", ("999",))
        assert row is None

    def test_fetch_one_with_no_params(self, db_with_data):
        """fetch_one works with no parameters."""
        row = db_with_data.fetch_one("SELECT COUNT(*) FROM test_data")
        assert row[0] == 3

    def test_fetch_all_returns_list(self, db_with_data):
        """fetch_all returns list of rows."""
        rows = db_with_data.fetch_all("SELECT * FROM test_data ORDER BY id")
        assert len(rows) == 3
        assert rows[0][0] == "1"
        assert rows[1][0] == "2"
        assert rows[2][0] == "3"

    def test_fetch_all_returns_empty_list(self, db_with_data):
        """fetch_all returns empty list when no matches."""
        rows = db_with_data.fetch_all("SELECT * FROM test_data WHERE value > ?", (1000,))
        assert rows == []

    def test_fetch_all_with_filter(self, db_with_data):
        """fetch_all works with filter parameters."""
        rows = db_with_data.fetch_all("SELECT * FROM test_data WHERE value >= ?", (200,))
        assert len(rows) == 2


# ============================================================================
# Test Write Operations
# ============================================================================


class TestWriteOperations:
    """Tests for execute_write and executemany methods."""

    def test_execute_write_inserts(self, db_with_table):
        """execute_write inserts data."""
        db_with_table.execute_write(
            "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
            ("1", "Test", 100),
        )

        row = db_with_table.fetch_one("SELECT * FROM test_data WHERE id = ?", ("1",))
        assert row is not None
        assert row[1] == "Test"

    def test_execute_write_updates(self, db_with_table):
        """execute_write updates data."""
        db_with_table.execute_write(
            "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
            ("1", "Original", 100),
        )
        db_with_table.execute_write(
            "UPDATE test_data SET name = ? WHERE id = ?",
            ("Updated", "1"),
        )

        row = db_with_table.fetch_one("SELECT * FROM test_data WHERE id = ?", ("1",))
        assert row[1] == "Updated"

    def test_execute_write_deletes(self, db_with_table):
        """execute_write deletes data."""
        db_with_table.execute_write(
            "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
            ("1", "Test", 100),
        )
        db_with_table.execute_write("DELETE FROM test_data WHERE id = ?", ("1",))

        row = db_with_table.fetch_one("SELECT * FROM test_data WHERE id = ?", ("1",))
        assert row is None

    def test_executemany_bulk_insert(self, db_with_table):
        """executemany inserts multiple rows."""
        params = [
            ("1", "First", 100),
            ("2", "Second", 200),
            ("3", "Third", 300),
        ]
        db_with_table.executemany(
            "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
            params,
        )

        count = db_with_table.fetch_one("SELECT COUNT(*) FROM test_data")
        assert count[0] == 3

    def test_executemany_empty_list(self, db_with_table):
        """executemany with empty list does nothing."""
        db_with_table.executemany(
            "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
            [],
        )

        count = db_with_table.fetch_one("SELECT COUNT(*) FROM test_data")
        assert count[0] == 0


# ============================================================================
# Test _get_connection (Backward Compatibility)
# ============================================================================


class TestGetConnection:
    """Tests for _get_connection method."""

    def test_returns_connection(self, db_with_table):
        """_get_connection returns a connection."""
        conn = db_with_table._get_connection()
        assert isinstance(conn, sqlite3.Connection)

    def test_connection_is_reused(self, db_with_table):
        """Same connection is returned on subsequent calls."""
        conn1 = db_with_table._get_connection()
        conn2 = db_with_table._get_connection()
        # They should be the same connection object
        assert conn1 is conn2


# ============================================================================
# Test Thread Safety
# ============================================================================


class TestThreadSafety:
    """Tests for thread-safe database access."""

    def test_concurrent_reads(self, db_with_table):
        """Concurrent reads work correctly."""
        # Add some data
        with db_with_table.connection() as conn:
            for i in range(10):
                conn.execute(
                    "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                    (str(i), f"Item {i}", i * 10),
                )

        results = []
        errors = []

        def reader():
            try:
                for _ in range(10):
                    count = db_with_table.fetch_one("SELECT COUNT(*) FROM test_data")
                    results.append(count[0])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 50
        assert all(count == 10 for count in results)

    def test_concurrent_writes(self, db_with_table):
        """Concurrent writes work correctly."""
        errors = []

        def writer(thread_id):
            try:
                for i in range(5):
                    db_with_table.execute_write(
                        "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                        (f"{thread_id}_{i}", f"Thread {thread_id}", i),
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        count = db_with_table.fetch_one("SELECT COUNT(*) FROM test_data")
        assert count[0] == 25

    def test_concurrent_read_write(self, db_with_table):
        """Concurrent reads and writes work correctly."""
        # Add initial data
        with db_with_table.connection() as conn:
            conn.execute(
                "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                ("counter", "Counter", 0),
            )

        errors = []
        reads = []

        def writer():
            try:
                for i in range(10):
                    db_with_table.execute_write(
                        "UPDATE test_data SET value = value + 1 WHERE id = ?",
                        ("counter",),
                    )
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(10):
                    row = db_with_table.fetch_one(
                        "SELECT value FROM test_data WHERE id = ?", ("counter",)
                    )
                    reads.append(row[0])
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        writer_thread = threading.Thread(target=writer)
        reader_threads = [threading.Thread(target=reader) for _ in range(3)]

        writer_thread.start()
        for t in reader_threads:
            t.start()

        writer_thread.join()
        for t in reader_threads:
            t.join()

        assert not errors
        # Final value should be 10
        row = db_with_table.fetch_one("SELECT value FROM test_data WHERE id = ?", ("counter",))
        assert row[0] == 10


# ============================================================================
# Test DatabaseManager Integration
# ============================================================================


class TestDatabaseManagerIntegration:
    """Tests for DatabaseManager integration."""

    def test_same_path_shares_manager(self, tmp_db_path):
        """Same path shares DatabaseManager instance."""
        db1 = BaseDatabase(tmp_db_path)
        db2 = BaseDatabase(tmp_db_path)
        assert db1._manager is db2._manager

    def test_different_paths_different_managers(self, tmp_path):
        """Different paths get different managers."""
        db1 = BaseDatabase(tmp_path / "db1.db")
        db2 = BaseDatabase(tmp_path / "db2.db")
        assert db1._manager is not db2._manager


# ============================================================================
# Test repr
# ============================================================================


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr_format(self, tmp_db_path):
        """__repr__ returns informative string."""
        db = BaseDatabase(tmp_db_path)
        repr_str = repr(db)
        assert "BaseDatabase" in repr_str
        assert str(tmp_db_path) in repr_str


# ============================================================================
# Test Subclassing
# ============================================================================


class TestSubclassing:
    """Tests for BaseDatabase subclassing."""

    def test_simple_subclass(self, tmp_db_path):
        """Simple subclass works correctly."""

        class MyDatabase(BaseDatabase):
            """Custom database wrapper."""

            pass

        db = MyDatabase(tmp_db_path)
        with db.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1)")

        row = db.fetch_one("SELECT id FROM test")
        assert row[0] == 1

    def test_subclass_with_custom_methods(self, tmp_db_path):
        """Subclass with custom methods works correctly."""

        class UserDatabase(BaseDatabase):
            """Database with user-specific methods."""

            def create_users_table(self):
                with self.connection() as conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            id TEXT PRIMARY KEY,
                            name TEXT NOT NULL
                        )
                    """)

            def add_user(self, user_id: str, name: str):
                with self.connection() as conn:
                    conn.execute(
                        "INSERT INTO users (id, name) VALUES (?, ?)",
                        (user_id, name),
                    )

            def get_user(self, user_id: str):
                return self.fetch_one("SELECT * FROM users WHERE id = ?", (user_id,))

        db = UserDatabase(tmp_db_path)
        db.create_users_table()
        db.add_user("1", "Alice")

        user = db.get_user("1")
        assert user is not None
        assert user[1] == "Alice"


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_sql_raises(self, db_with_table):
        """Invalid SQL raises sqlite3.OperationalError."""
        with pytest.raises(sqlite3.OperationalError):
            db_with_table.fetch_one("SELECT * FROM nonexistent_table")

    def test_syntax_error_raises(self, db_with_table):
        """SQL syntax error raises sqlite3.OperationalError."""
        with pytest.raises(sqlite3.OperationalError):
            db_with_table.fetch_one("SELEC * FROM test_data")

    def test_type_error_in_transaction(self, db_with_table):
        """ProgrammingError from incorrect bindings in transaction is propagated."""
        with pytest.raises(sqlite3.ProgrammingError):
            with db_with_table.transaction() as conn:
                conn.execute(
                    "INSERT INTO test_data (id, name, value) VALUES (?, ?, ?)",
                    ("1", "Test"),  # Missing value parameter - raises ProgrammingError
                )
