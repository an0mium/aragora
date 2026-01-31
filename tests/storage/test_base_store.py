"""
Tests for aragora.storage.base_store - SQLiteStore base class.

Tests cover:
- SQLiteStore initialization and validation
- Schema management integration
- CRUD helper methods (exists, count, delete_by_id)
- WHERE clause validation (SQL injection prevention)
- Schema versioning and migrations
- safe_add_column integration
- vacuum and table info methods
- Transaction edge cases (rollback, nested transactions)
"""

from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.base_store import SQLiteStore
from aragora.storage.schema import SchemaManager, DatabaseManager


# ============================================================================
# Test Store Implementations
# ============================================================================


class SimpleTestStore(SQLiteStore):
    """Simple test store for basic functionality."""

    SCHEMA_NAME = "simple_test"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            value INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_items_name ON items(name);
    """

    def add_item(self, item_id: str, name: str, value: int = 0) -> None:
        """Add an item to the store."""
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO items (id, name, value) VALUES (?, ?, ?)",
                (item_id, name, value),
            )

    def get_item(self, item_id: str) -> tuple | None:
        """Get an item by ID."""
        return self.fetch_one("SELECT * FROM items WHERE id = ?", (item_id,))


class MigratableTestStore(SQLiteStore):
    """Test store with migrations."""

    SCHEMA_NAME = "migratable_test"
    SCHEMA_VERSION = 2

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS records (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL
        );
    """

    def register_migrations(self, manager: SchemaManager) -> None:
        """Register schema migrations."""
        manager.register_migration(
            from_version=1,
            to_version=2,
            sql="ALTER TABLE records ADD COLUMN description TEXT;",
            description="Add description field",
        )

    def add_record(self, record_id: str, title: str, description: str = "") -> None:
        """Add a record."""
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO records (id, title, description) VALUES (?, ?, ?)",
                (record_id, title, description),
            )


class PostInitTestStore(SQLiteStore):
    """Test store with post-init hook."""

    SCHEMA_NAME = "post_init_test"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS data (
            key TEXT PRIMARY KEY,
            value TEXT
        );
    """

    def __init__(self, db_path, **kwargs):
        self.post_init_called = False
        self.cached_data = {}
        super().__init__(db_path, **kwargs)

    def _post_init(self) -> None:
        """Post-initialization hook."""
        self.post_init_called = True
        # Load some cached data
        rows = self.fetch_all("SELECT key, value FROM data")
        self.cached_data = {row[0]: row[1] for row in rows}


class NoSchemaNameStore(SQLiteStore):
    """Invalid store - missing SCHEMA_NAME."""

    SCHEMA_VERSION = 1
    INITIAL_SCHEMA = "CREATE TABLE test (id INTEGER);"


class NoInitialSchemaStore(SQLiteStore):
    """Invalid store - missing INITIAL_SCHEMA."""

    SCHEMA_NAME = "no_initial"
    SCHEMA_VERSION = 1


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


# ============================================================================
# Test SQLiteStore Initialization
# ============================================================================


class TestSQLiteStoreInit:
    """Tests for SQLiteStore initialization."""

    def test_creates_database_file(self, tmp_db_path):
        """Store creates database file if it doesn't exist."""
        assert not tmp_db_path.exists()
        store = SimpleTestStore(tmp_db_path)
        assert tmp_db_path.exists()

    def test_creates_parent_directories(self, tmp_path):
        """Store creates parent directories if needed."""
        nested_path = tmp_path / "a" / "b" / "c" / "test.db"
        assert not nested_path.parent.exists()
        store = SimpleTestStore(nested_path)
        assert nested_path.parent.exists()
        assert nested_path.exists()

    def test_initializes_schema(self, tmp_db_path):
        """Store initializes schema on construction."""
        store = SimpleTestStore(tmp_db_path)

        # Verify table was created
        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
            )
            assert cursor.fetchone() is not None

        # Verify index was created
        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_items_name'"
            )
            assert cursor.fetchone() is not None

    def test_sets_schema_version(self, tmp_db_path):
        """Store sets schema version after initialization."""
        store = SimpleTestStore(tmp_db_path)
        assert store.get_schema_version() == 1

    def test_auto_init_false_skips_schema(self, tmp_db_path):
        """auto_init=False skips schema initialization."""
        store = SimpleTestStore(tmp_db_path, auto_init=False)

        # Table should not exist
        with store.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
            )
            assert cursor.fetchone() is None

    def test_missing_schema_name_raises(self, tmp_db_path):
        """Missing SCHEMA_NAME raises ValueError."""
        with pytest.raises(ValueError, match="must define SCHEMA_NAME"):
            NoSchemaNameStore(tmp_db_path)

    def test_missing_initial_schema_raises(self, tmp_db_path):
        """Missing INITIAL_SCHEMA raises ValueError."""
        with pytest.raises(ValueError, match="must define INITIAL_SCHEMA"):
            NoInitialSchemaStore(tmp_db_path)

    def test_custom_timeout(self, tmp_db_path):
        """Custom timeout is respected."""
        store = SimpleTestStore(tmp_db_path, timeout=60.0)
        assert store._timeout == 60.0

    def test_path_accepts_string(self, tmp_path):
        """Store accepts string path."""
        db_path = str(tmp_path / "test.db")
        store = SimpleTestStore(db_path)
        assert store.db_path == Path(db_path)

    def test_path_accepts_pathlib(self, tmp_db_path):
        """Store accepts Path object."""
        store = SimpleTestStore(tmp_db_path)
        assert store.db_path == tmp_db_path


# ============================================================================
# Test Schema Migrations
# ============================================================================


class TestSchemasMigrations:
    """Tests for schema migration functionality."""

    def test_runs_migrations(self, tmp_db_path):
        """Store runs registered migrations."""
        store = MigratableTestStore(tmp_db_path)

        # Verify migration was applied (description column exists)
        with store.connection() as conn:
            cursor = conn.execute("PRAGMA table_info(records)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "description" in columns

    def test_migration_updates_version(self, tmp_db_path):
        """Migrations update schema version."""
        store = MigratableTestStore(tmp_db_path)
        assert store.get_schema_version() == 2

    def test_reopening_existing_db(self, tmp_db_path):
        """Reopening existing database doesn't re-run migrations."""
        # Create store once
        store1 = MigratableTestStore(tmp_db_path)
        store1.add_record("1", "Test", "Description")

        # Create a new store instance
        store2 = MigratableTestStore(tmp_db_path)

        # Data should still exist
        row = store2.fetch_one("SELECT * FROM records WHERE id = ?", ("1",))
        assert row is not None
        assert row[1] == "Test"

    def test_get_schema_version_uninitialized(self, tmp_db_path):
        """get_schema_version returns 0 for uninitialized store."""
        store = SimpleTestStore(tmp_db_path, auto_init=False)
        # No schema versions table yet
        assert store.get_schema_version() == 0


# ============================================================================
# Test Post-Init Hook
# ============================================================================


class TestPostInit:
    """Tests for _post_init hook."""

    def test_post_init_called(self, tmp_db_path):
        """_post_init is called after schema initialization."""
        store = PostInitTestStore(tmp_db_path)
        assert store.post_init_called is True

    def test_post_init_can_access_data(self, tmp_db_path):
        """_post_init can access database data."""
        # First create with some data
        store1 = PostInitTestStore(tmp_db_path)
        with store1.connection() as conn:
            conn.execute("INSERT INTO data (key, value) VALUES ('foo', 'bar')")

        # Create new instance - post_init should load the data
        store2 = PostInitTestStore(tmp_db_path)
        assert store2.cached_data == {"foo": "bar"}


# ============================================================================
# Test CRUD Helpers
# ============================================================================


class TestCRUDHelpers:
    """Tests for CRUD helper methods."""

    @pytest.fixture
    def store_with_data(self, tmp_db_path):
        """Create a store with test data."""
        store = SimpleTestStore(tmp_db_path)
        store.add_item("1", "Item One", 100)
        store.add_item("2", "Item Two", 200)
        store.add_item("3", "Item Three", 300)
        return store

    def test_exists_returns_true(self, store_with_data):
        """exists returns True for existing record."""
        assert store_with_data.exists("items", "id", "1") is True

    def test_exists_returns_false(self, store_with_data):
        """exists returns False for non-existing record."""
        assert store_with_data.exists("items", "id", "999") is False

    def test_exists_invalid_table_name(self, store_with_data):
        """exists rejects invalid table name."""
        with pytest.raises(ValueError, match="Invalid table name"):
            store_with_data.exists("items;DROP TABLE", "id", "1")

    def test_exists_invalid_column_name(self, store_with_data):
        """exists rejects invalid column name."""
        with pytest.raises(ValueError, match="Invalid column name"):
            store_with_data.exists("items", "id;--", "1")

    def test_count_all_records(self, store_with_data):
        """count returns total record count."""
        assert store_with_data.count("items") == 3

    def test_count_with_where_clause(self, store_with_data):
        """count with WHERE clause filters results."""
        count = store_with_data.count("items", "value > ?", (100,))
        assert count == 2

    def test_count_invalid_table_name(self, store_with_data):
        """count rejects invalid table name."""
        with pytest.raises(ValueError, match="Invalid table name"):
            store_with_data.count("items;DROP")

    def test_delete_by_id_success(self, store_with_data):
        """delete_by_id deletes existing record."""
        assert store_with_data.delete_by_id("items", "id", "1") is True
        assert store_with_data.exists("items", "id", "1") is False
        assert store_with_data.count("items") == 2

    def test_delete_by_id_not_found(self, store_with_data):
        """delete_by_id returns False for non-existing record."""
        assert store_with_data.delete_by_id("items", "id", "999") is False
        assert store_with_data.count("items") == 3

    def test_delete_by_id_invalid_table(self, store_with_data):
        """delete_by_id rejects invalid table name."""
        with pytest.raises(ValueError, match="Invalid table name"):
            store_with_data.delete_by_id("items;DROP", "id", "1")

    def test_delete_by_id_invalid_column(self, store_with_data):
        """delete_by_id rejects invalid column name."""
        with pytest.raises(ValueError, match="Invalid column name"):
            store_with_data.delete_by_id("items", "id'--", "1")


# ============================================================================
# Test WHERE Clause Validation
# ============================================================================


class TestWhereClauseValidation:
    """Tests for WHERE clause SQL injection prevention."""

    @pytest.fixture
    def store(self, tmp_db_path):
        """Create a test store."""
        return SimpleTestStore(tmp_db_path)

    def test_rejects_single_quotes(self, store):
        """WHERE clause with single quotes is rejected."""
        with pytest.raises(ValueError, match="unsafe pattern"):
            store.count("items", "name = 'test'")

    def test_rejects_double_quotes(self, store):
        """WHERE clause with double quotes is rejected."""
        with pytest.raises(ValueError, match="unsafe pattern"):
            store.count("items", 'name = "test"')

    def test_rejects_semicolon(self, store):
        """WHERE clause with semicolon is rejected."""
        with pytest.raises(ValueError, match="unsafe pattern"):
            store.count("items", "1=1; DROP TABLE items")

    def test_rejects_comment_dashes(self, store):
        """WHERE clause with -- comment is rejected."""
        with pytest.raises(ValueError, match="unsafe pattern"):
            store.count("items", "id = 1 --")

    def test_rejects_block_comment_start(self, store):
        """WHERE clause with /* is rejected."""
        with pytest.raises(ValueError, match="unsafe pattern"):
            store.count("items", "id = 1 /*")

    def test_rejects_block_comment_end(self, store):
        """WHERE clause with */ is rejected."""
        with pytest.raises(ValueError, match="unsafe pattern"):
            store.count("items", "*/ id = 1")

    def test_allows_parameterized_queries(self, store):
        """Parameterized queries are allowed."""
        store.add_item("1", "Test")
        count = store.count("items", "name = ?", ("Test",))
        assert count == 1

    def test_allows_is_null(self, store):
        """IS NULL clause is allowed."""
        count = store.count("items", "value IS NULL")
        assert count >= 0

    def test_allows_is_not_null(self, store):
        """IS NOT NULL clause is allowed."""
        count = store.count("items", "id IS NOT NULL")
        assert count >= 0


# ============================================================================
# Test safe_add_column Integration
# ============================================================================


class TestSafeAddColumn:
    """Tests for safe_add_column method."""

    @pytest.fixture
    def store(self, tmp_db_path):
        """Create a test store."""
        return SimpleTestStore(tmp_db_path)

    def test_adds_new_column(self, store):
        """safe_add_column adds a new column."""
        store.safe_add_column("items", "category", "TEXT")

        with store.connection() as conn:
            cursor = conn.execute("PRAGMA table_info(items)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "category" in columns

    def test_adds_column_with_default(self, store):
        """safe_add_column adds column with default value."""
        store.safe_add_column("items", "status", "TEXT", default="'active'")

        # Add a new row
        store.add_item("test", "Test Item")

        # Check default was applied
        row = store.fetch_one("SELECT status FROM items WHERE id = ?", ("test",))
        assert row[0] == "active"

    def test_column_already_exists(self, store):
        """safe_add_column is idempotent for existing columns."""
        # Add column first time
        store.safe_add_column("items", "extra", "TEXT")

        # Should not raise, just return
        store.safe_add_column("items", "extra", "TEXT")

        # Verify only one column
        with store.connection() as conn:
            cursor = conn.execute("PRAGMA table_info(items)")
            columns = [row[1] for row in cursor.fetchall()]
            assert columns.count("extra") == 1


# ============================================================================
# Test Table Info
# ============================================================================


class TestGetTableInfo:
    """Tests for get_table_info method."""

    @pytest.fixture
    def store(self, tmp_db_path):
        """Create a test store."""
        return SimpleTestStore(tmp_db_path)

    def test_returns_column_info(self, store):
        """get_table_info returns column information."""
        info = store.get_table_info("items")

        assert len(info) >= 4  # id, name, value, created_at

        # Find the 'id' column
        id_col = next(c for c in info if c["name"] == "id")
        assert id_col["type"] == "TEXT"
        assert id_col["pk"] is True

        # Find the 'value' column
        value_col = next(c for c in info if c["name"] == "value")
        assert value_col["type"] == "INTEGER"
        assert value_col["default"] == "0"

    def test_invalid_table_name(self, store):
        """get_table_info rejects invalid table name."""
        with pytest.raises(ValueError, match="Invalid table name"):
            store.get_table_info("items;DROP TABLE items")


# ============================================================================
# Test Vacuum
# ============================================================================


class TestVacuum:
    """Tests for vacuum method."""

    def test_vacuum_succeeds(self, tmp_db_path):
        """vacuum completes successfully."""
        store = SimpleTestStore(tmp_db_path)

        # Add and delete some data
        for i in range(100):
            store.add_item(str(i), f"Item {i}")
        for i in range(100):
            store.delete_by_id("items", "id", str(i))

        # Vacuum should not raise
        store.vacuum()


# ============================================================================
# Test Transaction Edge Cases
# ============================================================================


class TestTransactionEdgeCases:
    """Tests for transaction edge cases."""

    @pytest.fixture
    def store(self, tmp_db_path):
        """Create a test store."""
        return SimpleTestStore(tmp_db_path)

    def test_connection_auto_commits(self, store):
        """connection() context manager auto-commits on success."""
        with store.connection() as conn:
            conn.execute(
                "INSERT INTO items (id, name, value) VALUES (?, ?, ?)",
                ("1", "Test", 100),
            )

        # Should be committed
        row = store.fetch_one("SELECT * FROM items WHERE id = ?", ("1",))
        assert row is not None

    def test_connection_rollback_on_exception(self, store):
        """connection() rolls back on exception."""
        with pytest.raises(RuntimeError):
            with store.connection() as conn:
                conn.execute(
                    "INSERT INTO items (id, name, value) VALUES (?, ?, ?)",
                    ("1", "Test", 100),
                )
                raise RuntimeError("Test error")

        # Should be rolled back
        row = store.fetch_one("SELECT * FROM items WHERE id = ?", ("1",))
        assert row is None

    def test_transaction_explicit_commit(self, store):
        """transaction() context manager explicitly commits."""
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO items (id, name, value) VALUES (?, ?, ?)",
                ("1", "Test", 100),
            )

        # Should be committed
        row = store.fetch_one("SELECT * FROM items WHERE id = ?", ("1",))
        assert row is not None

    def test_transaction_rollback_on_exception(self, store):
        """transaction() rolls back on exception."""
        with pytest.raises(RuntimeError):
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO items (id, name, value) VALUES (?, ?, ?)",
                    ("1", "Test", 100),
                )
                raise RuntimeError("Test error")

        # Should be rolled back
        row = store.fetch_one("SELECT * FROM items WHERE id = ?", ("1",))
        assert row is None

    def test_partial_transaction_rollback(self, store):
        """Partial transaction is fully rolled back on error."""
        with pytest.raises(RuntimeError):
            with store.transaction() as conn:
                conn.execute(
                    "INSERT INTO items (id, name, value) VALUES (?, ?, ?)",
                    ("1", "First", 100),
                )
                conn.execute(
                    "INSERT INTO items (id, name, value) VALUES (?, ?, ?)",
                    ("2", "Second", 200),
                )
                raise RuntimeError("Error after both inserts")

        # Both should be rolled back
        assert store.count("items") == 0


# ============================================================================
# Test Concurrent Access
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent database access."""

    def test_concurrent_reads(self, tmp_db_path):
        """Concurrent reads work correctly."""
        store = SimpleTestStore(tmp_db_path)
        for i in range(10):
            store.add_item(str(i), f"Item {i}", i * 10)

        results = []
        errors = []

        def reader(thread_id):
            try:
                for _ in range(5):
                    count = store.count("items")
                    results.append((thread_id, count))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 25
        assert all(count == 10 for _, count in results)

    def test_concurrent_writes(self, tmp_db_path):
        """Concurrent writes work correctly."""
        store = SimpleTestStore(tmp_db_path)
        errors = []

        def writer(thread_id):
            try:
                for i in range(10):
                    store.add_item(f"{thread_id}_{i}", f"Item from thread {thread_id}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert store.count("items") == 50


# ============================================================================
# Test Fetch Methods
# ============================================================================


class TestFetchMethods:
    """Tests for fetch_one and fetch_all methods."""

    @pytest.fixture
    def store_with_data(self, tmp_db_path):
        """Create store with test data."""
        store = SimpleTestStore(tmp_db_path)
        store.add_item("1", "Alpha", 100)
        store.add_item("2", "Beta", 200)
        store.add_item("3", "Gamma", 300)
        return store

    def test_fetch_one_returns_row(self, store_with_data):
        """fetch_one returns a single row."""
        row = store_with_data.fetch_one("SELECT * FROM items WHERE id = ?", ("1",))
        assert row is not None
        assert row[0] == "1"
        assert row[1] == "Alpha"

    def test_fetch_one_returns_none(self, store_with_data):
        """fetch_one returns None for no results."""
        row = store_with_data.fetch_one("SELECT * FROM items WHERE id = ?", ("999",))
        assert row is None

    def test_fetch_all_returns_rows(self, store_with_data):
        """fetch_all returns all matching rows."""
        rows = store_with_data.fetch_all("SELECT * FROM items ORDER BY id")
        assert len(rows) == 3
        assert rows[0][0] == "1"
        assert rows[1][0] == "2"
        assert rows[2][0] == "3"

    def test_fetch_all_returns_empty(self, store_with_data):
        """fetch_all returns empty list for no results."""
        rows = store_with_data.fetch_all("SELECT * FROM items WHERE value > ?", (1000,))
        assert rows == []


# ============================================================================
# Test execute_write and executemany
# ============================================================================


class TestExecuteMethods:
    """Tests for execute_write and executemany methods."""

    @pytest.fixture
    def store(self, tmp_db_path):
        """Create a test store."""
        return SimpleTestStore(tmp_db_path)

    def test_execute_write(self, store):
        """execute_write performs write operations."""
        store.execute_write(
            "INSERT INTO items (id, name, value) VALUES (?, ?, ?)",
            ("1", "Test", 100),
        )

        row = store.fetch_one("SELECT * FROM items WHERE id = ?", ("1",))
        assert row is not None

    def test_executemany(self, store):
        """executemany inserts multiple rows."""
        params = [
            ("1", "First", 100),
            ("2", "Second", 200),
            ("3", "Third", 300),
        ]
        store.executemany(
            "INSERT INTO items (id, name, value) VALUES (?, ?, ?)",
            params,
        )

        assert store.count("items") == 3


# ============================================================================
# Test repr
# ============================================================================


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr(self, tmp_db_path):
        """__repr__ returns informative string."""
        store = SimpleTestStore(tmp_db_path)
        repr_str = repr(store)
        assert "SimpleTestStore" in repr_str
        assert str(tmp_db_path) in repr_str
