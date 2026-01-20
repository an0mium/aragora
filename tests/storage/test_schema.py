"""
Tests for aragora.storage.schema module.

Covers:
- SQL identifier validation
- Column type validation
- Default value validation
- WAL connection creation
- Migration dataclass
- SchemaManager (versioning, migrations)
- safe_add_column
- DatabaseManager (singleton, connections, transactions)
- create_performance_indexes
- analyze_tables
- ConnectionPool (acquire, release, context manager)
"""

import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.storage.schema import (
    VALID_COLUMN_TYPES,
    ConnectionPool,
    DatabaseManager,
    Migration,
    PERFORMANCE_INDEXES,
    SchemaManager,
    _validate_column_type,
    _validate_default_value,
    _validate_sql_identifier,
    analyze_tables,
    create_performance_indexes,
    get_wal_connection,
    safe_add_column,
)


# ============================================================================
# Validation Functions
# ============================================================================


class TestValidateSqlIdentifier:
    """Tests for _validate_sql_identifier."""

    def test_valid_simple_name(self):
        """Valid simple identifiers."""
        assert _validate_sql_identifier("users") is True
        assert _validate_sql_identifier("my_table") is True
        assert _validate_sql_identifier("Table1") is True

    def test_valid_underscore_prefix(self):
        """Identifiers can start with underscore."""
        assert _validate_sql_identifier("_internal") is True
        assert _validate_sql_identifier("_schema_versions") is True

    def test_invalid_empty(self):
        """Empty string is invalid."""
        assert _validate_sql_identifier("") is False

    def test_invalid_starts_with_number(self):
        """Cannot start with a number."""
        assert _validate_sql_identifier("1table") is False
        assert _validate_sql_identifier("123") is False

    def test_invalid_special_chars(self):
        """Special characters are not allowed."""
        assert _validate_sql_identifier("my-table") is False
        assert _validate_sql_identifier("my.table") is False
        assert _validate_sql_identifier("table;drop") is False
        assert _validate_sql_identifier("table'name") is False

    def test_invalid_too_long(self):
        """Names over 128 characters are invalid."""
        assert _validate_sql_identifier("a" * 128) is True  # Exactly 128 is ok
        assert _validate_sql_identifier("a" * 129) is False  # 129 is too long

    def test_sql_injection_attempts(self):
        """SQL injection patterns should fail validation."""
        assert _validate_sql_identifier("users; DROP TABLE users;--") is False
        assert _validate_sql_identifier("users' OR '1'='1") is False
        assert _validate_sql_identifier("users/**/") is False


class TestValidateColumnType:
    """Tests for _validate_column_type."""

    def test_valid_types(self):
        """All valid column types."""
        for col_type in VALID_COLUMN_TYPES:
            assert _validate_column_type(col_type) is True

    def test_valid_types_lowercase(self):
        """Types are case-insensitive."""
        assert _validate_column_type("text") is True
        assert _validate_column_type("integer") is True
        assert _validate_column_type("Real") is True

    def test_valid_with_length(self):
        """Types with length specifiers."""
        assert _validate_column_type("VARCHAR(255)") is True
        assert _validate_column_type("CHAR(10)") is True

    def test_invalid_types(self):
        """Invalid column types."""
        assert _validate_column_type("MONEY") is False
        assert _validate_column_type("XML") is False
        assert _validate_column_type("DROP TABLE") is False


class TestValidateDefaultValue:
    """Tests for _validate_default_value."""

    def test_null_values(self):
        """NULL is allowed."""
        assert _validate_default_value(None) is True
        assert _validate_default_value("NULL") is True
        assert _validate_default_value("null") is True

    def test_numeric_values(self):
        """Numeric literals are allowed."""
        assert _validate_default_value("0") is True
        assert _validate_default_value("42") is True
        assert _validate_default_value("-1") is True
        assert _validate_default_value("3.14") is True
        assert _validate_default_value("-0.5") is True

    def test_sql_functions(self):
        """SQL timestamp functions are allowed."""
        assert _validate_default_value("CURRENT_TIMESTAMP") is True
        assert _validate_default_value("CURRENT_DATE") is True
        assert _validate_default_value("CURRENT_TIME") is True

    def test_quoted_strings(self):
        """Single-quoted strings are allowed."""
        assert _validate_default_value("'hello'") is True
        assert _validate_default_value("''") is True

    def test_invalid_values(self):
        """Invalid default values."""
        assert _validate_default_value("SELECT 1") is False
        assert _validate_default_value("1; DROP TABLE") is False
        assert _validate_default_value("'test'; --") is False
        # Embedded quotes not allowed
        assert _validate_default_value("'it''s'") is False


# ============================================================================
# WAL Connection
# ============================================================================


class TestGetWalConnection:
    """Tests for get_wal_connection."""

    def test_creates_connection(self, tmp_path):
        """Creates a working connection."""
        db_path = tmp_path / "test.db"
        conn = get_wal_connection(db_path)
        try:
            assert conn is not None
            conn.execute("SELECT 1")
        finally:
            conn.close()

    def test_wal_mode_enabled(self, tmp_path):
        """WAL mode is enabled."""
        db_path = tmp_path / "test.db"
        conn = get_wal_connection(db_path)
        try:
            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"
        finally:
            conn.close()

    def test_row_factory_set(self, tmp_path):
        """Row factory is sqlite3.Row for dict-style access."""
        db_path = tmp_path / "test.db"
        conn = get_wal_connection(db_path)
        try:
            assert conn.row_factory == sqlite3.Row
            conn.execute("CREATE TABLE test (name TEXT)")
            conn.execute("INSERT INTO test VALUES ('foo')")
            row = conn.execute("SELECT name FROM test").fetchone()
            assert row["name"] == "foo"
        finally:
            conn.close()

    def test_custom_timeout(self, tmp_path):
        """Custom timeout is respected."""
        db_path = tmp_path / "test.db"
        conn = get_wal_connection(db_path, timeout=60.0)
        try:
            # Connection should be created with specified timeout
            assert conn is not None
        finally:
            conn.close()


# ============================================================================
# Migration
# ============================================================================


class TestMigration:
    """Tests for Migration dataclass."""

    def test_apply_sql(self, tmp_path):
        """Applying SQL migration."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE test (id INTEGER)")

            migration = Migration(
                from_version=1,
                to_version=2,
                sql="ALTER TABLE test ADD COLUMN name TEXT",
            )
            migration.apply(conn)

            # Verify column was added
            cursor = conn.execute("PRAGMA table_info(test)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "name" in columns
        finally:
            conn.close()

    def test_apply_function(self, tmp_path):
        """Applying function migration."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE test (id INTEGER)")

            called = []

            def migrate_fn(c):
                c.execute("ALTER TABLE test ADD COLUMN value TEXT")
                called.append(True)

            migration = Migration(
                from_version=1,
                to_version=2,
                function=migrate_fn,
            )
            migration.apply(conn)

            assert called == [True]
            cursor = conn.execute("PRAGMA table_info(test)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "value" in columns
        finally:
            conn.close()

    def test_apply_no_sql_or_function(self, tmp_path):
        """Migration without sql or function raises error."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            migration = Migration(from_version=1, to_version=2)
            with pytest.raises(ValueError, match="must have either sql or function"):
                migration.apply(conn)
        finally:
            conn.close()


# ============================================================================
# SchemaManager
# ============================================================================


class TestSchemaManager:
    """Tests for SchemaManager."""

    @pytest.fixture
    def db_conn(self, tmp_path):
        """Create a database connection."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        yield conn
        conn.close()

    def test_creates_version_table(self, db_conn):
        """Version table is created on init."""
        manager = SchemaManager(db_conn, "test_module")

        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_versions'"
        )
        assert cursor.fetchone() is not None

    def test_get_version_new_module(self, db_conn):
        """New module starts at version 0."""
        manager = SchemaManager(db_conn, "new_module")
        assert manager.get_version() == 0

    def test_set_and_get_version(self, db_conn):
        """Setting and getting version."""
        manager = SchemaManager(db_conn, "test_module")
        manager.set_version(5)
        assert manager.get_version() == 5

    def test_register_migration(self, db_conn):
        """Registering migrations."""
        manager = SchemaManager(db_conn, "test_module", current_version=3)

        manager.register_migration(1, 2, sql="SELECT 1")
        manager.register_migration(2, 3, sql="SELECT 2")

        assert len(manager.migrations) == 2
        assert manager.migrations[0].from_version == 1
        assert manager.migrations[1].from_version == 2

    def test_register_migration_sorts(self, db_conn):
        """Migrations are sorted by from_version."""
        manager = SchemaManager(db_conn, "test_module")

        manager.register_migration(3, 4, sql="SELECT 3")
        manager.register_migration(1, 2, sql="SELECT 1")
        manager.register_migration(2, 3, sql="SELECT 2")

        versions = [m.from_version for m in manager.migrations]
        assert versions == [1, 2, 3]

    def test_get_pending_migrations(self, db_conn):
        """Getting pending migrations."""
        manager = SchemaManager(db_conn, "test_module", current_version=3)
        manager.set_version(1)

        manager.register_migration(1, 2, sql="SELECT 1")
        manager.register_migration(2, 3, sql="SELECT 2")
        manager.register_migration(3, 4, sql="SELECT 3")  # Beyond target version

        pending = manager.get_pending_migrations()
        assert len(pending) == 2
        assert pending[0].to_version == 2
        assert pending[1].to_version == 3

    def test_ensure_schema_initial(self, db_conn):
        """Creating initial schema."""
        manager = SchemaManager(db_conn, "test_module", current_version=1)

        result = manager.ensure_schema(initial_schema="CREATE TABLE users (id INTEGER PRIMARY KEY)")

        assert result is True
        assert manager.get_version() == 1

        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        assert cursor.fetchone() is not None

    def test_ensure_schema_runs_migrations(self, db_conn):
        """Running pending migrations."""
        manager = SchemaManager(db_conn, "test_module", current_version=2)

        # Register migration BEFORE calling ensure_schema
        manager.register_migration(1, 2, sql="ALTER TABLE test ADD COLUMN name TEXT")

        # Create initial schema and run migrations
        result = manager.ensure_schema(initial_schema="CREATE TABLE test (id INTEGER)")
        assert result is True
        assert manager.get_version() == 2

        # Verify migration was applied
        cursor = db_conn.execute("PRAGMA table_info(test)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "name" in columns

    def test_ensure_schema_already_current(self, db_conn):
        """Schema already at current version."""
        manager = SchemaManager(db_conn, "test_module", current_version=1)
        manager.ensure_schema(initial_schema="CREATE TABLE test (id INTEGER)")

        # Second call should return False (no changes)
        result = manager.ensure_schema()
        assert result is False

    def test_ensure_schema_newer_version_warns(self, db_conn):
        """Database newer than code version logs warning."""
        manager = SchemaManager(db_conn, "test_module", current_version=1)
        manager.set_version(5)  # Pretend DB is at version 5

        result = manager.ensure_schema()
        assert result is False

    def test_validate_schema_all_present(self, db_conn):
        """Validating schema when all tables exist."""
        manager = SchemaManager(db_conn, "test_module")
        db_conn.execute("CREATE TABLE users (id INTEGER)")
        db_conn.execute("CREATE TABLE posts (id INTEGER)")

        result = manager.validate_schema(["users", "posts"])
        assert result["valid"] is True
        assert result["missing"] == []

    def test_validate_schema_missing_tables(self, db_conn):
        """Validating schema with missing tables."""
        manager = SchemaManager(db_conn, "test_module")
        db_conn.execute("CREATE TABLE users (id INTEGER)")

        result = manager.validate_schema(["users", "posts", "comments"])
        assert result["valid"] is False
        assert set(result["missing"]) == {"posts", "comments"}


# ============================================================================
# safe_add_column
# ============================================================================


class TestSafeAddColumn:
    """Tests for safe_add_column."""

    @pytest.fixture
    def db_conn(self, tmp_path):
        """Create a database connection with a test table."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        conn.commit()
        yield conn
        conn.close()

    def test_adds_column(self, db_conn):
        """Successfully adds a column."""
        result = safe_add_column(db_conn, "test", "name", "TEXT")

        assert result is True

        cursor = db_conn.execute("PRAGMA table_info(test)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "name" in columns

    def test_adds_column_with_default(self, db_conn):
        """Adds column with default value."""
        result = safe_add_column(db_conn, "test", "status", "TEXT", default="'active'")

        assert result is True

        # Insert a row and verify default
        db_conn.execute("INSERT INTO test (id) VALUES (1)")
        row = db_conn.execute("SELECT status FROM test WHERE id = 1").fetchone()
        assert row[0] == "active"

    def test_column_already_exists(self, db_conn):
        """Returns False if column already exists."""
        safe_add_column(db_conn, "test", "name", "TEXT")
        result = safe_add_column(db_conn, "test", "name", "TEXT")

        assert result is False

    def test_invalid_table_name(self, db_conn):
        """Rejects invalid table name."""
        with pytest.raises(ValueError, match="Invalid table name"):
            safe_add_column(db_conn, "test;DROP TABLE", "name", "TEXT")

    def test_invalid_column_name(self, db_conn):
        """Rejects invalid column name."""
        with pytest.raises(ValueError, match="Invalid column name"):
            safe_add_column(db_conn, "test", "name;--", "TEXT")

    def test_invalid_column_type(self, db_conn):
        """Rejects invalid column type."""
        with pytest.raises(ValueError, match="Invalid column type"):
            safe_add_column(db_conn, "test", "name", "MONEY")

    def test_invalid_default_value(self, db_conn):
        """Rejects invalid default value."""
        with pytest.raises(ValueError, match="Invalid default value"):
            safe_add_column(db_conn, "test", "name", "TEXT", default="1; DROP TABLE test")


# ============================================================================
# DatabaseManager
# ============================================================================


class TestDatabaseManager:
    """Tests for DatabaseManager."""

    @pytest.fixture(autouse=True)
    def cleanup_instances(self):
        """Clear instances before and after each test."""
        DatabaseManager.clear_instances()
        yield
        DatabaseManager.clear_instances()

    def test_singleton_pattern(self, tmp_path):
        """Same path returns same instance."""
        db_path = tmp_path / "test.db"

        manager1 = DatabaseManager.get_instance(db_path)
        manager2 = DatabaseManager.get_instance(db_path)

        assert manager1 is manager2

    def test_different_paths_different_instances(self, tmp_path):
        """Different paths return different instances."""
        db_path1 = tmp_path / "test1.db"
        db_path2 = tmp_path / "test2.db"

        manager1 = DatabaseManager.get_instance(db_path1)
        manager2 = DatabaseManager.get_instance(db_path2)

        assert manager1 is not manager2

    def test_get_connection(self, tmp_path):
        """Getting a connection."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        conn = manager.get_connection()
        assert conn is not None
        conn.execute("SELECT 1")

    def test_connection_context_manager(self, tmp_path):
        """Using connection context manager."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1)")

        # Verify commit happened
        with manager.connection() as conn:
            row = conn.execute("SELECT id FROM test").fetchone()
            assert row[0] == 1

    def test_connection_rollback_on_error(self, tmp_path):
        """Rollback on error in context manager."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")

        with pytest.raises(RuntimeError):
            with manager.connection() as conn:
                conn.execute("INSERT INTO test VALUES (1)")
                raise RuntimeError("Test error")

        # Verify rollback happened
        with manager.connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM test").fetchone()[0]
            assert count == 0

    def test_transaction_context_manager(self, tmp_path):
        """Using transaction context manager."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.transaction() as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1)")

        with manager.connection() as conn:
            row = conn.execute("SELECT id FROM test").fetchone()
            assert row[0] == 1

    def test_fresh_connection(self, tmp_path):
        """Using fresh connection context manager."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        with manager.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER)")

        with manager.fresh_connection() as conn:
            conn.execute("INSERT INTO test VALUES (1)")

        with manager.connection() as conn:
            row = conn.execute("SELECT id FROM test").fetchone()
            assert row[0] == 1

    def test_execute_method(self, tmp_path):
        """Using execute method."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        manager.execute("CREATE TABLE test (id INTEGER)")
        cursor = manager.execute("SELECT 1 as num")
        row = cursor.fetchone()
        assert row[0] == 1

    def test_fetch_one(self, tmp_path):
        """Using fetch_one method."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        manager.execute("CREATE TABLE test (id INTEGER, name TEXT)")
        manager.execute("INSERT INTO test VALUES (1, 'foo')")
        manager.get_connection().commit()

        row = manager.fetch_one("SELECT name FROM test WHERE id = ?", (1,))
        assert row[0] == "foo"

    def test_fetch_all(self, tmp_path):
        """Using fetch_all method."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        manager.execute("CREATE TABLE test (id INTEGER)")
        manager.execute("INSERT INTO test VALUES (1), (2), (3)")
        manager.get_connection().commit()

        rows = manager.fetch_all("SELECT id FROM test ORDER BY id")
        assert len(rows) == 3
        assert [r[0] for r in rows] == [1, 2, 3]

    def test_fetch_many(self, tmp_path):
        """Using fetch_many method."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        manager.execute("CREATE TABLE test (id INTEGER)")
        manager.execute("INSERT INTO test VALUES (1), (2), (3), (4), (5)")
        manager.get_connection().commit()

        rows = manager.fetch_many("SELECT id FROM test ORDER BY id", size=2)
        assert len(rows) == 2

    def test_pool_stats(self, tmp_path):
        """Getting pool statistics."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)

        # Use fresh_connection to trigger pool activity
        with manager.fresh_connection() as conn:
            conn.execute("SELECT 1")

        stats = manager.pool_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "returns" in stats
        assert "pool_size" in stats
        assert "max_pool_size" in stats

    def test_close(self, tmp_path):
        """Closing manager."""
        db_path = tmp_path / "test.db"
        manager = DatabaseManager.get_instance(db_path)
        manager.get_connection()  # Open a connection

        manager.close()
        # Connection should be None after close
        assert manager._conn is None


# ============================================================================
# create_performance_indexes
# ============================================================================


class TestCreatePerformanceIndexes:
    """Tests for create_performance_indexes."""

    @pytest.fixture
    def db_conn(self, tmp_path):
        """Create a database connection with test tables."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        # Create tables that match PERFORMANCE_INDEXES
        conn.execute(
            "CREATE TABLE memory_store (id INTEGER, agent_name TEXT, debate_id TEXT, timestamp TEXT)"
        )
        conn.execute(
            "CREATE TABLE votes (id INTEGER, agent_name TEXT, debate_id TEXT, round_num INTEGER)"
        )
        conn.commit()
        yield conn
        conn.close()

    def test_creates_indexes(self, db_conn):
        """Creates indexes on existing tables."""
        result = create_performance_indexes(db_conn, tables_to_index=["memory_store", "votes"])

        assert len(result["created"]) > 0
        assert "idx_memory_agent_debate" in result["created"]

    def test_skips_existing_indexes(self, db_conn):
        """Skips indexes that already exist."""
        # Create index first time
        result1 = create_performance_indexes(db_conn, tables_to_index=["memory_store"])
        created_first = result1["created"]

        # Try to create again
        result2 = create_performance_indexes(db_conn, tables_to_index=["memory_store"])

        # Should be in skipped, not created
        for idx in created_first:
            assert idx in result2["skipped"]
        assert len(result2["created"]) == 0

    def test_errors_for_missing_tables(self, db_conn):
        """Records errors for tables that don't exist."""
        # Call without filter - let it try to index tables that don't exist
        # (like 'continuum_memory', 'debates', etc. from PERFORMANCE_INDEXES)
        result = create_performance_indexes(db_conn)

        # Should have errors for missing tables
        assert len(result["errors"]) > 0
        assert any("does not exist" in e for e in result["errors"])

    def test_filters_by_tables_to_index(self, db_conn):
        """Only indexes specified tables."""
        result = create_performance_indexes(db_conn, tables_to_index=["votes"])

        # Should only have votes indexes
        for idx in result["created"]:
            assert "votes" in idx or "vote" in idx


# ============================================================================
# analyze_tables
# ============================================================================


class TestAnalyzeTables:
    """Tests for analyze_tables."""

    def test_runs_analyze(self, tmp_path):
        """ANALYZE command runs successfully."""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.execute("INSERT INTO test VALUES (1), (2), (3)")
            conn.commit()

            # Should not raise
            analyze_tables(conn)
        finally:
            conn.close()


# ============================================================================
# ConnectionPool
# ============================================================================


class TestConnectionPool:
    """Tests for ConnectionPool."""

    def test_acquire_and_release(self, tmp_path):
        """Acquiring and releasing connections."""
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path, max_connections=5)
        try:
            conn = pool.acquire()
            assert conn is not None
            conn.execute("SELECT 1")
            pool.release(conn)

            stats = pool.stats()
            assert stats["idle"] == 1
            assert stats["active"] == 0
        finally:
            pool.close()

    def test_connection_context_manager(self, tmp_path):
        """Using connection context manager."""
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path, max_connections=5)
        try:
            with pool.connection() as conn:
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.execute("INSERT INTO test VALUES (1)")

            # Verify committed
            with pool.connection() as conn:
                row = conn.execute("SELECT id FROM test").fetchone()
                assert row[0] == 1
        finally:
            pool.close()

    def test_connection_reuse(self, tmp_path):
        """Connections are reused from pool."""
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path, max_connections=5)
        try:
            # First acquire/release
            conn1 = pool.acquire()
            conn1_id = id(conn1)
            pool.release(conn1)

            # Second acquire should reuse
            conn2 = pool.acquire()
            conn2_id = id(conn2)
            pool.release(conn2)

            # Same connection object
            assert conn1_id == conn2_id
        finally:
            pool.close()

    def test_max_connections_limit(self, tmp_path):
        """Pool respects max_connections limit."""
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path, max_connections=2, timeout=1.0)
        try:
            conn1 = pool.acquire()
            conn2 = pool.acquire()

            # Third acquire should timeout since max is 2
            with pytest.raises(TimeoutError):
                pool.acquire(timeout=0.5)

            pool.release(conn1)
            pool.release(conn2)
        finally:
            pool.close()

    def test_pool_stats(self, tmp_path):
        """Getting pool statistics."""
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path, max_connections=5)
        try:
            conn = pool.acquire()
            stats = pool.stats()
            assert stats["active"] == 1
            assert stats["idle"] == 0
            assert stats["max"] == 5

            pool.release(conn)
            stats = pool.stats()
            assert stats["active"] == 0
            assert stats["idle"] == 1
        finally:
            pool.close()

    def test_concurrent_access(self, tmp_path):
        """Thread-safe concurrent access."""
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path, max_connections=3)

        # Create table first
        with pool.connection() as conn:
            conn.execute("CREATE TABLE test (id INTEGER, thread_id TEXT)")

        errors = []
        results = []

        def worker(thread_num):
            try:
                for _ in range(3):
                    with pool.connection() as conn:
                        conn.execute(
                            "INSERT INTO test VALUES (?, ?)",
                            (thread_num, threading.current_thread().name),
                        )
                        results.append(thread_num)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(results) == 15  # 5 threads * 3 iterations

        pool.close()

    def test_close_pool(self, tmp_path):
        """Closing pool prevents new acquisitions."""
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path)

        conn = pool.acquire()
        pool.release(conn)
        pool.close()

        # Importing DatabaseError for the proper exception type
        from aragora.exceptions import DatabaseError

        with pytest.raises(DatabaseError, match="closed"):
            pool.acquire()

    def test_release_connection_not_from_pool(self, tmp_path):
        """Releasing connection not from pool is handled gracefully."""
        db_path = tmp_path / "test.db"
        pool = ConnectionPool(db_path)
        try:
            # Create external connection
            external_conn = sqlite3.connect(db_path)

            # Should not raise, just log warning
            pool.release(external_conn)

            external_conn.close()
        finally:
            pool.close()


# ============================================================================
# Integration Tests
# ============================================================================


class TestSchemaIntegration:
    """Integration tests combining multiple components."""

    def test_full_migration_workflow(self, tmp_path):
        """Complete migration workflow."""
        db_path = tmp_path / "test.db"

        # Initial setup
        manager = DatabaseManager.get_instance(db_path)
        with manager.connection() as conn:
            schema_mgr = SchemaManager(conn, "app", current_version=2)

            # Register migration BEFORE ensure_schema
            schema_mgr.register_migration(
                1,
                2,
                sql="ALTER TABLE users ADD COLUMN email TEXT",
                description="Add email column",
            )

            # Create initial schema and run migrations in one call
            schema_mgr.ensure_schema(
                initial_schema="""
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY,
                        name TEXT NOT NULL
                    )
                """
            )

            # Verify
            assert schema_mgr.get_version() == 2
            cursor = conn.execute("PRAGMA table_info(users)")
            columns = {row[1] for row in cursor.fetchall()}
            assert "email" in columns

        DatabaseManager.clear_instances()

    def test_safe_add_column_with_schema_manager(self, tmp_path):
        """Using safe_add_column with SchemaManager."""
        db_path = tmp_path / "test.db"

        manager = DatabaseManager.get_instance(db_path)
        with manager.connection() as conn:
            conn.execute("CREATE TABLE settings (key TEXT PRIMARY KEY)")
            conn.commit()

            # Use safe_add_column for optional migration
            added = safe_add_column(conn, "settings", "value", "TEXT", default="''")
            assert added is True

            # Idempotent - second call should return False
            added = safe_add_column(conn, "settings", "value", "TEXT")
            assert added is False

        DatabaseManager.clear_instances()
