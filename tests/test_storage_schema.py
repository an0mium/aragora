"""
Tests for storage/schema.py - Schema versioning and database management.

Tests cover:
- SQL injection prevention (identifier, column type, default validation)
- SchemaManager (migrations, version tracking)
- safe_add_column function
- DatabaseManager (singleton, connection pooling)
- ConnectionPool (multi-threaded access)
"""

import pytest
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from aragora.storage.schema import (
    _validate_sql_identifier,
    _validate_column_type,
    _validate_default_value,
    VALID_COLUMN_TYPES,
    get_wal_connection,
    Migration,
    SchemaManager,
    safe_add_column,
    DatabaseManager,
    ConnectionPool,
    create_performance_indexes,
    analyze_tables,
    PERFORMANCE_INDEXES,
)


# ============================================================================
# SQL Injection Prevention Tests
# ============================================================================

class TestValidateSqlIdentifier:
    """Tests for SQL identifier validation to prevent injection."""

    def test_valid_identifiers(self):
        """Valid SQL identifiers should pass."""
        valid = [
            "users",
            "my_table",
            "_private",
            "Table123",
            "a",
            "A",
            "_",
        ]
        for name in valid:
            assert _validate_sql_identifier(name), f"Should be valid: {name}"

    def test_invalid_identifiers(self):
        """Invalid identifiers should be rejected."""
        invalid = [
            "",  # Empty
            "123table",  # Starts with number
            "user-name",  # Contains hyphen
            "table.name",  # Contains dot
            "my table",  # Contains space
            "table;DROP TABLE users;--",  # SQL injection attempt
            "table' OR '1'='1",  # SQL injection attempt
            "a" * 129,  # Too long
            None,  # None type
        ]
        for name in invalid:
            if name is not None:
                assert not _validate_sql_identifier(name), f"Should be invalid: {name}"

    def test_max_length_boundary(self):
        """Test 128 char boundary."""
        assert _validate_sql_identifier("a" * 128)  # Exactly 128
        assert not _validate_sql_identifier("a" * 129)  # 129 chars


class TestValidateColumnType:
    """Tests for column type validation."""

    def test_valid_column_types(self):
        """All whitelisted column types should pass."""
        for col_type in VALID_COLUMN_TYPES:
            assert _validate_column_type(col_type), f"Should be valid: {col_type}"

    def test_case_insensitive(self):
        """Column types should be case-insensitive."""
        assert _validate_column_type("text")
        assert _validate_column_type("TEXT")
        assert _validate_column_type("Text")

    def test_varchar_with_length(self):
        """VARCHAR(n) should be valid."""
        assert _validate_column_type("VARCHAR(255)")
        assert _validate_column_type("varchar(100)")
        assert _validate_column_type("CHAR(10)")

    def test_invalid_column_types(self):
        """Invalid column types should be rejected."""
        invalid = [
            "INVALID",
            "DROP TABLE",
            "TEXT; DROP TABLE users;--",
            "",
        ]
        for col_type in invalid:
            assert not _validate_column_type(col_type), f"Should be invalid: {col_type}"


class TestValidateDefaultValue:
    """Tests for default value validation."""

    def test_null_values(self):
        """NULL should be valid."""
        assert _validate_default_value(None)
        assert _validate_default_value("NULL")
        assert _validate_default_value("null")

    def test_numeric_values(self):
        """Numeric literals should be valid."""
        valid = ["0", "123", "-456", "3.14", "-0.5", "100"]
        for val in valid:
            assert _validate_default_value(val), f"Should be valid: {val}"

    def test_sql_functions(self):
        """Common SQL functions should be valid."""
        assert _validate_default_value("CURRENT_TIMESTAMP")
        assert _validate_default_value("CURRENT_DATE")
        assert _validate_default_value("CURRENT_TIME")
        assert _validate_default_value("current_timestamp")

    def test_quoted_strings(self):
        """Single-quoted strings should be valid."""
        assert _validate_default_value("'default'")
        assert _validate_default_value("'hello world'")
        assert _validate_default_value("''")  # Empty string

    def test_invalid_defaults(self):
        """Invalid defaults should be rejected."""
        invalid = [
            "DROP TABLE users",
            "'; DROP TABLE users;--",
            "'test",  # Unclosed quote
            "test'",  # Unclosed quote
            "NOW()",  # Not in whitelist
            "SELECT 1",
        ]
        for val in invalid:
            assert not _validate_default_value(val), f"Should be invalid: {val}"


# ============================================================================
# Migration Tests
# ============================================================================

class TestMigration:
    """Tests for Migration dataclass."""

    def test_migration_with_sql(self):
        """Migration can apply SQL script."""
        conn = sqlite3.connect(":memory:")
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

    def test_migration_with_function(self):
        """Migration can apply Python function."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER)")

        def add_column(c):
            c.execute("ALTER TABLE test ADD COLUMN name TEXT")

        migration = Migration(
            from_version=1,
            to_version=2,
            function=add_column,
        )
        migration.apply(conn)

        cursor = conn.execute("PRAGMA table_info(test)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "name" in columns

    def test_migration_requires_sql_or_function(self):
        """Migration without sql or function should raise."""
        conn = sqlite3.connect(":memory:")
        migration = Migration(from_version=1, to_version=2)

        with pytest.raises(ValueError, match="must have either sql or function"):
            migration.apply(conn)


# ============================================================================
# SchemaManager Tests
# ============================================================================

class TestSchemaManager:
    """Tests for SchemaManager."""

    def test_initial_version_is_zero(self):
        """Fresh database should have version 0."""
        conn = sqlite3.connect(":memory:")
        manager = SchemaManager(conn, "test_module", current_version=1)

        assert manager.get_version() == 0

    def test_ensure_schema_creates_initial(self):
        """ensure_schema with initial_schema creates tables."""
        conn = sqlite3.connect(":memory:")
        manager = SchemaManager(conn, "test_module", current_version=1)

        initial = "CREATE TABLE test (id INTEGER PRIMARY KEY)"
        result = manager.ensure_schema(initial_schema=initial)

        assert result is True
        assert manager.get_version() == 1

        # Verify table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test'")
        assert cursor.fetchone() is not None

    def test_ensure_schema_idempotent(self):
        """Calling ensure_schema twice should not fail."""
        conn = sqlite3.connect(":memory:")
        manager = SchemaManager(conn, "test_module", current_version=1)

        initial = "CREATE TABLE test (id INTEGER PRIMARY KEY)"
        manager.ensure_schema(initial_schema=initial)
        result = manager.ensure_schema(initial_schema=initial)

        assert result is False  # Already up to date

    def test_migration_runs_in_order(self):
        """Migrations should run in version order."""
        conn = sqlite3.connect(":memory:")
        manager = SchemaManager(conn, "test_module", current_version=3)

        initial = "CREATE TABLE test (id INTEGER PRIMARY KEY)"
        manager.register_migration(1, 2, sql="ALTER TABLE test ADD COLUMN v2 TEXT")
        manager.register_migration(2, 3, sql="ALTER TABLE test ADD COLUMN v3 TEXT")

        manager.ensure_schema(initial_schema=initial)

        assert manager.get_version() == 3

        # Verify both columns exist
        cursor = conn.execute("PRAGMA table_info(test)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "v2" in columns
        assert "v3" in columns

    def test_migration_rollback_on_error(self):
        """Failed migration should rollback."""
        conn = sqlite3.connect(":memory:")
        manager = SchemaManager(conn, "test_module", current_version=2)

        initial = "CREATE TABLE test (id INTEGER PRIMARY KEY)"
        manager.register_migration(1, 2, sql="INVALID SQL SYNTAX")

        # ensure_schema creates initial schema (v1) then tries migration (v1->v2) which fails
        with pytest.raises(sqlite3.OperationalError):
            manager.ensure_schema(initial_schema=initial)

        # Version should still be 1 (migration failed, but initial schema succeeded)
        assert manager.get_version() == 1

    def test_validate_schema(self):
        """validate_schema should check table existence."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE users (id INTEGER)")
        conn.execute("CREATE TABLE orders (id INTEGER)")

        manager = SchemaManager(conn, "test_module", current_version=1)

        result = manager.validate_schema(["users", "orders"])
        assert result["valid"] is True
        assert result["missing"] == []

        result = manager.validate_schema(["users", "products"])
        assert result["valid"] is False
        assert "products" in result["missing"]


# ============================================================================
# safe_add_column Tests
# ============================================================================

class TestSafeAddColumn:
    """Tests for safe_add_column function."""

    def test_adds_column_if_not_exists(self):
        """Column should be added if it doesn't exist."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER)")

        result = safe_add_column(conn, "test", "name", "TEXT")

        assert result is True
        cursor = conn.execute("PRAGMA table_info(test)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "name" in columns

    def test_skips_if_column_exists(self):
        """Should return False if column already exists."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")

        result = safe_add_column(conn, "test", "name", "TEXT")

        assert result is False

    def test_adds_default_value(self):
        """Column with default value should work."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER)")

        safe_add_column(conn, "test", "status", "TEXT", default="'active'")

        conn.execute("INSERT INTO test (id) VALUES (1)")
        cursor = conn.execute("SELECT status FROM test WHERE id = 1")
        assert cursor.fetchone()[0] == "active"

    def test_rejects_invalid_table_name(self):
        """Invalid table name should raise ValueError."""
        conn = sqlite3.connect(":memory:")

        with pytest.raises(ValueError, match="Invalid table name"):
            safe_add_column(conn, "drop table users;--", "name", "TEXT")

    def test_rejects_invalid_column_name(self):
        """Invalid column name should raise ValueError."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER)")

        with pytest.raises(ValueError, match="Invalid column name"):
            safe_add_column(conn, "test", "column; DROP TABLE", "TEXT")

    def test_rejects_invalid_column_type(self):
        """Invalid column type should raise ValueError."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER)")

        with pytest.raises(ValueError, match="Invalid column type"):
            safe_add_column(conn, "test", "name", "DROP TABLE")

    def test_rejects_invalid_default(self):
        """Invalid default value should raise ValueError."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER)")

        with pytest.raises(ValueError, match="Invalid default value"):
            safe_add_column(conn, "test", "name", "TEXT", default="'; DROP TABLE;--")


# ============================================================================
# DatabaseManager Tests
# ============================================================================

class TestDatabaseManager:
    """Tests for DatabaseManager."""

    def test_singleton_pattern(self):
        """get_instance should return same instance for same path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            manager1 = DatabaseManager.get_instance(db_path)
            manager2 = DatabaseManager.get_instance(db_path)

            assert manager1 is manager2

            DatabaseManager.clear_instances()

    def test_clear_instances(self):
        """clear_instances should close all managers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = DatabaseManager.get_instance(db_path)
            manager.get_connection()  # Ensure connection is created

            DatabaseManager.clear_instances()

            # Getting instance again should create new one
            manager2 = DatabaseManager.get_instance(db_path)
            assert manager2 is not manager

            DatabaseManager.clear_instances()

    def test_connection_context_manager_commits(self):
        """connection() context manager should commit on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = DatabaseManager.get_instance(db_path)

            with manager.connection() as conn:
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.execute("INSERT INTO test VALUES (1)")

            # Verify data persisted
            cursor = manager.get_connection().execute("SELECT * FROM test")
            assert cursor.fetchone() == (1,)

            DatabaseManager.clear_instances()

    def test_connection_context_manager_rollback_on_error(self):
        """connection() should rollback on exception."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = DatabaseManager.get_instance(db_path)

            manager.get_connection().execute("CREATE TABLE test (id INTEGER)")
            manager.get_connection().commit()

            with pytest.raises(ValueError):
                with manager.connection() as conn:
                    conn.execute("INSERT INTO test VALUES (1)")
                    raise ValueError("Test error")

            # Verify data was rolled back
            cursor = manager.get_connection().execute("SELECT COUNT(*) FROM test")
            assert cursor.fetchone()[0] == 0

            DatabaseManager.clear_instances()

    def test_fresh_connection_pooling(self):
        """fresh_connection should use connection pooling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = DatabaseManager.get_instance(db_path)

            # Use fresh_connection multiple times
            for _ in range(5):
                with manager.fresh_connection() as conn:
                    conn.execute("SELECT 1")

            stats = manager.pool_stats()
            assert stats["returns"] > 0  # Connections returned to pool

            DatabaseManager.clear_instances()

    def test_execute_convenience_method(self):
        """execute() should work for simple queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = DatabaseManager.get_instance(db_path)

            manager.execute("CREATE TABLE test (id INTEGER)")
            manager.execute("INSERT INTO test VALUES (?)", (42,))
            manager.get_connection().commit()

            cursor = manager.execute("SELECT * FROM test")
            assert cursor.fetchone() == (42,)

            DatabaseManager.clear_instances()

    def test_fetch_one(self):
        """fetch_one should return single row."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = DatabaseManager.get_instance(db_path)

            with manager.connection() as conn:
                conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
                conn.execute("INSERT INTO test VALUES (1, 'Alice')")

            result = manager.fetch_one("SELECT * FROM test WHERE id = ?", (1,))
            assert result == (1, "Alice")

            DatabaseManager.clear_instances()

    def test_fetch_all(self):
        """fetch_all should return all rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            manager = DatabaseManager.get_instance(db_path)

            with manager.connection() as conn:
                conn.execute("CREATE TABLE test (id INTEGER)")
                conn.executemany("INSERT INTO test VALUES (?)", [(1,), (2,), (3,)])

            results = manager.fetch_all("SELECT * FROM test ORDER BY id")
            assert results == [(1,), (2,), (3,)]

            DatabaseManager.clear_instances()


# ============================================================================
# ConnectionPool Tests
# ============================================================================

class TestConnectionPool:
    """Tests for ConnectionPool."""

    def test_acquire_and_release(self):
        """Basic acquire/release should work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            pool = ConnectionPool(db_path, max_connections=5)

            conn = pool.acquire()
            assert conn is not None

            stats = pool.stats()
            assert stats["active"] == 1

            pool.release(conn)

            stats = pool.stats()
            assert stats["active"] == 0
            assert stats["idle"] == 1

            pool.close()

    def test_connection_context_manager(self):
        """connection() context manager should auto-release."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            pool = ConnectionPool(db_path, max_connections=5)

            with pool.connection() as conn:
                conn.execute("CREATE TABLE test (id INTEGER)")

            stats = pool.stats()
            assert stats["active"] == 0
            assert stats["idle"] == 1

            pool.close()

    def test_max_connections_limit(self):
        """Pool should respect max_connections limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            pool = ConnectionPool(db_path, max_connections=2, timeout=1.0)

            conn1 = pool.acquire()
            conn2 = pool.acquire()

            stats = pool.stats()
            assert stats["active"] == 2

            # Third acquire should timeout
            with pytest.raises(TimeoutError):
                pool.acquire(timeout=0.5)

            pool.release(conn1)
            pool.release(conn2)
            pool.close()

    def test_concurrent_access(self):
        """Pool should handle concurrent access correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            pool = ConnectionPool(db_path, max_connections=5)

            # Create table first
            with pool.connection() as conn:
                conn.execute("CREATE TABLE test (id INTEGER, thread_id INTEGER)")

            errors = []
            success_count = [0]

            def worker(thread_id):
                try:
                    for i in range(10):
                        with pool.connection() as conn:
                            conn.execute(
                                "INSERT INTO test VALUES (?, ?)",
                                (i, thread_id),
                            )
                            time.sleep(0.01)
                    success_count[0] += 1
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert not errors, f"Thread errors: {errors}"
            assert success_count[0] == 5

            # Verify all inserts succeeded
            with pool.connection() as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM test")
                count = cursor.fetchone()[0]
                assert count == 50  # 5 threads * 10 inserts

            pool.close()

    def test_close_pool(self):
        """Closing pool should prevent new acquisitions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            pool = ConnectionPool(db_path, max_connections=5)

            pool.close()

            from aragora.exceptions import DatabaseError
            with pytest.raises(DatabaseError, match="pool is closed"):
                pool.acquire()


# ============================================================================
# Performance Index Tests
# ============================================================================

class TestPerformanceIndexes:
    """Tests for create_performance_indexes."""

    def test_creates_indexes_on_existing_tables(self):
        """Indexes should be created for existing tables."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE memory_store (id INTEGER, agent_name TEXT, debate_id TEXT, timestamp TEXT)")
        conn.commit()

        result = create_performance_indexes(conn, tables_to_index=["memory_store"])

        assert len(result["created"]) > 0
        assert len(result["errors"]) == 0

    def test_skips_missing_tables(self):
        """Missing tables should be reported as errors."""
        conn = sqlite3.connect(":memory:")

        result = create_performance_indexes(conn, tables_to_index=["nonexistent_table"])

        assert len(result["created"]) == 0
        # All indexes for nonexistent table should be in errors

    def test_idempotent(self):
        """Creating indexes twice should be safe."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE memory_store (id INTEGER, agent_name TEXT, debate_id TEXT, timestamp TEXT)")
        conn.commit()

        result1 = create_performance_indexes(conn, tables_to_index=["memory_store"])
        result2 = create_performance_indexes(conn, tables_to_index=["memory_store"])

        assert len(result1["created"]) > 0
        assert len(result2["created"]) == 0
        assert len(result2["skipped"]) > 0


class TestAnalyzeTables:
    """Tests for analyze_tables."""

    def test_analyze_runs(self):
        """ANALYZE should run without error."""
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        conn.commit()

        # Should not raise
        analyze_tables(conn)


# ============================================================================
# WAL Connection Tests
# ============================================================================

class TestWalConnection:
    """Tests for get_wal_connection."""

    def test_wal_mode_enabled(self):
        """Connection should have WAL mode enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = get_wal_connection(db_path)

            cursor = conn.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            assert mode.lower() == "wal"

            conn.close()

    def test_custom_timeout(self):
        """Custom timeout should be set."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = get_wal_connection(db_path, timeout=60.0)

            cursor = conn.execute("PRAGMA busy_timeout")
            timeout = cursor.fetchone()[0]
            assert timeout == 60000  # milliseconds

            conn.close()
