"""
Tests for the schema versioning module.

Tests SQLite schema migration and version tracking.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from aragora.storage.schema import SchemaManager, safe_add_column, Migration


@pytest.fixture
def db_conn():
    """Create a temporary in-memory database connection."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def db_file():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    conn = sqlite3.connect(str(path))
    yield conn, path
    conn.close()
    path.unlink()


class TestSchemaManager:
    """Tests for SchemaManager class."""

    def test_creates_version_table(self, db_conn):
        """Version table should be created automatically."""
        manager = SchemaManager(db_conn, "test_module")

        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='_schema_versions'"
        )
        assert cursor.fetchone() is not None

    def test_initial_version_is_zero(self, db_conn):
        """New database should have version 0."""
        manager = SchemaManager(db_conn, "test_module")
        assert manager.get_version() == 0

    def test_set_and_get_version(self, db_conn):
        """Should be able to set and get version."""
        manager = SchemaManager(db_conn, "test_module")

        manager.set_version(5)
        assert manager.get_version() == 5

        manager.set_version(10)
        assert manager.get_version() == 10

    def test_multiple_modules_tracked_separately(self, db_conn):
        """Different modules should have independent versions."""
        manager1 = SchemaManager(db_conn, "module_a")
        manager2 = SchemaManager(db_conn, "module_b")

        manager1.set_version(3)
        manager2.set_version(7)

        assert manager1.get_version() == 3
        assert manager2.get_version() == 7

    def test_ensure_schema_with_initial(self, db_conn):
        """ensure_schema should create initial tables."""
        manager = SchemaManager(db_conn, "test", current_version=1)

        initial_sql = """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            );
        """

        result = manager.ensure_schema(initial_schema=initial_sql)

        assert result is True
        assert manager.get_version() == 1

        # Table should exist
        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
        )
        assert cursor.fetchone() is not None

    def test_ensure_schema_already_current(self, db_conn):
        """ensure_schema should return False if already current."""
        manager = SchemaManager(db_conn, "test", current_version=1)
        manager.set_version(1)

        result = manager.ensure_schema()

        assert result is False

    def test_migration_with_sql(self, db_conn):
        """Migrations with SQL should be applied."""
        # Create initial schema
        db_conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY)")
        db_conn.commit()

        manager = SchemaManager(db_conn, "test", current_version=2)
        manager.set_version(1)

        manager.register_migration(
            from_version=1,
            to_version=2,
            sql="ALTER TABLE items ADD COLUMN name TEXT;",
            description="Add name column",
        )

        result = manager.ensure_schema()

        assert result is True
        assert manager.get_version() == 2

        # Check column exists
        cursor = db_conn.execute("PRAGMA table_info(items)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "name" in columns

    def test_migration_with_function(self, db_conn):
        """Migrations with functions should be applied."""
        db_conn.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, value TEXT)")
        db_conn.commit()

        manager = SchemaManager(db_conn, "test", current_version=2)
        manager.set_version(1)

        def migrate_v1_to_v2(conn):
            conn.execute("INSERT INTO data (value) VALUES ('migrated')")

        manager.register_migration(
            from_version=1,
            to_version=2,
            function=migrate_v1_to_v2,
        )

        manager.ensure_schema()

        cursor = db_conn.execute("SELECT value FROM data")
        assert cursor.fetchone()[0] == "migrated"

    def test_multiple_migrations(self, db_conn):
        """Multiple migrations should run in order."""
        db_conn.execute("CREATE TABLE log (id INTEGER PRIMARY KEY, step INTEGER)")
        db_conn.commit()

        manager = SchemaManager(db_conn, "test", current_version=4)
        manager.set_version(1)

        manager.register_migration(1, 2, sql="INSERT INTO log (step) VALUES (2);")
        manager.register_migration(2, 3, sql="INSERT INTO log (step) VALUES (3);")
        manager.register_migration(3, 4, sql="INSERT INTO log (step) VALUES (4);")

        manager.ensure_schema()

        assert manager.get_version() == 4

        cursor = db_conn.execute("SELECT step FROM log ORDER BY step")
        steps = [row[0] for row in cursor.fetchall()]
        assert steps == [2, 3, 4]

    def test_validate_schema(self, db_conn):
        """validate_schema should check for expected tables."""
        db_conn.execute("CREATE TABLE users (id INTEGER)")
        db_conn.execute("CREATE TABLE posts (id INTEGER)")
        db_conn.commit()

        manager = SchemaManager(db_conn, "test")
        manager.set_version(1)

        result = manager.validate_schema(["users", "posts", "comments"])

        assert result["valid"] is False
        assert "comments" in result["missing"]
        assert result["version"] == 1


class TestMigration:
    """Tests for Migration class."""

    def test_migration_apply_sql(self, db_conn):
        """Migration.apply should execute SQL."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")

        migration = Migration(
            from_version=1,
            to_version=2,
            sql="ALTER TABLE t ADD COLUMN x TEXT;",
        )

        migration.apply(db_conn)

        cursor = db_conn.execute("PRAGMA table_info(t)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "x" in columns

    def test_migration_apply_function(self, db_conn):
        """Migration.apply should execute function."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")

        def add_column(conn):
            conn.execute("ALTER TABLE t ADD COLUMN y TEXT")

        migration = Migration(
            from_version=1,
            to_version=2,
            function=add_column,
        )

        migration.apply(db_conn)

        cursor = db_conn.execute("PRAGMA table_info(t)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "y" in columns

    def test_migration_requires_sql_or_function(self, db_conn):
        """Migration without sql or function should raise."""
        migration = Migration(from_version=1, to_version=2)

        with pytest.raises(ValueError):
            migration.apply(db_conn)


class TestSafeAddColumn:
    """Tests for safe_add_column helper."""

    def test_adds_missing_column(self, db_conn):
        """Should add column if it doesn't exist."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")
        db_conn.commit()

        result = safe_add_column(db_conn, "t", "name", "TEXT")

        assert result is True

        cursor = db_conn.execute("PRAGMA table_info(t)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "name" in columns

    def test_skips_existing_column(self, db_conn):
        """Should not error if column already exists."""
        db_conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        db_conn.commit()

        result = safe_add_column(db_conn, "t", "name", "TEXT")

        assert result is False

    def test_adds_column_with_default(self, db_conn):
        """Should add column with default value."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")
        db_conn.execute("INSERT INTO t (id) VALUES (1)")
        db_conn.commit()

        safe_add_column(db_conn, "t", "status", "TEXT", default="'active'")

        cursor = db_conn.execute("SELECT status FROM t WHERE id = 1")
        assert cursor.fetchone()[0] == "active"


class TestPersistence:
    """Tests for schema version persistence."""

    def test_version_persists_across_connections(self, db_file):
        """Schema version should persist across database reconnections."""
        conn, path = db_file

        manager = SchemaManager(conn, "test", current_version=3)
        manager.set_version(3)
        conn.close()

        # Reconnect
        conn2 = sqlite3.connect(str(path))
        manager2 = SchemaManager(conn2, "test", current_version=3)

        assert manager2.get_version() == 3

        conn2.close()


class TestDatabaseManager:
    """Tests for DatabaseManager singleton pattern."""

    def test_singleton_pattern(self, db_file):
        """Same path should return same instance."""
        from aragora.storage.schema import DatabaseManager

        conn, path = db_file
        conn.close()  # Close the fixture connection

        try:
            manager1 = DatabaseManager.get_instance(path)
            manager2 = DatabaseManager.get_instance(path)

            assert manager1 is manager2
        finally:
            DatabaseManager.clear_instances()

    def test_different_paths_different_instances(self):
        """Different paths should return different instances."""
        from aragora.storage.schema import DatabaseManager

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f1:
            path1 = Path(f1.name)
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f2:
            path2 = Path(f2.name)

        try:
            manager1 = DatabaseManager.get_instance(path1)
            manager2 = DatabaseManager.get_instance(path2)

            assert manager1 is not manager2
        finally:
            DatabaseManager.clear_instances()
            path1.unlink()
            path2.unlink()

    def test_clear_instances(self, db_file):
        """clear_instances should remove all cached instances."""
        from aragora.storage.schema import DatabaseManager

        conn, path = db_file
        conn.close()

        try:
            manager1 = DatabaseManager.get_instance(path)
            DatabaseManager.clear_instances()
            manager2 = DatabaseManager.get_instance(path)

            # After clear, should get a new instance
            assert manager1 is not manager2
        finally:
            DatabaseManager.clear_instances()

    def test_connection_context_manager(self, db_file):
        """Context manager should commit on success."""
        from aragora.storage.schema import DatabaseManager

        conn, path = db_file
        conn.close()

        try:
            manager = DatabaseManager.get_instance(path)

            # Create table using context manager
            with manager.connection() as c:
                c.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
                c.execute("INSERT INTO test (id) VALUES (1)")

            # Verify data was committed
            with manager.connection() as c:
                cursor = c.execute("SELECT id FROM test")
                assert cursor.fetchone()[0] == 1
        finally:
            DatabaseManager.clear_instances()

    def test_connection_context_manager_rollback(self, db_file):
        """Context manager should rollback on error."""
        from aragora.storage.schema import DatabaseManager

        conn, path = db_file
        conn.close()

        try:
            manager = DatabaseManager.get_instance(path)

            # Create table
            with manager.connection() as c:
                c.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")

            # Try to insert then raise error
            with pytest.raises(ValueError):
                with manager.connection() as c:
                    c.execute("INSERT INTO test (id) VALUES (1)")
                    raise ValueError("Test error")

            # Data should not be committed
            with manager.connection() as c:
                cursor = c.execute("SELECT COUNT(*) FROM test")
                assert cursor.fetchone()[0] == 0
        finally:
            DatabaseManager.clear_instances()


class TestSafeAddColumnValidation:
    """Tests for SQL injection prevention in safe_add_column."""

    def test_rejects_invalid_table_name(self, db_conn):
        """Should reject table names with special characters."""
        db_conn.execute("CREATE TABLE valid_table (id INTEGER)")
        db_conn.commit()

        with pytest.raises(ValueError, match="Invalid table name"):
            safe_add_column(db_conn, "valid_table; DROP TABLE users; --", "col", "TEXT")

    def test_rejects_invalid_column_name(self, db_conn):
        """Should reject column names with special characters."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")
        db_conn.commit()

        with pytest.raises(ValueError, match="Invalid column name"):
            safe_add_column(db_conn, "t", "col; DROP TABLE t; --", "TEXT")

    def test_rejects_invalid_column_type(self, db_conn):
        """Should reject column types not in whitelist."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")
        db_conn.commit()

        with pytest.raises(ValueError, match="Invalid column type"):
            safe_add_column(db_conn, "t", "col", "TEXT; DROP TABLE t; --")

    def test_rejects_invalid_default_value(self, db_conn):
        """Should reject default values with injection attempts."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")
        db_conn.commit()

        with pytest.raises(ValueError, match="Invalid default value"):
            safe_add_column(db_conn, "t", "col", "TEXT", default="''; DROP TABLE t; --")

    def test_accepts_valid_identifiers(self, db_conn):
        """Should accept valid SQL identifiers."""
        db_conn.execute("CREATE TABLE valid_table_123 (id INTEGER)")
        db_conn.commit()

        result = safe_add_column(db_conn, "valid_table_123", "valid_column_456", "TEXT")
        assert result is True

    def test_accepts_valid_default_null(self, db_conn):
        """Should accept NULL as default."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")
        db_conn.commit()

        result = safe_add_column(db_conn, "t", "col", "TEXT", default="NULL")
        assert result is True

    def test_accepts_valid_default_numeric(self, db_conn):
        """Should accept numeric defaults."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")
        db_conn.commit()

        result = safe_add_column(db_conn, "t", "col", "INTEGER", default="42")
        assert result is True

    def test_accepts_valid_default_timestamp(self, db_conn):
        """Should accept CURRENT_TIMESTAMP default."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")
        db_conn.commit()

        result = safe_add_column(db_conn, "t", "col", "TIMESTAMP", default="CURRENT_TIMESTAMP")
        assert result is True

    def test_rejects_long_identifiers(self, db_conn):
        """Should reject identifiers longer than 128 characters."""
        db_conn.execute("CREATE TABLE t (id INTEGER)")
        db_conn.commit()

        long_name = "a" * 129
        with pytest.raises(ValueError, match="Invalid column name"):
            safe_add_column(db_conn, "t", long_name, "TEXT")
