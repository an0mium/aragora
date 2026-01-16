"""Tests for database backends."""

import pytest
import tempfile
import os
from pathlib import Path


class TestSQLiteBackend:
    """Test SQLiteBackend class."""

    def test_backend_creation(self):
        """Test SQLite backend can be created."""
        from aragora.storage.backends import SQLiteBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path)

            assert backend.backend_type == "sqlite"
            assert backend.db_path == db_path
            backend.close()

    def test_connection_context_manager(self):
        """Test connection context manager."""
        from aragora.storage.backends import SQLiteBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path)

            with backend.connection() as conn:
                conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")
                conn.execute("INSERT INTO test (name) VALUES (?)", ("test",))

            # Verify data persisted
            row = backend.fetch_one("SELECT name FROM test WHERE id = ?", (1,))
            assert row is not None
            assert row[0] == "test"
            backend.close()

    def test_fetch_all(self):
        """Test fetch_all method."""
        from aragora.storage.backends import SQLiteBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path)

            with backend.connection() as conn:
                conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO items (value) VALUES (?)", ("a",))
                conn.execute("INSERT INTO items (value) VALUES (?)", ("b",))
                conn.execute("INSERT INTO items (value) VALUES (?)", ("c",))

            rows = backend.fetch_all("SELECT value FROM items ORDER BY id")
            assert len(rows) == 3
            assert [r[0] for r in rows] == ["a", "b", "c"]
            backend.close()

    def test_execute_write(self):
        """Test execute_write method."""
        from aragora.storage.backends import SQLiteBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path)

            with backend.connection() as conn:
                conn.execute("CREATE TABLE counter (value INTEGER)")
                conn.execute("INSERT INTO counter (value) VALUES (?)", (0,))

            backend.execute_write("UPDATE counter SET value = ?", (42,))

            row = backend.fetch_one("SELECT value FROM counter")
            assert row[0] == 42
            backend.close()

    def test_executemany(self):
        """Test executemany method."""
        from aragora.storage.backends import SQLiteBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path)

            with backend.connection() as conn:
                conn.execute("CREATE TABLE bulk (id INTEGER PRIMARY KEY, name TEXT)")

            backend.executemany(
                "INSERT INTO bulk (name) VALUES (?)", [("a",), ("b",), ("c",), ("d",)]
            )

            rows = backend.fetch_all("SELECT name FROM bulk")
            assert len(rows) == 4
            backend.close()

    def test_rollback_on_error(self):
        """Test rollback on exception."""
        from aragora.storage.backends import SQLiteBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            backend = SQLiteBackend(db_path)

            with backend.connection() as conn:
                conn.execute("CREATE TABLE rollback_test (id INTEGER PRIMARY KEY, value TEXT)")
                conn.execute("INSERT INTO rollback_test (value) VALUES (?)", ("original",))

            try:
                with backend.connection() as conn:
                    conn.execute("UPDATE rollback_test SET value = ?", ("modified",))
                    raise ValueError("Simulated error")
            except ValueError:
                pass

            # Value should be rolled back
            row = backend.fetch_one("SELECT value FROM rollback_test WHERE id = ?", (1,))
            assert row[0] == "original"
            backend.close()


class TestPostgreSQLBackend:
    """Test PostgreSQLBackend class (mock tests without real PostgreSQL)."""

    def test_postgresql_available_flag(self):
        """Test POSTGRESQL_AVAILABLE flag is set correctly."""
        from aragora.storage.backends import POSTGRESQL_AVAILABLE

        # Just verify it's a boolean
        assert isinstance(POSTGRESQL_AVAILABLE, bool)

    def test_postgresql_backend_interface(self):
        """Test PostgreSQLBackend has same interface as SQLiteBackend."""
        from aragora.storage.backends import SQLiteBackend, PostgreSQLBackend, DatabaseBackend

        # Both should implement DatabaseBackend
        assert issubclass(SQLiteBackend, DatabaseBackend)
        assert issubclass(PostgreSQLBackend, DatabaseBackend)

        # Check same methods exist
        methods = [
            "connection",
            "fetch_one",
            "fetch_all",
            "execute_write",
            "executemany",
            "close",
            "backend_type",
        ]
        for method in methods:
            assert hasattr(SQLiteBackend, method)
            assert hasattr(PostgreSQLBackend, method)

    def test_placeholder_conversion(self):
        """Test SQL placeholder conversion."""
        from aragora.storage.backends import PostgreSQLBackend, POSTGRESQL_AVAILABLE
        from unittest.mock import MagicMock, patch

        if not POSTGRESQL_AVAILABLE:
            pytest.skip("psycopg2 not installed")

        # Create mock pool
        with patch("aragora.storage.backends.pg_pool") as mock_pool:
            mock_pool.ThreadedConnectionPool.return_value = MagicMock()

            backend = PostgreSQLBackend("postgresql://user:pass@localhost/db")

            # Test placeholder conversion
            sql = "SELECT * FROM users WHERE id = ? AND name = ?"
            converted = backend.convert_placeholder(sql)
            assert converted == "SELECT * FROM users WHERE id = %s AND name = %s"


class TestGetDatabaseBackend:
    """Test get_database_backend factory function."""

    def test_get_sqlite_backend(self):
        """Test getting SQLite backend."""
        from aragora.storage.backends import get_database_backend, reset_database_backend

        reset_database_backend()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            backend = get_database_backend(force_sqlite=True, db_path=db_path)

            assert backend.backend_type == "sqlite"
            reset_database_backend()

    def test_backend_caching(self):
        """Test backend instance is cached."""
        from aragora.storage.backends import get_database_backend, reset_database_backend

        reset_database_backend()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")

            backend1 = get_database_backend(force_sqlite=True, db_path=db_path)
            backend2 = get_database_backend()

            # Should be same instance
            assert backend1 is backend2
            reset_database_backend()

    def test_reset_backend(self):
        """Test reset_database_backend clears cache."""
        from aragora.storage.backends import get_database_backend, reset_database_backend

        reset_database_backend()

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")

            backend1 = get_database_backend(force_sqlite=True, db_path=db_path)
            reset_database_backend()
            backend2 = get_database_backend(force_sqlite=True, db_path=db_path)

            # Should be different instances
            assert backend1 is not backend2
            reset_database_backend()


class TestDatabaseSettings:
    """Test database settings for PostgreSQL."""

    def test_postgresql_settings(self):
        """Test PostgreSQL settings exist."""
        from aragora.config.settings import get_settings, reset_settings

        reset_settings()
        settings = get_settings()

        # Check PostgreSQL config fields exist
        assert hasattr(settings.database, "url")
        assert hasattr(settings.database, "backend")
        assert hasattr(settings.database, "pool_size")
        assert hasattr(settings.database, "pool_max_overflow")
        assert hasattr(settings.database, "is_postgresql")

        # Check defaults
        assert settings.database.backend == "sqlite"
        assert settings.database.url is None
        assert settings.database.is_postgresql is False

    def test_is_postgresql_property(self):
        """Test is_postgresql property logic."""
        from aragora.config.settings import get_settings, reset_settings

        reset_settings()

        # SQLite (default)
        settings = get_settings()
        assert settings.database.is_postgresql is False

        # Test with environment variables
        reset_settings()
        os.environ["ARAGORA_DB_BACKEND"] = "postgresql"
        os.environ["DATABASE_URL"] = "postgresql://localhost/test"

        try:
            reset_settings()
            settings = get_settings()
            assert settings.database.is_postgresql is True
        finally:
            # Clean up
            os.environ.pop("ARAGORA_DB_BACKEND", None)
            os.environ.pop("DATABASE_URL", None)
            reset_settings()
