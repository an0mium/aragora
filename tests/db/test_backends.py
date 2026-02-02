"""
Tests for Database Backend Abstraction Layer.

Covers:
- DatabaseConfig creation and parsing
- SQLiteBackend operations and pooling
- PostgresBackend operations (with mocking)
- SQL translation between backends
- Connection pooling behavior
- Health checks
- Global database instance management
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.db.backends import (
    DatabaseConfig,
    DatabaseBackend,
    SQLiteBackend,
    PostgresBackend,
    configure_database,
    get_database,
    ConnectionProtocol,
    CursorProtocol,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_db_dir(tmp_path):
    """Create a temporary directory for database files."""
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    return db_dir


@pytest.fixture
def sqlite_config(temp_db_dir):
    """Create a SQLite database configuration."""
    return DatabaseConfig(
        backend="sqlite",
        sqlite_path=str(temp_db_dir / "test.db"),
        pool_size=5,
        pool_timeout=10.0,
    )


@pytest.fixture
def postgres_config():
    """Create a PostgreSQL database configuration."""
    return DatabaseConfig(
        backend="postgres",
        pg_host="localhost",
        pg_port=5432,
        pg_database="test_db",
        pg_user="test_user",
        pg_password="test_pass",
        pg_ssl_mode="prefer",
        pool_size=10,
    )


@pytest.fixture
def sqlite_backend(sqlite_config):
    """Create a SQLite backend instance."""
    backend = SQLiteBackend(sqlite_config)
    yield backend
    backend.close()


@pytest.fixture
def clean_global_db():
    """Reset global database state before and after test."""
    import aragora.db.backends as backends_module

    original = backends_module._database
    backends_module._database = None
    yield
    if backends_module._database is not None:
        backends_module._database.close()
    backends_module._database = original


# -----------------------------------------------------------------------------
# DatabaseConfig Tests
# -----------------------------------------------------------------------------


class TestDatabaseConfigCreation:
    """Tests for DatabaseConfig initialization."""

    def test_default_values(self, temp_db_dir):
        """DatabaseConfig has sensible defaults."""
        with patch("aragora.config.resolve_db_path", return_value=str(temp_db_dir / "default.db")):
            config = DatabaseConfig()

        assert config.backend == "sqlite"
        assert config.pg_host == "localhost"
        assert config.pg_port == 5432
        assert config.pool_size == 20
        assert config.pool_max_overflow == 15

    def test_custom_values(self, temp_db_dir):
        """DatabaseConfig accepts custom values."""
        config = DatabaseConfig(
            backend="postgres",
            sqlite_path=str(temp_db_dir / "custom.db"),
            pg_host="db.example.com",
            pg_port=5433,
            pg_database="mydb",
            pool_size=50,
        )

        assert config.backend == "postgres"
        assert config.pg_host == "db.example.com"
        assert config.pg_port == 5433
        assert config.pg_database == "mydb"
        assert config.pool_size == 50


class TestDatabaseConfigFromEnv:
    """Tests for DatabaseConfig.from_env()."""

    def test_from_env_defaults(self, temp_db_dir):
        """from_env uses defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("aragora.config.resolve_db_path", return_value=str(temp_db_dir / "env.db")):
                config = DatabaseConfig.from_env()

        assert config.backend == "sqlite"

    def test_from_env_with_backend(self, temp_db_dir):
        """from_env respects ARAGORA_DB_BACKEND."""
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "postgres"}, clear=True):
            with patch("aragora.config.resolve_db_path", return_value=str(temp_db_dir / "env.db")):
                config = DatabaseConfig.from_env()

        assert config.backend == "postgres"

    def test_from_env_with_database_url(self, temp_db_dir):
        """from_env parses DATABASE_URL and auto-detects postgres."""
        url = "postgresql://user:pass@db.example.com:5433/mydb?sslmode=require"

        with patch.dict(os.environ, {"DATABASE_URL": url}, clear=True):
            with patch("aragora.config.resolve_db_path", return_value=str(temp_db_dir / "env.db")):
                config = DatabaseConfig.from_env()

        assert config.backend == "postgres"
        assert config.pg_host == "db.example.com"
        assert config.pg_port == 5433
        assert config.pg_database == "mydb"
        assert config.pg_user == "user"
        assert config.pg_password == "pass"
        assert config.pg_ssl_mode == "require"

    def test_from_env_with_postgres_url_scheme(self, temp_db_dir):
        """from_env handles postgres:// URL scheme."""
        url = "postgres://user:pass@host:5432/db"

        with patch.dict(os.environ, {"DATABASE_URL": url}, clear=True):
            with patch("aragora.config.resolve_db_path", return_value=str(temp_db_dir / "env.db")):
                config = DatabaseConfig.from_env()

        assert config.backend == "postgres"
        assert config.pg_host == "host"

    def test_from_env_with_pool_settings(self, temp_db_dir):
        """from_env reads pool settings from environment."""
        env = {
            "ARAGORA_DB_POOL_SIZE": "100",
            "ARAGORA_DB_POOL_MAX_OVERFLOW": "50",
            "ARAGORA_DB_POOL_TIMEOUT": "60.0",
        }

        with patch.dict(os.environ, env, clear=True):
            with patch("aragora.config.resolve_db_path", return_value=str(temp_db_dir / "env.db")):
                config = DatabaseConfig.from_env()

        assert config.pool_size == 100
        assert config.pool_max_overflow == 50
        assert config.pool_timeout == 60.0

    def test_from_env_url_with_special_chars(self, temp_db_dir):
        """from_env handles URL-encoded special characters in password."""
        # Password with special chars: p@ss%word!
        url = "postgresql://user:p%40ss%25word%21@host:5432/db"

        with patch.dict(os.environ, {"DATABASE_URL": url}, clear=True):
            with patch("aragora.config.resolve_db_path", return_value=str(temp_db_dir / "env.db")):
                config = DatabaseConfig.from_env()

        assert config.pg_password == "p@ss%word!"


class TestDatabaseConfigDSN:
    """Tests for DatabaseConfig.pg_dsn property."""

    def test_pg_dsn_format(self, postgres_config):
        """pg_dsn returns properly formatted connection string."""
        dsn = postgres_config.pg_dsn

        assert "host=localhost" in dsn
        assert "port=5432" in dsn
        assert "dbname=test_db" in dsn
        assert "user=test_user" in dsn
        assert "password=test_pass" in dsn
        assert "sslmode=prefer" in dsn


# -----------------------------------------------------------------------------
# SQLiteBackend Tests
# -----------------------------------------------------------------------------


class TestSQLiteBackendInitialization:
    """Tests for SQLiteBackend initialization."""

    def test_creates_parent_directory(self, temp_db_dir):
        """SQLiteBackend creates parent directory if needed."""
        nested_path = temp_db_dir / "nested" / "path" / "test.db"
        config = DatabaseConfig(
            backend="sqlite",
            sqlite_path=str(nested_path),
        )

        backend = SQLiteBackend(config)
        assert nested_path.parent.exists()
        backend.close()

    def test_placeholder_is_question_mark(self, sqlite_backend):
        """SQLite uses ? as placeholder."""
        assert sqlite_backend.placeholder == "?"


class TestSQLiteBackendConnection:
    """Tests for SQLiteBackend connection management."""

    def test_connect_creates_connection(self, sqlite_backend):
        """connect() returns a valid connection."""
        conn = sqlite_backend.connect()
        assert conn is not None

        # Verify it's usable
        cursor = conn.execute("SELECT 1")
        result = cursor.fetchone()
        assert result == (1,)

        conn.close()

    def test_connection_context_manager(self, sqlite_backend):
        """connection() context manager auto-commits."""
        with sqlite_backend.connection() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO test (id) VALUES (1)")

        # Verify commit happened
        row = sqlite_backend.fetch_one("SELECT id FROM test WHERE id = 1")
        assert row == (1,)

    def test_connection_context_manager_rollback_on_error(self, sqlite_backend):
        """connection() context manager rolls back on error."""
        # Create table first
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS test2 (id INTEGER PRIMARY KEY)")

        try:
            with sqlite_backend.connection() as conn:
                conn.execute("INSERT INTO test2 (id) VALUES (1)")
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify rollback happened
        row = sqlite_backend.fetch_one("SELECT id FROM test2 WHERE id = 1")
        assert row is None


class TestSQLiteBackendOperations:
    """Tests for SQLiteBackend SQL operations."""

    def test_execute_without_fetch(self, sqlite_backend):
        """execute() without fetch returns cursor."""
        cursor = sqlite_backend.execute(
            "CREATE TABLE IF NOT EXISTS exec_test (id INTEGER PRIMARY KEY)"
        )
        assert cursor is not None

    def test_execute_with_fetch(self, sqlite_backend):
        """execute() with fetch=True returns rows."""
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS exec_test2 (id INTEGER PRIMARY KEY)")
        sqlite_backend.execute("INSERT INTO exec_test2 (id) VALUES (1), (2), (3)")

        rows = sqlite_backend.execute("SELECT id FROM exec_test2 ORDER BY id", fetch=True)
        assert rows == [(1,), (2,), (3,)]

    def test_execute_with_params(self, sqlite_backend):
        """execute() handles parameterized queries."""
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS param_test (name TEXT)")
        sqlite_backend.execute("INSERT INTO param_test (name) VALUES (?)", ("test_name",))

        row = sqlite_backend.fetch_one("SELECT name FROM param_test")
        assert row == ("test_name",)

    def test_executemany(self, sqlite_backend):
        """executemany() inserts multiple rows."""
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS many_test (id INTEGER, name TEXT)")

        params = [(1, "one"), (2, "two"), (3, "three")]
        count = sqlite_backend.executemany("INSERT INTO many_test VALUES (?, ?)", params)

        rows = sqlite_backend.fetch_all("SELECT id, name FROM many_test ORDER BY id")
        assert len(rows) == 3

    def test_fetch_one_returns_single_row(self, sqlite_backend):
        """fetch_one() returns single row or None."""
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS fetch_one_test (id INTEGER)")
        sqlite_backend.execute("INSERT INTO fetch_one_test (id) VALUES (42)")

        row = sqlite_backend.fetch_one("SELECT id FROM fetch_one_test")
        assert row == (42,)

        no_row = sqlite_backend.fetch_one("SELECT id FROM fetch_one_test WHERE id = 999")
        assert no_row is None

    def test_fetch_all_returns_all_rows(self, sqlite_backend):
        """fetch_all() returns all matching rows."""
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS fetch_all_test (id INTEGER)")
        sqlite_backend.execute("INSERT INTO fetch_all_test (id) VALUES (1), (2), (3)")

        rows = sqlite_backend.fetch_all("SELECT id FROM fetch_all_test ORDER BY id")
        assert rows == [(1,), (2,), (3,)]

    def test_fetch_all_empty_result(self, sqlite_backend):
        """fetch_all() returns empty list when no matches."""
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS empty_test (id INTEGER)")

        rows = sqlite_backend.fetch_all("SELECT id FROM empty_test")
        assert rows == []


class TestSQLiteBackendPooling:
    """Tests for SQLiteBackend connection pooling."""

    def test_connection_returned_to_pool(self, sqlite_backend):
        """Connections are returned to pool after use."""
        # Use a connection
        with sqlite_backend.connection() as conn:
            conn.execute("SELECT 1")

        # Pool should have one connection
        assert len(sqlite_backend._pool) == 1

    def test_pool_size_limit(self, temp_db_dir):
        """Pool doesn't exceed max size."""
        config = DatabaseConfig(
            backend="sqlite",
            sqlite_path=str(temp_db_dir / "pool_test.db"),
            pool_size=2,
        )
        backend = SQLiteBackend(config)

        # Use multiple connections
        for _ in range(5):
            with backend.connection():
                pass

        # Pool should be at max
        assert len(backend._pool) <= 2
        backend.close()

    def test_broken_connection_discarded(self, sqlite_backend):
        """Broken connections are not returned to pool."""
        # Get a connection and close it to make it "broken"
        conn = sqlite_backend.connect()
        conn.close()

        # Force return to pool
        sqlite_backend._pool.append(conn)

        # Get another connection - should skip the broken one
        new_conn = sqlite_backend.connect()
        assert new_conn is not conn

        # Verify the broken one was removed
        assert conn not in sqlite_backend._pool
        new_conn.close()


class TestSQLiteBackendHealthCheck:
    """Tests for SQLiteBackend health check."""

    def test_health_check_success(self, sqlite_backend):
        """health_check() returns True for valid database."""
        assert sqlite_backend.health_check() is True

    def test_health_check_failure(self, sqlite_backend):
        """health_check() returns False on connection failure."""
        # Mock connection to simulate failure
        with patch.object(sqlite_backend, "connection", side_effect=sqlite3.Error("Test error")):
            assert sqlite_backend.health_check() is False


class TestSQLiteBackendClose:
    """Tests for SQLiteBackend close."""

    def test_close_clears_pool(self, sqlite_backend):
        """close() clears all pooled connections."""
        # Add some connections to pool
        with sqlite_backend.connection():
            pass

        assert len(sqlite_backend._pool) > 0

        sqlite_backend.close()

        assert len(sqlite_backend._pool) == 0


# -----------------------------------------------------------------------------
# PostgresBackend Tests
# -----------------------------------------------------------------------------


class TestPostgresBackendInitialization:
    """Tests for PostgresBackend initialization."""

    def test_placeholder_is_percent_s(self, postgres_config):
        """PostgreSQL uses %s as placeholder."""
        with patch("psycopg2.pool.ThreadedConnectionPool"):
            backend = PostgresBackend(postgres_config)
            assert backend.placeholder == "%s"
            backend.close()

    def test_initialization_without_psycopg2(self, postgres_config):
        """PostgresBackend handles missing psycopg2."""
        with patch.dict("sys.modules", {"psycopg2": None, "psycopg2.pool": None}):
            # Force reimport by deleting cached import
            import importlib
            import aragora.db.backends as backends

            importlib.reload(backends)

            backend = backends.PostgresBackend(postgres_config)
            assert backend._initialized is False

            # Re-reload to restore
            importlib.reload(backends)


class TestPostgresBackendSQLTranslation:
    """Tests for PostgresBackend SQL translation."""

    def test_translate_sql_question_marks(self, postgres_config):
        """translate_sql converts ? to %s."""
        with patch("psycopg2.pool.ThreadedConnectionPool"):
            backend = PostgresBackend(postgres_config)

            sql = "SELECT * FROM users WHERE id = ? AND name = ?"
            translated = backend.translate_sql(sql)

            assert translated == "SELECT * FROM users WHERE id = %s AND name = %s"
            backend.close()

    def test_translate_sql_no_placeholders(self, postgres_config):
        """translate_sql leaves SQL without placeholders unchanged."""
        with patch("psycopg2.pool.ThreadedConnectionPool"):
            backend = PostgresBackend(postgres_config)

            sql = "SELECT * FROM users"
            translated = backend.translate_sql(sql)

            assert translated == sql
            backend.close()


class TestPostgresBackendOperations:
    """Tests for PostgresBackend operations with mocked psycopg2."""

    @pytest.fixture
    def mock_postgres_backend(self, postgres_config):
        """Create a PostgresBackend with mocked psycopg2."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        with patch("psycopg2.pool.ThreadedConnectionPool", return_value=mock_pool):
            backend = PostgresBackend(postgres_config)
            backend._mock_pool = mock_pool
            backend._mock_conn = mock_conn
            backend._mock_cursor = mock_cursor
            yield backend
            backend.close()

    def test_connect_gets_from_pool(self, mock_postgres_backend):
        """connect() gets connection from pool."""
        conn = mock_postgres_backend.connect()
        assert conn == mock_postgres_backend._mock_conn
        mock_postgres_backend._mock_pool.getconn.assert_called_once()

    def test_execute_translates_sql(self, mock_postgres_backend):
        """execute() translates SQL placeholders."""
        mock_postgres_backend._mock_cursor.fetchall.return_value = [(1,)]

        mock_postgres_backend.execute("SELECT * FROM users WHERE id = ?", (1,), fetch=True)

        # Verify translated SQL was used
        call_args = mock_postgres_backend._mock_cursor.execute.call_args
        assert "%s" in call_args[0][0]
        assert "?" not in call_args[0][0]

    def test_fetch_one(self, mock_postgres_backend):
        """fetch_one() returns single row."""
        mock_postgres_backend._mock_cursor.fetchone.return_value = (42, "test")

        row = mock_postgres_backend.fetch_one("SELECT * FROM users WHERE id = ?", (1,))

        assert row == (42, "test")

    def test_fetch_all(self, mock_postgres_backend):
        """fetch_all() returns all rows."""
        mock_postgres_backend._mock_cursor.fetchall.return_value = [(1,), (2,), (3,)]

        rows = mock_postgres_backend.fetch_all("SELECT id FROM users")

        assert rows == [(1,), (2,), (3,)]

    def test_executemany(self, mock_postgres_backend):
        """executemany() executes with multiple param sets."""
        mock_postgres_backend._mock_cursor.rowcount = 3

        count = mock_postgres_backend.executemany(
            "INSERT INTO users (id) VALUES (?)", [(1,), (2,), (3,)]
        )

        mock_postgres_backend._mock_cursor.executemany.assert_called_once()

    def test_health_check_success(self, mock_postgres_backend):
        """health_check() returns True on success."""
        assert mock_postgres_backend.health_check() is True

    def test_health_check_failure(self, mock_postgres_backend):
        """health_check() returns False on error."""
        mock_postgres_backend._mock_cursor.execute.side_effect = Exception("Connection failed")

        assert mock_postgres_backend.health_check() is False


class TestPostgresBackendNotInitialized:
    """Tests for PostgresBackend when not properly initialized."""

    def test_connect_raises_when_not_initialized(self, postgres_config):
        """connect() raises ConfigurationError when not initialized."""
        from aragora.exceptions import ConfigurationError

        with patch("psycopg2.pool.ThreadedConnectionPool", side_effect=ImportError):
            backend = PostgresBackend(postgres_config)
            backend._initialized = False

            with pytest.raises(ConfigurationError):
                backend.connect()

    def test_connection_raises_when_not_initialized(self, postgres_config):
        """connection() context manager raises when not initialized."""
        from aragora.exceptions import ConfigurationError

        with patch("psycopg2.pool.ThreadedConnectionPool", side_effect=ImportError):
            backend = PostgresBackend(postgres_config)
            backend._initialized = False

            with pytest.raises(ConfigurationError):
                with backend.connection():
                    pass


# -----------------------------------------------------------------------------
# Global Database Instance Tests
# -----------------------------------------------------------------------------


class TestGlobalDatabaseInstance:
    """Tests for global database instance management."""

    def test_configure_database_sqlite(self, sqlite_config, clean_global_db):
        """configure_database creates SQLite backend."""
        backend = configure_database(sqlite_config)

        # Check by class name to avoid import path issues
        assert backend.__class__.__name__ == "SQLiteBackend"
        assert backend.health_check() is True

    def test_configure_database_postgres(self, postgres_config, clean_global_db):
        """configure_database creates PostgreSQL backend."""
        with patch("psycopg2.pool.ThreadedConnectionPool"):
            backend = configure_database(postgres_config)

            assert backend.__class__.__name__ == "PostgresBackend"

    def test_configure_database_from_env(self, temp_db_dir, clean_global_db):
        """configure_database uses env vars when no config provided."""
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "sqlite"}, clear=True):
            with patch("aragora.config.resolve_db_path", return_value=str(temp_db_dir / "env.db")):
                backend = configure_database()

                assert backend.__class__.__name__ == "SQLiteBackend"

    def test_get_database_initializes_once(self, sqlite_config, clean_global_db):
        """get_database initializes backend only once."""
        configure_database(sqlite_config)

        db1 = get_database()
        db2 = get_database()

        assert db1 is db2

    def test_configure_database_closes_previous(self, temp_db_dir, clean_global_db):
        """configure_database closes previous backend."""
        config1 = DatabaseConfig(
            backend="sqlite",
            sqlite_path=str(temp_db_dir / "first.db"),
        )
        config2 = DatabaseConfig(
            backend="sqlite",
            sqlite_path=str(temp_db_dir / "second.db"),
        )

        backend1 = configure_database(config1)

        # Add a connection to pool to verify it gets cleaned up
        with backend1.connection():
            pass

        backend2 = configure_database(config2)

        # backend1 should be closed
        assert len(backend1._pool) == 0
        assert backend1 is not backend2


# -----------------------------------------------------------------------------
# Concurrent Access Tests
# -----------------------------------------------------------------------------


class TestConcurrentAccess:
    """Tests for thread-safe concurrent database access."""

    def test_concurrent_sqlite_operations(self, sqlite_backend):
        """SQLite backend handles concurrent operations."""
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS concurrent_test (id INTEGER)")

        errors = []

        def worker(n):
            try:
                for i in range(10):
                    sqlite_backend.execute(
                        "INSERT INTO concurrent_test (id) VALUES (?)", (n * 100 + i,)
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify all inserts succeeded
        rows = sqlite_backend.fetch_all("SELECT COUNT(*) FROM concurrent_test")
        assert rows[0][0] == 50  # 5 threads * 10 inserts


# -----------------------------------------------------------------------------
# Protocol Compliance Tests
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests verifying backends comply with protocols."""

    def test_sqlite_backend_is_database_backend(self, sqlite_backend):
        """SQLiteBackend is a DatabaseBackend."""
        assert isinstance(sqlite_backend, DatabaseBackend)

    def test_postgres_backend_is_database_backend(self, postgres_config):
        """PostgresBackend is a DatabaseBackend."""
        with patch("psycopg2.pool.ThreadedConnectionPool"):
            backend = PostgresBackend(postgres_config)
            assert isinstance(backend, DatabaseBackend)
            backend.close()

    def test_backend_has_required_methods(self, sqlite_backend):
        """Backend has all required abstract methods."""
        required_methods = [
            "connect",
            "connection",
            "execute",
            "executemany",
            "fetch_one",
            "fetch_all",
            "health_check",
            "close",
            "translate_sql",
            "translate_params",
        ]

        for method_name in required_methods:
            assert hasattr(sqlite_backend, method_name)
            assert callable(getattr(sqlite_backend, method_name))


# -----------------------------------------------------------------------------
# Edge Cases and Error Handling
# -----------------------------------------------------------------------------


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error handling."""

    def test_empty_params(self, sqlite_backend):
        """Operations work with empty params."""
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS empty_params_test (id INTEGER)")

        # Empty tuple
        rows = sqlite_backend.fetch_all("SELECT * FROM empty_params_test", ())
        assert rows == []

    def test_dict_params(self, sqlite_backend):
        """Operations work with dict params."""
        sqlite_backend.execute("CREATE TABLE IF NOT EXISTS dict_params_test (name TEXT)")

        # Named parameters
        sqlite_backend.execute(
            "INSERT INTO dict_params_test (name) VALUES (:name)", {"name": "test"}
        )

        row = sqlite_backend.fetch_one("SELECT name FROM dict_params_test")
        assert row == ("test",)

    def test_translate_params_default(self, sqlite_backend):
        """translate_params returns params unchanged by default."""
        params = (1, 2, 3)
        assert sqlite_backend.translate_params(params) == params

    def test_translate_sql_default(self, sqlite_backend):
        """translate_sql returns SQL unchanged by default."""
        sql = "SELECT * FROM table"
        assert sqlite_backend.translate_sql(sql) == sql

    def test_database_url_parsing_error(self, temp_db_dir):
        """Invalid DATABASE_URL is handled gracefully."""
        with patch.dict(os.environ, {"DATABASE_URL": "invalid://url"}, clear=True):
            with patch("aragora.config.resolve_db_path", return_value=str(temp_db_dir / "env.db")):
                # Should not raise, just log warning
                config = DatabaseConfig.from_env()
                # Backend remains sqlite (default) since URL couldn't be parsed
                assert config.backend == "sqlite"
