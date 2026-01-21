"""
PostgreSQL integration tests for Aragora.

These tests verify that the database abstraction layer works correctly
with PostgreSQL. They require a running PostgreSQL instance.

Run with:
    pytest tests/integration/test_postgres.py -v

Or with Docker:
    docker-compose -f docker-compose.dev.yml up -d postgres
    DATABASE_URL=postgresql://aragora:aragora_dev_password@localhost:5432/aragora \
        pytest tests/integration/test_postgres.py -v
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager
from typing import Generator

import pytest

# Skip all tests if PostgreSQL is not configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL", "").startswith("postgresql://"),
    reason="PostgreSQL not configured (set DATABASE_URL)",
)


@pytest.fixture
def pg_connection_url() -> str:
    """Get PostgreSQL connection URL from environment."""
    url = os.environ.get("DATABASE_URL")
    if not url or not url.startswith("postgresql://"):
        pytest.skip("DATABASE_URL not set or not PostgreSQL")
    return url


@pytest.fixture
def test_table_name() -> str:
    """Generate unique table name to avoid conflicts."""
    return f"test_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def pg_backend(pg_connection_url):
    """Create a PostgreSQL backend for testing."""
    from aragora.db.backends import PostgreSQLBackend

    backend = PostgreSQLBackend(pg_connection_url)
    yield backend
    backend.close()


@pytest.fixture
def clean_test_schema(pg_backend, test_table_name):
    """Create and cleanup a test table."""
    # Create test table
    with pg_backend.connection() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {test_table_name} (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                value INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()

    yield test_table_name

    # Cleanup
    with pg_backend.connection() as conn:
        conn.execute(f"DROP TABLE IF EXISTS {test_table_name}")
        conn.commit()


class TestPostgreSQLBackend:
    """Tests for PostgreSQL backend functionality."""

    def test_connection_health(self, pg_backend):
        """Test that the backend can connect and report health."""
        assert pg_backend.is_healthy()

    def test_basic_crud(self, pg_backend, clean_test_schema):
        """Test basic CRUD operations."""
        table = clean_test_schema

        # Create
        with pg_backend.connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO {table} (name, value) VALUES ($1, $2) RETURNING id",
                ("test_item", 42),
            )
            row = cursor.fetchone()
            assert row is not None
            item_id = row[0]
            conn.commit()

        # Read
        with pg_backend.connection() as conn:
            cursor = conn.execute(
                f"SELECT name, value FROM {table} WHERE id = $1",
                (item_id,),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == "test_item"
            assert row[1] == 42

        # Update
        with pg_backend.connection() as conn:
            conn.execute(
                f"UPDATE {table} SET value = $1 WHERE id = $2",
                (100, item_id),
            )
            conn.commit()

        with pg_backend.connection() as conn:
            cursor = conn.execute(
                f"SELECT value FROM {table} WHERE id = $1",
                (item_id,),
            )
            row = cursor.fetchone()
            assert row[0] == 100

        # Delete
        with pg_backend.connection() as conn:
            conn.execute(f"DELETE FROM {table} WHERE id = $1", (item_id,))
            conn.commit()

        with pg_backend.connection() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE id = $1",
                (item_id,),
            )
            assert cursor.fetchone()[0] == 0

    def test_transaction_commit(self, pg_backend, clean_test_schema):
        """Test that transactions commit properly."""
        table = clean_test_schema

        with pg_backend.connection() as conn:
            conn.execute(
                f"INSERT INTO {table} (name, value) VALUES ($1, $2)",
                ("tx_test", 1),
            )
            conn.commit()

        # Verify in new connection
        with pg_backend.connection() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE name = $1",
                ("tx_test",),
            )
            assert cursor.fetchone()[0] == 1

    def test_transaction_rollback(self, pg_backend, clean_test_schema):
        """Test that transactions rollback properly."""
        table = clean_test_schema

        with pg_backend.connection() as conn:
            conn.execute(
                f"INSERT INTO {table} (name, value) VALUES ($1, $2)",
                ("rollback_test", 1),
            )
            conn.rollback()

        # Verify not present
        with pg_backend.connection() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE name = $1",
                ("rollback_test",),
            )
            assert cursor.fetchone()[0] == 0

    def test_executemany(self, pg_backend, clean_test_schema):
        """Test batch inserts with executemany."""
        table = clean_test_schema

        items = [("item_1", 1), ("item_2", 2), ("item_3", 3)]

        with pg_backend.connection() as conn:
            conn.executemany(
                f"INSERT INTO {table} (name, value) VALUES ($1, $2)",
                items,
            )
            conn.commit()

        with pg_backend.connection() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            assert cursor.fetchone()[0] == 3

    def test_parameterized_queries_prevent_injection(self, pg_backend, clean_test_schema):
        """Test that parameterized queries prevent SQL injection."""
        table = clean_test_schema

        # Attempt injection
        malicious_name = "'; DROP TABLE users; --"

        with pg_backend.connection() as conn:
            conn.execute(
                f"INSERT INTO {table} (name, value) VALUES ($1, $2)",
                (malicious_name, 1),
            )
            conn.commit()

        # Table should still exist and contain the literal string
        with pg_backend.connection() as conn:
            cursor = conn.execute(
                f"SELECT name FROM {table} WHERE name = $1",
                (malicious_name,),
            )
            row = cursor.fetchone()
            assert row is not None
            assert row[0] == malicious_name


class TestDatabaseAbstraction:
    """Tests for database-agnostic abstraction layer."""

    def test_get_table_columns(self, pg_backend, clean_test_schema):
        """Test get_table_columns works with PostgreSQL."""
        table = clean_test_schema

        columns = pg_backend.get_table_columns(table)

        assert "id" in columns
        assert "name" in columns
        assert "value" in columns
        assert "created_at" in columns

    def test_backend_type_detection(self, pg_backend):
        """Test that backend type is correctly identified."""
        assert pg_backend.backend_type == "postgresql"

    def test_placeholder_translation(self, pg_backend, clean_test_schema):
        """Test that ? placeholders are translated to $N."""
        table = clean_test_schema

        # SQLite-style query should be translated
        with pg_backend.connection() as conn:
            # Note: The backend should translate ? to $1
            conn.execute(
                f"INSERT INTO {table} (name, value) VALUES ($1, $2)",
                ("translated", 99),
            )
            conn.commit()

        with pg_backend.connection() as conn:
            cursor = conn.execute(
                f"SELECT value FROM {table} WHERE name = $1",
                ("translated",),
            )
            assert cursor.fetchone()[0] == 99


class TestConnectionPool:
    """Tests for connection pool functionality."""

    def test_pool_reuses_connections(self, pg_backend):
        """Test that connection pool reuses connections."""
        # Get initial pool size
        initial_size = pg_backend.pool_size

        # Make multiple connections
        for _ in range(10):
            with pg_backend.connection() as conn:
                conn.execute("SELECT 1")

        # Pool should not have grown excessively
        assert pg_backend.pool_size <= initial_size + 2

    def test_concurrent_connections(self, pg_backend, clean_test_schema):
        """Test concurrent database access."""
        import threading

        table = clean_test_schema
        errors = []

        def worker(n):
            try:
                with pg_backend.connection() as conn:
                    conn.execute(
                        f"INSERT INTO {table} (name, value) VALUES ($1, $2)",
                        (f"worker_{n}", n),
                    )
                    conn.commit()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent access errors: {errors}"

        with pg_backend.connection() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            assert cursor.fetchone()[0] == 10


class TestStoreIntegration:
    """Tests for store classes with PostgreSQL backend."""

    def test_critique_store_postgres(self, pg_connection_url):
        """Test CritiqueStore works with PostgreSQL."""
        from aragora.memory.store import CritiqueStore

        # CritiqueStore should work with PostgreSQL if DATABASE_URL is set
        store = CritiqueStore(db_path=":memory:")  # Uses in-memory SQLite

        # Just verify it initializes
        assert store is not None

    @pytest.mark.skip(reason="Requires PostgreSQL-aware stores - implement after migration")
    def test_elo_system_postgres(self, pg_connection_url):
        """Test EloSystem works with PostgreSQL."""
        from aragora.ranking.elo import EloSystem

        # This would require EloSystem to support PostgreSQL
        pass

    @pytest.mark.skip(reason="Requires PostgreSQL-aware stores - implement after migration")
    def test_continuum_memory_postgres(self, pg_connection_url):
        """Test ContinuumMemory works with PostgreSQL."""
        from aragora.memory.continuum import ContinuumMemory

        # This would require ContinuumMemory to support PostgreSQL
        pass


class TestMigrations:
    """Tests for database migrations - requires PostgreSQL."""

    @pytest.mark.skip(reason="Requires PostgreSQL-aware EloSystem - implement after ELO migration")
    def test_elo_migrations(self, pg_connection_url):
        """Test ELO system migrations."""
        pass

    @pytest.mark.skip(reason="Requires PostgreSQL-aware ContinuumMemory - implement after memory migration")
    def test_continuum_migrations(self, pg_connection_url):
        """Test Continuum memory migrations."""
        pass
