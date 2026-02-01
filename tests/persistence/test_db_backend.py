"""
Tests for aragora.persistence.db_backend - Unified Database Backend Adapter.

Tests cover:
- BackendCapabilities factory methods and properties
- SchemaInfo and ColumnInfo data classes
- SQLite type normalization
- UnifiedBackend facade (sync access, capabilities, schema introspection)
- get_unified_backend() factory with environment-based detection
- Backend reset for test isolation
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.persistence.db_backend import (
    BackendCapabilities,
    BackendType,
    ColumnInfo,
    SchemaInfo,
    UnifiedBackend,
    get_unified_backend,
    normalize_sqlite_type,
    reset_unified_backend,
    SQLITE_TO_PG_TYPE_MAP,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_backend():
    """Reset global backend state between tests."""
    reset_unified_backend()
    yield
    reset_unified_backend()


@pytest.fixture
def tmp_sqlite_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with a test table."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            age INTEGER,
            score REAL DEFAULT 0.0,
            active BOOLEAN DEFAULT 1,
            metadata JSON,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        "INSERT INTO users (name, email, age, score) VALUES (?, ?, ?, ?)",
        ("Alice", "alice@example.com", 30, 95.5),
    )
    conn.execute(
        "INSERT INTO users (name, email, age, score) VALUES (?, ?, ?, ?)",
        ("Bob", "bob@example.com", 25, 88.0),
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_sync_backend():
    """Create a mock sync DatabaseBackend."""
    backend = MagicMock()
    backend.backend_type = "sqlite"
    backend.fetch_one = MagicMock(return_value=(42,))
    backend.fetch_all = MagicMock(return_value=[(1, "test")])
    backend.execute_write = MagicMock()
    backend.convert_placeholder = MagicMock(side_effect=lambda sql: sql)
    backend.close = MagicMock()
    backend.get_table_columns = MagicMock(
        return_value=[
            {"name": "id", "type": "INTEGER", "notnull": True, "default": None, "pk": True},
            {"name": "name", "type": "TEXT", "notnull": True, "default": None, "pk": False},
            {"name": "data", "type": "JSON", "notnull": False, "default": "'{}'", "pk": False},
        ]
    )
    return backend


# ===========================================================================
# Test BackendCapabilities
# ===========================================================================


class TestBackendCapabilities:
    """Tests for BackendCapabilities factory methods and properties."""

    def test_sqlite_capabilities(self):
        """SQLite capabilities reflect single-writer limitations."""
        caps = BackendCapabilities.for_sqlite()

        assert caps.backend_type == BackendType.SQLITE
        assert caps.supports_concurrent_writes is False
        assert caps.supports_advisory_locks is False
        assert caps.supports_listen_notify is False
        assert caps.supports_jsonb is False
        assert caps.supports_full_text_search is True
        assert caps.supports_read_replicas is False
        assert caps.supports_savepoints is True
        assert caps.supports_returning is False
        assert caps.has_async_pool is False
        assert caps.max_concurrent_connections == 1

    def test_postgres_capabilities(self):
        """PostgreSQL capabilities reflect production features."""
        caps = BackendCapabilities.for_postgres(has_async_pool=True, max_connections=50)

        assert caps.backend_type == BackendType.POSTGRES
        assert caps.supports_concurrent_writes is True
        assert caps.supports_advisory_locks is True
        assert caps.supports_listen_notify is True
        assert caps.supports_jsonb is True
        assert caps.supports_full_text_search is True
        assert caps.supports_read_replicas is True
        assert caps.supports_savepoints is True
        assert caps.supports_returning is True
        assert caps.has_async_pool is True
        assert caps.max_concurrent_connections == 50

    def test_postgres_default_max_connections(self):
        """Default max connections for PostgreSQL is 20."""
        caps = BackendCapabilities.for_postgres()
        assert caps.max_concurrent_connections == 20

    def test_capabilities_are_frozen(self):
        """BackendCapabilities instances are immutable."""
        caps = BackendCapabilities.for_sqlite()
        with pytest.raises(AttributeError):
            caps.supports_concurrent_writes = True  # type: ignore[misc]


# ===========================================================================
# Test SchemaInfo and ColumnInfo
# ===========================================================================


class TestSchemaInfo:
    """Tests for SchemaInfo and ColumnInfo data classes."""

    def test_column_names(self):
        """column_names returns ordered list of names."""
        schema = SchemaInfo(
            table_name="users",
            columns=[
                ColumnInfo(name="id", type="BIGINT"),
                ColumnInfo(name="name", type="TEXT"),
                ColumnInfo(name="email", type="TEXT"),
            ],
        )
        assert schema.column_names() == ["id", "name", "email"]

    def test_has_column(self):
        """has_column checks column existence."""
        schema = SchemaInfo(
            table_name="users",
            columns=[
                ColumnInfo(name="id", type="BIGINT"),
                ColumnInfo(name="name", type="TEXT"),
            ],
        )
        assert schema.has_column("id") is True
        assert schema.has_column("name") is True
        assert schema.has_column("nonexistent") is False

    def test_empty_schema(self):
        """Empty schema returns empty lists."""
        schema = SchemaInfo(table_name="empty")
        assert schema.column_names() == []
        assert schema.has_column("any") is False

    def test_column_info_defaults(self):
        """ColumnInfo defaults are sensible."""
        col = ColumnInfo(name="test", type="TEXT")
        assert col.nullable is True
        assert col.default is None
        assert col.is_primary_key is False
        assert col.raw_type == ""


# ===========================================================================
# Test normalize_sqlite_type
# ===========================================================================


class TestNormalizeSqliteType:
    """Tests for SQLite type normalization."""

    def test_direct_mappings(self):
        """Standard types map correctly."""
        assert normalize_sqlite_type("TEXT") == "TEXT"
        assert normalize_sqlite_type("INTEGER") == "BIGINT"
        assert normalize_sqlite_type("REAL") == "DOUBLE PRECISION"
        assert normalize_sqlite_type("BLOB") == "BYTEA"
        assert normalize_sqlite_type("BOOLEAN") == "BOOLEAN"

    def test_case_insensitive(self):
        """Type normalization is case-insensitive."""
        assert normalize_sqlite_type("text") == "TEXT"
        assert normalize_sqlite_type("Integer") == "BIGINT"
        assert normalize_sqlite_type("REAL") == "DOUBLE PRECISION"

    def test_parenthesized_lengths(self):
        """Types with lengths strip the length specifier."""
        assert normalize_sqlite_type("VARCHAR(255)") == "TEXT"
        assert normalize_sqlite_type("CHAR(10)") == "TEXT"
        assert normalize_sqlite_type("NUMERIC(10,2)") == "NUMERIC"

    def test_affinity_rules(self):
        """SQLite affinity rules are applied correctly."""
        # INT affinity
        assert normalize_sqlite_type("BIGINT") == "BIGINT"
        assert normalize_sqlite_type("SMALLINT") == "BIGINT"
        assert normalize_sqlite_type("UNSIGNED BIG INT") == "BIGINT"
        assert normalize_sqlite_type("TINYINT") == "BIGINT"

        # TEXT affinity
        assert normalize_sqlite_type("CLOB") == "TEXT"
        assert normalize_sqlite_type("NVARCHAR(100)") == "TEXT"

        # REAL affinity
        assert normalize_sqlite_type("FLOAT") == "DOUBLE PRECISION"
        assert normalize_sqlite_type("DOUBLE") == "DOUBLE PRECISION"

    def test_timestamp_types(self):
        """Timestamp-like types map to TIMESTAMPTZ."""
        assert normalize_sqlite_type("DATETIME") == "TIMESTAMPTZ"
        assert normalize_sqlite_type("TIMESTAMP") == "TIMESTAMPTZ"

    def test_json_types(self):
        """JSON types map to JSONB."""
        assert normalize_sqlite_type("JSON") == "JSONB"
        assert normalize_sqlite_type("JSONB") == "JSONB"

    def test_empty_type(self):
        """Empty type defaults to TEXT."""
        assert normalize_sqlite_type("") == "TEXT"

    def test_unknown_type(self):
        """Unknown types default to TEXT."""
        assert normalize_sqlite_type("SOMECUSTOMTYPE") == "TEXT"

    def test_all_map_entries_are_valid(self):
        """All entries in the type map produce non-empty results."""
        for sqlite_type, pg_type in SQLITE_TO_PG_TYPE_MAP.items():
            assert pg_type, f"Empty mapping for {sqlite_type}"
            # Verify the map agrees with the function
            assert normalize_sqlite_type(sqlite_type) == pg_type


# ===========================================================================
# Test UnifiedBackend
# ===========================================================================


class TestUnifiedBackend:
    """Tests for the UnifiedBackend facade."""

    def test_backend_type_property(self, mock_sync_backend):
        """backend_type returns correct value."""
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(mock_sync_backend, caps)
        assert backend.backend_type == BackendType.SQLITE

    def test_is_sqlite(self, mock_sync_backend):
        """is_sqlite property works correctly."""
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(mock_sync_backend, caps)
        assert backend.is_sqlite is True
        assert backend.is_postgres is False

    def test_is_postgres(self, mock_sync_backend):
        """is_postgres property works correctly."""
        caps = BackendCapabilities.for_postgres()
        backend = UnifiedBackend(mock_sync_backend, caps)
        assert backend.is_postgres is True
        assert backend.is_sqlite is False

    def test_sync_access(self, mock_sync_backend):
        """Sync backend is accessible through .sync attribute."""
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(mock_sync_backend, caps)

        result = backend.sync.fetch_one("SELECT 1")
        assert result == (42,)
        mock_sync_backend.fetch_one.assert_called_once_with("SELECT 1")

    def test_translate_sql(self, mock_sync_backend):
        """translate_sql delegates to sync backend."""
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(mock_sync_backend, caps)

        backend.translate_sql("SELECT * FROM t WHERE id = ?")
        mock_sync_backend.convert_placeholder.assert_called_once()

    def test_get_schema_info_sqlite(self, mock_sync_backend):
        """get_schema_info returns normalized schema for SQLite."""
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(mock_sync_backend, caps)

        schema = backend.get_schema_info("test_table")

        assert schema.table_name == "test_table"
        assert len(schema.columns) == 3
        assert schema.columns[0].name == "id"
        assert schema.columns[0].type == "BIGINT"  # INTEGER -> BIGINT
        assert schema.columns[0].is_primary_key is True
        assert schema.columns[1].name == "name"
        assert schema.columns[1].type == "TEXT"
        assert schema.columns[2].name == "data"
        assert schema.columns[2].type == "JSONB"  # JSON -> JSONB

    def test_get_schema_info_postgres(self, mock_sync_backend):
        """get_schema_info returns uppercase types for PostgreSQL."""
        caps = BackendCapabilities.for_postgres()
        backend = UnifiedBackend(mock_sync_backend, caps)

        schema = backend.get_schema_info("test_table")

        # PostgreSQL types should be uppercased but not translated
        assert schema.columns[0].type == "INTEGER"
        assert schema.columns[1].type == "TEXT"

    @pytest.mark.asyncio
    async def test_get_async_pool_none_for_sqlite(self, mock_sync_backend):
        """get_async_pool returns None for SQLite backend."""
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(mock_sync_backend, caps)

        pool = await backend.get_async_pool()
        assert pool is None

    @pytest.mark.asyncio
    async def test_get_async_pool_with_factory(self, mock_sync_backend):
        """get_async_pool calls factory and caches result."""
        mock_pool = MagicMock()
        factory_called = 0

        async def mock_factory():
            nonlocal factory_called
            factory_called += 1
            return mock_pool

        caps = BackendCapabilities.for_postgres(has_async_pool=True)
        backend = UnifiedBackend(mock_sync_backend, caps, async_pool_factory=mock_factory)

        # First call creates pool
        pool = await backend.get_async_pool()
        assert pool is mock_pool
        assert factory_called == 1

        # Second call returns cached pool
        pool2 = await backend.get_async_pool()
        assert pool2 is mock_pool
        assert factory_called == 1  # Not called again

    @pytest.mark.asyncio
    async def test_get_async_pool_handles_factory_error(self, mock_sync_backend):
        """get_async_pool returns None if factory fails."""

        async def failing_factory():
            raise ConnectionError("Cannot connect")

        caps = BackendCapabilities.for_postgres(has_async_pool=True)
        backend = UnifiedBackend(mock_sync_backend, caps, async_pool_factory=failing_factory)

        pool = await backend.get_async_pool()
        assert pool is None

    def test_close(self, mock_sync_backend):
        """close() closes the sync backend."""
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(mock_sync_backend, caps)

        backend.close()
        mock_sync_backend.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_async(self, mock_sync_backend):
        """close_async() closes the async pool."""
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()

        caps = BackendCapabilities.for_postgres()
        backend = UnifiedBackend(mock_sync_backend, caps)
        backend._async_pool = mock_pool

        await backend.close_async()
        mock_pool.close.assert_called_once()
        assert backend._async_pool is None

    def test_repr(self, mock_sync_backend):
        """__repr__ includes type and pool status."""
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(mock_sync_backend, caps)

        repr_str = repr(backend)
        assert "sqlite" in repr_str
        assert "pool=no" in repr_str


# ===========================================================================
# Test get_unified_backend factory
# ===========================================================================


class TestGetUnifiedBackend:
    """Tests for the get_unified_backend factory function."""

    def test_default_creates_sqlite(self, tmp_path: Path):
        """Without PostgreSQL config, defaults to SQLite."""
        db_path = str(tmp_path / "test.db")
        with patch.dict(os.environ, {}, clear=True):
            # Ensure no PostgreSQL env vars
            for key in [
                "DATABASE_URL",
                "ARAGORA_POSTGRES_DSN",
                "SUPABASE_POSTGRES_DSN",
                "ARAGORA_DB_BACKEND",
            ]:
                os.environ.pop(key, None)

            backend = get_unified_backend(force_backend=BackendType.SQLITE, db_path=db_path)

        assert backend.is_sqlite is True
        assert backend.capabilities.supports_concurrent_writes is False
        backend.close()

    def test_force_sqlite(self, tmp_path: Path):
        """force_backend=SQLITE always creates SQLite."""
        db_path = str(tmp_path / "forced.db")
        backend = get_unified_backend(force_backend=BackendType.SQLITE, db_path=db_path)

        assert backend.is_sqlite is True
        backend.close()

    def test_env_override_sqlite(self, tmp_path: Path):
        """ARAGORA_DB_BACKEND=sqlite forces SQLite."""
        db_path = str(tmp_path / "env.db")
        with patch.dict(
            os.environ,
            {"ARAGORA_DB_BACKEND": "sqlite", "ARAGORA_DATA_DIR": str(tmp_path)},
        ):
            reset_unified_backend()
            backend = get_unified_backend(db_path=db_path)

        assert backend.is_sqlite is True
        backend.close()

    def test_singleton_behavior(self, tmp_path: Path):
        """get_unified_backend returns the same instance on repeated calls."""
        db_path = str(tmp_path / "singleton.db")
        backend1 = get_unified_backend(force_backend=BackendType.SQLITE, db_path=db_path)
        backend2 = get_unified_backend()

        assert backend1 is backend2
        backend1.close()

    def test_reset_clears_singleton(self, tmp_path: Path):
        """reset_unified_backend clears the cached instance."""
        db_path = str(tmp_path / "reset.db")
        backend1 = get_unified_backend(force_backend=BackendType.SQLITE, db_path=db_path)
        reset_unified_backend()

        db_path2 = str(tmp_path / "reset2.db")
        backend2 = get_unified_backend(force_backend=BackendType.SQLITE, db_path=db_path2)

        assert backend1 is not backend2
        backend2.close()

    def test_postgres_fallback_on_no_dsn(self, tmp_path: Path):
        """Requesting PostgreSQL without DSN falls back to SQLite."""
        with patch.dict(os.environ, {"ARAGORA_DATA_DIR": str(tmp_path)}, clear=False):
            # Remove all PG config
            for key in [
                "DATABASE_URL",
                "ARAGORA_POSTGRES_DSN",
                "SUPABASE_POSTGRES_DSN",
                "SUPABASE_URL",
            ]:
                os.environ.pop(key, None)

            reset_unified_backend()
            backend = get_unified_backend(force_backend=BackendType.POSTGRES)

        # Should fall back to SQLite since no DSN is available
        assert backend.is_sqlite is True
        backend.close()


# ===========================================================================
# Integration test with real SQLite
# ===========================================================================


class TestUnifiedBackendIntegration:
    """Integration tests using a real SQLite database."""

    def test_schema_introspection(self, tmp_sqlite_db: Path):
        """Schema introspection works with a real SQLite database."""
        from aragora.storage.backends import SQLiteBackend

        sync = SQLiteBackend(str(tmp_sqlite_db))
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(sync, caps)

        schema = backend.get_schema_info("users")

        assert schema.table_name == "users"
        assert schema.has_column("id")
        assert schema.has_column("name")
        assert schema.has_column("email")
        assert schema.has_column("age")
        assert schema.has_column("score")
        assert schema.has_column("active")

        # Check type translations
        id_col = next(c for c in schema.columns if c.name == "id")
        assert id_col.type == "BIGINT"
        assert id_col.is_primary_key is True

        name_col = next(c for c in schema.columns if c.name == "name")
        assert name_col.type == "TEXT"
        assert name_col.nullable is False

        score_col = next(c for c in schema.columns if c.name == "score")
        assert score_col.type == "DOUBLE PRECISION"

        backend.close()

    def test_sync_operations(self, tmp_sqlite_db: Path):
        """Sync operations work through the unified backend."""
        from aragora.storage.backends import SQLiteBackend

        sync = SQLiteBackend(str(tmp_sqlite_db))
        caps = BackendCapabilities.for_sqlite()
        backend = UnifiedBackend(sync, caps)

        # Read
        row = backend.sync.fetch_one("SELECT COUNT(*) FROM users")
        assert row[0] == 2

        rows = backend.sync.fetch_all("SELECT name FROM users ORDER BY name")
        names = [r[0] for r in rows]
        assert names == ["Alice", "Bob"]

        # Write
        backend.sync.execute_write(
            "INSERT INTO users (name, email, age, score) VALUES (?, ?, ?, ?)",
            ("Charlie", "charlie@example.com", 35, 92.0),
        )
        row = backend.sync.fetch_one("SELECT COUNT(*) FROM users")
        assert row[0] == 3

        backend.close()
