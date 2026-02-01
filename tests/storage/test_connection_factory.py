"""Tests for aragora.storage.connection_factory - Database connection factory.

Tests cover:
- StorageBackendType enum
- Database configuration resolution
- Supabase PostgreSQL DSN derivation
- Self-hosted PostgreSQL DSN detection
- Connection pooling (mocked)
- Connection reuse (pool caching)
- Error handling (missing configuration)
- Connection cleanup
- Environment detection (production/development)
- create_persistent_store helper
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.storage.connection_factory import (
    DatabaseConfig,
    StorageBackendType,
    close_all_pools,
    create_persistent_store,
    get_database_pool,
    get_database_pool_sync,
    get_selfhosted_postgres_dsn,
    get_supabase_postgres_dsn,
    is_production_environment,
    reset_pools,
    resolve_database_config,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_pool_state():
    """Reset pool state before and after each test."""
    reset_pools()
    yield
    reset_pools()


@pytest.fixture(autouse=True)
def clean_env():
    """Clean up environment variables that may interfere with tests."""
    env_vars_to_clear = [
        "ARAGORA_DB_BACKEND",
        "ARAGORA_POSTGRES_DSN",
        "DATABASE_URL",
        "SUPABASE_URL",
        "SUPABASE_DB_PASSWORD",
        "SUPABASE_POSTGRES_DSN",
        "SUPABASE_KEY",
        "ARAGORA_ENV",
        "ARAGORA_ALLOW_SQLITE_FALLBACK",
        "ARAGORA_REQUIRE_DISTRIBUTED",
    ]
    saved = {k: os.environ.get(k) for k in env_vars_to_clear}
    for k in env_vars_to_clear:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


@pytest.fixture
def mock_asyncpg_pool():
    """Create a mock asyncpg pool."""
    pool = MagicMock()
    pool.get_size.return_value = 10
    pool.get_idle_size.return_value = 5
    pool.close = AsyncMock()
    pool.terminate = MagicMock()

    mock_conn = MagicMock()
    mock_conn.fetchval = AsyncMock(return_value=1)

    async_cm = AsyncMock()
    async_cm.__aenter__.return_value = mock_conn
    async_cm.__aexit__.return_value = None
    pool.acquire.return_value = async_cm

    return pool


# ===========================================================================
# Test: StorageBackendType Enum
# ===========================================================================


class TestStorageBackendTypeEnum:
    """Tests for StorageBackendType enum values."""

    def test_supabase_value(self):
        """SUPABASE should have correct string value."""
        assert StorageBackendType.SUPABASE.value == "supabase"

    def test_postgres_value(self):
        """POSTGRES should have correct string value."""
        assert StorageBackendType.POSTGRES.value == "postgres"

    def test_sqlite_value(self):
        """SQLITE should have correct string value."""
        assert StorageBackendType.SQLITE.value == "sqlite"

    def test_enum_is_str(self):
        """Enum values should be usable as strings."""
        assert str(StorageBackendType.SUPABASE) == "StorageBackendType.SUPABASE"
        assert StorageBackendType.SUPABASE == "supabase"


# ===========================================================================
# Test: SQLite Connection (resolve_database_config)
# ===========================================================================


class TestSQLiteConnection:
    """Tests for SQLite database connection configuration."""

    def test_default_fallback_to_sqlite(self):
        """Should fall back to SQLite when no PostgreSQL is configured."""
        config = resolve_database_config("test_store", allow_sqlite=True)
        assert config.backend_type == StorageBackendType.SQLITE
        assert config.dsn is None
        assert config.is_supabase is False

    def test_explicit_sqlite_override(self):
        """Should use SQLite when explicitly requested."""
        with patch.dict(os.environ, {"ARAGORA_TEST_STORE_BACKEND": "sqlite"}):
            config = resolve_database_config("test_store", allow_sqlite=True)
            assert config.backend_type == StorageBackendType.SQLITE

    def test_sqlite_not_allowed_raises(self):
        """Should raise error when SQLite not allowed and no PostgreSQL."""
        with pytest.raises(RuntimeError) as exc_info:
            resolve_database_config("test_store", allow_sqlite=False)
        assert "No suitable database backend" in str(exc_info.value)

    def test_explicit_sqlite_when_not_allowed_raises(self):
        """Should raise when SQLite explicitly requested but not allowed."""
        with patch.dict(os.environ, {"ARAGORA_TEST_STORE_BACKEND": "sqlite"}):
            with pytest.raises(RuntimeError) as exc_info:
                resolve_database_config("test_store", allow_sqlite=False)
            assert "SQLite not allowed" in str(exc_info.value)


# ===========================================================================
# Test: PostgreSQL Connection (mock)
# ===========================================================================


class TestPostgreSQLConnection:
    """Tests for PostgreSQL database connection configuration (mocked)."""

    def test_selfhosted_postgres_via_dsn(self):
        """Should detect self-hosted PostgreSQL from ARAGORA_POSTGRES_DSN."""
        dsn = "postgresql://user:pass@localhost:5432/testdb"
        with patch.dict(os.environ, {"ARAGORA_POSTGRES_DSN": dsn}):
            config = resolve_database_config("test_store")
            assert config.backend_type == StorageBackendType.POSTGRES
            assert config.dsn == dsn
            assert config.is_supabase is False

    def test_selfhosted_postgres_via_database_url(self):
        """Should detect PostgreSQL from DATABASE_URL (PaaS convention)."""
        dsn = "postgresql://user:pass@db.railway.app:5432/app"
        with patch.dict(os.environ, {"DATABASE_URL": dsn}):
            config = resolve_database_config("test_store")
            assert config.backend_type == StorageBackendType.POSTGRES
            assert config.dsn == dsn
            assert config.is_supabase is False

    def test_aragora_dsn_takes_precedence_over_database_url(self):
        """ARAGORA_POSTGRES_DSN should take precedence over DATABASE_URL."""
        primary_dsn = "postgresql://primary:pass@localhost/db1"
        fallback_dsn = "postgresql://fallback:pass@localhost/db2"
        with patch.dict(
            os.environ,
            {"ARAGORA_POSTGRES_DSN": primary_dsn, "DATABASE_URL": fallback_dsn},
        ):
            config = resolve_database_config("test_store")
            assert config.dsn == primary_dsn

    def test_explicit_postgres_backend_requires_dsn(self):
        """Explicit postgres backend without DSN should fall back to auto-detect."""
        with patch.dict(os.environ, {"ARAGORA_TEST_STORE_BACKEND": "postgres"}):
            config = resolve_database_config("test_store", allow_sqlite=True)
            # Falls back to SQLite since no DSN is configured
            assert config.backend_type == StorageBackendType.SQLITE

    def test_explicit_postgres_without_dsn_and_no_sqlite_raises(self):
        """Explicit postgres without DSN and no SQLite allowed should raise."""
        with patch.dict(os.environ, {"ARAGORA_TEST_STORE_BACKEND": "postgres"}):
            with pytest.raises(RuntimeError) as exc_info:
                resolve_database_config("test_store", allow_sqlite=False)
            assert "ARAGORA_POSTGRES_DSN" in str(exc_info.value)


# ===========================================================================
# Test: Connection Pooling
# ===========================================================================


class TestConnectionPooling:
    """Tests for database connection pooling."""

    @pytest.mark.asyncio
    async def test_get_database_pool_returns_none_for_sqlite(self):
        """get_database_pool should return None pool for SQLite backend."""
        pool, config = await get_database_pool("test_store", allow_sqlite=True)
        assert pool is None
        assert config.backend_type == StorageBackendType.SQLITE

    @pytest.mark.asyncio
    async def test_get_database_pool_creates_pool_for_postgres(self, mock_asyncpg_pool):
        """get_database_pool should create pool for PostgreSQL backend."""
        dsn = "postgresql://user:pass@localhost:5432/testdb"

        with (
            patch.dict(os.environ, {"ARAGORA_POSTGRES_DSN": dsn}),
            patch(
                "aragora.storage.postgres_store.get_postgres_pool",
                new=AsyncMock(return_value=mock_asyncpg_pool),
            ),
        ):
            pool, config = await get_database_pool("test_store")
            assert pool is mock_asyncpg_pool
            assert config.backend_type == StorageBackendType.POSTGRES
            assert config.dsn == dsn

    def test_get_database_pool_sync_returns_config_for_sqlite(self):
        """Synchronous pool getter should work for SQLite."""
        pool, config = get_database_pool_sync("test_store", allow_sqlite=True)
        assert pool is None
        assert config.backend_type == StorageBackendType.SQLITE


# ===========================================================================
# Test: Connection Reuse
# ===========================================================================


class TestConnectionReuse:
    """Tests for connection pool caching and reuse."""

    @pytest.mark.asyncio
    async def test_pool_is_cached(self, mock_asyncpg_pool):
        """Same pool should be returned on subsequent calls."""
        dsn = "postgresql://user:pass@localhost:5432/testdb"

        with (
            patch.dict(os.environ, {"ARAGORA_POSTGRES_DSN": dsn}),
            patch(
                "aragora.storage.postgres_store.get_postgres_pool",
                new=AsyncMock(return_value=mock_asyncpg_pool),
            ) as mock_get_pool,
        ):
            pool1, _ = await get_database_pool("store1")
            pool2, _ = await get_database_pool("store2")

            # Both should get the same cached pool
            assert pool1 is pool2
            # get_postgres_pool should only be called once
            assert mock_get_pool.call_count == 1

    @pytest.mark.asyncio
    async def test_reset_pools_clears_cache(self, mock_asyncpg_pool):
        """reset_pools should clear the pool cache."""
        dsn = "postgresql://user:pass@localhost:5432/testdb"

        call_count = 0

        async def mock_get_pool(**kwargs):
            nonlocal call_count
            call_count += 1
            return mock_asyncpg_pool

        with (
            patch.dict(os.environ, {"ARAGORA_POSTGRES_DSN": dsn}),
            patch(
                "aragora.storage.postgres_store.get_postgres_pool",
                new=mock_get_pool,
            ),
        ):
            await get_database_pool("store1")
            assert call_count == 1

            reset_pools()

            await get_database_pool("store2")
            assert call_count == 2  # New pool created after reset


# ===========================================================================
# Test: Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in connection factory."""

    def test_unknown_backend_falls_back_to_autodetect(self):
        """Unknown backend type should fall back to auto-detection."""
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "mysql"}):
            config = resolve_database_config("test_store", allow_sqlite=True)
            # Falls back to SQLite since no supported backend is configured
            assert config.backend_type == StorageBackendType.SQLITE

    def test_memory_backend_falls_back_to_autodetect(self):
        """Memory backend (non-persistent) should fall back to auto-detection."""
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "memory"}):
            config = resolve_database_config("test_store", allow_sqlite=True)
            assert config.backend_type == StorageBackendType.SQLITE

    def test_redis_backend_falls_back_to_autodetect(self):
        """Redis backend (non-persistent) should fall back to auto-detection."""
        with patch.dict(os.environ, {"ARAGORA_DB_BACKEND": "redis"}):
            config = resolve_database_config("test_store", allow_sqlite=True)
            assert config.backend_type == StorageBackendType.SQLITE


# ===========================================================================
# Test: Connection Cleanup
# ===========================================================================


class TestConnectionCleanup:
    """Tests for connection cleanup and pool closing."""

    @pytest.mark.asyncio
    async def test_close_all_pools(self, mock_asyncpg_pool):
        """close_all_pools should close cached pools."""
        dsn = "postgresql://user:pass@localhost:5432/testdb"

        with (
            patch.dict(os.environ, {"ARAGORA_POSTGRES_DSN": dsn}),
            patch(
                "aragora.storage.postgres_store.get_postgres_pool",
                new=AsyncMock(return_value=mock_asyncpg_pool),
            ),
        ):
            await get_database_pool("test_store")
            await close_all_pools()

            mock_asyncpg_pool.close.assert_called_once()


# ===========================================================================
# Test: Supabase PostgreSQL DSN
# ===========================================================================


class TestSupabasePostgresDSN:
    """Tests for Supabase PostgreSQL DSN derivation."""

    def test_explicit_supabase_dsn(self):
        """Should use explicit SUPABASE_POSTGRES_DSN when provided."""
        dsn = "postgresql://postgres:secret@db.project.supabase.co:5432/postgres"
        with patch.dict(os.environ, {"SUPABASE_POSTGRES_DSN": dsn}):
            result = get_supabase_postgres_dsn()
            assert result == dsn

    def test_derived_from_url_and_password(self):
        """Should derive DSN from SUPABASE_URL and SUPABASE_DB_PASSWORD."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://myproject.supabase.co",
                "SUPABASE_DB_PASSWORD": "secretpass",
            },
        ):
            result = get_supabase_postgres_dsn()
            assert result is not None
            assert "myproject" in result
            assert "secretpass" in result
            assert result.startswith("postgresql://")

    def test_returns_none_without_password(self):
        """Should return None if password is missing."""
        with patch.dict(os.environ, {"SUPABASE_URL": "https://myproject.supabase.co"}):
            result = get_supabase_postgres_dsn()
            assert result is None

    def test_returns_none_without_url(self):
        """Should return None if URL is missing."""
        with patch.dict(os.environ, {"SUPABASE_DB_PASSWORD": "secretpass"}):
            result = get_supabase_postgres_dsn()
            assert result is None

    def test_supabase_backend_config(self):
        """Supabase backend should set is_supabase=True."""
        dsn = "postgresql://postgres:secret@db.project.supabase.co:5432/postgres"
        with patch.dict(os.environ, {"SUPABASE_POSTGRES_DSN": dsn}):
            config = resolve_database_config("test_store")
            assert config.backend_type == StorageBackendType.SUPABASE
            assert config.is_supabase is True
            assert config.dsn == dsn

    def test_supabase_takes_precedence_over_selfhosted(self):
        """Supabase should be preferred over self-hosted PostgreSQL."""
        supabase_dsn = "postgresql://postgres:secret@db.project.supabase.co:5432/postgres"
        selfhosted_dsn = "postgresql://user:pass@localhost:5432/db"
        with patch.dict(
            os.environ,
            {"SUPABASE_POSTGRES_DSN": supabase_dsn, "ARAGORA_POSTGRES_DSN": selfhosted_dsn},
        ):
            config = resolve_database_config("test_store")
            assert config.backend_type == StorageBackendType.SUPABASE
            assert config.dsn == supabase_dsn


# ===========================================================================
# Test: Self-hosted PostgreSQL DSN
# ===========================================================================


class TestSelfhostedPostgresDSN:
    """Tests for self-hosted PostgreSQL DSN detection."""

    def test_aragora_postgres_dsn(self):
        """Should return ARAGORA_POSTGRES_DSN when set."""
        dsn = "postgresql://user:pass@localhost:5432/mydb"
        with patch.dict(os.environ, {"ARAGORA_POSTGRES_DSN": dsn}):
            result = get_selfhosted_postgres_dsn()
            assert result == dsn

    def test_database_url_fallback(self):
        """Should return DATABASE_URL as fallback."""
        dsn = "postgresql://user:pass@db.example.com:5432/app"
        with patch.dict(os.environ, {"DATABASE_URL": dsn}):
            result = get_selfhosted_postgres_dsn()
            assert result == dsn

    def test_returns_none_when_not_configured(self):
        """Should return None when no DSN is configured."""
        result = get_selfhosted_postgres_dsn()
        assert result is None


# ===========================================================================
# Test: Production Environment Detection
# ===========================================================================


class TestProductionEnvironment:
    """Tests for production environment detection."""

    def test_development_is_not_production(self):
        """Development environment should not be detected as production."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            assert is_production_environment() is False

    def test_production_is_production(self):
        """Production environment should be detected."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            assert is_production_environment() is True

    def test_prod_shorthand_is_production(self):
        """Prod shorthand should be detected as production."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "prod"}):
            assert is_production_environment() is True

    def test_staging_is_production(self):
        """Staging environment should be treated as production."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "staging"}):
            assert is_production_environment() is True

    def test_live_is_production(self):
        """Live environment should be treated as production."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "live"}):
            assert is_production_environment() is True

    def test_default_is_not_production(self):
        """Default (no ARAGORA_ENV) should not be production."""
        assert is_production_environment() is False


# ===========================================================================
# Test: DatabaseConfig dataclass
# ===========================================================================


class TestDatabaseConfig:
    """Tests for DatabaseConfig dataclass."""

    def test_config_creation(self):
        """Should create DatabaseConfig with all fields."""
        config = DatabaseConfig(
            backend_type=StorageBackendType.POSTGRES,
            dsn="postgresql://localhost/db",
            is_supabase=False,
            pool=None,
        )
        assert config.backend_type == StorageBackendType.POSTGRES
        assert config.dsn == "postgresql://localhost/db"
        assert config.is_supabase is False
        assert config.pool is None

    def test_config_with_pool(self):
        """Should store pool reference in config."""
        mock_pool = MagicMock()
        config = DatabaseConfig(
            backend_type=StorageBackendType.SUPABASE,
            dsn="postgresql://db.supabase.co/postgres",
            is_supabase=True,
            pool=mock_pool,
        )
        assert config.pool is mock_pool


# ===========================================================================
# Test: create_persistent_store helper
# ===========================================================================


class TestCreatePersistentStore:
    """Tests for create_persistent_store helper function."""

    def test_creates_sqlite_store_as_fallback(self, tmp_path):
        """Should create SQLite store when no PostgreSQL is configured."""

        class MockSQLiteStore:
            def __init__(self, db_path):
                self.db_path = db_path

        class MockPostgresStore:
            def __init__(self, pool):
                self.pool = pool

        with patch.dict(os.environ, {"ARAGORA_DATA_DIR": str(tmp_path)}):
            store = create_persistent_store(
                "test_store",
                MockSQLiteStore,
                MockPostgresStore,
                "test.db",
            )
            assert isinstance(store, MockSQLiteStore)
            assert str(store.db_path).endswith("test.db")

    def test_creates_memory_store_when_requested(self):
        """Should create memory store when MEMORY backend is requested."""

        class MockSQLiteStore:
            pass

        class MockPostgresStore:
            pass

        class MockMemoryStore:
            pass

        with patch.dict(os.environ, {"ARAGORA_TEST_STORE_BACKEND": "memory"}):
            store = create_persistent_store(
                "test_store",
                MockSQLiteStore,
                MockPostgresStore,
                "test.db",
                memory_class=MockMemoryStore,
            )
            assert isinstance(store, MockMemoryStore)


# ===========================================================================
# Test: Backend Override
# ===========================================================================


class TestBackendOverride:
    """Tests for per-store backend overrides."""

    def test_per_store_backend_override(self):
        """Per-store backend should override global setting."""
        with patch.dict(
            os.environ,
            {
                "ARAGORA_DB_BACKEND": "supabase",
                "ARAGORA_MY_STORE_BACKEND": "sqlite",
            },
        ):
            config = resolve_database_config("my_store", allow_sqlite=True)
            assert config.backend_type == StorageBackendType.SQLITE

    def test_legacy_store_backend_naming(self):
        """Legacy ARAGORA_<STORE>_STORE_BACKEND should work."""
        with patch.dict(os.environ, {"ARAGORA_MY_STORE_STORE_BACKEND": "sqlite"}):
            config = resolve_database_config("my_store", allow_sqlite=True)
            assert config.backend_type == StorageBackendType.SQLITE

    def test_global_backend_applies_to_all_stores(self):
        """Global ARAGORA_DB_BACKEND should apply to stores without override."""
        dsn = "postgresql://user:pass@localhost/db"
        with patch.dict(
            os.environ,
            {"ARAGORA_DB_BACKEND": "postgres", "ARAGORA_POSTGRES_DSN": dsn},
        ):
            config = resolve_database_config("any_store")
            assert config.backend_type == StorageBackendType.POSTGRES
