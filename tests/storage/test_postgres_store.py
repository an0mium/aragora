"""
Tests for PostgreSQL store implementation.

Tests cover:
- Pool management (get_postgres_pool, close_postgres_pool)
- PostgresStore base class
- Schema versioning and migrations
- CRUD helper methods
- Table/column name injection prevention
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import aragora.storage.postgres_store as module


# =============================================================================
# Test Setup
# =============================================================================


@pytest.fixture
def mock_asyncpg_available():
    """Temporarily enable asyncpg availability for testing."""
    original = module.ASYNCPG_AVAILABLE
    module.ASYNCPG_AVAILABLE = True
    yield
    module.ASYNCPG_AVAILABLE = original


@pytest.fixture
def mock_pool():
    """Create a mock connection pool."""
    pool = MagicMock()
    pool.get_size.return_value = 10
    pool.get_min_size.return_value = 5
    pool.get_max_size.return_value = 20
    pool.get_idle_size.return_value = 8
    return pool


@pytest.fixture
def mock_connection():
    """Create a mock database connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.executemany = AsyncMock()
    return conn


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_pool_acquire_timeout(self):
        """POOL_ACQUIRE_TIMEOUT should be positive."""
        from aragora.storage.postgres_store import POOL_ACQUIRE_TIMEOUT

        assert POOL_ACQUIRE_TIMEOUT > 0
        assert isinstance(POOL_ACQUIRE_TIMEOUT, float)

    def test_pool_backpressure_threshold(self):
        """POOL_BACKPRESSURE_THRESHOLD should be between 0 and 1."""
        from aragora.storage.postgres_store import POOL_BACKPRESSURE_THRESHOLD

        assert 0 < POOL_BACKPRESSURE_THRESHOLD < 1

    def test_pool_circuit_breaker_threshold(self):
        """POOL_CIRCUIT_BREAKER_THRESHOLD should be positive integer."""
        from aragora.storage.postgres_store import POOL_CIRCUIT_BREAKER_THRESHOLD

        assert POOL_CIRCUIT_BREAKER_THRESHOLD > 0
        assert isinstance(POOL_CIRCUIT_BREAKER_THRESHOLD, int)

    def test_pool_circuit_breaker_cooldown(self):
        """POOL_CIRCUIT_BREAKER_COOLDOWN should be positive."""
        from aragora.storage.postgres_store import POOL_CIRCUIT_BREAKER_COOLDOWN

        assert POOL_CIRCUIT_BREAKER_COOLDOWN > 0


# =============================================================================
# PoolMetrics Tests
# =============================================================================


class TestPoolMetricsDataclass:
    """Tests for PoolMetrics dataclass."""

    def test_pool_metrics_construction(self):
        """PoolMetrics can be constructed with all fields."""
        from aragora.storage.postgres_store import PoolMetrics

        metrics = PoolMetrics(
            pool_size=15,
            pool_min_size=5,
            pool_max_size=20,
            free_connections=5,
            used_connections=10,
            utilization=0.5,
            backpressure=False,
            total_acquisitions=100,
            failed_acquisitions=5,
            timeouts=2,
            circuit_breaker_rejections=0,
            circuit_breaker_status="closed",
            avg_wait_time_ms=5.0,
            max_wait_time_ms=50.0,
        )

        assert metrics.pool_size == 15
        assert metrics.utilization == 0.5
        assert metrics.backpressure is False
        assert metrics.circuit_breaker_status == "closed"


# =============================================================================
# Pool Management Tests
# =============================================================================


class TestGetPostgresPool:
    """Tests for get_postgres_pool function."""

    @pytest.mark.asyncio
    async def test_raises_when_asyncpg_not_available(self):
        """Should raise RuntimeError when asyncpg is not installed."""
        from aragora.storage.postgres_store import get_postgres_pool

        original = module.ASYNCPG_AVAILABLE
        module.ASYNCPG_AVAILABLE = False

        try:
            with pytest.raises(RuntimeError, match="asyncpg"):
                await get_postgres_pool(dsn="postgresql://localhost/test")
        finally:
            module.ASYNCPG_AVAILABLE = original

    @pytest.mark.asyncio
    async def test_returns_existing_pool(self, mock_asyncpg_available):
        """Should return existing pool if already created."""
        from aragora.storage.postgres_store import get_postgres_pool

        # Set up existing pool
        mock_pool = MagicMock()
        original_pool = module._pool
        module._pool = mock_pool

        try:
            result = await get_postgres_pool()
            assert result is mock_pool
        finally:
            module._pool = original_pool

    @pytest.mark.asyncio
    async def test_raises_when_no_dsn_configured(self, mock_asyncpg_available):
        """Should raise RuntimeError when DSN not configured."""
        from aragora.storage.postgres_store import get_postgres_pool

        # Clear pool and environment
        original_pool = module._pool
        module._pool = None

        with patch.dict("os.environ", {}, clear=True):
            with patch.object(module, "asyncpg", MagicMock()) as mock_asyncpg:
                mock_asyncpg.create_pool = AsyncMock()

                try:
                    with pytest.raises(RuntimeError, match="DSN not configured"):
                        await get_postgres_pool(dsn=None)
                finally:
                    module._pool = original_pool

    @pytest.mark.asyncio
    async def test_uses_env_var_for_dsn(self, mock_asyncpg_available):
        """Should use ARAGORA_POSTGRES_DSN environment variable."""
        from aragora.storage.postgres_store import get_postgres_pool

        original_pool = module._pool
        module._pool = None

        mock_created_pool = MagicMock()
        with patch.dict("os.environ", {"ARAGORA_POSTGRES_DSN": "postgresql://test"}):
            with patch.object(module, "asyncpg") as mock_asyncpg:
                mock_asyncpg.create_pool = AsyncMock(return_value=mock_created_pool)

                try:
                    result = await get_postgres_pool()
                    assert result is mock_created_pool
                    mock_asyncpg.create_pool.assert_called_once()
                finally:
                    module._pool = original_pool

    @pytest.mark.asyncio
    async def test_uses_database_url_fallback(self, mock_asyncpg_available):
        """Should use DATABASE_URL as fallback."""
        from aragora.storage.postgres_store import get_postgres_pool

        original_pool = module._pool
        module._pool = None

        mock_created_pool = MagicMock()
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://fallback"}):
            with patch.object(module, "asyncpg") as mock_asyncpg:
                mock_asyncpg.create_pool = AsyncMock(return_value=mock_created_pool)

                try:
                    result = await get_postgres_pool()
                    assert result is mock_created_pool
                finally:
                    module._pool = original_pool


class TestClosePostgresPool:
    """Tests for close_postgres_pool function."""

    @pytest.mark.asyncio
    async def test_closes_existing_pool(self):
        """Should close and clear existing pool."""
        from aragora.storage.postgres_store import close_postgres_pool

        mock_pool = AsyncMock()
        original_pool = module._pool
        module._pool = mock_pool

        try:
            await close_postgres_pool()
            mock_pool.close.assert_called_once()
            assert module._pool is None
        finally:
            module._pool = original_pool

    @pytest.mark.asyncio
    async def test_noop_when_no_pool(self):
        """Should do nothing when pool not initialized."""
        from aragora.storage.postgres_store import close_postgres_pool

        original_pool = module._pool
        module._pool = None

        try:
            await close_postgres_pool()  # Should not raise
            assert module._pool is None
        finally:
            module._pool = original_pool


# =============================================================================
# PostgresStore Tests
# =============================================================================


class TestPostgresStoreInit:
    """Tests for PostgresStore initialization."""

    def test_raises_when_asyncpg_not_available(self, mock_pool):
        """Should raise RuntimeError when asyncpg not available."""
        from aragora.storage.postgres_store import PostgresStore

        original = module.ASYNCPG_AVAILABLE
        module.ASYNCPG_AVAILABLE = False

        try:

            class TestStore(PostgresStore):
                SCHEMA_NAME = "test"
                SCHEMA_VERSION = 1
                INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

            with pytest.raises(RuntimeError, match="asyncpg is required"):
                TestStore(mock_pool)
        finally:
            module.ASYNCPG_AVAILABLE = original

    def test_raises_when_schema_name_missing(self, mock_asyncpg_available, mock_pool):
        """Should raise ValueError when SCHEMA_NAME not defined."""
        from aragora.storage.postgres_store import PostgresStore

        class NoNameStore(PostgresStore):
            SCHEMA_NAME = ""
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        with pytest.raises(ValueError, match="must define SCHEMA_NAME"):
            NoNameStore(mock_pool)

    def test_raises_when_initial_schema_missing(self, mock_asyncpg_available, mock_pool):
        """Should raise ValueError when INITIAL_SCHEMA not defined."""
        from aragora.storage.postgres_store import PostgresStore

        class NoSchemaStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = ""

        with pytest.raises(ValueError, match="must define INITIAL_SCHEMA"):
            NoSchemaStore(mock_pool)

    def test_stores_pool_reference(self, mock_asyncpg_available, mock_pool):
        """Should store pool reference."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        store = TestStore(mock_pool)
        assert store._pool is mock_pool
        assert store._initialized is False


class TestPostgresStoreInitialize:
    """Tests for PostgresStore.initialize method."""

    @pytest.mark.asyncio
    async def test_creates_schema_version_table(self, mock_asyncpg_available, mock_pool):
        """Should create _schema_versions table on initialize."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)  # No existing version

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = TestStore(mock_pool, use_resilient=False)
        await store.initialize()

        # Should have executed schema version table creation
        calls = mock_conn.execute.call_args_list
        assert any("_schema_versions" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_runs_initial_schema_for_new_db(self, mock_asyncpg_available, mock_pool):
        """Should run INITIAL_SCHEMA for new database."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test_items (id TEXT PRIMARY KEY);"

        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=None)  # No existing version

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = TestStore(mock_pool, use_resilient=False)
        await store.initialize()

        # Should have executed initial schema
        calls = [str(call) for call in mock_conn.execute.call_args_list]
        assert any("test_items" in call for call in calls)

    @pytest.mark.asyncio
    async def test_skips_if_already_initialized(self, mock_asyncpg_available, mock_pool):
        """Should skip initialization if already done."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        mock_conn = AsyncMock()
        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = TestStore(mock_pool, use_resilient=False)
        store._initialized = True

        await store.initialize()  # Should return immediately

        # Pool should not have been accessed
        mock_pool.acquire.assert_not_called()


class TestPostgresStoreCRUD:
    """Tests for PostgresStore CRUD helper methods."""

    @pytest.fixture
    def test_store(self, mock_asyncpg_available, mock_pool, mock_connection):
        """Create a test store with mocked connection."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = TestStore(mock_pool, use_resilient=False)
        store._initialized = True
        return store

    @pytest.mark.asyncio
    async def test_fetch_one(self, test_store, mock_connection):
        """fetch_one should execute query and return single row."""
        mock_connection.fetchrow = AsyncMock(return_value={"id": "123", "name": "test"})

        result = await test_store.fetch_one("SELECT * FROM items WHERE id = $1", "123")

        assert result == {"id": "123", "name": "test"}
        mock_connection.fetchrow.assert_called_once_with("SELECT * FROM items WHERE id = $1", "123")

    @pytest.mark.asyncio
    async def test_fetch_one_returns_none(self, test_store, mock_connection):
        """fetch_one should return None when no results."""
        mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await test_store.fetch_one("SELECT * FROM items WHERE id = $1", "999")

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_all(self, test_store, mock_connection):
        """fetch_all should execute query and return all rows."""
        mock_connection.fetch = AsyncMock(
            return_value=[
                {"id": "1", "name": "first"},
                {"id": "2", "name": "second"},
            ]
        )

        result = await test_store.fetch_all("SELECT * FROM items")

        assert len(result) == 2
        assert result[0]["id"] == "1"
        mock_connection.fetch.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute(self, test_store, mock_connection):
        """execute should run write operation and return status."""
        mock_connection.execute = AsyncMock(return_value="INSERT 0 1")

        result = await test_store.execute("INSERT INTO items (id) VALUES ($1)", "123")

        assert result == "INSERT 0 1"
        mock_connection.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_executemany(self, test_store, mock_connection):
        """executemany should run statement with multiple parameter sets."""
        mock_connection.executemany = AsyncMock()

        args = [("1",), ("2",), ("3",)]
        await test_store.executemany("INSERT INTO items (id) VALUES ($1)", args)

        mock_connection.executemany.assert_called_once_with(
            "INSERT INTO items (id) VALUES ($1)", args
        )

    @pytest.mark.asyncio
    async def test_exists_when_found(self, test_store, mock_connection):
        """exists should return True when record found."""
        mock_connection.fetchrow = AsyncMock(return_value={"?column?": 1})

        result = await test_store.exists("items", "id", "123")

        assert result is True

    @pytest.mark.asyncio
    async def test_exists_when_not_found(self, test_store, mock_connection):
        """exists should return False when record not found."""
        mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await test_store.exists("items", "id", "999")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists_validates_table_name(self, test_store):
        """exists should reject invalid table names."""
        with pytest.raises(ValueError, match="Invalid table name"):
            await test_store.exists("items; DROP TABLE", "id", "123")

    @pytest.mark.asyncio
    async def test_exists_validates_column_name(self, test_store):
        """exists should reject invalid column names."""
        with pytest.raises(ValueError, match="Invalid column name"):
            await test_store.exists("items", "id = '1' OR '1'='1'; --", "123")

    @pytest.mark.asyncio
    async def test_count_without_where(self, test_store, mock_connection):
        """count should return total count without WHERE clause."""
        mock_connection.fetchrow = AsyncMock(return_value=(42,))

        result = await test_store.count("items")

        assert result == 42

    @pytest.mark.asyncio
    async def test_count_with_where(self, test_store, mock_connection):
        """count should return filtered count with WHERE clause."""
        mock_connection.fetchrow = AsyncMock(return_value=(5,))

        result = await test_store.count("items", "status = $1", "active")

        assert result == 5
        # Verify WHERE was added
        call_args = mock_connection.fetchrow.call_args
        assert "WHERE" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_count_validates_table_name(self, test_store):
        """count should reject invalid table names."""
        with pytest.raises(ValueError, match="Invalid table name"):
            await test_store.count("items; DROP TABLE users")

    @pytest.mark.asyncio
    async def test_count_returns_zero_when_none(self, test_store, mock_connection):
        """count should return 0 when no results."""
        mock_connection.fetchrow = AsyncMock(return_value=None)

        result = await test_store.count("items")

        assert result == 0

    @pytest.mark.asyncio
    async def test_delete_by_id_success(self, test_store, mock_connection):
        """delete_by_id should return True when record deleted."""
        mock_connection.execute = AsyncMock(return_value="DELETE 1")

        result = await test_store.delete_by_id("items", "id", "123")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_by_id_not_found(self, test_store, mock_connection):
        """delete_by_id should return False when no record deleted."""
        mock_connection.execute = AsyncMock(return_value="DELETE 0")

        result = await test_store.delete_by_id("items", "id", "999")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_by_id_validates_table_name(self, test_store):
        """delete_by_id should reject invalid table names."""
        with pytest.raises(ValueError, match="Invalid table name"):
            await test_store.delete_by_id("items;--", "id", "123")

    @pytest.mark.asyncio
    async def test_delete_by_id_validates_column_name(self, test_store):
        """delete_by_id should reject invalid column names."""
        with pytest.raises(ValueError, match="Invalid column name"):
            await test_store.delete_by_id("items", "id;DROP", "123")


class TestPostgresStoreSchemaVersion:
    """Tests for schema version management."""

    @pytest.mark.asyncio
    async def test_get_schema_version(self, mock_asyncpg_available, mock_pool, mock_connection):
        """get_schema_version should return current version."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 2
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        mock_connection.fetchrow = AsyncMock(return_value=(2,))

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = TestStore(mock_pool, use_resilient=False)
        version = await store.get_schema_version()

        assert version == 2

    @pytest.mark.asyncio
    async def test_get_schema_version_returns_zero_on_error(
        self, mock_asyncpg_available, mock_pool, mock_connection
    ):
        """get_schema_version should return 0 on error."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        mock_connection.fetchrow = AsyncMock(side_effect=RuntimeError("Connection failed"))

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = TestStore(mock_pool, use_resilient=False)
        version = await store.get_schema_version()

        assert version == 0


class TestPostgresStoreTransaction:
    """Tests for transaction context manager."""

    @pytest.mark.asyncio
    async def test_transaction_commits_on_success(
        self, mock_asyncpg_available, mock_pool, mock_connection
    ):
        """Transaction should commit on successful completion."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        # Create transaction mock
        mock_tx = AsyncMock()
        mock_connection.transaction = MagicMock(return_value=mock_tx)

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = TestStore(mock_pool, use_resilient=False)
        store._initialized = True

        async with store.transaction() as conn:
            assert conn is mock_connection

        mock_connection.transaction.assert_called_once()


class TestPostgresStoreConnection:
    """Tests for connection context manager."""

    @pytest.mark.asyncio
    async def test_connection_uses_resilient_when_enabled(
        self, mock_asyncpg_available, mock_pool, mock_connection
    ):
        """Connection should use resilient acquisition when enabled."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.get_size.return_value = 10
        mock_pool.get_max_size.return_value = 20

        store = TestStore(mock_pool, use_resilient=True)
        store._initialized = True

        # The resilient path should track metrics
        module.reset_pool_metrics()
        async with store.connection() as conn:
            assert conn is mock_connection

        # Resilient path tracks acquisitions
        assert module._pool_metrics["total_acquisitions"] >= 1

    @pytest.mark.asyncio
    async def test_connection_bypasses_resilient_when_disabled(
        self, mock_asyncpg_available, mock_pool, mock_connection
    ):
        """Connection should bypass resilience when disabled."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = TestStore(mock_pool, use_resilient=False)
        store._initialized = True

        module.reset_pool_metrics()
        async with store.connection() as conn:
            assert conn is mock_connection

        # Non-resilient path doesn't track metrics
        assert module._pool_metrics["total_acquisitions"] == 0


class TestValidTableColumnNames:
    """Tests for table/column name validation."""

    @pytest.fixture
    def test_store(self, mock_asyncpg_available, mock_pool, mock_connection):
        """Create a test store for validation tests."""
        from aragora.storage.postgres_store import PostgresStore

        class TestStore(PostgresStore):
            SCHEMA_NAME = "test_store"
            SCHEMA_VERSION = 1
            INITIAL_SCHEMA = "CREATE TABLE test (id TEXT);"

        mock_pool.acquire = MagicMock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        store = TestStore(mock_pool, use_resilient=False)
        store._initialized = True
        return store

    @pytest.mark.asyncio
    async def test_allows_valid_table_names(self, test_store, mock_connection):
        """Should allow alphanumeric table names with underscores."""
        mock_connection.fetchrow = AsyncMock(return_value=(1,))

        # These should all work
        await test_store.exists("users", "id", "1")
        await test_store.exists("user_profiles", "id", "1")
        await test_store.exists("MyTable123", "id", "1")

    @pytest.mark.asyncio
    async def test_rejects_sql_injection_table(self, test_store):
        """Should reject SQL injection attempts in table names."""
        injection_attempts = [
            "users; DROP TABLE users;--",
            "users' OR '1'='1",
            "users UNION SELECT * FROM secrets",
            "users\n; DELETE FROM users;",
        ]

        for table in injection_attempts:
            with pytest.raises(ValueError, match="Invalid table name"):
                await test_store.exists(table, "id", "1")

    @pytest.mark.asyncio
    async def test_rejects_sql_injection_column(self, test_store):
        """Should reject SQL injection attempts in column names."""
        injection_attempts = [
            "id; DROP TABLE users;--",
            "id' OR '1'='1",
            "id UNION SELECT * FROM secrets",
        ]

        for col in injection_attempts:
            with pytest.raises(ValueError, match="Invalid column name"):
                await test_store.exists("users", col, "1")


class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_all_exports_importable(self):
        """All items in __all__ should be importable."""
        from aragora.storage import postgres_store

        for name in postgres_store.__all__:
            assert hasattr(postgres_store, name), f"Missing export: {name}"

    def test_key_exports(self):
        """Key exports should be available."""
        from aragora.storage.postgres_store import (
            PostgresStore,
            get_postgres_pool,
            get_postgres_pool_from_settings,
            close_postgres_pool,
            acquire_connection_resilient,
            get_pool_metrics,
            is_pool_healthy,
            reset_pool_metrics,
            PoolMetrics,
            PoolExhaustedError,
            ASYNCPG_AVAILABLE,
        )

        assert PostgresStore is not None
        assert callable(get_postgres_pool)
        assert callable(get_postgres_pool_from_settings)
        assert callable(close_postgres_pool)
        assert callable(acquire_connection_resilient)
        assert callable(get_pool_metrics)
        assert callable(is_pool_healthy)
        assert callable(reset_pool_metrics)
        assert PoolMetrics is not None
        assert issubclass(PoolExhaustedError, Exception)
        assert isinstance(ASYNCPG_AVAILABLE, bool)
