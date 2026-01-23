"""
Tests for Redis and Postgres checkpoint store backends.

These tests verify the checkpoint store implementations work correctly
for production deployments.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.workflow.types import WorkflowCheckpoint


class TestRedisCheckpointStore:
    """Tests for RedisCheckpointStore."""

    def _make_checkpoint(
        self,
        workflow_id: str = "test_workflow",
        current_step: str = "step1",
    ) -> WorkflowCheckpoint:
        """Create a test checkpoint."""
        return WorkflowCheckpoint(
            id="cp_123",
            workflow_id=workflow_id,
            definition_id="def_abc",
            current_step=current_step,
            completed_steps=["init", "validate"],
            step_outputs={"init": {"status": "ok"}, "validate": {"passed": True}},
            context_state={"key": "value"},
            created_at=datetime.now(),
            checksum="abc123",
        )

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = MagicMock()
        redis.get.return_value = None
        redis.setex.return_value = True
        redis.delete.return_value = 1
        redis.zadd.return_value = 1
        redis.zrevrange.return_value = []
        redis.zrem.return_value = 1
        redis.expire.return_value = True
        return redis

    def test_redis_checkpoint_store_initialization(self, mock_redis):
        """Test RedisCheckpointStore initializes correctly."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore(ttl_hours=48)
            assert store._ttl_seconds == 48 * 3600

    def test_redis_checkpoint_store_not_available(self):
        """Test RedisCheckpointStore raises when Redis not available."""
        with patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            with pytest.raises(RuntimeError, match="Redis"):
                RedisCheckpointStore()

    @pytest.mark.asyncio
    async def test_redis_save_checkpoint(self, mock_redis):
        """Test saving a checkpoint to Redis."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            checkpoint = self._make_checkpoint()
            checkpoint_id = await store.save(checkpoint)

            # Should return a valid checkpoint ID
            assert checkpoint_id.startswith("test_workflow_")

            # Should call setex for data and metadata
            assert mock_redis.setex.call_count == 2

            # Should add to workflow index
            mock_redis.zadd.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_load_checkpoint(self, mock_redis):
        """Test loading a checkpoint from Redis."""
        checkpoint_data = {
            "id": "cp_123",
            "workflow_id": "test_workflow",
            "definition_id": "def_abc",
            "current_step": "step1",
            "completed_steps": ["init"],
            "step_outputs": {},
            "context_state": {},
            "created_at": datetime.now().isoformat(),
            "checksum": "abc",
        }
        mock_redis.get.side_effect = [
            json.dumps(checkpoint_data).encode(),  # Data
            json.dumps({"compressed": False}).encode(),  # Metadata
        ]

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            checkpoint = await store.load("test_workflow_123456")

            assert checkpoint is not None
            assert checkpoint.workflow_id == "test_workflow"
            assert checkpoint.current_step == "step1"

    @pytest.mark.asyncio
    async def test_redis_load_checkpoint_not_found(self, mock_redis):
        """Test loading non-existent checkpoint returns None."""
        mock_redis.get.return_value = None

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            checkpoint = await store.load("nonexistent_123")
            assert checkpoint is None

    @pytest.mark.asyncio
    async def test_redis_load_latest_checkpoint(self, mock_redis):
        """Test loading latest checkpoint for a workflow."""
        mock_redis.zrevrange.return_value = [b"test_workflow_999"]

        checkpoint_data = {
            "id": "cp_latest",
            "workflow_id": "test_workflow",
            "definition_id": "def_abc",
            "current_step": "final",
            "completed_steps": ["init", "process"],
            "step_outputs": {},
            "context_state": {},
            "created_at": datetime.now().isoformat(),
            "checksum": "",
        }
        mock_redis.get.side_effect = [
            json.dumps(checkpoint_data).encode(),
            json.dumps({"compressed": False}).encode(),
        ]

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            checkpoint = await store.load_latest("test_workflow")

            assert checkpoint is not None
            assert checkpoint.current_step == "final"
            mock_redis.zrevrange.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_list_checkpoints(self, mock_redis):
        """Test listing checkpoints for a workflow."""
        mock_redis.zrevrange.return_value = [
            b"test_workflow_999",
            b"test_workflow_888",
            b"test_workflow_777",
        ]

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            ids = await store.list_checkpoints("test_workflow")

            assert len(ids) == 3
            assert ids[0] == "test_workflow_999"

    @pytest.mark.asyncio
    async def test_redis_delete_checkpoint(self, mock_redis):
        """Test deleting a checkpoint."""
        mock_redis.delete.return_value = 2  # Data + metadata deleted

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            result = await store.delete("test_workflow_123")

            assert result is True
            mock_redis.delete.assert_called_once()
            mock_redis.zrem.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_compressed_checkpoint(self, mock_redis):
        """Test compressed checkpoint handling."""
        import zlib

        checkpoint_data = {
            "id": "cp_big",
            "workflow_id": "test_workflow",
            "definition_id": "def_abc",
            "current_step": "step1",
            "completed_steps": [],
            "step_outputs": {"big_data": "x" * 10000},  # Large data
            "context_state": {},
            "created_at": datetime.now().isoformat(),
            "checksum": "",
        }

        compressed = zlib.compress(json.dumps(checkpoint_data).encode())
        mock_redis.get.side_effect = [
            compressed,  # Compressed data
            json.dumps({"compressed": True}).encode(),  # Metadata
        ]

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            checkpoint = await store.load("test_workflow_123")

            assert checkpoint is not None
            assert checkpoint.workflow_id == "test_workflow"


class TestPostgresCheckpointStore:
    """Tests for PostgresCheckpointStore."""

    def _make_checkpoint(
        self,
        workflow_id: str = "test_workflow",
        current_step: str = "step1",
    ) -> WorkflowCheckpoint:
        """Create a test checkpoint."""
        return WorkflowCheckpoint(
            id="cp_123",
            workflow_id=workflow_id,
            definition_id="def_abc",
            current_step=current_step,
            completed_steps=["init", "validate"],
            step_outputs={"init": {"status": "ok"}},
            context_state={"key": "value"},
            created_at=datetime.now(),
            checksum="abc123",
        )

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = MagicMock()
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)

        # Make acquire() return an async context manager
        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        return pool, conn

    def test_postgres_checkpoint_store_not_available(self):
        """Test PostgresCheckpointStore raises when asyncpg not available."""
        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            with pytest.raises(RuntimeError, match="asyncpg"):
                PostgresCheckpointStore(MagicMock())

    @pytest.mark.asyncio
    async def test_postgres_initialize(self, mock_pool):
        """Test PostgresCheckpointStore initialization."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = None  # No existing schema

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            await store.initialize()

            assert store._initialized is True
            # Should create schema version table and main table
            assert conn.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_postgres_save_checkpoint(self, mock_pool):
        """Test saving a checkpoint to PostgreSQL."""
        pool, conn = mock_pool

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            checkpoint = self._make_checkpoint()
            checkpoint_id = await store.save(checkpoint)

            assert checkpoint_id.startswith("test_workflow_")
            conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_postgres_load_checkpoint(self, mock_pool):
        """Test loading a checkpoint from PostgreSQL."""
        pool, conn = mock_pool

        row = {
            "id": "test_workflow_123",
            "workflow_id": "test_workflow",
            "definition_id": "def_abc",
            "current_step": "step1",
            "completed_steps": ["init"],
            "step_outputs": {"init": {"status": "ok"}},
            "context_state": {},
            "created_at": datetime.now(),
            "checksum": "abc",
        }
        conn.fetchrow.return_value = row

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            checkpoint = await store.load("test_workflow_123")

            assert checkpoint is not None
            assert checkpoint.workflow_id == "test_workflow"

    @pytest.mark.asyncio
    async def test_postgres_load_checkpoint_not_found(self, mock_pool):
        """Test loading non-existent checkpoint returns None."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = None

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            checkpoint = await store.load("nonexistent_123")
            assert checkpoint is None

    @pytest.mark.asyncio
    async def test_postgres_load_latest(self, mock_pool):
        """Test loading latest checkpoint."""
        pool, conn = mock_pool

        row = {
            "id": "test_workflow_999",
            "workflow_id": "test_workflow",
            "definition_id": "def_abc",
            "current_step": "final",
            "completed_steps": ["init", "process"],
            "step_outputs": {},
            "context_state": {},
            "created_at": datetime.now(),
            "checksum": "",
        }
        conn.fetchrow.return_value = row

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            checkpoint = await store.load_latest("test_workflow")

            assert checkpoint is not None
            assert checkpoint.current_step == "final"

    @pytest.mark.asyncio
    async def test_postgres_list_checkpoints(self, mock_pool):
        """Test listing checkpoints."""
        pool, conn = mock_pool

        conn.fetch.return_value = [
            {"id": "test_workflow_999"},
            {"id": "test_workflow_888"},
        ]

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            ids = await store.list_checkpoints("test_workflow")

            assert len(ids) == 2
            assert ids[0] == "test_workflow_999"

    @pytest.mark.asyncio
    async def test_postgres_delete_checkpoint(self, mock_pool):
        """Test deleting a checkpoint."""
        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 1"

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            result = await store.delete("test_workflow_123")

            assert result is True

    @pytest.mark.asyncio
    async def test_postgres_cleanup_old_checkpoints(self, mock_pool):
        """Test cleaning up old checkpoints."""
        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 5"

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            count = await store.cleanup_old_checkpoints("test_workflow", keep_count=10)

            assert count == 5


class TestGetCheckpointStore:
    """Tests for get_checkpoint_store factory function."""

    def test_get_checkpoint_store_with_mound(self):
        """Test get_checkpoint_store with explicit mound."""
        from aragora.workflow.checkpoint_store import (
            get_checkpoint_store,
            KnowledgeMoundCheckpointStore,
        )

        mock_mound = MagicMock()
        store = get_checkpoint_store(mound=mock_mound)

        assert isinstance(store, KnowledgeMoundCheckpointStore)
        assert store.mound is mock_mound

    def test_get_checkpoint_store_fallback_to_file(self):
        """Test get_checkpoint_store falls back to file store."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import (
                get_checkpoint_store,
                FileCheckpointStore,
            )

            store = get_checkpoint_store(
                use_default_mound=False,
                prefer_redis=False,
                prefer_postgres=False,
            )

            assert isinstance(store, FileCheckpointStore)

    def test_get_checkpoint_store_with_redis(self):
        """Test get_checkpoint_store uses Redis when available."""
        mock_redis = MagicMock()

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import (
                get_checkpoint_store,
                RedisCheckpointStore,
            )

            store = get_checkpoint_store(
                use_default_mound=False,
                prefer_redis=True,
            )

            assert isinstance(store, RedisCheckpointStore)

    @pytest.mark.asyncio
    async def test_get_checkpoint_store_async(self):
        """Test async get_checkpoint_store_async factory."""
        mock_redis = MagicMock()

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store.get_redis_client",
                return_value=mock_redis,
            ),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import (
                get_checkpoint_store_async,
                RedisCheckpointStore,
            )

            store = await get_checkpoint_store_async(
                use_default_mound=False,
                prefer_redis=True,
            )

            assert isinstance(store, RedisCheckpointStore)


# =============================================================================
# CachingCheckpointStore Tests
# =============================================================================


class TestCachingCheckpointStore:
    """Tests for CachingCheckpointStore wrapper."""

    def _make_checkpoint(
        self,
        checkpoint_id: str = "cp_123",
        workflow_id: str = "test_workflow",
    ) -> WorkflowCheckpoint:
        """Create a test checkpoint."""
        return WorkflowCheckpoint(
            id=checkpoint_id,
            workflow_id=workflow_id,
            definition_id="def_abc",
            current_step="step1",
            completed_steps=["init"],
            step_outputs={"init": {"status": "ok"}},
            context_state={"key": "value"},
            created_at=datetime.now(),
            checksum="abc123",
        )

    @pytest.mark.asyncio
    async def test_caching_store_save_updates_cache(self):
        """Test save() updates the cache."""
        from aragora.workflow.checkpoint_store import CachingCheckpointStore

        # Mock backend store
        backend = AsyncMock()
        backend.save = AsyncMock(return_value="cp_123")

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)
        checkpoint = self._make_checkpoint()

        cp_id = await cached_store.save(checkpoint)

        assert cp_id == "cp_123"
        backend.save.assert_called_once_with(checkpoint)
        # Cache should have the checkpoint
        assert cached_store._cache.get("cp_123") is checkpoint

    @pytest.mark.asyncio
    async def test_caching_store_load_hits_cache(self):
        """Test load() returns from cache on hit."""
        from aragora.workflow.checkpoint_store import CachingCheckpointStore

        backend = AsyncMock()
        backend.load = AsyncMock(return_value=None)

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)
        checkpoint = self._make_checkpoint()

        # Pre-populate cache
        cached_store._cache.put("cp_123", checkpoint)

        result = await cached_store.load("cp_123")

        assert result is checkpoint
        # Backend should NOT be called
        backend.load.assert_not_called()
        # Cache stats should show hit
        assert cached_store.cache_stats["hits"] >= 1

    @pytest.mark.asyncio
    async def test_caching_store_load_misses_to_backend(self):
        """Test load() falls back to backend on cache miss."""
        from aragora.workflow.checkpoint_store import CachingCheckpointStore

        checkpoint = self._make_checkpoint()
        backend = AsyncMock()
        backend.load = AsyncMock(return_value=checkpoint)

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)

        result = await cached_store.load("cp_123")

        assert result is checkpoint
        backend.load.assert_called_once_with("cp_123")
        # Should now be in cache
        assert cached_store._cache.get("cp_123") is checkpoint

    @pytest.mark.asyncio
    async def test_caching_store_load_latest_always_hits_backend(self):
        """Test load_latest() always goes to backend."""
        from aragora.workflow.checkpoint_store import CachingCheckpointStore

        checkpoint = self._make_checkpoint()
        backend = AsyncMock()
        backend.load_latest = AsyncMock(return_value=checkpoint)

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)

        result = await cached_store.load_latest("test_workflow")

        assert result is checkpoint
        backend.load_latest.assert_called_once_with("test_workflow")
        # Should populate cache
        assert cached_store._cache.get("cp_123") is checkpoint

    @pytest.mark.asyncio
    async def test_caching_store_delete_removes_from_cache(self):
        """Test delete() removes from both backend and cache."""
        from aragora.workflow.checkpoint_store import CachingCheckpointStore

        backend = AsyncMock()
        backend.delete = AsyncMock(return_value=True)

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)
        checkpoint = self._make_checkpoint()

        # Pre-populate cache
        cached_store._cache.put("cp_123", checkpoint)

        result = await cached_store.delete("cp_123")

        assert result is True
        backend.delete.assert_called_once_with("cp_123")
        # Should be removed from cache
        assert cached_store._cache.get("cp_123") is None

    @pytest.mark.asyncio
    async def test_caching_store_list_checkpoints(self):
        """Test list_checkpoints() delegates to backend."""
        from aragora.workflow.checkpoint_store import CachingCheckpointStore

        backend = AsyncMock()
        backend.list_checkpoints = AsyncMock(return_value=["cp_1", "cp_2", "cp_3"])

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)

        result = await cached_store.list_checkpoints("test_workflow")

        assert result == ["cp_1", "cp_2", "cp_3"]
        backend.list_checkpoints.assert_called_once_with("test_workflow")

    def test_caching_store_clear_cache(self):
        """Test clear_cache() clears only cache, not backend."""
        from aragora.workflow.checkpoint_store import CachingCheckpointStore

        backend = MagicMock()
        cached_store = CachingCheckpointStore(backend, max_cache_size=10)
        checkpoint = self._make_checkpoint()

        # Populate cache
        cached_store._cache.put("cp_123", checkpoint)
        assert cached_store._cache.size == 1

        cached_store.clear_cache()

        assert cached_store._cache.size == 0

    def test_caching_store_cache_stats(self):
        """Test cache_stats property returns statistics."""
        from aragora.workflow.checkpoint_store import CachingCheckpointStore

        backend = MagicMock()
        cached_store = CachingCheckpointStore(backend, max_cache_size=10)

        stats = cached_store.cache_stats

        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["max_size"] == 10

    def test_caching_store_backend_store_property(self):
        """Test backend_store property returns wrapped store."""
        from aragora.workflow.checkpoint_store import CachingCheckpointStore

        backend = MagicMock()
        cached_store = CachingCheckpointStore(backend)

        assert cached_store.backend_store is backend


class TestCachingFactoryIntegration:
    """Tests for caching integration in factory functions."""

    def test_get_checkpoint_store_with_caching_enabled(self):
        """Test get_checkpoint_store wraps with cache when enabled."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import (
                get_checkpoint_store,
                CachingCheckpointStore,
                FileCheckpointStore,
            )

            store = get_checkpoint_store(
                use_default_mound=False,
                prefer_redis=False,
                prefer_postgres=False,
                enable_caching=True,
                cache_size=50,
            )

            assert isinstance(store, CachingCheckpointStore)
            assert isinstance(store.backend_store, FileCheckpointStore)
            assert store._cache._max_size == 50

    def test_get_checkpoint_store_caching_disabled_by_default(self):
        """Test get_checkpoint_store doesn't wrap with cache by default."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import (
                get_checkpoint_store,
                CachingCheckpointStore,
                FileCheckpointStore,
            )

            store = get_checkpoint_store(
                use_default_mound=False,
                prefer_redis=False,
                prefer_postgres=False,
            )

            assert not isinstance(store, CachingCheckpointStore)
            assert isinstance(store, FileCheckpointStore)

    def test_get_checkpoint_store_caching_via_env_var(self):
        """Test get_checkpoint_store enables caching via environment variable."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
            patch.dict("os.environ", {"ARAGORA_CHECKPOINT_CACHE": "true"}),
        ):
            from aragora.workflow.checkpoint_store import (
                get_checkpoint_store,
                CachingCheckpointStore,
            )

            store = get_checkpoint_store(
                use_default_mound=False,
                prefer_redis=False,
                prefer_postgres=False,
            )

            assert isinstance(store, CachingCheckpointStore)
