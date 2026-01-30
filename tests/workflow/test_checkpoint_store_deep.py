"""
Deep tests for Workflow Checkpoint Store - covering undertested areas.

Focuses on:
- LRUCheckpointCache: eviction at capacity, stats tracking, concurrent access, clear behavior
- CachingCheckpointStore: cache invalidation, backend fallback, cache stats, load_latest always backend
- RedisCheckpointStore: compression threshold, TTL handling, socket timeout, sorted set index, connection errors
- PostgresCheckpointStore: schema versioning, checksum validation, cleanup_old_checkpoints, timeout handling
- FileCheckpointStore: glob-based load_latest, concurrent file access, corrupted JSON handling
- KnowledgeMoundCheckpointStore: provenance chain, duck-typed delete, serialization/deserialization
- get_checkpoint_store() factory: backend selection chain, env var overrides, fallback
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import zlib
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow.checkpoint_store import (
    CachingCheckpointStore,
    ConnectionTimeoutError,
    FileCheckpointStore,
    LRUCheckpointCache,
    MAX_CHECKPOINT_CACHE_SIZE,
)
from aragora.workflow.types import WorkflowCheckpoint


# =============================================================================
# Helpers
# =============================================================================


def _make_checkpoint(
    checkpoint_id: str = "cp-001",
    workflow_id: str = "wf-test-123",
    definition_id: str = "def-001",
    current_step: str = "step_2",
    completed_steps: list[str] | None = None,
    step_outputs: dict | None = None,
    context_state: dict | None = None,
    checksum: str = "abc123",
) -> WorkflowCheckpoint:
    """Create a test checkpoint with defaults."""
    return WorkflowCheckpoint(
        id=checkpoint_id,
        workflow_id=workflow_id,
        definition_id=definition_id,
        current_step=current_step,
        completed_steps=completed_steps or ["step_1"],
        step_outputs=step_outputs or {"step_1": {"result": "ok"}},
        context_state=context_state or {"counter": 5},
        created_at=datetime(2024, 6, 15, 12, 0, 0),
        checksum=checksum,
    )


# =============================================================================
# LRUCheckpointCache - Deep Tests
# =============================================================================


class TestLRUCheckpointCacheDeep:
    """Deep tests for LRU eviction, stats tracking, and concurrent access."""

    def test_eviction_at_exact_capacity(self):
        """Eviction fires when inserting the (max_size + 1)th item."""
        cache = LRUCheckpointCache(max_size=3)
        for i in range(3):
            cache.put(f"cp-{i}", _make_checkpoint(checkpoint_id=f"cp-{i}"))
        assert cache.size == 3

        # Insert 4th item -> evicts cp-0
        cache.put("cp-3", _make_checkpoint(checkpoint_id="cp-3"))
        assert cache.size == 3
        assert cache.get("cp-0") is None
        assert cache.get("cp-3") is not None

    def test_eviction_preserves_recently_accessed_items(self):
        """Accessing an item prevents it from being evicted next."""
        cache = LRUCheckpointCache(max_size=3)
        for i in range(3):
            cache.put(f"cp-{i}", _make_checkpoint(checkpoint_id=f"cp-{i}"))

        # Touch cp-0 so it becomes most recently used
        cache.get("cp-0")

        # Insert cp-3 -> should evict cp-1 (oldest not accessed)
        cache.put("cp-3", _make_checkpoint(checkpoint_id="cp-3"))
        assert cache.get("cp-0") is not None
        assert cache.get("cp-1") is None

    def test_eviction_chain_multiple_inserts(self):
        """Multiple inserts beyond capacity evict in FIFO order."""
        cache = LRUCheckpointCache(max_size=2)
        cache.put("a", _make_checkpoint(checkpoint_id="a"))
        cache.put("b", _make_checkpoint(checkpoint_id="b"))
        cache.put("c", _make_checkpoint(checkpoint_id="c"))  # evicts a
        cache.put("d", _make_checkpoint(checkpoint_id="d"))  # evicts b

        assert cache.get("a") is None
        assert cache.get("b") is None
        assert cache.get("c") is not None
        assert cache.get("d") is not None

    def test_put_existing_key_does_not_evict(self):
        """Updating an existing key does not trigger eviction."""
        cache = LRUCheckpointCache(max_size=2)
        cache.put("a", _make_checkpoint(checkpoint_id="a"))
        cache.put("b", _make_checkpoint(checkpoint_id="b"))

        # Update 'a' (existing) - should NOT evict anything
        updated = _make_checkpoint(checkpoint_id="a", current_step="updated")
        cache.put("a", updated)

        assert cache.size == 2
        assert cache.get("a").current_step == "updated"
        assert cache.get("b") is not None

    def test_stats_accumulate_across_operations(self):
        """Stats correctly accumulate hits and misses over many operations."""
        cache = LRUCheckpointCache(max_size=10)
        cache.put("x", _make_checkpoint(checkpoint_id="x"))

        # 5 hits
        for _ in range(5):
            cache.get("x")
        # 3 misses
        for _ in range(3):
            cache.get("nonexistent")

        stats = cache.stats
        assert stats["hits"] == 5
        assert stats["misses"] == 3
        assert stats["hit_rate"] == pytest.approx(5 / 8)

    def test_clear_resets_size_but_not_stats(self):
        """Clear removes all entries but hit/miss stats are retained."""
        cache = LRUCheckpointCache(max_size=10)
        cache.put("a", _make_checkpoint(checkpoint_id="a"))
        cache.get("a")  # hit
        cache.get("b")  # miss

        cache.clear()

        assert cache.size == 0
        # Stats are not reset by clear
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1

    def test_remove_decreases_size(self):
        """Remove decreases size and subsequent get returns None."""
        cache = LRUCheckpointCache(max_size=5)
        cache.put("a", _make_checkpoint(checkpoint_id="a"))
        cache.put("b", _make_checkpoint(checkpoint_id="b"))
        assert cache.size == 2

        cache.remove("a")
        assert cache.size == 1
        assert cache.get("a") is None

    def test_size_one_cache(self):
        """Cache with max_size=1 always keeps only the latest entry."""
        cache = LRUCheckpointCache(max_size=1)
        cache.put("a", _make_checkpoint(checkpoint_id="a"))
        cache.put("b", _make_checkpoint(checkpoint_id="b"))

        assert cache.size == 1
        assert cache.get("a") is None
        assert cache.get("b") is not None


# =============================================================================
# CachingCheckpointStore - Deep Tests
# =============================================================================


class TestCachingCheckpointStoreDeep:
    """Deep tests for cache invalidation, backend fallback, and stats."""

    @pytest.fixture
    def backend(self):
        store = MagicMock()
        store.save = AsyncMock(return_value="cp-saved")
        store.load = AsyncMock(return_value=None)
        store.load_latest = AsyncMock(return_value=None)
        store.list_checkpoints = AsyncMock(return_value=[])
        store.delete = AsyncMock(return_value=True)
        return store

    @pytest.mark.asyncio
    async def test_cache_populated_after_backend_load(self, backend):
        """After a cache miss, backend result is cached for subsequent loads."""
        cp = _make_checkpoint()
        backend.load.return_value = cp
        cached = CachingCheckpointStore(backend, max_cache_size=10)

        # First load -> cache miss -> backend
        result1 = await cached.load("cp-001")
        assert backend.load.call_count == 1
        assert result1 is cp

        # Second load -> cache hit -> no backend call
        result2 = await cached.load("cp-001")
        assert backend.load.call_count == 1  # still 1
        assert result2 is cp

    @pytest.mark.asyncio
    async def test_backend_returns_none_not_cached(self, backend):
        """When backend returns None, nothing is cached."""
        backend.load.return_value = None
        cached = CachingCheckpointStore(backend, max_cache_size=10)

        result = await cached.load("missing")
        assert result is None

        # Call again -> should still go to backend
        await cached.load("missing")
        assert backend.load.call_count == 2

    @pytest.mark.asyncio
    async def test_load_latest_always_hits_backend_even_after_cache(self, backend):
        """load_latest always queries backend regardless of cache state."""
        cp = _make_checkpoint()
        backend.load_latest.return_value = cp
        cached = CachingCheckpointStore(backend, max_cache_size=10)

        await cached.load_latest("wf-test-123")
        await cached.load_latest("wf-test-123")
        await cached.load_latest("wf-test-123")

        assert backend.load_latest.call_count == 3

    @pytest.mark.asyncio
    async def test_load_latest_updates_cache_by_checkpoint_id(self, backend):
        """load_latest caches the returned checkpoint by its ID."""
        cp = _make_checkpoint(checkpoint_id="cp-latest")
        backend.load_latest.return_value = cp
        cached = CachingCheckpointStore(backend, max_cache_size=10)

        await cached.load_latest("wf-test-123")

        # Now load by ID should hit cache
        result = await cached.load("cp-latest")
        # Backend load should NOT have been called
        backend.load.assert_not_called()
        assert result is cp

    @pytest.mark.asyncio
    async def test_delete_invalidates_cache_entry(self, backend):
        """Delete removes the entry from cache so next load hits backend."""
        cp = _make_checkpoint()
        cached = CachingCheckpointStore(backend, max_cache_size=10)
        cached._cache.put("cp-001", cp)

        await cached.delete("cp-001")

        # Cache entry gone
        assert cached._cache.get("cp-001") is None

    @pytest.mark.asyncio
    async def test_save_populates_cache_with_returned_id(self, backend):
        """After save, the checkpoint is cached under the returned ID."""
        backend.save.return_value = "cp-new-id"
        cp = _make_checkpoint()
        cached = CachingCheckpointStore(backend, max_cache_size=10)

        result_id = await cached.save(cp)
        assert result_id == "cp-new-id"

        # Cached under returned ID
        assert cached._cache.get("cp-new-id") is cp

    @pytest.mark.asyncio
    async def test_cache_stats_after_mixed_operations(self, backend):
        """Stats reflect the real pattern of hits and misses."""
        cp = _make_checkpoint()
        backend.load.return_value = cp
        cached = CachingCheckpointStore(backend, max_cache_size=10)

        await cached.load("cp-001")  # miss -> backend
        await cached.load("cp-001")  # hit
        await cached.load("cp-001")  # hit
        await cached.load("other")   # miss -> backend

        stats = cached.cache_stats
        assert stats["hits"] == 2
        assert stats["misses"] == 2
        assert stats["hit_rate"] == pytest.approx(0.5)

    def test_clear_cache_resets_entries_but_backend_untouched(self, backend):
        """clear_cache empties the cache without touching the backend."""
        cached = CachingCheckpointStore(backend, max_cache_size=10)
        cached._cache.put("a", _make_checkpoint(checkpoint_id="a"))
        cached._cache.put("b", _make_checkpoint(checkpoint_id="b"))

        cached.clear_cache()

        assert cached._cache.size == 0
        # Backend should have no delete calls
        backend.delete.assert_not_called()


# =============================================================================
# RedisCheckpointStore - Deep Tests
# =============================================================================


class TestRedisCheckpointStoreDeep:
    """Deep tests for compression, TTL, socket timeout, sorted set, and errors."""

    @pytest.fixture
    def mock_redis(self):
        redis = MagicMock()
        redis.setex = MagicMock()
        redis.get = MagicMock(return_value=None)
        redis.zadd = MagicMock()
        redis.zrevrange = MagicMock(return_value=[])
        redis.delete = MagicMock(return_value=1)
        redis.zrem = MagicMock()
        redis.expire = MagicMock()
        redis.connection_pool = MagicMock()
        redis.connection_pool.connection_kwargs = {}
        return redis

    def _make_store(self, mock_redis, **kwargs):
        """Create a RedisCheckpointStore with mocked Redis."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=lambda: mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore
            store = RedisCheckpointStore(**kwargs)
            store._redis = mock_redis
            return store

    @pytest.mark.asyncio
    async def test_save_compresses_above_threshold(self, mock_redis):
        """Data above compress_threshold is zlib-compressed."""
        store = self._make_store(mock_redis, compress_threshold=100)

        # Create checkpoint with large step_outputs
        big_cp = _make_checkpoint(step_outputs={"big": "x" * 500})
        await store.save(big_cp)

        # First setex call is the data
        data_call = mock_redis.setex.call_args_list[0]
        stored_bytes = data_call[0][2]

        # Should be compressed -> decompressible
        decompressed = zlib.decompress(stored_bytes)
        assert b"workflow_id" in decompressed

        # Metadata should say compressed=True
        meta_call = mock_redis.setex.call_args_list[1]
        meta_bytes = meta_call[0][2]
        meta = json.loads(meta_bytes)
        assert meta["compressed"] is True

    @pytest.mark.asyncio
    async def test_save_does_not_compress_below_threshold(self, mock_redis):
        """Data below compress_threshold is stored as plain bytes."""
        store = self._make_store(mock_redis, compress_threshold=100000)

        cp = _make_checkpoint(step_outputs={"small": "ok"})
        await store.save(cp)

        # Data call
        data_call = mock_redis.setex.call_args_list[0]
        stored_bytes = data_call[0][2]

        # Should be plain UTF-8
        decoded = stored_bytes.decode("utf-8")
        assert "workflow_id" in decoded

        # Metadata says not compressed
        meta_call = mock_redis.setex.call_args_list[1]
        meta = json.loads(meta_call[0][2])
        assert meta["compressed"] is False

    @pytest.mark.asyncio
    async def test_load_decompresses_correctly(self, mock_redis):
        """Load correctly decompresses data flagged as compressed."""
        store = self._make_store(mock_redis)

        cp_data = {
            "workflow_id": "wf-1",
            "definition_id": "def-1",
            "current_step": "s1",
            "completed_steps": [],
            "step_outputs": {},
            "context_state": {},
            "created_at": "2024-06-15T12:00:00",
            "checksum": "",
        }
        compressed = zlib.compress(json.dumps(cp_data).encode("utf-8"))

        mock_redis.get.side_effect = [
            compressed,
            json.dumps({"compressed": True}).encode(),
        ]

        result = await store.load("wf-1_12345")
        assert result is not None
        assert result.workflow_id == "wf-1"

    @pytest.mark.asyncio
    async def test_load_handles_missing_meta_gracefully(self, mock_redis):
        """Load works when metadata key is missing (assumes uncompressed)."""
        store = self._make_store(mock_redis)

        cp_data = {
            "workflow_id": "wf-2",
            "definition_id": "def-2",
            "current_step": "s1",
            "completed_steps": [],
            "step_outputs": {},
            "context_state": {},
            "created_at": "2024-06-15T12:00:00",
            "checksum": "",
        }
        mock_redis.get.side_effect = [
            json.dumps(cp_data).encode("utf-8"),
            None,  # No metadata
        ]

        result = await store.load("wf-2_12345")
        assert result is not None
        assert result.workflow_id == "wf-2"

    @pytest.mark.asyncio
    async def test_ttl_seconds_calculation(self, mock_redis):
        """TTL is correctly converted from hours to seconds."""
        store = self._make_store(mock_redis, ttl_hours=48)
        assert store._ttl_seconds == 48 * 3600

    @pytest.mark.asyncio
    async def test_save_sets_ttl_on_data_and_index(self, mock_redis):
        """Save applies TTL to data key and expire on index key."""
        store = self._make_store(mock_redis, ttl_hours=12)
        expected_ttl = 12 * 3600

        cp = _make_checkpoint()
        await store.save(cp)

        # Data key TTL
        data_call = mock_redis.setex.call_args_list[0]
        assert data_call[0][1] == expected_ttl

        # Meta key TTL
        meta_call = mock_redis.setex.call_args_list[1]
        assert meta_call[0][1] == expected_ttl

        # Index expire
        mock_redis.expire.assert_called_once()
        expire_args = mock_redis.expire.call_args[0]
        assert expire_args[1] == expected_ttl

    @pytest.mark.asyncio
    async def test_socket_timeout_configuration(self, mock_redis):
        """Socket timeouts are applied to the connection pool."""
        store = self._make_store(
            mock_redis,
            socket_timeout=15.0,
            socket_connect_timeout=5.0,
        )
        # Reset to force lazy init path
        store._redis = None

        with patch(
            "aragora.workflow.checkpoint_store._get_redis_client",
            return_value=mock_redis,
        ):
            redis_client = store._get_redis()

        assert redis_client.connection_pool.connection_kwargs["socket_timeout"] == 15.0
        assert redis_client.connection_pool.connection_kwargs["socket_connect_timeout"] == 5.0

    @pytest.mark.asyncio
    async def test_save_connection_timeout_raises(self, mock_redis):
        """Save raises ConnectionTimeoutError on Redis timeout."""
        store = self._make_store(mock_redis)

        # Simulate a TimeoutError
        mock_redis.setex.side_effect = type("TimeoutError", (Exception,), {})("timed out")

        with pytest.raises(ConnectionTimeoutError, match="timed out"):
            await store.save(_make_checkpoint())

    @pytest.mark.asyncio
    async def test_load_connection_error_raises(self, mock_redis):
        """Load raises ConnectionTimeoutError on Redis connection error."""
        store = self._make_store(mock_redis)

        mock_redis.get.side_effect = type("ConnectionError", (Exception,), {})("refused")

        with pytest.raises(ConnectionTimeoutError, match="refused"):
            await store.load("cp-123")

    @pytest.mark.asyncio
    async def test_load_latest_empty_sorted_set(self, mock_redis):
        """load_latest returns None when sorted set is empty."""
        store = self._make_store(mock_redis)
        mock_redis.zrevrange.return_value = []

        result = await store.load_latest("wf-none")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_latest_decodes_bytes_checkpoint_id(self, mock_redis):
        """load_latest decodes bytes-type checkpoint IDs from sorted set."""
        store = self._make_store(mock_redis)

        cp_data = {
            "workflow_id": "wf-1",
            "definition_id": "def-1",
            "current_step": "final",
            "completed_steps": ["a", "b"],
            "step_outputs": {},
            "context_state": {},
            "created_at": "2024-06-15T12:00:00",
            "checksum": "",
        }
        mock_redis.zrevrange.return_value = [b"wf-1_99999"]
        mock_redis.get.side_effect = [
            json.dumps(cp_data).encode(),
            json.dumps({"compressed": False}).encode(),
        ]

        result = await store.load_latest("wf-1")
        assert result is not None
        assert result.current_step == "final"

    @pytest.mark.asyncio
    async def test_delete_removes_from_sorted_set_index(self, mock_redis):
        """Delete removes the checkpoint from the workflow sorted set index."""
        store = self._make_store(mock_redis)
        mock_redis.delete.return_value = 2

        await store.delete("wf-abc_12345")

        mock_redis.zrem.assert_called_once()
        zrem_args = mock_redis.zrem.call_args[0]
        assert "wf-abc" in zrem_args[0]  # index key includes workflow_id
        assert zrem_args[1] == "wf-abc_12345"

    @pytest.mark.asyncio
    async def test_list_checkpoints_returns_string_ids(self, mock_redis):
        """list_checkpoints converts bytes results to strings."""
        store = self._make_store(mock_redis)
        mock_redis.zrevrange.return_value = [b"wf_111", "wf_222"]

        ids = await store.list_checkpoints("wf")
        assert ids == ["wf_111", "wf_222"]

    @pytest.mark.asyncio
    async def test_list_checkpoints_error_returns_empty(self, mock_redis):
        """list_checkpoints returns empty list on error."""
        store = self._make_store(mock_redis)
        mock_redis.zrevrange.side_effect = RuntimeError("connection lost")

        ids = await store.list_checkpoints("wf")
        assert ids == []


# =============================================================================
# PostgresCheckpointStore - Deep Tests
# =============================================================================


class TestPostgresCheckpointStoreDeep:
    """Deep tests for schema versioning, checksum validation, cleanup, timeouts."""

    @pytest.fixture
    def mock_pool(self):
        pool = MagicMock()
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="INSERT 0 1")
        conn.fetch = AsyncMock(return_value=[])
        conn.fetchrow = AsyncMock(return_value=None)

        pool.acquire = MagicMock()
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

        return pool, conn

    def _make_store(self, pool):
        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore
            store = PostgresCheckpointStore(pool)
            return store

    @pytest.mark.asyncio
    async def test_initialize_skips_if_already_initialized(self, mock_pool):
        """Initialize is a no-op when already initialized."""
        pool, conn = mock_pool
        store = self._make_store(pool)
        store._initialized = True

        await store.initialize()

        # No SQL executed
        conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_creates_schema_when_version_zero(self, mock_pool):
        """Initialize runs INITIAL_SCHEMA when schema version is 0 (new DB)."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = None  # No schema version row

        store = self._make_store(pool)
        await store.initialize()

        assert store._initialized is True
        # Should execute: schema_versions table, initial schema, version insert
        assert conn.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_initialize_skips_schema_when_version_current(self, mock_pool):
        """Initialize does not re-run schema if version matches."""
        pool, conn = mock_pool
        conn.fetchrow.return_value = {"version": 1}  # Current version

        store = self._make_store(pool)
        await store.initialize()

        assert store._initialized is True
        # Only the schema_versions table creation, no INITIAL_SCHEMA
        # The execute is called once for CREATE TABLE IF NOT EXISTS _schema_versions
        assert conn.execute.call_count == 1

    @pytest.mark.asyncio
    async def test_load_checksum_mismatch_logs_warning(self, mock_pool):
        """Load logs a warning when stored checksum does not match computed."""
        pool, conn = mock_pool
        row = {
            "id": "cp-check",
            "workflow_id": "wf-1",
            "definition_id": "def-1",
            "current_step": "s1",
            "completed_steps": ["a"],
            "step_outputs": "{}",
            "context_state": "{}",
            "created_at": datetime(2024, 6, 15),
            "checksum": "intentionally_wrong_checksum",
        }
        conn.fetchrow.return_value = row

        store = self._make_store(pool)
        store._initialized = True

        import logging
        with patch.object(logging.getLogger("aragora.workflow.checkpoint_store"), "warning") as mock_warn:
            result = await store.load("cp-check")

        assert result is not None
        assert result.workflow_id == "wf-1"
        mock_warn.assert_called_once()
        assert "checksum mismatch" in mock_warn.call_args[0][0]

    @pytest.mark.asyncio
    async def test_load_no_checksum_skips_validation(self, mock_pool):
        """Load skips checksum validation when checksum is empty."""
        pool, conn = mock_pool
        row = {
            "id": "cp-no-cs",
            "workflow_id": "wf-1",
            "definition_id": "def-1",
            "current_step": "s1",
            "completed_steps": [],
            "step_outputs": "{}",
            "context_state": "{}",
            "created_at": datetime(2024, 6, 15),
            "checksum": "",
        }
        conn.fetchrow.return_value = row

        store = self._make_store(pool)
        store._initialized = True

        result = await store.load("cp-no-cs")
        assert result is not None
        # No error should have occurred

    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints_returns_count(self, mock_pool):
        """cleanup_old_checkpoints returns the number of deleted rows."""
        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 7"

        store = self._make_store(pool)
        store._initialized = True

        count = await store.cleanup_old_checkpoints("wf-1", keep_count=5)
        assert count == 7

    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints_zero_deleted(self, mock_pool):
        """cleanup_old_checkpoints returns 0 when nothing to clean."""
        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 0"

        store = self._make_store(pool)
        store._initialized = True

        count = await store.cleanup_old_checkpoints("wf-1", keep_count=100)
        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints_error_returns_zero(self, mock_pool):
        """cleanup_old_checkpoints returns 0 on error."""
        pool, conn = mock_pool
        conn.execute.side_effect = RuntimeError("DB down")

        store = self._make_store(pool)
        store._initialized = True

        count = await store.cleanup_old_checkpoints("wf-1", keep_count=5)
        assert count == 0

    @pytest.mark.asyncio
    async def test_save_timeout_raises_connection_timeout_error(self, mock_pool):
        """Save raises ConnectionTimeoutError on asyncio.TimeoutError."""
        pool, conn = mock_pool
        conn.execute.side_effect = asyncio.TimeoutError()

        store = self._make_store(pool)
        store._initialized = True

        with pytest.raises(ConnectionTimeoutError, match="timed out"):
            await store.save(_make_checkpoint())

    @pytest.mark.asyncio
    async def test_load_timeout_returns_none(self, mock_pool):
        """Load returns None on timeout instead of raising."""
        pool, conn = mock_pool
        conn.fetchrow.side_effect = asyncio.TimeoutError()

        store = self._make_store(pool)
        store._initialized = True

        result = await store.load("cp-timeout")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_latest_timeout_returns_none(self, mock_pool):
        """load_latest returns None on timeout."""
        pool, conn = mock_pool
        conn.fetchrow.side_effect = asyncio.TimeoutError()

        store = self._make_store(pool)
        store._initialized = True

        result = await store.load_latest("wf-timeout")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_no_rows(self, mock_pool):
        """Delete returns False when no rows affected."""
        pool, conn = mock_pool
        conn.execute.return_value = "DELETE 0"

        store = self._make_store(pool)
        store._initialized = True

        result = await store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_row_to_checkpoint_parses_string_step_outputs(self, mock_pool):
        """_row_to_checkpoint parses step_outputs when stored as JSON string."""
        pool, conn = mock_pool
        row = {
            "id": "cp-str",
            "workflow_id": "wf-1",
            "definition_id": "def-1",
            "current_step": "s1",
            "completed_steps": ["a"],
            "step_outputs": '{"step_a": {"val": 42}}',
            "context_state": '{"key": "v"}',
            "created_at": datetime(2024, 6, 15),
            "checksum": "",
        }
        conn.fetchrow.return_value = row

        store = self._make_store(pool)
        store._initialized = True

        result = await store.load("cp-str")
        assert result is not None
        assert result.step_outputs == {"step_a": {"val": 42}}
        assert result.context_state == {"key": "v"}


# =============================================================================
# FileCheckpointStore - Deep Tests
# =============================================================================


class TestFileCheckpointStoreDeep:
    """Deep tests for glob-based load_latest, concurrent access, corrupted JSON."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as d:
            yield d

    @pytest.mark.asyncio
    async def test_load_latest_picks_lexicographically_last_file(self, temp_dir):
        """load_latest returns the file that sorts last (latest timestamp)."""
        store = FileCheckpointStore(temp_dir)

        # Manually create files with known timestamps
        for ts in ["20240101_100000", "20240101_120000", "20240101_110000"]:
            cp_id = f"wf-sort_{ts}"
            data = {
                "workflow_id": "wf-sort",
                "definition_id": "def",
                "current_step": ts,
                "completed_steps": [],
                "step_outputs": {},
                "context_state": {},
                "created_at": "2024-01-01T10:00:00",
                "checksum": "",
            }
            (Path(temp_dir) / f"{cp_id}.json").write_text(json.dumps(data))

        latest = await store.load_latest("wf-sort")
        assert latest is not None
        # "20240101_120000" sorts last
        assert latest.current_step == "20240101_120000"

    @pytest.mark.asyncio
    async def test_load_corrupted_json_raises(self, temp_dir):
        """Loading a corrupted JSON file raises an error."""
        store = FileCheckpointStore(temp_dir)

        # Write corrupted JSON
        bad_path = Path(temp_dir) / "wf-bad_20240101_120000.json"
        bad_path.write_text("{invalid json content!!!")

        with pytest.raises(json.JSONDecodeError):
            await store.load("wf-bad_20240101_120000")

    @pytest.mark.asyncio
    async def test_load_latest_with_corrupted_file_raises(self, temp_dir):
        """load_latest raises when the latest file is corrupted."""
        store = FileCheckpointStore(temp_dir)

        bad_path = Path(temp_dir) / "wf-corrupt_20240101_120000.json"
        bad_path.write_text("NOT VALID JSON")

        with pytest.raises(json.JSONDecodeError):
            await store.load_latest("wf-corrupt")

    @pytest.mark.asyncio
    async def test_list_checkpoints_returns_stems_only(self, temp_dir):
        """list_checkpoints returns file stems without .json extension."""
        store = FileCheckpointStore(temp_dir)

        for i in range(3):
            cp = _make_checkpoint(
                workflow_id="wf-stems",
                current_step=f"step_{i}",
            )
            # Use manual file write to control naming
            data = {"workflow_id": "wf-stems", "definition_id": "d", "current_step": f"s{i}",
                     "completed_steps": [], "step_outputs": {}, "context_state": {},
                     "created_at": "2024-01-01T00:00:00", "checksum": ""}
            (Path(temp_dir) / f"wf-stems_{i}.json").write_text(json.dumps(data))

        ids = await store.list_checkpoints("wf-stems")
        assert len(ids) == 3
        for id_ in ids:
            assert not id_.endswith(".json")

    @pytest.mark.asyncio
    async def test_concurrent_saves_do_not_overwrite(self, temp_dir):
        """Concurrent saves with different timestamps produce separate files."""
        store = FileCheckpointStore(temp_dir)

        async def save_cp(step: str):
            cp = _make_checkpoint(workflow_id="wf-concurrent", current_step=step)
            return await store.save(cp)

        # Save with slight delay to ensure different timestamps
        id1 = await save_cp("step_a")
        await asyncio.sleep(1.1)  # FileCheckpointStore uses second-precision
        id2 = await save_cp("step_b")

        assert id1 != id2
        files = list(Path(temp_dir).glob("wf-concurrent_*.json"))
        assert len(files) == 2


# =============================================================================
# KnowledgeMoundCheckpointStore - Deep Tests
# =============================================================================


class TestKnowledgeMoundCheckpointStoreDeep:
    """Deep tests for provenance chain, duck-typed delete, serialization."""

    @pytest.fixture
    def mock_mound(self):
        mound = MagicMock()
        mound._workspace_id = "test-ws"
        mound.add_node = AsyncMock(return_value="node-abc")
        mound.get_node = AsyncMock(return_value=None)
        mound.query_by_provenance = AsyncMock(return_value=[])
        mound.delete_node = AsyncMock(return_value=True)
        return mound

    @pytest.mark.asyncio
    async def test_save_builds_provenance_chain(self, mock_mound):
        """Save constructs a ProvenanceChain with workflow metadata."""
        import builtins
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        captured_node = None

        async def capture_add_node(node):
            nonlocal captured_node
            captured_node = node
            return "node-prov"

        mock_mound.add_node = capture_add_node
        store = KnowledgeMoundCheckpointStore(mock_mound)

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "aragora.knowledge.mound":
                mock_module = MagicMock()
                # Use real classes that capture args
                mock_module.KnowledgeNode = lambda **kw: MagicMock(**kw)
                mock_module.MemoryTier = MagicMock()
                mock_module.MemoryTier.MEDIUM = "medium"
                mock_module.ProvenanceChain = lambda **kw: MagicMock(**kw)
                return mock_module
            return original_import(name, *args, **kwargs)

        cp = _make_checkpoint(
            workflow_id="wf-prov",
            current_step="step_3",
            completed_steps=["step_1", "step_2"],
        )

        with patch.object(builtins, "__import__", side_effect=mock_import):
            node_id = await store.save(cp)

        assert node_id == "node-prov"
        assert captured_node is not None

    @pytest.mark.asyncio
    async def test_load_wrong_node_type_returns_none(self, mock_mound):
        """Load returns None when node_type is not 'workflow_checkpoint'."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        node = MagicMock()
        node.node_type = "document"
        node.content = "{}"
        mock_mound.get_node.return_value = node

        store = KnowledgeMoundCheckpointStore(mock_mound)
        result = await store.load("node-wrong")
        assert result is None

    @pytest.mark.asyncio
    async def test_load_deserializes_checkpoint_fields(self, mock_mound):
        """Load correctly deserializes all checkpoint fields from JSON content."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        node = MagicMock()
        node.node_type = "workflow_checkpoint"
        node.content = json.dumps({
            "id": "cp-km",
            "workflow_id": "wf-km-1",
            "definition_id": "def-km",
            "current_step": "step_5",
            "completed_steps": ["step_1", "step_2", "step_3", "step_4"],
            "step_outputs": {"step_1": {"val": 42}},
            "context_state": {"mode": "test"},
            "created_at": "2024-06-15T10:30:00",
            "checksum": "km-checksum",
        })
        mock_mound.get_node.return_value = node

        store = KnowledgeMoundCheckpointStore(mock_mound)
        result = await store.load("node-km")

        assert result is not None
        assert result.workflow_id == "wf-km-1"
        assert result.current_step == "step_5"
        assert len(result.completed_steps) == 4
        assert result.step_outputs == {"step_1": {"val": 42}}
        assert result.context_state == {"mode": "test"}
        assert result.checksum == "km-checksum"

    @pytest.mark.asyncio
    async def test_delete_delegates_to_mound_delete_node(self, mock_mound):
        """Delete calls mound.delete_node with the checkpoint ID."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        store = KnowledgeMoundCheckpointStore(mock_mound)
        result = await store.delete("node-del")

        assert result is True
        mock_mound.delete_node.assert_called_once_with("node-del")

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_error(self, mock_mound):
        """Delete returns False when mound.delete_node raises."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        mock_mound.delete_node.side_effect = RuntimeError("storage error")

        store = KnowledgeMoundCheckpointStore(mock_mound)
        result = await store.delete("node-err")
        assert result is False

    @pytest.mark.asyncio
    async def test_load_latest_queries_by_provenance(self, mock_mound):
        """load_latest queries mound by provenance source_type and source_id."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        node = MagicMock()
        node.content = json.dumps({
            "workflow_id": "wf-lat",
            "definition_id": "def-lat",
            "current_step": "last",
            "completed_steps": [],
            "step_outputs": {},
            "context_state": {},
            "created_at": "2024-06-15T12:00:00",
            "checksum": "",
        })
        mock_mound.query_by_provenance.return_value = [node]

        store = KnowledgeMoundCheckpointStore(mock_mound)
        result = await store.load_latest("wf-lat")

        assert result is not None
        assert result.current_step == "last"
        mock_mound.query_by_provenance.assert_called_once_with(
            source_type="workflow_engine",
            source_id="wf-lat",
            node_type="workflow_checkpoint",
            limit=1,
        )

    @pytest.mark.asyncio
    async def test_workspace_id_defaults_to_mound_workspace(self, mock_mound):
        """Workspace ID defaults to mound's _workspace_id."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        store = KnowledgeMoundCheckpointStore(mock_mound)
        assert store.workspace_id == "test-ws"

    @pytest.mark.asyncio
    async def test_workspace_id_override(self, mock_mound):
        """Workspace ID can be explicitly overridden."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        store = KnowledgeMoundCheckpointStore(mock_mound, workspace_id="custom-ws")
        assert store.workspace_id == "custom-ws"


# =============================================================================
# get_checkpoint_store() Factory - Deep Tests
# =============================================================================


class TestGetCheckpointStoreFactoryDeep:
    """Deep tests for backend selection chain, env var overrides, fallback."""

    def test_redis_preferred_over_file_when_available(self):
        """Redis is selected over file when REDIS_AVAILABLE and prefer_redis=True."""
        mock_redis = MagicMock()

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch("aragora.workflow.checkpoint_store._get_redis_client", return_value=mock_redis),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
        ):
            from aragora.workflow.checkpoint_store import (
                RedisCheckpointStore,
                get_checkpoint_store,
            )

            store = get_checkpoint_store(
                use_default_mound=False,
                prefer_redis=True,
            )
            assert isinstance(store, RedisCheckpointStore)

    def test_file_fallback_when_redis_unavailable(self):
        """Falls back to FileCheckpointStore when Redis is unavailable."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import FileCheckpointStore, get_checkpoint_store

            store = get_checkpoint_store(use_default_mound=False)
            assert isinstance(store, FileCheckpointStore)

    def test_env_var_store_backend_redis(self):
        """ARAGORA_CHECKPOINT_STORE_BACKEND=redis forces Redis selection."""
        mock_redis = MagicMock()

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch("aragora.workflow.checkpoint_store._get_redis_client", return_value=mock_redis),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch.dict("os.environ", {"ARAGORA_CHECKPOINT_STORE_BACKEND": "redis"}),
        ):
            from aragora.workflow.checkpoint_store import (
                RedisCheckpointStore,
                get_checkpoint_store,
            )

            store = get_checkpoint_store(
                use_default_mound=False,
                prefer_redis=False,  # overridden by env var
            )
            assert isinstance(store, RedisCheckpointStore)

    def test_env_var_store_backend_file(self):
        """ARAGORA_CHECKPOINT_STORE_BACKEND=file forces file selection."""
        mock_redis = MagicMock()

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch("aragora.workflow.checkpoint_store._get_redis_client", return_value=mock_redis),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch.dict("os.environ", {"ARAGORA_CHECKPOINT_STORE_BACKEND": "file"}),
        ):
            from aragora.workflow.checkpoint_store import FileCheckpointStore, get_checkpoint_store

            store = get_checkpoint_store(use_default_mound=False)
            assert isinstance(store, FileCheckpointStore)

    def test_env_var_global_db_backend_postgres(self):
        """ARAGORA_DB_BACKEND=postgres enables postgres preference."""
        # Postgres path will fail (no real pool), falling back to file
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
            patch.dict("os.environ", {"ARAGORA_DB_BACKEND": "postgres"}),
        ):
            from aragora.workflow.checkpoint_store import FileCheckpointStore, get_checkpoint_store

            store = get_checkpoint_store(use_default_mound=False)
            # Since asyncpg is not available, falls through to file
            assert isinstance(store, FileCheckpointStore)

    def test_env_var_cache_true_enables_caching(self):
        """ARAGORA_CHECKPOINT_CACHE=true enables the caching layer."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
            patch.dict("os.environ", {"ARAGORA_CHECKPOINT_CACHE": "1"}),
        ):
            from aragora.workflow.checkpoint_store import CachingCheckpointStore, get_checkpoint_store

            store = get_checkpoint_store(use_default_mound=False)
            assert isinstance(store, CachingCheckpointStore)

    def test_mound_takes_priority_over_redis(self):
        """Explicitly provided mound beats Redis even when Redis is available."""
        mock_mound = MagicMock()
        mock_mound._workspace_id = "prio-test"
        mock_redis = MagicMock()

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch("aragora.workflow.checkpoint_store._get_redis_client", return_value=mock_redis),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import (
                KnowledgeMoundCheckpointStore,
                get_checkpoint_store,
            )

            store = get_checkpoint_store(mound=mock_mound)
            assert isinstance(store, KnowledgeMoundCheckpointStore)

    def test_default_mound_used_when_no_explicit_mound(self):
        """Default mound is used when no explicit mound provided."""
        mock_mound = MagicMock()
        mock_mound._workspace_id = "default-mound"

        with patch("aragora.workflow.checkpoint_store._default_mound", mock_mound):
            from aragora.workflow.checkpoint_store import (
                KnowledgeMoundCheckpointStore,
                get_checkpoint_store,
            )

            store = get_checkpoint_store(use_default_mound=True)
            assert isinstance(store, KnowledgeMoundCheckpointStore)

    def test_default_mound_skipped_when_use_default_false(self):
        """Default mound is skipped when use_default_mound=False."""
        mock_mound = MagicMock()
        mock_mound._workspace_id = "skip-me"

        with (
            patch("aragora.workflow.checkpoint_store._default_mound", mock_mound),
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
        ):
            from aragora.workflow.checkpoint_store import FileCheckpointStore, get_checkpoint_store

            store = get_checkpoint_store(use_default_mound=False)
            assert isinstance(store, FileCheckpointStore)

    def test_redis_failure_falls_back_to_file(self):
        """When Redis client returns None, falls back to file store."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch("aragora.workflow.checkpoint_store._get_redis_client", return_value=None),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
        ):
            from aragora.workflow.checkpoint_store import FileCheckpointStore, get_checkpoint_store

            store = get_checkpoint_store(use_default_mound=False)
            assert isinstance(store, FileCheckpointStore)
