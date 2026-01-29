"""
Tests for Workflow Checkpoint Store implementations.

Tests:
- LRU cache for checkpoints
- Caching wrapper for stores
- File-based checkpoint store
- Redis checkpoint store (mocked)
- PostgreSQL checkpoint store (mocked)
- KnowledgeMound checkpoint store (mocked)
- Factory functions
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.workflow.checkpoint_store import (
    CachingCheckpointStore,
    CheckpointValidationError,
    ConnectionTimeoutError,
    FileCheckpointStore,
    LRUCheckpointCache,
    MAX_CHECKPOINT_CACHE_SIZE,
)
from aragora.workflow.types import WorkflowCheckpoint


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_checkpoint():
    """Create a sample checkpoint for testing."""
    return WorkflowCheckpoint(
        id="cp-001",
        workflow_id="wf-test-123",
        definition_id="def-001",
        current_step="step_2",
        completed_steps=["step_1"],
        step_outputs={"step_1": {"result": "success", "data": {"key": "value"}}},
        context_state={"counter": 5, "flag": True},
        created_at=datetime(2024, 1, 15, 12, 0, 0),
        checksum="abc123",
    )


@pytest.fixture
def sample_checkpoint_2():
    """Create another sample checkpoint."""
    return WorkflowCheckpoint(
        id="cp-002",
        workflow_id="wf-test-123",
        definition_id="def-001",
        current_step="step_3",
        completed_steps=["step_1", "step_2"],
        step_outputs={
            "step_1": {"result": "success"},
            "step_2": {"result": "processed"},
        },
        context_state={"counter": 10},
        created_at=datetime(2024, 1, 15, 13, 0, 0),
        checksum="def456",
    )


@pytest.fixture
def sample_checkpoint_different_workflow():
    """Create checkpoint for a different workflow."""
    return WorkflowCheckpoint(
        id="cp-003",
        workflow_id="wf-other-456",
        definition_id="def-002",
        current_step="step_1",
        completed_steps=[],
        step_outputs={},
        context_state={},
        created_at=datetime(2024, 1, 15, 14, 0, 0),
        checksum="ghi789",
    )


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for file-based checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# LRUCheckpointCache Tests
# =============================================================================


class TestLRUCheckpointCache:
    """Tests for LRU checkpoint cache."""

    def test_init_default_size(self):
        """Initialize cache with default size."""
        cache = LRUCheckpointCache()
        assert cache._max_size == MAX_CHECKPOINT_CACHE_SIZE

    def test_init_custom_size(self):
        """Initialize cache with custom size."""
        cache = LRUCheckpointCache(max_size=50)
        assert cache._max_size == 50

    def test_put_and_get(self, sample_checkpoint):
        """Put and retrieve a checkpoint."""
        cache = LRUCheckpointCache()
        cache.put("cp-001", sample_checkpoint)

        result = cache.get("cp-001")
        assert result is not None
        assert result.id == "cp-001"
        assert result.workflow_id == "wf-test-123"

    def test_get_miss(self):
        """Get returns None for missing key."""
        cache = LRUCheckpointCache()
        result = cache.get("nonexistent")
        assert result is None

    def test_get_updates_lru_order(self, sample_checkpoint, sample_checkpoint_2):
        """Getting an item moves it to end (most recently used)."""
        cache = LRUCheckpointCache(max_size=2)

        # Add two items
        cache.put("cp-001", sample_checkpoint)
        cache.put("cp-002", sample_checkpoint_2)

        # Access first item (moves to end)
        cache.get("cp-001")

        # Add third item - should evict cp-002 (least recently used)
        cp3 = WorkflowCheckpoint(
            id="cp-003",
            workflow_id="wf-test",
            definition_id="def-001",
            current_step="step_1",
            completed_steps=[],
            step_outputs={},
            context_state={},
            created_at=datetime.now(),
            checksum="",
        )
        cache.put("cp-003", cp3)

        # cp-001 should still be there (was accessed recently)
        assert cache.get("cp-001") is not None
        # cp-002 should be evicted
        assert cache.get("cp-002") is None

    def test_put_updates_existing(self, sample_checkpoint):
        """Putting existing key updates the value."""
        cache = LRUCheckpointCache()
        cache.put("cp-001", sample_checkpoint)

        # Update with new checkpoint
        updated = WorkflowCheckpoint(
            id="cp-001",
            workflow_id="wf-updated",
            definition_id="def-001",
            current_step="step_new",
            completed_steps=["step_1", "step_2"],
            step_outputs={},
            context_state={},
            created_at=datetime.now(),
            checksum="",
        )
        cache.put("cp-001", updated)

        result = cache.get("cp-001")
        assert result.workflow_id == "wf-updated"
        assert result.current_step == "step_new"

    def test_lru_eviction(self, sample_checkpoint):
        """Oldest entries are evicted when cache is full."""
        cache = LRUCheckpointCache(max_size=3)

        # Fill cache
        for i in range(3):
            cp = WorkflowCheckpoint(
                id=f"cp-{i}",
                workflow_id="wf-test",
                definition_id="def-001",
                current_step="step_1",
                completed_steps=[],
                step_outputs={},
                context_state={},
                created_at=datetime.now(),
                checksum="",
            )
            cache.put(f"cp-{i}", cp)

        # Add one more - should evict cp-0
        cache.put("cp-3", sample_checkpoint)

        assert cache.get("cp-0") is None  # Evicted
        assert cache.get("cp-1") is not None
        assert cache.get("cp-2") is not None
        assert cache.get("cp-3") is not None

    def test_remove(self, sample_checkpoint):
        """Remove checkpoint from cache."""
        cache = LRUCheckpointCache()
        cache.put("cp-001", sample_checkpoint)

        assert cache.remove("cp-001") is True
        assert cache.get("cp-001") is None

    def test_remove_nonexistent(self):
        """Remove nonexistent key returns False."""
        cache = LRUCheckpointCache()
        assert cache.remove("nonexistent") is False

    def test_clear(self, sample_checkpoint, sample_checkpoint_2):
        """Clear entire cache."""
        cache = LRUCheckpointCache()
        cache.put("cp-001", sample_checkpoint)
        cache.put("cp-002", sample_checkpoint_2)

        cache.clear()

        assert cache.size == 0
        assert cache.get("cp-001") is None
        assert cache.get("cp-002") is None

    def test_size_property(self, sample_checkpoint, sample_checkpoint_2):
        """Size property returns current cache size."""
        cache = LRUCheckpointCache()
        assert cache.size == 0

        cache.put("cp-001", sample_checkpoint)
        assert cache.size == 1

        cache.put("cp-002", sample_checkpoint_2)
        assert cache.size == 2

    def test_stats_property(self, sample_checkpoint):
        """Stats property returns cache statistics."""
        cache = LRUCheckpointCache(max_size=100)
        cache.put("cp-001", sample_checkpoint)

        # Hit
        cache.get("cp-001")
        # Miss
        cache.get("nonexistent")

        stats = cache.stats

        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_stats_empty_cache(self):
        """Stats on empty cache don't divide by zero."""
        cache = LRUCheckpointCache()
        stats = cache.stats

        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0


# =============================================================================
# CachingCheckpointStore Tests
# =============================================================================


class TestCachingCheckpointStore:
    """Tests for caching wrapper around stores."""

    @pytest.fixture
    def mock_backend_store(self):
        """Create a mock backend store."""
        store = MagicMock()
        store.save = AsyncMock(return_value="cp-saved-001")
        store.load = AsyncMock(return_value=None)
        store.load_latest = AsyncMock(return_value=None)
        store.list_checkpoints = AsyncMock(return_value=[])
        store.delete = AsyncMock(return_value=True)
        return store

    @pytest.mark.asyncio
    async def test_save_updates_cache(self, mock_backend_store, sample_checkpoint):
        """Save updates both backend and cache."""
        cached_store = CachingCheckpointStore(mock_backend_store)

        checkpoint_id = await cached_store.save(sample_checkpoint)

        # Backend was called
        mock_backend_store.save.assert_called_once_with(sample_checkpoint)
        assert checkpoint_id == "cp-saved-001"

        # Cache has the checkpoint
        cached = cached_store._cache.get("cp-saved-001")
        assert cached is sample_checkpoint

    @pytest.mark.asyncio
    async def test_load_hits_cache(self, mock_backend_store, sample_checkpoint):
        """Load returns from cache when available."""
        cached_store = CachingCheckpointStore(mock_backend_store)

        # Pre-populate cache
        cached_store._cache.put("cp-001", sample_checkpoint)

        result = await cached_store.load("cp-001")

        # Backend NOT called (cache hit)
        mock_backend_store.load.assert_not_called()
        assert result is sample_checkpoint

    @pytest.mark.asyncio
    async def test_load_falls_through_on_miss(self, mock_backend_store, sample_checkpoint):
        """Load goes to backend on cache miss."""
        mock_backend_store.load.return_value = sample_checkpoint
        cached_store = CachingCheckpointStore(mock_backend_store)

        result = await cached_store.load("cp-001")

        # Backend called
        mock_backend_store.load.assert_called_once_with("cp-001")
        assert result is sample_checkpoint

        # Now cached for next time
        cached = cached_store._cache.get("cp-001")
        assert cached is sample_checkpoint

    @pytest.mark.asyncio
    async def test_load_latest_always_hits_backend(self, mock_backend_store, sample_checkpoint):
        """Load latest always goes to backend (don't track recency in cache)."""
        mock_backend_store.load_latest.return_value = sample_checkpoint
        cached_store = CachingCheckpointStore(mock_backend_store)

        # Pre-populate cache with old checkpoint
        old_cp = WorkflowCheckpoint(
            id="old-cp",
            workflow_id="wf-test-123",
            definition_id="def-001",
            current_step="step_1",
            completed_steps=[],
            step_outputs={},
            context_state={},
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            checksum="",
        )
        cached_store._cache.put("old-cp", old_cp)

        result = await cached_store.load_latest("wf-test-123")

        # Backend called even though cache has data
        mock_backend_store.load_latest.assert_called_once_with("wf-test-123")
        assert result is sample_checkpoint

    @pytest.mark.asyncio
    async def test_list_checkpoints_hits_backend(self, mock_backend_store):
        """List checkpoints always goes to backend."""
        mock_backend_store.list_checkpoints.return_value = ["cp-1", "cp-2", "cp-3"]
        cached_store = CachingCheckpointStore(mock_backend_store)

        result = await cached_store.list_checkpoints("wf-test")

        mock_backend_store.list_checkpoints.assert_called_once_with("wf-test")
        assert result == ["cp-1", "cp-2", "cp-3"]

    @pytest.mark.asyncio
    async def test_delete_clears_cache(self, mock_backend_store, sample_checkpoint):
        """Delete removes from both cache and backend."""
        cached_store = CachingCheckpointStore(mock_backend_store)

        # Pre-populate cache
        cached_store._cache.put("cp-001", sample_checkpoint)

        result = await cached_store.delete("cp-001")

        assert result is True
        mock_backend_store.delete.assert_called_once_with("cp-001")

        # Cache entry removed
        assert cached_store._cache.get("cp-001") is None

    def test_clear_cache(self, mock_backend_store, sample_checkpoint, sample_checkpoint_2):
        """Clear cache without affecting backend."""
        cached_store = CachingCheckpointStore(mock_backend_store)

        cached_store._cache.put("cp-001", sample_checkpoint)
        cached_store._cache.put("cp-002", sample_checkpoint_2)

        cached_store.clear_cache()

        assert cached_store._cache.size == 0

    def test_cache_stats(self, mock_backend_store):
        """Get cache statistics."""
        cached_store = CachingCheckpointStore(mock_backend_store, max_cache_size=50)

        stats = cached_store.cache_stats

        assert stats["max_size"] == 50

    def test_backend_store_property(self, mock_backend_store):
        """Access underlying backend store."""
        cached_store = CachingCheckpointStore(mock_backend_store)

        assert cached_store.backend_store is mock_backend_store


# =============================================================================
# FileCheckpointStore Tests
# =============================================================================


class TestFileCheckpointStore:
    """Tests for file-based checkpoint storage."""

    def test_init_creates_directory(self, temp_checkpoint_dir):
        """Initialize creates checkpoint directory."""
        store_dir = Path(temp_checkpoint_dir) / "new_checkpoints"
        store = FileCheckpointStore(str(store_dir))

        assert store_dir.exists()

    @pytest.mark.asyncio
    async def test_save_creates_file(self, temp_checkpoint_dir, sample_checkpoint):
        """Save creates checkpoint file."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        checkpoint_id = await store.save(sample_checkpoint)

        # File created
        files = list(Path(temp_checkpoint_dir).glob("*.json"))
        assert len(files) == 1

        # ID format: workflow_id_timestamp
        assert checkpoint_id.startswith("wf-test-123_")

    @pytest.mark.asyncio
    async def test_save_serializes_correctly(self, temp_checkpoint_dir, sample_checkpoint):
        """Save serializes checkpoint data correctly."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        checkpoint_id = await store.save(sample_checkpoint)
        file_path = Path(temp_checkpoint_dir) / f"{checkpoint_id}.json"

        data = json.loads(file_path.read_text())

        assert data["workflow_id"] == "wf-test-123"
        assert data["definition_id"] == "def-001"
        assert data["current_step"] == "step_2"
        assert data["completed_steps"] == ["step_1"]
        assert data["checkpoint_id"] == checkpoint_id

    @pytest.mark.asyncio
    async def test_load_existing_checkpoint(self, temp_checkpoint_dir, sample_checkpoint):
        """Load returns saved checkpoint."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        checkpoint_id = await store.save(sample_checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        assert loaded.workflow_id == "wf-test-123"
        assert loaded.current_step == "step_2"
        assert loaded.completed_steps == ["step_1"]

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, temp_checkpoint_dir):
        """Load returns None for nonexistent checkpoint."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        loaded = await store.load("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_latest(self, temp_checkpoint_dir, sample_checkpoint, sample_checkpoint_2):
        """Load latest returns most recent checkpoint."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        # Save two checkpoints with slight delay
        await store.save(sample_checkpoint)
        import asyncio

        await asyncio.sleep(0.01)  # Ensure different timestamps
        await store.save(sample_checkpoint_2)

        latest = await store.load_latest("wf-test-123")

        assert latest is not None
        assert latest.current_step == "step_3"  # From sample_checkpoint_2

    @pytest.mark.asyncio
    async def test_load_latest_no_checkpoints(self, temp_checkpoint_dir):
        """Load latest returns None when no checkpoints exist."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        latest = await store.load_latest("nonexistent-workflow")
        assert latest is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(
        self, temp_checkpoint_dir, sample_checkpoint, sample_checkpoint_2
    ):
        """List returns all checkpoint IDs for workflow."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        id1 = await store.save(sample_checkpoint)
        import asyncio

        # FileCheckpointStore uses second-precision timestamps for IDs
        # Need >1s delay to ensure different checkpoint IDs
        await asyncio.sleep(1.1)
        id2 = await store.save(sample_checkpoint_2)

        checkpoints = await store.list_checkpoints("wf-test-123")

        assert len(checkpoints) == 2
        assert id1 in checkpoints
        assert id2 in checkpoints

    @pytest.mark.asyncio
    async def test_list_checkpoints_filters_by_workflow(
        self, temp_checkpoint_dir, sample_checkpoint, sample_checkpoint_different_workflow
    ):
        """List only returns checkpoints for specified workflow."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        await store.save(sample_checkpoint)
        await store.save(sample_checkpoint_different_workflow)

        checkpoints = await store.list_checkpoints("wf-test-123")
        assert len(checkpoints) == 1

        other_checkpoints = await store.list_checkpoints("wf-other-456")
        assert len(other_checkpoints) == 1

    @pytest.mark.asyncio
    async def test_delete(self, temp_checkpoint_dir, sample_checkpoint):
        """Delete removes checkpoint file."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        checkpoint_id = await store.save(sample_checkpoint)
        assert await store.delete(checkpoint_id) is True

        # File removed
        file_path = Path(temp_checkpoint_dir) / f"{checkpoint_id}.json"
        assert not file_path.exists()

        # Load returns None
        assert await store.load(checkpoint_id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, temp_checkpoint_dir):
        """Delete returns False for nonexistent checkpoint."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        result = await store.delete("nonexistent")
        assert result is False


# =============================================================================
# Exception Tests
# =============================================================================


class TestCheckpointExceptions:
    """Tests for checkpoint-related exceptions."""

    def test_checkpoint_validation_error(self):
        """CheckpointValidationError can be raised and caught."""
        with pytest.raises(CheckpointValidationError):
            raise CheckpointValidationError("Invalid checkpoint data")

    def test_connection_timeout_error(self):
        """ConnectionTimeoutError can be raised and caught."""
        with pytest.raises(ConnectionTimeoutError):
            raise ConnectionTimeoutError("Connection timed out after 10s")

    def test_connection_timeout_error_message(self):
        """ConnectionTimeoutError preserves message."""
        error = ConnectionTimeoutError("Redis connection failed")
        assert str(error) == "Redis connection failed"


# =============================================================================
# RedisCheckpointStore Tests (Mocked)
# =============================================================================


class TestRedisCheckpointStore:
    """Tests for Redis checkpoint store with mocked Redis."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis = MagicMock()
        redis.setex = MagicMock()
        redis.get = MagicMock(return_value=None)
        redis.zadd = MagicMock()
        redis.zrevrange = MagicMock(return_value=[])
        redis.delete = MagicMock(return_value=1)
        redis.zrem = MagicMock()
        redis.expire = MagicMock()
        return redis

    @pytest.mark.asyncio
    async def test_save_stores_checkpoint(self, mock_redis, sample_checkpoint):
        """Save stores checkpoint in Redis."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=lambda: mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore(ttl_hours=24)
            store._redis = mock_redis

            checkpoint_id = await store.save(sample_checkpoint)

            # setex called for data and metadata
            assert mock_redis.setex.call_count == 2
            # zadd called for workflow index
            mock_redis.zadd.assert_called_once()

            assert checkpoint_id.startswith("wf-test-123_")

    @pytest.mark.asyncio
    async def test_load_returns_checkpoint(self, mock_redis, sample_checkpoint):
        """Load retrieves and deserializes checkpoint."""
        # Prepare mock data
        checkpoint_data = {
            "workflow_id": "wf-test-123",
            "definition_id": "def-001",
            "current_step": "step_2",
            "completed_steps": ["step_1"],
            "step_outputs": {},
            "context_state": {},
            "created_at": "2024-01-15T12:00:00",
            "checksum": "abc123",
        }
        mock_redis.get = MagicMock(
            side_effect=[
                json.dumps(checkpoint_data).encode(),  # Data
                json.dumps({"compressed": False}).encode(),  # Metadata
            ]
        )

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=lambda: mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            checkpoint = await store.load("wf-test-123_1234567890")

            assert checkpoint is not None
            assert checkpoint.workflow_id == "wf-test-123"
            assert checkpoint.current_step == "step_2"

    @pytest.mark.asyncio
    async def test_load_returns_none_for_missing(self, mock_redis):
        """Load returns None for missing checkpoint."""
        mock_redis.get = MagicMock(return_value=None)

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=lambda: mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            checkpoint = await store.load("nonexistent")
            assert checkpoint is None

    @pytest.mark.asyncio
    async def test_load_latest(self, mock_redis):
        """Load latest retrieves most recent checkpoint."""
        checkpoint_data = {
            "workflow_id": "wf-test",
            "definition_id": "def-001",
            "current_step": "step_3",
            "completed_steps": ["step_1", "step_2"],
            "step_outputs": {},
            "context_state": {},
            "created_at": "2024-01-15T14:00:00",
            "checksum": "",
        }

        mock_redis.zrevrange = MagicMock(return_value=[b"wf-test_1234567890"])
        mock_redis.get = MagicMock(
            side_effect=[
                json.dumps(checkpoint_data).encode(),
                json.dumps({"compressed": False}).encode(),
            ]
        )

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=lambda: mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            checkpoint = await store.load_latest("wf-test")

            mock_redis.zrevrange.assert_called()
            assert checkpoint is not None
            assert checkpoint.current_step == "step_3"

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, mock_redis):
        """List checkpoints returns IDs from sorted set."""
        mock_redis.zrevrange = MagicMock(
            return_value=[
                b"wf-test_111",
                b"wf-test_222",
                b"wf-test_333",
            ]
        )

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=lambda: mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            ids = await store.list_checkpoints("wf-test")

            assert ids == ["wf-test_111", "wf-test_222", "wf-test_333"]

    @pytest.mark.asyncio
    async def test_delete(self, mock_redis):
        """Delete removes checkpoint from Redis."""
        mock_redis.delete = MagicMock(return_value=2)  # 2 keys deleted (data + meta)

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=lambda: mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            result = await store.delete("wf-test_12345")

            assert result is True
            mock_redis.delete.assert_called()
            mock_redis.zrem.assert_called()


# =============================================================================
# KnowledgeMoundCheckpointStore Tests (Mocked)
# =============================================================================


class TestKnowledgeMoundCheckpointStore:
    """Tests for KnowledgeMound checkpoint store with mocked KM."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock KnowledgeMound."""
        mound = MagicMock()
        mound._workspace_id = "test-workspace"
        mound.add_node = AsyncMock(return_value="node-123")
        mound.get_node = AsyncMock(return_value=None)
        mound.query_by_provenance = AsyncMock(return_value=[])
        mound.delete_node = AsyncMock(return_value=True)
        return mound

    @pytest.mark.asyncio
    async def test_save(self, mock_mound, sample_checkpoint):
        """Save stores checkpoint in KnowledgeMound."""
        import builtins

        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        store = KnowledgeMoundCheckpointStore(mock_mound)

        # Mock the dynamic import of KM types
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "aragora.knowledge.mound":
                mock_module = MagicMock()
                mock_module.KnowledgeNode = MagicMock
                mock_module.MemoryTier = MagicMock()
                mock_module.MemoryTier.MEDIUM = "medium"
                mock_module.ProvenanceChain = MagicMock
                return mock_module
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            node_id = await store.save(sample_checkpoint)

        mock_mound.add_node.assert_called_once()
        assert node_id == "node-123"

    @pytest.mark.asyncio
    async def test_load(self, mock_mound, sample_checkpoint):
        """Load retrieves checkpoint from KnowledgeMound."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        # Create mock node
        mock_node = MagicMock()
        mock_node.node_type = "workflow_checkpoint"
        mock_node.content = json.dumps(
            {
                "workflow_id": "wf-test-123",
                "definition_id": "def-001",
                "current_step": "step_2",
                "completed_steps": ["step_1"],
                "step_outputs": {},
                "context_state": {},
                "created_at": "2024-01-15T12:00:00",
                "checksum": "",
            }
        )
        mock_mound.get_node.return_value = mock_node

        store = KnowledgeMoundCheckpointStore(mock_mound)
        checkpoint = await store.load("node-123")

        mock_mound.get_node.assert_called_once_with("node-123")
        assert checkpoint is not None
        assert checkpoint.workflow_id == "wf-test-123"

    @pytest.mark.asyncio
    async def test_load_returns_none_for_missing(self, mock_mound):
        """Load returns None for nonexistent node."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        mock_mound.get_node.return_value = None
        store = KnowledgeMoundCheckpointStore(mock_mound)

        checkpoint = await store.load("nonexistent")
        assert checkpoint is None

    @pytest.mark.asyncio
    async def test_load_returns_none_for_wrong_type(self, mock_mound):
        """Load returns None if node is not a checkpoint."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        mock_node = MagicMock()
        mock_node.node_type = "other_type"
        mock_mound.get_node.return_value = mock_node

        store = KnowledgeMoundCheckpointStore(mock_mound)
        checkpoint = await store.load("node-123")

        assert checkpoint is None

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, mock_mound):
        """List checkpoints queries by provenance."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        mock_node1 = MagicMock()
        mock_node1.id = "node-1"
        mock_node2 = MagicMock()
        mock_node2.id = "node-2"
        mock_mound.query_by_provenance.return_value = [mock_node1, mock_node2]

        store = KnowledgeMoundCheckpointStore(mock_mound)
        ids = await store.list_checkpoints("wf-test")

        assert ids == ["node-1", "node-2"]
        mock_mound.query_by_provenance.assert_called_with(
            source_type="workflow_engine",
            source_id="wf-test",
            node_type="workflow_checkpoint",
            limit=100,
        )

    @pytest.mark.asyncio
    async def test_delete(self, mock_mound):
        """Delete removes node from KnowledgeMound."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        store = KnowledgeMoundCheckpointStore(mock_mound)
        result = await store.delete("node-123")

        assert result is True
        mock_mound.delete_node.assert_called_once_with("node-123")


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestGetCheckpointStore:
    """Tests for get_checkpoint_store factory function."""

    def test_returns_file_store_as_fallback(self, temp_checkpoint_dir):
        """Returns FileCheckpointStore when no other backends available."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import get_checkpoint_store

            store = get_checkpoint_store(fallback_dir=temp_checkpoint_dir)

            assert isinstance(store, FileCheckpointStore)

    def test_uses_provided_mound(self, temp_checkpoint_dir):
        """Uses explicitly provided KnowledgeMound."""
        mock_mound = MagicMock()
        mock_mound._workspace_id = "test"

        from aragora.workflow.checkpoint_store import (
            KnowledgeMoundCheckpointStore,
            get_checkpoint_store,
        )

        store = get_checkpoint_store(mound=mock_mound, fallback_dir=temp_checkpoint_dir)

        assert isinstance(store, KnowledgeMoundCheckpointStore)

    def test_enables_caching(self, temp_checkpoint_dir):
        """Enables caching when requested."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import (
                CachingCheckpointStore,
                get_checkpoint_store,
            )

            store = get_checkpoint_store(
                fallback_dir=temp_checkpoint_dir,
                enable_caching=True,
                cache_size=50,
            )

            assert isinstance(store, CachingCheckpointStore)

    def test_cache_size_customizable(self, temp_checkpoint_dir):
        """Cache size is customizable."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            from aragora.workflow.checkpoint_store import (
                CachingCheckpointStore,
                get_checkpoint_store,
            )

            store = get_checkpoint_store(
                fallback_dir=temp_checkpoint_dir,
                enable_caching=True,
                cache_size=25,
            )

            assert isinstance(store, CachingCheckpointStore)
            assert store._cache._max_size == 25


class TestSetDefaultKnowledgeMound:
    """Tests for set/get default knowledge mound."""

    def test_set_and_get_default_mound(self):
        """Set and retrieve default mound."""
        from aragora.workflow.checkpoint_store import (
            get_default_knowledge_mound,
            set_default_knowledge_mound,
        )

        mock_mound = MagicMock()
        mock_mound._workspace_id = "test"

        # Store original value
        original = get_default_knowledge_mound()

        try:
            set_default_knowledge_mound(mock_mound)
            assert get_default_knowledge_mound() is mock_mound
        finally:
            # Restore original (or None)
            import aragora.workflow.checkpoint_store as cs

            cs._default_mound = original


# =============================================================================
# Integration Tests
# =============================================================================


class TestCheckpointStoreIntegration:
    """Integration tests for checkpoint stores."""

    @pytest.mark.asyncio
    async def test_file_store_roundtrip(self, temp_checkpoint_dir, sample_checkpoint):
        """Full save/load/delete cycle with FileCheckpointStore."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        # Save
        checkpoint_id = await store.save(sample_checkpoint)
        assert checkpoint_id is not None

        # Load
        loaded = await store.load(checkpoint_id)
        assert loaded is not None
        assert loaded.workflow_id == sample_checkpoint.workflow_id
        assert loaded.current_step == sample_checkpoint.current_step
        assert loaded.completed_steps == sample_checkpoint.completed_steps

        # Delete
        deleted = await store.delete(checkpoint_id)
        assert deleted is True

        # Verify deleted
        assert await store.load(checkpoint_id) is None

    @pytest.mark.asyncio
    async def test_caching_store_reduces_backend_calls(
        self, temp_checkpoint_dir, sample_checkpoint
    ):
        """Caching store reduces calls to backend."""
        backend = FileCheckpointStore(temp_checkpoint_dir)
        cached = CachingCheckpointStore(backend)

        # Save checkpoint
        checkpoint_id = await cached.save(sample_checkpoint)

        # Load multiple times - only first should hit backend
        load1 = await cached.load(checkpoint_id)
        load2 = await cached.load(checkpoint_id)
        load3 = await cached.load(checkpoint_id)

        assert load1 is not None
        assert load2 is not None
        assert load3 is not None

        # All loads should return same data
        assert load1.workflow_id == load2.workflow_id == load3.workflow_id

        # Check cache stats
        stats = cached.cache_stats
        assert stats["hits"] >= 2  # At least 2 cache hits (loads 2 and 3)

    @pytest.mark.asyncio
    async def test_multiple_workflows_isolated(
        self, temp_checkpoint_dir, sample_checkpoint, sample_checkpoint_different_workflow
    ):
        """Checkpoints for different workflows are isolated."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        # Save checkpoints for different workflows
        id1 = await store.save(sample_checkpoint)
        id2 = await store.save(sample_checkpoint_different_workflow)

        # List should only return checkpoints for specified workflow
        list1 = await store.list_checkpoints("wf-test-123")
        list2 = await store.list_checkpoints("wf-other-456")

        assert len(list1) == 1
        assert len(list2) == 1
        assert id1 in list1
        assert id2 in list2

        # Load latest should return correct checkpoint
        latest1 = await store.load_latest("wf-test-123")
        latest2 = await store.load_latest("wf-other-456")

        assert latest1.workflow_id == "wf-test-123"
        assert latest2.workflow_id == "wf-other-456"
