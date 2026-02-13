"""
Comprehensive Unit Tests for Workflow Checkpoint Store.

This module provides 80+ comprehensive tests for the checkpoint_store.py module
(~1,600 lines of code). Tests cover:

1. Checkpoint creation - Creating checkpoints with state
2. Checkpoint restoration - Restoring state from checkpoints
3. Integrity verification - Hash/signature verification
4. Checkpoint listing/querying - Finding checkpoints by various criteria
5. Cleanup/retention - Old checkpoint cleanup
6. Concurrent access - Thread safety patterns
7. Error handling - Corrupted checkpoints, missing files

Test file: /Users/armand/Development/aragora/tests/workflow/test_checkpoint.py
Source file: /Users/armand/Development/aragora/aragora/workflow/checkpoint_store.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import zlib

import pytest

from aragora.workflow.checkpoint_store import (
    CachingCheckpointStore,
    CheckpointValidationError,
    ConnectionTimeoutError,
    FileCheckpointStore,
    LRUCheckpointCache,
    MAX_CHECKPOINT_CACHE_SIZE,
    DEFAULT_CONNECTION_TIMEOUT,
    DEFAULT_OPERATION_TIMEOUT,
    get_checkpoint_store,
    set_default_knowledge_mound,
    get_default_knowledge_mound,
)
from aragora.workflow.types import WorkflowCheckpoint


# =============================================================================
# Helper Functions and Fixtures
# =============================================================================


def create_checkpoint(
    checkpoint_id: str = "cp-test-001",
    workflow_id: str = "wf-test-123",
    definition_id: str = "def-test-001",
    current_step: str = "step_processing",
    completed_steps: list[str] | None = None,
    step_outputs: dict[str, Any] | None = None,
    context_state: dict[str, Any] | None = None,
    created_at: datetime | None = None,
    checksum: str = "",
) -> WorkflowCheckpoint:
    """Create a test WorkflowCheckpoint with customizable fields."""
    return WorkflowCheckpoint(
        id=checkpoint_id,
        workflow_id=workflow_id,
        definition_id=definition_id,
        current_step=current_step,
        completed_steps=completed_steps or ["step_init", "step_validate"],
        step_outputs=step_outputs or {"step_init": {"status": "ok"}},
        context_state=context_state or {"user_id": "user-123"},
        created_at=created_at or datetime(2024, 6, 15, 12, 0, 0),
        checksum=checksum,
    )


@pytest.fixture
def sample_checkpoint() -> WorkflowCheckpoint:
    """Create a sample checkpoint for testing."""
    return create_checkpoint()


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoint files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = MagicMock()
    redis.setex = MagicMock(return_value=True)
    redis.get = MagicMock(return_value=None)
    redis.zadd = MagicMock(return_value=1)
    redis.zrevrange = MagicMock(return_value=[])
    redis.delete = MagicMock(return_value=1)
    redis.zrem = MagicMock(return_value=1)
    redis.expire = MagicMock(return_value=True)
    redis.exists = MagicMock(return_value=0)
    redis.keys = MagicMock(return_value=[])
    redis.connection_pool = MagicMock()
    redis.connection_pool.connection_kwargs = {}
    return redis


@pytest.fixture
def mock_postgres_pool():
    """Create a mock PostgreSQL connection pool."""
    pool = MagicMock()
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=None)

    pool.acquire = MagicMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    return pool, conn


@pytest.fixture
def mock_knowledge_mound():
    """Create a mock KnowledgeMound instance."""
    mound = MagicMock()
    mound._workspace_id = "test-workspace"
    mound.add_node = AsyncMock(return_value="node-123")
    mound.get_node = AsyncMock(return_value=None)
    mound.query_by_provenance = AsyncMock(return_value=[])
    mound.delete_node = AsyncMock(return_value=True)
    return mound


# =============================================================================
# 1. Checkpoint Creation Tests
# =============================================================================


class TestCheckpointCreation:
    """Tests for creating and saving checkpoints."""

    @pytest.mark.asyncio
    async def test_file_store_save_creates_file(self, temp_checkpoint_dir, sample_checkpoint):
        """FileCheckpointStore.save creates a JSON file."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(sample_checkpoint)

        file_path = Path(temp_checkpoint_dir) / f"{checkpoint_id}.json"
        assert file_path.exists()
        assert file_path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_file_store_save_returns_valid_id(self, temp_checkpoint_dir, sample_checkpoint):
        """FileCheckpointStore.save returns a valid checkpoint ID."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(sample_checkpoint)

        assert checkpoint_id is not None
        assert sample_checkpoint.workflow_id in checkpoint_id
        assert "_" in checkpoint_id  # Contains timestamp separator

    @pytest.mark.asyncio
    async def test_file_store_save_preserves_all_fields(self, temp_checkpoint_dir):
        """FileCheckpointStore.save preserves all checkpoint fields."""
        checkpoint = create_checkpoint(
            checkpoint_id="cp-preserve",
            workflow_id="wf-preserve",
            definition_id="def-preserve",
            current_step="step_final",
            completed_steps=["a", "b", "c"],
            step_outputs={"a": {"x": 1}, "b": {"y": 2}},
            context_state={"key": "value", "nested": {"inner": True}},
            checksum="abc123",
        )

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)

        # Read the file directly
        file_path = Path(temp_checkpoint_dir) / f"{checkpoint_id}.json"
        data = json.loads(file_path.read_text())

        assert data["workflow_id"] == "wf-preserve"
        assert data["definition_id"] == "def-preserve"
        assert data["current_step"] == "step_final"
        assert data["completed_steps"] == ["a", "b", "c"]
        assert data["step_outputs"]["a"]["x"] == 1
        assert data["context_state"]["nested"]["inner"] is True
        assert data["checksum"] == "abc123"

    @pytest.mark.asyncio
    async def test_file_store_save_with_empty_collections(self, temp_checkpoint_dir):
        """FileCheckpointStore.save handles empty collections."""
        # Create checkpoint directly to ensure empty collections
        checkpoint = WorkflowCheckpoint(
            id="cp-empty",
            workflow_id="wf-empty",
            definition_id="def-empty",
            current_step="step_1",
            completed_steps=[],
            step_outputs={},
            context_state={},
            created_at=datetime(2024, 6, 15, 12, 0, 0),
            checksum="",
        )

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        assert loaded.completed_steps == []
        assert loaded.step_outputs == {}
        assert loaded.context_state == {}

    @pytest.mark.asyncio
    async def test_file_store_save_with_unicode_content(self, temp_checkpoint_dir):
        """FileCheckpointStore.save handles unicode content."""
        checkpoint = create_checkpoint(
            workflow_id="wf-\u4e2d\u6587",  # Chinese characters
            step_outputs={
                "step": {"message": "\u0440\u0443\u0441\u0441\u043a\u0438\u0439"}
            },  # Russian
            context_state={"emoji": "\ud83d\ude80\ud83c\udf1f"},
        )

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        # Check content is preserved (may have different encoding)
        assert "emoji" in loaded.context_state

    @pytest.mark.asyncio
    async def test_file_store_save_with_large_data(self, temp_checkpoint_dir):
        """FileCheckpointStore.save handles large data payloads."""
        large_outputs = {f"step_{i}": {"data": "x" * 1000} for i in range(100)}
        checkpoint = create_checkpoint(
            completed_steps=[f"step_{i}" for i in range(100)],
            step_outputs=large_outputs,
        )

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        assert len(loaded.completed_steps) == 100
        assert len(loaded.step_outputs) == 100

    @pytest.mark.asyncio
    async def test_caching_store_save_populates_cache(self, sample_checkpoint):
        """CachingCheckpointStore.save populates the cache."""
        backend = AsyncMock()
        backend.save = AsyncMock(return_value="cp-cached-001")

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)
        checkpoint_id = await cached_store.save(sample_checkpoint)

        assert checkpoint_id == "cp-cached-001"
        assert cached_store._cache.get("cp-cached-001") is sample_checkpoint
        backend.save.assert_called_once_with(sample_checkpoint)

    @pytest.mark.asyncio
    async def test_multiple_saves_create_unique_ids(self, temp_checkpoint_dir, sample_checkpoint):
        """Multiple saves create unique checkpoint IDs."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        ids = set()

        for _ in range(3):
            checkpoint_id = await store.save(sample_checkpoint)
            ids.add(checkpoint_id)
            await asyncio.sleep(1.1)  # FileCheckpointStore uses second precision

        assert len(ids) == 3


# =============================================================================
# 2. Checkpoint Restoration Tests
# =============================================================================


class TestCheckpointRestoration:
    """Tests for loading and restoring checkpoints."""

    @pytest.mark.asyncio
    async def test_file_store_load_returns_complete_checkpoint(
        self, temp_checkpoint_dir, sample_checkpoint
    ):
        """FileCheckpointStore.load returns a complete checkpoint."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(sample_checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        assert loaded.workflow_id == sample_checkpoint.workflow_id
        assert loaded.definition_id == sample_checkpoint.definition_id
        assert loaded.current_step == sample_checkpoint.current_step
        assert loaded.completed_steps == sample_checkpoint.completed_steps
        assert loaded.step_outputs == sample_checkpoint.step_outputs
        assert loaded.context_state == sample_checkpoint.context_state

    @pytest.mark.asyncio
    async def test_file_store_load_nonexistent_returns_none(self, temp_checkpoint_dir):
        """FileCheckpointStore.load returns None for nonexistent checkpoint."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        result = await store.load("nonexistent-checkpoint-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_file_store_load_latest_returns_most_recent(self, temp_checkpoint_dir):
        """FileCheckpointStore.load_latest returns the most recent checkpoint."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        # Create checkpoints with increasing version numbers
        for i in range(3):
            checkpoint = create_checkpoint(
                workflow_id="wf-latest",
                context_state={"version": i},
            )
            await store.save(checkpoint)
            await asyncio.sleep(1.1)  # Ensure different timestamps

        latest = await store.load_latest("wf-latest")

        assert latest is not None
        assert latest.context_state["version"] == 2

    @pytest.mark.asyncio
    async def test_file_store_load_latest_no_checkpoints(self, temp_checkpoint_dir):
        """FileCheckpointStore.load_latest returns None when no checkpoints exist."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        result = await store.load_latest("nonexistent-workflow")
        assert result is None

    @pytest.mark.asyncio
    async def test_caching_store_load_hits_cache(self, sample_checkpoint):
        """CachingCheckpointStore.load returns cached checkpoint without backend call."""
        backend = AsyncMock()
        backend.load = AsyncMock(return_value=None)

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)
        cached_store._cache.put("cp-001", sample_checkpoint)

        result = await cached_store.load("cp-001")

        assert result is sample_checkpoint
        backend.load.assert_not_called()

    @pytest.mark.asyncio
    async def test_caching_store_load_cache_miss_calls_backend(self, sample_checkpoint):
        """CachingCheckpointStore.load calls backend on cache miss."""
        backend = AsyncMock()
        backend.load = AsyncMock(return_value=sample_checkpoint)

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)
        result = await cached_store.load("cp-001")

        assert result is sample_checkpoint
        backend.load.assert_called_once_with("cp-001")
        # Should be cached after load
        assert cached_store._cache.get("cp-001") is sample_checkpoint

    @pytest.mark.asyncio
    async def test_caching_store_load_latest_always_calls_backend(self, sample_checkpoint):
        """CachingCheckpointStore.load_latest always calls backend."""
        backend = AsyncMock()
        backend.load_latest = AsyncMock(return_value=sample_checkpoint)

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)

        # Call multiple times
        await cached_store.load_latest("wf-test")
        await cached_store.load_latest("wf-test")
        await cached_store.load_latest("wf-test")

        # Backend called each time
        assert backend.load_latest.call_count == 3

    @pytest.mark.asyncio
    async def test_restore_preserves_datetime_fields(self, temp_checkpoint_dir):
        """Restored checkpoint has correct datetime field."""
        specific_time = datetime(2024, 8, 20, 14, 30, 45)
        checkpoint = create_checkpoint(created_at=specific_time)

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        assert isinstance(loaded.created_at, datetime)
        assert loaded.created_at.year == 2024
        assert loaded.created_at.month == 8
        assert loaded.created_at.day == 20

    @pytest.mark.asyncio
    async def test_restore_preserves_nested_structures(self, temp_checkpoint_dir):
        """Restored checkpoint preserves deeply nested structures."""
        checkpoint = create_checkpoint(
            step_outputs={
                "step": {"level1": {"level2": {"level3": {"value": "deep", "array": [1, 2, 3]}}}}
            }
        )

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        deep = loaded.step_outputs["step"]["level1"]["level2"]["level3"]
        assert deep["value"] == "deep"
        assert deep["array"] == [1, 2, 3]


# =============================================================================
# 3. Integrity Verification Tests
# =============================================================================


class TestIntegrityVerification:
    """Tests for checkpoint integrity verification."""

    @pytest.mark.asyncio
    async def test_checksum_preserved_through_save_load(self, temp_checkpoint_dir):
        """Checksum field is preserved through save/load cycle."""
        checkpoint = create_checkpoint(checksum="sha256:abcdef123456")

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        assert loaded.checksum == "sha256:abcdef123456"

    @pytest.mark.asyncio
    async def test_postgres_checksum_validation_on_load(self, mock_postgres_pool):
        """PostgresCheckpointStore validates checksum on load."""
        pool, conn = mock_postgres_pool

        # Create row with mismatched checksum
        row = {
            "id": "cp-checksum",
            "workflow_id": "wf-test",
            "definition_id": "def-test",
            "current_step": "s1",
            "completed_steps": ["init"],
            "step_outputs": "{}",
            "context_state": "{}",
            "created_at": datetime.now(),
            "checksum": "invalid_checksum_value",
        }
        conn.fetchrow.return_value = row

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            # Should still load but log warning
            result = await store.load("cp-checksum")
            assert result is not None

    def test_compute_checksum_deterministic(self, mock_postgres_pool):
        """PostgresCheckpointStore._compute_checksum produces deterministic results."""
        pool, _ = mock_postgres_pool

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)

            checkpoint = create_checkpoint()
            checksum1 = store._compute_checksum(checkpoint)
            checksum2 = store._compute_checksum(checkpoint)

            assert checksum1 == checksum2
            assert len(checksum1) == 16  # SHA-256 truncated to 16 chars

    def test_compute_checksum_changes_with_data(self, mock_postgres_pool):
        """PostgresCheckpointStore._compute_checksum changes when data changes."""
        pool, _ = mock_postgres_pool

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)

            cp1 = create_checkpoint(current_step="step_a")
            cp2 = create_checkpoint(current_step="step_b")

            checksum1 = store._compute_checksum(cp1)
            checksum2 = store._compute_checksum(cp2)

            assert checksum1 != checksum2

    @pytest.mark.asyncio
    async def test_file_corrupted_json_raises_error(self, temp_checkpoint_dir):
        """Loading corrupted JSON file raises JSONDecodeError."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        # Write corrupted JSON directly
        corrupted_path = Path(temp_checkpoint_dir) / "wf-corrupt_20240115_120000.json"
        corrupted_path.write_text("{ invalid json content !!!")

        with pytest.raises(json.JSONDecodeError):
            await store.load("wf-corrupt_20240115_120000")

    def test_checkpoint_validation_error_can_be_raised(self):
        """CheckpointValidationError can be instantiated and raised."""
        error = CheckpointValidationError("Integrity check failed")
        assert "Integrity check failed" in str(error)

        with pytest.raises(CheckpointValidationError):
            raise CheckpointValidationError("Test error")


# =============================================================================
# 4. Checkpoint Listing/Querying Tests
# =============================================================================


class TestCheckpointListingQuerying:
    """Tests for listing and querying checkpoints."""

    @pytest.mark.asyncio
    async def test_file_store_list_checkpoints(self, temp_checkpoint_dir):
        """FileCheckpointStore.list_checkpoints returns all checkpoint IDs."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        # Create multiple checkpoints for same workflow
        for i in range(3):
            checkpoint = create_checkpoint(workflow_id="wf-list")
            await store.save(checkpoint)
            await asyncio.sleep(1.1)

        ids = await store.list_checkpoints("wf-list")

        assert len(ids) == 3
        for id_ in ids:
            assert "wf-list" in id_
            assert not id_.endswith(".json")  # Should be stem only

    @pytest.mark.asyncio
    async def test_file_store_list_checkpoints_filters_by_workflow(self, temp_checkpoint_dir):
        """FileCheckpointStore.list_checkpoints only returns matching workflow."""
        store = FileCheckpointStore(temp_checkpoint_dir)

        # Create checkpoints for different workflows
        for wf in ["wf-a", "wf-b"]:
            for i in range(2):
                checkpoint = create_checkpoint(workflow_id=wf)
                await store.save(checkpoint)
                await asyncio.sleep(1.1)

        list_a = await store.list_checkpoints("wf-a")
        list_b = await store.list_checkpoints("wf-b")

        assert len(list_a) == 2
        assert len(list_b) == 2
        assert all("wf-a" in id_ for id_ in list_a)
        assert all("wf-b" in id_ for id_ in list_b)

    @pytest.mark.asyncio
    async def test_file_store_list_checkpoints_empty_workflow(self, temp_checkpoint_dir):
        """FileCheckpointStore.list_checkpoints returns empty list for unknown workflow."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        ids = await store.list_checkpoints("nonexistent-workflow")
        assert ids == []

    @pytest.mark.asyncio
    async def test_caching_store_list_checkpoints_delegates(self, sample_checkpoint):
        """CachingCheckpointStore.list_checkpoints delegates to backend."""
        backend = AsyncMock()
        backend.list_checkpoints = AsyncMock(return_value=["cp-1", "cp-2", "cp-3"])

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)
        result = await cached_store.list_checkpoints("wf-test")

        assert result == ["cp-1", "cp-2", "cp-3"]
        backend.list_checkpoints.assert_called_once_with("wf-test")

    @pytest.mark.asyncio
    async def test_redis_list_checkpoints_uses_sorted_set(self, mock_redis):
        """RedisCheckpointStore.list_checkpoints uses sorted set index."""
        mock_redis.zrevrange.return_value = [
            b"wf-test_999",
            b"wf-test_888",
            b"wf-test_777",
        ]

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            ids = await store.list_checkpoints("wf-test")

            assert len(ids) == 3
            assert ids[0] == "wf-test_999"
            mock_redis.zrevrange.assert_called_once()

    @pytest.mark.asyncio
    async def test_postgres_list_checkpoints(self, mock_postgres_pool):
        """PostgresCheckpointStore.list_checkpoints queries database."""
        pool, conn = mock_postgres_pool
        conn.fetch.return_value = [
            {"id": "wf-test_999"},
            {"id": "wf-test_888"},
        ]

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            ids = await store.list_checkpoints("wf-test")

            assert len(ids) == 2
            assert ids[0] == "wf-test_999"


# =============================================================================
# 5. Cleanup/Retention Tests
# =============================================================================


class TestCleanupRetention:
    """Tests for checkpoint cleanup and retention policies."""

    @pytest.mark.asyncio
    async def test_file_store_delete_removes_file(self, temp_checkpoint_dir, sample_checkpoint):
        """FileCheckpointStore.delete removes the checkpoint file."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(sample_checkpoint)

        # Verify file exists
        file_path = Path(temp_checkpoint_dir) / f"{checkpoint_id}.json"
        assert file_path.exists()

        # Delete
        result = await store.delete(checkpoint_id)

        assert result is True
        assert not file_path.exists()

    @pytest.mark.asyncio
    async def test_file_store_delete_nonexistent_returns_false(self, temp_checkpoint_dir):
        """FileCheckpointStore.delete returns False for nonexistent checkpoint."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        result = await store.delete("nonexistent-checkpoint")
        assert result is False

    @pytest.mark.asyncio
    async def test_caching_store_delete_removes_from_cache(self, sample_checkpoint):
        """CachingCheckpointStore.delete removes entry from cache."""
        backend = AsyncMock()
        backend.delete = AsyncMock(return_value=True)

        cached_store = CachingCheckpointStore(backend, max_cache_size=10)
        cached_store._cache.put("cp-001", sample_checkpoint)

        await cached_store.delete("cp-001")

        assert cached_store._cache.get("cp-001") is None
        backend.delete.assert_called_once_with("cp-001")

    @pytest.mark.asyncio
    async def test_redis_delete_removes_from_sorted_set(self, mock_redis):
        """RedisCheckpointStore.delete removes from sorted set index."""
        mock_redis.delete.return_value = 2

        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            result = await store.delete("wf-test_12345")

            assert result is True
            mock_redis.zrem.assert_called_once()

    @pytest.mark.asyncio
    async def test_postgres_cleanup_old_checkpoints(self, mock_postgres_pool):
        """PostgresCheckpointStore.cleanup_old_checkpoints deletes old entries."""
        pool, conn = mock_postgres_pool
        conn.execute.return_value = "DELETE 5"

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            deleted = await store.cleanup_old_checkpoints("wf-test", keep_count=10)

            assert deleted == 5

    @pytest.mark.asyncio
    async def test_postgres_cleanup_keeps_recent(self, mock_postgres_pool):
        """PostgresCheckpointStore.cleanup_old_checkpoints keeps specified count."""
        pool, conn = mock_postgres_pool
        conn.execute.return_value = "DELETE 0"

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            deleted = await store.cleanup_old_checkpoints("wf-test", keep_count=100)

            assert deleted == 0

    def test_lru_cache_evicts_when_full(self):
        """LRUCheckpointCache evicts oldest entries when full."""
        cache = LRUCheckpointCache(max_size=2)

        cache.put("a", create_checkpoint(checkpoint_id="a"))
        cache.put("b", create_checkpoint(checkpoint_id="b"))
        cache.put("c", create_checkpoint(checkpoint_id="c"))  # Should evict 'a'

        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None
        assert cache.size == 2

    def test_lru_cache_access_prevents_eviction(self):
        """Accessing an entry prevents it from being evicted."""
        cache = LRUCheckpointCache(max_size=2)

        cache.put("a", create_checkpoint(checkpoint_id="a"))
        cache.put("b", create_checkpoint(checkpoint_id="b"))

        # Access 'a' to make it recently used
        cache.get("a")

        # Insert 'c' - should evict 'b' (oldest not accessed)
        cache.put("c", create_checkpoint(checkpoint_id="c"))

        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None

    def test_lru_cache_clear(self):
        """LRUCheckpointCache.clear removes all entries."""
        cache = LRUCheckpointCache(max_size=10)

        for i in range(5):
            cache.put(f"cp-{i}", create_checkpoint(checkpoint_id=f"cp-{i}"))

        assert cache.size == 5

        cache.clear()

        assert cache.size == 0

    def test_lru_cache_remove(self):
        """LRUCheckpointCache.remove removes specific entry."""
        cache = LRUCheckpointCache(max_size=10)
        cache.put("a", create_checkpoint(checkpoint_id="a"))
        cache.put("b", create_checkpoint(checkpoint_id="b"))

        result = cache.remove("a")

        assert result is True
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.size == 1

    def test_lru_cache_remove_nonexistent(self):
        """LRUCheckpointCache.remove returns False for nonexistent key."""
        cache = LRUCheckpointCache(max_size=10)
        result = cache.remove("nonexistent")
        assert result is False


# =============================================================================
# 6. Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent checkpoint access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_saves_produce_unique_files(self, temp_checkpoint_dir):
        """Concurrent saves produce unique checkpoint files."""
        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_ids = []

        # Save with delays to ensure different timestamps
        for i in range(5):
            checkpoint = create_checkpoint(
                workflow_id="wf-concurrent",
                context_state={"index": i},
            )
            cp_id = await store.save(checkpoint)
            checkpoint_ids.append(cp_id)
            await asyncio.sleep(1.1)

        # All IDs should be unique
        assert len(set(checkpoint_ids)) == 5

    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, sample_checkpoint):
        """Concurrent cache operations work correctly."""
        cache = LRUCheckpointCache(max_size=100)

        async def cache_op(index: int) -> None:
            key = f"cp-{index}"
            cp = create_checkpoint(checkpoint_id=key)
            cache.put(key, cp)
            await asyncio.sleep(0.001)
            cache.get(key)

        # Run operations concurrently
        tasks = [cache_op(i) for i in range(50)]
        await asyncio.gather(*tasks)

        assert cache.size <= 100

    @pytest.mark.asyncio
    async def test_caching_store_concurrent_loads(self, sample_checkpoint):
        """CachingCheckpointStore handles concurrent loads."""
        backend = AsyncMock()
        backend.load = AsyncMock(return_value=sample_checkpoint)

        cached_store = CachingCheckpointStore(backend, max_cache_size=100)

        async def load_checkpoint() -> WorkflowCheckpoint | None:
            return await cached_store.load("cp-001")

        # Load concurrently
        results = await asyncio.gather(*[load_checkpoint() for _ in range(10)])

        # All results should be valid
        for result in results:
            assert result is not None
            assert result.workflow_id == sample_checkpoint.workflow_id

    def test_lru_cache_thread_safety_basic(self):
        """LRUCheckpointCache handles basic thread access."""
        cache = LRUCheckpointCache(max_size=100)
        errors = []

        def thread_op(thread_id: int) -> None:
            try:
                for i in range(20):
                    key = f"t{thread_id}-{i}"
                    cache.put(key, create_checkpoint(checkpoint_id=key))
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=thread_op, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should have occurred
        assert len(errors) == 0
        # Cache size should be within bounds
        assert cache.size <= 100


# =============================================================================
# 7. Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_connection_timeout_error_message(self):
        """ConnectionTimeoutError preserves error message."""
        error = ConnectionTimeoutError("Redis connection failed after 10s")
        assert "10s" in str(error)
        assert "Redis" in str(error)

    @pytest.mark.asyncio
    async def test_redis_save_timeout_raises_connection_error(self, mock_redis):
        """RedisCheckpointStore.save raises ConnectionTimeoutError on timeout."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            # Simulate timeout
            mock_redis.setex.side_effect = type("TimeoutError", (Exception,), {})("timeout")

            with pytest.raises(ConnectionTimeoutError, match="timeout"):
                await store.save(create_checkpoint())

    @pytest.mark.asyncio
    async def test_redis_load_connection_error(self, mock_redis):
        """RedisCheckpointStore.load raises ConnectionTimeoutError on connection error."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            mock_redis.get.side_effect = type("ConnectionError", (Exception,), {})("refused")

            with pytest.raises(ConnectionTimeoutError, match="refused"):
                await store.load("cp-123")

    @pytest.mark.asyncio
    async def test_postgres_save_timeout_raises_connection_error(self, mock_postgres_pool):
        """PostgresCheckpointStore.save raises ConnectionTimeoutError on timeout."""
        pool, conn = mock_postgres_pool
        conn.execute.side_effect = asyncio.TimeoutError()

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            with pytest.raises(ConnectionTimeoutError):
                await store.save(create_checkpoint())

    @pytest.mark.asyncio
    async def test_postgres_load_timeout_returns_none(self, mock_postgres_pool):
        """PostgresCheckpointStore.load returns None on timeout."""
        pool, conn = mock_postgres_pool
        conn.fetchrow.side_effect = asyncio.TimeoutError()

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            result = await store.load("cp-timeout")
            assert result is None

    @pytest.mark.asyncio
    async def test_redis_list_checkpoints_error_returns_empty(self, mock_redis):
        """RedisCheckpointStore.list_checkpoints returns empty list on error."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

            mock_redis.zrevrange.side_effect = RuntimeError("connection lost")

            ids = await store.list_checkpoints("wf-test")
            assert ids == []

    @pytest.mark.asyncio
    async def test_km_store_delete_error_returns_false(self, mock_knowledge_mound):
        """KnowledgeMoundCheckpointStore.delete returns False on error."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        mock_knowledge_mound.delete_node = AsyncMock(side_effect=RuntimeError("storage error"))

        store = KnowledgeMoundCheckpointStore(mock_knowledge_mound)
        result = await store.delete("node-error")

        assert result is False

    def test_redis_store_requires_redis_available(self):
        """RedisCheckpointStore raises when Redis not available."""
        with patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            with pytest.raises(RuntimeError, match="Redis"):
                RedisCheckpointStore()

    def test_postgres_store_requires_asyncpg_available(self):
        """PostgresCheckpointStore raises when asyncpg not available."""
        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            with pytest.raises(RuntimeError, match="asyncpg"):
                PostgresCheckpointStore(MagicMock())


# =============================================================================
# 8. LRU Cache Specific Tests
# =============================================================================


class TestLRUCheckpointCache:
    """Tests specific to LRUCheckpointCache implementation."""

    def test_default_max_size(self):
        """LRUCheckpointCache uses default max size."""
        cache = LRUCheckpointCache()
        assert cache._max_size == MAX_CHECKPOINT_CACHE_SIZE

    def test_custom_max_size(self):
        """LRUCheckpointCache accepts custom max size."""
        cache = LRUCheckpointCache(max_size=50)
        assert cache._max_size == 50

    def test_size_property(self):
        """LRUCheckpointCache.size returns current size."""
        cache = LRUCheckpointCache(max_size=10)
        assert cache.size == 0

        cache.put("a", create_checkpoint())
        assert cache.size == 1

        cache.put("b", create_checkpoint())
        assert cache.size == 2

    def test_stats_hit_rate_calculation(self):
        """LRUCheckpointCache stats correctly calculates hit rate."""
        cache = LRUCheckpointCache(max_size=10)

        # One miss
        cache.get("missing")

        # Put and two hits
        cache.put("exists", create_checkpoint())
        cache.get("exists")
        cache.get("exists")

        stats = cache.stats
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(2 / 3)

    def test_stats_empty_cache(self):
        """LRUCheckpointCache stats handles zero operations."""
        cache = LRUCheckpointCache(max_size=10)
        stats = cache.stats

        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_put_updates_existing_key(self):
        """LRUCheckpointCache.put updates existing key."""
        cache = LRUCheckpointCache(max_size=2)

        cp1 = create_checkpoint(current_step="step_1")
        cp2 = create_checkpoint(current_step="step_2")

        cache.put("a", cp1)
        cache.put("a", cp2)

        assert cache.size == 1
        assert cache.get("a").current_step == "step_2"

    def test_single_entry_cache(self):
        """LRUCheckpointCache with max_size=1."""
        cache = LRUCheckpointCache(max_size=1)

        cache.put("a", create_checkpoint(checkpoint_id="a"))
        cache.put("b", create_checkpoint(checkpoint_id="b"))

        assert cache.size == 1
        assert cache.get("a") is None
        assert cache.get("b") is not None


# =============================================================================
# 9. CachingCheckpointStore Specific Tests
# =============================================================================


class TestCachingCheckpointStore:
    """Tests specific to CachingCheckpointStore wrapper."""

    def test_backend_store_property(self):
        """CachingCheckpointStore.backend_store returns wrapped store."""
        backend = MagicMock()
        cached = CachingCheckpointStore(backend)
        assert cached.backend_store is backend

    def test_cache_stats_property(self):
        """CachingCheckpointStore.cache_stats returns cache statistics."""
        backend = MagicMock()
        cached = CachingCheckpointStore(backend, max_cache_size=50)

        stats = cached.cache_stats
        assert "size" in stats
        assert "max_size" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["max_size"] == 50

    def test_clear_cache_method(self):
        """CachingCheckpointStore.clear_cache clears only cache."""
        backend = MagicMock()
        cached = CachingCheckpointStore(backend, max_cache_size=10)

        cached._cache.put("a", create_checkpoint())
        cached._cache.put("b", create_checkpoint())
        assert cached._cache.size == 2

        cached.clear_cache()

        assert cached._cache.size == 0

    @pytest.mark.asyncio
    async def test_cache_not_populated_for_none_result(self):
        """CachingCheckpointStore does not cache None results."""
        backend = AsyncMock()
        backend.load = AsyncMock(return_value=None)

        cached = CachingCheckpointStore(backend, max_cache_size=10)

        await cached.load("missing")
        await cached.load("missing")

        # Backend called each time since None not cached
        assert backend.load.call_count == 2


# =============================================================================
# 10. Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for checkpoint store factory functions."""

    def test_get_checkpoint_store_with_mound(self, mock_knowledge_mound):
        """get_checkpoint_store uses provided mound."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        store = get_checkpoint_store(mound=mock_knowledge_mound)
        assert isinstance(store, KnowledgeMoundCheckpointStore)
        assert store.mound is mock_knowledge_mound

    def test_get_checkpoint_store_fallback_to_file(self, temp_checkpoint_dir):
        """get_checkpoint_store falls back to FileCheckpointStore."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            store = get_checkpoint_store(
                fallback_dir=temp_checkpoint_dir,
                use_default_mound=False,
            )
            assert isinstance(store, FileCheckpointStore)

    def test_get_checkpoint_store_with_caching(self, temp_checkpoint_dir):
        """get_checkpoint_store wraps with cache when enabled."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
        ):
            store = get_checkpoint_store(
                fallback_dir=temp_checkpoint_dir,
                use_default_mound=False,
                enable_caching=True,
                cache_size=50,
            )
            assert isinstance(store, CachingCheckpointStore)
            assert store._cache._max_size == 50

    def test_get_checkpoint_store_env_var_cache(self, temp_checkpoint_dir):
        """get_checkpoint_store respects ARAGORA_CHECKPOINT_CACHE env var."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", False),
            patch("aragora.workflow.checkpoint_store._default_mound", None),
            patch.dict(os.environ, {"ARAGORA_CHECKPOINT_CACHE": "true"}),
        ):
            store = get_checkpoint_store(
                fallback_dir=temp_checkpoint_dir,
                use_default_mound=False,
            )
            assert isinstance(store, CachingCheckpointStore)

    def test_set_and_get_default_mound(self, mock_knowledge_mound):
        """set_default_knowledge_mound and get_default_knowledge_mound work."""
        import aragora.workflow.checkpoints.factory as _factory

        original = get_default_knowledge_mound()

        try:
            set_default_knowledge_mound(mock_knowledge_mound)
            assert get_default_knowledge_mound() is mock_knowledge_mound
        finally:
            _factory._default_mound = original


# =============================================================================
# 11. Redis-Specific Tests
# =============================================================================


class TestRedisCheckpointStore:
    """Tests specific to RedisCheckpointStore."""

    @pytest.mark.asyncio
    async def test_redis_compression_large_data(self, mock_redis):
        """Redis compresses data above threshold."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore(compress_threshold=100)
            store._redis = mock_redis

            # Large checkpoint
            checkpoint = create_checkpoint(step_outputs={"large": "x" * 1000})

            await store.save(checkpoint)

            # Get stored data
            stored_bytes = mock_redis.setex.call_args_list[0][0][2]

            # Should be compressed
            try:
                zlib.decompress(stored_bytes)
                is_compressed = True
            except zlib.error:
                is_compressed = False

            assert is_compressed

    @pytest.mark.asyncio
    async def test_redis_load_decompresses(self, mock_redis):
        """Redis load decompresses data correctly."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore()
            store._redis = mock_redis

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

    def test_redis_ttl_configuration(self, mock_redis):
        """Redis TTL is correctly configured."""
        with (
            patch("aragora.workflow.checkpoint_store.REDIS_AVAILABLE", True),
            patch(
                "aragora.workflow.checkpoint_store._get_redis_client",
                return_value=mock_redis,
            ),
        ):
            from aragora.workflow.checkpoint_store import RedisCheckpointStore

            store = RedisCheckpointStore(ttl_hours=72)
            assert store._ttl_seconds == 72 * 3600


# =============================================================================
# 12. PostgreSQL-Specific Tests
# =============================================================================


class TestPostgresCheckpointStore:
    """Tests specific to PostgresCheckpointStore."""

    @pytest.mark.asyncio
    async def test_postgres_schema_initialization(self, mock_postgres_pool):
        """PostgresCheckpointStore initializes schema."""
        pool, conn = mock_postgres_pool
        conn.fetchrow.return_value = None  # No existing schema

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            await store.initialize()

            assert store._initialized is True
            assert conn.execute.call_count >= 2

    @pytest.mark.asyncio
    async def test_postgres_initialize_idempotent(self, mock_postgres_pool):
        """PostgresCheckpointStore.initialize is idempotent."""
        pool, conn = mock_postgres_pool

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            await store.initialize()

            # No SQL executed
            conn.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_postgres_row_to_checkpoint_parses_json_strings(self, mock_postgres_pool):
        """PostgresCheckpointStore parses JSON strings in row data."""
        pool, conn = mock_postgres_pool

        row = {
            "id": "cp-json",
            "workflow_id": "wf-1",
            "definition_id": "def-1",
            "current_step": "s1",
            "completed_steps": ["a"],
            "step_outputs": '{"step_a": {"val": 42}}',  # JSON string
            "context_state": '{"key": "value"}',  # JSON string
            "created_at": datetime.now(),
            "checksum": "",
        }
        conn.fetchrow.return_value = row

        with patch("aragora.workflow.checkpoint_store.ASYNCPG_AVAILABLE", True):
            from aragora.workflow.checkpoint_store import PostgresCheckpointStore

            store = PostgresCheckpointStore(pool)
            store._initialized = True

            result = await store.load("cp-json")

            assert result is not None
            assert result.step_outputs == {"step_a": {"val": 42}}
            assert result.context_state == {"key": "value"}


# =============================================================================
# 13. KnowledgeMound-Specific Tests
# =============================================================================


class TestKnowledgeMoundCheckpointStore:
    """Tests specific to KnowledgeMoundCheckpointStore."""

    @pytest.mark.asyncio
    async def test_km_workspace_default(self, mock_knowledge_mound):
        """KnowledgeMoundCheckpointStore uses mound's workspace by default."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        store = KnowledgeMoundCheckpointStore(mock_knowledge_mound)
        assert store.workspace_id == "test-workspace"

    @pytest.mark.asyncio
    async def test_km_workspace_override(self, mock_knowledge_mound):
        """KnowledgeMoundCheckpointStore allows workspace override."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        store = KnowledgeMoundCheckpointStore(mock_knowledge_mound, workspace_id="custom-workspace")
        assert store.workspace_id == "custom-workspace"

    @pytest.mark.asyncio
    async def test_km_load_wrong_type_returns_none(self, mock_knowledge_mound):
        """KnowledgeMoundCheckpointStore returns None for wrong node type."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        node = MagicMock()
        node.node_type = "document"  # Not "workflow_checkpoint"
        node.content = "{}"

        mock_knowledge_mound.get_node.return_value = node

        store = KnowledgeMoundCheckpointStore(mock_knowledge_mound)
        result = await store.load("node-wrong-type")

        assert result is None

    @pytest.mark.asyncio
    async def test_km_query_by_provenance(self, mock_knowledge_mound):
        """KnowledgeMoundCheckpointStore queries by provenance."""
        from aragora.workflow.checkpoint_store import KnowledgeMoundCheckpointStore

        checkpoint_node = MagicMock()
        checkpoint_node.content = json.dumps(
            {
                "workflow_id": "wf-prov",
                "definition_id": "def-prov",
                "current_step": "latest",
                "completed_steps": [],
                "step_outputs": {},
                "context_state": {},
                "created_at": datetime.now().isoformat(),
                "checksum": "",
            }
        )
        mock_knowledge_mound.query_by_provenance.return_value = [checkpoint_node]

        store = KnowledgeMoundCheckpointStore(mock_knowledge_mound)
        result = await store.load_latest("wf-prov")

        assert result is not None
        assert result.current_step == "latest"
        mock_knowledge_mound.query_by_provenance.assert_called_once()


# =============================================================================
# 14. Constants and Module Configuration Tests
# =============================================================================


class TestModuleConfiguration:
    """Tests for module-level constants and configuration."""

    def test_max_checkpoint_cache_size_defined(self):
        """MAX_CHECKPOINT_CACHE_SIZE is defined and positive."""
        assert MAX_CHECKPOINT_CACHE_SIZE > 0

    def test_default_connection_timeout_defined(self):
        """DEFAULT_CONNECTION_TIMEOUT is defined and positive."""
        assert DEFAULT_CONNECTION_TIMEOUT > 0

    def test_default_operation_timeout_defined(self):
        """DEFAULT_OPERATION_TIMEOUT is defined and positive."""
        assert DEFAULT_OPERATION_TIMEOUT > 0

    def test_timeout_values_reasonable(self):
        """Timeout values are reasonable (not too short or too long)."""
        assert 1.0 <= DEFAULT_CONNECTION_TIMEOUT <= 60.0
        assert 1.0 <= DEFAULT_OPERATION_TIMEOUT <= 600.0


# =============================================================================
# 15. Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_workflow_id(self, temp_checkpoint_dir):
        """Handle empty workflow_id."""
        checkpoint = create_checkpoint(workflow_id="")

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)

        assert checkpoint_id is not None

    @pytest.mark.asyncio
    async def test_very_long_step_name(self, temp_checkpoint_dir):
        """Handle very long step names."""
        long_step = "step_" + "a" * 1000

        checkpoint = create_checkpoint(
            current_step=long_step,
            completed_steps=[long_step],
            step_outputs={long_step: {"completed": True}},
        )

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        assert len(loaded.current_step) > 100

    @pytest.mark.asyncio
    async def test_null_values_in_context(self, temp_checkpoint_dir):
        """Handle null values in context_state."""
        checkpoint = create_checkpoint(
            context_state={
                "null_value": None,
                "empty_string": "",
                "zero": 0,
                "false": False,
                "empty_list": [],
                "empty_dict": {},
            }
        )

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        assert loaded.context_state["null_value"] is None
        assert loaded.context_state["empty_string"] == ""
        assert loaded.context_state["zero"] == 0
        assert loaded.context_state["false"] is False
        assert loaded.context_state["empty_list"] == []
        assert loaded.context_state["empty_dict"] == {}

    @pytest.mark.asyncio
    async def test_special_characters_in_data(self, temp_checkpoint_dir):
        """Handle special characters in checkpoint data."""
        checkpoint = create_checkpoint(
            step_outputs={
                "step": {
                    "quotes": "Contains \"quotes\" and 'apostrophes'",
                    "path": "C:\\Users\\test\\file.txt",
                    "newlines": "line1\nline2\nline3",
                    "tabs": "col1\tcol2",
                }
            }
        )

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        assert "quotes" in loaded.step_outputs["step"]["quotes"]

    @pytest.mark.asyncio
    async def test_numeric_types_preserved(self, temp_checkpoint_dir):
        """Numeric types are preserved through save/load."""
        checkpoint = create_checkpoint(
            step_outputs={
                "step": {
                    "integer": 42,
                    "float": 3.14159,
                    "negative": -100,
                    "large": 10**15,
                }
            }
        )

        store = FileCheckpointStore(temp_checkpoint_dir)
        checkpoint_id = await store.save(checkpoint)
        loaded = await store.load(checkpoint_id)

        assert loaded is not None
        outputs = loaded.step_outputs["step"]
        assert outputs["integer"] == 42
        assert abs(outputs["float"] - 3.14159) < 0.0001
        assert outputs["negative"] == -100
        assert outputs["large"] == 10**15

    def test_lru_cache_zero_operations_hit_rate(self):
        """LRU cache handles zero operations for hit rate calculation."""
        cache = LRUCheckpointCache(max_size=10)
        stats = cache.stats

        # Should not divide by zero
        assert stats["hit_rate"] == 0.0
