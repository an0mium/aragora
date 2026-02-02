"""Tests for SupermemoryBackend."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.backends.supermemory import SupermemoryBackend, CacheEntry
from aragora.memory.protocols import MemoryEntry, MemoryQueryResult


@pytest.fixture
def mock_config():
    """Create a mock config."""
    config = MagicMock()
    config.sync_threshold = 0.7
    config.cache_ttl_seconds = 300.0
    config.get_container_tag.return_value = "aragora_patterns"
    return config


@pytest.fixture
def backend(mock_config):
    """Create a backend with mock config."""
    return SupermemoryBackend(
        config=mock_config,
        enable_external_sync=False,  # Disable external for unit tests
    )


@pytest.fixture
def sample_entry():
    """Create a fresh sample memory entry for each test."""

    def _create_entry():
        return MemoryEntry(
            id="test_1",
            content="Test memory content",
            tier="medium",
            weight=0.8,
            created_at=datetime.utcnow(),
            metadata={"key": "value"},
        )

    return _create_entry()


class TestSupermemoryBackendStore:
    """Test store operations."""

    @pytest.mark.asyncio
    async def test_store_entry(self, backend, sample_entry):
        """Test storing an entry."""
        entry_id = await backend.store(sample_entry)

        assert entry_id == "test_1"
        assert "test_1" in backend._local_entries
        assert "test_1" in backend._by_tier["medium"]

    @pytest.mark.asyncio
    async def test_store_generates_id(self, backend):
        """Test ID generation for entries without ID."""
        entry = MemoryEntry(id="", content="No ID", tier="fast", weight=0.5)
        entry_id = await backend.store(entry)

        assert entry_id.startswith("sm_")
        assert entry_id in backend._local_entries

    @pytest.mark.asyncio
    async def test_store_adds_to_cache(self, backend, sample_entry):
        """Test entry is cached after store."""
        await backend.store(sample_entry)

        assert "test_1" in backend._cache
        cached = backend._cache["test_1"]
        assert cached.entry.content == "Test memory content"


class TestSupermemoryBackendGet:
    """Test get operations."""

    @pytest.mark.asyncio
    async def test_get_existing(self, backend, sample_entry):
        """Test getting an existing entry."""
        await backend.store(sample_entry)
        retrieved = await backend.get("test_1")

        assert retrieved is not None
        assert retrieved.id == "test_1"
        assert retrieved.content == "Test memory content"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, backend):
        """Test getting a non-existent entry."""
        retrieved = await backend.get("nonexistent")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_from_cache(self, backend, sample_entry):
        """Test getting entry from cache."""
        await backend.store(sample_entry)

        # Clear local but keep cache
        backend._local_entries.clear()

        # Should still get from cache
        retrieved = await backend.get("test_1")
        assert retrieved is not None


class TestSupermemoryBackendUpdate:
    """Test update operations."""

    @pytest.mark.asyncio
    async def test_update_existing(self, backend, sample_entry):
        """Test updating an existing entry."""
        await backend.store(sample_entry)

        sample_entry.content = "Updated content"
        result = await backend.update(sample_entry)

        assert result is True
        updated = await backend.get("test_1")
        assert updated.content == "Updated content"

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, backend, sample_entry):
        """Test updating a non-existent entry."""
        result = await backend.update(sample_entry)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_changes_tier(self, backend, sample_entry):
        """Test updating entry tier."""
        await backend.store(sample_entry)
        assert "test_1" in backend._by_tier["medium"]

        sample_entry.tier = "slow"
        await backend.update(sample_entry)

        assert "test_1" not in backend._by_tier["medium"]
        assert "test_1" in backend._by_tier["slow"]


class TestSupermemoryBackendDelete:
    """Test delete operations."""

    @pytest.mark.asyncio
    async def test_delete_existing(self, backend, sample_entry):
        """Test deleting an existing entry."""
        await backend.store(sample_entry)
        result = await backend.delete("test_1")

        assert result is True
        assert "test_1" not in backend._local_entries
        assert "test_1" not in backend._cache

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, backend):
        """Test deleting a non-existent entry."""
        result = await backend.delete("nonexistent")
        assert result is False


class TestSupermemoryBackendQuery:
    """Test query operations."""

    @pytest.mark.asyncio
    async def test_query_all(self, backend):
        """Test querying all entries."""
        entries = [
            MemoryEntry(id=f"e{i}", content=f"Content {i}", tier="medium", weight=0.5 + i * 0.1)
            for i in range(5)
        ]
        for e in entries:
            await backend.store(e)

        result = await backend.query()

        assert isinstance(result, MemoryQueryResult)
        assert result.total_count == 5
        assert len(result.entries) == 5

    @pytest.mark.asyncio
    async def test_query_by_tier(self, backend):
        """Test querying by tier."""
        await backend.store(MemoryEntry(id="e1", content="Fast", tier="fast", weight=0.5))
        await backend.store(MemoryEntry(id="e2", content="Medium", tier="medium", weight=0.5))
        await backend.store(MemoryEntry(id="e3", content="Medium2", tier="medium", weight=0.6))

        result = await backend.query(tier="medium")

        assert result.total_count == 2
        assert all(e.tier == "medium" for e in result.entries)

    @pytest.mark.asyncio
    async def test_query_by_min_weight(self, backend):
        """Test querying by minimum weight."""
        await backend.store(MemoryEntry(id="e1", content="Low", tier="fast", weight=0.3))
        await backend.store(MemoryEntry(id="e2", content="High", tier="fast", weight=0.8))
        await backend.store(MemoryEntry(id="e3", content="Higher", tier="fast", weight=0.9))

        result = await backend.query(min_weight=0.7)

        assert result.total_count == 2
        assert all(e.weight >= 0.7 for e in result.entries)

    @pytest.mark.asyncio
    async def test_query_pagination(self, backend):
        """Test query pagination."""
        for i in range(10):
            await backend.store(
                MemoryEntry(id=f"e{i}", content=f"Content {i}", tier="fast", weight=0.5)
            )

        result = await backend.query(limit=3, offset=0)
        assert len(result.entries) == 3
        assert result.has_more is True

        result2 = await backend.query(limit=3, offset=9)
        assert len(result2.entries) == 1
        assert result2.has_more is False


class TestSupermemoryBackendBatch:
    """Test batch operations."""

    @pytest.mark.asyncio
    async def test_store_batch(self, backend):
        """Test batch store."""
        entries = [
            MemoryEntry(id=f"batch{i}", content=f"Content {i}", tier="fast", weight=0.5)
            for i in range(5)
        ]

        ids = await backend.store_batch(entries)

        assert len(ids) == 5
        assert all(id.startswith("batch") for id in ids)

    @pytest.mark.asyncio
    async def test_delete_batch(self, backend):
        """Test batch delete."""
        for i in range(5):
            await backend.store(
                MemoryEntry(id=f"del{i}", content=f"Content {i}", tier="fast", weight=0.5)
            )

        count = await backend.delete_batch(["del0", "del2", "del4", "nonexistent"])

        assert count == 3
        assert "del1" in backend._local_entries
        assert "del0" not in backend._local_entries


class TestSupermemoryBackendTier:
    """Test tier operations."""

    @pytest.mark.asyncio
    async def test_promote(self, backend, sample_entry):
        """Test tier promotion."""
        await backend.store(sample_entry)
        result = await backend.promote("test_1", "slow")

        assert result is True
        entry = await backend.get("test_1")
        assert entry.tier == "slow"
        assert "test_1" not in backend._by_tier["medium"]
        assert "test_1" in backend._by_tier["slow"]

    @pytest.mark.asyncio
    async def test_promote_same_tier(self, backend, sample_entry):
        """Test promoting to same tier."""
        await backend.store(sample_entry)
        result = await backend.promote("test_1", "medium")

        assert result is True

    @pytest.mark.asyncio
    async def test_count_by_tier(self, backend):
        """Test tier counts."""
        await backend.store(MemoryEntry(id="e1", content="Fast", tier="fast", weight=0.5))
        await backend.store(MemoryEntry(id="e2", content="Medium", tier="medium", weight=0.5))
        await backend.store(MemoryEntry(id="e3", content="Medium2", tier="medium", weight=0.5))

        counts = await backend.count_by_tier()

        assert counts["fast"] == 1
        assert counts["medium"] == 2
        assert counts["slow"] == 0


class TestSupermemoryBackendSimilarity:
    """Test similarity search."""

    @pytest.mark.asyncio
    async def test_search_similar(self, backend):
        """Test similarity search."""
        entries = [
            MemoryEntry(id="e1", content="A", tier="fast", weight=0.5, embedding=[1.0, 0.0, 0.0]),
            MemoryEntry(id="e2", content="B", tier="fast", weight=0.5, embedding=[0.0, 1.0, 0.0]),
            MemoryEntry(id="e3", content="C", tier="fast", weight=0.5, embedding=[0.9, 0.1, 0.0]),
        ]
        for e in entries:
            await backend.store(e)

        results = await backend.search_similar(
            query_embedding=[1.0, 0.0, 0.0],
            limit=10,
            min_similarity=0.5,
        )

        assert len(results) >= 1
        # First result should be most similar
        assert results[0][0].id in ["e1", "e3"]

    @pytest.mark.asyncio
    async def test_search_similar_with_tier_filter(self, backend):
        """Test similarity search with tier filter."""
        await backend.store(
            MemoryEntry(id="e1", content="A", tier="fast", weight=0.5, embedding=[1.0, 0.0])
        )
        await backend.store(
            MemoryEntry(id="e2", content="B", tier="slow", weight=0.5, embedding=[1.0, 0.0])
        )

        results = await backend.search_similar(
            query_embedding=[1.0, 0.0],
            limit=10,
            tier="fast",
        )

        assert len(results) == 1
        assert results[0][0].id == "e1"


class TestSupermemoryBackendMaintenance:
    """Test maintenance operations."""

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, backend):
        """Test cleaning up expired entries."""
        now = datetime.utcnow()
        expired = MemoryEntry(
            id="expired",
            content="Old",
            tier="fast",
            weight=0.5,
            expires_at=now - timedelta(hours=1),
        )
        valid = MemoryEntry(
            id="valid",
            content="New",
            tier="fast",
            weight=0.5,
            expires_at=now + timedelta(hours=1),
        )
        await backend.store(expired)
        await backend.store(valid)

        count = await backend.cleanup_expired()

        assert count == 1
        assert "expired" not in backend._local_entries
        assert "valid" in backend._local_entries

    @pytest.mark.asyncio
    async def test_vacuum(self, backend, sample_entry):
        """Test vacuum operation."""
        await backend.store(sample_entry)
        await backend.vacuum()  # Should not raise

    @pytest.mark.asyncio
    async def test_health_check(self, backend, sample_entry):
        """Test health check."""
        await backend.store(sample_entry)
        health = await backend.health_check()

        assert health.healthy is True
        assert "local_entries" in health.details
        assert health.details["local_entries"] == 1

    @pytest.mark.asyncio
    async def test_get_stats(self, backend):
        """Test getting statistics."""
        await backend.store(MemoryEntry(id="e1", content="A", tier="fast", weight=0.5))
        await backend.store(MemoryEntry(id="e2", content="B", tier="medium", weight=0.8))

        stats = await backend.get_stats()

        assert stats["total_entries"] == 2
        assert stats["tier_counts"]["fast"] == 1
        assert stats["tier_counts"]["medium"] == 1


class TestCacheEntry:
    """Test CacheEntry class."""

    def test_is_expired_fresh(self):
        """Test fresh entry is not expired."""
        import time

        cache_entry = CacheEntry(
            entry=MagicMock(),
            cached_at=time.time(),
            ttl_seconds=300.0,
        )

        assert cache_entry.is_expired() is False

    def test_is_expired_old(self):
        """Test old entry is expired."""
        import time

        cache_entry = CacheEntry(
            entry=MagicMock(),
            cached_at=time.time() - 400,  # 400 seconds ago
            ttl_seconds=300.0,
        )

        assert cache_entry.is_expired() is True


class TestSupermemoryBackendCacheEviction:
    """Test cache eviction."""

    @pytest.mark.asyncio
    async def test_cache_eviction_at_capacity(self, backend):
        """Test oldest entries are evicted when cache is full."""
        backend.cache_max_size = 3

        for i in range(5):
            await backend.store(MemoryEntry(id=f"e{i}", content=f"C{i}", tier="fast", weight=0.5))

        # Should only have 3 entries in cache (the last 3)
        assert len(backend._cache) == 3
        assert "e0" not in backend._cache
        assert "e1" not in backend._cache
        assert "e4" in backend._cache
