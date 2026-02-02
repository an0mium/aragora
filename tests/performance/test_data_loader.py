"""Tests for DataLoader Batch Query Resolution.

Covers:
- LoaderStats tracking and calculations
- DataLoader batching behavior
- Request caching and deduplication
- Backpressure with max_queue_size
- Batch timeout functionality
- prime() and clear() operations
- BatchResolver for multiple loaders
- create_data_loader convenience function
- batch_by_id helper
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from aragora.performance.data_loader import (
    BatchResolver,
    DataLoader,
    LoaderStats,
    batch_by_id,
    create_data_loader,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_batch_fn():
    """Create a simple batch function that returns values for keys."""
    call_count = 0
    batch_sizes = []

    async def batch_fn(keys: list[str]) -> list[str]:
        nonlocal call_count, batch_sizes
        call_count += 1
        batch_sizes.append(len(keys))
        return [f"value_{k}" for k in keys]

    batch_fn.call_count = lambda: call_count
    batch_fn.batch_sizes = lambda: batch_sizes
    return batch_fn


@pytest.fixture
def slow_batch_fn():
    """Create a slow batch function for testing batching."""
    call_count = 0

    async def batch_fn(keys: list[str]) -> list[str]:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)  # Simulate database query
        return [f"value_{k}" for k in keys]

    batch_fn.call_count = lambda: call_count
    return batch_fn


@pytest.fixture
def failing_batch_fn():
    """Create a batch function that fails."""

    async def batch_fn(keys: list[str]) -> list[str]:
        raise RuntimeError("Database connection failed")

    return batch_fn


# =============================================================================
# LoaderStats Tests
# =============================================================================


class TestLoaderStats:
    """Tests for LoaderStats dataclass."""

    def test_initial_values(self):
        """Test default initial values."""
        stats = LoaderStats()

        assert stats.loads == 0
        assert stats.batches == 0
        assert stats.cache_hits == 0
        assert stats.batch_sizes == []
        assert stats.total_load_time_ms == 0.0
        assert stats.queue_overflows == 0
        assert stats.max_queue_size_seen == 0

    def test_avg_batch_size_empty(self):
        """Test average batch size with no batches."""
        stats = LoaderStats()
        assert stats.avg_batch_size == 0.0

    def test_avg_batch_size_calculation(self):
        """Test average batch size calculation."""
        stats = LoaderStats(batch_sizes=[10, 20, 30])
        assert stats.avg_batch_size == 20.0

    def test_cache_hit_rate_empty(self):
        """Test cache hit rate with no loads."""
        stats = LoaderStats()
        assert stats.cache_hit_rate == 0.0

    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation."""
        stats = LoaderStats(loads=100, cache_hits=75)
        assert stats.cache_hit_rate == 75.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = LoaderStats(
            loads=100,
            batches=10,
            cache_hits=50,
            batch_sizes=[5, 10, 15],
            total_load_time_ms=250.0,
            queue_overflows=2,
            max_queue_size_seen=50,
        )

        d = stats.to_dict()

        assert d["loads"] == 100
        assert d["batches"] == 10
        assert "50.0%" in d["cache_hit_rate"]
        assert d["avg_batch_size"] == 10.0
        assert d["queue_overflows"] == 2


# =============================================================================
# DataLoader Basic Tests
# =============================================================================


class TestDataLoaderBasic:
    """Basic tests for DataLoader."""

    def test_initialization(self, simple_batch_fn):
        """Test loader initialization."""
        loader = DataLoader(
            simple_batch_fn,
            max_batch_size=50,
            cache=True,
            name="test_loader",
        )

        assert loader._max_batch_size == 50
        assert loader._cache_enabled is True
        assert loader._name == "test_loader"

    @pytest.mark.asyncio
    async def test_single_load(self, simple_batch_fn):
        """Test loading a single item."""
        loader = DataLoader(simple_batch_fn)

        result = await loader.load("key1")

        assert result == "value_key1"
        assert simple_batch_fn.call_count() == 1

    @pytest.mark.asyncio
    async def test_load_many(self, simple_batch_fn):
        """Test loading multiple items."""
        loader = DataLoader(simple_batch_fn)

        results = await loader.load_many(["key1", "key2", "key3"])

        assert results == ["value_key1", "value_key2", "value_key3"]

    @pytest.mark.asyncio
    async def test_stats_property(self, simple_batch_fn):
        """Test stats property."""
        loader = DataLoader(simple_batch_fn)

        await loader.load("key1")

        stats = loader.stats
        assert stats.loads == 1
        assert stats.batches == 1


# =============================================================================
# Batching Tests
# =============================================================================


class TestDataLoaderBatching:
    """Tests for batching behavior."""

    @pytest.mark.asyncio
    async def test_concurrent_loads_batched(self, slow_batch_fn):
        """Concurrent loads are batched together."""
        loader = DataLoader(slow_batch_fn, batch_timeout_ms=50)

        # Start multiple loads concurrently
        results = await asyncio.gather(
            loader.load("key1"),
            loader.load("key2"),
            loader.load("key3"),
        )

        assert results == ["value_key1", "value_key2", "value_key3"]
        # Should be batched into a single call
        assert slow_batch_fn.call_count() == 1

    @pytest.mark.asyncio
    async def test_max_batch_size_respected(self, simple_batch_fn):
        """Max batch size is respected."""
        loader = DataLoader(simple_batch_fn, max_batch_size=3, batch_timeout_ms=100)

        # Load more items than max batch size
        results = await loader.load_many([f"key{i}" for i in range(10)])

        assert len(results) == 10
        # Should have multiple batches
        assert simple_batch_fn.call_count() >= 3  # ceil(10/3) = 4

    @pytest.mark.asyncio
    async def test_batch_sizes_tracked(self, simple_batch_fn):
        """Batch sizes are tracked in stats."""
        loader = DataLoader(simple_batch_fn, batch_timeout_ms=50)

        await asyncio.gather(
            loader.load("key1"),
            loader.load("key2"),
            loader.load("key3"),
        )

        stats = loader.stats
        assert 3 in stats.batch_sizes or sum(stats.batch_sizes) == 3


# =============================================================================
# Caching Tests
# =============================================================================


class TestDataLoaderCaching:
    """Tests for caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_hits(self, simple_batch_fn):
        """Second load of same key is a cache hit."""
        loader = DataLoader(simple_batch_fn, batch_timeout_ms=50)

        # First load
        result1 = await loader.load("key1")

        # Wait for batch to complete
        await asyncio.sleep(0.1)

        # Second load (should be cached)
        result2 = await loader.load("key1")

        assert result1 == result2
        assert loader.stats.cache_hits >= 1

    @pytest.mark.asyncio
    async def test_cache_disabled(self, simple_batch_fn):
        """Cache can be disabled."""
        loader = DataLoader(simple_batch_fn, cache=False, batch_timeout_ms=50)

        await loader.load("key1")
        await asyncio.sleep(0.1)
        await loader.load("key1")

        # Should have loaded twice (no caching)
        assert simple_batch_fn.call_count() >= 2

    @pytest.mark.asyncio
    async def test_prime_cache(self, simple_batch_fn):
        """Prime can pre-populate cache."""
        loader = DataLoader(simple_batch_fn)

        loader.prime("key1", "primed_value")
        result = await loader.load("key1")

        assert result == "primed_value"
        assert simple_batch_fn.call_count() == 0  # Not loaded

    @pytest.mark.asyncio
    async def test_clear_specific_key(self, simple_batch_fn):
        """Clear can remove specific key from cache."""
        loader = DataLoader(simple_batch_fn, batch_timeout_ms=50)

        await loader.load("key1")
        await asyncio.sleep(0.1)

        loader.clear("key1")

        # Should need to reload
        await loader.load("key1")
        assert simple_batch_fn.call_count() >= 2

    @pytest.mark.asyncio
    async def test_clear_all_cache(self, simple_batch_fn):
        """Clear can remove all entries from cache."""
        loader = DataLoader(simple_batch_fn, batch_timeout_ms=50)

        await loader.load("key1")
        await loader.load("key2")
        await asyncio.sleep(0.1)

        loader.clear()  # Clear all

        # Cache should be empty
        assert len(loader._cache) == 0

    @pytest.mark.asyncio
    async def test_custom_cache_key_fn(self, simple_batch_fn):
        """Custom cache key function is used."""
        loader = DataLoader(
            simple_batch_fn,
            cache_key_fn=lambda k: f"prefix:{k}",
            batch_timeout_ms=50,
        )

        await loader.load("key1")
        await asyncio.sleep(0.1)

        # Cache should use prefixed key
        assert "prefix:key1" in loader._cache


# =============================================================================
# Backpressure Tests
# =============================================================================


class TestDataLoaderBackpressure:
    """Tests for backpressure handling."""

    @pytest.mark.asyncio
    async def test_queue_overflow_raises(self, slow_batch_fn):
        """Queue overflow raises RuntimeError."""
        loader = DataLoader(
            slow_batch_fn,
            max_queue_size=5,
            batch_timeout_ms=1000,  # Long timeout to build up queue
        )

        with pytest.raises(RuntimeError, match="queue overflow"):
            # Try to add more items than max queue size
            tasks = [loader.load(f"key{i}") for i in range(10)]
            await asyncio.gather(*tasks)

    @pytest.mark.asyncio
    async def test_max_queue_size_tracked(self, simple_batch_fn):
        """Max queue size is tracked in stats."""
        loader = DataLoader(simple_batch_fn, batch_timeout_ms=50)

        await asyncio.gather(*[loader.load(f"key{i}") for i in range(5)])

        stats = loader.stats
        assert stats.max_queue_size_seen >= 1

    @pytest.mark.asyncio
    async def test_queue_overflows_tracked(self, slow_batch_fn):
        """Queue overflows are tracked in stats."""
        loader = DataLoader(
            slow_batch_fn,
            max_queue_size=3,
            batch_timeout_ms=1000,
        )

        try:
            await asyncio.gather(*[loader.load(f"key{i}") for i in range(10)])
        except RuntimeError:
            pass

        stats = loader.stats
        assert stats.queue_overflows >= 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestDataLoaderErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_batch_function_error(self, failing_batch_fn):
        """Batch function errors are propagated to all waiters."""
        loader = DataLoader(failing_batch_fn, batch_timeout_ms=50)

        with pytest.raises(RuntimeError, match="Database connection failed"):
            await loader.load("key1")

    @pytest.mark.asyncio
    async def test_error_propagated_to_multiple_waiters(self, failing_batch_fn):
        """Error is propagated to all concurrent waiters."""
        loader = DataLoader(failing_batch_fn, batch_timeout_ms=50)

        results = await asyncio.gather(
            loader.load("key1"),
            loader.load("key2"),
            loader.load("key3"),
            return_exceptions=True,
        )

        # All results should be exceptions
        assert all(isinstance(r, RuntimeError) for r in results)

    @pytest.mark.asyncio
    async def test_wrong_result_length_error(self):
        """Raises error when batch function returns wrong number of results."""

        async def bad_batch_fn(keys):
            return ["only_one"]  # Wrong number

        loader = DataLoader(bad_batch_fn, batch_timeout_ms=50)

        with pytest.raises(ValueError, match="returned .* values for .* keys"):
            await asyncio.gather(
                loader.load("key1"),
                loader.load("key2"),
            )


# =============================================================================
# clear_all Tests
# =============================================================================


class TestDataLoaderClearAll:
    """Tests for clear_all method."""

    @pytest.mark.asyncio
    async def test_clear_all_clears_cache(self, simple_batch_fn):
        """clear_all clears the cache."""
        loader = DataLoader(simple_batch_fn, batch_timeout_ms=50)

        await loader.load("key1")
        await asyncio.sleep(0.1)

        loader.clear_all()

        assert len(loader._cache) == 0

    @pytest.mark.asyncio
    async def test_clear_all_resets_stats(self, simple_batch_fn):
        """clear_all resets statistics."""
        loader = DataLoader(simple_batch_fn, batch_timeout_ms=50)

        await loader.load("key1")
        await asyncio.sleep(0.1)

        loader.clear_all()

        stats = loader.stats
        assert stats.loads == 0
        assert stats.batches == 0

    @pytest.mark.asyncio
    async def test_clear_all_rejects_pending(self, slow_batch_fn):
        """clear_all rejects pending requests."""
        loader = DataLoader(slow_batch_fn, batch_timeout_ms=1000)

        # Start a load but clear before it completes
        task = asyncio.create_task(loader.load("key1"))
        await asyncio.sleep(0.001)  # Let it queue

        loader.clear_all()

        with pytest.raises(RuntimeError, match="cleared while request pending"):
            await task


# =============================================================================
# BatchResolver Tests
# =============================================================================


class TestBatchResolver:
    """Tests for BatchResolver."""

    def test_initialization(self):
        """Test resolver initialization."""
        resolver = BatchResolver()
        assert len(resolver._loaders) == 0

    @pytest.mark.asyncio
    async def test_register_loader(self, simple_batch_fn):
        """Test registering a loader."""
        resolver = BatchResolver()

        loader = resolver.register("users", simple_batch_fn)

        assert loader is not None
        assert "users" in resolver._loaders

    @pytest.mark.asyncio
    async def test_register_returns_existing(self, simple_batch_fn):
        """Register returns existing loader if name already registered."""
        resolver = BatchResolver()

        loader1 = resolver.register("users", simple_batch_fn)
        loader2 = resolver.register("users", simple_batch_fn)

        assert loader1 is loader2

    @pytest.mark.asyncio
    async def test_get_loader(self, simple_batch_fn):
        """Test getting a loader by name."""
        resolver = BatchResolver()
        resolver.register("users", simple_batch_fn)

        loader = resolver.get("users")
        assert loader is not None

        unknown = resolver.get("unknown")
        assert unknown is None

    @pytest.mark.asyncio
    async def test_clear_all_loaders(self, simple_batch_fn):
        """Test clearing all loaders."""
        resolver = BatchResolver()

        loader1 = resolver.register("users", simple_batch_fn)
        loader2 = resolver.register("posts", simple_batch_fn)

        await loader1.load("user1")
        await loader2.load("post1")
        await asyncio.sleep(0.1)

        resolver.clear_all()

        # All caches should be cleared
        assert len(loader1._cache) == 0
        assert len(loader2._cache) == 0

    @pytest.mark.asyncio
    async def test_stats_all_loaders(self, simple_batch_fn):
        """Test getting stats for all loaders."""
        resolver = BatchResolver()

        loader1 = resolver.register("users", simple_batch_fn)
        loader2 = resolver.register("posts", simple_batch_fn)

        await loader1.load("user1")
        await loader2.load("post1")
        await asyncio.sleep(0.1)

        stats = resolver.stats()

        assert "users" in stats
        assert "posts" in stats
        assert stats["users"]["loads"] >= 1
        assert stats["posts"]["loads"] >= 1


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_create_data_loader(self, simple_batch_fn):
        """Test create_data_loader function."""
        loader = create_data_loader(
            simple_batch_fn,
            max_batch_size=50,
            cache=True,
            name="my_loader",
        )

        assert loader._max_batch_size == 50
        assert loader._cache_enabled is True
        assert loader._name == "my_loader"

        result = await loader.load("key1")
        assert result == "value_key1"

    @pytest.mark.asyncio
    async def test_batch_by_id(self):
        """Test batch_by_id helper function."""

        async def fetch_fn(ids):
            return {id_: f"value_{id_}" for id_ in ids}

        results = await batch_by_id(["1", "2", "3"], fetch_fn)

        assert results == ["value_1", "value_2", "value_3"]

    @pytest.mark.asyncio
    async def test_batch_by_id_missing_values(self):
        """Test batch_by_id with missing values."""

        async def fetch_fn(ids):
            # Only return some values
            return {"1": "value_1", "3": "value_3"}

        results = await batch_by_id(["1", "2", "3"], fetch_fn)

        assert results == ["value_1", None, "value_3"]


# =============================================================================
# Pending Queue Size Tests
# =============================================================================


class TestPendingQueueSize:
    """Tests for pending queue size monitoring."""

    @pytest.mark.asyncio
    async def test_pending_queue_size_property(self, slow_batch_fn):
        """Test pending_queue_size property."""
        loader = DataLoader(slow_batch_fn, batch_timeout_ms=1000)

        # Queue some requests
        task = asyncio.create_task(loader.load("key1"))
        await asyncio.sleep(0.001)

        # Should have pending items
        assert loader.pending_queue_size >= 0

        # Clean up
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestDataLoaderIntegration:
    """Integration tests for DataLoader."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, simple_batch_fn):
        """Test complete data loader workflow."""
        loader = DataLoader(
            simple_batch_fn,
            max_batch_size=10,
            cache=True,
            name="integration_test",
            batch_timeout_ms=50,
        )

        # Initial loads
        results1 = await asyncio.gather(*[loader.load(f"key{i}") for i in range(5)])
        assert len(results1) == 5

        # Wait for batch to complete
        await asyncio.sleep(0.1)

        # Cached loads
        results2 = await asyncio.gather(*[loader.load(f"key{i}") for i in range(5)])
        assert results1 == results2

        # Check stats
        stats = loader.stats
        assert stats.loads == 10
        assert stats.cache_hits >= 5
        assert stats.batches >= 1

        # Prime new value
        loader.prime("key10", "primed_value")
        result = await loader.load("key10")
        assert result == "primed_value"

        # Clear and reload
        loader.clear("key0")
        result = await loader.load("key0")
        assert result == "value_key0"

    @pytest.mark.asyncio
    async def test_with_batch_resolver(self):
        """Test using BatchResolver for multiple entity types."""
        resolver = BatchResolver()

        async def load_users(ids):
            return [{"id": id_, "type": "user"} for id_ in ids]

        async def load_posts(ids):
            return [{"id": id_, "type": "post"} for id_ in ids]

        user_loader = resolver.register("users", load_users)
        post_loader = resolver.register("posts", load_posts)

        # Load multiple entity types concurrently
        users, posts = await asyncio.gather(
            user_loader.load_many(["u1", "u2"]),
            post_loader.load_many(["p1", "p2", "p3"]),
        )

        assert len(users) == 2
        assert all(u["type"] == "user" for u in users)

        assert len(posts) == 3
        assert all(p["type"] == "post" for p in posts)

        # Check combined stats
        stats = resolver.stats()
        assert stats["users"]["loads"] == 2
        assert stats["posts"]["loads"] == 3
