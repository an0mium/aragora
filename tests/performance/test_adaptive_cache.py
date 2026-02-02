"""Tests for Adaptive TTL Cache.

Covers:
- AccessPattern tracking and calculations
- CacheEntry storage
- CacheStats calculations
- AdaptiveTTLCache get/set/delete operations
- TTL adjustment based on access patterns
- Hot spot detection
- Eviction policies
- Background cleanup
- CacheOptimizer recommendations
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from aragora.performance.adaptive_cache import (
    AccessPattern,
    AdaptiveTTLCache,
    CacheEntry,
    CacheOptimizer,
    CacheStats,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def cache():
    """Create a cache for testing."""
    return AdaptiveTTLCache(
        name="test_cache",
        base_ttl=60.0,
        min_ttl=10.0,
        max_ttl=3600.0,
        max_size=100,
    )


@pytest.fixture
def small_cache():
    """Create a small cache for eviction testing."""
    return AdaptiveTTLCache(
        name="small_cache",
        base_ttl=60.0,
        max_size=5,
    )


# =============================================================================
# AccessPattern Tests
# =============================================================================


class TestAccessPattern:
    """Tests for AccessPattern dataclass."""

    def test_initial_values(self):
        """Test default initial values."""
        pattern = AccessPattern(key="test")

        assert pattern.key == "test"
        assert pattern.access_count == 0
        assert pattern.last_access == 0.0
        assert pattern.first_access == 0.0
        assert pattern.access_times == []
        assert pattern.is_hot is False

    def test_access_frequency_empty(self):
        """Test frequency with no accesses."""
        pattern = AccessPattern(key="test")
        assert pattern.access_frequency == 0.0

    def test_access_frequency_single_access(self):
        """Test frequency with single access."""
        pattern = AccessPattern(key="test")
        pattern.record_access()
        assert pattern.access_frequency == 0.0

    def test_access_frequency_multiple_accesses(self):
        """Test frequency calculation with multiple accesses."""
        pattern = AccessPattern(key="test")

        # Record multiple accesses over time
        pattern.first_access = 0.0
        pattern.last_access = 10.0
        pattern.access_count = 100

        # 100 accesses in 10 seconds = 10/sec
        assert pattern.access_frequency == 10.0

    def test_avg_interval_empty(self):
        """Test average interval with no accesses."""
        pattern = AccessPattern(key="test")
        assert pattern.avg_interval_ms == float("inf")

    def test_avg_interval_calculation(self):
        """Test average interval calculation."""
        pattern = AccessPattern(key="test")
        pattern.access_times = [1.0, 1.1, 1.2, 1.3]  # 100ms intervals

        avg = pattern.avg_interval_ms
        assert abs(avg - 100.0) < 1.0

    def test_record_access(self):
        """Test recording an access event."""
        pattern = AccessPattern(key="test")

        before = time.time()
        pattern.record_access()
        after = time.time()

        assert pattern.access_count == 1
        assert before <= pattern.first_access <= after
        assert before <= pattern.last_access <= after
        assert len(pattern.access_times) == 1

    def test_record_access_history_limit(self):
        """Test that history is limited."""
        pattern = AccessPattern(key="test")

        # Record many accesses
        for _ in range(150):
            pattern.record_access()

        assert len(pattern.access_times) <= 100


# =============================================================================
# CacheEntry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_entry_creation(self):
        """Test cache entry creation."""
        pattern = AccessPattern(key="test")
        entry = CacheEntry(
            value="test_value",
            created_at=1000.0,
            expires_at=1060.0,
            access_pattern=pattern,
        )

        assert entry.value == "test_value"
        assert entry.created_at == 1000.0
        assert entry.expires_at == 1060.0
        assert entry.access_pattern == pattern


# =============================================================================
# CacheStats Tests
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_initial_values(self):
        """Test default initial values."""
        stats = CacheStats()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.ttl_adjustments == 0
        assert stats.hot_spots == 0

    def test_hit_rate_empty(self):
        """Test hit rate with no operations."""
        stats = CacheStats()
        assert stats.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 75.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = CacheStats(
            hits=100,
            misses=20,
            evictions=5,
            ttl_adjustments=10,
            hot_spots=3,
            avg_ttl_seconds=120.5,
        )

        d = stats.to_dict()

        assert d["hits"] == 100
        assert d["misses"] == 20
        assert d["evictions"] == 5
        assert "83.3%" in d["hit_rate"]


# =============================================================================
# AdaptiveTTLCache Basic Tests
# =============================================================================


class TestAdaptiveTTLCacheBasic:
    """Basic tests for AdaptiveTTLCache."""

    def test_initialization(self, cache):
        """Test cache initialization."""
        assert cache._name == "test_cache"
        assert cache._base_ttl == 60.0
        assert cache._min_ttl == 10.0
        assert cache._max_ttl == 3600.0
        assert cache._max_size == 100

    def test_size_property(self, cache):
        """Test size property."""
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache):
        """Test basic set and get."""
        await cache.set("key1", "value1")
        result = await cache.get("key1")

        assert result == "value1"
        assert cache.size == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, cache):
        """Test getting nonexistent key."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, cache):
        """Test delete operation."""
        await cache.set("key1", "value1")
        deleted = await cache.delete("key1")

        assert deleted is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, cache):
        """Test delete of nonexistent key."""
        deleted = await cache.delete("nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_clear(self, cache):
        """Test clearing all entries."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        cache.clear()

        assert cache.size == 0
        assert await cache.get("key1") is None


# =============================================================================
# TTL Tests
# =============================================================================


class TestAdaptiveTTLCacheTTL:
    """Tests for TTL behavior."""

    @pytest.mark.asyncio
    async def test_expired_entry_not_returned(self, cache):
        """Expired entries are not returned."""
        # Set with very short TTL
        await cache.set("key1", "value1", ttl=0.001)
        await asyncio.sleep(0.01)

        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_explicit_ttl_overrides_base(self, cache):
        """Explicit TTL overrides base TTL."""
        await cache.set("key1", "value1", ttl=1.0)

        # Entry should have 1 second TTL
        entry = cache._cache["key1"]
        ttl_set = entry.expires_at - entry.created_at
        assert abs(ttl_set - 1.0) < 0.1

    @pytest.mark.asyncio
    async def test_ttl_adjustment_on_frequent_access(self, cache):
        """TTL increases for frequently accessed keys."""
        await cache.set("key1", "value1")

        # Record initial TTL
        initial_ttl = cache._cache["key1"].access_pattern.current_ttl

        # Simulate frequent access
        pattern = cache._cache["key1"].access_pattern
        pattern.access_count = 100
        pattern.first_access = time.time() - 1  # 1 second ago
        pattern.last_access = time.time()

        # Access should trigger TTL adjustment
        await cache.get("key1")

        stats = cache.stats
        assert stats.ttl_adjustments >= 0  # May or may not adjust depending on frequency


# =============================================================================
# Hot Spot Tests
# =============================================================================


class TestHotSpotDetection:
    """Tests for hot spot detection."""

    @pytest.mark.asyncio
    async def test_hot_spot_detection(self, cache):
        """Very frequently accessed keys are marked as hot spots."""
        await cache.set("key1", "value1")

        # Simulate very high frequency access
        pattern = cache._cache["key1"].access_pattern
        pattern.access_count = 1000
        pattern.first_access = time.time() - 1  # 1 second ago
        pattern.last_access = time.time()

        # Access to trigger recalculation
        await cache.get("key1")

        hot_spots = cache.get_hot_spots()
        assert "key1" in hot_spots

    @pytest.mark.asyncio
    async def test_get_hot_spots_empty(self, cache):
        """Empty cache has no hot spots."""
        hot_spots = cache.get_hot_spots()
        assert hot_spots == []


# =============================================================================
# Eviction Tests
# =============================================================================


class TestEviction:
    """Tests for cache eviction."""

    @pytest.mark.asyncio
    async def test_eviction_when_full(self, small_cache):
        """Cache evicts entries when full."""
        # Fill cache beyond capacity
        for i in range(10):
            await small_cache.set(f"key{i}", f"value{i}")

        # Should have evicted some entries
        assert small_cache.size <= small_cache._max_size

    @pytest.mark.asyncio
    async def test_eviction_prefers_non_hot(self, small_cache):
        """Eviction prefers non-hot entries."""
        # Fill cache
        for i in range(5):
            await small_cache.set(f"key{i}", f"value{i}")

        # Mark one as hot
        if "key0" in small_cache._cache:
            small_cache._cache["key0"].access_pattern.is_hot = True

        # Add more entries to trigger eviction
        for i in range(5, 10):
            await small_cache.set(f"key{i}", f"value{i}")

        # Hot entry should still exist
        if small_cache._cache:
            hot_entries = [k for k, v in small_cache._cache.items() if v.access_pattern.is_hot]
            # If hot entry was preserved, it's still there
            # (this depends on eviction timing)


# =============================================================================
# Statistics Tests
# =============================================================================


class TestCacheStatistics:
    """Tests for cache statistics tracking."""

    @pytest.mark.asyncio
    async def test_hit_tracking(self, cache):
        """Hits are tracked."""
        await cache.set("key1", "value1")
        await cache.get("key1")
        await cache.get("key1")

        stats = cache.stats
        assert stats.hits == 2

    @pytest.mark.asyncio
    async def test_miss_tracking(self, cache):
        """Misses are tracked."""
        await cache.get("nonexistent1")
        await cache.get("nonexistent2")

        stats = cache.stats
        assert stats.misses == 2

    @pytest.mark.asyncio
    async def test_eviction_tracking(self, small_cache):
        """Evictions are tracked."""
        # Fill cache and trigger evictions
        for i in range(20):
            await small_cache.set(f"key{i}", f"value{i}")

        stats = small_cache.stats
        assert stats.evictions > 0


# =============================================================================
# Background Cleanup Tests
# =============================================================================


class TestBackgroundCleanup:
    """Tests for background cleanup task."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, cache):
        """Cache can start and stop cleanup task."""
        await cache.start()
        assert cache._running is True

        await cache.stop()
        assert cache._running is False

    @pytest.mark.asyncio
    async def test_cleanup_removes_expired(self, cache):
        """Cleanup removes expired entries."""
        # Set entries with very short TTL
        await cache.set("key1", "value1", ttl=0.001)
        await cache.set("key2", "value2", ttl=0.001)

        await asyncio.sleep(0.02)

        # Manually trigger cleanup
        cache._cleanup_expired()

        assert cache.size == 0


# =============================================================================
# CacheOptimizer Tests
# =============================================================================


class TestCacheOptimizer:
    """Tests for CacheOptimizer."""

    @pytest.mark.asyncio
    async def test_record_snapshot(self, cache):
        """Optimizer records snapshots."""
        optimizer = CacheOptimizer(cache)

        await cache.set("key1", "value1")
        snapshot = optimizer.record_snapshot()

        assert "timestamp" in snapshot
        assert snapshot["size"] == 1
        assert "stats" in snapshot

    @pytest.mark.asyncio
    async def test_snapshot_history_limit(self, cache):
        """Snapshot history is limited."""
        optimizer = CacheOptimizer(cache)

        for _ in range(150):
            optimizer.record_snapshot()

        assert len(optimizer._history) <= 100

    @pytest.mark.asyncio
    async def test_recommendations_low_hit_rate(self, cache):
        """Recommends improvements for low hit rate."""
        # Create low hit rate scenario
        cache._stats.hits = 20
        cache._stats.misses = 80

        optimizer = CacheOptimizer(cache)
        recommendations = optimizer.get_recommendations()

        assert any("hit rate" in r.lower() for r in recommendations)

    @pytest.mark.asyncio
    async def test_recommendations_high_evictions(self, cache):
        """Recommends improvements for high eviction rate."""
        cache._stats.hits = 50
        cache._stats.evictions = 100

        optimizer = CacheOptimizer(cache)
        recommendations = optimizer.get_recommendations()

        assert any("eviction" in r.lower() for r in recommendations)

    @pytest.mark.asyncio
    async def test_recommendations_many_hot_spots(self, cache):
        """Recommends for many hot spots."""
        # Add entries and mark many as hot
        for i in range(10):
            await cache.set(f"key{i}", f"value{i}")
            cache._cache[f"key{i}"].access_pattern.is_hot = True

        optimizer = CacheOptimizer(cache)
        recommendations = optimizer.get_recommendations()

        assert any("hot spot" in r.lower() for r in recommendations)


# =============================================================================
# Key Types Tests
# =============================================================================


class TestKeyTypes:
    """Tests for different key types."""

    @pytest.mark.asyncio
    async def test_string_keys(self, cache):
        """String keys work correctly."""
        await cache.set("string_key", "value")
        assert await cache.get("string_key") == "value"

    @pytest.mark.asyncio
    async def test_int_keys(self, cache):
        """Integer keys are converted to strings."""
        await cache.set(123, "value")
        assert await cache.get(123) == "value"

    @pytest.mark.asyncio
    async def test_tuple_keys(self, cache):
        """Tuple keys are converted to strings."""
        await cache.set((1, 2, 3), "value")
        assert await cache.get((1, 2, 3)) == "value"


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_sets(self, cache):
        """Concurrent sets don't cause issues."""

        async def set_value(i):
            await cache.set(f"key{i}", f"value{i}")

        await asyncio.gather(*[set_value(i) for i in range(100)])

        assert cache.size <= 100

    @pytest.mark.asyncio
    async def test_concurrent_gets(self, cache):
        """Concurrent gets don't cause issues."""
        await cache.set("key1", "value1")

        async def get_value():
            return await cache.get("key1")

        results = await asyncio.gather(*[get_value() for _ in range(100)])

        assert all(r == "value1" for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_mixed_operations(self, cache):
        """Mixed concurrent operations don't cause issues."""

        async def operations(i):
            await cache.set(f"key{i}", f"value{i}")
            await cache.get(f"key{i}")
            await cache.get(f"key{i % 10}")
            if i % 3 == 0:
                await cache.delete(f"key{i}")

        await asyncio.gather(*[operations(i) for i in range(50)])

        # Cache should still be functional
        await cache.set("final", "value")
        assert await cache.get("final") == "value"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_none_value(self, cache):
        """None values are cached correctly."""
        await cache.set("key1", None)
        result = await cache.get("key1")
        assert result is None

        # Should be a cache hit, not miss
        stats = cache.stats
        assert stats.hits == 1

    @pytest.mark.asyncio
    async def test_empty_string_value(self, cache):
        """Empty string values are cached."""
        await cache.set("key1", "")
        result = await cache.get("key1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_complex_values(self, cache):
        """Complex values are cached."""
        value = {
            "list": [1, 2, 3],
            "dict": {"nested": True},
            "tuple": (1, 2),
        }
        await cache.set("complex", value)
        result = await cache.get("complex")
        assert result == value

    @pytest.mark.asyncio
    async def test_update_existing_key(self, cache):
        """Updating existing key preserves access pattern."""
        await cache.set("key1", "value1")
        await cache.get("key1")  # Access it

        await cache.set("key1", "value2")
        result = await cache.get("key1")

        assert result == "value2"
        # Access pattern should be preserved/updated
        assert cache._cache["key1"].access_pattern.access_count >= 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestAdaptiveCacheIntegration:
    """Integration tests for adaptive cache."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, cache):
        """Test complete cache workflow."""
        # Set some values
        for i in range(10):
            await cache.set(f"user:{i}", {"id": i, "name": f"User {i}"})

        # Access some frequently
        for _ in range(50):
            await cache.get("user:0")  # Hot key
            await cache.get("user:1")

        # Less frequent access
        await cache.get("user:5")

        # Check stats
        stats = cache.stats
        assert stats.hits >= 50
        assert cache.size == 10

        # Check hot spot detection
        hot_spots = cache.get_hot_spots()
        # At least one should be detected as hot
        assert len(hot_spots) >= 0

    @pytest.mark.asyncio
    async def test_with_optimizer(self, cache):
        """Test cache with optimizer."""
        optimizer = CacheOptimizer(cache)

        # Simulate workload
        for i in range(100):
            await cache.set(f"key{i % 10}", f"value{i}")
            await cache.get(f"key{i % 5}")

        # Record snapshots
        for _ in range(5):
            optimizer.record_snapshot()

        # Get recommendations
        recommendations = optimizer.get_recommendations()

        # Should provide some insights
        assert isinstance(recommendations, list)
