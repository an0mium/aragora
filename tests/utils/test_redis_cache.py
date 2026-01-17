"""Comprehensive tests for the redis_cache utility.

Tests cover:
- RedisTTLCache initialization and configuration
- Redis cache get/set operations
- TTL handling and expiration
- Connection handling (Redis availability detection)
- Error recovery and fallback behavior
- Cache invalidation (single key and clear operations)
- Statistics tracking
- Thread safety with concurrent operations
- HybridTTLCache factory function
"""

import json
import threading
import time
from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.utils.redis_cache import (
    HybridTTLCache,
    RedisTTLCache,
)


# ============================================================================
# RedisTTLCache Initialization Tests
# ============================================================================


class TestRedisTTLCacheInit:
    """Tests for RedisTTLCache initialization."""

    def test_default_initialization(self):
        """Cache should initialize with default values."""
        cache = RedisTTLCache()
        assert cache._prefix == "cache"
        assert cache._ttl_seconds == 300.0
        assert cache._maxsize == 1000

    def test_custom_prefix(self):
        """Cache should use custom prefix."""
        cache = RedisTTLCache(prefix="leaderboard")
        assert cache._prefix == "leaderboard"

    def test_custom_ttl(self):
        """Cache should use custom TTL."""
        cache = RedisTTLCache(ttl_seconds=60.0)
        assert cache._ttl_seconds == 60.0

    def test_custom_maxsize(self):
        """Cache should use custom maxsize."""
        cache = RedisTTLCache(maxsize=500)
        assert cache._maxsize == 500

    def test_all_custom_params(self):
        """Cache should accept all custom parameters."""
        cache = RedisTTLCache(
            prefix="test",
            ttl_seconds=120.0,
            maxsize=2000,
        )
        assert cache._prefix == "test"
        assert cache._ttl_seconds == 120.0
        assert cache._maxsize == 2000

    def test_initial_memory_cache_empty(self):
        """Memory cache should start empty."""
        cache = RedisTTLCache()
        assert len(cache._memory_cache) == 0

    def test_initial_statistics_zero(self):
        """Statistics should start at zero."""
        cache = RedisTTLCache()
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._redis_hits == 0
        assert cache._redis_misses == 0

    def test_redis_not_checked_initially(self):
        """Redis should not be checked on initialization (lazy)."""
        cache = RedisTTLCache()
        assert cache._redis_checked is False
        assert cache._redis is None


# ============================================================================
# Redis Key Generation Tests
# ============================================================================


class TestRedisKeyGeneration:
    """Tests for Redis key generation."""

    def test_redis_key_format(self):
        """Redis key should follow expected format."""
        cache = RedisTTLCache(prefix="test")
        key = cache._redis_key("mykey")
        assert key == "aragora:test:mykey"

    def test_redis_key_with_special_chars(self):
        """Redis key should handle special characters in key."""
        cache = RedisTTLCache(prefix="cache")
        key = cache._redis_key("user:123:profile")
        assert key == "aragora:cache:user:123:profile"

    def test_redis_key_with_empty_string(self):
        """Redis key should handle empty string."""
        cache = RedisTTLCache(prefix="test")
        key = cache._redis_key("")
        assert key == "aragora:test:"


# ============================================================================
# In-Memory Cache Tests (No Redis)
# ============================================================================


class TestInMemoryCacheOperations:
    """Tests for in-memory cache operations when Redis is unavailable."""

    @pytest.fixture
    def cache_no_redis(self):
        """Create cache with Redis unavailable."""
        cache = RedisTTLCache(prefix="test", ttl_seconds=10.0, maxsize=5)
        # Mark Redis as checked and unavailable
        cache._redis_checked = True
        cache._redis = None
        return cache

    def test_set_and_get(self, cache_no_redis):
        """Set and get should work with in-memory cache."""
        cache_no_redis.set("key1", {"value": 1})
        result = cache_no_redis.get("key1")
        assert result == {"value": 1}

    def test_get_nonexistent_key(self, cache_no_redis):
        """Get on nonexistent key should return None."""
        result = cache_no_redis.get("nonexistent")
        assert result is None

    def test_get_updates_miss_count(self, cache_no_redis):
        """Get miss should update statistics."""
        cache_no_redis.get("nonexistent")
        assert cache_no_redis._misses == 1

    def test_get_updates_hit_count(self, cache_no_redis):
        """Get hit should update statistics."""
        cache_no_redis.set("key1", "value1")
        cache_no_redis.get("key1")
        assert cache_no_redis._hits == 1

    def test_set_overwrites_existing(self, cache_no_redis):
        """Set should overwrite existing value."""
        cache_no_redis.set("key1", "old")
        cache_no_redis.set("key1", "new")
        assert cache_no_redis.get("key1") == "new"

    def test_ttl_expiration(self, cache_no_redis):
        """Expired entries should return None."""
        cache_no_redis._ttl_seconds = 0.1  # 100ms TTL
        cache_no_redis.set("key1", "value1")
        time.sleep(0.15)  # Wait for expiration
        result = cache_no_redis.get("key1")
        assert result is None

    def test_ttl_not_expired(self, cache_no_redis):
        """Non-expired entries should return value."""
        cache_no_redis._ttl_seconds = 10.0
        cache_no_redis.set("key1", "value1")
        time.sleep(0.01)
        result = cache_no_redis.get("key1")
        assert result == "value1"

    def test_expired_entry_removed(self, cache_no_redis):
        """Expired entries should be removed from cache."""
        cache_no_redis._ttl_seconds = 0.05
        cache_no_redis.set("key1", "value1")
        time.sleep(0.1)
        cache_no_redis.get("key1")  # This triggers removal
        assert "key1" not in cache_no_redis._memory_cache

    def test_maxsize_eviction(self, cache_no_redis):
        """Cache should evict oldest entries when full."""
        # maxsize is 5
        for i in range(7):
            cache_no_redis.set(f"key{i}", f"value{i}")
        # First 2 should be evicted
        assert len(cache_no_redis._memory_cache) == 5
        assert "key0" not in cache_no_redis._memory_cache
        assert "key1" not in cache_no_redis._memory_cache
        assert "key6" in cache_no_redis._memory_cache

    def test_lru_behavior_on_get(self, cache_no_redis):
        """Get should move key to end (LRU behavior)."""
        cache_no_redis.set("key1", "value1")
        cache_no_redis.set("key2", "value2")
        cache_no_redis.set("key3", "value3")
        # Access key1 to move it to end
        cache_no_redis.get("key1")
        keys = list(cache_no_redis._memory_cache.keys())
        assert keys[-1] == "key1"

    def test_complex_value_types(self, cache_no_redis):
        """Cache should handle complex value types."""
        test_values = [
            {"nested": {"dict": True}},
            [1, 2, 3, "list"],
            "string",
            123,
            45.67,
            None,
            True,
        ]
        for i, val in enumerate(test_values):
            cache_no_redis.set(f"key{i}", val)
            assert cache_no_redis.get(f"key{i}") == val


# ============================================================================
# Redis Cache Tests (Mocked Redis)
# ============================================================================


class TestRedisCacheOperations:
    """Tests for cache operations with mocked Redis."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        mock = MagicMock()
        mock.get.return_value = None
        mock.setex.return_value = True
        mock.delete.return_value = 0
        mock.keys.return_value = []
        return mock

    @pytest.fixture
    def cache_with_redis(self, mock_redis_client):
        """Create cache with mocked Redis available."""
        cache = RedisTTLCache(prefix="test", ttl_seconds=300.0)
        cache._redis_checked = True
        cache._redis = mock_redis_client
        return cache

    def test_get_from_redis(self, cache_with_redis, mock_redis_client):
        """Get should query Redis first."""
        mock_redis_client.get.return_value = json.dumps({"value": "from_redis"})
        result = cache_with_redis.get("key1")
        assert result == {"value": "from_redis"}
        mock_redis_client.get.assert_called_once_with("aragora:test:key1")

    def test_get_updates_redis_hit_count(self, cache_with_redis, mock_redis_client):
        """Redis hit should update redis_hits."""
        mock_redis_client.get.return_value = json.dumps("value")
        cache_with_redis.get("key1")
        assert cache_with_redis._redis_hits == 1
        assert cache_with_redis._hits == 1

    def test_get_updates_redis_miss_count(self, cache_with_redis, mock_redis_client):
        """Redis miss should update redis_misses."""
        mock_redis_client.get.return_value = None
        cache_with_redis.get("key1")
        assert cache_with_redis._redis_misses == 1
        assert cache_with_redis._misses == 1

    def test_set_writes_to_redis(self, cache_with_redis, mock_redis_client):
        """Set should write to Redis."""
        cache_with_redis.set("key1", {"data": "value"})
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == "aragora:test:key1"
        assert call_args[0][1] == 300  # TTL as int
        # Verify JSON serialization
        assert json.loads(call_args[0][2]) == {"data": "value"}

    def test_set_also_writes_to_memory(self, cache_with_redis, mock_redis_client):
        """Set should also write to memory cache as backup."""
        cache_with_redis.set("key1", "value1")
        assert "key1" in cache_with_redis._memory_cache

    def test_invalidate_from_redis(self, cache_with_redis, mock_redis_client):
        """Invalidate should delete from Redis."""
        mock_redis_client.delete.return_value = 1
        cache_with_redis._memory_cache["key1"] = (time.time(), "value1")
        result = cache_with_redis.invalidate("key1")
        assert result is True
        mock_redis_client.delete.assert_called_once_with("aragora:test:key1")

    def test_invalidate_from_memory(self, cache_with_redis, mock_redis_client):
        """Invalidate should delete from memory cache."""
        mock_redis_client.delete.return_value = 0
        cache_with_redis._memory_cache["key1"] = (time.time(), "value1")
        result = cache_with_redis.invalidate("key1")
        assert result is True
        assert "key1" not in cache_with_redis._memory_cache

    def test_invalidate_nonexistent(self, cache_with_redis, mock_redis_client):
        """Invalidate nonexistent key should return False."""
        mock_redis_client.delete.return_value = 0
        result = cache_with_redis.invalidate("nonexistent")
        assert result is False

    def test_clear_clears_redis(self, cache_with_redis, mock_redis_client):
        """Clear should delete all prefixed keys from Redis."""
        mock_redis_client.keys.return_value = ["aragora:test:key1", "aragora:test:key2"]
        cache_with_redis._memory_cache["key1"] = (time.time(), "v1")
        cache_with_redis._memory_cache["key2"] = (time.time(), "v2")
        count = cache_with_redis.clear()
        mock_redis_client.keys.assert_called_once_with("aragora:test:*")
        mock_redis_client.delete.assert_called_once()
        assert count == 2

    def test_clear_clears_memory(self, cache_with_redis, mock_redis_client):
        """Clear should clear memory cache."""
        cache_with_redis._memory_cache["key1"] = (time.time(), "v1")
        cache_with_redis.clear()
        assert len(cache_with_redis._memory_cache) == 0

    def test_clear_prefix_partial_clear(self, cache_with_redis, mock_redis_client):
        """Clear_prefix should only clear matching keys."""
        mock_redis_client.keys.return_value = ["aragora:test:user:1"]
        cache_with_redis._memory_cache["user:1"] = (time.time(), "v1")
        cache_with_redis._memory_cache["user:2"] = (time.time(), "v2")
        cache_with_redis._memory_cache["other:1"] = (time.time(), "v3")
        count = cache_with_redis.clear_prefix("user:")
        assert count == 2
        assert "user:1" not in cache_with_redis._memory_cache
        assert "user:2" not in cache_with_redis._memory_cache
        assert "other:1" in cache_with_redis._memory_cache


# ============================================================================
# Error Recovery and Fallback Tests
# ============================================================================


class TestErrorRecoveryAndFallback:
    """Tests for error recovery and fallback behavior."""

    @pytest.fixture
    def cache_with_failing_redis(self):
        """Create cache with Redis that raises errors."""
        cache = RedisTTLCache(prefix="test", ttl_seconds=300.0)
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("Redis connection error")
        mock_redis.setex.side_effect = Exception("Redis connection error")
        mock_redis.delete.side_effect = Exception("Redis connection error")
        mock_redis.keys.side_effect = Exception("Redis connection error")
        cache._redis_checked = True
        cache._redis = mock_redis
        return cache

    def test_get_fallback_on_redis_error(self, cache_with_failing_redis):
        """Get should fallback to memory on Redis error."""
        # Pre-populate memory cache
        cache_with_failing_redis._memory_cache["key1"] = (time.time(), "memory_value")
        result = cache_with_failing_redis.get("key1")
        assert result == "memory_value"

    def test_get_miss_on_redis_error_no_memory(self, cache_with_failing_redis):
        """Get should return None if both Redis and memory miss."""
        result = cache_with_failing_redis.get("nonexistent")
        assert result is None

    def test_set_continues_on_redis_error(self, cache_with_failing_redis):
        """Set should still write to memory on Redis error."""
        cache_with_failing_redis.set("key1", "value1")
        assert "key1" in cache_with_failing_redis._memory_cache

    def test_invalidate_continues_on_redis_error(self, cache_with_failing_redis):
        """Invalidate should still work on memory if Redis fails."""
        cache_with_failing_redis._memory_cache["key1"] = (time.time(), "v1")
        result = cache_with_failing_redis.invalidate("key1")
        assert result is True
        assert "key1" not in cache_with_failing_redis._memory_cache

    def test_clear_continues_on_redis_error(self, cache_with_failing_redis):
        """Clear should still work on memory if Redis fails."""
        cache_with_failing_redis._memory_cache["key1"] = (time.time(), "v1")
        cache_with_failing_redis._memory_cache["key2"] = (time.time(), "v2")
        count = cache_with_failing_redis.clear()
        assert count == 2
        assert len(cache_with_failing_redis._memory_cache) == 0

    def test_clear_prefix_continues_on_redis_error(self, cache_with_failing_redis):
        """Clear_prefix should still work on memory if Redis fails."""
        cache_with_failing_redis._memory_cache["user:1"] = (time.time(), "v1")
        count = cache_with_failing_redis.clear_prefix("user:")
        assert count == 1


# ============================================================================
# Redis Connection Lazy Initialization Tests
# ============================================================================


class TestRedisLazyInitialization:
    """Tests for lazy Redis initialization."""

    def test_get_redis_lazy_init(self):
        """_get_redis should lazily initialize on first call."""
        cache = RedisTTLCache(prefix="test")
        with patch(
            "aragora.utils.redis_cache.RedisTTLCache._get_redis"
        ) as mock_get_redis:
            mock_get_redis.return_value = None
            cache.get("key1")
            mock_get_redis.assert_called()

    def test_redis_import_error_handled(self):
        """ImportError from redis_config should be handled gracefully."""
        cache = RedisTTLCache(prefix="test")
        with patch(
            "aragora.server.redis_config.get_redis_client",
            side_effect=ImportError("No module"),
        ):
            cache._redis_checked = False  # Reset to force re-check
            result = cache._get_redis()
            assert result is None
            assert cache._redis_checked is True

    def test_redis_none_from_config(self):
        """None from get_redis_client should be handled."""
        cache = RedisTTLCache(prefix="test")
        with patch(
            "aragora.server.redis_config.get_redis_client",
            return_value=None,
        ):
            cache._redis_checked = False
            result = cache._get_redis()
            assert result is None
            assert cache._redis_checked is True

    def test_redis_client_cached_after_init(self):
        """Redis client should be cached after initialization."""
        cache = RedisTTLCache(prefix="test")
        mock_client = MagicMock()
        with patch(
            "aragora.server.redis_config.get_redis_client",
            return_value=mock_client,
        ):
            cache._redis_checked = False
            result1 = cache._get_redis()
            result2 = cache._get_redis()
            assert result1 is mock_client
            assert result2 is mock_client


# ============================================================================
# Statistics Tests
# ============================================================================


class TestCacheStatistics:
    """Tests for cache statistics."""

    @pytest.fixture
    def cache_with_data(self):
        """Create cache with some operations performed."""
        cache = RedisTTLCache(prefix="stats", ttl_seconds=300.0, maxsize=100)
        cache._redis_checked = True
        cache._redis = None  # No Redis
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.get("key1")  # hit
        cache.get("key3")  # miss
        return cache

    def test_stats_size(self, cache_with_data):
        """Stats should report correct size."""
        stats = cache_with_data.stats
        assert stats["size"] == 2

    def test_stats_maxsize(self, cache_with_data):
        """Stats should report maxsize."""
        stats = cache_with_data.stats
        assert stats["maxsize"] == 100

    def test_stats_ttl(self, cache_with_data):
        """Stats should report TTL."""
        stats = cache_with_data.stats
        assert stats["ttl_seconds"] == 300.0

    def test_stats_prefix(self, cache_with_data):
        """Stats should report prefix."""
        stats = cache_with_data.stats
        assert stats["prefix"] == "stats"

    def test_stats_hits(self, cache_with_data):
        """Stats should report hits."""
        stats = cache_with_data.stats
        assert stats["hits"] == 1

    def test_stats_misses(self, cache_with_data):
        """Stats should report misses."""
        stats = cache_with_data.stats
        assert stats["misses"] == 1

    def test_stats_hit_rate(self, cache_with_data):
        """Stats should calculate hit rate."""
        stats = cache_with_data.stats
        assert stats["hit_rate"] == 0.5  # 1 hit / 2 total

    def test_stats_hit_rate_zero_total(self):
        """Hit rate should be 0 when no operations."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        stats = cache.stats
        assert stats["hit_rate"] == 0.0

    def test_stats_using_redis(self):
        """Stats should report Redis availability."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = MagicMock()
        stats = cache.stats
        assert stats["using_redis"] is True

    def test_stats_not_using_redis(self):
        """Stats should report Redis unavailability."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        stats = cache.stats
        assert stats["using_redis"] is False

    def test_stats_redis_hits_misses(self):
        """Stats should report Redis-specific hits/misses."""
        cache = RedisTTLCache()
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps("value")
        cache._redis_checked = True
        cache._redis = mock_redis

        cache.get("key1")  # Redis hit
        mock_redis.get.return_value = None
        cache.get("key2")  # Redis miss

        stats = cache.stats
        assert stats["redis_hits"] == 1
        assert stats["redis_misses"] == 1


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_set_operations(self):
        """Concurrent set operations should be thread-safe."""
        cache = RedisTTLCache(prefix="thread", maxsize=1000)
        cache._redis_checked = True
        cache._redis = None

        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    cache.set(f"worker{worker_id}:key{i}", f"value{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(cache) <= 1000  # Should respect maxsize

    def test_concurrent_get_operations(self):
        """Concurrent get operations should be thread-safe."""
        cache = RedisTTLCache(prefix="thread", maxsize=1000)
        cache._redis_checked = True
        cache._redis = None

        # Pre-populate cache
        for i in range(100):
            cache.set(f"key{i}", f"value{i}")

        errors = []
        results = []
        lock = threading.Lock()

        def worker():
            try:
                for i in range(100):
                    result = cache.get(f"key{i}")
                    with lock:
                        results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 1000  # 10 threads * 100 gets

    def test_concurrent_mixed_operations(self):
        """Concurrent mixed operations should be thread-safe."""
        cache = RedisTTLCache(prefix="thread", maxsize=100)
        cache._redis_checked = True
        cache._redis = None

        errors = []

        def setter():
            try:
                for i in range(50):
                    cache.set(f"key{i}", f"value{i}")
            except Exception as e:
                errors.append(e)

        def getter():
            try:
                for i in range(50):
                    cache.get(f"key{i}")
            except Exception as e:
                errors.append(e)

        def invalidator():
            try:
                for i in range(50):
                    cache.invalidate(f"key{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=setter),
            threading.Thread(target=setter),
            threading.Thread(target=getter),
            threading.Thread(target=getter),
            threading.Thread(target=invalidator),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# __len__ Tests
# ============================================================================


class TestLenMethod:
    """Tests for __len__ method."""

    def test_len_empty_cache(self):
        """Empty cache should have length 0."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        assert len(cache) == 0

    def test_len_with_entries(self):
        """Length should match number of entries."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.set("key3", "v3")
        assert len(cache) == 3

    def test_len_after_invalidation(self):
        """Length should decrease after invalidation."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.invalidate("key1")
        assert len(cache) == 1


# ============================================================================
# JSON Serialization Tests
# ============================================================================


class TestJSONSerialization:
    """Tests for JSON serialization in Redis operations."""

    def test_serialize_dict(self):
        """Dict values should be JSON serialized."""
        cache = RedisTTLCache(prefix="json")
        mock_redis = MagicMock()
        cache._redis_checked = True
        cache._redis = mock_redis

        cache.set("key1", {"name": "test", "count": 42})
        call_args = mock_redis.setex.call_args[0]
        serialized = call_args[2]
        assert json.loads(serialized) == {"name": "test", "count": 42}

    def test_serialize_list(self):
        """List values should be JSON serialized."""
        cache = RedisTTLCache(prefix="json")
        mock_redis = MagicMock()
        cache._redis_checked = True
        cache._redis = mock_redis

        cache.set("key1", [1, 2, 3, "four"])
        call_args = mock_redis.setex.call_args[0]
        serialized = call_args[2]
        assert json.loads(serialized) == [1, 2, 3, "four"]

    def test_serialize_non_json_uses_str(self):
        """Non-JSON-serializable values should use str conversion."""
        cache = RedisTTLCache(prefix="json")
        mock_redis = MagicMock()
        cache._redis_checked = True
        cache._redis = mock_redis

        # datetime is not directly JSON-serializable but has __str__
        from datetime import datetime

        now = datetime(2024, 1, 15, 12, 0, 0)
        cache.set("key1", {"time": now})
        # Should not raise - uses default=str

    def test_deserialize_from_redis(self):
        """Values from Redis should be JSON deserialized."""
        cache = RedisTTLCache(prefix="json")
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({"nested": {"value": True}})
        cache._redis_checked = True
        cache._redis = mock_redis

        result = cache.get("key1")
        assert result == {"nested": {"value": True}}


# ============================================================================
# HybridTTLCache Factory Tests
# ============================================================================


class TestHybridTTLCacheFactory:
    """Tests for the HybridTTLCache factory function."""

    def test_returns_redis_ttl_cache(self):
        """HybridTTLCache should return RedisTTLCache instance."""
        cache = HybridTTLCache(prefix="hybrid")
        assert isinstance(cache, RedisTTLCache)

    def test_passes_parameters(self):
        """HybridTTLCache should pass parameters correctly."""
        cache = HybridTTLCache(
            prefix="custom",
            ttl_seconds=600.0,
            maxsize=500,
        )
        assert cache._prefix == "custom"
        assert cache._ttl_seconds == 600.0
        assert cache._maxsize == 500

    def test_default_parameters(self):
        """HybridTTLCache should use defaults when not specified."""
        cache = HybridTTLCache()
        assert cache._prefix == "cache"
        assert cache._ttl_seconds == 300.0
        assert cache._maxsize == 1000


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_set_none_value(self):
        """Setting None value should work."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        cache.set("key1", None)
        result = cache.get("key1")
        assert result is None  # Value is None

    def test_empty_string_key(self):
        """Empty string key should work."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        cache.set("", "value")
        result = cache.get("")
        assert result == "value"

    def test_very_large_value(self):
        """Very large values should work."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        large_value = {"data": "x" * 100000}
        cache.set("key1", large_value)
        result = cache.get("key1")
        assert result == large_value

    def test_unicode_keys_and_values(self):
        """Unicode keys and values should work."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        cache.set("key_\u4e2d\u6587", {"value": "\u0e01\u0e02\u0e03"})
        result = cache.get("key_\u4e2d\u6587")
        assert result == {"value": "\u0e01\u0e02\u0e03"}

    def test_ttl_zero(self):
        """TTL of 0 should immediately expire."""
        cache = RedisTTLCache(ttl_seconds=0)
        cache._redis_checked = True
        cache._redis = None
        cache.set("key1", "value1")
        # Should expire immediately (or nearly)
        time.sleep(0.01)
        result = cache.get("key1")
        assert result is None

    def test_negative_ttl_treated_as_zero(self):
        """Negative TTL should behave like zero/immediate expiry."""
        cache = RedisTTLCache(ttl_seconds=-1)
        cache._redis_checked = True
        cache._redis = None
        cache.set("key1", "value1")
        time.sleep(0.01)
        result = cache.get("key1")
        assert result is None

    def test_maxsize_one(self):
        """Maxsize of 1 should work correctly."""
        cache = RedisTTLCache(maxsize=1)
        cache._redis_checked = True
        cache._redis = None
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        assert len(cache) == 1
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"

    def test_repeated_invalidation(self):
        """Repeated invalidation of same key should be safe."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        cache.set("key1", "value1")
        assert cache.invalidate("key1") is True
        assert cache.invalidate("key1") is False
        assert cache.invalidate("key1") is False

    def test_clear_empty_cache(self):
        """Clearing empty cache should work."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        count = cache.clear()
        assert count == 0

    def test_clear_prefix_empty_match(self):
        """Clear_prefix with no matches should return 0."""
        cache = RedisTTLCache()
        cache._redis_checked = True
        cache._redis = None
        cache.set("key1", "value1")
        count = cache.clear_prefix("nonexistent:")
        assert count == 0
        assert cache.get("key1") == "value1"


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestIntegrationScenarios:
    """Tests for realistic usage scenarios."""

    def test_leaderboard_caching_scenario(self):
        """Simulate leaderboard caching scenario."""
        cache = RedisTTLCache(prefix="leaderboard", ttl_seconds=60.0, maxsize=100)
        cache._redis_checked = True
        cache._redis = None

        # Cache leaderboard data
        leaderboard = [
            {"agent": "claude", "elo": 1850, "wins": 42},
            {"agent": "gpt4", "elo": 1820, "wins": 38},
            {"agent": "gemini", "elo": 1790, "wins": 35},
        ]
        cache.set("global", leaderboard)

        # Retrieve and verify
        result = cache.get("global")
        assert len(result) == 3
        assert result[0]["agent"] == "claude"

        # Invalidate on update
        cache.invalidate("global")
        assert cache.get("global") is None

    def test_user_session_scenario(self):
        """Simulate user session caching scenario."""
        cache = RedisTTLCache(prefix="session", ttl_seconds=3600.0, maxsize=10000)
        cache._redis_checked = True
        cache._redis = None

        # Store sessions
        for i in range(100):
            cache.set(f"user:{i}", {"user_id": i, "logged_in": True})

        # Verify retrieval
        session = cache.get("user:50")
        assert session["user_id"] == 50

        # Clear user sessions
        count = cache.clear_prefix("user:")
        assert count == 100

    def test_api_response_caching_scenario(self):
        """Simulate API response caching scenario."""
        cache = RedisTTLCache(prefix="api", ttl_seconds=300.0)
        cache._redis_checked = True
        cache._redis = None

        # Cache API response
        response = {
            "status": "success",
            "data": {"debates": [{"id": 1}, {"id": 2}]},
            "meta": {"page": 1, "total": 100},
        }
        cache.set("debates:list:page1", response)

        # Hit the cache
        cached = cache.get("debates:list:page1")
        assert cached["status"] == "success"
        assert len(cached["data"]["debates"]) == 2

        # Check stats
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["hit_rate"] == 1.0
