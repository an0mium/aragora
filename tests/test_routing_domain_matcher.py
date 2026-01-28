"""Tests for routing.domain_matcher module.

Tests cover:
- _DomainCache TTL cache behavior
- Cache key normalization
- Cache eviction at capacity
- Hit/miss statistics tracking
"""

import time

import pytest

from aragora.routing.domain_matcher import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_CACHE_TTL,
    _DomainCache,
)


class TestDomainCacheDefaults:
    """Tests for default configuration values."""

    def test_default_cache_ttl(self):
        """Default TTL should be 1 hour (3600 seconds)."""
        assert DEFAULT_CACHE_TTL == 3600

    def test_default_cache_size(self):
        """Default max size should be 500 entries."""
        assert DEFAULT_CACHE_SIZE == 500


class TestDomainCache:
    """Tests for _DomainCache TTL cache."""

    def test_cache_hit(self):
        """Cache should return stored result."""
        cache = _DomainCache()
        result = [("security", 0.95), ("auth", 0.85)]
        cache.set("authentication systems", 2, result)
        assert cache.get("authentication systems", 2) == result

    def test_cache_miss_on_different_text(self):
        """Cache should miss for different text."""
        cache = _DomainCache()
        cache.set("authentication", 2, [("security", 0.9)])
        assert cache.get("authorization", 2) is None

    def test_cache_miss_on_different_top_n(self):
        """Cache should miss for different top_n."""
        cache = _DomainCache()
        cache.set("authentication", 2, [("security", 0.9)])
        assert cache.get("authentication", 3) is None

    def test_cache_miss_on_ttl_expiration(self):
        """Cache should miss after TTL expiration."""
        cache = _DomainCache(ttl_seconds=0)  # Immediate expiration
        cache.set("test", 1, [("domain", 0.9)])
        time.sleep(0.01)
        assert cache.get("test", 1) is None

    def test_cache_eviction_at_capacity(self):
        """Oldest entries should be evicted when cache is full."""
        cache = _DomainCache(max_size=10)
        # Fill cache beyond capacity
        for i in range(15):
            cache.set(f"query{i}", 1, [("domain", float(i))])
        assert len(cache._cache) <= 10

    def test_cache_eviction_removes_oldest(self):
        """Eviction should remove oldest entries first."""
        cache = _DomainCache(max_size=5)
        # Add entries with slight delays to ensure ordering
        for i in range(5):
            cache.set(f"query{i}", 1, [("domain", float(i))])
            time.sleep(0.001)  # Ensure timestamp differences

        # Add more entries to trigger eviction
        cache.set("new_query", 1, [("new", 1.0)])

        # Oldest entry should be gone
        assert cache.get("query0", 1) is None
        # Newer entries should still exist
        assert cache.get("new_query", 1) == [("new", 1.0)]

    def test_cache_statistics_initial(self):
        """Stats should be zero initially."""
        cache = _DomainCache()
        stats = cache.stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0
        assert stats["size"] == 0

    def test_cache_statistics_tracking(self):
        """Stats should track hits and misses."""
        cache = _DomainCache()
        cache.set("q1", 1, [("a", 0.9)])
        cache.get("q1", 1)  # hit
        cache.get("q2", 1)  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_statistics_multiple_hits(self):
        """Stats should accumulate correctly over multiple accesses."""
        cache = _DomainCache()
        cache.set("q1", 1, [("a", 0.9)])
        cache.get("q1", 1)  # hit
        cache.get("q1", 1)  # hit
        cache.get("q1", 1)  # hit
        cache.get("q2", 1)  # miss
        stats = cache.stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.75

    def test_clear_returns_count(self):
        """Clear should return number of cleared entries."""
        cache = _DomainCache()
        cache.set("q1", 1, [])
        cache.set("q2", 1, [])
        count = cache.clear()
        assert count == 2
        assert len(cache._cache) == 0

    def test_clear_empty_cache(self):
        """Clear on empty cache should return 0."""
        cache = _DomainCache()
        count = cache.clear()
        assert count == 0

    def test_key_normalization_lowercase(self):
        """Cache keys should normalize text to lowercase."""
        cache = _DomainCache()
        cache.set("HELLO WORLD", 2, [("test", 0.9)])
        assert cache.get("hello world", 2) == [("test", 0.9)]

    def test_key_normalization_strip(self):
        """Cache keys should strip whitespace."""
        cache = _DomainCache()
        cache.set("  hello world  ", 2, [("test", 0.9)])
        assert cache.get("hello world", 2) == [("test", 0.9)]

    def test_key_normalization_combined(self):
        """Cache keys should normalize (lowercase + strip) together."""
        cache = _DomainCache()
        cache.set("  HELLO WORLD  ", 2, [("test", 0.9)])
        assert cache.get("hello world", 2) == [("test", 0.9)]

    def test_key_normalization_truncation(self):
        """Cache keys should truncate long text to 500 chars."""
        cache = _DomainCache()
        long_text = "a" * 1000
        cache.set(long_text, 2, [("test", 0.9)])
        # Should still find it via the truncated key
        assert cache.get(long_text, 2) == [("test", 0.9)]

    def test_stats_includes_config(self):
        """Stats should include configuration values."""
        cache = _DomainCache(max_size=100, ttl_seconds=60)
        stats = cache.stats()
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 60

    def test_overwrite_existing_key(self):
        """Setting same key should overwrite existing entry."""
        cache = _DomainCache()
        cache.set("query", 2, [("old", 0.5)])
        cache.set("query", 2, [("new", 0.9)])
        assert cache.get("query", 2) == [("new", 0.9)]
        assert len(cache._cache) == 1

    def test_empty_result_list(self):
        """Cache should handle empty result lists."""
        cache = _DomainCache()
        cache.set("empty", 1, [])
        assert cache.get("empty", 1) == []

    def test_multiple_top_n_values_same_text(self):
        """Same text with different top_n should be separate entries."""
        cache = _DomainCache()
        cache.set("query", 1, [("a", 0.9)])
        cache.set("query", 2, [("a", 0.9), ("b", 0.8)])
        cache.set("query", 3, [("a", 0.9), ("b", 0.8), ("c", 0.7)])

        assert cache.get("query", 1) == [("a", 0.9)]
        assert cache.get("query", 2) == [("a", 0.9), ("b", 0.8)]
        assert cache.get("query", 3) == [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        assert len(cache._cache) == 3
