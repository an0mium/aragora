"""
Tests for aragora.routing.domain_matcher module.

Tests cover:
- _DomainCache TTL cache functionality
- DOMAIN_KEYWORDS structure and content
- DomainDetector keyword-based detection
- Cache hit/miss tracking
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.routing.domain_matcher import (
    DOMAIN_KEYWORDS,
    DEFAULT_CACHE_SIZE,
    DEFAULT_CACHE_TTL,
    _DomainCache,
    _domain_cache,
)


# =============================================================================
# TestDomainCache - TTL Cache Functionality
# =============================================================================


class TestDomainCacheInit:
    """Tests for _DomainCache initialization."""

    def test_default_parameters(self):
        """Should initialize with default parameters."""
        cache = _DomainCache()
        assert cache._max_size == DEFAULT_CACHE_SIZE
        assert cache._ttl == DEFAULT_CACHE_TTL
        assert cache._hits == 0
        assert cache._misses == 0

    def test_custom_parameters(self):
        """Should accept custom parameters."""
        cache = _DomainCache(max_size=100, ttl_seconds=60)
        assert cache._max_size == 100
        assert cache._ttl == 60


class TestDomainCacheMakeKey:
    """Tests for _DomainCache._make_key()."""

    def test_consistent_keys(self):
        """Same text and top_n should produce same key."""
        cache = _DomainCache()
        key1 = cache._make_key("test text", 3)
        key2 = cache._make_key("test text", 3)
        assert key1 == key2

    def test_case_insensitive(self):
        """Keys should be case-insensitive."""
        cache = _DomainCache()
        key1 = cache._make_key("Test Text", 3)
        key2 = cache._make_key("test text", 3)
        assert key1 == key2

    def test_strips_whitespace(self):
        """Keys should strip leading/trailing whitespace."""
        cache = _DomainCache()
        key1 = cache._make_key("  test  ", 3)
        key2 = cache._make_key("test", 3)
        assert key1 == key2

    def test_different_top_n_different_keys(self):
        """Different top_n should produce different keys."""
        cache = _DomainCache()
        key1 = cache._make_key("test", 3)
        key2 = cache._make_key("test", 5)
        assert key1 != key2

    def test_truncates_long_text(self):
        """Should truncate text longer than 500 chars."""
        cache = _DomainCache()
        long_text = "a" * 1000
        key = cache._make_key(long_text, 3)
        # Key should be 16 chars (sha256 hex truncated)
        assert len(key) == 16


class TestDomainCacheGetSet:
    """Tests for _DomainCache.get() and set()."""

    def test_get_returns_none_for_missing(self):
        """get() should return None for missing entries."""
        cache = _DomainCache()
        result = cache.get("nonexistent", 3)
        assert result is None
        assert cache._misses == 1

    def test_set_and_get(self):
        """set() should store value retrievable by get()."""
        cache = _DomainCache()
        result = [("security", 0.9), ("testing", 0.5)]
        cache.set("test text", 3, result)

        retrieved = cache.get("test text", 3)
        assert retrieved == result
        assert cache._hits == 1

    def test_expired_entries_return_none(self):
        """get() should return None for expired entries."""
        cache = _DomainCache(ttl_seconds=1)
        cache.set("test", 3, [("domain", 1.0)])

        # Wait for expiry
        time.sleep(1.1)

        result = cache.get("test", 3)
        assert result is None
        assert cache._misses == 1

    def test_eviction_at_capacity(self):
        """set() should evict oldest entries when at capacity."""
        cache = _DomainCache(max_size=10)

        # Fill cache
        for i in range(10):
            cache.set(f"text{i}", 3, [(f"domain{i}", 1.0)])
            time.sleep(0.01)  # Ensure different timestamps

        # Add one more (should evict oldest)
        cache.set("text_new", 3, [("new", 1.0)])

        # Cache should not exceed max_size
        assert len(cache._cache) <= 10


class TestDomainCacheStats:
    """Tests for _DomainCache.stats()."""

    def test_stats_initial(self):
        """stats() should return initial values."""
        cache = _DomainCache()
        stats = cache.stats()

        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_stats_after_operations(self):
        """stats() should reflect cache operations."""
        cache = _DomainCache()

        # One miss
        cache.get("missing", 3)

        # One hit
        cache.set("test", 3, [("domain", 1.0)])
        cache.get("test", 3)

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestDomainCacheClear:
    """Tests for _DomainCache.clear()."""

    def test_clear_removes_all(self):
        """clear() should remove all entries."""
        cache = _DomainCache()
        cache.set("text1", 3, [("domain", 1.0)])
        cache.set("text2", 3, [("domain", 1.0)])

        count = cache.clear()

        assert count == 2
        assert len(cache._cache) == 0

    def test_clear_returns_count(self):
        """clear() should return count of cleared entries."""
        cache = _DomainCache()
        cache.set("text1", 3, [("domain", 1.0)])

        count = cache.clear()
        assert count == 1


# =============================================================================
# TestDomainKeywords - Keyword Configuration
# =============================================================================


class TestDomainKeywords:
    """Tests for DOMAIN_KEYWORDS configuration."""

    def test_is_dict(self):
        """DOMAIN_KEYWORDS should be a dict."""
        assert isinstance(DOMAIN_KEYWORDS, dict)

    def test_has_core_domains(self):
        """Should have core domain categories."""
        expected_domains = ["security", "performance", "architecture", "testing", "api"]
        for domain in expected_domains:
            assert domain in DOMAIN_KEYWORDS

    def test_keywords_are_lists(self):
        """Each domain should have a list of keywords."""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            assert isinstance(keywords, list), f"{domain} should have a list of keywords"
            assert len(keywords) > 0, f"{domain} should have at least one keyword"

    def test_keywords_are_lowercase(self):
        """All keywords should be lowercase for matching."""
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                assert keyword == keyword.lower(), (
                    f"Keyword '{keyword}' in {domain} should be lowercase"
                )

    def test_security_keywords(self):
        """Security domain should have critical security keywords."""
        security_keywords = DOMAIN_KEYWORDS.get("security", [])
        critical = ["authentication", "authorization", "vulnerability", "xss", "sql injection"]
        for kw in critical:
            assert kw in security_keywords, f"Security should include '{kw}'"


# =============================================================================
# TestGlobalCache - Global Cache Instance
# =============================================================================


class TestGlobalCache:
    """Tests for global _domain_cache instance."""

    def test_global_cache_exists(self):
        """Global _domain_cache should exist."""
        assert _domain_cache is not None
        assert isinstance(_domain_cache, _DomainCache)

    def test_global_cache_is_singleton(self):
        """Global cache should be shared across imports."""
        from aragora.routing.domain_matcher import _domain_cache as cache2

        assert _domain_cache is cache2


# =============================================================================
# TestDomainCacheIntegration - End-to-End Scenarios
# =============================================================================


class TestDomainCacheIntegration:
    """Integration tests for domain cache."""

    def test_cache_hit_performance(self):
        """Cache hits should be fast."""
        cache = _DomainCache()

        # Set a value
        cache.set("performance test", 3, [("performance", 0.95)])

        # Time cache hits
        start = time.time()
        for _ in range(1000):
            cache.get("performance test", 3)
        elapsed = time.time() - start

        # 1000 cache hits should take < 100ms
        assert elapsed < 0.1

    def test_cache_workflow(self):
        """Test typical cache workflow."""
        cache = _DomainCache(max_size=100, ttl_seconds=60)

        # Miss -> compute -> store -> hit pattern
        result1 = cache.get("new task", 3)
        assert result1 is None  # Miss

        # Simulate expensive computation result
        computed = [("security", 0.8), ("api", 0.6)]
        cache.set("new task", 3, computed)

        result2 = cache.get("new task", 3)
        assert result2 == computed  # Hit

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
