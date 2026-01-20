"""Tests for RLM compressor performance optimizations.

Tests the LRU cache, concurrent LLM call control, and cache statistics.
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, patch

from aragora.rlm.compressor import (
    LRUCompressionCache,
    CacheEntry,
    HierarchicalCompressor,
    clear_compression_cache,
    get_compression_cache_stats,
    configure_compression_cache,
    get_call_semaphore,
)
from aragora.rlm.types import RLMContext, RLMConfig


class TestLRUCompressionCache:
    """Test LRU cache implementation."""

    def test_cache_init(self):
        """Test cache initialization."""
        cache = LRUCompressionCache(max_size=100, ttl_seconds=60.0)

        assert cache.max_size == 100
        assert cache.ttl_seconds == 60.0
        assert len(cache._cache) == 0

    def test_cache_set_and_get(self):
        """Test setting and getting cache entries."""
        cache = LRUCompressionCache(max_size=100, ttl_seconds=60.0)

        context = RLMContext(
            original_content="test",
            original_tokens=1,
        )

        cache.set("key1", context)
        result = cache.get("key1")

        assert result is context
        assert cache._hits == 1
        assert cache._misses == 0

    def test_cache_miss(self):
        """Test cache miss."""
        cache = LRUCompressionCache(max_size=100, ttl_seconds=60.0)

        result = cache.get("nonexistent")

        assert result is None
        assert cache._misses == 1

    def test_cache_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = LRUCompressionCache(max_size=100, ttl_seconds=0.1)  # 100ms TTL

        context = RLMContext(original_content="test", original_tokens=1)
        cache.set("key1", context)

        # Should get it immediately
        assert cache.get("key1") is context

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should be expired
        assert cache.get("key1") is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = LRUCompressionCache(max_size=3, ttl_seconds=3600.0)

        # Add 3 entries
        for i in range(3):
            context = RLMContext(original_content=f"test{i}", original_tokens=1)
            cache.set(f"key{i}", context)

        # Access key0 to make it recently used
        cache.get("key0")

        # Add a 4th entry - should evict key1 (least recently used)
        context = RLMContext(original_content="test3", original_tokens=1)
        cache.set("key3", context)

        # key1 should be evicted
        assert cache.get("key1") is None
        # key0 should still be there
        assert cache.get("key0") is not None
        # key2 should still be there
        assert cache.get("key2") is not None
        # key3 should be there
        assert cache.get("key3") is not None

    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = LRUCompressionCache(max_size=100, ttl_seconds=3600.0)

        for i in range(5):
            context = RLMContext(original_content=f"test{i}", original_tokens=1)
            cache.set(f"key{i}", context)

        assert len(cache._cache) == 5

        cache.clear()

        assert len(cache._cache) == 0

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCompressionCache(max_size=100, ttl_seconds=3600.0)

        context = RLMContext(original_content="test", original_tokens=1)
        cache.set("key1", context)

        # Hit
        cache.get("key1")
        # Miss
        cache.get("nonexistent")

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestGlobalCacheFunctions:
    """Test global cache utility functions."""

    def test_clear_compression_cache(self):
        """Test clearing the global cache."""
        # Configure a small cache for testing
        configure_compression_cache(max_size=10, ttl_seconds=60.0)

        # Add something (via module internals)
        from aragora.rlm import compressor

        context = RLMContext(original_content="test", original_tokens=1)
        compressor._compression_cache.set("test_key", context)

        # Clear it
        clear_compression_cache()

        # Should be empty
        stats = get_compression_cache_stats()
        assert stats["size"] == 0

    def test_get_compression_cache_stats(self):
        """Test getting cache stats."""
        configure_compression_cache(max_size=50, ttl_seconds=120.0)

        stats = get_compression_cache_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert stats["max_size"] == 50

    def test_configure_compression_cache(self):
        """Test configuring the cache."""
        configure_compression_cache(max_size=200, ttl_seconds=7200.0)

        stats = get_compression_cache_stats()

        assert stats["max_size"] == 200


class TestSemaphoreControl:
    """Test semaphore-based concurrency control."""

    def test_get_call_semaphore(self):
        """Test getting the call semaphore."""
        # Reset the semaphore
        from aragora.rlm import compressor

        compressor._call_semaphore = None

        semaphore = get_call_semaphore(max_concurrent=5)

        assert isinstance(semaphore, asyncio.Semaphore)

    def test_semaphore_reuse(self):
        """Test that semaphore is reused."""
        from aragora.rlm import compressor

        compressor._call_semaphore = None

        sem1 = get_call_semaphore(10)
        sem2 = get_call_semaphore(10)

        # Should be the same semaphore
        assert sem1 is sem2


class TestCompressorCacheIntegration:
    """Test compressor integration with cache."""

    @pytest.mark.asyncio
    async def test_compressor_uses_cache(self):
        """Test that compressor uses the cache."""
        configure_compression_cache(max_size=100, ttl_seconds=3600.0)
        clear_compression_cache()

        config = RLMConfig(cache_compressions=True)
        compressor = HierarchicalCompressor(config=config, agent_call=None)

        # Compress some content
        result1 = await compressor.compress("Test content for compression")

        # Stats should show cache set
        assert result1.cache_hits == 0

        # Compress again - should hit cache
        result2 = await compressor.compress("Test content for compression")

        assert result2.cache_hits == 1
        assert result2.time_seconds == 0.0  # Cached results are instant

    @pytest.mark.asyncio
    async def test_compressor_cache_disabled(self):
        """Test that compressor respects cache_compressions setting."""
        configure_compression_cache(max_size=100, ttl_seconds=3600.0)
        clear_compression_cache()

        config = RLMConfig(cache_compressions=False)
        compressor = HierarchicalCompressor(config=config, agent_call=None)

        # Compress twice
        result1 = await compressor.compress("Test content")
        result2 = await compressor.compress("Test content")

        # Neither should be cache hits since caching is disabled
        assert result1.cache_hits == 0
        assert result2.cache_hits == 0


class TestCacheEntry:
    """Test CacheEntry dataclass."""

    def test_cache_entry_creation(self):
        """Test creating a cache entry."""
        context = RLMContext(original_content="test", original_tokens=1)

        entry = CacheEntry(
            context=context,
            created_at=time.time(),
            access_count=0,
        )

        assert entry.context is context
        assert entry.access_count == 0

    def test_cache_entry_default_access_count(self):
        """Test default access count."""
        context = RLMContext(original_content="test", original_tokens=1)

        entry = CacheEntry(
            context=context,
            created_at=time.time(),
        )

        assert entry.access_count == 0
