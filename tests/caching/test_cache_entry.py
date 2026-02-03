"""
Tests for CacheEntry and CacheStats dataclasses.

Covers:
- CacheStats initialization and defaults
- CacheStats hit_rate property calculation
- CacheStats repr format
- CacheEntry creation with value and TTL
- CacheEntry expiration logic
- CacheEntry with no TTL (never expires)
- Edge cases: zero totals, boundary TTL values
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from aragora.caching.decorators import CacheEntry, CacheStats


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def empty_stats() -> CacheStats:
    """Create a CacheStats with default (zero) values."""
    return CacheStats()


@pytest.fixture
def populated_stats() -> CacheStats:
    """Create a CacheStats with non-zero values."""
    return CacheStats(hits=80, misses=20, size=50, maxsize=128, evictions=5)


# ===========================================================================
# Test: CacheStats Initialization
# ===========================================================================


class TestCacheStatsInit:
    """Tests for CacheStats default initialization."""

    def test_default_hits(self, empty_stats: CacheStats):
        """Default hits should be zero."""
        assert empty_stats.hits == 0

    def test_default_misses(self, empty_stats: CacheStats):
        """Default misses should be zero."""
        assert empty_stats.misses == 0

    def test_default_size(self, empty_stats: CacheStats):
        """Default size should be zero."""
        assert empty_stats.size == 0

    def test_default_maxsize(self, empty_stats: CacheStats):
        """Default maxsize should be zero."""
        assert empty_stats.maxsize == 0

    def test_default_evictions(self, empty_stats: CacheStats):
        """Default evictions should be zero."""
        assert empty_stats.evictions == 0

    def test_custom_values(self, populated_stats: CacheStats):
        """Custom values are stored correctly."""
        assert populated_stats.hits == 80
        assert populated_stats.misses == 20
        assert populated_stats.size == 50
        assert populated_stats.maxsize == 128
        assert populated_stats.evictions == 5

    def test_partial_initialization(self):
        """Only some fields can be specified."""
        stats = CacheStats(hits=10, misses=5)
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.size == 0
        assert stats.maxsize == 0
        assert stats.evictions == 0


# ===========================================================================
# Test: CacheStats hit_rate Property
# ===========================================================================


class TestCacheStatsHitRate:
    """Tests for the hit_rate calculated property."""

    def test_hit_rate_zero_total(self, empty_stats: CacheStats):
        """Hit rate is 0.0 when no accesses have been made."""
        assert empty_stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        """Hit rate is 100% when all accesses are hits."""
        stats = CacheStats(hits=100, misses=0)
        assert stats.hit_rate == 100.0

    def test_hit_rate_all_misses(self):
        """Hit rate is 0% when all accesses are misses."""
        stats = CacheStats(hits=0, misses=100)
        assert stats.hit_rate == 0.0

    def test_hit_rate_mixed(self, populated_stats: CacheStats):
        """Hit rate is calculated correctly for mixed access."""
        # 80 hits / (80 + 20) total = 80%
        assert populated_stats.hit_rate == 80.0

    def test_hit_rate_fifty_percent(self):
        """Hit rate is 50% when hits and misses are equal."""
        stats = CacheStats(hits=50, misses=50)
        assert stats.hit_rate == 50.0

    def test_hit_rate_precision(self):
        """Hit rate retains floating point precision."""
        stats = CacheStats(hits=1, misses=2)
        expected = (1 / 3) * 100
        assert abs(stats.hit_rate - expected) < 1e-10

    def test_hit_rate_returns_float(self, empty_stats: CacheStats):
        """Hit rate always returns a float."""
        assert isinstance(empty_stats.hit_rate, float)


# ===========================================================================
# Test: CacheStats __repr__
# ===========================================================================


class TestCacheStatsRepr:
    """Tests for CacheStats string representation."""

    def test_repr_contains_hits(self, populated_stats: CacheStats):
        """Repr includes hit count."""
        assert "hits=80" in repr(populated_stats)

    def test_repr_contains_misses(self, populated_stats: CacheStats):
        """Repr includes miss count."""
        assert "misses=20" in repr(populated_stats)

    def test_repr_contains_size(self, populated_stats: CacheStats):
        """Repr includes size."""
        assert "size=50" in repr(populated_stats)

    def test_repr_contains_maxsize(self, populated_stats: CacheStats):
        """Repr includes maxsize."""
        assert "maxsize=128" in repr(populated_stats)

    def test_repr_contains_evictions(self, populated_stats: CacheStats):
        """Repr includes eviction count."""
        assert "evictions=5" in repr(populated_stats)

    def test_repr_contains_hit_rate(self, populated_stats: CacheStats):
        """Repr includes hit rate percentage."""
        assert "hit_rate=80.0%" in repr(populated_stats)

    def test_repr_format(self, empty_stats: CacheStats):
        """Repr follows expected format."""
        result = repr(empty_stats)
        assert result.startswith("CacheStats(")
        assert result.endswith(")")


# ===========================================================================
# Test: CacheEntry Initialization
# ===========================================================================


class TestCacheEntryInit:
    """Tests for CacheEntry creation."""

    def test_basic_creation(self):
        """CacheEntry can be created with value and timestamp."""
        entry = CacheEntry(value="hello", created_at=time.time())
        assert entry.value == "hello"

    def test_created_at_stored(self):
        """Timestamp is stored correctly."""
        now = time.time()
        entry = CacheEntry(value=42, created_at=now)
        assert entry.created_at == now

    def test_default_ttl_is_none(self):
        """Default TTL is None (never expires)."""
        entry = CacheEntry(value="test", created_at=time.time())
        assert entry.ttl_seconds is None

    def test_custom_ttl(self):
        """Custom TTL is stored correctly."""
        entry = CacheEntry(value="test", created_at=time.time(), ttl_seconds=60.0)
        assert entry.ttl_seconds == 60.0

    def test_stores_various_types(self):
        """CacheEntry can store any value type."""
        now = time.time()

        int_entry = CacheEntry(value=42, created_at=now)
        assert int_entry.value == 42

        list_entry = CacheEntry(value=[1, 2, 3], created_at=now)
        assert list_entry.value == [1, 2, 3]

        dict_entry = CacheEntry(value={"key": "val"}, created_at=now)
        assert dict_entry.value == {"key": "val"}

        none_entry = CacheEntry(value=None, created_at=now)
        assert none_entry.value is None

    def test_stores_empty_string(self):
        """CacheEntry can store empty string as value."""
        entry = CacheEntry(value="", created_at=time.time())
        assert entry.value == ""

    def test_stores_zero(self):
        """CacheEntry can store zero as value."""
        entry = CacheEntry(value=0, created_at=time.time())
        assert entry.value == 0

    def test_stores_false(self):
        """CacheEntry can store False as value."""
        entry = CacheEntry(value=False, created_at=time.time())
        assert entry.value is False


# ===========================================================================
# Test: CacheEntry Expiration
# ===========================================================================


class TestCacheEntryExpiration:
    """Tests for CacheEntry expiration logic."""

    def test_no_ttl_never_expires(self):
        """Entry without TTL never expires."""
        entry = CacheEntry(value="test", created_at=0.0, ttl_seconds=None)
        assert entry.is_expired() is False

    def test_not_expired_within_ttl(self):
        """Entry is not expired within TTL window."""
        entry = CacheEntry(value="test", created_at=time.time(), ttl_seconds=300.0)
        assert entry.is_expired() is False

    def test_expired_after_ttl(self):
        """Entry is expired after TTL has passed."""
        # Created 10 seconds ago, TTL of 5 seconds
        entry = CacheEntry(
            value="test",
            created_at=time.time() - 10.0,
            ttl_seconds=5.0,
        )
        assert entry.is_expired() is True

    def test_expired_at_boundary(self):
        """Entry at exactly the TTL boundary is expired (> check)."""
        # Created exactly ttl_seconds ago -- time() - created > ttl means expired
        past = time.time() - 1.0
        entry = CacheEntry(value="test", created_at=past, ttl_seconds=0.5)
        assert entry.is_expired() is True

    def test_zero_ttl_expires_immediately(self):
        """Entry with zero TTL expires immediately."""
        entry = CacheEntry(
            value="test",
            created_at=time.time() - 0.001,
            ttl_seconds=0.0,
        )
        assert entry.is_expired() is True

    def test_large_ttl_not_expired(self):
        """Entry with very large TTL does not expire quickly."""
        entry = CacheEntry(
            value="test",
            created_at=time.time() - 3600.0,  # 1 hour ago
            ttl_seconds=86400.0,  # 24 hours
        )
        assert entry.is_expired() is False

    def test_expired_uses_current_time(self):
        """Expiration check uses current time, not cached time."""
        now = time.time()
        entry = CacheEntry(value="test", created_at=now, ttl_seconds=1.0)

        # Patch time.time to simulate future
        with patch("aragora.caching.decorators.time") as mock_time:
            mock_time.time.return_value = now + 2.0
            assert entry.is_expired() is True

        with patch("aragora.caching.decorators.time") as mock_time:
            mock_time.time.return_value = now + 0.5
            assert entry.is_expired() is False
