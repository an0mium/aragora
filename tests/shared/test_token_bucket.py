"""Tests for unified token bucket rate limiter.

Tests cover:
- Token bucket initialization
- Synchronous acquisition (try_acquire, acquire)
- Asynchronous acquisition (acquire_async)
- Token release
- API header updates
- Statistics tracking
- Keyed token bucket (per-key limiting)
- Thread safety
- Edge cases
"""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from aragora.shared.rate_limiting.token_bucket import (
    KeyedTokenBucket,
    TokenBucket,
    TokenBucketStats,
)


class TestTokenBucketInit:
    """Tests for TokenBucket initialization."""

    def test_default_values(self):
        """Should initialize with sensible defaults."""
        bucket = TokenBucket()
        assert bucket.rate_per_minute == 60.0
        assert bucket.burst == 10
        assert bucket.name == "default"

    def test_custom_values(self):
        """Should accept custom configuration."""
        bucket = TokenBucket(rate_per_minute=120.0, burst=20, name="test")
        assert bucket.rate_per_minute == 120.0
        assert bucket.burst == 20
        assert bucket.name == "test"

    def test_starts_with_full_burst(self):
        """Bucket should start with full burst capacity."""
        bucket = TokenBucket(burst=15)
        assert bucket.available_tokens == 15.0


class TestTryAcquire:
    """Tests for non-blocking try_acquire."""

    def test_success_when_tokens_available(self):
        """Should return True when tokens are available."""
        bucket = TokenBucket(rate_per_minute=60, burst=10)
        assert bucket.try_acquire() is True

    def test_consumes_token(self):
        """Should consume one token on success."""
        bucket = TokenBucket(rate_per_minute=60, burst=5)
        initial = bucket.available_tokens

        bucket.try_acquire()

        # Allow for small timing differences
        assert bucket.available_tokens < initial

    def test_fails_when_empty(self):
        """Should return False when no tokens available."""
        bucket = TokenBucket(rate_per_minute=60, burst=2)

        # Exhaust all tokens
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is False

    def test_multiple_tokens(self):
        """Should support acquiring multiple tokens at once."""
        bucket = TokenBucket(rate_per_minute=60, burst=10)

        assert bucket.try_acquire(tokens=5) is True
        assert bucket.available_tokens < 6  # Should have ~5 or fewer left

        assert bucket.try_acquire(tokens=10) is False  # Not enough

    def test_tracks_statistics(self):
        """Should track acquired and rejected counts."""
        bucket = TokenBucket(rate_per_minute=60, burst=2)

        bucket.try_acquire()  # acquired
        bucket.try_acquire()  # acquired
        bucket.try_acquire()  # rejected

        stats = bucket.stats
        assert stats["acquired"] == 2
        assert stats["rejected"] == 1


class TestAcquireBlocking:
    """Tests for blocking acquire with timeout."""

    def test_returns_immediately_if_available(self):
        """Should return immediately when tokens available."""
        bucket = TokenBucket(rate_per_minute=60, burst=10)
        start = time.monotonic()

        result = bucket.acquire(timeout=5.0)

        elapsed = time.monotonic() - start
        assert result is True
        assert elapsed < 0.1  # Should be nearly instant

    def test_waits_for_refill(self):
        """Should wait for tokens to refill."""
        # 600 per minute = 10 per second = 0.1 sec per token
        bucket = TokenBucket(rate_per_minute=600, burst=1)

        # Exhaust token
        bucket.try_acquire()

        start = time.monotonic()
        result = bucket.acquire(timeout=1.0)
        elapsed = time.monotonic() - start

        assert result is True
        assert elapsed >= 0.05  # Should have waited for refill

    def test_times_out_when_exhausted(self):
        """Should return False when timeout exceeded."""
        bucket = TokenBucket(rate_per_minute=6, burst=1)  # Very slow refill

        # Exhaust token
        bucket.try_acquire()

        start = time.monotonic()
        result = bucket.acquire(timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed >= 0.1  # Should have waited full timeout

    def test_zero_timeout_is_non_blocking(self):
        """Zero timeout should behave like try_acquire."""
        bucket = TokenBucket(rate_per_minute=60, burst=1)
        bucket.try_acquire()  # Exhaust

        result = bucket.acquire(timeout=0.0)
        assert result is False


class TestAcquireAsync:
    """Tests for async acquire_async."""

    @pytest.mark.asyncio
    async def test_returns_immediately_if_available(self):
        """Should return immediately when tokens available."""
        bucket = TokenBucket(rate_per_minute=60, burst=10)
        start = time.monotonic()

        result = await bucket.acquire_async(timeout=5.0)

        elapsed = time.monotonic() - start
        assert result is True
        assert elapsed < 0.1  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_waits_for_refill(self):
        """Should wait for tokens to refill without blocking event loop."""
        # 600 per minute = 10 per second = 0.1 sec per token
        bucket = TokenBucket(rate_per_minute=600, burst=1)

        # Exhaust token
        bucket.try_acquire()

        start = time.monotonic()
        result = await bucket.acquire_async(timeout=1.0)
        elapsed = time.monotonic() - start

        assert result is True
        assert elapsed >= 0.05  # Should have waited for refill

    @pytest.mark.asyncio
    async def test_times_out_when_exhausted(self):
        """Should return False when timeout exceeded."""
        bucket = TokenBucket(rate_per_minute=6, burst=1)  # Very slow refill

        # Exhaust token
        bucket.try_acquire()

        start = time.monotonic()
        result = await bucket.acquire_async(timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed >= 0.1  # Should have waited full timeout

    @pytest.mark.asyncio
    async def test_concurrent_async_acquire(self):
        """Multiple async acquires should work concurrently."""
        bucket = TokenBucket(rate_per_minute=600, burst=5)

        async def acquire_task():
            return await bucket.acquire_async(timeout=2.0)

        # Launch 5 concurrent acquires
        results = await asyncio.gather(*[acquire_task() for _ in range(5)])

        # All should succeed (bucket has 5 burst capacity)
        assert all(results)


class TestRelease:
    """Tests for token release."""

    def test_release_restores_token(self):
        """Release should restore a token to the bucket."""
        bucket = TokenBucket(rate_per_minute=60, burst=5)

        # Exhaust tokens
        for _ in range(5):
            bucket.try_acquire()

        assert bucket.try_acquire() is False

        # Release one
        bucket.release()

        assert bucket.try_acquire() is True

    def test_release_respects_burst_limit(self):
        """Release should not exceed burst capacity."""
        bucket = TokenBucket(rate_per_minute=60, burst=5)

        # Release extra tokens
        for _ in range(10):
            bucket.release()

        # Should not exceed burst
        assert bucket.available_tokens <= 5.0


class TestUpdateFromHeaders:
    """Tests for API header updates."""

    def test_updates_from_standard_headers(self):
        """Should parse standard rate limit headers."""
        bucket = TokenBucket()

        headers = {
            "X-RateLimit-Limit": "100",
            "X-RateLimit-Remaining": "50",
            "X-RateLimit-Reset": "1700000000.0",
        }

        bucket.update_from_headers(headers)

        stats = bucket.stats
        assert stats["api_limit"] == 100
        assert stats["api_remaining"] == 50

    def test_updates_from_lowercase_headers(self):
        """Should parse lowercase header variants."""
        bucket = TokenBucket()

        headers = {
            "x-ratelimit-limit": "200",
            "x-ratelimit-remaining": "75",
        }

        bucket.update_from_headers(headers)

        stats = bucket.stats
        assert stats["api_limit"] == 200
        assert stats["api_remaining"] == 75

    def test_ignores_invalid_values(self):
        """Should ignore non-numeric header values."""
        bucket = TokenBucket()

        headers = {
            "X-RateLimit-Limit": "not-a-number",
            "X-RateLimit-Remaining": "50",
        }

        bucket.update_from_headers(headers)

        stats = bucket.stats
        assert stats["api_limit"] is None  # Invalid, not set
        assert stats["api_remaining"] == 50


class TestStatistics:
    """Tests for statistics tracking."""

    def test_stats_include_all_fields(self):
        """Stats should include all expected fields."""
        bucket = TokenBucket(rate_per_minute=120, burst=15, name="test")

        bucket.try_acquire()
        bucket.try_acquire()

        stats = bucket.stats
        assert "name" in stats
        assert "rate_per_minute" in stats
        assert "burst" in stats
        assert "tokens_available" in stats
        assert "acquired" in stats
        assert "rejected" in stats
        assert "api_limit" in stats
        assert "api_remaining" in stats

        assert stats["name"] == "test"
        assert stats["rate_per_minute"] == 120
        assert stats["burst"] == 15
        assert stats["acquired"] == 2

    def test_reset_stats(self):
        """reset_stats should clear counters."""
        bucket = TokenBucket()

        bucket.try_acquire()
        bucket.try_acquire()

        bucket.reset_stats()

        stats = bucket.stats
        assert stats["acquired"] == 0
        assert stats["rejected"] == 0


class TestKeyedTokenBucket:
    """Tests for per-key rate limiting."""

    def test_separate_limits_per_key(self):
        """Each key should have independent limits."""
        limiter = KeyedTokenBucket(rate_per_minute=60, burst=2)

        # Key1 uses its limit
        assert limiter.try_acquire("key1") is True
        assert limiter.try_acquire("key1") is True
        assert limiter.try_acquire("key1") is False

        # Key2 is independent
        assert limiter.try_acquire("key2") is True
        assert limiter.try_acquire("key2") is True
        assert limiter.try_acquire("key2") is False

    def test_tracks_active_keys(self):
        """Should track number of active keys."""
        limiter = KeyedTokenBucket()

        limiter.try_acquire("a")
        limiter.try_acquire("b")
        limiter.try_acquire("c")

        stats = limiter.stats
        assert stats["active_keys"] == 3

    def test_aggregate_stats(self):
        """Should track aggregate statistics across all keys."""
        limiter = KeyedTokenBucket(rate_per_minute=60, burst=1)

        limiter.try_acquire("a")  # acquired
        limiter.try_acquire("b")  # acquired
        limiter.try_acquire("a")  # rejected (exhausted)

        stats = limiter.stats
        assert stats["acquired"] == 2
        assert stats["rejected"] == 1

    def test_get_key_stats(self):
        """Should return stats for specific key."""
        limiter = KeyedTokenBucket()

        limiter.try_acquire("mykey")
        limiter.try_acquire("mykey")

        key_stats = limiter.get_key_stats("mykey")
        assert key_stats is not None
        assert key_stats["acquired"] == 2

    def test_get_key_stats_nonexistent(self):
        """Should return None for nonexistent key."""
        limiter = KeyedTokenBucket()

        assert limiter.get_key_stats("nonexistent") is None

    @pytest.mark.asyncio
    async def test_async_acquire_keyed(self):
        """Should support async acquire for keyed buckets."""
        limiter = KeyedTokenBucket(rate_per_minute=600, burst=5)

        result = await limiter.acquire_async("user123", timeout=1.0)
        assert result is True


class TestThreadSafety:
    """Tests for thread-safe operation."""

    def test_concurrent_try_acquire(self):
        """Multiple threads can try_acquire safely."""
        bucket = TokenBucket(rate_per_minute=60000, burst=1000)
        acquired_count = 0
        lock = threading.Lock()

        def acquire_thread():
            nonlocal acquired_count
            if bucket.try_acquire():
                with lock:
                    acquired_count += 1

        threads = [threading.Thread(target=acquire_thread) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have acquired exactly burst capacity
        assert acquired_count == 100  # All should succeed

    def test_thread_pool_usage(self):
        """Should work correctly with thread pool executor."""
        bucket = TokenBucket(rate_per_minute=60000, burst=50)
        results = []
        lock = threading.Lock()

        def worker():
            result = bucket.try_acquire()
            with lock:
                results.append(result)
            return result

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker) for _ in range(50)]
            for f in futures:
                f.result()

        # All 50 should succeed
        assert len(results) == 50
        assert all(results)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_burst(self):
        """Zero burst should never allow acquisition."""
        bucket = TokenBucket(rate_per_minute=60, burst=0)
        assert bucket.try_acquire() is False

    def test_very_high_rate(self):
        """Very high rate should still work correctly."""
        bucket = TokenBucket(rate_per_minute=100000, burst=1000)
        assert bucket.try_acquire() is True

    def test_very_low_rate(self):
        """Very low rate should still work correctly."""
        bucket = TokenBucket(rate_per_minute=0.1, burst=1)
        assert bucket.try_acquire() is True
        # Next should fail (refill is very slow)
        assert bucket.try_acquire() is False

    def test_float_tokens(self):
        """Should handle fractional token counts correctly."""
        bucket = TokenBucket(rate_per_minute=30, burst=5)

        # Use some tokens
        bucket.try_acquire()
        bucket.try_acquire()

        # Wait a bit for partial refill
        time.sleep(0.1)

        # Should have partial tokens
        available = bucket.available_tokens
        assert available > 3.0  # Started with 5, used 2, got some refill

    def test_multiple_release(self):
        """Multiple releases should not corrupt state."""
        bucket = TokenBucket(rate_per_minute=60, burst=5)

        # Release many times
        for _ in range(100):
            bucket.release()

        # Should be capped at burst
        assert bucket.available_tokens <= 5.0


class TestTokenBucketStats:
    """Tests for TokenBucketStats dataclass."""

    def test_initial_values(self):
        """Should start with zero values."""
        stats = TokenBucketStats()
        assert stats.acquired == 0
        assert stats.rejected == 0
        assert stats.total_wait_ms == 0.0

    def test_record_acquired(self):
        """Should track acquired count and wait time."""
        stats = TokenBucketStats()
        stats.record_acquired(wait_ms=10.0)
        stats.record_acquired(wait_ms=20.0)

        assert stats.acquired == 2
        assert stats.total_wait_ms == 30.0

    def test_record_rejected(self):
        """Should track rejected count."""
        stats = TokenBucketStats()
        stats.record_rejected()
        stats.record_rejected()

        assert stats.rejected == 2

    def test_to_dict(self):
        """Should convert to dictionary correctly."""
        stats = TokenBucketStats()
        stats.record_acquired(wait_ms=100.0)
        stats.record_acquired(wait_ms=200.0)
        stats.record_rejected()

        d = stats.to_dict()
        assert d["acquired"] == 2
        assert d["rejected"] == 1
        assert d["total_wait_ms"] == 300.0
        assert d["avg_wait_ms"] == 150.0

    def test_avg_wait_zero_acquired(self):
        """Should handle zero acquired for avg calculation."""
        stats = TokenBucketStats()
        d = stats.to_dict()
        assert d["avg_wait_ms"] == 0.0
