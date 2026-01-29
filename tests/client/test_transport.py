"""
Tests for HTTP transport utilities.

Tests cover:
- RetryConfig exponential backoff calculation
- RetryConfig jitter application
- RetryConfig max backoff cap
- RateLimiter request throttling (sync and async)
- RateLimiter disabled when rps=0
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from aragora.client.transport import RateLimiter, RetryConfig


# ============================================================================
# RetryConfig Tests
# ============================================================================


class TestRetryConfigDefaults:
    """Tests for RetryConfig default values."""

    def test_default_max_retries(self):
        """Test default max_retries is 3."""
        config = RetryConfig()
        assert config.max_retries == 3

    def test_default_backoff_factor(self):
        """Test default backoff_factor is 0.5."""
        config = RetryConfig()
        assert config.backoff_factor == 0.5

    def test_default_max_backoff(self):
        """Test default max_backoff is 30.0."""
        config = RetryConfig()
        assert config.max_backoff == 30.0

    def test_default_retry_statuses(self):
        """Test default retry_statuses includes 429 and 5xx codes."""
        config = RetryConfig()
        assert 429 in config.retry_statuses
        assert 500 in config.retry_statuses
        assert 502 in config.retry_statuses
        assert 503 in config.retry_statuses
        assert 504 in config.retry_statuses

    def test_default_jitter_enabled(self):
        """Test default jitter is True."""
        config = RetryConfig()
        assert config.jitter is True


class TestRetryConfigCustomization:
    """Tests for RetryConfig customization."""

    def test_custom_max_retries(self):
        """Test custom max_retries."""
        config = RetryConfig(max_retries=5)
        assert config.max_retries == 5

    def test_custom_backoff_factor(self):
        """Test custom backoff_factor."""
        config = RetryConfig(backoff_factor=1.0)
        assert config.backoff_factor == 1.0

    def test_custom_max_backoff(self):
        """Test custom max_backoff."""
        config = RetryConfig(max_backoff=60.0)
        assert config.max_backoff == 60.0

    def test_custom_retry_statuses(self):
        """Test custom retry_statuses."""
        config = RetryConfig(retry_statuses=(429, 503))
        assert config.retry_statuses == (429, 503)
        assert 500 not in config.retry_statuses

    def test_jitter_disabled(self):
        """Test jitter can be disabled."""
        config = RetryConfig(jitter=False)
        assert config.jitter is False


class TestRetryConfigGetDelay:
    """Tests for RetryConfig.get_delay() exponential backoff."""

    def test_first_attempt_delay(self):
        """Test delay for first retry attempt (attempt=0)."""
        config = RetryConfig(backoff_factor=0.5, jitter=False)
        # delay = 0.5 * (2^0) = 0.5
        delay = config.get_delay(0)
        assert delay == 0.5

    def test_second_attempt_delay(self):
        """Test delay for second retry attempt (attempt=1)."""
        config = RetryConfig(backoff_factor=0.5, jitter=False)
        # delay = 0.5 * (2^1) = 1.0
        delay = config.get_delay(1)
        assert delay == 1.0

    def test_third_attempt_delay(self):
        """Test delay for third retry attempt (attempt=2)."""
        config = RetryConfig(backoff_factor=0.5, jitter=False)
        # delay = 0.5 * (2^2) = 2.0
        delay = config.get_delay(2)
        assert delay == 2.0

    def test_exponential_growth(self):
        """Test delay grows exponentially."""
        config = RetryConfig(backoff_factor=1.0, jitter=False, max_backoff=1000.0)
        delays = [config.get_delay(i) for i in range(5)]
        # Expected: 1, 2, 4, 8, 16
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_max_backoff_cap(self):
        """Test delay is capped at max_backoff."""
        config = RetryConfig(backoff_factor=0.5, max_backoff=5.0, jitter=False)
        # attempt=10: 0.5 * 2^10 = 512, but capped at 5.0
        delay = config.get_delay(10)
        assert delay == 5.0

    def test_max_backoff_exactly_reached(self):
        """Test delay when exponential equals max_backoff."""
        config = RetryConfig(backoff_factor=1.0, max_backoff=8.0, jitter=False)
        # attempt=3: 1 * 2^3 = 8.0, exactly at max
        delay = config.get_delay(3)
        assert delay == 8.0
        # attempt=4: 1 * 2^4 = 16.0, capped at 8.0
        delay = config.get_delay(4)
        assert delay == 8.0

    def test_jitter_produces_range(self):
        """Test jitter produces delays in range [0.5*base, 1.5*base)."""
        config = RetryConfig(backoff_factor=1.0, jitter=True, max_backoff=100.0)
        # Base delay for attempt=2 is 4.0
        # With jitter: delay = 4.0 * (0.5 + random()) -> [2.0, 6.0)
        delays = [config.get_delay(2) for _ in range(100)]

        # All delays should be in range [2.0, 6.0)
        assert all(2.0 <= d < 6.0 for d in delays)

        # Delays should vary (not all the same)
        assert len(set(delays)) > 1

    def test_jitter_randomness(self):
        """Test jitter produces different values on each call."""
        config = RetryConfig(backoff_factor=1.0, jitter=True)
        delays = [config.get_delay(0) for _ in range(10)]
        # Should have some variation
        unique_delays = set(delays)
        assert len(unique_delays) > 1, "Jitter should produce varying delays"

    def test_zero_attempt(self):
        """Test delay for attempt=0."""
        config = RetryConfig(backoff_factor=2.0, jitter=False)
        # delay = 2.0 * 2^0 = 2.0
        delay = config.get_delay(0)
        assert delay == 2.0


# ============================================================================
# RateLimiter Tests
# ============================================================================


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_init_with_positive_rps(self):
        """Test initialization with positive RPS."""
        limiter = RateLimiter(10.0)
        assert limiter.rps == 10.0
        assert limiter.min_interval == 0.1  # 1/10

    def test_init_with_fractional_rps(self):
        """Test initialization with fractional RPS."""
        limiter = RateLimiter(0.5)
        assert limiter.rps == 0.5
        assert limiter.min_interval == 2.0  # 1/0.5

    def test_init_with_zero_rps(self):
        """Test initialization with zero RPS (disabled)."""
        limiter = RateLimiter(0)
        assert limiter.rps == 0
        assert limiter.min_interval == 0

    def test_init_with_negative_rps(self):
        """Test initialization with negative RPS (disabled)."""
        limiter = RateLimiter(-1)
        assert limiter.rps == -1
        assert limiter.min_interval == 0

    def test_last_request_initialized_to_zero(self):
        """Test _last_request is initialized to 0."""
        limiter = RateLimiter(10)
        assert limiter._last_request == 0


class TestRateLimiterWaitSync:
    """Tests for RateLimiter.wait() synchronous method."""

    def test_wait_disabled_when_rps_zero(self):
        """Test wait() returns immediately when rps=0."""
        limiter = RateLimiter(0)
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        assert elapsed < 0.01  # Should be nearly instant

    def test_wait_disabled_when_rps_negative(self):
        """Test wait() returns immediately when rps<0."""
        limiter = RateLimiter(-5)
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        assert elapsed < 0.01  # Should be nearly instant

    def test_first_request_immediate(self):
        """Test first request goes through immediately."""
        limiter = RateLimiter(10)  # 100ms interval
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        # First request should be nearly instant (no previous request)
        assert elapsed < 0.05

    def test_sequential_requests_throttled(self):
        """Test sequential requests are throttled by min_interval."""
        limiter = RateLimiter(10)  # 100ms interval

        limiter.wait()  # First request
        start = time.time()
        limiter.wait()  # Second request should wait
        elapsed = time.time() - start

        # Should have waited approximately 100ms
        assert 0.08 <= elapsed <= 0.15

    def test_updates_last_request_time(self):
        """Test wait() updates _last_request timestamp."""
        limiter = RateLimiter(10)
        assert limiter._last_request == 0

        limiter.wait()

        assert limiter._last_request > 0

    def test_multiple_rapid_requests(self):
        """Test multiple rapid requests are properly throttled."""
        limiter = RateLimiter(20)  # 50ms interval

        times = []
        for _ in range(4):
            limiter.wait()
            times.append(time.time())

        # Check intervals between requests
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]

        # All intervals should be approximately 50ms (within tolerance)
        for interval in intervals:
            assert 0.04 <= interval <= 0.08


class TestRateLimiterWaitAsync:
    """Tests for RateLimiter.wait_async() asynchronous method."""

    @pytest.mark.asyncio
    async def test_wait_async_disabled_when_rps_zero(self):
        """Test wait_async() returns immediately when rps=0."""
        limiter = RateLimiter(0)
        start = asyncio.get_event_loop().time()
        await limiter.wait_async()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.01

    @pytest.mark.asyncio
    async def test_wait_async_disabled_when_rps_negative(self):
        """Test wait_async() returns immediately when rps<0."""
        limiter = RateLimiter(-5)
        start = asyncio.get_event_loop().time()
        await limiter.wait_async()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.01

    @pytest.mark.asyncio
    async def test_first_async_request_immediate(self):
        """Test first async request goes through immediately."""
        limiter = RateLimiter(10)  # 100ms interval
        start = asyncio.get_event_loop().time()
        await limiter.wait_async()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_sequential_async_requests_throttled(self):
        """Test sequential async requests are throttled."""
        limiter = RateLimiter(10)  # 100ms interval

        await limiter.wait_async()  # First request
        start = asyncio.get_event_loop().time()
        await limiter.wait_async()  # Second request should wait
        elapsed = asyncio.get_event_loop().time() - start

        assert 0.08 <= elapsed <= 0.15

    @pytest.mark.asyncio
    async def test_async_updates_last_request_time(self):
        """Test wait_async() updates _last_request timestamp."""
        limiter = RateLimiter(10)
        assert limiter._last_request == 0

        await limiter.wait_async()

        assert limiter._last_request > 0

    @pytest.mark.asyncio
    async def test_async_uses_asyncio_sleep(self):
        """Test wait_async() uses asyncio.sleep (non-blocking)."""
        limiter = RateLimiter(5)  # 200ms interval

        await limiter.wait_async()

        # This should use asyncio.sleep, allowing other tasks to run
        async def concurrent_task():
            return "completed"

        # Both should complete without blocking each other
        start = asyncio.get_event_loop().time()
        result, _ = await asyncio.gather(
            concurrent_task(),
            limiter.wait_async(),
        )
        elapsed = asyncio.get_event_loop().time() - start

        assert result == "completed"
        assert elapsed < 0.3  # Should complete within reasonable time


class TestRateLimiterEdgeCases:
    """Tests for RateLimiter edge cases."""

    def test_high_rps_minimal_delay(self):
        """Test high RPS results in minimal delay."""
        limiter = RateLimiter(1000)  # 1ms interval
        assert limiter.min_interval == 0.001

        start = time.time()
        for _ in range(5):
            limiter.wait()
        elapsed = time.time() - start

        # Should complete quickly with 1ms intervals
        assert elapsed < 0.1

    def test_very_low_rps_long_delay(self):
        """Test very low RPS results in long delay."""
        limiter = RateLimiter(0.5)  # 2s interval
        assert limiter.min_interval == 2.0

        # First request should be immediate
        start = time.time()
        limiter.wait()
        elapsed = time.time() - start
        assert elapsed < 0.1

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_sync_and_async_update_last_request(self):
        """Test that both sync and async methods update _last_request.

        Note: The current implementation uses different time sources
        (time.time() for sync vs asyncio loop time for async).
        This test verifies both methods update the timestamp.
        """
        # Test sync method updates _last_request
        sync_limiter = RateLimiter(10)
        assert sync_limiter._last_request == 0
        sync_limiter.wait()
        assert sync_limiter._last_request > 0

        # Test async method updates _last_request (separate limiter to avoid timing issues)
        async_limiter = RateLimiter(10)
        assert async_limiter._last_request == 0
        await async_limiter.wait_async()
        assert async_limiter._last_request > 0


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestRetryConfigWithRateLimiter:
    """Tests combining RetryConfig and RateLimiter patterns."""

    def test_typical_production_config(self):
        """Test typical production configuration."""
        retry = RetryConfig(
            max_retries=3,
            backoff_factor=0.5,
            max_backoff=30.0,
            jitter=True,
        )
        limiter = RateLimiter(10)  # 10 RPS

        assert retry.max_retries == 3
        assert limiter.min_interval == 0.1

    def test_aggressive_retry_config(self):
        """Test aggressive retry configuration."""
        retry = RetryConfig(
            max_retries=5,
            backoff_factor=0.1,
            max_backoff=5.0,
            jitter=False,
        )

        # Delays: 0.1, 0.2, 0.4, 0.8, 1.6
        delays = [retry.get_delay(i) for i in range(5)]
        assert delays == [0.1, 0.2, 0.4, 0.8, 1.6]

    def test_conservative_retry_config(self):
        """Test conservative retry configuration."""
        retry = RetryConfig(
            max_retries=2,
            backoff_factor=2.0,
            max_backoff=60.0,
            jitter=False,
        )

        # Delays: 2.0, 4.0
        delays = [retry.get_delay(i) for i in range(2)]
        assert delays == [2.0, 4.0]
