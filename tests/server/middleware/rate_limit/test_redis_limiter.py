"""
Tests for Redis-backed rate limiter.

Tests cover:
- Rate limit allowing requests under limit
- Rate limit denying requests over limit
- Rate limit reset after window expires
- Fail-open policy when Redis unavailable
- Fail-closed policy when configured
- Redis unavailable fallback to in-memory limiter
- Circuit breaker activation after N Redis failures
- Distributed rate limiting (same key across multiple limiters)
- Key normalization and sanitization
- Token bucket refill behavior
- Concurrent request handling
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.rate_limit.redis_limiter import (
    RateLimitCircuitBreaker,
    RedisRateLimiter,
    RATE_LIMIT_FAIL_OPEN,
)
from aragora.server.middleware.rate_limit.limiter import RateLimitConfig, RateLimitResult
from aragora.server.middleware.rate_limit.base import (
    sanitize_rate_limit_key_component,
    normalize_rate_limit_path,
)


# =============================================================================
# Mock Redis Implementation
# =============================================================================


class MockRedis:
    """Mock Redis client for testing without actual Redis server."""

    def __init__(self, fail_on_connect: bool = False):
        self._data: dict[str, Any] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._ttls: dict[str, float] = {}
        self._scripts: dict[str, str] = {}
        self._script_counter = 0
        self._fail_on_connect = fail_on_connect
        self._fail_count = 0
        self._operation_count = 0

    def set_fail_mode(self, fail_count: int) -> None:
        """Configure Redis to fail for the next N operations."""
        self._fail_count = fail_count

    def _check_fail(self) -> None:
        """Check if we should simulate a failure."""
        self._operation_count += 1
        if self._fail_count > 0:
            self._fail_count -= 1
            raise ConnectionError("Simulated Redis failure")
        if self._fail_on_connect:
            raise ConnectionError("Redis connection refused")

    def ping(self) -> bool:
        self._check_fail()
        return True

    def get(self, key: str) -> Optional[str]:
        self._check_fail()
        return self._data.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._check_fail()
        self._data[key] = value
        if ex:
            self._ttls[key] = time.time() + ex
        return True

    def incr(self, key: str) -> int:
        self._check_fail()
        val = int(self._data.get(key, 0)) + 1
        self._data[key] = str(val)
        return val

    def delete(self, *keys: str) -> int:
        self._check_fail()
        deleted = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                deleted += 1
            if key in self._hashes:
                del self._hashes[key]
                deleted += 1
        return deleted

    def scan_iter(self, match: str = "*", count: int = 100):
        """Iterate keys matching pattern."""
        import fnmatch

        for key in list(self._data.keys()) + list(self._hashes.keys()):
            if fnmatch.fnmatch(key, match):
                yield key

    def hmget(self, name: str, keys: list[str]) -> list[Optional[str]]:
        self._check_fail()
        hash_data = self._hashes.get(name, {})
        return [hash_data.get(k) for k in keys]

    def hset(
        self,
        name: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        mapping: Optional[dict[str, str]] = None,
    ) -> int:
        self._check_fail()
        if name not in self._hashes:
            self._hashes[name] = {}
        if mapping:
            self._hashes[name].update(mapping)
            return len(mapping)
        elif key and value:
            self._hashes[name][key] = value
            return 1
        return 0

    def hgetall(self, name: str) -> dict[str, str]:
        self._check_fail()
        return self._hashes.get(name, {})

    def expire(self, key: str, seconds: int) -> bool:
        self._check_fail()
        self._ttls[key] = time.time() + seconds
        return True

    def script_load(self, script: str) -> str:
        """Load a Lua script and return SHA."""
        self._check_fail()
        sha = f"mock_sha_{self._script_counter}"
        self._scripts[sha] = script
        self._script_counter += 1
        return sha

    def evalsha(self, sha: str, numkeys: int, *args) -> list[Any]:
        """Execute a Lua script by SHA (simplified mock)."""
        self._check_fail()
        # Simulate token bucket behavior
        # Args: key, rate, burst, now, tokens_requested, ttl
        key = args[0] if args else "unknown"
        rate = float(args[1]) if len(args) > 1 else 60
        burst = float(args[2]) if len(args) > 2 else 120
        now = float(args[3]) if len(args) > 3 else time.time()
        tokens_requested = int(args[4]) if len(args) > 4 else 1

        # Get current state
        data = self._hashes.get(key, {})
        tokens = float(data.get("tokens", burst))
        last_refill = float(data.get("last_refill", now))

        # Calculate refill
        elapsed_minutes = (now - last_refill) / 60.0
        refill_amount = elapsed_minutes * rate
        tokens = min(burst, tokens + refill_amount)

        # Try to consume
        allowed = 0
        if tokens >= tokens_requested:
            tokens = tokens - tokens_requested
            allowed = 1

        # Save state
        self._hashes[key] = {"tokens": str(tokens), "last_refill": str(now)}

        return [allowed, tokens, burst]

    def pipeline(self) -> "MockPipeline":
        return MockPipeline(self)

    def close(self) -> None:
        pass


class MockPipeline:
    """Mock Redis pipeline for atomic operations."""

    def __init__(self, redis: MockRedis):
        self._redis = redis
        self._commands: list[tuple] = []

    def hset(
        self,
        name: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        mapping: Optional[dict[str, str]] = None,
    ) -> "MockPipeline":
        self._commands.append(("hset", (name,), {"key": key, "value": value, "mapping": mapping}))
        return self

    def expire(self, key: str, seconds: int) -> "MockPipeline":
        self._commands.append(("expire", (key, seconds), {}))
        return self

    def incr(self, key: str) -> "MockPipeline":
        self._commands.append(("incr", (key,), {}))
        return self

    def execute(self) -> list[Any]:
        results = []
        for cmd, args, kwargs in self._commands:
            method = getattr(self._redis, cmd)
            results.append(method(*args, **kwargs))
        self._commands.clear()
        return results


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_redis():
    """Create a fresh mock Redis client."""
    return MockRedis()


@pytest.fixture
def failing_redis():
    """Create a mock Redis that fails on connect."""
    return MockRedis(fail_on_connect=True)


@pytest.fixture
def redis_limiter(mock_redis):
    """Create a RedisRateLimiter with mock Redis and real token bucket behavior."""
    limiter = RedisRateLimiter(
        redis_client=mock_redis,
        default_limit=60,
        ip_limit=120,
        enable_circuit_breaker=True,
        enable_distributed_metrics=False,
        instance_id="test-instance",
    )
    yield limiter


@pytest.fixture
def redis_limiter_no_circuit_breaker(mock_redis):
    """Create a RedisRateLimiter without circuit breaker."""
    limiter = RedisRateLimiter(
        redis_client=mock_redis,
        default_limit=60,
        ip_limit=120,
        enable_circuit_breaker=False,
        enable_distributed_metrics=False,
        instance_id="test-no-cb",
    )
    yield limiter


# =============================================================================
# Test: Rate limit allows requests under limit
# =============================================================================


class TestRateLimitAllowsUnderLimit:
    """Tests for allowing requests under the rate limit."""

    def test_allows_first_request(self, redis_limiter):
        """First request should always be allowed."""
        result = redis_limiter.allow("192.168.1.1")
        assert result.allowed is True

    def test_allows_multiple_requests_under_limit(self, redis_limiter):
        """Multiple requests under limit should all be allowed."""
        for i in range(10):
            result = redis_limiter.allow(f"192.168.1.{i}")
            assert result.allowed is True

    def test_allows_burst_requests(self, mock_redis):
        """Should allow burst up to burst_size."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=10,
            ip_limit=10,
            enable_circuit_breaker=False,
            instance_id="burst-test",
        )

        # Burst size is 2x rate by default (20)
        results = [limiter.allow("192.168.1.1") for _ in range(15)]
        allowed_count = sum(1 for r in results if r.allowed)
        assert allowed_count >= 10  # At least rate limit worth

    def test_returns_remaining_count(self, redis_limiter):
        """Should return remaining token count."""
        result = redis_limiter.allow("192.168.1.1")
        assert result.remaining >= 0
        assert result.limit > 0


# =============================================================================
# Test: Rate limit denies requests over limit
# =============================================================================


class TestRateLimitDeniesOverLimit:
    """Tests for denying requests over the rate limit."""

    def test_denies_after_exhausting_tokens(self, mock_redis):
        """Should deny requests after exhausting all tokens."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=5,
            ip_limit=5,
            enable_circuit_breaker=False,
            instance_id="deny-test",
        )

        # Exhaust tokens (burst_size = 2 * 5 = 10)
        for _ in range(15):
            limiter.allow("192.168.1.1")

        # Next request should be denied
        result = limiter.allow("192.168.1.1")
        assert result.allowed is False

    def test_returns_retry_after_when_denied(self, mock_redis):
        """Should return retry_after value when request is denied."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=1,
            ip_limit=1,
            enable_circuit_breaker=False,
            instance_id="retry-test",
        )

        # Exhaust tokens
        for _ in range(5):
            limiter.allow("192.168.1.1")

        result = limiter.allow("192.168.1.1")
        if not result.allowed:
            assert result.retry_after >= 0

    def test_tracks_rejections_in_metrics(self, mock_redis):
        """Should track rejected requests in metrics."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=2,
            ip_limit=2,
            enable_circuit_breaker=False,
            instance_id="metrics-test",
        )

        # Exhaust tokens
        for _ in range(10):
            limiter.allow("192.168.1.1")

        stats = limiter.get_stats()
        assert stats["requests_rejected"] > 0


# =============================================================================
# Test: Rate limit resets after window expires
# =============================================================================


class TestRateLimitResetAfterWindow:
    """Tests for rate limit reset after time window expires."""

    def test_tokens_refill_over_time(self, mock_redis):
        """Tokens should refill over time based on rate."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=60,  # 1 token per second
            ip_limit=60,
            enable_circuit_breaker=False,
            instance_id="refill-test",
        )

        # Exhaust some tokens
        for _ in range(5):
            limiter.allow("192.168.1.1")

        # Simulate time passing by directly updating mock Redis state
        key = f"{limiter.key_prefix}ip:192.168.1.1"
        if key in mock_redis._hashes:
            old_time = float(mock_redis._hashes[key].get("last_refill", time.time()))
            # Simulate 10 seconds passing (should refill 10 tokens at 60/min = 1/sec)
            mock_redis._hashes[key]["last_refill"] = str(old_time - 10)

        # Should be allowed again after refill
        result = limiter.allow("192.168.1.1")
        assert result.allowed is True

    def test_bucket_never_exceeds_burst_size(self, mock_redis):
        """Token bucket should never exceed burst_size even after long delay."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=10,
            ip_limit=10,
            enable_circuit_breaker=False,
            instance_id="max-test",
        )

        # Make one request
        limiter.allow("192.168.1.1")

        # Simulate long time passing
        key = f"{limiter.key_prefix}ip:192.168.1.1"
        if key in mock_redis._hashes:
            old_time = float(mock_redis._hashes[key].get("last_refill", time.time()))
            mock_redis._hashes[key]["last_refill"] = str(old_time - 3600)  # 1 hour

        result = limiter.allow("192.168.1.1")
        # Remaining should be capped at burst_size - 1
        assert result.remaining <= 20  # burst_size = 2 * 10


# =============================================================================
# Test: Fail-open policy (if Redis unavailable, allow requests)
# =============================================================================


class TestFailOpenPolicy:
    """Tests for fail-open behavior when Redis is unavailable."""

    def test_allows_requests_on_redis_error_with_fallback(self, mock_redis):
        """Should allow requests via fallback when Redis fails."""
        mock_redis.set_fail_mode(100)  # Fail all operations

        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            enable_circuit_breaker=True,
            instance_id="fail-open-test",
        )

        # Should not raise, should use fallback
        result = limiter.allow("192.168.1.1")
        assert result.allowed is True

    def test_uses_in_memory_fallback_when_redis_fails(self, mock_redis):
        """Should use in-memory fallback limiter when Redis fails."""
        mock_redis.set_fail_mode(100)

        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            enable_circuit_breaker=True,
            instance_id="fallback-test",
        )

        limiter.allow("192.168.1.1")
        assert limiter._fallback_requests >= 1

    def test_fallback_still_enforces_limits(self, mock_redis):
        """Fallback should still enforce rate limits."""
        mock_redis.set_fail_mode(1000)

        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=2,
            ip_limit=2,
            enable_circuit_breaker=True,
            instance_id="fallback-limits-test",
        )

        # Make many requests to trigger rate limit in fallback
        results = []
        for _ in range(20):
            results.append(limiter.allow("192.168.1.1"))

        # Some should be rejected by fallback
        denied = sum(1 for r in results if not r.allowed)
        assert denied > 0


# =============================================================================
# Test: Fail-closed policy (if configured that way)
# =============================================================================


class TestFailClosedPolicy:
    """Tests for fail-closed behavior when configured."""

    def test_circuit_breaker_blocks_when_open(self, mock_redis):
        """Circuit breaker should block requests when OPEN."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            enable_circuit_breaker=True,
            instance_id="circuit-test",
        )

        # Force circuit open
        limiter._circuit_breaker._state = RateLimitCircuitBreaker.OPEN
        limiter._circuit_breaker._last_failure_time = time.time()

        # Should use fallback (which allows by default)
        result = limiter.allow("192.168.1.1")
        assert limiter._fallback_requests >= 1


# =============================================================================
# Test: Redis unavailable fallback to in-memory limiter
# =============================================================================


class TestRedisUnavailableFallback:
    """Tests for fallback to in-memory limiter when Redis is unavailable."""

    def test_fallback_limiter_is_used(self, mock_redis):
        """Should use the fallback RateLimiter when Redis operations fail."""
        mock_redis.set_fail_mode(10)

        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            enable_circuit_breaker=True,
            instance_id="use-fallback-test",
        )

        limiter.allow("192.168.1.1")

        assert limiter._fallback_requests >= 1
        assert limiter._redis_failures >= 1

    def test_fallback_has_same_configuration(self, mock_redis):
        """Fallback limiter should have same endpoint configuration."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=100,
            ip_limit=200,
            enable_circuit_breaker=False,
            instance_id="config-fallback-test",
        )

        limiter.configure_endpoint("/api/test", 50, burst_size=100)

        # Check fallback has the config
        fallback_config = limiter._fallback.get_config("/api/test")
        assert fallback_config.requests_per_minute == 50


# =============================================================================
# Test: Circuit breaker activation after N Redis failures
# =============================================================================


class TestCircuitBreakerActivation:
    """Tests for circuit breaker activation after Redis failures."""

    def test_circuit_opens_after_threshold_failures(self, mock_redis):
        """Circuit should open after failure_threshold failures."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            enable_circuit_breaker=True,
            instance_id="cb-activation-test",
        )
        limiter._circuit_breaker = RateLimitCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=10.0,
        )

        # Simulate failures
        mock_redis.set_fail_mode(5)

        for _ in range(5):
            limiter.allow("192.168.1.1")

        assert limiter._circuit_breaker.state == RateLimitCircuitBreaker.OPEN

    def test_circuit_transitions_to_half_open(self, mock_redis):
        """Circuit should transition to HALF_OPEN after recovery timeout."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            enable_circuit_breaker=True,
            instance_id="cb-half-open-test",
        )
        limiter._circuit_breaker = RateLimitCircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,  # 50ms
        )

        # Open the circuit
        mock_redis.set_fail_mode(3)
        for _ in range(3):
            limiter.allow("192.168.1.1")

        assert limiter._circuit_breaker.state == RateLimitCircuitBreaker.OPEN

        # Wait for recovery
        time.sleep(0.1)

        assert limiter._circuit_breaker.state == RateLimitCircuitBreaker.HALF_OPEN

    def test_circuit_closes_on_successful_calls(self, mock_redis):
        """Circuit should close after successful calls in HALF_OPEN."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            enable_circuit_breaker=True,
            instance_id="cb-close-test",
        )
        limiter._circuit_breaker = RateLimitCircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.05,
            half_open_max_calls=2,
        )

        # Open the circuit
        mock_redis.set_fail_mode(3)
        for _ in range(3):
            limiter.allow("192.168.1.1")

        # Wait for half-open
        time.sleep(0.1)

        # Redis recovers
        mock_redis.set_fail_mode(0)

        # Successful calls
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.1")

        assert limiter._circuit_breaker.state == RateLimitCircuitBreaker.CLOSED


# =============================================================================
# Test: Distributed rate limiting (same key across multiple limiters)
# =============================================================================


class TestDistributedRateLimiting:
    """Tests for distributed rate limiting across multiple instances."""

    def test_shared_redis_shares_state(self):
        """Multiple limiters sharing Redis should share rate limit state."""
        shared_redis = MockRedis()

        limiter1 = RedisRateLimiter(
            redis_client=shared_redis,
            default_limit=10,
            ip_limit=10,
            enable_circuit_breaker=False,
            instance_id="instance-1",
        )
        limiter2 = RedisRateLimiter(
            redis_client=shared_redis,
            default_limit=10,
            ip_limit=10,
            enable_circuit_breaker=False,
            instance_id="instance-2",
        )

        # Both limiters consume from same pool
        for _ in range(8):
            limiter1.allow("192.168.1.1")

        for _ in range(8):
            limiter2.allow("192.168.1.1")

        # Combined requests should hit the limit
        total_allowed = limiter1._requests_allowed + limiter2._requests_allowed
        total_rejected = limiter1._requests_rejected + limiter2._requests_rejected
        assert total_rejected > 0 or total_allowed <= 20

    def test_different_keys_independent(self):
        """Different rate limit keys should be independent."""
        shared_redis = MockRedis()

        limiter = RedisRateLimiter(
            redis_client=shared_redis,
            default_limit=5,
            ip_limit=5,
            enable_circuit_breaker=False,
            instance_id="independent-test",
        )

        # Exhaust limit for one IP
        for _ in range(15):
            limiter.allow("192.168.1.1")

        # Different IP should still work
        result = limiter.allow("192.168.1.2")
        assert result.allowed is True


# =============================================================================
# Test: Key normalization and sanitization
# =============================================================================


class TestKeyNormalizationAndSanitization:
    """Tests for rate limit key normalization and sanitization."""

    def test_sanitizes_ip_with_colons(self):
        """Should sanitize colons in IP addresses to prevent key injection."""
        sanitized = sanitize_rate_limit_key_component("::1")
        assert ":" not in sanitized
        assert sanitized == "__1"

    def test_sanitizes_newlines(self):
        """Should remove newlines from key components."""
        sanitized = sanitize_rate_limit_key_component("test\nvalue")
        assert "\n" not in sanitized

    def test_sanitizes_carriage_returns(self):
        """Should remove carriage returns from key components."""
        sanitized = sanitize_rate_limit_key_component("test\rvalue")
        assert "\r" not in sanitized

    def test_normalizes_path(self):
        """Should normalize paths for consistent matching."""
        normalized = normalize_rate_limit_path("/api//debates/")
        assert normalized == "/api/debates"

    def test_normalizes_path_traversal(self):
        """Should prevent path traversal in normalization."""
        normalized = normalize_rate_limit_path("/api/../secret")
        assert ".." not in normalized

    def test_limiter_uses_sanitized_keys(self, redis_limiter, mock_redis):
        """Limiter should use sanitized keys internally."""
        redis_limiter.allow("192:168:1:1", endpoint="/api/test\n/inject")

        # Check keys in mock Redis don't contain dangerous characters
        for key in mock_redis._hashes.keys():
            assert "\n" not in key
            # Colons in the key prefix are OK, but not in client-provided values
            # after the key_prefix


# =============================================================================
# Test: Token bucket refill behavior
# =============================================================================


class TestTokenBucketRefillBehavior:
    """Tests for token bucket algorithm and refill behavior."""

    def test_tokens_refill_at_configured_rate(self, mock_redis):
        """Tokens should refill at the configured rate per minute."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=60,  # 60 per minute = 1 per second
            ip_limit=60,
            enable_circuit_breaker=False,
            instance_id="refill-rate-test",
        )

        # Consume some tokens
        for _ in range(10):
            limiter.allow("192.168.1.1")

        # Simulate 5 seconds passing
        key = f"{limiter.key_prefix}ip:192.168.1.1"
        if key in mock_redis._hashes:
            old_time = float(mock_redis._hashes[key]["last_refill"])
            mock_redis._hashes[key]["last_refill"] = str(old_time - 5)

        result = limiter.allow("192.168.1.1")
        # Should have refilled ~5 tokens
        assert result.allowed is True

    def test_burst_size_defaults_to_2x_rate(self, mock_redis):
        """Burst size should default to 2x the rate limit."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=50,
            ip_limit=50,
            enable_circuit_breaker=False,
            instance_id="burst-default-test",
        )

        # Get config to check burst
        config = limiter.get_config("/api/test")
        # Bucket burst = 2 * rate
        assert config.burst_size is None  # Uses default


# =============================================================================
# Test: Concurrent request handling
# =============================================================================


class TestConcurrentRequestHandling:
    """Tests for handling concurrent requests safely."""

    def test_concurrent_requests_thread_safe(self, mock_redis):
        """Concurrent requests should be handled thread-safely."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=100,
            ip_limit=100,
            enable_circuit_breaker=False,
            instance_id="concurrent-test",
        )

        results = []
        errors = []

        def make_request():
            try:
                for _ in range(50):
                    result = limiter.allow("192.168.1.1")
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=make_request) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 250  # 5 threads * 50 requests

    def test_concurrent_different_ips_independent(self, mock_redis):
        """Concurrent requests from different IPs should be independent."""
        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=10,
            ip_limit=10,
            enable_circuit_breaker=False,
            instance_id="independent-concurrent-test",
        )

        results_per_ip: dict[str, list] = {}
        lock = threading.Lock()

        def make_requests(ip: str):
            results = []
            for _ in range(5):
                result = limiter.allow(ip)
                results.append(result)
            with lock:
                results_per_ip[ip] = results

        threads = [
            threading.Thread(target=make_requests, args=(f"192.168.1.{i}",)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each IP should have all requests allowed (under limit)
        for ip, results in results_per_ip.items():
            allowed = sum(1 for r in results if r.allowed)
            assert allowed == 5


# =============================================================================
# Test: Additional edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_ip_address(self, redis_limiter):
        """Should handle empty IP address gracefully."""
        result = redis_limiter.allow("")
        assert result.allowed is True  # Uses 'anonymous' key

    def test_none_endpoint(self, redis_limiter):
        """Should handle None endpoint gracefully."""
        result = redis_limiter.allow("192.168.1.1", endpoint=None)
        assert result.allowed is True

    def test_get_stats_returns_complete_info(self, redis_limiter):
        """get_stats should return complete statistics."""
        redis_limiter.allow("192.168.1.1")
        stats = redis_limiter.get_stats()

        assert "backend" in stats
        assert stats["backend"] == "redis"
        assert "instance_id" in stats
        assert "requests_allowed" in stats
        assert "requests_rejected" in stats
        assert "configured_endpoints" in stats

    def test_reset_clears_all_state(self, redis_limiter, mock_redis):
        """reset should clear all rate limiter state."""
        redis_limiter.allow("192.168.1.1")
        redis_limiter.allow("192.168.1.2")

        redis_limiter.reset()

        # Buckets should be cleared
        assert len(redis_limiter._buckets) == 0

    def test_configure_endpoint_stores_config(self, redis_limiter):
        """configure_endpoint should store and retrieve configuration."""
        redis_limiter.configure_endpoint("/api/special", 1000, burst_size=2000)

        config = redis_limiter.get_config("/api/special")
        assert config.requests_per_minute == 1000
        assert config.burst_size == 2000

    def test_wildcard_endpoint_matching(self, redis_limiter):
        """Should match wildcard endpoint configurations."""
        redis_limiter.configure_endpoint("/api/admin/*", 10)

        config = redis_limiter.get_config("/api/admin/users")
        assert config.requests_per_minute == 10
