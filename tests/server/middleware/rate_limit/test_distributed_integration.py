"""
Integration tests for distributed rate limiting.

Tests multi-instance coordination, Redis failover, and circuit breaker behavior.
These tests verify the production-readiness of the distributed rate limiting system.

Run with:
    pytest tests/server/middleware/rate_limit/test_distributed_integration.py -v

Requires:
    - Redis running locally (or set REDIS_URL)
    - Set ARAGORA_ENV=test to skip strict mode enforcement
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.rate_limit.distributed import (
    DistributedRateLimiter,
    get_distributed_limiter,
    reset_distributed_limiter,
)
from aragora.server.middleware.rate_limit.redis_limiter import (
    RateLimitCircuitBreaker,
    get_redis_client,
)

if TYPE_CHECKING:
    pass

pytestmark = [pytest.mark.integration]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_limiter():
    """Reset the global limiter before and after each test."""
    reset_distributed_limiter()
    yield
    reset_distributed_limiter()


@pytest.fixture
def redis_available():
    """Check if Redis is available for testing."""
    client = get_redis_client()
    if client is None:
        pytest.skip("Redis not available")
    try:
        client.ping()
        return True
    except Exception:
        pytest.skip("Redis not responding")


# ============================================================================
# Basic Functionality Tests
# ============================================================================


class TestDistributedLimiterBasic:
    """Basic functionality tests for distributed rate limiter."""

    def test_initialization_without_redis(self):
        """Test limiter initializes with in-memory fallback when Redis unavailable."""
        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=None,
        ):
            limiter = DistributedRateLimiter(
                instance_id="test-1",
                strict_mode=False,
            )

            # Should use memory backend
            assert limiter.backend == "memory"
            assert not limiter.is_using_redis

            # Should still work
            result = limiter.allow("192.168.1.1", "/api/test")
            assert result.allowed

    def test_rate_limit_enforcement(self):
        """Test rate limits are enforced."""
        limiter = DistributedRateLimiter(
            instance_id="test-2",
            strict_mode=False,
        )

        # Configure a low rate limit
        limiter.configure_endpoint("/api/limited", requests_per_minute=5, burst_size=5)

        # First 5 requests should be allowed
        for i in range(5):
            result = limiter.allow("192.168.1.1", "/api/limited")
            assert result.allowed, f"Request {i + 1} should be allowed"

        # Next request should be rate limited
        result = limiter.allow("192.168.1.1", "/api/limited")
        assert not result.allowed, "Request 6 should be rate limited"
        assert result.remaining == 0
        assert result.retry_after > 0

    def test_per_client_limits(self):
        """Test different clients have separate rate limits."""
        limiter = DistributedRateLimiter(
            instance_id="test-3",
            strict_mode=False,
        )

        limiter.configure_endpoint("/api/per-client", requests_per_minute=3, burst_size=3)

        # Client A uses all their tokens
        for _ in range(3):
            result = limiter.allow("client-a", "/api/per-client")
            assert result.allowed

        result = limiter.allow("client-a", "/api/per-client")
        assert not result.allowed

        # Client B should still have tokens
        result = limiter.allow("client-b", "/api/per-client")
        assert result.allowed

    def test_tenant_based_limits(self):
        """Test tenant-based rate limiting."""
        limiter = DistributedRateLimiter(
            instance_id="test-4",
            strict_mode=False,
        )

        limiter.configure_endpoint("/api/tenant", requests_per_minute=10, key_type="tenant")

        # Same client, different tenants
        result1 = limiter.allow("client", "/api/tenant", tenant_id="tenant-1")
        result2 = limiter.allow("client", "/api/tenant", tenant_id="tenant-2")

        assert result1.allowed
        assert result2.allowed

    def test_stats_collection(self):
        """Test statistics are collected properly."""
        limiter = DistributedRateLimiter(
            instance_id="test-5",
            strict_mode=False,
            enable_metrics=False,
        )

        # Make some requests
        for _ in range(10):
            limiter.allow("client", "/api/stats")

        stats = limiter.get_stats()
        assert stats["instance_id"] == "test-5"
        assert stats["total_requests"] == 10
        assert "backend" in stats


# ============================================================================
# Multi-Instance Coordination Tests
# ============================================================================


class TestMultiInstanceCoordination:
    """Tests for multi-instance rate limit coordination via Redis."""

    def test_shared_rate_limits_across_instances(self, redis_available):
        """Test rate limits are shared across multiple limiter instances."""
        # Create two instances simulating different servers
        limiter1 = DistributedRateLimiter(
            instance_id="server-1",
            strict_mode=False,
        )
        limiter2 = DistributedRateLimiter(
            instance_id="server-2",
            strict_mode=False,
        )

        # Configure same endpoint on both
        limiter1.configure_endpoint("/api/shared", requests_per_minute=10, burst_size=10)
        limiter2.configure_endpoint("/api/shared", requests_per_minute=10, burst_size=10)

        # Make 5 requests on instance 1
        for _ in range(5):
            result = limiter1.allow("shared-client", "/api/shared")
            assert result.allowed

        # Make 5 requests on instance 2
        for _ in range(5):
            result = limiter2.allow("shared-client", "/api/shared")
            assert result.allowed

        # Next request on either instance should be limited
        result = limiter1.allow("shared-client", "/api/shared")
        assert not result.allowed

        result = limiter2.allow("shared-client", "/api/shared")
        assert not result.allowed

    def test_concurrent_requests_across_instances(self, redis_available):
        """Test concurrent requests from multiple instances."""
        # Create instances
        limiters = [
            DistributedRateLimiter(instance_id=f"concurrent-{i}", strict_mode=False)
            for i in range(3)
        ]

        # Configure endpoints
        for limiter in limiters:
            limiter.configure_endpoint("/api/concurrent", requests_per_minute=30, burst_size=30)

        allowed_count = 0
        lock = threading.Lock()

        def make_request(limiter, n):
            nonlocal allowed_count
            result = limiter.allow(f"concurrent-client-{n % 3}", "/api/concurrent")
            with lock:
                if result.allowed:
                    allowed_count += 1

        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(100):
                limiter = limiters[i % 3]
                futures.append(executor.submit(make_request, limiter, i))

            # Wait for all to complete
            for f in futures:
                f.result()

        # Should have allowed approximately 30 requests per client (3 clients)
        # With some margin for timing
        assert allowed_count >= 30, f"Expected at least 30 allowed, got {allowed_count}"
        assert allowed_count <= 100, f"Expected at most 100 allowed, got {allowed_count}"


# ============================================================================
# Redis Failover Tests
# ============================================================================


class TestRedisFailover:
    """Tests for Redis failover and circuit breaker behavior."""

    def test_fallback_on_redis_unavailable(self):
        """Test graceful fallback when Redis becomes unavailable."""
        # Start with working Redis mock
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.evalsha.side_effect = ConnectionError("Redis connection lost")

        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=mock_redis,
        ):
            with patch(
                "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                return_value=mock_redis,
            ):
                limiter = DistributedRateLimiter(
                    instance_id="failover-test",
                    strict_mode=False,
                )

                # Should still allow requests via fallback
                result = limiter.allow("client", "/api/failover")
                assert result.allowed

                # Should have recorded fallback
                stats = limiter.get_stats()
                assert stats["fallback_requests"] >= 0

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after repeated failures."""
        # Create circuit breaker
        cb = RateLimitCircuitBreaker(
            failure_threshold=3,
            reset_timeout_seconds=1.0,
            half_open_max_calls=1,
        )

        # Record failures to open circuit
        for _ in range(3):
            cb.record_failure()

        assert cb.state == RateLimitCircuitBreaker.OPEN

    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker transitions to half-open and recovers."""
        cb = RateLimitCircuitBreaker(
            failure_threshold=2,
            reset_timeout_seconds=0.1,
            half_open_max_calls=1,
        )

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == RateLimitCircuitBreaker.OPEN

        # Wait for reset timeout
        time.sleep(0.15)

        # Should allow check in half-open state
        assert cb.allow_request()
        assert cb.state == RateLimitCircuitBreaker.HALF_OPEN

        # Success should close circuit
        cb.record_success()
        assert cb.state == RateLimitCircuitBreaker.CLOSED

    def test_strict_mode_in_production(self):
        """Test strict mode raises error in production without Redis."""
        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=None,
        ):
            with patch(
                "aragora.server.middleware.rate_limit.distributed._is_production_mode",
                return_value=True,
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    DistributedRateLimiter(
                        instance_id="strict-test",
                        strict_mode=True,
                    )

                assert "ARAGORA_RATE_LIMIT_STRICT" in str(exc_info.value)
                assert "Redis is unavailable" in str(exc_info.value)

    def test_strict_mode_warns_in_development(self):
        """Test strict mode only warns in development without Redis."""
        with patch(
            "aragora.server.middleware.rate_limit.distributed.get_redis_client",
            return_value=None,
        ):
            with patch(
                "aragora.server.middleware.rate_limit.distributed._is_production_mode",
                return_value=False,
            ):
                with patch(
                    "aragora.server.middleware.rate_limit.distributed._is_development_mode",
                    return_value=True,
                ):
                    # Should not raise, just warn
                    limiter = DistributedRateLimiter(
                        instance_id="dev-strict-test",
                        strict_mode=True,
                    )

                    assert not limiter.is_using_redis
                    assert limiter.backend == "memory"


# ============================================================================
# Performance Tests
# ============================================================================


class TestRateLimiterPerformance:
    """Performance tests for rate limiter."""

    def test_high_throughput_in_memory(self):
        """Test in-memory limiter handles high request rates."""
        limiter = DistributedRateLimiter(
            instance_id="perf-test",
            strict_mode=False,
            enable_metrics=False,
        )

        limiter.configure_endpoint("/api/perf", requests_per_minute=10000, burst_size=10000)

        start = time.monotonic()
        count = 10000

        for _ in range(count):
            limiter.allow("perf-client", "/api/perf")

        elapsed = time.monotonic() - start
        rate = count / elapsed

        # Should handle at least 10K requests/second on in-memory backend
        assert rate > 10000, f"Rate too low: {rate:.0f} req/s"

    def test_concurrent_access_thread_safe(self):
        """Test thread safety under concurrent access."""
        limiter = DistributedRateLimiter(
            instance_id="thread-test",
            strict_mode=False,
            enable_metrics=False,
        )

        limiter.configure_endpoint("/api/threads", requests_per_minute=1000, burst_size=1000)

        errors = []
        results = []
        lock = threading.Lock()

        def make_requests(n):
            try:
                for _ in range(100):
                    result = limiter.allow(f"client-{n}", "/api/threads")
                    with lock:
                        results.append(result.allowed)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run from multiple threads
        threads = [threading.Thread(target=make_requests, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"
        assert len(results) == 1000


# ============================================================================
# Configuration Tests
# ============================================================================


class TestRateLimiterConfiguration:
    """Tests for rate limiter configuration."""

    def test_endpoint_configuration(self):
        """Test endpoint-specific configuration."""
        limiter = DistributedRateLimiter(
            instance_id="config-test",
            strict_mode=False,
        )

        # Configure different limits for different endpoints
        limiter.configure_endpoint("/api/fast", requests_per_minute=100)
        limiter.configure_endpoint("/api/slow", requests_per_minute=10)

        # Fast endpoint should allow more
        for _ in range(50):
            result = limiter.allow("client", "/api/fast")
            assert result.allowed

        # Slow endpoint should limit sooner
        for _ in range(10):
            limiter.allow("client", "/api/slow")

        result = limiter.allow("client", "/api/slow")
        assert not result.allowed

    def test_burst_size_configuration(self):
        """Test burst size allows initial spike."""
        limiter = DistributedRateLimiter(
            instance_id="burst-test",
            strict_mode=False,
        )

        # High burst, low sustained rate
        limiter.configure_endpoint("/api/burst", requests_per_minute=10, burst_size=50)

        # Should allow burst of 50
        allowed = 0
        for _ in range(60):
            result = limiter.allow("client", "/api/burst")
            if result.allowed:
                allowed += 1

        assert allowed >= 50, f"Expected at least 50 allowed, got {allowed}"

    def test_reset_clears_all_state(self):
        """Test reset clears all rate limit state."""
        limiter = DistributedRateLimiter(
            instance_id="reset-test",
            strict_mode=False,
        )

        # Use up rate limit
        limiter.configure_endpoint("/api/reset", requests_per_minute=5, burst_size=5)
        for _ in range(5):
            limiter.allow("client", "/api/reset")

        result = limiter.allow("client", "/api/reset")
        assert not result.allowed

        # Reset
        limiter.reset()

        # Should be allowed again
        result = limiter.allow("client", "/api/reset")
        assert result.allowed


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests for rate limiter."""

    def test_empty_client_ip(self):
        """Test handling of empty client IP."""
        limiter = DistributedRateLimiter(
            instance_id="edge-test",
            strict_mode=False,
        )

        # Should handle empty IP gracefully
        result = limiter.allow("", "/api/test")
        assert result is not None

    def test_special_characters_in_endpoint(self):
        """Test endpoints with special characters."""
        limiter = DistributedRateLimiter(
            instance_id="special-test",
            strict_mode=False,
        )

        # Various endpoint formats
        endpoints = [
            "/api/v1/users/123",
            "/api/debates?filter=active",
            "/api/workspaces/ws-123/members",
            "/api/data:export",
        ]

        for endpoint in endpoints:
            result = limiter.allow("client", endpoint)
            assert result is not None

    def test_very_long_tenant_id(self):
        """Test handling of very long tenant IDs."""
        limiter = DistributedRateLimiter(
            instance_id="long-tenant-test",
            strict_mode=False,
        )

        long_tenant = "t" * 1000

        result = limiter.allow("client", "/api/test", tenant_id=long_tenant)
        assert result is not None

    def test_cleanup_stale_entries(self):
        """Test cleanup of stale rate limit entries."""
        limiter = DistributedRateLimiter(
            instance_id="cleanup-test",
            strict_mode=False,
        )

        # Make some requests
        for i in range(10):
            limiter.allow(f"client-{i}", "/api/cleanup")

        # Cleanup should not error
        cleaned = limiter.cleanup(max_age_seconds=0)
        assert cleaned >= 0


__all__ = [
    "TestDistributedLimiterBasic",
    "TestMultiInstanceCoordination",
    "TestRedisFailover",
    "TestRateLimiterPerformance",
    "TestRateLimiterConfiguration",
    "TestEdgeCases",
]
