"""
API rate limits benchmark tests.

Measures performance of rate limiting under various load conditions
and stress tests the rate limiting infrastructure.
"""

import asyncio
import time
import pytest
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import threading


# =============================================================================
# Mock Rate Limiter Implementation
# =============================================================================


class TokenBucketRateLimiter:
    """Token bucket rate limiter for benchmarking."""

    def __init__(self, rate_per_second: float, burst_size: int):
        self.rate_per_second = rate_per_second
        self.burst_size = burst_size
        self.tokens = float(burst_size)
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def try_acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.burst_size, self.tokens + elapsed * self.rate_per_second
            )
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_wait_time(self) -> float:
        """Get time to wait for next token."""
        if self.tokens >= 1:
            return 0
        tokens_needed = 1 - self.tokens
        return tokens_needed / self.rate_per_second


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for benchmarking."""

    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            now = time.time()
            cutoff = now - self.window_seconds

            # Remove old requests
            self.requests = [t for t in self.requests if t > cutoff]

            if len(self.requests) >= self.limit:
                return False

            self.requests.append(now)
            return True


# =============================================================================
# Rate Limiter Performance Tests
# =============================================================================


class TestRateLimiterPerformance:
    """Test rate limiter performance characteristics."""

    def test_token_bucket_acquisition_speed(self):
        """Measure token bucket acquisition performance."""
        limiter = TokenBucketRateLimiter(rate_per_second=10000, burst_size=10000)

        num_acquisitions = 10000
        start = time.time()

        for _ in range(num_acquisitions):
            limiter.try_acquire()

        elapsed = time.time() - start
        acquisitions_per_second = num_acquisitions / elapsed

        # Should be very fast (in-memory operations)
        assert acquisitions_per_second > 100000

    def test_sliding_window_acquisition_speed(self):
        """Measure sliding window acquisition performance."""
        limiter = SlidingWindowRateLimiter(limit=10000, window_seconds=1.0)

        num_acquisitions = 10000
        start = time.time()

        for _ in range(num_acquisitions):
            limiter.try_acquire()

        elapsed = time.time() - start
        acquisitions_per_second = num_acquisitions / elapsed

        # Slightly slower due to list operations
        assert acquisitions_per_second > 1000

    def test_concurrent_acquisition_performance(self):
        """Measure concurrent acquisition performance."""
        limiter = TokenBucketRateLimiter(rate_per_second=10000, burst_size=10000)
        successes = []
        lock = threading.Lock()

        def acquire():
            result = limiter.try_acquire()
            with lock:
                successes.append(result)

        threads = []
        num_threads = 100
        acquisitions_per_thread = 100

        start = time.time()

        for _ in range(num_threads):
            for _ in range(acquisitions_per_thread):
                t = threading.Thread(target=acquire)
                threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        elapsed = time.time() - start
        total_acquisitions = num_threads * acquisitions_per_thread
        acquisitions_per_second = total_acquisitions / elapsed

        # Should handle concurrent access reasonably well
        assert acquisitions_per_second > 500


# =============================================================================
# Multi-Client Rate Limit Tests
# =============================================================================


class TestMultiClientRateLimiting:
    """Test rate limiting with multiple clients."""

    def test_per_client_isolation(self):
        """Each client should have separate rate limits."""
        client_limiters: Dict[str, TokenBucketRateLimiter] = {}

        def get_limiter(client_id: str) -> TokenBucketRateLimiter:
            if client_id not in client_limiters:
                client_limiters[client_id] = TokenBucketRateLimiter(
                    rate_per_second=10, burst_size=10
                )
            return client_limiters[client_id]

        # Each client should get 10 tokens
        results = {}
        for client in ["client-1", "client-2", "client-3"]:
            limiter = get_limiter(client)
            count = 0
            for _ in range(15):
                if limiter.try_acquire():
                    count += 1
            results[client] = count

        # Each client should have gotten exactly 10 tokens
        assert all(count == 10 for count in results.values())

    def test_many_clients_performance(self):
        """Test performance with many concurrent clients."""
        num_clients = 100
        client_limiters: Dict[str, TokenBucketRateLimiter] = {}

        def get_limiter(client_id: str) -> TokenBucketRateLimiter:
            if client_id not in client_limiters:
                client_limiters[client_id] = TokenBucketRateLimiter(
                    rate_per_second=100, burst_size=10
                )
            return client_limiters[client_id]

        start = time.time()

        for i in range(num_clients):
            client_id = f"client-{i}"
            limiter = get_limiter(client_id)
            for _ in range(10):
                limiter.try_acquire()

        elapsed = time.time() - start
        ops_per_second = (num_clients * 10) / elapsed

        assert len(client_limiters) == num_clients
        assert ops_per_second > 10000


# =============================================================================
# Tier-Based Rate Limit Tests
# =============================================================================


@dataclass
class TierConfig:
    """Rate limit configuration for a tier."""

    name: str
    rate_per_second: float
    burst_size: int


class TestTierBasedRateLimiting:
    """Test tier-based rate limiting performance."""

    def test_tier_enforcement(self):
        """Test that different tiers have different limits."""
        tiers = {
            "free": TierConfig("free", rate_per_second=1, burst_size=5),
            "starter": TierConfig("starter", rate_per_second=5, burst_size=20),
            "professional": TierConfig("professional", rate_per_second=20, burst_size=50),
            "enterprise": TierConfig("enterprise", rate_per_second=100, burst_size=200),
        }

        results = {}
        for tier_name, config in tiers.items():
            limiter = TokenBucketRateLimiter(
                rate_per_second=config.rate_per_second, burst_size=config.burst_size
            )
            count = 0
            for _ in range(100):
                if limiter.try_acquire():
                    count += 1
            results[tier_name] = count

        # Higher tiers should get more requests through
        assert results["free"] < results["starter"]
        assert results["starter"] < results["professional"]
        assert results["professional"] < results["enterprise"]

    def test_tier_upgrade_handling(self):
        """Test upgrading tier mid-session."""
        # Start with free tier
        limiter = TokenBucketRateLimiter(rate_per_second=1, burst_size=5)

        # Use up free tier limit
        for _ in range(5):
            limiter.try_acquire()

        # Should be rate limited
        assert limiter.try_acquire() is False

        # Upgrade to professional (new limiter)
        limiter = TokenBucketRateLimiter(rate_per_second=20, burst_size=50)

        # Should have new capacity
        count = 0
        for _ in range(50):
            if limiter.try_acquire():
                count += 1

        assert count == 50


# =============================================================================
# Endpoint-Specific Rate Limit Tests
# =============================================================================


class TestEndpointRateLimiting:
    """Test endpoint-specific rate limiting."""

    def test_different_endpoints_different_limits(self):
        """Different endpoints should have different limits."""
        endpoint_limits = {
            "/api/debates": (10, 20),  # rate_per_second, burst
            "/api/gauntlet": (2, 5),  # Expensive endpoint
            "/api/health": (100, 200),  # Cheap endpoint
        }

        limiters = {
            endpoint: TokenBucketRateLimiter(rate, burst)
            for endpoint, (rate, burst) in endpoint_limits.items()
        }

        # Burst requests to each endpoint
        results = {}
        for endpoint, limiter in limiters.items():
            count = 0
            for _ in range(50):
                if limiter.try_acquire():
                    count += 1
            results[endpoint] = count

        assert results["/api/gauntlet"] < results["/api/debates"]
        assert results["/api/debates"] < results["/api/health"]

    def test_endpoint_rate_limit_performance(self):
        """Test performance of endpoint-specific rate limiting."""
        num_endpoints = 10
        limiters = {
            f"/api/endpoint-{i}": TokenBucketRateLimiter(
                rate_per_second=100, burst_size=100
            )
            for i in range(num_endpoints)
        }

        num_requests = 10000
        start = time.time()

        for i in range(num_requests):
            endpoint = f"/api/endpoint-{i % num_endpoints}"
            limiters[endpoint].try_acquire()

        elapsed = time.time() - start
        requests_per_second = num_requests / elapsed

        assert requests_per_second > 50000


# =============================================================================
# Rate Limit Recovery Tests
# =============================================================================


class TestRateLimitRecovery:
    """Test rate limit token recovery performance."""

    @pytest.mark.asyncio
    async def test_token_recovery_timing(self):
        """Test that tokens recover at correct rate."""
        limiter = TokenBucketRateLimiter(rate_per_second=10, burst_size=10)

        # Exhaust all tokens
        for _ in range(10):
            limiter.try_acquire()

        assert limiter.try_acquire() is False

        # Wait for 1 token to recover (0.1 seconds at 10/sec)
        await asyncio.sleep(0.15)

        # Should have at least 1 token now
        assert limiter.try_acquire() is True

    @pytest.mark.asyncio
    async def test_recovery_under_load(self):
        """Test token recovery while under continuous load."""
        limiter = TokenBucketRateLimiter(rate_per_second=100, burst_size=10)

        # Exhaust burst
        for _ in range(10):
            limiter.try_acquire()

        # Try to acquire at rate slightly below limit
        successes = 0
        for _ in range(50):
            if limiter.try_acquire():
                successes += 1
            await asyncio.sleep(0.015)  # ~67 req/sec, below 100 limit

        # Should get most requests through due to recovery
        assert successes > 30


# =============================================================================
# Stress Tests
# =============================================================================


class TestRateLimitStress:
    """Stress tests for rate limiting infrastructure."""

    def test_high_volume_requests(self):
        """Test handling of high volume requests."""
        limiter = TokenBucketRateLimiter(rate_per_second=1000, burst_size=100)

        num_requests = 100000
        allowed = 0

        start = time.time()

        for _ in range(num_requests):
            if limiter.try_acquire():
                allowed += 1

        elapsed = time.time() - start
        requests_per_second = num_requests / elapsed

        # Should process requests quickly (in-memory operations)
        assert requests_per_second > 100000
        # Burst + some recovered tokens during test
        # At 1000/sec rate and ~0.4s elapsed, expect ~100 + 400 = 500 max
        assert 100 <= allowed <= 600

    def test_sustained_rate_limiting(self):
        """Test sustained rate limiting over time."""
        limiter = TokenBucketRateLimiter(rate_per_second=1000, burst_size=100)

        batches = 10
        requests_per_batch = 1000
        total_allowed = 0

        for batch in range(batches):
            for _ in range(requests_per_batch):
                if limiter.try_acquire():
                    total_allowed += 1
            # Small delay between batches for token recovery
            time.sleep(0.01)

        # Should allow burst + some recovered tokens
        assert total_allowed > 100  # More than just burst
        assert total_allowed < batches * requests_per_batch  # But not all


# =============================================================================
# Latency Tests
# =============================================================================


class TestRateLimitLatency:
    """Test rate limit check latency."""

    def test_acquire_latency_distribution(self):
        """Measure rate limit check latency distribution."""
        limiter = TokenBucketRateLimiter(rate_per_second=10000, burst_size=10000)
        latencies = []

        for _ in range(10000):
            start = time.time()
            limiter.try_acquire()
            latencies.append((time.time() - start) * 1000000)  # microseconds

        latencies.sort()

        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        # Should be sub-microsecond for P50
        assert p50 < 10  # 10 microseconds
        assert p95 < 50  # 50 microseconds
        assert p99 < 100  # 100 microseconds

    def test_blocked_request_latency(self):
        """Measure latency for blocked requests."""
        limiter = TokenBucketRateLimiter(rate_per_second=10, burst_size=10)

        # Exhaust tokens
        for _ in range(10):
            limiter.try_acquire()

        # Measure blocked request latency
        latencies = []
        for _ in range(1000):
            start = time.time()
            limiter.try_acquire()  # Will be blocked
            latencies.append((time.time() - start) * 1000000)  # microseconds

        avg_latency = sum(latencies) / len(latencies)

        # Blocked requests should still be fast
        assert avg_latency < 10  # 10 microseconds


# =============================================================================
# Memory Usage Tests
# =============================================================================


class TestRateLimitMemory:
    """Test rate limit memory usage."""

    def test_sliding_window_memory_bounded(self):
        """Sliding window memory should be bounded."""
        import sys

        limiter = SlidingWindowRateLimiter(limit=1000, window_seconds=60.0)

        # Fill up the window
        for _ in range(1000):
            limiter.try_acquire()

        size_full = sys.getsizeof(limiter.requests)

        # Wait for window to slide (simulate)
        limiter.requests = [
            t for t in limiter.requests if t > time.time() - 60
        ]

        # Add more (should evict old)
        for _ in range(500):
            limiter.try_acquire()

        size_after = sys.getsizeof(limiter.requests)

        # Memory should not grow unbounded
        # (Actual size depends on list implementation)
        assert len(limiter.requests) <= 1000

    def test_many_limiters_memory(self):
        """Test memory with many rate limiters."""
        import sys

        limiters = []
        for i in range(10000):
            limiters.append(
                TokenBucketRateLimiter(rate_per_second=100, burst_size=10)
            )

        # Each limiter should be small
        sample_size = sys.getsizeof(limiters[0])
        assert sample_size < 1000  # Less than 1KB per limiter

        # Total memory for 10k limiters should be reasonable
        total_estimate = sample_size * len(limiters)
        assert total_estimate < 10 * 1024 * 1024  # Less than 10MB
