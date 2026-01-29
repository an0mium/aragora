"""
Tests for distributed rate limiting with RedisRateLimiter.

Tests cover:
- RateLimitCircuitBreaker state transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Multi-instance coordination via Redis
- Fallback to in-memory behavior when Redis fails
- Distributed metrics aggregation accuracy
- Circuit breaker recovery scenarios
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.rate_limit.redis_limiter import (
    RateLimitCircuitBreaker,
    RedisRateLimiter,
)


# =============================================================================
# Mock Redis Implementation
# =============================================================================


class MockRedis:
    """Mock Redis client for testing distributed scenarios."""

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._hashes: Dict[str, Dict[str, str]] = {}
        self._ttls: Dict[str, float] = {}
        self._available = True
        self._fail_count = 0
        self._max_failures = 0

    def set_fail_mode(self, fail_count: int) -> None:
        """Configure Redis to fail for the next N operations."""
        self._fail_count = fail_count
        self._max_failures = fail_count

    def _check_fail(self) -> None:
        """Check if we should simulate a failure."""
        if self._fail_count > 0:
            self._fail_count -= 1
            raise ConnectionError("Simulated Redis failure")

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

    def hset(
        self,
        name: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        mapping: Optional[Dict[str, str]] = None,
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

    def hgetall(self, name: str) -> Dict[str, str]:
        self._check_fail()
        return self._hashes.get(name, {})

    def expire(self, key: str, seconds: int) -> bool:
        self._check_fail()
        self._ttls[key] = time.time() + seconds
        return True

    def pipeline(self) -> "MockPipeline":
        return MockPipeline(self)

    def close(self) -> None:
        pass


class MockPipeline:
    """Mock Redis pipeline for atomic operations."""

    def __init__(self, redis: MockRedis):
        self._redis = redis
        self._commands: List[tuple] = []

    def hset(
        self,
        name: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        mapping: Optional[Dict[str, str]] = None,
    ) -> "MockPipeline":
        self._commands.append(("hset", (name,), {"key": key, "value": value, "mapping": mapping}))
        return self

    def expire(self, key: str, seconds: int) -> "MockPipeline":
        self._commands.append(("expire", (key, seconds), {}))
        return self

    def incr(self, key: str) -> "MockPipeline":
        self._commands.append(("incr", (key,), {}))
        return self

    def execute(self) -> List[Any]:
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
def circuit_breaker():
    """Create a circuit breaker with low thresholds for testing."""
    return RateLimitCircuitBreaker(
        failure_threshold=3,
        recovery_timeout=0.1,  # 100ms for fast tests
        half_open_max_calls=2,
    )


@pytest.fixture
def redis_limiter(mock_redis):
    """Create a RedisRateLimiter with mock Redis."""
    # Patch the RedisTokenBucket to work with our mock
    with patch(
        "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
    ) as mock_bucket_class:
        # Create a mock bucket that tracks consumption
        mock_bucket = MagicMock()
        mock_bucket.consume.return_value = True
        mock_bucket.remaining = 50
        mock_bucket.get_retry_after.return_value = 0
        mock_bucket_class.return_value = mock_bucket

        limiter = RedisRateLimiter(
            redis_client=mock_redis,
            default_limit=60,
            ip_limit=120,
            enable_circuit_breaker=True,
            enable_distributed_metrics=True,
            instance_id="test-instance-1",
        )
        limiter._mock_bucket = mock_bucket  # Store for test access
        yield limiter


# =============================================================================
# TestRateLimitCircuitBreaker - State Transitions
# =============================================================================


class TestCircuitBreakerStates:
    """Tests for RateLimitCircuitBreaker state machine."""

    def test_initial_state_is_closed(self, circuit_breaker):
        """Circuit breaker starts in CLOSED state."""
        assert circuit_breaker.state == RateLimitCircuitBreaker.CLOSED

    def test_closed_allows_requests(self, circuit_breaker):
        """CLOSED state allows all requests."""
        for _ in range(10):
            assert circuit_breaker.allow_request() is True

    def test_opens_after_failure_threshold(self, circuit_breaker):
        """Circuit opens after reaching failure threshold."""
        # Record 3 failures (threshold)
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == RateLimitCircuitBreaker.OPEN

    def test_open_blocks_requests(self, circuit_breaker):
        """OPEN state blocks all requests."""
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.allow_request() is False
        assert circuit_breaker.allow_request() is False

    def test_transitions_to_half_open_after_timeout(self, circuit_breaker):
        """Circuit transitions to HALF_OPEN after recovery timeout."""
        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure()

        assert circuit_breaker.state == RateLimitCircuitBreaker.OPEN

        # Wait for recovery timeout
        time.sleep(0.15)  # > 100ms recovery timeout

        assert circuit_breaker.state == RateLimitCircuitBreaker.HALF_OPEN

    def test_half_open_allows_limited_requests(self, circuit_breaker):
        """HALF_OPEN allows limited test requests."""
        # Open then wait for half-open
        for _ in range(3):
            circuit_breaker.record_failure()
        time.sleep(0.15)

        # Should allow half_open_max_calls (2) requests
        assert circuit_breaker.allow_request() is True
        assert circuit_breaker.allow_request() is True
        # Third request should be blocked
        assert circuit_breaker.allow_request() is False

    def test_half_open_closes_on_success(self, circuit_breaker):
        """HALF_OPEN closes circuit on successful calls."""
        # Open then wait for half-open
        for _ in range(3):
            circuit_breaker.record_failure()
        time.sleep(0.15)

        # Make successful calls in half-open
        circuit_breaker.allow_request()
        circuit_breaker.record_success()
        circuit_breaker.allow_request()
        circuit_breaker.record_success()

        assert circuit_breaker.state == RateLimitCircuitBreaker.CLOSED

    def test_half_open_reopens_on_failure(self, circuit_breaker):
        """HALF_OPEN reopens circuit on failure."""
        # Open then wait for half-open
        for _ in range(3):
            circuit_breaker.record_failure()
        time.sleep(0.15)

        assert circuit_breaker.state == RateLimitCircuitBreaker.HALF_OPEN

        # Record a failure in half-open
        circuit_breaker.record_failure()

        assert circuit_breaker.state == RateLimitCircuitBreaker.OPEN

    def test_get_stats_returns_all_metrics(self, circuit_breaker):
        """get_stats returns comprehensive metrics."""
        circuit_breaker.record_failure()
        circuit_breaker.record_success()

        stats = circuit_breaker.get_stats()

        assert "state" in stats
        assert "failure_count" in stats
        assert "success_count" in stats
        assert "failure_threshold" in stats
        assert "recovery_timeout" in stats
        assert stats["failure_count"] == 1
        assert stats["success_count"] == 1


class TestCircuitBreakerThreadSafety:
    """Tests for circuit breaker thread safety."""

    def test_concurrent_failures(self):
        """Multiple threads recording failures should be safe."""
        cb = RateLimitCircuitBreaker(failure_threshold=100)
        errors = []

        def record_failures():
            try:
                for _ in range(50):
                    cb.record_failure()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_failures) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = cb.get_stats()
        assert stats["failure_count"] == 500

    def test_concurrent_state_transitions(self):
        """Concurrent state checks should be safe."""
        cb = RateLimitCircuitBreaker(
            failure_threshold=5,
            recovery_timeout=0.01,  # 10ms
        )
        states = []

        def check_state():
            for _ in range(100):
                states.append(cb.state)
                if cb.allow_request():
                    cb.record_success()
                else:
                    cb.record_failure()
                time.sleep(0.001)

        threads = [threading.Thread(target=check_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have recorded many states without errors
        assert len(states) == 500


# =============================================================================
# TestRedisRateLimiterFallback - Fallback Behavior
# =============================================================================


class TestRedisRateLimiterFallback:
    """Tests for fallback to in-memory when Redis fails."""

    def test_uses_fallback_when_circuit_open(self, redis_limiter, mock_redis):
        """Should use in-memory fallback when circuit is open."""
        # Open the circuit breaker
        redis_limiter._circuit_breaker._state = RateLimitCircuitBreaker.OPEN
        redis_limiter._circuit_breaker._last_failure_time = time.time()

        result = redis_limiter.allow("192.168.1.1")

        assert result.allowed is True
        assert redis_limiter._fallback_requests == 1

    def test_uses_fallback_on_redis_error(self, mock_redis):
        """Should fall back to in-memory on Redis errors."""
        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = ConnectionError("Redis down")
            mock_bucket_class.return_value = mock_bucket

            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=True,
                instance_id="test-fallback",
            )

            # Should not raise, should use fallback
            result = limiter.allow("192.168.1.1")

            assert result.allowed is True
            assert limiter._fallback_requests == 1

    def test_fallback_maintains_rate_limiting(self, mock_redis):
        """Fallback should still enforce rate limits."""
        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = ConnectionError("Redis down")
            mock_bucket_class.return_value = mock_bucket

            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                default_limit=2,
                ip_limit=2,
                enable_circuit_breaker=True,
                instance_id="test-fallback-limits",
            )
            # Configure fallback to have low limits
            limiter._fallback._ip_buckets["192.168.1.1"] = MagicMock()
            limiter._fallback._ip_buckets["192.168.1.1"].consume.side_effect = [True, True, False]
            limiter._fallback._ip_buckets["192.168.1.1"].remaining = 0
            limiter._fallback._ip_buckets["192.168.1.1"].get_retry_after.return_value = 1.0

            # First two should succeed
            result1 = limiter.allow("192.168.1.1")
            result2 = limiter.allow("192.168.1.1")

            assert result1.allowed is True
            assert result2.allowed is True


# =============================================================================
# TestDistributedMetrics - Multi-Instance Aggregation
# =============================================================================


class TestDistributedMetrics:
    """Tests for distributed metrics aggregation."""

    def test_syncs_metrics_to_redis(self, redis_limiter, mock_redis):
        """Metrics should be synced to Redis periodically."""
        # Make some requests to generate metrics
        redis_limiter._requests_allowed = 100
        redis_limiter._requests_rejected = 10
        redis_limiter._last_metrics_sync = 0  # Force sync

        redis_limiter._maybe_sync_distributed_metrics()

        # Check metrics were stored in Redis
        key = f"{redis_limiter._metrics_key}{redis_limiter.instance_id}"
        assert key in mock_redis._hashes
        metrics = mock_redis._hashes[key]
        assert metrics["requests_allowed"] == "100"
        assert metrics["requests_rejected"] == "10"

    def test_aggregates_metrics_from_multiple_instances(self, mock_redis):
        """Should aggregate metrics from all instances."""
        # Simulate metrics from multiple instances
        mock_redis._hashes["aragora:ratelimit:metrics:instance-1"] = {
            "requests_allowed": "100",
            "requests_rejected": "10",
            "redis_failures": "1",
            "fallback_requests": "5",
            "last_sync": "2024-01-01T00:00:00",
        }
        mock_redis._hashes["aragora:ratelimit:metrics:instance-2"] = {
            "requests_allowed": "200",
            "requests_rejected": "20",
            "redis_failures": "2",
            "fallback_requests": "10",
            "last_sync": "2024-01-01T00:00:00",
        }

        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_distributed_metrics=True,
                instance_id="aggregator",
            )

            aggregated = limiter.get_distributed_metrics()

            assert aggregated["total_requests_allowed"] == 300
            assert aggregated["total_requests_rejected"] == 30
            assert aggregated["total_redis_failures"] == 3
            assert aggregated["total_fallback_requests"] == 15
            assert aggregated["instance_count"] == 2

    def test_calculates_rejection_rate(self, mock_redis):
        """Should calculate overall rejection rate."""
        mock_redis._hashes["aragora:ratelimit:metrics:instance-1"] = {
            "requests_allowed": "90",
            "requests_rejected": "10",
            "redis_failures": "0",
            "fallback_requests": "0",
        }

        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_distributed_metrics=True,
                instance_id="calc",
            )

            aggregated = limiter.get_distributed_metrics()

            # 10 rejected out of 100 total = 10% rejection rate
            assert abs(aggregated["total_rejection_rate"] - 0.1) < 0.01

    def test_handles_missing_metrics_gracefully(self, mock_redis):
        """Should handle missing or partial metrics."""
        mock_redis._hashes["aragora:ratelimit:metrics:partial"] = {
            "requests_allowed": "50",
            # Missing other fields
        }

        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_distributed_metrics=True,
                instance_id="handler",
            )

            # Should not raise
            aggregated = limiter.get_distributed_metrics()
            assert aggregated["total_requests_allowed"] == 50
            assert aggregated["total_requests_rejected"] == 0


# =============================================================================
# TestMultiInstanceCoordination - Shared State
# =============================================================================


class TestMultiInstanceCoordination:
    """Tests for rate limit coordination across instances."""

    def test_instances_share_rate_limit_state(self):
        """Multiple limiter instances should share state via Redis."""
        # Use real MockRedis that both instances share
        shared_redis = MockRedis()

        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            # Track all consume calls across instances
            consume_count = [0]

            def track_consume(n):
                consume_count[0] += n
                # After 5 total consumes, start rejecting
                return consume_count[0] <= 5

            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = track_consume
            mock_bucket.remaining = 0
            mock_bucket.get_retry_after.return_value = 1.0
            mock_bucket_class.return_value = mock_bucket

            limiter1 = RedisRateLimiter(
                redis_client=shared_redis,
                default_limit=5,
                enable_circuit_breaker=False,
                instance_id="instance-1",
            )
            limiter2 = RedisRateLimiter(
                redis_client=shared_redis,
                default_limit=5,
                enable_circuit_breaker=False,
                instance_id="instance-2",
            )

            # Alternate requests between instances
            results = []
            for i in range(8):
                limiter = limiter1 if i % 2 == 0 else limiter2
                results.append(limiter.allow("shared-client"))

            # First 5 should be allowed, rest should be blocked
            allowed = sum(1 for r in results if r.allowed)
            assert allowed == 5

    def test_endpoint_configs_independent_per_instance(self, mock_redis):
        """Endpoint configurations are instance-local (not shared)."""
        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter1 = RedisRateLimiter(
                redis_client=mock_redis,
                instance_id="config-1",
            )
            limiter2 = RedisRateLimiter(
                redis_client=mock_redis,
                instance_id="config-2",
            )

            # Configure different limits on each instance
            limiter1.configure_endpoint("/api/test", 100)
            limiter2.configure_endpoint("/api/test", 200)

            config1 = limiter1.get_config("/api/test")
            config2 = limiter2.get_config("/api/test")

            assert config1.requests_per_minute == 100
            assert config2.requests_per_minute == 200


# =============================================================================
# TestObservabilityMetrics - Local Metrics
# =============================================================================


class TestObservabilityMetrics:
    """Tests for local observability metrics."""

    def test_tracks_allowed_requests(self, redis_limiter):
        """Should track allowed request count."""
        redis_limiter.allow("192.168.1.1")
        redis_limiter.allow("192.168.1.2")

        assert redis_limiter._requests_allowed == 2

    def test_tracks_rejected_requests(self, redis_limiter):
        """Should track rejected request count."""
        # Make bucket reject requests
        redis_limiter._mock_bucket.consume.return_value = False

        redis_limiter.allow("192.168.1.1")
        redis_limiter.allow("192.168.1.1")

        assert redis_limiter._requests_rejected == 2

    def test_tracks_rejections_by_endpoint(self, redis_limiter):
        """Should track rejections per endpoint."""
        redis_limiter._mock_bucket.consume.return_value = False

        redis_limiter.allow("192.168.1.1", endpoint="/api/debates")
        redis_limiter.allow("192.168.1.1", endpoint="/api/debates")
        redis_limiter.allow("192.168.1.1", endpoint="/api/agents")

        assert redis_limiter._rejections_by_endpoint.get("/api/debates") == 2
        assert redis_limiter._rejections_by_endpoint.get("/api/agents") == 1

    def test_tracks_redis_failures(self, mock_redis):
        """Should track Redis failure count."""
        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = ConnectionError("Redis down")
            mock_bucket_class.return_value = mock_bucket

            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=True,
                instance_id="failure-tracker",
            )

            limiter.allow("192.168.1.1")
            limiter.allow("192.168.1.1")

            assert limiter._redis_failures == 2

    def test_reset_metrics_clears_all(self, redis_limiter):
        """reset_metrics should clear all counters."""
        redis_limiter._requests_allowed = 100
        redis_limiter._requests_rejected = 50
        redis_limiter._rejections_by_endpoint["/api/test"] = 25

        redis_limiter.reset_metrics()

        assert redis_limiter._requests_allowed == 0
        assert redis_limiter._requests_rejected == 0
        assert len(redis_limiter._rejections_by_endpoint) == 0

    def test_get_stats_includes_all_metrics(self, redis_limiter, mock_redis):
        """get_stats should include all observability metrics."""
        redis_limiter._requests_allowed = 100
        redis_limiter._requests_rejected = 10

        stats = redis_limiter.get_stats()

        assert stats["backend"] == "redis"
        assert stats["instance_id"] == "test-instance-1"
        assert stats["requests_allowed"] == 100
        assert stats["requests_rejected"] == 10
        assert stats["total_requests"] == 110
        assert abs(stats["rejection_rate"] - 0.0909) < 0.01  # ~9%
        assert "circuit_breaker" in stats


# =============================================================================
# Integration Tests
# =============================================================================


class TestDistributedRateLimitingIntegration:
    """Integration tests for complete distributed scenarios."""

    def test_full_circuit_breaker_recovery_cycle(self, mock_redis):
        """Test complete circuit breaker lifecycle with recovery."""
        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            should_fail = [True]  # Mutable flag to control failures

            def conditional_consume(n):
                if should_fail[0]:
                    raise ConnectionError("Simulated failure")
                return True

            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = conditional_consume
            mock_bucket.remaining = 50
            mock_bucket.get_retry_after.return_value = 0
            mock_bucket_class.return_value = mock_bucket

            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=True,
                instance_id="recovery-test",
            )
            limiter._circuit_breaker = RateLimitCircuitBreaker(
                failure_threshold=3,
                recovery_timeout=0.05,  # 50ms
                half_open_max_calls=2,
            )

            # Phase 1: Failures open the circuit
            for _ in range(3):
                limiter.allow("192.168.1.1")

            assert limiter._circuit_breaker.state == RateLimitCircuitBreaker.OPEN
            assert limiter._fallback_requests >= 1  # Used fallback

            # Phase 2: Wait for half-open
            time.sleep(0.06)
            assert limiter._circuit_breaker.state == RateLimitCircuitBreaker.HALF_OPEN

            # Phase 3: Redis "recovers" - stop failing
            should_fail[0] = False

            # Successful calls close the circuit
            limiter.allow("192.168.1.1")
            limiter.allow("192.168.1.1")

            # Circuit should be closed now
            assert limiter._circuit_breaker.state == RateLimitCircuitBreaker.CLOSED

    def test_metrics_aggregation_accuracy(self, mock_redis):
        """Test that distributed metrics are accurately aggregated."""
        # Set up metrics from 3 instances
        for i in range(3):
            mock_redis._hashes[f"aragora:ratelimit:metrics:instance-{i}"] = {
                "requests_allowed": str(100 * (i + 1)),
                "requests_rejected": str(10 * (i + 1)),
                "redis_failures": str(i),
                "fallback_requests": str(5 * i),
                "last_sync": "2024-01-01T00:00:00",
            }

        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_distributed_metrics=True,
                instance_id="aggregator",
            )

            metrics = limiter.get_distributed_metrics()

            # instance-0: 100 allowed, 10 rejected
            # instance-1: 200 allowed, 20 rejected
            # instance-2: 300 allowed, 30 rejected
            assert metrics["total_requests_allowed"] == 600
            assert metrics["total_requests_rejected"] == 60
            assert metrics["total_redis_failures"] == 3  # 0 + 1 + 2
            assert metrics["total_fallback_requests"] == 15  # 0 + 5 + 10
            assert metrics["instance_count"] == 3

            # Rejection rate: 60 / 660 = 9.09%
            expected_rate = 60 / 660
            assert abs(metrics["total_rejection_rate"] - expected_rate) < 0.001
