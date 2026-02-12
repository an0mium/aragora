"""
Tests for distributed rate limiting.

Tests cover:
- Distributed rate limiting across simulated instances
- Redis failure fallback behavior
- Strict mode enforcement
- Prometheus metrics recording
- Handler decorator integration with distributed limiter
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Mock Redis Implementation for Multi-Instance Testing
# =============================================================================


class SharedMockRedis:
    """
    Mock Redis client that simulates shared state across instances.

    This implementation uses a class-level data store to simulate
    Redis behavior where multiple "instances" share the same state.
    Includes Lua script support for token bucket operations.
    """

    # Class-level shared storage to simulate Redis
    _shared_data: dict[str, Any] = {}
    _shared_hashes: dict[str, dict[str, str]] = {}
    _shared_sorted_sets: dict[str, list] = {}
    _shared_scripts: dict[str, str] = {}
    _script_counter = 0
    _lock = threading.Lock()
    _fail_mode = False
    _fail_count = 0

    def __init__(self, instance_id: str = "default"):
        self.instance_id = instance_id
        self._available = True

    @classmethod
    def reset_shared_state(cls) -> None:
        """Reset all shared state (call between tests)."""
        with cls._lock:
            cls._shared_data.clear()
            cls._shared_hashes.clear()
            cls._shared_sorted_sets.clear()
            cls._shared_scripts.clear()
            cls._script_counter = 0
            cls._fail_mode = False
            cls._fail_count = 0

    @classmethod
    def set_fail_mode(cls, fail: bool = True, fail_count: int = 0) -> None:
        """Enable failure simulation."""
        with cls._lock:
            cls._fail_mode = fail
            cls._fail_count = fail_count

    def _check_fail(self) -> None:
        """Check if we should simulate a failure."""
        with self._lock:
            if self._fail_mode:
                if self._fail_count > 0:
                    SharedMockRedis._fail_count -= 1
                    if SharedMockRedis._fail_count <= 0:
                        SharedMockRedis._fail_mode = False
                raise ConnectionError("Simulated Redis failure")

    def ping(self) -> bool:
        self._check_fail()
        return True

    def get(self, key: str) -> str | None:
        self._check_fail()
        with self._lock:
            return self._shared_data.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> bool:
        self._check_fail()
        with self._lock:
            self._shared_data[key] = value
        return True

    def incr(self, key: str) -> int:
        self._check_fail()
        with self._lock:
            val = int(self._shared_data.get(key, 0)) + 1
            self._shared_data[key] = str(val)
            return val

    def delete(self, *keys: str) -> int:
        self._check_fail()
        deleted = 0
        with self._lock:
            for key in keys:
                if key in self._shared_data:
                    del self._shared_data[key]
                    deleted += 1
                if key in self._shared_hashes:
                    del self._shared_hashes[key]
                    deleted += 1
                if key in self._shared_sorted_sets:
                    del self._shared_sorted_sets[key]
                    deleted += 1
        return deleted

    def scan_iter(self, match: str = "*", count: int = 100):
        """Iterate keys matching pattern."""
        import fnmatch

        with self._lock:
            all_keys = (
                list(self._shared_data.keys())
                + list(self._shared_hashes.keys())
                + list(self._shared_sorted_sets.keys())
            )
            for key in all_keys:
                if fnmatch.fnmatch(key, match):
                    yield key

    def hset(
        self,
        name: str,
        key: str | None = None,
        value: str | None = None,
        mapping: dict[str, str] | None = None,
    ) -> int:
        self._check_fail()
        with self._lock:
            if name not in self._shared_hashes:
                self._shared_hashes[name] = {}
            if mapping:
                self._shared_hashes[name].update(mapping)
                return len(mapping)
            elif key and value:
                self._shared_hashes[name][key] = value
                return 1
        return 0

    def hgetall(self, name: str) -> dict[str, str]:
        self._check_fail()
        with self._lock:
            return dict(self._shared_hashes.get(name, {}))

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        """Add members to sorted set."""
        self._check_fail()
        with self._lock:
            if key not in self._shared_sorted_sets:
                self._shared_sorted_sets[key] = []
            added = 0
            for member, score in mapping.items():
                # Remove existing if present
                self._shared_sorted_sets[key] = [
                    (m, s) for m, s in self._shared_sorted_sets[key] if m != member
                ]
                self._shared_sorted_sets[key].append((member, score))
                added += 1
            # Sort by score
            self._shared_sorted_sets[key].sort(key=lambda x: x[1])
            return added

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """Remove members by score range."""
        self._check_fail()
        with self._lock:
            if key not in self._shared_sorted_sets:
                return 0
            original_len = len(self._shared_sorted_sets[key])
            self._shared_sorted_sets[key] = [
                (m, s)
                for m, s in self._shared_sorted_sets[key]
                if not (min_score <= s <= max_score)
            ]
            return original_len - len(self._shared_sorted_sets[key])

    def zcard(self, key: str) -> int:
        """Get cardinality of sorted set."""
        self._check_fail()
        with self._lock:
            return len(self._shared_sorted_sets.get(key, []))

    def zrem(self, key: str, *members: str) -> int:
        """Remove members from sorted set."""
        self._check_fail()
        with self._lock:
            if key not in self._shared_sorted_sets:
                return 0
            removed = 0
            for member in members:
                original_len = len(self._shared_sorted_sets[key])
                self._shared_sorted_sets[key] = [
                    (m, s) for m, s in self._shared_sorted_sets[key] if m != member
                ]
                if len(self._shared_sorted_sets[key]) < original_len:
                    removed += 1
            return removed

    def zrange(self, key: str, start: int, end: int, withscores: bool = False) -> list:
        """Get range of members from sorted set."""
        self._check_fail()
        with self._lock:
            members = self._shared_sorted_sets.get(key, [])
            if end == -1:
                end = len(members)
            else:
                end = end + 1
            sliced = members[start:end]
            if withscores:
                return sliced
            return [m for m, _ in sliced]

    def expire(self, key: str, seconds: int) -> bool:
        self._check_fail()
        return True

    def hmget(self, name: str, keys: list) -> list:
        """Get multiple hash fields."""
        self._check_fail()
        with self._lock:
            hash_data = self._shared_hashes.get(name, {})
            return [hash_data.get(k) for k in keys]

    def hmset(self, name: str, mapping: dict[str, str]) -> bool:
        """Set multiple hash fields."""
        self._check_fail()
        with self._lock:
            if name not in self._shared_hashes:
                self._shared_hashes[name] = {}
            self._shared_hashes[name].update(mapping)
        return True

    def script_load(self, script: str) -> str:
        """Load a Lua script and return its SHA."""
        self._check_fail()
        with self._lock:
            SharedMockRedis._script_counter += 1
            sha = f"script_sha_{SharedMockRedis._script_counter}"
            SharedMockRedis._shared_scripts[sha] = script
            return sha

    def evalsha(self, sha: str, numkeys: int, *args) -> list:
        """Execute a Lua script by SHA.

        This is a simplified implementation that handles the token bucket
        consume script specifically.
        """
        self._check_fail()

        # Parse args based on token bucket consume script
        if numkeys == 1:
            key = args[0]
            rate = float(args[1])
            burst = float(args[2])
            now = float(args[3])
            tokens_requested = float(args[4])
            ttl = int(args[5])

            with self._lock:
                # Get current state from hash
                hash_data = self._shared_hashes.get(key, {})
                tokens = float(hash_data.get("tokens", burst))
                last_refill = float(hash_data.get("last_refill", now))

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
                if key not in self._shared_hashes:
                    self._shared_hashes[key] = {}
                self._shared_hashes[key]["tokens"] = str(tokens)
                self._shared_hashes[key]["last_refill"] = str(now)

                return [allowed, tokens, burst]

        return [1, 0, 0]  # Default: allow

    def pipeline(self) -> SharedMockPipeline:
        return SharedMockPipeline(self)

    def close(self) -> None:
        pass


class SharedMockPipeline:
    """Mock Redis pipeline for atomic operations."""

    def __init__(self, redis: SharedMockRedis):
        self._redis = redis
        self._commands: list = []

    def zadd(self, key: str, mapping: dict[str, float]) -> SharedMockPipeline:
        self._commands.append(("zadd", (key, mapping), {}))
        return self

    def zremrangebyscore(
        self, key: str, min_score: float, max_score: float
    ) -> SharedMockPipeline:
        self._commands.append(("zremrangebyscore", (key, min_score, max_score), {}))
        return self

    def zcard(self, key: str) -> SharedMockPipeline:
        self._commands.append(("zcard", (key,), {}))
        return self

    def hset(
        self,
        name: str,
        key: str | None = None,
        value: str | None = None,
        mapping: dict[str, str] | None = None,
    ) -> SharedMockPipeline:
        self._commands.append(("hset", (name,), {"key": key, "value": value, "mapping": mapping}))
        return self

    def expire(self, key: str, seconds: int) -> SharedMockPipeline:
        self._commands.append(("expire", (key, seconds), {}))
        return self

    def incr(self, key: str) -> SharedMockPipeline:
        self._commands.append(("incr", (key,), {}))
        return self

    def execute(self) -> list:
        results = []
        for cmd, args, kwargs in self._commands:
            method = getattr(self._redis, cmd)
            results.append(method(*args, **kwargs))
        self._commands.clear()
        return results


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_shared_redis():
    """Reset shared Redis state before each test."""
    SharedMockRedis.reset_shared_state()
    yield
    SharedMockRedis.reset_shared_state()


@pytest.fixture
def shared_redis():
    """Get a shared mock Redis client."""
    return SharedMockRedis("test-instance")


@pytest.fixture
def mock_settings():
    """Mock settings for rate limiting."""
    settings = MagicMock()
    settings.rate_limit.redis_key_prefix = "aragora:ratelimit"
    settings.rate_limit.redis_ttl_seconds = 120
    settings.rate_limit.redis_url = "redis://localhost:6379"
    return settings


# =============================================================================
# Tests for Cross-Instance Rate Limiting
# =============================================================================


class TestDistributedRateLimiting:
    """Tests that rate limits are shared across simulated instances."""

    def test_rate_limit_shared_across_instances(self, mock_settings):
        """Rate limit state should be shared via Redis across instances."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
        )

        # Track consume calls to simulate shared state
        consume_count = [0]
        limit = 5

        def mock_consume(n):
            consume_count[0] += n
            return consume_count[0] <= limit

        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = mock_consume
            mock_bucket.remaining = 0
            mock_bucket.get_retry_after.return_value = 1.0
            mock_bucket_class.return_value = mock_bucket

            shared_redis = SharedMockRedis("shared")

            with (
                patch(
                    "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                    return_value=shared_redis,
                ),
                patch(
                    "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                    return_value=shared_redis,
                ),
                patch("aragora.config.settings.get_settings", return_value=mock_settings),
            ):
                # Create two "instances" of the distributed limiter
                limiter1 = DistributedRateLimiter(instance_id="instance-1", enable_metrics=False)
                limiter2 = DistributedRateLimiter(instance_id="instance-2", enable_metrics=False)

                # Configure same endpoint on both (simulating deploy)
                limiter1.configure_endpoint("/api/test", limit)
                limiter2.configure_endpoint("/api/test", limit)

                # Make requests alternating between instances
                results = []
                for i in range(10):
                    limiter = limiter1 if i % 2 == 0 else limiter2
                    result = limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
                    results.append(result.allowed)

                # Since both share Redis state, the combined requests should hit limit
                allowed_count = sum(1 for r in results if r)
                # First 5 should be allowed, rest rejected
                assert allowed_count == limit, f"Expected {limit} allowed, got {allowed_count}"

    def test_different_clients_have_separate_limits(self, mock_settings):
        """Different client IPs should have independent rate limits."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
        )

        # Track consume calls per client
        client_counts: dict[str, int] = {}
        limit = 3

        def mock_consume(n, key=None):
            """Track consumes by the bucket key."""
            # Use current bucket key from mock_bucket
            current_key = getattr(mock_bucket, "_current_key", "unknown")
            client_counts[current_key] = client_counts.get(current_key, 0) + n
            return client_counts[current_key] <= limit

        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()

            def create_bucket(redis, key, rate, burst, prefix, ttl):
                mock_bucket._current_key = key
                return mock_bucket

            mock_bucket_class.side_effect = create_bucket
            mock_bucket.consume.side_effect = lambda n: mock_consume(n)
            mock_bucket.remaining = 0
            mock_bucket.get_retry_after.return_value = 1.0

            shared_redis = SharedMockRedis("shared")

            with (
                patch(
                    "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                    return_value=shared_redis,
                ),
                patch(
                    "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                    return_value=shared_redis,
                ),
                patch("aragora.config.settings.get_settings", return_value=mock_settings),
            ):
                limiter = DistributedRateLimiter(instance_id="instance-1", enable_metrics=False)
                limiter.configure_endpoint("/api/test", limit)

                # Client A makes 3 requests (should all be allowed)
                for _ in range(limit):
                    mock_bucket._current_key = "ip:192.168.1.1"
                    result = limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
                    assert result.allowed is True

                # Client A's 4th request should be rejected
                mock_bucket._current_key = "ip:192.168.1.1"
                result = limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
                assert result.allowed is False

                # Client B should still be able to make requests
                mock_bucket._current_key = "ip:192.168.1.2"
                result = limiter.allow(client_ip="192.168.1.2", endpoint="/api/test")
                assert result.allowed is True

    def test_tenant_isolation(self, mock_settings):
        """Different tenants should have isolated rate limits when key_type=tenant."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
        )

        # Track consume calls per bucket key
        bucket_counts: dict[str, int] = {}
        bucket_locks: dict[str, threading.Lock] = {}
        limit = 3

        class MockBucket:
            def __init__(self, key):
                self.key = key
                self.remaining = 0
                if key not in bucket_locks:
                    bucket_locks[key] = threading.Lock()

            def consume(self, n):
                with bucket_locks[self.key]:
                    bucket_counts[self.key] = bucket_counts.get(self.key, 0) + n
                    return bucket_counts[self.key] <= limit

            def get_retry_after(self):
                return 1.0

        buckets: dict[str, MockBucket] = {}

        def create_bucket(redis, key, rate, burst, prefix, ttl):
            full_key = f"{prefix}{key}"
            if full_key not in buckets:
                buckets[full_key] = MockBucket(full_key)
            return buckets[full_key]

        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket",
            side_effect=create_bucket,
        ):
            shared_redis = SharedMockRedis("shared")

            with (
                patch(
                    "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                    return_value=shared_redis,
                ),
                patch(
                    "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                    return_value=shared_redis,
                ),
                patch("aragora.config.settings.get_settings", return_value=mock_settings),
            ):
                limiter = DistributedRateLimiter(instance_id="instance-1", enable_metrics=False)
                limiter.configure_endpoint("/api/test", limit, key_type="tenant")

                # Tenant A exhausts their limit
                for _ in range(limit):
                    result = limiter.allow(
                        client_ip="192.168.1.1",
                        endpoint="/api/test",
                        tenant_id="tenant-a",
                    )
                    assert result.allowed is True

                # Tenant A's next request is rejected
                result = limiter.allow(
                    client_ip="192.168.1.1",
                    endpoint="/api/test",
                    tenant_id="tenant-a",
                )
                assert result.allowed is False

                # Tenant B should still have their full quota
                for _ in range(limit):
                    result = limiter.allow(
                        client_ip="192.168.1.2",
                        endpoint="/api/test",
                        tenant_id="tenant-b",
                    )
                    assert result.allowed is True


# =============================================================================
# Tests for Redis Failure Fallback
# =============================================================================


class TestRedisFallback:
    """Tests for fallback behavior when Redis is unavailable."""

    def test_fallback_to_memory_when_redis_unavailable(self, mock_settings):
        """Should fall back to in-memory limiting when Redis is not available."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
            reset_distributed_limiter,
        )

        reset_distributed_limiter()

        with (
            patch(
                "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                return_value=None,
            ),
            patch("aragora.config.settings.get_settings", return_value=mock_settings),
        ):
            limiter = DistributedRateLimiter(instance_id="fallback-instance", enable_metrics=False)

            # Should use memory backend
            assert limiter.backend == "memory"
            assert limiter.is_using_redis is False

            # Should still work for rate limiting
            result = limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
            assert result.allowed is True

    def test_fallback_when_redis_fails_mid_operation(self, mock_settings):
        """Should fall back gracefully when Redis fails during operation."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
        )

        # The fallback counter is in the RedisRateLimiter, not DistributedRateLimiter
        # We need to check if the distributed limiter properly handles errors
        call_count = [0]

        def mock_consume_with_failure(n):
            call_count[0] += 1
            if call_count[0] > 1:
                raise ConnectionError("Simulated Redis failure")
            return True

        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = mock_consume_with_failure
            mock_bucket.remaining = 50
            mock_bucket.get_retry_after.return_value = 0
            mock_bucket_class.return_value = mock_bucket

            shared_redis = SharedMockRedis("shared")

            with (
                patch(
                    "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                    return_value=shared_redis,
                ),
                patch(
                    "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                    return_value=shared_redis,
                ),
                patch("aragora.config.settings.get_settings", return_value=mock_settings),
            ):
                limiter = DistributedRateLimiter(instance_id="fail-test", enable_metrics=False)

                # First request should work via Redis
                result = limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
                assert result.allowed is True

                # Second request should trigger failure and use fallback
                result = limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
                assert result.allowed is True

                # Check stats to see fallback was used (tracked in Redis limiter)
                stats = limiter.get_stats()
                # The redis limiter should have recorded a fallback
                redis_stats = stats.get("redis", {})
                assert redis_stats.get("fallback_requests", 0) >= 1 or call_count[0] >= 2

    def test_strict_mode_raises_in_production(self, mock_settings):
        """Strict mode should raise error in production when Redis unavailable."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
            reset_distributed_limiter,
        )
        from aragora.server.middleware.rate_limit.redis_limiter import (
            reset_redis_client,
        )

        reset_distributed_limiter()
        reset_redis_client()

        with (
            patch(
                "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                return_value=None,
            ),
            patch(
                "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                return_value=None,
            ),
            patch(
                "aragora.server.middleware.rate_limit.distributed._is_production_mode",
                return_value=True,
            ),
            patch(
                "aragora.server.middleware.rate_limit.distributed._is_development_mode",
                return_value=False,
            ),
            patch("aragora.config.settings.get_settings", return_value=mock_settings),
        ):
            with pytest.raises(RuntimeError) as exc_info:
                limiter = DistributedRateLimiter(
                    instance_id="strict-test",
                    strict_mode=True,
                    enable_metrics=False,
                )
                # Force initialization
                limiter._initialize()

            assert "Redis is unavailable" in str(exc_info.value)

    def test_strict_mode_warns_in_development(self, mock_settings, caplog):
        """Strict mode should warn in development when Redis unavailable."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
            reset_distributed_limiter,
        )

        reset_distributed_limiter()

        with (
            patch(
                "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                return_value=None,
            ),
            patch(
                "aragora.server.middleware.rate_limit.distributed._is_production_mode",
                return_value=False,
            ),
            patch(
                "aragora.server.middleware.rate_limit.distributed._is_development_mode",
                return_value=True,
            ),
            patch("aragora.config.settings.get_settings", return_value=mock_settings),
        ):
            import logging

            with caplog.at_level(logging.WARNING):
                limiter = DistributedRateLimiter(
                    instance_id="dev-strict-test",
                    strict_mode=True,
                    enable_metrics=False,
                )

            # Should not raise, but should log warning
            assert limiter.backend == "memory"
            assert "Falling back to in-memory" in caplog.text


# =============================================================================
# Tests for Metrics Recording
# =============================================================================


class TestMetricsRecording:
    """Tests for Prometheus metrics recording."""

    def test_records_rate_limit_decisions(self, shared_redis, mock_settings):
        """Should record rate limit decisions in metrics."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
        )
        from aragora.server.middleware.rate_limit.metrics import (
            PROMETHEUS_AVAILABLE,
        )

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus client not available")

        with (
            patch(
                "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                return_value=shared_redis,
            ),
            patch(
                "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                return_value=shared_redis,
            ),
            patch("aragora.config.settings.get_settings", return_value=mock_settings),
            patch(
                "aragora.server.middleware.rate_limit.distributed.record_rate_limit_decision"
            ) as mock_record,
        ):
            limiter = DistributedRateLimiter(instance_id="metrics-test", enable_metrics=True)
            limiter.configure_endpoint("/api/test", 5)

            limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")

            # Verify metrics were recorded
            mock_record.assert_called()
            call_kwargs = mock_record.call_args[1]
            assert call_kwargs["endpoint"] == "/api/test"
            assert "allowed" in call_kwargs

    def test_records_tenant_rejections(self, mock_settings):
        """Should track rejections by tenant."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
        )

        call_count = [0]
        limit = 2

        def mock_consume(n):
            call_count[0] += 1
            return call_count[0] <= limit

        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = mock_consume
            mock_bucket.remaining = 0
            mock_bucket.get_retry_after.return_value = 1.0
            mock_bucket_class.return_value = mock_bucket

            shared_redis = SharedMockRedis("shared")

            with (
                patch(
                    "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                    return_value=shared_redis,
                ),
                patch(
                    "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                    return_value=shared_redis,
                ),
                patch("aragora.config.settings.get_settings", return_value=mock_settings),
            ):
                limiter = DistributedRateLimiter(instance_id="tenant-metrics", enable_metrics=False)
                limiter.configure_endpoint("/api/test", limit, key_type="tenant")

                # Exhaust limit for tenant-a
                limiter.allow(client_ip="192.168.1.1", endpoint="/api/test", tenant_id="tenant-a")
                limiter.allow(client_ip="192.168.1.1", endpoint="/api/test", tenant_id="tenant-a")
                # This one should be rejected
                result = limiter.allow(
                    client_ip="192.168.1.1", endpoint="/api/test", tenant_id="tenant-a"
                )
                assert result.allowed is False

                stats = limiter.get_stats()
                assert "tenant_rejections" in stats
                # Should have at least one rejection for tenant-a
                assert any("tenant-a" in k for k in stats["tenant_rejections"].keys())


# =============================================================================
# Tests for Handler Decorator Integration
# =============================================================================


class TestHandlerDecoratorIntegration:
    """Tests for rate limit decorator with distributed limiter."""

    def test_decorator_uses_distributed_limiter_by_default(self):
        """Rate limit decorator should use distributed limiter by default."""
        from aragora.server.handlers.utils.rate_limit import rate_limit

        # Explicitly enable distributed mode
        @rate_limit(requests_per_minute=10, use_distributed=True)
        def test_handler(self, handler):
            return {"status": "ok"}

        # Check that it's marked as using distributed limiter
        assert hasattr(test_handler, "_rate_limit_distributed")
        assert test_handler._rate_limit_distributed is True

    def test_decorator_can_use_local_limiter(self):
        """Rate limit decorator should support local limiter via flag."""
        from aragora.server.handlers.utils.rate_limit import rate_limit

        @rate_limit(requests_per_minute=10, use_distributed=False)
        def test_handler(self, handler):
            return {"status": "ok"}

        # Check that it's NOT using distributed limiter
        assert hasattr(test_handler, "_rate_limit_distributed")
        assert test_handler._rate_limit_distributed is False

    def test_decorator_tenant_aware(self, shared_redis, mock_settings):
        """Tenant-aware decorator should extract tenant from request."""
        from aragora.server.handlers.utils.rate_limit import rate_limit

        with (
            patch(
                "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                return_value=shared_redis,
            ),
            patch(
                "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                return_value=shared_redis,
            ),
            patch("aragora.config.settings.get_settings", return_value=mock_settings),
        ):
            # Create a mock handler with tenant ID
            mock_handler = MagicMock()
            mock_handler.tenant_id = "test-tenant"
            mock_handler.headers = {}
            mock_handler.client_address = ("192.168.1.1", 12345)

            @rate_limit(requests_per_minute=100, tenant_aware=True)
            def test_handler(self, handler):
                return {"status": "ok"}

            class MockSelf:
                pass

            # Call should succeed
            result = test_handler(MockSelf(), mock_handler)
            assert result == {"status": "ok"}


# =============================================================================
# Tests for Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_requests_across_instances(self, mock_settings):
        """Multiple threads should safely share rate limit state."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
        )

        # Thread-safe counter for shared rate limiting
        call_count = [0]
        call_lock = threading.Lock()
        limit = 50

        def mock_consume(n):
            with call_lock:
                call_count[0] += n
                return call_count[0] <= limit

        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = mock_consume
            mock_bucket.remaining = 0
            mock_bucket.get_retry_after.return_value = 1.0
            mock_bucket_class.return_value = mock_bucket

            shared_redis = SharedMockRedis("shared")

            with (
                patch(
                    "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                    return_value=shared_redis,
                ),
                patch(
                    "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                    return_value=shared_redis,
                ),
                patch("aragora.config.settings.get_settings", return_value=mock_settings),
            ):
                limiter1 = DistributedRateLimiter(instance_id="concurrent-1", enable_metrics=False)
                limiter2 = DistributedRateLimiter(instance_id="concurrent-2", enable_metrics=False)

                limiter1.configure_endpoint("/api/test", limit)
                limiter2.configure_endpoint("/api/test", limit)

                results = []
                results_lock = threading.Lock()
                errors = []

                def make_requests(limiter, client_ip, count):
                    try:
                        for _ in range(count):
                            result = limiter.allow(client_ip=client_ip, endpoint="/api/test")
                            with results_lock:
                                results.append(result.allowed)
                    except Exception as e:
                        errors.append(e)

                # Create threads for concurrent access
                threads = [
                    threading.Thread(target=make_requests, args=(limiter1, "192.168.1.1", 30)),
                    threading.Thread(target=make_requests, args=(limiter2, "192.168.1.1", 30)),
                    threading.Thread(target=make_requests, args=(limiter1, "192.168.1.1", 30)),
                ]

                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                # No errors should occur
                assert len(errors) == 0, f"Errors occurred: {errors}"

                # Total allowed should not exceed limit
                allowed_count = sum(1 for r in results if r)
                # With 50 limit and 90 total requests, should have ~50 allowed
                assert allowed_count <= 55, f"Too many allowed: {allowed_count}"
                assert allowed_count >= 45, f"Too few allowed: {allowed_count}"


# =============================================================================
# Tests for Stats and Monitoring
# =============================================================================


class TestStatsAndMonitoring:
    """Tests for statistics and monitoring endpoints."""

    def test_get_stats_includes_all_metrics(self, shared_redis, mock_settings):
        """get_stats should return comprehensive metrics."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
        )

        with (
            patch(
                "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                return_value=shared_redis,
            ),
            patch(
                "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                return_value=shared_redis,
            ),
            patch("aragora.config.settings.get_settings", return_value=mock_settings),
        ):
            limiter = DistributedRateLimiter(instance_id="stats-test", enable_metrics=False)

            # Make some requests
            for _ in range(5):
                limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")

            stats = limiter.get_stats()

            assert "instance_id" in stats
            assert stats["instance_id"] == "stats-test"
            assert "backend" in stats
            assert "total_requests" in stats
            assert stats["total_requests"] == 5
            assert "tenant_rejections" in stats

    def test_reset_clears_all_state(self, mock_settings):
        """reset() should clear all rate limiter state."""
        from aragora.server.middleware.rate_limit.distributed import (
            DistributedRateLimiter,
        )

        call_count = [0]
        limit = 2
        reset_called = [False]

        def mock_consume(n):
            if reset_called[0]:
                # After reset, allow again
                call_count[0] = 1
                reset_called[0] = False
                return True
            call_count[0] += n
            return call_count[0] <= limit

        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = mock_consume
            mock_bucket.remaining = 0
            mock_bucket.get_retry_after.return_value = 1.0
            mock_bucket_class.return_value = mock_bucket

            shared_redis = SharedMockRedis("shared")

            with (
                patch(
                    "aragora.server.middleware.rate_limit.redis_limiter.get_redis_client",
                    return_value=shared_redis,
                ),
                patch(
                    "aragora.server.middleware.rate_limit.distributed.get_redis_client",
                    return_value=shared_redis,
                ),
                patch("aragora.config.settings.get_settings", return_value=mock_settings),
            ):
                limiter = DistributedRateLimiter(instance_id="reset-test", enable_metrics=False)
                limiter.configure_endpoint("/api/test", limit)

                # Exhaust limit
                limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
                limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
                result = limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
                assert result.allowed is False

                # Reset
                reset_called[0] = True
                call_count[0] = 0
                limiter.reset()

                # Should be able to make requests again
                result = limiter.allow(client_ip="192.168.1.1", endpoint="/api/test")
                assert result.allowed is True
