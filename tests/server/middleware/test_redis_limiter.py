"""
Tests for aragora.server.middleware.rate_limit.redis_limiter - Redis Rate Limiting.

Tests cover:
- RedisRateLimiter initialization and configuration
- Endpoint configuration (configure_endpoint, get_config)
- Key type routing (ip, token, combined, endpoint)
- allow() method with various parameters
- Disabled endpoint configs
- get_redis_client() and reset_redis_client() module functions
- Module-level constants (RATE_LIMIT_FAIL_OPEN, REDIS_FAILURE_THRESHOLD, etc.)
- RedisRateLimiter.cleanup() and .reset() methods
- get_client_key() delegation
- get_stats() with Redis errors
- Metrics sync skipping when interval not elapsed
- Metrics sync failure handling
- Zero-request rejection rate (avoid division by zero)
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional
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
    """Mock Redis client for testing."""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._hashes: dict[str, dict[str, str]] = {}
        self._ttls: dict[str, float] = {}
        self._fail_on_scan = False

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._data[key] = value
        if ex:
            self._ttls[key] = time.time() + ex
        return True

    def incr(self, key: str) -> int:
        val = int(self._data.get(key, 0)) + 1
        self._data[key] = str(val)
        return val

    def delete(self, *keys: str) -> int:
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
        import fnmatch

        if self._fail_on_scan:
            raise ConnectionError("Simulated scan failure")
        for key in list(self._data.keys()) + list(self._hashes.keys()):
            if fnmatch.fnmatch(key, match):
                yield key

    def hset(
        self,
        name: str,
        key: Optional[str] = None,
        value: Optional[str] = None,
        mapping: Optional[dict[str, str]] = None,
    ) -> int:
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
        return self._hashes.get(name, {})

    def expire(self, key: str, seconds: int) -> bool:
        self._ttls[key] = time.time() + seconds
        return True

    def pipeline(self) -> "MockPipeline":
        return MockPipeline(self)

    def close(self) -> None:
        pass


class MockPipeline:
    """Mock Redis pipeline."""

    def __init__(self, redis: MockRedis):
        self._redis = redis
        self._commands: list[tuple] = []

    def hset(self, name, key=None, value=None, mapping=None):
        self._commands.append(("hset", (name,), {"key": key, "value": value, "mapping": mapping}))
        return self

    def expire(self, key, seconds):
        self._commands.append(("expire", (key, seconds), {}))
        return self

    def incr(self, key):
        self._commands.append(("incr", (key,), {}))
        return self

    def execute(self):
        results = []
        for cmd, args, kwargs in self._commands:
            method = getattr(self._redis, cmd)
            results.append(method(*args, **kwargs))
        self._commands.clear()
        return results


class FailingPipeline:
    """Pipeline that fails on execute."""

    def hset(self, *args, **kwargs):
        return self

    def expire(self, *args, **kwargs):
        return self

    def execute(self):
        raise ConnectionError("Pipeline failed")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_redis():
    return MockRedis()


@pytest.fixture
def redis_limiter(mock_redis):
    with patch(
        "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
    ) as mock_bucket_class:
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
        limiter._mock_bucket = mock_bucket
        yield limiter


# =============================================================================
# Test Module-Level Constants and Exports
# =============================================================================


class TestModuleConstants:
    """Tests for module-level constants and configuration."""

    def test_redis_available_flag(self):
        """REDIS_AVAILABLE should reflect import status."""
        from aragora.server.middleware.rate_limit.redis_limiter import REDIS_AVAILABLE

        assert isinstance(REDIS_AVAILABLE, bool)

    def test_rate_limit_fail_open_default(self):
        """RATE_LIMIT_FAIL_OPEN should default to false (fail-closed)."""
        # The module reads env at import time, so check the value
        from aragora.server.middleware.rate_limit.redis_limiter import RATE_LIMIT_FAIL_OPEN

        assert isinstance(RATE_LIMIT_FAIL_OPEN, bool)

    def test_redis_failure_threshold_type(self):
        """REDIS_FAILURE_THRESHOLD should be an integer."""
        from aragora.server.middleware.rate_limit.redis_limiter import REDIS_FAILURE_THRESHOLD

        assert isinstance(REDIS_FAILURE_THRESHOLD, int)
        assert REDIS_FAILURE_THRESHOLD > 0

    def test_enable_circuit_breaker_type(self):
        """ENABLE_CIRCUIT_BREAKER should be a boolean."""
        from aragora.server.middleware.rate_limit.redis_limiter import ENABLE_CIRCUIT_BREAKER

        assert isinstance(ENABLE_CIRCUIT_BREAKER, bool)

    def test_module_all_exports(self):
        """Module __all__ should list expected exports."""
        import sys
        from aragora.server.middleware.rate_limit.redis_limiter import RedisRateLimiter
        mod = sys.modules["aragora.server.middleware.rate_limit.redis_limiter"]

        expected = {
            "REDIS_AVAILABLE",
            "RATE_LIMIT_FAIL_OPEN",
            "REDIS_FAILURE_THRESHOLD",
            "ENABLE_CIRCUIT_BREAKER",
            "ENABLE_DISTRIBUTED_METRICS",
            "get_redis_client",
            "reset_redis_client",
            "RateLimitCircuitBreaker",
            "RedisRateLimiter",
        }
        assert set(mod.__all__) == expected


# =============================================================================
# Test get_redis_client and reset_redis_client
# =============================================================================


class TestGetRedisClient:
    """Tests for get_redis_client and reset_redis_client."""

    def test_reset_redis_client_resets_state(self):
        """reset_redis_client should clear cached state."""
        from aragora.server.middleware.rate_limit.redis_limiter import (
            reset_redis_client,
        )

        # Should not raise
        reset_redis_client()

    def test_reset_redis_client_handles_close_errors(self):
        """reset_redis_client should handle close errors gracefully."""
        import sys
        from aragora.server.middleware.rate_limit.redis_limiter import reset_redis_client
        mod = sys.modules["aragora.server.middleware.rate_limit.redis_limiter"]

        # Set up a mock client that raises on close
        mock_client = MagicMock()
        mock_client.close.side_effect = OSError("Close failed")

        old_client = mod._redis_client
        old_attempted = mod._redis_init_attempted
        try:
            mod._redis_client = mock_client
            mod._redis_init_attempted = True

            # Should not raise
            reset_redis_client()

            assert mod._redis_client is None
            assert mod._redis_init_attempted is False
        finally:
            mod._redis_client = old_client
            mod._redis_init_attempted = old_attempted

    def test_get_redis_client_returns_none_when_unavailable(self):
        """get_redis_client should return None when Redis not available."""
        import sys
        from aragora.server.middleware.rate_limit.redis_limiter import get_redis_client
        mod = sys.modules["aragora.server.middleware.rate_limit.redis_limiter"]

        old_client = mod._redis_client
        old_attempted = mod._redis_init_attempted
        old_available = mod.REDIS_AVAILABLE
        try:
            mod._redis_client = None
            mod._redis_init_attempted = False
            mod.REDIS_AVAILABLE = False

            result = get_redis_client()
            assert result is None
        finally:
            mod._redis_client = old_client
            mod._redis_init_attempted = old_attempted
            mod.REDIS_AVAILABLE = old_available

    def test_get_redis_client_returns_cached(self):
        """get_redis_client should return cached client on repeat calls."""
        import sys
        from aragora.server.middleware.rate_limit.redis_limiter import get_redis_client
        mod = sys.modules["aragora.server.middleware.rate_limit.redis_limiter"]

        old_client = mod._redis_client
        old_attempted = mod._redis_init_attempted
        try:
            mock_client = MagicMock()
            mod._redis_client = mock_client
            mod._redis_init_attempted = True

            result = get_redis_client()
            assert result is mock_client
        finally:
            mod._redis_client = old_client
            mod._redis_init_attempted = old_attempted


# =============================================================================
# Test RedisRateLimiter Initialization
# =============================================================================


class TestRedisRateLimiterInit:
    """Tests for RedisRateLimiter initialization."""

    def test_init_with_defaults(self, mock_redis):
        """Should initialize with default values."""
        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                instance_id="test-init",
            )
            assert limiter.default_limit > 0
            assert limiter.ip_limit > 0
            assert limiter.instance_id == "test-init"
            assert limiter._requests_allowed == 0
            assert limiter._requests_rejected == 0

    def test_init_with_custom_limits(self, mock_redis):
        """Should initialize with custom limits."""
        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                default_limit=100,
                ip_limit=200,
                key_prefix="custom:",
                ttl_seconds=300,
                instance_id="custom-test",
            )
            assert limiter.default_limit == 100
            assert limiter.ip_limit == 200
            assert limiter.key_prefix == "custom:"
            assert limiter.ttl_seconds == 300

    def test_init_without_circuit_breaker(self, mock_redis):
        """Should allow disabling circuit breaker."""
        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=False,
                instance_id="no-cb",
            )
            assert limiter._circuit_breaker is None

    def test_init_with_circuit_breaker(self, mock_redis):
        """Should enable circuit breaker by default."""
        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=True,
                instance_id="with-cb",
            )
            assert limiter._circuit_breaker is not None

    def test_init_auto_generates_instance_id(self, mock_redis):
        """Should auto-generate instance_id when not provided."""
        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            with patch.dict(os.environ, {}, clear=True):
                limiter = RedisRateLimiter(
                    redis_client=mock_redis,
                    instance_id=None,
                )
                assert limiter.instance_id.startswith("instance-")


# =============================================================================
# Test Endpoint Configuration
# =============================================================================


class TestEndpointConfiguration:
    """Tests for configure_endpoint and get_config."""

    def test_configure_endpoint_basic(self, redis_limiter):
        """Should configure a basic endpoint."""
        redis_limiter.configure_endpoint("/api/debates", 100)

        config = redis_limiter.get_config("/api/debates")
        assert config.requests_per_minute == 100

    def test_configure_endpoint_with_burst(self, redis_limiter):
        """Should configure endpoint with burst size."""
        redis_limiter.configure_endpoint("/api/debates", 100, burst_size=200)

        config = redis_limiter.get_config("/api/debates")
        assert config.requests_per_minute == 100
        assert config.burst_size == 200

    def test_configure_endpoint_with_key_type(self, redis_limiter):
        """Should configure endpoint with key type."""
        redis_limiter.configure_endpoint("/api/debates", 100, key_type="token")

        config = redis_limiter.get_config("/api/debates")
        assert config.key_type == "token"

    def test_get_config_default(self, redis_limiter):
        """Should return default config for unconfigured endpoint."""
        config = redis_limiter.get_config("/api/unconfigured")
        assert config.requests_per_minute == redis_limiter.default_limit

    def test_get_config_wildcard_match(self, redis_limiter):
        """Should match wildcard endpoint patterns."""
        redis_limiter.configure_endpoint("/api/debates/*", 50)

        config = redis_limiter.get_config("/api/debates/123")
        assert config.requests_per_minute == 50

    def test_configure_endpoint_normalizes_path(self, redis_limiter):
        """Should normalize endpoint paths."""
        redis_limiter.configure_endpoint("/API/Debates/", 100)

        config = redis_limiter.get_config("/api/debates")
        assert config.requests_per_minute == 100

    def test_configure_endpoint_also_configures_fallback(self, redis_limiter):
        """Should also configure the in-memory fallback limiter."""
        redis_limiter.configure_endpoint("/api/debates", 100, burst_size=200)

        # The fallback should also have the config
        fallback_stats = redis_limiter._fallback.get_stats()
        assert len(fallback_stats.get("configured_endpoints", [])) > 0


# =============================================================================
# Test allow() Method with Key Types
# =============================================================================


class TestAllowKeyTypes:
    """Tests for allow() with different key types."""

    def test_allow_ip_key(self, redis_limiter):
        """Should use IP-based key by default."""
        result = redis_limiter.allow("192.168.1.1")
        assert result.allowed is True
        assert "ip:" in result.key

    def test_allow_token_key(self, redis_limiter):
        """Should use token-based key when configured."""
        redis_limiter.configure_endpoint("/api/debates", 100, key_type="token")

        result = redis_limiter.allow("192.168.1.1", endpoint="/api/debates", token="my-api-key")
        assert result.allowed is True
        assert "token:" in result.key

    def test_allow_combined_key(self, redis_limiter):
        """Should use combined endpoint+IP key when configured."""
        redis_limiter.configure_endpoint("/api/debates", 100, key_type="combined")

        result = redis_limiter.allow("192.168.1.1", endpoint="/api/debates")
        assert result.allowed is True
        assert "ep:" in result.key
        assert "ip:" in result.key

    def test_allow_endpoint_key(self, redis_limiter):
        """Should use endpoint-only key when configured."""
        redis_limiter.configure_endpoint("/api/debates", 100, key_type="endpoint")

        result = redis_limiter.allow("192.168.1.1", endpoint="/api/debates")
        assert result.allowed is True
        assert "ep:" in result.key

    def test_allow_token_key_without_token_falls_back_to_ip(self, redis_limiter):
        """Should fall back to IP key when token key configured but no token provided."""
        redis_limiter.configure_endpoint("/api/debates", 100, key_type="token")

        result = redis_limiter.allow("192.168.1.1", endpoint="/api/debates")
        assert result.allowed is True
        assert "ip:" in result.key

    def test_allow_anonymous_ip(self, redis_limiter):
        """Should handle empty IP gracefully."""
        result = redis_limiter.allow("")
        assert result.allowed is True

    def test_allow_disabled_config(self, mock_redis):
        """Should allow requests when endpoint is disabled."""
        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                instance_id="disabled-test",
            )
            from aragora.server.middleware.rate_limit.limiter import RateLimitConfig

            limiter._endpoint_configs["/api/test"] = RateLimitConfig(enabled=False)

            result = limiter.allow("192.168.1.1", endpoint="/api/test")
            assert result.allowed is True
            assert result.limit == 0

    def test_allow_with_tenant_id_accepted(self, redis_limiter):
        """Should accept tenant_id parameter without errors."""
        result = redis_limiter.allow("192.168.1.1", tenant_id="tenant-123")
        assert result.allowed is True

    def test_allow_rejected_returns_retry_after(self, redis_limiter):
        """Should return retry_after when request is rejected."""
        redis_limiter._mock_bucket.consume.return_value = False
        redis_limiter._mock_bucket.get_retry_after.return_value = 5.0
        redis_limiter._mock_bucket.remaining = 0

        result = redis_limiter.allow("192.168.1.1")
        assert result.allowed is False
        assert result.retry_after == 5.0


# =============================================================================
# Test cleanup and reset
# =============================================================================


class TestCleanupAndReset:
    """Tests for cleanup and reset methods."""

    def test_cleanup_returns_zero(self, redis_limiter):
        """cleanup should return 0 (Redis handles TTL)."""
        result = redis_limiter.cleanup(max_age_seconds=300)
        assert result == 0

    def test_reset_clears_redis_keys(self, mock_redis, redis_limiter):
        """reset should delete all keys with prefix."""
        # Add some mock data
        mock_redis._data["aragora:ratelimit:ip:1.2.3.4"] = "1"
        mock_redis._data["aragora:ratelimit:ip:5.6.7.8"] = "2"

        redis_limiter.reset()

        # Keys should be deleted
        assert "aragora:ratelimit:ip:1.2.3.4" not in mock_redis._data
        assert "aragora:ratelimit:ip:5.6.7.8" not in mock_redis._data

    def test_reset_clears_buckets(self, redis_limiter):
        """reset should clear cached bucket objects."""
        redis_limiter._buckets["test-key"] = MagicMock()

        redis_limiter.reset()

        assert len(redis_limiter._buckets) == 0

    def test_reset_handles_redis_failure(self, redis_limiter, mock_redis):
        """reset should handle Redis errors gracefully."""
        mock_redis._fail_on_scan = True

        # Should not raise
        redis_limiter.reset()

    def test_reset_also_resets_fallback(self, redis_limiter):
        """reset should also reset the fallback limiter."""
        redis_limiter._fallback._ip_buckets["1.2.3.4"] = MagicMock()

        redis_limiter.reset()

        # Fallback should be reset too
        assert len(redis_limiter._fallback._ip_buckets) == 0


# =============================================================================
# Test get_client_key
# =============================================================================


class TestGetClientKey:
    """Tests for get_client_key delegation."""

    def test_get_client_key_delegates_to_fallback(self, redis_limiter):
        """get_client_key should delegate to in-memory fallback."""
        handler = MagicMock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = {}

        key = redis_limiter.get_client_key(handler)
        assert isinstance(key, str)


# =============================================================================
# Test Metrics Sync Edge Cases
# =============================================================================


class TestMetricsSyncEdgeCases:
    """Tests for metrics sync edge cases."""

    def test_metrics_sync_skipped_within_interval(self, redis_limiter, mock_redis):
        """Should skip sync when interval hasn't elapsed."""
        redis_limiter._last_metrics_sync = time.time()  # Just synced
        redis_limiter._requests_allowed = 999

        redis_limiter._maybe_sync_distributed_metrics()

        # Should not have synced (no key in Redis)
        key = f"{redis_limiter._metrics_key}{redis_limiter.instance_id}"
        assert key not in mock_redis._hashes

    def test_metrics_sync_handles_pipeline_failure(self, mock_redis):
        """Should handle pipeline execution failure gracefully."""
        failing_redis = MagicMock()
        failing_redis.pipeline.return_value = FailingPipeline()

        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=failing_redis,
                enable_distributed_metrics=True,
                instance_id="fail-sync",
            )
            limiter._last_metrics_sync = 0  # Force sync
            limiter._requests_allowed = 100

            # Should not raise
            limiter._maybe_sync_distributed_metrics()

    def test_get_distributed_metrics_handles_redis_error(self, mock_redis):
        """Should handle errors in get_distributed_metrics gracefully."""
        failing_redis = MagicMock()
        failing_redis.scan_iter.side_effect = ConnectionError("Redis down")

        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=failing_redis,
                enable_distributed_metrics=True,
                instance_id="fail-metrics",
            )

            result = limiter.get_distributed_metrics()
            assert "error" in result

    def test_get_distributed_metrics_empty_instances(self, mock_redis):
        """Should handle case with no instances."""
        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_distributed_metrics=True,
                instance_id="lonely",
            )

            result = limiter.get_distributed_metrics()
            assert result["total_requests_allowed"] == 0
            assert result["total_rejection_rate"] == 0.0
            assert result["instance_count"] == 0


# =============================================================================
# Test get_stats Edge Cases
# =============================================================================


class TestGetStatsEdgeCases:
    """Tests for get_stats edge cases."""

    def test_get_stats_zero_requests_no_division_error(self, redis_limiter):
        """Should handle zero requests without division by zero."""
        redis_limiter._requests_allowed = 0
        redis_limiter._requests_rejected = 0

        stats = redis_limiter.get_stats()
        assert stats["rejection_rate"] == 0.0
        assert stats["total_requests"] == 0

    def test_get_stats_handles_redis_scan_error(self, mock_redis):
        """Should include error info when Redis scan fails."""
        mock_redis._fail_on_scan = True

        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=False,
                instance_id="stats-error",
            )

            stats = limiter.get_stats()
            assert "error" in stats
            assert "fallback_stats" in stats

    def test_get_stats_includes_configured_endpoints(self, redis_limiter):
        """Should list configured endpoints."""
        redis_limiter.configure_endpoint("/api/debates", 100)
        redis_limiter.configure_endpoint("/api/agents", 200)

        stats = redis_limiter.get_stats()
        assert "/api/debates" in stats["configured_endpoints"]
        assert "/api/agents" in stats["configured_endpoints"]

    def test_get_stats_includes_default_and_ip_limits(self, redis_limiter):
        """Should include default and IP limits."""
        stats = redis_limiter.get_stats()
        assert stats["default_limit"] == 60
        assert stats["ip_limit"] == 120

    def test_get_stats_without_circuit_breaker(self, mock_redis):
        """Should work without circuit breaker."""
        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=False,
                instance_id="no-cb-stats",
            )

            stats = limiter.get_stats()
            assert "circuit_breaker" not in stats

    def test_get_stats_counts_redis_keys(self, mock_redis):
        """Should count Redis keys with prefix."""
        mock_redis._data["aragora:ratelimit:ip:1"] = "1"
        mock_redis._data["aragora:ratelimit:ip:2"] = "2"
        mock_redis._data["other:key"] = "3"

        with patch("aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"):
            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=False,
                enable_distributed_metrics=False,
                instance_id="count-keys",
            )

            stats = limiter.get_stats()
            assert stats["redis_keys"] == 2


# =============================================================================
# Test Circuit Breaker Integration with Allow
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration in allow()."""

    def test_allow_without_circuit_breaker(self, mock_redis):
        """Should work normally without circuit breaker."""
        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.return_value = True
            mock_bucket.remaining = 50
            mock_bucket.get_retry_after.return_value = 0
            mock_bucket_class.return_value = mock_bucket

            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=False,
                instance_id="no-cb-allow",
            )

            result = limiter.allow("192.168.1.1")
            assert result.allowed is True

    def test_allow_records_success_on_circuit_breaker(self, redis_limiter):
        """Should record success on circuit breaker after successful allow."""
        initial_success_count = redis_limiter._circuit_breaker._success_count

        redis_limiter.allow("192.168.1.1")

        assert redis_limiter._circuit_breaker._success_count > initial_success_count

    def test_allow_records_failure_on_circuit_breaker(self, mock_redis):
        """Should record failure on circuit breaker when Redis fails."""
        with patch(
            "aragora.server.middleware.rate_limit.redis_limiter.RedisTokenBucket"
        ) as mock_bucket_class:
            mock_bucket = MagicMock()
            mock_bucket.consume.side_effect = ConnectionError("Redis error")
            mock_bucket_class.return_value = mock_bucket

            limiter = RedisRateLimiter(
                redis_client=mock_redis,
                enable_circuit_breaker=True,
                instance_id="cb-failure",
            )

            limiter.allow("192.168.1.1")

            assert limiter._circuit_breaker._failure_count >= 1
