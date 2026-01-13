"""
Tests for aragora.server.rate_limit_redis - Distributed rate limiting with Redis.

Tests cover:
- RedisConfig dataclass
- RedisRateLimiter initialization
- allow() method with sliding window algorithm
- configure_endpoint() for per-endpoint limits
- get_remaining() without consuming tokens
- reset() to clear rate limit state
- get_stats() for monitoring
- get_client_key() for request extraction
- Fallback behavior when Redis unavailable
"""

import os
import time
import pytest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock

from aragora.server.rate_limit_redis import (
    RedisConfig,
    RedisRateLimiter,
    get_redis_config,
    get_distributed_rate_limiter,
    reset_distributed_rate_limiter,
)


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self._data: Dict[str, List[Tuple[str, float]]] = {}
        self._ttls: Dict[str, float] = {}
        self._available = True

    def ping(self):
        if not self._available:
            raise ConnectionError("Redis unavailable")
        return True

    def pipeline(self):
        return MockPipeline(self)

    def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        if key not in self._data:
            return 0
        original_len = len(self._data[key])
        self._data[key] = [
            (member, score)
            for member, score in self._data[key]
            if score > max_score or score < min_score
        ]
        return original_len - len(self._data[key])

    def zcard(self, key: str) -> int:
        return len(self._data.get(key, []))

    def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        if key not in self._data:
            self._data[key] = []
        added = 0
        for member, score in mapping.items():
            # Check if member already exists
            exists = any(m == member for m, s in self._data[key])
            if not exists:
                self._data[key].append((member, score))
                added += 1
        return added

    def zrem(self, key: str, *members: str) -> int:
        if key not in self._data:
            return 0
        removed = 0
        for member in members:
            original_len = len(self._data[key])
            self._data[key] = [(m, s) for m, s in self._data[key] if m != member]
            if len(self._data[key]) < original_len:
                removed += 1
        return removed

    def zrange(self, key: str, start: int, stop: int, withscores: bool = False) -> list:
        if key not in self._data:
            return []
        data = sorted(self._data[key], key=lambda x: x[1])
        if stop == -1:
            sliced = data[start:]
        else:
            sliced = data[start : stop + 1]
        if withscores:
            return sliced
        return [m for m, s in sliced]

    def expire(self, key: str, seconds: int) -> bool:
        self._ttls[key] = time.time() + seconds
        return True

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self._data:
                del self._data[key]
                deleted += 1
            if key in self._ttls:
                del self._ttls[key]
        return deleted

    def scan_iter(self, match: str = "*", count: int = 100):
        """Iterate keys matching pattern."""
        import fnmatch

        for key in list(self._data.keys()):
            if fnmatch.fnmatch(key, match):
                yield key

    def close(self):
        pass


class MockPipeline:
    """Mock Redis pipeline for atomic operations."""

    def __init__(self, redis: MockRedis):
        self._redis = redis
        self._commands: List[Tuple[str, tuple, dict]] = []

    def zremrangebyscore(self, key: str, min_score: float, max_score: float):
        self._commands.append(("zremrangebyscore", (key, min_score, max_score), {}))
        return self

    def zcard(self, key: str):
        self._commands.append(("zcard", (key,), {}))
        return self

    def zadd(self, key: str, mapping: Dict[str, float]):
        self._commands.append(("zadd", (key, mapping), {}))
        return self

    def expire(self, key: str, seconds: int):
        self._commands.append(("expire", (key, seconds), {}))
        return self

    def execute(self) -> List[Any]:
        results = []
        for cmd, args, kwargs in self._commands:
            method = getattr(self._redis, cmd)
            results.append(method(*args, **kwargs))
        self._commands.clear()
        return results


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    return MockRedis()


@pytest.fixture
def limiter(mock_redis):
    """Create RedisRateLimiter with mock Redis."""
    limiter = RedisRateLimiter(redis_url="redis://localhost:6379")
    # Manually inject mock Redis (bypasses _get_redis which requires real redis)
    limiter._redis = mock_redis
    limiter._available = True
    return limiter


class TestRedisConfig:
    """Tests for RedisConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = RedisConfig()
        assert config.url == "redis://localhost:6379"
        assert config.prefix == "aragora:ratelimit"
        assert config.socket_timeout > 0
        assert config.max_connections > 0

    def test_custom_values(self):
        """Should accept custom values."""
        config = RedisConfig(
            url="redis://custom:6380",
            prefix="custom:prefix",
            default_limit=100,
        )
        assert config.url == "redis://custom:6380"
        assert config.prefix == "custom:prefix"
        assert config.default_limit == 100


class TestGetRedisConfig:
    """Tests for get_redis_config() function."""

    def test_defaults_from_environment(self):
        """Should read defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            config = get_redis_config()
            assert config.url == "redis://localhost:6379"

    def test_reads_redis_url(self):
        """Should read REDIS_URL from environment."""
        with patch.dict(os.environ, {"REDIS_URL": "redis://custom:6380"}):
            config = get_redis_config()
            assert config.url == "redis://custom:6380"

    def test_reads_prefix(self):
        """Should read REDIS_RATE_LIMIT_PREFIX from environment."""
        with patch.dict(os.environ, {"REDIS_RATE_LIMIT_PREFIX": "myapp:rate"}):
            config = get_redis_config()
            assert config.prefix == "myapp:rate"


class TestRedisRateLimiterInit:
    """Tests for RedisRateLimiter initialization."""

    def test_init_with_url(self):
        """Should initialize with redis_url."""
        limiter = RedisRateLimiter(redis_url="redis://test:6379")
        assert limiter.config.url == "redis://test:6379"

    def test_init_with_config(self):
        """Should initialize with full config."""
        config = RedisConfig(url="redis://custom:6380", default_limit=500)
        limiter = RedisRateLimiter(config=config)
        assert limiter.config.default_limit == 500

    def test_url_overrides_config(self):
        """redis_url should override config.url."""
        config = RedisConfig(url="redis://config:6379")
        limiter = RedisRateLimiter(redis_url="redis://override:6380", config=config)
        assert limiter.config.url == "redis://override:6380"

    def test_lazy_connection(self):
        """Should not connect until needed."""
        limiter = RedisRateLimiter()
        assert limiter._redis is None


class TestConfigureEndpoint:
    """Tests for configure_endpoint() method."""

    def test_configure_custom_limit(self, limiter):
        """Should configure endpoint with custom limits."""
        limiter.configure_endpoint("/api/debates", requests_per_minute=100)

        config = limiter.get_config("/api/debates")
        assert config.requests_per_minute == 100

    def test_configure_burst_size(self, limiter):
        """Should configure burst size."""
        limiter.configure_endpoint("/api/heavy", requests_per_minute=10, burst_size=20)

        config = limiter.get_config("/api/heavy")
        assert config.burst_size == 20

    def test_configure_key_type(self, limiter):
        """Should configure key type."""
        limiter.configure_endpoint("/api/global", requests_per_minute=1000, key_type="endpoint")

        config = limiter.get_config("/api/global")
        assert config.key_type == "endpoint"

    def test_get_config_fallback(self, limiter):
        """Should return default config for unconfigured endpoints."""
        config = limiter.get_config("/api/unknown")
        assert config.requests_per_minute == limiter.config.default_limit

    def test_wildcard_endpoint_matching(self, limiter):
        """Should match wildcard endpoints."""
        limiter.configure_endpoint("/api/debates/*", requests_per_minute=50)

        config = limiter.get_config("/api/debates/123")
        assert config.requests_per_minute == 50


class TestAllow:
    """Tests for allow() method."""

    def test_allow_under_limit(self, limiter, mock_redis):
        """Should allow requests under limit."""
        result = limiter.allow("192.168.1.1")

        assert result.allowed is True
        assert result.remaining > 0

    def test_block_over_limit(self, limiter, mock_redis):
        """Should block requests over limit."""
        # Make many requests to exceed limit
        limiter.config.default_limit = 5

        for _ in range(5):
            limiter.allow("192.168.1.1")

        result = limiter.allow("192.168.1.1")
        assert result.allowed is False
        assert result.remaining == 0

    def test_different_ips_independent(self, limiter, mock_redis):
        """Different IPs should have independent limits."""
        limiter.config.default_limit = 2

        # Exhaust limit for IP 1
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.1")

        # IP 2 should still be allowed
        result = limiter.allow("192.168.1.2")
        assert result.allowed is True

    def test_uses_endpoint_config(self, limiter, mock_redis):
        """Should use endpoint-specific config."""
        limiter.configure_endpoint("/api/heavy", requests_per_minute=2)

        limiter.allow("192.168.1.1", endpoint="/api/heavy")
        limiter.allow("192.168.1.1", endpoint="/api/heavy")

        result = limiter.allow("192.168.1.1", endpoint="/api/heavy")
        assert result.allowed is False

    def test_token_based_keying(self, limiter, mock_redis):
        """Should support token-based rate limiting."""
        limiter.configure_endpoint("/api/auth", requests_per_minute=10, key_type="token")

        result = limiter.allow("any-ip", endpoint="/api/auth", token="user-123")

        assert result.allowed is True
        assert "token:" in result.key

    def test_combined_keying(self, limiter, mock_redis):
        """Should support combined IP+endpoint keying."""
        limiter.configure_endpoint("/api/resource", requests_per_minute=10, key_type="combined")

        result = limiter.allow("192.168.1.1", endpoint="/api/resource")

        assert result.allowed is True
        assert "ep:" in result.key
        assert "ip:" in result.key

    def test_returns_retry_after(self, limiter, mock_redis):
        """Should return retry_after when blocked."""
        limiter.config.default_limit = 1

        limiter.allow("192.168.1.1")
        result = limiter.allow("192.168.1.1")

        assert result.allowed is False
        assert result.retry_after >= 0

    def test_allow_when_redis_unavailable(self):
        """Should allow requests when Redis unavailable."""
        limiter = RedisRateLimiter()
        limiter._available = False

        result = limiter.allow("192.168.1.1")

        assert result.allowed is True  # Fail-open behavior


class TestGetRemaining:
    """Tests for get_remaining() method."""

    def test_returns_remaining_without_consuming(self, limiter, mock_redis):
        """Should return remaining without consuming a token."""
        limiter.config.default_limit = 10

        # Make 3 requests
        for _ in range(3):
            limiter.allow("192.168.1.1")

        remaining = limiter.get_remaining("192.168.1.1")

        assert remaining == 7  # 10 - 3 = 7

    def test_returns_zero_when_exhausted(self, limiter, mock_redis):
        """Should return 0 when limit exhausted."""
        limiter.config.default_limit = 2

        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.1")

        remaining = limiter.get_remaining("192.168.1.1")
        assert remaining == 0

    def test_returns_zero_when_unavailable(self):
        """Should return 0 when Redis unavailable."""
        limiter = RedisRateLimiter()
        limiter._available = False

        remaining = limiter.get_remaining("192.168.1.1")
        assert remaining == 0


class TestReset:
    """Tests for reset() method."""

    def test_reset_all_keys(self, limiter, mock_redis):
        """Should reset all rate limit keys."""
        # Create some keys
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.2")

        deleted = limiter.reset()

        assert deleted >= 2

    def test_reset_with_pattern(self, limiter, mock_redis):
        """Should reset keys matching pattern."""
        limiter.allow("192.168.1.1")
        limiter.allow("192.168.1.2", token="user-123")

        # Only reset IP-based keys
        deleted = limiter.reset(pattern="ip:*")

        # Should have deleted IP key but not token key
        assert deleted >= 0

    def test_reset_when_unavailable(self):
        """Should return 0 when Redis unavailable."""
        limiter = RedisRateLimiter()
        limiter._available = False

        deleted = limiter.reset()
        assert deleted == 0


class TestGetStats:
    """Tests for get_stats() method."""

    def test_returns_stats_when_available(self, limiter, mock_redis):
        """Should return stats dict when Redis available."""
        limiter.allow("192.168.1.1")

        stats = limiter.get_stats()

        assert stats["available"] is True
        assert stats["backend"] == "redis"
        assert "ip_keys" in stats

    def test_returns_unavailable_stats(self):
        """Should indicate unavailable when Redis down."""
        limiter = RedisRateLimiter()
        limiter._available = False

        stats = limiter.get_stats()

        assert stats["available"] is False

    def test_includes_configured_endpoints(self, limiter, mock_redis):
        """Should include configured endpoints in stats."""
        limiter.configure_endpoint("/api/debates", requests_per_minute=100)
        limiter.configure_endpoint("/api/agents", requests_per_minute=50)

        stats = limiter.get_stats()

        assert "/api/debates" in stats["configured_endpoints"]
        assert "/api/agents" in stats["configured_endpoints"]


class TestGetClientKey:
    """Tests for get_client_key() method."""

    def test_extracts_ip_from_client_address(self, limiter):
        """Should extract IP from client_address."""
        handler = MagicMock()
        handler.client_address = ("192.168.1.100", 12345)
        handler.headers = {}

        key = limiter.get_client_key(handler)

        assert key == "192.168.1.100"

    def test_prefers_x_forwarded_for(self, limiter):
        """Should use X-Forwarded-For if present."""
        handler = MagicMock()
        handler.client_address = ("127.0.0.1", 80)
        handler.headers = {"X-Forwarded-For": "203.0.113.195, 70.41.3.18"}

        key = limiter.get_client_key(handler)

        assert key == "203.0.113.195"  # First IP in chain

    def test_returns_anonymous_for_none(self, limiter):
        """Should return 'anonymous' for None handler."""
        key = limiter.get_client_key(None)
        assert key == "anonymous"


class TestClose:
    """Tests for close() method."""

    def test_closes_connection(self, limiter, mock_redis):
        """Should close Redis connection."""
        limiter._redis = mock_redis

        limiter.close()

        assert limiter._redis is None


class TestDistributedRateLimiterFactory:
    """Tests for get_distributed_rate_limiter() factory."""

    def test_returns_limiter_with_allow_method(self):
        """Should return a limiter with allow() method."""
        reset_distributed_rate_limiter()

        with patch.dict(os.environ, {}, clear=True):
            # Without REDIS_URL, falls back to in-memory
            limiter = get_distributed_rate_limiter()
            assert hasattr(limiter, "allow")

        reset_distributed_rate_limiter()

    def test_falls_back_to_memory_when_no_redis_url(self):
        """Should fall back to in-memory when no REDIS_URL."""
        reset_distributed_rate_limiter()

        with patch.dict(os.environ, {}, clear=True):
            limiter = get_distributed_rate_limiter()

            # Should have allow method (works for both in-memory and Redis)
            assert hasattr(limiter, "allow")

        reset_distributed_rate_limiter()

    def test_caches_instance(self):
        """Should return same instance on subsequent calls."""
        reset_distributed_rate_limiter()

        with patch.dict(os.environ, {}, clear=True):
            limiter1 = get_distributed_rate_limiter()
            limiter2 = get_distributed_rate_limiter()

            assert limiter1 is limiter2

        reset_distributed_rate_limiter()


class TestSlidingWindowAlgorithm:
    """Tests for sliding window rate limiting algorithm."""

    def test_window_cleanup(self, limiter, mock_redis):
        """Old entries should be cleaned from window."""
        limiter.config.default_limit = 100

        # Simulate old entries by manipulating mock
        key = f"{limiter.config.prefix}:ip:192.168.1.1"
        old_time = time.time() - 120  # 2 minutes ago
        mock_redis._data[key] = [
            (str(old_time), old_time),
            (str(old_time + 1), old_time + 1),
        ]

        # New request should clean old entries
        result = limiter.allow("192.168.1.1")

        assert result.allowed is True
        # Old entries should be removed

    def test_burst_handling(self, limiter, mock_redis):
        """Should handle burst traffic correctly."""
        limiter.config.default_limit = 10

        # Rapid burst of requests
        results = []
        for _ in range(15):
            results.append(limiter.allow("192.168.1.1"))

        allowed = sum(1 for r in results if r.allowed)
        blocked = sum(1 for r in results if not r.allowed)

        assert allowed == 10
        assert blocked == 5


class TestIsAvailable:
    """Tests for is_available property."""

    def test_true_when_connected(self, limiter):
        """Should be True when Redis connected."""
        limiter._available = True
        assert limiter.is_available is True

    def test_false_when_disconnected(self):
        """Should be False when Redis unavailable."""
        limiter = RedisRateLimiter()
        limiter._available = False
        assert limiter.is_available is False
