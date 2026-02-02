"""
Tests for Rate Limiter Registry.

Covers:
- RateLimiterRegistry initialization
- Default limiter creation and caching
- Named limiter management
- Redis detection and fallback
- Cleanup and reset functionality
- Global functions (get_rate_limiter, etc.)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def clean_registry():
    """Reset registry before and after test."""
    from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

    reset_rate_limiters()
    yield
    reset_rate_limiters()


@pytest.fixture
def mock_redis_unavailable():
    """Mock Redis as unavailable."""
    with patch(
        "aragora.server.middleware.rate_limit.registry.get_redis_client",
        return_value=None,
    ):
        yield


@pytest.fixture
def mock_redis_available():
    """Mock Redis as available."""
    mock_client = MagicMock()
    with patch(
        "aragora.server.middleware.rate_limit.registry.get_redis_client",
        return_value=mock_client,
    ):
        with patch(
            "aragora.server.middleware.rate_limit.registry.RedisRateLimiter"
        ) as mock_redis_limiter:
            mock_limiter = MagicMock()
            mock_redis_limiter.return_value = mock_limiter
            yield mock_limiter


# -----------------------------------------------------------------------------
# RateLimiterRegistry Tests
# -----------------------------------------------------------------------------


class TestRateLimiterRegistryInitialization:
    """Tests for RateLimiterRegistry initialization."""

    def test_registry_starts_empty(self, clean_registry):
        """Registry starts with no limiters."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry

        registry = RateLimiterRegistry()

        assert registry._default_limiter is None
        assert len(registry._limiters) == 0
        assert registry._use_redis is None


class TestRateLimiterRegistryDefaultLimiter:
    """Tests for default limiter management."""

    def test_get_default_creates_limiter(self, clean_registry, mock_redis_unavailable):
        """get_default() creates a limiter when none exists."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry
        from aragora.server.middleware.rate_limit.limiter import RateLimiter

        registry = RateLimiterRegistry()
        limiter = registry.get_default()

        assert limiter is not None
        assert isinstance(limiter, RateLimiter)
        assert registry._default_limiter is limiter

    def test_get_default_caches_limiter(self, clean_registry, mock_redis_unavailable):
        """get_default() returns same limiter on repeated calls."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry

        registry = RateLimiterRegistry()
        limiter1 = registry.get_default()
        limiter2 = registry.get_default()

        assert limiter1 is limiter2

    def test_get_default_configures_endpoints(self, clean_registry, mock_redis_unavailable):
        """get_default() configures standard endpoint limits."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry

        registry = RateLimiterRegistry()
        limiter = registry.get_default()

        # Check some configured endpoints
        assert "/api/debates" in limiter._endpoint_configs
        assert "/api/agent/*" in limiter._endpoint_configs


class TestRateLimiterRegistryRedisDetection:
    """Tests for Redis availability detection."""

    def test_uses_memory_when_redis_unavailable(self, clean_registry, mock_redis_unavailable):
        """Falls back to in-memory limiter when Redis unavailable."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry
        from aragora.server.middleware.rate_limit.limiter import RateLimiter

        registry = RateLimiterRegistry()
        limiter = registry.get_default()

        assert isinstance(limiter, RateLimiter)
        assert registry._use_redis is False
        assert registry.is_using_redis is False

    def test_is_using_redis_triggers_init(self, clean_registry, mock_redis_unavailable):
        """is_using_redis property triggers initialization if needed."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry

        registry = RateLimiterRegistry()
        assert registry._use_redis is None

        # Accessing property should trigger init
        result = registry.is_using_redis

        assert result is False
        assert registry._default_limiter is not None


class TestRateLimiterRegistryNamedLimiters:
    """Tests for named limiter management."""

    def test_get_creates_named_limiter(self, clean_registry, mock_redis_unavailable):
        """get() creates a new named limiter."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry

        registry = RateLimiterRegistry()
        limiter = registry.get("test_limiter", requests_per_minute=100)

        assert limiter is not None
        assert "test_limiter" in registry._limiters

    def test_get_caches_named_limiter(self, clean_registry, mock_redis_unavailable):
        """get() returns same limiter for same name."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry

        registry = RateLimiterRegistry()
        limiter1 = registry.get("test_limiter", requests_per_minute=100)
        limiter2 = registry.get("test_limiter", requests_per_minute=200)  # Different config

        assert limiter1 is limiter2  # Should return cached, ignore new config

    def test_get_respects_requests_per_minute(self, clean_registry, mock_redis_unavailable):
        """get() respects requests_per_minute parameter."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry

        registry = RateLimiterRegistry()
        limiter = registry.get("custom_rate", requests_per_minute=42)

        assert limiter.default_limit == 42


class TestRateLimiterRegistryCleanup:
    """Tests for cleanup functionality."""

    def test_cleanup_calls_all_limiters(self, clean_registry, mock_redis_unavailable):
        """cleanup() calls cleanup on all limiters."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry

        registry = RateLimiterRegistry()
        registry.get_default()
        registry.get("limiter1")
        registry.get("limiter2")

        # Should not raise
        removed = registry.cleanup(max_age_seconds=0)

        assert isinstance(removed, int)

    def test_reset_clears_all_state(self, clean_registry, mock_redis_unavailable):
        """reset() clears all limiters and state."""
        from aragora.server.middleware.rate_limit.registry import RateLimiterRegistry

        registry = RateLimiterRegistry()
        registry.get_default()
        registry.get("test_limiter")

        assert len(registry._limiters) == 1
        assert registry._default_limiter is not None

        registry.reset()

        assert len(registry._limiters) == 0
        assert registry._default_limiter is None


# -----------------------------------------------------------------------------
# Global Function Tests
# -----------------------------------------------------------------------------


class TestGetRateLimiter:
    """Tests for get_rate_limiter global function."""

    def test_get_default_limiter(self, clean_registry, mock_redis_unavailable):
        """get_rate_limiter() with default name returns default limiter."""
        from aragora.server.middleware.rate_limit.registry import get_rate_limiter

        limiter = get_rate_limiter()

        assert limiter is not None

    def test_get_named_limiter(self, clean_registry, mock_redis_unavailable):
        """get_rate_limiter() with name returns named limiter."""
        from aragora.server.middleware.rate_limit.registry import get_rate_limiter

        limiter1 = get_rate_limiter("test_api", requests_per_minute=30)
        limiter2 = get_rate_limiter("test_api")

        assert limiter1 is limiter2


class TestCleanupRateLimiters:
    """Tests for cleanup_rate_limiters global function."""

    def test_cleanup_works(self, clean_registry, mock_redis_unavailable):
        """cleanup_rate_limiters() executes without error."""
        from aragora.server.middleware.rate_limit.registry import (
            cleanup_rate_limiters,
            get_rate_limiter,
        )

        # Create some limiters first
        get_rate_limiter()
        get_rate_limiter("test")

        # Should not raise
        removed = cleanup_rate_limiters()
        assert isinstance(removed, int)


class TestResetRateLimiters:
    """Tests for reset_rate_limiters global function."""

    def test_reset_clears_registry(self, clean_registry, mock_redis_unavailable):
        """reset_rate_limiters() clears the registry."""
        from aragora.server.middleware.rate_limit.registry import (
            get_rate_limiter,
            reset_rate_limiters,
        )

        limiter1 = get_rate_limiter("test_limiter")

        reset_rate_limiters()

        # Should get a new limiter after reset
        limiter2 = get_rate_limiter("test_limiter")
        assert limiter2 is not limiter1


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestRateLimiterRegistryIntegration:
    """Integration tests for registry with actual limiter operations."""

    def test_limiter_allows_requests(self, clean_registry, mock_redis_unavailable):
        """Limiter from registry allows requests under limit."""
        from aragora.server.middleware.rate_limit.registry import get_rate_limiter

        limiter = get_rate_limiter("test", requests_per_minute=60)

        result = limiter.allow("192.168.1.1")
        assert result.allowed is True

    def test_limiter_denies_over_limit(self, clean_registry, mock_redis_unavailable):
        """Limiter from registry denies requests over limit."""
        from aragora.server.middleware.rate_limit.registry import get_rate_limiter

        limiter = get_rate_limiter("strict_test", requests_per_minute=1)

        # First request should be allowed
        result1 = limiter.allow("192.168.1.2")
        assert result1.allowed is True

        # Exhaust burst (burst = 2x rate = 2)
        limiter.allow("192.168.1.2")
        limiter.allow("192.168.1.2")

        # Next request should be denied
        result2 = limiter.allow("192.168.1.2")
        assert result2.allowed is False

    def test_different_ips_independent(self, clean_registry, mock_redis_unavailable):
        """Different IPs have independent limits."""
        from aragora.server.middleware.rate_limit.registry import get_rate_limiter

        limiter = get_rate_limiter("ip_test", requests_per_minute=1)

        # Exhaust IP1
        for _ in range(5):
            limiter.allow("192.168.1.1")

        # IP2 should still be allowed
        result = limiter.allow("192.168.1.2")
        assert result.allowed is True


class TestRateLimiterEndpointConfiguration:
    """Tests for endpoint-specific configuration."""

    def test_endpoint_config_from_default(self, clean_registry, mock_redis_unavailable):
        """Default limiter has endpoint configurations."""
        from aragora.server.middleware.rate_limit.registry import get_rate_limiter

        limiter = get_rate_limiter()

        # Should have configured endpoints
        configs = limiter._endpoint_configs
        assert len(configs) > 0

        # Check specific endpoints
        assert any("debates" in k for k in configs.keys())
        assert any("knowledge" in k for k in configs.keys())

    def test_endpoint_matching_wildcard(self, clean_registry, mock_redis_unavailable):
        """Endpoint wildcard matching works."""
        from aragora.server.middleware.rate_limit.registry import get_rate_limiter

        limiter = get_rate_limiter()

        # Configure a specific wildcard endpoint
        limiter.configure_endpoint("/api/test/*", 10, key_type="ip")

        # Check it's stored
        assert "/api/test/*" in limiter._endpoint_configs
