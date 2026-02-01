"""
Tests for Tenant Rate Limiter.

Tests cover:
- TenantRateLimiter class (per-tenant rate limiting)
- Tenant isolation (one tenant can't affect another)
- Integration with QuotaManager for tenant-specific limits
- Fallback to default limits when no tenant context
- Convenience functions
"""

from __future__ import annotations

import pytest
import threading
from unittest.mock import Mock, patch, MagicMock

from aragora.server.middleware.rate_limit import (
    TokenBucket,
    RateLimiter,
    reset_rate_limiters,
)
from aragora.server.middleware.rate_limit.tenant_limiter import (
    TenantRateLimiter,
    TenantRateLimitConfig,
    get_tenant_rate_limiter,
    reset_tenant_rate_limiter,
    check_tenant_rate_limit,
)
from aragora.tenancy.context import TenantContext


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_limiters():
    """Reset rate limiters before and after each test."""
    reset_rate_limiters()
    reset_tenant_rate_limiter()
    yield
    reset_rate_limiters()
    reset_tenant_rate_limiter()


@pytest.fixture
def tenant_limiter():
    """Create a fresh tenant rate limiter for testing."""
    config = TenantRateLimitConfig(
        default_limit=60,
        default_burst=120,
        use_quota_manager=False,  # Disable for predictable testing
    )
    return TenantRateLimiter(config=config)


@pytest.fixture
def strict_tenant_limiter():
    """Create a tenant limiter with very strict limits for testing rate limiting."""
    config = TenantRateLimitConfig(
        default_limit=2,
        default_burst=2,
        use_quota_manager=False,
    )
    return TenantRateLimiter(config=config)


# ============================================================================
# TenantRateLimitConfig Tests
# ============================================================================


class TestTenantRateLimitConfig:
    """Tests for TenantRateLimitConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TenantRateLimitConfig()
        assert config.default_limit == 60
        assert config.default_burst is None
        assert config.use_quota_manager is True
        assert config.fallback_to_default is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TenantRateLimitConfig(
            default_limit=100,
            default_burst=200,
            use_quota_manager=False,
            fallback_to_default=False,
        )
        assert config.default_limit == 100
        assert config.default_burst == 200
        assert config.use_quota_manager is False
        assert config.fallback_to_default is False


# ============================================================================
# TenantRateLimiter Tests
# ============================================================================


class TestTenantRateLimiter:
    """Tests for TenantRateLimiter class."""

    def test_init_defaults(self, tenant_limiter):
        """Test initialization with defaults."""
        assert tenant_limiter.config.default_limit == 60
        assert tenant_limiter.config.default_burst == 120

    def test_allow_under_limit(self, tenant_limiter):
        """Test allow returns True under limit."""
        result = tenant_limiter.allow(tenant_id="tenant-1")
        assert result.allowed is True
        assert result.remaining > 0
        assert "tenant:" in result.key

    def test_allow_over_limit(self, strict_tenant_limiter):
        """Test allow returns False when limit exceeded."""
        # Exhaust the tenant's limit (2 requests max)
        strict_tenant_limiter.allow(tenant_id="tenant-1")
        strict_tenant_limiter.allow(tenant_id="tenant-1")

        # Third request should be rate limited
        result = strict_tenant_limiter.allow(tenant_id="tenant-1")

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after > 0

    def test_tenant_isolation(self, strict_tenant_limiter):
        """Test that one tenant cannot affect another tenant's rate limit."""
        # Exhaust tenant-1's limit
        strict_tenant_limiter.allow(tenant_id="tenant-1")
        strict_tenant_limiter.allow(tenant_id="tenant-1")
        result1 = strict_tenant_limiter.allow(tenant_id="tenant-1")

        # tenant-2 should still have their full quota
        result2 = strict_tenant_limiter.allow(tenant_id="tenant-2")

        assert result1.allowed is False, "tenant-1 should be rate limited"
        assert result2.allowed is True, "tenant-2 should not be affected by tenant-1"

    def test_tenant_isolation_many_tenants(self, strict_tenant_limiter):
        """Test isolation with many tenants."""
        # Exhaust limits for multiple tenants
        for i in range(5):
            tenant_id = f"tenant-{i}"
            strict_tenant_limiter.allow(tenant_id=tenant_id)
            strict_tenant_limiter.allow(tenant_id=tenant_id)
            result = strict_tenant_limiter.allow(tenant_id=tenant_id)
            assert result.allowed is False, f"{tenant_id} should be rate limited"

        # New tenant should still have full quota
        new_result = strict_tenant_limiter.allow(tenant_id="tenant-new")
        assert new_result.allowed is True, "New tenant should have full quota"

    def test_allow_with_tenant_context(self, tenant_limiter):
        """Test that tenant ID is read from context when not provided."""
        with TenantContext(tenant_id="context-tenant"):
            result = tenant_limiter.allow()  # No tenant_id provided

        assert result.allowed is True
        assert "tenant:context-tenant" in result.key

    def test_allow_explicit_tenant_overrides_context(self, tenant_limiter):
        """Test that explicit tenant_id overrides context."""
        with TenantContext(tenant_id="context-tenant"):
            result = tenant_limiter.allow(tenant_id="explicit-tenant")

        assert "tenant:explicit-tenant" in result.key

    def test_allow_no_tenant_fallback_to_default(self, tenant_limiter):
        """Test fallback to default bucket when no tenant context."""
        result = tenant_limiter.allow()  # No tenant context

        assert result.allowed is True
        assert result.key == "default"

    def test_allow_no_tenant_no_fallback(self):
        """Test behavior when no fallback is configured."""
        config = TenantRateLimitConfig(
            fallback_to_default=False,
            use_quota_manager=False,
        )
        limiter = TenantRateLimiter(config=config)

        result = limiter.allow()  # No tenant context

        assert result.allowed is True
        assert result.key == "no_tenant"
        assert result.limit == 0

    def test_key_format(self, tenant_limiter):
        """Test result key format includes tenant."""
        result = tenant_limiter.allow(tenant_id="my-tenant-id")
        assert result.key == "tenant:my-tenant-id"

    def test_metrics_tracking(self, strict_tenant_limiter):
        """Test that metrics are tracked per tenant."""
        # Make some requests
        strict_tenant_limiter.allow(tenant_id="tenant-a")
        strict_tenant_limiter.allow(tenant_id="tenant-a")
        strict_tenant_limiter.allow(tenant_id="tenant-a")  # Rate limited
        strict_tenant_limiter.allow(tenant_id="tenant-b")

        stats = strict_tenant_limiter.get_stats()

        assert stats["total_requests"] == 4
        assert stats["total_rejections"] == 1
        assert stats["requests_by_tenant"]["tenant-a"] == 3
        assert stats["requests_by_tenant"]["tenant-b"] == 1
        assert stats["rejections_by_tenant"]["tenant-a"] == 1

    def test_get_tenant_stats(self, strict_tenant_limiter):
        """Test getting stats for a specific tenant."""
        strict_tenant_limiter.allow(tenant_id="tenant-x")
        strict_tenant_limiter.allow(tenant_id="tenant-x")

        stats = strict_tenant_limiter.get_tenant_stats("tenant-x")

        assert stats["tenant_id"] == "tenant-x"
        assert stats["rate_limit"] == 2
        assert stats["burst_size"] == 2
        assert stats["requests"] == 2
        assert stats["remaining"] == 0

    def test_reset_clears_all(self, tenant_limiter):
        """Test reset clears all state."""
        tenant_limiter.allow(tenant_id="tenant-1")
        tenant_limiter.allow(tenant_id="tenant-2")

        tenant_limiter.reset()

        stats = tenant_limiter.get_stats()
        assert stats["tenant_count"] == 0
        assert stats["total_requests"] == 0

    def test_reset_tenant(self, strict_tenant_limiter):
        """Test resetting a specific tenant."""
        # Exhaust tenant-1's limit
        strict_tenant_limiter.allow(tenant_id="tenant-1")
        strict_tenant_limiter.allow(tenant_id="tenant-1")
        result1 = strict_tenant_limiter.allow(tenant_id="tenant-1")
        assert result1.allowed is False

        # Reset tenant-1
        strict_tenant_limiter.reset_tenant("tenant-1")

        # tenant-1 should have full quota again
        result2 = strict_tenant_limiter.allow(tenant_id="tenant-1")
        assert result2.allowed is True

    def test_cleanup_removes_stale(self, tenant_limiter):
        """Test cleanup removes stale tenant buckets."""
        import time

        tenant_limiter.allow(tenant_id="tenant-old")
        tenant_limiter.allow(tenant_id="tenant-new")

        # Make the old bucket stale
        tenant_limiter._tenant_buckets["tenant-old"].last_refill = time.monotonic() - 600

        removed = tenant_limiter.cleanup(max_age_seconds=300)

        assert removed == 1
        assert "tenant-old" not in tenant_limiter._tenant_buckets
        assert "tenant-new" in tenant_limiter._tenant_buckets

    def test_thread_safety(self, tenant_limiter):
        """Test concurrent access from multiple threads."""
        results = []
        errors = []

        def make_request(tenant_id: str):
            try:
                for _ in range(10):
                    result = tenant_limiter.allow(tenant_id=tenant_id)
                    results.append(result.allowed)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=make_request, args=(f"tenant-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50  # 5 threads x 10 requests each


# ============================================================================
# QuotaManager Integration Tests
# ============================================================================


class TestTenantRateLimiterQuotaIntegration:
    """Tests for TenantRateLimiter integration with QuotaManager."""

    def test_uses_quota_manager_limits(self):
        """Test that limits are fetched from QuotaManager."""
        mock_quota_manager = Mock()
        mock_limit = Mock()
        mock_limit.limit = 100
        mock_limit.burst_limit = 200
        mock_quota_manager._get_limits_for_tenant.return_value = {"api_requests": mock_limit}

        config = TenantRateLimitConfig(use_quota_manager=True)
        limiter = TenantRateLimiter(config=config, quota_manager=mock_quota_manager)

        result = limiter.allow(tenant_id="premium-tenant")

        assert result.limit == 100
        mock_quota_manager._get_limits_for_tenant.assert_called_with("premium-tenant")

    def test_fallback_when_quota_manager_fails(self):
        """Test fallback to default limits when QuotaManager fails."""
        mock_quota_manager = Mock()
        mock_quota_manager._get_limits_for_tenant.side_effect = RuntimeError("DB error")

        config = TenantRateLimitConfig(
            default_limit=50,
            default_burst=100,
            use_quota_manager=True,
        )
        limiter = TenantRateLimiter(config=config, quota_manager=mock_quota_manager)

        result = limiter.allow(tenant_id="tenant-1")

        assert result.allowed is True
        assert result.limit == 50  # Default limit used

    def test_different_tenants_different_limits(self):
        """Test that different tenants can have different limits."""
        mock_quota_manager = Mock()

        def get_limits(tenant_id):
            mock_limit = Mock()
            if tenant_id == "premium":
                mock_limit.limit = 1000
                mock_limit.burst_limit = 2000
            else:
                mock_limit.limit = 10
                mock_limit.burst_limit = 20
            return {"api_requests": mock_limit}

        mock_quota_manager._get_limits_for_tenant.side_effect = get_limits

        config = TenantRateLimitConfig(use_quota_manager=True)
        limiter = TenantRateLimiter(config=config, quota_manager=mock_quota_manager)

        premium_result = limiter.allow(tenant_id="premium")
        free_result = limiter.allow(tenant_id="free")

        assert premium_result.limit == 1000
        assert free_result.limit == 10


# ============================================================================
# Convenience Functions Tests
# ============================================================================


class TestTenantRateLimitConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_tenant_rate_limiter_singleton(self):
        """Test get_tenant_rate_limiter returns singleton."""
        limiter1 = get_tenant_rate_limiter()
        limiter2 = get_tenant_rate_limiter()

        assert limiter1 is limiter2

    def test_reset_tenant_rate_limiter(self):
        """Test reset_tenant_rate_limiter clears singleton."""
        limiter1 = get_tenant_rate_limiter()
        limiter1.allow(tenant_id="test")

        reset_tenant_rate_limiter()

        limiter2 = get_tenant_rate_limiter()
        assert limiter1 is not limiter2
        assert limiter2.get_stats()["total_requests"] == 0

    def test_check_tenant_rate_limit(self):
        """Test check_tenant_rate_limit convenience function."""
        reset_tenant_rate_limiter()

        result = check_tenant_rate_limit(tenant_id="test-tenant")

        assert result.allowed is True
        assert "tenant:" in result.key

    def test_check_tenant_rate_limit_uses_context(self):
        """Test check_tenant_rate_limit reads from tenant context."""
        reset_tenant_rate_limiter()

        with TenantContext(tenant_id="ctx-tenant"):
            result = check_tenant_rate_limit()

        assert "tenant:ctx-tenant" in result.key


# ============================================================================
# RateLimiter Tenant Key Type Tests
# ============================================================================


class TestRateLimiterTenantKeyType:
    """Tests for tenant key_type in RateLimiter."""

    def test_allow_with_tenant_key_type(self):
        """Test RateLimiter uses tenant key when configured."""
        limiter = RateLimiter(default_limit=60, ip_limit=120)
        limiter.configure_endpoint("/api/tenant-aware", 30, key_type="tenant")

        result = limiter.allow(
            client_ip="192.168.1.1",
            endpoint="/api/tenant-aware",
            tenant_id="my-tenant",
        )

        assert result.allowed is True
        assert result.key == "tenant:my-tenant"

    def test_tenant_key_type_isolation(self):
        """Test tenant isolation with tenant key_type."""
        limiter = RateLimiter(default_limit=2, ip_limit=2)
        limiter.configure_endpoint("/api/limited", 2, key_type="tenant")

        # Exhaust tenant-1's limit
        limiter.allow("192.168.1.1", "/api/limited", tenant_id="tenant-1")
        limiter.allow("192.168.1.1", "/api/limited", tenant_id="tenant-1")
        limiter.allow("192.168.1.1", "/api/limited", tenant_id="tenant-1")
        limiter.allow("192.168.1.1", "/api/limited", tenant_id="tenant-1")
        limiter.allow("192.168.1.1", "/api/limited", tenant_id="tenant-1")

        # tenant-2 should still have quota (same IP!)
        result = limiter.allow("192.168.1.1", "/api/limited", tenant_id="tenant-2")

        assert result.allowed is True

    def test_tenant_key_type_falls_back_to_ip(self):
        """Test that tenant key_type falls back to IP when no tenant provided."""
        limiter = RateLimiter(default_limit=60, ip_limit=120)
        limiter.configure_endpoint("/api/tenant-aware", 30, key_type="tenant")

        # No tenant_id provided
        result = limiter.allow("192.168.1.1", "/api/tenant-aware")

        # Should fall back to IP-based limiting
        assert result.key == "ip:192.168.1.1"

    def test_tenant_buckets_tracked(self):
        """Test that tenant buckets are tracked separately."""
        limiter = RateLimiter(default_limit=60, ip_limit=120)
        limiter.configure_endpoint("/api/test", 30, key_type="tenant")

        limiter.allow("192.168.1.1", "/api/test", tenant_id="tenant-a")
        limiter.allow("192.168.1.2", "/api/test", tenant_id="tenant-b")
        limiter.allow("192.168.1.3", "/api/test", tenant_id="tenant-c")

        stats = limiter.get_stats()

        assert stats["tenant_buckets"] == 3


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestTenantRateLimiterEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_tenant_id(self, tenant_limiter):
        """Test handling of empty string tenant ID."""
        result = tenant_limiter.allow(tenant_id="")

        # Empty string should be treated as no tenant
        assert result.key == "default"

    def test_special_characters_in_tenant_id(self, tenant_limiter):
        """Test handling of special characters in tenant ID."""
        # Colons should be sanitized
        result = tenant_limiter.allow(tenant_id="tenant:with:colons")

        assert result.allowed is True
        assert ":" not in result.key.replace("tenant:", "")  # Only the prefix colon

    def test_very_long_tenant_id(self, tenant_limiter):
        """Test handling of very long tenant IDs."""
        long_id = "t" * 1000
        result = tenant_limiter.allow(tenant_id=long_id)

        assert result.allowed is True

    def test_lru_eviction(self):
        """Test LRU eviction when max tenants reached."""
        config = TenantRateLimitConfig(
            default_limit=60,
            use_quota_manager=False,
        )
        limiter = TenantRateLimiter(config=config)
        limiter._max_tenants = 3  # Set very low max

        # Add 4 tenants (exceeds max of 3)
        limiter.allow(tenant_id="tenant-1")
        limiter.allow(tenant_id="tenant-2")
        limiter.allow(tenant_id="tenant-3")
        limiter.allow(tenant_id="tenant-4")

        # First tenant should have been evicted
        assert "tenant-1" not in limiter._tenant_buckets
        assert "tenant-4" in limiter._tenant_buckets

    def test_concurrent_tenant_creation(self):
        """Test concurrent creation of tenant buckets."""
        config = TenantRateLimitConfig(
            default_limit=60,
            use_quota_manager=False,
        )
        limiter = TenantRateLimiter(config=config)

        errors = []
        results = []

        def create_tenant(tenant_id: str):
            try:
                result = limiter.allow(tenant_id=tenant_id)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(100):
            t = threading.Thread(target=create_tenant, args=(f"tenant-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 100
        assert all(r.allowed for r in results)
