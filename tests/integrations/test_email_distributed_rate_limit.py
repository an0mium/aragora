"""
Tests for distributed email rate limiting.

Tests for:
- Token bucket algorithm
- Sliding window counters
- Per-provider limits
- Multi-tenant isolation
- Redis and local fallback modes
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.integrations.email_rate_limiter import (
    DEFAULT_PROVIDER_LIMITS,
    EmailRateLimiter,
    ProviderLimits,
    RateLimitResult,
    UsageStats,
    get_email_rate_limiter,
    reset_email_rate_limiter,
    set_email_rate_limiter,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def limiter():
    """Create a rate limiter with Redis disabled."""
    rl = EmailRateLimiter()
    rl._redis_checked = True  # Skip Redis check
    rl._redis = None
    return rl


@pytest.fixture
def custom_limits():
    """Create custom provider limits for testing."""
    return ProviderLimits(
        per_minute=10,
        per_hour=50,
        per_day=100,
        burst_size=5,
        refill_rate=1.0,
    )


@pytest.fixture(autouse=True)
def reset_global_limiter():
    """Reset global limiter before and after each test."""
    reset_email_rate_limiter()
    yield
    reset_email_rate_limiter()


# =============================================================================
# ProviderLimits Tests
# =============================================================================


class TestProviderLimits:
    """Tests for ProviderLimits configuration."""

    def test_default_limits_exist(self):
        """Test default limits exist for common providers."""
        assert "gmail" in DEFAULT_PROVIDER_LIMITS
        assert "microsoft" in DEFAULT_PROVIDER_LIMITS
        assert "sendgrid" in DEFAULT_PROVIDER_LIMITS
        assert "ses" in DEFAULT_PROVIDER_LIMITS
        assert "smtp" in DEFAULT_PROVIDER_LIMITS

    def test_gmail_limits(self):
        """Test Gmail has appropriate limits."""
        limits = DEFAULT_PROVIDER_LIMITS["gmail"]
        assert limits.per_day == 500
        assert limits.per_hour <= 100
        assert limits.burst_size > 0

    def test_sendgrid_limits_higher(self):
        """Test SendGrid has higher limits than Gmail."""
        gmail = DEFAULT_PROVIDER_LIMITS["gmail"]
        sendgrid = DEFAULT_PROVIDER_LIMITS["sendgrid"]
        assert sendgrid.per_day > gmail.per_day
        assert sendgrid.per_hour > gmail.per_hour


# =============================================================================
# RateLimitResult Tests
# =============================================================================


class TestRateLimitResult:
    """Tests for RateLimitResult."""

    def test_allowed_result(self):
        """Test allowed result structure."""
        result = RateLimitResult(allowed=True, remaining=10, limit_type="ok")
        assert result.allowed is True
        assert result.remaining == 10

    def test_denied_result(self):
        """Test denied result structure."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            retry_after=30.0,
            limit_type="minute",
        )
        assert result.allowed is False
        assert result.retry_after == 30.0

    def test_to_dict(self):
        """Test serialization to dict."""
        result = RateLimitResult(allowed=True, remaining=5)
        data = result.to_dict()
        assert data["allowed"] is True
        assert data["remaining"] == 5


# =============================================================================
# UsageStats Tests
# =============================================================================


class TestUsageStats:
    """Tests for UsageStats."""

    def test_remaining_calculation(self):
        """Test remaining count calculations."""
        stats = UsageStats(
            tenant_id="t1",
            provider="gmail",
            minute_count=5,
            hour_count=40,
            day_count=400,
            limits=ProviderLimits(per_minute=10, per_hour=100, per_day=500),
        )
        assert stats.minute_remaining == 5
        assert stats.hour_remaining == 60
        assert stats.day_remaining == 100

    def test_remaining_never_negative(self):
        """Test remaining never goes negative."""
        stats = UsageStats(
            tenant_id="t1",
            provider="gmail",
            minute_count=100,  # Over limit
            hour_count=0,
            day_count=0,
            limits=ProviderLimits(per_minute=10, per_hour=100, per_day=500),
        )
        assert stats.minute_remaining == 0

    def test_to_dict(self):
        """Test serialization to dict."""
        stats = UsageStats(
            tenant_id="t1",
            provider="gmail",
            minute_count=5,
            hour_count=10,
            day_count=50,
            bucket_tokens=3.5,
        )
        data = stats.to_dict()
        assert data["tenant_id"] == "t1"
        assert data["provider"] == "gmail"
        assert "minute" in data
        assert "hour" in data
        assert "day" in data
        assert "burst" in data


# =============================================================================
# EmailRateLimiter Basic Tests
# =============================================================================


class TestEmailRateLimiterBasic:
    """Basic tests for EmailRateLimiter."""

    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter):
        """Test successful acquisition."""
        result = await limiter.acquire("tenant_1", "gmail")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_acquire_multiple(self, limiter):
        """Test acquiring multiple tokens."""
        result = await limiter.acquire("tenant_1", "gmail", count=3)
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_get_usage(self, limiter):
        """Test getting usage statistics."""
        await limiter.acquire("tenant_1", "gmail")
        await limiter.acquire("tenant_1", "gmail")

        stats = await limiter.get_usage("tenant_1", "gmail")
        assert stats.minute_count == 2
        assert stats.hour_count == 2
        assert stats.day_count == 2

    @pytest.mark.asyncio
    async def test_reset(self, limiter):
        """Test resetting rate limits."""
        await limiter.acquire("tenant_1", "gmail")
        await limiter.acquire("tenant_1", "gmail")

        await limiter.reset("tenant_1", "gmail")

        stats = await limiter.get_usage("tenant_1", "gmail")
        assert stats.minute_count == 0
        assert stats.hour_count == 0


# =============================================================================
# Rate Limit Enforcement Tests
# =============================================================================


class TestRateLimitEnforcement:
    """Tests for rate limit enforcement."""

    @pytest.mark.asyncio
    async def test_minute_limit_enforced(self, limiter, custom_limits):
        """Test minute limit is enforced."""
        limiter.set_tenant_limits("tenant_1", "gmail", custom_limits)

        # Use up the minute limit
        for _ in range(10):
            result = await limiter.acquire("tenant_1", "gmail")
            assert result.allowed is True

        # Next request should be denied
        result = await limiter.acquire("tenant_1", "gmail")
        assert result.allowed is False
        assert result.limit_type == "minute"
        assert result.retry_after > 0

    @pytest.mark.asyncio
    async def test_burst_limit_enforced(self, limiter):
        """Test burst limit is enforced."""
        # Create limiter with very small burst
        limits = ProviderLimits(
            per_minute=100,
            per_hour=1000,
            per_day=10000,
            burst_size=3,
            refill_rate=0.1,
        )
        limiter.set_tenant_limits("tenant_1", "test", limits)

        # Use up burst
        for i in range(3):
            result = await limiter.acquire("tenant_1", "test")
            assert result.allowed is True, f"Request {i + 1} should be allowed"

        # Next should hit burst limit
        result = await limiter.acquire("tenant_1", "test")
        assert result.allowed is False
        assert result.limit_type == "burst"

    @pytest.mark.asyncio
    async def test_remaining_count_decreases(self, limiter, custom_limits):
        """Test remaining count decreases with each request."""
        limiter.set_tenant_limits("tenant_1", "gmail", custom_limits)

        stats_before = await limiter.get_usage("tenant_1", "gmail")
        initial_remaining = stats_before.minute_remaining

        await limiter.acquire("tenant_1", "gmail")

        stats_after = await limiter.get_usage("tenant_1", "gmail")
        assert stats_after.minute_remaining == initial_remaining - 1


# =============================================================================
# Multi-tenant Isolation Tests
# =============================================================================


class TestMultiTenantIsolation:
    """Tests for multi-tenant rate limit isolation."""

    @pytest.mark.asyncio
    async def test_tenants_have_separate_limits(self, limiter):
        """Test each tenant has their own limits."""
        limits = ProviderLimits(per_minute=5, per_hour=50, per_day=100)
        limiter.set_tenant_limits("tenant_a", "gmail", limits)
        limiter.set_tenant_limits("tenant_b", "gmail", limits)

        # Exhaust tenant A's limit
        for _ in range(5):
            await limiter.acquire("tenant_a", "gmail")

        # Tenant A should be blocked
        result_a = await limiter.acquire("tenant_a", "gmail")
        assert result_a.allowed is False

        # Tenant B should still be allowed
        result_b = await limiter.acquire("tenant_b", "gmail")
        assert result_b.allowed is True

    @pytest.mark.asyncio
    async def test_tenant_specific_limits(self, limiter):
        """Test tenant-specific limit overrides."""
        # Tenant A gets low limits
        limiter.set_tenant_limits(
            "tenant_a", "gmail", ProviderLimits(per_minute=2, per_hour=10, per_day=50)
        )
        # Tenant B gets high limits
        limiter.set_tenant_limits(
            "tenant_b", "gmail", ProviderLimits(per_minute=100, per_hour=1000, per_day=10000)
        )

        # Tenant A hits limit quickly
        await limiter.acquire("tenant_a", "gmail")
        await limiter.acquire("tenant_a", "gmail")
        result_a = await limiter.acquire("tenant_a", "gmail")
        assert result_a.allowed is False

        # Tenant B has plenty of room
        for _ in range(10):
            result_b = await limiter.acquire("tenant_b", "gmail")
            assert result_b.allowed is True

    @pytest.mark.asyncio
    async def test_usage_isolated_between_tenants(self, limiter):
        """Test usage stats are isolated between tenants."""
        await limiter.acquire("tenant_a", "gmail")
        await limiter.acquire("tenant_a", "gmail")
        await limiter.acquire("tenant_b", "gmail")

        stats_a = await limiter.get_usage("tenant_a", "gmail")
        stats_b = await limiter.get_usage("tenant_b", "gmail")

        assert stats_a.minute_count == 2
        assert stats_b.minute_count == 1


# =============================================================================
# Per-Provider Tests
# =============================================================================


class TestPerProviderLimits:
    """Tests for per-provider rate limits."""

    @pytest.mark.asyncio
    async def test_different_providers_have_separate_limits(self, limiter):
        """Test each provider has separate limits."""
        limits = ProviderLimits(per_minute=5, per_hour=50, per_day=100)
        limiter.set_tenant_limits("tenant_1", "gmail", limits)
        limiter.set_tenant_limits("tenant_1", "sendgrid", limits)

        # Exhaust gmail limit
        for _ in range(5):
            await limiter.acquire("tenant_1", "gmail")

        # Gmail should be blocked
        result_gmail = await limiter.acquire("tenant_1", "gmail")
        assert result_gmail.allowed is False

        # SendGrid should still work
        result_sg = await limiter.acquire("tenant_1", "sendgrid")
        assert result_sg.allowed is True

    @pytest.mark.asyncio
    async def test_default_provider_limits_applied(self, limiter):
        """Test default provider limits are applied."""
        # Without custom limits, defaults should apply
        stats = await limiter.get_usage("tenant_1", "gmail")
        default_gmail = DEFAULT_PROVIDER_LIMITS["gmail"]

        assert stats.limits.per_day == default_gmail.per_day
        assert stats.limits.per_hour == default_gmail.per_hour


# =============================================================================
# Token Bucket Tests
# =============================================================================


class TestTokenBucket:
    """Tests for token bucket algorithm."""

    @pytest.mark.asyncio
    async def test_bucket_refills_over_time(self, limiter):
        """Test token bucket refills over time."""
        # Set up fast refill for testing
        limits = ProviderLimits(
            per_minute=100,
            per_hour=1000,
            per_day=10000,
            burst_size=5,
            refill_rate=10.0,  # 10 tokens per second
        )
        limiter.set_tenant_limits("tenant_1", "test", limits)

        # Exhaust burst
        for _ in range(5):
            await limiter.acquire("tenant_1", "test")

        # Should be blocked
        result = await limiter.acquire("tenant_1", "test")
        assert result.allowed is False

        # Wait for refill (0.2 seconds = 2 tokens)
        await asyncio.sleep(0.2)

        # Should be allowed now
        result = await limiter.acquire("tenant_1", "test")
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_bucket_capped_at_burst_size(self, limiter):
        """Test bucket doesn't exceed burst size."""
        limits = ProviderLimits(
            burst_size=5,
            refill_rate=100.0,  # Very fast refill
        )
        limiter.set_tenant_limits("tenant_1", "test", limits)

        # Wait for refill
        await asyncio.sleep(0.1)

        stats = await limiter.get_usage("tenant_1", "test")
        assert stats.bucket_tokens <= limits.burst_size


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access handling."""

    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self, limiter):
        """Test concurrent rate limit acquisitions."""
        limits = ProviderLimits(
            per_minute=50,
            per_hour=500,
            per_day=5000,
            burst_size=50,
            refill_rate=10.0,
        )
        limiter.set_tenant_limits("tenant_1", "test", limits)

        # Run many concurrent acquisitions
        tasks = [limiter.acquire("tenant_1", "test") for _ in range(30)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        allowed_count = sum(1 for r in results if r.allowed)
        assert allowed_count == 30

        # Verify counts are accurate
        stats = await limiter.get_usage("tenant_1", "test")
        assert stats.minute_count == 30

    @pytest.mark.asyncio
    async def test_concurrent_different_tenants(self, limiter):
        """Test concurrent access from different tenants."""
        # Run concurrent acquisitions from multiple tenants
        tasks = []
        for i in range(5):
            for _ in range(10):
                tasks.append(limiter.acquire(f"tenant_{i}", "gmail"))

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(r.allowed for r in results)

        # Each tenant should have 10 requests
        for i in range(5):
            stats = await limiter.get_usage(f"tenant_{i}", "gmail")
            assert stats.minute_count == 10


# =============================================================================
# Global Factory Tests
# =============================================================================


class TestGlobalFactory:
    """Tests for global rate limiter factory."""

    def test_get_limiter_returns_instance(self):
        """Test get_email_rate_limiter returns a limiter."""
        limiter = get_email_rate_limiter()
        assert limiter is not None
        assert isinstance(limiter, EmailRateLimiter)

    def test_get_limiter_returns_singleton(self):
        """Test get_email_rate_limiter returns singleton."""
        limiter1 = get_email_rate_limiter()
        limiter2 = get_email_rate_limiter()
        assert limiter1 is limiter2

    def test_set_custom_limiter(self):
        """Test setting a custom limiter."""
        custom = EmailRateLimiter()
        set_email_rate_limiter(custom)

        retrieved = get_email_rate_limiter()
        assert retrieved is custom

    def test_reset_limiter(self):
        """Test resetting the global limiter."""
        limiter1 = get_email_rate_limiter()
        reset_email_rate_limiter()
        limiter2 = get_email_rate_limiter()

        assert limiter1 is not limiter2


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_unknown_provider_uses_defaults(self, limiter):
        """Test unknown provider uses default limits."""
        result = await limiter.acquire("tenant_1", "unknown_provider")
        assert result.allowed is True

        stats = await limiter.get_usage("tenant_1", "unknown_provider")
        assert stats.limits.per_minute == ProviderLimits().per_minute

    @pytest.mark.asyncio
    async def test_acquire_zero_count_allowed(self, limiter):
        """Test acquiring zero tokens is allowed (noop)."""
        result = await limiter.acquire("tenant_1", "gmail", count=0)
        assert result.allowed is True
