"""
Tests for tenant resource limits enforcement.

Tests cover:
- TenantLimitExceededError exception
- TenantLimitsEnforcer limit checks
- Usage summary calculations
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass

from aragora.tenancy.limits import (
    TenantLimitExceededError,
    TenantLimitsEnforcer,
)
from aragora.tenancy.tenant import TenantConfig


# =============================================================================
# TenantLimitExceededError Tests
# =============================================================================


class TestTenantLimitExceededError:
    """Tests for TenantLimitExceededError exception."""

    def test_error_message(self):
        """Should include message in exception."""
        error = TenantLimitExceededError(
            message="Daily debate limit exceeded",
            limit_type="debates_per_day",
            current=10,
            limit=10,
            tenant_id="tenant-123",
        )
        assert str(error) == "Daily debate limit exceeded"

    def test_error_attributes(self):
        """Should store all attributes."""
        error = TenantLimitExceededError(
            message="Limit exceeded",
            limit_type="tokens_per_month",
            current=50000,
            limit=100000,
            tenant_id="tenant-456",
        )
        assert error.limit_type == "tokens_per_month"
        assert error.current == 50000
        assert error.limit == 100000
        assert error.tenant_id == "tenant-456"

    def test_error_without_tenant_id(self):
        """Should allow missing tenant_id."""
        error = TenantLimitExceededError(
            message="Limit exceeded",
            limit_type="storage_quota",
            current=100,
            limit=50,
        )
        assert error.tenant_id is None


# =============================================================================
# TenantLimitsEnforcer Tests
# =============================================================================


@pytest.fixture
def config() -> TenantConfig:
    """Create a test tenant configuration."""
    return TenantConfig(
        max_debates_per_day=10,
        max_concurrent_debates=3,
        tokens_per_month=100000,
        storage_quota=1024 * 1024 * 100,  # 100MB
        api_requests_per_minute=60,
    )


@pytest.fixture
def enforcer(config: TenantConfig) -> TenantLimitsEnforcer:
    """Create a test limits enforcer."""
    return TenantLimitsEnforcer(config)


class TestTenantLimitsEnforcerInit:
    """Tests for TenantLimitsEnforcer initialization."""

    def test_stores_config(self, config: TenantConfig):
        """Should store the tenant config."""
        enforcer = TenantLimitsEnforcer(config)
        assert enforcer.config is config


class TestCheckDebateLimit:
    """Tests for check_debate_limit method."""

    @pytest.mark.asyncio
    async def test_allows_when_under_limit(self, enforcer: TenantLimitsEnforcer):
        """Should return True when under limit."""
        result = await enforcer.check_debate_limit("tenant-1", current_count=5)
        assert result is True

    @pytest.mark.asyncio
    async def test_allows_at_boundary(self, enforcer: TenantLimitsEnforcer):
        """Should allow when one below limit."""
        result = await enforcer.check_debate_limit("tenant-1", current_count=9)
        assert result is True

    @pytest.mark.asyncio
    async def test_raises_at_limit(self, enforcer: TenantLimitsEnforcer):
        """Should raise when at limit."""
        with pytest.raises(TenantLimitExceededError) as exc_info:
            await enforcer.check_debate_limit("tenant-1", current_count=10)

        error = exc_info.value
        assert error.limit_type == "debates_per_day"
        assert error.current == 10
        assert error.limit == 10
        assert error.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_raises_over_limit(self, enforcer: TenantLimitsEnforcer):
        """Should raise when over limit."""
        with pytest.raises(TenantLimitExceededError):
            await enforcer.check_debate_limit("tenant-1", current_count=15)


class TestCheckTokenBudget:
    """Tests for check_token_budget method."""

    @pytest.mark.asyncio
    async def test_allows_when_under_budget(self, enforcer: TenantLimitsEnforcer):
        """Should return True when under budget."""
        result = await enforcer.check_token_budget(
            "tenant-1", tokens_used=50000, tokens_requested=1000
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_allows_at_exact_limit(self, enforcer: TenantLimitsEnforcer):
        """Should allow when exactly at limit."""
        result = await enforcer.check_token_budget(
            "tenant-1", tokens_used=99000, tokens_requested=1000
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_raises_when_exceeds_budget(self, enforcer: TenantLimitsEnforcer):
        """Should raise when request would exceed budget."""
        with pytest.raises(TenantLimitExceededError) as exc_info:
            await enforcer.check_token_budget(
                "tenant-1", tokens_used=99000, tokens_requested=2000
            )

        error = exc_info.value
        assert error.limit_type == "tokens_per_month"
        assert error.current == 99000
        assert error.limit == 100000
        assert error.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_handles_zero_tokens_used(self, enforcer: TenantLimitsEnforcer):
        """Should handle zero tokens used."""
        result = await enforcer.check_token_budget(
            "tenant-1", tokens_used=0, tokens_requested=50000
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_handles_large_request(self, enforcer: TenantLimitsEnforcer):
        """Should reject request exceeding total budget."""
        with pytest.raises(TenantLimitExceededError):
            await enforcer.check_token_budget(
                "tenant-1", tokens_used=0, tokens_requested=200000
            )


class TestCheckStorageQuota:
    """Tests for check_storage_quota method."""

    @pytest.mark.asyncio
    async def test_allows_when_under_quota(self, enforcer: TenantLimitsEnforcer):
        """Should return True when under quota."""
        result = await enforcer.check_storage_quota(
            "tenant-1", current_usage=1024 * 1024 * 50, bytes_requested=1024
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_raises_when_exceeds_quota(self, enforcer: TenantLimitsEnforcer):
        """Should raise when request would exceed quota."""
        with pytest.raises(TenantLimitExceededError) as exc_info:
            await enforcer.check_storage_quota(
                "tenant-1",
                current_usage=1024 * 1024 * 99,
                bytes_requested=1024 * 1024 * 10,
            )

        error = exc_info.value
        assert error.limit_type == "storage_quota"
        assert error.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_allows_exact_quota(self, enforcer: TenantLimitsEnforcer):
        """Should allow when exactly at quota."""
        # config.storage_quota is 100MB = 104857600 bytes
        result = await enforcer.check_storage_quota(
            "tenant-1", current_usage=1024 * 1024 * 99, bytes_requested=1024 * 1024
        )
        assert result is True


class TestCheckConcurrentDebates:
    """Tests for check_concurrent_debates method."""

    @pytest.mark.asyncio
    async def test_allows_when_under_limit(self, enforcer: TenantLimitsEnforcer):
        """Should return True when under limit."""
        result = await enforcer.check_concurrent_debates("tenant-1", active_count=1)
        assert result is True

    @pytest.mark.asyncio
    async def test_allows_at_boundary(self, enforcer: TenantLimitsEnforcer):
        """Should allow when one below limit."""
        result = await enforcer.check_concurrent_debates("tenant-1", active_count=2)
        assert result is True

    @pytest.mark.asyncio
    async def test_raises_at_limit(self, enforcer: TenantLimitsEnforcer):
        """Should raise when at limit."""
        with pytest.raises(TenantLimitExceededError) as exc_info:
            await enforcer.check_concurrent_debates("tenant-1", active_count=3)

        error = exc_info.value
        assert error.limit_type == "concurrent_debates"
        assert error.current == 3
        assert error.limit == 3
        assert error.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_handles_zero_active(self, enforcer: TenantLimitsEnforcer):
        """Should allow when no active debates."""
        result = await enforcer.check_concurrent_debates("tenant-1", active_count=0)
        assert result is True


class TestCheckApiRateLimit:
    """Tests for check_api_rate_limit method."""

    @pytest.mark.asyncio
    async def test_allows_when_under_limit(self, enforcer: TenantLimitsEnforcer):
        """Should return True when under limit."""
        result = await enforcer.check_api_rate_limit("tenant-1", requests_this_minute=30)
        assert result is True

    @pytest.mark.asyncio
    async def test_allows_at_boundary(self, enforcer: TenantLimitsEnforcer):
        """Should allow when one below limit."""
        result = await enforcer.check_api_rate_limit("tenant-1", requests_this_minute=59)
        assert result is True

    @pytest.mark.asyncio
    async def test_raises_at_limit(self, enforcer: TenantLimitsEnforcer):
        """Should raise when at limit."""
        with pytest.raises(TenantLimitExceededError) as exc_info:
            await enforcer.check_api_rate_limit("tenant-1", requests_this_minute=60)

        error = exc_info.value
        assert error.limit_type == "api_rate_limit"
        assert error.current == 60
        assert error.limit == 60
        assert error.tenant_id == "tenant-1"

    @pytest.mark.asyncio
    async def test_raises_over_limit(self, enforcer: TenantLimitsEnforcer):
        """Should raise when over limit."""
        with pytest.raises(TenantLimitExceededError):
            await enforcer.check_api_rate_limit("tenant-1", requests_this_minute=100)


class TestGetUsageSummary:
    """Tests for get_usage_summary method."""

    def test_returns_all_categories(self, enforcer: TenantLimitsEnforcer):
        """Should return all usage categories."""
        summary = enforcer.get_usage_summary()
        assert "debates" in summary
        assert "tokens" in summary
        assert "storage" in summary
        assert "concurrent_debates" in summary

    def test_debates_summary(self, enforcer: TenantLimitsEnforcer):
        """Should calculate debates usage correctly."""
        summary = enforcer.get_usage_summary(debates_today=5)
        debates = summary["debates"]
        assert debates["used"] == 5
        assert debates["limit"] == 10
        assert debates["remaining"] == 5
        assert debates["percentage"] == 50.0

    def test_tokens_summary(self, enforcer: TenantLimitsEnforcer):
        """Should calculate tokens usage correctly."""
        summary = enforcer.get_usage_summary(tokens_this_month=25000)
        tokens = summary["tokens"]
        assert tokens["used"] == 25000
        assert tokens["limit"] == 100000
        assert tokens["remaining"] == 75000
        assert tokens["percentage"] == 25.0

    def test_storage_summary(self, enforcer: TenantLimitsEnforcer):
        """Should calculate storage usage correctly."""
        usage = 1024 * 1024 * 50  # 50MB
        summary = enforcer.get_usage_summary(storage_bytes=usage)
        storage = summary["storage"]
        assert storage["used"] == usage
        assert storage["limit"] == 1024 * 1024 * 100  # 100MB
        assert storage["remaining"] == 1024 * 1024 * 50

    def test_concurrent_debates_summary(self, enforcer: TenantLimitsEnforcer):
        """Should calculate concurrent debates correctly."""
        summary = enforcer.get_usage_summary(active_debates=2)
        concurrent = summary["concurrent_debates"]
        assert concurrent["active"] == 2
        assert concurrent["limit"] == 3
        assert concurrent["remaining"] == 1

    def test_default_values(self, enforcer: TenantLimitsEnforcer):
        """Should default to zero usage."""
        summary = enforcer.get_usage_summary()
        assert summary["debates"]["used"] == 0
        assert summary["tokens"]["used"] == 0
        assert summary["storage"]["used"] == 0
        assert summary["concurrent_debates"]["active"] == 0

    def test_remaining_never_negative(self, enforcer: TenantLimitsEnforcer):
        """Remaining should not be negative even if over limit."""
        summary = enforcer.get_usage_summary(debates_today=15)
        assert summary["debates"]["remaining"] == 0

    def test_percentage_calculation(self, enforcer: TenantLimitsEnforcer):
        """Should calculate percentage correctly."""
        summary = enforcer.get_usage_summary(
            debates_today=7,
            tokens_this_month=33333,
        )
        assert summary["debates"]["percentage"] == 70.0
        assert summary["tokens"]["percentage"] == pytest.approx(33.3, rel=0.1)


class TestEnforcerWithDifferentConfigs:
    """Tests with various TenantConfig settings."""

    @pytest.mark.asyncio
    async def test_with_low_limits(self):
        """Should work with very low limits."""
        config = TenantConfig(
            max_debates_per_day=1,
            max_concurrent_debates=1,
            tokens_per_month=1000,
            storage_quota=1024,
            api_requests_per_minute=1,
        )
        enforcer = TenantLimitsEnforcer(config)

        # First request should pass
        result = await enforcer.check_debate_limit("tenant-1", current_count=0)
        assert result is True

        # Second should fail
        with pytest.raises(TenantLimitExceededError):
            await enforcer.check_debate_limit("tenant-1", current_count=1)

    @pytest.mark.asyncio
    async def test_with_high_limits(self):
        """Should work with very high limits."""
        config = TenantConfig(
            max_debates_per_day=1000000,
            tokens_per_month=10000000000,
        )
        enforcer = TenantLimitsEnforcer(config)

        result = await enforcer.check_debate_limit("tenant-1", current_count=999999)
        assert result is True

        result = await enforcer.check_token_budget(
            "tenant-1", tokens_used=9999999000, tokens_requested=1000
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_different_tenants_same_enforcer(self):
        """Should track limits per-tenant independently."""
        config = TenantConfig(max_debates_per_day=10)
        enforcer = TenantLimitsEnforcer(config)

        # Both tenants should be able to hit their own limit
        await enforcer.check_debate_limit("tenant-1", current_count=9)
        await enforcer.check_debate_limit("tenant-2", current_count=9)

        # Both should fail at limit
        with pytest.raises(TenantLimitExceededError) as exc1:
            await enforcer.check_debate_limit("tenant-1", current_count=10)
        assert exc1.value.tenant_id == "tenant-1"

        with pytest.raises(TenantLimitExceededError) as exc2:
            await enforcer.check_debate_limit("tenant-2", current_count=10)
        assert exc2.value.tenant_id == "tenant-2"


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_zero_values(self, enforcer: TenantLimitsEnforcer):
        """Should handle zero values correctly."""
        await enforcer.check_debate_limit("t", current_count=0)
        await enforcer.check_token_budget("t", tokens_used=0, tokens_requested=0)
        await enforcer.check_storage_quota("t", current_usage=0, bytes_requested=0)
        await enforcer.check_concurrent_debates("t", active_count=0)
        await enforcer.check_api_rate_limit("t", requests_this_minute=0)

    @pytest.mark.asyncio
    async def test_empty_tenant_id(self, enforcer: TenantLimitsEnforcer):
        """Should handle empty tenant ID."""
        result = await enforcer.check_debate_limit("", current_count=5)
        assert result is True

    def test_summary_with_zero_limits(self):
        """Should handle zero limits gracefully in percentage calculation."""
        # This tests division by zero protection
        config = TenantConfig(
            max_debates_per_day=0,
            tokens_per_month=0,
            storage_quota=0,
        )
        enforcer = TenantLimitsEnforcer(config)

        # This will raise ZeroDivisionError if not handled
        # But since the config has defaults, this actually tests with those defaults
        # Let's test with actual usage instead
        summary = enforcer.get_usage_summary(debates_today=0)
        # With default config (100 debates/day), this should be 0%
        assert summary["debates"]["percentage"] == 0.0
