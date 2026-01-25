"""Tests for tenant quota management."""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from aragora.tenancy.context import TenantContext
from aragora.tenancy.quotas import (
    QuotaConfig,
    QuotaExceeded,
    QuotaLimit,
    QuotaManager,
    QuotaPeriod,
    QuotaStatus,
    UsageRecord,
)
from aragora.tenancy.tenant import Tenant, TenantConfig, TenantTier


class TestQuotaPeriod:
    """Tests for QuotaPeriod enum."""

    def test_all_periods_exist(self):
        """Test all expected periods exist."""
        assert QuotaPeriod.MINUTE.value == "minute"
        assert QuotaPeriod.HOUR.value == "hour"
        assert QuotaPeriod.DAY.value == "day"
        assert QuotaPeriod.WEEK.value == "week"
        assert QuotaPeriod.MONTH.value == "month"
        assert QuotaPeriod.UNLIMITED.value == "unlimited"

    def test_period_count(self):
        """Test expected number of periods."""
        assert len(QuotaPeriod) == 6


class TestQuotaLimit:
    """Tests for QuotaLimit dataclass."""

    def test_create_basic_limit(self):
        """Test creating a basic quota limit."""
        limit = QuotaLimit(
            resource="api_requests",
            limit=100,
        )

        assert limit.resource == "api_requests"
        assert limit.limit == 100
        assert limit.period == QuotaPeriod.DAY
        assert limit.soft_limit is None
        assert limit.burst_limit is None

    def test_create_full_limit(self):
        """Test creating a fully specified quota limit."""
        limit = QuotaLimit(
            resource="tokens",
            limit=1000,
            period=QuotaPeriod.MONTH,
            soft_limit=800,
            burst_limit=1500,
        )

        assert limit.resource == "tokens"
        assert limit.limit == 1000
        assert limit.period == QuotaPeriod.MONTH
        assert limit.soft_limit == 800
        assert limit.burst_limit == 1500

    def test_period_seconds_minute(self):
        """Test period_seconds for MINUTE."""
        limit = QuotaLimit("test", 100, QuotaPeriod.MINUTE)
        assert limit.period_seconds == 60

    def test_period_seconds_hour(self):
        """Test period_seconds for HOUR."""
        limit = QuotaLimit("test", 100, QuotaPeriod.HOUR)
        assert limit.period_seconds == 3600

    def test_period_seconds_day(self):
        """Test period_seconds for DAY."""
        limit = QuotaLimit("test", 100, QuotaPeriod.DAY)
        assert limit.period_seconds == 86400

    def test_period_seconds_week(self):
        """Test period_seconds for WEEK."""
        limit = QuotaLimit("test", 100, QuotaPeriod.WEEK)
        assert limit.period_seconds == 604800

    def test_period_seconds_month(self):
        """Test period_seconds for MONTH (30 days)."""
        limit = QuotaLimit("test", 100, QuotaPeriod.MONTH)
        assert limit.period_seconds == 2592000

    def test_period_seconds_unlimited(self):
        """Test period_seconds for UNLIMITED."""
        limit = QuotaLimit("test", 100, QuotaPeriod.UNLIMITED)
        assert limit.period_seconds == 0


class TestQuotaExceeded:
    """Tests for QuotaExceeded exception."""

    def test_basic_exception(self):
        """Test basic QuotaExceeded creation."""
        exc = QuotaExceeded(
            message="Quota exceeded",
            resource="api_requests",
            limit=100,
            current=95,
            period=QuotaPeriod.MINUTE,
        )

        assert str(exc) == "Quota exceeded"
        assert exc.resource == "api_requests"
        assert exc.limit == 100
        assert exc.current == 95
        assert exc.period == QuotaPeriod.MINUTE
        assert exc.tenant_id is None
        assert exc.retry_after is None

    def test_exception_with_tenant_and_retry(self):
        """Test QuotaExceeded with tenant and retry_after."""
        exc = QuotaExceeded(
            message="Rate limit",
            resource="tokens",
            limit=1000,
            current=1000,
            period=QuotaPeriod.MONTH,
            tenant_id="tenant-123",
            retry_after=3600,
        )

        assert exc.tenant_id == "tenant-123"
        assert exc.retry_after == 3600

    def test_exception_inheritance(self):
        """Test exception inherits from Exception."""
        assert issubclass(QuotaExceeded, Exception)


class TestQuotaConfig:
    """Tests for QuotaConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = QuotaConfig()

        assert config.limits == []
        assert config.strict_enforcement is True
        assert config.warn_at_threshold == 0.8
        assert config.enable_rate_limiting is True
        assert config.rate_limit_window == 60
        assert config.persist_usage is True

    def test_default_limits(self):
        """Test default_limits factory creates expected limits."""
        config = QuotaConfig.default_limits()

        # Should have limits for key resources
        resource_names = [limit.resource for limit in config.limits]
        assert "api_requests" in resource_names
        assert "debates" in resource_names
        assert "tokens" in resource_names
        assert "storage_bytes" in resource_names
        assert "users" in resource_names
        assert "connectors" in resource_names

    def test_default_limits_values(self):
        """Test specific default limit values."""
        config = QuotaConfig.default_limits()

        # Find specific limits
        limits_by_resource = {(lim.resource, lim.period): lim for lim in config.limits}

        # API requests per minute
        api_minute = limits_by_resource.get(("api_requests", QuotaPeriod.MINUTE))
        assert api_minute is not None
        assert api_minute.limit == 60
        assert api_minute.burst_limit == 100

        # Debates per day
        debates = limits_by_resource.get(("debates", QuotaPeriod.DAY))
        assert debates is not None
        assert debates.limit == 100

        # Tokens per month
        tokens = limits_by_resource.get(("tokens", QuotaPeriod.MONTH))
        assert tokens is not None
        assert tokens.limit == 1_000_000


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_create_usage_record(self):
        """Test creating a usage record."""
        now = datetime.now()
        record = UsageRecord(
            resource="api_requests",
            tenant_id="tenant-123",
            count=50,
            period_start=now,
            period=QuotaPeriod.MINUTE,
        )

        assert record.resource == "api_requests"
        assert record.tenant_id == "tenant-123"
        assert record.count == 50
        assert record.period_start == now
        assert record.period == QuotaPeriod.MINUTE


class TestQuotaStatus:
    """Tests for QuotaStatus dataclass."""

    def test_create_quota_status(self):
        """Test creating a quota status."""
        reset_time = datetime.now() + timedelta(hours=1)
        status = QuotaStatus(
            resource="api_requests",
            limit=100,
            current=75,
            remaining=25,
            period=QuotaPeriod.HOUR,
            period_resets_at=reset_time,
            percentage_used=0.75,
            is_exceeded=False,
            is_warning=False,
        )

        assert status.resource == "api_requests"
        assert status.limit == 100
        assert status.current == 75
        assert status.remaining == 25
        assert status.percentage_used == 0.75
        assert status.is_exceeded is False
        assert status.is_warning is False

    def test_exceeded_status(self):
        """Test status when quota is exceeded."""
        status = QuotaStatus(
            resource="tokens",
            limit=1000,
            current=1100,
            remaining=0,
            period=QuotaPeriod.MONTH,
            period_resets_at=None,
            percentage_used=1.1,
            is_exceeded=True,
            is_warning=True,
        )

        assert status.is_exceeded is True
        assert status.remaining == 0


class TestQuotaManager:
    """Tests for QuotaManager class."""

    def test_init_with_default_config(self):
        """Test initializing with default configuration."""
        manager = QuotaManager()

        assert manager.config is not None
        assert len(manager.config.limits) > 0

    def test_init_with_custom_config(self):
        """Test initializing with custom configuration."""
        config = QuotaConfig(
            limits=[
                QuotaLimit("custom_resource", 50, QuotaPeriod.HOUR),
            ],
            strict_enforcement=False,
        )
        manager = QuotaManager(config)

        assert len(manager.config.limits) == 1
        assert manager.config.strict_enforcement is False


class TestQuotaManagerCheckQuota:
    """Tests for QuotaManager.check_quota method."""

    @pytest.mark.asyncio
    async def test_check_quota_no_tenant(self):
        """Test check_quota returns True when no tenant context."""
        manager = QuotaManager()
        result = await manager.check_quota("api_requests", 100)
        assert result is True

    @pytest.mark.asyncio
    async def test_check_quota_under_limit(self):
        """Test check_quota returns True when under limit."""
        config = QuotaConfig(
            limits=[QuotaLimit("api_requests", 100, QuotaPeriod.MINUTE)],
        )
        manager = QuotaManager(config)

        tenant = Tenant(id="test", name="Test", slug="test")
        with TenantContext(tenant=tenant):
            result = await manager.check_quota("api_requests", 50)
            assert result is True

    @pytest.mark.asyncio
    async def test_check_quota_unknown_resource(self):
        """Test check_quota returns True for unknown resource."""
        manager = QuotaManager(QuotaConfig(limits=[]))

        tenant = Tenant(id="test", name="Test", slug="test")
        with TenantContext(tenant=tenant):
            result = await manager.check_quota("unknown_resource", 100)
            assert result is True

    @pytest.mark.asyncio
    async def test_check_quota_with_tenant_id(self):
        """Test check_quota with explicit tenant_id."""
        config = QuotaConfig(
            limits=[QuotaLimit("tokens", 1000, QuotaPeriod.MONTH)],
        )
        manager = QuotaManager(config)

        result = await manager.check_quota("tokens", 500, tenant_id="explicit-tenant")
        assert result is True


class TestQuotaManagerConsume:
    """Tests for QuotaManager.consume method."""

    @pytest.mark.asyncio
    async def test_consume_no_tenant(self):
        """Test consume does nothing when no tenant context."""
        manager = QuotaManager()
        # Should not raise
        await manager.consume("api_requests", 100)

    @pytest.mark.asyncio
    async def test_consume_raises_when_exceeded(self):
        """Test consume raises QuotaExceeded when limit exceeded."""
        config = QuotaConfig(
            limits=[QuotaLimit("api_requests", 10, QuotaPeriod.MINUTE)],
            strict_enforcement=True,
        )
        manager = QuotaManager(config)

        # Create tenant with matching config limit
        tenant_config = TenantConfig(api_requests_per_minute=10)
        tenant = Tenant(id="test", name="Test", slug="test", config=tenant_config)
        with TenantContext(tenant=tenant):
            # First consume up to limit
            for _ in range(10):
                await manager.consume("api_requests", 1)

            # This should exceed the limit
            with pytest.raises(QuotaExceeded) as exc_info:
                await manager.consume("api_requests", 1)

            assert exc_info.value.resource == "api_requests"
            assert exc_info.value.limit == 10

    @pytest.mark.asyncio
    async def test_consume_no_exception_when_not_strict(self):
        """Test consume doesn't raise when strict_enforcement is False."""
        config = QuotaConfig(
            limits=[QuotaLimit("api_requests", 10, QuotaPeriod.MINUTE)],
            strict_enforcement=False,
        )
        manager = QuotaManager(config)

        tenant = Tenant(id="test", name="Test", slug="test")
        with TenantContext(tenant=tenant):
            # Consume more than limit
            await manager.consume("api_requests", 100)
            # Should not raise


class TestQuotaManagerPeriodCalculation:
    """Tests for period calculation methods."""

    def test_get_period_key_minute(self):
        """Test period key for MINUTE."""
        manager = QuotaManager()
        key = manager._get_period_key(QuotaPeriod.MINUTE)

        # Should be in format YYYY-MM-DD-HH-MM
        parts = key.split("-")
        assert len(parts) == 5

    def test_get_period_key_hour(self):
        """Test period key for HOUR."""
        manager = QuotaManager()
        key = manager._get_period_key(QuotaPeriod.HOUR)

        # Should be in format YYYY-MM-DD-HH
        parts = key.split("-")
        assert len(parts) == 4

    def test_get_period_key_day(self):
        """Test period key for DAY."""
        manager = QuotaManager()
        key = manager._get_period_key(QuotaPeriod.DAY)

        # Should be in format YYYY-MM-DD
        parts = key.split("-")
        assert len(parts) == 3

    def test_get_period_key_month(self):
        """Test period key for MONTH."""
        manager = QuotaManager()
        key = manager._get_period_key(QuotaPeriod.MONTH)

        # Should be in format YYYY-MM
        parts = key.split("-")
        assert len(parts) == 2

    def test_get_period_key_unlimited(self):
        """Test period key for UNLIMITED."""
        manager = QuotaManager()
        key = manager._get_period_key(QuotaPeriod.UNLIMITED)
        assert key == "unlimited"

    def test_get_period_start_day(self):
        """Test period start for DAY."""
        manager = QuotaManager()
        start = manager._get_period_start(QuotaPeriod.DAY)

        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0

    def test_get_period_start_month(self):
        """Test period start for MONTH."""
        manager = QuotaManager()
        start = manager._get_period_start(QuotaPeriod.MONTH)

        assert start.day == 1
        assert start.hour == 0

    def test_get_period_end_hour(self):
        """Test period end for HOUR."""
        manager = QuotaManager()
        end = manager._get_period_end(QuotaPeriod.HOUR)
        start = manager._get_period_start(QuotaPeriod.HOUR)

        assert end is not None
        assert end - start == timedelta(hours=1)

    def test_get_period_end_day(self):
        """Test period end for DAY."""
        manager = QuotaManager()
        end = manager._get_period_end(QuotaPeriod.DAY)
        start = manager._get_period_start(QuotaPeriod.DAY)

        assert end is not None
        assert end - start == timedelta(days=1)

    def test_get_period_end_unlimited(self):
        """Test period end for UNLIMITED returns None."""
        manager = QuotaManager()
        end = manager._get_period_end(QuotaPeriod.UNLIMITED)
        assert end is None


class TestQuotaManagerTenantIntegration:
    """Tests for QuotaManager integration with tenant configuration."""

    @pytest.mark.asyncio
    async def test_uses_tenant_config_limits(self):
        """Test that manager uses tenant-specific limits from config."""
        manager = QuotaManager()

        # Create tenant with custom config
        config = TenantConfig(
            max_debates_per_day=5,
            api_requests_per_minute=10,
        )
        tenant = Tenant(
            id="custom-tenant",
            name="Custom",
            slug="custom",
            config=config,
        )

        with TenantContext(tenant=tenant):
            limits = manager._get_limits_for_tenant("custom-tenant")

            assert limits["debates"].limit == 5
            assert limits["api_requests"].limit == 10

    @pytest.mark.asyncio
    async def test_limits_cache(self):
        """Test that limits are cached per tenant."""
        manager = QuotaManager()

        tenant = Tenant(id="cache-test", name="Cache", slug="cache")
        with TenantContext(tenant=tenant):
            # First call populates cache
            limits1 = manager._get_limits_for_tenant("cache-test")
            # Second call uses cache
            limits2 = manager._get_limits_for_tenant("cache-test")

            assert limits1 is limits2

    @pytest.mark.asyncio
    async def test_different_tenants_different_limits(self):
        """Test that different tenants can have different limits."""
        manager = QuotaManager()

        config_free = TenantConfig.for_tier(TenantTier.FREE)
        config_pro = TenantConfig.for_tier(TenantTier.PROFESSIONAL)

        free_tenant = Tenant(id="free-tenant", name="Free", slug="free", config=config_free)
        pro_tenant = Tenant(id="pro-tenant", name="Pro", slug="pro", config=config_pro)

        with TenantContext(tenant=free_tenant):
            free_limits = manager._get_limits_for_tenant("free-tenant")

        # Clear cache to get pro limits
        manager._limits_cache.clear()

        with TenantContext(tenant=pro_tenant):
            pro_limits = manager._get_limits_for_tenant("pro-tenant")

        # Pro should have higher limits
        assert pro_limits["debates"].limit > free_limits["debates"].limit
        assert pro_limits["tokens"].limit > free_limits["tokens"].limit
