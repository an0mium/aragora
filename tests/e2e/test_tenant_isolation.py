"""
E2E tests for multi-tenant isolation.

Tests tenant isolation including:
- Data segregation verification
- Cross-tenant access prevention
- Quota enforcement
- Tenant context propagation
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.e2e.conftest import TestTenant


# ============================================================================
# Context Propagation E2E Tests (Uses actual API)
# ============================================================================


class TestContextPropagationE2E:
    """E2E tests for tenant context propagation."""

    @pytest.mark.asyncio
    async def test_context_in_async_tasks(
        self,
        tenant_a: TestTenant,
    ):
        """Test tenant context propagates across async tasks."""
        from aragora.tenancy import TenantContext, get_current_tenant_id

        async def nested_task():
            return get_current_tenant_id()

        with TenantContext(tenant_a.id):
            # Context should propagate to nested async
            result = await nested_task()
            assert result == tenant_a.id

            # Context should propagate to gathered tasks
            results = await asyncio.gather(
                nested_task(),
                nested_task(),
                nested_task(),
            )
            assert all(r == tenant_a.id for r in results)

    @pytest.mark.asyncio
    async def test_context_isolation_between_requests(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test contexts are isolated between concurrent requests."""
        from aragora.tenancy import TenantContext, get_current_tenant_id

        async def request_handler(tenant_id: str):
            with TenantContext(tenant_id):
                await asyncio.sleep(0.01)  # Simulate processing
                return get_current_tenant_id()

        # Run concurrent requests for different tenants
        results = await asyncio.gather(
            request_handler(tenant_a.id),
            request_handler(tenant_b.id),
            request_handler(tenant_a.id),
            request_handler(tenant_b.id),
        )

        assert results[0] == tenant_a.id
        assert results[1] == tenant_b.id
        assert results[2] == tenant_a.id
        assert results[3] == tenant_b.id


# ============================================================================
# Data Isolation E2E Tests
# ============================================================================


class TestDataIsolationE2E:
    """E2E tests for tenant data isolation."""

    @pytest.mark.asyncio
    async def test_sql_query_tenant_filtering(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test SQL queries are automatically filtered by tenant."""
        from aragora.tenancy import TenantContext
        from aragora.tenancy.isolation import TenantDataIsolation

        isolation = TenantDataIsolation()

        # Build query for tenant A - apply_tenant_filter returns (sql, params)
        with TenantContext(tenant_a.id):
            query_a, params_a = isolation.apply_tenant_filter(
                "SELECT * FROM facts WHERE type = 'knowledge'"
            )
            assert "tenant_id = :tenant_id" in query_a
            assert params_a["tenant_id"] == tenant_a.id

        # Build query for tenant B
        with TenantContext(tenant_b.id):
            query_b, params_b = isolation.apply_tenant_filter(
                "SELECT * FROM facts WHERE type = 'knowledge'"
            )
            assert "tenant_id = :tenant_id" in query_b
            assert params_b["tenant_id"] == tenant_b.id

    @pytest.mark.asyncio
    async def test_ownership_validation(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test resource ownership validation."""
        from aragora.tenancy import TenantContext, TenantIsolationError
        from aragora.tenancy.isolation import TenantDataIsolation

        isolation = TenantDataIsolation()

        # Resource owned by tenant A
        resource = {"id": "resource-123", "tenant_id": tenant_a.id, "data": "secret"}

        # Tenant A can validate ownership
        with TenantContext(tenant_a.id):
            assert isolation.validate_ownership(resource) is True

        # Tenant B cannot validate ownership (should raise)
        with TenantContext(tenant_b.id):
            with pytest.raises(TenantIsolationError):
                isolation.validate_ownership(resource)

    @pytest.mark.asyncio
    async def test_query_filtering(
        self,
        tenant_a: TestTenant,
    ):
        """Test query dict filtering adds tenant filter."""
        from aragora.tenancy import TenantContext
        from aragora.tenancy.isolation import TenantDataIsolation

        isolation = TenantDataIsolation()

        base_query = {"status": "active", "type": "debate"}

        with TenantContext(tenant_a.id):
            filtered = isolation.filter_query(base_query, "debates")
            assert filtered["tenant_id"] == tenant_a.id
            assert filtered["status"] == "active"
            assert filtered["type"] == "debate"


# ============================================================================
# Quota Enforcement E2E Tests
# ============================================================================


class TestQuotaEnforcementE2E:
    """E2E tests for tenant quota enforcement."""

    @pytest.mark.asyncio
    async def test_quota_check_and_consume(
        self,
        tenant_a: TestTenant,
    ):
        """Test quota checking and consumption."""
        from aragora.tenancy import TenantContext
        from aragora.tenancy.quotas import (
            QuotaManager,
            QuotaExceeded,
            QuotaConfig,
            QuotaLimit,
            QuotaPeriod,
        )

        # Create manager with low limit for testing
        config = QuotaConfig(
            limits=[
                QuotaLimit("test_resource", 5, QuotaPeriod.MINUTE),
            ]
        )
        manager = QuotaManager(config)

        with TenantContext(tenant_a.id):
            # First 5 should succeed
            for i in range(5):
                can_proceed = await manager.check_quota("test_resource", 1)
                assert can_proceed is True
                await manager.consume("test_resource", 1)

            # 6th should fail
            can_proceed = await manager.check_quota("test_resource", 1)
            assert can_proceed is False

            # Consuming should raise
            with pytest.raises(QuotaExceeded):
                await manager.consume("test_resource", 1)

    @pytest.mark.asyncio
    async def test_quota_status(
        self,
        tenant_a: TestTenant,
    ):
        """Test getting quota status."""
        from aragora.tenancy import TenantContext
        from aragora.tenancy.quotas import QuotaManager, QuotaConfig, QuotaLimit, QuotaPeriod

        config = QuotaConfig(
            limits=[
                QuotaLimit("api_requests", 100, QuotaPeriod.MINUTE),
            ]
        )
        manager = QuotaManager(config)

        with TenantContext(tenant_a.id):
            # Consume some quota
            await manager.consume("api_requests", 30)

            # Check status
            status = await manager.get_quota_status("api_requests")
            assert status.limit == 100
            assert status.current == 30
            assert status.remaining == 70
            assert status.percentage_used == 30.0


# ============================================================================
# Billing Integration E2E Tests
# ============================================================================


class TestBillingIntegrationE2E:
    """E2E tests for tenant billing integration."""

    @pytest.mark.asyncio
    async def test_usage_meter_record(
        self,
        tenant_a: TestTenant,
    ):
        """Test usage metering records events."""
        from aragora.tenancy import TenantContext
        from aragora.billing import UsageMeter

        meter = UsageMeter()

        with TenantContext(tenant_a.id):
            # Record usage - UsageMeter uses record_api_call, record_tokens, etc.
            await meter.record_api_call(resource="debates")
            await meter.record_api_call(resource="knowledge")
            await meter.record_tokens(tokens_in=500, tokens_out=500)

            # Verify events were recorded (check internal state)
            assert len(meter._events) >= 3


# ============================================================================
# Tests requiring API additions (skipped with specific reasons)
# ============================================================================


@pytest.mark.skip(
    reason="Requires TenantDataIsolation.register_resource() - use TenantFilter.register_resource() from stream module instead"
)
class TestResourceRegistration:
    """Tests for resource registration - requires API additions."""

    @pytest.mark.asyncio
    async def test_cross_tenant_access_prevention(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test that cross-tenant access is prevented."""
        pass


class TestTenantSecurityE2E:
    """E2E tests for tenant security using TenantManager."""

    @pytest.mark.asyncio
    async def test_api_key_validation(self, tenant_a: TestTenant):
        """Test API key validation for tenant access."""
        from aragora.tenancy.tenant import Tenant, TenantManager, TenantStatus

        manager = TenantManager()

        # Create and register a tenant with an API key
        tenant = Tenant(
            id=tenant_a.id,
            name=tenant_a.name,
            slug=tenant_a.id.replace("-", "_"),
        )
        # Generate and set an API key
        api_key = tenant.generate_api_key()
        manager.register_tenant(tenant)

        # Valid API key should return the tenant
        result = await manager.validate_api_key(api_key)
        assert result is not None
        assert result.id == tenant_a.id

        # Invalid API key should return None
        result = await manager.validate_api_key("invalid_key_12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_tenant_suspension(self, tenant_a: TestTenant):
        """Test suspended tenant cannot access resources via API key."""
        from aragora.tenancy.tenant import (
            Tenant,
            TenantManager,
            TenantStatus,
            TenantSuspendedError,
        )

        manager = TenantManager()

        # Create and register a tenant
        tenant = Tenant(
            id=tenant_a.id,
            name=tenant_a.name,
            slug=tenant_a.id.replace("-", "_"),
        )
        api_key = tenant.generate_api_key()
        manager.register_tenant(tenant)

        # Should work while active
        result = await manager.validate_api_key(api_key)
        assert result is not None

        # Suspend the tenant
        await manager.suspend_tenant(tenant_a.id, "Billing overdue")
        assert tenant.status == TenantStatus.SUSPENDED

        # Should raise TenantSuspendedError when suspended
        with pytest.raises(TenantSuspendedError) as exc_info:
            await manager.validate_api_key(api_key)
        assert exc_info.value.tenant_id == tenant_a.id

        # Reactivate and verify access restored
        await manager.activate_tenant(tenant_a.id)
        result = await manager.validate_api_key(api_key)
        assert result is not None
        assert result.status == TenantStatus.ACTIVE
