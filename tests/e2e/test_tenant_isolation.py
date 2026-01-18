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
import pytest_asyncio

from tests.e2e.conftest import TestTenant


# ============================================================================
# Data Isolation E2E Tests
# ============================================================================


class TestDataIsolationE2E:
    """E2E tests for tenant data isolation."""

    @pytest.mark.asyncio
    async def test_tenant_data_segregation(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test that tenant data is properly segregated."""
        from aragora.tenancy import TenantContext
        from aragora.knowledge.mound.facade import KnowledgeMound

        # Create data for tenant A
        with TenantContext(tenant_a.id):
            mound_a = KnowledgeMound()
            await mound_a.add_node(
                title="Tenant A Secret",
                content="Confidential information for Tenant A",
                node_type="document",
            )

        # Create data for tenant B
        with TenantContext(tenant_b.id):
            mound_b = KnowledgeMound()
            await mound_b.add_node(
                title="Tenant B Secret",
                content="Confidential information for Tenant B",
                node_type="document",
            )

        # Verify tenant A can only see their data
        with TenantContext(tenant_a.id):
            results_a = await mound_a.search("Secret")
            assert all("Tenant A" in r.title for r in results_a)
            assert not any("Tenant B" in r.title for r in results_a)

        # Verify tenant B can only see their data
        with TenantContext(tenant_b.id):
            results_b = await mound_b.search("Secret")
            assert all("Tenant B" in r.title for r in results_b)
            assert not any("Tenant A" in r.title for r in results_b)

    @pytest.mark.asyncio
    async def test_cross_tenant_access_prevention(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test that cross-tenant access is prevented."""
        from aragora.tenancy import TenantContext, TenantIsolationError
        from aragora.tenancy.isolation import TenantDataIsolation

        isolation = TenantDataIsolation()

        # Create resource owned by tenant A
        resource_id = "resource-123"
        await isolation.register_resource(resource_id, tenant_a.id)

        # Tenant A can access their resource
        with TenantContext(tenant_a.id):
            assert await isolation.can_access(resource_id)

        # Tenant B cannot access tenant A's resource
        with TenantContext(tenant_b.id):
            assert not await isolation.can_access(resource_id)

            # Attempting direct access should raise error
            with pytest.raises(TenantIsolationError):
                await isolation.get_resource(resource_id)

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

        # Build query for tenant A
        with TenantContext(tenant_a.id):
            query_a = isolation.apply_tenant_filter(
                "SELECT * FROM facts WHERE type = 'knowledge'"
            )
            assert f"tenant_id = '{tenant_a.id}'" in query_a

        # Build query for tenant B
        with TenantContext(tenant_b.id):
            query_b = isolation.apply_tenant_filter(
                "SELECT * FROM facts WHERE type = 'knowledge'"
            )
            assert f"tenant_id = '{tenant_b.id}'" in query_b

    @pytest.mark.asyncio
    async def test_debate_isolation(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test debates are isolated between tenants."""
        from aragora.tenancy import TenantContext
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        # Run debate for tenant A
        with TenantContext(tenant_a.id):
            env_a = Environment(task="Tenant A Topic")
            protocol = DebateProtocol(rounds=1)
            mock_agents = [MagicMock()]
            mock_agents[0].generate = AsyncMock(return_value="Response")

            arena_a = Arena(env_a, mock_agents, protocol)
            result_a = await arena_a.run()

            debate_id_a = result_a.debate_id

        # Run debate for tenant B
        with TenantContext(tenant_b.id):
            env_b = Environment(task="Tenant B Topic")
            arena_b = Arena(env_b, mock_agents, protocol)
            result_b = await arena_b.run()

            debate_id_b = result_b.debate_id

        # Verify debates are separate
        assert debate_id_a != debate_id_b

        # Tenant A cannot see tenant B's debate
        with TenantContext(tenant_a.id):
            from aragora.debate.store import DebateStore
            store = DebateStore()
            with pytest.raises(Exception):  # Should fail or return None
                await store.get_debate(debate_id_b)


# ============================================================================
# Quota Enforcement E2E Tests
# ============================================================================


class TestQuotaEnforcementE2E:
    """E2E tests for tenant quota enforcement."""

    @pytest.mark.asyncio
    async def test_api_rate_limiting(
        self,
        tenant_a: TestTenant,
    ):
        """Test API rate limits are enforced per tenant."""
        from aragora.tenancy import TenantContext
        from aragora.tenancy.quotas import QuotaManager, QuotaExceeded

        # Configure low rate limit for testing
        manager = QuotaManager()
        manager.configure_tenant(tenant_a.id, {
            "api_requests_per_minute": 5,
        })

        with TenantContext(tenant_a.id):
            # First 5 requests should succeed
            for _ in range(5):
                await manager.check_quota("api_requests")

            # 6th request should fail
            with pytest.raises(QuotaExceeded):
                await manager.check_quota("api_requests")

    @pytest.mark.asyncio
    async def test_storage_quota(
        self,
        tenant_a: TestTenant,
    ):
        """Test storage quotas are enforced."""
        from aragora.tenancy import TenantContext
        from aragora.tenancy.quotas import QuotaManager, QuotaExceeded

        manager = QuotaManager()
        manager.configure_tenant(tenant_a.id, {
            "storage_bytes": 1024,  # 1KB limit
        })

        with TenantContext(tenant_a.id):
            # First addition should succeed
            await manager.record_usage("storage", 500)

            # Adding more should fail
            with pytest.raises(QuotaExceeded):
                await manager.record_usage("storage", 600)

    @pytest.mark.asyncio
    async def test_debate_round_limits(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test debate round limits differ by tier."""
        from aragora.tenancy import TenantContext
        from aragora.tenancy.quotas import QuotaManager

        manager = QuotaManager()

        # Enterprise tier (tenant A) has higher limits
        manager.configure_tenant(tenant_a.id, {
            "max_debate_rounds": 100,
            "tier": "enterprise",
        })

        # Standard tier (tenant B) has lower limits
        manager.configure_tenant(tenant_b.id, {
            "max_debate_rounds": 10,
            "tier": "standard",
        })

        with TenantContext(tenant_a.id):
            limit_a = await manager.get_limit("max_debate_rounds")
            assert limit_a == 100

        with TenantContext(tenant_b.id):
            limit_b = await manager.get_limit("max_debate_rounds")
            assert limit_b == 10

    @pytest.mark.asyncio
    async def test_connector_limits(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test connector count limits by tier."""
        from aragora.tenancy import TenantContext
        from aragora.tenancy.quotas import QuotaManager, QuotaExceeded

        manager = QuotaManager()
        manager.configure_tenant(tenant_b.id, {
            "max_connectors": 2,
            "tier": "standard",
        })

        with TenantContext(tenant_b.id):
            # Add 2 connectors
            await manager.record_usage("connectors", 1)
            await manager.record_usage("connectors", 1)

            # 3rd connector should fail
            with pytest.raises(QuotaExceeded):
                await manager.record_usage("connectors", 1)


# ============================================================================
# Context Propagation E2E Tests
# ============================================================================


class TestContextPropagationE2E:
    """E2E tests for tenant context propagation."""

    @pytest.mark.asyncio
    async def test_context_in_async_tasks(
        self,
        tenant_a: TestTenant,
    ):
        """Test tenant context propagates across async tasks."""
        from aragora.tenancy import TenantContext, get_current_tenant

        async def nested_task():
            return get_current_tenant()

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
    async def test_context_in_thread_pool(
        self,
        tenant_a: TestTenant,
    ):
        """Test tenant context propagates to thread pool."""
        import concurrent.futures
        from aragora.tenancy import TenantContext, get_current_tenant

        def sync_task():
            return get_current_tenant()

        with TenantContext(tenant_a.id):
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = await loop.run_in_executor(pool, sync_task)
                assert result == tenant_a.id

    @pytest.mark.asyncio
    async def test_context_isolation_between_requests(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test contexts are isolated between concurrent requests."""
        from aragora.tenancy import TenantContext, get_current_tenant

        async def request_handler(tenant_id: str):
            with TenantContext(tenant_id):
                await asyncio.sleep(0.1)  # Simulate processing
                return get_current_tenant()

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
# Billing Integration E2E Tests
# ============================================================================


class TestBillingIntegrationE2E:
    """E2E tests for tenant billing integration."""

    @pytest.mark.asyncio
    async def test_usage_tracking_per_tenant(
        self,
        tenant_a: TestTenant,
        tenant_b: TestTenant,
    ):
        """Test usage is tracked separately per tenant."""
        from aragora.tenancy import TenantContext
        from aragora.billing import UsageMeter

        meter = UsageMeter()

        # Record usage for tenant A
        with TenantContext(tenant_a.id):
            await meter.record("api_calls", 100)
            await meter.record("tokens", 5000)

        # Record usage for tenant B
        with TenantContext(tenant_b.id):
            await meter.record("api_calls", 50)
            await meter.record("tokens", 2500)

        # Verify separate tracking
        summary_a = await meter.get_summary(tenant_a.id)
        summary_b = await meter.get_summary(tenant_b.id)

        assert summary_a["api_calls"] == 100
        assert summary_b["api_calls"] == 50

    @pytest.mark.asyncio
    async def test_billing_period_rollup(
        self,
        tenant_a: TestTenant,
    ):
        """Test usage rollup per billing period."""
        from aragora.tenancy import TenantContext
        from aragora.billing import UsageMeter, BillingPeriod

        meter = UsageMeter()

        with TenantContext(tenant_a.id):
            # Record over time
            await meter.record("api_calls", 10)
            await meter.record("api_calls", 20)
            await meter.record("api_calls", 30)

        # Get period summary
        summary = await meter.get_period_summary(
            tenant_a.id,
            period=BillingPeriod.CURRENT_MONTH,
        )

        assert summary["api_calls"] == 60

    @pytest.mark.asyncio
    async def test_cost_projection(
        self,
        tenant_a: TestTenant,
    ):
        """Test cost projection based on usage."""
        from aragora.tenancy import TenantContext
        from aragora.billing import UsageMeter

        meter = UsageMeter()

        with TenantContext(tenant_a.id):
            await meter.record("tokens", 1_000_000)  # 1M tokens

        projection = await meter.project_monthly_cost(tenant_a.id)

        assert projection is not None
        assert projection["estimated_cost"] > 0


# ============================================================================
# Security E2E Tests
# ============================================================================


class TestTenantSecurityE2E:
    """E2E tests for tenant security."""

    @pytest.mark.asyncio
    async def test_api_key_validation(
        self,
        tenant_a: TestTenant,
    ):
        """Test API key validation for tenant access."""
        from aragora.tenancy.tenant import TenantManager

        manager = TenantManager()

        # Valid key should work
        tenant = await manager.validate_api_key(tenant_a.api_key)
        assert tenant is not None
        assert tenant.id == tenant_a.id

        # Invalid key should fail
        tenant = await manager.validate_api_key("invalid_key_123")
        assert tenant is None

    @pytest.mark.asyncio
    async def test_tenant_suspension(
        self,
        tenant_a: TestTenant,
    ):
        """Test suspended tenant cannot access resources."""
        from aragora.tenancy import TenantContext
        from aragora.tenancy.tenant import TenantManager, TenantSuspendedError

        manager = TenantManager()

        # Suspend tenant
        await manager.suspend_tenant(tenant_a.id, reason="Payment overdue")

        # Access should fail
        with pytest.raises(TenantSuspendedError):
            with TenantContext(tenant_a.id):
                # Any operation should fail
                pass

        # Unsuspend
        await manager.unsuspend_tenant(tenant_a.id)

        # Access should work again
        with TenantContext(tenant_a.id):
            pass  # Should not raise

    @pytest.mark.asyncio
    async def test_audit_logging(
        self,
        tenant_a: TestTenant,
    ):
        """Test tenant actions are audit logged."""
        from aragora.tenancy import TenantContext
        from aragora.observability.log_types import AuditEntry
        from aragora.observability.log_backends import AuditLogBackend

        # Mock audit backend
        audit_entries: List[AuditEntry] = []

        with patch("aragora.tenancy.context.get_audit_backend") as mock:
            backend = MagicMock(spec=AuditLogBackend)

            async def capture_entry(entry: AuditEntry):
                audit_entries.append(entry)

            backend.append = capture_entry
            mock.return_value = backend

            with TenantContext(tenant_a.id, audit_enabled=True):
                # Perform some action
                from aragora.knowledge.mound.facade import KnowledgeMound
                mound = KnowledgeMound()
                await mound.add_node(
                    title="Test",
                    content="Test content",
                    node_type="document",
                )

            # Should have audit entry
            assert len(audit_entries) > 0
            assert any(e.tenant_id == tenant_a.id for e in audit_entries)
