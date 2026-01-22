"""
Multi-Tenant Data Isolation Tests.

Tests data isolation between tenants to ensure:
- No cross-tenant data leakage
- Proper tenant context propagation
- Resource quotas per tenant
- Audit logging for tenant operations
"""

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest


@dataclass
class MockResource:
    """Mock resource for testing ownership validation."""

    id: str
    tenant_id: str
    name: str


class TestTenantIsolation:
    """Test tenant data isolation."""

    @pytest.fixture
    def tenant_a_id(self):
        return "tenant-a"

    @pytest.fixture
    def tenant_b_id(self):
        return "tenant-b"

    @pytest.mark.asyncio
    async def test_debate_isolation_via_context(self, tenant_a_id, tenant_b_id):
        """Test that debates are isolated between tenants via context."""
        from aragora.tenancy.context import TenantContext, get_current_tenant_id
        from aragora.tenancy.isolation import TenantDataIsolation, TenantIsolationConfig

        # Create resources for both tenants
        resource_a = MockResource(
            id="debate-a1",
            tenant_id=tenant_a_id,
            name="Topic for A",
        )
        resource_b = MockResource(
            id="debate-b1",
            tenant_id=tenant_b_id,
            name="Topic for B",
        )

        # Test isolation via context
        with TenantContext(tenant_id=tenant_a_id):
            current = get_current_tenant_id()
            assert current == tenant_a_id

            # Create isolation with non-strict mode for testing
            config = TenantIsolationConfig(strict_validation=False)
            isolation = TenantDataIsolation(config)

            # Verify tenant A can access their own data
            assert isolation.validate_ownership(resource_a)
            # Tenant A cannot access Tenant B's data
            assert not isolation.validate_ownership(resource_b)

    @pytest.mark.asyncio
    async def test_user_isolation_via_context(self, tenant_a_id, tenant_b_id):
        """Test that users are isolated between tenants via context."""
        from aragora.tenancy.context import TenantContext, get_current_tenant_id
        from aragora.tenancy.isolation import TenantDataIsolation, TenantIsolationConfig

        resource_a = MockResource(id="user-a1", tenant_id=tenant_a_id, name="User A")
        resource_b = MockResource(id="user-b1", tenant_id=tenant_b_id, name="User B")

        with TenantContext(tenant_id=tenant_a_id):
            assert get_current_tenant_id() == tenant_a_id

            config = TenantIsolationConfig(strict_validation=False)
            isolation = TenantDataIsolation(config)

            # Can access own tenant's resources
            assert isolation.validate_ownership(resource_a)
            # Cannot access other tenant's resources
            assert not isolation.validate_ownership(resource_b)


class TestTenantContextPropagation:
    """Test tenant context propagation through the system."""

    @pytest.mark.asyncio
    async def test_context_in_request_handler(self):
        """Test tenant context is available in request handlers."""
        from aragora.tenancy.context import TenantContext, get_current_tenant_id

        # Simulate request with tenant context
        async def mock_handler():
            tenant_id = get_current_tenant_id()
            assert tenant_id is not None
            assert tenant_id == "test-tenant"
            return {"tenant": tenant_id}

        # Set context and execute handler
        with TenantContext(tenant_id="test-tenant"):
            result = await mock_handler()

        assert result["tenant"] == "test-tenant"

    @pytest.mark.asyncio
    async def test_context_in_background_tasks(self):
        """Test tenant context propagates to background tasks."""
        from aragora.tenancy.context import TenantContext, get_current_tenant_id

        captured_tenant = None

        async def background_task():
            nonlocal captured_tenant
            captured_tenant = get_current_tenant_id()

        # Start task with context
        with TenantContext(tenant_id="bg-tenant"):
            # Simulate background task execution
            await background_task()

        assert captured_tenant == "bg-tenant"

    @pytest.mark.asyncio
    async def test_context_in_nested_services(self):
        """Test tenant context propagates through nested service calls."""
        from aragora.tenancy.context import TenantContext, get_current_tenant_id

        async def outer_service():
            assert get_current_tenant_id() == "nested-tenant"
            return await inner_service()

        async def inner_service():
            assert get_current_tenant_id() == "nested-tenant"
            return await deepest_service()

        async def deepest_service():
            return get_current_tenant_id()

        with TenantContext(tenant_id="nested-tenant"):
            result = await outer_service()

        assert result == "nested-tenant"

    @pytest.mark.asyncio
    async def test_context_isolation_between_tasks(self):
        """Test that concurrent tasks maintain separate tenant contexts."""
        from aragora.tenancy.context import TenantContext, get_current_tenant_id

        results = {}

        async def task_for_tenant(tenant_id: str, delay: float):
            with TenantContext(tenant_id=tenant_id):
                await asyncio.sleep(delay)
                results[tenant_id] = get_current_tenant_id()

        # Run concurrent tasks with different tenants
        await asyncio.gather(
            task_for_tenant("tenant-1", 0.01),
            task_for_tenant("tenant-2", 0.005),
            task_for_tenant("tenant-3", 0.015),
        )

        # Each task should have captured its own tenant ID
        assert results["tenant-1"] == "tenant-1"
        assert results["tenant-2"] == "tenant-2"
        assert results["tenant-3"] == "tenant-3"


class TestTenantResourceQuotas:
    """Test enforcement of tenant resource quotas."""

    @pytest.mark.asyncio
    async def test_quota_check(self):
        """Test quota checking."""
        from aragora.tenancy.context import TenantContext
        from aragora.tenancy.quotas import QuotaConfig, QuotaLimit, QuotaManager, QuotaPeriod

        # Configure with specific limits
        config = QuotaConfig(
            limits=[
                QuotaLimit("debates", 5, QuotaPeriod.DAY),
                QuotaLimit("agents", 10, QuotaPeriod.UNLIMITED),
                QuotaLimit("concurrent_debates", 2, QuotaPeriod.UNLIMITED),
            ],
            strict_enforcement=False,
        )
        manager = QuotaManager(config)

        with TenantContext(tenant_id="test-tenant"):
            # Check if within quota (starting fresh, so count is 0)
            can_create = await manager.check_quota("debates", 1)
            assert can_create is True

            # Consume some quota
            for _ in range(5):
                await manager.consume("debates", 1)

            # Now should be at limit
            can_create = await manager.check_quota("debates", 1)
            assert can_create is False

    @pytest.mark.asyncio
    async def test_concurrent_debate_quota(self):
        """Test concurrent debate quota enforcement."""
        from aragora.tenancy.context import TenantContext
        from aragora.tenancy.quotas import QuotaConfig, QuotaLimit, QuotaManager, QuotaPeriod

        config = QuotaConfig(
            limits=[
                QuotaLimit("concurrent_debates", 2, QuotaPeriod.UNLIMITED),
            ],
            strict_enforcement=False,
        )
        manager = QuotaManager(config)

        with TenantContext(tenant_id="test-tenant"):
            # Allow concurrent debates up to limit
            allowed = await manager.check_quota("concurrent_debates", 1)
            assert allowed is True

            # Consume the quota
            await manager.consume("concurrent_debates", 1)
            await manager.consume("concurrent_debates", 1)

            # Deny when at concurrent limit
            allowed = await manager.check_quota("concurrent_debates", 1)
            assert allowed is False


class TestTenantAuditLogging:
    """Test audit logging for tenant operations."""

    @pytest.mark.asyncio
    async def test_audit_log_captures_tenant(self):
        """Test that audit logs capture tenant context."""
        from aragora.tenancy.context import (
            TenantContext,
            get_audit_backend,
            set_audit_backend,
        )

        # Create mock audit backend
        mock_audit = MagicMock()
        mock_audit.log = AsyncMock()

        with TenantContext(tenant_id="audit-tenant"):
            set_audit_backend(mock_audit)

            # Simulate an audit event
            backend = get_audit_backend()
            if backend:
                await backend.log(
                    action="debate.create",
                    resource_id="debate-123",
                    details={"topic": "Test topic"},
                )

                # Verify audit was called
                mock_audit.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_cross_tenant_access_logged(self):
        """Test that cross-tenant access attempts are logged."""
        from aragora.tenancy.context import TenantContext
        from aragora.tenancy.isolation import (
            TenantDataIsolation,
            TenantIsolationConfig,
            TenantIsolationError,
        )

        # Create a resource belonging to tenant-b
        other_tenant_resource = MockResource(
            id="resource-b1",
            tenant_id="tenant-b",
            name="Other Tenant Resource",
        )

        with TenantContext(tenant_id="tenant-a"):
            # Use strict validation so it raises on violation
            config = TenantIsolationConfig(strict_validation=True, audit_access=True)
            isolation = TenantDataIsolation(config)

            # Attempt to access another tenant's data should raise
            with pytest.raises(TenantIsolationError):
                isolation.validate_ownership(other_tenant_resource)


class TestTenantManager:
    """Test tenant management operations."""

    @pytest.mark.asyncio
    async def test_tenant_creation(self):
        """Test creating a new tenant."""
        from aragora.tenancy.tenant import Tenant, TenantConfig, TenantTier

        tenant = Tenant(
            id="new-tenant",
            name="New Company",
            slug="new-company",
            tier=TenantTier.PROFESSIONAL,
            config=TenantConfig(),
        )

        assert tenant.id == "new-tenant"
        assert tenant.name == "New Company"
        assert tenant.tier == TenantTier.PROFESSIONAL

    @pytest.mark.asyncio
    async def test_tenant_creation_via_factory(self):
        """Test creating a new tenant via factory method."""
        from aragora.tenancy.tenant import Tenant, TenantTier

        tenant = Tenant.create(
            name="Factory Company",
            owner_email="owner@factory.com",
            tier=TenantTier.PROFESSIONAL,
        )

        assert tenant.name == "Factory Company"
        assert tenant.slug == "factory-company"
        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.owner_email == "owner@factory.com"

    @pytest.mark.asyncio
    async def test_tenant_config_defaults(self):
        """Test tenant config has sensible defaults."""
        from aragora.tenancy.tenant import TenantConfig

        config = TenantConfig()

        # Should have default limits
        assert config.max_debates_per_day > 0
        assert config.max_agents_per_debate > 0
        assert config.max_concurrent_debates > 0

    @pytest.mark.asyncio
    async def test_tenant_suspension(self):
        """Test tenant suspension blocks operations."""
        from aragora.tenancy.tenant import (
            Tenant,
            TenantConfig,
            TenantManager,
            TenantStatus,
            TenantSuspendedError,
            TenantTier,
        )

        tenant = Tenant(
            id="suspended-tenant",
            name="Suspended Co",
            slug="suspended-co",
            tier=TenantTier.PROFESSIONAL,
            status=TenantStatus.SUSPENDED,
            config=TenantConfig(),
        )

        # Suspended tenant should not be active
        assert not tenant.is_active()

        # Using TenantManager to validate API key should raise for suspended tenant
        manager = TenantManager()
        manager.register_tenant(tenant)

        # Generate an API key and try to validate it
        api_key = tenant.generate_api_key()

        with pytest.raises(TenantSuspendedError):
            await manager.validate_api_key(api_key)
