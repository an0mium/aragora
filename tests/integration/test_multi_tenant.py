"""
Multi-Tenant Data Isolation Tests.

Tests data isolation between tenants to ensure:
- No cross-tenant data leakage
- Proper tenant context propagation
- Resource limits per tenant
- Audit logging for tenant operations
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class MockTenant:
    """Mock tenant for testing."""

    id: str
    name: str
    domain: str
    tier: str = "professional"
    status: str = "active"


@dataclass
class MockDebate:
    """Mock debate for testing."""

    id: str
    tenant_id: str
    topic: str
    status: str = "active"


class TestTenantIsolation:
    """Test tenant data isolation."""

    @pytest.fixture
    def tenant_a(self):
        return MockTenant(
            id="tenant-a",
            name="Company A",
            domain="company-a.com",
        )

    @pytest.fixture
    def tenant_b(self):
        return MockTenant(
            id="tenant-b",
            name="Company B",
            domain="company-b.com",
        )

    @pytest.mark.asyncio
    async def test_debate_isolation(self, tenant_a, tenant_b):
        """Test that debates are isolated between tenants."""
        from aragora.storage.debate_store import DebateStore

        store = DebateStore()

        # Create debates for both tenants
        debate_a = MockDebate(
            id="debate-a1",
            tenant_id=tenant_a.id,
            topic="Topic for A",
        )
        debate_b = MockDebate(
            id="debate-b1",
            tenant_id=tenant_b.id,
            topic="Topic for B",
        )

        with patch.object(store, "_db") as mock_db:
            mock_db.fetch_all = AsyncMock(return_value=[debate_a.__dict__])

            # Fetch debates for tenant A - should only see tenant A's debates
            debates = await store.list_debates(tenant_id=tenant_a.id)

            # Verify query included tenant filter
            mock_db.fetch_all.assert_called_once()
            call_args = str(mock_db.fetch_all.call_args)
            assert tenant_a.id in call_args

    @pytest.mark.asyncio
    async def test_user_isolation(self, tenant_a, tenant_b):
        """Test that users are isolated between tenants."""
        from aragora.storage.user_store import UserStore

        store = UserStore()

        with patch.object(store, "_db") as mock_db:
            mock_db.fetch_all = AsyncMock(return_value=[])

            # Fetch users for tenant A
            await store.list_users(tenant_id=tenant_a.id)

            # Verify tenant filter was applied
            call_args = str(mock_db.fetch_all.call_args)
            assert tenant_a.id in call_args

    @pytest.mark.asyncio
    async def test_api_key_isolation(self, tenant_a, tenant_b):
        """Test that API keys are isolated between tenants."""
        from aragora.storage.api_key_store import APIKeyStore

        store = APIKeyStore()

        with patch.object(store, "_db") as mock_db:
            mock_db.fetch_one = AsyncMock(return_value=None)

            # Validate API key - should check tenant ownership
            result = await store.validate_key(
                key="test-key",
                tenant_id=tenant_a.id,
            )

            # Verify tenant was checked
            mock_db.fetch_one.assert_called_once()
            call_args = str(mock_db.fetch_one.call_args)
            assert tenant_a.id in call_args or "tenant" in call_args.lower()


class TestTenantContextPropagation:
    """Test tenant context propagation through the system."""

    @pytest.mark.asyncio
    async def test_context_in_request_handler(self):
        """Test tenant context is available in request handlers."""
        from aragora.tenancy.context import TenantContext, set_tenant_context

        # Simulate request with tenant context
        async def mock_handler():
            ctx = TenantContext.current()
            assert ctx is not None
            assert ctx.tenant_id == "test-tenant"
            return {"tenant": ctx.tenant_id}

        # Set context and execute handler
        with TenantContext(tenant_id="test-tenant"):
            result = await mock_handler()

        assert result["tenant"] == "test-tenant"

    @pytest.mark.asyncio
    async def test_context_in_background_tasks(self):
        """Test tenant context propagates to background tasks."""
        from aragora.tenancy.context import TenantContext

        captured_tenant = None

        async def background_task():
            nonlocal captured_tenant
            ctx = TenantContext.current()
            captured_tenant = ctx.tenant_id if ctx else None

        # Start task with context
        with TenantContext(tenant_id="bg-tenant"):
            # Simulate background task execution
            await background_task()

        assert captured_tenant == "bg-tenant"

    @pytest.mark.asyncio
    async def test_context_in_nested_services(self):
        """Test tenant context propagates through nested service calls."""
        from aragora.tenancy.context import TenantContext

        async def outer_service():
            assert TenantContext.current().tenant_id == "nested-tenant"
            return await inner_service()

        async def inner_service():
            assert TenantContext.current().tenant_id == "nested-tenant"
            return await deepest_service()

        async def deepest_service():
            ctx = TenantContext.current()
            return ctx.tenant_id

        with TenantContext(tenant_id="nested-tenant"):
            result = await outer_service()

        assert result == "nested-tenant"


class TestTenantResourceLimits:
    """Test enforcement of tenant resource limits."""

    @pytest.mark.asyncio
    async def test_debate_creation_limit(self):
        """Test debate creation respects tenant limits."""
        from aragora.tenancy.limits import ResourceLimiter
        from aragora.tenancy.tenant import TenantConfig

        config = TenantConfig(max_debates_per_day=5)
        limiter = ResourceLimiter(config)

        # Allow up to limit
        for i in range(5):
            allowed = await limiter.can_create_debate(
                tenant_id="test-tenant",
                current_count=i,
            )
            assert allowed is True

        # Deny over limit
        allowed = await limiter.can_create_debate(
            tenant_id="test-tenant",
            current_count=5,
        )
        assert allowed is False

    @pytest.mark.asyncio
    async def test_concurrent_debate_limit(self):
        """Test concurrent debate limit enforcement."""
        from aragora.tenancy.limits import ResourceLimiter
        from aragora.tenancy.tenant import TenantConfig

        config = TenantConfig(max_concurrent_debates=2)
        limiter = ResourceLimiter(config)

        # Allow concurrent debates up to limit
        allowed = await limiter.can_start_debate(
            tenant_id="test-tenant",
            active_count=1,
        )
        assert allowed is True

        # Deny when at concurrent limit
        allowed = await limiter.can_start_debate(
            tenant_id="test-tenant",
            active_count=2,
        )
        assert allowed is False

    @pytest.mark.asyncio
    async def test_storage_quota_enforcement(self):
        """Test storage quota enforcement."""
        from aragora.tenancy.limits import ResourceLimiter
        from aragora.tenancy.tenant import TenantConfig

        config = TenantConfig(storage_quota=1024 * 1024)  # 1MB
        limiter = ResourceLimiter(config)

        # Allow storage under quota
        allowed = await limiter.can_store(
            tenant_id="test-tenant",
            current_usage=500 * 1024,  # 500KB
            requested_bytes=200 * 1024,  # 200KB
        )
        assert allowed is True

        # Deny storage that would exceed quota
        allowed = await limiter.can_store(
            tenant_id="test-tenant",
            current_usage=900 * 1024,
            requested_bytes=200 * 1024,
        )
        assert allowed is False


class TestTenantTierFeatures:
    """Test tier-based feature access."""

    @pytest.mark.asyncio
    async def test_free_tier_features(self):
        """Test free tier feature restrictions."""
        from aragora.tenancy.tenant import TenantConfig, TenantTier

        config = TenantConfig.for_tier(TenantTier.FREE)

        assert config.enable_sso is False
        assert config.enable_custom_agents is False
        assert config.max_debates_per_day == 10
        assert config.max_users == 3

    @pytest.mark.asyncio
    async def test_enterprise_tier_features(self):
        """Test enterprise tier features."""
        from aragora.tenancy.tenant import TenantConfig, TenantTier

        config = TenantConfig.for_tier(TenantTier.ENTERPRISE)

        assert config.enable_sso is True
        assert config.enable_custom_agents is True
        assert config.enable_audit_log is True
        assert config.max_users >= 100

    @pytest.mark.asyncio
    async def test_feature_gate_enforcement(self):
        """Test feature gate enforcement based on tier."""
        from aragora.tenancy.features import FeatureGate
        from aragora.tenancy.tenant import TenantTier

        # Free tier user
        free_gate = FeatureGate(tier=TenantTier.FREE)
        assert free_gate.is_enabled("basic_debates") is True
        assert free_gate.is_enabled("sso") is False
        assert free_gate.is_enabled("custom_agents") is False

        # Enterprise tier user
        enterprise_gate = FeatureGate(tier=TenantTier.ENTERPRISE)
        assert enterprise_gate.is_enabled("basic_debates") is True
        assert enterprise_gate.is_enabled("sso") is True
        assert enterprise_gate.is_enabled("custom_agents") is True


class TestCrossTenantAccess:
    """Test cross-tenant access prevention."""

    @pytest.mark.asyncio
    async def test_cross_tenant_debate_access_denied(self):
        """Test that accessing another tenant's debate is denied."""
        from aragora.tenancy.isolation import IsolationEnforcer

        enforcer = IsolationEnforcer()

        # Tenant A trying to access Tenant B's debate
        with pytest.raises(PermissionError) as exc_info:
            await enforcer.verify_debate_access(
                requesting_tenant="tenant-a",
                debate_tenant="tenant-b",
                debate_id="debate-123",
            )

        assert "tenant" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_cross_tenant_user_access_denied(self):
        """Test that accessing another tenant's users is denied."""
        from aragora.tenancy.isolation import IsolationEnforcer

        enforcer = IsolationEnforcer()

        with pytest.raises(PermissionError):
            await enforcer.verify_user_access(
                requesting_tenant="tenant-a",
                user_tenant="tenant-b",
                user_id="user-123",
            )

    @pytest.mark.asyncio
    async def test_same_tenant_access_allowed(self):
        """Test that same-tenant access is allowed."""
        from aragora.tenancy.isolation import IsolationEnforcer

        enforcer = IsolationEnforcer()

        # Should not raise
        result = await enforcer.verify_debate_access(
            requesting_tenant="tenant-a",
            debate_tenant="tenant-a",
            debate_id="debate-123",
        )

        assert result is True


class TestTenantAuditLogging:
    """Test tenant audit logging."""

    @pytest.mark.asyncio
    async def test_tenant_creation_logged(self):
        """Test that tenant creation is audit logged."""
        from aragora.audit.tenant_audit import TenantAuditLogger

        logger = TenantAuditLogger()

        await logger.log_tenant_created(
            tenant_id="new-tenant",
            name="New Corp",
            tier="professional",
            created_by="admin-user",
        )

        logs = await logger.get_logs(tenant_id="new-tenant", limit=10)
        assert len(logs) >= 1
        assert logs[0]["event_type"] == "tenant_created"

    @pytest.mark.asyncio
    async def test_tenant_tier_change_logged(self):
        """Test that tier changes are audit logged."""
        from aragora.audit.tenant_audit import TenantAuditLogger

        logger = TenantAuditLogger()

        await logger.log_tier_change(
            tenant_id="upgraded-tenant",
            old_tier="starter",
            new_tier="enterprise",
            changed_by="admin-user",
        )

        logs = await logger.get_logs(tenant_id="upgraded-tenant", limit=10)
        tier_logs = [log for log in logs if log["event_type"] == "tier_changed"]
        assert len(tier_logs) >= 1
        assert tier_logs[0]["old_tier"] == "starter"
        assert tier_logs[0]["new_tier"] == "enterprise"

    @pytest.mark.asyncio
    async def test_cross_tenant_attempt_logged(self):
        """Test that cross-tenant access attempts are logged."""
        from aragora.audit.tenant_audit import TenantAuditLogger

        logger = TenantAuditLogger()

        await logger.log_access_denied(
            requesting_tenant="tenant-a",
            target_tenant="tenant-b",
            resource_type="debate",
            resource_id="debate-123",
            user_id="user-456",
        )

        logs = await logger.get_security_logs(limit=10)
        denied_logs = [log for log in logs if log["event_type"] == "cross_tenant_access_denied"]
        assert len(denied_logs) >= 1


class TestTenantDataMigration:
    """Test tenant data migration scenarios."""

    @pytest.mark.asyncio
    async def test_tenant_data_export(self):
        """Test exporting all tenant data."""
        from aragora.tenancy.data import TenantDataExporter

        exporter = TenantDataExporter()

        with patch.object(exporter, "_fetch_all_data") as mock_fetch:
            mock_fetch.return_value = {
                "debates": [{"id": "1", "topic": "Test"}],
                "users": [{"id": "u1", "email": "test@test.com"}],
                "settings": {"theme": "dark"},
            }

            data = await exporter.export_tenant_data("tenant-to-export")

        assert "debates" in data
        assert "users" in data
        assert len(data["debates"]) == 1

    @pytest.mark.asyncio
    async def test_tenant_data_deletion(self):
        """Test GDPR-compliant tenant data deletion."""
        from aragora.tenancy.data import TenantDataManager

        manager = TenantDataManager()

        with patch.object(manager, "_delete_tenant_resources") as mock_delete:
            mock_delete.return_value = {
                "debates_deleted": 10,
                "users_deleted": 5,
                "storage_freed_bytes": 1024 * 1024,
            }

            result = await manager.delete_tenant_data(
                tenant_id="tenant-to-delete",
                confirmation="DELETE-tenant-to-delete",
            )

        assert result["debates_deleted"] == 10
        mock_delete.assert_called_once()


# Markers for running specific test groups
# NOTE: The skip marker is defined at the top of the file and takes precedence
