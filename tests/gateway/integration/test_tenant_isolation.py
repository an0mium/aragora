"""
Integration tests for multi-tenant isolation in the gateway.

Tests that tenant data, agents, and resources are properly isolated:
- Tenant A cannot see Tenant B's agents
- Tenant A cannot access Tenant B's credentials
- Debate results are only visible to owning tenant
- Quotas are enforced independently per tenant
- Metrics are properly tagged with tenant_id
- Cross-tenant access attempts are explicitly denied
"""

import pytest
import asyncio
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

from tests.gateway.integration.conftest import (
    register_external_agent,
    TenantContext,
    MockAgent,
    MockCredentialVault,
)


class TenantIsolationError(Exception):
    """Raised when tenant isolation is violated."""

    def __init__(self, message: str, requesting_tenant: str, target_tenant: str):
        self.requesting_tenant = requesting_tenant
        self.target_tenant = target_tenant
        super().__init__(message)


@dataclass
class TenantResourceRegistry:
    """Registry that enforces tenant isolation for resources."""

    agents: dict[str, dict] = field(default_factory=dict)
    credentials: dict[str, dict] = field(default_factory=dict)
    debate_results: dict[str, dict] = field(default_factory=dict)
    quotas: dict[str, dict] = field(default_factory=dict)
    metrics: list[dict] = field(default_factory=list)

    def register_agent(
        self,
        tenant_id: str,
        agent_name: str,
        agent_config: dict,
    ) -> None:
        """Register an agent for a tenant."""
        key = f"{tenant_id}:{agent_name}"
        self.agents[key] = {
            **agent_config,
            "tenant_id": tenant_id,
            "name": agent_name,
        }

    def get_agent(
        self,
        requesting_tenant: str,
        agent_name: str,
        target_tenant: str | None = None,
    ) -> dict | None:
        """Get an agent, enforcing tenant isolation."""
        target = target_tenant or requesting_tenant
        key = f"{target}:{agent_name}"

        if key not in self.agents:
            return None

        agent = self.agents[key]

        # Enforce isolation
        if agent["tenant_id"] != requesting_tenant:
            raise TenantIsolationError(
                f"Tenant {requesting_tenant} cannot access agent owned by {agent['tenant_id']}",
                requesting_tenant=requesting_tenant,
                target_tenant=agent["tenant_id"],
            )

        return agent

    def list_agents(self, tenant_id: str) -> list[dict]:
        """List agents visible to a tenant."""
        return [agent for agent in self.agents.values() if agent["tenant_id"] == tenant_id]

    def store_credential(
        self,
        tenant_id: str,
        credential_id: str,
        value: str,
        metadata: dict | None = None,
    ) -> None:
        """Store a credential for a tenant."""
        key = f"{tenant_id}:{credential_id}"
        self.credentials[key] = {
            "tenant_id": tenant_id,
            "credential_id": credential_id,
            "value": value,
            "metadata": metadata or {},
        }

    def get_credential(
        self,
        requesting_tenant: str,
        credential_id: str,
        target_tenant: str | None = None,
    ) -> str | None:
        """Get a credential, enforcing tenant isolation."""
        target = target_tenant or requesting_tenant
        key = f"{target}:{credential_id}"

        if key not in self.credentials:
            return None

        cred = self.credentials[key]

        # Enforce isolation
        if cred["tenant_id"] != requesting_tenant:
            raise TenantIsolationError(
                f"Tenant {requesting_tenant} cannot access credential owned by {cred['tenant_id']}",
                requesting_tenant=requesting_tenant,
                target_tenant=cred["tenant_id"],
            )

        return cred["value"]

    def store_debate_result(
        self,
        tenant_id: str,
        debate_id: str,
        result: dict,
    ) -> None:
        """Store a debate result for a tenant."""
        key = f"{tenant_id}:{debate_id}"
        self.debate_results[key] = {
            "tenant_id": tenant_id,
            "debate_id": debate_id,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_debate_result(
        self,
        requesting_tenant: str,
        debate_id: str,
        target_tenant: str | None = None,
    ) -> dict | None:
        """Get a debate result, enforcing tenant isolation."""
        target = target_tenant or requesting_tenant
        key = f"{target}:{debate_id}"

        if key not in self.debate_results:
            return None

        result = self.debate_results[key]

        # Enforce isolation
        if result["tenant_id"] != requesting_tenant:
            raise TenantIsolationError(
                f"Tenant {requesting_tenant} cannot access debate result owned by {result['tenant_id']}",
                requesting_tenant=requesting_tenant,
                target_tenant=result["tenant_id"],
            )

        return result["result"]

    def set_quota(
        self,
        tenant_id: str,
        quota_type: str,
        limit: int,
    ) -> None:
        """Set a quota for a tenant."""
        if tenant_id not in self.quotas:
            self.quotas[tenant_id] = {}
        self.quotas[tenant_id][quota_type] = {
            "limit": limit,
            "used": 0,
        }

    def consume_quota(
        self,
        tenant_id: str,
        quota_type: str,
        amount: int = 1,
    ) -> bool:
        """Consume quota for a tenant. Returns False if exceeded."""
        if tenant_id not in self.quotas:
            return True  # No quota set

        if quota_type not in self.quotas[tenant_id]:
            return True  # No quota for this type

        quota = self.quotas[tenant_id][quota_type]
        if quota["used"] + amount > quota["limit"]:
            return False

        quota["used"] += amount
        return True

    def get_quota_status(self, tenant_id: str) -> dict:
        """Get quota status for a tenant."""
        return self.quotas.get(tenant_id, {})

    def record_metric(
        self,
        tenant_id: str,
        metric_type: str,
        value: Any,
    ) -> None:
        """Record a metric tagged with tenant_id."""
        self.metrics.append(
            {
                "tenant_id": tenant_id,
                "type": metric_type,
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_metrics(self, tenant_id: str) -> list[dict]:
        """Get metrics for a specific tenant."""
        return [m for m in self.metrics if m["tenant_id"] == tenant_id]


class TestTenantIsolation:
    """Integration tests for multi-tenant isolation."""

    @pytest.fixture
    def registry(self) -> TenantResourceRegistry:
        """Create a fresh tenant resource registry."""
        return TenantResourceRegistry()

    @pytest.mark.asyncio
    async def test_tenant_agents_isolated(
        self,
        tenant_context: TenantContext,
        alt_tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that agents are isolated between tenants."""
        # Register agent for tenant A
        registry.register_agent(
            tenant_id=tenant_context.tenant_id,
            agent_name="tenant-a-agent",
            agent_config={"framework": "crewai", "url": "https://a.example.com"},
        )

        # Register agent for tenant B
        registry.register_agent(
            tenant_id=alt_tenant_context.tenant_id,
            agent_name="tenant-b-agent",
            agent_config={"framework": "autogen", "url": "https://b.example.com"},
        )

        # Tenant A should see only their agents
        a_agents = registry.list_agents(tenant_context.tenant_id)
        assert len(a_agents) == 1
        assert a_agents[0]["name"] == "tenant-a-agent"

        # Tenant B should see only their agents
        b_agents = registry.list_agents(alt_tenant_context.tenant_id)
        assert len(b_agents) == 1
        assert b_agents[0]["name"] == "tenant-b-agent"

        # Tenant A cannot see tenant B's agent
        assert "tenant-b-agent" not in [a["name"] for a in a_agents]

    @pytest.mark.asyncio
    async def test_tenant_credentials_isolated(
        self,
        tenant_context: TenantContext,
        alt_tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that credentials are isolated between tenants."""
        # Store credentials for each tenant
        registry.store_credential(
            tenant_id=tenant_context.tenant_id,
            credential_id="api-key",
            value="secret-a-123",
        )
        registry.store_credential(
            tenant_id=alt_tenant_context.tenant_id,
            credential_id="api-key",
            value="secret-b-456",
        )

        # Tenant A can access their own credential
        a_cred = registry.get_credential(
            requesting_tenant=tenant_context.tenant_id,
            credential_id="api-key",
        )
        assert a_cred == "secret-a-123"

        # Tenant A cannot access tenant B's credential
        with pytest.raises(TenantIsolationError) as exc_info:
            registry.get_credential(
                requesting_tenant=tenant_context.tenant_id,
                credential_id="api-key",
                target_tenant=alt_tenant_context.tenant_id,
            )
        assert exc_info.value.requesting_tenant == tenant_context.tenant_id
        assert exc_info.value.target_tenant == alt_tenant_context.tenant_id

    @pytest.mark.asyncio
    async def test_tenant_debate_results_isolated(
        self,
        tenant_context: TenantContext,
        alt_tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that debate results are only visible to owning tenant."""
        # Store debate results for each tenant
        registry.store_debate_result(
            tenant_id=tenant_context.tenant_id,
            debate_id="debate-1",
            result={"consensus": "Option A", "confidence": 0.9},
        )
        registry.store_debate_result(
            tenant_id=alt_tenant_context.tenant_id,
            debate_id="debate-2",
            result={"consensus": "Option B", "confidence": 0.85},
        )

        # Tenant A can access their own debate result
        a_result = registry.get_debate_result(
            requesting_tenant=tenant_context.tenant_id,
            debate_id="debate-1",
        )
        assert a_result["consensus"] == "Option A"

        # Tenant A cannot access tenant B's debate result
        with pytest.raises(TenantIsolationError):
            registry.get_debate_result(
                requesting_tenant=tenant_context.tenant_id,
                debate_id="debate-2",
                target_tenant=alt_tenant_context.tenant_id,
            )

        # Tenant A accessing their own result with explicit target works
        a_result_explicit = registry.get_debate_result(
            requesting_tenant=tenant_context.tenant_id,
            debate_id="debate-1",
            target_tenant=tenant_context.tenant_id,
        )
        assert a_result_explicit["consensus"] == "Option A"

    @pytest.mark.asyncio
    async def test_tenant_quota_enforcement(
        self,
        tenant_context: TenantContext,
        alt_tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that tenant quotas are enforced independently."""
        # Set different quotas for each tenant
        registry.set_quota(tenant_context.tenant_id, "api_calls", limit=5)
        registry.set_quota(alt_tenant_context.tenant_id, "api_calls", limit=10)

        # Consume quota for tenant A
        for _ in range(5):
            assert registry.consume_quota(tenant_context.tenant_id, "api_calls") is True

        # Tenant A's quota should be exhausted
        assert registry.consume_quota(tenant_context.tenant_id, "api_calls") is False

        # Tenant B's quota should still be available
        assert registry.consume_quota(alt_tenant_context.tenant_id, "api_calls") is True

        # Verify quota status
        a_status = registry.get_quota_status(tenant_context.tenant_id)
        b_status = registry.get_quota_status(alt_tenant_context.tenant_id)

        assert a_status["api_calls"]["used"] == 5
        assert a_status["api_calls"]["limit"] == 5
        assert b_status["api_calls"]["used"] == 1
        assert b_status["api_calls"]["limit"] == 10

    @pytest.mark.asyncio
    async def test_tenant_metrics_isolated(
        self,
        tenant_context: TenantContext,
        alt_tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that metrics are tagged with tenant_id and isolated."""
        # Record metrics for each tenant
        registry.record_metric(
            tenant_context.tenant_id,
            "request_count",
            10,
        )
        registry.record_metric(
            tenant_context.tenant_id,
            "latency_ms",
            150.5,
        )
        registry.record_metric(
            alt_tenant_context.tenant_id,
            "request_count",
            25,
        )

        # Get metrics for tenant A
        a_metrics = registry.get_metrics(tenant_context.tenant_id)
        assert len(a_metrics) == 2
        assert all(m["tenant_id"] == tenant_context.tenant_id for m in a_metrics)

        # Get metrics for tenant B
        b_metrics = registry.get_metrics(alt_tenant_context.tenant_id)
        assert len(b_metrics) == 1
        assert b_metrics[0]["tenant_id"] == alt_tenant_context.tenant_id
        assert b_metrics[0]["value"] == 25

    @pytest.mark.asyncio
    async def test_cross_tenant_access_denied(
        self,
        tenant_context: TenantContext,
        alt_tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that explicit cross-tenant access attempts fail."""
        # Register resources for tenant B
        registry.register_agent(
            tenant_id=alt_tenant_context.tenant_id,
            agent_name="protected-agent",
            agent_config={"framework": "crewai"},
        )
        registry.store_credential(
            tenant_id=alt_tenant_context.tenant_id,
            credential_id="protected-key",
            value="super-secret",
        )
        registry.store_debate_result(
            tenant_id=alt_tenant_context.tenant_id,
            debate_id="protected-debate",
            result={"data": "sensitive"},
        )

        # All cross-tenant access attempts should fail
        with pytest.raises(TenantIsolationError) as exc:
            registry.get_agent(
                requesting_tenant=tenant_context.tenant_id,
                agent_name="protected-agent",
                target_tenant=alt_tenant_context.tenant_id,
            )
        assert "cannot access" in str(exc.value)

        with pytest.raises(TenantIsolationError):
            registry.get_credential(
                requesting_tenant=tenant_context.tenant_id,
                credential_id="protected-key",
                target_tenant=alt_tenant_context.tenant_id,
            )

        with pytest.raises(TenantIsolationError):
            registry.get_debate_result(
                requesting_tenant=tenant_context.tenant_id,
                debate_id="protected-debate",
                target_tenant=alt_tenant_context.tenant_id,
            )


class TestTenantIsolationEdgeCases:
    """Edge case tests for tenant isolation."""

    @pytest.fixture
    def registry(self) -> TenantResourceRegistry:
        """Create a fresh tenant resource registry."""
        return TenantResourceRegistry()

    @pytest.mark.asyncio
    async def test_nonexistent_resource_returns_none(
        self,
        tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that accessing nonexistent resources returns None, not error."""
        # Accessing nonexistent agent
        result = registry.get_agent(
            requesting_tenant=tenant_context.tenant_id,
            agent_name="nonexistent-agent",
        )
        assert result is None

        # Accessing nonexistent credential
        cred = registry.get_credential(
            requesting_tenant=tenant_context.tenant_id,
            credential_id="nonexistent-key",
        )
        assert cred is None

        # Accessing nonexistent debate result
        debate = registry.get_debate_result(
            requesting_tenant=tenant_context.tenant_id,
            debate_id="nonexistent-debate",
        )
        assert debate is None

    @pytest.mark.asyncio
    async def test_same_resource_name_different_tenants(
        self,
        tenant_context: TenantContext,
        alt_tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that same resource names work independently per tenant."""
        # Both tenants create resources with the same name
        registry.register_agent(
            tenant_id=tenant_context.tenant_id,
            agent_name="my-agent",
            agent_config={"version": "1.0"},
        )
        registry.register_agent(
            tenant_id=alt_tenant_context.tenant_id,
            agent_name="my-agent",
            agent_config={"version": "2.0"},
        )

        # Each tenant gets their own version
        a_agent = registry.get_agent(
            requesting_tenant=tenant_context.tenant_id,
            agent_name="my-agent",
        )
        b_agent = registry.get_agent(
            requesting_tenant=alt_tenant_context.tenant_id,
            agent_name="my-agent",
        )

        assert a_agent["version"] == "1.0"
        assert b_agent["version"] == "2.0"

    @pytest.mark.asyncio
    async def test_quota_no_limit_allows_all(
        self,
        tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that tenants without quotas are not limited."""
        # No quota set for tenant
        for _ in range(100):
            assert registry.consume_quota(tenant_context.tenant_id, "api_calls") is True

    @pytest.mark.asyncio
    async def test_isolation_error_contains_details(
        self,
        tenant_context: TenantContext,
        alt_tenant_context: TenantContext,
        registry: TenantResourceRegistry,
    ):
        """Test that isolation errors contain useful debugging info."""
        registry.register_agent(
            tenant_id=alt_tenant_context.tenant_id,
            agent_name="target-agent",
            agent_config={},
        )

        with pytest.raises(TenantIsolationError) as exc_info:
            registry.get_agent(
                requesting_tenant=tenant_context.tenant_id,
                agent_name="target-agent",
                target_tenant=alt_tenant_context.tenant_id,
            )

        error = exc_info.value
        assert error.requesting_tenant == tenant_context.tenant_id
        assert error.target_tenant == alt_tenant_context.tenant_id
        assert tenant_context.tenant_id in str(error)
        assert alt_tenant_context.tenant_id in str(error)
