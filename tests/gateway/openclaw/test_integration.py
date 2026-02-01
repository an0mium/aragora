"""
End-to-end integration tests for the OpenClaw gateway.

Tests component interaction across the full gateway stack:
- OpenClawGatewayAdapter (session management, task execution)
- AuthBridge (authentication, permission mapping, session lifecycle)
- ActionFilter (allowlist/denylist, risk scoring)
- CredentialVault (store, retrieve, rotate, tenant isolation)
- DecisionRouter (risk-based routing to debate vs. execute)
- GatewayHealthChecker (readiness probes)
- EnterpriseProxy (circuit breaker, recovery)

Only external HTTP calls to OpenClaw are mocked; all internal component
interactions are exercised for real.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# -- Gateway adapter and protocol types --
from aragora.gateway.openclaw.adapter import (
    GatewayResult,
    OpenClawGatewayAdapter,
)
from aragora.gateway.openclaw.capabilities import CapabilityFilter
from aragora.gateway.openclaw.protocol import (
    AragoraRequest,
    AuthorizationContext,
    TenantContext,
)
from aragora.gateway.openclaw.sandbox import SandboxConfig, SandboxResult, SandboxStatus

# -- Auth bridge --
from aragora.gateway.enterprise.auth_bridge import (
    AuthBridge,
    AuthContext,
    AuditEntry,
    BridgedSession,
    PermissionMapping,
    PermissionDeniedError,
    SessionExpiredError,
    SessionLifecycleHook,
    TokenType,
)

# -- Action filter --
from aragora.gateway.openclaw.action_filter import (
    ActionFilter,
    ActionRule,
    FilterDecision,
    RiskLevel as ActionRiskLevel,
)

# -- Credential vault --
from aragora.gateway.openclaw.credential_vault import (
    CredentialAccessDeniedError,
    CredentialNotFoundError,
    CredentialRateLimiter,
    CredentialType,
    CredentialVault,
    RotationPolicy,
    TenantIsolationError,
    reset_credential_vault,
)

# -- Decision router --
from aragora.gateway.decision_router import (
    DecisionRouter,
    RouteDecision,
    RouteDestination,
    RoutingCriteria,
    RoutingRule,
    RiskLevel as RouterRiskLevel,
    TenantRoutingConfig,
)

# -- Health checker --
from aragora.gateway.health import (
    ComponentHealth,
    GatewayHealthChecker,
    GatewayHealthStatus,
    HealthStatus,
)

# -- Enterprise proxy types --
from aragora.gateway.enterprise.proxy import (
    CircuitBreakerSettings,
    CircuitOpenError,
    ExternalFrameworkConfig,
    ProxyConfig,
    ProxyResponse,
)


# =============================================================================
# Shared Fixtures / Helpers
# =============================================================================


class MockRBACChecker:
    """Reusable mock RBAC checker for integration tests."""

    def __init__(self, allow_all: bool = True) -> None:
        self.allow_all = allow_all
        self.checked_permissions: list[tuple[str, str]] = []

    def check_permission(
        self, actor_id: str, permission: str, resource_id: str | None = None
    ) -> bool:
        self.checked_permissions.append((actor_id, permission))
        return self.allow_all

    async def check_permission_async(
        self, actor_id: str, permission: str, resource_id: str | None = None
    ) -> bool:
        return self.check_permission(actor_id, permission, resource_id)


class MockAuditLogger:
    """Captures audit events for assertion."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def log(
        self,
        event_type: str,
        actor_id: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "actor_id": actor_id,
                "resource_id": resource_id,
                "details": details,
                "severity": severity,
            }
        )

    async def log_async(
        self,
        event_type: str,
        actor_id: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        self.log(event_type, actor_id, resource_id, details, severity)


class MockVaultAuditLogger:
    """Audit logger that satisfies CredentialVault's AuditLoggerProtocol."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def log_event(
        self,
        event_type: str,
        actor_id: str,
        tenant_id: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        severity: str = "info",
    ) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "actor_id": actor_id,
                "tenant_id": tenant_id,
                "resource_id": resource_id,
                "details": details,
                "severity": severity,
            }
        )


@dataclass
class FakeAuthContext:
    """Lightweight auth context satisfying CredentialVault's AuthorizationContextProtocol."""

    user_id: str
    org_id: str | None = None
    roles: set[str] = field(default_factory=set)
    permissions: set[str] = field(default_factory=set)

    def has_permission(self, permission_key: str) -> bool:
        if "credentials:admin" in self.permissions:
            return True
        return permission_key in self.permissions

    def has_role(self, role_name: str) -> bool:
        return role_name in self.roles


class TrackingLifecycleHook(SessionLifecycleHook):
    """Track session lifecycle events during integration tests."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.accessed: list[str] = []
        self.expired: list[str] = []
        self.destroyed: list[tuple[str, str]] = []

    async def on_session_created(self, session: BridgedSession, context: AuthContext) -> None:
        self.created.append(session.session_id)

    async def on_session_accessed(self, session: BridgedSession) -> None:
        self.accessed.append(session.session_id)

    async def on_session_expired(self, session: BridgedSession) -> None:
        self.expired.append(session.session_id)

    async def on_session_destroyed(self, session: BridgedSession, reason: str = "logout") -> None:
        self.destroyed.append((session.session_id, reason))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def auth_context() -> AuthorizationContext:
    """Standard authorization context for adapter tests."""
    return AuthorizationContext(
        actor_id="user-integ-001",
        actor_type="user",
        permissions={"gateway.execute"},
    )


@pytest.fixture
def tenant_context() -> TenantContext:
    return TenantContext(
        tenant_id="tenant-integ",
        organization_id="org-integ",
        enabled_capabilities={"text_generation", "search_internal"},
    )


@pytest.fixture
def adapter() -> OpenClawGatewayAdapter:
    return OpenClawGatewayAdapter(
        openclaw_endpoint="http://localhost:8081",
        rbac_checker=MockRBACChecker(),
        audit_logger=MockAuditLogger(),
        sandbox_config=SandboxConfig(),
        capability_filter=CapabilityFilter(),
    )


@pytest.fixture
def auth_bridge() -> AuthBridge:
    return AuthBridge(
        permission_mappings=[
            PermissionMapping(
                aragora_permission="debates.create",
                external_action="create_conversation",
            ),
            PermissionMapping(
                aragora_permission="debates.read",
                external_action="view_conversation",
            ),
            PermissionMapping(
                aragora_permission="gateway.execute",
                external_action="execute_task",
            ),
        ],
        action_allowlist={
            "create_conversation",
            "view_conversation",
            "execute_task",
        },
        enable_audit=True,
    )


@pytest.fixture
def action_filter() -> ActionFilter:
    return ActionFilter(
        tenant_id="tenant-integ",
        allowed_actions={
            "browser.navigate",
            "browser.click",
            "filesystem.read",
            "text_generation",
        },
        enable_audit=True,
    )


@pytest.fixture
def vault_audit() -> MockVaultAuditLogger:
    return MockVaultAuditLogger()


@pytest.fixture
def credential_vault(vault_audit: MockVaultAuditLogger) -> CredentialVault:
    reset_credential_vault()
    return CredentialVault(
        audit_logger=vault_audit,
        rate_limiter=CredentialRateLimiter(
            max_per_minute=100,
            max_per_hour=500,
        ),
    )


@pytest.fixture
def decision_router() -> DecisionRouter:
    return DecisionRouter(
        criteria=RoutingCriteria(
            financial_threshold=10000.0,
            risk_levels={RouterRiskLevel.HIGH, RouterRiskLevel.CRITICAL},
        ),
        enable_audit=True,
    )


@pytest.fixture
def health_checker() -> GatewayHealthChecker:
    return GatewayHealthChecker(
        openclaw_endpoint="http://localhost:8081",
        check_timeout=2.0,
    )


# =============================================================================
# 1. Full Session Lifecycle
# =============================================================================


class TestFullSessionLifecycle:
    """Create session -> execute action -> check result -> close session."""

    @pytest.mark.asyncio
    async def test_session_create_execute_close(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """End-to-end: create session, run task, get result, close."""
        request = AragoraRequest(
            content="Summarise quarterly report",
            capabilities=["text_generation"],
        )

        with patch.object(adapter.sandbox, "execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = SandboxResult(
                status=SandboxStatus.COMPLETED,
                output={"text": "Q4 revenue up 12%."},
                execution_time_ms=200,
            )
            result = await adapter.execute_task(request, auth_context)

        assert result.success is True
        assert result.response is not None
        assert result.response.status == "completed"

    @pytest.mark.asyncio
    async def test_auth_bridge_session_lifecycle(self, auth_bridge: AuthBridge) -> None:
        """AuthBridge: create context -> create session -> verify session -> destroy."""
        context = AuthContext(
            user_id="user-lifecycle",
            email="lifecycle@test.com",
            permissions={"debates.create", "gateway.execute"},
            roles={"admin"},
        )

        session = await auth_bridge.create_session(context, external_session_id="ext-01")
        assert session.session_id
        assert session.auth_context is not None
        assert session.auth_context.user_id == "user-lifecycle"

        # Verify session exists and can be retrieved
        fetched = await auth_bridge.get_session(session.session_id)
        assert fetched is not None
        assert fetched.session_id == session.session_id

        # Destroy session
        destroyed = await auth_bridge.destroy_session(session.session_id)
        assert destroyed is True

        # Should be gone after destruction
        gone = await auth_bridge.get_session(session.session_id)
        assert gone is None

    @pytest.mark.asyncio
    async def test_lifecycle_hooks_invoked(self) -> None:
        """Session lifecycle hooks are called during create/access/destroy."""
        hook = TrackingLifecycleHook()
        bridge = AuthBridge(enable_audit=True, lifecycle_hooks=[hook])

        context = AuthContext(
            user_id="hook-user",
            email="hook@test.com",
            permissions=set(),
        )

        session = await bridge.create_session(context)
        assert session.session_id in hook.created

        # Access the session via verify_request
        ctx = await bridge.verify_request(session_id=session.session_id)
        assert ctx.user_id == "hook-user"
        assert session.session_id in hook.accessed

        # Destroy
        await bridge.destroy_session(session.session_id, reason="test-cleanup")
        assert (session.session_id, "test-cleanup") in hook.destroyed

    @pytest.mark.asyncio
    async def test_expired_session_cleaned_on_access(self) -> None:
        """Expired sessions are cleaned up and raise on next access."""
        bridge = AuthBridge(session_duration=1, enable_audit=True)
        context = AuthContext(
            user_id="expiry-user",
            email="expiry@test.com",
            permissions=set(),
        )
        session = await bridge.create_session(context, duration=1)

        # Force expiry
        session.expires_at = time.time() - 10

        with pytest.raises(SessionExpiredError):
            await bridge.verify_request(session_id=session.session_id)

    @pytest.mark.asyncio
    async def test_session_metadata_preserved(self, auth_bridge: AuthBridge) -> None:
        """Metadata attached to session is preserved across retrieval."""
        context = AuthContext(
            user_id="meta-user",
            email="meta@test.com",
            permissions=set(),
        )
        session = await auth_bridge.create_session(
            context,
            metadata={"origin": "slack", "channel": "#general"},
        )
        fetched = await auth_bridge.get_session(session.session_id)
        assert fetched is not None
        assert fetched.metadata["origin"] == "slack"
        assert fetched.metadata["channel"] == "#general"

    @pytest.mark.asyncio
    async def test_adapter_task_with_tenant_context(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Task execution propagates tenant context correctly."""
        request = AragoraRequest(
            content="Tenant-scoped query",
            capabilities=["text_generation"],
            context={"tenant_id": "tenant-integ"},
        )

        with patch.object(adapter.sandbox, "execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = SandboxResult(
                status=SandboxStatus.COMPLETED,
                output={"text": "Response"},
                execution_time_ms=100,
            )
            result = await adapter.execute_task(request, auth_context)

        assert result.success is True


# =============================================================================
# 2. Security Flow
# =============================================================================


class TestSecurityFlow:
    """Auth verification -> RBAC check -> action filter -> audit log."""

    @pytest.mark.asyncio
    async def test_auth_then_rbac_then_action_filter(
        self, auth_bridge: AuthBridge, action_filter: ActionFilter
    ) -> None:
        """Full chain: authenticate, check permission mapping, filter action."""
        context = AuthContext(
            user_id="sec-user",
            email="sec@test.com",
            permissions={"gateway.execute"},
            roles=set(),
        )
        session = await auth_bridge.create_session(context)
        retrieved_ctx = await auth_bridge.verify_request(session_id=session.session_id)

        # Permission mapping: gateway.execute -> execute_task
        allowed = auth_bridge.is_action_allowed(retrieved_ctx, "execute_task")
        assert allowed is True

        # Action filter: browser.navigate is in tenant allowlist
        decision = action_filter.check_action("browser.navigate")
        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_denied_action_produces_audit(
        self, auth_bridge: AuthBridge, action_filter: ActionFilter
    ) -> None:
        """Blocked action generates audit entries in both bridge and filter."""
        context = AuthContext(
            user_id="denied-user",
            email="denied@test.com",
            permissions=set(),  # No permissions
        )
        session = await auth_bridge.create_session(context)
        retrieved_ctx = await auth_bridge.verify_request(session_id=session.session_id)

        # Bridge should deny (no mapping for 'execute_task')
        allowed = auth_bridge.is_action_allowed(retrieved_ctx, "execute_task")
        assert allowed is False

        # Filter blocks action not in allowlist
        decision = action_filter.check_action("system.rm_rf")
        assert decision.allowed is False
        assert decision.risk_level == ActionRiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_critical_action_always_blocked(self, action_filter: ActionFilter) -> None:
        """Critical denylist actions are blocked even if in allowlist."""
        # Add system.rm_rf to allowlist explicitly -- should still be blocked
        action_filter._allowed_actions.add("system.rm_rf")
        decision = action_filter.check_action("system.rm_rf")
        assert decision.allowed is False
        assert decision.risk_level == ActionRiskLevel.CRITICAL

    @pytest.mark.asyncio
    async def test_permission_denied_raises(self, auth_bridge: AuthBridge) -> None:
        """check_action raises PermissionDeniedError for unauthorized actions."""
        context = AuthContext(
            user_id="no-perm-user",
            email="noperm@test.com",
            permissions=set(),
        )
        with pytest.raises(PermissionDeniedError):
            auth_bridge.check_action(context, "create_conversation")

    @pytest.mark.asyncio
    async def test_denylist_overrides_allowlist_in_bridge(self, auth_bridge: AuthBridge) -> None:
        """Bridge action_denylist blocks even if action is in allowlist."""
        bridge = AuthBridge(
            permission_mappings=[
                PermissionMapping("gateway.execute", "execute_task"),
            ],
            action_allowlist={"execute_task"},
            action_denylist={"execute_task"},
            enable_audit=True,
        )
        context = AuthContext(
            user_id="deny-test",
            email="deny@test.com",
            permissions={"gateway.execute"},
        )
        assert bridge.is_action_allowed(context, "execute_task") is False

    @pytest.mark.asyncio
    async def test_audit_log_records_auth_events(self, auth_bridge: AuthBridge) -> None:
        """Auth bridge audit log captures authentication and session events."""
        context = AuthContext(
            user_id="audit-user",
            email="audit@test.com",
            permissions=set(),
        )
        session = await auth_bridge.create_session(context)
        await auth_bridge.verify_request(session_id=session.session_id)
        await auth_bridge.destroy_session(session.session_id)

        # At least 3 events: create, verify (authentication), destroy
        log = await auth_bridge.get_audit_log(user_id="audit-user")
        event_types = [e["event_type"] for e in log]
        assert "session" in event_types
        assert "authentication" in event_types

    @pytest.mark.asyncio
    async def test_adapter_permission_denied_flow(self, auth_context: AuthorizationContext) -> None:
        """Adapter rejects task when RBAC checker denies permission."""
        adapter = OpenClawGatewayAdapter(
            openclaw_endpoint="http://localhost:8081",
            rbac_checker=MockRBACChecker(allow_all=False),
            audit_logger=MockAuditLogger(),
        )
        request = AragoraRequest(content="should be denied")
        result = await adapter.execute_task(request, auth_context)
        assert result.success is False
        assert result.blocked_reason == "permission_denied"

    @pytest.mark.asyncio
    async def test_action_filter_stats_updated(self, action_filter: ActionFilter) -> None:
        """Filter statistics are updated after check_action calls."""
        action_filter.check_action("browser.navigate")
        action_filter.check_action("system.rm_rf")

        stats = action_filter._stats
        assert stats["total_checks"] >= 2
        assert stats["allowed"] >= 1
        assert stats["blocked"] >= 1


# =============================================================================
# 3. Decision Routing Flow
# =============================================================================


class TestDecisionRoutingFlow:
    """Request -> risk assessment -> route to debate or execute."""

    @pytest.mark.asyncio
    async def test_low_value_routes_to_execute(self, decision_router: DecisionRouter) -> None:
        """Low-value request is routed to direct execution."""
        decision = await decision_router.route(
            {"action": "query", "amount": 50},
            context={"tenant_id": "t1"},
        )
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_high_value_routes_to_debate(self, decision_router: DecisionRouter) -> None:
        """Large financial transaction is routed to debate."""
        decision = await decision_router.route(
            {"action": "transfer", "amount": 50000},
            context={"tenant_id": "t1"},
        )
        assert decision.destination == RouteDestination.DEBATE
        assert "financial" in decision.reason.lower() or len(decision.criteria_matched) > 0

    @pytest.mark.asyncio
    async def test_risk_level_triggers_debate(self, decision_router: DecisionRouter) -> None:
        """High-risk action triggers debate routing."""
        decision = await decision_router.route(
            {"action": "deploy", "risk_level": "critical"},
            context={"tenant_id": "t1"},
        )
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_explicit_debate_keyword(self, decision_router: DecisionRouter) -> None:
        """Request containing debate keyword is routed to debate."""
        decision = await decision_router.route(
            {"content": "We need to debate this proposal"},
            context={"tenant_id": "t1"},
        )
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_explicit_execute_keyword(self, decision_router: DecisionRouter) -> None:
        """Request containing execute keyword is routed to execution."""
        decision = await decision_router.route(
            {"content": "Just execute the backup job"},
            context={"tenant_id": "t1"},
        )
        assert decision.destination == RouteDestination.EXECUTE

    @pytest.mark.asyncio
    async def test_custom_rule_overrides_default(self, decision_router: DecisionRouter) -> None:
        """Custom routing rule takes precedence over defaults."""
        decision_router.add_rule(
            RoutingRule(
                rule_id="always-debate-deploy",
                condition=lambda req: req.get("action") == "deploy",
                destination=RouteDestination.DEBATE,
                priority=200,
                reason="Deployments always require consensus",
            )
        )
        decision = await decision_router.route(
            {"action": "deploy", "amount": 5},
            context={"tenant_id": "t1"},
        )
        assert decision.destination == RouteDestination.DEBATE
        assert decision.rule_id == "always-debate-deploy"

    @pytest.mark.asyncio
    async def test_tenant_specific_routing_config(self, decision_router: DecisionRouter) -> None:
        """Tenant-specific criteria override global criteria."""
        tenant_config = TenantRoutingConfig(
            tenant_id="strict-tenant",
            criteria=RoutingCriteria(financial_threshold=100.0),
        )
        await decision_router.add_tenant_config(tenant_config)

        # Amount=500 exceeds tenant threshold (100) but not global (10000)
        decision = await decision_router.route(
            {"action": "purchase", "amount": 500},
            context={"tenant_id": "strict-tenant"},
        )
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_compliance_flag_triggers_debate(self, decision_router: DecisionRouter) -> None:
        """Compliance flags (e.g., PII) route to debate."""
        decision = await decision_router.route(
            {"action": "export_data", "compliance_flags": ["pii"]},
            context={"tenant_id": "t1"},
        )
        assert decision.destination == RouteDestination.DEBATE

    @pytest.mark.asyncio
    async def test_routing_metrics_increment(self, decision_router: DecisionRouter) -> None:
        """Routing metrics are updated after routing decisions."""
        await decision_router.route({"action": "query"})
        await decision_router.route({"action": "transfer", "amount": 50000})
        metrics = decision_router._metrics
        assert metrics.total_requests >= 2

    @pytest.mark.asyncio
    async def test_decision_serialization(self, decision_router: DecisionRouter) -> None:
        """RouteDecision.to_dict() produces a valid dictionary."""
        decision = await decision_router.route({"action": "query"})
        d = decision.to_dict()
        assert "destination" in d
        assert "reason" in d
        assert "request_id" in d


# =============================================================================
# 4. Credential Flow
# =============================================================================


class TestCredentialFlow:
    """Store credential -> use in session -> rotate -> verify new value."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(
        self, credential_vault: CredentialVault, vault_audit: MockVaultAuditLogger
    ) -> None:
        """Store a credential and retrieve its decrypted value."""
        ctx = FakeAuthContext(
            user_id="cred-user",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:read"},
        )

        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-test-key-12345",
            auth_context=ctx,
            description="OpenAI API key for integration tests",
        )

        value = await credential_vault.get_credential_value(cred_id, auth_context=ctx)
        assert value == "sk-test-key-12345"

        # Verify audit events recorded
        event_types = [e["event_type"] for e in vault_audit.events]
        assert "credential_created" in event_types
        assert "credential_accessed" in event_types

    @pytest.mark.asyncio
    async def test_rotate_credential_changes_value(self, credential_vault: CredentialVault) -> None:
        """After rotation, retrieving the credential returns the new value."""
        ctx = FakeAuthContext(
            user_id="rotate-user",
            org_id="tenant-a",
            permissions={
                "credentials:create",
                "credentials:read",
                "credentials:rotate",
            },
        )

        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="anthropic",
            credential_type=CredentialType.API_KEY,
            value="old-key",
            auth_context=ctx,
        )

        # Rotate
        rotated = await credential_vault.rotate_credential(
            cred_id, new_value="new-key", auth_context=ctx
        )
        assert rotated.metadata.version == 2
        assert rotated.metadata.access_count == 0  # Reset after rotation

        # Retrieve should return new value
        value = await credential_vault.get_credential_value(cred_id, auth_context=ctx)
        assert value == "new-key"

    @pytest.mark.asyncio
    async def test_credential_metadata_tracking(self, credential_vault: CredentialVault) -> None:
        """Access count and last_accessed_by are updated on retrieval."""
        ctx = FakeAuthContext(
            user_id="meta-user",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:read"},
        )

        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-track",
            auth_context=ctx,
        )

        await credential_vault.get_credential_value(cred_id, auth_context=ctx)
        await credential_vault.get_credential_value(cred_id, auth_context=ctx)

        cred = await credential_vault.get_credential(cred_id, auth_context=ctx)
        assert cred.metadata.access_count == 2
        assert cred.metadata.last_accessed_by == "meta-user"

    @pytest.mark.asyncio
    async def test_credential_listing_by_framework(self, credential_vault: CredentialVault) -> None:
        """List credentials filtered by framework."""
        ctx = FakeAuthContext(
            user_id="list-user",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:list"},
        )

        await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-1",
            auth_context=ctx,
        )
        await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="anthropic",
            credential_type=CredentialType.API_KEY,
            value="sk-2",
            auth_context=ctx,
        )

        openai_creds = await credential_vault.list_credentials(
            tenant_id="tenant-a",
            framework="openai",
            auth_context=ctx,
        )
        assert len(openai_creds) == 1
        assert openai_creds[0].framework == "openai"

    @pytest.mark.asyncio
    async def test_credential_deletion(self, credential_vault: CredentialVault) -> None:
        """Deleted credential is no longer retrievable."""
        ctx = FakeAuthContext(
            user_id="del-user",
            org_id="tenant-a",
            permissions={
                "credentials:create",
                "credentials:read",
                "credentials:delete",
            },
        )

        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-del",
            auth_context=ctx,
        )

        await credential_vault.delete_credential(cred_id, auth_context=ctx)

        with pytest.raises(CredentialNotFoundError):
            await credential_vault.get_credential(cred_id, auth_context=ctx)

    @pytest.mark.asyncio
    async def test_rotation_policy_detection(self, credential_vault: CredentialVault) -> None:
        """Credentials with rotation due are found by get_credentials_needing_rotation."""
        ctx = FakeAuthContext(
            user_id="rot-user",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:list"},
        )

        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-rot",
            auth_context=ctx,
            rotation_policy=RotationPolicy(interval_days=1),
        )

        # Force the creation date to the past so rotation is due
        cred = credential_vault._credentials[cred_id]
        cred.metadata.created_at = datetime.now(timezone.utc) - timedelta(days=10)

        needing = await credential_vault.get_credentials_needing_rotation(
            tenant_id="tenant-a",
            auth_context=ctx,
        )
        assert any(c.credential_id == cred_id for c in needing)


# =============================================================================
# 5. Failure Scenarios
# =============================================================================


class TestFailureScenarios:
    """Circuit breaker trip -> recovery probe -> resume."""

    @pytest.mark.asyncio
    async def test_adapter_sandbox_failure_returns_error(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Sandbox execution failure propagates as a failed GatewayResult."""
        request = AragoraRequest(
            content="Generate report",
            capabilities=["text_generation"],
        )

        with patch.object(adapter.sandbox, "execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = SandboxResult(
                status=SandboxStatus.FAILED,
                output=None,
                error="Timeout exceeded",
                execution_time_ms=5000,
            )
            result = await adapter.execute_task(request, auth_context)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_adapter_sandbox_exception_propagates(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Unexpected exception in sandbox propagates to caller."""
        request = AragoraRequest(
            content="Test exception handling",
            capabilities=["text_generation"],
        )

        with patch.object(adapter.sandbox, "execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = RuntimeError("Unexpected sandbox crash")
            with pytest.raises(RuntimeError, match="Unexpected sandbox crash"):
                await adapter.execute_task(request, auth_context)

    @pytest.mark.asyncio
    async def test_capability_blocked_action(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Blocked capability is rejected before sandbox execution."""
        request = AragoraRequest(
            content="Execute dangerous command",
            capabilities=["shell_execute"],
        )
        result = await adapter.execute_task(request, auth_context)
        assert result.success is False
        assert result.blocked_reason == "capability_blocked"

    @pytest.mark.asyncio
    async def test_credential_vault_rate_limiting(self, vault_audit: MockVaultAuditLogger) -> None:
        """Credential access is rate-limited."""
        vault = CredentialVault(
            audit_logger=vault_audit,
            rate_limiter=CredentialRateLimiter(
                max_per_minute=3,
                max_per_hour=10,
                lockout_duration_seconds=60,
            ),
        )
        ctx = FakeAuthContext(
            user_id="rl-user",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:read"},
        )

        cred_id = await vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-rate",
            auth_context=ctx,
        )

        # First 3 accesses should work
        for _ in range(3):
            await vault.get_credential_value(cred_id, auth_context=ctx)

        # 4th should be rate limited
        from aragora.gateway.openclaw.credential_vault import CredentialRateLimitedError

        with pytest.raises(CredentialRateLimitedError):
            await vault.get_credential_value(cred_id, auth_context=ctx)

    @pytest.mark.asyncio
    async def test_health_check_with_unavailable_openclaw(
        self, health_checker: GatewayHealthChecker
    ) -> None:
        """Health check reports unhealthy when OpenClaw is unreachable."""
        # OpenClaw at localhost:8081 is not running, so check should fail or
        # report unhealthy/unknown.
        result = await health_checker.check_openclaw()
        assert result.status in (HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN)

    @pytest.mark.asyncio
    async def test_health_check_readiness_aggregation(
        self, health_checker: GatewayHealthChecker
    ) -> None:
        """Readiness probe aggregates all component statuses."""
        status = await health_checker.check_readiness()
        assert isinstance(status, GatewayHealthStatus)
        assert "openclaw" in status.components
        # At least some components should be checked
        assert len(status.components) >= 1

    @pytest.mark.asyncio
    async def test_health_liveness_caches_result(
        self, health_checker: GatewayHealthChecker
    ) -> None:
        """Liveness probe returns cached readiness result."""
        # First call fills cache
        status1 = await health_checker.check_readiness()
        # Liveness should return cached
        status2 = await health_checker.check_liveness()
        assert status2.checked_at == status1.checked_at

    @pytest.mark.asyncio
    async def test_custom_health_check_registration(
        self, health_checker: GatewayHealthChecker
    ) -> None:
        """Custom health check can be registered and executed."""

        async def custom_check() -> ComponentHealth:
            return ComponentHealth(
                name="custom_db",
                status=HealthStatus.HEALTHY,
                latency_ms=1.0,
            )

        health_checker.register_check("custom_db", custom_check)
        status = await health_checker.check_readiness()
        assert "custom_db" in status.components
        assert status.components["custom_db"].status == HealthStatus.HEALTHY

        # Cleanup
        health_checker.unregister_check("custom_db")


# =============================================================================
# 6. Multi-Tenant Isolation
# =============================================================================


class TestMultiTenantIsolation:
    """Two tenants cannot access each other's sessions/credentials."""

    @pytest.mark.asyncio
    async def test_credential_tenant_isolation(self, credential_vault: CredentialVault) -> None:
        """Tenant A cannot access Tenant B's credentials."""
        ctx_a = FakeAuthContext(
            user_id="user-a",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:read"},
        )
        ctx_b = FakeAuthContext(
            user_id="user-b",
            org_id="tenant-b",
            permissions={"credentials:create", "credentials:read"},
        )

        # Tenant A stores a credential
        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-tenant-a-secret",
            auth_context=ctx_a,
        )

        # Tenant B tries to access it
        with pytest.raises(TenantIsolationError):
            await credential_vault.get_credential(cred_id, auth_context=ctx_b)

    @pytest.mark.asyncio
    async def test_credential_listing_respects_tenant(
        self, credential_vault: CredentialVault
    ) -> None:
        """List credentials only returns items for the caller's tenant."""
        ctx_a = FakeAuthContext(
            user_id="user-a",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:list"},
        )
        ctx_b = FakeAuthContext(
            user_id="user-b",
            org_id="tenant-b",
            permissions={"credentials:create", "credentials:list"},
        )

        await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-a",
            auth_context=ctx_a,
        )
        await credential_vault.store_credential(
            tenant_id="tenant-b",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-b",
            auth_context=ctx_b,
        )

        # Tenant A should only see their own
        creds_a = await credential_vault.list_credentials(auth_context=ctx_a)
        assert all(c.tenant_id == "tenant-a" for c in creds_a)

        # Tenant B should only see their own
        creds_b = await credential_vault.list_credentials(auth_context=ctx_b)
        assert all(c.tenant_id == "tenant-b" for c in creds_b)

    @pytest.mark.asyncio
    async def test_credential_rotation_denied_cross_tenant(
        self, credential_vault: CredentialVault
    ) -> None:
        """Tenant B cannot rotate Tenant A's credential."""
        ctx_a = FakeAuthContext(
            user_id="user-a",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:read", "credentials:rotate"},
        )
        ctx_b = FakeAuthContext(
            user_id="user-b",
            org_id="tenant-b",
            permissions={"credentials:read", "credentials:rotate"},
        )

        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-cross-tenant",
            auth_context=ctx_a,
        )

        with pytest.raises(TenantIsolationError):
            await credential_vault.rotate_credential(
                cred_id, new_value="sk-hacked", auth_context=ctx_b
            )

    @pytest.mark.asyncio
    async def test_admin_cross_tenant_access(self, credential_vault: CredentialVault) -> None:
        """Admin role grants cross-tenant credential access."""
        ctx_a = FakeAuthContext(
            user_id="user-a",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:read"},
        )
        admin_ctx = FakeAuthContext(
            user_id="admin-user",
            org_id="tenant-admin",
            roles={"admin"},
            permissions={"credentials:read"},
        )

        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-admin-access",
            auth_context=ctx_a,
        )

        # Admin should be able to read cross-tenant
        cred = await credential_vault.get_credential(cred_id, auth_context=admin_ctx)
        assert cred.credential_id == cred_id

    @pytest.mark.asyncio
    async def test_action_filter_tenant_scoping(self) -> None:
        """Separate action filters per tenant enforce different allowlists."""
        filter_a = ActionFilter(
            tenant_id="tenant-a",
            allowed_actions={"browser.navigate", "browser.click"},
        )
        filter_b = ActionFilter(
            tenant_id="tenant-b",
            allowed_actions={"browser.navigate"},
        )

        # browser.click allowed for tenant-a but not tenant-b
        decision_a = filter_a.check_action("browser.click")
        assert decision_a.allowed is True

        decision_b = filter_b.check_action("browser.click")
        assert decision_b.allowed is False
        assert "not in tenant allowlist" in decision_b.reason

    @pytest.mark.asyncio
    async def test_decision_router_tenant_isolation(self, decision_router: DecisionRouter) -> None:
        """Different tenants get different routing based on their config."""
        await decision_router.add_tenant_config(
            TenantRoutingConfig(
                tenant_id="cautious-tenant",
                criteria=RoutingCriteria(financial_threshold=50.0),
            )
        )

        # Same request, different tenants
        decision_cautious = await decision_router.route(
            {"action": "purchase", "amount": 100},
            context={"tenant_id": "cautious-tenant"},
        )
        decision_default = await decision_router.route(
            {"action": "purchase", "amount": 100},
            context={"tenant_id": "relaxed-tenant"},
        )

        assert decision_cautious.destination == RouteDestination.DEBATE
        assert decision_default.destination == RouteDestination.EXECUTE


# =============================================================================
# 7. Audit Trail Integrity
# =============================================================================


class TestAuditTrailIntegrity:
    """Full request/response chain with hash verification."""

    @pytest.mark.asyncio
    async def test_vault_audit_complete_lifecycle(
        self, credential_vault: CredentialVault, vault_audit: MockVaultAuditLogger
    ) -> None:
        """Full credential lifecycle produces complete audit trail."""
        ctx = FakeAuthContext(
            user_id="audit-life",
            org_id="tenant-a",
            permissions={
                "credentials:create",
                "credentials:read",
                "credentials:rotate",
                "credentials:delete",
            },
        )

        # Create
        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-audit",
            auth_context=ctx,
        )

        # Access
        await credential_vault.get_credential_value(cred_id, auth_context=ctx)

        # Rotate
        await credential_vault.rotate_credential(
            cred_id, new_value="sk-audit-new", auth_context=ctx
        )

        # Delete
        await credential_vault.delete_credential(cred_id, auth_context=ctx)

        # Verify audit trail
        events = [e["event_type"] for e in vault_audit.events]
        assert "credential_created" in events
        assert "credential_accessed" in events
        assert "credential_rotated" in events
        assert "credential_deleted" in events

        # All events should reference correct actor
        for event in vault_audit.events:
            assert event["actor_id"] == "audit-life"

    @pytest.mark.asyncio
    async def test_bridge_audit_trail_timestamps(self, auth_bridge: AuthBridge) -> None:
        """Audit entries have monotonically increasing timestamps."""
        context = AuthContext(
            user_id="ts-user",
            email="ts@test.com",
            permissions={"gateway.execute"},
        )
        session = await auth_bridge.create_session(context)
        await auth_bridge.verify_request(session_id=session.session_id)
        await auth_bridge.destroy_session(session.session_id)

        log = await auth_bridge.get_audit_log(user_id="ts-user")
        timestamps = [e["timestamp"] for e in log]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    @pytest.mark.asyncio
    async def test_action_filter_decision_ids_unique(self, action_filter: ActionFilter) -> None:
        """Each filter decision has a unique decision_id."""
        decisions = []
        for action in ["browser.navigate", "browser.click", "system.rm_rf", "filesystem.read"]:
            d = action_filter.check_action(action)
            decisions.append(d)

        ids = [d.decision_id for d in decisions]
        assert len(ids) == len(set(ids)), "Decision IDs must be unique"

    @pytest.mark.asyncio
    async def test_filter_decision_hash_integrity(self, action_filter: ActionFilter) -> None:
        """Decision ID is a deterministic hash of action + tenant + timestamp."""
        decision = action_filter.check_action("browser.navigate")
        # Verify the decision_id was generated from the expected data
        data = f"{decision.action}:{decision.tenant_id}:{decision.timestamp}"
        expected_hash = hashlib.sha256(data.encode()).hexdigest()[:16]
        assert decision.decision_id == expected_hash

    @pytest.mark.asyncio
    async def test_routing_audit_entries(self, decision_router: DecisionRouter) -> None:
        """Decision router produces audit entries for all routing decisions."""
        await decision_router.route({"action": "simple_query"})
        await decision_router.route({"action": "transfer", "amount": 50000})

        assert len(decision_router._audit_log) >= 2
        for entry in decision_router._audit_log:
            assert entry.request_id
            assert entry.timestamp is not None

    @pytest.mark.asyncio
    async def test_bridge_audit_filtered_by_event_type(self, auth_bridge: AuthBridge) -> None:
        """Audit log can be filtered by event type."""
        ctx = AuthContext(
            user_id="filter-user",
            email="filter@test.com",
            permissions=set(),
        )
        session = await auth_bridge.create_session(ctx)
        await auth_bridge.verify_request(session_id=session.session_id)

        auth_events = await auth_bridge.get_audit_log(event_type="authentication")
        session_events = await auth_bridge.get_audit_log(event_type="session")

        assert all(e["event_type"] == "authentication" for e in auth_events)
        assert all(e["event_type"] == "session" for e in session_events)

    @pytest.mark.asyncio
    async def test_vault_audit_denied_access_logged(
        self, credential_vault: CredentialVault, vault_audit: MockVaultAuditLogger
    ) -> None:
        """Denied credential access attempts are logged with warning severity."""
        owner_ctx = FakeAuthContext(
            user_id="owner",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:read"},
        )
        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-denied",
            auth_context=owner_ctx,
        )

        # User without read permission
        no_perm_ctx = FakeAuthContext(
            user_id="intruder",
            org_id="tenant-a",
            permissions=set(),
        )
        with pytest.raises(CredentialAccessDeniedError):
            await credential_vault.get_credential(cred_id, auth_context=no_perm_ctx)

        # Verify denial was logged
        denied_events = [
            e for e in vault_audit.events if e["event_type"] == "credential_access_denied"
        ]
        assert len(denied_events) >= 1
        assert denied_events[0]["severity"] == "warning"
        assert denied_events[0]["actor_id"] == "intruder"


# =============================================================================
# 8. Cross-Component Integration
# =============================================================================


class TestCrossComponentIntegration:
    """Tests that span multiple gateway components in a single flow."""

    @pytest.mark.asyncio
    async def test_auth_to_route_to_execute(
        self, auth_bridge: AuthBridge, decision_router: DecisionRouter
    ) -> None:
        """Full flow: auth -> route decision -> simulated execution."""
        # Authenticate
        context = AuthContext(
            user_id="cross-user",
            email="cross@test.com",
            permissions={"gateway.execute"},
        )
        session = await auth_bridge.create_session(context)
        auth_ctx = await auth_bridge.verify_request(session_id=session.session_id)

        # Route request
        decision = await decision_router.route(
            {"action": "simple_query", "amount": 10},
            context={"tenant_id": "cross-tenant"},
        )

        # Low-value query should go to execute
        assert decision.destination == RouteDestination.EXECUTE
        assert auth_ctx.user_id == "cross-user"

    @pytest.mark.asyncio
    async def test_auth_to_filter_to_execute(
        self,
        auth_bridge: AuthBridge,
        action_filter: ActionFilter,
        adapter: OpenClawGatewayAdapter,
        auth_context: AuthorizationContext,
    ) -> None:
        """Auth bridge -> action filter -> adapter execution chain."""
        context = AuthContext(
            user_id="chain-user",
            email="chain@test.com",
            permissions={"gateway.execute"},
        )
        session = await auth_bridge.create_session(context)
        _ = await auth_bridge.verify_request(session_id=session.session_id)

        # Filter allows browser.navigate
        filter_decision = action_filter.check_action("browser.navigate")
        assert filter_decision.allowed is True

        # Execute via adapter
        request = AragoraRequest(
            content="Navigate to page",
            capabilities=["text_generation"],
        )

        with patch.object(adapter.sandbox, "execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = SandboxResult(
                status=SandboxStatus.COMPLETED,
                output={"text": "Page loaded"},
                execution_time_ms=50,
            )
            result = await adapter.execute_task(request, auth_context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_credential_used_with_adapter(
        self,
        credential_vault: CredentialVault,
        adapter: OpenClawGatewayAdapter,
        auth_context: AuthorizationContext,
    ) -> None:
        """Credential retrieved from vault is available for adapter task."""
        vault_ctx = FakeAuthContext(
            user_id="cred-adapter-user",
            org_id="tenant-integ",
            permissions={"credentials:create", "credentials:read", "credentials:list"},
        )

        await credential_vault.store_credential(
            tenant_id="tenant-integ",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-integ-test",
            auth_context=vault_ctx,
        )

        # Get credentials for execution
        creds = await credential_vault.get_credentials_for_execution(
            tenant_id="tenant-integ",
            auth_context=vault_ctx,
        )
        assert "openai" in creds
        assert creds["openai"] == "sk-integ-test"

        # Execute task (credential would be injected in real system)
        request = AragoraRequest(
            content="Use credential for task",
            capabilities=["text_generation"],
            context={"api_key": creds["openai"]},
        )

        with patch.object(adapter.sandbox, "execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = SandboxResult(
                status=SandboxStatus.COMPLETED,
                output={"text": "Done"},
                execution_time_ms=75,
            )
            result = await adapter.execute_task(request, auth_context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_health_check_serialization(self, health_checker: GatewayHealthChecker) -> None:
        """Health check results serialize to a valid dict."""
        status = await health_checker.check_readiness()
        d = status.to_dict()
        assert "status" in d
        assert "components" in d
        assert "ready" in d
        assert isinstance(d["ready"], bool)

    @pytest.mark.asyncio
    async def test_token_exchange_with_scoped_actions(self, auth_bridge: AuthBridge) -> None:
        """Token exchange generates token with allowed actions as scope."""
        context = AuthContext(
            user_id="exchange-user",
            email="exchange@test.com",
            permissions={"gateway.execute"},
        )

        result = await auth_bridge.exchange_token(
            context,
            target_audience="external-api",
            token_lifetime=1800,
        )

        assert result.access_token
        assert result.token_type == "Bearer"
        assert result.expires_in == 1800
        assert result.audience == "external-api"

    @pytest.mark.asyncio
    async def test_bridge_stats_reflect_state(self, auth_bridge: AuthBridge) -> None:
        """AuthBridge.get_stats() reflects current session and mapping counts."""
        ctx = AuthContext(
            user_id="stats-user",
            email="stats@test.com",
            permissions=set(),
        )
        await auth_bridge.create_session(ctx)
        stats = auth_bridge.get_stats()

        assert stats["permission_mappings"] == 3
        assert stats["total_sessions"] >= 1
        assert stats["audit_enabled"] is True

    @pytest.mark.asyncio
    async def test_vault_stats_after_operations(self, credential_vault: CredentialVault) -> None:
        """CredentialVault.get_stats() reflects stored credentials."""
        ctx = FakeAuthContext(
            user_id="stats-cred",
            org_id="tenant-a",
            permissions={"credentials:create"},
        )

        await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-stats",
            auth_context=ctx,
        )

        stats = credential_vault.get_stats()
        assert stats["total_credentials"] >= 1
        assert "openai" in stats["by_framework"]
        assert "tenant-a" in stats["by_tenant"]

    @pytest.mark.asyncio
    async def test_routing_with_multiple_criteria_match(
        self, decision_router: DecisionRouter
    ) -> None:
        """Request matching multiple criteria still produces a single decision."""
        decision = await decision_router.route(
            {
                "action": "large_transfer",
                "amount": 100000,
                "risk_level": "critical",
                "compliance_flags": ["pii", "financial"],
                "content": "We need to debate this",
            },
            context={"tenant_id": "multi-criteria"},
        )
        assert decision.destination == RouteDestination.DEBATE
        assert len(decision.criteria_matched) >= 1

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, auth_bridge: AuthBridge) -> None:
        """Expired sessions are removed by cleanup_expired_sessions."""
        ctx = AuthContext(
            user_id="cleanup-user",
            email="cleanup@test.com",
            permissions=set(),
        )
        s1 = await auth_bridge.create_session(ctx, duration=1)
        s2 = await auth_bridge.create_session(ctx, duration=3600)

        # Force s1 expiry
        s1.expires_at = time.time() - 10

        count = await auth_bridge.cleanup_expired_sessions()
        assert count >= 1

        # s1 should be gone, s2 should remain
        assert await auth_bridge.get_session(s1.session_id) is None
        assert await auth_bridge.get_session(s2.session_id) is not None

    @pytest.mark.asyncio
    async def test_agent_allowlist_enforcement(self, credential_vault: CredentialVault) -> None:
        """Credential with agent allowlist rejects unlisted agents."""
        ctx = FakeAuthContext(
            user_id="agent-user",
            org_id="tenant-a",
            permissions={"credentials:create", "credentials:read"},
        )

        cred_id = await credential_vault.store_credential(
            tenant_id="tenant-a",
            framework="openai",
            credential_type=CredentialType.API_KEY,
            value="sk-agent",
            auth_context=ctx,
            allowed_agents=["claude", "gpt4"],
        )

        # Allowed agent
        value = await credential_vault.get_credential_value(
            cred_id, auth_context=ctx, agent_name="claude"
        )
        assert value == "sk-agent"

        # Disallowed agent
        with pytest.raises(CredentialAccessDeniedError):
            await credential_vault.get_credential_value(
                cred_id, auth_context=ctx, agent_name="unknown-agent"
            )
