"""
Tests for OpenClaw gateway adapter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.openclaw.adapter import (
    DeviceHandle,
    DeviceRegistration,
    GatewayResult,
    OpenClawGatewayAdapter,
    PluginInstallRequest,
)
from aragora.gateway.openclaw.capabilities import CapabilityFilter
from aragora.gateway.openclaw.protocol import (
    AragoraRequest,
    AuthorizationContext,
    TenantContext,
)
from aragora.gateway.openclaw.sandbox import SandboxConfig, SandboxResult, SandboxStatus


class MockRBACChecker:
    """Mock RBAC checker for tests."""

    def __init__(self, allow_all: bool = True):
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
    """Mock audit logger for tests."""

    def __init__(self):
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


class MockApprovalGate:
    """Mock approval gate for tests."""

    def __init__(self, approve_all: bool = True):
        self.approve_all = approve_all

    async def check_approval(
        self,
        gate: str,
        actor_id: str,
        resource_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        if self.approve_all:
            return True, "approved"
        return False, "approval required"


class TestOpenClawGatewayAdapter:
    """Tests for OpenClawGatewayAdapter."""

    @pytest.fixture
    def auth_context(self) -> AuthorizationContext:
        """Create auth context for tests."""
        return AuthorizationContext(
            actor_id="user-123",
            actor_type="user",
            permissions={"gateway.execute"},
        )

    @pytest.fixture
    def tenant_context(self) -> TenantContext:
        """Create tenant context for tests."""
        return TenantContext(
            tenant_id="tenant-456",
            organization_id="org-789",
            enabled_capabilities={"text_generation", "search_internal"},
        )

    @pytest.fixture
    def adapter(self) -> OpenClawGatewayAdapter:
        """Create adapter with mocks."""
        return OpenClawGatewayAdapter(
            openclaw_endpoint="http://localhost:8081",
            rbac_checker=MockRBACChecker(),
            audit_logger=MockAuditLogger(),
            sandbox_config=SandboxConfig(),
            capability_filter=CapabilityFilter(),
        )

    @pytest.fixture
    def adapter_no_rbac(self) -> OpenClawGatewayAdapter:
        """Create adapter without RBAC checker."""
        return OpenClawGatewayAdapter(
            openclaw_endpoint="http://localhost:8081",
            audit_logger=MockAuditLogger(),
        )

    @pytest.mark.asyncio
    async def test_execute_task_success(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Test successful task execution."""
        request = AragoraRequest(
            content="Generate a summary",
            capabilities=["text_generation"],
        )

        # Mock sandbox execution
        with patch.object(adapter.sandbox, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = SandboxResult(
                status=SandboxStatus.COMPLETED,
                output={"text": "Summary here"},
                execution_time_ms=150,
            )

            result = await adapter.execute_task(request, auth_context)

        assert result.success is True
        assert result.response is not None
        assert result.response.status == "completed"

    @pytest.mark.asyncio
    async def test_execute_task_permission_denied(self, auth_context: AuthorizationContext) -> None:
        """Test task execution with permission denied."""
        adapter = OpenClawGatewayAdapter(
            openclaw_endpoint="http://localhost:8081",
            rbac_checker=MockRBACChecker(allow_all=False),
            audit_logger=MockAuditLogger(),
        )

        request = AragoraRequest(content="Test")

        result = await adapter.execute_task(request, auth_context)

        assert result.success is False
        assert result.blocked_reason == "permission_denied"
        assert "gateway.execute" in result.error

    @pytest.mark.asyncio
    async def test_execute_task_capability_blocked(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Test task execution with blocked capability."""
        request = AragoraRequest(
            content="Execute shell command",
            capabilities=["shell_execute"],  # Blocked by default
        )

        result = await adapter.execute_task(request, auth_context)

        assert result.success is False
        assert result.blocked_reason == "capability_blocked"
        assert "shell_execute" in result.error

    @pytest.mark.asyncio
    async def test_execute_task_approval_required(self, auth_context: AuthorizationContext) -> None:
        """Test task execution with capability requiring approval."""
        adapter = OpenClawGatewayAdapter(
            openclaw_endpoint="http://localhost:8081",
            rbac_checker=MockRBACChecker(),
            audit_logger=MockAuditLogger(),
            approval_gate=MockApprovalGate(approve_all=False),
        )

        request = AragoraRequest(
            content="Write to file",
            capabilities=["file_system_write"],  # Requires approval
        )

        result = await adapter.execute_task(request, auth_context)

        assert result.success is False
        assert result.blocked_reason == "approval_required"

    @pytest.mark.asyncio
    async def test_execute_task_with_tenant_context(
        self,
        adapter: OpenClawGatewayAdapter,
        auth_context: AuthorizationContext,
        tenant_context: TenantContext,
    ) -> None:
        """Test task execution with tenant context."""
        request = AragoraRequest(
            content="Search query",
            capabilities=["text_generation"],
        )

        with patch.object(adapter.sandbox, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = SandboxResult(
                status=SandboxStatus.COMPLETED,
                output={"text": "Results"},
            )

            result = await adapter.execute_task(request, auth_context, tenant_context)

        assert result.success is True
        # Verify tenant context was passed through

    @pytest.mark.asyncio
    async def test_execute_task_audit_logging(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Test that task execution logs audit events."""
        request = AragoraRequest(content="Test")

        with patch.object(adapter.sandbox, "execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = SandboxResult(
                status=SandboxStatus.COMPLETED,
                output={},
            )

            await adapter.execute_task(request, auth_context)

        # Check audit logger was called
        audit_logger = adapter.audit_logger
        assert len(audit_logger.events) >= 2  # submit + complete


class TestDeviceManagement:
    """Tests for device registration/unregistration."""

    @pytest.fixture
    def adapter(self) -> OpenClawGatewayAdapter:
        return OpenClawGatewayAdapter(
            openclaw_endpoint="http://localhost:8081",
            rbac_checker=MockRBACChecker(),
            audit_logger=MockAuditLogger(),
        )

    @pytest.fixture
    def auth_context(self) -> AuthorizationContext:
        return AuthorizationContext(actor_id="user-123")

    @pytest.mark.asyncio
    async def test_register_device(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Test device registration."""
        device = DeviceRegistration(
            device_id="device-001",
            device_name="My Laptop",
            device_type="desktop",
            capabilities=["text_generation"],
        )

        result = await adapter.register_device(device, auth_context)

        assert result.success is True
        assert "device_handle" in result.metadata
        assert result.metadata["device_handle"]["device_id"] == "device-001"

    @pytest.mark.asyncio
    async def test_register_device_permission_denied(
        self, auth_context: AuthorizationContext
    ) -> None:
        """Test device registration with permission denied."""
        adapter = OpenClawGatewayAdapter(
            openclaw_endpoint="http://localhost:8081",
            rbac_checker=MockRBACChecker(allow_all=False),
        )

        device = DeviceRegistration(
            device_id="device-001",
            device_name="My Laptop",
            device_type="desktop",
        )

        result = await adapter.register_device(device, auth_context)

        assert result.success is False
        assert result.blocked_reason == "permission_denied"

    @pytest.mark.asyncio
    async def test_unregister_device(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Test device unregistration."""
        result = await adapter.unregister_device("device-001", auth_context)

        assert result.success is True


class TestPluginManagement:
    """Tests for plugin installation/uninstallation."""

    @pytest.fixture
    def adapter(self) -> OpenClawGatewayAdapter:
        return OpenClawGatewayAdapter(
            openclaw_endpoint="http://localhost:8081",
            rbac_checker=MockRBACChecker(),
            audit_logger=MockAuditLogger(),
            sandbox_config=SandboxConfig(
                plugin_allowlist_mode=True,
                allowed_plugins=["allowed-plugin"],
            ),
        )

    @pytest.fixture
    def auth_context(self) -> AuthorizationContext:
        return AuthorizationContext(actor_id="user-123")

    @pytest.mark.asyncio
    async def test_install_plugin_allowed(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Test installing allowed plugin."""
        plugin = PluginInstallRequest(
            plugin_id="allowed-plugin",
            plugin_name="Allowed Plugin",
            version="1.0.0",
            source="marketplace",
        )

        result = await adapter.install_plugin(plugin, auth_context)

        assert result.success is True
        assert "installed_at" in result.metadata

    @pytest.mark.asyncio
    async def test_install_plugin_not_in_allowlist(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Test installing plugin not in allowlist."""
        plugin = PluginInstallRequest(
            plugin_id="unauthorized-plugin",
            plugin_name="Unauthorized Plugin",
            version="1.0.0",
            source="marketplace",
        )

        result = await adapter.install_plugin(plugin, auth_context)

        assert result.success is False
        assert result.blocked_reason == "plugin_not_allowed"

    @pytest.mark.asyncio
    async def test_uninstall_plugin(
        self, adapter: OpenClawGatewayAdapter, auth_context: AuthorizationContext
    ) -> None:
        """Test plugin uninstallation."""
        result = await adapter.uninstall_plugin("some-plugin", auth_context)

        assert result.success is True


class TestConfigurationManagement:
    """Tests for configuration management."""

    def test_update_sandbox_config(self) -> None:
        """Test updating sandbox configuration."""
        adapter = OpenClawGatewayAdapter()

        new_config = SandboxConfig(max_memory_mb=1024)
        adapter.update_sandbox_config(new_config)

        assert adapter.sandbox_config.max_memory_mb == 1024

    def test_update_capability_filter(self) -> None:
        """Test updating capability filter."""
        adapter = OpenClawGatewayAdapter()

        new_filter = CapabilityFilter(blocked_override={"text_generation"})
        adapter.update_capability_filter(new_filter)

        assert "text_generation" in adapter.capability_filter.blocked_override

    def test_enable_tenant_capability(self) -> None:
        """Test enabling capability for tenant."""
        adapter = OpenClawGatewayAdapter()
        tenant = TenantContext(tenant_id="t1")

        adapter.enable_tenant_capability("browser_automation", tenant)

        assert "browser_automation" in adapter.capability_filter.tenant_enabled
        assert "browser_automation" in tenant.enabled_capabilities

    def test_add_plugin_to_allowlist(self) -> None:
        """Test adding plugin to allowlist."""
        adapter = OpenClawGatewayAdapter()

        adapter.add_plugin_to_allowlist("new-plugin")

        assert "new-plugin" in adapter.sandbox_config.allowed_plugins


class TestGatewayResult:
    """Tests for GatewayResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = GatewayResult(
            success=True,
            request_id="req-123",
            metadata={"execution_time_ms": 150},
        )

        assert result.success is True
        assert result.request_id == "req-123"
        assert result.error is None

    def test_failure_result(self) -> None:
        """Test failure result."""
        result = GatewayResult(
            success=False,
            request_id="req-456",
            error="Something went wrong",
            blocked_reason="capability_blocked",
        )

        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.blocked_reason == "capability_blocked"
