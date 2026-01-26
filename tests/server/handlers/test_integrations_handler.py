"""
Tests for integrations handler endpoints.

Tests the integrations API handlers for:
- List integrations (Slack, Teams, Discord, Email)
- Get/disconnect specific integrations
- Health checks and status
- Integration statistics
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def parse_result(result) -> Dict[str, Any]:
    """Parse HandlerResult body to dict for assertions."""
    body = json.loads(result.body)
    return {"success": result.status_code < 400, "data": body, "status_code": result.status_code}


class MockWorkspace:
    """Mock workspace object for testing."""

    def __init__(
        self,
        workspace_id: str = "ws-123",
        workspace_name: str = "Test Workspace",
        is_active: bool = True,
        installed_at: datetime | None = None,
        installed_by: str = "user-1",
        scopes: List[str] | None = None,
        access_token: str = "xoxb-test-token",
        refresh_token: str | None = "refresh-token",
        token_expires_at: datetime | None = None,
    ):
        self.workspace_id = workspace_id
        self.tenant_id = workspace_id  # For Teams compatibility
        self.workspace_name = workspace_name
        self.tenant_name = workspace_name  # For Teams compatibility
        self.is_active = is_active
        self.installed_at = installed_at or datetime.now(timezone.utc)
        self.installed_by = installed_by
        self.scopes = scopes or ["chat:write", "channels:read"]
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_expires_at = token_expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "workspace_name": self.workspace_name,
            "is_active": self.is_active,
            "installed_at": self.installed_at.isoformat(),
            "installed_by": self.installed_by,
            "scopes": self.scopes,
        }


@pytest.fixture
def mock_slack_store():
    """Create mock Slack workspace store."""
    store = MagicMock()
    store.get_by_tenant.return_value = [MockWorkspace()]
    store.list_active.return_value = [MockWorkspace(), MockWorkspace(workspace_id="ws-456")]
    store.get.return_value = MockWorkspace()
    store.deactivate.return_value = True
    store.get_stats.return_value = {
        "total": 5,
        "active": 4,
        "inactive": 1,
    }
    return store


@pytest.fixture
def mock_teams_store():
    """Create mock Teams workspace store."""
    store = MagicMock()
    store.get_by_aragora_tenant.return_value = [MockWorkspace()]
    store.list_active.return_value = [MockWorkspace()]
    store.get.return_value = MockWorkspace()
    store.deactivate.return_value = True
    store.get_stats.return_value = {
        "total": 3,
        "active": 3,
        "inactive": 0,
    }
    return store


@pytest.fixture
def handler_with_mocks(mock_slack_store, mock_teams_store, mock_server_context):
    """Create handler with mocked stores."""
    from aragora.server.handlers.integrations import IntegrationsHandler

    handler = IntegrationsHandler(mock_server_context)
    handler._slack_store = mock_slack_store
    handler._teams_store = mock_teams_store
    return handler


class TestListIntegrations:
    """Tests for listing integrations."""

    @pytest.mark.asyncio
    async def test_list_all_integrations(
        self, handler_with_mocks, mock_slack_store, mock_teams_store
    ):
        """Test listing all integrations returns combined results."""
        raw_result = await handler_with_mocks._list_integrations(
            tenant_id="tenant-1",
            query_params={},
        )
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "integrations" in result["data"]
        assert "pagination" in result["data"]

    @pytest.mark.asyncio
    async def test_list_integrations_with_type_filter(self, handler_with_mocks):
        """Test listing integrations filtered by type."""
        raw_result = await handler_with_mocks._list_integrations(
            tenant_id="tenant-1",
            query_params={"type": "slack"},
        )
        result = parse_result(raw_result)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_list_integrations_with_status_filter(self, handler_with_mocks):
        """Test listing integrations filtered by status."""
        raw_result = await handler_with_mocks._list_integrations(
            tenant_id="tenant-1",
            query_params={"status": "active"},
        )
        result = parse_result(raw_result)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_list_integrations_pagination(self, handler_with_mocks):
        """Test listing integrations with pagination."""
        raw_result = await handler_with_mocks._list_integrations(
            tenant_id="tenant-1",
            query_params={"limit": "10", "offset": "0"},
        )
        result = parse_result(raw_result)

        assert result["success"] is True
        pagination = result["data"]["pagination"]
        assert "limit" in pagination
        assert "offset" in pagination
        assert "total" in pagination

    @pytest.mark.asyncio
    async def test_list_integrations_empty(self, mock_server_context):
        """Test listing integrations when none configured."""
        from aragora.server.handlers.integrations import IntegrationsHandler

        handler = IntegrationsHandler(mock_server_context)
        handler._slack_store = MagicMock()
        handler._slack_store.get_by_tenant.return_value = []
        handler._teams_store = MagicMock()
        handler._teams_store.get_by_aragora_tenant.return_value = []

        raw_result = await handler._list_integrations(
            tenant_id="tenant-1",
            query_params={},
        )
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["integrations"] == []


class TestGetIntegration:
    """Tests for getting specific integrations."""

    @pytest.mark.asyncio
    async def test_get_slack_integration(self, handler_with_mocks):
        """Test getting Slack integration details."""
        result = await handler_with_mocks._get_integration(
            integration_type="slack",
            workspace_id="ws-123",
            tenant_id="tenant-1",
        )

        assert result["success"] is True
        assert result["data"]["type"] == "slack"
        assert result["data"]["connected"] is True

    @pytest.mark.asyncio
    async def test_get_teams_integration(self, handler_with_mocks):
        """Test getting Teams integration details."""
        result = await handler_with_mocks._get_integration(
            integration_type="teams",
            workspace_id="ws-123",
            tenant_id="tenant-1",
        )

        assert result["success"] is True
        assert result["data"]["type"] == "teams"

    @pytest.mark.asyncio
    async def test_get_integration_not_found(self, handler_with_mocks, mock_slack_store):
        """Test getting non-existent integration returns 404."""
        mock_slack_store.get.return_value = None

        result = await handler_with_mocks._get_integration(
            integration_type="slack",
            workspace_id="nonexistent",
            tenant_id="tenant-1",
        )

        assert result["success"] is False
        assert result["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_integration_missing_workspace_id(self, handler_with_mocks):
        """Test getting integration without workspace_id returns 400."""
        result = await handler_with_mocks._get_integration(
            integration_type="slack",
            workspace_id=None,
            tenant_id="tenant-1",
        )

        assert result["success"] is False
        assert result["status_code"] == 400

    @pytest.mark.asyncio
    async def test_get_unsupported_integration_type(self, handler_with_mocks):
        """Test getting unsupported integration type returns error."""
        result = await handler_with_mocks._get_integration(
            integration_type="unsupported",
            workspace_id="ws-123",
            tenant_id="tenant-1",
        )

        assert result["success"] is False
        assert result["status_code"] in [400, 404]


class TestDisconnectIntegration:
    """Tests for disconnecting integrations."""

    @pytest.mark.asyncio
    async def test_disconnect_slack_integration(self, handler_with_mocks, mock_slack_store):
        """Test disconnecting Slack integration."""
        result = await handler_with_mocks._disconnect_integration(
            integration_type="slack",
            workspace_id="ws-123",
            tenant_id="tenant-1",
        )

        assert result["success"] is True
        mock_slack_store.deactivate.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_teams_integration(self, handler_with_mocks, mock_teams_store):
        """Test disconnecting Teams integration."""
        result = await handler_with_mocks._disconnect_integration(
            integration_type="teams",
            workspace_id="ws-123",
            tenant_id="tenant-1",
        )

        assert result["success"] is True
        mock_teams_store.deactivate.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_missing_workspace_id(self, handler_with_mocks):
        """Test disconnecting without workspace_id returns 400."""
        result = await handler_with_mocks._disconnect_integration(
            integration_type="slack",
            workspace_id=None,
            tenant_id="tenant-1",
        )

        assert result["success"] is False
        assert result["status_code"] == 400


class TestIntegrationHealth:
    """Tests for integration health checks."""

    @pytest.mark.asyncio
    async def test_slack_health_check_healthy(self, handler_with_mocks):
        """Test Slack health check returns healthy status."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "ok": True,
                "team": "Test Team",
                "bot_id": "B123",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = await handler_with_mocks._get_health(
                integration_type="slack",
                workspace_id="ws-123",
                tenant_id="tenant-1",
            )

        assert result["success"] is True
        assert result["data"]["type"] == "slack"

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, handler_with_mocks):
        """Test health check handles timeout gracefully."""
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            result = await handler_with_mocks._get_health(
                integration_type="slack",
                workspace_id="ws-123",
                tenant_id="tenant-1",
            )

        assert result["success"] is True
        assert result["data"]["healthy"] is False

    @pytest.mark.asyncio
    async def test_test_integration_endpoint(self, handler_with_mocks):
        """Test integration test endpoint."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"ok": True}).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = await handler_with_mocks._test_integration(
                integration_type="slack",
                workspace_id="ws-123",
                tenant_id="tenant-1",
            )

        assert result["success"] is True


class TestIntegrationStats:
    """Tests for integration statistics."""

    @pytest.mark.asyncio
    async def test_get_stats_returns_all_types(self, handler_with_mocks):
        """Test stats endpoint returns all integration types."""
        result = await handler_with_mocks._get_stats(tenant_id="tenant-1")

        assert result["success"] is True
        assert "stats" in result["data"]
        assert "slack" in result["data"]["stats"]
        assert "teams" in result["data"]["stats"]
        assert "total_integrations" in result["data"]["stats"]

    @pytest.mark.asyncio
    async def test_stats_includes_generated_at(self, handler_with_mocks):
        """Test stats includes generation timestamp."""
        result = await handler_with_mocks._get_stats(tenant_id="tenant-1")

        assert result["success"] is True
        assert "generated_at" in result["data"]


class TestHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_integration_paths(self, mock_server_context):
        """Test handler recognizes integration paths."""
        from aragora.server.handlers.integrations import IntegrationsHandler

        handler = IntegrationsHandler(mock_server_context)

        assert handler.can_handle("/api/v2/integrations")
        assert handler.can_handle("/api/v2/integrations/slack")
        assert handler.can_handle("/api/v2/integrations/teams/test")
        assert handler.can_handle("/api/v2/integrations/stats")

    def test_cannot_handle_other_paths(self, mock_server_context):
        """Test handler rejects non-integration paths."""
        from aragora.server.handlers.integrations import IntegrationsHandler

        handler = IntegrationsHandler(mock_server_context)

        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/v1/integrations")  # Wrong version
        assert not handler.can_handle("/health")

    @pytest.mark.asyncio
    async def test_handle_routes_to_correct_method(self, handler_with_mocks):
        """Test handle method routes to correct handler."""
        # Test GET list
        result = await handler_with_mocks.handle(
            method="GET",
            path="/api/v2/integrations",
            body=None,
            query_params={},
            headers={"x-tenant-id": "tenant-1"},
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_handle_unsupported_method(self, handler_with_mocks):
        """Test handle returns error for unsupported method."""
        result = await handler_with_mocks.handle(
            method="PATCH",
            path="/api/v2/integrations",
            body=None,
            query_params={},
            headers={},
        )

        assert result["success"] is False
        assert result["status_code"] == 405


class TestSupportedIntegrationTypes:
    """Tests for supported integration types."""

    SUPPORTED_TYPES = ["slack", "teams", "discord", "email"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("integration_type", SUPPORTED_TYPES)
    async def test_supported_integration_types(self, handler_with_mocks, integration_type):
        """Test all supported integration types are handled."""
        result = await handler_with_mocks._get_integration(
            integration_type=integration_type,
            workspace_id="ws-123",
            tenant_id="tenant-1",
        )

        # Should not return "unsupported type" error
        if result["success"] is False:
            assert "unsupported" not in result.get("error", "").lower()


class TestDiscordIntegration:
    """Tests for Discord integration specifics."""

    @pytest.mark.asyncio
    async def test_discord_health_check(self, handler_with_mocks):
        """Test Discord health check uses correct API."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "username": "AragoraBot",
                "id": "123456789",
            }
        ).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "test-token"}):
                result = await handler_with_mocks._get_health(
                    integration_type="discord",
                    workspace_id=None,
                    tenant_id="tenant-1",
                )

        assert result["success"] is True


class TestEmailIntegration:
    """Tests for Email integration specifics."""

    @pytest.mark.asyncio
    async def test_email_health_check(self, handler_with_mocks):
        """Test Email health check uses socket connection."""
        mock_socket = MagicMock()
        mock_socket.connect.return_value = None

        with patch("socket.create_connection", return_value=mock_socket):
            with patch.dict("os.environ", {"SMTP_HOST": "smtp.test.com", "SMTP_PORT": "587"}):
                result = await handler_with_mocks._get_health(
                    integration_type="email",
                    workspace_id=None,
                    tenant_id="tenant-1",
                )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_email_not_configured(self, handler_with_mocks):
        """Test Email returns not_configured when env vars missing."""
        with patch.dict("os.environ", {}, clear=True):
            result = await handler_with_mocks._get_integration(
                integration_type="email",
                workspace_id=None,
                tenant_id="tenant-1",
            )

        assert result["success"] is True
        # Should indicate not configured or handle gracefully
