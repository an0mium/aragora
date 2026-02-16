"""Tests for Integration Management Handler.

Tests the IntegrationsHandler which provides REST API endpoints for
managing platform integrations (Slack, Teams, Discord, Email).
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.integration_management import IntegrationsHandler


def parse_body(result: HandlerResult) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


def get_error_fields(body: dict) -> tuple[str | None, str | None]:
    """Extract error message/code from either legacy or structured format."""
    error = body.get("error")
    if isinstance(error, dict):
        return error.get("message"), error.get("code")
    return error, body.get("code")


@pytest.fixture
def server_context():
    """Create mock server context."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
    }


@pytest.fixture
def handler(server_context):
    """Create integration management handler with mock context."""
    return IntegrationsHandler(server_context)


class TestIntegrationsHandlerCanHandle:
    """Tests for can_handle method."""

    def test_handles_list_integrations(self, handler):
        """Test that handler can handle list integrations path."""
        assert handler.can_handle("/api/v2/integrations", "GET") is True

    def test_handles_integration_type(self, handler):
        """Test that handler can handle specific integration type path."""
        assert handler.can_handle("/api/v2/integrations/slack", "GET") is True
        assert handler.can_handle("/api/v2/integrations/teams", "GET") is True

    def test_handles_delete(self, handler):
        """Test that handler can handle DELETE requests."""
        assert handler.can_handle("/api/v2/integrations/slack", "DELETE") is True

    def test_handles_post(self, handler):
        """Test that handler can handle POST requests."""
        assert handler.can_handle("/api/v2/integrations/slack/test", "POST") is True

    def test_rejects_unrelated_path(self, handler):
        """Test that handler rejects unrelated paths."""
        assert handler.can_handle("/api/debates", "GET") is False
        assert handler.can_handle("/api/v1/integrations/slack", "GET") is False


class TestIntegrationsHandlerValidation:
    """Tests for input validation."""

    @pytest.mark.asyncio
    async def test_invalid_path(self, handler):
        """Test that invalid path returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"
        mock_handler.headers = {}

        result = await handler.handle(
            "/api/v2/integrations/slack/invalid",
            {},
            mock_handler,
        )

        assert result.status_code == 400
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "INVALID_PATH"

    @pytest.mark.asyncio
    async def test_unsupported_integration(self, handler):
        """Test that unsupported integration returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"
        mock_handler.headers = {}

        result = await handler.handle(
            "/api/v2/integrations/unknown/health",
            {},
            mock_handler,
        )

        assert result.status_code == 400
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "UNSUPPORTED_INTEGRATION"

    @pytest.mark.asyncio
    async def test_not_found(self, handler):
        """Test that unknown endpoint returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.command = "PATCH"  # Unsupported method
        mock_handler.headers = {}

        result = await handler.handle(
            "/api/v2/integrations",
            {},
            mock_handler,
        )

        assert result.status_code == 404
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "NOT_FOUND"


class TestIntegrationTypeValidation:
    """Tests for integration type validation."""

    @pytest.mark.asyncio
    async def test_get_unknown_type(self, handler):
        """Test that getting unknown integration type returns proper error code."""
        result = await handler._get_integration("unknown_type", None, None)

        assert result.status_code == 400
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "UNKNOWN_INTEGRATION_TYPE"


class TestSlackIntegrationValidation:
    """Tests for Slack integration validation."""

    @pytest.mark.asyncio
    async def test_slack_workspace_not_found(self, handler):
        """Test that missing Slack workspace returns proper error code."""
        with patch.object(handler, "_get_slack_store") as mock_store:
            mock_store.return_value.get.return_value = None
            result = await handler._get_integration("slack", "nonexistent", None)

        assert result.status_code == 404
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "SLACK_WORKSPACE_NOT_FOUND"


class TestTeamsIntegrationValidation:
    """Tests for Teams integration validation."""

    @pytest.mark.asyncio
    async def test_teams_tenant_not_found(self, handler):
        """Test that missing Teams tenant returns proper error code."""
        with patch.object(handler, "_get_teams_store") as mock_store:
            mock_store.return_value.get.return_value = None
            result = await handler._get_integration("teams", "nonexistent", None)

        assert result.status_code == 404
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "TEAMS_TENANT_NOT_FOUND"


class TestDisconnectIntegration:
    """Tests for disconnect integration endpoint."""

    @pytest.mark.asyncio
    async def test_disconnect_missing_workspace_id(self, handler):
        """Test that disconnect without workspace_id returns proper error code."""
        result = await handler._disconnect_integration("slack", None, None)

        assert result.status_code == 400
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "MISSING_WORKSPACE_ID"

    @pytest.mark.asyncio
    async def test_disconnect_slack_not_found(self, handler):
        """Test that disconnecting non-existent Slack workspace returns proper error code."""
        with patch.object(handler, "_get_slack_store") as mock_store:
            mock_store.return_value.get.return_value = None
            result = await handler._disconnect_integration("slack", "nonexistent", None)

        assert result.status_code == 404
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "SLACK_WORKSPACE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_disconnect_slack_failed(self, handler):
        """Test that failed Slack disconnect returns proper error code."""
        mock_workspace = MagicMock()
        mock_workspace.workspace_name = "Test Workspace"

        with patch.object(handler, "_get_slack_store") as mock_store:
            mock_store.return_value.get.return_value = mock_workspace
            mock_store.return_value.deactivate.return_value = False
            result = await handler._disconnect_integration("slack", "ws123", None)

        assert result.status_code == 500
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "DISCONNECT_FAILED"

    @pytest.mark.asyncio
    async def test_disconnect_unsupported(self, handler):
        """Test that disconnecting unsupported integration returns proper error code."""
        result = await handler._disconnect_integration("discord", "id123", None)

        assert result.status_code == 400
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "UNSUPPORTED_DISCONNECT"


class TestTestIntegration:
    """Tests for test integration endpoint."""

    @pytest.mark.asyncio
    async def test_slack_missing_workspace_id(self, handler):
        """Test that testing Slack without workspace_id returns proper error code."""
        result = await handler._test_integration("slack", None, None)

        assert result.status_code == 400
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "MISSING_WORKSPACE_ID"

    @pytest.mark.asyncio
    async def test_slack_workspace_not_found(self, handler):
        """Test that testing non-existent Slack workspace returns proper error code."""
        with patch.object(handler, "_get_slack_store") as mock_store:
            mock_store.return_value.get.return_value = None
            result = await handler._test_integration("slack", "nonexistent", None)

        assert result.status_code == 404
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "SLACK_WORKSPACE_NOT_FOUND"

    @pytest.mark.asyncio
    async def test_teams_missing_workspace_id(self, handler):
        """Test that testing Teams without workspace_id returns proper error code."""
        result = await handler._test_integration("teams", None, None)

        assert result.status_code == 400
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "MISSING_WORKSPACE_ID"

    @pytest.mark.asyncio
    async def test_unsupported_test(self, handler):
        """Test that testing unsupported integration returns proper error code."""
        result = await handler._test_integration("unknown", "id123", None)

        assert result.status_code == 400
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "UNSUPPORTED_TEST"


class TestHealthEndpoint:
    """Tests for health endpoint."""

    @pytest.mark.asyncio
    async def test_health_unknown_integration_type(self, handler):
        """Test that health for unknown integration type returns proper error code."""
        result = await handler._get_health("unknown", None, None)

        assert result.status_code == 400
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "UNKNOWN_INTEGRATION_TYPE"

    @pytest.mark.asyncio
    async def test_health_slack_workspace_not_found(self, handler):
        """Test that health for non-existent Slack workspace returns proper error code."""
        with patch.object(handler, "_get_slack_store") as mock_store:
            mock_store.return_value.get.return_value = None
            result = await handler._get_health("slack", "nonexistent", None)

        assert result.status_code == 404
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "SLACK_WORKSPACE_NOT_FOUND"


class TestInternalError:
    """Tests for internal error handling."""

    @pytest.mark.asyncio
    async def test_internal_error(self, handler):
        """Test that internal error returns proper error code."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"
        mock_handler.headers = {}

        with patch.object(handler, "_get_slack_store") as mock_store:
            mock_store.side_effect = RuntimeError("Database error")
            result = await handler.handle(
                "/api/v2/integrations/slack/health",
                {"workspace_id": "ws123"},
                mock_handler,
            )

        assert result.status_code == 500
        body = parse_body(result)
        _, code = get_error_fields(body)
        assert code == "INTERNAL_ERROR"
