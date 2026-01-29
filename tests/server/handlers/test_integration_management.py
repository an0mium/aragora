"""
Tests for IntegrationsHandler in aragora.server.handlers.integration_management.

Covers all REST endpoints:
    GET    /api/v2/integrations            - List all integrations
    GET    /api/v2/integrations/:type       - Get specific integration
    DELETE /api/v2/integrations/:type       - Disconnect integration
    POST   /api/v2/integrations/:type/test  - Test connectivity
    GET    /api/v2/integrations/:type/health - Health check
    GET    /api/v2/integrations/stats       - Statistics
"""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.integration_management import (
    SUPPORTED_INTEGRATIONS,
    IntegrationsHandler,
    create_integrations_handler,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_slack_workspace(
    workspace_id="ws-slack-001",
    workspace_name="Test Slack",
    tenant_id="tenant-001",
    is_active=True,
    installed_at="2026-01-10T00:00:00Z",
    installed_by="user-001",
    scopes="chat:write,channels:read",
    refresh_token="refresh-tok",
    token_expires_at="2026-02-10T00:00:00Z",
    access_token="xoxb-test-token",
):
    ws = MagicMock()
    ws.workspace_id = workspace_id
    ws.workspace_name = workspace_name
    ws.tenant_id = tenant_id
    ws.is_active = is_active
    ws.installed_at = installed_at
    ws.installed_by = installed_by
    ws.scopes = scopes
    ws.refresh_token = refresh_token
    ws.token_expires_at = token_expires_at
    ws.access_token = access_token
    ws.to_dict.return_value = {
        "workspace_id": workspace_id,
        "workspace_name": workspace_name,
        "is_active": is_active,
    }
    return ws


def _make_teams_workspace(
    tenant_id="tenant-teams-001",
    tenant_name="Test Teams Org",
    is_active=True,
    installed_at="2026-01-12T00:00:00Z",
    installed_by="user-002",
    scopes="User.Read,Team.ReadBasic.All",
    refresh_token="teams-refresh",
    token_expires_at="2026-02-12T00:00:00Z",
    access_token="eyJ-teams-token",
):
    ws = MagicMock()
    ws.workspace_id = tenant_id
    ws.tenant_id = tenant_id
    ws.tenant_name = tenant_name
    ws.is_active = is_active
    ws.installed_at = installed_at
    ws.installed_by = installed_by
    ws.scopes = scopes
    ws.refresh_token = refresh_token
    ws.token_expires_at = token_expires_at
    ws.access_token = access_token
    ws.to_dict.return_value = {
        "tenant_id": tenant_id,
        "tenant_name": tenant_name,
        "is_active": is_active,
    }
    return ws


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def handler(mock_server_context):
    h = IntegrationsHandler(server_context=mock_server_context)
    # Pre-inject mock stores so the handler never calls the real lazy loaders
    h._slack_store = MagicMock()
    h._teams_store = MagicMock()
    return h


# ============================================================================
# Init and Routing
# ============================================================================


class TestInitAndRouting:
    def test_can_handle_integrations_root(self, handler):
        assert handler.can_handle("/api/v2/integrations", "GET") is True

    def test_can_handle_integration_type(self, handler):
        assert handler.can_handle("/api/v2/integrations/slack", "GET") is True

    def test_can_handle_delete(self, handler):
        assert handler.can_handle("/api/v2/integrations/slack", "DELETE") is True

    def test_can_handle_post(self, handler):
        assert handler.can_handle("/api/v2/integrations/slack/test", "POST") is True

    def test_cannot_handle_put(self, handler):
        assert handler.can_handle("/api/v2/integrations/slack", "PUT") is False

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v2/debates", "GET") is False

    def test_routes_defined(self, handler):
        assert "/api/v2/integrations" in handler.ROUTES
        assert "/api/v2/integrations/*" in handler.ROUTES

    def test_supported_integrations_set(self):
        assert SUPPORTED_INTEGRATIONS == {"slack", "teams", "discord", "email"}


# ============================================================================
# Factory Function
# ============================================================================


class TestFactoryFunction:
    def test_create_integrations_handler(self, mock_server_context):
        h = create_integrations_handler(mock_server_context)
        assert isinstance(h, IntegrationsHandler)


# ============================================================================
# List Integrations
# ============================================================================


class TestListIntegrations:
    @pytest.mark.asyncio
    async def test_list_all_integrations(self, handler):
        slack_ws = _make_slack_workspace()
        teams_ws = _make_teams_workspace()
        handler._slack_store.list_active.return_value = [slack_ws]
        handler._teams_store.list_active.return_value = [teams_ws]

        result = await handler.handle("GET", "/api/v2/integrations")
        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["integrations"]) == 2
        types = {i["type"] for i in data["integrations"]}
        assert types == {"slack", "teams"}
        assert data["pagination"]["total"] == 2

    @pytest.mark.asyncio
    async def test_list_with_type_filter_slack(self, handler):
        slack_ws = _make_slack_workspace()
        handler._slack_store.list_active.return_value = [slack_ws]

        result = await handler.handle("GET", "/api/v2/integrations", query_params={"type": "slack"})
        data = json.loads(result.body)
        assert all(i["type"] == "slack" for i in data["integrations"])

    @pytest.mark.asyncio
    async def test_list_with_status_filter_active(self, handler):
        active_ws = _make_slack_workspace(is_active=True)
        inactive_ws = _make_slack_workspace(workspace_id="ws-slack-002", is_active=False)
        handler._slack_store.list_active.return_value = [active_ws, inactive_ws]
        handler._teams_store.list_active.return_value = []

        result = await handler.handle(
            "GET", "/api/v2/integrations", query_params={"status": "active"}
        )
        data = json.loads(result.body)
        assert all(i["status"] == "active" for i in data["integrations"])

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, handler):
        workspaces = [_make_slack_workspace(workspace_id=f"ws-{i}") for i in range(5)]
        handler._slack_store.list_active.return_value = workspaces
        handler._teams_store.list_active.return_value = []

        result = await handler.handle(
            "GET", "/api/v2/integrations", query_params={"limit": "2", "offset": "0"}
        )
        data = json.loads(result.body)
        assert len(data["integrations"]) == 2
        assert data["pagination"]["has_more"] is True
        assert data["pagination"]["total"] == 5

    @pytest.mark.asyncio
    async def test_list_with_tenant_id(self, handler):
        slack_ws = _make_slack_workspace()
        handler._slack_store.get_by_tenant.return_value = [slack_ws]
        handler._teams_store.get_by_aragora_tenant.return_value = []

        result = await handler.handle(
            "GET",
            "/api/v2/integrations",
            headers={"X-Tenant-ID": "tenant-001"},
        )
        data = json.loads(result.body)
        handler._slack_store.get_by_tenant.assert_called_once_with("tenant-001")
        assert result.status_code == 200


# ============================================================================
# Get Specific Integration
# ============================================================================


class TestGetIntegration:
    @pytest.mark.asyncio
    async def test_get_slack_by_workspace_id(self, handler):
        ws = _make_slack_workspace()
        handler._slack_store.get.return_value = ws

        with patch.object(handler, "_check_slack_health", return_value={"status": "healthy"}):
            result = await handler.handle(
                "GET",
                "/api/v2/integrations/slack",
                query_params={"workspace_id": "ws-slack-001"},
            )
        data = json.loads(result.body)
        assert data["type"] == "slack"
        assert data["connected"] is True
        assert "workspace" in data

    @pytest.mark.asyncio
    async def test_get_slack_not_found(self, handler):
        handler._slack_store.get.return_value = None
        result = await handler.handle(
            "GET",
            "/api/v2/integrations/slack",
            query_params={"workspace_id": "nonexistent"},
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_slack_list_all(self, handler):
        ws = _make_slack_workspace()
        handler._slack_store.list_active.return_value = [ws]

        result = await handler.handle("GET", "/api/v2/integrations/slack")
        data = json.loads(result.body)
        assert data["type"] == "slack"
        assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_get_teams_by_workspace_id(self, handler):
        ws = _make_teams_workspace()
        handler._teams_store.get.return_value = ws

        with patch.object(handler, "_check_teams_health", return_value={"status": "healthy"}):
            result = await handler.handle(
                "GET",
                "/api/v2/integrations/teams",
                query_params={"workspace_id": "tenant-teams-001"},
            )
        data = json.loads(result.body)
        assert data["type"] == "teams"
        assert data["connected"] is True

    @pytest.mark.asyncio
    async def test_get_teams_not_found(self, handler):
        handler._teams_store.get.return_value = None
        result = await handler.handle(
            "GET",
            "/api/v2/integrations/teams",
            query_params={"workspace_id": "nonexistent"},
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_discord_configured(self, handler):
        with patch.dict("os.environ", {"DISCORD_BOT_TOKEN": "bot-token-123"}):
            result = await handler.handle("GET", "/api/v2/integrations/discord")
        data = json.loads(result.body)
        assert data["type"] == "discord"
        assert data["connected"] is True

    @pytest.mark.asyncio
    async def test_get_discord_not_configured(self, handler):
        with patch.dict("os.environ", {}, clear=True):
            result = await handler.handle("GET", "/api/v2/integrations/discord")
        data = json.loads(result.body)
        assert data["connected"] is False

    @pytest.mark.asyncio
    async def test_get_email_configured(self, handler):
        with patch.dict("os.environ", {"SMTP_HOST": "smtp.example.com"}):
            result = await handler.handle("GET", "/api/v2/integrations/email")
        data = json.loads(result.body)
        assert data["type"] == "email"
        assert data["connected"] is True
        assert data["smtp_host"] == "smtp.example.com"

    @pytest.mark.asyncio
    async def test_get_email_not_configured(self, handler):
        with patch.dict("os.environ", {}, clear=True):
            result = await handler.handle("GET", "/api/v2/integrations/email")
        data = json.loads(result.body)
        assert data["connected"] is False


# ============================================================================
# Disconnect Integration
# ============================================================================


class TestDisconnectIntegration:
    @pytest.mark.asyncio
    async def test_disconnect_slack_success(self, handler):
        ws = _make_slack_workspace()
        handler._slack_store.get.return_value = ws
        handler._slack_store.deactivate.return_value = True

        result = await handler.handle(
            "DELETE",
            "/api/v2/integrations/slack",
            body={"workspace_id": "ws-slack-001"},
        )
        data = json.loads(result.body)
        assert result.status_code == 200
        assert data["disconnected"] is True
        assert data["type"] == "slack"

    @pytest.mark.asyncio
    async def test_disconnect_slack_not_found(self, handler):
        handler._slack_store.get.return_value = None

        result = await handler.handle(
            "DELETE",
            "/api/v2/integrations/slack",
            body={"workspace_id": "nonexistent"},
        )
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_disconnect_missing_workspace_id(self, handler):
        result = await handler.handle("DELETE", "/api/v2/integrations/slack", body={})
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "workspace_id" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_disconnect_teams_success(self, handler):
        ws = _make_teams_workspace()
        handler._teams_store.get.return_value = ws
        handler._teams_store.deactivate.return_value = True

        result = await handler.handle(
            "DELETE",
            "/api/v2/integrations/teams",
            body={"workspace_id": "tenant-teams-001"},
        )
        data = json.loads(result.body)
        assert data["disconnected"] is True
        assert data["type"] == "teams"

    @pytest.mark.asyncio
    async def test_disconnect_slack_deactivate_fails(self, handler):
        ws = _make_slack_workspace()
        handler._slack_store.get.return_value = ws
        handler._slack_store.deactivate.return_value = False

        result = await handler.handle(
            "DELETE",
            "/api/v2/integrations/slack",
            body={"workspace_id": "ws-slack-001"},
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_disconnect_unsupported_type(self, handler):
        result = await handler.handle(
            "DELETE",
            "/api/v2/integrations/discord",
            body={"workspace_id": "some-id"},
        )
        assert result.status_code == 400


# ============================================================================
# Test Integration Connectivity
# ============================================================================


class TestTestIntegration:
    @pytest.mark.asyncio
    async def test_slack_connectivity(self, handler):
        ws = _make_slack_workspace()
        handler._slack_store.get.return_value = ws

        with patch.object(handler, "_check_slack_health", return_value={"status": "healthy"}):
            result = await handler.handle(
                "POST",
                "/api/v2/integrations/slack/test",
                body={"workspace_id": "ws-slack-001"},
            )
        data = json.loads(result.body)
        assert data["type"] == "slack"
        assert data["test_result"]["status"] == "healthy"
        assert "tested_at" in data

    @pytest.mark.asyncio
    async def test_slack_missing_workspace_id(self, handler):
        result = await handler.handle("POST", "/api/v2/integrations/slack/test", body={})
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_teams_connectivity(self, handler):
        ws = _make_teams_workspace()
        handler._teams_store.get.return_value = ws

        with patch.object(handler, "_check_teams_health", return_value={"status": "healthy"}):
            result = await handler.handle(
                "POST",
                "/api/v2/integrations/teams/test",
                body={"workspace_id": "tenant-teams-001"},
            )
        data = json.loads(result.body)
        assert data["type"] == "teams"
        assert data["test_result"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_discord_connectivity(self, handler):
        with patch.object(
            handler,
            "_check_discord_health",
            return_value={"status": "healthy", "bot_name": "ArgoraBot"},
        ):
            result = await handler.handle("POST", "/api/v2/integrations/discord/test")
        data = json.loads(result.body)
        assert data["type"] == "discord"
        assert data["test_result"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_email_connectivity(self, handler):
        with patch.object(
            handler,
            "_check_email_health",
            return_value={"status": "healthy", "smtp_host": "smtp.example.com"},
        ):
            result = await handler.handle("POST", "/api/v2/integrations/email/test")
        data = json.loads(result.body)
        assert data["type"] == "email"
        assert data["test_result"]["status"] == "healthy"


# ============================================================================
# Health Check
# ============================================================================


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_slack_aggregate_health_all_healthy(self, handler):
        ws1 = _make_slack_workspace(workspace_id="ws-1")
        ws2 = _make_slack_workspace(workspace_id="ws-2")
        handler._slack_store.list_active.return_value = [ws1, ws2]

        with patch.object(handler, "_check_slack_health", return_value={"status": "healthy"}):
            result = await handler.handle("GET", "/api/v2/integrations/slack/health")
        data = json.loads(result.body)
        assert data["type"] == "slack"
        assert data["healthy"] is True
        assert data["status"] == "healthy"
        assert len(data["workspaces"]) == 2

    @pytest.mark.asyncio
    async def test_slack_aggregate_health_degraded(self, handler):
        ws1 = _make_slack_workspace(workspace_id="ws-1")
        ws2 = _make_slack_workspace(workspace_id="ws-2")
        handler._slack_store.list_active.return_value = [ws1, ws2]

        health_responses = [{"status": "healthy"}, {"status": "unhealthy"}]
        call_count = 0

        async def mock_check(ws):
            nonlocal call_count
            resp = health_responses[call_count]
            call_count += 1
            return resp

        with patch.object(handler, "_check_slack_health", side_effect=mock_check):
            result = await handler.handle("GET", "/api/v2/integrations/slack/health")
        data = json.loads(result.body)
        assert data["status"] == "degraded"
        assert data["healthy"] is False

    @pytest.mark.asyncio
    async def test_slack_no_workspaces(self, handler):
        handler._slack_store.list_active.return_value = []
        result = await handler.handle("GET", "/api/v2/integrations/slack/health")
        data = json.loads(result.body)
        assert data["status"] == "not_configured"
        assert data["healthy"] is False

    @pytest.mark.asyncio
    async def test_slack_specific_workspace_health(self, handler):
        ws = _make_slack_workspace()
        handler._slack_store.get.return_value = ws

        with patch.object(handler, "_check_slack_health", return_value={"status": "healthy"}):
            result = await handler.handle(
                "GET",
                "/api/v2/integrations/slack/health",
                query_params={"workspace_id": "ws-slack-001"},
            )
        data = json.loads(result.body)
        assert data["workspace_id"] == "ws-slack-001"
        assert data["healthy"] is True

    @pytest.mark.asyncio
    async def test_teams_health_no_workspaces(self, handler):
        handler._teams_store.list_active.return_value = []
        result = await handler.handle("GET", "/api/v2/integrations/teams/health")
        data = json.loads(result.body)
        assert data["status"] == "not_configured"

    @pytest.mark.asyncio
    async def test_teams_specific_workspace_health(self, handler):
        ws = _make_teams_workspace()
        handler._teams_store.get.return_value = ws

        with patch.object(handler, "_check_teams_health", return_value={"status": "healthy"}):
            result = await handler.handle(
                "GET",
                "/api/v2/integrations/teams/health",
                query_params={"workspace_id": "tenant-teams-001"},
            )
        data = json.loads(result.body)
        assert data["tenant_id"] == "tenant-teams-001"
        assert data["healthy"] is True

    @pytest.mark.asyncio
    async def test_discord_health(self, handler):
        with patch.object(
            handler,
            "_check_discord_health",
            return_value={"status": "healthy", "bot_name": "ArgoraBot"},
        ):
            result = await handler.handle("GET", "/api/v2/integrations/discord/health")
        data = json.loads(result.body)
        assert data["type"] == "discord"
        assert data["healthy"] is True

    @pytest.mark.asyncio
    async def test_email_health(self, handler):
        with patch.object(
            handler,
            "_check_email_health",
            return_value={"status": "healthy", "smtp_host": "smtp.example.com"},
        ):
            result = await handler.handle("GET", "/api/v2/integrations/email/health")
        data = json.loads(result.body)
        assert data["type"] == "email"
        assert data["healthy"] is True


# ============================================================================
# Stats
# ============================================================================


class TestStats:
    @pytest.mark.asyncio
    async def test_get_stats(self, handler):
        handler._slack_store.get_stats.return_value = {
            "active_workspaces": 3,
            "total_workspaces": 5,
        }
        handler._teams_store.get_stats.return_value = {
            "active_workspaces": 2,
            "total_workspaces": 4,
        }

        result = await handler.handle("GET", "/api/v2/integrations/stats")
        data = json.loads(result.body)
        assert result.status_code == 200
        assert data["stats"]["slack"]["active_workspaces"] == 3
        assert data["stats"]["teams"]["active_workspaces"] == 2
        assert data["stats"]["total_integrations"] == 5
        assert "generated_at" in data


# ============================================================================
# Error Handling
# ============================================================================


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_integration_type(self, handler):
        result = await handler.handle("GET", "/api/v2/integrations/jira")
        assert result.status_code == 400
        data = json.loads(result.body)
        assert (
            "Unknown integration" in data.get("error", "")
            or "jira" in data.get("error", "").lower()
        )

    @pytest.mark.asyncio
    async def test_invalid_path_too_short(self, handler):
        # Path that starts with /api/v2/integrations/ but splits to < 5 parts
        # The path "/api/v2/integrations/" splits to ['', 'api', 'v2', 'integrations', '']
        # which has 5 parts with empty last. Let's use the root path which is handled differently.
        # The 404 is returned when no route matches
        result = await handler.handle("GET", "/api/v2/integrations/unknown/unknown/unknown")
        # "unknown" is not in SUPPORTED_INTEGRATIONS
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_internal_error_is_caught(self, handler):
        handler._slack_store.list_active.side_effect = RuntimeError("DB connection lost")
        handler._teams_store.list_active.return_value = []

        result = await handler.handle("GET", "/api/v2/integrations")
        assert result.status_code == 500
        data = json.loads(result.body)
        assert "Internal error" in data.get("error", "") or "DB connection" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_not_found_for_unmatched_route(self, handler):
        # A path that doesn't match any route pattern in handle()
        # The base path with a non-matching method scenario is tricky;
        # but handle returns 404 when nothing matches at the end
        result = await handler.handle("POST", "/api/v2/integrations")
        assert result.status_code == 404
