"""
Tests for OAuthWizardHandler - unified OAuth wizard for SME onboarding.

Covers:
- GET /api/v2/integrations/wizard - Get wizard configuration
- GET /api/v2/integrations/wizard/providers - List all providers
- GET /api/v2/integrations/wizard/status - Get integration status
- POST /api/v2/integrations/wizard/validate - Validate configuration
- POST /api/v2/integrations/wizard/{provider}/test - Test connection
- GET /api/v2/integrations/wizard/{provider}/workspaces - List workspaces
- POST /api/v2/integrations/wizard/{provider}/disconnect - Disconnect
- RBAC permission enforcement
"""

from __future__ import annotations

import json
import os
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.oauth_wizard import (
    OAuthWizardHandler,
    PROVIDERS,
    create_oauth_wizard_handler,
)
from aragora.rbac.models import AuthorizationContext


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def server_context():
    """Create a mock server context."""
    return {"config": {"debug": True}}


@pytest.fixture
def auth_context():
    """Create a test authorization context with connector permissions."""
    return AuthorizationContext(
        user_id="user-123",
        org_id="org-456",
        roles={"admin"},
        permissions={"connector:read", "connector:create", "connector:delete"},
    )


@pytest.fixture
def read_only_context():
    """Create a test authorization context with read-only permissions."""
    return AuthorizationContext(
        user_id="user-readonly",
        org_id="org-456",
        roles={"viewer"},
        permissions={"connector:read"},
    )


@pytest.fixture
def no_permission_context():
    """Create a test authorization context with no connector permissions."""
    return AuthorizationContext(
        user_id="user-none",
        org_id="org-456",
        roles={"guest"},
        permissions=set(),
    )


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = "GET"
    handler.headers = {"Authorization": "Bearer test-token"}
    return handler


@pytest.fixture
def oauth_handler(server_context):
    """Create an OAuthWizardHandler instance."""
    return OAuthWizardHandler(server_context)


# -----------------------------------------------------------------------------
# Initialization Tests
# -----------------------------------------------------------------------------


class TestOAuthWizardHandlerInit:
    """Tests for OAuthWizardHandler initialization."""

    def test_init_with_server_context(self, server_context):
        """Handler initializes with server context."""
        handler = OAuthWizardHandler(server_context)
        assert handler.ctx == server_context

    def test_resource_type(self, oauth_handler):
        """Handler has correct resource type."""
        assert oauth_handler.RESOURCE_TYPE == "connector"

    def test_routes(self, oauth_handler):
        """Handler has correct routes."""
        assert "/api/v2/integrations/wizard" in oauth_handler.ROUTES
        assert "/api/v2/integrations/wizard/*" in oauth_handler.ROUTES

    def test_can_handle_wizard_path(self, oauth_handler):
        """can_handle returns True for wizard paths."""
        assert oauth_handler.can_handle("/api/v2/integrations/wizard") is True
        assert oauth_handler.can_handle("/api/v2/integrations/wizard/providers") is True
        assert oauth_handler.can_handle("/api/v2/integrations/wizard/status") is True
        assert oauth_handler.can_handle("/api/v2/integrations/wizard/slack/test") is True

    def test_can_handle_non_wizard_path(self, oauth_handler):
        """can_handle returns False for non-wizard paths."""
        assert oauth_handler.can_handle("/api/v2/integrations") is False
        assert oauth_handler.can_handle("/api/v2/debates") is False


class TestFactoryFunction:
    """Tests for create_oauth_wizard_handler factory function."""

    def test_create_handler(self, server_context):
        """Factory creates handler instance."""
        handler = create_oauth_wizard_handler(server_context)
        assert isinstance(handler, OAuthWizardHandler)
        assert handler.ctx == server_context


# -----------------------------------------------------------------------------
# RBAC Permission Tests
# -----------------------------------------------------------------------------


class TestRBACPermissions:
    """Tests for RBAC permission enforcement."""

    @pytest.mark.asyncio
    async def test_unauthenticated_request_returns_401(self, oauth_handler, mock_handler):
        """Unauthenticated requests return 401."""
        with patch.object(oauth_handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            from aragora.server.handlers.secure import UnauthorizedError

            mock_auth.side_effect = UnauthorizedError("Token required")

            result = await oauth_handler.handle(
                "/api/v2/integrations/wizard",
                {},
                mock_handler,
            )

            assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_permission_denied_returns_403(
        self, oauth_handler, mock_handler, no_permission_context
    ):
        """Requests without required permission return 403."""
        with patch.object(
            oauth_handler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=no_permission_context,
        ):
            with patch.object(
                oauth_handler, "check_permission", side_effect=Exception("Permission denied")
            ):
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard",
                    {},
                    mock_handler,
                )

                assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_read_permission_allows_get(self, oauth_handler, mock_handler, read_only_context):
        """Read-only context can access GET endpoints."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=read_only_context,
        ):
            with patch.object(oauth_handler, "check_permission") as mock_check:
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard",
                    {},
                    mock_handler,
                )

                mock_check.assert_called_with(read_only_context, "connector:read")
                assert result.status_code == 200


# -----------------------------------------------------------------------------
# GET /api/v2/integrations/wizard Tests
# -----------------------------------------------------------------------------


class TestGetWizardConfig:
    """Tests for GET /api/v2/integrations/wizard endpoint."""

    @pytest.mark.asyncio
    async def test_get_wizard_config_success(self, oauth_handler, mock_handler, auth_context):
        """Get wizard config returns provider information."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard",
                    {},
                    mock_handler,
                )

                assert result.status_code == 200
                data = json.loads(result.body)
                assert "wizard" in data
                assert "providers" in data["wizard"]
                assert "summary" in data["wizard"]
                assert "generated_at" in data

    @pytest.mark.asyncio
    async def test_wizard_config_contains_all_providers(
        self, oauth_handler, mock_handler, auth_context
    ):
        """Wizard config contains all defined providers."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard",
                    {},
                    mock_handler,
                )

                data = json.loads(result.body)
                provider_ids = [p["id"] for p in data["wizard"]["providers"]]
                for expected_id in PROVIDERS.keys():
                    assert expected_id in provider_ids

    @pytest.mark.asyncio
    async def test_wizard_config_has_recommended_order(
        self, oauth_handler, mock_handler, auth_context
    ):
        """Wizard config includes recommended setup order."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard",
                    {},
                    mock_handler,
                )

                data = json.loads(result.body)
                assert "recommended_order" in data["wizard"]
                assert "slack" in data["wizard"]["recommended_order"]


# -----------------------------------------------------------------------------
# GET /api/v2/integrations/wizard/providers Tests
# -----------------------------------------------------------------------------


class TestListProviders:
    """Tests for GET /api/v2/integrations/wizard/providers endpoint."""

    @pytest.mark.asyncio
    async def test_list_providers_returns_all(self, oauth_handler, mock_handler, auth_context):
        """List providers returns all providers."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard/providers",
                    {},
                    mock_handler,
                )

                assert result.status_code == 200
                data = json.loads(result.body)
                assert "providers" in data
                assert "total" in data
                assert data["total"] == len(PROVIDERS)

    @pytest.mark.asyncio
    async def test_list_providers_filter_by_category(
        self, oauth_handler, mock_handler, auth_context
    ):
        """List providers can filter by category."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard/providers",
                    {"category": "communication"},
                    mock_handler,
                )

                assert result.status_code == 200
                data = json.loads(result.body)
                for provider in data["providers"]:
                    assert provider["category"] == "communication"

    @pytest.mark.asyncio
    async def test_list_providers_filter_by_configured(
        self, oauth_handler, mock_handler, auth_context
    ):
        """List providers can filter by configured status."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard/providers",
                    {"configured": "false"},
                    mock_handler,
                )

                assert result.status_code == 200
                data = json.loads(result.body)
                for provider in data["providers"]:
                    assert provider["configured"] is False


# -----------------------------------------------------------------------------
# GET /api/v2/integrations/wizard/status Tests
# -----------------------------------------------------------------------------


class TestGetStatus:
    """Tests for GET /api/v2/integrations/wizard/status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_returns_all_providers(
        self, oauth_handler, mock_handler, auth_context
    ):
        """Get status returns status for all providers."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(
                    oauth_handler, "_check_connection", new_callable=AsyncMock, return_value=None
                ):
                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/status",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "statuses" in data
                    assert "summary" in data
                    assert len(data["statuses"]) == len(PROVIDERS)

    @pytest.mark.asyncio
    async def test_status_includes_summary(self, oauth_handler, mock_handler, auth_context):
        """Status response includes summary counts."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(
                    oauth_handler, "_check_connection", new_callable=AsyncMock, return_value=None
                ):
                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/status",
                        {},
                        mock_handler,
                    )

                    data = json.loads(result.body)
                    summary = data["summary"]
                    assert "total" in summary
                    assert "configured" in summary
                    assert "connected" in summary
                    assert "needs_attention" in summary


# -----------------------------------------------------------------------------
# POST /api/v2/integrations/wizard/validate Tests
# -----------------------------------------------------------------------------


class TestValidateConfig:
    """Tests for POST /api/v2/integrations/wizard/validate endpoint."""

    @pytest.mark.asyncio
    async def test_validate_missing_provider_returns_400(
        self, oauth_handler, mock_handler, auth_context
    ):
        """Validate without provider returns 400."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(oauth_handler, "read_json_body", return_value={}):
                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/validate",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_validate_unknown_provider_returns_400(
        self, oauth_handler, mock_handler, auth_context
    ):
        """Validate with unknown provider returns 400."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"provider": "unknown_provider"}
                ):
                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/validate",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_validate_slack_missing_env_vars(self, oauth_handler, mock_handler, auth_context):
        """Validate Slack without env vars returns invalid."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"provider": "slack"}
                ):
                    with patch.dict(os.environ, {}, clear=True):
                        result = await oauth_handler.handle(
                            "/api/v2/integrations/wizard/validate",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["valid"] is False
                        assert len(data["checks"]) > 0

    @pytest.mark.asyncio
    async def test_validate_slack_with_env_vars(self, oauth_handler, mock_handler, auth_context):
        """Validate Slack with env vars returns valid."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"provider": "slack"}
                ):
                    with patch.dict(
                        os.environ,
                        {
                            "SLACK_CLIENT_ID": "test-client-id",
                            "SLACK_CLIENT_SECRET": "test-client-secret",
                        },
                    ):
                        result = await oauth_handler.handle(
                            "/api/v2/integrations/wizard/validate",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["valid"] is True


# -----------------------------------------------------------------------------
# POST /api/v2/integrations/wizard/{provider}/test Tests
# -----------------------------------------------------------------------------


class TestTestConnection:
    """Tests for POST /api/v2/integrations/wizard/{provider}/test endpoint."""

    @pytest.mark.asyncio
    async def test_test_unknown_provider_returns_404(
        self, oauth_handler, mock_handler, auth_context
    ):
        """Test unknown provider returns 404."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(oauth_handler, "read_json_body", return_value={}):
                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/unknown/test",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_test_slack_no_workspaces(self, oauth_handler, mock_handler, auth_context):
        """Test Slack with no workspaces returns failure."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(oauth_handler, "read_json_body", return_value={}):
                    with patch.object(
                        oauth_handler,
                        "_test_slack_api",
                        new_callable=AsyncMock,
                        return_value={"success": False, "error": "No active workspaces"},
                    ):
                        result = await oauth_handler.handle(
                            "/api/v2/integrations/wizard/slack/test",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["test_result"]["success"] is False

    @pytest.mark.asyncio
    async def test_test_slack_success(self, oauth_handler, mock_handler, auth_context):
        """Test Slack with active workspace returns success."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(oauth_handler, "read_json_body", return_value={}):
                    with patch.object(
                        oauth_handler,
                        "_test_slack_api",
                        new_callable=AsyncMock,
                        return_value={"success": True, "team": "Test Team", "team_id": "T123"},
                    ):
                        result = await oauth_handler.handle(
                            "/api/v2/integrations/wizard/slack/test",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["test_result"]["success"] is True
                        assert data["provider"] == "slack"


# -----------------------------------------------------------------------------
# GET /api/v2/integrations/wizard/{provider}/workspaces Tests
# -----------------------------------------------------------------------------


class TestListWorkspaces:
    """Tests for GET /api/v2/integrations/wizard/{provider}/workspaces endpoint."""

    @pytest.mark.asyncio
    async def test_list_workspaces_unknown_provider_returns_404(
        self, oauth_handler, mock_handler, auth_context
    ):
        """List workspaces for unknown provider returns 404."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard/unknown/workspaces",
                    {},
                    mock_handler,
                )

                assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_list_slack_workspaces(self, oauth_handler, mock_handler, auth_context):
        """List Slack workspaces returns workspace list."""
        mock_handler.command = "GET"

        mock_workspaces = [
            {
                "workspace_id": "W123",
                "name": "Test Workspace",
                "is_active": True,
                "created_at": "2024-01-01",
            },
        ]

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch(
                    "aragora.storage.slack_workspace_store.get_slack_workspace_store"
                ) as mock_store:
                    mock_store.return_value.list_active.return_value = mock_workspaces

                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/slack/workspaces",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["provider"] == "slack"
                    assert data["count"] == 1

    @pytest.mark.asyncio
    async def test_list_teams_tenants(self, oauth_handler, mock_handler, auth_context):
        """List Teams tenants returns tenant list."""
        mock_handler.command = "GET"

        mock_tenants = [
            {
                "tenant_id": "T123",
                "name": "Test Tenant",
                "is_active": True,
                "created_at": "2024-01-01",
            },
        ]

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch(
                    "aragora.storage.teams_tenant_store.get_teams_tenant_store"
                ) as mock_store:
                    mock_store.return_value.list_active.return_value = mock_tenants

                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/teams/workspaces",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert data["provider"] == "teams"
                    assert data["count"] == 1


# -----------------------------------------------------------------------------
# POST /api/v2/integrations/wizard/{provider}/disconnect Tests
# -----------------------------------------------------------------------------


class TestDisconnectProvider:
    """Tests for POST /api/v2/integrations/wizard/{provider}/disconnect endpoint."""

    @pytest.mark.asyncio
    async def test_disconnect_unknown_provider_returns_404(
        self, oauth_handler, mock_handler, auth_context
    ):
        """Disconnect unknown provider returns 404."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(oauth_handler, "read_json_body", return_value={}):
                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/unknown/disconnect",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_disconnect_slack_missing_workspace_id_returns_400(
        self, oauth_handler, mock_handler, auth_context
    ):
        """Disconnect Slack without workspace_id returns 400."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(oauth_handler, "read_json_body", return_value={}):
                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/slack/disconnect",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_disconnect_slack_success(self, oauth_handler, mock_handler, auth_context):
        """Disconnect Slack workspace succeeds."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"workspace_id": "W123"}
                ):
                    with patch(
                        "aragora.storage.slack_workspace_store.get_slack_workspace_store"
                    ) as mock_store:
                        result = await oauth_handler.handle(
                            "/api/v2/integrations/wizard/slack/disconnect",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["disconnected"] is True
                        mock_store.return_value.deactivate.assert_called_with("W123")

    @pytest.mark.asyncio
    async def test_disconnect_teams_success(self, oauth_handler, mock_handler, auth_context):
        """Disconnect Teams tenant succeeds."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"tenant_id": "T123"}
                ):
                    with patch(
                        "aragora.storage.teams_tenant_store.get_teams_tenant_store"
                    ) as mock_store:
                        result = await oauth_handler.handle(
                            "/api/v2/integrations/wizard/teams/disconnect",
                            {},
                            mock_handler,
                        )

                        assert result.status_code == 200
                        data = json.loads(result.body)
                        assert data["disconnected"] is True
                        mock_store.return_value.deactivate.assert_called_with("T123")

    @pytest.mark.asyncio
    async def test_disconnect_requires_delete_permission(
        self, oauth_handler, mock_handler, read_only_context
    ):
        """Disconnect requires connector:delete permission."""
        mock_handler.command = "POST"

        with patch.object(
            oauth_handler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=read_only_context,
        ):
            with patch.object(
                oauth_handler, "check_permission", side_effect=Exception("Permission denied")
            ):
                with patch.object(
                    oauth_handler, "read_json_body", return_value={"workspace_id": "W123"}
                ):
                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard/slack/disconnect",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 403


# -----------------------------------------------------------------------------
# Provider Configuration Check Tests
# -----------------------------------------------------------------------------


class TestProviderConfigCheck:
    """Tests for _check_provider_config method."""

    def test_check_unconfigured_provider(self, oauth_handler):
        """Check unconfigured provider returns errors."""
        with patch.dict(os.environ, {}, clear=True):
            status = oauth_handler._check_provider_config("slack", PROVIDERS["slack"])

            assert status["configured"] is False
            assert len(status["errors"]) > 0
            assert "Missing required" in status["errors"][0]

    def test_check_configured_provider(self, oauth_handler):
        """Check configured provider returns success."""
        with patch.dict(
            os.environ,
            {
                "SLACK_CLIENT_ID": "test-id",
                "SLACK_CLIENT_SECRET": "test-secret",
            },
        ):
            status = oauth_handler._check_provider_config("slack", PROVIDERS["slack"])

            assert status["configured"] is True
            assert len(status["errors"]) == 0

    def test_check_partially_configured_provider(self, oauth_handler):
        """Check partially configured provider returns warnings."""
        with patch.dict(
            os.environ,
            {
                "SLACK_CLIENT_ID": "test-id",
                "SLACK_CLIENT_SECRET": "test-secret",
                # Missing optional: SLACK_REDIRECT_URI, SLACK_SCOPES
            },
            clear=True,
        ):
            status = oauth_handler._check_provider_config("slack", PROVIDERS["slack"])

            assert status["configured"] is True
            assert len(status["warnings"]) > 0


# -----------------------------------------------------------------------------
# Connection Check Tests
# -----------------------------------------------------------------------------


class TestConnectionChecks:
    """Tests for connection check methods."""

    @pytest.mark.asyncio
    async def test_check_connection_unchecked_provider(self, oauth_handler):
        """Check connection for provider without health check."""
        result = await oauth_handler._check_connection("gmail")

        assert result["status"] == "unchecked"

    @pytest.mark.asyncio
    async def test_check_slack_connection_no_workspaces(self, oauth_handler):
        """Check Slack connection with no workspaces."""
        with patch("aragora.storage.slack_workspace_store.get_slack_workspace_store") as mock_store:
            mock_store.return_value.list_active.return_value = []

            result = await oauth_handler._check_slack_connection()

            assert result["status"] == "not_connected"

    @pytest.mark.asyncio
    async def test_check_slack_connection_with_workspaces(self, oauth_handler):
        """Check Slack connection with active workspaces."""
        with patch("aragora.storage.slack_workspace_store.get_slack_workspace_store") as mock_store:
            mock_store.return_value.list_active.side_effect = [
                [{"workspace_id": "W123"}],  # First call (limit=1)
                [{"workspace_id": "W123"}, {"workspace_id": "W456"}],  # Second call (limit=100)
            ]

            result = await oauth_handler._check_slack_connection()

            assert result["status"] == "connected"
            assert result["workspaces"] == 2

    @pytest.mark.asyncio
    async def test_check_discord_connection_not_configured(self, oauth_handler):
        """Check Discord connection without bot token."""
        with patch.dict(os.environ, {}, clear=True):
            result = await oauth_handler._check_discord_connection()

            assert result["status"] == "not_configured"

    @pytest.mark.asyncio
    async def test_check_discord_connection_configured(self, oauth_handler):
        """Check Discord connection with bot token."""
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            result = await oauth_handler._check_discord_connection()

            assert result["status"] == "configured"


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_not_found_returns_404(self, oauth_handler, mock_handler, auth_context):
        """Unknown path returns 404."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                result = await oauth_handler.handle(
                    "/api/v2/integrations/wizard/unknown/path/here",
                    {},
                    mock_handler,
                )

                assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_internal_error_returns_500(self, oauth_handler, mock_handler, auth_context):
        """Internal error returns 500."""
        mock_handler.command = "GET"

        with patch.object(
            oauth_handler, "get_auth_context", new_callable=AsyncMock, return_value=auth_context
        ):
            with patch.object(oauth_handler, "check_permission"):
                with patch.object(
                    oauth_handler,
                    "_get_wizard_config",
                    new_callable=AsyncMock,
                    side_effect=Exception("Internal error"),
                ):
                    result = await oauth_handler.handle(
                        "/api/v2/integrations/wizard",
                        {},
                        mock_handler,
                    )

                    assert result.status_code == 500


# -----------------------------------------------------------------------------
# Provider Data Tests
# -----------------------------------------------------------------------------


class TestProviderData:
    """Tests for PROVIDERS data structure."""

    def test_all_providers_have_required_fields(self):
        """All providers have required fields."""
        required_fields = [
            "name",
            "description",
            "category",
            "setup_time_minutes",
            "features",
            "required_env_vars",
            "optional_env_vars",
            "oauth_scopes",
            "install_url",
            "docs_url",
        ]

        for provider_id, provider in PROVIDERS.items():
            for field in required_fields:
                assert field in provider, f"Provider {provider_id} missing field {field}"

    def test_provider_categories_are_valid(self):
        """Provider categories are valid."""
        valid_categories = {"communication", "development"}

        for provider_id, provider in PROVIDERS.items():
            assert provider["category"] in valid_categories, (
                f"Provider {provider_id} has invalid category {provider['category']}"
            )

    def test_provider_features_are_lists(self):
        """Provider features are lists."""
        for provider_id, provider in PROVIDERS.items():
            assert isinstance(provider["features"], list)
            assert len(provider["features"]) > 0

    def test_slack_provider_config(self):
        """Slack provider has correct configuration."""
        slack = PROVIDERS["slack"]
        assert slack["name"] == "Slack"
        assert "SLACK_CLIENT_ID" in slack["required_env_vars"]
        assert "SLACK_CLIENT_SECRET" in slack["required_env_vars"]
        assert slack["install_url"] == "/api/integrations/slack/install"

    def test_teams_provider_config(self):
        """Teams provider has correct configuration."""
        teams = PROVIDERS["teams"]
        assert teams["name"] == "Microsoft Teams"
        assert "TEAMS_CLIENT_ID" in teams["required_env_vars"]
        assert "TEAMS_CLIENT_SECRET" in teams["required_env_vars"]
