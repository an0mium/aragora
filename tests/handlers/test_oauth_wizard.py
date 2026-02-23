"""Tests for OAuth Wizard Handler (aragora/server/handlers/oauth_wizard.py).

Covers all routes and behavior of the OAuthWizardHandler class:
- can_handle() routing
- GET  /api/v2/integrations/wizard              - Get wizard configuration
- GET  /api/v2/integrations/wizard/providers    - List all available providers
- GET  /api/v2/integrations/wizard/status       - Get status of all integrations
- POST /api/v2/integrations/wizard/validate     - Validate configuration
- POST /api/v2/integrations/wizard/{provider}/test - Test connection
- GET  /api/v2/integrations/wizard/{provider}/workspaces - List workspaces
- POST /api/v2/integrations/wizard/{provider}/disconnect - Disconnect provider
- Authentication and permission enforcement
- Error handling and edge cases
"""

from __future__ import annotations

import json
import os
import socket
from collections import defaultdict
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.oauth_wizard import (
    CONNECTOR_CREATE,
    CONNECTOR_DELETE,
    CONNECTOR_READ,
    OAuthWizardHandler,
    PROVIDERS,
    create_oauth_wizard_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the decoded JSON body from a HandlerResult."""
    if hasattr(result, "body") and isinstance(result.body, bytes):
        return json.loads(result.body.decode("utf-8"))
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


def _make_handler(method: str = "GET", body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with command and optional JSON body."""
    mock = MagicMock()
    mock.command = method
    mock.client_address = ("127.0.0.1", 12345)
    mock.headers = {"Content-Length": "0", "X-Forwarded-For": "127.0.0.1"}
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        mock.rfile.read.return_value = body_bytes
        mock.headers = {
            "Content-Length": str(len(body_bytes)),
            "X-Forwarded-For": "127.0.0.1",
        }
    else:
        mock.rfile.read.return_value = b"{}"
        mock.headers["Content-Length"] = "2"
    return mock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an OAuthWizardHandler with minimal server context."""
    return OAuthWizardHandler({})


@pytest.fixture
def mock_http_get():
    """Create a mock GET HTTP handler."""
    return _make_handler("GET")


@pytest.fixture
def mock_http_post():
    """Create a mock POST HTTP handler with empty body."""
    return _make_handler("POST", {})


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove all provider env vars for clean tests."""
    env_vars = [
        "SLACK_CLIENT_ID",
        "SLACK_CLIENT_SECRET",
        "SLACK_REDIRECT_URI",
        "SLACK_SCOPES",
        "TEAMS_CLIENT_ID",
        "TEAMS_CLIENT_SECRET",
        "TEAMS_REDIRECT_URI",
        "TEAMS_SCOPES",
        "DISCORD_BOT_TOKEN",
        "DISCORD_CLIENT_ID",
        "DISCORD_CLIENT_SECRET",
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_PASSWORD",
        "SMTP_FROM",
        "GOOGLE_OAUTH_CLIENT_ID",
        "GOOGLE_OAUTH_CLIENT_SECRET",
        "GOOGLE_OAUTH_REDIRECT_URI",
        "GITHUB_APP_ID",
        "GITHUB_PRIVATE_KEY",
        "GITHUB_WEBHOOK_SECRET",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset distributed rate limiter state between tests."""
    try:
        from aragora.server.middleware.rate_limit.distributed import get_distributed_limiter

        limiter = get_distributed_limiter()
        if hasattr(limiter, "_buckets"):
            limiter._buckets = defaultdict(dict)
        if hasattr(limiter, "_requests"):
            limiter._requests = defaultdict(list)
    except (ImportError, AttributeError):
        pass
    yield


# ============================================================================
# Routing Tests
# ============================================================================


class TestCanHandle:
    """Tests for can_handle() routing."""

    def test_handles_wizard_root(self, handler):
        assert handler.can_handle("/api/v2/integrations/wizard")

    def test_handles_wizard_providers(self, handler):
        assert handler.can_handle("/api/v2/integrations/wizard/providers")

    def test_handles_wizard_status(self, handler):
        assert handler.can_handle("/api/v2/integrations/wizard/status")

    def test_handles_wizard_validate(self, handler):
        assert handler.can_handle("/api/v2/integrations/wizard/validate")

    def test_handles_provider_test(self, handler):
        assert handler.can_handle("/api/v2/integrations/wizard/slack/test")

    def test_handles_provider_workspaces(self, handler):
        assert handler.can_handle("/api/v2/integrations/wizard/slack/workspaces")

    def test_handles_provider_disconnect(self, handler):
        assert handler.can_handle("/api/v2/integrations/wizard/teams/disconnect")

    def test_rejects_other_paths(self, handler):
        assert not handler.can_handle("/api/v1/integrations/wizard")
        assert not handler.can_handle("/api/v2/integrations/other")
        assert not handler.can_handle("/api/v2/oauth")
        assert not handler.can_handle("/other")

    def test_handles_any_method(self, handler):
        assert handler.can_handle("/api/v2/integrations/wizard", "GET")
        assert handler.can_handle("/api/v2/integrations/wizard", "POST")

    def test_handles_deep_nested_paths(self, handler):
        assert handler.can_handle("/api/v2/integrations/wizard/slack/test/extra")


# ============================================================================
# Factory Tests
# ============================================================================


class TestFactory:
    """Tests for the handler factory function."""

    def test_create_handler(self):
        h = create_oauth_wizard_handler({})
        assert isinstance(h, OAuthWizardHandler)

    def test_create_handler_with_context(self):
        ctx = {"user_store": MagicMock()}
        h = create_oauth_wizard_handler(ctx)
        assert isinstance(h, OAuthWizardHandler)

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "connector"

    def test_routes(self, handler):
        assert "/api/v2/integrations/wizard" in handler.ROUTES
        assert "/api/v2/integrations/wizard/*" in handler.ROUTES


# ============================================================================
# GET /api/v2/integrations/wizard Tests
# ============================================================================


class TestGetWizardConfig:
    """Tests for the wizard configuration endpoint."""

    @pytest.mark.asyncio
    async def test_returns_wizard_config(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        assert _status(result) == 200
        body = _body(result)
        assert "wizard" in body
        assert "generated_at" in body

    @pytest.mark.asyncio
    async def test_wizard_config_has_providers(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        providers = body["wizard"]["providers"]
        assert len(providers) == len(PROVIDERS)

    @pytest.mark.asyncio
    async def test_wizard_config_has_summary(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        summary = body["wizard"]["summary"]
        assert "total_providers" in summary
        assert "configured" in summary
        assert "ready_to_use" in summary
        assert summary["total_providers"] == len(PROVIDERS)

    @pytest.mark.asyncio
    async def test_wizard_config_has_recommended_order(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        order = body["wizard"]["recommended_order"]
        assert "slack" in order
        assert "teams" in order
        assert "email" in order

    @pytest.mark.asyncio
    async def test_wizard_config_sorted_by_category(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        providers = body["wizard"]["providers"]
        categories = [p["category"] for p in providers]
        assert categories == sorted(categories)

    @pytest.mark.asyncio
    async def test_wizard_config_no_providers_configured(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        assert body["wizard"]["summary"]["configured"] == 0

    @pytest.mark.asyncio
    async def test_wizard_config_with_configured_provider(
        self, handler, mock_http_get, monkeypatch
    ):
        monkeypatch.setenv("SLACK_CLIENT_ID", "test-id")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "test-secret")
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        assert body["wizard"]["summary"]["configured"] >= 1

    @pytest.mark.asyncio
    async def test_wizard_providers_have_status(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        for provider in body["wizard"]["providers"]:
            assert "status" in provider
            assert "configured" in provider["status"]
            assert "errors" in provider["status"]

    @pytest.mark.asyncio
    async def test_wizard_version(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        assert body["wizard"]["version"] == "1.0"


# ============================================================================
# GET /api/v2/integrations/wizard/providers Tests
# ============================================================================


class TestListProviders:
    """Tests for the list providers endpoint."""

    @pytest.mark.asyncio
    async def test_list_all_providers(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/providers", {}, mock_http_get)
        assert _status(result) == 200
        body = _body(result)
        assert "providers" in body
        assert "total" in body
        assert body["total"] == len(PROVIDERS)

    @pytest.mark.asyncio
    async def test_provider_fields(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/providers", {}, mock_http_get)
        body = _body(result)
        for p in body["providers"]:
            assert "id" in p
            assert "name" in p
            assert "description" in p
            assert "category" in p
            assert "setup_time_minutes" in p
            assert "features" in p
            assert "configured" in p
            assert "install_url" in p
            assert "docs_url" in p

    @pytest.mark.asyncio
    async def test_filter_by_category_communication(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/providers",
            {"category": "communication"},
            mock_http_get,
        )
        body = _body(result)
        for p in body["providers"]:
            assert p["category"] == "communication"
        assert body["total"] > 0

    @pytest.mark.asyncio
    async def test_filter_by_category_development(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/providers",
            {"category": "development"},
            mock_http_get,
        )
        body = _body(result)
        for p in body["providers"]:
            assert p["category"] == "development"

    @pytest.mark.asyncio
    async def test_filter_by_unknown_category(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/providers",
            {"category": "nonexistent"},
            mock_http_get,
        )
        body = _body(result)
        assert body["total"] == 0
        assert body["providers"] == []

    @pytest.mark.asyncio
    async def test_filter_configured_true(self, handler, mock_http_get, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = await handler.handle(
            "/api/v2/integrations/wizard/providers",
            {"configured": "true"},
            mock_http_get,
        )
        body = _body(result)
        for p in body["providers"]:
            assert p["configured"] is True

    @pytest.mark.asyncio
    async def test_filter_configured_false(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/providers",
            {"configured": "false"},
            mock_http_get,
        )
        body = _body(result)
        for p in body["providers"]:
            assert p["configured"] is False
        assert body["total"] == len(PROVIDERS)

    @pytest.mark.asyncio
    async def test_filter_combined_category_and_configured(
        self, handler, mock_http_get, monkeypatch
    ):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = await handler.handle(
            "/api/v2/integrations/wizard/providers",
            {"category": "communication", "configured": "true"},
            mock_http_get,
        )
        body = _body(result)
        for p in body["providers"]:
            assert p["category"] == "communication"
            assert p["configured"] is True

    @pytest.mark.asyncio
    async def test_empty_query_params(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/providers", {}, mock_http_get)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_providers_include_all_known(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/providers", {}, mock_http_get)
        body = _body(result)
        ids = {p["id"] for p in body["providers"]}
        for provider_id in PROVIDERS:
            assert provider_id in ids


# ============================================================================
# GET /api/v2/integrations/wizard/status Tests
# ============================================================================


class TestGetStatus:
    """Tests for the integration status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_returns_200(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_status_has_statuses_list(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        body = _body(result)
        assert "statuses" in body
        assert len(body["statuses"]) == len(PROVIDERS)

    @pytest.mark.asyncio
    async def test_status_has_summary(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        body = _body(result)
        summary = body["summary"]
        assert "total" in summary
        assert "configured" in summary
        assert "connected" in summary
        assert "needs_attention" in summary
        assert summary["total"] == len(PROVIDERS)

    @pytest.mark.asyncio
    async def test_status_has_checked_at(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        body = _body(result)
        assert "checked_at" in body

    @pytest.mark.asyncio
    async def test_status_unconfigured_no_connection(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        body = _body(result)
        for s in body["statuses"]:
            assert s["configuration"]["configured"] is False
            assert s["connection"] is None

    @pytest.mark.asyncio
    async def test_status_configured_checks_connection(self, handler, mock_http_get, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        body = _body(result)
        discord_status = next(s for s in body["statuses"] if s["provider_id"] == "discord")
        assert discord_status["configuration"]["configured"] is True
        assert discord_status["connection"] is not None
        assert discord_status["connection"]["status"] == "configured"

    @pytest.mark.asyncio
    async def test_status_each_entry_has_fields(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        body = _body(result)
        for s in body["statuses"]:
            assert "provider_id" in s
            assert "name" in s
            assert "category" in s
            assert "configuration" in s

    @pytest.mark.asyncio
    async def test_status_slack_connected(self, handler, mock_http_get, monkeypatch):
        monkeypatch.setenv("SLACK_CLIENT_ID", "test-id")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "test-secret")
        mock_store = MagicMock()
        mock_store.list_active = MagicMock(side_effect=[["ws1"], ["ws1", "ws2"]])
        with patch(
            "aragora.server.handlers.oauth_wizard.OAuthWizardHandler._check_slack_connection",
            new_callable=AsyncMock,
            return_value={"status": "connected", "workspaces": 2},
        ):
            result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        body = _body(result)
        slack_status = next(s for s in body["statuses"] if s["provider_id"] == "slack")
        assert slack_status["configuration"]["configured"] is True


# ============================================================================
# POST /api/v2/integrations/wizard/validate Tests
# ============================================================================


class TestValidateConfig:
    """Tests for the validate configuration endpoint."""

    @pytest.mark.asyncio
    async def test_validate_missing_provider(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert "required" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_validate_unknown_provider(self, handler):
        mock = _make_handler("POST", {"provider": "unknown_service"})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert "Unknown provider" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_validate_slack_missing_env(self, handler):
        mock = _make_handler("POST", {"provider": "slack"})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["provider"] == "slack"
        assert body["valid"] is False
        assert any(c["name"] == "SLACK_CLIENT_ID" for c in body["checks"])
        assert any(c["name"] == "SLACK_CLIENT_SECRET" for c in body["checks"])

    @pytest.mark.asyncio
    async def test_validate_slack_with_env_vars(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_CLIENT_ID", "test-id")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "test-secret")
        mock = _make_handler("POST", {"provider": "slack"})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is True
        assert all(c["present"] for c in body["checks"] if c["required"])

    @pytest.mark.asyncio
    async def test_validate_slack_with_config_values(self, handler):
        mock = _make_handler(
            "POST",
            {
                "provider": "slack",
                "config": {"SLACK_CLIENT_ID": "id", "SLACK_CLIENT_SECRET": "secret"},
            },
        )
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_checks_optional_vars(self, handler):
        mock = _make_handler(
            "POST",
            {
                "provider": "slack",
                "config": {
                    "SLACK_CLIENT_ID": "id",
                    "SLACK_CLIENT_SECRET": "secret",
                },
            },
        )
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        optional_checks = [c for c in body["checks"] if not c["required"]]
        assert len(optional_checks) > 0

    @pytest.mark.asyncio
    async def test_validate_recommendations_valid(self, handler):
        mock = _make_handler(
            "POST",
            {
                "provider": "slack",
                "config": {"SLACK_CLIENT_ID": "id", "SLACK_CLIENT_SECRET": "secret"},
            },
        )
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert len(body["recommendations"]) >= 1
        assert "looks good" in body["recommendations"][0].lower()

    @pytest.mark.asyncio
    async def test_validate_recommendations_invalid(self, handler):
        mock = _make_handler("POST", {"provider": "slack"})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert len(body["recommendations"]) >= 2
        assert "environment variables" in body["recommendations"][0].lower()

    @pytest.mark.asyncio
    async def test_validate_email_provider(self, handler, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        mock = _make_handler("POST", {"provider": "email"})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is True
        assert body["provider"] == "email"

    @pytest.mark.asyncio
    async def test_validate_github_provider(self, handler):
        mock = _make_handler(
            "POST",
            {
                "provider": "github",
                "config": {"GITHUB_APP_ID": "123", "GITHUB_PRIVATE_KEY": "key"},
            },
        )
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is True
        assert body["provider"] == "github"

    @pytest.mark.asyncio
    async def test_validate_discord_provider(self, handler, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        mock = _make_handler("POST", {"provider": "discord"})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_teams_missing_required(self, handler):
        mock = _make_handler("POST", {"provider": "teams"})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_gmail_provider(self, handler):
        mock = _make_handler(
            "POST",
            {
                "provider": "gmail",
                "config": {
                    "GOOGLE_OAUTH_CLIENT_ID": "id",
                    "GOOGLE_OAUTH_CLIENT_SECRET": "secret",
                },
            },
        )
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is True
        assert body["provider"] == "gmail"

    @pytest.mark.asyncio
    async def test_validate_partial_config(self, handler):
        mock = _make_handler(
            "POST",
            {
                "provider": "slack",
                "config": {"SLACK_CLIENT_ID": "id"},
            },
        )
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is False
        id_check = next(c for c in body["checks"] if c["name"] == "SLACK_CLIENT_ID")
        assert id_check["present"] is True
        secret_check = next(c for c in body["checks"] if c["name"] == "SLACK_CLIENT_SECRET")
        assert secret_check["present"] is False

    @pytest.mark.asyncio
    async def test_validate_each_provider(self, handler):
        for provider_id in PROVIDERS:
            mock = _make_handler("POST", {"provider": provider_id})
            result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
            body = _body(result)
            assert body["provider"] == provider_id
            assert "checks" in body


# ============================================================================
# POST /api/v2/integrations/wizard/{provider}/test Tests
# ============================================================================


class TestTestConnection:
    """Tests for the test connection endpoint."""

    @pytest.mark.asyncio
    async def test_unknown_provider(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/nonexistent/test", {}, mock)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_slack_test_returns_result(self, handler):
        mock = _make_handler("POST", {})
        with patch.object(handler, "_test_slack_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"success": True, "team": "TestTeam"}
            result = await handler.handle("/api/v2/integrations/wizard/slack/test", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["provider"] == "slack"
        assert body["test_result"]["success"] is True
        assert "tested_at" in body

    @pytest.mark.asyncio
    async def test_teams_test_returns_result(self, handler):
        mock = _make_handler("POST", {})
        with patch.object(handler, "_test_teams_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"success": True, "display_name": "TestUser"}
            result = await handler.handle("/api/v2/integrations/wizard/teams/test", {}, mock)
        assert _status(result) == 200
        body = _body(result)
        assert body["provider"] == "teams"
        assert body["test_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_discord_test_returns_result(self, handler):
        mock = _make_handler("POST", {})
        with patch.object(handler, "_test_discord_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"success": True, "bot_name": "TestBot"}
            result = await handler.handle("/api/v2/integrations/wizard/discord/test", {}, mock)
        body = _body(result)
        assert body["provider"] == "discord"
        assert body["test_result"]["success"] is True

    @pytest.mark.asyncio
    async def test_unimplemented_provider_test(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/email/test", {}, mock)
        body = _body(result)
        assert body["provider"] == "email"
        assert body["test_result"]["success"] is False
        assert "not implemented" in body["test_result"]["error"].lower()

    @pytest.mark.asyncio
    async def test_test_connection_exception(self, handler):
        mock = _make_handler("POST", {})
        with patch.object(
            handler,
            "_test_slack_api",
            new_callable=AsyncMock,
            side_effect=ConnectionError("network down"),
        ):
            result = await handler.handle("/api/v2/integrations/wizard/slack/test", {}, mock)
        body = _body(result)
        assert body["test_result"]["success"] is False
        assert "failed" in body["test_result"]["error"].lower()

    @pytest.mark.asyncio
    async def test_github_test_not_implemented(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/github/test", {}, mock)
        body = _body(result)
        assert body["test_result"]["success"] is False

    @pytest.mark.asyncio
    async def test_gmail_test_not_implemented(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/gmail/test", {}, mock)
        body = _body(result)
        assert body["test_result"]["success"] is False


# ============================================================================
# GET /api/v2/integrations/wizard/{provider}/workspaces Tests
# ============================================================================


class TestListWorkspaces:
    """Tests for the list workspaces endpoint."""

    @pytest.mark.asyncio
    async def test_unknown_provider(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/nonexistent/workspaces", {}, mock_http_get
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_slack_workspaces(self, handler, mock_http_get):
        with patch.object(handler, "_get_slack_workspaces", new_callable=AsyncMock) as mock_ws:
            mock_ws.return_value = [
                {"id": "W1", "name": "Test", "is_active": True, "connected_at": None}
            ]
            result = await handler.handle(
                "/api/v2/integrations/wizard/slack/workspaces", {}, mock_http_get
            )
        body = _body(result)
        assert body["provider"] == "slack"
        assert body["count"] == 1
        assert len(body["workspaces"]) == 1

    @pytest.mark.asyncio
    async def test_teams_tenants(self, handler, mock_http_get):
        with patch.object(handler, "_get_teams_tenants", new_callable=AsyncMock) as mock_t:
            mock_t.return_value = [
                {"id": "T1", "name": "Corp", "is_active": True, "connected_at": None}
            ]
            result = await handler.handle(
                "/api/v2/integrations/wizard/teams/workspaces", {}, mock_http_get
            )
        body = _body(result)
        assert body["provider"] == "teams"
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_unsupported_provider_workspaces(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/email/workspaces", {}, mock_http_get
        )
        body = _body(result)
        assert body["provider"] == "email"
        assert body["workspaces"] == []
        assert "not available" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_workspaces_exception(self, handler, mock_http_get):
        with patch.object(
            handler,
            "_get_slack_workspaces",
            new_callable=AsyncMock,
            side_effect=ImportError("module not found"),
        ):
            result = await handler.handle(
                "/api/v2/integrations/wizard/slack/workspaces", {}, mock_http_get
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_empty_workspaces(self, handler, mock_http_get):
        with patch.object(handler, "_get_slack_workspaces", new_callable=AsyncMock) as mock_ws:
            mock_ws.return_value = []
            result = await handler.handle(
                "/api/v2/integrations/wizard/slack/workspaces", {}, mock_http_get
            )
        body = _body(result)
        assert body["count"] == 0
        assert body["workspaces"] == []

    @pytest.mark.asyncio
    async def test_discord_workspaces_not_supported(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/discord/workspaces", {}, mock_http_get
        )
        body = _body(result)
        assert body["workspaces"] == []

    @pytest.mark.asyncio
    async def test_github_workspaces_not_supported(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/github/workspaces", {}, mock_http_get
        )
        body = _body(result)
        assert body["workspaces"] == []


# ============================================================================
# POST /api/v2/integrations/wizard/{provider}/disconnect Tests
# ============================================================================


class TestDisconnect:
    """Tests for the disconnect endpoint."""

    @pytest.mark.asyncio
    async def test_disconnect_unknown_provider(self, handler):
        mock = _make_handler("POST", {"workspace_id": "W1"})
        result = await handler.handle(
            "/api/v2/integrations/wizard/nonexistent/disconnect", {}, mock
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_disconnect_slack_success(self, handler):
        mock = _make_handler("POST", {"workspace_id": "W123"})
        with patch.object(handler, "_disconnect_slack_workspace", new_callable=AsyncMock) as mock_d:
            mock_d.return_value = {"success": True, "message": "Workspace W123 disconnected"}
            result = await handler.handle("/api/v2/integrations/wizard/slack/disconnect", {}, mock)
        body = _body(result)
        assert body["provider"] == "slack"
        assert body["disconnected"] is True

    @pytest.mark.asyncio
    async def test_disconnect_slack_missing_workspace_id(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/slack/disconnect", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert "workspace_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_disconnect_teams_success(self, handler):
        mock = _make_handler("POST", {"tenant_id": "T123"})
        with patch.object(handler, "_disconnect_teams_tenant", new_callable=AsyncMock) as mock_d:
            mock_d.return_value = {"success": True, "message": "Tenant T123 disconnected"}
            result = await handler.handle("/api/v2/integrations/wizard/teams/disconnect", {}, mock)
        body = _body(result)
        assert body["disconnected"] is True

    @pytest.mark.asyncio
    async def test_disconnect_teams_missing_tenant_id(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/teams/disconnect", {}, mock)
        assert _status(result) == 400
        body = _body(result)
        assert "tenant_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_disconnect_discord_success(self, handler):
        mock = _make_handler("POST", {"guild_id": "G123"})
        with patch.object(handler, "_disconnect_discord_guild", new_callable=AsyncMock) as mock_d:
            mock_d.return_value = {"success": True, "message": "Guild G123 disconnected"}
            result = await handler.handle(
                "/api/v2/integrations/wizard/discord/disconnect", {}, mock
            )
        body = _body(result)
        assert body["disconnected"] is True

    @pytest.mark.asyncio
    async def test_disconnect_discord_missing_guild_id(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/discord/disconnect", {}, mock)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_disconnect_gmail_success(self, handler):
        mock = _make_handler("POST", {"user_id": "user@example.com"})
        with patch.object(handler, "_disconnect_gmail_account", new_callable=AsyncMock) as mock_d:
            mock_d.return_value = {"success": True, "message": "Gmail disconnected"}
            result = await handler.handle("/api/v2/integrations/wizard/gmail/disconnect", {}, mock)
        body = _body(result)
        assert body["disconnected"] is True

    @pytest.mark.asyncio
    async def test_disconnect_gmail_default_user(self, handler):
        mock = _make_handler("POST", {})
        with patch.object(handler, "_disconnect_gmail_account", new_callable=AsyncMock) as mock_d:
            mock_d.return_value = {"success": True, "message": "Gmail disconnected"}
            result = await handler.handle("/api/v2/integrations/wizard/gmail/disconnect", {}, mock)
        mock_d.assert_called_once_with("default")

    @pytest.mark.asyncio
    async def test_disconnect_email_success(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/email/disconnect", {}, mock)
        body = _body(result)
        assert body["disconnected"] is True
        assert "cleared" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_disconnect_unsupported_provider(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/github/disconnect", {}, mock)
        assert _status(result) == 501

    @pytest.mark.asyncio
    async def test_disconnect_exception(self, handler):
        mock = _make_handler("POST", {"workspace_id": "W1"})
        with patch.object(
            handler,
            "_disconnect_slack_workspace",
            new_callable=AsyncMock,
            side_effect=RuntimeError("store error"),
        ):
            result = await handler.handle("/api/v2/integrations/wizard/slack/disconnect", {}, mock)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_disconnect_slack_failure_result(self, handler):
        mock = _make_handler("POST", {"workspace_id": "W1"})
        with patch.object(handler, "_disconnect_slack_workspace", new_callable=AsyncMock) as mock_d:
            mock_d.return_value = {"success": False, "message": "Not found"}
            result = await handler.handle("/api/v2/integrations/wizard/slack/disconnect", {}, mock)
        body = _body(result)
        assert body["disconnected"] is False


# ============================================================================
# Authentication Tests
# ============================================================================


class TestAuthentication:
    """Tests for authentication requirements."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_providers_returns_401(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/providers", {}, mock_http_get)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_validate_returns_401(self, handler):
        mock = _make_handler("POST", {"provider": "slack"})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_unauthenticated_status_returns_401(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_null_handler_defaults_to_get(self, handler):
        result = await handler.handle("/api/v2/integrations/wizard", {}, None)
        assert _status(result) == 401


# ============================================================================
# Permission Tests
# ============================================================================


class TestPermissions:
    """Tests for RBAC permission enforcement."""

    @pytest.mark.asyncio
    async def test_get_wizard_requires_read(self, handler, mock_http_get):
        with patch.object(handler, "check_permission") as mock_perm:
            await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
            mock_perm.assert_called()
            call_args = mock_perm.call_args
            assert call_args[0][1] == CONNECTOR_READ

    @pytest.mark.asyncio
    async def test_post_validate_requires_create(self, handler):
        mock = _make_handler("POST", {"provider": "slack"})
        with patch.object(handler, "check_permission") as mock_perm:
            await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
            mock_perm.assert_called()
            call_args = mock_perm.call_args
            assert call_args[0][1] == CONNECTOR_CREATE

    @pytest.mark.asyncio
    async def test_post_disconnect_requires_delete(self, handler):
        mock = _make_handler("POST", {"workspace_id": "W1"})
        with patch.object(handler, "check_permission") as mock_perm:
            with patch.object(
                handler,
                "_disconnect_slack_workspace",
                new_callable=AsyncMock,
                return_value={"success": True, "message": "ok"},
            ):
                await handler.handle("/api/v2/integrations/wizard/slack/disconnect", {}, mock)
            mock_perm.assert_called()
            call_args = mock_perm.call_args
            assert call_args[0][1] == CONNECTOR_DELETE

    @pytest.mark.asyncio
    async def test_post_test_requires_create(self, handler):
        mock = _make_handler("POST", {})
        with patch.object(handler, "check_permission") as mock_perm:
            with patch.object(
                handler,
                "_test_slack_api",
                new_callable=AsyncMock,
                return_value={"success": True},
            ):
                await handler.handle("/api/v2/integrations/wizard/slack/test", {}, mock)
            mock_perm.assert_called()
            call_args = mock_perm.call_args
            assert call_args[0][1] == CONNECTOR_CREATE

    @pytest.mark.asyncio
    async def test_permission_denied_returns_403(self, handler, mock_http_get):
        from aragora.rbac.decorators import PermissionDeniedError

        with patch.object(handler, "check_permission", side_effect=PermissionDeniedError("denied")):
            result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_role_required_returns_403(self, handler, mock_http_get):
        from aragora.rbac.decorators import RoleRequiredError

        with patch.object(
            handler,
            "check_permission",
            side_effect=RoleRequiredError("admin role required", {"admin"}, set()),
        ):
            result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        assert _status(result) == 403


# ============================================================================
# Not Found Tests
# ============================================================================


class TestNotFound:
    """Tests for unmatched routes returning 404."""

    @pytest.mark.asyncio
    async def test_unknown_sub_path(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/unknown_action", {}, mock_http_get
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_on_validate(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock_http_get)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_on_test_endpoint(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard/slack/test", {}, mock_http_get)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_on_disconnect_endpoint(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/slack/disconnect", {}, mock_http_get
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_on_workspaces(self, handler):
        mock = _make_handler("POST", {})
        result = await handler.handle("/api/v2/integrations/wizard/slack/workspaces", {}, mock)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_deep_unknown_path(self, handler, mock_http_get):
        result = await handler.handle(
            "/api/v2/integrations/wizard/slack/test/extra/deep", {}, mock_http_get
        )
        assert _status(result) == 404


# ============================================================================
# Internal Connection Check Tests
# ============================================================================


class TestCheckConnection:
    """Tests for internal _check_connection and provider-specific checks."""

    @pytest.mark.asyncio
    async def test_check_discord_not_configured(self, handler):
        result = await handler._check_discord_connection()
        assert result["status"] == "not_configured"

    @pytest.mark.asyncio
    async def test_check_discord_configured(self, handler, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
        result = await handler._check_discord_connection()
        assert result["status"] == "configured"
        assert "Bot token present" in result["note"]

    @pytest.mark.asyncio
    async def test_check_email_not_configured(self, handler):
        result = await handler._check_email_connection()
        assert result["status"] == "not_configured"

    @pytest.mark.asyncio
    async def test_check_email_connected(self, handler, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        with patch("socket.socket") as mock_sock:
            mock_instance = MagicMock()
            mock_instance.connect_ex.return_value = 0
            mock_sock.return_value = mock_instance
            result = await handler._check_email_connection()
        assert result["status"] == "connected"
        assert result["smtp_host"] == "smtp.example.com"

    @pytest.mark.asyncio
    async def test_check_email_unreachable(self, handler, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        with patch("socket.socket") as mock_sock:
            mock_instance = MagicMock()
            mock_instance.connect_ex.return_value = 1
            mock_sock.return_value = mock_instance
            result = await handler._check_email_connection()
        assert result["status"] == "unreachable"

    @pytest.mark.asyncio
    async def test_check_email_connection_error(self, handler, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        with patch("socket.socket") as mock_sock:
            mock_instance = MagicMock()
            mock_instance.connect_ex.side_effect = OSError("connection refused")
            mock_sock.return_value = mock_instance
            result = await handler._check_email_connection()
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_check_email_custom_port(self, handler, monkeypatch):
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("SMTP_PORT", "465")
        with patch("socket.socket") as mock_sock:
            mock_instance = MagicMock()
            mock_instance.connect_ex.return_value = 0
            mock_sock.return_value = mock_instance
            result = await handler._check_email_connection()
        assert result["smtp_port"] == 465

    @pytest.mark.asyncio
    async def test_check_connection_unknown_provider(self, handler):
        result = await handler._check_connection("gmail")
        assert result["status"] == "unchecked"

    @pytest.mark.asyncio
    async def test_check_connection_exception(self, handler):
        with patch.object(
            handler,
            "_check_slack_connection",
            new_callable=AsyncMock,
            side_effect=ConnectionError("failed"),
        ):
            result = await handler._check_connection("slack")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_check_slack_connection_import_error(self, handler):
        with patch.object(
            handler,
            "_check_slack_connection",
            new_callable=AsyncMock,
            side_effect=ImportError("no module"),
        ):
            result = await handler._check_connection("slack")
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_check_teams_connection_route(self, handler):
        with patch.object(handler, "_check_teams_connection", new_callable=AsyncMock) as mock_t:
            mock_t.return_value = {"status": "connected", "tenants": 3}
            result = await handler._check_connection("teams")
        assert result["status"] == "connected"


# ============================================================================
# Provider Config Check Tests
# ============================================================================


class TestCheckProviderConfig:
    """Tests for _check_provider_config."""

    def test_all_required_missing(self, handler):
        provider = PROVIDERS["slack"]
        result = handler._check_provider_config("slack", provider)
        assert result["configured"] is False
        assert len(result["errors"]) > 0
        assert result["required_vars_present"] == 0
        assert result["required_vars_total"] == 2

    def test_all_required_present(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_CLIENT_ID", "id")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "secret")
        provider = PROVIDERS["slack"]
        result = handler._check_provider_config("slack", provider)
        assert result["configured"] is True
        assert result["errors"] == []
        assert result["required_vars_present"] == 2

    def test_partial_required(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_CLIENT_ID", "id")
        provider = PROVIDERS["slack"]
        result = handler._check_provider_config("slack", provider)
        assert result["configured"] is False
        assert result["required_vars_present"] == 1

    def test_optional_warnings(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_CLIENT_ID", "id")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "secret")
        provider = PROVIDERS["slack"]
        result = handler._check_provider_config("slack", provider)
        assert len(result["warnings"]) > 0

    def test_no_optional_warnings_when_set(self, handler, monkeypatch):
        monkeypatch.setenv("SLACK_CLIENT_ID", "id")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "secret")
        monkeypatch.setenv("SLACK_REDIRECT_URI", "http://example.com/callback")
        monkeypatch.setenv("SLACK_SCOPES", "channels:read")
        provider = PROVIDERS["slack"]
        result = handler._check_provider_config("slack", provider)
        assert result["warnings"] == []

    def test_discord_single_required(self, handler, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
        provider = PROVIDERS["discord"]
        result = handler._check_provider_config("discord", provider)
        assert result["configured"] is True
        assert result["required_vars_present"] == 1
        assert result["required_vars_total"] == 1


# ============================================================================
# Internal Slack/Teams Connection Tests
# ============================================================================


class TestSlackConnection:
    """Tests for Slack connection checking."""

    @pytest.mark.asyncio
    async def test_slack_connected(self, handler):
        mock_store = MagicMock()
        mock_store.list_active = MagicMock(side_effect=[["ws1"], ["ws1", "ws2"]])
        with patch(
            "aragora.server.handlers.oauth_wizard.OAuthWizardHandler._check_slack_connection",
            new_callable=AsyncMock,
            return_value={"status": "connected", "workspaces": 2},
        ) as mock_check:
            result = await handler._check_connection("slack")
        assert result["status"] == "connected"

    @pytest.mark.asyncio
    async def test_slack_not_connected(self, handler):
        with patch(
            "aragora.server.handlers.oauth_wizard.OAuthWizardHandler._check_slack_connection",
            new_callable=AsyncMock,
            return_value={"status": "not_connected", "reason": "No active workspaces"},
        ):
            result = await handler._check_connection("slack")
        assert result["status"] == "not_connected"


class TestTeamsConnection:
    """Tests for Teams connection checking."""

    @pytest.mark.asyncio
    async def test_teams_connected(self, handler):
        with patch(
            "aragora.server.handlers.oauth_wizard.OAuthWizardHandler._check_teams_connection",
            new_callable=AsyncMock,
            return_value={"status": "connected", "tenants": 1},
        ):
            result = await handler._check_connection("teams")
        assert result["status"] == "connected"

    @pytest.mark.asyncio
    async def test_teams_not_connected(self, handler):
        with patch(
            "aragora.server.handlers.oauth_wizard.OAuthWizardHandler._check_teams_connection",
            new_callable=AsyncMock,
            return_value={"status": "not_connected", "reason": "No active tenants"},
        ):
            result = await handler._check_connection("teams")
        assert result["status"] == "not_connected"


# ============================================================================
# Internal API Test Method Tests
# ============================================================================


class TestSlackApiTest:
    """Tests for _test_slack_api."""

    @pytest.mark.asyncio
    async def test_slack_no_workspaces(self, handler):
        mock_store = MagicMock()
        mock_store.list_active.return_value = []
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_store,
        ):
            result = await handler._test_slack_api()
        assert result["success"] is False
        assert "No active workspaces" in result["error"]

    @pytest.mark.asyncio
    async def test_slack_no_access_token(self, handler):
        mock_store = MagicMock()
        mock_store.list_active.return_value = [{"workspace_id": "W1"}]
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_store,
        ):
            result = await handler._test_slack_api()
        assert result["success"] is False
        assert "No access token" in result["error"]

    @pytest.mark.asyncio
    async def test_slack_import_error(self, handler):
        """When the slack store module is not available, returns failure."""
        import builtins

        original_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if "slack_workspace_store" in name:
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=patched_import):
            result = await handler._test_slack_api()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_discord_no_bot_token(self, handler):
        result = await handler._test_discord_api()
        assert result["success"] is False
        assert "not configured" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_discord_import_error(self, handler, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
        import builtins

        original_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            if "http_client_pool" in name:
                raise ImportError("no module")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=patched_import):
            result = await handler._test_discord_api()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_teams_no_tenants(self, handler):
        mock_store = MagicMock()
        mock_store.list_active.return_value = []
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_store,
        ):
            result = await handler._test_teams_api()
        assert result["success"] is False
        assert "No active tenants" in result["error"]

    @pytest.mark.asyncio
    async def test_teams_no_access_token(self, handler):
        mock_store = MagicMock()
        mock_store.list_active.return_value = [{"tenant_id": "T1"}]
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_store,
        ):
            result = await handler._test_teams_api()
        assert result["success"] is False
        assert "No access token" in result["error"]


# ============================================================================
# Disconnect Internal Method Tests
# ============================================================================


class TestDisconnectInternals:
    """Tests for internal disconnect methods."""

    @pytest.mark.asyncio
    async def test_disconnect_slack_workspace(self, handler):
        mock_store = MagicMock()
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_store,
        ):
            result = await handler._disconnect_slack_workspace("W123")
        mock_store.deactivate.assert_called_once_with("W123")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_disconnect_teams_tenant(self, handler):
        mock_store = MagicMock()
        with patch(
            "aragora.storage.teams_tenant_store.get_teams_tenant_store",
            return_value=mock_store,
        ):
            result = await handler._disconnect_teams_tenant("T123")
        mock_store.deactivate.assert_called_once_with("T123")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_disconnect_discord_guild_success(self, handler):
        mock_store = MagicMock()
        mock_store.deactivate.return_value = True
        with patch(
            "aragora.storage.discord_guild_store.get_discord_guild_store",
            return_value=mock_store,
        ):
            result = await handler._disconnect_discord_guild("G123")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_disconnect_discord_guild_not_found(self, handler):
        mock_store = MagicMock()
        mock_store.deactivate.return_value = False
        with patch(
            "aragora.storage.discord_guild_store.get_discord_guild_store",
            return_value=mock_store,
        ):
            result = await handler._disconnect_discord_guild("G999")
        assert result["success"] is False
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_disconnect_gmail_success(self, handler):
        mock_store = AsyncMock()
        mock_store.delete.return_value = True
        with patch(
            "aragora.storage.integration_store.get_integration_store",
            return_value=mock_store,
        ):
            result = await handler._disconnect_gmail_account("user@example.com")
        mock_store.delete.assert_called_once_with("gmail", "user@example.com")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_disconnect_gmail_not_found(self, handler):
        mock_store = AsyncMock()
        mock_store.delete.return_value = False
        with patch(
            "aragora.storage.integration_store.get_integration_store",
            return_value=mock_store,
        ):
            result = await handler._disconnect_gmail_account("unknown@example.com")
        assert result["success"] is False


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling across all endpoints."""

    @pytest.mark.asyncio
    async def test_connection_error_returns_500(self, handler, mock_http_get):
        with patch.object(
            handler,
            "_get_wizard_config",
            new_callable=AsyncMock,
            side_effect=ConnectionError("network down"),
        ):
            result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_timeout_error_returns_500(self, handler, mock_http_get):
        with patch.object(
            handler,
            "_get_wizard_config",
            new_callable=AsyncMock,
            side_effect=TimeoutError("timed out"),
        ):
            result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_value_error_returns_500(self, handler, mock_http_get):
        with patch.object(
            handler,
            "_list_providers",
            new_callable=AsyncMock,
            side_effect=ValueError("bad value"),
        ):
            result = await handler.handle(
                "/api/v2/integrations/wizard/providers", {}, mock_http_get
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, handler, mock_http_get):
        with patch.object(
            handler,
            "_get_status",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected"),
        ):
            result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        assert _status(result) == 500

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_null_handler_with_post_body(self, handler):
        """Null handler defaults method to GET and auth fails without mock."""
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, None)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_query_params_default_empty(self, handler, mock_http_get):
        result = await handler.handle("/api/v2/integrations/wizard", None, mock_http_get)
        assert _status(result) == 200


# ============================================================================
# Provider Data Integrity Tests
# ============================================================================


class TestProviderData:
    """Tests validating the PROVIDERS constant data integrity."""

    def test_all_providers_have_required_fields(self):
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
        for pid, pdata in PROVIDERS.items():
            for field in required_fields:
                assert field in pdata, f"{pid} missing field: {field}"

    def test_all_providers_have_features_list(self):
        for pid, pdata in PROVIDERS.items():
            assert isinstance(pdata["features"], list), f"{pid} features should be a list"
            assert len(pdata["features"]) > 0, f"{pid} should have features"

    def test_all_providers_have_valid_categories(self):
        valid_categories = {"communication", "development"}
        for pid, pdata in PROVIDERS.items():
            assert pdata["category"] in valid_categories, f"{pid} has invalid category"

    def test_provider_count(self):
        assert len(PROVIDERS) == 6

    def test_provider_ids(self):
        expected = {"slack", "teams", "discord", "email", "gmail", "github"}
        assert set(PROVIDERS.keys()) == expected

    def test_setup_times_positive(self):
        for pid, pdata in PROVIDERS.items():
            assert pdata["setup_time_minutes"] > 0

    def test_email_has_no_oauth(self):
        assert PROVIDERS["email"]["oauth_scopes"] == []
        assert PROVIDERS["email"]["install_url"] is None

    def test_docs_urls_well_formed(self):
        for pid, pdata in PROVIDERS.items():
            assert pdata["docs_url"].startswith("https://")


# ============================================================================
# MaybeAwait Helper Tests
# ============================================================================


class TestMaybeAwait:
    """Tests for the _maybe_await helper."""

    @pytest.mark.asyncio
    async def test_non_awaitable_value(self, handler):
        result = await handler._maybe_await(42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_none_value(self, handler):
        result = await handler._maybe_await(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_awaitable_value(self, handler):
        async def coro():
            return "hello"

        result = await handler._maybe_await(coro())
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_string_value(self, handler):
        result = await handler._maybe_await("test")
        assert result == "test"

    @pytest.mark.asyncio
    async def test_list_value(self, handler):
        result = await handler._maybe_await([1, 2, 3])
        assert result == [1, 2, 3]


# ============================================================================
# NormalizeMockSideEffect Tests
# ============================================================================


class TestNormalizeMockSideEffect:
    """Tests for _normalize_mock_side_effect."""

    def test_list_side_effect_converted_to_iter(self, handler):
        mock = MagicMock()
        mock.side_effect = [1, 2, 3]
        handler._normalize_mock_side_effect(mock)
        assert not isinstance(mock.side_effect, list)

    def test_non_list_side_effect_unchanged(self, handler):
        mock = MagicMock()
        original = lambda: None
        mock.side_effect = original
        handler._normalize_mock_side_effect(mock)
        assert mock.side_effect is original

    def test_no_side_effect(self, handler):
        mock = MagicMock()
        mock.side_effect = None
        handler._normalize_mock_side_effect(mock)
        assert mock.side_effect is None

    def test_plain_object(self, handler):
        obj = object()
        handler._normalize_mock_side_effect(obj)
        # Should not raise


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Miscellaneous edge case tests."""

    @pytest.mark.asyncio
    async def test_wizard_config_multiple_configured(self, handler, mock_http_get, monkeypatch):
        monkeypatch.setenv("SLACK_CLIENT_ID", "id")
        monkeypatch.setenv("SLACK_CLIENT_SECRET", "secret")
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        assert body["wizard"]["summary"]["configured"] >= 3

    @pytest.mark.asyncio
    async def test_validate_empty_config_dict(self, handler):
        mock = _make_handler("POST", {"provider": "slack", "config": {}})
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_with_extra_config_keys(self, handler):
        mock = _make_handler(
            "POST",
            {
                "provider": "discord",
                "config": {"DISCORD_BOT_TOKEN": "tok", "EXTRA_KEY": "value"},
            },
        )
        result = await handler.handle("/api/v2/integrations/wizard/validate", {}, mock)
        body = _body(result)
        assert body["valid"] is True

    @pytest.mark.asyncio
    async def test_multiple_providers_configured_status(self, handler, mock_http_get, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        result = await handler.handle("/api/v2/integrations/wizard/status", {}, mock_http_get)
        body = _body(result)
        configured = [s for s in body["statuses"] if s["configuration"]["configured"]]
        assert len(configured) >= 2

    @pytest.mark.asyncio
    async def test_filter_configured_true_uppercase(self, handler, mock_http_get, monkeypatch):
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
        result = await handler.handle(
            "/api/v2/integrations/wizard/providers",
            {"configured": "True"},
            mock_http_get,
        )
        body = _body(result)
        for p in body["providers"]:
            assert p["configured"] is True

    @pytest.mark.asyncio
    async def test_ready_to_use_count(self, handler, mock_http_get, monkeypatch):
        """Ready to use = configured AND no errors."""
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "token")
        result = await handler.handle("/api/v2/integrations/wizard", {}, mock_http_get)
        body = _body(result)
        ready = body["wizard"]["summary"]["ready_to_use"]
        configured = body["wizard"]["summary"]["configured"]
        assert ready <= configured
