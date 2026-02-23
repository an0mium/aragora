"""Tests for Slack OAuth handler (aragora/server/handlers/social/slack_oauth.py).

Covers all routes and behavior of the SlackOAuthHandler class:
- can_handle() route matching for all static and dynamic routes
- GET  /api/integrations/slack/install   - Initiate OAuth flow
- GET  /api/integrations/slack/preview   - Consent preview page
- GET  /api/integrations/slack/callback  - OAuth callback from Slack
- POST /api/integrations/slack/uninstall - Handle app removal webhook
- GET  /api/integrations/slack/workspaces - List workspaces
- GET  /api/integrations/slack/workspaces/{id}/status - Workspace token status
- POST /api/integrations/slack/workspaces/{id}/refresh - Token refresh
- Method not allowed responses
- Permission denied paths
- Error handling and edge cases
- OAuth state management helpers
- Dual-signature handle() calling conventions
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _html(result) -> str:
    """Extract HTML body string from a HandlerResult."""
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8")
    return raw


# ---------------------------------------------------------------------------
# Lazy import so conftest auto-auth patches run first
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    import aragora.server.handlers.social.slack_oauth as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.SlackOAuthHandler


@pytest.fixture
def handler(handler_cls):
    """Create a SlackOAuthHandler with empty context."""
    return handler_cls(ctx={})


@pytest.fixture(autouse=True)
def _reset_module_globals(handler_module, monkeypatch):
    """Reset module-level globals between tests so state does not leak."""
    handler_module._oauth_states_fallback.clear()
    handler_module._slack_oauth_audit = None
    yield
    handler_module._oauth_states_fallback.clear()
    handler_module._slack_oauth_audit = None


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch, handler_module):
    """Set a default non-production environment and Slack credentials."""
    monkeypatch.setattr(handler_module, "SLACK_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(handler_module, "SLACK_CLIENT_SECRET", "test-client-secret")
    monkeypatch.setattr(handler_module, "SLACK_REDIRECT_URI", "https://example.com/callback")
    monkeypatch.setattr(handler_module, "ARAGORA_ENV", "test")
    monkeypatch.setattr(handler_module, "SLACK_SCOPES", "channels:history,chat:write")


@pytest.fixture
def mock_state_store():
    """Create a mock OAuth state store."""
    store = MagicMock()
    store.generate.return_value = "test-state-token-abc123"
    store.validate_and_consume.return_value = {
        "tenant_id": "tenant-1",
        "provider": "slack",
        "created_at": time.time(),
    }
    return store


@pytest.fixture(autouse=True)
def _patch_state_store(monkeypatch, mock_state_store, handler_module):
    """Patch _get_state_store to return our mock."""
    monkeypatch.setattr(handler_module, "_get_state_store", lambda: mock_state_store)


@pytest.fixture
def mock_workspace():
    """Create a mock SlackWorkspace object."""
    ws = MagicMock()
    ws.workspace_id = "W123"
    ws.workspace_name = "Test Workspace"
    ws.access_token = "xoxb-test"
    ws.bot_user_id = "B123"
    ws.installed_at = time.time()
    ws.installed_by = "U456"
    ws.scopes = ["channels:history", "chat:write"]
    ws.tenant_id = "tenant-1"
    ws.is_active = True
    ws.refresh_token = "xoxr-refresh"
    ws.token_expires_at = time.time() + 7200
    return ws


@pytest.fixture
def mock_workspace_store(mock_workspace):
    """Create a mock workspace store."""
    store = MagicMock()
    store.save.return_value = True
    store.get.return_value = mock_workspace
    store.list_active.return_value = [mock_workspace]
    store.deactivate.return_value = True
    return store


def _make_httpx_mock(response_json, status_code=200):
    """Build a mock httpx AsyncClient + response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_json
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client, mock_response


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_install_path(self, handler):
        assert handler.can_handle("/api/integrations/slack/install")

    def test_preview_path(self, handler):
        assert handler.can_handle("/api/integrations/slack/preview")

    def test_callback_path(self, handler):
        assert handler.can_handle("/api/integrations/slack/callback")

    def test_uninstall_path(self, handler):
        assert handler.can_handle("/api/integrations/slack/uninstall")

    def test_workspaces_path(self, handler):
        assert handler.can_handle("/api/integrations/slack/workspaces")

    def test_workspace_status_path(self, handler):
        assert handler.can_handle("/api/integrations/slack/workspaces/W123/status")

    def test_workspace_refresh_path(self, handler):
        assert handler.can_handle("/api/integrations/slack/workspaces/W123/refresh")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_path(self, handler):
        assert not handler.can_handle("/api/integrations/slack")

    def test_rejects_extra_suffix(self, handler):
        assert not handler.can_handle("/api/integrations/slack/install/extra")

    def test_rejects_wrong_prefix(self, handler):
        assert not handler.can_handle("/api/v1/integrations/slack/install")

    def test_rejects_typo(self, handler):
        assert not handler.can_handle("/api/integrations/slak/install")

    def test_workspace_status_dynamic_id(self, handler):
        assert handler.can_handle("/api/integrations/slack/workspaces/any-id-here/status")

    def test_workspace_refresh_dynamic_id(self, handler):
        assert handler.can_handle("/api/integrations/slack/workspaces/abc-xyz/refresh")

    def test_rejects_workspace_unknown_action(self, handler):
        assert not handler.can_handle("/api/integrations/slack/workspaces/W123/delete")

    def test_rejects_workspace_missing_action(self, handler):
        """Workspaces with ID but no sub-action should not match patterns."""
        assert not handler.can_handle("/api/integrations/slack/workspaces/W123")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/")


# ============================================================================
# Handler initialization and factory
# ============================================================================


class TestInit:
    """Test handler initialization."""

    def test_default_ctx(self, handler_cls):
        h = handler_cls()
        assert h.ctx == {}

    def test_custom_ctx(self, handler_cls):
        ctx = {"key": "value"}
        h = handler_cls(ctx=ctx)
        assert h.ctx == ctx

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "connector"

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) >= 5

    def test_route_patterns_count(self, handler):
        assert len(handler.ROUTE_PATTERNS) == 2


class TestFactory:
    """Test the factory function."""

    def test_create_handler(self, handler_module):
        h = handler_module.create_slack_oauth_handler({"server": True})
        assert isinstance(h, handler_module.SlackOAuthHandler)
        assert h.ctx == {"server": True}


# ============================================================================
# GET /api/integrations/slack/install
# ============================================================================


class TestInstall:
    """Tests for the /install endpoint."""

    @pytest.mark.asyncio
    async def test_returns_302_redirect(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        assert _status(result) == 302

    @pytest.mark.asyncio
    async def test_redirect_location_contains_slack_oauth_url(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        location = result.headers.get("Location", "")
        assert "slack.com/oauth/v2/authorize" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_client_id(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        location = result.headers.get("Location", "")
        assert "client_id=test-client-id" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_state(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        location = result.headers.get("Location", "")
        assert "state=test-state-token-abc123" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_redirect_uri(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        location = result.headers.get("Location", "")
        assert "redirect_uri=" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_scopes(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        location = result.headers.get("Location", "")
        assert "scope=" in location

    @pytest.mark.asyncio
    async def test_cache_control_no_store(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        assert result.headers.get("Cache-Control") == "no-store"

    @pytest.mark.asyncio
    async def test_content_type_html(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        assert "text/html" in result.content_type

    @pytest.mark.asyncio
    async def test_with_tenant_id(self, handler, mock_state_store):
        result = await handler.handle(
            "GET", "/api/integrations/slack/install", {}, {"tenant_id": "t-123"}, {}, None
        )
        assert _status(result) == 302
        mock_state_store.generate.assert_called_once()
        call_kwargs = mock_state_store.generate.call_args
        metadata = call_kwargs[1].get("metadata") or call_kwargs.kwargs.get("metadata")
        assert metadata["tenant_id"] == "t-123"

    @pytest.mark.asyncio
    async def test_stores_state_in_fallback(self, handler, handler_module):
        await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        assert "test-state-token-abc123" in handler_module._oauth_states_fallback

    @pytest.mark.asyncio
    async def test_fallback_state_contains_provider(self, handler, handler_module):
        await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        state_data = handler_module._oauth_states_fallback["test-state-token-abc123"]
        assert state_data["provider"] == "slack"

    @pytest.mark.asyncio
    async def test_no_client_id_returns_503(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "SLACK_CLIENT_ID", None)
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_state_store_failure_returns_503(self, handler, mock_state_store):
        mock_state_store.generate.side_effect = RuntimeError("store down")
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_state_store_value_error_returns_503(self, handler, mock_state_store):
        mock_state_store.generate.side_effect = ValueError("bad input")
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_method_not_allowed_post(self, handler):
        result = await handler.handle("POST", "/api/integrations/slack/install", {}, {}, {}, None)
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_dev_fallback_redirect_uri_localhost(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "SLACK_REDIRECT_URI", None)
        monkeypatch.setattr(handler_module, "ARAGORA_ENV", "development")
        result = await handler.handle(
            "GET", "/api/integrations/slack/install", {}, {"host": "localhost:3000"}, {}, None
        )
        assert _status(result) == 302
        location = result.headers.get("Location", "")
        assert "localhost" in location
        assert "3000" in location

    @pytest.mark.asyncio
    async def test_dev_fallback_127_0_0_1(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "SLACK_REDIRECT_URI", None)
        monkeypatch.setattr(handler_module, "ARAGORA_ENV", "dev")
        result = await handler.handle(
            "GET", "/api/integrations/slack/install", {}, {"host": "127.0.0.1:8080"}, {}, None
        )
        assert _status(result) == 302

    @pytest.mark.asyncio
    async def test_dev_fallback_rejects_non_localhost(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "SLACK_REDIRECT_URI", None)
        monkeypatch.setattr(handler_module, "ARAGORA_ENV", "development")
        result = await handler.handle(
            "GET", "/api/integrations/slack/install", {}, {"host": "evil.com:8080"}, {}, None
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_production_requires_redirect_uri(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "SLACK_REDIRECT_URI", None)
        monkeypatch.setattr(handler_module, "ARAGORA_ENV", "production")
        result = await handler.handle("GET", "/api/integrations/slack/install", {}, {}, {}, None)
        assert _status(result) == 500


# ============================================================================
# GET /api/integrations/slack/preview
# ============================================================================


class TestPreview:
    """Tests for the /preview consent page endpoint."""

    @pytest.mark.asyncio
    async def test_returns_200(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_returns_html(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        assert "text/html" in result.content_type

    @pytest.mark.asyncio
    async def test_html_contains_install_link(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        html = _html(result)
        assert "/api/integrations/slack/install" in html

    @pytest.mark.asyncio
    async def test_html_contains_scope_names(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        html = _html(result)
        assert "Read Channel Messages" in html
        assert "Send Messages" in html

    @pytest.mark.asyncio
    async def test_html_contains_data_notice(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        html = _html(result)
        assert "How We Handle Your Data" in html

    @pytest.mark.asyncio
    async def test_html_contains_title(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        html = _html(result)
        assert "Install Aragora" in html

    @pytest.mark.asyncio
    async def test_preview_with_tenant_id(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/preview", {}, {"tenant_id": "t-abc"}, {}, None
        )
        html = _html(result)
        assert "tenant_id=t-abc" in html

    @pytest.mark.asyncio
    async def test_preview_without_tenant_id(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        html = _html(result)
        assert "tenant_id=" not in html

    @pytest.mark.asyncio
    async def test_preview_no_client_id_returns_503(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "SLACK_CLIENT_ID", None)
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_cache_control(self, handler):
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        assert result.headers.get("Cache-Control") == "no-store"

    @pytest.mark.asyncio
    async def test_method_not_allowed_post(self, handler):
        result = await handler.handle("POST", "/api/integrations/slack/preview", {}, {}, {}, None)
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_unknown_scope_shown_as_required(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "SLACK_SCOPES", "custom:scope")
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        html = _html(result)
        assert "Custom Scope" in html
        assert "Required" in html

    @pytest.mark.asyncio
    async def test_optional_scopes_section(self, handler, handler_module, monkeypatch):
        """Scopes marked optional in SCOPE_DESCRIPTIONS show as Optional."""
        monkeypatch.setattr(handler_module, "SLACK_SCOPES", "commands")
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        html = _html(result)
        assert "Optional" in html

    @pytest.mark.asyncio
    async def test_all_default_scopes(self, handler, handler_module, monkeypatch):
        """All default scopes render correctly."""
        monkeypatch.setattr(
            handler_module, "SLACK_SCOPES",
            "channels:history,chat:write,commands,users:read,team:read,channels:read",
        )
        result = await handler.handle("GET", "/api/integrations/slack/preview", {}, {}, {}, None)
        html = _html(result)
        assert "Read Channel Messages" in html
        assert "Slash Commands" in html
        assert "View User Information" in html
        assert "View Workspace Info" in html
        assert "List Channels" in html


# ============================================================================
# GET /api/integrations/slack/callback
# ============================================================================


class TestCallback:
    """Tests for the OAuth callback endpoint."""

    @pytest.mark.asyncio
    async def test_error_param_returns_400(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {"error": "access_denied"}, {}, None,
        )
        assert _status(result) == 400
        body = _body(result)
        assert "access_denied" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_missing_code_returns_400(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {"state": "abc"}, {}, None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_state_returns_400(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {"code": "abc"}, {}, None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_both_code_and_state_returns_400(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {}, {}, None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_state_returns_400(self, handler, mock_state_store):
        mock_state_store.validate_and_consume.return_value = None
        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {"code": "abc", "state": "invalid"}, {}, None,
        )
        assert _status(result) == 400
        body = _body(result)
        assert "expired" in body.get("error", "").lower() or "invalid" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_successful_callback(self, handler, mock_workspace_store):
        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-new-token",
            "team": {"id": "W123", "name": "My Team"},
            "bot_user_id": "B789",
            "authed_user": {"id": "U456"},
            "scope": "channels:history,chat:write",
        })

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ), \
             patch(
                 "aragora.storage.slack_workspace_store.SlackWorkspace",
                 return_value=MagicMock(),
             ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/callback", {},
                {"code": "test-code", "state": "test-state-token-abc123"}, {}, None,
            )
        assert _status(result) == 200
        html = _html(result)
        assert "Connected" in html
        assert "My Team" in html

    @pytest.mark.asyncio
    async def test_callback_slack_api_error(self, handler):
        mock_client, _ = _make_httpx_mock({"ok": False, "error": "invalid_code"})

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.handle(
                "GET", "/api/integrations/slack/callback", {},
                {"code": "bad-code", "state": "test-state-token-abc123"}, {}, None,
            )
        assert _status(result) == 400
        body = _body(result)
        assert "invalid_code" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_callback_missing_workspace_id_returns_500(self, handler):
        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-token",
            "team": {"id": "", "name": "No ID"},
            "bot_user_id": "B789",
            "authed_user": {"id": "U456"},
            "scope": "channels:history",
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.handle(
                "GET", "/api/integrations/slack/callback", {},
                {"code": "code", "state": "test-state-token-abc123"}, {}, None,
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_missing_access_token_returns_500(self, handler):
        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "",
            "team": {"id": "W123", "name": "No Token"},
            "bot_user_id": "B789",
            "authed_user": {"id": "U456"},
            "scope": "channels:history",
        })

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.handle(
                "GET", "/api/integrations/slack/callback", {},
                {"code": "code", "state": "test-state-token-abc123"}, {}, None,
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_no_credentials_returns_503(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "SLACK_CLIENT_ID", None)
        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {"code": "code", "state": "test-state-token-abc123"}, {}, None,
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_no_secret_returns_503(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "SLACK_CLIENT_SECRET", None)
        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {"code": "code", "state": "test-state-token-abc123"}, {}, None,
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_workspace_store_import_error(self, handler):
        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-token",
            "team": {"id": "W999", "name": "Store Fail"},
            "bot_user_id": "B1",
            "authed_user": {"id": "U1"},
            "scope": "channels:history",
        })

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            result = await handler.handle(
                "GET", "/api/integrations/slack/callback", {},
                {"code": "code", "state": "test-state-token-abc123"}, {}, None,
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_workspace_save_fails(self, handler, mock_workspace_store):
        mock_workspace_store.save.return_value = False

        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-token",
            "team": {"id": "W999", "name": "Save Fail"},
            "bot_user_id": "B1",
            "authed_user": {"id": "U1"},
            "scope": "channels:history",
        })

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ), \
             patch(
                 "aragora.storage.slack_workspace_store.SlackWorkspace",
                 return_value=MagicMock(),
             ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/callback", {},
                {"code": "code", "state": "test-state-token-abc123"}, {}, None,
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_method_not_allowed(self, handler):
        result = await handler.handle(
            "POST", "/api/integrations/slack/callback", {}, {}, {}, None,
        )
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_callback_with_refresh_token(self, handler, mock_workspace_store):
        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-token",
            "team": {"id": "W123", "name": "Refresh Team"},
            "bot_user_id": "B789",
            "authed_user": {"id": "U456"},
            "scope": "channels:history",
            "refresh_token": "xoxr-refresh-token",
            "expires_in": 43200,
        })

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ), \
             patch(
                 "aragora.storage.slack_workspace_store.SlackWorkspace",
                 return_value=MagicMock(),
             ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/callback", {},
                {"code": "code", "state": "test-state-token-abc123"}, {}, None,
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_fallback_state_validation(self, handler, handler_module, mock_state_store):
        """When centralized store returns None, fall back to in-memory."""
        mock_state_store.validate_and_consume.return_value = None
        handler_module._oauth_states_fallback["fallback-state"] = {
            "tenant_id": "t-1",
            "provider": "slack",
            "created_at": time.time(),
        }

        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-token",
            "team": {"id": "W111", "name": "Fallback"},
            "bot_user_id": "B1",
            "authed_user": {"id": "U1"},
            "scope": "channels:history",
        })

        mock_ws_store = MagicMock()
        mock_ws_store.save.return_value = True

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_ws_store,
             ), \
             patch(
                 "aragora.storage.slack_workspace_store.SlackWorkspace",
                 return_value=MagicMock(),
             ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/callback", {},
                {"code": "code", "state": "fallback-state"}, {}, None,
            )
        assert _status(result) == 200
        # Fallback state should have been consumed (popped)
        assert "fallback-state" not in handler_module._oauth_states_fallback

    @pytest.mark.asyncio
    async def test_callback_state_from_oauth_state_object(self, handler, mock_state_store, mock_workspace_store):
        """When state store returns an OAuthState object instead of dict."""
        state_obj = MagicMock()
        state_obj.metadata = {"tenant_id": "t-obj", "provider": "slack"}
        mock_state_store.validate_and_consume.return_value = state_obj

        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-token",
            "team": {"id": "W123", "name": "ObjState"},
            "bot_user_id": "B1",
            "authed_user": {"id": "U1"},
            "scope": "channels:history",
        })

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ), \
             patch(
                 "aragora.storage.slack_workspace_store.SlackWorkspace",
                 return_value=MagicMock(),
             ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/callback", {},
                {"code": "code", "state": "test-state-token-abc123"}, {}, None,
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_error_with_audit_logger(self, handler, handler_module):
        """Error callback triggers audit log if audit logger is available."""
        mock_audit = MagicMock()
        handler_module._slack_oauth_audit = mock_audit

        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {"error": "access_denied"}, {}, None,
        )
        assert _status(result) == 400
        mock_audit.log_oauth.assert_called_once()
        call_kwargs = mock_audit.log_oauth.call_args[1]
        assert call_kwargs["action"] == "install"
        assert call_kwargs["success"] is False


# ============================================================================
# POST /api/integrations/slack/uninstall
# ============================================================================


class TestUninstall:
    """Tests for the app uninstall webhook endpoint."""

    @pytest.mark.asyncio
    async def test_app_uninstalled_event(self, handler, mock_workspace_store, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "")
        body = {
            "team_id": "W123",
            "event": {"type": "app_uninstalled", "team_id": "W123"},
        }
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/uninstall", body, {}, {}, None,
            )
        assert _status(result) == 200
        body_data = _body(result)
        assert body_data.get("ok") is True
        mock_workspace_store.deactivate.assert_called_once_with("W123")

    @pytest.mark.asyncio
    async def test_app_uninstalled_uses_event_team_id(self, handler, mock_workspace_store, monkeypatch):
        """When body.team_id is absent, falls back to event.team_id."""
        monkeypatch.setenv("ARAGORA_ENV", "test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "")
        body = {
            "event": {"type": "app_uninstalled", "team_id": "W789"},
        }
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/uninstall", body, {}, {}, None,
            )
        assert _status(result) == 200
        mock_workspace_store.deactivate.assert_called_once_with("W789")

    @pytest.mark.asyncio
    async def test_tokens_revoked_event(self, handler, mock_workspace_store, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "")
        body = {
            "team_id": "W123",
            "event": {
                "type": "tokens_revoked",
                "tokens": {"bot": ["xoxb-old"]},
            },
        }
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/uninstall", body, {}, {}, None,
            )
        assert _status(result) == 200
        mock_workspace_store.deactivate.assert_called_once_with("W123")

    @pytest.mark.asyncio
    async def test_tokens_revoked_no_bot_tokens_no_deactivation(self, handler, mock_workspace_store, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "")
        body = {
            "team_id": "W123",
            "event": {
                "type": "tokens_revoked",
                "tokens": {"bot": []},
            },
        }
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/uninstall", body, {}, {}, None,
            )
        assert _status(result) == 200
        mock_workspace_store.deactivate.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_event_still_acks(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "")
        body = {"event": {"type": "unknown_event"}}
        result = await handler.handle(
            "POST", "/api/integrations/slack/uninstall", body, {}, {}, None,
        )
        assert _status(result) == 200
        assert _body(result).get("ok") is True

    @pytest.mark.asyncio
    async def test_missing_signing_secret_in_production(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "")
        body = {"event": {"type": "app_uninstalled"}}
        result = await handler.handle(
            "POST", "/api/integrations/slack/uninstall", body, {}, {}, None,
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_valid_signature_accepted(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        signing_secret = "test-signing-secret"
        monkeypatch.setenv("SLACK_SIGNING_SECRET", signing_secret)

        body = {"event": {"type": "app_uninstalled"}, "team_id": "W123"}
        timestamp = str(int(time.time()))
        body_str = json.dumps(body, separators=(",", ":"))
        sig_basestring = f"v0:{timestamp}:{body_str}"
        computed_sig = "v0=" + hmac.new(
            signing_secret.encode(), sig_basestring.encode(), hashlib.sha256
        ).hexdigest()

        headers = {
            "x-slack-request-timestamp": timestamp,
            "x-slack-signature": computed_sig,
        }

        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=MagicMock(),
        ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/uninstall", body, {}, headers, None,
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_invalid_signature_rejected(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "real-secret")

        body = {"event": {"type": "app_uninstalled"}}
        timestamp = str(int(time.time()))
        headers = {
            "x-slack-request-timestamp": timestamp,
            "x-slack-signature": "v0=invalid_signature_hex",
        }
        result = await handler.handle(
            "POST", "/api/integrations/slack/uninstall", body, {}, headers, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_missing_signature_headers(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "some-secret")
        body = {"event": {"type": "app_uninstalled"}}
        result = await handler.handle(
            "POST", "/api/integrations/slack/uninstall", body, {}, {}, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_missing_timestamp_header(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "some-secret")
        headers = {"x-slack-signature": "v0=somesig"}
        body = {"event": {"type": "app_uninstalled"}}
        result = await handler.handle(
            "POST", "/api/integrations/slack/uninstall", body, {}, headers, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_expired_timestamp_rejected(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "some-secret")
        old_timestamp = str(int(time.time()) - 600)
        headers = {
            "x-slack-request-timestamp": old_timestamp,
            "x-slack-signature": "v0=somesig",
        }
        body = {"event": {"type": "app_uninstalled"}}
        result = await handler.handle(
            "POST", "/api/integrations/slack/uninstall", body, {}, headers, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_invalid_timestamp_rejected(self, handler, monkeypatch):
        monkeypatch.setenv("ARAGORA_ENV", "production")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "some-secret")
        headers = {
            "x-slack-request-timestamp": "not-a-number",
            "x-slack-signature": "v0=somesig",
        }
        body = {"event": {"type": "app_uninstalled"}}
        result = await handler.handle(
            "POST", "/api/integrations/slack/uninstall", body, {}, headers, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_method_not_allowed_get(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/uninstall", {}, {}, {}, None,
        )
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_uninstall_store_import_error(self, handler, monkeypatch):
        """Store import error is handled gracefully during uninstall."""
        monkeypatch.setenv("ARAGORA_ENV", "test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "")
        body = {
            "team_id": "W123",
            "event": {"type": "app_uninstalled", "team_id": "W123"},
        }
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            result = await handler.handle(
                "POST", "/api/integrations/slack/uninstall", body, {}, {}, None,
            )
        # Still acks - graceful degradation
        assert _status(result) == 200


# ============================================================================
# GET /api/integrations/slack/workspaces
# ============================================================================


class TestListWorkspaces:
    """Tests for the workspace listing endpoint."""

    @pytest.mark.asyncio
    async def test_returns_200(self, handler, mock_workspace_store):
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_returns_workspace_list(self, handler, mock_workspace_store):
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        body = _body(result)
        assert "workspaces" in body
        assert body["total"] == 1
        assert body["workspaces"][0]["workspace_id"] == "W123"

    @pytest.mark.asyncio
    async def test_workspace_fields_complete(self, handler, mock_workspace_store):
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        ws = _body(result)["workspaces"][0]
        expected_fields = [
            "workspace_id", "workspace_name", "is_active", "token_status",
            "token_expires_at", "installed_at", "installed_by", "scopes", "tenant_id",
        ]
        for field in expected_fields:
            assert field in ws, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_token_status_valid(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.token_expires_at = time.time() + 7200
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["workspaces"][0]["token_status"] == "valid"

    @pytest.mark.asyncio
    async def test_token_status_expired(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.token_expires_at = time.time() - 100
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["workspaces"][0]["token_status"] == "expired"

    @pytest.mark.asyncio
    async def test_token_status_expiring_soon(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.token_expires_at = time.time() + 1800  # 30 min
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["workspaces"][0]["token_status"] == "expiring_soon"

    @pytest.mark.asyncio
    async def test_token_status_no_expiration(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.token_expires_at = None
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["workspaces"][0]["token_status"] == "valid"

    @pytest.mark.asyncio
    async def test_store_import_error_returns_503(self, handler):
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_empty_workspace_list(self, handler, mock_workspace_store):
        mock_workspace_store.list_active.return_value = []
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["total"] == 0
        assert body["workspaces"] == []

    @pytest.mark.asyncio
    async def test_method_not_allowed_post(self, handler):
        result = await handler.handle(
            "POST", "/api/integrations/slack/workspaces", {}, {}, {}, None,
        )
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_store_value_error_returns_500(self, handler, mock_workspace_store):
        mock_workspace_store.list_active.side_effect = ValueError("bad data")
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_store_type_error_returns_500(self, handler, mock_workspace_store):
        mock_workspace_store.list_active.side_effect = TypeError("wrong type")
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
            )
        assert _status(result) == 500


# ============================================================================
# GET /api/integrations/slack/workspaces/{id}/status
# ============================================================================


class TestWorkspaceStatus:
    """Tests for the workspace status endpoint."""

    @pytest.mark.asyncio
    async def test_returns_200(self, handler, mock_workspace_store):
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_returns_workspace_details(self, handler, mock_workspace_store):
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["workspace_id"] == "W123"
        assert body["workspace_name"] == "Test Workspace"
        assert body["is_active"] is True

    @pytest.mark.asyncio
    async def test_status_fields_complete(self, handler, mock_workspace_store):
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        body = _body(result)
        expected_fields = [
            "workspace_id", "workspace_name", "is_active", "token_status",
            "token_expires_at", "expires_in_seconds", "has_refresh_token",
            "scopes", "installed_at", "installed_by", "bot_user_id", "tenant_id",
        ]
        for field in expected_fields:
            assert field in body, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_token_expired(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.token_expires_at = time.time() - 3600
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["token_status"] == "expired"
        assert body["expires_in_seconds"] < 0

    @pytest.mark.asyncio
    async def test_token_expiring_soon(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.token_expires_at = time.time() + 1800
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["token_status"] == "expiring_soon"

    @pytest.mark.asyncio
    async def test_token_expiring_today(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.token_expires_at = time.time() + 7200  # 2 hours
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["token_status"] == "expiring_today"

    @pytest.mark.asyncio
    async def test_token_no_expiration(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.token_expires_at = None
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["token_status"] == "valid"
        assert body["expires_in_seconds"] is None

    @pytest.mark.asyncio
    async def test_has_refresh_token(self, handler, mock_workspace_store):
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["has_refresh_token"] is True

    @pytest.mark.asyncio
    async def test_no_refresh_token(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.refresh_token = None
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        body = _body(result)
        assert body["has_refresh_token"] is False

    @pytest.mark.asyncio
    async def test_workspace_not_found(self, handler, mock_workspace_store):
        mock_workspace_store.get.return_value = None
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W999/status", {}, {}, {}, None,
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_store_import_error_returns_503(self, handler):
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_store_error_returns_500(self, handler, mock_workspace_store):
        mock_workspace_store.get.side_effect = ValueError("corrupt data")
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_method_not_allowed_post(self, handler):
        result = await handler.handle(
            "POST", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
        )
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_different_workspace_id_param(self, handler, mock_workspace_store, mock_workspace):
        """Workspace ID is extracted from path and passed to store."""
        mock_workspace.workspace_id = "W-CUSTOM-42"
        mock_workspace_store.get.return_value = mock_workspace
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "GET", "/api/integrations/slack/workspaces/W-CUSTOM-42/status", {}, {}, {}, None,
            )
        assert _status(result) == 200
        mock_workspace_store.get.assert_called_with("W-CUSTOM-42")


# ============================================================================
# POST /api/integrations/slack/workspaces/{id}/refresh
# ============================================================================


class TestRefreshToken:
    """Tests for the token refresh endpoint."""

    @pytest.mark.asyncio
    async def test_successful_refresh(self, handler, mock_workspace_store):
        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-new-token",
            "refresh_token": "xoxr-new-refresh",
            "expires_in": 43200,
        })

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["workspace_id"] == "W123"
        assert body["expires_in_seconds"] == 43200

    @pytest.mark.asyncio
    async def test_workspace_not_found(self, handler, mock_workspace_store):
        mock_workspace_store.get.return_value = None
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W999/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_no_refresh_token_returns_400(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.refresh_token = None
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 400
        body = _body(result)
        assert "refresh token" in body.get("error", "").lower() or "Re-installation" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_inactive_workspace_returns_400(self, handler, mock_workspace, mock_workspace_store):
        mock_workspace.is_active = False
        with patch(
            "aragora.storage.slack_workspace_store.get_slack_workspace_store",
            return_value=mock_workspace_store,
        ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_slack_api_error_during_refresh(self, handler, mock_workspace_store):
        mock_client, _ = _make_httpx_mock({"ok": False, "error": "token_expired"})

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 400
        body = _body(result)
        assert "token_expired" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_httpx_error_during_refresh(self, handler, mock_workspace_store):
        import httpx

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("connection failed"))

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 502

    @pytest.mark.asyncio
    async def test_save_failure_returns_500(self, handler, mock_workspace_store):
        mock_workspace_store.save.return_value = False

        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-new",
            "expires_in": 43200,
        })

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_store_import_error_returns_503(self, handler):
        with patch.dict("sys.modules", {"aragora.storage.slack_workspace_store": None}):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_method_not_allowed_get(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
        )
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_refresh_without_expires_in(self, handler, mock_workspace_store):
        """When Slack omits expires_in, token_expires_at should be None."""
        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-new-token",
        })

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body["token_expires_at"] is None
        assert body["expires_in_seconds"] is None

    @pytest.mark.asyncio
    async def test_refresh_updates_workspace_tokens(self, handler, mock_workspace, mock_workspace_store):
        """Verify that the workspace object's tokens are updated."""
        mock_client, _ = _make_httpx_mock({
            "ok": True,
            "access_token": "xoxb-updated-token",
            "refresh_token": "xoxr-updated-refresh",
            "expires_in": 86400,
        })

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 200
        assert mock_workspace.access_token == "xoxb-updated-token"
        assert mock_workspace.refresh_token == "xoxr-updated-refresh"
        mock_workspace_store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_refresh_connection_error_returns_500(self, handler, mock_workspace_store):
        """ConnectionError during refresh returns 500."""
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=ConnectionError("no route to host"))

        with patch("httpx.AsyncClient", return_value=mock_client), \
             patch(
                 "aragora.storage.slack_workspace_store.get_slack_workspace_store",
                 return_value=mock_workspace_store,
             ):
            result = await handler.handle(
                "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
            )
        assert _status(result) == 500


# ============================================================================
# Unknown / not-found routes
# ============================================================================


class TestNotFound:
    """Test that unmatched paths return 404."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/unknown", {}, {}, {}, None,
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_close_path_returns_404(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/settings", {}, {}, {}, None,
        )
        assert _status(result) == 404


# ============================================================================
# Permission enforcement
# ============================================================================


class TestPermissions:
    """Test RBAC permission checks on protected routes."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_install_requires_auth_in_production(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "ARAGORA_ENV", "production")
        result = await handler.handle(
            "GET", "/api/integrations/slack/install", {}, {}, {}, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_preview_requires_auth(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/preview", {}, {}, {}, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_workspaces_requires_auth(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/workspaces", {}, {}, {}, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_workspace_status_requires_auth(self, handler):
        result = await handler.handle(
            "GET", "/api/integrations/slack/workspaces/W123/status", {}, {}, {}, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_workspace_refresh_requires_auth(self, handler):
        result = await handler.handle(
            "POST", "/api/integrations/slack/workspaces/W123/refresh", {}, {}, {}, None,
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_callback_does_not_require_auth(self, handler):
        """Callback endpoint works even without auth (validates state instead)."""
        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {"error": "access_denied"}, {}, None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_install_in_dev_mode_skips_auth(self, handler, handler_module, monkeypatch):
        """In development mode, install endpoint works without auth."""
        monkeypatch.setattr(handler_module, "ARAGORA_ENV", "development")
        result = await handler.handle(
            "GET", "/api/integrations/slack/install", {}, {}, {}, None,
        )
        assert _status(result) == 302

    @pytest.mark.asyncio
    async def test_install_in_local_mode_skips_auth(self, handler, handler_module, monkeypatch):
        """In local mode, install endpoint works without auth."""
        monkeypatch.setattr(handler_module, "ARAGORA_ENV", "local")
        result = await handler.handle(
            "GET", "/api/integrations/slack/install", {}, {}, {}, None,
        )
        assert _status(result) == 302


# ============================================================================
# _check_permission helper
# ============================================================================


class TestCheckPermission:
    """Tests for the _check_permission method."""

    def test_no_permissions_returns_true(self, handler):
        assert handler._check_permission(MagicMock()) is True

    def test_require_all_true_all_pass(self, handler):
        with patch.object(handler, "check_permission", return_value=True):
            assert handler._check_permission(
                MagicMock(), "perm1", "perm2", require_all=True,
            ) is True

    def test_require_all_false_any_pass(self, handler):
        call_count = 0

        def side_effect(ctx, perm):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from aragora.server.handlers.secure import ForbiddenError

                raise ForbiddenError("nope", permission=perm)
            return True

        with patch.object(handler, "check_permission", side_effect=side_effect):
            assert handler._check_permission(MagicMock(), "perm1", "perm2") is True

    def test_require_all_false_none_pass_raises(self, handler):
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(
            handler, "check_permission",
            side_effect=ForbiddenError("denied", permission="perm1"),
        ):
            with pytest.raises(ForbiddenError):
                handler._check_permission(MagicMock(), "perm1", "perm2")

    def test_require_all_true_one_fails_raises(self, handler):
        from aragora.server.handlers.secure import ForbiddenError

        call_count = 0

        def side_effect(ctx, perm):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ForbiddenError("denied", permission=perm)
            return True

        with patch.object(handler, "check_permission", side_effect=side_effect):
            with pytest.raises(ForbiddenError):
                handler._check_permission(
                    MagicMock(), "perm1", "perm2", require_all=True,
                )


# ============================================================================
# Cleanup helper
# ============================================================================


class TestCleanupFallbackStates:
    """Tests for the _cleanup_oauth_states_fallback function."""

    def test_removes_expired_states(self, handler_module):
        now = time.time()
        handler_module._oauth_states_fallback["old"] = {
            "created_at": now - 99999,
            "tenant_id": None,
        }
        handler_module._oauth_states_fallback["fresh"] = {
            "created_at": now,
            "tenant_id": None,
        }
        handler_module._cleanup_oauth_states_fallback(now=now)
        assert "old" not in handler_module._oauth_states_fallback
        assert "fresh" in handler_module._oauth_states_fallback

    def test_no_states_noop(self, handler_module):
        handler_module._cleanup_oauth_states_fallback()
        assert handler_module._oauth_states_fallback == {}

    def test_all_expired_clears_all(self, handler_module):
        now = time.time()
        handler_module._oauth_states_fallback["a"] = {"created_at": now - 99999}
        handler_module._oauth_states_fallback["b"] = {"created_at": now - 88888}
        handler_module._cleanup_oauth_states_fallback(now=now)
        assert len(handler_module._oauth_states_fallback) == 0

    def test_uses_current_time_by_default(self, handler_module):
        handler_module._oauth_states_fallback["recent"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        handler_module._cleanup_oauth_states_fallback()
        # Should still be present - created just now
        assert "recent" in handler_module._oauth_states_fallback


# ============================================================================
# Audit logger helper
# ============================================================================


class TestAuditLogger:
    """Tests for the _get_oauth_audit_logger lazy init."""

    def test_returns_none_when_import_fails(self, handler_module):
        handler_module._slack_oauth_audit = None
        with patch.dict("sys.modules", {"aragora.audit.slack_audit": None}):
            result = handler_module._get_oauth_audit_logger()
        assert result is None

    def test_caches_logger(self, handler_module):
        handler_module._slack_oauth_audit = None
        mock_logger = MagicMock()
        with patch(
            "aragora.audit.slack_audit.get_slack_audit_logger",
            return_value=mock_logger,
        ):
            first = handler_module._get_oauth_audit_logger()
            second = handler_module._get_oauth_audit_logger()
        assert first is second is mock_logger

    def test_returns_existing_logger(self, handler_module):
        existing = MagicMock()
        handler_module._slack_oauth_audit = existing
        assert handler_module._get_oauth_audit_logger() is existing


# ============================================================================
# handle() calling conventions
# ============================================================================


class TestHandleCallingConventions:
    """Test the dual-signature handle() method."""

    @pytest.mark.asyncio
    async def test_direct_call_convention(self, handler):
        """Direct: handle(method, path, body, query_params, headers, handler)."""
        result = await handler.handle(
            "GET", "/api/integrations/slack/callback", {},
            {"error": "denied"}, {}, None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_registry_call_convention(self, handler):
        """Registry: handle(path, query_params, handler)."""
        mock_h = MagicMock()
        mock_h.command = "GET"
        mock_h.headers = {"Content-Length": "0"}
        mock_h.rfile = MagicMock()
        mock_h.rfile.read.return_value = b"{}"

        result = await handler.handle(
            "/api/integrations/slack/callback",
            {"error": ["denied"]},
            mock_h,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_registry_call_with_body(self, handler, monkeypatch):
        """Registry call that reads body from handler."""
        monkeypatch.setenv("ARAGORA_ENV", "test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "")

        body_dict = {"event": {"type": "unknown"}}
        body_bytes = json.dumps(body_dict).encode()

        mock_h = MagicMock()
        mock_h.command = "POST"
        mock_h.headers = {"Content-Length": str(len(body_bytes))}
        mock_h.rfile = MagicMock()
        mock_h.rfile.read.return_value = body_bytes

        result = await handler.handle(
            "/api/integrations/slack/uninstall",
            {},
            mock_h,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_registry_call_with_dict_query_params(self, handler):
        """Registry call normalizes list-valued query params."""
        result = await handler.handle(
            "/api/integrations/slack/callback",
            {"error": ["access_denied"], "extra": "val"},
            None,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_registry_call_no_handler(self, handler):
        """Registry call with no handler defaults to GET."""
        result = await handler.handle(
            "/api/integrations/slack/callback",
            {"error": ["denied"]},
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_registry_call_invalid_body_json(self, handler, monkeypatch):
        """When handler body is not valid JSON, body defaults to {}."""
        monkeypatch.setenv("ARAGORA_ENV", "test")
        monkeypatch.setenv("SLACK_SIGNING_SECRET", "")

        mock_h = MagicMock()
        mock_h.command = "POST"
        mock_h.headers = {"Content-Length": "10"}
        mock_h.rfile = MagicMock()
        mock_h.rfile.read.return_value = b"not-json!!"

        result = await handler.handle(
            "/api/integrations/slack/uninstall",
            {},
            mock_h,
        )
        # Should still process (body will be {} with no event)
        assert _status(result) == 200


# ============================================================================
# Constants and module-level setup
# ============================================================================


class TestConstants:
    """Verify module-level constants are correctly defined."""

    def test_permission_constants(self, handler_module):
        assert handler_module.PERM_SLACK_OAUTH_INSTALL == "slack:oauth:install"
        assert handler_module.PERM_SLACK_OAUTH_CALLBACK == "slack:oauth:callback"
        assert handler_module.PERM_SLACK_OAUTH_DISCONNECT == "slack:oauth:disconnect"
        assert handler_module.PERM_SLACK_WORKSPACE_MANAGE == "slack:workspace:manage"
        assert handler_module.PERM_SLACK_ADMIN == "slack:admin"

    def test_legacy_permission_constants(self, handler_module):
        assert handler_module.CONNECTOR_READ == "connectors.read"
        assert handler_module.CONNECTOR_AUTHORIZE == "connectors.authorize"

    def test_default_scopes_defined(self, handler_module):
        assert "channels:history" in handler_module.DEFAULT_SCOPES
        assert "chat:write" in handler_module.DEFAULT_SCOPES

    def test_slack_oauth_urls(self, handler_module):
        assert "slack.com/oauth/v2/authorize" in handler_module.SLACK_OAUTH_AUTHORIZE_URL
        assert "slack.com/api/oauth.v2.access" in handler_module.SLACK_OAUTH_TOKEN_URL

    def test_scope_descriptions_dict(self, handler_module):
        assert "channels:history" in handler_module.SCOPE_DESCRIPTIONS
        assert "chat:write" in handler_module.SCOPE_DESCRIPTIONS
        desc = handler_module.SCOPE_DESCRIPTIONS["channels:history"]
        assert "name" in desc
        assert "description" in desc
        assert "required" in desc

    def test_scope_descriptions_all_have_required_fields(self, handler_module):
        for scope, desc in handler_module.SCOPE_DESCRIPTIONS.items():
            assert "name" in desc, f"Missing 'name' in {scope}"
            assert "description" in desc, f"Missing 'description' in {scope}"
            assert "required" in desc, f"Missing 'required' in {scope}"

    def test_routes_list(self, handler_module):
        assert len(handler_module.SlackOAuthHandler.ROUTES) >= 5

    def test_route_patterns_list(self, handler_module):
        assert len(handler_module.SlackOAuthHandler.ROUTE_PATTERNS) == 2
