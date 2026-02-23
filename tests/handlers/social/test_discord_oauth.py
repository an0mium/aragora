"""Tests for Discord OAuth handler (aragora/server/handlers/social/discord_oauth.py).

Covers all routes and behavior of the DiscordOAuthHandler class:
- can_handle() route matching for all static routes
- GET  /api/integrations/discord/install   - Initiate OAuth flow
- GET  /api/integrations/discord/callback  - Handle OAuth callback from Discord
- POST /api/integrations/discord/uninstall - Handle guild removal webhook
- Method not allowed responses
- Permission denied paths
- Error handling and edge cases
- OAuth state management
- Dual-signature handle() calling conventions
"""

from __future__ import annotations

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
    import aragora.server.handlers.social.discord_oauth as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.DiscordOAuthHandler


@pytest.fixture
def handler(handler_cls):
    """Create a DiscordOAuthHandler with empty context."""
    return handler_cls(ctx={})


@pytest.fixture(autouse=True)
def _reset_module_globals(handler_module):
    """Reset module-level globals between tests so state does not leak."""
    handler_module._oauth_states.clear()
    yield
    handler_module._oauth_states.clear()


@pytest.fixture(autouse=True)
def _patch_env(monkeypatch, handler_module):
    """Set default Discord OAuth credentials for tests."""
    monkeypatch.setattr(handler_module, "DISCORD_CLIENT_ID", "test-discord-client-id")
    monkeypatch.setattr(handler_module, "DISCORD_CLIENT_SECRET", "test-discord-secret")
    monkeypatch.setattr(
        handler_module,
        "DISCORD_REDIRECT_URI",
        "https://example.com/api/integrations/discord/callback",
    )
    monkeypatch.setattr(handler_module, "DISCORD_SCOPES", "bot applications.commands")
    monkeypatch.setattr(handler_module, "DISCORD_BOT_PERMISSIONS", "274877975616")


# ---------------------------------------------------------------------------
# httpx mock helpers
# ---------------------------------------------------------------------------


def _make_httpx_mock(response_json: dict, status_code: int = 200):
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
    mock_client.get = AsyncMock(return_value=mock_response)
    return mock_client, mock_response


# ---------------------------------------------------------------------------
# Mock guild store helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_guild():
    """Create a mock DiscordGuild object."""
    guild = MagicMock()
    guild.guild_id = "G123456"
    guild.guild_name = "Test Guild"
    guild.access_token = "discord-access-token"
    guild.refresh_token = "discord-refresh-token"
    guild.bot_user_id = "B999"
    guild.installed_at = time.time()
    guild.installed_by = None
    guild.scopes = ["bot", "applications.commands"]
    guild.tenant_id = "tenant-1"
    guild.is_active = True
    guild.expires_at = time.time() + 604800
    return guild


@pytest.fixture
def mock_guild_store(mock_guild):
    """Create a mock guild store."""
    store = MagicMock()
    store.save.return_value = True
    store.get.return_value = mock_guild
    store.deactivate.return_value = True
    return store


# ============================================================================
# can_handle routing
# ============================================================================


class TestCanHandle:
    """Verify that can_handle correctly accepts or rejects paths."""

    def test_install_path(self, handler):
        assert handler.can_handle("/api/integrations/discord/install")

    def test_callback_path(self, handler):
        assert handler.can_handle("/api/integrations/discord/callback")

    def test_uninstall_path(self, handler):
        assert handler.can_handle("/api/integrations/discord/uninstall")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_path(self, handler):
        assert not handler.can_handle("/api/integrations/discord")

    def test_rejects_extra_suffix(self, handler):
        assert not handler.can_handle("/api/integrations/discord/install/extra")

    def test_rejects_wrong_prefix(self, handler):
        assert not handler.can_handle("/api/v1/integrations/discord/install")

    def test_rejects_typo(self, handler):
        assert not handler.can_handle("/api/integrations/discrd/install")

    def test_rejects_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_slack_path(self, handler):
        assert not handler.can_handle("/api/integrations/slack/install")

    def test_rejects_teams_path(self, handler):
        assert not handler.can_handle("/api/integrations/teams/install")


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
        assert len(handler.ROUTES) == 3

    def test_routes_list(self, handler):
        expected = [
            "/api/integrations/discord/install",
            "/api/integrations/discord/callback",
            "/api/integrations/discord/uninstall",
        ]
        for route in expected:
            assert route in handler.ROUTES


class TestFactory:
    """Test the factory function."""

    def test_create_handler(self, handler_module):
        h = handler_module.create_discord_oauth_handler({"server": True})
        assert isinstance(h, handler_module.DiscordOAuthHandler)


# ============================================================================
# GET /api/integrations/discord/install
# ============================================================================


class TestInstall:
    """Tests for the /install endpoint."""

    @pytest.mark.asyncio
    async def test_returns_302_redirect(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        assert _status(result) == 302

    @pytest.mark.asyncio
    async def test_redirect_location_contains_discord_oauth_url(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        location = result.headers.get("Location", "")
        assert "discord.com/api/oauth2/authorize" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_client_id(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        location = result.headers.get("Location", "")
        assert "client_id=test-discord-client-id" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_state(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        location = result.headers.get("Location", "")
        assert "state=" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_redirect_uri(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        location = result.headers.get("Location", "")
        assert "redirect_uri=" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_scopes(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        location = result.headers.get("Location", "")
        assert "scope=" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_permissions(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        location = result.headers.get("Location", "")
        assert "permissions=" in location

    @pytest.mark.asyncio
    async def test_redirect_location_contains_response_type_code(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        location = result.headers.get("Location", "")
        assert "response_type=code" in location

    @pytest.mark.asyncio
    async def test_cache_control_no_store(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        assert result.headers.get("Cache-Control") == "no-store"

    @pytest.mark.asyncio
    async def test_content_type_html(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        assert "text/html" in result.content_type

    @pytest.mark.asyncio
    async def test_state_stored_in_memory(self, handler, handler_module):
        await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        assert len(handler_module._oauth_states) == 1

    @pytest.mark.asyncio
    async def test_state_contains_created_at(self, handler, handler_module):
        await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        state_key = list(handler_module._oauth_states.keys())[0]
        state_data = handler_module._oauth_states[state_key]
        assert "created_at" in state_data
        assert isinstance(state_data["created_at"], float)

    @pytest.mark.asyncio
    async def test_state_contains_tenant_id(self, handler, handler_module):
        await handler.handle(
            "/api/integrations/discord/install",
            {"tenant_id": "t-abc"},
            None,
            method="GET",
        )
        state_key = list(handler_module._oauth_states.keys())[0]
        state_data = handler_module._oauth_states[state_key]
        assert state_data["tenant_id"] == "t-abc"

    @pytest.mark.asyncio
    async def test_state_tenant_id_none_when_not_provided(self, handler, handler_module):
        await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        state_key = list(handler_module._oauth_states.keys())[0]
        state_data = handler_module._oauth_states[state_key]
        assert state_data["tenant_id"] is None

    @pytest.mark.asyncio
    async def test_no_client_id_returns_503(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "DISCORD_CLIENT_ID", None)
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_no_client_id_error_message(self, handler, handler_module, monkeypatch):
        monkeypatch.setattr(handler_module, "DISCORD_CLIENT_ID", None)
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        body = _body(result)
        assert "not configured" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_method_not_allowed_post(self, handler):
        result = await handler.handle("/api/integrations/discord/install", {}, None, method="POST")
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_expired_states_cleaned_up(self, handler, handler_module):
        """Old states (>10 min) should be cleaned up during install."""
        old_time = time.time() - 700  # 11+ minutes ago
        handler_module._oauth_states["old-state"] = {
            "created_at": old_time,
            "tenant_id": None,
        }
        handler_module._oauth_states["fresh-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        await handler.handle("/api/integrations/discord/install", {}, None, method="GET")
        assert "old-state" not in handler_module._oauth_states
        assert "fresh-state" in handler_module._oauth_states
        # Plus the new state just created
        assert len(handler_module._oauth_states) == 2

    @pytest.mark.asyncio
    async def test_fallback_redirect_uri_localhost(self, handler, handler_module, monkeypatch):
        """When DISCORD_REDIRECT_URI is unset, localhost fallback should be used."""
        monkeypatch.setattr(handler_module, "DISCORD_REDIRECT_URI", None)
        result = await handler.handle(
            "/api/integrations/discord/install",
            {"host": "localhost:8080"},
            None,
            method="GET",
        )
        assert _status(result) == 302
        location = result.headers.get("Location", "")
        assert "localhost" in location

    @pytest.mark.asyncio
    async def test_fallback_redirect_uri_uses_http_for_localhost(
        self, handler, handler_module, monkeypatch
    ):
        monkeypatch.setattr(handler_module, "DISCORD_REDIRECT_URI", None)
        result = await handler.handle(
            "/api/integrations/discord/install",
            {"host": "localhost:3000"},
            None,
            method="GET",
        )
        assert _status(result) == 302
        location = result.headers.get("Location", "")
        assert "http%3A%2F%2Flocalhost" in location or "http://localhost" in location

    @pytest.mark.asyncio
    async def test_fallback_redirect_uri_uses_https_for_non_localhost(
        self, handler, handler_module, monkeypatch
    ):
        monkeypatch.setattr(handler_module, "DISCORD_REDIRECT_URI", None)
        result = await handler.handle(
            "/api/integrations/discord/install",
            {"host": "myapp.example.com"},
            None,
            method="GET",
        )
        assert _status(result) == 302
        location = result.headers.get("Location", "")
        assert "https" in location


# ============================================================================
# GET /api/integrations/discord/callback
# ============================================================================


class TestCallback:
    """Tests for the OAuth callback endpoint."""

    @pytest.mark.asyncio
    async def test_error_param_returns_400(self, handler):
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"error": "access_denied"},
            None,
            method="GET",
        )
        assert _status(result) == 400
        body = _body(result)
        assert "access_denied" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_error_with_description(self, handler):
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"error": "access_denied", "error_description": "User denied the request"},
            None,
            method="GET",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_missing_code_returns_400(self, handler):
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"state": "some-state"},
            None,
            method="GET",
        )
        assert _status(result) == 400
        body = _body(result)
        assert "code" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_state_returns_400(self, handler):
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"code": "auth-code"},
            None,
            method="GET",
        )
        assert _status(result) == 400
        body = _body(result)
        assert "state" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_missing_both_code_and_state_returns_400(self, handler):
        result = await handler.handle("/api/integrations/discord/callback", {}, None, method="GET")
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_state_returns_400(self, handler):
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"code": "auth-code", "state": "invalid-state"},
            None,
            method="GET",
        )
        assert _status(result) == 400
        body = _body(result)
        assert (
            "state" in body.get("error", "").lower() or "expired" in body.get("error", "").lower()
        )

    @pytest.mark.asyncio
    async def test_state_consumed_on_use(self, handler, handler_module):
        """State token should be removed after being used."""
        handler_module._oauth_states["test-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        # This will fail at token exchange but the state should still be consumed
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"code": "auth-code", "state": "test-state"},
            None,
            method="GET",
        )
        assert "test-state" not in handler_module._oauth_states

    @pytest.mark.asyncio
    async def test_successful_callback(self, handler, handler_module, mock_guild_store):
        """Full successful callback flow."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": "tenant-1",
        }

        token_response = {
            "access_token": "discord-access-token",
            "refresh_token": "discord-refresh-token",
            "expires_in": 604800,
            "scope": "bot applications.commands",
            "guild": {"id": "G123", "name": "My Discord Server"},
        }
        mock_client, _ = _make_httpx_mock(token_response)

        # Mock the /users/@me response
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-user-123"}
        mock_client.get = AsyncMock(return_value=me_response)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                return_value=MagicMock(),
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "auth-code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 200
        html = _html(result)
        assert "Connected" in html
        assert "My Discord Server" in html

    @pytest.mark.asyncio
    async def test_callback_html_content_type(self, handler, handler_module, mock_guild_store):
        """Successful callback returns text/html."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }

        token_response = {
            "access_token": "token",
            "refresh_token": "refresh",
            "expires_in": 604800,
            "scope": "bot",
            "guild": {"id": "G123", "name": "Guild"},
        }
        mock_client, _ = _make_httpx_mock(token_response)
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                return_value=MagicMock(),
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert "text/html" in result.content_type

    @pytest.mark.asyncio
    async def test_callback_no_credentials_returns_503(self, handler, handler_module, monkeypatch):
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        monkeypatch.setattr(handler_module, "DISCORD_CLIENT_ID", None)
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"code": "code", "state": "valid-state"},
            None,
            method="GET",
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_no_secret_returns_503(self, handler, handler_module, monkeypatch):
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        monkeypatch.setattr(handler_module, "DISCORD_CLIENT_SECRET", None)
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"code": "code", "state": "valid-state"},
            None,
            method="GET",
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_httpx_import_error(self, handler, handler_module):
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        with patch.dict("sys.modules", {"httpx": None}):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_token_exchange_connection_error(self, handler, handler_module):
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=ConnectionError("network down"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_token_exchange_timeout(self, handler, handler_module):
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=TimeoutError("timed out"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_token_exchange_os_error(self, handler, handler_module):
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=OSError("socket error"))

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_no_access_token_returns_500(self, handler, handler_module):
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        # Response without access_token
        mock_client, _ = _make_httpx_mock(
            {
                "refresh_token": "rt",
                "expires_in": 604800,
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_no_guild_id_returns_500(self, handler, handler_module):
        """When no guild_id can be determined, returns 500."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        # No guild in response and no guild_id in query params
        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "refresh_token": "rt",
                "expires_in": 604800,
                "scope": "bot",
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_guild_id_from_query_param(
        self, handler, handler_module, mock_guild_store
    ):
        """Guild ID can come from query params instead of token response."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "refresh_token": "rt",
                "expires_in": 604800,
                "scope": "bot",
                # No guild in response
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                return_value=MagicMock(),
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state", "guild_id": "G-FROM-QP"},
                None,
                method="GET",
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_guild_store_import_error(self, handler, handler_module):
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "refresh_token": "rt",
                "expires_in": 604800,
                "scope": "bot",
                "guild": {"id": "G123", "name": "Guild"},
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch.dict("sys.modules", {"aragora.storage.discord_guild_store": None}),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_callback_guild_save_fails(self, handler, handler_module, mock_guild_store):
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        mock_guild_store.save.return_value = False

        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "refresh_token": "rt",
                "expires_in": 604800,
                "scope": "bot",
                "guild": {"id": "G123", "name": "Save Fail Guild"},
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                return_value=MagicMock(),
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_callback_me_endpoint_failure_fallback(
        self, handler, handler_module, mock_guild_store
    ):
        """If /users/@me fails, bot_user_id falls back to DISCORD_CLIENT_ID."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "refresh_token": "rt",
                "expires_in": 604800,
                "scope": "bot",
                "guild": {"id": "G123", "name": "Guild"},
            }
        )
        # /users/@me raises an error
        mock_client.get = AsyncMock(side_effect=ConnectionError("api down"))

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                return_value=MagicMock(),
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_method_not_allowed_post(self, handler):
        result = await handler.handle("/api/integrations/discord/callback", {}, None, method="POST")
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_callback_with_tenant_id_from_state(
        self, handler, handler_module, mock_guild_store
    ):
        """Tenant ID from state data is passed through to guild creation."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": "t-from-state",
        }
        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "refresh_token": "rt",
                "expires_in": 604800,
                "scope": "bot",
                "guild": {"id": "G123", "name": "Guild"},
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        mock_guild_cls = MagicMock()
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                mock_guild_cls,
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 200
        # Verify tenant_id was passed to DiscordGuild constructor
        call_kwargs = mock_guild_cls.call_args[1]
        assert call_kwargs["tenant_id"] == "t-from-state"

    @pytest.mark.asyncio
    async def test_callback_redirect_uri_localhost_restriction(
        self, handler, handler_module, monkeypatch
    ):
        """When DISCORD_REDIRECT_URI is not set, non-localhost hosts are rejected."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        monkeypatch.setattr(handler_module, "DISCORD_REDIRECT_URI", None)
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"code": "code", "state": "valid-state", "host": "evil.com"},
            None,
            method="GET",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_callback_redirect_uri_localhost_allowed(
        self, handler, handler_module, monkeypatch, mock_guild_store
    ):
        """Localhost is allowed when DISCORD_REDIRECT_URI is not set."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        monkeypatch.setattr(handler_module, "DISCORD_REDIRECT_URI", None)

        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "refresh_token": "rt",
                "expires_in": 604800,
                "scope": "bot",
                "guild": {"id": "G123", "name": "Guild"},
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                return_value=MagicMock(),
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state", "host": "localhost:8080"},
                None,
                method="GET",
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_redirect_uri_127_0_0_1_allowed(
        self, handler, handler_module, monkeypatch, mock_guild_store
    ):
        """127.0.0.1 is allowed when DISCORD_REDIRECT_URI is not set."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        monkeypatch.setattr(handler_module, "DISCORD_REDIRECT_URI", None)

        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "refresh_token": "rt",
                "expires_in": 604800,
                "scope": "bot",
                "guild": {"id": "G123", "name": "Guild"},
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                return_value=MagicMock(),
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state", "host": "127.0.0.1:8080"},
                None,
                method="GET",
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_redirect_uri_ipv6_localhost_allowed(
        self, handler, handler_module, monkeypatch, mock_guild_store
    ):
        """[::1] is allowed when DISCORD_REDIRECT_URI is not set."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        monkeypatch.setattr(handler_module, "DISCORD_REDIRECT_URI", None)

        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "refresh_token": "rt",
                "expires_in": 604800,
                "scope": "bot",
                "guild": {"id": "G123", "name": "Guild"},
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                return_value=MagicMock(),
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state", "host": "[::1]:8080"},
                None,
                method="GET",
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_callback_default_expires_in(self, handler, handler_module, mock_guild_store):
        """When expires_in is not in the response, default of 604800 is used."""
        handler_module._oauth_states["valid-state"] = {
            "created_at": time.time(),
            "tenant_id": None,
        }
        mock_client, _ = _make_httpx_mock(
            {
                "access_token": "token",
                "scope": "bot",
                "guild": {"id": "G123", "name": "Guild"},
                # No expires_in
            }
        )
        me_response = MagicMock()
        me_response.status_code = 200
        me_response.json.return_value = {"id": "bot-id"}
        mock_client.get = AsyncMock(return_value=me_response)

        mock_guild_cls = MagicMock()
        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            patch(
                "aragora.storage.discord_guild_store.get_discord_guild_store",
                return_value=mock_guild_store,
            ),
            patch(
                "aragora.storage.discord_guild_store.DiscordGuild",
                mock_guild_cls,
            ),
        ):
            result = await handler.handle(
                "/api/integrations/discord/callback",
                {"code": "code", "state": "valid-state"},
                None,
                method="GET",
            )
        assert _status(result) == 200


# ============================================================================
# POST /api/integrations/discord/uninstall
# ============================================================================


class TestUninstall:
    """Tests for the bot removal webhook endpoint."""

    @pytest.mark.asyncio
    async def test_successful_uninstall(self, handler, mock_guild_store):
        with patch(
            "aragora.storage.discord_guild_store.get_discord_guild_store",
            return_value=mock_guild_store,
        ):
            result = await handler.handle(
                "/api/integrations/discord/uninstall",
                {},
                None,
                method="POST",
                body={"guild_id": "G123"},
            )
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True
        assert body.get("guild_id") == "G123"
        mock_guild_store.deactivate.assert_called_once_with("G123")

    @pytest.mark.asyncio
    async def test_missing_guild_id_returns_400(self, handler):
        result = await handler.handle(
            "/api/integrations/discord/uninstall",
            {},
            None,
            method="POST",
            body={},
        )
        assert _status(result) == 400
        body = _body(result)
        assert "guild_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_store_import_error_handled_gracefully(self, handler):
        """Uninstall should still return success even if store import fails."""
        with patch.dict("sys.modules", {"aragora.storage.discord_guild_store": None}):
            result = await handler.handle(
                "/api/integrations/discord/uninstall",
                {},
                None,
                method="POST",
                body={"guild_id": "G123"},
            )
        assert _status(result) == 200
        body = _body(result)
        assert body.get("ok") is True

    @pytest.mark.asyncio
    async def test_method_not_allowed_get(self, handler):
        result = await handler.handle("/api/integrations/discord/uninstall", {}, None, method="GET")
        assert _status(result) == 405

    @pytest.mark.asyncio
    async def test_uninstall_with_handler_body(self, handler, mock_guild_store):
        """Uninstall should work when body comes from mock HTTP handler."""
        body_dict = {"guild_id": "G999"}
        body_bytes = json.dumps(body_dict).encode()

        mock_h = MagicMock()
        mock_h.command = "POST"
        mock_h.headers = {"Content-Length": str(len(body_bytes))}
        mock_h.rfile = MagicMock()
        mock_h.rfile.read.return_value = body_bytes

        with patch(
            "aragora.storage.discord_guild_store.get_discord_guild_store",
            return_value=mock_guild_store,
        ):
            result = await handler.handle(
                "/api/integrations/discord/uninstall",
                {},
                mock_h,
            )
        assert _status(result) == 200
        body = _body(result)
        assert body.get("guild_id") == "G999"


# ============================================================================
# Unknown / not-found routes
# ============================================================================


class TestNotFound:
    """Test that unmatched paths return 404."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self, handler):
        result = await handler.handle("/api/integrations/discord/unknown", {}, None, method="GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_null_path_returns_404(self, handler):
        result = await handler.handle(method="GET")
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_integration_returns_404(self, handler):
        result = await handler.handle("/api/integrations/slack/install", {}, None, method="GET")
        assert _status(result) == 404


# ============================================================================
# Permission enforcement
# ============================================================================


class TestPermissions:
    """Test RBAC permission checks on protected routes."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_install_requires_auth(self, handler):
        result = await handler.handle(
            "/api/integrations/discord/install", {}, MagicMock(), method="GET"
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_callback_does_not_require_auth(self, handler):
        """Callback endpoint works even without auth (validates state instead)."""
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"error": "access_denied"},
            None,
            method="GET",
        )
        assert _status(result) == 400  # 400 not 401

    @pytest.mark.asyncio
    async def test_uninstall_does_not_require_auth(self, handler, mock_guild_store):
        """Uninstall webhook from Discord does not require auth."""
        with patch(
            "aragora.storage.discord_guild_store.get_discord_guild_store",
            return_value=mock_guild_store,
        ):
            result = await handler.handle(
                "/api/integrations/discord/uninstall",
                {},
                None,
                method="POST",
                body={"guild_id": "G123"},
            )
        assert _status(result) != 401

    @pytest.mark.asyncio
    async def test_install_permission_denied(self, handler):
        """When check_permission raises ForbiddenError, install returns 403."""
        from aragora.server.handlers.secure import ForbiddenError

        with patch.object(handler, "check_permission", side_effect=ForbiddenError("denied")):
            result = await handler.handle(
                "/api/integrations/discord/install", {}, None, method="GET"
            )
        assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_install_permission_error(self, handler):
        """When check_permission raises PermissionError, install returns 403."""
        with patch.object(handler, "check_permission", side_effect=PermissionError("denied")):
            result = await handler.handle(
                "/api/integrations/discord/install", {}, None, method="GET"
            )
        assert _status(result) == 403


# ============================================================================
# handle() calling conventions
# ============================================================================


class TestHandleCallingConventions:
    """Test the dual-signature handle() method."""

    @pytest.mark.asyncio
    async def test_kwargs_style(self, handler):
        """handle(path, query_params, handler, method=...)."""
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"error": "denied"},
            None,
            method="GET",
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_method_path_args(self, handler):
        """handle(method, path) where method looks like HTTP verb."""
        result = await handler.handle(
            "GET",
            "/api/integrations/discord/callback",
            {"error": "denied"},
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_path_only(self, handler):
        """handle(path, query_params) defaulting to GET."""
        result = await handler.handle(
            "/api/integrations/discord/callback",
            {"error": "denied"},
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handler_with_command_attr(self, handler, mock_guild_store):
        """When handler has .command attr, it determines the method."""
        body_dict = {"guild_id": "G123"}
        body_bytes = json.dumps(body_dict).encode()

        mock_h = MagicMock()
        mock_h.command = "POST"
        mock_h.headers = {"Content-Length": str(len(body_bytes))}
        mock_h.rfile = MagicMock()
        mock_h.rfile.read.return_value = body_bytes

        with patch(
            "aragora.storage.discord_guild_store.get_discord_guild_store",
            return_value=mock_guild_store,
        ):
            result = await handler.handle(
                "/api/integrations/discord/uninstall",
                {},
                mock_h,
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_body_param_kwarg(self, handler, mock_guild_store):
        """Body can be provided via body= kwarg."""
        with patch(
            "aragora.storage.discord_guild_store.get_discord_guild_store",
            return_value=mock_guild_store,
        ):
            result = await handler.handle(
                "/api/integrations/discord/uninstall",
                {},
                None,
                method="POST",
                body={"guild_id": "G-KWARG"},
            )
        assert _status(result) == 200
        body = _body(result)
        assert body.get("guild_id") == "G-KWARG"


# ============================================================================
# Module-level constants
# ============================================================================


class TestConstants:
    """Verify module-level constants are correctly defined."""

    def test_permission_constants(self, handler_module):
        assert handler_module.CONNECTOR_READ == "connectors.read"
        assert handler_module.CONNECTOR_AUTHORIZE == "connectors.authorize"

    def test_default_scopes(self, handler_module):
        assert handler_module.DEFAULT_SCOPES == "bot applications.commands"

    def test_default_permissions(self, handler_module):
        assert handler_module.DEFAULT_PERMISSIONS == "274877975616"

    def test_discord_oauth_urls(self, handler_module):
        assert "discord.com/api/oauth2/authorize" in handler_module.DISCORD_OAUTH_AUTHORIZE_URL
        assert "discord.com/api/oauth2/token" in handler_module.DISCORD_OAUTH_TOKEN_URL

    def test_discord_api_base(self, handler_module):
        assert "discord.com/api/v10" in handler_module.DISCORD_API_BASE

    def test_routes_list(self, handler_module):
        routes = handler_module.DiscordOAuthHandler.ROUTES
        assert "/api/integrations/discord/install" in routes
        assert "/api/integrations/discord/callback" in routes
        assert "/api/integrations/discord/uninstall" in routes
