"""
Tests for DiscordOAuthHandler - Discord OAuth bot installation flow.

Tests cover:
- Install endpoint (redirect to Discord)
- OAuth callback (token exchange, guild storage)
- Uninstall endpoint
- State token CSRF protection
- Error handling
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.discord_oauth import (
    DiscordOAuthHandler,
    _oauth_states,
    create_discord_oauth_handler,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def mock_server_context():
    """Create a mock server context."""
    return MagicMock()


@pytest.fixture
def oauth_handler(mock_server_context):
    """Create an OAuth handler for testing."""
    return DiscordOAuthHandler(mock_server_context)


@pytest.fixture
def cleanup_oauth_states():
    """Clean up OAuth states after tests."""
    yield
    _oauth_states.clear()


def parse_handler_response(result) -> Dict[str, Any]:
    """Parse handler result body as JSON."""
    if hasattr(result, "body") and result.body:
        body = result.body
        if isinstance(body, bytes):
            try:
                return json.loads(body.decode())
            except json.JSONDecodeError:
                return {}
        return json.loads(body) if body else {}
    return {}


# ===========================================================================
# Handler Routing Tests
# ===========================================================================


class TestDiscordOAuthHandlerRouting:
    """Tests for request routing."""

    def test_can_handle_install(self, oauth_handler):
        """Test can_handle for install endpoint."""
        assert oauth_handler.can_handle("/api/integrations/discord/install") is True

    def test_can_handle_callback(self, oauth_handler):
        """Test can_handle for callback endpoint."""
        assert oauth_handler.can_handle("/api/integrations/discord/callback") is True

    def test_can_handle_uninstall(self, oauth_handler):
        """Test can_handle for uninstall endpoint."""
        assert oauth_handler.can_handle("/api/integrations/discord/uninstall") is True

    def test_cannot_handle_other_paths(self, oauth_handler):
        """Test can_handle returns False for other paths."""
        assert oauth_handler.can_handle("/api/discord/install") is False
        assert oauth_handler.can_handle("/api/v2/discord/oauth") is False

    def test_routes_attribute(self, oauth_handler):
        """Test ROUTES includes all endpoints."""
        assert "/api/integrations/discord/install" in oauth_handler.ROUTES
        assert "/api/integrations/discord/callback" in oauth_handler.ROUTES
        assert "/api/integrations/discord/uninstall" in oauth_handler.ROUTES


# ===========================================================================
# Install Endpoint Tests
# ===========================================================================


class TestDiscordOAuthInstall:
    """Tests for OAuth install endpoint."""

    @pytest.mark.asyncio
    async def test_install_no_client_id(self, oauth_handler):
        """Test install without DISCORD_CLIENT_ID configured."""
        with patch("aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", ""):
            result = await oauth_handler.handle("GET", "/api/integrations/discord/install")

        assert result.status_code == 503
        data = parse_handler_response(result)
        assert "not configured" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_install_redirect(self, oauth_handler, cleanup_oauth_states):
        """Test install redirects to Discord OAuth."""
        with patch(
            "aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "test-client-id"
        ):
            result = await oauth_handler.handle("GET", "/api/integrations/discord/install")

        assert result.status_code == 302
        assert "Location" in result.headers
        assert "discord.com" in result.headers["Location"]
        assert "client_id=test-client-id" in result.headers["Location"]

    @pytest.mark.asyncio
    async def test_install_generates_state(self, oauth_handler, cleanup_oauth_states):
        """Test install generates state token."""
        initial_count = len(_oauth_states)

        with patch(
            "aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "test-client-id"
        ):
            result = await oauth_handler.handle("GET", "/api/integrations/discord/install")

        assert len(_oauth_states) == initial_count + 1
        assert "state=" in result.headers["Location"]

    @pytest.mark.asyncio
    async def test_install_with_tenant_id(self, oauth_handler, cleanup_oauth_states):
        """Test install stores tenant_id in state."""
        with patch(
            "aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "test-client-id"
        ):
            result = await oauth_handler.handle(
                "GET",
                "/api/integrations/discord/install",
                query_params={"tenant_id": "tenant-001"},
            )

        # Find the new state
        for state, data in _oauth_states.items():
            if data.get("tenant_id") == "tenant-001":
                assert True
                return

        pytest.fail("tenant_id not stored in state")

    @pytest.mark.asyncio
    async def test_install_cleans_old_states(self, oauth_handler, cleanup_oauth_states):
        """Test install cleans up expired states."""
        # Add old state
        old_state = "old-state-token"
        _oauth_states[old_state] = {"created_at": time.time() - 700}  # 11+ minutes old

        with patch(
            "aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "test-client-id"
        ):
            await oauth_handler.handle("GET", "/api/integrations/discord/install")

        assert old_state not in _oauth_states

    @pytest.mark.asyncio
    async def test_install_includes_permissions(self, oauth_handler, cleanup_oauth_states):
        """Test install includes bot permissions."""
        with patch(
            "aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "test-client-id"
        ):
            result = await oauth_handler.handle("GET", "/api/integrations/discord/install")

        assert "permissions=" in result.headers["Location"]

    @pytest.mark.asyncio
    async def test_install_method_not_allowed(self, oauth_handler):
        """Test install rejects non-GET methods."""
        result = await oauth_handler.handle("POST", "/api/integrations/discord/install")

        assert result.status_code == 405


# ===========================================================================
# Callback Endpoint Tests
# ===========================================================================


class TestDiscordOAuthCallback:
    """Tests for OAuth callback endpoint."""

    @pytest.mark.asyncio
    async def test_callback_error_from_discord(self, oauth_handler):
        """Test callback handles error from Discord."""
        result = await oauth_handler.handle(
            "GET",
            "/api/integrations/discord/callback",
            query_params={"error": "access_denied"},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "denied" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_missing_code(self, oauth_handler):
        """Test callback requires authorization code."""
        result = await oauth_handler.handle(
            "GET",
            "/api/integrations/discord/callback",
            query_params={"state": "some-state"},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "code" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_missing_state(self, oauth_handler):
        """Test callback requires state parameter."""
        result = await oauth_handler.handle(
            "GET",
            "/api/integrations/discord/callback",
            query_params={"code": "auth-code"},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "state" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_callback_invalid_state(self, oauth_handler, cleanup_oauth_states):
        """Test callback rejects invalid state token."""
        result = await oauth_handler.handle(
            "GET",
            "/api/integrations/discord/callback",
            query_params={"code": "auth-code", "state": "invalid-state"},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert (
            "expired" in data.get("error", "").lower() or "invalid" in data.get("error", "").lower()
        )

    @pytest.mark.asyncio
    async def test_callback_no_client_secret(self, oauth_handler, cleanup_oauth_states):
        """Test callback fails without client secret."""
        state = "valid-state"
        _oauth_states[state] = {"created_at": time.time()}

        with patch("aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "id"):
            with patch("aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_SECRET", ""):
                result = await oauth_handler.handle(
                    "GET",
                    "/api/integrations/discord/callback",
                    query_params={"code": "auth-code", "state": state},
                )

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_callback_token_exchange_error(self, oauth_handler, cleanup_oauth_states):
        """Test callback handles token exchange errors."""
        state = "valid-state"
        _oauth_states[state] = {"created_at": time.time()}

        with patch("aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "id"):
            with patch(
                "aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_SECRET", "secret"
            ):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                        side_effect=Exception("Network error")
                    )
                    result = await oauth_handler.handle(
                        "GET",
                        "/api/integrations/discord/callback",
                        query_params={"code": "auth-code", "state": state},
                    )

        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_callback_success(self, oauth_handler, cleanup_oauth_states):
        """Test successful callback stores guild."""
        state = "valid-state"
        _oauth_states[state] = {"created_at": time.time(), "tenant_id": "tenant-001"}

        mock_token_response = MagicMock()
        mock_token_response.json.return_value = {
            "access_token": "test-access-token",
            "refresh_token": "test-refresh-token",
            "expires_in": 604800,
            "scope": "bot applications.commands",
            "guild": {"id": "123456789", "name": "Test Server"},
        }
        mock_token_response.raise_for_status = MagicMock()

        mock_me_response = MagicMock()
        mock_me_response.status_code = 200
        mock_me_response.json.return_value = {"id": "bot-001"}

        mock_store = MagicMock()
        mock_store.save.return_value = True

        with patch("aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "id"):
            with patch(
                "aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_SECRET", "secret"
            ):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_instance = MagicMock()
                    mock_instance.post = AsyncMock(return_value=mock_token_response)
                    mock_instance.get = AsyncMock(return_value=mock_me_response)
                    mock_client.return_value.__aenter__.return_value = mock_instance
                    with patch(
                        "aragora.storage.discord_guild_store.get_discord_guild_store",
                        return_value=mock_store,
                    ):
                        result = await oauth_handler.handle(
                            "GET",
                            "/api/integrations/discord/callback",
                            query_params={"code": "auth-code", "state": state},
                        )

        assert result.status_code == 200
        assert result.content_type == "text/html"
        assert b"Connected" in result.body
        assert b"Test Server" in result.body
        mock_store.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_callback_method_not_allowed(self, oauth_handler):
        """Test callback rejects non-GET methods."""
        result = await oauth_handler.handle("POST", "/api/integrations/discord/callback")

        assert result.status_code == 405


# ===========================================================================
# Uninstall Endpoint Tests
# ===========================================================================


class TestDiscordOAuthUninstall:
    """Tests for uninstall endpoint."""

    @pytest.mark.asyncio
    async def test_uninstall_missing_guild_id(self, oauth_handler):
        """Test uninstall requires guild_id."""
        result = await oauth_handler.handle(
            "POST",
            "/api/integrations/discord/uninstall",
            body={},
        )

        assert result.status_code == 400
        data = parse_handler_response(result)
        assert "guild_id" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_uninstall_success(self, oauth_handler):
        """Test successful uninstall deactivates guild."""
        mock_store = MagicMock()

        with patch(
            "aragora.storage.discord_guild_store.get_discord_guild_store",
            return_value=mock_store,
        ):
            result = await oauth_handler.handle(
                "POST",
                "/api/integrations/discord/uninstall",
                body={"guild_id": "123456789"},
            )

        assert result.status_code == 200
        data = parse_handler_response(result)
        assert data.get("ok") is True
        assert data.get("guild_id") == "123456789"
        mock_store.deactivate.assert_called_once_with("123456789")

    @pytest.mark.asyncio
    async def test_uninstall_store_unavailable(self, oauth_handler):
        """Test uninstall handles store import error gracefully."""
        with patch(
            "aragora.storage.discord_guild_store.get_discord_guild_store",
            side_effect=ImportError("Store not available"),
        ):
            result = await oauth_handler.handle(
                "POST",
                "/api/integrations/discord/uninstall",
                body={"guild_id": "123456789"},
            )

        assert result.status_code == 200  # Still acknowledges request

    @pytest.mark.asyncio
    async def test_uninstall_method_not_allowed(self, oauth_handler):
        """Test uninstall rejects non-POST methods."""
        result = await oauth_handler.handle("GET", "/api/integrations/discord/uninstall")

        assert result.status_code == 405


# ===========================================================================
# Factory Function Tests
# ===========================================================================


class TestDiscordOAuthHandlerFactory:
    """Tests for handler factory function."""

    def test_create_discord_oauth_handler(self, mock_server_context):
        """Test factory creates handler."""
        handler = create_discord_oauth_handler(mock_server_context)

        assert isinstance(handler, DiscordOAuthHandler)


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestDiscordOAuthHandlerErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handle_not_found(self, oauth_handler):
        """Test handle returns 404 for unknown path."""
        result = await oauth_handler.handle("GET", "/api/integrations/discord/unknown")

        assert result.status_code == 404


# ===========================================================================
# State Token Tests
# ===========================================================================


class TestDiscordOAuthState:
    """Tests for OAuth state token handling."""

    @pytest.mark.asyncio
    async def test_state_consumed_after_callback(self, oauth_handler, cleanup_oauth_states):
        """Test state token is consumed after callback attempt."""
        state = "valid-state"
        _oauth_states[state] = {"created_at": time.time()}

        # Make callback fail early but still consume state
        with patch("aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "id"):
            with patch(
                "aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_SECRET", "secret"
            ):
                with patch("httpx.AsyncClient") as mock_client:
                    mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                        side_effect=Exception("Error")
                    )
                    await oauth_handler.handle(
                        "GET",
                        "/api/integrations/discord/callback",
                        query_params={"code": "code", "state": state},
                    )

        # State should be consumed (removed)
        assert state not in _oauth_states

    @pytest.mark.asyncio
    async def test_state_includes_timestamp(self, oauth_handler, cleanup_oauth_states):
        """Test state includes creation timestamp."""
        with patch("aragora.server.handlers.social.discord_oauth.DISCORD_CLIENT_ID", "id"):
            before = time.time()
            await oauth_handler.handle("GET", "/api/integrations/discord/install")
            after = time.time()

        for state, data in _oauth_states.items():
            assert "created_at" in data
            assert before <= data["created_at"] <= after
