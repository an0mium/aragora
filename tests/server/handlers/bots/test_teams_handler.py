"""Tests for Microsoft Teams bot handler."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.teams import TeamsHandler


# =============================================================================
# Test Handler Initialization
# =============================================================================


class TestTeamsHandlerInit:
    """Tests for Teams handler initialization."""

    def test_handler_routes(self):
        """Should define correct routes."""
        handler = TeamsHandler({})
        assert "/api/v1/bots/teams/messages" in handler.ROUTES
        assert "/api/v1/bots/teams/status" in handler.ROUTES

    def test_can_handle_messages_route(self):
        """Should handle messages route."""
        handler = TeamsHandler({})
        assert handler.can_handle("/api/v1/bots/teams/messages") is True

    def test_can_handle_status_route(self):
        """Should handle status route."""
        handler = TeamsHandler({})
        assert handler.can_handle("/api/v1/bots/teams/status") is True

    def test_cannot_handle_unknown_route(self):
        """Should not handle unknown routes."""
        handler = TeamsHandler({})
        assert handler.can_handle("/api/v1/bots/unknown") is False


# =============================================================================
# Test Status Endpoint
# =============================================================================


class TestTeamsStatus:
    """Tests for Teams status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_without_credentials(self):
        """Should return status showing not configured when no credentials."""
        handler = TeamsHandler({})

        # Mock auth context
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=["bots.read"])
            with patch.object(handler, "check_permission"):
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "enabled" in body
        assert "app_id_configured" in body
        assert "password_configured" in body
        assert "sdk_available" in body

    @pytest.mark.asyncio
    async def test_get_status_requires_auth(self):
        """Should require authentication for status endpoint."""
        from aragora.server.handlers.secure import UnauthorizedError

        handler = TeamsHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("No auth")
            mock_handler = MagicMock()
            result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_get_status_requires_permission(self):
        """Should require bots.read permission."""
        from aragora.server.handlers.secure import ForbiddenError

        handler = TeamsHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock()
            with patch.object(handler, "check_permission") as mock_check:
                mock_check.side_effect = ForbiddenError("Missing permission")
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 403


# =============================================================================
# Test Messages Endpoint
# =============================================================================


class TestTeamsMessages:
    """Tests for Teams message webhook endpoint."""

    @pytest.mark.asyncio
    async def test_handle_post_without_bot(self):
        """Should return 503 when bot not configured."""
        handler = TeamsHandler({})

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": "2"}
        mock_request.rfile.read.return_value = b"{}"

        result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body
        assert "not configured" in body["error"]

    @pytest.mark.asyncio
    async def test_handle_post_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        handler = TeamsHandler({})

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": "12"}
        mock_request.rfile.read.return_value = b"not valid json"

        # Mock bot to be available - return an async mock
        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_bot:
            mock_bot.return_value = MagicMock()
            result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_unknown_path(self):
        """Should return None for unknown paths."""
        handler = TeamsHandler({})
        mock_request = MagicMock()

        result = await handler.handle_post("/api/v1/bots/unknown", {}, mock_request)

        assert result is None


# =============================================================================
# Test Bot Framework Availability Check
# =============================================================================


class TestBotFrameworkCheck:
    """Tests for Bot Framework SDK availability check."""

    def test_check_botframework_not_available(self):
        """Should detect when Bot Framework SDK is not installed."""
        from aragora.server.handlers.bots.teams import _check_botframework_available

        # This may pass or fail depending on whether botbuilder is installed
        available, error = _check_botframework_available()
        assert isinstance(available, bool)
        if not available:
            assert error is not None


# =============================================================================
# Test Lazy Bot Initialization
# =============================================================================


class TestBotLazyInit:
    """Tests for lazy bot initialization."""

    @pytest.mark.asyncio
    async def test_ensure_bot_without_credentials(self):
        """Should return None when credentials not configured."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", ""):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", ""):
                result = await handler._ensure_bot()

        assert result is None

    @pytest.mark.asyncio
    async def test_ensure_bot_caches_result(self):
        """Should cache bot initialization result."""
        handler = TeamsHandler({})
        handler._bot_initialized = True
        handler._bot = "cached_bot"

        result = await handler._ensure_bot()

        assert result == "cached_bot"
