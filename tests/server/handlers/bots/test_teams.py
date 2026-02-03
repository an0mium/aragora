"""Tests for Microsoft Teams bot handler.

Tests cover:
- Handler initialization and routing
- Status endpoint (RBAC protected)
- Message handling with Bot Framework
- Activity processing (messages, card actions)
- Authentication and error handling
- Bot lazy initialization
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.teams import (
    TeamsHandler,
    _check_botframework_available,
)

# Check if botbuilder SDK is available
HAS_BOTBUILDER = "botbuilder" in sys.modules or _check_botframework_available()[0]
requires_botbuilder = pytest.mark.skipif(not HAS_BOTBUILDER, reason="botbuilder SDK not installed")


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

    def test_cannot_handle_partial_match(self):
        """Should not handle partial route matches."""
        handler = TeamsHandler({})
        assert handler.can_handle("/api/v1/bots/teams") is False
        assert handler.can_handle("/api/v1/bots/teams/messages/extra") is False

    def test_bot_platform_attribute(self):
        """Should have correct bot_platform attribute."""
        handler = TeamsHandler({})
        assert handler.bot_platform == "teams"

    def test_initial_bot_state(self):
        """Should start with bot not initialized."""
        handler = TeamsHandler({})
        assert handler._bot is None
        assert handler._bot_initialized is False


# =============================================================================
# Test Status Endpoint
# =============================================================================


class TestTeamsStatus:
    """Tests for Teams status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_without_credentials(self):
        """Should return status showing not configured when no credentials."""
        handler = TeamsHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=["bots.read"])
            with patch.object(handler, "check_permission"):
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["platform"] == "teams"
        assert "enabled" in body
        assert "app_id_configured" in body
        assert "password_configured" in body
        assert "sdk_available" in body

    @pytest.mark.asyncio
    async def test_get_status_with_credentials_configured(self):
        """Should show enabled when credentials are configured."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "test-app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "test-password"):
                with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                    mock_auth.return_value = MagicMock(permissions=["bots.read"])
                    with patch.object(handler, "check_permission"):
                        mock_handler = MagicMock()
                        result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["app_id_configured"] is True
        assert body["password_configured"] is True

    @pytest.mark.asyncio
    async def test_get_status_requires_auth(self):
        """Should require authentication for status endpoint."""
        from aragora.server.handlers.utils.auth import UnauthorizedError

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
        from aragora.server.handlers.utils.auth import ForbiddenError

        handler = TeamsHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock()
            with patch.object(handler, "check_permission") as mock_check:
                mock_check.side_effect = ForbiddenError("Missing permission")
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_status_includes_sdk_error_when_unavailable(self):
        """Should include SDK error message when botbuilder not available."""
        handler = TeamsHandler({})

        with patch(
            "aragora.server.handlers.bots.teams._check_botframework_available",
            return_value=(False, "botbuilder-core not installed"),
        ):
            with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
                mock_auth.return_value = MagicMock(permissions=["bots.read"])
                with patch.object(handler, "check_permission"):
                    mock_handler = MagicMock()
                    result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)

        assert result is not None
        body = json.loads(result.body)
        assert body["sdk_available"] is False
        assert body["sdk_error"] == "botbuilder-core not installed"


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
        assert "details" in body

    @pytest.mark.asyncio
    async def test_handle_post_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        handler = TeamsHandler({})

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": "14"}
        mock_request.rfile.read.return_value = b"not valid json"

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_bot:
            mock_bot.return_value = MagicMock()
            result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_post_empty_body(self):
        """Should handle empty request body."""
        handler = TeamsHandler({})

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": "0"}
        mock_request.rfile.read.return_value = b""

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

    @requires_botbuilder
    @pytest.mark.asyncio
    async def test_handle_post_with_valid_activity(self):
        """Should process valid Bot Framework activity."""
        handler = TeamsHandler({})

        activity_data = {
            "type": "message",
            "id": "activity-123",
            "timestamp": "2024-01-15T10:00:00Z",
            "channelId": "msteams",
            "from": {"id": "user-123", "name": "Test User"},
            "conversation": {"id": "conv-123"},
            "recipient": {"id": "bot-123"},
            "text": "Hello bot!",
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(activity_data))),
            "Authorization": "Bearer test-token",
        }
        mock_request.rfile.read.return_value = json.dumps(activity_data).encode()

        mock_adapter = MagicMock()
        mock_adapter.authenticate_request = AsyncMock()
        mock_adapter.process_activity = AsyncMock()

        mock_bot = MagicMock()
        mock_bot.get_adapter.return_value = mock_adapter
        mock_bot.on_turn = AsyncMock()

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_bot
            with patch("botbuilder.schema.Activity") as mock_activity_cls:
                mock_activity = MagicMock()
                mock_activity_cls.deserialize.return_value = mock_activity
                result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Activity Processing
# =============================================================================


@requires_botbuilder
class TestActivityProcessing:
    """Tests for Bot Framework activity processing (requires botbuilder SDK)."""

    @pytest.mark.asyncio
    async def test_authentication_failure_invalid_token(self):
        """Should return 401 on invalid auth token."""
        handler = TeamsHandler({})

        activity_data = {"type": "message", "text": "test"}

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(activity_data))),
            "Authorization": "Bearer invalid-token",
        }
        mock_request.rfile.read.return_value = json.dumps(activity_data).encode()

        mock_adapter = MagicMock()
        mock_adapter.authenticate_request = AsyncMock(side_effect=ValueError("Invalid token"))

        mock_bot = MagicMock()
        mock_bot.get_adapter.return_value = mock_adapter

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_bot
            with patch("botbuilder.schema.Activity") as mock_activity_cls:
                mock_activity_cls.deserialize.return_value = MagicMock()
                result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 401
        body = json.loads(result.body)
        assert "Invalid authentication token" in body["error"]

    @pytest.mark.asyncio
    async def test_authentication_failure_unexpected_error(self):
        """Should return 401 on unexpected auth error."""
        handler = TeamsHandler({})

        activity_data = {"type": "message", "text": "test"}

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(activity_data))),
            "Authorization": "Bearer token",
        }
        mock_request.rfile.read.return_value = json.dumps(activity_data).encode()

        mock_adapter = MagicMock()
        mock_adapter.authenticate_request = AsyncMock(side_effect=RuntimeError("Unexpected error"))

        mock_bot = MagicMock()
        mock_bot.get_adapter.return_value = mock_adapter

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_bot
            with patch("botbuilder.schema.Activity") as mock_activity_cls:
                mock_activity_cls.deserialize.return_value = MagicMock()
                result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 401
        body = json.loads(result.body)
        assert "Unauthorized" in body["error"]

    @pytest.mark.asyncio
    async def test_activity_processing_data_error(self):
        """Should return 400 on data processing error."""
        handler = TeamsHandler({})

        activity_data = {"type": "message", "text": "test"}

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(activity_data))),
            "Authorization": "Bearer token",
        }
        mock_request.rfile.read.return_value = json.dumps(activity_data).encode()

        mock_adapter = MagicMock()
        mock_adapter.authenticate_request = AsyncMock()
        mock_adapter.process_activity = AsyncMock(side_effect=ValueError("Invalid activity data"))

        mock_bot = MagicMock()
        mock_bot.get_adapter.return_value = mock_adapter
        mock_bot.on_turn = AsyncMock()

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_bot
            with patch("botbuilder.schema.Activity") as mock_activity_cls:
                mock_activity_cls.deserialize.return_value = MagicMock()
                result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_activity_processing_internal_error(self):
        """Should return 500 on internal processing error."""
        handler = TeamsHandler({})

        activity_data = {"type": "message", "text": "test"}

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(activity_data))),
            "Authorization": "Bearer token",
        }
        mock_request.rfile.read.return_value = json.dumps(activity_data).encode()

        mock_adapter = MagicMock()
        mock_adapter.authenticate_request = AsyncMock()
        mock_adapter.process_activity = AsyncMock(side_effect=RuntimeError("Internal failure"))

        mock_bot = MagicMock()
        mock_bot.get_adapter.return_value = mock_adapter
        mock_bot.on_turn = AsyncMock()

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_bot
            with patch("botbuilder.schema.Activity") as mock_activity_cls:
                mock_activity_cls.deserialize.return_value = MagicMock()
                result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 500
        body = json.loads(result.body)
        assert "Internal processing error" in body["error"]


# =============================================================================
# Test Card Actions
# =============================================================================


@requires_botbuilder
class TestCardActions:
    """Tests for Teams Adaptive Card action handling (requires botbuilder SDK)."""

    @pytest.mark.asyncio
    async def test_handle_card_action_invoke(self):
        """Should handle invoke activity for card actions."""
        handler = TeamsHandler({})

        activity_data = {
            "type": "invoke",
            "name": "adaptiveCard/action",
            "value": {
                "action": "vote",
                "debateId": "debate-123",
                "vote": "agree",
            },
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(activity_data))),
            "Authorization": "Bearer token",
        }
        mock_request.rfile.read.return_value = json.dumps(activity_data).encode()

        mock_adapter = MagicMock()
        mock_adapter.authenticate_request = AsyncMock()
        mock_adapter.process_activity = AsyncMock()

        mock_bot = MagicMock()
        mock_bot.get_adapter.return_value = mock_adapter
        mock_bot.on_turn = AsyncMock()

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_bot
            with patch("botbuilder.schema.Activity") as mock_activity_cls:
                mock_activity_cls.deserialize.return_value = MagicMock()
                result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_message_reaction(self):
        """Should handle message reaction activity."""
        handler = TeamsHandler({})

        activity_data = {
            "type": "messageReaction",
            "reactionsAdded": [{"type": "like"}],
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(activity_data))),
            "Authorization": "Bearer token",
        }
        mock_request.rfile.read.return_value = json.dumps(activity_data).encode()

        mock_adapter = MagicMock()
        mock_adapter.authenticate_request = AsyncMock()
        mock_adapter.process_activity = AsyncMock()

        mock_bot = MagicMock()
        mock_bot.get_adapter.return_value = mock_adapter
        mock_bot.on_turn = AsyncMock()

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_bot
            with patch("botbuilder.schema.Activity") as mock_activity_cls:
                mock_activity_cls.deserialize.return_value = MagicMock()
                result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Bot Framework Availability Check
# =============================================================================


class TestBotFrameworkCheck:
    """Tests for Bot Framework SDK availability check."""

    def test_check_botframework_returns_tuple(self):
        """Should return a tuple of (bool, Optional[str])."""
        available, error = _check_botframework_available()
        assert isinstance(available, bool)
        if not available:
            assert error is not None
            assert isinstance(error, str)
        else:
            assert error is None

    def test_check_botframework_with_import_error(self):
        """Should return error message when SDK not installed."""
        with patch.dict("sys.modules", {"botbuilder.core": None}):
            with patch(
                "aragora.server.handlers.bots.teams._check_botframework_available"
            ) as mock_check:
                mock_check.return_value = (False, "botbuilder-core not installed")
                available, error = mock_check()
                assert available is False
                assert "botbuilder" in error


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
        assert handler._bot_initialized is True

    @pytest.mark.asyncio
    async def test_ensure_bot_without_app_id(self):
        """Should return None when app ID not configured."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", None):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                result = await handler._ensure_bot()

        assert result is None

    @pytest.mark.asyncio
    async def test_ensure_bot_without_password(self):
        """Should return None when password not configured."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", None):
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

    @pytest.mark.asyncio
    async def test_ensure_bot_creates_bot_with_credentials(self):
        """Should create TeamsBot when credentials are configured."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                result = await handler._ensure_bot()

        # The handler creates a TeamsBot directly when credentials are configured
        assert result is not None
        assert handler._bot_initialized is True
        assert handler._bot is not None

    @pytest.mark.asyncio
    async def test_ensure_bot_only_initializes_once(self):
        """Should only initialize bot once even when called multiple times."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                result1 = await handler._ensure_bot()
                result2 = await handler._ensure_bot()

        # Both calls should return the same bot instance
        assert result1 is result2
        assert handler._bot_initialized is True

    @pytest.mark.asyncio
    async def test_ensure_bot_returns_cached_bot(self):
        """Should return cached bot after first initialization."""
        handler = TeamsHandler({})
        handler._bot_initialized = True
        mock_bot = MagicMock()
        handler._bot = mock_bot

        result = await handler._ensure_bot()

        assert result is mock_bot

    @pytest.mark.asyncio
    async def test_ensure_bot_returns_none_without_app_id_only(self):
        """Should return None when only app ID is missing."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", None):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                result = await handler._ensure_bot()

        assert result is None
        assert handler._bot_initialized is True


# =============================================================================
# Test Bot Enabled Check
# =============================================================================


class TestBotEnabledCheck:
    """Tests for _is_bot_enabled method."""

    def test_is_bot_enabled_false_no_credentials(self):
        """Should return False when no credentials configured."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", ""):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", ""):
                assert handler._is_bot_enabled() is False

    def test_is_bot_enabled_false_missing_app_id(self):
        """Should return False when app ID missing."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", None):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                assert handler._is_bot_enabled() is False

    def test_is_bot_enabled_false_missing_password(self):
        """Should return False when password missing."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", None):
                assert handler._is_bot_enabled() is False

    def test_is_bot_enabled_true_with_credentials(self):
        """Should return True when both credentials configured."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                assert handler._is_bot_enabled() is True


# =============================================================================
# Test Handle Method Routing
# =============================================================================


class TestHandleMethodRouting:
    """Tests for handle method routing."""

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_path(self):
        """Should return None for paths not in ROUTES."""
        handler = TeamsHandler({})
        mock_handler = MagicMock()

        result = await handler.handle("/api/v1/unknown", {}, mock_handler)

        assert result is None

    @pytest.mark.asyncio
    async def test_handle_routes_to_status(self):
        """Should route status path to status handler."""
        handler = TeamsHandler({})

        with patch.object(handler, "handle_status_request", new_callable=AsyncMock) as mock_status:
            mock_status.return_value = MagicMock(status_code=200)
            mock_handler = MagicMock()
            result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)

        mock_status.assert_called_once_with(mock_handler)
        assert result is not None


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in message processing."""

    @pytest.mark.asyncio
    async def test_handle_general_exception(self):
        """Should handle general exceptions in message processing."""
        handler = TeamsHandler({})

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": "10"}
        mock_request.rfile.read.side_effect = IOError("Read error")

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = MagicMock()
            result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

        assert result is not None
        # Exception is handled by _handle_webhook_exception
        # 503 is returned for connection/IO errors, 400/500 for other errors
        assert result.status_code in (400, 500, 503)

    @requires_botbuilder
    @pytest.mark.asyncio
    async def test_audit_webhook_auth_failure_called(self):
        """Should audit authentication failures."""
        handler = TeamsHandler({})

        activity_data = {"type": "message", "text": "test"}

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(activity_data))),
            "Authorization": "Bearer invalid",
        }
        mock_request.rfile.read.return_value = json.dumps(activity_data).encode()

        mock_adapter = MagicMock()
        mock_adapter.authenticate_request = AsyncMock(side_effect=ValueError("Invalid token"))

        mock_bot = MagicMock()
        mock_bot.get_adapter.return_value = mock_adapter

        with patch.object(handler, "_ensure_bot", new_callable=AsyncMock) as mock_ensure:
            mock_ensure.return_value = mock_bot
            with patch("botbuilder.schema.Activity") as mock_activity_cls:
                mock_activity_cls.deserialize.return_value = MagicMock()
                with patch.object(handler, "_audit_webhook_auth_failure") as mock_audit:
                    await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_request)

                    mock_audit.assert_called_once_with("auth_token", "invalid_token")


# =============================================================================
# Test TeamsBot Class
# =============================================================================


class TestTeamsBot:
    """Tests for TeamsBot class methods."""

    def test_teams_bot_initialization(self):
        """Should initialize with app credentials."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        assert bot.app_id == "test-app-id"
        assert bot.app_password == "test-password"
        assert bot._connector is None

    def test_teams_bot_defaults_from_env(self):
        """Should use environment variables as defaults."""
        from aragora.server.handlers.bots.teams import TeamsBot

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "env-app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "env-password"):
                bot = TeamsBot()

                assert bot.app_id == "env-app-id"
                assert bot.app_password == "env-password"


class TestTeamsBotActivityProcessing:
    """Tests for TeamsBot activity processing methods."""

    @pytest.mark.asyncio
    async def test_process_activity_message(self):
        """Should process message activity."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "type": "message",
            "id": "activity-123",
            "text": "<at>Bot</at> help",
            "from": {"id": "user-123", "name": "Test User"},
            "conversation": {"id": "conv-123", "conversationType": "channel"},
            "serviceUrl": "https://smba.trafficmanager.net/test/",
            "entities": [{"type": "mention"}],
        }

        with patch.object(bot, "_handle_message", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = {}
            with patch(
                "aragora.server.handlers.bots.teams._verify_teams_token",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await bot.process_activity(activity, "Bearer test-token")

        mock_handle.assert_called_once_with(activity)
        assert result == {}

    @pytest.mark.asyncio
    async def test_process_activity_invoke(self):
        """Should process invoke activity."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "type": "invoke",
            "name": "adaptiveCard/action",
            "value": {"action": "vote", "debate_id": "debate-123", "agent": "Claude"},
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://smba.trafficmanager.net/test/",
        }

        with patch.object(bot, "_handle_invoke", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = {"status": 200, "body": {}}
            with patch(
                "aragora.server.handlers.bots.teams._verify_teams_token",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await bot.process_activity(activity, "Bearer test-token")

        mock_handle.assert_called_once_with(activity)

    @pytest.mark.asyncio
    async def test_process_activity_conversation_update(self):
        """Should process conversation update activity."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "type": "conversationUpdate",
            "membersAdded": [{"id": "bot-123"}],
            "recipient": {"id": "bot-123"},
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://smba.trafficmanager.net/test/",
        }

        with patch.object(
            bot, "_handle_conversation_update", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = {}
            with patch(
                "aragora.server.handlers.bots.teams._verify_teams_token",
                new_callable=AsyncMock,
                return_value=True,
            ):
                result = await bot.process_activity(activity, "Bearer test-token")

        mock_handle.assert_called_once_with(activity)

    @pytest.mark.asyncio
    async def test_process_activity_invalid_token(self):
        """Should raise ValueError for invalid token."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "type": "message",
            "text": "test",
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
        }

        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with pytest.raises(ValueError, match="Invalid authentication token"):
                await bot.process_activity(activity, "Bearer invalid-token")


class TestTeamsBotCommands:
    """Tests for TeamsBot command handling."""

    @pytest.mark.asyncio
    async def test_handle_command_debate(self):
        """Should handle debate command."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        with patch.object(bot, "_cmd_debate", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = {}
            activity = {
                "conversation": {"id": "conv-123"},
                "serviceUrl": "https://test/",
            }
            await bot._handle_command(
                command="debate",
                args="Should we use microservices?",
                conversation_id="conv-123",
                user_id="user-123",
                service_url="https://test/",
                activity=activity,
            )

        mock_cmd.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_command_status(self):
        """Should handle status command."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        with patch.object(bot, "_cmd_status", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = {}
            activity = {}
            await bot._handle_command(
                command="status",
                args="",
                conversation_id="conv-123",
                user_id="user-123",
                service_url="https://test/",
                activity=activity,
            )

        mock_cmd.assert_called_once_with(activity)

    @pytest.mark.asyncio
    async def test_handle_command_help(self):
        """Should handle help command."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        with patch.object(bot, "_cmd_help", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = {}
            activity = {}
            await bot._handle_command(
                command="help",
                args="",
                conversation_id="conv-123",
                user_id="user-123",
                service_url="https://test/",
                activity=activity,
            )

        mock_cmd.assert_called_once_with(activity)

    @pytest.mark.asyncio
    async def test_handle_command_leaderboard(self):
        """Should handle leaderboard command."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        with patch.object(bot, "_cmd_leaderboard", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = {}
            activity = {}
            await bot._handle_command(
                command="leaderboard",
                args="",
                conversation_id="conv-123",
                user_id="user-123",
                service_url="https://test/",
                activity=activity,
            )

        mock_cmd.assert_called_once_with(activity)

    @pytest.mark.asyncio
    async def test_handle_command_unknown(self):
        """Should handle unknown command."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        with patch.object(bot, "_cmd_unknown", new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = {}
            activity = {}
            await bot._handle_command(
                command="foobar",
                args="",
                conversation_id="conv-123",
                user_id="user-123",
                service_url="https://test/",
                activity=activity,
            )

        mock_cmd.assert_called_once_with("foobar", activity)


class TestTeamsBotCardActions:
    """Tests for TeamsBot card action handling."""

    @pytest.mark.asyncio
    async def test_handle_vote_action(self):
        """Should handle vote card action."""
        from aragora.server.handlers.bots.teams import TeamsBot, _user_votes

        # Clear votes before test
        _user_votes.clear()

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://test/",
        }

        # Mock RBAC to allow all permissions
        with patch.object(bot, "_check_permission", return_value=None):
            with patch("aragora.server.handlers.bots.teams.audit_data"):
                result = await bot._handle_vote(
                    debate_id="debate-456",
                    agent="Claude",
                    user_id="user-123",
                    activity=activity,
                )

        assert result["status"] == 200
        assert "debate-456" in _user_votes
        assert _user_votes["debate-456"]["user-123"] == "Claude"

        # Clean up
        _user_votes.clear()

    @pytest.mark.asyncio
    async def test_handle_vote_action_invalid_data(self):
        """Should return error for invalid vote data."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {}

        # Mock RBAC to allow all permissions so we test validation logic
        with patch.object(bot, "_check_permission", return_value=None):
            result = await bot._handle_vote(
                debate_id="",
                agent="",
                user_id="user-123",
                activity=activity,
            )

        assert result["status"] == 400
        assert "Invalid vote data" in result["body"]["value"]

    @pytest.mark.asyncio
    async def test_handle_summary_action(self):
        """Should handle summary card action."""
        from aragora.server.handlers.bots.teams import TeamsBot, _active_debates

        # Set up active debate
        _active_debates["debate-789"] = {
            "topic": "Test topic",
            "started_at": 1000,
        }

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {}

        result = await bot._handle_summary(
            debate_id="debate-789",
            activity=activity,
        )

        assert result["status"] == 200
        assert result["body"]["type"] == "application/vnd.microsoft.card.adaptive"

        # Clean up
        _active_debates.clear()

    @pytest.mark.asyncio
    async def test_handle_summary_debate_not_found(self):
        """Should return message when debate not found."""
        from aragora.server.handlers.bots.teams import TeamsBot, _active_debates

        _active_debates.clear()

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {}

        result = await bot._handle_summary(
            debate_id="nonexistent",
            activity=activity,
        )

        assert result["status"] == 200
        assert "not found" in result["body"]["value"]


class TestTeamsBotInvokeHandling:
    """Tests for TeamsBot invoke handling."""

    @pytest.mark.asyncio
    async def test_handle_invoke_card_action(self):
        """Should route card action invokes correctly."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "name": "adaptiveCard/action",
            "value": {"action": "help"},
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://test/",
        }

        with patch.object(bot, "_handle_card_action", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = {"status": 200, "body": {}}
            await bot._handle_invoke(activity)

        mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_invoke_compose_extension_query(self):
        """Should handle compose extension query."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "name": "composeExtension/query",
            "value": {"parameters": [{"name": "query", "value": "test"}]},
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://test/",
        }

        with patch.object(
            bot, "_handle_compose_extension_query", new_callable=AsyncMock
        ) as mock_handle:
            mock_handle.return_value = {"status": 200, "body": {}}
            await bot._handle_invoke(activity)

        mock_handle.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_invoke_task_fetch(self):
        """Should handle task module fetch."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "name": "task/fetch",
            "value": {"commandId": "startDebate"},
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://test/",
        }

        with patch.object(bot, "_handle_task_module_fetch", new_callable=AsyncMock) as mock_handle:
            mock_handle.return_value = {"status": 200, "body": {"task": {}}}
            await bot._handle_invoke(activity)

        mock_handle.assert_called_once()


class TestTeamsBotHelperFunctions:
    """Tests for TeamsBot helper functions and card builders."""

    def test_build_debate_card(self):
        """Should build valid debate card."""
        from aragora.server.handlers.bots.teams import build_debate_card

        card = build_debate_card(
            debate_id="debate-123",
            topic="Should we use AI for decisions?",
            agents=["Claude", "GPT-4", "Gemini"],
            current_round=2,
            total_rounds=5,
            include_vote_buttons=True,
        )

        assert card["type"] == "AdaptiveCard"
        assert card["version"] == "1.4"
        assert len(card["body"]) > 0
        assert len(card["actions"]) > 0

    def test_build_debate_card_without_vote_buttons(self):
        """Should build card without vote buttons."""
        from aragora.server.handlers.bots.teams import build_debate_card

        card = build_debate_card(
            debate_id="debate-123",
            topic="Test topic",
            agents=["Claude"],
            current_round=1,
            total_rounds=3,
            include_vote_buttons=False,
        )

        # Actions should be None or empty when no vote buttons
        assert card.get("actions") is None or len(card["actions"]) == 0

    def test_build_consensus_card(self):
        """Should build valid consensus card."""
        from aragora.server.handlers.bots.teams import build_consensus_card

        card = build_consensus_card(
            debate_id="debate-123",
            topic="Important decision",
            consensus_reached=True,
            confidence=0.85,
            winner="Claude",
            final_answer="The decision is to proceed with option A.",
            vote_counts={"Claude": 5, "GPT-4": 2},
        )

        assert card["type"] == "AdaptiveCard"
        assert len(card["body"]) > 0
        assert len(card["actions"]) > 0

    def test_build_consensus_card_no_consensus(self):
        """Should build card for no consensus case."""
        from aragora.server.handlers.bots.teams import build_consensus_card

        card = build_consensus_card(
            debate_id="debate-456",
            topic="Contentious issue",
            consensus_reached=False,
            confidence=0.45,
            winner=None,
            final_answer=None,
            vote_counts={},
        )

        assert card["type"] == "AdaptiveCard"
        # Check header shows no consensus
        header = card["body"][0]
        assert "No Consensus" in header["text"]

    def test_get_debate_vote_counts(self):
        """Should calculate vote counts correctly."""
        from aragora.server.handlers.bots.teams import get_debate_vote_counts, _user_votes

        _user_votes.clear()
        _user_votes["debate-test"] = {
            "user-1": "Claude",
            "user-2": "Claude",
            "user-3": "GPT-4",
        }

        counts = get_debate_vote_counts("debate-test")

        assert counts["Claude"] == 2
        assert counts["GPT-4"] == 1

        # Clean up
        _user_votes.clear()

    def test_get_debate_vote_counts_empty(self):
        """Should return empty dict for unknown debate."""
        from aragora.server.handlers.bots.teams import get_debate_vote_counts, _user_votes

        _user_votes.clear()

        counts = get_debate_vote_counts("nonexistent-debate")

        assert counts == {}

    def test_get_conversation_reference(self):
        """Should retrieve stored conversation reference."""
        from aragora.server.handlers.bots.teams import (
            get_conversation_reference,
            _conversation_references,
        )

        _conversation_references.clear()
        _conversation_references["conv-test"] = {
            "service_url": "https://test/",
            "conversation": {"id": "conv-test"},
            "bot": {"id": "bot-123"},
        }

        ref = get_conversation_reference("conv-test")

        assert ref is not None
        assert ref["service_url"] == "https://test/"

        # Clean up
        _conversation_references.clear()

    def test_get_conversation_reference_not_found(self):
        """Should return None for unknown conversation."""
        from aragora.server.handlers.bots.teams import (
            get_conversation_reference,
            _conversation_references,
        )

        _conversation_references.clear()

        ref = get_conversation_reference("unknown-conv")

        assert ref is None


class TestTeamsBotProactiveMessaging:
    """Tests for TeamsBot proactive messaging."""

    @pytest.mark.asyncio
    async def test_send_proactive_message_with_text(self):
        """Should send proactive text message."""
        from aragora.server.handlers.bots.teams import TeamsBot, _conversation_references

        _conversation_references.clear()
        _conversation_references["conv-proactive"] = {
            "service_url": "https://test/",
            "conversation": {"id": "conv-proactive"},
        }

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        mock_connector = MagicMock()
        mock_connector.send_message = AsyncMock()

        with patch.object(bot, "_get_connector", new_callable=AsyncMock) as mock_get_conn:
            mock_get_conn.return_value = mock_connector

            result = await bot.send_proactive_message(
                conversation_id="conv-proactive",
                text="Hello from bot!",
            )

        assert result is True
        mock_connector.send_message.assert_called_once()

        # Clean up
        _conversation_references.clear()

    @pytest.mark.asyncio
    async def test_send_proactive_message_no_reference(self):
        """Should return False when no conversation reference."""
        from aragora.server.handlers.bots.teams import TeamsBot, _conversation_references

        _conversation_references.clear()

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        result = await bot.send_proactive_message(
            conversation_id="unknown-conv",
            text="Test message",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_proactive_message_no_connector(self):
        """Should return False when connector unavailable."""
        from aragora.server.handlers.bots.teams import TeamsBot, _conversation_references

        _conversation_references.clear()
        _conversation_references["conv-test"] = {
            "service_url": "https://test/",
        }

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        with patch.object(bot, "_get_connector", new_callable=AsyncMock) as mock_get_conn:
            mock_get_conn.return_value = None

            result = await bot.send_proactive_message(
                conversation_id="conv-test",
                text="Test message",
            )

        assert result is False

        # Clean up
        _conversation_references.clear()


class TestTeamsBotMessageHandling:
    """Tests for TeamsBot message handling."""

    @pytest.mark.asyncio
    async def test_handle_message_with_mention(self):
        """Should handle message with @mention."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "text": "<at>Aragora</at> help",
            "conversation": {"id": "conv-123", "conversationType": "channel"},
            "from": {"id": "user-123", "name": "Test User"},
            "serviceUrl": "https://test/",
            "entities": [{"type": "mention"}],
        }

        # Mock RBAC to allow all permissions
        with patch.object(bot, "_check_permission", return_value=None):
            with patch.object(bot, "_send_typing", new_callable=AsyncMock):
                with patch.object(bot, "_handle_command", new_callable=AsyncMock) as mock_cmd:
                    mock_cmd.return_value = {}
                    await bot._handle_message(activity)

        mock_cmd.assert_called_once()
        call_args = mock_cmd.call_args
        assert call_args[1]["command"] == "help"

    @pytest.mark.asyncio
    async def test_handle_message_personal_conversation(self):
        """Should treat all messages as commands in personal conversation."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "text": "What is the best programming language?",
            "conversation": {"id": "conv-123", "conversationType": "personal"},
            "from": {"id": "user-123", "name": "Test User"},
            "serviceUrl": "https://test/",
            "entities": [],
        }

        # Mock RBAC to allow all permissions
        with patch.object(bot, "_check_permission", return_value=None):
            with patch.object(bot, "_send_typing", new_callable=AsyncMock):
                with patch.object(bot, "_handle_command", new_callable=AsyncMock) as mock_cmd:
                    mock_cmd.return_value = {}
                    await bot._handle_message(activity)

        mock_cmd.assert_called_once()
        # In personal scope, unknown commands default to debate
        call_args = mock_cmd.call_args
        assert call_args[1]["command"] == "debate"

    @pytest.mark.asyncio
    async def test_handle_message_no_mention_in_group(self):
        """Should prompt for @mention in group chat without mention."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "text": "Hello there",
            "conversation": {"id": "conv-123", "conversationType": "channel"},
            "from": {"id": "user-123", "name": "Test User"},
            "serviceUrl": "https://test/",
            "entities": [],
        }

        # Mock RBAC to allow all permissions
        with patch.object(bot, "_check_permission", return_value=None):
            with patch.object(bot, "_send_typing", new_callable=AsyncMock):
                with patch.object(bot, "_send_reply", new_callable=AsyncMock) as mock_reply:
                    await bot._handle_message(activity)

        mock_reply.assert_called_once()
        reply_text = mock_reply.call_args[0][1]
        assert "@Aragora" in reply_text


class TestTeamsBotRBAC:
    """Tests for RBAC permission checks in TeamsBot."""

    @pytest.mark.asyncio
    async def test_rbac_denies_message_without_permission(self):
        """Should deny message processing when RBAC returns permission denied."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "text": "test message",
            "conversation": {"id": "conv-123", "conversationType": "personal"},
            "from": {"id": "user-123", "name": "Test User"},
            "serviceUrl": "https://test/",
            "entities": [],
        }

        # Mock RBAC to deny permission
        perm_error = {
            "error": "permission_denied",
            "message": "Permission denied: teams:messages:read",
        }
        with patch.object(bot, "_check_permission", return_value=perm_error):
            with patch.object(bot, "_send_reply", new_callable=AsyncMock) as mock_reply:
                result = await bot._handle_message(activity)

        # Should send permission denied message
        mock_reply.assert_called_once()
        reply_text = mock_reply.call_args[0][1]
        assert "permission" in reply_text.lower()
        assert result == {}

    @pytest.mark.asyncio
    async def test_rbac_denies_vote_without_permission(self):
        """Should deny vote action when RBAC returns permission denied."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://test/",
        }

        # Mock RBAC to deny permission
        perm_error = {"error": "permission_denied", "message": "Not allowed"}
        with patch.object(bot, "_check_permission", return_value=perm_error):
            result = await bot._handle_vote(
                debate_id="debate-456",
                agent="Claude",
                user_id="user-123",
                activity=activity,
            )

        # Should return 403 with permission denied card
        assert result["status"] == 403
        assert "Permission Denied" in str(result["body"])

    @pytest.mark.asyncio
    async def test_rbac_denies_debate_creation_without_permission(self):
        """Should deny debate creation when RBAC returns permission denied."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        activity = {
            "from": {"id": "user-123"},
            "conversation": {"id": "conv-123"},
            "serviceUrl": "https://test/",
        }

        # Mock RBAC to deny permission
        perm_error = {
            "error": "permission_denied",
            "message": "Permission denied: teams:debates:create",
        }
        with patch.object(bot, "_check_permission", return_value=perm_error):
            with patch.object(bot, "_send_reply", new_callable=AsyncMock) as mock_reply:
                result = await bot._cmd_debate(
                    topic="test topic",
                    conversation_id="conv-123",
                    user_id="user-123",
                    service_url="https://test/",
                    thread_id=None,
                    activity=activity,
                )

        # Should send permission denied message
        mock_reply.assert_called_once()
        reply_text = mock_reply.call_args[0][1]
        assert "permission" in reply_text.lower()
        assert result == {}

    def test_rbac_helper_methods_exist(self):
        """TeamsBot should have RBAC helper methods."""
        from aragora.server.handlers.bots.teams import TeamsBot

        bot = TeamsBot(app_id="test-app-id", app_password="test-password")

        assert hasattr(bot, "_check_permission")
        assert hasattr(bot, "_get_auth_context_from_activity")
        assert hasattr(bot, "_validate_tenant")
        assert callable(bot._check_permission)
        assert callable(bot._get_auth_context_from_activity)
        assert callable(bot._validate_tenant)

    def test_rbac_permission_constants_exist(self):
        """Should have RBAC permission constants defined."""
        from aragora.server.handlers.bots.teams import (
            PERM_TEAMS_ADMIN,
            PERM_TEAMS_CARDS_RESPOND,
            PERM_TEAMS_DEBATES_CREATE,
            PERM_TEAMS_DEBATES_VOTE,
            PERM_TEAMS_MESSAGES_READ,
            PERM_TEAMS_MESSAGES_SEND,
        )

        assert PERM_TEAMS_MESSAGES_READ == "teams:messages:read"
        assert PERM_TEAMS_MESSAGES_SEND == "teams:messages:send"
        assert PERM_TEAMS_DEBATES_CREATE == "teams:debates:create"
        assert PERM_TEAMS_DEBATES_VOTE == "teams:debates:vote"
        assert PERM_TEAMS_CARDS_RESPOND == "teams:cards:respond"
        assert PERM_TEAMS_ADMIN == "teams:admin"


class TestAgentDisplayNames:
    """Tests for agent display names mapping."""

    def test_agent_display_names(self):
        """Should map agent IDs to display names."""
        from aragora.server.handlers.bots.teams import AGENT_DISPLAY_NAMES

        assert AGENT_DISPLAY_NAMES["claude"] == "Claude"
        assert AGENT_DISPLAY_NAMES["gpt4"] == "GPT-4"
        assert AGENT_DISPLAY_NAMES["gemini"] == "Gemini"
        assert AGENT_DISPLAY_NAMES["anthropic-api"] == "Claude"
        assert AGENT_DISPLAY_NAMES["openai-api"] == "GPT-4"
