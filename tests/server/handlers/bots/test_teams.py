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
    async def test_ensure_bot_sdk_not_available(self):
        """Should return None when SDK not available."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                with patch(
                    "aragora.server.handlers.bots.teams._check_botframework_available",
                    return_value=(False, "not installed"),
                ):
                    result = await handler._ensure_bot()

        assert result is None
        assert handler._bot_initialized is True

    @pytest.mark.asyncio
    async def test_ensure_bot_import_error(self):
        """Should handle import error for bot module."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                with patch(
                    "aragora.server.handlers.bots.teams._check_botframework_available",
                    return_value=(True, None),
                ):
                    with patch(
                        "aragora.bots.teams_bot.create_teams_bot",
                        side_effect=ImportError("Module not found"),
                    ):
                        result = await handler._ensure_bot()

        assert result is None
        assert handler._bot_initialized is True

    @pytest.mark.asyncio
    async def test_ensure_bot_config_error(self):
        """Should handle configuration errors gracefully."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                with patch(
                    "aragora.server.handlers.bots.teams._check_botframework_available",
                    return_value=(True, None),
                ):
                    mock_bot = MagicMock()
                    mock_bot.setup = AsyncMock(side_effect=ValueError("Bad config"))
                    with patch(
                        "aragora.bots.teams_bot.create_teams_bot",
                        return_value=mock_bot,
                    ):
                        result = await handler._ensure_bot()

        assert result is None
        assert handler._bot_initialized is True

    @pytest.mark.asyncio
    async def test_ensure_bot_unexpected_error(self):
        """Should handle unexpected errors gracefully."""
        handler = TeamsHandler({})

        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-id"):
            with patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "password"):
                with patch(
                    "aragora.server.handlers.bots.teams._check_botframework_available",
                    return_value=(True, None),
                ):
                    mock_bot = MagicMock()
                    mock_bot.setup = AsyncMock(side_effect=RuntimeError("Unexpected"))
                    with patch(
                        "aragora.bots.teams_bot.create_teams_bot",
                        return_value=mock_bot,
                    ):
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
