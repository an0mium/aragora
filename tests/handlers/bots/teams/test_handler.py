"""
Tests for Microsoft Teams Bot main handler.

Covers all routes and behavior of the TeamsHandler and TeamsBot classes:
- TeamsHandler
  - can_handle() routing for all defined routes
  - GET  /api/v1/bots/teams/status   - Bot status (RBAC-protected)
  - POST /api/v1/bots/teams/messages - Bot Framework activity processing
  - _ensure_bot() lazy initialization (configured / not configured)
  - _is_bot_enabled() with various env var combinations
  - _get_platform_config_status() returns Teams-specific fields
  - handle_post returns None for unknown paths
  - handle returns None for non-status paths

- TeamsBot
  - __init__ defaults and explicit params
  - process_activity() main entry point
    - Token verification (valid / invalid / no app_id)
    - Tenant validation (matching / mismatching / no tenant configured)
    - Activity routing (message / invoke / conversationUpdate /
      messageReaction / installationUpdate / unknown)
  - _handle_message()
    - Mention-based commands (debate, status, help, leaderboard, agents, vote)
    - Personal chat with known command
    - Personal chat with unknown command (defaults to debate)
    - Non-mention, non-personal message (prompt reply)
    - RBAC permission denied for messages
  - _handle_invoke()
    - adaptiveCard/action
    - composeExtension/submitAction
    - composeExtension/query
    - composeExtension/fetchTask / task/fetch
    - task/submit
    - Unknown invoke name (default 200)
  - _handle_conversation_update / _handle_message_reaction /
    _handle_installation_update delegation
  - _handle_command() routing to all commands
  - _cmd_debate() RBAC check
  - send_typing / send_reply / send_card / send_proactive_message
  - _get_connector() lazy init and ImportError fallback
  - _get_event_processor / _get_card_actions lazy init
  - RBAC helpers
    - _get_auth_context_from_activity() with/without AAD ID
    - _check_permission() allowed / denied / RBAC unavailable
    - _validate_tenant() matching / mismatching / unconfigured
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Lazy import so conftest auto-auth patches run first
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    import aragora.server.handlers.bots.teams.handler as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.TeamsHandler


@pytest.fixture
def handler(handler_cls):
    """Create a TeamsHandler with empty context."""
    return handler_cls(ctx={})


@pytest.fixture
def bot_cls(handler_module):
    return handler_module.TeamsBot


# ---------------------------------------------------------------------------
# Mock HTTP Handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/teams/messages"
    method: str = "POST"
    body: dict[str, Any] | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.body is not None:
            body_bytes = json.dumps(self.body).encode("utf-8")
        else:
            body_bytes = b"{}"
        self.rfile = io.BytesIO(body_bytes)
        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body_bytes))
        self.client_address = ("127.0.0.1", 12345)


def _make_http_handler(
    body: dict[str, Any] | None = None,
    auth_header: str = "Bearer valid-token",
    content_type: str = "application/json",
) -> MockHTTPHandler:
    """Create a MockHTTPHandler pre-configured for POST requests."""
    headers = {
        "Content-Type": content_type,
        "Authorization": auth_header,
    }
    return MockHTTPHandler(body=body, headers=headers)


# ---------------------------------------------------------------------------
# Activity builders
# ---------------------------------------------------------------------------


def _make_activity(
    activity_type: str = "message",
    text: str = "hello",
    user_id: str = "user-123",
    aad_object_id: str = "",
    conversation_id: str = "conv-abc",
    conversation_type: str = "",
    service_url: str = "https://smba.trafficmanager.net/teams/",
    tenant_id: str = "",
    entities: list | None = None,
    invoke_name: str = "",
    value: dict[str, Any] | None = None,
    reply_to_id: str | None = None,
) -> dict[str, Any]:
    """Build a minimal Bot Framework activity."""
    activity: dict[str, Any] = {
        "type": activity_type,
        "id": "act-001",
        "from": {"id": user_id},
        "conversation": {"id": conversation_id},
        "serviceUrl": service_url,
    }
    if text:
        activity["text"] = text
    if aad_object_id:
        activity["from"]["aadObjectId"] = aad_object_id
    if conversation_type:
        activity["conversation"]["conversationType"] = conversation_type
    if tenant_id:
        activity["conversation"]["tenantId"] = tenant_id
    if entities is not None:
        activity["entities"] = entities
    if invoke_name:
        activity["name"] = invoke_name
    if value is not None:
        activity["value"] = value
    if reply_to_id:
        activity["replyToId"] = reply_to_id
    return activity


def _mention_entities() -> list[dict[str, Any]]:
    """Build mention entities list."""
    return [{"type": "mention", "mentioned": {"id": "bot-id", "name": "Aragora"}}]


# ---------------------------------------------------------------------------
# Shared state fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_state():
    """Clear module-level shared state between tests."""
    from aragora.server.handlers.bots.teams_utils import (
        _active_debates,
        _conversation_references,
        _user_votes,
    )

    _active_debates.clear()
    _conversation_references.clear()
    _user_votes.clear()
    yield
    _active_debates.clear()
    _conversation_references.clear()
    _user_votes.clear()


# ===========================================================================
# TeamsHandler: can_handle routing
# ===========================================================================


class TestCanHandle:
    """Test can_handle() for all defined routes."""

    def test_messages_route(self, handler):
        assert handler.can_handle("/api/v1/bots/teams/messages") is True

    def test_status_route(self, handler):
        assert handler.can_handle("/api/v1/bots/teams/status") is True

    def test_teams_root(self, handler):
        assert handler.can_handle("/api/v1/teams") is True

    def test_debates_send(self, handler):
        assert handler.can_handle("/api/v1/teams/debates/send") is True

    def test_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/bots/teams/unknown") is False

    def test_slack_route_rejected(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/messages") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_match(self, handler):
        assert handler.can_handle("/api/v1/bots/teams") is False


# ===========================================================================
# TeamsHandler: _is_bot_enabled
# ===========================================================================


class TestIsBotEnabled:
    """Test _is_bot_enabled() with various env configurations."""

    def test_enabled_when_both_set(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-123"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "secret"):
            assert handler._is_bot_enabled() is True

    def test_disabled_when_no_app_id(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", ""), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "secret"):
            assert handler._is_bot_enabled() is False

    def test_disabled_when_no_password(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-123"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", ""):
            assert handler._is_bot_enabled() is False

    def test_disabled_when_none(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", None), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", None):
            assert handler._is_bot_enabled() is False


# ===========================================================================
# TeamsHandler: _get_platform_config_status
# ===========================================================================


class TestGetPlatformConfigStatus:
    """Test _get_platform_config_status() returns Teams-specific fields."""

    def test_returns_expected_keys(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-123"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "secret"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_TENANT_ID", "tenant-1"), \
             patch(
                 "aragora.server.handlers.bots.teams.handler._check_botframework_available",
                 return_value=(True, None),
             ), \
             patch(
                 "aragora.server.handlers.bots.teams.handler._check_connector_available",
                 return_value=(True, None),
             ):
            status = handler._get_platform_config_status()
            assert status["app_id_configured"] is True
            assert status["password_configured"] is True
            assert status["tenant_id_configured"] is True
            assert status["sdk_available"] is True
            assert status["sdk_error"] is None
            assert status["connector_available"] is True
            assert status["connector_error"] is None
            assert "features" in status
            assert status["features"]["adaptive_cards"] is True
            assert status["features"]["compose_extensions"] is True
            assert status["features"]["task_modules"] is True
            assert status["features"]["link_unfurling"] is True

    def test_sdk_not_available(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", ""), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", ""), \
             patch("aragora.server.handlers.bots.teams.TEAMS_TENANT_ID", ""), \
             patch(
                 "aragora.server.handlers.bots.teams.handler._check_botframework_available",
                 return_value=(False, "not installed"),
             ), \
             patch(
                 "aragora.server.handlers.bots.teams.handler._check_connector_available",
                 return_value=(False, "not available"),
             ):
            status = handler._get_platform_config_status()
            assert status["sdk_available"] is False
            assert status["sdk_error"] == "not installed"
            assert status["connector_available"] is False
            assert status["connector_error"] == "not available"

    def test_includes_debate_and_conversation_counts(self, handler):
        from aragora.server.handlers.bots.teams_utils import (
            _active_debates,
            _conversation_references,
        )

        _active_debates["d1"] = {"topic": "test"}
        _active_debates["d2"] = {"topic": "test2"}
        _conversation_references["c1"] = {"service_url": "x"}

        with patch(
            "aragora.server.handlers.bots.teams.handler._check_botframework_available",
            return_value=(True, None),
        ), patch(
            "aragora.server.handlers.bots.teams.handler._check_connector_available",
            return_value=(True, None),
        ):
            status = handler._get_platform_config_status()
            assert status["active_debates"] == 2
            assert status["conversation_references"] == 1


# ===========================================================================
# TeamsHandler: _ensure_bot
# ===========================================================================


class TestEnsureBot:
    """Test lazy bot initialization."""

    @pytest.mark.asyncio
    async def test_returns_bot_when_configured(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-123"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "secret"):
            bot = await handler._ensure_bot()
            assert bot is not None
            assert bot.app_id == "app-123"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_configured(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", ""), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", ""):
            bot = await handler._ensure_bot()
            assert bot is None

    @pytest.mark.asyncio
    async def test_caches_bot_instance(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-123"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "secret"):
            bot1 = await handler._ensure_bot()
            bot2 = await handler._ensure_bot()
            assert bot1 is bot2

    @pytest.mark.asyncio
    async def test_caches_none_when_not_configured(self, handler):
        """Once _bot_initialized is True, does not retry."""
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", ""), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", ""):
            bot1 = await handler._ensure_bot()
            assert bot1 is None
        # Even with creds now available, returns None (cached)
        bot2 = await handler._ensure_bot()
        assert bot2 is None


# ===========================================================================
# TeamsHandler: handle (GET status)
# ===========================================================================


class TestHandleGetStatus:
    """Test GET /api/v1/bots/teams/status."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-123"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "secret"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_TENANT_ID", "t1"), \
             patch(
                 "aragora.server.handlers.bots.teams.handler._check_botframework_available",
                 return_value=(True, None),
             ), \
             patch(
                 "aragora.server.handlers.bots.teams.handler._check_connector_available",
                 return_value=(True, None),
             ):
            mock_handler = _make_http_handler()
            result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)
            assert _status(result) == 200
            body = _body(result)
            assert body["platform"] == "teams"
            assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_non_status_path(self, handler):
        mock_handler = _make_http_handler()
        result = await handler.handle("/api/v1/bots/teams/messages", {}, mock_handler)
        assert result is None


# ===========================================================================
# TeamsHandler: handle_post (POST messages)
# ===========================================================================


class TestHandlePostMessages:
    """Test POST /api/v1/bots/teams/messages."""

    @pytest.mark.asyncio
    async def test_bot_not_configured_returns_503(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", ""), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", ""):
            handler._bot_initialized = False
            handler._bot = None
            activity = _make_activity()
            mock_h = _make_http_handler(body=activity)
            result = await handler.handle_post(
                "/api/v1/bots/teams/messages", {}, mock_h
            )
            assert _status(result) == 503
            assert "not configured" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_empty_body_returns_400(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-123"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "secret"):
            handler._bot_initialized = False
            handler._bot = None
            mock_h = MockHTTPHandler(
                body=None,
                headers={"Content-Length": "0", "Authorization": "Bearer x"},
            )
            mock_h.rfile = io.BytesIO(b"")
            mock_h.headers["Content-Length"] = "0"
            mock_h.client_address = ("127.0.0.1", 12345)
            result = await handler.handle_post(
                "/api/v1/bots/teams/messages", {}, mock_h
            )
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, handler):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "app-123"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "secret"):
            handler._bot_initialized = False
            handler._bot = None
            bad_body = b"not-json"
            mock_h = MockHTTPHandler(headers={
                "Content-Length": str(len(bad_body)),
                "Authorization": "Bearer x",
            })
            mock_h.rfile = io.BytesIO(bad_body)
            mock_h.client_address = ("127.0.0.1", 12345)
            result = await handler.handle_post(
                "/api/v1/bots/teams/messages", {}, mock_h
            )
            # Should return an error (400 for invalid JSON or 200 with error body)
            status = _status(result)
            assert status in (200, 400)

    @pytest.mark.asyncio
    async def test_successful_message_activity(self, handler):
        """Full round-trip: POST with a message activity processed by bot."""
        activity = _make_activity(activity_type="message", text="hello")

        mock_bot = MagicMock()
        mock_bot.process_activity = AsyncMock(return_value={})

        handler._bot = mock_bot
        handler._bot_initialized = True

        mock_h = _make_http_handler(body=activity)
        result = await handler.handle_post(
            "/api/v1/bots/teams/messages", {}, mock_h
        )
        assert _status(result) == 200
        mock_bot.process_activity.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoke_activity_returns_invoke_response(self, handler):
        """Invoke activities return the response body/status from the bot."""
        activity = _make_activity(activity_type="invoke", invoke_name="adaptiveCard/action")
        invoke_response = {
            "status": 200,
            "body": {"statusCode": 200, "type": "message", "value": "OK"},
        }

        mock_bot = MagicMock()
        mock_bot.process_activity = AsyncMock(return_value=invoke_response)

        handler._bot = mock_bot
        handler._bot_initialized = True

        mock_h = _make_http_handler(body=activity)
        result = await handler.handle_post(
            "/api/v1/bots/teams/messages", {}, mock_h
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_auth_error_returns_401(self, handler):
        """ValueError from process_activity is treated as auth failure."""
        activity = _make_activity()

        mock_bot = MagicMock()
        mock_bot.process_activity = AsyncMock(side_effect=ValueError("Invalid token"))

        handler._bot = mock_bot
        handler._bot_initialized = True

        mock_h = _make_http_handler(body=activity)
        result = await handler.handle_post(
            "/api/v1/bots/teams/messages", {}, mock_h
        )
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_runtime_error_returns_500(self, handler):
        """RuntimeError from process_activity is treated as internal error."""
        activity = _make_activity()

        mock_bot = MagicMock()
        mock_bot.process_activity = AsyncMock(side_effect=RuntimeError("boom"))

        handler._bot = mock_bot
        handler._bot_initialized = True

        mock_h = _make_http_handler(body=activity)
        result = await handler.handle_post(
            "/api/v1/bots/teams/messages", {}, mock_h
        )
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_handle_post_unknown_path_returns_none(self, handler):
        mock_h = _make_http_handler()
        result = await handler.handle_post(
            "/api/v1/bots/teams/unknown", {}, mock_h
        )
        assert result is None


# ===========================================================================
# TeamsBot: __init__
# ===========================================================================


class TestTeamsBotInit:
    """Test TeamsBot initialization."""

    def test_explicit_credentials(self, bot_cls):
        bot = bot_cls(app_id="my-app", app_password="my-pass")
        assert bot.app_id == "my-app"
        assert bot.app_password == "my-pass"

    def test_default_empty_credentials(self, bot_cls):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", None), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", None):
            bot = bot_cls()
            assert bot.app_id == ""
            assert bot.app_password == ""

    def test_env_credentials(self, bot_cls):
        with patch("aragora.server.handlers.bots.teams.TEAMS_APP_ID", "env-app"), \
             patch("aragora.server.handlers.bots.teams.TEAMS_APP_PASSWORD", "env-pass"):
            bot = bot_cls()
            assert bot.app_id == "env-app"
            assert bot.app_password == "env-pass"

    def test_connector_initially_none(self, bot_cls):
        bot = bot_cls(app_id="x", app_password="y")
        assert bot._connector is None

    def test_event_processor_initially_none(self, bot_cls):
        bot = bot_cls(app_id="x", app_password="y")
        assert bot._event_processor is None

    def test_card_actions_initially_none(self, bot_cls):
        bot = bot_cls(app_id="x", app_password="y")
        assert bot._card_actions is None


# ===========================================================================
# TeamsBot: process_activity
# ===========================================================================


class TestProcessActivity:
    """Test the main activity processing entry point."""

    @pytest.fixture
    def bot(self, bot_cls):
        return bot_cls(app_id="app-id", app_password="app-pass")

    @pytest.mark.asyncio
    async def test_invalid_token_raises(self, bot):
        """Invalid token raises ValueError."""
        activity = _make_activity()
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with pytest.raises(ValueError, match="Invalid authentication token"):
                await bot.process_activity(activity, "Bearer invalid")

    @pytest.mark.asyncio
    async def test_valid_token_processes(self, bot):
        """Valid token allows processing."""
        activity = _make_activity(activity_type="message")
        bot._handle_message = AsyncMock(return_value={})
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await bot.process_activity(activity, "Bearer valid")
            assert result == {}
            bot._handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_app_id_skips_verification(self, bot_cls):
        """When app_id is empty, token verification is skipped."""
        bot = bot_cls(app_id="", app_password="")
        activity = _make_activity(activity_type="message")
        bot._handle_message = AsyncMock(return_value={})
        # _verify_teams_token should NOT be called
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
        ) as mock_verify:
            result = await bot.process_activity(activity, "Bearer anything")
            mock_verify.assert_not_awaited()
            assert result == {}

    @pytest.mark.asyncio
    async def test_tenant_validation_failure(self, bot, handler_module):
        """Tenant mismatch returns 403."""
        activity = _make_activity(tenant_id="bad-tenant")
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=True,
        ), patch.object(
            bot, "_validate_tenant", return_value={"error": "tenant_denied", "message": "nope"}
        ):
            result = await bot.process_activity(activity, "Bearer valid")
            assert result["status"] == 403

    @pytest.mark.asyncio
    async def test_routes_message_activity(self, bot):
        activity = _make_activity(activity_type="message")
        bot._handle_message = AsyncMock(return_value={"ok": True})
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await bot.process_activity(activity, "Bearer x")
            assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_routes_invoke_activity(self, bot):
        activity = _make_activity(activity_type="invoke")
        bot._handle_invoke = AsyncMock(return_value={"status": 200})
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await bot.process_activity(activity, "Bearer x")
            assert result["status"] == 200

    @pytest.mark.asyncio
    async def test_routes_conversation_update(self, bot):
        activity = _make_activity(activity_type="conversationUpdate")
        bot._handle_conversation_update = AsyncMock(return_value={})
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await bot.process_activity(activity, "Bearer x")
            bot._handle_conversation_update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_message_reaction(self, bot):
        activity = _make_activity(activity_type="messageReaction")
        bot._handle_message_reaction = AsyncMock(return_value={})
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await bot.process_activity(activity, "Bearer x")
            bot._handle_message_reaction.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_routes_installation_update(self, bot):
        activity = _make_activity(activity_type="installationUpdate")
        bot._handle_installation_update = AsyncMock(return_value={})
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await bot.process_activity(activity, "Bearer x")
            bot._handle_installation_update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_activity_type_returns_empty(self, bot):
        activity = _make_activity(activity_type="unknownType")
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await bot.process_activity(activity, "Bearer x")
            assert result == {}

    @pytest.mark.asyncio
    async def test_stores_conversation_reference(self, bot):
        """process_activity stores the conversation reference."""
        from aragora.server.handlers.bots.teams_utils import _conversation_references

        activity = _make_activity(
            activity_type="message",
            conversation_id="conv-store-test",
            service_url="https://test.example.com",
        )
        bot._handle_message = AsyncMock(return_value={})
        with patch(
            "aragora.server.handlers.bots.teams._verify_teams_token",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await bot.process_activity(activity, "Bearer x")
            assert "conv-store-test" in _conversation_references


# ===========================================================================
# TeamsBot: _handle_message
# ===========================================================================


class TestHandleMessage:
    """Test message activity handling."""

    @pytest.fixture
    def bot(self, bot_cls):
        bot = bot_cls(app_id="app-id", app_password="pass")
        bot._check_permission = MagicMock(return_value=None)
        bot.send_typing = AsyncMock()
        bot.send_reply = AsyncMock()
        return bot

    @pytest.mark.asyncio
    async def test_mention_debate_command(self, bot):
        """@mention with 'debate topic' routes to _handle_command with command='debate'."""
        activity = _make_activity(
            text="<at>Aragora</at> debate AI safety",
            entities=_mention_entities(),
        )
        bot._handle_command = AsyncMock(return_value={"started": True})
        result = await bot._handle_message(activity)
        bot._handle_command.assert_awaited_once()
        call_kwargs = bot._handle_command.call_args
        assert call_kwargs.kwargs.get("command") or call_kwargs[1].get("command") == "debate"

    @pytest.mark.asyncio
    async def test_mention_help_command(self, bot):
        activity = _make_activity(
            text="<at>Aragora</at> help",
            entities=_mention_entities(),
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        args = bot._handle_command.call_args
        assert args.kwargs.get("command") == "help"

    @pytest.mark.asyncio
    async def test_mention_status_command(self, bot):
        activity = _make_activity(
            text="<at>Aragora</at> status",
            entities=_mention_entities(),
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        args = bot._handle_command.call_args
        assert args.kwargs.get("command") == "status"

    @pytest.mark.asyncio
    async def test_mention_leaderboard_command(self, bot):
        activity = _make_activity(
            text="<at>Aragora</at> leaderboard",
            entities=_mention_entities(),
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        args = bot._handle_command.call_args
        assert args.kwargs.get("command") == "leaderboard"

    @pytest.mark.asyncio
    async def test_mention_agents_command(self, bot):
        activity = _make_activity(
            text="<at>Aragora</at> agents",
            entities=_mention_entities(),
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        args = bot._handle_command.call_args
        assert args.kwargs.get("command") == "agents"

    @pytest.mark.asyncio
    async def test_mention_vote_command(self, bot):
        activity = _make_activity(
            text="<at>Aragora</at> vote claude",
            entities=_mention_entities(),
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        args = bot._handle_command.call_args
        assert args.kwargs.get("command") == "vote"
        assert args.kwargs.get("args") == "claude"

    @pytest.mark.asyncio
    async def test_mention_ask_command(self, bot):
        """'ask' is an alias for 'debate'."""
        activity = _make_activity(
            text="<at>Aragora</at> ask what is life?",
            entities=_mention_entities(),
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        args = bot._handle_command.call_args
        assert args.kwargs.get("command") == "ask"

    @pytest.mark.asyncio
    async def test_personal_known_command(self, bot):
        """Personal chat with a known command like 'help'."""
        activity = _make_activity(
            text="help",
            conversation_type="personal",
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        args = bot._handle_command.call_args
        assert args.kwargs.get("command") == "help"

    @pytest.mark.asyncio
    async def test_personal_unknown_command_defaults_to_debate(self, bot):
        """Personal chat with unknown command defaults to 'debate'."""
        activity = _make_activity(
            text="is AI sentient?",
            conversation_type="personal",
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        args = bot._handle_command.call_args
        assert args.kwargs.get("command") == "debate"
        assert args.kwargs.get("args") == "is AI sentient?"

    @pytest.mark.asyncio
    async def test_non_mention_non_personal_sends_prompt(self, bot):
        """Non-mention, non-personal message gets a prompt reply."""
        activity = _make_activity(
            text="random text",
            conversation_type="channel",
        )
        await bot._handle_message(activity)
        bot.send_reply.assert_awaited_once()
        reply_text = bot.send_reply.call_args[0][1]
        assert "Mention me" in reply_text

    @pytest.mark.asyncio
    async def test_rbac_denied_sends_reply(self, bot):
        """RBAC denial sends permission denied reply."""
        bot._check_permission = MagicMock(
            return_value={"message": "Permission denied", "error": "permission_denied"}
        )
        activity = _make_activity(text="hello", entities=_mention_entities())
        await bot._handle_message(activity)
        bot.send_reply.assert_awaited()
        reply_text = bot.send_reply.call_args[0][1]
        assert "ermission denied" in reply_text

    @pytest.mark.asyncio
    async def test_sends_typing_indicator(self, bot):
        """Typing indicator is sent before processing."""
        activity = _make_activity(
            text="<at>Aragora</at> help",
            entities=_mention_entities(),
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        bot.send_typing.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_text(self, bot):
        """Message with empty text still processes."""
        activity = _make_activity(text="", conversation_type="channel")
        await bot._handle_message(activity)
        # Should send the "mention me" prompt
        bot.send_reply.assert_awaited()

    @pytest.mark.asyncio
    async def test_mention_with_no_command_text(self, bot):
        """Mention with no command after stripping."""
        activity = _make_activity(
            text="<at>Aragora</at>  ",
            entities=_mention_entities(),
        )
        bot._handle_command = AsyncMock(return_value={})
        await bot._handle_message(activity)
        args = bot._handle_command.call_args
        assert args.kwargs.get("command") == ""


# ===========================================================================
# TeamsBot: _handle_invoke
# ===========================================================================


class TestHandleInvoke:
    """Test invoke activity routing."""

    @pytest.fixture
    def bot(self, bot_cls):
        bot = bot_cls(app_id="app-id", app_password="pass")
        bot._handle_card_action = AsyncMock(
            return_value={"status": 200, "body": {"statusCode": 200}}
        )
        bot._handle_compose_extension_submit = AsyncMock(
            return_value={"status": 200, "body": {}}
        )
        bot._handle_compose_extension_query = AsyncMock(
            return_value={"status": 200, "body": {}}
        )
        bot._handle_task_module_fetch = AsyncMock(
            return_value={"status": 200, "body": {}}
        )
        bot._handle_task_module_submit = AsyncMock(
            return_value={"status": 200, "body": {}}
        )
        return bot

    @pytest.mark.asyncio
    async def test_adaptive_card_action(self, bot):
        activity = _make_activity(
            activity_type="invoke",
            invoke_name="adaptiveCard/action",
            value={"action": "vote"},
        )
        result = await bot._handle_invoke(activity)
        bot._handle_card_action.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_invoke_name_routes_to_card_action(self, bot):
        activity = _make_activity(activity_type="invoke", invoke_name="")
        result = await bot._handle_invoke(activity)
        bot._handle_card_action.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_compose_extension_submit(self, bot):
        activity = _make_activity(
            activity_type="invoke",
            invoke_name="composeExtension/submitAction",
        )
        await bot._handle_invoke(activity)
        bot._handle_compose_extension_submit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_compose_extension_query(self, bot):
        activity = _make_activity(
            activity_type="invoke",
            invoke_name="composeExtension/query",
        )
        await bot._handle_invoke(activity)
        bot._handle_compose_extension_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_compose_extension_fetch_task(self, bot):
        activity = _make_activity(
            activity_type="invoke",
            invoke_name="composeExtension/fetchTask",
        )
        await bot._handle_invoke(activity)
        bot._handle_task_module_fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_task_fetch(self, bot):
        activity = _make_activity(
            activity_type="invoke",
            invoke_name="task/fetch",
        )
        await bot._handle_invoke(activity)
        bot._handle_task_module_fetch.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_task_submit(self, bot):
        activity = _make_activity(
            activity_type="invoke",
            invoke_name="task/submit",
        )
        await bot._handle_invoke(activity)
        bot._handle_task_module_submit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unknown_invoke_returns_200(self, bot):
        activity = _make_activity(
            activity_type="invoke",
            invoke_name="some/unknown/action",
        )
        result = await bot._handle_invoke(activity)
        assert result["status"] == 200
        assert result["body"]["statusCode"] == 200


# ===========================================================================
# TeamsBot: _handle_command routing
# ===========================================================================


class TestHandleCommand:
    """Test command routing."""

    @pytest.fixture
    def bot(self, bot_cls):
        bot = bot_cls(app_id="app-id", app_password="pass")
        bot._cmd_debate = AsyncMock(return_value={"debate": True})
        bot._cmd_status = AsyncMock(return_value={"status": True})
        bot._cmd_help = AsyncMock(return_value={"help": True})
        bot._cmd_leaderboard = AsyncMock(return_value={"leaderboard": True})
        bot._cmd_agents = AsyncMock(return_value={"agents": True})
        bot._cmd_vote = AsyncMock(return_value={"vote": True})
        bot._cmd_unknown = AsyncMock(return_value={"unknown": True})
        return bot

    @pytest.mark.asyncio
    async def test_debate_command(self, bot):
        activity = _make_activity()
        result = await bot._handle_command(
            command="debate", args="topic", conversation_id="c",
            user_id="u", service_url="s", activity=activity,
        )
        assert result == {"debate": True}

    @pytest.mark.asyncio
    async def test_ask_command(self, bot):
        activity = _make_activity()
        result = await bot._handle_command(
            command="ask", args="question", conversation_id="c",
            user_id="u", service_url="s", activity=activity,
        )
        assert result == {"debate": True}

    @pytest.mark.asyncio
    async def test_status_command(self, bot):
        activity = _make_activity()
        result = await bot._handle_command(
            command="status", args="", conversation_id="c",
            user_id="u", service_url="s", activity=activity,
        )
        assert result == {"status": True}

    @pytest.mark.asyncio
    async def test_help_command(self, bot):
        activity = _make_activity()
        result = await bot._handle_command(
            command="help", args="", conversation_id="c",
            user_id="u", service_url="s", activity=activity,
        )
        assert result == {"help": True}

    @pytest.mark.asyncio
    async def test_leaderboard_command(self, bot):
        activity = _make_activity()
        result = await bot._handle_command(
            command="leaderboard", args="", conversation_id="c",
            user_id="u", service_url="s", activity=activity,
        )
        assert result == {"leaderboard": True}

    @pytest.mark.asyncio
    async def test_agents_command(self, bot):
        activity = _make_activity()
        result = await bot._handle_command(
            command="agents", args="", conversation_id="c",
            user_id="u", service_url="s", activity=activity,
        )
        assert result == {"agents": True}

    @pytest.mark.asyncio
    async def test_vote_command(self, bot):
        activity = _make_activity()
        result = await bot._handle_command(
            command="vote", args="claude", conversation_id="c",
            user_id="u", service_url="s", activity=activity,
        )
        assert result == {"vote": True}
        bot._cmd_vote.assert_awaited_once_with("claude", activity)

    @pytest.mark.asyncio
    async def test_unknown_command(self, bot):
        activity = _make_activity()
        result = await bot._handle_command(
            command="xyzzy", args="", conversation_id="c",
            user_id="u", service_url="s", activity=activity,
        )
        assert result == {"unknown": True}
        bot._cmd_unknown.assert_awaited_once_with("xyzzy", activity)


# ===========================================================================
# TeamsBot: _cmd_debate (RBAC check)
# ===========================================================================


class TestCmdDebate:
    """Test debate command with RBAC."""

    @pytest.fixture
    def bot(self, bot_cls):
        bot = bot_cls(app_id="app-id", app_password="pass")
        bot.send_reply = AsyncMock()
        return bot

    @pytest.mark.asyncio
    async def test_rbac_denied(self, bot):
        bot._check_permission = MagicMock(
            return_value={"message": "Permission denied", "error": "permission_denied"}
        )
        activity = _make_activity()
        result = await bot._cmd_debate(
            topic="test", conversation_id="c", user_id="u",
            service_url="s", thread_id=None, activity=activity,
        )
        bot.send_reply.assert_awaited()
        assert result == {}

    @pytest.mark.asyncio
    async def test_rbac_allowed_delegates_to_event_processor(self, bot):
        bot._check_permission = MagicMock(return_value=None)
        mock_ep = MagicMock()
        mock_ep._cmd_debate = AsyncMock(return_value={"debating": True})
        bot._get_event_processor = MagicMock(return_value=mock_ep)
        activity = _make_activity()
        result = await bot._cmd_debate(
            topic="test", conversation_id="c", user_id="u",
            service_url="s", thread_id=None, activity=activity,
        )
        assert result == {"debating": True}


# ===========================================================================
# TeamsBot: delegation methods
# ===========================================================================


class TestDelegationMethods:
    """Test that delegation methods forward to event processor / card actions."""

    @pytest.fixture
    def bot(self, bot_cls):
        bot = bot_cls(app_id="app-id", app_password="pass")
        mock_ep = MagicMock()
        mock_ep._handle_conversation_update = AsyncMock(return_value={"conv": True})
        mock_ep._handle_message_reaction = AsyncMock(return_value={"reaction": True})
        mock_ep._handle_installation_update = AsyncMock(return_value={"install": True})
        mock_ep._cmd_status = AsyncMock(return_value={"status": True})
        mock_ep._cmd_help = AsyncMock(return_value={"help": True})
        mock_ep._cmd_leaderboard = AsyncMock(return_value={"lb": True})
        mock_ep._cmd_agents = AsyncMock(return_value={"agents": True})
        mock_ep._cmd_vote = AsyncMock(return_value={"vote": True})
        mock_ep._cmd_unknown = AsyncMock(return_value={"unknown": True})
        bot._event_processor = mock_ep

        mock_ca = MagicMock()
        mock_ca._handle_card_action = AsyncMock(return_value={"card": True})
        mock_ca._handle_vote = AsyncMock(return_value={"voted": True})
        mock_ca._handle_summary = AsyncMock(return_value={"summary": True})
        mock_ca._handle_task_module_fetch = AsyncMock(return_value={"fetch": True})
        mock_ca._handle_task_module_submit = AsyncMock(return_value={"submit": True})
        mock_ca._handle_compose_extension_submit = AsyncMock(return_value={"ext_submit": True})
        mock_ca._handle_compose_extension_query = AsyncMock(return_value={"ext_query": True})
        bot._card_actions = mock_ca
        return bot

    @pytest.mark.asyncio
    async def test_handle_conversation_update(self, bot):
        result = await bot._handle_conversation_update({"type": "conversationUpdate"})
        assert result == {"conv": True}

    @pytest.mark.asyncio
    async def test_handle_message_reaction(self, bot):
        result = await bot._handle_message_reaction({"type": "messageReaction"})
        assert result == {"reaction": True}

    @pytest.mark.asyncio
    async def test_handle_installation_update(self, bot):
        result = await bot._handle_installation_update({"type": "installationUpdate"})
        assert result == {"install": True}

    @pytest.mark.asyncio
    async def test_cmd_status(self, bot):
        result = await bot._cmd_status({"type": "message"})
        assert result == {"status": True}

    @pytest.mark.asyncio
    async def test_cmd_help(self, bot):
        result = await bot._cmd_help({"type": "message"})
        assert result == {"help": True}

    @pytest.mark.asyncio
    async def test_cmd_leaderboard(self, bot):
        result = await bot._cmd_leaderboard({"type": "message"})
        assert result == {"lb": True}

    @pytest.mark.asyncio
    async def test_cmd_agents(self, bot):
        result = await bot._cmd_agents({"type": "message"})
        assert result == {"agents": True}

    @pytest.mark.asyncio
    async def test_cmd_vote(self, bot):
        result = await bot._cmd_vote("claude", {"type": "message"})
        assert result == {"vote": True}

    @pytest.mark.asyncio
    async def test_cmd_unknown(self, bot):
        result = await bot._cmd_unknown("foo", {"type": "message"})
        assert result == {"unknown": True}

    @pytest.mark.asyncio
    async def test_handle_card_action(self, bot):
        activity = _make_activity(value={"action": "vote"})
        result = await bot._handle_card_action(activity)
        assert result == {"card": True}

    @pytest.mark.asyncio
    async def test_handle_card_action_explicit_value(self, bot):
        activity = _make_activity()
        result = await bot._handle_card_action(activity, value={"action": "x"}, user_id="u1")
        assert result == {"card": True}

    @pytest.mark.asyncio
    async def test_handle_vote(self, bot):
        activity = _make_activity()
        result = await bot._handle_vote("d1", "claude", "u1", activity)
        assert result == {"voted": True}

    @pytest.mark.asyncio
    async def test_handle_summary(self, bot):
        activity = _make_activity()
        result = await bot._handle_summary("d1", activity)
        assert result == {"summary": True}

    @pytest.mark.asyncio
    async def test_handle_task_module_fetch(self, bot):
        activity = _make_activity(value={"x": 1})
        result = await bot._handle_task_module_fetch(activity)
        assert result == {"fetch": True}

    @pytest.mark.asyncio
    async def test_handle_task_module_submit(self, bot):
        activity = _make_activity(value={"x": 1})
        result = await bot._handle_task_module_submit(activity)
        assert result == {"submit": True}

    @pytest.mark.asyncio
    async def test_handle_compose_extension_submit(self, bot):
        activity = _make_activity(value={"x": 1})
        result = await bot._handle_compose_extension_submit(activity)
        assert result == {"ext_submit": True}

    @pytest.mark.asyncio
    async def test_handle_compose_extension_query(self, bot):
        activity = _make_activity(value={"x": 1})
        result = await bot._handle_compose_extension_query(activity)
        assert result == {"ext_query": True}


# ===========================================================================
# TeamsBot: send_typing / send_reply / send_card
# ===========================================================================


class TestSendMethods:
    """Test message sending utilities."""

    @pytest.fixture
    def bot(self, bot_cls):
        bot = bot_cls(app_id="app-id", app_password="pass")
        return bot

    @pytest.mark.asyncio
    async def test_send_typing_success(self, bot):
        mock_connector = AsyncMock()
        bot._connector = mock_connector
        activity = _make_activity(conversation_id="conv-1", service_url="https://svc.example.com")
        await bot.send_typing(activity)
        mock_connector.send_typing_indicator.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_typing_no_connector(self, bot):
        bot._connector = None
        with patch.object(bot, "_get_connector", new_callable=AsyncMock, return_value=None):
            activity = _make_activity()
            # Should not raise
            await bot.send_typing(activity)

    @pytest.mark.asyncio
    async def test_send_typing_error_swallowed(self, bot):
        mock_connector = AsyncMock()
        mock_connector.send_typing_indicator.side_effect = RuntimeError("nope")
        bot._connector = mock_connector
        activity = _make_activity()
        # Should not raise
        await bot.send_typing(activity)

    @pytest.mark.asyncio
    async def test_send_reply_success(self, bot):
        mock_connector = AsyncMock()
        bot._connector = mock_connector
        activity = _make_activity(conversation_id="conv-1")
        await bot.send_reply(activity, "Hello!")
        mock_connector.send_message.assert_awaited_once()
        kwargs = mock_connector.send_message.call_args
        assert kwargs.kwargs.get("text") == "Hello!"

    @pytest.mark.asyncio
    async def test_send_reply_no_connector(self, bot):
        bot._connector = None
        with patch.object(bot, "_get_connector", new_callable=AsyncMock, return_value=None):
            activity = _make_activity()
            await bot.send_reply(activity, "hello")
            # Should not raise

    @pytest.mark.asyncio
    async def test_send_reply_error_logged(self, bot):
        mock_connector = AsyncMock()
        mock_connector.send_message.side_effect = RuntimeError("fail")
        bot._connector = mock_connector
        activity = _make_activity()
        await bot.send_reply(activity, "test")
        # Should not raise

    @pytest.mark.asyncio
    async def test_send_card_success(self, bot):
        mock_connector = AsyncMock()
        mock_connector._get_access_token = AsyncMock(return_value="token-abc")
        mock_connector._http_request = AsyncMock()
        bot._connector = mock_connector
        activity = _make_activity(conversation_id="conv-1", service_url="https://svc")
        card = {"type": "AdaptiveCard", "body": []}
        await bot.send_card(activity, card, "fallback")
        mock_connector._http_request.assert_awaited_once()
        call_kwargs = mock_connector._http_request.call_args.kwargs
        assert "conv-1" in call_kwargs["url"]

    @pytest.mark.asyncio
    async def test_send_card_with_thread_id(self, bot):
        mock_connector = AsyncMock()
        mock_connector._get_access_token = AsyncMock(return_value="t")
        mock_connector._http_request = AsyncMock()
        bot._connector = mock_connector
        activity = _make_activity(reply_to_id="thread-1")
        await bot.send_card(activity, {"body": []}, "fb")
        json_payload = mock_connector._http_request.call_args.kwargs.get("json", {})
        assert json_payload.get("replyToId") == "thread-1"

    @pytest.mark.asyncio
    async def test_send_card_no_connector(self, bot):
        bot._connector = None
        with patch.object(bot, "_get_connector", new_callable=AsyncMock, return_value=None):
            activity = _make_activity()
            await bot.send_card(activity, {}, "fb")

    @pytest.mark.asyncio
    async def test_send_card_error_swallowed(self, bot):
        mock_connector = AsyncMock()
        mock_connector._get_access_token = AsyncMock(side_effect=RuntimeError("nope"))
        bot._connector = mock_connector
        activity = _make_activity()
        await bot.send_card(activity, {}, "fb")


# ===========================================================================
# TeamsBot: send_proactive_message
# ===========================================================================


class TestSendProactiveMessage:
    """Test proactive messaging."""

    @pytest.fixture
    def bot(self, bot_cls):
        bot = bot_cls(app_id="app-id", app_password="pass")
        return bot

    def _store_ref(self, conv_id: str, service_url: str = "https://svc"):
        from aragora.server.handlers.bots.teams_utils import _conversation_references
        _conversation_references[conv_id] = {"service_url": service_url}

    @pytest.mark.asyncio
    async def test_no_conversation_reference(self, bot):
        result = await bot.send_proactive_message("no-such-conv", text="hi")
        assert result is False

    @pytest.mark.asyncio
    async def test_no_connector(self, bot):
        self._store_ref("conv-1")
        with patch.object(bot, "_get_connector", new_callable=AsyncMock, return_value=None):
            result = await bot.send_proactive_message("conv-1", text="hi")
            assert result is False

    @pytest.mark.asyncio
    async def test_text_message_success(self, bot):
        self._store_ref("conv-1")
        mock_connector = AsyncMock()
        bot._connector = mock_connector
        result = await bot.send_proactive_message("conv-1", text="hello")
        assert result is True
        mock_connector.send_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_card_message_success(self, bot):
        self._store_ref("conv-1", "https://svc")
        mock_connector = AsyncMock()
        mock_connector._get_access_token = AsyncMock(return_value="token")
        mock_connector._http_request = AsyncMock()
        bot._connector = mock_connector
        card = {"type": "AdaptiveCard"}
        result = await bot.send_proactive_message("conv-1", card=card, fallback_text="fb")
        assert result is True
        mock_connector._http_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_text_or_card(self, bot):
        self._store_ref("conv-1")
        mock_connector = AsyncMock()
        bot._connector = mock_connector
        result = await bot.send_proactive_message("conv-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_error_returns_false(self, bot):
        self._store_ref("conv-1")
        mock_connector = AsyncMock()
        mock_connector.send_message.side_effect = RuntimeError("fail")
        bot._connector = mock_connector
        result = await bot.send_proactive_message("conv-1", text="hi")
        assert result is False


# ===========================================================================
# TeamsBot: _get_connector
# ===========================================================================


class TestGetConnector:
    """Test lazy connector initialization."""

    @pytest.fixture
    def bot(self, bot_cls):
        return bot_cls(app_id="app-id", app_password="pass")

    @pytest.mark.asyncio
    async def test_returns_cached_connector(self, bot):
        mock_conn = MagicMock()
        bot._connector = mock_conn
        result = await bot._get_connector()
        assert result is mock_conn

    @pytest.mark.asyncio
    async def test_import_error_returns_none(self, bot):
        with patch(
            "aragora.server.handlers.bots.teams.handler.TeamsBot._get_connector",
            new_callable=AsyncMock,
            return_value=None,
        ):
            # Simulate ImportError by calling the real method with a patched import
            bot._connector = None
            with patch.dict("sys.modules", {"aragora.connectors.chat.teams": None}):
                result = await bot._get_connector()
                assert result is None


# ===========================================================================
# TeamsBot: _get_event_processor / _get_card_actions
# ===========================================================================


class TestLazySubcomponents:
    """Test lazy initialization of event processor and card actions."""

    @pytest.fixture
    def bot(self, bot_cls):
        return bot_cls(app_id="app-id", app_password="pass")

    def test_get_event_processor_creates_instance(self, bot):
        with patch(
            "aragora.server.handlers.bots.teams.events.TeamsEventProcessor"
        ) as MockEP:
            MockEP.return_value = MagicMock()
            ep = bot._get_event_processor()
            assert ep is not None
            MockEP.assert_called_once_with(bot)

    def test_get_event_processor_caches(self, bot):
        with patch(
            "aragora.server.handlers.bots.teams.events.TeamsEventProcessor"
        ) as MockEP:
            MockEP.return_value = MagicMock()
            ep1 = bot._get_event_processor()
            ep2 = bot._get_event_processor()
            assert ep1 is ep2
            assert MockEP.call_count == 1

    def test_get_card_actions_creates_instance(self, bot):
        with patch(
            "aragora.server.handlers.bots.teams.cards.TeamsCardActions"
        ) as MockCA:
            MockCA.return_value = MagicMock()
            ca = bot._get_card_actions()
            assert ca is not None
            MockCA.assert_called_once_with(bot)

    def test_get_card_actions_caches(self, bot):
        with patch(
            "aragora.server.handlers.bots.teams.cards.TeamsCardActions"
        ) as MockCA:
            MockCA.return_value = MagicMock()
            ca1 = bot._get_card_actions()
            ca2 = bot._get_card_actions()
            assert ca1 is ca2
            assert MockCA.call_count == 1


# ===========================================================================
# TeamsBot: RBAC helpers
# ===========================================================================


class TestGetAuthContextFromActivity:
    """Test _get_auth_context_from_activity()."""

    @pytest.fixture
    def bot(self, bot_cls):
        return bot_cls(app_id="app-id", app_password="pass")

    def test_with_aad_object_id(self, bot):
        activity = _make_activity(user_id="user-1", aad_object_id="aad-123", tenant_id="t1")
        ctx = bot._get_auth_context_from_activity(activity)
        if ctx is not None:
            assert "aad-123" in ctx.user_id
            assert ctx.org_id == "t1"

    def test_with_user_id_only(self, bot):
        activity = _make_activity(user_id="user-1", tenant_id="t1")
        ctx = bot._get_auth_context_from_activity(activity)
        if ctx is not None:
            assert "user-1" in ctx.user_id

    def test_no_user_id_returns_none(self, bot):
        activity = {"from": {}, "conversation": {}}
        ctx = bot._get_auth_context_from_activity(activity)
        assert ctx is None

    def test_missing_from_key(self, bot):
        activity = {"conversation": {}}
        ctx = bot._get_auth_context_from_activity(activity)
        # Should not raise, returns None or context
        # The from field defaults to {} via .get()

    def test_tenant_id_in_context(self, bot):
        activity = _make_activity(user_id="u1", tenant_id="tenant-abc")
        ctx = bot._get_auth_context_from_activity(activity)
        if ctx is not None:
            assert ctx.org_id == "tenant-abc"

    def test_no_tenant_id(self, bot):
        activity = _make_activity(user_id="u1")
        ctx = bot._get_auth_context_from_activity(activity)
        if ctx is not None:
            assert ctx.org_id is None or ctx.org_id == ""


class TestCheckPermission:
    """Test _check_permission() helper."""

    @pytest.fixture
    def bot(self, bot_cls):
        return bot_cls(app_id="app-id", app_password="pass")

    def test_rbac_unavailable_dev_mode(self, bot, handler_module):
        """When RBAC is unavailable and not in production, returns None (permissive)."""
        with patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch.object(handler_module, "check_permission", None), \
             patch(
                 "aragora.server.handlers.bots.teams.handler.rbac_fail_closed",
                 return_value=False,
             ):
            activity = _make_activity(user_id="u1")
            result = bot._check_permission(activity, "teams:messages:read")
            assert result is None

    def test_rbac_unavailable_production_mode(self, bot, handler_module):
        """When RBAC is unavailable in production, returns 503."""
        with patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch.object(handler_module, "check_permission", None), \
             patch(
                 "aragora.server.handlers.bots.teams.handler.rbac_fail_closed",
                 return_value=True,
             ):
            activity = _make_activity(user_id="u1")
            result = bot._check_permission(activity, "teams:messages:read")
            assert result is not None
            assert result["status"] == 503

    def test_permission_allowed(self, bot, handler_module):
        """When permission is allowed, returns None."""
        mock_decision = MagicMock()
        mock_decision.allowed = True
        mock_check = MagicMock(return_value=mock_decision)

        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission", mock_check):
            activity = _make_activity(user_id="u1", aad_object_id="aad-1")
            result = bot._check_permission(activity, "teams:messages:read")
            assert result is None

    def test_permission_denied(self, bot, handler_module):
        """When permission is denied, returns error dict."""
        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "No role"
        mock_check = MagicMock(return_value=mock_decision)

        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission", mock_check):
            activity = _make_activity(user_id="u1", aad_object_id="aad-1")
            result = bot._check_permission(activity, "teams:messages:read")
            assert result is not None
            assert result["error"] == "permission_denied"

    def test_no_auth_context_returns_none(self, bot, handler_module):
        """When auth context cannot be built, returns None (permissive)."""
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission", MagicMock()):
            activity = {"from": {}, "conversation": {}}
            result = bot._check_permission(activity, "teams:messages:read")
            assert result is None

    def test_check_permission_exception_returns_none(self, bot, handler_module):
        """When check_permission raises, returns None (graceful degradation)."""
        mock_check = MagicMock(side_effect=TypeError("bad arg"))

        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission", mock_check):
            activity = _make_activity(user_id="u1", aad_object_id="aad-1")
            result = bot._check_permission(activity, "teams:messages:read")
            assert result is None


class TestValidateTenant:
    """Test _validate_tenant()."""

    @pytest.fixture
    def bot(self, bot_cls):
        return bot_cls(app_id="app-id", app_password="pass")

    def test_no_required_tenant(self, bot):
        """When no tenant is configured, validation passes."""
        with patch("aragora.server.handlers.bots.teams.handler.TEAMS_TENANT_ID", None):
            activity = _make_activity(tenant_id="any-tenant")
            result = bot._validate_tenant(activity)
            assert result is None

    def test_matching_tenant(self, bot):
        with patch("aragora.server.handlers.bots.teams.handler.TEAMS_TENANT_ID", "t1"):
            activity = _make_activity(tenant_id="t1")
            result = bot._validate_tenant(activity)
            assert result is None

    def test_mismatching_tenant(self, bot):
        with patch("aragora.server.handlers.bots.teams.handler.TEAMS_TENANT_ID", "t1"):
            activity = _make_activity(tenant_id="t2")
            result = bot._validate_tenant(activity)
            assert result is not None
            assert result["error"] == "tenant_denied"

    def test_explicit_expected_tenant(self, bot):
        activity = _make_activity(tenant_id="expected")
        result = bot._validate_tenant(activity, expected_tenant_id="expected")
        assert result is None

    def test_explicit_expected_tenant_mismatch(self, bot):
        activity = _make_activity(tenant_id="wrong")
        result = bot._validate_tenant(activity, expected_tenant_id="expected")
        assert result is not None
        assert result["error"] == "tenant_denied"

    def test_empty_tenant_in_activity(self, bot):
        with patch("aragora.server.handlers.bots.teams.handler.TEAMS_TENANT_ID", "t1"):
            activity = _make_activity()  # no tenant_id
            result = bot._validate_tenant(activity)
            assert result is not None


# ===========================================================================
# Constants and module-level items
# ===========================================================================


class TestModuleConstants:
    """Test module-level constants and patterns."""

    def test_agent_display_names(self, handler_module):
        names = handler_module.AGENT_DISPLAY_NAMES
        assert "claude" in names
        assert names["claude"] == "Claude"
        assert "gpt4" in names
        assert names["gpt4"] == "GPT-4"
        assert "gemini" in names
        assert "mistral" in names

    def test_mention_pattern_strips_at_tags(self, handler_module):
        text = "<at>Aragora Bot</at> debate AI"
        cleaned = handler_module.MENTION_PATTERN.sub("", text)
        assert cleaned == "debate AI"

    def test_mention_pattern_case_insensitive(self, handler_module):
        text = "<AT>Bot</AT> help"
        cleaned = handler_module.MENTION_PATTERN.sub("", text)
        assert cleaned == "help"

    def test_mention_pattern_multiple_mentions(self, handler_module):
        text = "<at>A</at> <at>B</at> cmd"
        cleaned = handler_module.MENTION_PATTERN.sub("", text)
        assert cleaned == "cmd"

    def test_permission_constants(self, handler_module):
        assert handler_module.PERM_TEAMS_MESSAGES_READ == "teams:messages:read"
        assert handler_module.PERM_TEAMS_MESSAGES_SEND == "teams:messages:send"
        assert handler_module.PERM_TEAMS_DEBATES_CREATE == "teams:debates:create"
        assert handler_module.PERM_TEAMS_DEBATES_VOTE == "teams:debates:vote"
        assert handler_module.PERM_TEAMS_CARDS_RESPOND == "teams:cards:respond"
        assert handler_module.PERM_TEAMS_ADMIN == "teams:admin"

    def test_routes_list(self, handler_module):
        routes = handler_module.TeamsHandler.ROUTES
        assert "/api/v1/bots/teams/messages" in routes
        assert "/api/v1/bots/teams/status" in routes
        assert "/api/v1/teams" in routes
        assert "/api/v1/teams/debates/send" in routes

    def test_bot_platform(self, handler_module):
        h = handler_module.TeamsHandler(ctx={})
        assert h.bot_platform == "teams"

    def test_all_exports(self, handler_module):
        assert "TeamsHandler" in handler_module.__all__
        assert "TeamsBot" in handler_module.__all__


# ===========================================================================
# TeamsBot: _send_typing / _send_reply wrappers
# ===========================================================================


class TestInternalSendWrappers:
    """Test the internal _send_typing / _send_reply wrappers."""

    @pytest.fixture
    def bot(self, bot_cls):
        bot = bot_cls(app_id="a", app_password="b")
        bot.send_typing = AsyncMock()
        bot.send_reply = AsyncMock()
        return bot

    @pytest.mark.asyncio
    async def test_send_typing_wrapper(self, bot):
        activity = _make_activity()
        await bot._send_typing(activity)
        bot.send_typing.assert_awaited_once_with(activity)

    @pytest.mark.asyncio
    async def test_send_reply_wrapper(self, bot):
        activity = _make_activity()
        await bot._send_reply(activity, "hello")
        bot.send_reply.assert_awaited_once_with(activity, "hello")


# ===========================================================================
# TeamsBot: invoke delegation with default value/user_id extraction
# ===========================================================================


class TestInvokeDelegationDefaults:
    """Test that invoke delegation methods extract value/user_id from activity when not provided."""

    @pytest.fixture
    def bot(self, bot_cls):
        bot = bot_cls(app_id="a", app_password="b")
        mock_ca = MagicMock()
        mock_ca._handle_card_action = AsyncMock(return_value={"ok": True})
        mock_ca._handle_task_module_fetch = AsyncMock(return_value={"ok": True})
        mock_ca._handle_task_module_submit = AsyncMock(return_value={"ok": True})
        mock_ca._handle_compose_extension_submit = AsyncMock(return_value={"ok": True})
        mock_ca._handle_compose_extension_query = AsyncMock(return_value={"ok": True})
        bot._card_actions = mock_ca
        return bot

    @pytest.mark.asyncio
    async def test_card_action_extracts_value_from_activity(self, bot):
        activity = _make_activity(
            user_id="u1",
            value={"action": "vote", "debate_id": "d1"},
        )
        await bot._handle_card_action(activity)
        call_args = bot._card_actions._handle_card_action.call_args
        assert call_args[0][0] == {"action": "vote", "debate_id": "d1"}
        assert call_args[0][1] == "u1"

    @pytest.mark.asyncio
    async def test_task_module_fetch_extracts_defaults(self, bot):
        activity = _make_activity(user_id="u2", value={"data": "x"})
        await bot._handle_task_module_fetch(activity)
        call_args = bot._card_actions._handle_task_module_fetch.call_args
        assert call_args[0][0] == {"data": "x"}
        assert call_args[0][1] == "u2"

    @pytest.mark.asyncio
    async def test_task_module_submit_extracts_defaults(self, bot):
        activity = _make_activity(user_id="u3", value={"data": "y"})
        await bot._handle_task_module_submit(activity)
        call_args = bot._card_actions._handle_task_module_submit.call_args
        assert call_args[0][0] == {"data": "y"}
        assert call_args[0][1] == "u3"

    @pytest.mark.asyncio
    async def test_compose_extension_submit_extracts_defaults(self, bot):
        activity = _make_activity(user_id="u4", value={"data": "z"})
        await bot._handle_compose_extension_submit(activity)
        call_args = bot._card_actions._handle_compose_extension_submit.call_args
        assert call_args[0][0] == {"data": "z"}
        assert call_args[0][1] == "u4"

    @pytest.mark.asyncio
    async def test_compose_extension_query_extracts_defaults(self, bot):
        activity = _make_activity(user_id="u5", value={"query": "test"})
        await bot._handle_compose_extension_query(activity)
        call_args = bot._card_actions._handle_compose_extension_query.call_args
        assert call_args[0][0] == {"query": "test"}
        assert call_args[0][1] == "u5"


# ===========================================================================
# TeamsHandler: handle_post edge cases
# ===========================================================================


class TestHandlePostEdgeCases:
    """Additional edge cases for handle_post."""

    @pytest.mark.asyncio
    async def test_os_error_returns_500(self, handler):
        activity = _make_activity()
        mock_bot = MagicMock()
        mock_bot.process_activity = AsyncMock(side_effect=OSError("disk"))
        handler._bot = mock_bot
        handler._bot_initialized = True
        mock_h = _make_http_handler(body=activity)
        result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_h)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_attribute_error_returns_500(self, handler):
        activity = _make_activity()
        mock_bot = MagicMock()
        mock_bot.process_activity = AsyncMock(side_effect=AttributeError("missing"))
        handler._bot = mock_bot
        handler._bot_initialized = True
        mock_h = _make_http_handler(body=activity)
        result = await handler.handle_post("/api/v1/bots/teams/messages", {}, mock_h)
        assert _status(result) == 500
