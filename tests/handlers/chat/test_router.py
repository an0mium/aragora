"""Tests for ChatWebhookRouter and ChatHandler.

Covers:
- ChatWebhookRouter: platform detection (headers + body), connector caching,
  webhook verification, event parsing, webhook handling, verification challenges,
  event processing (commands, interactions, messages, voice), help text, status,
  input source mapping, decision routing
- ChatHandler: can_handle routing, handle() for status and method checks,
  handle_post() for platform detection from path/headers/body, rate limiting,
  large body rejection, auto-detection fallback
- Singleton get_webhook_router and _create_decision_router_debate_starter
- Utility functions: _handle_task_exception, create_tracked_task
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.chat.models import (
    BotCommand,
    ChatChannel,
    ChatMessage,
    ChatUser,
    FileAttachment,
    UserInteraction,
    InteractionType,
    VoiceMessage,
    WebhookEvent,
)
from aragora.server.handlers.chat.router import (
    ChatHandler,
    ChatWebhookRouter,
    _create_decision_router_debate_starter,
    _handle_task_exception,
    create_tracked_task,
    get_webhook_router,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Mock HTTP request handler for ChatHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
        headers: dict[str, str] | None = None,
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        if headers:
            self.headers.update(headers)
        self.rfile = MagicMock()

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


def _make_user(is_bot: bool = False, user_id: str = "U001") -> ChatUser:
    user = ChatUser(id=user_id, platform="slack", username="testuser", is_bot=is_bot)
    # Router code accesses .name (which ChatUser doesn't define); add it for compat
    user.name = user.username  # type: ignore[attr-defined]
    return user


def _make_channel(channel_id: str = "C001") -> ChatChannel:
    return ChatChannel(id=channel_id, platform="slack", name="general")


def _make_message(
    content: str = "Hello world", is_bot: bool = False
) -> ChatMessage:
    return ChatMessage(
        id="msg-001",
        platform="slack",
        channel=_make_channel(),
        author=_make_user(is_bot=is_bot),
        content=content,
    )


def _make_command(
    name: str = "aragora",
    args: list[str] | None = None,
    user: ChatUser | None = None,
    channel: ChatChannel | None = None,
) -> BotCommand:
    return BotCommand(
        name=name,
        text=f"/{name} {' '.join(args or [])}",
        args=args or [],
        user=user or _make_user(),
        channel=channel or _make_channel(),
        platform="slack",
    )


def _make_interaction(action_id: str = "btn-1") -> UserInteraction:
    return UserInteraction(
        id="int-001",
        interaction_type=InteractionType.BUTTON_CLICK,
        action_id=action_id,
        user=_make_user(),
        channel=_make_channel(),
        platform="slack",
    )


def _make_voice() -> VoiceMessage:
    return VoiceMessage(
        id="voice-001",
        channel=_make_channel(),
        author=_make_user(),
        duration_seconds=5.0,
        file=FileAttachment(id="f1", filename="audio.ogg", content_type="audio/ogg", size=1024),
        platform="slack",
    )


def _make_event(
    platform: str = "slack",
    event_type: str = "message",
    message: ChatMessage | None = None,
    command: BotCommand | None = None,
    interaction: UserInteraction | None = None,
    voice_message: VoiceMessage | None = None,
    challenge: str | None = None,
) -> WebhookEvent:
    return WebhookEvent(
        platform=platform,
        event_type=event_type,
        message=message,
        command=command,
        interaction=interaction,
        voice_message=voice_message,
        challenge=challenge,
    )


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- Platform detection (headers)
# ---------------------------------------------------------------------------


class TestDetectPlatformHeaders:
    """Test header-based platform detection."""

    def test_detect_slack(self):
        router = ChatWebhookRouter()
        headers = {"X-Slack-Signature": "v0=abc", "X-Slack-Request-Timestamp": "12345"}
        assert router.detect_platform(headers) == "slack"

    def test_detect_discord(self):
        router = ChatWebhookRouter()
        headers = {"X-Signature-Ed25519": "sig", "X-Signature-Timestamp": "12345"}
        assert router.detect_platform(headers) == "discord"

    def test_detect_telegram(self):
        router = ChatWebhookRouter()
        headers = {"X-Telegram-Bot-Api-Secret-Token": "secret"}
        assert router.detect_platform(headers) == "telegram"

    def test_detect_whatsapp(self):
        router = ChatWebhookRouter()
        headers = {"X-Hub-Signature-256": "sha256=abc"}
        assert router.detect_platform(headers) == "whatsapp"

    def test_detect_teams_bearer(self):
        router = ChatWebhookRouter()
        headers = {"Authorization": "Bearer some-token"}
        assert router.detect_platform(headers) == "teams"

    def test_detect_none_empty_headers(self):
        router = ChatWebhookRouter()
        assert router.detect_platform({}) is None

    def test_detect_none_unknown_headers(self):
        router = ChatWebhookRouter()
        headers = {"X-Custom": "val"}
        assert router.detect_platform(headers) is None

    def test_slack_priority_over_others(self):
        """Slack header should win if present alongside others."""
        router = ChatWebhookRouter()
        headers = {
            "X-Slack-Signature": "v0=abc",
            "X-Hub-Signature-256": "sha256=xyz",
        }
        assert router.detect_platform(headers) == "slack"

    def test_discord_priority_over_whatsapp(self):
        router = ChatWebhookRouter()
        headers = {
            "X-Signature-Ed25519": "sig",
            "X-Hub-Signature-256": "sha256=xyz",
        }
        assert router.detect_platform(headers) == "discord"


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- Platform detection (body)
# ---------------------------------------------------------------------------


class TestDetectPlatformBody:
    """Test body-based platform detection."""

    def test_detect_telegram_message(self):
        router = ChatWebhookRouter()
        body = json.dumps({"update_id": 1, "message": {"text": "hi"}}).encode()
        assert router.detect_platform_from_body({}, body) == "telegram"

    def test_detect_telegram_callback_query(self):
        router = ChatWebhookRouter()
        body = json.dumps({"update_id": 1, "callback_query": {}}).encode()
        assert router.detect_platform_from_body({}, body) == "telegram"

    def test_detect_telegram_inline_query(self):
        router = ChatWebhookRouter()
        body = json.dumps({"update_id": 1, "inline_query": {}}).encode()
        assert router.detect_platform_from_body({}, body) == "telegram"

    def test_detect_telegram_edited_message(self):
        router = ChatWebhookRouter()
        body = json.dumps({"update_id": 1, "edited_message": {}}).encode()
        assert router.detect_platform_from_body({}, body) == "telegram"

    def test_detect_telegram_channel_post(self):
        router = ChatWebhookRouter()
        body = json.dumps({"update_id": 1, "channel_post": {}}).encode()
        assert router.detect_platform_from_body({}, body) == "telegram"

    def test_telegram_update_id_only_no_match(self):
        """update_id alone without a recognized key should not detect telegram."""
        router = ChatWebhookRouter()
        body = json.dumps({"update_id": 1}).encode()
        assert router.detect_platform_from_body({}, body) is None

    def test_detect_whatsapp(self):
        router = ChatWebhookRouter()
        body = json.dumps({"object": "whatsapp_business_account", "entry": []}).encode()
        assert router.detect_platform_from_body({}, body) == "whatsapp"

    def test_facebook_page_not_whatsapp(self):
        router = ChatWebhookRouter()
        body = json.dumps({"object": "page", "entry": []}).encode()
        assert router.detect_platform_from_body({}, body) is None

    def test_instagram_not_whatsapp(self):
        router = ChatWebhookRouter()
        body = json.dumps({"object": "instagram", "entry": []}).encode()
        assert router.detect_platform_from_body({}, body) is None

    def test_detect_teams_service_url(self):
        router = ChatWebhookRouter()
        body = json.dumps(
            {"type": "message", "serviceUrl": "https://smba.trafficmanager.net/emea/"}
        ).encode()
        assert router.detect_platform_from_body({}, body) == "teams"

    def test_detect_teams_channel_id(self):
        router = ChatWebhookRouter()
        body = json.dumps({"type": "message", "channelId": "msteams"}).encode()
        assert router.detect_platform_from_body({}, body) == "teams"

    def test_detect_google_chat_message(self):
        router = ChatWebhookRouter()
        body = json.dumps(
            {"type": "MESSAGE", "message": {}, "space": {"name": "spaces/1"}}
        ).encode()
        assert router.detect_platform_from_body({}, body) == "google_chat"

    def test_detect_google_chat_added_to_space(self):
        router = ChatWebhookRouter()
        body = json.dumps({"type": "ADDED_TO_SPACE", "space": {}}).encode()
        assert router.detect_platform_from_body({}, body) == "google_chat"

    def test_detect_google_chat_removed(self):
        router = ChatWebhookRouter()
        body = json.dumps({"type": "REMOVED_FROM_SPACE", "space": {}}).encode()
        assert router.detect_platform_from_body({}, body) == "google_chat"

    def test_detect_google_chat_card_clicked(self):
        router = ChatWebhookRouter()
        body = json.dumps({"type": "CARD_CLICKED", "message": {}}).encode()
        assert router.detect_platform_from_body({}, body) == "google_chat"

    def test_detect_discord_from_body(self):
        router = ChatWebhookRouter()
        body = json.dumps({"type": 1, "application_id": "123"}).encode()
        assert router.detect_platform_from_body({}, body) == "discord"

    def test_discord_requires_int_type(self):
        """Discord detection requires type to be an int, not a string."""
        router = ChatWebhookRouter()
        body = json.dumps({"type": "1", "application_id": "123"}).encode()
        assert router.detect_platform_from_body({}, body) is None

    def test_detect_slack_from_body(self):
        router = ChatWebhookRouter()
        body = json.dumps({"token": "abc", "team_id": "T1", "api_app_id": "A1"}).encode()
        assert router.detect_platform_from_body({}, body) == "slack"

    def test_slack_partial_keys_no_match(self):
        router = ChatWebhookRouter()
        body = json.dumps({"token": "abc", "team_id": "T1"}).encode()
        assert router.detect_platform_from_body({}, body) is None

    def test_invalid_json(self):
        router = ChatWebhookRouter()
        assert router.detect_platform_from_body({}, b"not json") is None

    def test_invalid_utf8(self):
        router = ChatWebhookRouter()
        assert router.detect_platform_from_body({}, b"\xff\xfe") is None

    def test_unknown_structure(self):
        router = ChatWebhookRouter()
        body = json.dumps({"random": "stuff"}).encode()
        assert router.detect_platform_from_body({}, body) is None


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- Connector caching
# ---------------------------------------------------------------------------


class TestConnectorCaching:
    """Test connector caching behaviour."""

    def test_caches_connector(self):
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn) as p:
            c1 = router.get_connector("slack")
            c2 = router.get_connector("slack")
            assert c1 is c2
            assert p.call_count == 1

    def test_returns_none_for_unconfigured(self):
        router = ChatWebhookRouter()
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=None):
            assert router.get_connector("unknown") is None

    def test_different_platforms_cached_separately(self):
        router = ChatWebhookRouter()
        slack_conn = MagicMock(name="slack_conn")
        discord_conn = MagicMock(name="discord_conn")

        def side_effect(p):
            return slack_conn if p == "slack" else discord_conn

        with patch("aragora.server.handlers.chat.router.get_connector", side_effect=side_effect):
            s = router.get_connector("slack")
            d = router.get_connector("discord")
            assert s is slack_conn
            assert d is discord_conn


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- Webhook verification
# ---------------------------------------------------------------------------


class TestVerifyWebhook:
    """Test webhook signature verification delegation."""

    def test_verify_delegates_to_connector(self):
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            assert router.verify_webhook("slack", {"h": "v"}, b"body") is True
            mock_conn.verify_webhook.assert_called_once_with({"h": "v"}, b"body")

    def test_verify_returns_false_no_connector(self):
        router = ChatWebhookRouter()
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=None):
            assert router.verify_webhook("unknown", {}, b"") is False

    def test_verify_returns_false_on_fail(self):
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = False
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            assert router.verify_webhook("slack", {}, b"body") is False


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- Event parsing
# ---------------------------------------------------------------------------


class TestParseEvent:
    """Test webhook event parsing."""

    def test_parse_delegates_to_connector(self):
        router = ChatWebhookRouter()
        expected = _make_event()
        mock_conn = MagicMock()
        mock_conn.parse_webhook_event.return_value = expected
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = router.parse_event("slack", {}, b"body")
            assert result is expected

    def test_parse_returns_error_event_no_connector(self):
        router = ChatWebhookRouter()
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=None):
            result = router.parse_event("unknown", {}, b"")
            assert result.event_type == "error"
            assert "No connector" in result.raw_payload.get("error", "")


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- handle_webhook
# ---------------------------------------------------------------------------


class TestHandleWebhook:
    """Test handle_webhook flow."""

    @pytest.mark.asyncio
    async def test_failed_verification_returns_401(self):
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = False
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await router.handle_webhook("slack", {}, b"body")
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_verification_challenge_handled(self):
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        event = _make_event(challenge="test-challenge")
        mock_conn.parse_webhook_event.return_value = event
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await router.handle_webhook("slack", {}, b"body")
            assert result["challenge"] == "test-challenge"

    @pytest.mark.asyncio
    async def test_normal_event_processed(self):
        handler = AsyncMock()
        router = ChatWebhookRouter(event_handler=handler)
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        event = _make_event()
        mock_conn.parse_webhook_event.return_value = event
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await router.handle_webhook("slack", {}, b"body")
            assert result["success"] is True


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- Verification challenges
# ---------------------------------------------------------------------------


class TestVerificationChallenges:
    """Test _handle_verification for every platform."""

    def test_slack_challenge(self):
        router = ChatWebhookRouter()
        event = _make_event(platform="slack", challenge="xyz")
        result = router._handle_verification("slack", event)
        assert result == {"challenge": "xyz"}

    def test_discord_pong(self):
        router = ChatWebhookRouter()
        event = _make_event(platform="discord", challenge="ignored")
        result = router._handle_verification("discord", event)
        assert result == {"type": 1}

    def test_google_chat(self):
        router = ChatWebhookRouter()
        event = _make_event(platform="google_chat", challenge="c")
        result = router._handle_verification("google_chat", event)
        assert result == {"success": True}

    def test_telegram(self):
        router = ChatWebhookRouter()
        event = _make_event(platform="telegram", challenge="c")
        result = router._handle_verification("telegram", event)
        assert result == {"success": True}

    def test_whatsapp_with_challenge(self):
        router = ChatWebhookRouter()
        event = _make_event(platform="whatsapp", challenge="hub-challenge-val")
        result = router._handle_verification("whatsapp", event)
        assert result == {"hub.challenge": "hub-challenge-val"}

    def test_whatsapp_no_challenge(self):
        router = ChatWebhookRouter()
        event = _make_event(platform="whatsapp")
        result = router._handle_verification("whatsapp", event)
        assert result == {"success": True}

    def test_unknown_platform_verification(self):
        router = ChatWebhookRouter()
        event = _make_event(platform="some_other")
        result = router._handle_verification("some_other", event)
        assert result == {"success": True}


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- _process_event routing
# ---------------------------------------------------------------------------


class TestProcessEvent:
    """Test _process_event dispatching."""

    @pytest.mark.asyncio
    async def test_command_dispatched(self):
        router = ChatWebhookRouter()
        cmd = _make_command(name="test")
        event = _make_event(command=cmd)
        with patch.object(router, "_handle_command", new_callable=AsyncMock) as m:
            m.return_value = {"success": True}
            result = await router._process_event(event)
            m.assert_called_once_with(event)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_interaction_dispatched(self):
        router = ChatWebhookRouter()
        event = _make_event(interaction=_make_interaction())
        with patch.object(router, "_handle_interaction", new_callable=AsyncMock) as m:
            m.return_value = {"success": True}
            result = await router._process_event(event)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_dispatched(self):
        router = ChatWebhookRouter()
        event = _make_event(message=_make_message())
        with patch.object(router, "_handle_message", new_callable=AsyncMock) as m:
            m.return_value = {"success": True}
            result = await router._process_event(event)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_dispatched(self):
        router = ChatWebhookRouter()
        event = _make_event(voice_message=_make_voice())
        with patch.object(router, "_handle_voice", new_callable=AsyncMock) as m:
            m.return_value = {"success": True}
            result = await router._process_event(event)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_generic_event_calls_event_handler(self):
        handler = AsyncMock()
        router = ChatWebhookRouter(event_handler=handler)
        event = _make_event()  # No command/interaction/message/voice
        result = await router._process_event(event)
        handler.assert_called_once_with(event)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_generic_event_handler_error_caught(self):
        handler = AsyncMock(side_effect=RuntimeError("boom"))
        router = ChatWebhookRouter(event_handler=handler)
        event = _make_event()
        result = await router._process_event(event)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_generic_no_event_handler(self):
        router = ChatWebhookRouter()
        event = _make_event()
        result = await router._process_event(event)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- _handle_command
# ---------------------------------------------------------------------------


class TestHandleCommand:
    """Test _handle_command."""

    @pytest.mark.asyncio
    async def test_none_command_returns_success(self):
        router = ChatWebhookRouter()
        event = _make_event(command=None)
        event.command = None
        result = await router._handle_command(event)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_aragora_command_routes(self):
        router = ChatWebhookRouter()
        cmd = _make_command(name="aragora", args=["help"])
        event = _make_event(command=cmd)
        with patch.object(router, "_handle_aragora_command", new_callable=AsyncMock) as m:
            m.return_value = {"success": True}
            await router._handle_command(event)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_command_routes(self):
        router = ChatWebhookRouter()
        cmd = _make_command(name="debate", args=["topic"])
        event = _make_event(command=cmd)
        with patch.object(router, "_handle_aragora_command", new_callable=AsyncMock) as m:
            m.return_value = {"success": True}
            await router._handle_command(event)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_review_command_routes(self):
        router = ChatWebhookRouter()
        cmd = _make_command(name="review")
        event = _make_event(command=cmd)
        with patch.object(router, "_handle_aragora_command", new_callable=AsyncMock) as m:
            m.return_value = {"success": True}
            await router._handle_command(event)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_gauntlet_command_routes(self):
        router = ChatWebhookRouter()
        cmd = _make_command(name="gauntlet")
        event = _make_event(command=cmd)
        with patch.object(router, "_handle_aragora_command", new_callable=AsyncMock) as m:
            m.return_value = {"success": True}
            await router._handle_command(event)
            m.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_command_calls_event_handler(self):
        handler = AsyncMock()
        router = ChatWebhookRouter(event_handler=handler)
        cmd = _make_command(name="custom_cmd")
        event = _make_event(command=cmd)
        await router._handle_command(event)
        handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_unknown_command_handler_error_caught(self):
        handler = AsyncMock(side_effect=TypeError("bad"))
        router = ChatWebhookRouter(event_handler=handler)
        cmd = _make_command(name="custom_cmd")
        event = _make_event(command=cmd)
        result = await router._handle_command(event)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- _handle_aragora_command subcommands
# ---------------------------------------------------------------------------


class TestHandleAragonaCommand:
    """Test _handle_aragora_command subcommands."""

    def _router_with_connector(self):
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        mock_conn.format_blocks.return_value = [{"type": "section", "text": "hi"}]
        mock_conn.respond_to_command = AsyncMock()
        router._connectors["slack"] = mock_conn
        return router, mock_conn

    @pytest.mark.asyncio
    async def test_help_subcommand(self):
        router, conn = self._router_with_connector()
        cmd = _make_command(name="aragora", args=["help"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        assert result["success"] is True
        conn.respond_to_command.assert_called_once()
        call_args = conn.respond_to_command.call_args
        assert "help" in call_args[0][1].lower() or "Available Commands" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_help_default_no_args(self):
        """When no args, subcommand defaults to 'help'."""
        router, conn = self._router_with_connector()
        cmd = _make_command(name="aragora", args=[])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        assert result["success"] is True
        conn.respond_to_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_status_subcommand(self):
        router, conn = self._router_with_connector()
        cmd = _make_command(name="aragora", args=["status"])
        event = _make_event(command=cmd)
        with patch(
            "aragora.server.handlers.chat.router.get_configured_platforms",
            return_value=["slack"],
        ):
            result = await router._handle_aragora_command(event)
            assert result["success"] is True
            conn.respond_to_command.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_subcommand_no_topic(self):
        router, conn = self._router_with_connector()
        cmd = _make_command(name="aragora", args=["debate"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        assert result["success"] is True
        text = conn.respond_to_command.call_args[0][1]
        assert "provide a debate topic" in text.lower() or "Usage" in text

    @pytest.mark.asyncio
    async def test_debate_subcommand_with_topic_legacy_starter(self):
        starter = AsyncMock(return_value={"debate_id": "d-123"})
        router = ChatWebhookRouter(debate_starter=starter)
        mock_conn = MagicMock()
        mock_conn.respond_to_command = AsyncMock()
        router._connectors["slack"] = mock_conn
        router._decision_router = None

        cmd = _make_command(name="aragora", args=["debate", "Should", "we", "refactor?"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        assert result["success"] is True
        starter.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_subcommand_legacy_starter_error(self):
        starter = AsyncMock(side_effect=RuntimeError("fail"))
        router = ChatWebhookRouter(debate_starter=starter)
        mock_conn = MagicMock()
        mock_conn.respond_to_command = AsyncMock()
        router._connectors["slack"] = mock_conn
        router._decision_router = None

        cmd = _make_command(name="aragora", args=["debate", "topic"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        assert result["success"] is True
        text = conn_text(mock_conn)
        assert "failed" in text.lower() or "Failed" in text

    @pytest.mark.asyncio
    async def test_debate_subcommand_no_starter(self):
        router = ChatWebhookRouter()
        router._decision_router = None
        mock_conn = MagicMock()
        mock_conn.respond_to_command = AsyncMock()
        router._connectors["slack"] = mock_conn

        cmd = _make_command(name="aragora", args=["debate", "topic"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        text = conn_text(mock_conn)
        assert "not configured" in text.lower() or "not available" in text.lower()

    @pytest.mark.asyncio
    async def test_start_alias(self):
        """'start' should behave like 'debate'."""
        router = ChatWebhookRouter()
        router._decision_router = None
        mock_conn = MagicMock()
        mock_conn.respond_to_command = AsyncMock()
        router._connectors["slack"] = mock_conn

        cmd = _make_command(name="aragora", args=["start", "my topic"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_gauntlet_subcommand_no_topic(self):
        router, conn = self._router_with_connector()
        cmd = _make_command(name="aragora", args=["gauntlet"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        text = conn_text(conn)
        assert "provide a topic" in text.lower() or "Usage" in text

    @pytest.mark.asyncio
    async def test_gauntlet_no_decision_router(self):
        router, conn = self._router_with_connector()
        router._decision_router = None
        cmd = _make_command(name="aragora", args=["gauntlet", "test topic"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        text = conn_text(conn)
        assert "not available" in text.lower()

    @pytest.mark.asyncio
    async def test_workflow_subcommand_no_name(self):
        router, conn = self._router_with_connector()
        cmd = _make_command(name="aragora", args=["workflow"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        text = conn_text(conn)
        assert "provide a workflow" in text.lower() or "Usage" in text

    @pytest.mark.asyncio
    async def test_workflow_no_decision_router(self):
        router, conn = self._router_with_connector()
        router._decision_router = None
        cmd = _make_command(name="aragora", args=["workflow", "security-audit"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        text = conn_text(conn)
        assert "not available" in text.lower()

    @pytest.mark.asyncio
    async def test_unknown_subcommand(self):
        router, conn = self._router_with_connector()
        cmd = _make_command(name="aragora", args=["foobar"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        text = conn_text(conn)
        assert "unknown command" in text.lower() or "Unknown" in text

    @pytest.mark.asyncio
    async def test_respond_to_command_error_caught(self):
        router, conn = self._router_with_connector()
        conn.respond_to_command = AsyncMock(side_effect=ConnectionError("net"))
        cmd = _make_command(name="aragora", args=["help"])
        event = _make_event(command=cmd)
        result = await router._handle_aragora_command(event)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_null_command(self):
        router, _ = self._router_with_connector()
        event = _make_event()
        event.command = None
        result = await router._handle_aragora_command(event)
        assert result.get("success") is False

    @pytest.mark.asyncio
    async def test_null_connector(self):
        router = ChatWebhookRouter()
        cmd = _make_command(name="aragora", args=["help"])
        event = _make_event(command=cmd, platform="unconfigured")
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=None):
            result = await router._handle_aragora_command(event)
            assert result.get("success") is False


def conn_text(conn) -> str:
    """Extract response text sent via respond_to_command."""
    if conn.respond_to_command.call_args:
        return conn.respond_to_command.call_args[0][1]
    return ""


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- _handle_interaction
# ---------------------------------------------------------------------------


class TestHandleInteraction:
    """Test _handle_interaction."""

    @pytest.mark.asyncio
    async def test_none_interaction(self):
        router = ChatWebhookRouter()
        event = _make_event()
        event.interaction = None
        result = await router._handle_interaction(event)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_approval_router_called(self):
        router = ChatWebhookRouter()
        interaction = _make_interaction()
        event = _make_event(interaction=interaction)
        mock_conn = MagicMock()
        router._connectors["slack"] = mock_conn

        mock_approval_router = MagicMock()
        mock_approval_router.handle_interaction = AsyncMock(return_value=True)
        # ApprovalInteractionRouter is imported inside the method body
        with patch(
            "aragora.approvals.interaction_router.ApprovalInteractionRouter",
            return_value=mock_approval_router,
        ):
            router._approval_router = None  # Force init
            result = await router._handle_interaction(event)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_approval_router_not_handled_falls_through(self):
        handler = AsyncMock()
        router = ChatWebhookRouter(event_handler=handler)
        interaction = _make_interaction()
        event = _make_event(interaction=interaction)
        mock_conn = MagicMock()
        router._connectors["slack"] = mock_conn

        mock_approval = MagicMock()
        mock_approval.handle_interaction = AsyncMock(return_value=False)
        router._approval_router = mock_approval

        result = await router._handle_interaction(event)
        handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_approval_router_import_error_no_connector(self):
        """When no connector is available, approval routing is skipped."""
        handler = AsyncMock()
        router = ChatWebhookRouter(event_handler=handler)
        interaction = _make_interaction()
        event = _make_event(interaction=interaction, platform="missing")
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=None):
            result = await router._handle_interaction(event)
            assert result["success"] is True
            handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_interaction_event_handler_error_caught(self):
        handler = AsyncMock(side_effect=ValueError("bad"))
        router = ChatWebhookRouter(event_handler=handler)
        interaction = _make_interaction()
        event = _make_event(interaction=interaction, platform="noconn")
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=None):
            result = await router._handle_interaction(event)
            assert result["success"] is True


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- _handle_message
# ---------------------------------------------------------------------------


class TestHandleMessage:
    """Test _handle_message."""

    @pytest.mark.asyncio
    async def test_none_message(self):
        router = ChatWebhookRouter()
        event = _make_event()
        event.message = None
        result = await router._handle_message(event)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_bot_message_skipped(self):
        handler = AsyncMock()
        router = ChatWebhookRouter(event_handler=handler)
        event = _make_event(message=_make_message(is_bot=True))
        result = await router._handle_message(event)
        assert result["success"] is True
        handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_human_message_passed_to_handler(self):
        handler = AsyncMock()
        router = ChatWebhookRouter(event_handler=handler)
        event = _make_event(message=_make_message(is_bot=False))
        result = await router._handle_message(event)
        handler.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_message_handler_error_caught(self):
        handler = AsyncMock(side_effect=ConnectionError("conn"))
        router = ChatWebhookRouter(event_handler=handler)
        event = _make_event(message=_make_message())
        result = await router._handle_message(event)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_no_handler_returns_success(self):
        router = ChatWebhookRouter()
        event = _make_event(message=_make_message())
        result = await router._handle_message(event)
        assert result["success"] is True


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- _handle_voice
# ---------------------------------------------------------------------------


class TestHandleVoice:
    """Test _handle_voice."""

    @pytest.mark.asyncio
    async def test_none_voice(self):
        router = ChatWebhookRouter()
        event = _make_event()
        event.voice_message = None
        result = await router._handle_voice(event)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_transcription_attempted(self):
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        router._connectors["slack"] = mock_conn

        mock_bridge = MagicMock()
        mock_bridge.transcribe_voice_message = AsyncMock(return_value="Hello transcript")

        event = _make_event(voice_message=_make_voice())
        with patch(
            "aragora.connectors.chat.get_voice_bridge",
            return_value=mock_bridge,
        ):
            result = await router._handle_voice(event)
            assert result["success"] is True
            mock_bridge.transcribe_voice_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_voice_import_error_caught(self):
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        router._connectors["slack"] = mock_conn
        event = _make_event(voice_message=_make_voice())

        with patch(
            "aragora.connectors.chat.get_voice_bridge",
            side_effect=ImportError("no bridge"),
        ):
            result = await router._handle_voice(event)
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_voice_connection_error_caught(self):
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        router._connectors["slack"] = mock_conn

        mock_bridge = MagicMock()
        mock_bridge.transcribe_voice_message = AsyncMock(
            side_effect=ConnectionError("net fail")
        )
        event = _make_event(voice_message=_make_voice())

        with patch(
            "aragora.connectors.chat.get_voice_bridge",
            return_value=mock_bridge,
        ):
            result = await router._handle_voice(event)
            assert result["success"] is True


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- Help text and status
# ---------------------------------------------------------------------------


class TestHelpAndStatus:
    """Test _get_help_text and _get_status."""

    def test_help_text_contains_commands(self):
        router = ChatWebhookRouter()
        text = router._get_help_text()
        assert "/aragora help" in text
        assert "/aragora status" in text
        assert "/aragora debate" in text
        assert "/aragora gauntlet" in text
        assert "/aragora workflow" in text

    @pytest.mark.asyncio
    async def test_status_connected(self):
        router = ChatWebhookRouter()
        with patch(
            "aragora.server.handlers.chat.router.get_configured_platforms",
            return_value=["slack", "discord"],
        ):
            status = await router._get_status()
            assert status["connected"] is True
            assert "slack" in status["platforms"]

    @pytest.mark.asyncio
    async def test_status_disconnected(self):
        router = ChatWebhookRouter()
        with patch(
            "aragora.server.handlers.chat.router.get_configured_platforms",
            return_value=[],
        ):
            status = await router._get_status()
            assert status["connected"] is False


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- _get_input_source
# ---------------------------------------------------------------------------


class TestGetInputSource:
    """Test _get_input_source platform mapping."""

    def test_known_platforms(self):
        router = ChatWebhookRouter()
        with patch(
            "aragora.server.handlers.chat.router.DECISION_ROUTER_AVAILABLE", True
        ), patch("aragora.server.handlers.chat.router.InputSource") as mock_is:
            mock_is.SLACK = "SLACK"
            mock_is.DISCORD = "DISCORD"
            mock_is.TEAMS = "TEAMS"
            mock_is.GOOGLE_CHAT = "GOOGLE_CHAT"
            mock_is.TELEGRAM = "TELEGRAM"
            mock_is.WHATSAPP = "WHATSAPP"
            mock_is.HTTP_API = "HTTP_API"

            assert router._get_input_source("slack") == "SLACK"
            assert router._get_input_source("discord") == "DISCORD"
            assert router._get_input_source("teams") == "TEAMS"
            assert router._get_input_source("google_chat") == "GOOGLE_CHAT"
            assert router._get_input_source("telegram") == "TELEGRAM"
            assert router._get_input_source("whatsapp") == "WHATSAPP"

    def test_unknown_platform_defaults_to_http_api(self):
        router = ChatWebhookRouter()
        with patch(
            "aragora.server.handlers.chat.router.DECISION_ROUTER_AVAILABLE", True
        ), patch("aragora.server.handlers.chat.router.InputSource") as mock_is:
            mock_is.HTTP_API = "HTTP_API"
            mock_is.SLACK = "SLACK"
            mock_is.DISCORD = "DISCORD"
            mock_is.TEAMS = "TEAMS"
            mock_is.GOOGLE_CHAT = "GOOGLE_CHAT"
            mock_is.TELEGRAM = "TELEGRAM"
            mock_is.WHATSAPP = "WHATSAPP"
            assert router._get_input_source("some_other") == "HTTP_API"

    def test_case_insensitive(self):
        router = ChatWebhookRouter()
        with patch(
            "aragora.server.handlers.chat.router.DECISION_ROUTER_AVAILABLE", True
        ), patch("aragora.server.handlers.chat.router.InputSource") as mock_is:
            mock_is.SLACK = "SLACK"
            mock_is.HTTP_API = "HTTP_API"
            mock_is.DISCORD = "DISCORD"
            mock_is.TEAMS = "TEAMS"
            mock_is.GOOGLE_CHAT = "GOOGLE_CHAT"
            mock_is.TELEGRAM = "TELEGRAM"
            mock_is.WHATSAPP = "WHATSAPP"
            assert router._get_input_source("SLACK") == "SLACK"
            assert router._get_input_source("Slack") == "SLACK"

    def test_returns_none_when_not_available(self):
        router = ChatWebhookRouter()
        with patch(
            "aragora.server.handlers.chat.router.DECISION_ROUTER_AVAILABLE", False
        ):
            assert router._get_input_source("slack") is None


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- Constructor
# ---------------------------------------------------------------------------


class TestRouterInit:
    """Test ChatWebhookRouter construction."""

    def test_defaults(self):
        router = ChatWebhookRouter()
        assert router.event_handler is None
        assert router.debate_starter is None
        assert router._connectors == {}

    def test_with_callbacks(self):
        eh = AsyncMock()
        ds = AsyncMock()
        router = ChatWebhookRouter(event_handler=eh, debate_starter=ds)
        assert router.event_handler is eh
        assert router.debate_starter is ds

    def test_explicit_decision_router(self):
        mock_dr = MagicMock()
        router = ChatWebhookRouter(decision_router=mock_dr)
        assert router._decision_router is mock_dr

    def test_decision_router_auto_init_fallback(self):
        """When DECISION_ROUTER_AVAILABLE and no explicit router, tries get_decision_router."""
        with patch(
            "aragora.server.handlers.chat.router.DECISION_ROUTER_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.chat.router.get_decision_router",
            side_effect=RuntimeError("not ready"),
        ):
            router = ChatWebhookRouter()
            assert router._decision_router is None


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- PLATFORM_SIGNATURES
# ---------------------------------------------------------------------------


class TestPlatformSignatures:
    """Test PLATFORM_SIGNATURES class attribute."""

    def test_all_platforms_defined(self):
        expected = {"slack", "discord", "teams", "google_chat", "telegram", "whatsapp"}
        assert set(ChatWebhookRouter.PLATFORM_SIGNATURES.keys()) == expected

    def test_each_has_at_least_one_header(self):
        for platform, headers in ChatWebhookRouter.PLATFORM_SIGNATURES.items():
            assert len(headers) >= 1, f"{platform} has no signature headers"


# ---------------------------------------------------------------------------
# ChatHandler -- can_handle
# ---------------------------------------------------------------------------


class TestChatHandlerCanHandle:
    """Test ChatHandler.can_handle routing."""

    def test_status_route(self):
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/chat/status") is True

    def test_generic_webhook_route(self):
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/chat/webhook") is True

    def test_slack_webhook(self):
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/chat/slack/webhook") is True

    def test_teams_webhook(self):
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/chat/teams/webhook") is True

    def test_discord_webhook(self):
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/chat/discord/webhook") is True

    def test_google_chat_webhook(self):
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/chat/google_chat/webhook") is True

    def test_telegram_webhook(self):
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/chat/telegram/webhook") is True

    def test_whatsapp_webhook(self):
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/chat/whatsapp/webhook") is True

    def test_non_chat_path(self):
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/debates/123") is False

    def test_chat_prefix_matches(self):
        """Any /api/v1/chat/* path should match."""
        handler = ChatHandler(ctx={})
        assert handler.can_handle("/api/v1/chat/custom_platform/webhook") is True


# ---------------------------------------------------------------------------
# ChatHandler -- handle (sync routing)
# ---------------------------------------------------------------------------


class TestChatHandlerHandle:
    """Test ChatHandler.handle method."""

    def test_status_returns_result(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="GET")
        with patch(
            "aragora.server.handlers.chat.router.get_configured_platforms",
            return_value=["slack"],
        ), patch(
            "aragora.server.handlers.chat.router.get_registry",
        ) as mock_reg:
            mock_registry = MagicMock()
            mock_registry.all.return_value = {}
            mock_reg.return_value = mock_registry
            result = handler.handle("/api/v1/chat/status", {}, mock_h)
            assert result is not None
            body = _body(result)
            assert "configured_platforms" in body

    def test_non_post_webhook_returns_405(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="GET")
        result = handler.handle("/api/v1/chat/slack/webhook", {}, mock_h)
        assert _status(result) == 405

    def test_post_returns_none_for_handle_post(self):
        """POST requests should return None from handle() to delegate to handle_post()."""
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST")
        result = handler.handle("/api/v1/chat/slack/webhook", {}, mock_h)
        assert result is None


# ---------------------------------------------------------------------------
# ChatHandler -- handle_post
# ---------------------------------------------------------------------------


class TestChatHandlerHandlePost:
    """Test ChatHandler.handle_post method."""

    @pytest.mark.asyncio
    async def test_slack_platform_from_path(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={"text": "hi"})
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event()

        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await handler.handle_post(
                "/api/v1/chat/slack/webhook", {"text": "hi"}, mock_h
            )
            assert result is not None
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_teams_platform_from_path(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={})
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event(platform="teams")

        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await handler.handle_post(
                "/api/v1/chat/teams/webhook", {}, mock_h
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_discord_platform_from_path(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={})
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event(platform="discord")

        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await handler.handle_post(
                "/api/v1/chat/discord/webhook", {}, mock_h
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_google_chat_platform_from_path(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={})
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event(platform="google_chat")

        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await handler.handle_post(
                "/api/v1/chat/google_chat/webhook", {}, mock_h
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_telegram_platform_from_path(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={})
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event(platform="telegram")

        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await handler.handle_post(
                "/api/v1/chat/telegram/webhook", {}, mock_h
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_whatsapp_platform_from_path(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={})
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event(platform="whatsapp")

        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await handler.handle_post(
                "/api/v1/chat/whatsapp/webhook", {}, mock_h
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_auto_detect_from_headers(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(
            method="POST",
            body={},
            headers={"X-Slack-Signature": "v0=abc"},
        )
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event()

        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await handler.handle_post(
                "/api/v1/chat/webhook", {}, mock_h
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_auto_detect_from_body(self):
        handler = ChatHandler(ctx={})
        body = {"update_id": 1, "message": {"text": "hi"}}
        mock_h = MockHTTPHandler(method="POST", body=body)
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event(platform="telegram")

        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await handler.handle_post(
                "/api/v1/chat/webhook", body, mock_h
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unknown_platform_returns_400(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={"random": "data"})
        result = await handler.handle_post(
            "/api/v1/chat/webhook", {"random": "data"}, mock_h
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={})

        with patch(
            "aragora.server.handlers.chat.router._chat_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = await handler.handle_post(
                "/api/v1/chat/slack/webhook", {}, mock_h
            )
            assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_large_body_rejected(self):
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={})
        # Set content-length to exceed 10MB limit
        mock_h.headers["Content-Length"] = str(11 * 1024 * 1024)

        with patch(
            "aragora.server.handlers.chat.router._chat_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            result = await handler.handle_post(
                "/api/v1/chat/slack/webhook", {}, mock_h
            )
            assert _status(result) == 413

    @pytest.mark.asyncio
    async def test_invalid_content_length(self):
        """Invalid content-length should default to 0, not crash."""
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={})
        mock_h.headers["Content-Length"] = "not_a_number"
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event()

        with patch(
            "aragora.server.handlers.chat.router._chat_limiter"
        ) as mock_limiter, patch(
            "aragora.server.handlers.chat.router.get_connector", return_value=mock_conn
        ):
            mock_limiter.is_allowed.return_value = True
            # Should not crash - content_length defaults to 0
            result = await handler.handle_post(
                "/api/v1/chat/slack/webhook", {}, mock_h
            )
            assert result is not None

    @pytest.mark.asyncio
    async def test_webhook_auth_failure_from_router(self):
        """When webhook verification fails, the inner router returns 401."""
        router = ChatWebhookRouter()
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = False
        with patch("aragora.server.handlers.chat.router.get_connector", return_value=mock_conn):
            result = await router.handle_webhook("slack", {}, b"body")
            # Inner router returns HandlerResult with 401
            assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_no_rfile_uses_body_dict(self):
        """When handler has no rfile, raw_body is built from body dict."""
        handler = ChatHandler(ctx={})
        mock_h = MockHTTPHandler(method="POST", body={"key": "val"})
        delattr(mock_h, "rfile")
        mock_conn = MagicMock()
        mock_conn.verify_webhook.return_value = True
        mock_conn.parse_webhook_event.return_value = _make_event()

        with patch(
            "aragora.server.handlers.chat.router._chat_limiter"
        ) as mock_limiter, patch(
            "aragora.server.handlers.chat.router.get_connector", return_value=mock_conn
        ):
            mock_limiter.is_allowed.return_value = True
            result = await handler.handle_post(
                "/api/v1/chat/slack/webhook", {"key": "val"}, mock_h
            )
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# ChatHandler -- _get_status
# ---------------------------------------------------------------------------


class TestChatHandlerGetStatus:
    """Test ChatHandler._get_status."""

    def test_status_structure(self):
        handler = ChatHandler(ctx={})
        with patch(
            "aragora.server.handlers.chat.router.get_configured_platforms",
            return_value=["slack"],
        ), patch(
            "aragora.server.handlers.chat.router.get_registry",
        ) as mock_reg:
            mock_registry = MagicMock()
            mock_conn = MagicMock()
            mock_conn.platform_display_name = "Slack"
            mock_conn.is_configured = True
            mock_registry.all.return_value = {"slack": mock_conn}
            mock_reg.return_value = mock_registry

            result = handler._get_status()
            body = _body(result)
            assert body["enabled"] is True
            assert "slack" in body["configured_platforms"]
            assert "slack" in body["connectors"]

    def test_status_no_platforms(self):
        handler = ChatHandler(ctx={})
        with patch(
            "aragora.server.handlers.chat.router.get_configured_platforms",
            return_value=[],
        ), patch(
            "aragora.server.handlers.chat.router.get_registry",
        ) as mock_reg:
            mock_registry = MagicMock()
            mock_registry.all.return_value = {}
            mock_reg.return_value = mock_registry

            result = handler._get_status()
            body = _body(result)
            assert body["enabled"] is False


# ---------------------------------------------------------------------------
# Singleton get_webhook_router
# ---------------------------------------------------------------------------


class TestGetWebhookRouter:
    """Test get_webhook_router singleton."""

    def test_creates_singleton(self):
        import aragora.server.handlers.chat.router as mod

        mod._router = None
        try:
            with patch(
                "aragora.server.handlers.chat.router._create_decision_router_debate_starter"
            ) as mock_create:
                mock_create.return_value = AsyncMock()
                r1 = get_webhook_router()
                r2 = get_webhook_router()
                assert r1 is r2
                mock_create.assert_called_once()
        finally:
            mod._router = None

    def test_custom_starter_skips_decision_router(self):
        import aragora.server.handlers.chat.router as mod

        mod._router = None
        try:
            custom = AsyncMock()
            with patch(
                "aragora.server.handlers.chat.router._create_decision_router_debate_starter"
            ) as mock_create:
                r = get_webhook_router(debate_starter=custom)
                mock_create.assert_not_called()
                assert r.debate_starter is custom
        finally:
            mod._router = None

    def test_disable_decision_router(self):
        import aragora.server.handlers.chat.router as mod

        mod._router = None
        try:
            with patch(
                "aragora.server.handlers.chat.router._create_decision_router_debate_starter"
            ) as mock_create:
                r = get_webhook_router(use_decision_router=False)
                mock_create.assert_not_called()
                assert r.debate_starter is None
        finally:
            mod._router = None

    def test_with_event_handler(self):
        import aragora.server.handlers.chat.router as mod

        mod._router = None
        try:
            eh = AsyncMock()
            with patch(
                "aragora.server.handlers.chat.router._create_decision_router_debate_starter"
            ) as mock_create:
                mock_create.return_value = AsyncMock()
                r = get_webhook_router(event_handler=eh)
                assert r.event_handler is eh
        finally:
            mod._router = None


# ---------------------------------------------------------------------------
# _create_decision_router_debate_starter
# ---------------------------------------------------------------------------


class TestCreateDecisionRouterDebateStarter:
    """Test the DecisionRouter debate starter factory."""

    def test_returns_callable(self):
        starter = _create_decision_router_debate_starter()
        assert callable(starter)

    @pytest.mark.asyncio
    async def test_successful_debate(self):
        starter = _create_decision_router_debate_starter()
        with patch("aragora.core.get_decision_router") as mock_get:
            mock_router = MagicMock()
            mock_result = MagicMock()
            mock_result.request_id = "req-1"
            mock_result.success = True
            mock_result.answer = "Answer"
            mock_result.confidence = 0.85
            mock_result.debate_result = MagicMock()
            mock_result.debate_result.debate_id = "d-1"
            mock_router.route = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_router

            result = await starter(
                topic="test topic", platform="slack", channel="C1", user="U1"
            )
            assert result["debate_id"] == "d-1"
            assert result["status"] == "completed"
            assert result["topic"] == "test topic"
            assert result["answer"] == "Answer"

    @pytest.mark.asyncio
    async def test_failed_debate(self):
        starter = _create_decision_router_debate_starter()
        with patch("aragora.core.get_decision_router") as mock_get:
            mock_router = MagicMock()
            mock_result = MagicMock()
            mock_result.request_id = "req-2"
            mock_result.success = False
            mock_result.answer = None
            mock_result.confidence = 0.0
            mock_result.debate_result = None
            mock_router.route = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_router

            result = await starter(
                topic="topic", platform="discord", channel="C2", user="U2"
            )
            assert result["status"] == "failed"
            assert result["debate_id"] == "req-2"

    @pytest.mark.asyncio
    async def test_import_error_returns_minimal(self):
        starter = _create_decision_router_debate_starter()
        with patch("aragora.core.get_decision_router", side_effect=ImportError("nope")):
            result = await starter(
                topic="t", platform="slack", channel="C", user="U"
            )
            assert result["debate_id"] == "pending"
            assert result["topic"] == "t"

    @pytest.mark.asyncio
    async def test_runtime_error_raised(self):
        starter = _create_decision_router_debate_starter()
        with patch(
            "aragora.core.get_decision_router",
            side_effect=RuntimeError("fail"),
        ):
            with pytest.raises(RuntimeError):
                await starter(
                    topic="t", platform="slack", channel="C", user="U"
                )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """Test _handle_task_exception and create_tracked_task."""

    def test_handle_task_exception_cancelled(self):
        task = MagicMock()
        task.cancelled.return_value = True
        _handle_task_exception(task, "test_task")
        # Should not raise

    def test_handle_task_exception_with_exception(self):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("boom")
        _handle_task_exception(task, "test_task")
        # Should log but not raise

    def test_handle_task_exception_no_exception(self):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        _handle_task_exception(task, "test_task")
        # Should not raise

    @pytest.mark.asyncio
    async def test_create_tracked_task(self):
        async def noop():
            return 42

        task = create_tracked_task(noop(), name="test_tracked")
        assert task.get_name() == "test_tracked"
        result = await task
        assert result == 42

    @pytest.mark.asyncio
    async def test_create_tracked_task_exception_logged(self):
        async def fail():
            raise ValueError("oops")

        task = create_tracked_task(fail(), name="fail_task")
        with pytest.raises(ValueError):
            await task


# ---------------------------------------------------------------------------
# ChatWebhookRouter -- _route_decision
# ---------------------------------------------------------------------------


class TestRouteDecision:
    """Test _route_decision method."""

    @pytest.mark.asyncio
    async def test_raises_when_unavailable(self):
        router = ChatWebhookRouter()
        router._decision_router = None
        with patch(
            "aragora.server.handlers.chat.router.DECISION_ROUTER_AVAILABLE", False
        ):
            cmd = _make_command()
            event = _make_event(command=cmd)
            with pytest.raises(RuntimeError, match="not available"):
                await router._route_decision(
                    content="topic",
                    decision_type="DEBATE",
                    event=event,
                    command=cmd,
                )

    @pytest.mark.asyncio
    async def test_routes_successfully(self):
        mock_dr = AsyncMock()
        mock_result = MagicMock()
        mock_result.decision_id = "dec-1"
        mock_dr.route = AsyncMock(return_value=mock_result)
        router = ChatWebhookRouter(decision_router=mock_dr)

        cmd = _make_command()
        event = _make_event(command=cmd)

        with patch(
            "aragora.server.handlers.chat.router.DECISION_ROUTER_AVAILABLE", True
        ), patch(
            "aragora.server.handlers.chat.router.ResponseChannel", MagicMock()
        ), patch(
            "aragora.server.handlers.chat.router.RequestContext", MagicMock()
        ), patch(
            "aragora.server.handlers.chat.router.DecisionRequest", MagicMock(return_value=MagicMock())
        ):
            result = await router._route_decision(
                content="topic",
                decision_type="DEBATE",
                event=event,
                command=cmd,
            )
            mock_dr.route.assert_called_once()


# ---------------------------------------------------------------------------
# ChatHandler -- ROUTES attribute
# ---------------------------------------------------------------------------


class TestChatHandlerRoutes:
    """Test ROUTES class attribute."""

    def test_routes_list(self):
        expected_routes = [
            "/api/v1/chat/webhook",
            "/api/v1/chat/status",
            "/api/v1/chat/slack/webhook",
            "/api/v1/chat/teams/webhook",
            "/api/v1/chat/discord/webhook",
            "/api/v1/chat/google_chat/webhook",
            "/api/v1/chat/telegram/webhook",
            "/api/v1/chat/whatsapp/webhook",
        ]
        assert ChatHandler.ROUTES == expected_routes

    def test_handler_has_router(self):
        handler = ChatHandler(ctx={})
        assert isinstance(handler.router, ChatWebhookRouter)

    def test_handler_accepts_server_context(self):
        handler = ChatHandler(ctx={"key": "val"})
        assert handler.ctx == {"key": "val"}

    def test_handler_default_ctx(self):
        handler = ChatHandler()
        assert handler.ctx == {}
