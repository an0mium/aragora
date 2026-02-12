"""Tests for aragora.server.handlers.chat.router - Chat Router Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# Mock Classes
# ===========================================================================


@dataclass
class MockUser:
    """Mock authenticated user."""

    user_id: str = "user-123"
    id: str = "user-123"
    org_id: str = "org-123"
    email: str = "test@example.com"
    name: str = "Test User"


class MockHandler:
    """Mock HTTP request handler."""

    def __init__(
        self,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        path: str = "/",
        method: str = "GET",
    ):
        self._body = body
        self.headers = headers or {"Content-Length": str(len(body))}
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)
        self.client_address = ("127.0.0.1", 12345)

    @classmethod
    def with_json_body(cls, data: dict[str, Any], **kwargs) -> "MockHandler":
        body = json.dumps(data).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        return cls(body=body, headers=headers, **kwargs)

    def get_argument(self, name: str, default: str = None) -> str | None:
        return default


class MockWebhookEvent:
    """Mock webhook event."""

    def __init__(
        self,
        platform: str = "slack",
        event_type: str = "message",
        is_verification: bool = False,
        challenge: str | None = None,
    ):
        self.platform = platform
        self.event_type = event_type
        self.is_verification = is_verification
        self.challenge = challenge
        self.command = None
        self.interaction = None
        self.message = None
        self.voice_message = None
        self.raw_payload = {}


class MockConnector:
    """Mock chat platform connector."""

    def __init__(self, platform: str = "slack"):
        self.platform = platform
        self.platform_display_name = platform.title()
        self.is_configured = True

    def verify_webhook(self, headers: dict[str, str], body: bytes) -> bool:
        return True

    def parse_webhook_event(self, headers: dict[str, str], body: bytes) -> MockWebhookEvent:
        return MockWebhookEvent(platform=self.platform)

    def format_blocks(self, title: str, body: str = None, fields: list = None):
        return [{"type": "section", "text": {"type": "mrkdwn", "text": title}}]

    async def respond_to_command(self, command, text: str, blocks=None, ephemeral=False):
        pass


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_router_state():
    """Reset router state before each test."""
    try:
        from aragora.server.handlers.chat import router

        router._router = None
    except Exception:
        pass
    yield


@pytest.fixture
def mock_connector():
    return MockConnector()


@pytest.fixture
def mock_registry(mock_connector):
    registry = MagicMock()
    registry.all.return_value = {"slack": mock_connector}
    return registry


# ===========================================================================
# ChatWebhookRouter Tests
# ===========================================================================


class TestChatWebhookRouter:
    """Tests for ChatWebhookRouter class."""

    def test_router_initialization(self):
        """Test router can be initialized."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        assert router is not None

    def test_router_with_event_handler(self):
        """Test router with custom event handler."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        async def custom_handler(event):
            pass

        router = ChatWebhookRouter(event_handler=custom_handler)
        assert router.event_handler == custom_handler

    def test_router_with_debate_starter(self):
        """Test router with custom debate starter."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        async def custom_starter(**kwargs):
            return {"debate_id": "test-123"}

        router = ChatWebhookRouter(debate_starter=custom_starter)
        assert router.debate_starter == custom_starter

    @pytest.mark.asyncio
    async def test_route_decision_passes_attachments(self):
        """Ensure chat attachments are forwarded into DecisionRequest."""
        from types import SimpleNamespace

        from aragora.core.decision import DecisionType
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        captured: dict[str, Any] = {}

        class DummyDecisionRouter:
            async def route(self, request):
                captured["request"] = request
                return SimpleNamespace(success=True)

        router = ChatWebhookRouter(decision_router=DummyDecisionRouter())

        event = SimpleNamespace(
            platform="slack",
            message=SimpleNamespace(attachments=[{"filename": "notes.txt", "content": "hello"}]),
        )
        command = SimpleNamespace(
            channel="C123",
            thread_ts=None,
            user=SimpleNamespace(id="U123", name="alice"),
        )

        await router._route_decision("hello", DecisionType.DEBATE, event, command)

        request = captured.get("request")
        assert request is not None
        assert request.attachments == [{"filename": "notes.txt", "content": "hello"}]


# ===========================================================================
# Platform Detection Tests
# ===========================================================================


class TestPlatformDetection:
    """Tests for platform detection from headers."""

    def test_detect_slack_from_signature(self):
        """Test detecting Slack from signature header."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        headers = {"X-Slack-Signature": "v0=abc123"}
        platform = router.detect_platform(headers)
        assert platform == "slack"

    def test_detect_discord_from_signature(self):
        """Test detecting Discord from signature header."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        headers = {"X-Signature-Ed25519": "abc123"}
        platform = router.detect_platform(headers)
        assert platform == "discord"

    def test_detect_telegram_from_secret_token(self):
        """Test detecting Telegram from secret token header."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        headers = {"X-Telegram-Bot-Api-Secret-Token": "secret123"}
        platform = router.detect_platform(headers)
        assert platform == "telegram"

    def test_detect_whatsapp_from_hub_signature(self):
        """Test detecting WhatsApp from hub signature header."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        headers = {"X-Hub-Signature-256": "sha256=abc123"}
        platform = router.detect_platform(headers)
        assert platform == "whatsapp"

    def test_detect_teams_from_bearer(self):
        """Test detecting Teams from Bearer auth."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        headers = {"Authorization": "Bearer token123"}
        platform = router.detect_platform(headers)
        assert platform == "teams"

    def test_detect_unknown_platform(self):
        """Test return None for unknown platform."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        headers = {"Content-Type": "application/json"}
        platform = router.detect_platform(headers)
        assert platform is None


# ===========================================================================
# Body-Based Platform Detection Tests
# ===========================================================================


class TestBodyPlatformDetection:
    """Tests for platform detection from body."""

    def test_detect_telegram_from_body(self):
        """Test detecting Telegram from body structure."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        body = json.dumps({"update_id": 123, "message": {"text": "hello"}}).encode()
        platform = router.detect_platform_from_body({}, body)
        assert platform == "telegram"

    def test_detect_whatsapp_from_body(self):
        """Test detecting WhatsApp from body structure."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        body = json.dumps({"object": "whatsapp_business_account", "entry": []}).encode()
        platform = router.detect_platform_from_body({}, body)
        assert platform == "whatsapp"

    def test_detect_teams_from_body(self):
        """Test detecting Teams from body structure."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        body = json.dumps({"channelId": "msteams", "type": "message"}).encode()
        platform = router.detect_platform_from_body({}, body)
        assert platform == "teams"

    def test_detect_google_chat_from_body(self):
        """Test detecting Google Chat from body structure."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        body = json.dumps({"type": "MESSAGE", "space": {}, "message": {}}).encode()
        platform = router.detect_platform_from_body({}, body)
        assert platform == "google_chat"

    def test_invalid_json_body(self):
        """Test handling invalid JSON body."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        body = b"not json"
        platform = router.detect_platform_from_body({}, body)
        assert platform is None


# ===========================================================================
# Verification Handling Tests
# ===========================================================================


class TestVerificationHandling:
    """Tests for webhook verification handling."""

    def test_slack_verification_challenge(self):
        """Test Slack URL verification challenge."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        event = MockWebhookEvent(platform="slack", is_verification=True)
        event.challenge = "test-challenge"
        result = router._handle_verification("slack", event)
        assert result == {"challenge": "test-challenge"}

    def test_discord_verification_pong(self):
        """Test Discord PING/PONG verification."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        event = MockWebhookEvent(platform="discord", is_verification=True)
        result = router._handle_verification("discord", event)
        assert result == {"type": 1}

    def test_whatsapp_verification_hub_challenge(self):
        """Test WhatsApp hub challenge verification."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        event = MockWebhookEvent(platform="whatsapp", is_verification=True)
        event.challenge = "hub-challenge-token"
        result = router._handle_verification("whatsapp", event)
        assert result == {"hub.challenge": "hub-challenge-token"}


# ===========================================================================
# Get Webhook Router Tests
# ===========================================================================


class TestGetWebhookRouter:
    """Tests for get_webhook_router singleton."""

    def test_get_webhook_router_returns_router(self):
        """Test get_webhook_router returns a router."""
        from aragora.server.handlers.chat.router import get_webhook_router

        router = get_webhook_router()
        assert router is not None

    def test_get_webhook_router_returns_same_instance(self):
        """Test get_webhook_router returns singleton."""
        from aragora.server.handlers.chat.router import get_webhook_router

        router1 = get_webhook_router()
        router2 = get_webhook_router()
        assert router1 is router2


# ===========================================================================
# Help Text Tests
# ===========================================================================


class TestHelpText:
    """Tests for help text generation."""

    def test_get_help_text(self):
        """Test help text is generated."""
        from aragora.server.handlers.chat.router import ChatWebhookRouter

        router = ChatWebhookRouter()
        help_text = router._get_help_text()
        assert "Available Commands" in help_text
        assert "/aragora help" in help_text
        assert "/aragora debate" in help_text
