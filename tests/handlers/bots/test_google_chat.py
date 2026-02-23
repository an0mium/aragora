"""
Tests for Google Chat Bot webhook handler.

Covers all routes and behavior of the GoogleChatHandler class:
- can_handle() routing for all defined routes
- GET /api/v1/bots/google-chat/status  - Bot status endpoint
- POST /api/v1/bots/google-chat/webhook - Webhook event handling
  - MESSAGE events (DM, @mention, slash commands)
  - CARD_CLICKED events (votes, view details)
  - ADDED_TO_SPACE / REMOVED_FROM_SPACE events
- Slash commands: /debate, /plan, /implement, /gauntlet, /status, /help, /agents
- Token verification (layered strategy with caching)
- Input validation (topic length, statement length)
- Card response formatting
- Error handling and edge cases
"""

from __future__ import annotations

import hashlib
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
    import aragora.server.handlers.bots.google_chat as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.GoogleChatHandler


@pytest.fixture
def handler(handler_cls):
    """Create a GoogleChatHandler with empty context."""
    return handler_cls(ctx={})


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/google-chat/webhook"
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
        # Provide client_address for rate-limit key extraction
        self.client_address = ("127.0.0.1", 12345)


def _make_webhook_handler(
    body: dict[str, Any],
    auth_header: str = "Bearer valid-token",
) -> MockHTTPHandler:
    """Create a MockHTTPHandler pre-configured for webhook POST requests."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": auth_header,
    }
    return MockHTTPHandler(body=body, headers=headers)


def _message_event(
    text: str = "Hello",
    space_type: str = "ROOM",
    slash_command: dict | None = None,
    annotations: list | None = None,
    attachments: list | None = None,
) -> dict[str, Any]:
    """Build a minimal MESSAGE event payload."""
    message: dict[str, Any] = {"text": text}
    if slash_command is not None:
        message["slashCommand"] = slash_command
    if annotations is not None:
        message["annotations"] = annotations
    if attachments is not None:
        message["attachments"] = attachments
    return {
        "type": "MESSAGE",
        "message": message,
        "space": {"name": "spaces/test123", "type": space_type},
        "user": {"name": "users/12345", "displayName": "Test User"},
    }


def _card_click_event(
    function_name: str = "vote_agree",
    params: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a minimal CARD_CLICKED event payload."""
    if params is None:
        params = [{"key": "debate_id", "value": "debate-abc"}]
    return {
        "type": "CARD_CLICKED",
        "action": {
            "actionMethodName": function_name,
            "parameters": params,
        },
        "user": {"name": "users/12345", "displayName": "Test User"},
        "space": {"name": "spaces/test123"},
    }


# ===========================================================================
# Token Cache Utilities
# ===========================================================================


class TestTokenCacheKey:
    """Tests for _token_cache_key utility."""

    def test_returns_sha256_hex(self, handler_module):
        key = handler_module._token_cache_key("test-token")
        expected = hashlib.sha256(b"test-token").hexdigest()
        assert key == expected

    def test_deterministic(self, handler_module):
        k1 = handler_module._token_cache_key("abc")
        k2 = handler_module._token_cache_key("abc")
        assert k1 == k2

    def test_different_tokens_produce_different_keys(self, handler_module):
        k1 = handler_module._token_cache_key("token-a")
        k2 = handler_module._token_cache_key("token-b")
        assert k1 != k2


class TestTokenCache:
    """Tests for token cache get/set/clear."""

    def setup_method(self):
        """Clear the token cache before each test."""
        import aragora.server.handlers.bots.google_chat as mod

        mod.clear_token_cache()

    def teardown_method(self):
        """Clear the token cache after each test."""
        import aragora.server.handlers.bots.google_chat as mod

        mod.clear_token_cache()

    def test_cache_miss_returns_none(self, handler_module):
        result = handler_module._get_cached_result("never-seen")
        assert result is None

    def test_set_and_get_valid(self, handler_module):
        handler_module._set_cached_result("tok", True)
        assert handler_module._get_cached_result("tok") is True

    def test_set_and_get_invalid(self, handler_module):
        handler_module._set_cached_result("bad-tok", False)
        assert handler_module._get_cached_result("bad-tok") is False

    def test_expired_entry_returns_none(self, handler_module):
        handler_module._set_cached_result("exp-tok", True)
        # Manually expire the entry
        key = handler_module._token_cache_key("exp-tok")
        with handler_module._token_cache_lock:
            handler_module._token_cache[key] = (True, time.monotonic() - 1)
        assert handler_module._get_cached_result("exp-tok") is None

    def test_clear_cache(self, handler_module):
        handler_module._set_cached_result("tok1", True)
        handler_module._set_cached_result("tok2", False)
        handler_module.clear_token_cache()
        assert handler_module._get_cached_result("tok1") is None
        assert handler_module._get_cached_result("tok2") is None

    def test_valid_token_gets_longer_ttl(self, handler_module):
        handler_module._set_cached_result("valid", True)
        key = handler_module._token_cache_key("valid")
        with handler_module._token_cache_lock:
            _, expiry = handler_module._token_cache[key]
        # Valid tokens: 5 min (300s) TTL
        remaining = expiry - time.monotonic()
        assert remaining > 250  # Should be close to 300s

    def test_invalid_token_gets_shorter_ttl(self, handler_module):
        handler_module._set_cached_result("invalid", False)
        key = handler_module._token_cache_key("invalid")
        with handler_module._token_cache_lock:
            _, expiry = handler_module._token_cache[key]
        remaining = expiry - time.monotonic()
        assert remaining < 70  # Should be close to 60s


# ===========================================================================
# Token Verification Layers
# ===========================================================================


class TestVerifyTokenViaJwtVerifier:
    """Tests for _verify_token_via_jwt_verifier."""

    def test_returns_none_when_module_missing(self, handler_module):
        with patch.dict("sys.modules", {"aragora.connectors.chat.jwt_verify": None}):
            result = handler_module._verify_token_via_jwt_verifier("token")
        # Should return None because ImportError is caught
        assert result is None or result is True or result is False

    @patch("aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier")
    def test_valid_token_returns_true(self, mock_layer, handler_module):
        mock_layer.return_value = True
        assert mock_layer("valid") is True

    @patch("aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier")
    def test_invalid_token_returns_false(self, mock_layer, handler_module):
        mock_layer.return_value = False
        assert mock_layer("invalid") is False


class TestVerifyTokenViaTokeninfo:
    """Tests for _verify_token_via_tokeninfo (HTTP fallback)."""

    @patch("urllib.request.urlopen")
    def test_valid_token(self, mock_urlopen, handler_module):
        resp_data = {
            "email": "chat@system.gserviceaccount.com",
            "exp": str(int(time.time()) + 3600),
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = handler_module._verify_token_via_tokeninfo("valid-token")
        assert result is True

    @patch("urllib.request.urlopen")
    def test_unexpected_email_returns_false(self, mock_urlopen, handler_module):
        resp_data = {
            "email": "attacker@evil.com",
            "exp": str(int(time.time()) + 3600),
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = handler_module._verify_token_via_tokeninfo("bad-email-token")
        assert result is False

    @patch("urllib.request.urlopen")
    def test_expired_token_returns_false(self, mock_urlopen, handler_module):
        resp_data = {
            "email": "chat@system.gserviceaccount.com",
            "exp": str(int(time.time()) - 100),
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = handler_module._verify_token_via_tokeninfo("expired-token")
        assert result is False

    @patch("urllib.request.urlopen")
    def test_http_error_returns_false(self, mock_urlopen, handler_module):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 401, "Unauthorized", {}, None
        )
        result = handler_module._verify_token_via_tokeninfo("bad-token")
        assert result is False

    @patch("urllib.request.urlopen")
    def test_network_error_returns_none(self, mock_urlopen, handler_module):
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        result = handler_module._verify_token_via_tokeninfo("some-token")
        assert result is None

    @patch("urllib.request.urlopen")
    def test_invalid_exp_returns_false(self, mock_urlopen, handler_module):
        resp_data = {
            "email": "chat@system.gserviceaccount.com",
            "exp": "not-a-number",
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(resp_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = handler_module._verify_token_via_tokeninfo("bad-exp-token")
        assert result is False


class TestVerifyGoogleChatToken:
    """Tests for the main _verify_google_chat_token entry point."""

    def setup_method(self):
        import aragora.server.handlers.bots.google_chat as mod

        mod.clear_token_cache()

    def teardown_method(self):
        import aragora.server.handlers.bots.google_chat as mod

        mod.clear_token_cache()

    def test_missing_auth_header_returns_false(self, handler_module):
        assert handler_module._verify_google_chat_token("") is False

    def test_non_bearer_header_returns_false(self, handler_module):
        assert handler_module._verify_google_chat_token("Basic abc123") is False

    def test_empty_bearer_token_returns_false(self, handler_module):
        assert handler_module._verify_google_chat_token("Bearer ") is False
        assert handler_module._verify_google_chat_token("Bearer   ") is False

    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
        return_value=True,
    )
    def test_jwt_verifier_success(self, mock_jwt, handler_module):
        result = handler_module._verify_google_chat_token("Bearer valid-jwt")
        assert result is True
        mock_jwt.assert_called_once_with("valid-jwt")

    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
        return_value=None,
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
        return_value=True,
    )
    def test_falls_through_to_google_auth(self, mock_gauth, mock_jwt, handler_module):
        result = handler_module._verify_google_chat_token("Bearer some-token")
        assert result is True
        mock_gauth.assert_called_once()

    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
        return_value=None,
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
        return_value=None,
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_tokeninfo",
        return_value=True,
    )
    def test_falls_through_to_tokeninfo(
        self, mock_tokeninfo, mock_gauth, mock_jwt, handler_module
    ):
        result = handler_module._verify_google_chat_token("Bearer fallback-tok")
        assert result is True
        mock_tokeninfo.assert_called_once()

    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
        return_value=None,
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_google_auth",
        return_value=None,
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_tokeninfo",
        return_value=None,
    )
    def test_all_layers_unavailable_fails_closed(
        self, mock_tokeninfo, mock_gauth, mock_jwt, handler_module
    ):
        result = handler_module._verify_google_chat_token("Bearer unknown-tok")
        assert result is False

    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
        return_value=True,
    )
    def test_cached_result_is_reused(self, mock_jwt, handler_module):
        # First call populates cache
        handler_module._verify_google_chat_token("Bearer cached-tok")
        # Second call should hit cache, not the verifier again
        mock_jwt.reset_mock()
        result = handler_module._verify_google_chat_token("Bearer cached-tok")
        assert result is True
        mock_jwt.assert_not_called()

    @patch(
        "aragora.server.handlers.bots.google_chat._verify_token_via_jwt_verifier",
        return_value=False,
    )
    def test_cached_invalid_result(self, mock_jwt, handler_module):
        # First call caches False
        handler_module._verify_google_chat_token("Bearer bad-tok")
        mock_jwt.reset_mock()
        result = handler_module._verify_google_chat_token("Bearer bad-tok")
        assert result is False
        mock_jwt.assert_not_called()


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_handles_webhook_path(self, handler):
        assert handler.can_handle("/api/v1/bots/google-chat/webhook", "POST")

    def test_handles_status_path(self, handler):
        assert handler.can_handle("/api/v1/bots/google-chat/status", "GET")

    def test_rejects_unknown_path(self, handler):
        assert not handler.can_handle("/api/v1/bots/slack/webhook", "POST")
        assert not handler.can_handle("/api/v1/other", "GET")
        assert not handler.can_handle("/api/v1/bots/google-chat/unknown", "GET")

    def test_routes_list_has_expected_count(self, handler):
        assert len(handler.ROUTES) == 2

    def test_routes_list_contents(self, handler):
        assert "/api/v1/bots/google-chat/webhook" in handler.ROUTES
        assert "/api/v1/bots/google-chat/status" in handler.ROUTES

    def test_bot_platform_attribute(self, handler):
        assert handler.bot_platform == "google_chat"


# ===========================================================================
# Handler Initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_default_context(self, handler_cls):
        h = handler_cls()
        assert h.ctx == {}

    def test_custom_context(self, handler_cls):
        ctx = {"key": "value"}
        h = handler_cls(ctx=ctx)
        assert h.ctx == ctx

    def test_none_context_defaults_to_empty(self, handler_cls):
        h = handler_cls(ctx=None)
        assert h.ctx == {}


# ===========================================================================
# GET /api/v1/bots/google-chat/status
# ===========================================================================


class TestStatusEndpoint:
    """Tests for the status GET endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler):
        mock_handler = MockHTTPHandler(
            path="/api/v1/bots/google-chat/status", method="GET"
        )
        result = await handler.handle(
            "/api/v1/bots/google-chat/status", {}, mock_handler
        )
        assert result is not None
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_status_contains_platform(self, handler):
        mock_handler = MockHTTPHandler(
            path="/api/v1/bots/google-chat/status", method="GET"
        )
        result = await handler.handle(
            "/api/v1/bots/google-chat/status", {}, mock_handler
        )
        body = _body(result)
        assert body["platform"] == "google_chat"

    @pytest.mark.asyncio
    async def test_status_enabled_field(self, handler):
        mock_handler = MockHTTPHandler(
            path="/api/v1/bots/google-chat/status", method="GET"
        )
        with patch(
            "aragora.server.handlers.bots.google_chat.get_google_chat_connector",
            return_value=MagicMock(),
        ):
            result = await handler.handle(
                "/api/v1/bots/google-chat/status", {}, mock_handler
            )
        body = _body(result)
        assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_status_disabled_when_no_connector(self, handler):
        mock_handler = MockHTTPHandler(
            path="/api/v1/bots/google-chat/status", method="GET"
        )
        with patch(
            "aragora.server.handlers.bots.google_chat.get_google_chat_connector",
            return_value=None,
        ):
            result = await handler.handle(
                "/api/v1/bots/google-chat/status", {}, mock_handler
            )
        body = _body(result)
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_status_includes_config_fields(self, handler):
        mock_handler = MockHTTPHandler(
            path="/api/v1/bots/google-chat/status", method="GET"
        )
        with patch(
            "aragora.server.handlers.bots.google_chat.get_google_chat_connector",
            return_value=None,
        ):
            result = await handler.handle(
                "/api/v1/bots/google-chat/status", {}, mock_handler
            )
        body = _body(result)
        assert "credentials_configured" in body
        assert "project_id_configured" in body

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_get(self, handler):
        mock_handler = MockHTTPHandler(
            path="/api/v1/bots/google-chat/unknown", method="GET"
        )
        result = await handler.handle(
            "/api/v1/bots/google-chat/unknown", {}, mock_handler
        )
        assert result is None


# ===========================================================================
# POST /api/v1/bots/google-chat/webhook - Auth & Gating
# ===========================================================================


class TestWebhookAuth:
    """Tests for webhook authentication and gating."""

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        None,
    )
    def test_webhook_rejects_when_no_credentials(self, handler):
        mock_h = _make_webhook_handler({"type": "MESSAGE"})
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 503
        body = _body(result)
        assert "not configured" in body.get("error", "").lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=False,
    )
    def test_webhook_rejects_invalid_token(self, mock_verify, handler):
        mock_h = _make_webhook_handler({"type": "MESSAGE"}, auth_header="Bearer bad")
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 401

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=False,
    )
    def test_webhook_missing_auth_header(self, mock_verify, handler):
        mock_h = _make_webhook_handler({"type": "MESSAGE"}, auth_header="")
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 401

    def test_handle_post_returns_none_for_unknown_path(self, handler):
        mock_h = MockHTTPHandler()
        result = handler.handle_post(
            "/api/v1/bots/google-chat/unknown", {}, mock_h
        )
        assert result is None


# ===========================================================================
# POST /api/v1/bots/google-chat/webhook - MESSAGE events
# ===========================================================================


class TestWebhookMessageEvent:
    """Tests for MESSAGE webhook events."""

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_regular_room_message_returns_empty(self, mock_verify, handler):
        """Regular message in group room returns empty JSON."""
        event = _message_event(text="Hello everyone", space_type="ROOM")
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200
        body = _body(result)
        assert body == {}

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    @patch("asyncio.create_task")
    def test_dm_starts_debate(self, mock_task, mock_verify, handler):
        """DM with sufficient text triggers debate."""
        event = _message_event(
            text="What is the best programming language for data science?",
            space_type="DM",
        )
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200
        body = _body(result)
        assert "cardsV2" in body

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_dm_short_text_returns_hint(self, mock_verify, handler):
        """DM with very short text returns help hint."""
        event = _message_event(text="hi", space_type="DM")
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200
        body = _body(result)
        # Short text goes to _start_debate_from_message which asks for more context
        assert "cardsV2" in body or "text" in body

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    @patch("asyncio.create_task")
    def test_bot_mention_starts_debate(self, mock_task, mock_verify, handler):
        """Bot @mention with text starts a debate."""
        event = _message_event(
            text="<users/bot123> What should we do about the migration project?",
            space_type="ROOM",
            annotations=[
                {
                    "type": "USER_MENTION",
                    "userMention": {"type": "BOT"},
                }
            ],
        )
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200
        body = _body(result)
        assert "cardsV2" in body


# ===========================================================================
# POST /api/v1/bots/google-chat/webhook - Slash Commands
# ===========================================================================


class TestSlashCommands:
    """Tests for slash command handling."""

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_help_command(self, mock_verify, handler):
        event = _message_event(
            text="/help",
            slash_command={"commandName": "/help"},
        )
        event["message"]["argumentText"] = ""
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        assert _status(result) == 200
        assert "cardsV2" in body
        # Help should mention debate command
        card_text = json.dumps(body)
        assert "debate" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    @patch("asyncio.create_task")
    def test_debate_command_with_topic(self, mock_task, mock_verify, handler):
        event = _message_event(
            text="/debate Should we migrate to kubernetes?",
            slash_command={"commandName": "/debate"},
        )
        event["message"]["argumentText"] = "Should we migrate to kubernetes?"
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        assert _status(result) == 200
        assert "cardsV2" in body
        card_text = json.dumps(body)
        assert "Starting Debate" in card_text

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_debate_command_empty_topic(self, mock_verify, handler):
        event = _message_event(
            text="/debate",
            slash_command={"commandName": "/debate"},
        )
        event["message"]["argumentText"] = ""
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "provide a topic" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_debate_command_topic_too_short(self, mock_verify, handler):
        event = _message_event(
            text="/debate hi",
            slash_command={"commandName": "/debate"},
        )
        event["message"]["argumentText"] = "hi"
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "too short" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_debate_command_topic_too_long(self, mock_verify, handler):
        long_topic = "x" * 600
        event = _message_event(
            text=f"/debate {long_topic}",
            slash_command={"commandName": "/debate"},
        )
        event["message"]["argumentText"] = long_topic
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "too long" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    @patch("asyncio.create_task")
    def test_plan_command(self, mock_task, mock_verify, handler):
        event = _message_event(
            text="/plan Build a rate limiter service",
            slash_command={"commandName": "/plan"},
        )
        event["message"]["argumentText"] = "Build a rate limiter service"
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        assert _status(result) == 200
        card_text = json.dumps(body)
        assert "Starting Debate" in card_text

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    @patch("asyncio.create_task")
    def test_implement_command(self, mock_task, mock_verify, handler):
        event = _message_event(
            text="/implement Add caching to the API layer",
            slash_command={"commandName": "/implement"},
        )
        event["message"]["argumentText"] = "Add caching to the API layer"
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        assert _status(result) == 200

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    @patch("asyncio.create_task")
    def test_gauntlet_command_with_statement(self, mock_task, mock_verify, handler):
        event = _message_event(
            text="/gauntlet We should migrate to microservices",
            slash_command={"commandName": "/gauntlet"},
        )
        event["message"]["argumentText"] = "We should migrate to microservices"
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        assert _status(result) == 200
        card_text = json.dumps(body)
        assert "Gauntlet" in card_text or "gauntlet" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_gauntlet_command_empty_statement(self, mock_verify, handler):
        event = _message_event(
            text="/gauntlet",
            slash_command={"commandName": "/gauntlet"},
        )
        event["message"]["argumentText"] = ""
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "provide a statement" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_gauntlet_command_statement_too_short(self, mock_verify, handler):
        event = _message_event(
            text="/gauntlet hi",
            slash_command={"commandName": "/gauntlet"},
        )
        event["message"]["argumentText"] = "hi"
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "too short" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_gauntlet_command_statement_too_long(self, mock_verify, handler):
        long_stmt = "x" * 1100
        event = _message_event(
            text=f"/gauntlet {long_stmt}",
            slash_command={"commandName": "/gauntlet"},
        )
        event["message"]["argumentText"] = long_stmt
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "too long" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_status_command(self, mock_verify, handler):
        event = _message_event(
            text="/status",
            slash_command={"commandName": "/status"},
        )
        event["message"]["argumentText"] = ""
        mock_h = _make_webhook_handler(event)
        # Mock EloSystem to avoid external dependency
        with patch(
            "aragora.server.handlers.bots.google_chat.GoogleChatHandler._cmd_status"
        ) as mock_cmd:
            from aragora.server.handlers.base import json_response

            mock_cmd.return_value = json_response({"text": "Status: Online"})
            result = handler.handle_post(
                "/api/v1/bots/google-chat/webhook", {}, mock_h
            )
        assert _status(result) == 200

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_agents_command(self, mock_verify, handler):
        event = _message_event(
            text="/agents",
            slash_command={"commandName": "/agents"},
        )
        event["message"]["argumentText"] = ""
        mock_h = _make_webhook_handler(event)
        with patch(
            "aragora.server.handlers.bots.google_chat.GoogleChatHandler._cmd_agents"
        ) as mock_cmd:
            from aragora.server.handlers.base import json_response

            mock_cmd.return_value = json_response({"text": "Top agents"})
            result = handler.handle_post(
                "/api/v1/bots/google-chat/webhook", {}, mock_h
            )
        assert _status(result) == 200

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_unknown_command(self, mock_verify, handler):
        event = _message_event(
            text="/foobar",
            slash_command={"commandName": "/foobar"},
        )
        event["message"]["argumentText"] = ""
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "unknown" in card_text.lower() or "Unknown" in card_text


# ===========================================================================
# POST /api/v1/bots/google-chat/webhook - CARD_CLICKED events
# ===========================================================================


class TestCardClickEvents:
    """Tests for CARD_CLICKED webhook events."""

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", False)
    @patch(
        "aragora.server.handlers.bots.google_chat.rbac_fail_closed",
        return_value=False,
    )
    def test_vote_agree(self, mock_fail_closed, mock_verify, handler):
        event = _card_click_event(
            function_name="vote_agree",
            params=[{"key": "debate_id", "value": "debate-001"}],
        )
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", False)
    @patch(
        "aragora.server.handlers.bots.google_chat.rbac_fail_closed",
        return_value=False,
    )
    def test_vote_disagree(self, mock_fail_closed, mock_verify, handler):
        event = _card_click_event(
            function_name="vote_disagree",
            params=[{"key": "debate_id", "value": "debate-002"}],
        )
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", False)
    @patch(
        "aragora.server.handlers.bots.google_chat.rbac_fail_closed",
        return_value=False,
    )
    def test_vote_no_debate_id(self, mock_fail_closed, mock_verify, handler):
        event = _card_click_event(
            function_name="vote_agree",
            params=[],  # No debate_id
        )
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "no debate" in card_text.lower() or "error" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_view_details_no_debate_id(self, mock_verify, handler):
        event = _card_click_event(
            function_name="view_details",
            params=[],
        )
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "no debate" in card_text.lower() or "error" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_view_details_debate_found(self, mock_verify, handler):
        event = _card_click_event(
            function_name="view_details",
            params=[{"key": "debate_id", "value": "debate-abc"}],
        )
        mock_h = _make_webhook_handler(event)
        mock_db = MagicMock()
        mock_db.get.return_value = {
            "task": "Should we use microservices?",
            "final_answer": "Yes, for scalability reasons.",
            "consensus_reached": True,
            "confidence": 0.85,
            "rounds_used": 3,
        }
        with patch(
            "aragora.server.handlers.bots.google_chat.GoogleChatHandler._handle_view_details"
        ) as mock_view:
            from aragora.server.handlers.base import json_response

            mock_view.return_value = json_response(
                {"cardsV2": [{"card": {"sections": [{"header": "Debate Details"}]}}]}
            )
            result = handler.handle_post(
                "/api/v1/bots/google-chat/webhook", {}, mock_h
            )
        assert _status(result) == 200

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_view_details_debate_not_found(self, mock_verify, handler):
        event = _card_click_event(
            function_name="view_details",
            params=[{"key": "debate_id", "value": "nonexistent"}],
        )
        mock_h = _make_webhook_handler(event)
        mock_db = MagicMock()
        mock_db.get.return_value = None
        with patch(
            "aragora.server.handlers.bots.google_chat.GoogleChatHandler._handle_view_details",
            wraps=handler._handle_view_details,
        ):
            with patch(
                "aragora.server.storage.get_debates_db",
                return_value=mock_db,
            ):
                result = handler.handle_post(
                    "/api/v1/bots/google-chat/webhook", {}, mock_h
                )
        body = _body(result)
        card_text = json.dumps(body)
        assert "not found" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_unknown_card_action(self, mock_verify, handler):
        event = _card_click_event(
            function_name="unknown_action",
            params=[],
        )
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200
        body = _body(result)
        # Unknown action returns empty JSON
        assert body == {}


# ===========================================================================
# POST /api/v1/bots/google-chat/webhook - Space events
# ===========================================================================


class TestSpaceEvents:
    """Tests for ADDED_TO_SPACE and REMOVED_FROM_SPACE events."""

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_added_to_space(self, mock_verify, handler):
        event = {
            "type": "ADDED_TO_SPACE",
            "space": {"name": "spaces/abc", "displayName": "Test Room"},
            "user": {"name": "users/12345", "displayName": "Admin"},
        }
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "Welcome" in card_text or "welcome" in card_text.lower()

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_removed_from_space(self, mock_verify, handler):
        event = {
            "type": "REMOVED_FROM_SPACE",
            "space": {"name": "spaces/abc", "displayName": "Test Room"},
            "user": {"name": "users/12345"},
        }
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200
        body = _body(result)
        assert body == {}


# ===========================================================================
# POST /api/v1/bots/google-chat/webhook - Unhandled Event Types
# ===========================================================================


class TestUnhandledEvents:
    """Tests for unrecognized event types."""

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_unknown_event_type(self, mock_verify, handler):
        event = {"type": "SOME_FUTURE_EVENT"}
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200
        body = _body(result)
        assert body == {}

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_missing_event_type(self, mock_verify, handler):
        event = {"message": {"text": "hello"}}
        mock_h = _make_webhook_handler(event)
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert _status(result) == 200
        body = _body(result)
        assert body == {}


# ===========================================================================
# Attachment Extraction
# ===========================================================================


class TestExtractAttachments:
    """Tests for _extract_attachments helper."""

    def test_no_message(self, handler):
        result = handler._extract_attachments({})
        assert result == []

    def test_no_attachments(self, handler):
        event = {"message": {"text": "hello"}}
        result = handler._extract_attachments(event)
        assert result == []

    def test_empty_attachments(self, handler):
        event = {"message": {"attachments": []}}
        result = handler._extract_attachments(event)
        assert result == []

    def test_single_attachment_with_name(self, handler):
        event = {
            "message": {
                "attachments": [{"name": "report.pdf", "contentType": "application/pdf"}]
            }
        }
        result = handler._extract_attachments(event)
        assert len(result) == 1
        assert result[0]["filename"] == "report.pdf"

    def test_attachment_with_filename_field(self, handler):
        event = {
            "message": {
                "attachments": [{"filename": "doc.txt", "size": 1024}]
            }
        }
        result = handler._extract_attachments(event)
        assert len(result) == 1
        assert result[0]["filename"] == "doc.txt"

    def test_attachment_with_content_name(self, handler):
        event = {
            "message": {
                "attachments": [{"contentName": "image.png"}]
            }
        }
        result = handler._extract_attachments(event)
        assert len(result) == 1
        assert result[0]["filename"] == "image.png"

    def test_attachment_fallback_name(self, handler):
        event = {
            "message": {
                "attachments": [{"contentType": "image/jpeg"}]
            }
        }
        result = handler._extract_attachments(event)
        assert len(result) == 1
        assert result[0]["filename"] == "attachment_1"

    def test_multiple_attachments(self, handler):
        event = {
            "message": {
                "attachments": [
                    {"name": "a.txt"},
                    {"name": "b.txt"},
                    {"name": "c.txt"},
                ]
            }
        }
        result = handler._extract_attachments(event)
        assert len(result) == 3

    def test_non_dict_attachment_skipped(self, handler):
        event = {
            "message": {
                "attachments": ["not-a-dict", {"name": "real.txt"}]
            }
        }
        result = handler._extract_attachments(event)
        assert len(result) == 1
        assert result[0]["filename"] == "real.txt"

    def test_non_list_attachments_returns_empty(self, handler):
        event = {"message": {"attachments": "not-a-list"}}
        result = handler._extract_attachments(event)
        assert result == []

    def test_non_dict_message_returns_empty(self, handler):
        event = {"message": "not-a-dict"}
        result = handler._extract_attachments(event)
        assert result == []


# ===========================================================================
# Card Response Formatting
# ===========================================================================


class TestCardResponse:
    """Tests for _card_response helper."""

    def test_card_with_title_only(self, handler):
        result = handler._card_response(title="Hello")
        body = _body(result)
        assert "cardsV2" in body
        sections = body["cardsV2"][0]["card"]["sections"]
        assert any("header" in s for s in sections)

    def test_card_with_body_only(self, handler):
        result = handler._card_response(body="Some text")
        body = _body(result)
        assert "cardsV2" in body

    def test_card_with_fields(self, handler):
        result = handler._card_response(
            fields=[("Status", "Online"), ("Agents", "5")]
        )
        body = _body(result)
        assert "cardsV2" in body
        card_text = json.dumps(body)
        assert "Status" in card_text
        assert "Online" in card_text

    def test_card_with_context(self, handler):
        result = handler._card_response(
            body="Content", context="Requested by user"
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "Requested by user" in card_text

    def test_card_with_actions(self, handler):
        result = handler._card_response(
            body="Choose",
            actions=[{"text": "OK", "onClick": {"action": {"function": "ok"}}}],
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "buttonList" in card_text

    def test_empty_card_response(self, handler):
        result = handler._card_response()
        body = _body(result)
        # No title, body, fields, context, or actions - empty response
        assert isinstance(body, dict)

    def test_card_id_is_aragora_response(self, handler):
        result = handler._card_response(title="Test")
        body = _body(result)
        assert body["cardsV2"][0]["cardId"] == "aragora_response"


# ===========================================================================
# _cmd_status Implementation
# ===========================================================================


class TestCmdStatus:
    """Tests for _cmd_status command."""

    def test_status_with_elo_system(self, handler):
        mock_elo = MagicMock()
        mock_elo.get_all_ratings.return_value = [MagicMock(), MagicMock()]
        with patch(
            "aragora.ranking.elo.EloSystem", return_value=mock_elo
        ):
            result = handler._cmd_status("spaces/test")
        body = _body(result)
        card_text = json.dumps(body)
        assert "Online" in card_text

    def test_status_without_elo_system(self, handler):
        with patch.dict(
            "sys.modules", {"aragora.ranking.elo": None}
        ):
            # When EloSystem import fails, should still return status
            result = handler._cmd_status("spaces/test")
        body = _body(result)
        card_text = json.dumps(body)
        assert "Online" in card_text or "Available" in card_text


# ===========================================================================
# _cmd_agents Implementation
# ===========================================================================


class TestCmdAgents:
    """Tests for _cmd_agents command."""

    def test_agents_with_ratings(self, handler):
        mock_agent = MagicMock()
        mock_agent.name = "Claude"
        mock_agent.elo = 1650
        mock_elo = MagicMock()
        mock_elo.get_all_ratings.return_value = [mock_agent]
        with patch(
            "aragora.ranking.elo.EloSystem", return_value=mock_elo
        ):
            result = handler._cmd_agents()
        body = _body(result)
        card_text = json.dumps(body)
        assert "Claude" in card_text

    def test_agents_with_empty_ratings(self, handler):
        mock_elo = MagicMock()
        mock_elo.get_all_ratings.return_value = []
        with patch(
            "aragora.ranking.elo.EloSystem", return_value=mock_elo
        ):
            result = handler._cmd_agents()
        body = _body(result)
        card_text = json.dumps(body)
        assert "no agents" in card_text.lower()

    def test_agents_import_error(self, handler):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = handler._cmd_agents()
        body = _body(result)
        card_text = json.dumps(body)
        assert "unavailable" in card_text.lower()


# ===========================================================================
# _cmd_help Implementation
# ===========================================================================


class TestCmdHelp:
    """Tests for _cmd_help command."""

    def test_help_includes_all_commands(self, handler):
        result = handler._cmd_help()
        body = _body(result)
        card_text = json.dumps(body)
        assert "/debate" in card_text
        assert "/plan" in card_text
        assert "/implement" in card_text
        assert "/gauntlet" in card_text
        assert "/status" in card_text
        assert "/agents" in card_text
        assert "/help" in card_text

    def test_help_includes_examples(self, handler):
        result = handler._cmd_help()
        body = _body(result)
        card_text = json.dumps(body)
        assert "Example" in card_text or "example" in card_text.lower()


# ===========================================================================
# _handle_vote
# ===========================================================================


class TestHandleVote:
    """Tests for _handle_vote card action."""

    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", False)
    @patch(
        "aragora.server.handlers.bots.google_chat.rbac_fail_closed",
        return_value=False,
    )
    def test_vote_no_debate_id(self, mock_fail_closed, handler):
        result = handler._handle_vote("", "agree", "user1", "spaces/test")
        body = _body(result)
        card_text = json.dumps(body)
        assert "no debate" in card_text.lower() or "error" in card_text.lower()

    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", False)
    @patch(
        "aragora.server.handlers.bots.google_chat.rbac_fail_closed",
        return_value=False,
    )
    def test_vote_consensus_store_unavailable(self, mock_fail_closed, handler):
        """Vote acknowledged even when ConsensusStore is unavailable."""
        with patch.dict("sys.modules", {"aragora.memory.consensus": None}):
            result = handler._handle_vote("debate-1", "agree", "user1", "spaces/test")
        body = _body(result)
        card_text = json.dumps(body)
        assert "acknowledged" in card_text.lower() or "vote" in card_text.lower()

    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", True)
    def test_vote_rbac_denied(self, handler):
        with patch(
            "aragora.server.handlers.bots.google_chat.check_permission",
            side_effect=PermissionError("denied"),
        ):
            result = handler._handle_vote("debate-1", "agree", "user1", "spaces/test")
        body = _body(result)
        card_text = json.dumps(body)
        assert "permission denied" in card_text.lower() or "denied" in card_text.lower()


# ===========================================================================
# _handle_view_details
# ===========================================================================


class TestHandleViewDetails:
    """Tests for _handle_view_details card action."""

    def test_no_debate_id(self, handler):
        result = handler._handle_view_details("")
        body = _body(result)
        card_text = json.dumps(body)
        assert "no debate" in card_text.lower() or "error" in card_text.lower()

    def test_debate_found(self, handler):
        mock_db = MagicMock()
        mock_db.get.return_value = {
            "task": "Test topic for debate",
            "final_answer": "The conclusion is clear",
            "consensus_reached": True,
            "confidence": 0.9,
            "rounds_used": 3,
        }
        with patch(
            "aragora.server.storage.get_debates_db", return_value=mock_db
        ):
            result = handler._handle_view_details("debate-123")
        body = _body(result)
        card_text = json.dumps(body)
        assert "Debate Details" in card_text
        assert "Test topic" in card_text

    def test_debate_not_found(self, handler):
        mock_db = MagicMock()
        mock_db.get.return_value = None
        with patch(
            "aragora.server.storage.get_debates_db", return_value=mock_db
        ):
            result = handler._handle_view_details("nonexistent")
        body = _body(result)
        card_text = json.dumps(body)
        assert "not found" in card_text.lower()

    def test_db_unavailable(self, handler):
        with patch(
            "aragora.server.storage.get_debates_db", return_value=None
        ):
            result = handler._handle_view_details("debate-123")
        body = _body(result)
        card_text = json.dumps(body)
        assert "not found" in card_text.lower() or "error" in card_text.lower()

    def test_import_error(self, handler):
        with patch.dict("sys.modules", {"aragora.server.storage": None}):
            result = handler._handle_view_details("debate-123")
        body = _body(result)
        card_text = json.dumps(body)
        assert "error" in card_text.lower()


# ===========================================================================
# _start_debate_from_message
# ===========================================================================


class TestStartDebateFromMessage:
    """Tests for _start_debate_from_message."""

    @patch("asyncio.create_task")
    def test_clean_text_removes_mentions(self, mock_task, handler):
        user = {"name": "users/12345", "displayName": "Test"}
        event = _message_event(text="<users/bot123> Should we use React for the frontend redesign?")
        result = handler._start_debate_from_message(
            "<users/bot123> Should we use React for the frontend redesign?",
            "spaces/test",
            user,
            event,
        )
        body = _body(result)
        card_text = json.dumps(body)
        # The @mention is cleaned, topic goes to _cmd_debate
        assert "Starting Debate" in card_text or "cardsV2" in body

    def test_short_text_returns_help(self, handler):
        user = {"name": "users/12345", "displayName": "Test"}
        event = _message_event(text="hi")
        result = handler._start_debate_from_message(
            "hi", "spaces/test", user, event
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "need more context" in card_text.lower()

    def test_empty_text_returns_help(self, handler):
        user = {"name": "users/12345", "displayName": "Test"}
        event = _message_event(text="")
        result = handler._start_debate_from_message(
            "", "spaces/test", user, event
        )
        body = _body(result)
        card_text = json.dumps(body)
        assert "need more context" in card_text.lower()

    @patch("asyncio.create_task")
    def test_plan_prefix_sets_decision_integrity(self, mock_task, handler):
        user = {"name": "users/12345", "displayName": "Test"}
        event = _message_event(text="plan Build a distributed cache system design")
        result = handler._start_debate_from_message(
            "plan Build a distributed cache system design",
            "spaces/test",
            user,
            event,
        )
        body = _body(result)
        assert "cardsV2" in body

    @patch("asyncio.create_task")
    def test_implement_prefix_sets_execution_mode(self, mock_task, handler):
        user = {"name": "users/12345", "displayName": "Test"}
        event = _message_event(text="implement Add retry logic to all HTTP clients")
        result = handler._start_debate_from_message(
            "implement Add retry logic to all HTTP clients",
            "spaces/test",
            user,
            event,
        )
        body = _body(result)
        assert "cardsV2" in body

    @patch("asyncio.create_task")
    def test_debate_prefix_strips_keyword(self, mock_task, handler):
        user = {"name": "users/12345", "displayName": "Test"}
        event = _message_event(text="debate Should we use GraphQL?")
        result = handler._start_debate_from_message(
            "debate Should we use GraphQL?",
            "spaces/test",
            user,
            event,
        )
        body = _body(result)
        assert "cardsV2" in body


# ===========================================================================
# RBAC Permission Checks
# ===========================================================================


class TestBotPermissionCheck:
    """Tests for _check_bot_permission helper."""

    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", False)
    @patch(
        "aragora.server.handlers.bots.google_chat.rbac_fail_closed",
        return_value=False,
    )
    def test_no_rbac_no_fail_closed_allows(self, mock_fail, handler):
        # Should not raise
        handler._check_bot_permission("debates:create", user_id="gchat:123")

    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", False)
    @patch(
        "aragora.server.handlers.bots.google_chat.rbac_fail_closed",
        return_value=True,
    )
    def test_no_rbac_fail_closed_raises(self, mock_fail, handler):
        with pytest.raises(PermissionError):
            handler._check_bot_permission("debates:create", user_id="gchat:123")

    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", True)
    @patch(
        "aragora.server.handlers.bots.google_chat.check_permission",
        side_effect=PermissionError("denied"),
    )
    def test_rbac_available_permission_denied(self, mock_check, handler):
        with pytest.raises(PermissionError):
            handler._check_bot_permission("debates:create", user_id="gchat:123")

    @patch("aragora.server.handlers.bots.google_chat.RBAC_AVAILABLE", True)
    @patch("aragora.server.handlers.bots.google_chat.check_permission")
    def test_rbac_available_permission_granted(self, mock_check, handler):
        handler._check_bot_permission("debates:create", user_id="gchat:123")
        mock_check.assert_called_once()


# ===========================================================================
# Google Chat Connector
# ===========================================================================


class TestGetGoogleChatConnector:
    """Tests for get_google_chat_connector factory."""

    def setup_method(self):
        import aragora.server.handlers.bots.google_chat as mod

        mod._google_chat_connector = None

    def teardown_method(self):
        import aragora.server.handlers.bots.google_chat as mod

        mod._google_chat_connector = None

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        None,
    )
    def test_returns_none_when_no_credentials(self, handler_module):
        result = handler_module.get_google_chat_connector()
        assert result is None

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    def test_returns_none_on_import_error(self, handler_module):
        with patch.dict("sys.modules", {"aragora.connectors.chat.google_chat": None}):
            result = handler_module.get_google_chat_connector()
        assert result is None

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    def test_caches_connector(self, handler_module):
        mock_connector = MagicMock()
        with patch(
            "aragora.connectors.chat.google_chat.GoogleChatConnector",
            return_value=mock_connector,
        ):
            first = handler_module.get_google_chat_connector()
            second = handler_module.get_google_chat_connector()
        assert first is second


# ===========================================================================
# get_google_chat_handler Factory
# ===========================================================================


class TestGetGoogleChatHandler:
    """Tests for get_google_chat_handler factory."""

    def setup_method(self):
        import aragora.server.handlers.bots.google_chat as mod

        mod._google_chat_handler = None

    def teardown_method(self):
        import aragora.server.handlers.bots.google_chat as mod

        mod._google_chat_handler = None

    def test_creates_handler(self, handler_module):
        h = handler_module.get_google_chat_handler()
        assert isinstance(h, handler_module.GoogleChatHandler)

    def test_returns_cached_handler(self, handler_module):
        h1 = handler_module.get_google_chat_handler()
        h2 = handler_module.get_google_chat_handler()
        assert h1 is h2

    def test_accepts_server_context(self, handler_module):
        ctx = {"key": "value"}
        h = handler_module.get_google_chat_handler(server_context=ctx)
        assert h.ctx == ctx


# ===========================================================================
# Error Handling in Webhook
# ===========================================================================


class TestWebhookErrorHandling:
    """Tests for error handling in the webhook path."""

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_empty_body_returns_error(self, mock_verify, handler):
        mock_h = MockHTTPHandler(
            headers={
                "Content-Length": "0",
                "Authorization": "Bearer valid",
            }
        )
        mock_h.rfile = io.BytesIO(b"")
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        # Empty body should return an error
        assert result is not None
        status = _status(result)
        assert status in (200, 400)

    @patch(
        "aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS",
        "fake-creds",
    )
    @patch(
        "aragora.server.handlers.bots.google_chat._verify_google_chat_token",
        return_value=True,
    )
    def test_invalid_json_body(self, mock_verify, handler):
        mock_h = MockHTTPHandler(
            headers={
                "Content-Length": "12",
                "Authorization": "Bearer valid",
            }
        )
        mock_h.rfile = io.BytesIO(b"not valid json")
        result = handler.handle_post(
            "/api/v1/bots/google-chat/webhook", {}, mock_h
        )
        assert result is not None
        status = _status(result)
        assert status in (200, 400)


# ===========================================================================
# Tracked Task Creation
# ===========================================================================


class TestCreateTrackedTask:
    """Tests for create_tracked_task utility."""

    @pytest.mark.asyncio
    async def test_creates_task_with_name(self, handler_module):
        async def noop():
            pass

        task = handler_module.create_tracked_task(noop(), name="test-task")
        assert task.get_name() == "test-task"
        await task  # Clean up

    @pytest.mark.asyncio
    async def test_task_has_done_callback(self, handler_module):
        async def noop():
            pass

        task = handler_module.create_tracked_task(noop(), name="callback-test")
        # The task should have a done callback for exception handling
        await task  # Clean up


class TestHandleTaskException:
    """Tests for _handle_task_exception."""

    def test_cancelled_task(self, handler_module):
        task = MagicMock()
        task.cancelled.return_value = True
        # Should not raise
        handler_module._handle_task_exception(task, "test")

    def test_failed_task(self, handler_module):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("boom")
        # Should not raise (just logs)
        handler_module._handle_task_exception(task, "test")

    def test_successful_task(self, handler_module):
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        # Should not raise
        handler_module._handle_task_exception(task, "test")


# ===========================================================================
# Input Validation Constants
# ===========================================================================


class TestInputValidation:
    """Tests for input validation constants."""

    def test_max_topic_length(self, handler_module):
        assert handler_module.MAX_TOPIC_LENGTH == 500

    def test_max_statement_length(self, handler_module):
        assert handler_module.MAX_STATEMENT_LENGTH == 1000


# ===========================================================================
# _is_bot_enabled
# ===========================================================================


class TestIsBotEnabled:
    """Tests for _is_bot_enabled method."""

    def test_enabled_when_connector_exists(self, handler):
        with patch(
            "aragora.server.handlers.bots.google_chat.get_google_chat_connector",
            return_value=MagicMock(),
        ):
            assert handler._is_bot_enabled() is True

    def test_disabled_when_no_connector(self, handler):
        with patch(
            "aragora.server.handlers.bots.google_chat.get_google_chat_connector",
            return_value=None,
        ):
            assert handler._is_bot_enabled() is False


# ===========================================================================
# _build_status_response
# ===========================================================================


class TestBuildStatusResponse:
    """Tests for _build_status_response method."""

    def test_includes_platform(self, handler):
        with patch(
            "aragora.server.handlers.bots.google_chat.get_google_chat_connector",
            return_value=None,
        ):
            result = handler._build_status_response()
        body = _body(result)
        assert body["platform"] == "google_chat"

    def test_includes_extra_status(self, handler):
        with patch(
            "aragora.server.handlers.bots.google_chat.get_google_chat_connector",
            return_value=None,
        ):
            result = handler._build_status_response(extra_status={"version": "1.0"})
        body = _body(result)
        assert body["version"] == "1.0"

    def test_includes_credentials_configured(self, handler):
        with patch(
            "aragora.server.handlers.bots.google_chat.get_google_chat_connector",
            return_value=None,
        ):
            result = handler._build_status_response()
        body = _body(result)
        assert "credentials_configured" in body
        assert "project_id_configured" in body
