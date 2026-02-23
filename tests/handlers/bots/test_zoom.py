"""
Tests for Zoom Bot endpoint handler.

Covers all routes and behavior of the ZoomHandler class:
- can_handle() routing for all defined routes
- GET  /api/v1/bots/zoom/status  - Bot status (RBAC protected)
- POST /api/v1/bots/zoom/events  - Webhook event handling
  - endpoint.url_validation (HMAC-SHA256 challenge/response)
  - bot_notification (chat messages, RBAC on debates:create)
  - meeting.ended
  - bot_installed
  - Unknown event types
- Webhook signature verification (x-zm-signature / x-zm-request-timestamp)
- _ensure_bot lazy initialization (import, config, error paths)
- _is_bot_enabled / _get_platform_config_status
- _check_bot_permission RBAC integration
- Error handling: JSONDecodeError, ValueError, KeyError, TypeError, RuntimeError, OSError
- Handler initialization and module exports
- Security tests: path traversal, injection, oversized body
"""

from __future__ import annotations

import io
import json
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
    import aragora.server.handlers.bots.zoom as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.ZoomHandler


@pytest.fixture
def handler(handler_cls):
    """Create a ZoomHandler with empty context."""
    return handler_cls(ctx={})


# ---------------------------------------------------------------------------
# Mock HTTP Handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/zoom/events"
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


def _make_event_handler(
    body: dict[str, Any],
    signature: str = "v0=abcd1234",
    timestamp: str = "1700000000",
) -> MockHTTPHandler:
    """Create a MockHTTPHandler pre-configured for event POST requests."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "x-zm-signature": signature,
        "x-zm-request-timestamp": timestamp,
    }
    return MockHTTPHandler(body=body, headers=headers)


# ---------------------------------------------------------------------------
# Event body builders
# ---------------------------------------------------------------------------


def _url_validation_event(plain_token: str = "test-token-123") -> dict[str, Any]:
    """Build a Zoom URL validation event."""
    return {
        "event": "endpoint.url_validation",
        "payload": {"plainToken": plain_token},
    }


def _bot_notification_event(
    user_jid: str = "user-jid-123",
    message: str = "Hello bot",
    to_jid: str = "bot-jid-456",
) -> dict[str, Any]:
    """Build a Zoom bot_notification event."""
    return {
        "event": "bot_notification",
        "payload": {
            "userJid": user_jid,
            "toJid": to_jid,
            "cmd": message,
            "accountId": "account-001",
        },
    }


def _meeting_ended_event(meeting_id: str = "meeting-001") -> dict[str, Any]:
    """Build a Zoom meeting.ended event."""
    return {
        "event": "meeting.ended",
        "payload": {
            "object": {
                "id": meeting_id,
                "topic": "Test Meeting",
            }
        },
    }


def _bot_installed_event() -> dict[str, Any]:
    """Build a Zoom bot_installed event."""
    return {
        "event": "bot_installed",
        "payload": {
            "accountId": "account-001",
        },
    }


# ===========================================================================
# can_handle()
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_events_route(self, handler):
        assert handler.can_handle("/api/v1/bots/zoom/events", "POST") is True

    def test_status_route(self, handler):
        assert handler.can_handle("/api/v1/bots/zoom/status", "GET") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/webhook", "POST") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/", "GET") is False

    def test_partial_match_no_trailing(self, handler):
        assert handler.can_handle("/api/v1/bots/zoom/eventsXYZ", "POST") is False

    def test_different_version(self, handler):
        assert handler.can_handle("/api/v2/bots/zoom/events", "POST") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("", "GET") is False

    def test_routes_list_complete(self, handler):
        """ROUTES list contains exactly the expected paths."""
        assert set(handler.ROUTES) == {
            "/api/v1/bots/zoom/events",
            "/api/v1/bots/zoom/status",
        }


# ===========================================================================
# GET /api/v1/bots/zoom/status
# ===========================================================================


class TestStatusEndpoint:
    """Tests for the status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_status_body_has_platform(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        body = _body(result)
        assert body["platform"] == "zoom"

    @pytest.mark.asyncio
    async def test_status_body_has_enabled_field(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "client-123"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "secret-456"):
            result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        body = _body(result)
        assert "enabled" in body
        assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_status_has_client_id_configured(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "client-123"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None):
            result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        body = _body(result)
        assert body["client_id_configured"] is True

    @pytest.mark.asyncio
    async def test_status_has_client_secret_configured(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        with patch.object(handler_module, "ZOOM_CLIENT_SECRET", "secret-456"), \
             patch.object(handler_module, "ZOOM_CLIENT_ID", None):
            result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        body = _body(result)
        assert body["client_secret_configured"] is True

    @pytest.mark.asyncio
    async def test_status_has_bot_jid_configured(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        with patch.object(handler_module, "ZOOM_BOT_JID", "bot-jid-789"):
            result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        body = _body(result)
        assert body["bot_jid_configured"] is True

    @pytest.mark.asyncio
    async def test_status_has_secret_token_configured(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", "token-abc"):
            result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        body = _body(result)
        assert body["secret_token_configured"] is True

    @pytest.mark.asyncio
    async def test_status_disabled_when_not_configured(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        with patch.object(handler_module, "ZOOM_CLIENT_ID", None), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None):
            result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        body = _body(result)
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_status_disabled_partial_config(self, handler, handler_module):
        """Enabled requires both client_id AND client_secret."""
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "client-123"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None):
            result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        body = _body(result)
        assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_non_status_get(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/events", method="GET")
        result = await handler.handle("/api/v1/bots/zoom/events", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_path(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/unknown", method="GET")
        result = await handler.handle("/api/v1/bots/zoom/unknown", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_status_all_unconfigured(self, handler, handler_module):
        """All config fields False when nothing is set."""
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        with patch.object(handler_module, "ZOOM_CLIENT_ID", None), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None), \
             patch.object(handler_module, "ZOOM_BOT_JID", None), \
             patch.object(handler_module, "ZOOM_SECRET_TOKEN", None):
            result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        body = _body(result)
        assert body["client_id_configured"] is False
        assert body["client_secret_configured"] is False
        assert body["bot_jid_configured"] is False
        assert body["secret_token_configured"] is False


# ===========================================================================
# POST /api/v1/bots/zoom/events - URL Validation
# ===========================================================================


class TestURLValidation:
    """Tests for endpoint.url_validation event handling."""

    @pytest.mark.asyncio
    async def test_url_validation_returns_200(self, handler, handler_module):
        event = _url_validation_event()
        http_handler = _make_event_handler(event, signature="", timestamp="")
        # No signature needed for URL validation
        http_handler.headers.pop("x-zm-signature", None)
        http_handler.headers.pop("x-zm-request-timestamp", None)
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", "mysecret"):
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_url_validation_returns_plain_token(self, handler, handler_module):
        event = _url_validation_event(plain_token="my-token-value")
        http_handler = _make_event_handler(event)
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", "mysecret"):
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        body = _body(result)
        assert body["plainToken"] == "my-token-value"

    @pytest.mark.asyncio
    async def test_url_validation_returns_encrypted_token(self, handler, handler_module):
        """encryptedToken should be HMAC-SHA256 of plainToken with secret."""
        import hashlib
        import hmac

        secret = "test-secret-key"
        plain_token = "test-plain-token"
        expected = hmac.new(
            secret.encode(), plain_token.encode(), hashlib.sha256
        ).hexdigest()

        event = _url_validation_event(plain_token=plain_token)
        http_handler = _make_event_handler(event)
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", secret):
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        body = _body(result)
        assert body["encryptedToken"] == expected

    @pytest.mark.asyncio
    async def test_url_validation_no_secret_token(self, handler, handler_module):
        """Without ZOOM_SECRET_TOKEN, URL validation returns 503."""
        event = _url_validation_event()
        http_handler = _make_event_handler(event)
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", None):
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_url_validation_empty_plain_token(self, handler, handler_module):
        """URL validation with empty plainToken still processes."""
        event = _url_validation_event(plain_token="")
        http_handler = _make_event_handler(event)
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", "secret"):
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        body = _body(result)
        assert body["plainToken"] == ""
        assert "encryptedToken" in body


# ===========================================================================
# POST /api/v1/bots/zoom/events - Signature Verification
# ===========================================================================


class TestSignatureVerification:
    """Tests for webhook signature verification on non-validation events."""

    @pytest.mark.asyncio
    async def test_missing_signature_returns_401(self, handler, handler_module):
        """Events without signature header return 401."""
        event = _bot_notification_event()
        http_handler = _make_event_handler(event, signature="", timestamp="123")
        # Remove signature header
        http_handler.headers.pop("x-zm-signature", None)
        http_handler.headers["x-zm-signature"] = ""
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = False
            handler._bot = None
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 401
        body = _body(result)
        assert "Missing signature" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_invalid_signature_returns_401(self, handler, handler_module):
        """Invalid signature verification fails with 401."""
        event = _bot_notification_event()
        http_handler = _make_event_handler(event, signature="bad-sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = False

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 401
        body = _body(result)
        assert "Invalid signature" in body.get("error", "")

    @pytest.mark.asyncio
    async def test_valid_signature_passes(self, handler, handler_module):
        """Valid signature allows event processing."""
        event = _bot_notification_event()
        http_handler = _make_event_handler(event, signature="valid-sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"ok": True})

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch("aragora.server.handlers.bots.zoom.rbac_fail_closed", return_value=False):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_signature_present_but_no_bot_returns_503(self, handler, handler_module):
        """Signature present but bot not configured returns 503."""
        event = _bot_notification_event()
        http_handler = _make_event_handler(event, signature="some-sig", timestamp="123")

        with patch.object(handler_module, "ZOOM_CLIENT_ID", None), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None):
            handler._bot_initialized = False
            handler._bot = None
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 503
        body = _body(result)
        assert "not configured" in body.get("error", "").lower()


# ===========================================================================
# POST /api/v1/bots/zoom/events - bot_notification (RBAC)
# ===========================================================================


class TestBotNotification:
    """Tests for bot_notification event handling."""

    @pytest.mark.asyncio
    async def test_bot_notification_success(self, handler, handler_module):
        """Successful bot_notification processes event."""
        event = _bot_notification_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"text": "Hello!"})

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch("aragora.server.handlers.bots.zoom.rbac_fail_closed", return_value=False):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["text"] == "Hello!"

    @pytest.mark.asyncio
    async def test_bot_notification_rbac_denied(self, handler, handler_module):
        """RBAC denial for bot_notification returns permission_denied."""
        event = _bot_notification_event(user_jid="user-abc")
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch.object(handler, "_check_bot_permission", side_effect=PermissionError("denied")):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        body = _body(result)
        assert body.get("error") == "permission_denied"

    @pytest.mark.asyncio
    async def test_bot_notification_empty_user_jid(self, handler, handler_module):
        """bot_notification with empty userJid still processes."""
        event = _bot_notification_event(user_jid="")
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"ok": True})

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch("aragora.server.handlers.bots.zoom.rbac_fail_closed", return_value=False):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 200


# ===========================================================================
# POST /api/v1/bots/zoom/events - Other event types
# ===========================================================================


class TestOtherEventTypes:
    """Tests for meeting.ended, bot_installed, and unknown event types."""

    @pytest.mark.asyncio
    async def test_meeting_ended_success(self, handler, handler_module):
        event = _meeting_ended_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"acknowledged": True})

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 200
        body = _body(result)
        assert body["acknowledged"] is True

    @pytest.mark.asyncio
    async def test_bot_installed_success(self, handler, handler_module):
        event = _bot_installed_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"installed": True})

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unknown_event_type(self, handler, handler_module):
        """Unknown event type is still processed by bot.handle_event."""
        event = {"event": "unknown.event", "payload": {}}
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"ok": True})

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_no_event_field(self, handler, handler_module):
        """Event without 'event' field - empty string event type."""
        event = {"payload": {}}
        http_handler = _make_event_handler(event, signature="", timestamp="")
        # No signature => 401 for non-url_validation
        result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_event_without_bot_returns_503(self, handler, handler_module):
        """Non-url_validation event with signature and verified but no bot returns 503."""
        event = _meeting_ended_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        # Bot is None but signature passes because bot not available for verification
        # Actually, if signature is present and bot is None, returns 503 from sig check
        with patch.object(handler_module, "ZOOM_CLIENT_ID", None), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None):
            handler._bot_initialized = False
            handler._bot = None
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 503


# ===========================================================================
# _ensure_bot
# ===========================================================================


class TestEnsureBot:
    """Tests for lazy bot initialization."""

    def test_returns_none_when_not_configured(self, handler, handler_module):
        """Returns None when ZOOM_CLIENT_ID or ZOOM_CLIENT_SECRET is missing."""
        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", None), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None):
            result = handler._ensure_bot()
        assert result is None
        assert handler._bot_initialized is True

    def test_returns_none_when_partial_config(self, handler, handler_module):
        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None):
            result = handler._ensure_bot()
        assert result is None

    def test_returns_bot_on_success(self, handler, handler_module):
        handler._bot_initialized = False
        mock_bot = MagicMock()
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch("aragora.bots.zoom_bot.create_zoom_bot", return_value=mock_bot):
            result = handler._ensure_bot()
        assert result is mock_bot
        assert handler._bot is mock_bot

    def test_caches_after_first_call(self, handler, handler_module):
        """Second call returns cached bot without re-initializing."""
        mock_bot = MagicMock()
        handler._bot_initialized = True
        handler._bot = mock_bot
        result = handler._ensure_bot()
        assert result is mock_bot

    def test_import_error_returns_none(self, handler, handler_module):
        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch.dict("sys.modules", {"aragora.bots.zoom_bot": None}):
            result = handler._ensure_bot()
        assert result is None
        assert handler._bot_initialized is True

    def test_value_error_returns_none(self, handler, handler_module):
        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch(
                 "aragora.bots.zoom_bot.create_zoom_bot",
                 side_effect=ValueError("bad config"),
             ):
            result = handler._ensure_bot()
        assert result is None

    def test_key_error_returns_none(self, handler, handler_module):
        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch(
                 "aragora.bots.zoom_bot.create_zoom_bot",
                 side_effect=KeyError("missing key"),
             ):
            result = handler._ensure_bot()
        assert result is None

    def test_type_error_returns_none(self, handler, handler_module):
        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch(
                 "aragora.bots.zoom_bot.create_zoom_bot",
                 side_effect=TypeError("wrong type"),
             ):
            result = handler._ensure_bot()
        assert result is None

    def test_runtime_error_returns_none(self, handler, handler_module):
        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch(
                 "aragora.bots.zoom_bot.create_zoom_bot",
                 side_effect=RuntimeError("failed"),
             ):
            result = handler._ensure_bot()
        assert result is None

    def test_os_error_returns_none(self, handler, handler_module):
        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch(
                 "aragora.bots.zoom_bot.create_zoom_bot",
                 side_effect=OSError("network error"),
             ):
            result = handler._ensure_bot()
        assert result is None

    def test_attribute_error_returns_none(self, handler, handler_module):
        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch(
                 "aragora.bots.zoom_bot.create_zoom_bot",
                 side_effect=AttributeError("missing attr"),
             ):
            result = handler._ensure_bot()
        assert result is None


# ===========================================================================
# _is_bot_enabled
# ===========================================================================


class TestIsBotEnabled:
    """Tests for _is_bot_enabled."""

    def test_enabled_when_both_set(self, handler, handler_module):
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            assert handler._is_bot_enabled() is True

    def test_disabled_when_client_id_not_set(self, handler, handler_module):
        with patch.object(handler_module, "ZOOM_CLIENT_ID", None), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            assert handler._is_bot_enabled() is False

    def test_disabled_when_client_secret_not_set(self, handler, handler_module):
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None):
            assert handler._is_bot_enabled() is False

    def test_disabled_when_neither_set(self, handler, handler_module):
        with patch.object(handler_module, "ZOOM_CLIENT_ID", None), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None):
            assert handler._is_bot_enabled() is False

    def test_disabled_when_empty_string(self, handler, handler_module):
        with patch.object(handler_module, "ZOOM_CLIENT_ID", ""), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", ""):
            assert handler._is_bot_enabled() is False


# ===========================================================================
# _get_platform_config_status
# ===========================================================================


class TestPlatformConfigStatus:
    """Tests for _get_platform_config_status."""

    def test_all_configured(self, handler, handler_module):
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch.object(handler_module, "ZOOM_BOT_JID", "jid"), \
             patch.object(handler_module, "ZOOM_SECRET_TOKEN", "token"):
            status = handler._get_platform_config_status()
        assert status["client_id_configured"] is True
        assert status["client_secret_configured"] is True
        assert status["bot_jid_configured"] is True
        assert status["secret_token_configured"] is True

    def test_none_configured(self, handler, handler_module):
        with patch.object(handler_module, "ZOOM_CLIENT_ID", None), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None), \
             patch.object(handler_module, "ZOOM_BOT_JID", None), \
             patch.object(handler_module, "ZOOM_SECRET_TOKEN", None):
            status = handler._get_platform_config_status()
        assert status["client_id_configured"] is False
        assert status["client_secret_configured"] is False
        assert status["bot_jid_configured"] is False
        assert status["secret_token_configured"] is False

    def test_partial_configured(self, handler, handler_module):
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", None), \
             patch.object(handler_module, "ZOOM_BOT_JID", "jid"), \
             patch.object(handler_module, "ZOOM_SECRET_TOKEN", None):
            status = handler._get_platform_config_status()
        assert status["client_id_configured"] is True
        assert status["client_secret_configured"] is False
        assert status["bot_jid_configured"] is True
        assert status["secret_token_configured"] is False


# ===========================================================================
# RBAC Permission Checks
# ===========================================================================


class TestRBACPermissions:
    """Tests for _check_bot_permission RBAC integration."""

    def test_rbac_not_available_non_production(self, handler, handler_module):
        """When RBAC is unavailable and not production, should pass."""
        with patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch("aragora.server.handlers.bots.zoom.rbac_fail_closed", return_value=False):
            handler._check_bot_permission("debates:create", user_id="zoom:123")

    def test_rbac_not_available_production(self, handler, handler_module):
        """When RBAC is unavailable in production, should raise."""
        with patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch("aragora.server.handlers.bots.zoom.rbac_fail_closed", return_value=True):
            with pytest.raises(PermissionError):
                handler._check_bot_permission("debates:create", user_id="zoom:123")

    def test_rbac_available_permission_granted(self, handler, handler_module):
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission") as mock_check:
            mock_check.return_value = None
            handler._check_bot_permission("debates:create", user_id="zoom:123")

    def test_rbac_available_permission_denied(self, handler, handler_module):
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission") as mock_check:
            mock_check.side_effect = PermissionError("Denied")
            with pytest.raises(PermissionError):
                handler._check_bot_permission("debates:create", user_id="zoom:123")

    def test_rbac_with_auth_context_in_context(self, handler, handler_module):
        """When auth_context is provided in context dict, it should be used."""
        mock_auth_ctx = MagicMock()
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission") as mock_check:
            handler._check_bot_permission(
                "debates:create",
                context={"auth_context": mock_auth_ctx},
            )
            mock_check.assert_called_once_with(mock_auth_ctx, "debates:create")

    def test_rbac_no_user_id_no_context(self, handler, handler_module):
        """When no user_id and no auth_context, check_permission not called."""
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission") as mock_check:
            handler._check_bot_permission("debates:create")
            mock_check.assert_not_called()

    def test_rbac_constructs_auth_context_with_user_id(self, handler, handler_module):
        """When user_id provided and no auth_context, builds AuthorizationContext."""
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission") as mock_check, \
             patch.object(handler_module, "AuthorizationContext") as mock_ctx_cls:
            mock_ctx_instance = MagicMock()
            mock_ctx_cls.return_value = mock_ctx_instance
            handler._check_bot_permission("bots.read", user_id="zoom:user-abc")
            mock_ctx_cls.assert_called_once_with(
                user_id="zoom:user-abc",
                roles={"bot_user"},
            )
            mock_check.assert_called_once_with(mock_ctx_instance, "bots.read")


# ===========================================================================
# Error Handling
# ===========================================================================


class TestEventErrorHandling:
    """Tests for exception handling in _handle_events."""

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, handler):
        """Invalid JSON body returns 400."""
        http_handler = MockHTTPHandler()
        http_handler.rfile = io.BytesIO(b"not json at all")
        http_handler.headers["Content-Length"] = "15"
        http_handler.headers["x-zm-signature"] = "abc"
        http_handler.headers["x-zm-request-timestamp"] = "123"
        result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_empty_body(self, handler):
        """Empty body returns 400."""
        http_handler = MockHTTPHandler()
        http_handler.rfile = io.BytesIO(b"")
        http_handler.headers["Content-Length"] = "0"
        http_handler.headers["x-zm-signature"] = "abc"
        http_handler.headers["x-zm-request-timestamp"] = "123"
        result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_value_error_in_processing(self, handler, handler_module):
        """ValueError during event processing returns error via _handle_webhook_exception."""
        event = _meeting_ended_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(side_effect=ValueError("bad value"))

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        # ValueError is caught by the try/except in _handle_events
        assert _status(result) in (400, 500)

    @pytest.mark.asyncio
    async def test_key_error_in_processing(self, handler, handler_module):
        """KeyError during event processing returns error."""
        event = _meeting_ended_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(side_effect=KeyError("missing"))

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) in (400, 500)

    @pytest.mark.asyncio
    async def test_type_error_in_processing(self, handler, handler_module):
        """TypeError during event processing returns error."""
        event = _meeting_ended_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(side_effect=TypeError("wrong type"))

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) in (400, 500)

    @pytest.mark.asyncio
    async def test_runtime_error_in_processing(self, handler, handler_module):
        """RuntimeError during event processing returns error."""
        event = _meeting_ended_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(side_effect=RuntimeError("boom"))

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) in (400, 500)

    @pytest.mark.asyncio
    async def test_os_error_in_processing(self, handler, handler_module):
        """OSError during event processing returns error."""
        event = _meeting_ended_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(side_effect=OSError("io error"))

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) in (400, 500, 503)

    @pytest.mark.asyncio
    async def test_json_decode_error_in_processing(self, handler, handler_module):
        """json.JSONDecodeError during processing returns error."""
        event = _meeting_ended_event()
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(
            side_effect=json.JSONDecodeError("bad", "doc", 0)
        )

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) in (400, 500)


# ===========================================================================
# handle_post routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for handle_post routing."""

    @pytest.mark.asyncio
    async def test_handle_post_returns_none_for_unknown_path(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/unknown")
        result = await handler.handle_post("/api/v1/bots/unknown", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_post_routes_events(self, handler, handler_module):
        """handle_post routes to _handle_events for events path."""
        event = _url_validation_event()
        http_handler = _make_event_handler(event)
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", "secret"):
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert result is not None
        assert _status(result) == 200


# ===========================================================================
# Handler Initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for ZoomHandler initialization."""

    def test_default_ctx(self, handler_cls):
        h = handler_cls()
        assert h.ctx == {}

    def test_custom_ctx(self, handler_cls):
        ctx = {"storage": MagicMock()}
        h = handler_cls(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx(self, handler_cls):
        h = handler_cls(ctx=None)
        assert h.ctx == {}

    def test_bot_platform(self, handler):
        assert handler.bot_platform == "zoom"

    def test_routes_defined(self, handler):
        assert "/api/v1/bots/zoom/events" in handler.ROUTES
        assert "/api/v1/bots/zoom/status" in handler.ROUTES

    def test_bot_initially_none(self, handler):
        assert handler._bot is None

    def test_bot_not_initialized(self, handler):
        assert handler._bot_initialized is False


# ===========================================================================
# Module-level __all__
# ===========================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports(self, handler_module):
        assert "ZoomHandler" in handler_module.__all__

    def test_zoom_handler_in_module(self, handler_module):
        assert hasattr(handler_module, "ZoomHandler")


# ===========================================================================
# Security Tests
# ===========================================================================


class TestSecurity:
    """Security-oriented tests."""

    @pytest.mark.asyncio
    async def test_path_traversal_in_event(self, handler):
        """Path traversal attempt in event path does not match route."""
        assert handler.can_handle("/api/v1/bots/zoom/../slack/events", "POST") is False

    @pytest.mark.asyncio
    async def test_null_byte_in_path(self, handler):
        """Null byte in path does not match route."""
        assert handler.can_handle("/api/v1/bots/zoom/events\x00", "POST") is False

    @pytest.mark.asyncio
    async def test_oversized_body_handled(self, handler):
        """Oversized Content-Length triggers ValueError, caught by error handler."""
        http_handler = MockHTTPHandler()
        http_handler.headers["Content-Length"] = str(20 * 1024 * 1024)  # 20MB
        http_handler.headers["x-zm-signature"] = "sig"
        http_handler.headers["x-zm-request-timestamp"] = "123"
        result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        # handle_errors decorator catches ValueError
        assert result is not None
        assert _status(result) in (400, 500)

    @pytest.mark.asyncio
    async def test_negative_content_length(self, handler):
        """Negative Content-Length treated as 0 (empty body)."""
        http_handler = MockHTTPHandler()
        http_handler.headers["Content-Length"] = "-1"
        http_handler.headers["x-zm-signature"] = "sig"
        http_handler.headers["x-zm-request-timestamp"] = "123"
        result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 400  # empty body error

    @pytest.mark.asyncio
    async def test_non_numeric_content_length(self, handler):
        """Non-numeric Content-Length treated as 0 (empty body)."""
        http_handler = MockHTTPHandler()
        http_handler.headers["Content-Length"] = "abc"
        http_handler.headers["x-zm-signature"] = "sig"
        http_handler.headers["x-zm-request-timestamp"] = "123"
        result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 400  # empty body error

    @pytest.mark.asyncio
    async def test_script_injection_in_event_field(self, handler, handler_module):
        """Script injection in event field doesn't cause execution."""
        event = {
            "event": "<script>alert('xss')</script>",
            "payload": {},
        }
        http_handler = _make_event_handler(event, signature="", timestamp="")
        # No signature => 401
        result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_deeply_nested_json(self, handler, handler_module):
        """Deeply nested JSON doesn't cause stack overflow."""
        event = {"event": "endpoint.url_validation", "payload": {"plainToken": "tok"}}
        # Add deep nesting in payload
        nested = event
        for _ in range(50):
            nested["nested"] = {"level": True}
            nested = nested["nested"]
        http_handler = _make_event_handler(event)
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", "secret"):
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 200


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_handle_with_empty_query_params(self, handler):
        """handle() works with empty query params."""
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        result = await handler.handle("/api/v1/bots/zoom/status", {}, http_handler)
        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_with_query_params(self, handler):
        """handle() works with populated query params."""
        http_handler = MockHTTPHandler(path="/api/v1/bots/zoom/status", method="GET")
        result = await handler.handle(
            "/api/v1/bots/zoom/status", {"format": "json"}, http_handler
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_url_validation_with_special_chars_in_token(self, handler, handler_module):
        """URL validation handles special characters in plainToken."""
        event = _url_validation_event(plain_token="tok/en+with=special&chars")
        http_handler = _make_event_handler(event)
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", "secret"):
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        body = _body(result)
        assert body["plainToken"] == "tok/en+with=special&chars"

    @pytest.mark.asyncio
    async def test_url_validation_with_unicode_token(self, handler, handler_module):
        """URL validation handles unicode in plainToken."""
        event = _url_validation_event(plain_token="token-\u00e9\u00e8\u00ea")
        http_handler = _make_event_handler(event)
        with patch.object(handler_module, "ZOOM_SECRET_TOKEN", "secret"):
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        body = _body(result)
        assert body["plainToken"] == "token-\u00e9\u00e8\u00ea"

    @pytest.mark.asyncio
    async def test_bot_notification_with_large_message(self, handler, handler_module):
        """bot_notification with large message payload processes correctly."""
        event = _bot_notification_event(message="x" * 10000)
        http_handler = _make_event_handler(event, signature="sig", timestamp="123")
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"ok": True})

        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch("aragora.server.handlers.bots.zoom.rbac_fail_closed", return_value=False):
            handler._bot_initialized = True
            handler._bot = mock_bot
            result = await handler.handle_post("/api/v1/bots/zoom/events", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_multiple_events_reuse_handler(self, handler, handler_module):
        """Handler processes multiple events sequentially."""
        mock_bot = MagicMock()
        mock_bot.verify_webhook.return_value = True
        mock_bot.handle_event = AsyncMock(return_value={"ok": True})

        handler._bot_initialized = True
        handler._bot = mock_bot

        for event_fn in [_meeting_ended_event, _bot_installed_event]:
            event = event_fn()
            http_handler = _make_event_handler(event, signature="sig", timestamp="123")
            with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
                 patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"):
                result = await handler.handle_post(
                    "/api/v1/bots/zoom/events", {}, http_handler
                )
            assert _status(result) == 200

    def test_ensure_bot_called_only_once(self, handler, handler_module):
        """_ensure_bot only initializes once even on multiple calls."""
        call_count = 0
        original_bot = MagicMock()

        def mock_create():
            nonlocal call_count
            call_count += 1
            return original_bot

        handler._bot_initialized = False
        with patch.object(handler_module, "ZOOM_CLIENT_ID", "cid"), \
             patch.object(handler_module, "ZOOM_CLIENT_SECRET", "csec"), \
             patch("aragora.bots.zoom_bot.create_zoom_bot", side_effect=mock_create):
            handler._ensure_bot()
            handler._ensure_bot()
            handler._ensure_bot()
        assert call_count == 1
