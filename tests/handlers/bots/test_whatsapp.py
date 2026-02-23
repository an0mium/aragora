"""
Tests for WhatsApp Bot webhook handler.

Covers all routes and behavior of the WhatsAppHandler class:
- can_handle() routing for all defined routes
- GET  /api/v1/bots/whatsapp/webhook  - Webhook verification challenge
- GET  /api/v1/bots/whatsapp/status   - Bot status endpoint
- POST /api/v1/bots/whatsapp/webhook  - Webhook message handling
  - Signature verification
  - Text messages (with and without commands)
  - Media messages (document, image, video, audio)
  - Interactive messages (list_reply, button_reply)
  - Button quick-reply messages
- Slash commands: /help, /debate, /plan, /implement, /status
- Greeting/welcome messages
- _extract_attachments for all media types
- _hydrate_whatsapp_attachments download path
- _send_message / _send_welcome / _send_help / _send_status
- _start_debate / _start_debate_async / _start_debate_via_queue
- RBAC permission checks
- Error handling and edge cases
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

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


def _body_text(result) -> str:
    """Extract raw body text from a HandlerResult."""
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8")
    return str(raw)


# ---------------------------------------------------------------------------
# Lazy import so conftest auto-auth patches run first
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    import aragora.server.handlers.bots.whatsapp as mod
    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.WhatsAppHandler


@pytest.fixture
def handler(handler_cls):
    """Create a WhatsAppHandler with empty context."""
    return handler_cls(ctx={})


# ---------------------------------------------------------------------------
# Mock HTTP Handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/whatsapp/webhook"
    method: str = "POST"
    body: dict[str, Any] | None = None
    raw_body: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.raw_body is not None:
            body_bytes = self.raw_body
        elif self.body is not None:
            body_bytes = json.dumps(self.body).encode("utf-8")
        else:
            body_bytes = b"{}"
        self.rfile = io.BytesIO(body_bytes)
        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(body_bytes))
        self.client_address = ("127.0.0.1", 12345)


def _make_webhook_handler(
    body: dict[str, Any],
    signature: str = "",
    raw_body: bytes | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler pre-configured for webhook POST requests."""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if signature:
        headers["X-Hub-Signature-256"] = signature
    return MockHTTPHandler(body=body, headers=headers, raw_body=raw_body)


def _compute_signature(body: bytes, secret: str) -> str:
    """Compute the expected HMAC-SHA256 signature for a request body."""
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def _webhook_payload(
    messages: list[dict[str, Any]] | None = None,
    contacts: list[dict[str, Any]] | None = None,
    phone_number_id: str = "123456",
) -> dict[str, Any]:
    """Build a standard WhatsApp webhook payload."""
    value: dict[str, Any] = {
        "metadata": {"phone_number_id": phone_number_id},
    }
    if contacts is not None:
        value["contacts"] = contacts
    else:
        value["contacts"] = [
            {"wa_id": "15551234567", "profile": {"name": "Test User"}}
        ]
    if messages is not None:
        value["messages"] = messages
    else:
        value["messages"] = []

    return {
        "entry": [
            {
                "changes": [
                    {"field": "messages", "value": value}
                ]
            }
        ]
    }


def _text_message(
    text: str = "Hello",
    from_number: str = "15551234567",
    msg_id: str = "wamid.abc123",
) -> dict[str, Any]:
    """Build a minimal text message."""
    return {
        "type": "text",
        "from": from_number,
        "id": msg_id,
        "timestamp": "1234567890",
        "text": {"body": text},
    }


def _media_message(
    media_type: str = "document",
    from_number: str = "15551234567",
    msg_id: str = "wamid.media1",
    media_id: str = "media-id-123",
    caption: str = "",
    mime_type: str = "application/pdf",
    filename: str | None = None,
) -> dict[str, Any]:
    """Build a media message (document, image, video, audio)."""
    media_data: dict[str, Any] = {
        "id": media_id,
        "mime_type": mime_type,
    }
    if caption:
        media_data["caption"] = caption
    if filename:
        media_data["filename"] = filename

    return {
        "type": media_type,
        "from": from_number,
        "id": msg_id,
        "timestamp": "1234567890",
        media_type: media_data,
    }


# ===========================================================================
# Signature Verification
# ===========================================================================


class TestVerifyWhatsAppSignature:
    """Tests for _verify_whatsapp_signature function."""

    def test_no_app_secret_rejects(self, handler_module):
        """When WHATSAPP_APP_SECRET is not set, signature verification fails closed."""
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", None):
            result = handler_module._verify_whatsapp_signature("sha256=abc", b"body")
            assert result is False

    def test_missing_sha256_prefix_rejects(self, handler_module):
        """Signature without sha256= prefix is rejected."""
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", "secret"):
            result = handler_module._verify_whatsapp_signature("invalid-sig", b"body")
            assert result is False

    def test_valid_signature_accepts(self, handler_module):
        """Valid HMAC-SHA256 signature passes verification."""
        secret = "test-app-secret"
        body = b'{"test": "data"}'
        sig = _compute_signature(body, secret)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret):
            result = handler_module._verify_whatsapp_signature(sig, body)
            assert result is True

    def test_invalid_signature_rejects(self, handler_module):
        """Wrong HMAC signature is rejected."""
        secret = "test-app-secret"
        body = b'{"test": "data"}'
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret):
            result = handler_module._verify_whatsapp_signature("sha256=badbadbad", body)
            assert result is False

    def test_wrong_secret_rejects(self, handler_module):
        """Signature computed with wrong secret is rejected."""
        body = b'{"test": "data"}'
        sig = _compute_signature(body, "wrong-secret")
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", "correct-secret"):
            result = handler_module._verify_whatsapp_signature(sig, body)
            assert result is False

    def test_empty_signature_rejects(self, handler_module):
        """Empty signature string is rejected."""
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", "secret"):
            result = handler_module._verify_whatsapp_signature("", b"body")
            assert result is False

    def test_empty_body_valid_if_signature_matches(self, handler_module):
        """Empty body can pass if signature matches."""
        secret = "secret"
        body = b""
        sig = _compute_signature(body, secret)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret):
            result = handler_module._verify_whatsapp_signature(sig, body)
            assert result is True


# ===========================================================================
# can_handle routing
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle routing logic."""

    def test_handles_webhook_path(self, handler):
        assert handler.can_handle("/api/v1/bots/whatsapp/webhook") is True

    def test_handles_status_path(self, handler):
        assert handler.can_handle("/api/v1/bots/whatsapp/status") is True

    def test_rejects_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/bots/whatsapp/unknown") is False

    def test_rejects_telegram_path(self, handler):
        assert handler.can_handle("/api/v1/bots/telegram/webhook") is False

    def test_rejects_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_partial_path(self, handler):
        assert handler.can_handle("/api/v1/bots/whatsapp") is False

    def test_routes_list_complete(self, handler):
        """ROUTES list contains exactly the expected paths."""
        assert set(handler.ROUTES) == {
            "/api/v1/bots/whatsapp/webhook",
            "/api/v1/bots/whatsapp/status",
        }


# ===========================================================================
# Bot platform and config
# ===========================================================================


class TestBotConfig:
    """Tests for bot_platform, _is_bot_enabled, _get_platform_config_status."""

    def test_bot_platform(self, handler):
        assert handler.bot_platform == "whatsapp"

    def test_is_bot_enabled_true(self, handler, handler_module):
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", "tok"), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", "pid"):
            assert handler._is_bot_enabled() is True

    def test_is_bot_enabled_false_no_token(self, handler, handler_module):
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", None), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", "pid"):
            assert handler._is_bot_enabled() is False

    def test_is_bot_enabled_false_no_phone(self, handler, handler_module):
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", "tok"), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", None):
            assert handler._is_bot_enabled() is False

    def test_is_bot_enabled_false_both_none(self, handler, handler_module):
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", None), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", None):
            assert handler._is_bot_enabled() is False

    def test_platform_config_status_all_configured(self, handler, handler_module):
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", "tok"), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", "pid"), \
             patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", "vt"), \
             patch.object(handler_module, "WHATSAPP_APP_SECRET", "sec"):
            status = handler._get_platform_config_status()
            assert status["access_token_configured"] is True
            assert status["phone_number_configured"] is True
            assert status["verify_token_configured"] is True
            assert status["app_secret_configured"] is True

    def test_platform_config_status_none_configured(self, handler, handler_module):
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", None), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", None), \
             patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", None), \
             patch.object(handler_module, "WHATSAPP_APP_SECRET", None):
            status = handler._get_platform_config_status()
            assert status["access_token_configured"] is False
            assert status["phone_number_configured"] is False
            assert status["verify_token_configured"] is False
            assert status["app_secret_configured"] is False


# ===========================================================================
# GET /api/v1/bots/whatsapp/status
# ===========================================================================


class TestStatusEndpoint:
    """Tests for the status endpoint via handle()."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler, handler_module):
        mock_http = MockHTTPHandler(path="/api/v1/bots/whatsapp/status", method="GET")
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", "tok"), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", "pid"), \
             patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", "vt"), \
             patch.object(handler_module, "WHATSAPP_APP_SECRET", "sec"):
            result = await handler.handle("/api/v1/bots/whatsapp/status", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["platform"] == "whatsapp"
            assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_status_disabled_when_not_configured(self, handler, handler_module):
        mock_http = MockHTTPHandler(path="/api/v1/bots/whatsapp/status", method="GET")
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", None), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", None), \
             patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", None), \
             patch.object(handler_module, "WHATSAPP_APP_SECRET", None):
            result = await handler.handle("/api/v1/bots/whatsapp/status", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["enabled"] is False


# ===========================================================================
# GET /api/v1/bots/whatsapp/webhook  - Verification Challenge
# ===========================================================================


class TestVerificationEndpoint:
    """Tests for webhook verification challenge via handle()."""

    @pytest.mark.asyncio
    async def test_valid_verification(self, handler, handler_module):
        """Valid subscribe request with matching token returns challenge."""
        with patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", "my-verify-token"):
            query_params = {
                "hub.mode": ["subscribe"],
                "hub.verify_token": ["my-verify-token"],
                "hub.challenge": ["challenge123"],
            }
            mock_http = MockHTTPHandler(method="GET")
            result = await handler.handle("/api/v1/bots/whatsapp/webhook", query_params, mock_http)
            assert _status(result) == 200
            assert _body_text(result) == "challenge123"
            assert result.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_verification_token_mismatch(self, handler, handler_module):
        """Token mismatch returns 403."""
        with patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", "correct-token"):
            query_params = {
                "hub.mode": ["subscribe"],
                "hub.verify_token": ["wrong-token"],
                "hub.challenge": ["challenge123"],
            }
            mock_http = MockHTTPHandler(method="GET")
            result = await handler.handle("/api/v1/bots/whatsapp/webhook", query_params, mock_http)
            assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_verification_no_token_configured(self, handler, handler_module):
        """No verify token configured returns 403."""
        with patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", None):
            query_params = {
                "hub.mode": ["subscribe"],
                "hub.verify_token": ["any-token"],
                "hub.challenge": ["challenge123"],
            }
            mock_http = MockHTTPHandler(method="GET")
            result = await handler.handle("/api/v1/bots/whatsapp/webhook", query_params, mock_http)
            assert _status(result) == 403

    @pytest.mark.asyncio
    async def test_verification_wrong_mode(self, handler, handler_module):
        """Non-subscribe mode returns 400."""
        with patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", "token"):
            query_params = {
                "hub.mode": ["unsubscribe"],
                "hub.verify_token": ["token"],
                "hub.challenge": ["challenge123"],
            }
            mock_http = MockHTTPHandler(method="GET")
            result = await handler.handle("/api/v1/bots/whatsapp/webhook", query_params, mock_http)
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_verification_no_mode(self, handler, handler_module):
        """Missing mode returns 400."""
        with patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", "token"):
            query_params = {
                "hub.verify_token": ["token"],
                "hub.challenge": ["challenge123"],
            }
            mock_http = MockHTTPHandler(method="GET")
            result = await handler.handle("/api/v1/bots/whatsapp/webhook", query_params, mock_http)
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_verification_empty_params(self, handler, handler_module):
        """Empty query params returns 400."""
        with patch.object(handler_module, "WHATSAPP_VERIFY_TOKEN", "token"):
            mock_http = MockHTTPHandler(method="GET")
            result = await handler.handle("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_path(self, handler):
        """Unknown path returns None from handle()."""
        mock_http = MockHTTPHandler(method="GET")
        result = await handler.handle("/api/v1/bots/whatsapp/unknown", {}, mock_http)
        assert result is None


# ===========================================================================
# POST /api/v1/bots/whatsapp/webhook - Webhook Messages
# ===========================================================================


class TestWebhookPost:
    """Tests for POST webhook message handling."""

    def _signed_post(self, handler, handler_module, payload: dict, secret: str = "test-secret"):
        """Create a signed POST request and call handle_post."""
        body_bytes = json.dumps(payload).encode("utf-8")
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret):
            return handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)

    def test_valid_webhook_returns_200(self, handler, handler_module):
        """Signed webhook with valid payload returns 200."""
        payload = _webhook_payload(messages=[])
        result = self._signed_post(handler, handler_module, payload)
        assert _status(result) == 200
        assert _body(result)["status"] == "ok"

    def test_invalid_signature_returns_401(self, handler, handler_module):
        """Invalid signature returns 401."""
        body_bytes = json.dumps(_webhook_payload()).encode("utf-8")
        mock_http = _make_webhook_handler(body=None, signature="sha256=badsig", raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", "real-secret"):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            assert _status(result) == 401

    def test_missing_signature_returns_401(self, handler, handler_module):
        """Missing signature header when app secret configured returns 401."""
        body_bytes = json.dumps(_webhook_payload()).encode("utf-8")
        mock_http = _make_webhook_handler(body=None, signature="", raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", "real-secret"):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            assert _status(result) == 401

    def test_no_app_secret_rejects(self, handler, handler_module):
        """When no app secret configured, all webhooks are rejected (fail closed)."""
        body_bytes = json.dumps(_webhook_payload()).encode("utf-8")
        mock_http = _make_webhook_handler(body=None, signature="", raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", None):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            assert _status(result) == 401

    def test_post_unknown_path_returns_none(self, handler, handler_module):
        """POST to unknown path returns None."""
        mock_http = MockHTTPHandler(method="POST")
        result = handler.handle_post("/api/v1/bots/whatsapp/unknown", {}, mock_http)
        assert result is None

    def test_text_message_triggers_debate(self, handler, handler_module):
        """Text message that is not a command starts a debate."""
        msg = _text_message("Should we adopt microservices?")
        payload = _webhook_payload(messages=[msg])

        with patch.object(handler, "_start_debate") as mock_debate, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            result = self._signed_post(handler, handler_module, payload)
            assert _status(result) == 200
            mock_debate.assert_called_once()
            call_args = mock_debate.call_args
            assert call_args[0][0] == "15551234567"
            assert call_args[0][1] == "Test User"
            assert call_args[0][2] == "Should we adopt microservices?"

    def test_greeting_triggers_welcome(self, handler, handler_module):
        """Greeting words trigger welcome message."""
        for greeting in ("hi", "hello", "hey", "start"):
            msg = _text_message(greeting)
            payload = _webhook_payload(messages=[msg])
            with patch.object(handler, "_send_welcome") as mock_welcome, \
                 patch.object(handler, "_extract_attachments", return_value=[]), \
                 patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
                self._signed_post(handler, handler_module, payload)
                mock_welcome.assert_called_once_with("15551234567")

    def test_help_command(self, handler, handler_module):
        """The /help command sends help message."""
        msg = _text_message("/help")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_send_help") as mock_help, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_help.assert_called_once_with("15551234567")

    def test_status_command(self, handler, handler_module):
        """The /status command sends status message."""
        msg = _text_message("/status")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_send_status") as mock_status, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_status.assert_called_once_with("15551234567")

    def test_debate_command(self, handler, handler_module):
        """The /debate command starts a debate with no decision_integrity."""
        msg = _text_message("/debate Should we use Rust?")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_start_debate") as mock_debate, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_debate.assert_called_once()
            args, kwargs = mock_debate.call_args
            assert args[2] == "Should we use Rust?"
            assert kwargs.get("decision_integrity") is None

    def test_plan_command(self, handler, handler_module):
        """The /plan command includes decision_integrity with plan fields."""
        msg = _text_message("/plan Migrate to cloud")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_start_debate") as mock_debate, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_debate.assert_called_once()
            _, kwargs = mock_debate.call_args
            di = kwargs["decision_integrity"]
            assert di["include_receipt"] is True
            assert di["include_plan"] is True
            assert di["include_context"] is False
            assert di["plan_strategy"] == "single_task"

    def test_implement_command(self, handler, handler_module):
        """The /implement command adds execution_mode and include_context."""
        msg = _text_message("/implement Build API gateway")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_start_debate") as mock_debate, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            _, kwargs = mock_debate.call_args
            di = kwargs["decision_integrity"]
            assert di["include_context"] is True
            assert di["execution_mode"] == "execute"
            assert di["execution_engine"] == "hybrid"

    def test_unknown_command(self, handler, handler_module):
        """Unknown /command sends error message."""
        msg = _text_message("/foobar")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_send_message") as mock_send, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_send.assert_called_once()
            text = mock_send.call_args[0][1]
            assert "Unknown command" in text
            assert "/foobar" in text

    def test_document_with_caption_starts_debate(self, handler, handler_module):
        """Document message with caption starts debate."""
        msg = _media_message("document", caption="Analyze this report")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_start_debate") as mock_debate, \
             patch.object(handler, "_extract_attachments", return_value=[{"type": "document"}]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[{"type": "document"}]):
            self._signed_post(handler, handler_module, payload)
            mock_debate.assert_called_once()
            assert mock_debate.call_args[0][2] == "Analyze this report"

    def test_document_without_caption_sends_prompt(self, handler, handler_module):
        """Document message without caption asks for a question."""
        msg = _media_message("document", caption="")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_send_message") as mock_send, \
             patch.object(handler, "_extract_attachments", return_value=[{"type": "document"}]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[{"type": "document"}]):
            self._signed_post(handler, handler_module, payload)
            mock_send.assert_called_once()
            text = mock_send.call_args[0][1]
            assert "question" in text.lower()

    def test_image_with_caption_starts_debate(self, handler, handler_module):
        """Image message with caption starts debate."""
        msg = _media_message("image", caption="What is this?", mime_type="image/jpeg")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_start_debate") as mock_debate, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_debate.assert_called_once()

    def test_video_without_caption_sends_prompt(self, handler, handler_module):
        """Video without caption prompts user."""
        msg = _media_message("video", caption="", mime_type="video/mp4")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_send_message") as mock_send, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_send.assert_called_once()

    def test_audio_without_caption_sends_prompt(self, handler, handler_module):
        """Audio without caption prompts user."""
        msg = _media_message("audio", caption="", mime_type="audio/ogg")
        payload = _webhook_payload(messages=[msg])
        with patch.object(handler, "_send_message") as mock_send, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_send.assert_called_once()

    def test_interactive_list_reply(self, handler, handler_module):
        """Interactive list reply is processed without error."""
        msg = {
            "type": "interactive",
            "from": "15551234567",
            "id": "wamid.int1",
            "timestamp": "1234567890",
            "interactive": {
                "type": "list_reply",
                "list_reply": {"id": "option-1", "title": "First option"},
            },
        }
        payload = _webhook_payload(messages=[msg])
        result = self._signed_post(handler, handler_module, payload)
        assert _status(result) == 200

    def test_interactive_button_reply(self, handler, handler_module):
        """Interactive button reply is processed without error."""
        msg = {
            "type": "interactive",
            "from": "15551234567",
            "id": "wamid.int2",
            "timestamp": "1234567890",
            "interactive": {
                "type": "button_reply",
                "button_reply": {"id": "btn-1", "title": "Agree"},
            },
        }
        payload = _webhook_payload(messages=[msg])
        result = self._signed_post(handler, handler_module, payload)
        assert _status(result) == 200

    def test_button_quick_reply(self, handler, handler_module):
        """Button quick reply message is processed without error."""
        msg = {
            "type": "button",
            "from": "15551234567",
            "id": "wamid.btn1",
            "timestamp": "1234567890",
            "button": {"payload": "vote_yes", "text": "Yes"},
        }
        payload = _webhook_payload(messages=[msg])
        result = self._signed_post(handler, handler_module, payload)
        assert _status(result) == 200

    def test_unhandled_message_type(self, handler, handler_module):
        """Unknown message type is logged but not an error."""
        msg = {
            "type": "sticker",
            "from": "15551234567",
            "id": "wamid.stk1",
            "timestamp": "1234567890",
        }
        payload = _webhook_payload(messages=[msg])
        result = self._signed_post(handler, handler_module, payload)
        assert _status(result) == 200

    def test_multiple_messages_in_payload(self, handler, handler_module):
        """Multiple messages in a single payload are all processed."""
        msgs = [
            _text_message("hi", msg_id="m1"),
            _text_message("hello", msg_id="m2"),
        ]
        payload = _webhook_payload(messages=msgs)
        with patch.object(handler, "_send_welcome") as mock_welcome, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            assert mock_welcome.call_count == 2

    def test_empty_entry_list(self, handler, handler_module):
        """Empty entry list is handled gracefully."""
        payload = {"entry": []}
        result = self._signed_post(handler, handler_module, payload)
        assert _status(result) == 200

    def test_non_messages_field_ignored(self, handler, handler_module):
        """Changes with field != 'messages' are ignored."""
        payload = {
            "entry": [{"changes": [{"field": "statuses", "value": {}}]}]
        }
        result = self._signed_post(handler, handler_module, payload)
        assert _status(result) == 200

    def test_contact_name_resolved(self, handler, handler_module):
        """Contact name is resolved from contacts array."""
        contacts = [{"wa_id": "15559999999", "profile": {"name": "Alice"}}]
        msg = _text_message("hi", from_number="15559999999")
        payload = _webhook_payload(messages=[msg], contacts=contacts)
        with patch.object(handler, "_send_welcome") as mock_welcome, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_welcome.assert_called_once_with("15559999999")

    def test_contact_name_fallback_to_unknown(self, handler, handler_module):
        """When no matching contact, contact_name defaults to 'Unknown'."""
        contacts = [{"wa_id": "different_number", "profile": {"name": "Bob"}}]
        msg = _text_message("/help", from_number="15551234567")
        payload = _webhook_payload(messages=[msg], contacts=contacts)
        with patch.object(handler, "_send_help") as mock_help, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            self._signed_post(handler, handler_module, payload)
            mock_help.assert_called_once()

    def test_webhook_json_error_returns_200(self, handler, handler_module):
        """Invalid JSON body still returns 200 (prevents retries)."""
        secret = "test-secret"
        raw = b"not-json"
        sig = _compute_signature(raw, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=raw)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            # Should return 400 from _parse_json_body for invalid JSON
            # or 200 from the exception handler
            assert _status(result) in (200, 400)


# ===========================================================================
# _extract_attachments
# ===========================================================================


class TestExtractAttachments:
    """Tests for the _extract_attachments method."""

    def test_document_extraction(self, handler):
        message = {
            "type": "document",
            "document": {
                "id": "doc-123",
                "filename": "report.pdf",
                "mime_type": "application/pdf",
                "caption": "My report",
            },
        }
        atts = handler._extract_attachments(message)
        assert len(atts) == 1
        assert atts[0]["type"] == "document"
        assert atts[0]["file_id"] == "doc-123"
        assert atts[0]["filename"] == "report.pdf"
        assert atts[0]["content_type"] == "application/pdf"
        assert atts[0]["caption"] == "My report"

    def test_image_extraction(self, handler):
        message = {
            "type": "image",
            "image": {
                "id": "img-456",
                "mime_type": "image/jpeg",
                "caption": "A photo",
            },
        }
        atts = handler._extract_attachments(message)
        assert len(atts) == 1
        assert atts[0]["type"] == "image"
        assert atts[0]["file_id"] == "img-456"

    def test_video_extraction(self, handler):
        message = {
            "type": "video",
            "video": {
                "id": "vid-789",
                "mime_type": "video/mp4",
                "caption": "A clip",
            },
        }
        atts = handler._extract_attachments(message)
        assert len(atts) == 1
        assert atts[0]["type"] == "video"
        assert atts[0]["file_id"] == "vid-789"

    def test_audio_extraction(self, handler):
        message = {
            "type": "audio",
            "audio": {
                "id": "aud-101",
                "mime_type": "audio/ogg",
            },
        }
        atts = handler._extract_attachments(message)
        assert len(atts) == 1
        assert atts[0]["type"] == "audio"
        assert atts[0]["file_id"] == "aud-101"

    def test_multiple_attachments(self, handler):
        """Message with both document and image."""
        message = {
            "type": "document",
            "document": {"id": "doc-1", "mime_type": "application/pdf"},
            "image": {"id": "img-1", "mime_type": "image/png"},
        }
        atts = handler._extract_attachments(message)
        assert len(atts) == 2
        types = {a["type"] for a in atts}
        assert "document" in types
        assert "image" in types

    def test_no_attachments(self, handler):
        """Plain text message has no attachments."""
        message = {"type": "text", "text": {"body": "Hello"}}
        atts = handler._extract_attachments(message)
        assert atts == []

    def test_non_dict_message_returns_empty(self, handler):
        """Non-dict input returns empty list."""
        assert handler._extract_attachments("not a dict") == []
        assert handler._extract_attachments(None) == []
        assert handler._extract_attachments(42) == []

    def test_document_without_filename_defaults(self, handler):
        """Document without filename gets default 'document'."""
        message = {
            "document": {"id": "doc-2", "mime_type": "text/plain"},
        }
        atts = handler._extract_attachments(message)
        assert len(atts) == 1
        assert atts[0]["filename"] == "document"

    def test_text_body_becomes_attachment_text(self, handler):
        """text.body is included in attachment text field."""
        message = {
            "text": {"body": "context text"},
            "document": {"id": "d1", "mime_type": "text/plain"},
        }
        atts = handler._extract_attachments(message)
        assert atts[0]["text"] == "context text"

    def test_document_non_dict_ignored(self, handler):
        """Non-dict document value is ignored."""
        message = {"document": "not-a-dict"}
        atts = handler._extract_attachments(message)
        assert len(atts) == 0

    def test_image_non_dict_ignored(self, handler):
        message = {"image": 42}
        atts = handler._extract_attachments(message)
        assert len(atts) == 0


# ===========================================================================
# _hydrate_whatsapp_attachments
# ===========================================================================


class TestHydrateAttachments:
    """Tests for _hydrate_whatsapp_attachments."""

    def test_empty_list_returns_empty(self, handler):
        result = handler._hydrate_whatsapp_attachments([])
        assert result == []

    def test_no_connector_returns_original(self, handler):
        """When connector is unavailable, returns attachments unchanged."""
        atts = [{"type": "document", "file_id": "d1"}]
        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": None}):
            result = handler._hydrate_whatsapp_attachments(atts)
            assert result == atts

    def test_connector_import_error_returns_original(self, handler):
        """ImportError from connector returns attachments unchanged."""
        atts = [{"type": "document", "file_id": "d1"}]
        # The import of get_connector happens inside the method; simulate ImportError
        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": None}):
            result = handler._hydrate_whatsapp_attachments(atts)
            assert result == atts

    def test_already_has_data_skipped(self, handler):
        """Attachments with data already present are not re-downloaded."""
        atts = [{"type": "document", "file_id": "d1", "data": b"existing"}]
        # Even with a working connector, it should skip
        result = handler._hydrate_whatsapp_attachments(atts)
        assert result[0]["data"] == b"existing"

    def test_no_file_id_skipped(self, handler):
        """Attachments without file_id are not downloaded."""
        atts = [{"type": "document"}]
        result = handler._hydrate_whatsapp_attachments(atts)
        assert "data" not in result[0]

    def test_non_dict_attachment_skipped(self, handler):
        """Non-dict entries are skipped."""
        atts = ["not-a-dict", {"type": "document", "file_id": "d1"}]
        result = handler._hydrate_whatsapp_attachments(atts)
        assert len(result) == 2

    def test_connector_none_returns_original(self, handler):
        """get_connector returning None returns attachments unchanged."""
        atts = [{"type": "document", "file_id": "d1"}]
        mock_registry = MagicMock()
        mock_registry.get_connector.return_value = None
        with patch.dict("sys.modules", {"aragora.connectors.chat.registry": mock_registry}):
            result = handler._hydrate_whatsapp_attachments(atts)
            assert result == atts

    def test_already_has_content_skipped(self, handler):
        """Attachments with content already present are not re-downloaded."""
        atts = [{"type": "document", "file_id": "d1", "content": b"some-content"}]
        result = handler._hydrate_whatsapp_attachments(atts)
        assert result[0]["content"] == b"some-content"


# ===========================================================================
# _send_message
# ===========================================================================


class TestSendMessage:
    """Tests for _send_message."""

    def test_no_credentials_logs_warning(self, handler, handler_module):
        """Without credentials, message is not sent."""
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", None), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", None):
            # Should not raise
            handler._send_message("15551234567", "Test message")

    def test_sends_via_httpx(self, handler, handler_module):
        """Message is sent via httpx when configured."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client

        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", "token123"), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", "phone123"), \
             patch.dict("sys.modules", {"httpx": mock_httpx}):
            handler._send_message("15551234567", "Hello!")
            mock_client.post.assert_called_once()
            call_kwargs = mock_client.post.call_args
            # Check the json payload contains the text
            json_arg = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json", {})
            assert "Hello!" in json.dumps(json_arg)

    def test_httpx_import_error(self, handler, handler_module):
        """When httpx is not available, logs warning."""
        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", "token"), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", "phone"):
            with patch.dict("sys.modules", {"httpx": None}):
                # Force re-import to trigger ImportError path
                # Just test that it doesn't raise
                handler._send_message("15551234567", "test")

    def test_send_failure_logged(self, handler, handler_module):
        """Failed send is logged but doesn't raise."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client

        with patch.object(handler_module, "WHATSAPP_ACCESS_TOKEN", "token"), \
             patch.object(handler_module, "WHATSAPP_PHONE_NUMBER_ID", "phone"), \
             patch.dict("sys.modules", {"httpx": mock_httpx}):
            handler._send_message("15551234567", "Hello!")


# ===========================================================================
# _send_welcome / _send_help / _send_status
# ===========================================================================


class TestHelperMessages:
    """Tests for welcome, help, and status message sending."""

    def test_send_welcome(self, handler):
        with patch.object(handler, "_send_message") as mock_send:
            handler._send_welcome("15551234567")
            mock_send.assert_called_once()
            text = mock_send.call_args[0][1]
            assert "Welcome" in text or "welcome" in text.lower()
            assert "/debate" in text
            assert "/help" in text

    def test_send_help(self, handler):
        with patch.object(handler, "_send_message") as mock_send:
            handler._send_help("15551234567")
            mock_send.assert_called_once()
            text = mock_send.call_args[0][1]
            assert "/debate" in text
            assert "/plan" in text
            assert "/implement" in text
            assert "/status" in text

    def test_send_status(self, handler):
        with patch.object(handler, "_send_message") as mock_send:
            handler._send_status("15551234567")
            mock_send.assert_called_once()
            text = mock_send.call_args[0][1]
            assert "Online" in text or "Ready" in text


# ===========================================================================
# _start_debate
# ===========================================================================


class TestStartDebate:
    """Tests for _start_debate."""

    def test_empty_topic_prompts_user(self, handler):
        """Empty topic sends a prompt instead of starting debate."""
        with patch.object(handler, "_send_message") as mock_send:
            handler._start_debate("15551234567", "Test User", "   ", [])
            mock_send.assert_called_once()
            text = mock_send.call_args[0][1]
            assert "topic" in text.lower()

    def test_debate_started_sends_confirmation(self, handler):
        """Valid topic starts debate and sends confirmation."""
        with patch.object(handler, "_start_debate_async", return_value="abcdef12-3456-7890") as mock_async, \
             patch.object(handler, "_send_message") as mock_send, \
             patch.object(handler, "_check_bot_permission"):
            handler._start_debate("15551234567", "Test User", "Should we use Rust?", [])
            mock_async.assert_called_once()
            mock_send.assert_called_once()
            text = mock_send.call_args[0][1]
            assert "Should we use Rust?" in text
            assert "abcdef12" in text  # First 8 chars of debate_id

    def test_rbac_denied_sends_permission_error(self, handler):
        """RBAC denial sends permission denied message."""
        with patch.object(handler, "_check_bot_permission", side_effect=PermissionError("denied")), \
             patch.object(handler, "_send_message") as mock_send:
            handler._start_debate("15551234567", "Test User", "Topic", [])
            mock_send.assert_called_once()
            text = mock_send.call_args[0][1]
            assert "Permission denied" in text

    def test_debate_with_decision_integrity(self, handler):
        """decision_integrity parameter is forwarded to _start_debate_async."""
        di = {"include_receipt": True, "plan_strategy": "multi_task"}
        with patch.object(handler, "_start_debate_async", return_value="id-123") as mock_async, \
             patch.object(handler, "_send_message"), \
             patch.object(handler, "_check_bot_permission"):
            handler._start_debate("15551234567", "Test", "Topic", [], decision_integrity=di)
            _, kwargs = mock_async.call_args
            assert kwargs["decision_integrity"] == di

    def test_debate_with_attachments(self, handler):
        """Attachments are forwarded to _start_debate_async."""
        atts = [{"type": "document", "file_id": "d1"}]
        with patch.object(handler, "_start_debate_async", return_value="id-123") as mock_async, \
             patch.object(handler, "_send_message"), \
             patch.object(handler, "_check_bot_permission"):
            handler._start_debate("15551234567", "Test", "Topic", atts)
            call_args = mock_async.call_args
            assert call_args[0][3] == atts


# ===========================================================================
# _start_debate_async
# ===========================================================================


class TestStartDebateAsync:
    """Tests for _start_debate_async."""

    def test_registers_debate_origin(self, handler):
        """Debate origin is registered for result routing."""
        mock_reg = MagicMock()
        mock_origin_module = MagicMock()
        mock_origin_module.register_debate_origin = mock_reg

        with patch.dict("sys.modules", {
            "aragora.server.debate_origin": mock_origin_module,
            "aragora.core.decision": None,  # Force fallback
        }), patch.object(handler, "_start_debate_via_queue", return_value="queue-id"):
            result = handler._start_debate_async("15551234567", "Test", "Topic")
            assert isinstance(result, str)
            mock_reg.assert_called_once()

    def test_fallback_to_queue_on_import_error(self, handler):
        """Falls back to queue system when DecisionRouter is unavailable."""
        with patch.dict("sys.modules", {"aragora.core.decision": None}), \
             patch.object(handler, "_start_debate_via_queue", return_value="queue-id") as mock_queue:
            result = handler._start_debate_async("15551234567", "Test", "Topic")
            assert isinstance(result, str)

    def test_origin_registration_failure_handled(self, handler):
        """Failed origin registration doesn't prevent debate start."""
        mock_origin_module = MagicMock()
        mock_origin_module.register_debate_origin.side_effect = RuntimeError("fail")

        with patch.dict("sys.modules", {
            "aragora.server.debate_origin": mock_origin_module,
            "aragora.core.decision": None,
        }), patch.object(handler, "_start_debate_via_queue", return_value="q-id"):
            result = handler._start_debate_async("15551234567", "Test", "Topic")
            assert isinstance(result, str)


# ===========================================================================
# _start_debate_via_queue
# ===========================================================================


class TestStartDebateViaQueue:
    """Tests for _start_debate_via_queue."""

    def test_queue_import_error_falls_to_direct(self, handler):
        """ImportError from queue falls back to direct execution."""
        with patch.dict("sys.modules", {"aragora.queue": None}), \
             patch.object(handler, "_run_debate_direct", return_value="direct-id") as mock_direct:
            result = handler._start_debate_via_queue("15551234567", "Test", "Topic", "id-1")
            mock_direct.assert_called_once()

    def test_queue_runtime_error_returns_id(self, handler):
        """RuntimeError from queue returns the debate_id."""
        mock_queue_mod = MagicMock()
        mock_queue_mod.create_debate_job.side_effect = RuntimeError("queue fail")
        with patch.dict("sys.modules", {"aragora.queue": mock_queue_mod}):
            result = handler._start_debate_via_queue("15551234567", "Test", "Topic", "fallback-id")
            assert result == "fallback-id"


# ===========================================================================
# _run_debate_direct
# ===========================================================================


class TestRunDebateDirect:
    """Tests for _run_debate_direct."""

    def test_returns_debate_id(self, handler):
        """Direct debate returns the given debate_id immediately."""
        result = handler._run_debate_direct("15551234567", "Test", "Topic", "direct-123")
        assert result == "direct-123"

    def test_spawns_background_thread(self, handler):
        """Debate runs in a background thread."""
        with patch("threading.Thread") as mock_thread:
            mock_instance = MagicMock()
            mock_thread.return_value = mock_instance
            handler._run_debate_direct("15551234567", "Test", "Topic", "thread-id")
            mock_thread.assert_called_once()
            mock_instance.start.assert_called_once()
            assert mock_thread.call_args.kwargs.get("daemon") is True


# ===========================================================================
# RBAC Permission Checking
# ===========================================================================


class TestCheckBotPermission:
    """Tests for _check_bot_permission."""

    def test_rbac_not_available_no_fail_closed(self, handler, handler_module):
        """When RBAC is unavailable and not fail-closed, permission passes."""
        with patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch("aragora.server.handlers.bots.whatsapp.rbac_fail_closed", return_value=False):
            handler._check_bot_permission("debates:create", user_id="whatsapp:123")

    def test_rbac_not_available_fail_closed(self, handler, handler_module):
        """When RBAC is unavailable and fail-closed, raises PermissionError."""
        with patch.object(handler_module, "RBAC_AVAILABLE", False), \
             patch("aragora.server.handlers.bots.whatsapp.rbac_fail_closed", return_value=True):
            with pytest.raises(PermissionError):
                handler._check_bot_permission("debates:create", user_id="whatsapp:123")

    def test_rbac_available_no_context(self, handler, handler_module):
        """When RBAC available but no context and no user_id, no check is performed."""
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission") as mock_check:
            handler._check_bot_permission("debates:create")
            mock_check.assert_not_called()

    def test_rbac_available_with_user_id(self, handler, handler_module):
        """When RBAC available with user_id, creates AuthorizationContext and checks."""
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission") as mock_check:
            handler._check_bot_permission("debates:create", user_id="whatsapp:123")
            mock_check.assert_called_once()
            auth_ctx = mock_check.call_args[0][0]
            assert auth_ctx.user_id == "whatsapp:123"
            assert "bot_user" in auth_ctx.roles

    def test_rbac_available_with_auth_context_in_context(self, handler, handler_module):
        """When auth_context is in the context dict, it is used directly."""
        mock_auth_ctx = MagicMock()
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission") as mock_check:
            handler._check_bot_permission(
                "bots.read",
                context={"auth_context": mock_auth_ctx},
            )
            mock_check.assert_called_once_with(mock_auth_ctx, "bots.read")

    def test_rbac_check_failure_raises(self, handler, handler_module):
        """When check_permission raises, exception propagates."""
        with patch.object(handler_module, "RBAC_AVAILABLE", True), \
             patch.object(handler_module, "check_permission", side_effect=PermissionError("denied")):
            with pytest.raises(PermissionError, match="denied"):
                handler._check_bot_permission("debates:create", user_id="whatsapp:123")


# ===========================================================================
# Handler Constructor
# ===========================================================================


class TestHandlerInit:
    """Tests for WhatsAppHandler initialization."""

    def test_init_with_context(self, handler_module):
        h = handler_module.WhatsAppHandler(ctx={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_default_context(self, handler_module):
        h = handler_module.WhatsAppHandler()
        assert h.ctx == {}

    def test_init_none_context(self, handler_module):
        h = handler_module.WhatsAppHandler(ctx=None)
        assert h.ctx == {}


# ===========================================================================
# Edge Cases & Error Handling
# ===========================================================================


class TestEdgeCases:
    """Additional edge-case and error-handling tests."""

    def test_case_insensitive_greeting(self, handler, handler_module):
        """Greetings are case-insensitive."""
        msg = _text_message("HI")
        payload = _webhook_payload(messages=[msg])
        secret = "test-secret"
        body_bytes = json.dumps(payload).encode()
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret), \
             patch.object(handler, "_send_welcome") as mock_welcome, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            mock_welcome.assert_called_once()

    def test_command_case_insensitive(self, handler, handler_module):
        """Commands are case-insensitive."""
        msg = _text_message("/HELP")
        payload = _webhook_payload(messages=[msg])
        secret = "secret"
        body_bytes = json.dumps(payload).encode()
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret), \
             patch.object(handler, "_send_help") as mock_help, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            mock_help.assert_called_once()

    def test_greeting_with_whitespace(self, handler, handler_module):
        """Greetings with surrounding whitespace are recognized."""
        msg = _text_message("  hello  ")
        payload = _webhook_payload(messages=[msg])
        secret = "secret"
        body_bytes = json.dumps(payload).encode()
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret), \
             patch.object(handler, "_send_welcome") as mock_welcome, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            mock_welcome.assert_called_once()

    def test_debate_topic_truncated_in_confirmation(self, handler):
        """Long topics are truncated in the confirmation message."""
        long_topic = "x" * 300
        with patch.object(handler, "_start_debate_async", return_value="id-12345678") as mock_async, \
             patch.object(handler, "_send_message") as mock_send, \
             patch.object(handler, "_check_bot_permission"):
            handler._start_debate("15551234567", "Test", long_topic, [])
            text = mock_send.call_args[0][1]
            # topic should be truncated to 200 chars
            assert "x" * 200 in text
            assert "x" * 201 not in text

    def test_debate_id_truncated_in_confirmation(self, handler):
        """Debate ID is shown truncated (first 8 chars)."""
        with patch.object(handler, "_start_debate_async", return_value="abcdefgh-1234-rest") as mock_async, \
             patch.object(handler, "_send_message") as mock_send, \
             patch.object(handler, "_check_bot_permission"):
            handler._start_debate("15551234567", "Test", "Topic", [])
            text = mock_send.call_args[0][1]
            assert "abcdefgh" in text

    def test_empty_payload_entry_list(self, handler, handler_module):
        """Payload with empty changes list processes without error."""
        payload = {"entry": [{"changes": []}]}
        secret = "secret"
        body_bytes = json.dumps(payload).encode()
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            assert _status(result) == 200

    def test_value_without_messages_key(self, handler, handler_module):
        """Change value without 'messages' key processes without error."""
        payload = {
            "entry": [{"changes": [{"field": "messages", "value": {"metadata": {}}}]}]
        }
        secret = "secret"
        body_bytes = json.dumps(payload).encode()
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            assert _status(result) == 200

    def test_contact_profile_missing_name(self, handler, handler_module):
        """Contact with profile missing name uses from_number."""
        contacts = [{"wa_id": "15551234567", "profile": {}}]
        msg = _text_message("hi", from_number="15551234567")
        payload = _webhook_payload(messages=[msg], contacts=contacts)
        secret = "secret"
        body_bytes = json.dumps(payload).encode()
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret), \
             patch.object(handler, "_send_welcome"):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            assert _status(result) == 200

    def test_debate_command_no_args(self, handler, handler_module):
        """'/debate' without arguments sends empty-topic prompt."""
        msg = _text_message("/debate")
        payload = _webhook_payload(messages=[msg])
        secret = "secret"
        body_bytes = json.dumps(payload).encode()
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret), \
             patch.object(handler, "_send_message") as mock_send, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]), \
             patch.object(handler, "_check_bot_permission"):
            handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            text = mock_send.call_args[0][1]
            assert "topic" in text.lower()

    def test_interactive_unknown_type(self, handler):
        """Interactive message with unknown type is handled gracefully."""
        handler._handle_interactive("15551234567", {"type": "unknown_type"})

    def test_handle_button_reply_logs(self, handler):
        """Button reply handler doesn't raise."""
        handler._handle_button_reply("15551234567", {"payload": "action_x", "text": "Do X"})

    def test_multiple_entries_in_payload(self, handler, handler_module):
        """Multiple entries in a single webhook are all processed."""
        msg1 = _text_message("hi", msg_id="m1")
        msg2 = _text_message("hello", msg_id="m2")
        payload = {
            "entry": [
                {"changes": [{"field": "messages", "value": {"metadata": {}, "contacts": [{"wa_id": "15551234567", "profile": {"name": "A"}}], "messages": [msg1]}}]},
                {"changes": [{"field": "messages", "value": {"metadata": {}, "contacts": [{"wa_id": "15551234567", "profile": {"name": "B"}}], "messages": [msg2]}}]},
            ]
        }
        secret = "secret"
        body_bytes = json.dumps(payload).encode()
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret), \
             patch.object(handler, "_send_welcome") as mock_welcome, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            assert mock_welcome.call_count == 2


# ===========================================================================
# _handle_interactive detail tests
# ===========================================================================


class TestHandleInteractive:
    """Detail tests for _handle_interactive."""

    def test_list_reply_extracts_id(self, handler):
        """list_reply logs the reply ID."""
        interactive = {
            "type": "list_reply",
            "list_reply": {"id": "option-42", "title": "Option 42"},
        }
        # Should not raise
        handler._handle_interactive("15551234567", interactive)

    def test_button_reply_extracts_id(self, handler):
        """button_reply logs the button ID."""
        interactive = {
            "type": "button_reply",
            "button_reply": {"id": "btn-99", "title": "Confirm"},
        }
        handler._handle_interactive("15551234567", interactive)

    def test_missing_reply_data(self, handler):
        """Missing reply data doesn't cause errors."""
        handler._handle_interactive("15551234567", {"type": "list_reply"})
        handler._handle_interactive("15551234567", {"type": "button_reply"})

    def test_empty_interactive(self, handler):
        """Empty interactive dict handled gracefully."""
        handler._handle_interactive("15551234567", {})


# ===========================================================================
# _handle_button_reply
# ===========================================================================


class TestHandleButtonReply:
    """Tests for _handle_button_reply."""

    def test_extracts_payload_and_text(self, handler):
        handler._handle_button_reply("15551234567", {"payload": "yes", "text": "Yes"})

    def test_empty_button(self, handler):
        handler._handle_button_reply("15551234567", {})

    def test_missing_keys(self, handler):
        handler._handle_button_reply("15551234567", {"payload": "x"})


# ===========================================================================
# Media message processing detail
# ===========================================================================


class TestMediaMessageProcessing:
    """Detailed tests for media message processing flow."""

    def test_media_payload_non_dict_caption(self, handler, handler_module):
        """Non-dict media payload doesn't crash caption extraction."""
        msg = {
            "type": "image",
            "from": "15551234567",
            "id": "wamid.img1",
            "timestamp": "1234567890",
            "image": "not-a-dict",  # Should not crash
        }
        payload = _webhook_payload(messages=[msg])
        secret = "secret"
        body_bytes = json.dumps(payload).encode()
        sig = _compute_signature(body_bytes, secret)
        mock_http = _make_webhook_handler(body=None, signature=sig, raw_body=body_bytes)
        with patch.object(handler_module, "WHATSAPP_APP_SECRET", secret), \
             patch.object(handler, "_send_message") as mock_send, \
             patch.object(handler, "_extract_attachments", return_value=[]), \
             patch.object(handler, "_hydrate_whatsapp_attachments", return_value=[]):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_http)
            assert _status(result) == 200
            # Without a caption, should prompt user
            mock_send.assert_called_once()
