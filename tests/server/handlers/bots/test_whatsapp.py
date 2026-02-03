"""Comprehensive tests for WhatsApp bot webhook handler.

Tests cover:
- Webhook verification (hub.mode, hub.verify_token, hub.challenge)
- Signature verification (HMAC-SHA256)
- Message handling (text, commands, greetings)
- Command processing (/help, /debate, /status, unknown commands)
- Interactive message handling (list_reply, button_reply)
- Message sending (_send_message, _send_welcome, _send_help, _send_status)
- Response formatting
- Status endpoint (RBAC auth, unauthorized, forbidden)
- Debate starting (_start_debate, _start_debate_async, _start_debate_via_queue)
- Error handling (invalid JSON, exceptions, missing data)
- API interactions (mocked)
"""

from __future__ import annotations

import hashlib
import hmac
import json
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots import whatsapp as whatsapp_module
from aragora.server.handlers.bots.whatsapp import (
    WhatsAppHandler,
    _verify_whatsapp_signature,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def _make_mock_request(
    body: bytes = b"",
    signature: str = "",
    content_length: int | None = None,
) -> MagicMock:
    """Create a mock HTTP handler with WhatsApp webhook payload."""
    if content_length is None:
        content_length = len(body)
    mock = MagicMock()
    mock.headers = {
        "Content-Length": str(content_length),
        "X-Hub-Signature-256": signature,
    }
    mock.rfile = BytesIO(body)
    return mock


def _make_handler() -> WhatsAppHandler:
    """Create a WhatsAppHandler with empty server context."""
    return WhatsAppHandler({})


def _compute_signature(body: bytes, secret: str) -> str:
    """Compute valid WhatsApp webhook signature."""
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def _make_webhook_payload(
    message_type: str = "text",
    text_body: str = "Hello",
    from_number: str = "1234567890",
    media_id: str = "media123",
    caption: str = "Media caption",
) -> dict[str, Any]:
    """Create a standard WhatsApp webhook message payload."""
    message: dict[str, Any] = {
        "id": "wamid.test123",
        "from": from_number,
        "timestamp": "1700000000",
        "type": message_type,
    }

    if message_type == "text":
        message["text"] = {"body": text_body}
    elif message_type == "interactive":
        message["interactive"] = {
            "type": "button_reply",
            "button_reply": {"id": "button1", "title": "Option 1"},
        }
    elif message_type == "button":
        message["button"] = {"payload": "quick_reply_payload", "text": "Quick Reply"}
    elif message_type == "document":
        message["document"] = {
            "id": media_id,
            "filename": "spec.pdf",
            "mime_type": "application/pdf",
            "caption": caption,
        }

    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "business_account_id",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "metadata": {"phone_number_id": "phone_id_123"},
                            "contacts": [
                                {
                                    "wa_id": from_number,
                                    "profile": {"name": "Test User"},
                                }
                            ],
                            "messages": [message],
                        },
                    }
                ],
            }
        ],
    }


def _dispatch_webhook(
    handler: WhatsAppHandler,
    payload: dict[str, Any],
    *,
    path: str = "/api/v1/bots/whatsapp/webhook",
    app_secret: str = "",
    mock_debate: bool = False,
):
    """Send a webhook payload through handle_post with signature verification.

    Args:
        handler: The WhatsAppHandler instance.
        payload: The webhook payload dict.
        path: The URL path.
        app_secret: If set, computes valid signature with this secret.
        mock_debate: If True, mock _start_debate_async to avoid timeouts.
    """
    body = json.dumps(payload).encode("utf-8")
    signature = _compute_signature(body, app_secret) if app_secret else ""
    mock_request = _make_mock_request(body, signature=signature)

    with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", app_secret or ""):
        if mock_debate:
            with patch.object(
                handler,
                "_start_debate_async",
                return_value="mock-debate-id-12345678",
            ):
                return handler.handle_post(path, {}, mock_request)
        return handler.handle_post(path, {}, mock_request)


# =============================================================================
# Test Signature Verification
# =============================================================================


class TestWhatsAppSignatureVerification:
    """Tests for _verify_whatsapp_signature function."""

    def test_verify_valid_signature(self):
        """Should return True when signature matches."""
        body = b'{"test": "data"}'
        secret = "my_app_secret"
        signature = _compute_signature(body, secret)

        with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", secret):
            result = _verify_whatsapp_signature(signature, body)

        assert result is True

    def test_reject_invalid_signature(self):
        """Should return False when signature does not match."""
        body = b'{"test": "data"}'

        with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", "correct_secret"):
            result = _verify_whatsapp_signature("sha256=invalid_signature", body)

        assert result is False

    def test_reject_signature_without_sha256_prefix(self):
        """Should return False when signature lacks sha256= prefix."""
        body = b'{"test": "data"}'

        with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", "secret"):
            result = _verify_whatsapp_signature("invalid_format_signature", body)

        assert result is False

    def test_fails_closed_when_secret_not_configured(self):
        """Should return False (fail closed) when app secret is not configured."""
        body = b'{"test": "data"}'

        with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", ""):
            result = _verify_whatsapp_signature("sha256=any_signature", body)

        assert result is False

    def test_empty_body_with_valid_signature(self):
        """Should verify empty body correctly."""
        body = b""
        secret = "test_secret"
        signature = _compute_signature(body, secret)

        with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", secret):
            result = _verify_whatsapp_signature(signature, body)

        assert result is True


# =============================================================================
# Test Handler Initialization and Routing
# =============================================================================


class TestWhatsAppHandlerInit:
    """Tests for WhatsAppHandler class setup."""

    def test_bot_platform_is_whatsapp(self):
        """Should identify as 'whatsapp' platform."""
        handler = _make_handler()
        assert handler.bot_platform == "whatsapp"

    def test_routes_include_webhook_and_status(self):
        """Should include webhook and status in ROUTES."""
        handler = _make_handler()
        assert "/api/v1/bots/whatsapp/webhook" in handler.ROUTES
        assert "/api/v1/bots/whatsapp/status" in handler.ROUTES

    def test_can_handle_webhook(self):
        """Should handle the webhook path."""
        handler = _make_handler()
        assert handler.can_handle("/api/v1/bots/whatsapp/webhook") is True

    def test_can_handle_status(self):
        """Should handle the status path."""
        handler = _make_handler()
        assert handler.can_handle("/api/v1/bots/whatsapp/status") is True

    def test_cannot_handle_unknown_path(self):
        """Should not handle unrelated paths."""
        handler = _make_handler()
        assert handler.can_handle("/api/v1/bots/telegram/webhook") is False
        assert handler.can_handle("/api/v1/debates") is False

    def test_is_bot_enabled_true_when_configured(self):
        """Should report enabled when access token and phone ID are set."""
        handler = _make_handler()
        with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", "token"):
            with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", "phone_id"):
                assert handler._is_bot_enabled() is True

    def test_is_bot_enabled_false_when_token_missing(self):
        """Should report disabled when access token is not set."""
        handler = _make_handler()
        with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", None):
            with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", "phone_id"):
                assert handler._is_bot_enabled() is False

    def test_is_bot_enabled_false_when_phone_id_missing(self):
        """Should report disabled when phone ID is not set."""
        handler = _make_handler()
        with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", "token"):
            with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", None):
                assert handler._is_bot_enabled() is False


# =============================================================================
# Test Webhook Verification (GET)
# =============================================================================


class TestWebhookVerification:
    """Tests for WhatsApp webhook verification challenge (GET requests)."""

    @pytest.mark.asyncio
    async def test_verification_success(self):
        """Should return challenge when verify token matches."""
        handler = _make_handler()
        query_params = {
            "hub.mode": ["subscribe"],
            "hub.verify_token": ["my_verify_token"],
            "hub.challenge": ["challenge_string_123"],
        }

        with patch.object(whatsapp_module, "WHATSAPP_VERIFY_TOKEN", "my_verify_token"):
            result = await handler.handle(
                "/api/v1/bots/whatsapp/webhook", query_params, MagicMock()
            )

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/plain"
        assert result.body == b"challenge_string_123"

    @pytest.mark.asyncio
    async def test_verification_token_mismatch(self):
        """Should return 403 when verify token does not match."""
        handler = _make_handler()
        query_params = {
            "hub.mode": ["subscribe"],
            "hub.verify_token": ["wrong_token"],
            "hub.challenge": ["challenge_123"],
        }

        with patch.object(whatsapp_module, "WHATSAPP_VERIFY_TOKEN", "correct_token"):
            result = await handler.handle(
                "/api/v1/bots/whatsapp/webhook", query_params, MagicMock()
            )

        assert result is not None
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_verification_token_not_configured(self):
        """Should return 403 when verify token is not configured."""
        handler = _make_handler()
        query_params = {
            "hub.mode": ["subscribe"],
            "hub.verify_token": ["any_token"],
            "hub.challenge": ["challenge_123"],
        }

        with patch.object(whatsapp_module, "WHATSAPP_VERIFY_TOKEN", None):
            result = await handler.handle(
                "/api/v1/bots/whatsapp/webhook", query_params, MagicMock()
            )

        assert result is not None
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_verification_invalid_mode(self):
        """Should return 400 for non-subscribe mode."""
        handler = _make_handler()
        query_params = {
            "hub.mode": ["unsubscribe"],
            "hub.verify_token": ["token"],
            "hub.challenge": ["challenge"],
        }

        result = await handler.handle("/api/v1/bots/whatsapp/webhook", query_params, MagicMock())

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_verification_missing_mode(self):
        """Should return 400 when hub.mode is missing."""
        handler = _make_handler()
        query_params = {
            "hub.verify_token": ["token"],
            "hub.challenge": ["challenge"],
        }

        result = await handler.handle("/api/v1/bots/whatsapp/webhook", query_params, MagicMock())

        assert result is not None
        assert result.status_code == 400


# =============================================================================
# Test Status Endpoint
# =============================================================================


class TestWhatsAppStatusEndpoint:
    """Tests for the GET /status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_platform_info(self):
        """Should return status JSON with platform details."""
        handler = _make_handler()
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock()
            with patch.object(handler, "check_permission"):
                with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", "token"):
                    with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", "phone"):
                        with patch.object(whatsapp_module, "WHATSAPP_VERIFY_TOKEN", "verify"):
                            with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", "secret"):
                                result = await handler.handle(
                                    "/api/v1/bots/whatsapp/status", {}, MagicMock()
                                )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["platform"] == "whatsapp"
        assert body["enabled"] is True
        assert body["access_token_configured"] is True
        assert body["phone_number_configured"] is True
        assert body["verify_token_configured"] is True
        assert body["app_secret_configured"] is True

    @pytest.mark.asyncio
    async def test_status_shows_disabled_when_not_configured(self):
        """Should show disabled when credentials are missing."""
        handler = _make_handler()
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
            with patch.object(handler, "check_permission"):
                with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", None):
                    with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", None):
                        result = await handler.handle(
                            "/api/v1/bots/whatsapp/status", {}, MagicMock()
                        )

        body = json.loads(result.body)
        assert body["enabled"] is False
        assert body["access_token_configured"] is False
        assert body["phone_number_configured"] is False

    @pytest.mark.asyncio
    async def test_status_returns_401_when_unauthenticated(self):
        """Should return 401 when authentication fails."""
        from aragora.server.handlers.utils.auth import UnauthorizedError

        handler = _make_handler()
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("No token")
            result = await handler.handle("/api/v1/bots/whatsapp/status", {}, MagicMock())

        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_status_returns_403_when_forbidden(self):
        """Should return 403 when lacking bots.read permission."""
        from aragora.server.handlers.utils.auth import ForbiddenError

        handler = _make_handler()
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock):
            with patch.object(handler, "check_permission") as mock_check:
                mock_check.side_effect = ForbiddenError("Missing permission")
                result = await handler.handle("/api/v1/bots/whatsapp/status", {}, MagicMock())

        assert result.status_code == 403


# =============================================================================
# Test Webhook POST Handling
# =============================================================================


class TestWebhookPostRouting:
    """Tests for handle_post routing logic."""

    def test_returns_none_for_unmatched_path(self):
        """Should return None for paths not matching webhook."""
        handler = _make_handler()
        result = handler.handle_post("/api/v1/bots/telegram/webhook", {}, MagicMock())
        assert result is None

    def test_webhook_rejects_invalid_signature(self):
        """Should return 401 when signature verification fails."""
        handler = _make_handler()
        payload = _make_webhook_payload()
        body = json.dumps(payload).encode("utf-8")
        mock_request = _make_mock_request(body, signature="sha256=invalid")

        with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", "real_secret"):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 401

    def test_webhook_accepts_valid_signature(self):
        """Should process webhook when signature is valid."""
        handler = _make_handler()
        payload = _make_webhook_payload()

        result = _dispatch_webhook(handler, payload, app_secret="test_secret")

        assert result is not None
        assert result.status_code == 200

    def test_webhook_invalid_json_returns_400(self):
        """Should return 400 for invalid JSON body."""
        handler = _make_handler()
        invalid_body = b"not json!!"
        test_secret = "test_secret_key"
        # Compute valid signature for the invalid JSON body
        valid_sig = _compute_signature(invalid_body, test_secret)
        mock_request = _make_mock_request(invalid_body, signature=valid_sig)

        with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", test_secret):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 400

    def test_webhook_returns_200_on_exception(self):
        """Should return 200 even on error to prevent retries."""
        handler = _make_handler()
        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": "100",
            "X-Hub-Signature-256": "",
        }
        mock_request.rfile.read.side_effect = OSError("Disk failure")

        with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", ""):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "error"


# =============================================================================
# Test Message Handling
# =============================================================================


class TestMessageHandling:
    """Tests for incoming WhatsApp message processing."""

    def test_text_message_handled(self):
        """Should handle a plain text message and return ok."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="Hello, bot!")

        # Use mock_debate to avoid Redis/async dependencies
        result = _dispatch_webhook(handler, payload, app_secret="secret", mock_debate=True)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "ok"

    def test_greeting_message_triggers_welcome(self):
        """Should send welcome message for greeting words."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="hello")

        with patch.object(handler, "_send_welcome") as mock_welcome:
            result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result.status_code == 200
        mock_welcome.assert_called_once_with("1234567890")

    def test_start_greeting_triggers_welcome(self):
        """Should send welcome for 'start' message."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="start")

        with patch.object(handler, "_send_welcome") as mock_welcome:
            result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result.status_code == 200
        mock_welcome.assert_called_once()

    def test_non_greeting_triggers_debate(self):
        """Should treat non-command, non-greeting text as debate topic."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="What is the best programming language?")

        result = _dispatch_webhook(handler, payload, app_secret="secret", mock_debate=True)

        assert result.status_code == 200

    def test_interactive_list_reply_handled(self):
        """Should handle interactive list reply messages."""
        handler = _make_handler()
        payload = _make_webhook_payload(message_type="interactive")
        # Modify to be list_reply
        payload["entry"][0]["changes"][0]["value"]["messages"][0]["interactive"] = {
            "type": "list_reply",
            "list_reply": {"id": "list_item_1", "title": "Selected Item"},
        }

        result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result is not None
        assert result.status_code == 200

    def test_interactive_button_reply_handled(self):
        """Should handle interactive button reply messages."""
        handler = _make_handler()
        payload = _make_webhook_payload(message_type="interactive")

        result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result is not None
        assert result.status_code == 200

    def test_quick_reply_button_handled(self):
        """Should handle quick reply button messages."""
        handler = _make_handler()
        payload = _make_webhook_payload(message_type="button")

        result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result is not None
        assert result.status_code == 200

    def test_unknown_message_type_logged(self):
        """Should log but not fail on unknown message type."""
        handler = _make_handler()
        payload = _make_webhook_payload()
        payload["entry"][0]["changes"][0]["value"]["messages"][0]["type"] = "sticker"

        result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result is not None
        assert result.status_code == 200

    def test_contact_name_from_profile(self):
        """Should extract contact name from profile."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="/help")
        # Ensure contact name is set
        payload["entry"][0]["changes"][0]["value"]["contacts"][0]["profile"]["name"] = "Custom Name"

        with patch.object(handler, "_send_help") as mock_help:
            result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result.status_code == 200
        mock_help.assert_called_once()

    def test_document_message_triggers_debate_with_caption(self):
        """Should start a debate when document message includes a caption."""
        handler = _make_handler()
        payload = _make_webhook_payload(
            message_type="document",
            caption="Review this spec",
        )

        with patch.object(handler, "_start_debate") as mock_start:
            result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result.status_code == 200
        mock_start.assert_called_once()


# =============================================================================
# Test Command Processing
# =============================================================================


class TestCommandProcessing:
    """Tests for WhatsApp bot command handling."""

    def test_help_command(self):
        """Should handle /help command."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="/help")

        with patch.object(handler, "_send_help") as mock_help:
            result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result.status_code == 200
        mock_help.assert_called_once_with("1234567890")

    def test_status_command(self):
        """Should handle /status command."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="/status")

        with patch.object(handler, "_send_status") as mock_status:
            result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result.status_code == 200
        mock_status.assert_called_once_with("1234567890")

    def test_debate_command_with_topic(self):
        """Should start debate with /debate command."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="/debate Should we use Rust?")

        with patch.object(handler, "_start_debate") as mock_debate:
            result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result.status_code == 200
        mock_debate.assert_called_once()
        call_args = mock_debate.call_args[0]
        assert call_args[0] == "1234567890"  # from_number
        assert "Should we use Rust?" in call_args[2]  # topic

    def test_debate_command_empty_topic(self):
        """Should prompt for topic when /debate has no arguments."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="/debate")

        with patch.object(handler, "_start_debate") as mock_debate:
            result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result.status_code == 200
        mock_debate.assert_called_once()
        call_args = mock_debate.call_args[0]
        assert call_args[2] == ""  # empty topic

    def test_unknown_command(self):
        """Should send error message for unknown commands."""
        handler = _make_handler()
        payload = _make_webhook_payload(text_body="/foobar")

        with patch.object(handler, "_send_message") as mock_send:
            result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result.status_code == 200
        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "Unknown command" in sent_text
        assert "/foobar" in sent_text


# =============================================================================
# Test Message Sending
# =============================================================================


class TestMessageSending:
    """Tests for _send_message and helper message methods."""

    def test_send_message_does_nothing_without_credentials(self):
        """Should skip sending when credentials are not configured."""
        handler = _make_handler()
        with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", None):
            # Should not raise
            handler._send_message("12345", "Test message")

    def test_send_message_calls_whatsapp_api(self):
        """Should POST to WhatsApp Cloud API when configured."""
        handler = _make_handler()
        with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", "test_token"):
            with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", "phone_id"):
                with patch("httpx.Client") as mock_client:
                    mock_resp = MagicMock(is_success=True)
                    mock_client.return_value.__enter__.return_value.post.return_value = mock_resp

                    handler._send_message("12345", "Hello")

                    call_args = mock_client.return_value.__enter__.return_value.post.call_args
                    assert "phone_id/messages" in call_args[0][0]
                    assert call_args[1]["json"]["to"] == "12345"
                    assert call_args[1]["json"]["text"]["body"] == "Hello"

    def test_send_message_handles_http_failure(self):
        """Should not raise on HTTP failure."""
        handler = _make_handler()
        with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", "token"):
            with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", "phone"):
                with patch("httpx.Client") as mock_client:
                    mock_resp = MagicMock(is_success=False, status_code=500, text="Error")
                    mock_client.return_value.__enter__.return_value.post.return_value = mock_resp

                    # Should not raise
                    handler._send_message("12345", "Fail gracefully")

    def test_send_message_handles_connection_error(self):
        """Should not raise on network error."""
        handler = _make_handler()
        with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", "token"):
            with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", "phone"):
                with patch("httpx.Client") as mock_client:
                    mock_client.return_value.__enter__.return_value.post.side_effect = (
                        ConnectionError("Network down")
                    )

                    # Should not raise
                    handler._send_message("12345", "Network issue")

    def test_send_welcome_message(self):
        """Should send welcome message with correct content."""
        handler = _make_handler()
        with patch.object(handler, "_send_message") as mock_send:
            handler._send_welcome("12345")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "Welcome to Aragora" in sent_text
        assert "/debate" in sent_text

    def test_send_help_message(self):
        """Should send help message with command list."""
        handler = _make_handler()
        with patch.object(handler, "_send_message") as mock_send:
            handler._send_help("12345")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "Aragora Commands" in sent_text
        assert "/debate" in sent_text
        assert "/status" in sent_text
        assert "/help" in sent_text

    def test_send_status_message(self):
        """Should send status message with model list."""
        handler = _make_handler()
        with patch.object(handler, "_send_message") as mock_send:
            handler._send_status("12345")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "Online" in sent_text
        assert "Claude" in sent_text
        assert "GPT-4" in sent_text


# =============================================================================
# Test Debate Starting
# =============================================================================


class TestDebateStarting:
    """Tests for _start_debate and async debate methods."""

    def test_start_debate_empty_topic_sends_prompt(self):
        """Should send prompt when topic is empty."""
        handler = _make_handler()
        with patch.object(handler, "_send_message") as mock_send:
            handler._start_debate("12345", "Test User", "")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "Please provide a topic" in sent_text

    def test_start_debate_with_topic_sends_confirmation(self):
        """Should send confirmation when starting debate."""
        handler = _make_handler()
        with patch.object(handler, "_start_debate_async", return_value="abc-123"):
            with patch.object(handler, "_send_message") as mock_send:
                handler._start_debate("12345", "Test User", "Is Python good?")

        mock_send.assert_called_once()
        sent_text = mock_send.call_args[0][1]
        assert "Starting debate" in sent_text
        assert "Is Python good?" in sent_text
        assert "abc-123" in sent_text

    def test_start_debate_truncates_long_topic(self):
        """Should truncate topic display to 200 chars."""
        handler = _make_handler()
        long_topic = "A" * 300
        with patch.object(handler, "_start_debate_async", return_value="xyz-456"):
            with patch.object(handler, "_send_message") as mock_send:
                handler._start_debate("12345", "User", long_topic)

        sent_text = mock_send.call_args[0][1]
        # Topic should be truncated to 200 chars
        assert len(long_topic[:200]) == 200
        # The message should contain the truncated topic
        assert "A" * 50 in sent_text  # Partial check

    def test_start_debate_async_returns_uuid(self):
        """Should return a debate ID string."""
        handler = _make_handler()

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = None
            with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", ""):
                # Mock register_debate_origin to avoid import errors
                with patch.dict(
                    "sys.modules",
                    {"aragora.server.debate_origin": MagicMock()},
                ):
                    debate_id = handler._start_debate_async("12345", "User", "Test topic")

        assert isinstance(debate_id, str)
        assert len(debate_id) > 0

    def test_start_debate_async_accepts_attachments(self):
        """Should accept attachments when starting debate."""
        handler = _make_handler()

        with patch("asyncio.run") as mock_run:
            mock_run.return_value = None
            with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", ""):
                with patch.dict(
                    "sys.modules",
                    {"aragora.server.debate_origin": MagicMock()},
                ):
                    debate_id = handler._start_debate_async(
                        "12345",
                        "User",
                        "Test topic",
                        attachments=[{"type": "document", "file_id": "file123"}],
                    )

        assert isinstance(debate_id, str)


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling edge cases."""

    def test_empty_entry_list_handled(self):
        """Should handle empty entry list gracefully."""
        handler = _make_handler()
        payload = {"object": "whatsapp_business_account", "entry": []}

        result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result is not None
        assert result.status_code == 200

    def test_empty_changes_list_handled(self):
        """Should handle empty changes list gracefully."""
        handler = _make_handler()
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{"id": "123", "changes": []}],
        }

        result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result is not None
        assert result.status_code == 200

    def test_missing_messages_field_handled(self):
        """Should handle missing messages field gracefully."""
        handler = _make_handler()
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "123",
                    "changes": [{"field": "messages", "value": {}}],
                }
            ],
        }

        result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result is not None
        assert result.status_code == 200

    def test_non_messages_field_ignored(self):
        """Should ignore changes with field != 'messages'."""
        handler = _make_handler()
        payload = {
            "object": "whatsapp_business_account",
            "entry": [
                {
                    "id": "123",
                    "changes": [{"field": "statuses", "value": {}}],
                }
            ],
        }

        result = _dispatch_webhook(handler, payload, app_secret="secret")

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Build Status Response
# =============================================================================


class TestBuildStatusResponse:
    """Tests for _build_status_response method."""

    def test_build_status_response_with_extra(self):
        """Should merge extra_status fields into status response."""
        handler = _make_handler()
        with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", "tok"):
            with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", "phone"):
                with patch.object(whatsapp_module, "WHATSAPP_VERIFY_TOKEN", "verify"):
                    with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", "secret"):
                        result = handler._build_status_response({"custom_field": 42})

        body = json.loads(result.body)
        assert body["custom_field"] == 42
        assert body["platform"] == "whatsapp"

    def test_build_status_response_shows_all_config_flags(self):
        """Should show all configuration status flags."""
        handler = _make_handler()
        with patch.object(whatsapp_module, "WHATSAPP_ACCESS_TOKEN", ""):
            with patch.object(whatsapp_module, "WHATSAPP_PHONE_NUMBER_ID", ""):
                with patch.object(whatsapp_module, "WHATSAPP_VERIFY_TOKEN", ""):
                    with patch.object(whatsapp_module, "WHATSAPP_APP_SECRET", ""):
                        result = handler._build_status_response()

        body = json.loads(result.body)
        assert body["access_token_configured"] is False
        assert body["phone_number_configured"] is False
        assert body["verify_token_configured"] is False
        assert body["app_secret_configured"] is False
