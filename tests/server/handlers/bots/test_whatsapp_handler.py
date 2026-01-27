"""Tests for WhatsApp bot handler."""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.whatsapp import (
    WhatsAppHandler,
    _verify_whatsapp_signature,
)


# =============================================================================
# Test Signature Verification
# =============================================================================


class TestWhatsAppSignatureVerification:
    """Tests for WhatsApp webhook signature verification."""

    def test_verify_signature_no_secret(self):
        """Should pass when no app secret is configured."""
        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", ""):
            result = _verify_whatsapp_signature("sha256=anything", b"body")
        assert result is True

    def test_verify_signature_valid(self):
        """Should verify valid signature."""
        secret = "test_secret_123"
        body = b'{"test": "data"}'
        expected_sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", secret):
            result = _verify_whatsapp_signature(f"sha256={expected_sig}", body)
        assert result is True

    def test_verify_signature_invalid(self):
        """Should reject invalid signature."""
        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", "secret"):
            result = _verify_whatsapp_signature("sha256=invalid", b"body")
        assert result is False

    def test_verify_signature_missing_prefix(self):
        """Should reject signature without sha256= prefix."""
        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", "secret"):
            result = _verify_whatsapp_signature("noprefixhash", b"body")
        assert result is False


# =============================================================================
# Test Handler Initialization
# =============================================================================


class TestWhatsAppHandlerInit:
    """Tests for WhatsApp handler initialization."""

    def test_handler_routes(self):
        """Should define correct routes."""
        handler = WhatsAppHandler({})
        assert "/api/v1/bots/whatsapp/webhook" in handler.ROUTES
        assert "/api/v1/bots/whatsapp/status" in handler.ROUTES

    def test_can_handle_webhook_route(self):
        """Should handle webhook route."""
        handler = WhatsAppHandler({})
        assert handler.can_handle("/api/v1/bots/whatsapp/webhook") is True

    def test_can_handle_status_route(self):
        """Should handle status route."""
        handler = WhatsAppHandler({})
        assert handler.can_handle("/api/v1/bots/whatsapp/status") is True


# =============================================================================
# Test Status Endpoint
# =============================================================================


class TestWhatsAppStatus:
    """Tests for WhatsApp status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Should return status information."""
        handler = WhatsAppHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=["bots:read"])
            with patch.object(handler, "check_permission"):
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/whatsapp/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["platform"] == "whatsapp"
        assert "enabled" in body
        assert "access_token_configured" in body
        assert "phone_number_configured" in body


# =============================================================================
# Test Webhook Verification Challenge
# =============================================================================


class TestWhatsAppVerification:
    """Tests for WhatsApp webhook verification challenge."""

    @pytest.mark.asyncio
    async def test_verification_challenge_success(self):
        """Should respond to verification challenge correctly."""
        handler = WhatsAppHandler({})

        query_params = {
            "hub.mode": ["subscribe"],
            "hub.verify_token": ["test_verify_token"],
            "hub.challenge": ["challenge_string_123"],
        }

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_VERIFY_TOKEN",
            "test_verify_token",
        ):
            mock_handler = MagicMock()
            result = await handler.handle(
                "/api/v1/bots/whatsapp/webhook", query_params, mock_handler
            )

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/plain"
        assert result.body == b"challenge_string_123"

    @pytest.mark.asyncio
    async def test_verification_challenge_invalid_token(self):
        """Should reject verification with invalid token."""
        handler = WhatsAppHandler({})

        query_params = {
            "hub.mode": ["subscribe"],
            "hub.verify_token": ["wrong_token"],
            "hub.challenge": ["challenge_string"],
        }

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_VERIFY_TOKEN",
            "correct_token",
        ):
            mock_handler = MagicMock()
            result = await handler.handle(
                "/api/v1/bots/whatsapp/webhook", query_params, mock_handler
            )

        assert result is not None
        assert result.status_code == 403


# =============================================================================
# Test Webhook Message Handling
# =============================================================================


class TestWhatsAppWebhook:
    """Tests for WhatsApp webhook message handling."""

    def test_handle_text_message(self):
        """Should handle incoming text message."""
        handler = WhatsAppHandler({})

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "metadata": {"phone_number_id": "123456"},
                                "contacts": [
                                    {"wa_id": "1234567890", "profile": {"name": "Test User"}}
                                ],
                                "messages": [
                                    {
                                        "type": "text",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "timestamp": "1234567890",
                                        "text": {"body": "Hello bot"},
                                    }
                                ],
                            },
                        }
                    ]
                }
            ]
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(payload))),
            "X-Hub-Signature-256": "",
        }
        mock_request.rfile.read.return_value = json.dumps(payload).encode()

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", ""):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
                result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "ok"

    def test_handle_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        handler = WhatsAppHandler({})

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": "15",
            "X-Hub-Signature-256": "",
        }
        mock_request.rfile.read.return_value = b"not valid json"

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", ""):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 400

    def test_handle_interactive_message(self):
        """Should handle interactive message response."""
        handler = WhatsAppHandler({})

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "metadata": {"phone_number_id": "123456"},
                                "messages": [
                                    {
                                        "type": "interactive",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "interactive": {
                                            "type": "button_reply",
                                            "button_reply": {"id": "btn1", "title": "Yes"},
                                        },
                                    }
                                ],
                            },
                        }
                    ]
                }
            ]
        }

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(payload))),
            "X-Hub-Signature-256": "",
        }
        mock_request.rfile.read.return_value = json.dumps(payload).encode()

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", ""):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Signature Rejection
# =============================================================================


class TestWhatsAppSignatureRejection:
    """Tests for WhatsApp signature rejection."""

    def test_reject_invalid_signature(self):
        """Should reject requests with invalid signatures."""
        handler = WhatsAppHandler({})

        payload = {"entry": []}

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(payload))),
            "X-Hub-Signature-256": "sha256=invalid",
        }
        mock_request.rfile.read.return_value = json.dumps(payload).encode()

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", "secret"):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 401
