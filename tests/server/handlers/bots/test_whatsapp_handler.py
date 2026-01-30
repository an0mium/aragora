"""Tests for WhatsApp bot handler.

Comprehensive test coverage for WhatsApp Cloud API webhook handling:
- Signature verification
- Webhook verification challenge
- Message handling (text, interactive, button)
- Status endpoint with RBAC
- Error handling and edge cases
"""

import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.whatsapp import (
    WhatsAppHandler,
    _verify_whatsapp_signature,
)


# Test secrets for webhook signature verification
TEST_WHATSAPP_SECRET = "test_whatsapp_secret_12345"


@pytest.fixture
def whatsapp_secret():
    """Fixture that patches WHATSAPP_APP_SECRET for tests."""
    with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET):
        yield TEST_WHATSAPP_SECRET


def _compute_signature(body: bytes, secret: str = TEST_WHATSAPP_SECRET) -> str:
    """Compute valid HMAC-SHA256 signature for test payloads."""
    sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={sig}"


def _create_signed_request(payload: dict, secret: str = TEST_WHATSAPP_SECRET) -> MagicMock:
    """Create a mock request with valid signature for webhook tests."""
    body = json.dumps(payload).encode()
    mock_request = MagicMock()
    mock_request.headers = {
        "Content-Length": str(len(body)),
        "X-Hub-Signature-256": _compute_signature(body, secret),
    }
    mock_request.rfile.read.return_value = body
    return mock_request


# =============================================================================
# Test Signature Verification
# =============================================================================


class TestWhatsAppSignatureVerification:
    """Tests for WhatsApp webhook signature verification."""

    def test_verify_signature_no_secret(self):
        """Should reject when no app secret is configured (fail-closed security)."""
        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", ""):
            result = _verify_whatsapp_signature("sha256=anything", b"body")
        assert result is False  # Fail closed - reject unverifiable requests

    def test_verify_signature_none_secret(self):
        """Should reject when app secret is None (fail-closed security)."""
        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", None):
            result = _verify_whatsapp_signature("sha256=anything", b"body")
        assert result is False  # Fail closed - reject unverifiable requests

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

    def test_verify_signature_empty_signature(self):
        """Should reject empty signature."""
        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", "secret"):
            result = _verify_whatsapp_signature("", b"body")
        assert result is False

    def test_verify_signature_timing_safe_comparison(self):
        """Should use timing-safe comparison to prevent timing attacks."""
        secret = "test_secret"
        body = b"test body"
        expected_sig = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        # Create a signature that differs only in last character
        almost_correct = expected_sig[:-1] + ("a" if expected_sig[-1] != "a" else "b")

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", secret):
            result = _verify_whatsapp_signature(f"sha256={almost_correct}", body)
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

    def test_cannot_handle_unknown_route(self):
        """Should not handle unknown routes."""
        handler = WhatsAppHandler({})
        assert handler.can_handle("/api/v1/bots/whatsapp/unknown") is False
        assert handler.can_handle("/api/v1/bots/telegram/webhook") is False

    def test_bot_platform_identifier(self):
        """Should have correct platform identifier."""
        handler = WhatsAppHandler({})
        assert handler.bot_platform == "whatsapp"

    def test_is_bot_enabled_when_configured(self):
        """Should report enabled when both tokens are configured."""
        handler = WhatsAppHandler({})
        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", "token"):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123"):
                assert handler._is_bot_enabled() is True

    def test_is_bot_disabled_without_token(self):
        """Should report disabled when access token is missing."""
        handler = WhatsAppHandler({})
        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123"):
                assert handler._is_bot_enabled() is False

    def test_is_bot_disabled_without_phone_number(self):
        """Should report disabled when phone number ID is missing."""
        handler = WhatsAppHandler({})
        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", "token"):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_PHONE_NUMBER_ID", ""):
                assert handler._is_bot_enabled() is False


# =============================================================================
# Test Status Endpoint
# =============================================================================


class TestWhatsAppStatus:
    """Tests for WhatsApp status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status_authenticated(self):
        """Should return status information when authenticated."""
        handler = WhatsAppHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=["bots.read"])
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
        assert "verify_token_configured" in body
        assert "app_secret_configured" in body

    @pytest.mark.asyncio
    async def test_get_status_unauthorized(self):
        """Should return 401 when not authenticated."""
        handler = WhatsAppHandler({})

        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("No token")
            mock_handler = MagicMock()
            result = await handler.handle("/api/v1/bots/whatsapp/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_get_status_forbidden(self):
        """Should return 403 when lacking permission."""
        handler = WhatsAppHandler({})

        from aragora.server.handlers.utils.auth import ForbiddenError

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=[])
            with patch.object(handler, "check_permission") as mock_check:
                mock_check.side_effect = ForbiddenError("Missing bots.read permission")
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/whatsapp/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 403


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

    @pytest.mark.asyncio
    async def test_verification_no_verify_token_configured(self):
        """Should return 403 when verify token is not configured."""
        handler = WhatsAppHandler({})

        query_params = {
            "hub.mode": ["subscribe"],
            "hub.verify_token": ["any_token"],
            "hub.challenge": ["challenge"],
        }

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_VERIFY_TOKEN", ""):
            mock_handler = MagicMock()
            result = await handler.handle(
                "/api/v1/bots/whatsapp/webhook", query_params, mock_handler
            )

        assert result is not None
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_verification_invalid_mode(self):
        """Should reject verification with invalid mode."""
        handler = WhatsAppHandler({})

        query_params = {
            "hub.mode": ["unsubscribe"],
            "hub.verify_token": ["token"],
            "hub.challenge": ["challenge"],
        }

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_VERIFY_TOKEN", "token"):
            mock_handler = MagicMock()
            result = await handler.handle(
                "/api/v1/bots/whatsapp/webhook", query_params, mock_handler
            )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_verification_missing_params(self):
        """Should handle missing verification parameters."""
        handler = WhatsAppHandler({})

        # Missing hub.mode
        query_params = {
            "hub.verify_token": ["token"],
            "hub.challenge": ["challenge"],
        }

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_VERIFY_TOKEN", "token"):
            mock_handler = MagicMock()
            result = await handler.handle(
                "/api/v1/bots/whatsapp/webhook", query_params, mock_handler
            )

        assert result is not None
        assert result.status_code == 400


# =============================================================================
# Test Webhook Message Handling
# =============================================================================


class TestWhatsAppWebhook:
    """Tests for WhatsApp webhook message handling."""

    @pytest.fixture(autouse=True)
    def setup_signed_requests(self):
        """Auto-patch signature verification for all webhook tests.

        This allows tests to focus on message handling logic without
        needing to generate valid HMAC signatures for each test payload.
        """
        with patch(
            "aragora.server.handlers.bots.whatsapp._verify_whatsapp_signature",
            return_value=True,
        ):
            yield

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

        mock_request = _create_signed_request(payload)

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
                with patch.object(handler, "_start_debate_async", return_value="debate-123"):
                    result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "ok"

    def test_handle_help_command(self):
        """Should handle /help command."""
        handler = WhatsAppHandler({})

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "metadata": {"phone_number_id": "123456"},
                                "contacts": [{"wa_id": "1234567890", "profile": {"name": "User"}}],
                                "messages": [
                                    {
                                        "type": "text",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "text": {"body": "/help"},
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
                with patch.object(handler, "_send_message") as mock_send:
                    result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        # Verify help message was sent
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert "1234567890" in args
        assert "Commands" in args[1] or "help" in args[1].lower()

    def test_handle_debate_command(self):
        """Should handle /debate command."""
        handler = WhatsAppHandler({})

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "metadata": {"phone_number_id": "123456"},
                                "contacts": [{"wa_id": "1234567890", "profile": {"name": "User"}}],
                                "messages": [
                                    {
                                        "type": "text",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "text": {"body": "/debate Should we use microservices?"},
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
                with patch.object(handler, "_start_debate", return_value=None) as mock_debate:
                    result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        mock_debate.assert_called_once()
        call_args = mock_debate.call_args[0]
        assert "Should we use microservices?" in call_args[2]

    def test_handle_status_command(self):
        """Should handle /status command."""
        handler = WhatsAppHandler({})

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "metadata": {"phone_number_id": "123456"},
                                "contacts": [{"wa_id": "1234567890", "profile": {"name": "User"}}],
                                "messages": [
                                    {
                                        "type": "text",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "text": {"body": "/status"},
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
                with patch.object(handler, "_send_message") as mock_send:
                    result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert "Status" in args[1] or "Online" in args[1]

    def test_handle_greeting_message(self):
        """Should handle greeting messages (hi, hello, hey)."""
        handler = WhatsAppHandler({})

        for greeting in ["hi", "hello", "hey", "start"]:
            payload = {
                "entry": [
                    {
                        "changes": [
                            {
                                "field": "messages",
                                "value": {
                                    "metadata": {"phone_number_id": "123456"},
                                    "contacts": [
                                        {"wa_id": "1234567890", "profile": {"name": "User"}}
                                    ],
                                    "messages": [
                                        {
                                            "type": "text",
                                            "from": "1234567890",
                                            "id": "msg123",
                                            "text": {"body": greeting},
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

            with patch(
                "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
            ):
                with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
                    with patch.object(handler, "_send_message") as mock_send:
                        result = handler.handle_post(
                            "/api/v1/bots/whatsapp/webhook", {}, mock_request
                        )

            assert result is not None
            assert result.status_code == 200

    def test_handle_unknown_command(self):
        """Should handle unknown commands."""
        handler = WhatsAppHandler({})

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "metadata": {"phone_number_id": "123456"},
                                "contacts": [{"wa_id": "1234567890", "profile": {"name": "User"}}],
                                "messages": [
                                    {
                                        "type": "text",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "text": {"body": "/unknowncommand"},
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
                with patch.object(handler, "_send_message") as mock_send:
                    result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert "Unknown command" in args[1]

    def test_handle_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        handler = WhatsAppHandler({})

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": "15",
            "X-Hub-Signature-256": "",
        }
        mock_request.rfile.read.return_value = b"not valid json"

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 400

    def test_handle_interactive_list_reply(self):
        """Should handle interactive list reply message."""
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
                                            "type": "list_reply",
                                            "list_reply": {"id": "option1", "title": "Option 1"},
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_interactive_button_reply(self):
        """Should handle interactive button reply message."""
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_button_message(self):
        """Should handle quick reply button message."""
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
                                        "type": "button",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "button": {"payload": "vote_yes", "text": "Yes"},
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_unhandled_message_type(self):
        """Should handle unrecognized message types gracefully."""
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
                                        "type": "sticker",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "sticker": {"id": "sticker123"},
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_empty_payload(self):
        """Should handle empty payload gracefully."""
        handler = WhatsAppHandler({})

        payload = {"entry": []}

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": str(len(json.dumps(payload))),
            "X-Hub-Signature-256": "",
        }
        mock_request.rfile.read.return_value = json.dumps(payload).encode()

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "ok"

    def test_handle_non_message_field(self):
        """Should ignore non-message field changes."""
        handler = WhatsAppHandler({})

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "field": "status",
                            "value": {"status": "read"},
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
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
            with patch.object(handler, "_audit_webhook_auth_failure") as mock_audit:
                result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 401
        mock_audit.assert_called_once_with("signature")


# =============================================================================
# Test Message Sending
# =============================================================================


class TestWhatsAppMessageSending:
    """Tests for WhatsApp message sending functionality."""

    def test_send_message_without_credentials(self):
        """Should not attempt to send when credentials are missing."""
        handler = WhatsAppHandler({})

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_PHONE_NUMBER_ID", ""):
                # Should not raise, just log warning
                handler._send_message("1234567890", "Test message")

    def test_send_message_with_credentials(self):
        """Should send message when credentials are configured."""
        handler = WhatsAppHandler({})

        with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", "token"):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_PHONE_NUMBER_ID", "123"):
                with patch("httpx.Client") as mock_client:
                    mock_response = MagicMock()
                    mock_response.is_success = True
                    mock_client.return_value.__enter__.return_value.post.return_value = (
                        mock_response
                    )

                    handler._send_message("1234567890", "Test message")

                    # Verify the API was called
                    mock_client.return_value.__enter__.return_value.post.assert_called_once()


# =============================================================================
# Test Debate Starting
# =============================================================================


class TestWhatsAppDebateStarting:
    """Tests for WhatsApp debate initiation."""

    def test_start_debate_empty_topic(self):
        """Should prompt for topic when topic is empty."""
        handler = WhatsAppHandler({})

        with patch.object(handler, "_send_message") as mock_send:
            handler._start_debate("1234567890", "User", "")

        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert "provide a topic" in args[1].lower()

    def test_start_debate_with_topic(self):
        """Should start debate with provided topic."""
        handler = WhatsAppHandler({})

        with patch.object(handler, "_start_debate_async", return_value="debate-123") as mock_async:
            with patch.object(handler, "_send_message") as mock_send:
                handler._start_debate("1234567890", "User", "Is AI good?")

        mock_async.assert_called_once()
        mock_send.assert_called_once()
        args = mock_send.call_args[0]
        assert "Starting debate" in args[1]


# =============================================================================
# Test Error Handling
# =============================================================================


class TestWhatsAppErrorHandling:
    """Tests for WhatsApp webhook error handling."""

    @pytest.fixture(autouse=True)
    def setup_signed_requests(self):
        """Auto-patch signature verification for error handling tests."""
        with patch(
            "aragora.server.handlers.bots.whatsapp._verify_whatsapp_signature",
            return_value=True,
        ):
            yield

    def test_handle_exception_returns_200(self):
        """Should return 200 on exception to prevent retries."""
        handler = WhatsAppHandler({})

        mock_request = MagicMock()
        mock_request.headers = {
            "Content-Length": "100",
            "X-Hub-Signature-256": "",
        }
        # Cause exception during body read
        mock_request.rfile.read.side_effect = Exception("Read error")

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        # Should return 200 with error status to prevent webhook retries
        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["status"] == "error"

    def test_handle_post_unknown_path(self):
        """Should return None for unknown paths."""
        handler = WhatsAppHandler({})

        mock_request = MagicMock()
        result = handler.handle_post("/api/v1/bots/whatsapp/unknown", {}, mock_request)

        assert result is None


# =============================================================================
# Test Contact Name Resolution
# =============================================================================


class TestWhatsAppContactResolution:
    """Tests for WhatsApp contact name resolution."""

    @pytest.fixture(autouse=True)
    def setup_signed_requests(self):
        """Auto-patch signature verification for contact resolution tests."""
        with patch(
            "aragora.server.handlers.bots.whatsapp._verify_whatsapp_signature",
            return_value=True,
        ):
            yield

    def test_resolve_contact_name_from_contacts(self):
        """Should resolve contact name from contacts array."""
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
                                    {"wa_id": "1234567890", "profile": {"name": "John Doe"}}
                                ],
                                "messages": [
                                    {
                                        "type": "text",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "text": {
                                            "body": "What is the weather?"
                                        },  # Not a greeting to trigger debate
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
                with patch.object(handler, "_start_debate") as mock_debate:
                    result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        mock_debate.assert_called_once()
        call_args = mock_debate.call_args[0]
        assert call_args[1] == "John Doe"  # Contact name should be resolved

    def test_fallback_to_unknown_contact(self):
        """Should fallback to 'Unknown' when contact not found."""
        handler = WhatsAppHandler({})

        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "field": "messages",
                            "value": {
                                "metadata": {"phone_number_id": "123456"},
                                "contacts": [],  # No contacts
                                "messages": [
                                    {
                                        "type": "text",
                                        "from": "1234567890",
                                        "id": "msg123",
                                        "text": {
                                            "body": "Explain quantum computing"
                                        },  # Not a greeting to trigger debate
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

        with patch(
            "aragora.server.handlers.bots.whatsapp.WHATSAPP_APP_SECRET", TEST_WHATSAPP_SECRET
        ):
            with patch("aragora.server.handlers.bots.whatsapp.WHATSAPP_ACCESS_TOKEN", ""):
                with patch.object(handler, "_start_debate") as mock_debate:
                    result = handler.handle_post("/api/v1/bots/whatsapp/webhook", {}, mock_request)

        assert result is not None
        mock_debate.assert_called_once()
        call_args = mock_debate.call_args[0]
        assert call_args[1] == "Unknown"  # Should fallback to Unknown
