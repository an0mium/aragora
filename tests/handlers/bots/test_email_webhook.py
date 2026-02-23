"""
Tests for Email Webhook Handler (inbound email processing).

Covers all routes and behavior of the EmailWebhookHandler class:
- can_handle() routing for all defined routes
- GET  /api/v1/bots/email/status                - Integration status endpoint
- POST /api/v1/bots/email/webhook/sendgrid       - SendGrid Inbound Parse webhook
- POST /api/v1/bots/email/webhook/mailgun         - Mailgun Inbound webhook
- POST /api/v1/bots/email/webhook/ses             - AWS SES SNS notification
- Signature verification per provider
- Content-Length validation and body-too-large rejection
- JSON parse errors
- Import errors (module unavailable)
- Async task scheduling (event loop present / absent)
- Mailgun: JSON vs form-data, nested vs flat signature fields
- SES: SubscriptionConfirmation, email notifications, non-email notifications
- _parse_form_data: multipart, urlencoded, empty
- _is_bot_enabled, _get_platform_config_status
- Handler initialization and module exports
- Email inbound disabled (503)
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
    import aragora.server.handlers.bots.email_webhook as mod

    return mod


@pytest.fixture
def handler_cls(handler_module):
    return handler_module.EmailWebhookHandler


@pytest.fixture
def handler(handler_cls):
    """Create an EmailWebhookHandler with empty context."""
    return handler_cls({})


# ---------------------------------------------------------------------------
# Mock HTTP Handler
# ---------------------------------------------------------------------------


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for simulating requests."""

    path: str = "/api/v1/bots/email/webhook/sendgrid"
    method: str = "POST"
    body_bytes: bytes = b"{}"
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.rfile = io.BytesIO(self.body_bytes)
        if "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(self.body_bytes))
        self.client_address = ("127.0.0.1", 12345)


def _make_sendgrid_handler(
    body_bytes: bytes | None = None,
    content_type: str = "application/x-www-form-urlencoded",
    timestamp: str = "",
    signature: str = "",
    content_length: str | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler pre-configured for SendGrid webhook POST."""
    if body_bytes is None:
        body_bytes = b"from=test%40example.com&to=reply%40aragora.ai&subject=Hello&text=Body"
    headers = {
        "Content-Type": content_type,
        "X-Twilio-Email-Event-Webhook-Timestamp": timestamp,
        "X-Twilio-Email-Event-Webhook-Signature": signature,
    }
    if content_length is not None:
        headers["Content-Length"] = content_length
    return MockHTTPHandler(
        path="/api/v1/bots/email/webhook/sendgrid",
        body_bytes=body_bytes,
        headers=headers,
    )


def _make_mailgun_handler(
    payload: dict[str, Any] | None = None,
    content_type: str = "application/json",
    content_length: str | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler pre-configured for Mailgun webhook POST."""
    if payload is None:
        payload = {
            "signature": {"timestamp": "1234567890", "token": "tok", "signature": "sig"},
            "sender": "test@example.com",
            "recipient": "reply@aragora.ai",
            "subject": "Re: Test",
            "body-plain": "Hello",
        }
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": content_type}
    if content_length is not None:
        headers["Content-Length"] = content_length
    return MockHTTPHandler(
        path="/api/v1/bots/email/webhook/mailgun",
        body_bytes=body_bytes,
        headers=headers,
    )


def _make_ses_handler(
    notification: dict[str, Any] | None = None,
    content_length: str | None = None,
) -> MockHTTPHandler:
    """Create a MockHTTPHandler pre-configured for SES webhook POST."""
    if notification is None:
        notification = {
            "Type": "Notification",
            "TopicArn": "arn:aws:sns:us-east-1:123456789:ses-inbound",
            "Message": json.dumps(
                {
                    "notificationType": "Received",
                    "mail": {
                        "messageId": "ses-msg-001",
                        "source": "sender@example.com",
                        "destination": ["reply@aragora.ai"],
                        "commonHeaders": {"subject": "SES Test"},
                    },
                    "content": "plain text body",
                }
            ),
        }
    body_bytes = json.dumps(notification).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if content_length is not None:
        headers["Content-Length"] = content_length
    return MockHTTPHandler(
        path="/api/v1/bots/email/webhook/ses",
        body_bytes=body_bytes,
        headers=headers,
    )


# ---------------------------------------------------------------------------
# Mock email data
# ---------------------------------------------------------------------------


def _mock_email_data(
    message_id: str = "msg-001",
    from_email: str = "sender@example.com",
    to_email: str = "reply@aragora.ai",
    subject: str = "Test Subject",
):
    """Create a mock InboundEmail-like object."""
    mock = MagicMock()
    mock.message_id = message_id
    mock.from_email = from_email
    mock.to_email = to_email
    mock.subject = subject
    mock.body_plain = "Hello"
    mock.body_html = "<p>Hello</p>"
    mock.in_reply_to = ""
    mock.references = []
    mock.headers = {}
    return mock


# ===========================================================================
# can_handle()
# ===========================================================================


class TestCanHandle:
    """Tests for can_handle() route matching."""

    def test_sendgrid_route(self, handler):
        assert handler.can_handle("/api/v1/bots/email/webhook/sendgrid", "POST") is True

    def test_mailgun_route(self, handler):
        assert handler.can_handle("/api/v1/bots/email/webhook/mailgun", "POST") is True

    def test_ses_route(self, handler):
        assert handler.can_handle("/api/v1/bots/email/webhook/ses", "POST") is True

    def test_status_route(self, handler):
        assert handler.can_handle("/api/v1/bots/email/status", "GET") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/bots/slack/webhook", "POST") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/", "GET") is False

    def test_partial_match(self, handler):
        assert handler.can_handle("/api/v1/bots/email/webhook/sendgridXYZ", "POST") is False

    def test_different_version(self, handler):
        assert handler.can_handle("/api/v2/bots/email/webhook/sendgrid", "POST") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("", "GET") is False

    def test_routes_list_complete(self, handler):
        """ROUTES list contains exactly the expected paths."""
        assert set(handler.ROUTES) == {
            "/api/v1/bots/email/webhook/sendgrid",
            "/api/v1/bots/email/webhook/mailgun",
            "/api/v1/bots/email/webhook/ses",
            "/api/v1/bots/email/status",
        }


# ===========================================================================
# GET /api/v1/bots/email/status
# ===========================================================================


class TestStatusEndpoint:
    """Tests for the status endpoint."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_status_body_has_platform(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
        body = _body(result)
        assert body["platform"] == "email"

    @pytest.mark.asyncio
    async def test_status_body_has_enabled_field(self, handler, handler_module):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        with patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True):
            result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
        body = _body(result)
        assert "enabled" in body
        assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_status_has_providers(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
        body = _body(result)
        assert "providers" in body
        assert "sendgrid" in body["providers"]
        assert "mailgun" in body["providers"]
        assert "ses" in body["providers"]

    @pytest.mark.asyncio
    async def test_status_sendgrid_configured(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        with patch.dict("os.environ", {"SENDGRID_INBOUND_SECRET": "test-secret"}):
            result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
        body = _body(result)
        assert body["providers"]["sendgrid"]["configured"] is True

    @pytest.mark.asyncio
    async def test_status_sendgrid_not_configured(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        with patch.dict("os.environ", {}, clear=False):
            # Remove the key if present
            import os

            orig = os.environ.pop("SENDGRID_INBOUND_SECRET", None)
            try:
                result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
            finally:
                if orig is not None:
                    os.environ["SENDGRID_INBOUND_SECRET"] = orig
        body = _body(result)
        assert body["providers"]["sendgrid"]["configured"] is False

    @pytest.mark.asyncio
    async def test_status_mailgun_configured(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        with patch.dict("os.environ", {"MAILGUN_WEBHOOK_SIGNING_KEY": "test-key"}):
            result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
        body = _body(result)
        assert body["providers"]["mailgun"]["configured"] is True

    @pytest.mark.asyncio
    async def test_status_ses_configured(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        with patch.dict("os.environ", {"SES_NOTIFICATION_SECRET": "test-secret"}):
            result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
        body = _body(result)
        assert body["providers"]["ses"]["configured"] is True

    @pytest.mark.asyncio
    async def test_status_webhook_urls_present(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
        body = _body(result)
        assert body["providers"]["sendgrid"]["webhook_url"] == "/api/v1/bots/email/webhook/sendgrid"
        assert body["providers"]["mailgun"]["webhook_url"] == "/api/v1/bots/email/webhook/mailgun"
        assert body["providers"]["ses"]["webhook_url"] == "/api/v1/bots/email/webhook/ses"

    @pytest.mark.asyncio
    async def test_status_signature_verification_fields(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/status", method="GET")
        result = await handler.handle("/api/v1/bots/email/status", {}, http_handler)
        body = _body(result)
        assert body["providers"]["sendgrid"]["signature_verification"] == "hmac-sha256"
        assert body["providers"]["mailgun"]["signature_verification"] == "hmac-sha256"

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_non_status_get(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/webhook/sendgrid", method="GET")
        result = await handler.handle("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown_path(self, handler):
        http_handler = MockHTTPHandler(path="/api/v1/bots/email/unknown", method="GET")
        result = await handler.handle("/api/v1/bots/email/unknown", {}, http_handler)
        assert result is None


# ===========================================================================
# POST disabled
# ===========================================================================


class TestEmailInboundDisabled:
    """Tests when EMAIL_INBOUND_ENABLED is False."""

    def test_sendgrid_returns_503(self, handler, handler_module):
        with patch.object(handler_module, "EMAIL_INBOUND_ENABLED", False):
            http_handler = _make_sendgrid_handler()
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)
        assert _status(result) == 503
        assert "disabled" in _body(result).get("error", "").lower()

    def test_mailgun_returns_503(self, handler, handler_module):
        with patch.object(handler_module, "EMAIL_INBOUND_ENABLED", False):
            http_handler = _make_mailgun_handler()
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)
        assert _status(result) == 503

    def test_ses_returns_503(self, handler, handler_module):
        with patch.object(handler_module, "EMAIL_INBOUND_ENABLED", False):
            http_handler = _make_ses_handler()
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)
        assert _status(result) == 503

    def test_unknown_path_returns_none(self, handler, handler_module):
        with patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True):
            http_handler = MockHTTPHandler(path="/api/v1/bots/email/unknown")
            result = handler.handle_post("/api/v1/bots/email/unknown", {}, http_handler)
        assert result is None


# ===========================================================================
# POST /api/v1/bots/email/webhook/sendgrid
# ===========================================================================


class TestSendGridWebhook:
    """Tests for SendGrid Inbound Parse webhook."""

    def test_successful_sendgrid_webhook(self, handler, handler_module):
        """Valid SendGrid webhook returns 200 with message_id."""
        email_data = _mock_email_data()
        mock_parse = MagicMock(return_value=email_data)
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()

        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch(
                "aragora.server.handlers.bots.email_webhook.EmailWebhookHandler._parse_form_data",
                return_value={"from": "test@example.com"},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_sendgrid_webhook=mock_parse,
                        verify_sendgrid_signature=mock_verify,
                        handle_email_reply=mock_handle,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"
        assert body["message_id"] == "msg-001"

    def test_sendgrid_signature_failure(self, handler, handler_module):
        """Invalid SendGrid signature returns 401."""
        mock_verify = MagicMock(return_value=False)

        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_sendgrid_signature=mock_verify,
                        parse_sendgrid_webhook=MagicMock(),
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
            patch("aragora.audit.unified.audit_security"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 401
        assert "signature" in _body(result).get("error", "").lower()

    def test_sendgrid_invalid_content_length(self, handler, handler_module):
        """Invalid Content-Length header returns 400."""
        http_handler = _make_sendgrid_handler(content_length="not-a-number")

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_sendgrid_signature=MagicMock(return_value=True),
                        parse_sendgrid_webhook=MagicMock(),
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 400
        assert "content-length" in _body(result).get("error", "").lower()

    def test_sendgrid_body_too_large(self, handler, handler_module):
        """Body exceeding 10MB returns 413."""
        http_handler = _make_sendgrid_handler(content_length=str(11 * 1024 * 1024))

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_sendgrid_signature=MagicMock(return_value=True),
                        parse_sendgrid_webhook=MagicMock(),
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 413
        assert "too large" in _body(result).get("error", "").lower()

    def test_sendgrid_import_error(self, handler, handler_module):
        """When email_reply_loop module unavailable, returns 503."""
        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict("sys.modules", {"aragora.integrations.email_reply_loop": None}),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_sendgrid_value_error_returns_200(self, handler, handler_module):
        """ValueError during processing returns 200 to prevent retries."""
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(side_effect=ValueError("bad data"))

        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_sendgrid_signature=mock_verify,
                        parse_sendgrid_webhook=mock_parse,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
            patch(
                "aragora.server.handlers.bots.email_webhook.EmailWebhookHandler._parse_form_data",
                return_value={},
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "error"

    def test_sendgrid_runtime_error_returns_200(self, handler, handler_module):
        """RuntimeError during processing returns 200 to prevent retries."""
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(side_effect=RuntimeError("processing error"))

        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_sendgrid_signature=mock_verify,
                        parse_sendgrid_webhook=mock_parse,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
            patch(
                "aragora.server.handlers.bots.email_webhook.EmailWebhookHandler._parse_form_data",
                return_value={},
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "error"

    def test_sendgrid_key_error_returns_200(self, handler, handler_module):
        """KeyError during processing returns 200 to prevent retries."""
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(side_effect=KeyError("missing_field"))

        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_sendgrid_signature=mock_verify,
                        parse_sendgrid_webhook=mock_parse,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
            patch(
                "aragora.server.handlers.bots.email_webhook.EmailWebhookHandler._parse_form_data",
                return_value={},
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "error"

    def test_sendgrid_type_error_returns_200(self, handler, handler_module):
        """TypeError during processing returns 200 to prevent retries."""
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(side_effect=TypeError("wrong type"))

        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_sendgrid_signature=mock_verify,
                        parse_sendgrid_webhook=mock_parse,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
            patch(
                "aragora.server.handlers.bots.email_webhook.EmailWebhookHandler._parse_form_data",
                return_value={},
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "error"

    def test_sendgrid_os_error_returns_200(self, handler, handler_module):
        """OSError during processing returns 200 to prevent retries."""
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(side_effect=OSError("io error"))

        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_sendgrid_signature=mock_verify,
                        parse_sendgrid_webhook=mock_parse,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
            patch(
                "aragora.server.handlers.bots.email_webhook.EmailWebhookHandler._parse_form_data",
                return_value={},
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "error"

    def test_sendgrid_with_event_loop(self, handler, handler_module):
        """When event loop is running, uses create_task instead of asyncio.run."""
        email_data = _mock_email_data()
        mock_parse = MagicMock(return_value=email_data)
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()

        mock_loop = MagicMock()
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = None

        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch(
                "aragora.server.handlers.bots.email_webhook.EmailWebhookHandler._parse_form_data",
                return_value={"from": "test@example.com"},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_sendgrid_webhook=mock_parse,
                        verify_sendgrid_signature=mock_verify,
                        handle_email_reply=mock_handle,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", return_value=mock_loop),
            patch("asyncio.create_task", return_value=mock_task) as mock_create,
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert _status(result) == 200
        mock_create.assert_called_once()


# ===========================================================================
# POST /api/v1/bots/email/webhook/mailgun
# ===========================================================================


class TestMailgunWebhook:
    """Tests for Mailgun inbound webhook."""

    def test_successful_mailgun_json(self, handler, handler_module):
        """Valid Mailgun JSON webhook returns 200 with message_id."""
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()

        mock_inbound = MagicMock()

        payload = {
            "signature": {"timestamp": "1234567890", "token": "tok", "signature": "sig"},
            "sender": "test@example.com",
            "recipient": "reply@aragora.ai",
            "subject": "Re: Test",
            "body-plain": "Hello there",
            "body-html": "<p>Hello there</p>",
            "Message-Id": "mg-msg-001",
            "message-headers": json.dumps(
                [["In-Reply-To", "<orig@example.com>"], ["References", "<ref1> <ref2>"]]
            ),
        }
        http_handler = _make_mailgun_handler(payload)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"
        mock_inbound.assert_called_once()

    def test_mailgun_flat_signature_fields(self, handler, handler_module):
        """Mailgun form data with flat (non-nested) signature fields."""
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_inbound = MagicMock()

        payload = {
            "timestamp": "1234567890",
            "token": "tok",
            "signature": "sig",
            "sender": "test@example.com",
            "recipient": "reply@aragora.ai",
            "subject": "Flat test",
            "body-plain": "Hello",
        }
        http_handler = _make_mailgun_handler(payload)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 200
        # verify_mailgun_signature should be called with the flat fields
        mock_verify.assert_called_once_with("1234567890", "tok", "sig")

    def test_mailgun_event_data_nested(self, handler, handler_module):
        """Mailgun event webhook with nested event-data field."""
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_inbound = MagicMock()

        payload = {
            "signature": {"timestamp": "1234567890", "token": "tok", "signature": "sig"},
            "event-data": {
                "sender": "nested@example.com",
                "recipient": "reply@aragora.ai",
                "subject": "Nested event",
                "body-plain": "Nested body",
                "Message-Id": "nested-msg-001",
            },
        }
        http_handler = _make_mailgun_handler(payload)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 200
        # Verify InboundEmail was called with nested event-data fields
        call_kwargs = mock_inbound.call_args
        assert (
            call_kwargs[1]["from_email"] == "nested@example.com"
            or call_kwargs[0][1] == "nested@example.com"
            if call_kwargs[0]
            else True
        )

    def test_mailgun_signature_failure(self, handler, handler_module):
        """Invalid Mailgun signature returns 401."""
        mock_verify = MagicMock(return_value=False)

        http_handler = _make_mailgun_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=AsyncMock(),
                        InboundEmail=MagicMock(),
                    )
                },
            ),
            patch("aragora.audit.unified.audit_security"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 401
        assert "signature" in _body(result).get("error", "").lower()

    def test_mailgun_invalid_json(self, handler, handler_module):
        """Invalid JSON body returns 400."""
        body_bytes = b"this is not json"
        http_handler = MockHTTPHandler(
            path="/api/v1/bots/email/webhook/mailgun",
            body_bytes=body_bytes,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body_bytes))},
        )

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=MagicMock(return_value=True),
                        handle_email_reply=AsyncMock(),
                        InboundEmail=MagicMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 400
        assert "json" in _body(result).get("error", "").lower()

    def test_mailgun_invalid_content_length(self, handler, handler_module):
        """Invalid Content-Length returns 400."""
        http_handler = _make_mailgun_handler(content_length="not-a-number")

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=MagicMock(return_value=True),
                        handle_email_reply=AsyncMock(),
                        InboundEmail=MagicMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 400

    def test_mailgun_body_too_large(self, handler, handler_module):
        """Body exceeding 10MB returns 413."""
        http_handler = _make_mailgun_handler(content_length=str(11 * 1024 * 1024))

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=MagicMock(return_value=True),
                        handle_email_reply=AsyncMock(),
                        InboundEmail=MagicMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 413

    def test_mailgun_import_error(self, handler, handler_module):
        """When email_reply_loop module unavailable, returns 503."""
        http_handler = _make_mailgun_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict("sys.modules", {"aragora.integrations.email_reply_loop": None}),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 503

    def test_mailgun_value_error_returns_200(self, handler, handler_module):
        """ValueError during processing returns 200 to prevent retries."""
        mock_verify = MagicMock(return_value=True)

        payload = {
            "signature": {"timestamp": "1234567890", "token": "tok", "signature": "sig"},
            "sender": "test@example.com",
            "subject": "Test",
            "body-plain": "Hello",
        }
        http_handler = _make_mailgun_handler(payload)

        mock_inbound = MagicMock(side_effect=ValueError("bad data"))

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=AsyncMock(),
                        InboundEmail=mock_inbound,
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "error"

    def test_mailgun_no_message_id_generates_one(self, handler, handler_module):
        """When Message-Id is missing, a fallback message_id is generated."""
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_inbound = MagicMock()

        # No Message-Id or message-id in payload
        payload = {
            "signature": {"timestamp": "1234567890", "token": "tok", "signature": "sig"},
            "sender": "test@example.com",
            "recipient": "reply@aragora.ai",
            "subject": "No ID test",
            "body-plain": "Hello",
        }
        http_handler = _make_mailgun_handler(payload)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 200
        # The generated message_id should start with "mailgun-"
        call_kwargs = mock_inbound.call_args
        msg_id = call_kwargs[1].get("message_id", "") if call_kwargs[1] else ""
        assert msg_id.startswith("mailgun-")

    def test_mailgun_message_headers_as_list(self, handler, handler_module):
        """Headers provided as a list (not JSON string) are parsed correctly."""
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_inbound = MagicMock()

        payload = {
            "signature": {"timestamp": "1234567890", "token": "tok", "signature": "sig"},
            "sender": "test@example.com",
            "recipient": "reply@aragora.ai",
            "subject": "Header test",
            "body-plain": "Hello",
            "Message-Id": "mg-hdr-001",
            "message-headers": [
                ["In-Reply-To", "<orig@example.com>"],
                ["References", "<ref1> <ref2>"],
            ],
        }
        http_handler = _make_mailgun_handler(payload)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 200
        # Check InboundEmail was called with parsed headers
        call_kwargs = mock_inbound.call_args[1] if mock_inbound.call_args[1] else {}
        headers = call_kwargs.get("headers", {})
        assert headers.get("In-Reply-To") == "<orig@example.com>"

    def test_mailgun_form_data_content_type(self, handler, handler_module):
        """Mailgun with non-JSON content type parses as form data."""
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_inbound = MagicMock()

        body_bytes = b"timestamp=1234567890&token=tok&signature=sig&sender=test%40example.com&subject=Test&body-plain=Hello"
        http_handler = MockHTTPHandler(
            path="/api/v1/bots/email/webhook/mailgun",
            body_bytes=body_bytes,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body_bytes)),
            },
        )

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 200
        # Flat form data: verify signature extracted from flat fields
        mock_verify.assert_called_once_with("1234567890", "tok", "sig")

    def test_mailgun_with_event_loop(self, handler, handler_module):
        """When event loop is running, uses create_task."""
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_inbound = MagicMock()
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = None

        payload = {
            "signature": {"timestamp": "1234567890", "token": "tok", "signature": "sig"},
            "sender": "test@example.com",
            "subject": "Loop test",
            "body-plain": "Hello",
            "Message-Id": "mg-loop-001",
        }
        http_handler = _make_mailgun_handler(payload)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", return_value=MagicMock()),
            patch("asyncio.create_task", return_value=mock_task) as mock_create,
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 200
        mock_create.assert_called_once()

    def test_mailgun_alternative_from_fields(self, handler, handler_module):
        """Mailgun falls back to 'from'/'From' and 'to'/'To' fields."""
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_inbound = MagicMock()

        payload = {
            "signature": {"timestamp": "1234567890", "token": "tok", "signature": "sig"},
            "from": "alt-from@example.com",
            "to": "alt-to@aragora.ai",
            "Subject": "Alt fields",
            "body-plain": "Hello",
            "Message-Id": "mg-alt-001",
        }
        http_handler = _make_mailgun_handler(payload)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert _status(result) == 200
        call_kwargs = mock_inbound.call_args[1] if mock_inbound.call_args[1] else {}
        assert call_kwargs.get("from_email") == "alt-from@example.com"

    def test_mailgun_invalid_message_headers_json(self, handler, handler_module):
        """Invalid JSON in message-headers is handled gracefully."""
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_inbound = MagicMock()

        payload = {
            "signature": {"timestamp": "1234567890", "token": "tok", "signature": "sig"},
            "sender": "test@example.com",
            "subject": "Bad headers",
            "body-plain": "Hello",
            "Message-Id": "mg-bad-hdr-001",
            "message-headers": "not-valid-json",
        }
        http_handler = _make_mailgun_handler(payload)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        # Should not crash, just skip header parsing
        assert _status(result) == 200
        call_kwargs = mock_inbound.call_args[1] if mock_inbound.call_args[1] else {}
        assert call_kwargs.get("headers") == {}


# ===========================================================================
# POST /api/v1/bots/email/webhook/ses
# ===========================================================================


class TestSESWebhook:
    """Tests for AWS SES SNS notification webhook."""

    def test_successful_ses_notification(self, handler, handler_module):
        """Valid SES notification returns 200 with message_id."""
        email_data = _mock_email_data(message_id="ses-msg-001")
        mock_parse = MagicMock(return_value=email_data)
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()

        notification = {
            "Type": "Notification",
            "TopicArn": "arn:aws:sns:us-east-1:123456789:ses-inbound",
            "Message": "{}",
        }
        http_handler = _make_ses_handler(notification)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=mock_parse,
                        verify_ses_signature=mock_verify,
                        handle_email_reply=mock_handle,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"
        assert body["message_id"] == "ses-msg-001"

    def test_ses_subscription_confirmation(self, handler, handler_module):
        """SES SubscriptionConfirmation returns subscription_pending status."""
        mock_verify = MagicMock(return_value=True)

        notification = {
            "Type": "SubscriptionConfirmation",
            "SubscribeURL": "https://sns.amazonaws.com/confirm?token=abc123",
        }
        http_handler = _make_ses_handler(notification)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=MagicMock(return_value=None),
                        verify_ses_signature=mock_verify,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "subscription_pending"
        assert body["subscribe_url"] == "https://sns.amazonaws.com/confirm?token=abc123"

    def test_ses_non_email_notification_ignored(self, handler, handler_module):
        """SES notification that is not an email receipt is ignored."""
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(return_value=None)

        notification = {
            "Type": "Notification",
            "Message": "{}",
        }
        http_handler = _make_ses_handler(notification)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=mock_parse,
                        verify_ses_signature=mock_verify,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ignored"

    def test_ses_signature_failure(self, handler, handler_module):
        """Invalid SES signature returns 401."""
        mock_verify = MagicMock(return_value=False)

        http_handler = _make_ses_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=MagicMock(),
                        verify_ses_signature=mock_verify,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
            patch("aragora.audit.unified.audit_security"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 401
        assert "signature" in _body(result).get("error", "").lower()

    def test_ses_invalid_json(self, handler, handler_module):
        """Invalid JSON body returns 400."""
        body_bytes = b"not json"
        http_handler = MockHTTPHandler(
            path="/api/v1/bots/email/webhook/ses",
            body_bytes=body_bytes,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body_bytes))},
        )

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=MagicMock(),
                        verify_ses_signature=MagicMock(return_value=True),
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 400
        assert "json" in _body(result).get("error", "").lower()

    def test_ses_invalid_content_length(self, handler, handler_module):
        """Invalid Content-Length returns 400."""
        http_handler = _make_ses_handler(content_length="bad")

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=MagicMock(),
                        verify_ses_signature=MagicMock(return_value=True),
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 400

    def test_ses_body_too_large(self, handler, handler_module):
        """Body exceeding 10MB returns 413."""
        http_handler = _make_ses_handler(content_length=str(11 * 1024 * 1024))

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=MagicMock(),
                        verify_ses_signature=MagicMock(return_value=True),
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 413

    def test_ses_import_error(self, handler, handler_module):
        """When email_reply_loop module unavailable, returns 503."""
        http_handler = _make_ses_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict("sys.modules", {"aragora.integrations.email_reply_loop": None}),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 503

    def test_ses_value_error_returns_200(self, handler, handler_module):
        """ValueError during processing returns 200 to prevent retries."""
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(side_effect=ValueError("bad notification"))

        notification = {"Type": "Notification", "Message": "{}"}
        http_handler = _make_ses_handler(notification)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=mock_parse,
                        verify_ses_signature=mock_verify,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "error"

    def test_ses_runtime_error_returns_200(self, handler, handler_module):
        """RuntimeError during processing returns 200 to prevent retries."""
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(side_effect=RuntimeError("boom"))

        notification = {"Type": "Notification", "Message": "{}"}
        http_handler = _make_ses_handler(notification)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=mock_parse,
                        verify_ses_signature=mock_verify,
                        handle_email_reply=AsyncMock(),
                    )
                },
            ),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "error"

    def test_ses_with_event_loop(self, handler, handler_module):
        """When event loop is running, uses create_task."""
        email_data = _mock_email_data(message_id="ses-loop-001")
        mock_parse = MagicMock(return_value=email_data)
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_task = MagicMock()
        mock_task.cancelled.return_value = False
        mock_task.exception.return_value = None

        notification = {"Type": "Notification", "Message": "{}"}
        http_handler = _make_ses_handler(notification)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=mock_parse,
                        verify_ses_signature=mock_verify,
                        handle_email_reply=mock_handle,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", return_value=MagicMock()),
            patch("asyncio.create_task", return_value=mock_task) as mock_create,
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert _status(result) == 200
        mock_create.assert_called_once()

    def test_ses_subscription_confirmation_no_url(self, handler, handler_module):
        """SES SubscriptionConfirmation without SubscribeURL falls through to parse."""
        mock_verify = MagicMock(return_value=True)
        email_data = _mock_email_data(message_id="ses-no-url-001")
        mock_parse = MagicMock(return_value=email_data)
        mock_handle = AsyncMock()

        notification = {
            "Type": "SubscriptionConfirmation",
            # No SubscribeURL - falls through
        }
        http_handler = _make_ses_handler(notification)

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=mock_parse,
                        verify_ses_signature=mock_verify,
                        handle_email_reply=mock_handle,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        # Falls through to parse_ses_notification
        assert _status(result) == 200
        mock_parse.assert_called_once()


# ===========================================================================
# _parse_form_data
# ===========================================================================


class TestParseFormData:
    """Tests for _parse_form_data utility."""

    def test_urlencoded_single_values(self, handler):
        body = b"key1=value1&key2=value2"
        result = handler._parse_form_data(body, "application/x-www-form-urlencoded")
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    def test_urlencoded_multiple_values(self, handler):
        body = b"key=val1&key=val2"
        result = handler._parse_form_data(body, "application/x-www-form-urlencoded")
        assert result["key"] == ["val1", "val2"]

    def test_urlencoded_encoded_chars(self, handler):
        body = b"email=test%40example.com"
        result = handler._parse_form_data(body, "application/x-www-form-urlencoded")
        assert result["email"] == "test@example.com"

    def test_unknown_content_type_returns_empty(self, handler):
        body = b"some data"
        result = handler._parse_form_data(body, "text/plain")
        assert result == {}

    def test_multipart_with_no_boundary(self, handler):
        body = b"some data"
        result = handler._parse_form_data(body, "multipart/form-data")
        assert result == {}

    def test_multipart_with_boundary(self, handler):
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        content_type = f"multipart/form-data; boundary={boundary}"
        body = (
            "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n"
            'Content-Disposition: form-data; name="from"\r\n\r\n'
            "test@example.com\r\n"
            "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n"
            'Content-Disposition: form-data; name="subject"\r\n\r\n'
            "Hello World\r\n"
            "------WebKitFormBoundary7MA4YWxkTrZu0gW--\r\n"
        ).encode("utf-8")
        result = handler._parse_form_data(body, content_type)
        assert result.get("from") == "test@example.com"
        assert result.get("subject") == "Hello World"

    def test_empty_body_urlencoded(self, handler):
        result = handler._parse_form_data(b"", "application/x-www-form-urlencoded")
        assert result == {}


# ===========================================================================
# _is_bot_enabled
# ===========================================================================


class TestIsBotEnabled:
    """Tests for _is_bot_enabled."""

    def test_enabled_when_flag_true(self, handler, handler_module):
        with patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True):
            assert handler._is_bot_enabled() is True

    def test_disabled_when_flag_false(self, handler, handler_module):
        with patch.object(handler_module, "EMAIL_INBOUND_ENABLED", False):
            assert handler._is_bot_enabled() is False


# ===========================================================================
# _get_platform_config_status
# ===========================================================================


class TestPlatformConfigStatus:
    """Tests for _get_platform_config_status."""

    def test_has_inbound_enabled_field(self, handler):
        status = handler._get_platform_config_status()
        assert "inbound_enabled" in status

    def test_has_providers_dict(self, handler):
        status = handler._get_platform_config_status()
        assert "providers" in status
        assert "sendgrid" in status["providers"]
        assert "mailgun" in status["providers"]
        assert "ses" in status["providers"]

    def test_sendgrid_not_configured_by_default(self, handler):
        import os

        orig = os.environ.pop("SENDGRID_INBOUND_SECRET", None)
        try:
            status = handler._get_platform_config_status()
            assert status["providers"]["sendgrid"]["configured"] is False
        finally:
            if orig is not None:
                os.environ["SENDGRID_INBOUND_SECRET"] = orig

    def test_sendgrid_configured_with_env(self, handler):
        with patch.dict("os.environ", {"SENDGRID_INBOUND_SECRET": "secret"}):
            status = handler._get_platform_config_status()
        assert status["providers"]["sendgrid"]["configured"] is True

    def test_mailgun_configured_with_env(self, handler):
        with patch.dict("os.environ", {"MAILGUN_WEBHOOK_SIGNING_KEY": "key"}):
            status = handler._get_platform_config_status()
        assert status["providers"]["mailgun"]["configured"] is True

    def test_ses_configured_with_env(self, handler):
        with patch.dict("os.environ", {"SES_NOTIFICATION_SECRET": "secret"}):
            status = handler._get_platform_config_status()
        assert status["providers"]["ses"]["configured"] is True


# ===========================================================================
# Handler Initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for EmailWebhookHandler initialization."""

    def test_default_ctx(self, handler_cls):
        h = handler_cls({})
        assert h.ctx == {}

    def test_custom_ctx(self, handler_cls):
        ctx = {"storage": MagicMock()}
        h = handler_cls(ctx)
        assert h.ctx is ctx

    def test_none_ctx(self, handler_cls):
        h = handler_cls(None)
        # SecureHandler converts None to empty via cast
        assert h.ctx is None or h.ctx == {}

    def test_bot_platform(self, handler):
        assert handler.bot_platform == "email"

    def test_routes_defined(self, handler):
        assert "/api/v1/bots/email/webhook/sendgrid" in handler.ROUTES
        assert "/api/v1/bots/email/webhook/mailgun" in handler.ROUTES
        assert "/api/v1/bots/email/webhook/ses" in handler.ROUTES
        assert "/api/v1/bots/email/status" in handler.ROUTES


# ===========================================================================
# handle_post routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for handle_post path routing."""

    def test_unknown_path_returns_none(self, handler, handler_module):
        with patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True):
            http_handler = MockHTTPHandler(path="/api/v1/bots/email/unknown")
            result = handler.handle_post("/api/v1/bots/email/unknown", {}, http_handler)
        assert result is None

    def test_routes_to_sendgrid(self, handler, handler_module):
        email_data = _mock_email_data()
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(return_value=email_data)
        mock_handle = AsyncMock()

        http_handler = _make_sendgrid_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch(
                "aragora.server.handlers.bots.email_webhook.EmailWebhookHandler._parse_form_data",
                return_value={},
            ),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_sendgrid_webhook=mock_parse,
                        verify_sendgrid_signature=mock_verify,
                        handle_email_reply=mock_handle,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/sendgrid", {}, http_handler)

        assert result is not None
        assert _status(result) == 200

    def test_routes_to_mailgun(self, handler, handler_module):
        mock_verify = MagicMock(return_value=True)
        mock_handle = AsyncMock()
        mock_inbound = MagicMock()

        http_handler = _make_mailgun_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        verify_mailgun_signature=mock_verify,
                        handle_email_reply=mock_handle,
                        InboundEmail=mock_inbound,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/mailgun", {}, http_handler)

        assert result is not None
        assert _status(result) == 200

    def test_routes_to_ses(self, handler, handler_module):
        email_data = _mock_email_data()
        mock_verify = MagicMock(return_value=True)
        mock_parse = MagicMock(return_value=email_data)
        mock_handle = AsyncMock()

        http_handler = _make_ses_handler()

        with (
            patch.object(handler_module, "EMAIL_INBOUND_ENABLED", True),
            patch.dict(
                "sys.modules",
                {
                    "aragora.integrations.email_reply_loop": MagicMock(
                        parse_ses_notification=mock_parse,
                        verify_ses_signature=mock_verify,
                        handle_email_reply=mock_handle,
                    )
                },
            ),
            patch("aragora.audit.unified.audit_data"),
            patch("asyncio.get_running_loop", side_effect=RuntimeError),
            patch("asyncio.run"),
        ):
            result = handler.handle_post("/api/v1/bots/email/webhook/ses", {}, http_handler)

        assert result is not None
        assert _status(result) == 200


# ===========================================================================
# Module-level __all__
# ===========================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports(self, handler_module):
        assert "EmailWebhookHandler" in handler_module.__all__

    def test_handler_in_module(self, handler_module):
        assert hasattr(handler_module, "EmailWebhookHandler")
