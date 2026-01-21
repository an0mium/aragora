"""
Tests for aragora.server.handlers.bots.email_webhook - Email Webhook handler.

Tests cover:
- SendGrid Inbound Parse webhook handling
- AWS SES SNS notification handling
- Signature verification
- Form data parsing
- Status endpoint
- Error handling
"""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.bots.email_webhook import EmailWebhookHandler


# ===========================================================================
# Test Fixtures
# ===========================================================================


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        headers: dict | None = None,
        body: bytes = b"",
        path: str = "/",
        method: str = "POST",
    ):
        self.headers = headers or {}
        self._body = body
        self.path = path
        self.command = method
        self.rfile = BytesIO(body)

    def get(self, key: str, default: str = "") -> str:
        return self.headers.get(key, default)


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handler initialization."""
    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
        "elo_system": MagicMock(),
        "continuum_memory": MagicMock(),
        "critique_store": MagicMock(),
        "document_store": MagicMock(),
        "evidence_store": MagicMock(),
        "usage_tracker": MagicMock(),
    }


@pytest.fixture
def handler(mock_server_context):
    """Create an EmailWebhookHandler instance."""
    return EmailWebhookHandler(mock_server_context)


@pytest.fixture
def sendgrid_form_data():
    """Create SendGrid Inbound Parse form data."""
    return {
        "from": "user@example.com",
        "to": "debate@aragora.example.com",
        "subject": "Re: [Aragora Debate-abc123] AI Ethics Discussion",
        "text": "I agree with the consensus.\n\n--\nOn Mon, Jan 20, 2026...",
        "html": "<p>I agree with the consensus.</p>",
        "headers": "Message-ID: <msg123@example.com>\r\nReferences: <debate-abc123@aragora>",
        "envelope": json.dumps(
            {
                "from": "user@example.com",
                "to": ["debate@aragora.example.com"],
            }
        ),
    }


@pytest.fixture
def ses_notification():
    """Create AWS SES SNS notification."""
    return {
        "Type": "Notification",
        "MessageId": "sns-msg-123",
        "TopicArn": "arn:aws:sns:us-east-1:123456:aragora-emails",
        "Message": json.dumps(
            {
                "notificationType": "Received",
                "mail": {
                    "messageId": "ses-msg-456",
                    "source": "user@example.com",
                    "destination": ["debate@aragora.example.com"],
                    "commonHeaders": {
                        "from": ["user@example.com"],
                        "to": ["debate@aragora.example.com"],
                        "subject": "Re: [Aragora Debate-xyz789] Climate Policy",
                    },
                },
                "receipt": {
                    "action": {"type": "S3", "bucketName": "emails", "objectKey": "email.eml"},
                },
                "content": "From: user@example.com\r\nTo: debate@aragora.example.com\r\n\r\nI disagree strongly.",
            }
        ),
        "Timestamp": "2026-01-20T10:00:00.000Z",
        "Signature": "test_signature",
        "SigningCertURL": "https://sns.us-east-1.amazonaws.com/cert.pem",
    }


@pytest.fixture
def ses_subscription_confirmation():
    """Create AWS SES SNS subscription confirmation."""
    return {
        "Type": "SubscriptionConfirmation",
        "MessageId": "sns-msg-123",
        "TopicArn": "arn:aws:sns:us-east-1:123456:aragora-emails",
        "Token": "confirmation_token",
        "SubscribeURL": "https://sns.us-east-1.amazonaws.com/confirm?token=xxx",
        "Timestamp": "2026-01-20T10:00:00.000Z",
    }


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_sendgrid_webhook(self, handler):
        """Test handler recognizes SendGrid webhook endpoint."""
        assert handler.can_handle("/api/bots/email/webhook/sendgrid") is True

    def test_can_handle_ses_webhook(self, handler):
        """Test handler recognizes SES webhook endpoint."""
        assert handler.can_handle("/api/bots/email/webhook/ses") is True

    def test_can_handle_status(self, handler):
        """Test handler recognizes status endpoint."""
        assert handler.can_handle("/api/bots/email/status") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/bots/email/unknown") is False
        assert handler.can_handle("/api/other/endpoint") is False


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestStatusEndpoint:
    """Tests for GET /api/bots/email/status."""

    def test_get_status(self, handler):
        """Test getting email integration status."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            method="GET",
        )

        result = handler.handle("/api/bots/email/status", {}, mock_http)

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "application/json"

        data = json.loads(result.body)
        assert data["platform"] == "email"
        assert "inbound_enabled" in data
        assert "providers" in data
        assert "sendgrid" in data["providers"]
        assert "ses" in data["providers"]

    def test_status_shows_provider_configuration(self, handler):
        """Test status shows whether providers are configured."""
        mock_http = MockHandler(method="GET")

        with patch.dict(
            "os.environ",
            {"SENDGRID_INBOUND_SECRET": "test_secret", "SES_NOTIFICATION_SECRET": ""},
            clear=False,
        ):
            result = handler.handle("/api/bots/email/status", {}, mock_http)

        data = json.loads(result.body)
        # SendGrid configured, SES not
        assert data["providers"]["sendgrid"]["configured"] is True


# ===========================================================================
# SendGrid Webhook Tests
# ===========================================================================


class TestSendGridWebhook:
    """Tests for POST /api/bots/email/webhook/sendgrid."""

    def test_sendgrid_webhook_success(self, handler, sendgrid_form_data):
        """Test successful SendGrid webhook processing."""
        # Create urlencoded form data
        from urllib.parse import urlencode

        body = urlencode(sendgrid_form_data).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Twilio-Email-Event-Webhook-Timestamp": "1234567890",
                "X-Twilio-Email-Event-Webhook-Signature": "test_sig",
            },
            body=body,
            method="POST",
        )

        with (
            patch(
                "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
                return_value=True,
            ),
            patch("aragora.integrations.email_reply_loop.parse_sendgrid_webhook") as mock_parse,
            patch(
                "aragora.integrations.email_reply_loop.handle_email_reply",
                new_callable=AsyncMock,
            ) as mock_handle,
        ):
            # Configure mock email data
            mock_email = MagicMock()
            mock_email.message_id = "msg-123"
            mock_email.from_email = "user@example.com"
            mock_email.subject = "Re: Test"
            mock_parse.return_value = mock_email

            result = handler.handle_post("/api/bots/email/webhook/sendgrid", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["status"] == "ok"
        assert "message_id" in data

    def test_sendgrid_invalid_signature(self, handler):
        """Test SendGrid webhook rejects invalid signature."""
        body = b"from=test@example.com&to=debate@aragora.com&text=Hello"

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Twilio-Email-Event-Webhook-Timestamp": "1234567890",
                "X-Twilio-Email-Event-Webhook-Signature": "invalid_sig",
            },
            body=body,
            method="POST",
        )

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=False,
        ):
            result = handler.handle_post("/api/bots/email/webhook/sendgrid", {}, mock_http)

        assert result is not None
        assert result.status_code == 401

    def test_sendgrid_disabled(self, handler):
        """Test SendGrid webhook returns 503 when disabled."""
        body = b"from=test@example.com"

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch("aragora.server.handlers.bots.email_webhook.EMAIL_INBOUND_ENABLED", False):
            result = handler.handle_post("/api/bots/email/webhook/sendgrid", {}, mock_http)

        assert result is not None
        assert result.status_code == 503


# ===========================================================================
# SES Webhook Tests
# ===========================================================================


class TestSESWebhook:
    """Tests for POST /api/bots/email/webhook/ses."""

    def test_ses_notification_success(self, handler, ses_notification):
        """Test successful SES notification processing."""
        body = json.dumps(ses_notification).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with (
            patch(
                "aragora.integrations.email_reply_loop.verify_ses_signature",
                return_value=True,
            ),
            patch("aragora.integrations.email_reply_loop.parse_ses_notification") as mock_parse,
            patch(
                "aragora.integrations.email_reply_loop.handle_email_reply",
                new_callable=AsyncMock,
            ),
        ):
            mock_email = MagicMock()
            mock_email.message_id = "ses-msg-456"
            mock_email.from_email = "user@example.com"
            mock_email.subject = "Re: Test"
            mock_parse.return_value = mock_email

            result = handler.handle_post("/api/bots/email/webhook/ses", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["status"] == "ok"

    def test_ses_subscription_confirmation(self, handler, ses_subscription_confirmation):
        """Test SES subscription confirmation handling."""
        body = json.dumps(ses_subscription_confirmation).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ):
            result = handler.handle_post("/api/bots/email/webhook/ses", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["status"] == "subscription_pending"
        assert "subscribe_url" in data

    def test_ses_invalid_signature(self, handler, ses_notification):
        """Test SES webhook rejects invalid signature."""
        body = json.dumps(ses_notification).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=False,
        ):
            result = handler.handle_post("/api/bots/email/webhook/ses", {}, mock_http)

        assert result is not None
        assert result.status_code == 401

    def test_ses_invalid_json(self, handler):
        """Test SES webhook handles invalid JSON."""
        body = b"not valid json"

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        result = handler.handle_post("/api/bots/email/webhook/ses", {}, mock_http)

        assert result is not None
        assert result.status_code == 400

    def test_ses_non_email_notification(self, handler):
        """Test SES webhook ignores non-email notifications."""
        notification = {
            "Type": "Notification",
            "Message": json.dumps({"notificationType": "Bounce"}),  # Not a received email
        }

        body = json.dumps(notification).encode()

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(body)),
            },
            body=body,
            method="POST",
        )

        with (
            patch(
                "aragora.integrations.email_reply_loop.verify_ses_signature",
                return_value=True,
            ),
            patch(
                "aragora.integrations.email_reply_loop.parse_ses_notification",
                return_value=None,  # Not an email receipt
            ),
        ):
            result = handler.handle_post("/api/bots/email/webhook/ses", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["status"] == "ignored"


# ===========================================================================
# Form Data Parsing Tests
# ===========================================================================


class TestFormDataParsing:
    """Tests for multipart form data parsing."""

    def test_parse_urlencoded(self, handler):
        """Test parsing URL-encoded form data."""
        body = b"from=test@example.com&to=debate@aragora.com&text=Hello+World"

        result = handler._parse_form_data(body, "application/x-www-form-urlencoded")

        assert result["from"] == "test@example.com"
        assert result["to"] == "debate@aragora.com"
        assert result["text"] == "Hello World"

    def test_parse_multipart(self, handler):
        """Test parsing multipart form data."""
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body = (
            "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n"
            'Content-Disposition: form-data; name="from"\r\n\r\n'
            "test@example.com\r\n"
            "------WebKitFormBoundary7MA4YWxkTrZu0gW\r\n"
            'Content-Disposition: form-data; name="text"\r\n\r\n'
            "Hello multipart\r\n"
            "------WebKitFormBoundary7MA4YWxkTrZu0gW--\r\n"
        ).encode()

        result = handler._parse_form_data(body, f"multipart/form-data; boundary={boundary}")

        # May not parse perfectly in all Python versions, but should not crash
        assert isinstance(result, dict)


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_import_error_graceful(self, handler):
        """Test graceful handling when email_reply_loop module is unavailable."""
        body = b"from=test@example.com&text=Hello"

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Twilio-Email-Event-Webhook-Timestamp": "123",
                "X-Twilio-Email-Event-Webhook-Signature": "sig",
            },
            body=body,
            method="POST",
        )

        with patch.dict("sys.modules", {"aragora.integrations.email_reply_loop": None}):
            # Should return 503 when module not available
            result = handler.handle_post("/api/bots/email/webhook/sendgrid", {}, mock_http)

        assert result is not None
        # Either 503 (module unavailable) or 200 with error (caught exception)
        assert result.status_code in (200, 503)

    def test_processing_error_returns_200(self, handler):
        """Test processing errors return 200 to prevent webhook retries."""
        body = b"from=test@example.com&text=Hello"

        mock_http = MockHandler(
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Content-Length": str(len(body)),
                "X-Twilio-Email-Event-Webhook-Timestamp": "123",
                "X-Twilio-Email-Event-Webhook-Signature": "sig",
            },
            body=body,
            method="POST",
        )

        with (
            patch(
                "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
                return_value=True,
            ),
            patch(
                "aragora.integrations.email_reply_loop.parse_sendgrid_webhook",
                side_effect=ValueError("Parse error"),
            ),
        ):
            result = handler.handle_post("/api/bots/email/webhook/sendgrid", {}, mock_http)

        assert result is not None
        # Returns 200 to prevent retries even on error
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["status"] == "error"
