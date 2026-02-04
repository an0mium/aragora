"""
Tests for Email Webhook HTTP Handler.

Tests for:
- SendGrid Inbound Parse handling
- SendGrid Event webhook handling
- Mailgun webhook handling
- AWS SES SNS notification handling
- Signature verification
- Error handling
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.integrations.email_webhook import (
    EmailWebhookHandler,
    _extract_header,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create email webhook handler."""
    return EmailWebhookHandler()


@pytest.fixture
def mock_request():
    """Create a mock aiohttp request."""
    request = MagicMock()
    request.headers = {}
    return request


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestExtractHeader:
    """Tests for _extract_header helper."""

    def test_extract_existing_header(self):
        """Test extracting an existing header."""
        headers = "From: test@example.com\nSubject: Test\nMessage-ID: <abc123>"
        assert _extract_header(headers, "Message-ID") == "<abc123>"
        assert _extract_header(headers, "From") == "test@example.com"

    def test_extract_case_insensitive(self):
        """Test case-insensitive header extraction."""
        headers = "MESSAGE-ID: <abc123>\nContent-Type: text/plain"
        assert _extract_header(headers, "message-id") == "<abc123>"
        assert _extract_header(headers, "Message-Id") == "<abc123>"

    def test_extract_missing_header(self):
        """Test extracting non-existent header."""
        headers = "From: test@example.com"
        assert _extract_header(headers, "Message-ID") == ""


# =============================================================================
# SendGrid Tests
# =============================================================================


class TestSendGridWebhook:
    """Tests for SendGrid webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_sendgrid_inbound(self, handler, mock_request):
        """Test handling SendGrid Inbound Parse webhook."""
        mock_request.headers = {"Content-Type": "multipart/form-data; boundary=----"}

        # Mock the post data
        form_data = MagicMock()
        form_data.get.side_effect = lambda key, default="": {
            "from": "sender@example.com",
            "to": "recipient@aragora.ai",
            "subject": "Test Email",
            "text": "Hello, this is a test email.",
            "html": "<p>Hello, this is a test email.</p>",
            "envelope": '{"to": ["recipient@aragora.ai"]}',
            "headers": "Message-ID: <test123>\nIn-Reply-To: <prev456>",
        }.get(key, default)

        mock_request.post = AsyncMock(return_value=form_data)

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new_callable=AsyncMock,
            return_value={"debate_id": "debate_123"},
        ):
            response = await handler.handle_sendgrid(mock_request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["status"] == "processed"
            assert "message_id" in body

    @pytest.mark.asyncio
    async def test_handle_sendgrid_event(self, handler, mock_request):
        """Test handling SendGrid Event webhook."""
        mock_request.headers = {
            "Content-Type": "application/json",
            "X-Twilio-Email-Event-Webhook-Signature": "",
            "X-Twilio-Email-Event-Webhook-Timestamp": "",
        }

        events = [
            {"event": "delivered", "email": "recipient@example.com"},
            {"event": "open", "email": "recipient@example.com"},
        ]
        mock_request.read = AsyncMock(return_value=json.dumps(events).encode())

        response = await handler._handle_sendgrid_event(mock_request)

        assert response.status == 200
        body = json.loads(response.body)
        assert body["events_processed"] == 2

    @pytest.mark.asyncio
    async def test_handle_sendgrid_bounce_event(self, handler, mock_request):
        """Test handling SendGrid bounce event."""
        mock_request.headers = {"Content-Type": "application/json"}

        events = [
            {"event": "bounce", "email": "bounced@example.com", "reason": "mailbox full"},
        ]
        mock_request.read = AsyncMock(return_value=json.dumps(events).encode())

        response = await handler._handle_sendgrid_event(mock_request)

        assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_sendgrid_unsupported_content_type(self, handler, mock_request):
        """Test handling unsupported content type."""
        mock_request.headers = {"Content-Type": "text/plain"}

        response = await handler.handle_sendgrid(mock_request)

        assert response.status == 415


# =============================================================================
# Mailgun Tests
# =============================================================================


class TestMailgunWebhook:
    """Tests for Mailgun webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_mailgun_valid(self, handler, mock_request):
        """Test handling valid Mailgun webhook."""
        form_data = MagicMock()
        form_data.get.side_effect = lambda key, default="": {
            "sender": "sender@example.com",
            "recipient": "recipient@aragora.ai",
            "subject": "Test Email",
            "body-plain": "Hello from Mailgun",
            "body-html": "<p>Hello from Mailgun</p>",
            "timestamp": "1234567890",
            "token": "test_token",
            "signature": "test_signature",
            "Message-Id": "<mailgun123>",
            "In-Reply-To": "",
            "References": "",
        }.get(key, default)

        mock_request.post = AsyncMock(return_value=form_data)

        with patch(
            "aragora.integrations.email_reply_loop.verify_mailgun_signature",
            return_value=True,
        ):
            with patch(
                "aragora.integrations.email_reply_loop.process_inbound_email",
                new_callable=AsyncMock,
                return_value={"debate_id": "debate_456"},
            ):
                response = await handler.handle_mailgun(mock_request)

                assert response.status == 200
                body = json.loads(response.body)
                assert body["status"] == "processed"

    @pytest.mark.asyncio
    async def test_handle_mailgun_invalid_signature(self, handler, mock_request):
        """Test handling Mailgun webhook with invalid signature."""
        form_data = MagicMock()
        form_data.get.side_effect = lambda key, default="": {
            "timestamp": "1234567890",
            "token": "test_token",
            "signature": "bad_signature",
        }.get(key, default)

        mock_request.post = AsyncMock(return_value=form_data)

        with patch(
            "aragora.integrations.email_reply_loop.verify_mailgun_signature",
            return_value=False,
        ):
            response = await handler.handle_mailgun(mock_request)

            assert response.status == 401


# =============================================================================
# AWS SES Tests
# =============================================================================


class TestSESWebhook:
    """Tests for AWS SES webhook handling."""

    @pytest.mark.asyncio
    async def test_handle_ses_subscription_confirmation(self, handler, mock_request):
        """Test handling SNS subscription confirmation."""
        message = {
            "Type": "SubscriptionConfirmation",
            "SubscribeURL": "https://sns.us-east-1.amazonaws.com/...",
        }
        mock_request.read = AsyncMock(return_value=json.dumps(message).encode())

        with patch("aiohttp.ClientSession") as MockSession:
            # Create proper async context manager mocks
            mock_resp = MagicMock()
            mock_resp.status = 200

            # Response context manager
            class MockRespContext:
                async def __aenter__(self):
                    return mock_resp

                async def __aexit__(self, *args):
                    pass

            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=MockRespContext())

            # Session context manager
            class MockSessionContext:
                async def __aenter__(self):
                    return mock_session

                async def __aexit__(self, *args):
                    pass

            MockSession.return_value = MockSessionContext()

            response = await handler.handle_ses(mock_request)

            assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_ses_suspicious_subscription_url(self, handler, mock_request):
        """Test rejecting suspicious subscription URL."""
        message = {
            "Type": "SubscriptionConfirmation",
            "SubscribeURL": "https://evil.com/steal-data",
        }
        mock_request.read = AsyncMock(return_value=json.dumps(message).encode())

        response = await handler.handle_ses(mock_request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_handle_ses_notification(self, handler, mock_request):
        """Test handling SES notification."""
        message = {
            "Type": "Notification",
            "Message": json.dumps(
                {
                    "notificationType": "Delivery",
                    "mail": {"messageId": "test123"},
                }
            ),
        }
        mock_request.read = AsyncMock(return_value=json.dumps(message).encode())

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ):
            response = await handler.handle_ses(mock_request)

            assert response.status == 200

    @pytest.mark.asyncio
    async def test_handle_ses_bounce(self, handler, mock_request):
        """Test handling SES bounce notification."""
        message = {
            "Type": "Notification",
            "Message": json.dumps(
                {
                    "notificationType": "Bounce",
                    "bounce": {
                        "bounceType": "Permanent",
                        "bouncedRecipients": [{"emailAddress": "bounced@example.com"}],
                    },
                }
            ),
        }
        mock_request.read = AsyncMock(return_value=json.dumps(message).encode())

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ):
            response = await handler.handle_ses(mock_request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["type"] == "bounce"

    @pytest.mark.asyncio
    async def test_handle_ses_invalid_json(self, handler, mock_request):
        """Test handling invalid JSON in SES webhook."""
        mock_request.read = AsyncMock(return_value=b"not json")

        response = await handler.handle_ses(mock_request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_handle_ses_unknown_type(self, handler, mock_request):
        """Test handling unknown SES message type."""
        message = {"Type": "UnknownType"}
        mock_request.read = AsyncMock(return_value=json.dumps(message).encode())

        response = await handler.handle_ses(mock_request)

        assert response.status == 400


# =============================================================================
# Statistics Tests
# =============================================================================


class TestHandlerStats:
    """Tests for handler statistics."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self, handler, mock_request):
        """Test that stats are tracked correctly."""
        assert handler._processed_count == 0
        assert handler._error_count == 0

        mock_request.headers = {"Content-Type": "multipart/form-data"}
        form_data = MagicMock()
        form_data.get.side_effect = lambda key, default="": {
            "from": "sender@example.com",
            "to": "recipient@aragora.ai",
            "subject": "Test",
            "text": "Test body",
        }.get(key, default)
        mock_request.post = AsyncMock(return_value=form_data)

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new_callable=AsyncMock,
            return_value={},
        ):
            await handler.handle_sendgrid(mock_request)

            stats = handler.get_stats()
            assert stats["processed_count"] == 1
            assert stats["error_count"] == 0

    def test_get_stats(self, handler):
        """Test get_stats returns correct structure."""
        stats = handler.get_stats()

        assert "processed_count" in stats
        assert "error_count" in stats
        assert "last_error" in stats
