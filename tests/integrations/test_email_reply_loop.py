"""
Tests for aragora.integrations.email_reply_loop - Email Reply Loop processing.

Tests cover:
- InboundEmail dataclass and parsing
- Debate ID extraction from headers/subject
- Email body cleaning (quote removal)
- SendGrid webhook parsing
- SES notification parsing
- Email reply processing
- Signature verification
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.integrations.email_reply_loop import (
    InboundEmail,
    EmailReplyOrigin,
    parse_raw_email,
    parse_sendgrid_webhook,
    parse_ses_notification,
    verify_sendgrid_signature,
    verify_ses_signature,
    process_inbound_email,
    handle_email_reply,
    register_email_origin,
    register_reply_handler,
    setup_email_reply_loop,
)


# ===========================================================================
# InboundEmail Tests
# ===========================================================================


class TestInboundEmail:
    """Tests for InboundEmail dataclass."""

    def test_create_inbound_email(self):
        """Test creating an InboundEmail instance."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Re: Test Subject",
            body_plain="This is the reply content.",
            body_html="<p>This is the reply content.</p>",
            headers={"References": "<original@aragora.com>"},
            received_at=datetime.now(),
        )

        assert email.message_id == "msg-123"
        assert email.from_email == "user@example.com"
        assert "debate@aragora.com" in email.to_email
        assert email.subject == "Re: Test Subject"
        assert email.body_plain == "This is the reply content."

    def test_debate_id_from_subject(self):
        """Test extracting debate ID from subject line."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Re: [Aragora Debate-id:abc123] AI Ethics Discussion",
            body_plain="My reply",
            headers={},
        )

        assert email.debate_id == "abc123"

    def test_debate_id_from_in_reply_to(self):
        """Test extracting debate ID from In-Reply-To header."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Re: Topic",
            body_plain="My reply",
            headers={},
            in_reply_to="<debate_id=def456@aragora.example.com>",
        )

        assert email.debate_id == "def456"

    def test_debate_id_from_references(self):
        """Test extracting debate ID from References header."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Re: Discussion",  # No debate ID in subject
            body_plain="My reply",
            headers={},
            references=["<debate-id:xyz789@aragora.example.com>"],
        )

        assert email.debate_id == "xyz789"

    def test_debate_id_from_custom_header(self):
        """Test extracting debate ID from X-Aragora-Debate-Id header."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Re: Discussion",
            body_plain="My reply",
            headers={"X-Aragora-Debate-Id": "custom-debate-id"},
        )

        assert email.debate_id == "custom-debate-id"

    def test_debate_id_none_when_not_found(self):
        """Test debate ID is None when not found."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Random Subject",
            body_plain="Random email",
            headers={},
        )

        assert email.debate_id is None

    def test_cleaned_body_removes_quotes(self):
        """Test cleaned_body removes quoted reply content."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Re: Test",
            body_plain="My actual reply.\n\n> This is quoted text\n> from the original.",
            headers={},
        )

        cleaned = email.cleaned_body
        assert "My actual reply" in cleaned
        assert "> This is quoted text" not in cleaned

    def test_cleaned_body_removes_signature(self):
        """Test cleaned_body removes email signature."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Re: Test",
            body_plain="My reply content.\n\n--\nBest regards,\nUser Name",
            headers={},
        )

        cleaned = email.cleaned_body
        assert "My reply content" in cleaned
        assert "Best regards" not in cleaned

    def test_cleaned_body_removes_on_date_wrote(self):
        """Test cleaned_body removes 'On X wrote:' pattern."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Re: Test",
            body_plain="I agree!\n\nOn Mon, Jan 20, 2026 at 10:00 AM Someone wrote:\n> Original message",
            headers={},
        )

        cleaned = email.cleaned_body
        assert "I agree" in cleaned
        assert "> Original message" not in cleaned

    def test_to_dict(self):
        """Test to_dict conversion."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Test Subject",
            body_plain="Test body",
        )

        result = email.to_dict()

        assert result["message_id"] == "msg-123"
        assert result["from_email"] == "user@example.com"
        assert result["to_email"] == "debate@aragora.com"
        assert result["subject"] == "Test Subject"
        assert "received_at" in result


# ===========================================================================
# EmailReplyOrigin Tests
# ===========================================================================


class TestEmailReplyOrigin:
    """Tests for EmailReplyOrigin tracking."""

    def test_create_reply_origin(self):
        """Test creating an EmailReplyOrigin."""
        origin = EmailReplyOrigin(
            debate_id="debate-123",
            message_id="<sent-msg@aragora.com>",
            recipient_email="user@example.com",
            recipient_name="Test User",
        )

        assert origin.debate_id == "debate-123"
        assert origin.recipient_email == "user@example.com"
        assert origin.message_id == "<sent-msg@aragora.com>"
        assert origin.recipient_name == "Test User"
        assert origin.reply_received is False

    def test_register_email_origin(self):
        """Test registering an email origin."""
        origin = register_email_origin(
            debate_id="debate-abc",
            message_id="<test-msg@aragora.com>",
            recipient_email="test@example.com",
            metadata={"extra": "data"},
        )

        assert origin is not None
        assert origin.debate_id == "debate-abc"
        assert origin.message_id == "<test-msg@aragora.com>"
        assert origin.metadata["extra"] == "data"

    def test_to_dict(self):
        """Test EmailReplyOrigin serialization to dict."""
        origin = EmailReplyOrigin(
            debate_id="debate-456",
            message_id="<dict-test@aragora.com>",
            recipient_email="dict@example.com",
            recipient_name="Dict User",
            metadata={"key": "value"},
        )

        result = origin.to_dict()

        assert result["debate_id"] == "debate-456"
        assert result["message_id"] == "<dict-test@aragora.com>"
        assert result["recipient_email"] == "dict@example.com"
        assert result["recipient_name"] == "Dict User"
        assert result["reply_received"] is False
        assert result["metadata"]["key"] == "value"
        assert "sent_at" in result

    def test_from_dict(self):
        """Test EmailReplyOrigin deserialization from dict."""
        data = {
            "debate_id": "debate-789",
            "message_id": "<from-dict@aragora.com>",
            "recipient_email": "from@example.com",
            "recipient_name": "From User",
            "sent_at": "2026-01-20T10:00:00",
            "reply_received": True,
            "reply_received_at": "2026-01-20T11:00:00",
            "metadata": {"restored": True},
        }

        origin = EmailReplyOrigin.from_dict(data)

        assert origin.debate_id == "debate-789"
        assert origin.message_id == "<from-dict@aragora.com>"
        assert origin.recipient_email == "from@example.com"
        assert origin.reply_received is True
        assert origin.metadata["restored"] is True

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = EmailReplyOrigin(
            debate_id="roundtrip-123",
            message_id="<roundtrip@aragora.com>",
            recipient_email="roundtrip@example.com",
            recipient_name="Roundtrip User",
            metadata={"round": "trip"},
        )

        data = original.to_dict()
        restored = EmailReplyOrigin.from_dict(data)

        assert restored.debate_id == original.debate_id
        assert restored.message_id == original.message_id
        assert restored.recipient_email == original.recipient_email
        assert restored.recipient_name == original.recipient_name
        assert restored.metadata == original.metadata


# ===========================================================================
# Redis Persistence Tests
# ===========================================================================


class TestEmailOriginRedisPersistence:
    """Tests for Redis persistence of email origins."""

    @patch("aragora.integrations.email_reply_loop._store_email_origin_redis")
    def test_register_origin_stores_to_redis(self, mock_store_redis):
        """Test that registering an origin attempts Redis storage."""
        origin = register_email_origin(
            debate_id="redis-test-1",
            message_id="<redis-store@aragora.com>",
            recipient_email="redis@example.com",
        )

        # Should attempt to store in Redis
        mock_store_redis.assert_called_once()
        call_args = mock_store_redis.call_args[0]
        assert call_args[0].debate_id == "redis-test-1"

    @patch("aragora.integrations.email_reply_loop._store_email_origin_redis")
    def test_register_origin_continues_on_redis_failure(self, mock_store_redis):
        """Test that registration succeeds even if Redis fails."""
        mock_store_redis.side_effect = Exception("Redis unavailable")

        # Should not raise, just log warning
        origin = register_email_origin(
            debate_id="redis-fail-test",
            message_id="<redis-fail@aragora.com>",
            recipient_email="fail@example.com",
        )

        assert origin is not None
        assert origin.debate_id == "redis-fail-test"

    @pytest.mark.asyncio
    @patch("aragora.integrations.email_reply_loop._load_email_origin_redis")
    async def test_get_origin_falls_back_to_redis(self, mock_load_redis):
        """Test that get_origin_by_reply checks Redis when not in memory."""
        from aragora.integrations.email_reply_loop import get_origin_by_reply, _reply_origins

        # Ensure not in memory
        test_msg_id = "<redis-fallback@aragora.com>"
        _reply_origins.pop(test_msg_id, None)

        # Mock Redis returning an origin
        mock_origin = EmailReplyOrigin(
            debate_id="redis-loaded",
            message_id=test_msg_id,
            recipient_email="loaded@example.com",
        )
        mock_load_redis.return_value = mock_origin

        result = await get_origin_by_reply(test_msg_id)

        assert result is not None
        assert result.debate_id == "redis-loaded"
        mock_load_redis.assert_called_once_with(test_msg_id)

    @pytest.mark.asyncio
    @patch("aragora.integrations.email_reply_loop._load_email_origin_redis")
    async def test_get_origin_caches_redis_result(self, mock_load_redis):
        """Test that Redis results are cached in memory."""
        from aragora.integrations.email_reply_loop import get_origin_by_reply, _reply_origins

        test_msg_id = "<cache-test@aragora.com>"
        _reply_origins.pop(test_msg_id, None)

        mock_origin = EmailReplyOrigin(
            debate_id="cached",
            message_id=test_msg_id,
            recipient_email="cache@example.com",
        )
        mock_load_redis.return_value = mock_origin

        # First call should hit Redis
        result1 = await get_origin_by_reply(test_msg_id)
        assert mock_load_redis.call_count == 1

        # Second call should use cache
        result2 = await get_origin_by_reply(test_msg_id)
        assert mock_load_redis.call_count == 1  # Not called again

        assert result1.debate_id == result2.debate_id

    @pytest.mark.asyncio
    async def test_get_origin_prefers_memory(self):
        """Test that in-memory origins are returned without Redis call."""
        from aragora.integrations.email_reply_loop import get_origin_by_reply, _reply_origins

        test_msg_id = "<memory-pref@aragora.com>"
        memory_origin = EmailReplyOrigin(
            debate_id="from-memory",
            message_id=test_msg_id,
            recipient_email="memory@example.com",
        )
        _reply_origins[test_msg_id] = memory_origin

        with patch("aragora.integrations.email_reply_loop._load_email_origin_redis") as mock_load:
            result = await get_origin_by_reply(test_msg_id)

            assert result.debate_id == "from-memory"
            mock_load.assert_not_called()

        # Cleanup
        _reply_origins.pop(test_msg_id, None)


# ===========================================================================
# Parse Raw Email Tests
# ===========================================================================


class TestParseRawEmail:
    """Tests for parsing raw email content."""

    def test_parse_simple_email(self):
        """Test parsing a simple email."""
        raw = (
            b"From: sender@example.com\r\n"
            b"To: recipient@aragora.com\r\n"
            b"Subject: Test Email\r\n"
            b"Message-ID: <raw-msg-123@example.com>\r\n"
            b"\r\n"
            b"This is the email body."
        )

        email = parse_raw_email(raw)

        assert email.from_email == "sender@example.com"
        assert "recipient@aragora.com" in email.to_email
        assert email.subject == "Test Email"
        assert email.message_id == "<raw-msg-123@example.com>"
        assert "This is the email body" in email.body_plain

    def test_parse_email_with_name(self):
        """Test parsing email with display name."""
        raw = (
            b"From: Sender Name <sender@example.com>\r\n"
            b"To: recipient@aragora.com\r\n"
            b"Subject: Named Sender Test\r\n"
            b"Message-ID: <named-123@example.com>\r\n"
            b"\r\n"
            b"Email body."
        )

        email = parse_raw_email(raw)

        assert email.from_email == "sender@example.com"
        assert email.from_name == "Sender Name"

    def test_parse_multipart_email(self):
        """Test parsing a multipart email with text and HTML."""
        raw = (
            b"From: sender@example.com\r\n"
            b"To: recipient@aragora.com\r\n"
            b"Subject: Multipart Test\r\n"
            b"Message-ID: <multi-123@example.com>\r\n"
            b"Content-Type: multipart/alternative; boundary=boundary123\r\n"
            b"\r\n"
            b"--boundary123\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n"
            b"Plain text body.\r\n"
            b"--boundary123\r\n"
            b"Content-Type: text/html\r\n"
            b"\r\n"
            b"<p>HTML body.</p>\r\n"
            b"--boundary123--\r\n"
        )

        email = parse_raw_email(raw)

        assert email.from_email == "sender@example.com"
        assert "Plain text body" in email.body_plain
        assert "<p>HTML body.</p>" in email.body_html


# ===========================================================================
# Parse SendGrid Webhook Tests
# ===========================================================================


class TestParseSendGridWebhook:
    """Tests for parsing SendGrid Inbound Parse webhook data."""

    def test_parse_sendgrid_webhook(self):
        """Test parsing SendGrid webhook form data."""
        data = {
            "from": "User Name <user@example.com>",
            "to": "debate@aragora.com",
            "subject": "Re: [Aragora Debate_id:sg123] Topic",
            "text": "My SendGrid reply.",
            "html": "<p>My SendGrid reply.</p>",
            "headers": "Message-ID: <sg-msg-456@example.com>\r\nDate: Mon, 20 Jan 2026 10:00:00 +0000",
            "envelope": json.dumps(
                {
                    "from": "user@example.com",
                    "to": ["debate@aragora.com"],
                }
            ),
        }

        email = parse_sendgrid_webhook(data)

        assert "user@example.com" in email.from_email
        assert email.subject == "Re: [Aragora Debate_id:sg123] Topic"
        assert email.body_plain == "My SendGrid reply."
        assert email.debate_id == "sg123"

    def test_parse_sendgrid_minimal_data(self):
        """Test parsing SendGrid webhook with minimal data."""
        data = {
            "from": "user@example.com",
            "text": "Just a message",
        }

        email = parse_sendgrid_webhook(data)

        assert email.from_email == "user@example.com"
        assert email.body_plain == "Just a message"
        assert email.message_id  # Should have generated ID

    def test_parse_sendgrid_with_headers(self):
        """Test parsing SendGrid webhook extracts headers."""
        data = {
            "from": "user@example.com",
            "text": "Reply text",
            "headers": (
                "Message-ID: <custom-id@example.com>\r\n"
                "In-Reply-To: <original@aragora.com>\r\n"
                "References: <ref1@aragora.com> <ref2@aragora.com>"
            ),
        }

        email = parse_sendgrid_webhook(data)

        assert email.message_id == "<custom-id@example.com>"
        assert email.in_reply_to == "<original@aragora.com>"
        assert len(email.references) == 2


# ===========================================================================
# Parse SES Notification Tests
# ===========================================================================


class TestParseSESNotification:
    """Tests for parsing AWS SES SNS notifications."""

    def test_parse_ses_notification_with_content(self):
        """Test parsing SES notification with full email content."""
        notification = {
            "Type": "Notification",
            "Message": json.dumps(
                {
                    "notificationType": "Received",
                    "mail": {
                        "messageId": "ses-abc123",
                        "source": "sender@example.com",
                        "destination": ["debate@aragora.com"],
                        "commonHeaders": {
                            "from": ["Sender Name <sender@example.com>"],
                            "to": ["debate@aragora.com"],
                            "subject": "Re: [Aragora Debate_id:ses456] Discussion",
                        },
                    },
                    "content": (
                        "From: sender@example.com\r\n"
                        "To: debate@aragora.com\r\n"
                        "Subject: Re: [Aragora Debate_id:ses456] Discussion\r\n"
                        "Message-ID: <ses-content@example.com>\r\n"
                        "\r\n"
                        "SES email content."
                    ),
                }
            ),
        }

        email = parse_ses_notification(notification)

        assert email is not None
        assert "sender@example.com" in email.from_email
        assert "SES email content" in email.body_plain

    def test_parse_ses_notification_headers_only(self):
        """Test parsing SES notification without full content."""
        notification = {
            "Type": "Notification",
            "Message": json.dumps(
                {
                    "notificationType": "Received",
                    "mail": {
                        "messageId": "ses-headers-only",
                        "source": "sender@example.com",
                        "destination": ["debate@aragora.com"],
                        "commonHeaders": {
                            "from": ["sender@example.com"],
                            "to": ["debate@aragora.com"],
                            "subject": "Test Subject",
                        },
                        "headers": [
                            {"name": "Message-ID", "value": "<ses-hdr@example.com>"},
                            {"name": "In-Reply-To", "value": "<original@aragora.com>"},
                        ],
                    },
                }
            ),
        }

        email = parse_ses_notification(notification)

        assert email is not None
        assert email.message_id == "ses-headers-only"

    def test_parse_ses_non_received_returns_none(self):
        """Test that non-Received notifications return None."""
        notification = {
            "Type": "Notification",
            "Message": json.dumps(
                {
                    "notificationType": "Bounce",
                    "bounce": {"bounceType": "Permanent"},
                }
            ),
        }

        result = parse_ses_notification(notification)
        assert result is None

    def test_parse_ses_subscription_confirmation(self):
        """Test that subscription confirmation returns None."""
        notification = {
            "Type": "SubscriptionConfirmation",
            "SubscribeURL": "https://example.com/confirm",
        }

        result = parse_ses_notification(notification)
        assert result is None


# ===========================================================================
# Signature Verification Tests
# ===========================================================================


class TestSignatureVerification:
    """Tests for webhook signature verification."""

    def test_verify_sendgrid_signature_no_secret(self):
        """Test SendGrid verification passes when no secret configured."""
        result = verify_sendgrid_signature(b"payload", "timestamp", "signature")
        # Should pass when no secret is configured (development mode)
        assert result is True

    def test_verify_ses_signature_valid(self):
        """Test SES verification with valid message structure."""
        notification = {
            "Type": "Notification",
            "MessageId": "msg-123",
            "TopicArn": "arn:aws:sns:us-east-1:123456:topic",
            "Timestamp": "2026-01-20T10:00:00.000Z",
        }

        result = verify_ses_signature(notification)
        assert result is True

    def test_verify_ses_signature_missing_fields(self):
        """Test SES verification fails with missing fields."""
        notification = {
            "Type": "Notification",
            # Missing required fields
        }

        result = verify_ses_signature(notification)
        assert result is False

    def test_verify_ses_signature_invalid_arn(self):
        """Test SES verification fails with invalid ARN."""
        notification = {
            "Type": "Notification",
            "MessageId": "msg-123",
            "TopicArn": "not-a-valid-arn",
            "Timestamp": "2026-01-20T10:00:00.000Z",
        }

        result = verify_ses_signature(notification)
        assert result is False


# ===========================================================================
# Email Reply Processing Tests
# ===========================================================================


class TestEmailReplyProcessing:
    """Tests for processing inbound email replies."""

    @pytest.mark.asyncio
    async def test_process_inbound_email_no_debate_id(self):
        """Test processing email without debate ID returns False."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Random Email",
            body_plain="This is not a debate reply.",
            headers={},
        )

        result = await process_inbound_email(email)

        # Should return False when no debate ID found
        assert result is False

    @pytest.mark.asyncio
    async def test_process_inbound_email_empty_body(self):
        """Test processing email with empty cleaned body returns False."""
        email = InboundEmail(
            message_id="msg-123",
            from_email="user@example.com",
            to_email="debate@aragora.com",
            subject="Re: [Aragora Debate_id:test123] Topic",
            body_plain="> All quoted text\n> No new content",
            headers={},
        )

        result = await process_inbound_email(email)

        # Should return False when cleaned body is empty
        assert result is False

    @pytest.mark.asyncio
    async def test_handle_email_reply_default_processing(self):
        """Test handle_email_reply falls through to process_inbound_email."""
        email = InboundEmail(
            message_id="msg-456",
            from_email="replier@example.com",
            to_email="debate@aragora.com",
            subject="Re: Test",
            body_plain="My reply content.",
            headers={},
        )

        # No debate ID, so should return False
        result = await handle_email_reply(email)
        assert result is False

    def test_register_reply_handler(self):
        """Test registering a custom reply handler."""
        handler_called = []

        def custom_handler(email: InboundEmail) -> bool:
            handler_called.append(email.message_id)
            return True

        register_reply_handler(custom_handler)

        # Handler should be registered
        from aragora.integrations.email_reply_loop import _reply_handlers

        assert custom_handler in _reply_handlers


# ===========================================================================
# Setup Tests
# ===========================================================================


class TestSetup:
    """Tests for setup functions."""

    def test_setup_email_reply_loop(self):
        """Test setup_email_reply_loop runs without error."""
        # Should not raise
        setup_email_reply_loop()


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    """Integration tests for email reply loop."""

    def test_sendgrid_to_debate_flow(self):
        """Test complete flow from SendGrid webhook to debate ID extraction."""
        # Simulate SendGrid webhook data
        webhook_data = {
            "from": "participant@example.com",
            "to": "debates@aragora.com",
            "subject": "Re: [Aragora Debate-id:flow123] Climate Policy Analysis",
            "text": "I believe the evidence strongly supports action.\n\n> Previous message here",
            "headers": "Message-ID: <sendgrid-flow@example.com>",
        }

        # Parse the webhook
        email = parse_sendgrid_webhook(webhook_data)

        # Verify extraction
        assert email.debate_id == "flow123"
        assert "I believe the evidence" in email.cleaned_body
        assert "> Previous message" not in email.cleaned_body

    def test_ses_to_debate_flow(self):
        """Test complete flow from SES notification to debate ID extraction."""
        # Simulate SES notification
        notification = {
            "Type": "Notification",
            "Message": json.dumps(
                {
                    "notificationType": "Received",
                    "mail": {
                        "messageId": "ses-flow-456",
                        "source": "participant@example.com",
                        "destination": ["debates@aragora.com"],
                        "commonHeaders": {
                            "from": ["participant@example.com"],
                            "to": ["debates@aragora.com"],
                            "subject": "Re: [Aragora Debate_id:sesflow789] AI Safety",
                        },
                    },
                    "content": (
                        "From: participant@example.com\r\n"
                        "Subject: Re: [Aragora Debate_id:sesflow789] AI Safety\r\n"
                        "Message-ID: <ses-flow@example.com>\r\n"
                        "\r\n"
                        "The safety implications are significant."
                    ),
                }
            ),
        }

        # Parse the notification
        email = parse_ses_notification(notification)

        # Verify extraction
        assert email is not None
        assert email.debate_id == "sesflow789"
        assert "safety implications" in email.body_plain.lower()
