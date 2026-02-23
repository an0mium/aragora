"""Tests for email webhook handler (aragora/server/handlers/integrations/email_webhook.py).

Covers all endpoints and behavior:
- SendGrid Inbound Parse (multipart/form-data)
- SendGrid Event Webhook (JSON array of events)
- Mailgun webhook (multipart/form-data with signature)
- AWS SES SNS notifications (Subscription, Notification, email receipt)
- SES bounce/complaint feedback handling
- Status/stats endpoint
- Utility functions: _extract_header, _parse_email_address
- register_email_webhook_routes helper
- Error handling, validation, edge cases
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.integrations.email_webhook import (
    EmailWebhookHandler,
    _extract_header,
    _parse_email_address,
    register_email_webhook_routes,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract the parsed body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("body", result)
    return json.loads(result.body.decode("utf-8"))


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    return result.status_code


# ---------------------------------------------------------------------------
# Mock request builders
# ---------------------------------------------------------------------------


class MockMultipartData(dict):
    """Dict subclass that mimics aiohttp multipart post() result."""
    pass


class MockAiohttpRequest:
    """Mock aiohttp request for testing email webhook handlers."""

    def __init__(
        self,
        *,
        content_type: str = "application/json",
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        post_data: dict[str, Any] | None = None,
    ):
        self.headers = headers or {}
        if "Content-Type" not in self.headers:
            self.headers["Content-Type"] = content_type
        self._body = body or b"{}"
        self._post_data = MockMultipartData(post_data or {})

    async def read(self) -> bytes:
        return self._body

    async def post(self) -> MockMultipartData:
        return self._post_data

    async def json(self) -> Any:
        return json.loads(self._body)


def _make_sendgrid_inbound_request(
    *,
    sender: str = "Alice <alice@example.com>",
    to: str = "debate@aragora.com",
    subject: str = "Test debate",
    text: str = "This is a test email body",
    html: str = "<p>This is a test email body</p>",
    envelope: str | None = None,
    headers_raw: str = "",
) -> MockAiohttpRequest:
    """Build a mock SendGrid Inbound Parse request."""
    post = {
        "from": sender,
        "to": to,
        "subject": subject,
        "text": text,
        "html": html,
        "envelope": envelope or json.dumps({"to": [to], "from": sender}),
        "headers": headers_raw,
    }
    return MockAiohttpRequest(
        content_type="multipart/form-data; boundary=----abc123",
        post_data=post,
    )


def _make_sendgrid_event_request(
    events: list[dict[str, Any]] | None = None,
    *,
    signature: str = "",
    timestamp: str = "",
) -> MockAiohttpRequest:
    """Build a mock SendGrid Event Webhook request."""
    body = json.dumps(events or [{"event": "delivered", "email": "test@example.com"}]).encode()
    headers = {"Content-Type": "application/json"}
    if signature:
        headers["X-Twilio-Email-Event-Webhook-Signature"] = signature
    if timestamp:
        headers["X-Twilio-Email-Event-Webhook-Timestamp"] = timestamp
    return MockAiohttpRequest(content_type="application/json", headers=headers, body=body)


def _make_mailgun_request(
    *,
    sender: str = "Bob <bob@example.com>",
    recipient: str = "debate@aragora.com",
    subject: str = "Mailgun test",
    body_plain: str = "Mailgun body",
    body_html: str = "<p>Mailgun body</p>",
    message_id: str = "<msg-001@mailgun.com>",
    in_reply_to: str = "",
    references: str = "",
    mg_timestamp: str = "1234567890",
    mg_token: str = "token123",
    mg_signature: str = "sig123",
) -> MockAiohttpRequest:
    """Build a mock Mailgun webhook request."""
    post = {
        "sender": sender,
        "recipient": recipient,
        "subject": subject,
        "body-plain": body_plain,
        "body-html": body_html,
        "Message-Id": message_id,
        "In-Reply-To": in_reply_to,
        "References": references,
        "timestamp": mg_timestamp,
        "token": mg_token,
        "signature": mg_signature,
    }
    return MockAiohttpRequest(
        content_type="multipart/form-data; boundary=----xyz",
        post_data=post,
    )


def _make_ses_request(message: dict[str, Any]) -> MockAiohttpRequest:
    """Build a mock SES SNS notification request."""
    body = json.dumps(message).encode()
    return MockAiohttpRequest(content_type="application/json", body=body)


# ---------------------------------------------------------------------------
# Mock InboundEmail to avoid importing the real one (has dependencies)
# ---------------------------------------------------------------------------


@dataclass
class MockInboundEmail:
    """Mock InboundEmail matching the real dataclass interface."""

    message_id: str = ""
    from_email: str = ""
    from_name: str = ""
    to_email: str = ""
    subject: str = ""
    body_plain: str = ""
    body_html: str = ""
    in_reply_to: str = ""
    references: list[str] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    attachments: list[dict[str, Any]] = field(default_factory=list)
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_data: bytes | None = None

    @property
    def debate_id(self) -> str | None:
        return self.headers.get("X-Aragora-Debate-Id")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler() -> EmailWebhookHandler:
    """Create a fresh EmailWebhookHandler instance."""
    return EmailWebhookHandler()


# ---------------------------------------------------------------------------
# Tests: Utility functions
# ---------------------------------------------------------------------------


class TestExtractHeader:
    """Tests for the _extract_header utility."""

    def test_extract_existing_header(self):
        raw = "From: alice@example.com\nMessage-ID: <abc123>\nSubject: test"
        assert _extract_header(raw, "Message-ID") == "<abc123>"

    def test_extract_header_case_insensitive(self):
        raw = "message-id: <abc123>"
        assert _extract_header(raw, "Message-ID") == "<abc123>"

    def test_extract_missing_header(self):
        raw = "From: alice@example.com\nSubject: test"
        assert _extract_header(raw, "Message-ID") == ""

    def test_extract_header_empty_string(self):
        assert _extract_header("", "Message-ID") == ""

    def test_extract_header_with_colon_in_value(self):
        raw = "References: <ref1:abc@example.com>"
        assert _extract_header(raw, "References") == "<ref1:abc@example.com>"

    def test_extract_in_reply_to(self):
        raw = "In-Reply-To: <reply-123@example.com>"
        assert _extract_header(raw, "In-Reply-To") == "<reply-123@example.com>"

    def test_extract_references_multi_value(self):
        raw = "References: <ref1@ex.com> <ref2@ex.com>"
        assert _extract_header(raw, "References") == "<ref1@ex.com> <ref2@ex.com>"


class TestParseEmailAddress:
    """Tests for the _parse_email_address utility."""

    def test_name_angle_bracket_format(self):
        email, name = _parse_email_address("Alice Smith <alice@example.com>")
        assert email == "alice@example.com"
        assert name == "Alice Smith"

    def test_quoted_name_format(self):
        email, name = _parse_email_address('"Alice Smith" <alice@example.com>')
        assert email == "alice@example.com"
        assert name == "Alice Smith"

    def test_plain_email(self):
        email, name = _parse_email_address("alice@example.com")
        assert email == "alice@example.com"
        assert name == ""

    def test_empty_string(self):
        email, name = _parse_email_address("")
        assert email == ""
        assert name == ""

    def test_no_match_returns_address_as_is(self):
        email, name = _parse_email_address("not-an-email")
        assert email == "not-an-email"
        assert name == ""

    def test_email_with_dots_and_dashes(self):
        email, name = _parse_email_address("Test User <test.user-01@sub.example.com>")
        assert email == "test.user-01@sub.example.com"
        assert name == "Test User"


# ---------------------------------------------------------------------------
# Tests: SendGrid Inbound Parse
# ---------------------------------------------------------------------------


class TestSendGridInbound:
    """Tests for SendGrid Inbound Parse webhook handling."""

    @pytest.mark.asyncio
    async def test_sendgrid_inbound_success(self, handler):
        """Process a valid SendGrid inbound email."""
        request = _make_sendgrid_inbound_request()
        mock_process = AsyncMock(return_value=True)

        with patch(
            "aragora.server.handlers.integrations.email_webhook._handle_sendgrid_inbound",
            new=mock_process,
            create=True,
        ):
            with patch(
                "aragora.integrations.email_reply_loop.process_inbound_email",
                new=mock_process,
            ):
                with patch(
                    "aragora.integrations.email_reply_loop.InboundEmail",
                    MockInboundEmail,
                ):
                    result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "processed"
        assert "message_id" in body

    @pytest.mark.asyncio
    async def test_sendgrid_inbound_process_failure_returns_202(self, handler):
        """Processing error still returns 2xx to prevent SendGrid retries."""
        request = _make_sendgrid_inbound_request()
        mock_process = AsyncMock(side_effect=RuntimeError("processing failed"))

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=mock_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 202
        body = _body(result)
        assert body["status"] == "error"
        assert handler._error_count == 1

    @pytest.mark.asyncio
    async def test_sendgrid_inbound_extracts_email_fields(self, handler):
        """Verify all email fields are extracted from multipart form data."""
        captured_email = {}

        async def capture_process(email_data):
            captured_email["from_email"] = email_data.from_email
            captured_email["from_name"] = email_data.from_name
            captured_email["to_email"] = email_data.to_email
            captured_email["subject"] = email_data.subject
            captured_email["body_plain"] = email_data.body_plain
            captured_email["body_html"] = email_data.body_html
            return True

        request = _make_sendgrid_inbound_request(
            sender="Test User <test@example.com>",
            to="debate@aragora.com",
            subject="My Debate Topic",
            text="Plain text body",
            html="<p>HTML body</p>",
        )

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        assert captured_email["from_email"] == "test@example.com"
        assert captured_email["from_name"] == "Test User"
        assert captured_email["to_email"] == "debate@aragora.com"
        assert captured_email["subject"] == "My Debate Topic"
        assert captured_email["body_plain"] == "Plain text body"
        assert captured_email["body_html"] == "<p>HTML body</p>"

    @pytest.mark.asyncio
    async def test_sendgrid_inbound_extracts_thread_headers(self, handler):
        """Verify Message-ID, In-Reply-To, and References are extracted."""
        captured_email = {}

        async def capture_process(email_data):
            captured_email["message_id"] = email_data.message_id
            captured_email["in_reply_to"] = email_data.in_reply_to
            captured_email["references"] = email_data.references
            return True

        headers_raw = (
            "Message-ID: <msg-42@example.com>\n"
            "In-Reply-To: <msg-41@example.com>\n"
            "References: <msg-40@example.com> <msg-41@example.com>"
        )
        request = _make_sendgrid_inbound_request(headers_raw=headers_raw)

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        assert captured_email["message_id"] == "<msg-42@example.com>"
        assert captured_email["in_reply_to"] == "<msg-41@example.com>"
        assert captured_email["references"] == ["<msg-40@example.com>", "<msg-41@example.com>"]

    @pytest.mark.asyncio
    async def test_sendgrid_inbound_invalid_envelope_json(self, handler):
        """Invalid envelope JSON is handled gracefully."""
        request = _make_sendgrid_inbound_request(envelope="{bad json}")
        mock_process = AsyncMock(return_value=True)

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=mock_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_sendgrid(request)

        # Should not crash, still processes
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_sendgrid_inbound_missing_optional_fields(self, handler):
        """Request with minimal fields is handled gracefully."""
        request = MockAiohttpRequest(
            content_type="multipart/form-data; boundary=----abc",
            post_data={"from": "sender@ex.com"},
        )
        mock_process = AsyncMock(return_value=True)

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=mock_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_sendgrid_inbound_increments_processed_count(self, handler):
        """Successful processing increments the processed counter."""
        assert handler._processed_count == 0
        request = _make_sendgrid_inbound_request()
        mock_process = AsyncMock(return_value=True)

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=mock_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            await handler.handle_sendgrid(request)

        assert handler._processed_count == 1


# ---------------------------------------------------------------------------
# Tests: SendGrid Event Webhook
# ---------------------------------------------------------------------------


class TestSendGridEvent:
    """Tests for SendGrid Event Webhook handling."""

    @pytest.mark.asyncio
    async def test_sendgrid_event_single_event(self, handler):
        """Process a single event in an array."""
        events = [{"event": "delivered", "email": "user@example.com"}]
        request = _make_sendgrid_event_request(events)

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "processed"
        assert body["events_processed"] == 1

    @pytest.mark.asyncio
    async def test_sendgrid_event_multiple_events(self, handler):
        """Process multiple events in a batch."""
        events = [
            {"event": "delivered", "email": "a@example.com"},
            {"event": "open", "email": "b@example.com"},
            {"event": "click", "email": "c@example.com"},
        ]
        request = _make_sendgrid_event_request(events)

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        assert _body(result)["events_processed"] == 3

    @pytest.mark.asyncio
    async def test_sendgrid_event_bounce_triggers_handler(self, handler):
        """Bounce event calls _handle_email_event."""
        events = [{"event": "bounce", "email": "bounced@example.com"}]
        request = _make_sendgrid_event_request(events)

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ), patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        mock_event.assert_called_once_with("bounce", "bounced@example.com", events[0])

    @pytest.mark.asyncio
    async def test_sendgrid_event_spamreport_triggers_handler(self, handler):
        """Spamreport event calls _handle_email_event."""
        events = [{"event": "spamreport", "email": "spam@example.com"}]
        request = _make_sendgrid_event_request(events)

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ), patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            result = await handler.handle_sendgrid(request)

        mock_event.assert_called_once_with("spamreport", "spam@example.com", events[0])

    @pytest.mark.asyncio
    async def test_sendgrid_event_unsubscribe_triggers_handler(self, handler):
        """Unsubscribe event calls _handle_email_event."""
        events = [{"event": "unsubscribe", "email": "unsub@example.com"}]
        request = _make_sendgrid_event_request(events)

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ), patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            result = await handler.handle_sendgrid(request)

        mock_event.assert_called_once_with("unsubscribe", "unsub@example.com", events[0])

    @pytest.mark.asyncio
    async def test_sendgrid_event_dropped_triggers_handler(self, handler):
        """Dropped event calls _handle_email_event."""
        events = [{"event": "dropped", "email": "dropped@example.com"}]
        request = _make_sendgrid_event_request(events)

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ), patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            result = await handler.handle_sendgrid(request)

        mock_event.assert_called_once_with("dropped", "dropped@example.com", events[0])

    @pytest.mark.asyncio
    async def test_sendgrid_event_delivered_does_not_trigger_handler(self, handler):
        """Delivered event does NOT call _handle_email_event (not a problem event)."""
        events = [{"event": "delivered", "email": "ok@example.com"}]
        request = _make_sendgrid_event_request(events)

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ), patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            result = await handler.handle_sendgrid(request)

        mock_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_sendgrid_event_invalid_signature_returns_401(self, handler):
        """Invalid signature is rejected with 401."""
        events = [{"event": "delivered", "email": "test@example.com"}]
        request = _make_sendgrid_event_request(events, signature="bad_sig", timestamp="12345")

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=False,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 401
        assert "signature" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_sendgrid_event_no_signature_skips_verification(self, handler):
        """Missing signature header means verification is skipped."""
        events = [{"event": "delivered", "email": "test@example.com"}]
        # No signature header at all
        request = _make_sendgrid_event_request(events)

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=False,
        ) as mock_verify:
            result = await handler.handle_sendgrid(request)

        # Should succeed because signature header is empty -> verification skipped
        assert _status(result) == 200
        mock_verify.assert_not_called()

    @pytest.mark.asyncio
    async def test_sendgrid_event_invalid_json_returns_400(self, handler):
        """Invalid JSON body returns 400."""
        request = MockAiohttpRequest(
            content_type="application/json",
            body=b"not valid json{{{",
        )

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sendgrid_event_single_object_wrapped_in_list(self, handler):
        """A single event object (not in array) is wrapped into a list."""
        # The handler wraps non-list into [event]
        request = MockAiohttpRequest(
            content_type="application/json",
            body=json.dumps({"event": "delivered", "email": "test@example.com"}).encode(),
        )

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        assert _body(result)["events_processed"] == 1


# ---------------------------------------------------------------------------
# Tests: SendGrid content type dispatch
# ---------------------------------------------------------------------------


class TestSendGridContentTypeDispatch:
    """Tests for content type routing in handle_sendgrid."""

    @pytest.mark.asyncio
    async def test_unsupported_content_type_returns_415(self, handler):
        """Unsupported content type returns 415."""
        request = MockAiohttpRequest(content_type="text/plain", body=b"hello")
        result = await handler.handle_sendgrid(request)
        assert _status(result) == 415

    @pytest.mark.asyncio
    async def test_multipart_routes_to_inbound(self, handler):
        """multipart/form-data routes to _handle_sendgrid_inbound."""
        request = _make_sendgrid_inbound_request()
        mock_process = AsyncMock(return_value=True)

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=mock_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        assert _body(result)["status"] == "processed"

    @pytest.mark.asyncio
    async def test_json_routes_to_event(self, handler):
        """application/json routes to _handle_sendgrid_event."""
        events = [{"event": "delivered", "email": "test@example.com"}]
        request = _make_sendgrid_event_request(events)

        with patch(
            "aragora.integrations.email_reply_loop.verify_sendgrid_signature",
            return_value=True,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        assert _body(result)["events_processed"] == 1

    @pytest.mark.asyncio
    async def test_sendgrid_top_level_exception_returns_500(self, handler):
        """Exception in handle_sendgrid catches and returns 500."""
        request = MockAiohttpRequest(content_type="application/json", body=b"[]")

        # Make json.loads raise by providing non-bytes read
        async def bad_read():
            raise RuntimeError("read error")

        request.read = bad_read

        result = await handler.handle_sendgrid(request)
        assert _status(result) == 500
        assert handler._error_count == 1
        assert handler._last_error == "Internal server error"


# ---------------------------------------------------------------------------
# Tests: Mailgun webhook
# ---------------------------------------------------------------------------


class TestMailgunWebhook:
    """Tests for Mailgun webhook handling."""

    @pytest.mark.asyncio
    async def test_mailgun_success(self, handler):
        """Process a valid Mailgun webhook."""
        request = _make_mailgun_request()
        mock_process = AsyncMock(return_value=True)

        with patch(
            "aragora.integrations.email_reply_loop.verify_mailgun_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=mock_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_mailgun(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "processed"
        assert "message_id" in body

    @pytest.mark.asyncio
    async def test_mailgun_invalid_signature_returns_401(self, handler):
        """Invalid Mailgun signature returns 401."""
        request = _make_mailgun_request()

        with patch(
            "aragora.integrations.email_reply_loop.verify_mailgun_signature",
            return_value=False,
        ):
            result = await handler.handle_mailgun(request)

        assert _status(result) == 401
        assert "signature" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_mailgun_extracts_fields(self, handler):
        """Verify all email fields are extracted from Mailgun webhook."""
        captured = {}

        async def capture(email_data):
            captured["from_email"] = email_data.from_email
            captured["from_name"] = email_data.from_name
            captured["to_email"] = email_data.to_email
            captured["subject"] = email_data.subject
            captured["body_plain"] = email_data.body_plain
            captured["body_html"] = email_data.body_html
            captured["message_id"] = email_data.message_id
            captured["in_reply_to"] = email_data.in_reply_to
            captured["references"] = email_data.references
            return True

        request = _make_mailgun_request(
            sender="Jane Doe <jane@example.com>",
            recipient="inbox@aragora.com",
            subject="Mailgun Subject",
            body_plain="Plain text",
            body_html="<p>HTML</p>",
            message_id="<mg-123@example.com>",
            in_reply_to="<mg-122@example.com>",
            references="<mg-121@example.com> <mg-122@example.com>",
        )

        with patch(
            "aragora.integrations.email_reply_loop.verify_mailgun_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_mailgun(request)

        assert _status(result) == 200
        assert captured["from_email"] == "jane@example.com"
        assert captured["from_name"] == "Jane Doe"
        assert captured["to_email"] == "inbox@aragora.com"
        assert captured["subject"] == "Mailgun Subject"
        assert captured["body_plain"] == "Plain text"
        assert captured["body_html"] == "<p>HTML</p>"
        assert captured["message_id"] == "<mg-123@example.com>"
        assert captured["in_reply_to"] == "<mg-122@example.com>"
        assert captured["references"] == ["<mg-121@example.com>", "<mg-122@example.com>"]

    @pytest.mark.asyncio
    async def test_mailgun_increments_processed_count(self, handler):
        """Successful mailgun processing increments counter."""
        assert handler._processed_count == 0
        request = _make_mailgun_request()

        with patch(
            "aragora.integrations.email_reply_loop.verify_mailgun_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            AsyncMock(return_value=True),
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            await handler.handle_mailgun(request)

        assert handler._processed_count == 1

    @pytest.mark.asyncio
    async def test_mailgun_exception_returns_500(self, handler):
        """Exception in mailgun handler returns 500 and increments error count."""
        request = _make_mailgun_request()

        with patch(
            "aragora.integrations.email_reply_loop.verify_mailgun_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            AsyncMock(side_effect=ValueError("mailgun error")),
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_mailgun(request)

        assert _status(result) == 500
        assert handler._error_count == 1
        assert handler._last_error == "Internal server error"

    @pytest.mark.asyncio
    async def test_mailgun_uses_stripped_text_fallback(self, handler):
        """When body-plain is empty, falls back to stripped-text."""
        captured = {}

        async def capture(email_data):
            captured["body_plain"] = email_data.body_plain
            return True

        # body-plain empty, stripped-text has content
        post_data = {
            "sender": "test@example.com",
            "recipient": "inbox@aragora.com",
            "subject": "Test",
            "body-plain": "",
            "stripped-text": "Stripped text fallback",
            "body-html": "",
            "stripped-html": "",
            "Message-Id": "<msg@ex.com>",
            "In-Reply-To": "",
            "References": "",
            "timestamp": "123",
            "token": "tok",
            "signature": "sig",
        }
        request = MockAiohttpRequest(
            content_type="multipart/form-data",
            post_data=post_data,
        )

        with patch(
            "aragora.integrations.email_reply_loop.verify_mailgun_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_mailgun(request)

        assert _status(result) == 200
        # The handler does `data.get("body-plain", "") or data.get("stripped-text", "")`
        # Empty string is falsy, so it falls back to stripped-text
        assert captured["body_plain"] == "Stripped text fallback"


# ---------------------------------------------------------------------------
# Tests: AWS SES webhook
# ---------------------------------------------------------------------------


class TestSESWebhook:
    """Tests for AWS SES SNS notification handling."""

    @pytest.mark.asyncio
    async def test_ses_subscription_confirmation(self, handler):
        """SNS SubscriptionConfirmation auto-confirms via URL."""
        message = {
            "Type": "SubscriptionConfirmation",
            "SubscribeURL": "https://sns.us-east-1.amazonaws.com/confirm?token=abc",
        }
        request = _make_ses_request(message)

        mock_resp = AsyncMock()
        mock_resp.status = 200

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        assert _body(result)["status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_ses_subscription_missing_url_returns_400(self, handler):
        """Missing SubscribeURL returns 400."""
        message = {
            "Type": "SubscriptionConfirmation",
        }
        request = _make_ses_request(message)
        result = await handler.handle_ses(request)

        assert _status(result) == 400
        assert "SubscribeURL" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_ses_subscription_suspicious_url_returns_400(self, handler):
        """Non-AWS subscription URL is rejected."""
        message = {
            "Type": "SubscriptionConfirmation",
            "SubscribeURL": "https://evil.example.com/confirm",
        }
        request = _make_ses_request(message)
        result = await handler.handle_ses(request)

        assert _status(result) == 400
        assert "Invalid subscription URL" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_ses_subscription_confirmation_failure(self, handler):
        """Failed SNS confirmation returns 500."""
        message = {
            "Type": "SubscriptionConfirmation",
            "SubscribeURL": "https://sns.us-east-1.amazonaws.com/confirm?token=abc",
        }
        request = _make_ses_request(message)

        mock_resp = AsyncMock()
        mock_resp.status = 403

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_resp),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle_ses(request)

        assert _status(result) == 500
        assert "failed" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_ses_subscription_network_error(self, handler):
        """Network error during confirmation returns 500."""
        message = {
            "Type": "SubscriptionConfirmation",
            "SubscribeURL": "https://sns.us-east-1.amazonaws.com/confirm?token=abc",
        }
        request = _make_ses_request(message)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(side_effect=OSError("connection refused")),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await handler.handle_ses(request)

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_ses_notification_bounce(self, handler):
        """Bounce notification processes feedback."""
        notification = {
            "notificationType": "Bounce",
            "bounce": {
                "bounceType": "Permanent",
                "bouncedRecipients": [
                    {"emailAddress": "bad@example.com"},
                ],
            },
        }
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ), patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["type"] == "bounce"
        mock_event.assert_called_once_with(
            "bounce", "bad@example.com", {"emailAddress": "bad@example.com"}
        )

    @pytest.mark.asyncio
    async def test_ses_notification_complaint(self, handler):
        """Complaint notification processes feedback."""
        notification = {
            "notificationType": "Complaint",
            "complaint": {
                "complainedRecipients": [
                    {"emailAddress": "complainer@example.com"},
                ],
            },
        }
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ), patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["type"] == "complaint"
        mock_event.assert_called_once_with(
            "complaint", "complainer@example.com", {"emailAddress": "complainer@example.com"}
        )

    @pytest.mark.asyncio
    async def test_ses_notification_bounce_multiple_recipients(self, handler):
        """Bounce with multiple recipients calls handler for each."""
        notification = {
            "notificationType": "Bounce",
            "bounce": {
                "bounceType": "Permanent",
                "bouncedRecipients": [
                    {"emailAddress": "a@example.com"},
                    {"emailAddress": "b@example.com"},
                ],
            },
        }
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ), patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        assert mock_event.call_count == 2

    @pytest.mark.asyncio
    async def test_ses_notification_email_receipt(self, handler):
        """SES email receipt (with mail+content) is processed."""
        notification = {
            "mail": {
                "messageId": "ses-msg-001",
                "commonHeaders": {
                    "from": ["Sender <sender@example.com>"],
                    "to": ["inbox@aragora.com"],
                    "subject": "SES email test",
                    "messageId": "ses-msg-001",
                },
            },
            "content": "This is the raw email content",
        }
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)
        mock_process = AsyncMock(return_value=True)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=mock_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "processed"
        assert body["message_id"] == "ses-msg-001"

    @pytest.mark.asyncio
    async def test_ses_notification_email_receipt_extracts_sender(self, handler):
        """SES email receipt correctly parses sender name and email."""
        captured = {}

        async def capture(email_data):
            captured["from_email"] = email_data.from_email
            captured["from_name"] = email_data.from_name
            captured["to_email"] = email_data.to_email
            captured["subject"] = email_data.subject
            return True

        notification = {
            "mail": {
                "commonHeaders": {
                    "from": ["Jane Smith <jane@example.com>"],
                    "to": ["debates@aragora.com"],
                    "subject": "Important debate",
                },
            },
            "content": "Email content here",
        }
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        assert captured["from_email"] == "jane@example.com"
        assert captured["from_name"] == "Jane Smith"
        assert captured["to_email"] == "debates@aragora.com"
        assert captured["subject"] == "Important debate"

    @pytest.mark.asyncio
    async def test_ses_notification_delivery_acknowledged(self, handler):
        """Delivery notification (non-bounce, no mail+content) is acknowledged."""
        notification = {
            "notificationType": "Delivery",
        }
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ):
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        assert _body(result)["status"] == "acknowledged"

    @pytest.mark.asyncio
    async def test_ses_notification_invalid_inner_json(self, handler):
        """Invalid JSON in notification Message field is handled gracefully."""
        message = {
            "Type": "Notification",
            "Message": "not valid json{{{",
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ):
            result = await handler.handle_ses(request)

        # Should not crash - empty notification is acknowledged
        assert _status(result) == 200
        assert _body(result)["status"] == "acknowledged"

    @pytest.mark.asyncio
    async def test_ses_unknown_message_type_returns_400(self, handler):
        """Unknown SNS message type returns 400."""
        message = {"Type": "UnknownType"}
        request = _make_ses_request(message)
        result = await handler.handle_ses(request)

        assert _status(result) == 400
        assert "Unknown message type" in _body(result).get("error", "")

    @pytest.mark.asyncio
    async def test_ses_invalid_json_body_returns_400(self, handler):
        """Invalid JSON in request body returns 400."""
        request = MockAiohttpRequest(
            content_type="application/json",
            body=b"not json!!",
        )
        result = await handler.handle_ses(request)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_ses_exception_returns_500(self, handler):
        """Generic exception returns 500 and increments error count."""
        # Cause a RuntimeError by making read() raise
        async def bad_read():
            raise RuntimeError("connection lost")

        request = MockAiohttpRequest(content_type="application/json")
        request.read = bad_read

        result = await handler.handle_ses(request)
        assert _status(result) == 500
        assert handler._error_count == 1
        assert handler._last_error == "Internal server error"

    @pytest.mark.asyncio
    async def test_ses_signature_failure_continues(self, handler):
        """SES signature verification failure logs warning but continues."""
        notification = {"notificationType": "Delivery"}
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=False,
        ):
            result = await handler.handle_ses(request)

        # Should still process (signature verification is optional for SNS)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_ses_email_receipt_truncates_content(self, handler):
        """Content longer than 10000 chars is truncated."""
        captured = {}

        async def capture(email_data):
            captured["body_plain"] = email_data.body_plain
            return True

        long_content = "x" * 20000
        notification = {
            "mail": {
                "commonHeaders": {
                    "from": ["test@example.com"],
                    "to": ["inbox@aragora.com"],
                    "subject": "Long content",
                },
            },
            "content": long_content,
        }
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        assert len(captured["body_plain"]) == 10000

    @pytest.mark.asyncio
    async def test_ses_email_receipt_empty_from_list(self, handler):
        """Empty from/to lists are handled gracefully."""
        captured = {}

        async def capture(email_data):
            captured["from_email"] = email_data.from_email
            captured["to_email"] = email_data.to_email
            return True

        notification = {
            "mail": {
                "commonHeaders": {
                    "from": [],
                    "to": [],
                    "subject": "No sender",
                },
            },
            "content": "content",
        }
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        assert captured["from_email"] == ""
        assert captured["to_email"] == ""


# ---------------------------------------------------------------------------
# Tests: _handle_email_event
# ---------------------------------------------------------------------------


class TestHandleEmailEvent:
    """Tests for the _handle_email_event internal method."""

    @pytest.mark.asyncio
    async def test_handle_email_event_logs_without_error(self, handler):
        """_handle_email_event completes without error (logs only)."""
        # This method currently just logs
        await handler._handle_email_event("bounce", "user@example.com", {"type": "hard"})
        # No exception means success

    @pytest.mark.asyncio
    async def test_handle_email_event_various_types(self, handler):
        """Various event types are handled without error."""
        for event_type in ("bounce", "complaint", "unsubscribe", "dropped", "spamreport"):
            await handler._handle_email_event(event_type, f"{event_type}@ex.com", {})


# ---------------------------------------------------------------------------
# Tests: _handle_ses_feedback
# ---------------------------------------------------------------------------


class TestSESFeedback:
    """Tests for SES bounce/complaint feedback handler."""

    @pytest.mark.asyncio
    async def test_ses_bounce_feedback(self, handler):
        """Bounce feedback calls _handle_email_event for each recipient."""
        notification = {
            "notificationType": "Bounce",
            "bounce": {
                "bounceType": "Transient",
                "bouncedRecipients": [
                    {"emailAddress": "bounce1@example.com"},
                    {"emailAddress": "bounce2@example.com"},
                ],
            },
        }

        with patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            await handler._handle_ses_feedback(notification)

        assert mock_event.call_count == 2
        mock_event.assert_any_call(
            "bounce", "bounce1@example.com", {"emailAddress": "bounce1@example.com"}
        )
        mock_event.assert_any_call(
            "bounce", "bounce2@example.com", {"emailAddress": "bounce2@example.com"}
        )

    @pytest.mark.asyncio
    async def test_ses_complaint_feedback(self, handler):
        """Complaint feedback calls _handle_email_event for each recipient."""
        notification = {
            "notificationType": "Complaint",
            "complaint": {
                "complainedRecipients": [
                    {"emailAddress": "complainer@example.com"},
                ],
            },
        }

        with patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            await handler._handle_ses_feedback(notification)

        mock_event.assert_called_once_with(
            "complaint", "complainer@example.com", {"emailAddress": "complainer@example.com"}
        )

    @pytest.mark.asyncio
    async def test_ses_feedback_empty_recipients(self, handler):
        """Empty recipient lists are handled gracefully."""
        notification = {
            "notificationType": "Bounce",
            "bounce": {
                "bouncedRecipients": [],
            },
        }

        with patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            await handler._handle_ses_feedback(notification)

        mock_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_ses_feedback_unknown_type(self, handler):
        """Unknown notification type is handled gracefully (no crash)."""
        notification = {"notificationType": "Unknown"}

        with patch.object(handler, "_handle_email_event", new_callable=AsyncMock) as mock_event:
            await handler._handle_ses_feedback(notification)

        mock_event.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: get_stats and handle_status
# ---------------------------------------------------------------------------


class TestStatsAndStatus:
    """Tests for statistics and status endpoints."""

    def test_get_stats_initial(self, handler):
        """Initial stats are all zero."""
        stats = handler.get_stats()
        assert stats["processed_count"] == 0
        assert stats["error_count"] == 0
        assert stats["last_error"] is None

    def test_get_stats_after_processing(self, handler):
        """Stats reflect processing activity."""
        handler._processed_count = 5
        handler._error_count = 2
        handler._last_error = "test error"

        stats = handler.get_stats()
        assert stats["processed_count"] == 5
        assert stats["error_count"] == 2
        assert stats["last_error"] == "test error"

    @pytest.mark.asyncio
    async def test_handle_status_returns_ok(self, handler):
        """Status endpoint returns 200 with stats and timestamp."""
        request = MockAiohttpRequest()
        result = await handler.handle_status(request)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "ok"
        assert "stats" in body
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_handle_status_includes_stats(self, handler):
        """Status endpoint includes current stats."""
        handler._processed_count = 10
        handler._error_count = 3
        handler._last_error = "some error"

        request = MockAiohttpRequest()
        result = await handler.handle_status(request)

        body = _body(result)
        stats = body["stats"]
        assert stats["processed_count"] == 10
        assert stats["error_count"] == 3
        assert stats["last_error"] == "some error"


# ---------------------------------------------------------------------------
# Tests: register_email_webhook_routes
# ---------------------------------------------------------------------------


class TestRegisterRoutes:
    """Tests for the register_email_webhook_routes function."""

    def test_register_routes_adds_four_routes(self):
        """Registration adds 4 routes: 3 POST + 1 GET."""
        mock_app = MagicMock()
        handler = register_email_webhook_routes(mock_app)

        assert isinstance(handler, EmailWebhookHandler)
        assert mock_app.router.add_post.call_count == 3
        assert mock_app.router.add_get.call_count == 1

    def test_register_routes_correct_paths(self):
        """Routes are registered at the correct paths."""
        mock_app = MagicMock()
        register_email_webhook_routes(mock_app)

        post_calls = [c.args[0] for c in mock_app.router.add_post.call_args_list]
        assert "/webhooks/email/sendgrid" in post_calls
        assert "/webhooks/email/mailgun" in post_calls
        assert "/webhooks/email/ses" in post_calls

        get_calls = [c.args[0] for c in mock_app.router.add_get.call_args_list]
        assert "/webhooks/email/status" in get_calls


# ---------------------------------------------------------------------------
# Tests: Handler initialization
# ---------------------------------------------------------------------------


class TestHandlerInit:
    """Tests for EmailWebhookHandler initialization."""

    def test_initial_state(self):
        """Handler initializes with zero counters."""
        handler = EmailWebhookHandler()
        assert handler._processed_count == 0
        assert handler._error_count == 0
        assert handler._last_error is None

    def test_multiple_handlers_independent(self):
        """Multiple handler instances have independent state."""
        h1 = EmailWebhookHandler()
        h2 = EmailWebhookHandler()
        h1._processed_count = 5
        assert h2._processed_count == 0


# ---------------------------------------------------------------------------
# Tests: Edge cases and combined scenarios
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and integration-style tests."""

    @pytest.mark.asyncio
    async def test_sendgrid_inbound_empty_sender(self, handler):
        """Empty sender in inbound parse is handled gracefully."""
        request = _make_sendgrid_inbound_request(sender="")
        mock_process = AsyncMock(return_value=True)

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=mock_process,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_sendgrid_inbound_generates_message_id_when_missing(self, handler):
        """When Message-ID header is absent, a sendgrid-timestamp ID is generated."""
        captured = {}

        async def capture(email_data):
            captured["message_id"] = email_data.message_id
            return True

        request = _make_sendgrid_inbound_request(headers_raw="")

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        assert captured["message_id"].startswith("sendgrid-")

    @pytest.mark.asyncio
    async def test_mailgun_generates_message_id_when_missing(self, handler):
        """When Message-Id is empty, a mailgun-timestamp ID is generated."""
        captured = {}

        async def capture(email_data):
            captured["message_id"] = email_data.message_id
            return True

        request = _make_mailgun_request(message_id="")

        with patch(
            "aragora.integrations.email_reply_loop.verify_mailgun_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_mailgun(request)

        assert _status(result) == 200
        assert captured["message_id"].startswith("mailgun-")

    @pytest.mark.asyncio
    async def test_sendgrid_to_from_envelope_fallback(self, handler):
        """When 'to' field is empty, falls back to envelope 'to'."""
        captured = {}

        async def capture(email_data):
            captured["to_email"] = email_data.to_email
            return True

        envelope = json.dumps({"to": ["envelope@aragora.com"], "from": "test@ex.com"})
        request = _make_sendgrid_inbound_request(to="", envelope=envelope)

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_sendgrid(request)

        assert _status(result) == 200
        assert captured["to_email"] == "envelope@aragora.com"

    @pytest.mark.asyncio
    async def test_sendgrid_empty_references_gives_empty_list(self, handler):
        """Empty References header yields an empty list."""
        captured = {}

        async def capture(email_data):
            captured["references"] = email_data.references
            return True

        request = _make_sendgrid_inbound_request(headers_raw="From: test@example.com")

        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            await handler.handle_sendgrid(request)

        assert captured["references"] == []

    @pytest.mark.asyncio
    async def test_multiple_sendgrid_calls_accumulate_stats(self, handler):
        """Multiple successful calls accumulate processed count."""
        mock_process = AsyncMock(return_value=True)

        for i in range(3):
            request = _make_sendgrid_inbound_request()
            with patch(
                "aragora.integrations.email_reply_loop.process_inbound_email",
                new=mock_process,
            ), patch(
                "aragora.integrations.email_reply_loop.InboundEmail",
                MockInboundEmail,
            ):
                await handler.handle_sendgrid(request)

        assert handler._processed_count == 3

    @pytest.mark.asyncio
    async def test_mixed_success_and_error_accumulation(self, handler):
        """Stats correctly accumulate mixed successes and failures."""
        # Success
        request = _make_sendgrid_inbound_request()
        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            AsyncMock(return_value=True),
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            await handler.handle_sendgrid(request)

        # Failure (RuntimeError in process)
        request2 = _make_sendgrid_inbound_request()
        with patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            AsyncMock(side_effect=RuntimeError("fail")),
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            await handler.handle_sendgrid(request2)

        assert handler._processed_count == 1
        assert handler._error_count == 1

    @pytest.mark.asyncio
    async def test_ses_email_receipt_missing_common_headers(self, handler):
        """SES email receipt with missing commonHeaders is handled."""
        captured = {}

        async def capture(email_data):
            captured["from_email"] = email_data.from_email
            captured["subject"] = email_data.subject
            return True

        notification = {
            "mail": {},
            "content": "raw content",
        }
        message = {
            "Type": "Notification",
            "Message": json.dumps(notification),
        }
        request = _make_ses_request(message)

        with patch(
            "aragora.integrations.email_reply_loop.verify_ses_signature",
            return_value=True,
        ), patch(
            "aragora.integrations.email_reply_loop.process_inbound_email",
            new=capture,
        ), patch(
            "aragora.integrations.email_reply_loop.InboundEmail",
            MockInboundEmail,
        ):
            result = await handler.handle_ses(request)

        assert _status(result) == 200
        assert captured["from_email"] == ""
        assert captured["subject"] == ""
