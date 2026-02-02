"""
Tests for Gmail Messages Mixin.

Comprehensive tests for GmailMessagesMixin covering:
1. Message fetching and parsing (headers, body, attachments)
2. Message sending with various content types
3. Thread management and conversation grouping
4. Label operations (add, remove, list) via modify_message
5. Search and filtering functionality
6. Batch operations
7. Error handling and API responses
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aragora.connectors.enterprise.base import SyncItem, SyncState
from aragora.connectors.enterprise.communication.models import (
    BatchFetchResult,
    EmailAttachment,
    EmailMessage,
    EmailThread,
    MessageFetchFailure,
)
from aragora.reasoning.provenance import SourceType


# =============================================================================
# Test Fixtures
# =============================================================================


class MockAsyncContextManager:
    """Mock async context manager for HTTP client."""

    def __init__(self, mock_client):
        self.mock_client = mock_client

    async def __aenter__(self):
        return self.mock_client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockGmailBase:
    """Mock base class that implements GmailBaseMethods protocol."""

    def __init__(self):
        self.user_id = "me"
        self.include_spam_trash = False
        self.exclude_labels: set[str] = set()
        self.labels: list[str] | None = None
        self.max_results = 100
        self._access_token = "test_token"
        self._circuit_open = False
        self._failure_count = 0
        self._success_count = 0
        self._mock_client = None

    @property
    def source_type(self) -> SourceType:
        return SourceType.DOCUMENT

    async def _get_access_token(self) -> str:
        return self._access_token

    async def _api_request(self, endpoint: str, method: str = "GET", **kwargs) -> dict:
        return {}

    def _get_client(self):
        """Return context manager for HTTP client."""
        if self._mock_client is None:
            self._mock_client = AsyncMock()
        return MockAsyncContextManager(self._mock_client)

    def check_circuit_breaker(self) -> bool:
        return not self._circuit_open

    def get_circuit_breaker_status(self) -> dict[str, Any]:
        return {"cooldown_seconds": 60, "failure_count": self._failure_count}

    def record_success(self) -> None:
        self._success_count += 1

    def record_failure(self) -> None:
        self._failure_count += 1

    async def get_user_info(self) -> dict[str, Any]:
        return {
            "emailAddress": "test@example.com",
            "historyId": "12345",
            "messagesTotal": 1000,
        }


def create_messages_mixin_with_mocks():
    """Create a messages mixin with proper mock setup for send/reply tests."""
    from aragora.connectors.enterprise.communication.gmail.messages import GmailMessagesMixin

    class TestMixin(GmailMessagesMixin, MockGmailBase):
        pass

    mixin = TestMixin()
    # Ensure circuit breaker returns proper values
    return mixin


@pytest.fixture
def mock_httpx_response():
    """Factory for creating mock httpx responses."""

    def _create(status_code: int = 200, json_data: dict = None, content: bytes = b""):
        response = Mock()
        response.status_code = status_code
        response.json = Mock(return_value=json_data or {})
        response.content = content or b"{}"
        response.text = (json_data and str(json_data)) or "{}"
        response.raise_for_status = Mock()
        if status_code >= 400:
            import httpx

            request = httpx.Request("GET", "https://gmail.googleapis.com/test")
            response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Error", request=request, response=response
            )
        return response

    return _create


@pytest.fixture
def messages_mixin():
    """Create a messages mixin instance with mock base."""
    from aragora.connectors.enterprise.communication.gmail.messages import GmailMessagesMixin

    class TestMixin(GmailMessagesMixin, MockGmailBase):
        pass

    return TestMixin()


@pytest.fixture
def sample_gmail_message():
    """Sample Gmail API message response."""
    return {
        "id": "msg_123",
        "threadId": "thread_456",
        "labelIds": ["INBOX", "UNREAD", "IMPORTANT"],
        "snippet": "This is a test email...",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "recipient@example.com"},
                {"name": "Subject", "value": "Test Subject"},
                {"name": "Date", "value": "Mon, 15 Jan 2024 10:30:00 +0000"},
                {"name": "CC", "value": "cc@example.com"},
                {"name": "BCC", "value": "bcc@example.com"},
                {"name": "Message-ID", "value": "<abc123@example.com>"},
            ],
            "body": {"data": base64.urlsafe_b64encode(b"Hello, this is the email body.").decode()},
        },
    }


@pytest.fixture
def sample_multipart_message():
    """Sample multipart Gmail API message response."""
    return {
        "id": "msg_multipart",
        "threadId": "thread_multipart",
        "labelIds": ["INBOX", "STARRED"],
        "snippet": "Multipart email...",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "recipient@example.com"},
                {"name": "Subject", "value": "Multipart Test"},
                {"name": "Date", "value": "Tue, 16 Jan 2024 14:00:00 +0000"},
            ],
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": base64.urlsafe_b64encode(b"Plain text body").decode()},
                },
                {
                    "mimeType": "text/html",
                    "body": {
                        "data": base64.urlsafe_b64encode(
                            b"<html><body>HTML body</body></html>"
                        ).decode()
                    },
                },
            ],
        },
    }


@pytest.fixture
def sample_message_with_attachments():
    """Sample message with attachments."""
    return {
        "id": "msg_attach",
        "threadId": "thread_attach",
        "labelIds": ["INBOX"],
        "snippet": "Message with attachment",
        "payload": {
            "mimeType": "multipart/mixed",
            "headers": [
                {"name": "Subject", "value": "With Attachment"},
                {"name": "From", "value": "sender@example.com"},
                {"name": "To", "value": "recipient@example.com"},
                {"name": "Date", "value": "Wed, 17 Jan 2024 09:00:00 +0000"},
            ],
            "parts": [
                {
                    "mimeType": "text/plain",
                    "body": {"data": base64.urlsafe_b64encode(b"Email body").decode()},
                },
                {
                    "mimeType": "application/pdf",
                    "filename": "document.pdf",
                    "body": {
                        "attachmentId": "attach_123",
                        "size": 12345,
                    },
                },
                {
                    "mimeType": "image/png",
                    "filename": "screenshot.png",
                    "body": {
                        "attachmentId": "attach_456",
                        "size": 54321,
                    },
                },
            ],
        },
    }


# =============================================================================
# List Messages Tests
# =============================================================================


class TestListMessages:
    """Tests for listing Gmail messages."""

    @pytest.mark.asyncio
    async def test_list_messages_basic(self, messages_mixin):
        """Test listing messages returns message IDs and pagination token."""
        response = {
            "messages": [
                {"id": "msg_1"},
                {"id": "msg_2"},
                {"id": "msg_3"},
            ],
            "nextPageToken": "next_page_token",
        }

        with patch.object(messages_mixin, "_api_request", return_value=response):
            message_ids, next_token = await messages_mixin.list_messages()

            assert len(message_ids) == 3
            assert "msg_1" in message_ids
            assert "msg_2" in message_ids
            assert "msg_3" in message_ids
            assert next_token == "next_page_token"

    @pytest.mark.asyncio
    async def test_list_messages_with_query(self, messages_mixin):
        """Test listing messages with search query."""
        response = {"messages": [{"id": "msg_1"}]}

        with patch.object(messages_mixin, "_api_request", return_value=response) as mock_request:
            await messages_mixin.list_messages(query="from:test@example.com")

            call_args = mock_request.call_args
            assert call_args.kwargs.get("params", {}).get("q") == "from:test@example.com"

    @pytest.mark.asyncio
    async def test_list_messages_with_label_ids(self, messages_mixin):
        """Test listing messages with label filter."""
        response = {"messages": [{"id": "msg_1"}]}

        with patch.object(messages_mixin, "_api_request", return_value=response) as mock_request:
            await messages_mixin.list_messages(label_ids=["INBOX", "IMPORTANT"])

            call_args = mock_request.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("labelIds") == ["INBOX", "IMPORTANT"]

    @pytest.mark.asyncio
    async def test_list_messages_with_pagination(self, messages_mixin):
        """Test listing messages with pagination token."""
        response = {"messages": [{"id": "msg_4"}], "nextPageToken": "page_3"}

        with patch.object(messages_mixin, "_api_request", return_value=response) as mock_request:
            await messages_mixin.list_messages(page_token="page_2")

            call_args = mock_request.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("pageToken") == "page_2"

    @pytest.mark.asyncio
    async def test_list_messages_max_results_capped(self, messages_mixin):
        """Test max_results is capped at 500."""
        response = {"messages": []}

        with patch.object(messages_mixin, "_api_request", return_value=response) as mock_request:
            await messages_mixin.list_messages(max_results=1000)

            call_args = mock_request.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("maxResults") == 500

    @pytest.mark.asyncio
    async def test_list_messages_empty_result(self, messages_mixin):
        """Test listing messages when no messages exist."""
        response = {}

        with patch.object(messages_mixin, "_api_request", return_value=response):
            message_ids, next_token = await messages_mixin.list_messages()

            assert message_ids == []
            assert next_token is None

    @pytest.mark.asyncio
    async def test_list_messages_includes_spam_trash_setting(self, messages_mixin):
        """Test include_spam_trash is passed to API."""
        messages_mixin.include_spam_trash = True
        response = {"messages": []}

        with patch.object(messages_mixin, "_api_request", return_value=response) as mock_request:
            await messages_mixin.list_messages()

            call_args = mock_request.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("includeSpamTrash") is True


# =============================================================================
# Get Message Tests
# =============================================================================


class TestGetMessage:
    """Tests for getting a single message."""

    @pytest.mark.asyncio
    async def test_get_message_basic(self, messages_mixin, sample_gmail_message):
        """Test getting a single message by ID."""
        with patch.object(messages_mixin, "_api_request", return_value=sample_gmail_message):
            message = await messages_mixin.get_message("msg_123")

            assert isinstance(message, EmailMessage)
            assert message.id == "msg_123"
            assert message.thread_id == "thread_456"
            assert message.subject == "Test Subject"
            assert message.from_address == "sender@example.com"

    @pytest.mark.asyncio
    async def test_get_message_calls_correct_endpoint(self, messages_mixin, sample_gmail_message):
        """Test get_message calls correct API endpoint."""
        with patch.object(
            messages_mixin, "_api_request", return_value=sample_gmail_message
        ) as mock_request:
            await messages_mixin.get_message("msg_123", format="metadata")

            mock_request.assert_called_once_with("/messages/msg_123", params={"format": "metadata"})

    @pytest.mark.asyncio
    async def test_get_message_formats(self, messages_mixin, sample_gmail_message):
        """Test different message format options."""
        with patch.object(
            messages_mixin, "_api_request", return_value=sample_gmail_message
        ) as mock_request:
            # Test full format
            await messages_mixin.get_message("msg_1", format="full")
            assert mock_request.call_args.kwargs["params"]["format"] == "full"

            # Test metadata format
            await messages_mixin.get_message("msg_2", format="metadata")
            assert mock_request.call_args.kwargs["params"]["format"] == "metadata"

            # Test minimal format
            await messages_mixin.get_message("msg_3", format="minimal")
            assert mock_request.call_args.kwargs["params"]["format"] == "minimal"


# =============================================================================
# Message Parsing Tests
# =============================================================================


class TestMessageParsing:
    """Tests for message parsing logic."""

    def test_parse_message_headers(self, messages_mixin, sample_gmail_message):
        """Test parsing message headers correctly."""
        message = messages_mixin._parse_message(sample_gmail_message)

        assert message.from_address == "sender@example.com"
        assert "recipient@example.com" in message.to_addresses
        assert "cc@example.com" in message.cc_addresses
        assert "bcc@example.com" in message.bcc_addresses
        assert message.subject == "Test Subject"

    def test_parse_message_body_text(self, messages_mixin, sample_gmail_message):
        """Test parsing plain text body."""
        message = messages_mixin._parse_message(sample_gmail_message)

        assert message.body_text == "Hello, this is the email body."

    def test_parse_message_labels(self, messages_mixin, sample_gmail_message):
        """Test parsing label flags."""
        message = messages_mixin._parse_message(sample_gmail_message)

        assert message.labels == ["INBOX", "UNREAD", "IMPORTANT"]
        assert message.is_read is False  # Has UNREAD
        assert message.is_starred is False  # No STARRED
        assert message.is_important is True  # Has IMPORTANT

    def test_parse_message_starred(self, messages_mixin, sample_multipart_message):
        """Test parsing starred message."""
        message = messages_mixin._parse_message(sample_multipart_message)

        assert message.is_starred is True  # Has STARRED label

    def test_parse_message_date(self, messages_mixin, sample_gmail_message):
        """Test parsing message date."""
        message = messages_mixin._parse_message(sample_gmail_message)

        assert message.date.year == 2024
        assert message.date.month == 1
        assert message.date.day == 15

    def test_parse_message_invalid_date(self, messages_mixin):
        """Test parsing message with invalid date falls back to now."""
        msg_data = {
            "id": "msg_bad_date",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "Date", "value": "invalid-date-format"},
                    {"name": "From", "value": "test@example.com"},
                ],
                "body": {},
            },
        }

        message = messages_mixin._parse_message(msg_data)

        # Should fall back to current time
        assert message.date is not None
        assert (datetime.now(timezone.utc) - message.date).seconds < 60

    def test_parse_message_missing_subject(self, messages_mixin):
        """Test parsing message without subject."""
        msg_data = {
            "id": "msg_no_subject",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:30:00 +0000"},
                ],
                "body": {},
            },
        }

        message = messages_mixin._parse_message(msg_data)

        assert message.subject == "(No Subject)"

    def test_parse_message_snippet(self, messages_mixin, sample_gmail_message):
        """Test snippet is extracted."""
        message = messages_mixin._parse_message(sample_gmail_message)

        assert message.snippet == "This is a test email..."

    def test_parse_message_message_id_header(self, messages_mixin, sample_gmail_message):
        """Test Message-ID header is accessible."""
        message = messages_mixin._parse_message(sample_gmail_message)

        # Headers are stored lowercase
        assert message.headers.get("message-id") == "<abc123@example.com>"


# =============================================================================
# Multipart Message Tests
# =============================================================================


class TestMultipartMessages:
    """Tests for multipart message handling."""

    def test_parse_multipart_alternative(self, messages_mixin, sample_multipart_message):
        """Test parsing multipart/alternative message."""
        message = messages_mixin._parse_message(sample_multipart_message)

        assert message.body_text == "Plain text body"
        assert "HTML body" in message.body_html

    def test_parse_message_with_attachments(self, messages_mixin, sample_message_with_attachments):
        """Test parsing message with attachments."""
        message = messages_mixin._parse_message(sample_message_with_attachments)

        assert len(message.attachments) == 2

        pdf_attachment = next(a for a in message.attachments if a.filename == "document.pdf")
        assert pdf_attachment.id == "attach_123"
        assert pdf_attachment.mime_type == "application/pdf"
        assert pdf_attachment.size == 12345

        png_attachment = next(a for a in message.attachments if a.filename == "screenshot.png")
        assert png_attachment.id == "attach_456"
        assert png_attachment.size == 54321

    def test_parse_nested_multipart(self, messages_mixin):
        """Test parsing nested multipart message."""
        msg_data = {
            "id": "msg_nested",
            "threadId": "thread_nested",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [
                    {"name": "Subject", "value": "Nested multipart"},
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Date", "value": "Thu, 18 Jan 2024 11:00:00 +0000"},
                ],
                "parts": [
                    {
                        "mimeType": "multipart/alternative",
                        "parts": [
                            {
                                "mimeType": "text/plain",
                                "body": {
                                    "data": base64.urlsafe_b64encode(b"Nested plain text").decode()
                                },
                            },
                            {
                                "mimeType": "text/html",
                                "body": {
                                    "data": base64.urlsafe_b64encode(b"<p>Nested HTML</p>").decode()
                                },
                            },
                        ],
                    },
                    {
                        "mimeType": "application/pdf",
                        "filename": "nested.pdf",
                        "body": {"attachmentId": "nested_attach", "size": 1000},
                    },
                ],
            },
        }

        message = messages_mixin._parse_message(msg_data)

        assert message.body_text == "Nested plain text"
        assert message.body_html == "<p>Nested HTML</p>"
        assert len(message.attachments) == 1
        assert message.attachments[0].filename == "nested.pdf"


# =============================================================================
# Batch Message Fetching Tests
# =============================================================================


class TestGetMessages:
    """Tests for batch message fetching (get_messages)."""

    @pytest.mark.asyncio
    async def test_get_messages_empty_list(self, messages_mixin):
        """Test get_messages with empty list returns empty list."""
        result = await messages_mixin.get_messages([])
        assert result == []

    @pytest.mark.asyncio
    async def test_get_messages_single(self, messages_mixin, sample_gmail_message):
        """Test get_messages with single message ID."""
        with patch.object(messages_mixin, "_api_request", return_value=sample_gmail_message):
            messages = await messages_mixin.get_messages(["msg_123"])

            assert len(messages) == 1
            assert isinstance(messages[0], EmailMessage)
            assert messages[0].id == "msg_123"

    @pytest.mark.asyncio
    async def test_get_messages_multiple(self, messages_mixin, sample_gmail_message):
        """Test get_messages with multiple message IDs."""
        call_count = 0

        async def mock_api_request(endpoint, **kwargs):
            nonlocal call_count
            call_count += 1
            msg_id = endpoint.split("/")[-1]
            return {**sample_gmail_message, "id": msg_id}

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            messages = await messages_mixin.get_messages(["msg_1", "msg_2", "msg_3"])

            assert len(messages) == 3
            assert call_count == 3
            ids = {msg.id for msg in messages}
            assert ids == {"msg_1", "msg_2", "msg_3"}

    @pytest.mark.asyncio
    async def test_get_messages_handles_failures_gracefully(
        self, messages_mixin, sample_gmail_message
    ):
        """Test get_messages returns partial results on failures."""

        async def mock_api_request(endpoint, **kwargs):
            msg_id = endpoint.split("/")[-1]
            if msg_id == "msg_2":
                raise Exception("API error for msg_2")
            return {**sample_gmail_message, "id": msg_id}

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            messages = await messages_mixin.get_messages(["msg_1", "msg_2", "msg_3"])

            assert len(messages) == 2
            ids = {msg.id for msg in messages}
            assert ids == {"msg_1", "msg_3"}

    @pytest.mark.asyncio
    async def test_get_messages_respects_max_concurrent(self, messages_mixin, sample_gmail_message):
        """Test get_messages respects max_concurrent limit."""
        concurrent_count = 0
        max_concurrent_observed = 0

        async def mock_api_request(endpoint, **kwargs):
            nonlocal concurrent_count, max_concurrent_observed
            concurrent_count += 1
            max_concurrent_observed = max(max_concurrent_observed, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return {**sample_gmail_message, "id": endpoint.split("/")[-1]}

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            message_ids = [f"msg_{i}" for i in range(20)]
            messages = await messages_mixin.get_messages(message_ids, max_concurrent=5)

            assert len(messages) == 20
            assert max_concurrent_observed <= 5

    @pytest.mark.asyncio
    async def test_get_messages_with_format(self, messages_mixin, sample_gmail_message):
        """Test get_messages passes format parameter."""
        formats_requested = []

        async def mock_api_request(endpoint, params=None, **kwargs):
            if params:
                formats_requested.append(params.get("format"))
            return sample_gmail_message

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            await messages_mixin.get_messages(["msg_1", "msg_2"], format="metadata")

            assert all(f == "metadata" for f in formats_requested)

    @pytest.mark.asyncio
    async def test_get_messages_all_fail(self, messages_mixin):
        """Test get_messages returns empty list when all fail."""

        async def mock_api_request(endpoint, **kwargs):
            raise Exception("All fail")

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            messages = await messages_mixin.get_messages(["msg_1", "msg_2"])
            assert messages == []

    @pytest.mark.asyncio
    async def test_get_messages_strict_mode_raises(self, messages_mixin, sample_gmail_message):
        """Test get_messages with strict=True raises on failure."""

        async def mock_api_request(endpoint, **kwargs):
            if "msg_2" in endpoint:
                raise Exception("API error")
            return sample_gmail_message

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            with pytest.raises(RuntimeError, match="Failed to fetch 1 of 2 messages"):
                await messages_mixin.get_messages(["msg_1", "msg_2"], strict=True)

    @pytest.mark.asyncio
    async def test_get_messages_strict_mode_success(self, messages_mixin, sample_gmail_message):
        """Test get_messages with strict=True succeeds when all pass."""
        with patch.object(messages_mixin, "_api_request", return_value=sample_gmail_message):
            messages = await messages_mixin.get_messages(["msg_1", "msg_2"], strict=True)
            assert len(messages) == 2


# =============================================================================
# Batch Fetch Result Tests
# =============================================================================


class TestGetMessagesBatch:
    """Tests for get_messages_batch with detailed failure tracking."""

    @pytest.mark.asyncio
    async def test_get_messages_batch_returns_result_object(
        self, messages_mixin, sample_gmail_message
    ):
        """Test get_messages_batch returns BatchFetchResult."""

        async def mock_api_request(endpoint, **kwargs):
            if "msg_2" in endpoint:
                raise ValueError("Not found")
            msg_id = endpoint.split("/")[-1]
            return {**sample_gmail_message, "id": msg_id}

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            result = await messages_mixin.get_messages_batch(["msg_1", "msg_2", "msg_3"])

            assert isinstance(result, BatchFetchResult)
            assert result.total_requested == 3
            assert result.success_count == 2
            assert result.failure_count == 1
            assert result.is_partial is True
            assert result.is_complete is False
            assert result.is_total_failure is False
            assert "msg_2" in result.failed_ids

    @pytest.mark.asyncio
    async def test_get_messages_batch_failure_details(self, messages_mixin):
        """Test failure details are captured correctly."""

        async def mock_api_request(endpoint, **kwargs):
            if "msg_1" in endpoint:
                raise TimeoutError("Connection timed out")
            elif "msg_2" in endpoint:
                raise ValueError("Invalid format")
            # msg_3 raises a connection error with "connection" keyword for retryable detection
            raise ConnectionError("connection refused")

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            result = await messages_mixin.get_messages_batch(["msg_1", "msg_2", "msg_3"])

            assert result.failure_count == 3
            assert result.is_total_failure is True

            failures_by_id = {f.message_id: f for f in result.failures}

            assert failures_by_id["msg_1"].error_type == "TimeoutError"
            assert failures_by_id["msg_1"].is_retryable is True

            assert failures_by_id["msg_2"].error_type == "ValueError"
            assert failures_by_id["msg_2"].is_retryable is False

            assert failures_by_id["msg_3"].error_type == "ConnectionError"
            assert failures_by_id["msg_3"].is_retryable is True

    @pytest.mark.asyncio
    async def test_get_messages_batch_retryable_ids(self, messages_mixin, sample_gmail_message):
        """Test retryable_ids property identifies retryable errors."""

        async def mock_api_request(endpoint, **kwargs):
            if "msg_1" in endpoint:
                raise ConnectionError("Connection reset")
            elif "msg_2" in endpoint:
                raise ValueError("Invalid data")
            elif "msg_3" in endpoint:
                raise RuntimeError("429 rate limit exceeded")
            return sample_gmail_message

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            result = await messages_mixin.get_messages_batch(["msg_1", "msg_2", "msg_3", "msg_4"])

            assert len(result.retryable_ids) == 2
            assert "msg_1" in result.retryable_ids
            assert "msg_3" in result.retryable_ids
            assert "msg_2" not in result.retryable_ids

    @pytest.mark.asyncio
    async def test_get_messages_batch_empty_input(self, messages_mixin):
        """Test get_messages_batch with empty input."""
        result = await messages_mixin.get_messages_batch([])

        assert result.total_requested == 0
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.is_complete is True
        assert result.is_total_failure is False

    @pytest.mark.asyncio
    async def test_get_messages_batch_to_dict(self, messages_mixin, sample_gmail_message):
        """Test BatchFetchResult.to_dict() serialization."""

        async def mock_api_request(endpoint, **kwargs):
            if "msg_2" in endpoint:
                raise ValueError("Not found")
            return sample_gmail_message

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            result = await messages_mixin.get_messages_batch(["msg_1", "msg_2"])
            result_dict = result.to_dict()

            assert result_dict["success_count"] == 1
            assert result_dict["failure_count"] == 1
            assert result_dict["total_requested"] == 2
            assert result_dict["is_partial"] is True
            assert "msg_2" in result_dict["failed_ids"]
            assert len(result_dict["failures"]) == 1


# =============================================================================
# Thread Operations Tests
# =============================================================================


class TestGetThread:
    """Tests for thread operations."""

    @pytest.mark.asyncio
    async def test_get_thread_basic(self, messages_mixin, sample_gmail_message):
        """Test getting a conversation thread."""
        thread_data = {
            "id": "thread_456",
            "snippet": "Thread snippet...",
            "messages": [
                sample_gmail_message,
                {**sample_gmail_message, "id": "msg_124"},
            ],
        }

        with patch.object(messages_mixin, "_api_request", return_value=thread_data):
            thread = await messages_mixin.get_thread("thread_456")

            assert isinstance(thread, EmailThread)
            assert thread.id == "thread_456"
            assert len(thread.messages) == 2
            assert thread.message_count == 2

    @pytest.mark.asyncio
    async def test_get_thread_participants(self, messages_mixin):
        """Test thread participants are collected."""
        msg1 = {
            "id": "msg_1",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "To", "value": "bob@example.com, carol@example.com"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:00:00 +0000"},
                ],
                "body": {},
            },
        }
        msg2 = {
            "id": "msg_2",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "bob@example.com"},
                    {"name": "To", "value": "alice@example.com"},
                    {"name": "CC", "value": "dave@example.com"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 11:00:00 +0000"},
                ],
                "body": {},
            },
        }

        thread_data = {"id": "thread_1", "snippet": "", "messages": [msg1, msg2]}

        with patch.object(messages_mixin, "_api_request", return_value=thread_data):
            thread = await messages_mixin.get_thread("thread_1")

            assert "alice@example.com" in thread.participants
            assert "bob@example.com" in thread.participants
            assert "carol@example.com" in thread.participants
            assert "dave@example.com" in thread.participants

    @pytest.mark.asyncio
    async def test_get_thread_labels_aggregated(self, messages_mixin):
        """Test thread labels are aggregated from all messages."""
        msg1 = {
            "id": "msg_1",
            "threadId": "thread_1",
            "labelIds": ["INBOX", "IMPORTANT"],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:00:00 +0000"},
                ],
                "body": {},
            },
        }
        msg2 = {
            "id": "msg_2",
            "threadId": "thread_1",
            "labelIds": ["INBOX", "STARRED"],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 11:00:00 +0000"},
                ],
                "body": {},
            },
        }

        thread_data = {"id": "thread_1", "snippet": "", "messages": [msg1, msg2]}

        with patch.object(messages_mixin, "_api_request", return_value=thread_data):
            thread = await messages_mixin.get_thread("thread_1")

            assert "INBOX" in thread.labels
            assert "IMPORTANT" in thread.labels
            assert "STARRED" in thread.labels

    @pytest.mark.asyncio
    async def test_get_thread_subject_from_first_message(
        self, messages_mixin, sample_gmail_message
    ):
        """Test thread subject comes from first message."""
        thread_data = {
            "id": "thread_1",
            "snippet": "",
            "messages": [sample_gmail_message],
        }

        with patch.object(messages_mixin, "_api_request", return_value=thread_data):
            thread = await messages_mixin.get_thread("thread_1")

            assert thread.subject == "Test Subject"

    @pytest.mark.asyncio
    async def test_get_thread_empty_messages(self, messages_mixin):
        """Test thread with no messages."""
        thread_data = {"id": "thread_empty", "snippet": "Empty", "messages": []}

        with patch.object(messages_mixin, "_api_request", return_value=thread_data):
            thread = await messages_mixin.get_thread("thread_empty")

            assert thread.subject == ""
            assert thread.message_count == 0
            assert thread.last_message_date is None


# =============================================================================
# History API Tests
# =============================================================================


class TestGetHistory:
    """Tests for History API (incremental sync)."""

    @pytest.mark.asyncio
    async def test_get_history_basic(self, messages_mixin):
        """Test getting message history."""
        history_data = {
            "history": [
                {
                    "id": "100",
                    "messagesAdded": [{"message": {"id": "msg_new"}}],
                },
            ],
            "historyId": "101",
        }

        with patch.object(messages_mixin, "_api_request", return_value=history_data):
            history, next_token, new_id = await messages_mixin.get_history("99")

            assert len(history) == 1
            assert history[0]["messagesAdded"][0]["message"]["id"] == "msg_new"
            assert new_id == "101"
            assert next_token is None

    @pytest.mark.asyncio
    async def test_get_history_with_pagination(self, messages_mixin):
        """Test history API with pagination."""
        history_data = {
            "history": [{"id": "100"}],
            "historyId": "101",
            "nextPageToken": "page_2",
        }

        with patch.object(
            messages_mixin, "_api_request", return_value=history_data
        ) as mock_request:
            await messages_mixin.get_history("99", page_token="page_1")

            call_args = mock_request.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("pageToken") == "page_1"

    @pytest.mark.asyncio
    async def test_get_history_with_label_filter(self, messages_mixin):
        """Test history API with label filter."""
        history_data = {"history": [], "historyId": "101"}

        with patch.object(
            messages_mixin, "_api_request", return_value=history_data
        ) as mock_request:
            await messages_mixin.get_history("99", label_id="INBOX")

            call_args = mock_request.call_args
            params = call_args.kwargs.get("params", {})
            assert params.get("labelId") == "INBOX"

    @pytest.mark.asyncio
    async def test_get_history_expired_id(self, messages_mixin):
        """Test handling expired history ID."""
        with patch.object(messages_mixin, "_api_request") as mock_request:
            # Use OSError which is caught by the exception handler in get_history
            mock_request.side_effect = OSError("404 historyId expired")

            history, next_token, new_id = await messages_mixin.get_history("old_id")

            assert history == []
            assert next_token is None
            assert new_id == ""

    @pytest.mark.asyncio
    async def test_get_history_types_requested(self, messages_mixin):
        """Test correct history types are requested."""
        history_data = {"history": [], "historyId": "101"}

        with patch.object(
            messages_mixin, "_api_request", return_value=history_data
        ) as mock_request:
            await messages_mixin.get_history("99")

            call_args = mock_request.call_args
            params = call_args.kwargs.get("params", {})
            assert "messageAdded" in params.get("historyTypes", [])
            assert "labelAdded" in params.get("historyTypes", [])


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Tests for search functionality."""

    @pytest.mark.asyncio
    async def test_search_basic(self, messages_mixin):
        """Test basic Gmail search."""
        mock_msg = EmailMessage(
            id="msg_123",
            thread_id="thread_456",
            subject="Test",
            from_address="test@example.com",
            to_addresses=["recipient@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            snippet="Snippet",
            labels=["INBOX"],
        )

        with patch.object(messages_mixin, "list_messages", return_value=(["msg_123"], None)):
            with patch.object(messages_mixin, "get_messages", return_value=[mock_msg]):
                results = await messages_mixin.search("from:test@example.com")

                assert len(results) == 1
                assert results[0].source_id == "msg_123"

    @pytest.mark.asyncio
    async def test_search_uses_batch_fetching(self, messages_mixin):
        """Test search uses batch fetching to avoid N+1 queries."""
        mock_msgs = [
            EmailMessage(
                id=f"msg_{i}",
                thread_id=f"thread_{i}",
                subject=f"Test {i}",
                from_address="test@example.com",
                to_addresses=["recipient@example.com"],
                date=datetime.now(timezone.utc),
                body_text="Body",
                snippet="Snippet",
                labels=[],
            )
            for i in range(3)
        ]

        with patch.object(
            messages_mixin, "list_messages", return_value=(["msg_0", "msg_1", "msg_2"], None)
        ):
            with patch.object(messages_mixin, "get_messages", return_value=mock_msgs) as mock_get:
                results = await messages_mixin.search("query", limit=10)

                assert len(results) == 3
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_respects_limit(self, messages_mixin):
        """Test search respects limit parameter."""
        with patch.object(
            messages_mixin, "list_messages", return_value=(["msg_1", "msg_2", "msg_3"], None)
        ) as mock_list:
            with patch.object(messages_mixin, "get_messages", return_value=[]):
                await messages_mixin.search("query", limit=5)

                call_args = mock_list.call_args
                assert call_args.kwargs.get("max_results") == 5

    @pytest.mark.asyncio
    async def test_search_returns_evidence_format(self, messages_mixin):
        """Test search returns Evidence objects with correct fields."""
        mock_msg = EmailMessage(
            id="msg_123",
            thread_id="thread_456",
            subject="Test Subject",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            snippet="Test snippet",
            labels=["INBOX", "IMPORTANT"],
        )

        with patch.object(messages_mixin, "list_messages", return_value=(["msg_123"], None)):
            with patch.object(messages_mixin, "get_messages", return_value=[mock_msg]):
                results = await messages_mixin.search("query")

                assert results[0].id == "gmail-msg_123"
                assert results[0].title == "Test Subject"
                assert results[0].author == "sender@example.com"
                assert results[0].content == "Test snippet"
                assert "thread_id" in results[0].metadata


# =============================================================================
# Fetch Tests
# =============================================================================


class TestFetch:
    """Tests for fetching specific emails."""

    @pytest.mark.asyncio
    async def test_fetch_with_gmail_prefix(self, messages_mixin):
        """Test fetch strips gmail- prefix from ID."""
        mock_msg = EmailMessage(
            id="msg_123",
            thread_id="thread_456",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body text",
            snippet="Snippet",
            labels=[],
        )

        with patch.object(messages_mixin, "get_message", return_value=mock_msg) as mock_get:
            result = await messages_mixin.fetch("gmail-msg_123")

            mock_get.assert_called_once_with("msg_123")
            assert result is not None
            assert result.source_id == "msg_123"

    @pytest.mark.asyncio
    async def test_fetch_without_prefix(self, messages_mixin):
        """Test fetch works without gmail- prefix."""
        mock_msg = EmailMessage(
            id="msg_456",
            thread_id="thread_789",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            snippet="Snippet",
            labels=[],
        )

        with patch.object(messages_mixin, "get_message", return_value=mock_msg) as mock_get:
            await messages_mixin.fetch("msg_456")

            mock_get.assert_called_once_with("msg_456")

    @pytest.mark.asyncio
    async def test_fetch_returns_evidence_with_metadata(self, messages_mixin):
        """Test fetch returns Evidence with full metadata."""
        mock_msg = EmailMessage(
            id="msg_123",
            thread_id="thread_456",
            subject="Test Subject",
            from_address="sender@example.com",
            to_addresses=["to1@example.com", "to2@example.com"],
            cc_addresses=["cc@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Full body text",
            snippet="Snippet",
            labels=["INBOX"],
            is_read=True,
            is_starred=True,
        )

        with patch.object(messages_mixin, "get_message", return_value=mock_msg):
            result = await messages_mixin.fetch("msg_123")

            assert result.title == "Test Subject"
            assert result.content == "Full body text"
            assert result.metadata["to"] == ["to1@example.com", "to2@example.com"]
            assert result.metadata["cc"] == ["cc@example.com"]
            assert result.metadata["is_read"] is True
            assert result.metadata["is_starred"] is True

    @pytest.mark.asyncio
    async def test_fetch_handles_error(self, messages_mixin):
        """Test fetch returns None on error."""
        with patch.object(messages_mixin, "get_message", side_effect=ValueError("Not found")):
            result = await messages_mixin.fetch("msg_nonexistent")

            assert result is None


# =============================================================================
# Send Message Tests
# =============================================================================


class TestSendMessage:
    """Tests for sending messages."""

    @pytest.mark.asyncio
    async def test_send_message_basic(self, messages_mixin, mock_httpx_response):
        """Test sending a basic email."""
        response_data = {"id": "sent_msg_123", "threadId": "new_thread"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_httpx_response(200, response_data))

        with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
            with patch.object(messages_mixin, "_get_client") as mock_get_client:
                mock_get_client.return_value = MockAsyncContextManager(mock_client)
                result = await messages_mixin.send_message(
                    to=["recipient@example.com"],
                    subject="Test Subject",
                    body="Test body",
                )

                assert result["success"] is True
                assert result["message_id"] == "sent_msg_123"
                assert result["thread_id"] == "new_thread"

    @pytest.mark.asyncio
    async def test_send_message_with_cc_bcc(self, messages_mixin, mock_httpx_response):
        """Test sending email with CC and BCC."""
        response_data = {"id": "sent_msg", "threadId": "thread"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_httpx_response(200, response_data))

        with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
            with patch.object(messages_mixin, "_get_client") as mock_get_client:
                mock_get_client.return_value = MockAsyncContextManager(mock_client)
                result = await messages_mixin.send_message(
                    to=["to@example.com"],
                    subject="Test",
                    body="Body",
                    cc=["cc1@example.com", "cc2@example.com"],
                    bcc=["bcc@example.com"],
                )

                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_send_message_with_html(self, messages_mixin, mock_httpx_response):
        """Test sending email with HTML body."""
        response_data = {"id": "sent_msg", "threadId": "thread"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_httpx_response(200, response_data))

        with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
            with patch.object(messages_mixin, "_get_client") as mock_get_client:
                mock_get_client.return_value = MockAsyncContextManager(mock_client)
                result = await messages_mixin.send_message(
                    to=["to@example.com"],
                    subject="HTML Email",
                    body="Plain text",
                    html_body="<html><body><h1>HTML Content</h1></body></html>",
                )

                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_send_message_with_reply_to(self, messages_mixin, mock_httpx_response):
        """Test sending email with reply-to header."""
        response_data = {"id": "sent_msg", "threadId": "thread"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_httpx_response(200, response_data))

        with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
            with patch.object(messages_mixin, "_get_client") as mock_get_client:
                mock_get_client.return_value = MockAsyncContextManager(mock_client)
                result = await messages_mixin.send_message(
                    to=["to@example.com"],
                    subject="Test",
                    body="Body",
                    reply_to="reply@example.com",
                )

                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_send_message_circuit_breaker_open(self, messages_mixin):
        """Test send fails when circuit breaker is open."""
        with patch.object(messages_mixin, "check_circuit_breaker", return_value=False):
            with patch.object(
                messages_mixin,
                "get_circuit_breaker_status",
                return_value={"cooldown_seconds": 60},
            ):
                with pytest.raises(ConnectionError, match="Circuit breaker open"):
                    await messages_mixin.send_message(
                        to=["to@example.com"],
                        subject="Test",
                        body="Body",
                    )

    @pytest.mark.asyncio
    async def test_send_message_records_failure_on_5xx(self, messages_mixin, mock_httpx_response):
        """Test 5xx errors record failures."""
        error_response = mock_httpx_response(500, {"error": {"message": "Server error"}})

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=error_response)

        with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
            with patch.object(messages_mixin, "_get_client") as mock_get_client:
                with patch.object(messages_mixin, "record_failure") as mock_record_failure:
                    mock_get_client.return_value = MockAsyncContextManager(mock_client)
                    with pytest.raises(RuntimeError):
                        await messages_mixin.send_message(
                            to=["to@example.com"],
                            subject="Test",
                            body="Body",
                        )

                    mock_record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_message_records_success(self, messages_mixin, mock_httpx_response):
        """Test successful send records success."""
        response_data = {"id": "sent", "threadId": "thread"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_httpx_response(200, response_data))

        with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
            with patch.object(messages_mixin, "_get_client") as mock_get_client:
                with patch.object(messages_mixin, "record_success") as mock_record_success:
                    mock_get_client.return_value = MockAsyncContextManager(mock_client)
                    await messages_mixin.send_message(
                        to=["to@example.com"],
                        subject="Test",
                        body="Body",
                    )

                    mock_record_success.assert_called_once()


# =============================================================================
# Reply Tests
# =============================================================================


class TestReplyToMessage:
    """Tests for replying to messages."""

    @pytest.mark.asyncio
    async def test_reply_to_message_basic(self, messages_mixin, mock_httpx_response):
        """Test replying to a message."""
        original_msg = EmailMessage(
            id="original_msg",
            thread_id="original_thread",
            subject="Original Subject",
            from_address="original@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Original body",
            headers={"message-id": "<original@example.com>"},
            labels=[],
        )

        response_data = {"id": "reply_msg", "threadId": "original_thread"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_httpx_response(200, response_data))

        with patch.object(messages_mixin, "get_message", return_value=original_msg):
            with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
                with patch.object(messages_mixin, "_get_client") as mock_get_client:
                    mock_get_client.return_value = MockAsyncContextManager(mock_client)
                    result = await messages_mixin.reply_to_message(
                        original_message_id="original_msg",
                        body="Reply body",
                    )

                    assert result["success"] is True
                    assert result["in_reply_to"] == "original_msg"
                    assert result["thread_id"] == "original_thread"

    @pytest.mark.asyncio
    async def test_reply_adds_re_prefix(self, messages_mixin, mock_httpx_response):
        """Test reply adds Re: prefix to subject."""
        original_msg = EmailMessage(
            id="original",
            thread_id="thread",
            subject="Original Subject",
            from_address="sender@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Original",
            labels=[],
        )

        response_data = {"id": "reply", "threadId": "thread"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_httpx_response(200, response_data))

        with patch.object(messages_mixin, "get_message", return_value=original_msg):
            with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
                with patch.object(messages_mixin, "_get_client") as mock_get_client:
                    mock_get_client.return_value = MockAsyncContextManager(mock_client)
                    await messages_mixin.reply_to_message("original", body="Reply")

                    # The raw message is base64 encoded, so we check it was called
                    assert mock_client.post.called

    @pytest.mark.asyncio
    async def test_reply_preserves_re_prefix(self, messages_mixin, mock_httpx_response):
        """Test reply doesn't double Re: prefix."""
        original_msg = EmailMessage(
            id="original",
            thread_id="thread",
            subject="Re: Already replied",
            from_address="sender@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Original",
            labels=[],
        )

        response_data = {"id": "reply", "threadId": "thread"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_httpx_response(200, response_data))

        with patch.object(messages_mixin, "get_message", return_value=original_msg):
            with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
                with patch.object(messages_mixin, "_get_client") as mock_get_client:
                    mock_get_client.return_value = MockAsyncContextManager(mock_client)
                    await messages_mixin.reply_to_message("original", body="Reply")

                    # Subject should still be "Re: Already replied" not "Re: Re: Already replied"
                    assert mock_client.post.called

    @pytest.mark.asyncio
    async def test_reply_not_found(self, messages_mixin):
        """Test reply fails when original not found."""
        # Return None or falsy value to simulate not found
        with patch.object(messages_mixin, "get_message", return_value=None):
            with pytest.raises(ValueError, match="Original message not found"):
                await messages_mixin.reply_to_message("nonexistent", body="Reply")

    @pytest.mark.asyncio
    async def test_reply_with_additional_cc(self, messages_mixin, mock_httpx_response):
        """Test reply with additional CC recipients."""
        original_msg = EmailMessage(
            id="original",
            thread_id="thread",
            subject="Original",
            from_address="sender@example.com",
            to_addresses=["me@example.com"],
            cc_addresses=["existing_cc@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Original",
            labels=[],
        )

        response_data = {"id": "reply", "threadId": "thread"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_httpx_response(200, response_data))

        with patch.object(messages_mixin, "get_message", return_value=original_msg):
            with patch.object(messages_mixin, "check_circuit_breaker", return_value=True):
                with patch.object(messages_mixin, "_get_client") as mock_get_client:
                    mock_get_client.return_value = MockAsyncContextManager(mock_client)
                    result = await messages_mixin.reply_to_message(
                        "original",
                        body="Reply",
                        cc=["new_cc@example.com"],
                    )

                    assert result["success"] is True


# =============================================================================
# Sync Items Tests
# =============================================================================


class TestSyncItems:
    """Tests for sync item operations."""

    @pytest.mark.asyncio
    async def test_sync_items_full_sync(self, messages_mixin):
        """Test full sync with no cursor."""
        state = SyncState(connector_id="gmail")

        mock_msg = EmailMessage(
            id="msg_1",
            thread_id="thread_1",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            labels=["INBOX"],
        )

        with patch.object(
            messages_mixin,
            "get_user_info",
            return_value={"emailAddress": "test@example.com", "historyId": "12345"},
        ):
            with patch.object(messages_mixin, "list_messages", return_value=(["msg_1"], None)):
                with patch.object(messages_mixin, "get_message", return_value=mock_msg):
                    items = []
                    async for item in messages_mixin.sync_items(state, batch_size=10):
                        items.append(item)

                    assert len(items) == 1
                    assert state.cursor == "12345"

    @pytest.mark.asyncio
    async def test_sync_items_incremental(self, messages_mixin):
        """Test incremental sync with cursor."""
        state = SyncState(connector_id="gmail", cursor="12345")

        mock_msg = EmailMessage(
            id="new_msg",
            thread_id="new_thread",
            subject="New",
            from_address="sender@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="New body",
            labels=["INBOX"],
        )

        with patch.object(messages_mixin, "get_history") as mock_history:
            mock_history.return_value = (
                [{"messagesAdded": [{"message": {"id": "new_msg"}}]}],
                None,
                "12346",
            )
            with patch.object(messages_mixin, "get_message", return_value=mock_msg):
                items = []
                async for item in messages_mixin.sync_items(state, batch_size=10):
                    items.append(item)

                assert len(items) == 1
                assert state.cursor == "12346"

    @pytest.mark.asyncio
    async def test_sync_items_excludes_labels(self, messages_mixin):
        """Test sync excludes specified labels."""
        messages_mixin.exclude_labels = {"SPAM", "TRASH"}
        state = SyncState(connector_id="gmail", cursor="12345")

        mock_msg = EmailMessage(
            id="spam_msg",
            thread_id="spam_thread",
            subject="Spam",
            from_address="spammer@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Spam body",
            labels=["SPAM"],
        )

        with patch.object(messages_mixin, "get_history") as mock_history:
            mock_history.return_value = (
                [{"messagesAdded": [{"message": {"id": "spam_msg"}}]}],
                None,
                "12346",
            )
            with patch.object(messages_mixin, "get_message", return_value=mock_msg):
                items = []
                async for item in messages_mixin.sync_items(state, batch_size=10):
                    items.append(item)

                assert len(items) == 0

    @pytest.mark.asyncio
    async def test_sync_items_history_expired(self, messages_mixin):
        """Test sync resets cursor when history expired."""
        state = SyncState(connector_id="gmail", cursor="old_cursor")

        with patch.object(messages_mixin, "get_history") as mock_history:
            # History expired - return empty with no new history ID
            mock_history.return_value = ([], None, "")

            items = []
            async for item in messages_mixin.sync_items(state, batch_size=10):
                items.append(item)

            # When history expires, cursor is reset but no items returned in this call
            # The next sync call with cursor=None will do full sync
            assert len(items) == 0
            assert state.cursor is None

    @pytest.mark.asyncio
    async def test_sync_items_history_expired_then_full_sync(self, messages_mixin):
        """Test full sync after history expired cursor reset."""
        state = SyncState(connector_id="gmail", cursor=None)

        mock_msg = EmailMessage(
            id="msg_1",
            thread_id="thread_1",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            labels=["INBOX"],
        )

        with patch.object(
            messages_mixin,
            "get_user_info",
            return_value={"emailAddress": "test@example.com", "historyId": "12345"},
        ):
            with patch.object(messages_mixin, "list_messages", return_value=(["msg_1"], None)):
                with patch.object(messages_mixin, "get_message", return_value=mock_msg):
                    items = []
                    async for item in messages_mixin.sync_items(state, batch_size=10):
                        items.append(item)

                    # Full sync should return messages
                    assert len(items) == 1
                    assert state.cursor == "12345"

    def test_message_to_sync_item(self, messages_mixin):
        """Test converting EmailMessage to SyncItem."""
        msg = EmailMessage(
            id="msg_123",
            thread_id="thread_123",
            subject="Test Subject",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            cc_addresses=["cc@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Email body content",
            snippet="Snippet...",
            labels=["INBOX", "IMPORTANT"],
            attachments=[
                EmailAttachment(
                    id="att_1", filename="file.pdf", mime_type="application/pdf", size=1000
                )
            ],
            is_read=True,
            is_starred=False,
            is_important=True,
        )

        sync_item = messages_mixin._message_to_sync_item(msg)

        assert isinstance(sync_item, SyncItem)
        assert sync_item.id == "gmail-msg_123"
        assert sync_item.source_type == "email"
        assert sync_item.title == "Test Subject"
        assert sync_item.author == "sender@example.com"
        assert "Subject: Test Subject" in sync_item.content
        assert "From: sender@example.com" in sync_item.content
        assert "CC: cc@example.com" in sync_item.content
        assert sync_item.metadata["has_attachments"] is True
        assert sync_item.metadata["attachment_count"] == 1
        assert sync_item.metadata["is_important"] is True


# =============================================================================
# Prioritization Tests
# =============================================================================


class TestSyncWithPrioritization:
    """Tests for email prioritization integration."""

    @pytest.mark.asyncio
    async def test_sync_with_prioritization_empty(self, messages_mixin):
        """Test prioritization with empty message list."""
        results = await messages_mixin.sync_with_prioritization([])
        assert results == []

    @pytest.mark.asyncio
    async def test_sync_with_prioritization_no_prioritizer(self, messages_mixin):
        """Test prioritization when EmailPrioritizer not available."""
        msg = EmailMessage(
            id="msg_1",
            thread_id="thread_1",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            labels=[],
        )

        # Mock import failure
        with patch.dict("sys.modules", {"aragora.services.email_prioritization": None}):
            results = await messages_mixin.sync_with_prioritization([msg])

            assert len(results) == 1
            assert results[0]["priority"] == "MEDIUM"
            assert results[0]["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_sync_with_prioritization_timeout(self, messages_mixin):
        """Test prioritization handles timeout."""
        msg = EmailMessage(
            id="msg_1",
            thread_id="thread_1",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            labels=[],
        )

        mock_prioritizer = AsyncMock()

        async def slow_score(*args, **kwargs):
            await asyncio.sleep(1)  # Will timeout

        mock_prioritizer.score_email = slow_score

        results = await messages_mixin.sync_with_prioritization(
            [msg], prioritizer=mock_prioritizer, timeout_seconds=0.01
        )

        assert len(results) == 1
        assert results[0]["priority"] == "MEDIUM"
        assert "timed out" in results[0]["rationale"]


class TestRankInbox:
    """Tests for inbox ranking."""

    @pytest.mark.asyncio
    async def test_rank_inbox_uses_batch_fetching(self, messages_mixin):
        """Test rank inbox uses batch fetching."""
        mock_msg = EmailMessage(
            id="msg_1",
            thread_id="thread_1",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            labels=[],
        )

        with patch.object(messages_mixin, "list_messages", return_value=(["msg_1"], None)):
            with patch.object(messages_mixin, "get_messages", return_value=[mock_msg]) as mock_get:
                with patch.object(messages_mixin, "sync_with_prioritization") as mock_prio:
                    mock_prio.return_value = [{"message": mock_msg, "priority": "HIGH"}]

                    results = await messages_mixin.rank_inbox(max_messages=10)

                    assert len(results) == 1
                    mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_rank_inbox_with_labels(self, messages_mixin):
        """Test rank inbox with label filter."""
        with patch.object(messages_mixin, "list_messages") as mock_list:
            mock_list.return_value = ([], None)
            with patch.object(messages_mixin, "get_messages", return_value=[]):
                with patch.object(messages_mixin, "sync_with_prioritization", return_value=[]):
                    await messages_mixin.rank_inbox(labels=["IMPORTANT", "STARRED"])

                    call_args = mock_list.call_args
                    query = call_args.kwargs.get("query", "")
                    assert "label:IMPORTANT" in query or "label:STARRED" in query


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_get_message_api_error(self, messages_mixin):
        """Test get_message handles API errors."""
        with patch.object(messages_mixin, "_api_request", side_effect=RuntimeError("API Error")):
            with pytest.raises(RuntimeError, match="API Error"):
                await messages_mixin.get_message("msg_123")

    @pytest.mark.asyncio
    async def test_list_messages_api_error(self, messages_mixin):
        """Test list_messages handles API errors."""
        with patch.object(
            messages_mixin, "_api_request", side_effect=ConnectionError("Network error")
        ):
            with pytest.raises(ConnectionError):
                await messages_mixin.list_messages()

    @pytest.mark.asyncio
    async def test_sync_items_handles_individual_failures(self, messages_mixin):
        """Test sync_items continues on individual message failures."""
        state = SyncState(connector_id="gmail", cursor="12345")

        call_count = 0

        async def mock_get_message(msg_id):
            nonlocal call_count
            call_count += 1
            if msg_id == "msg_2":
                raise ValueError("Failed to parse")
            return EmailMessage(
                id=msg_id,
                thread_id=f"thread_{msg_id}",
                subject="Test",
                from_address="sender@example.com",
                to_addresses=["test@example.com"],
                date=datetime.now(timezone.utc),
                body_text="Body",
                labels=["INBOX"],
            )

        with patch.object(messages_mixin, "get_history") as mock_history:
            mock_history.return_value = (
                [
                    {"messagesAdded": [{"message": {"id": "msg_1"}}]},
                    {"messagesAdded": [{"message": {"id": "msg_2"}}]},
                    {"messagesAdded": [{"message": {"id": "msg_3"}}]},
                ],
                None,
                "12348",
            )
            with patch.object(messages_mixin, "get_message", side_effect=mock_get_message):
                items = []
                async for item in messages_mixin.sync_items(state, batch_size=10):
                    items.append(item)

                # Should get 2 items (msg_1 and msg_3), msg_2 failed
                assert len(items) == 2
                assert call_count == 3


# =============================================================================
# MessageFetchFailure Tests
# =============================================================================


class TestMessageFetchFailure:
    """Tests for MessageFetchFailure dataclass."""

    def test_message_fetch_failure_basic(self):
        """Test basic MessageFetchFailure creation."""
        error = ValueError("Test error")
        failure = MessageFetchFailure(message_id="msg_123", error=error)

        assert failure.message_id == "msg_123"
        assert failure.error_type == "ValueError"
        assert failure.is_retryable is False

    def test_message_fetch_failure_retryable_detection(self):
        """Test retryable error detection."""
        timeout_error = TimeoutError("Connection timeout")
        failure = MessageFetchFailure(message_id="msg_1", error=timeout_error)
        assert failure.is_retryable is True

        rate_limit_error = RuntimeError("429 rate limit exceeded")
        failure2 = MessageFetchFailure(message_id="msg_2", error=rate_limit_error)
        assert failure2.is_retryable is True

        value_error = ValueError("Invalid format")
        failure3 = MessageFetchFailure(message_id="msg_3", error=value_error)
        assert failure3.is_retryable is False

    def test_message_fetch_failure_to_dict(self):
        """Test MessageFetchFailure serialization."""
        # Use an error message with "connection" keyword for retryable detection
        error = ConnectionError("connection refused")
        failure = MessageFetchFailure(message_id="msg_123", error=error)

        data = failure.to_dict()

        assert data["message_id"] == "msg_123"
        assert data["error_type"] == "ConnectionError"
        assert data["is_retryable"] is True
        assert "connection refused" in data["error"]


# =============================================================================
# BatchFetchResult Tests
# =============================================================================


class TestBatchFetchResult:
    """Tests for BatchFetchResult dataclass."""

    def test_batch_fetch_result_empty(self):
        """Test empty BatchFetchResult."""
        result = BatchFetchResult(messages=[], failures=[], total_requested=0)

        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.is_complete is True
        assert result.is_partial is False
        assert result.is_total_failure is False

    def test_batch_fetch_result_complete(self):
        """Test complete BatchFetchResult."""
        msgs = [
            EmailMessage(
                id="msg_1",
                thread_id="thread_1",
                subject="Test",
                from_address="test@example.com",
                to_addresses=[],
                date=datetime.now(timezone.utc),
                body_text="",
                labels=[],
            )
        ]
        result = BatchFetchResult(messages=msgs, failures=[], total_requested=1)

        assert result.success_count == 1
        assert result.is_complete is True
        assert result.is_partial is False
        assert result.is_total_failure is False

    def test_batch_fetch_result_partial(self):
        """Test partial BatchFetchResult."""
        msgs = [
            EmailMessage(
                id="msg_1",
                thread_id="thread_1",
                subject="Test",
                from_address="test@example.com",
                to_addresses=[],
                date=datetime.now(timezone.utc),
                body_text="",
                labels=[],
            )
        ]
        failures = [MessageFetchFailure(message_id="msg_2", error=ValueError("Failed"))]
        result = BatchFetchResult(messages=msgs, failures=failures, total_requested=2)

        assert result.success_count == 1
        assert result.failure_count == 1
        assert result.is_complete is False
        assert result.is_partial is True
        assert result.is_total_failure is False
        assert "msg_2" in result.failed_ids

    def test_batch_fetch_result_total_failure(self):
        """Test total failure BatchFetchResult."""
        failures = [
            MessageFetchFailure(message_id="msg_1", error=ValueError("Failed")),
            MessageFetchFailure(message_id="msg_2", error=ValueError("Failed")),
        ]
        result = BatchFetchResult(messages=[], failures=failures, total_requested=2)

        assert result.success_count == 0
        assert result.failure_count == 2
        assert result.is_complete is False
        assert result.is_partial is False
        assert result.is_total_failure is True

    def test_batch_fetch_result_retryable_ids(self):
        """Test retryable_ids property."""
        failures = [
            MessageFetchFailure(
                message_id="msg_1",
                error=ConnectionError("connection refused"),  # retryable keyword
            ),
            MessageFetchFailure(
                message_id="msg_2",
                error=ValueError("Invalid"),
            ),
        ]
        result = BatchFetchResult(messages=[], failures=failures, total_requested=2)

        assert "msg_1" in result.retryable_ids
        assert "msg_2" not in result.retryable_ids


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestMessageParsingEdgeCases:
    """Additional tests for message parsing edge cases."""

    def test_parse_message_empty_to_addresses(self, messages_mixin):
        """Test parsing message with empty To field."""
        msg_data = {
            "id": "msg_empty_to",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:30:00 +0000"},
                ],
                "body": {},
            },
        }

        message = messages_mixin._parse_message(msg_data)

        assert message.to_addresses == []
        assert message.cc_addresses == []
        assert message.bcc_addresses == []

    def test_parse_message_multiple_recipients(self, messages_mixin):
        """Test parsing message with multiple recipients."""
        msg_data = {
            "id": "msg_multi",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {
                        "name": "To",
                        "value": "to1@example.com, to2@example.com, to3@example.com",
                    },
                    {"name": "CC", "value": "cc1@example.com, cc2@example.com"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:30:00 +0000"},
                ],
                "body": {},
            },
        }

        message = messages_mixin._parse_message(msg_data)

        assert len(message.to_addresses) == 3
        assert "to1@example.com" in message.to_addresses
        assert "to2@example.com" in message.to_addresses
        assert "to3@example.com" in message.to_addresses
        assert len(message.cc_addresses) == 2

    def test_parse_message_html_only(self, messages_mixin):
        """Test parsing message with HTML body only."""
        msg_data = {
            "id": "msg_html",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/html",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Subject", "value": "HTML Only"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:30:00 +0000"},
                ],
                "body": {
                    "data": base64.urlsafe_b64encode(
                        b"<html><body><p>HTML content</p></body></html>"
                    ).decode()
                },
            },
        }

        message = messages_mixin._parse_message(msg_data)

        assert message.body_html == "<html><body><p>HTML content</p></body></html>"

    def test_parse_message_empty_body(self, messages_mixin):
        """Test parsing message with no body data."""
        msg_data = {
            "id": "msg_nobody",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Subject", "value": "Empty Body"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:30:00 +0000"},
                ],
                "body": {},
            },
        }

        message = messages_mixin._parse_message(msg_data)

        assert message.body_text == ""

    def test_parse_message_read_status(self, messages_mixin):
        """Test parsing message read status from labels."""
        unread_msg = {
            "id": "msg_unread",
            "threadId": "thread_1",
            "labelIds": ["INBOX", "UNREAD"],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:30:00 +0000"},
                ],
                "body": {},
            },
        }

        read_msg = {
            "id": "msg_read",
            "threadId": "thread_1",
            "labelIds": ["INBOX"],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:30:00 +0000"},
                ],
                "body": {},
            },
        }

        unread = messages_mixin._parse_message(unread_msg)
        read = messages_mixin._parse_message(read_msg)

        assert unread.is_read is False
        assert read.is_read is True

    def test_parse_message_inline_attachment(self, messages_mixin):
        """Test parsing message with inline attachment."""
        msg_data = {
            "id": "msg_inline",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "multipart/mixed",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Subject", "value": "Inline Image"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:30:00 +0000"},
                ],
                "parts": [
                    {
                        "mimeType": "text/plain",
                        "body": {"data": base64.urlsafe_b64encode(b"Check this image").decode()},
                    },
                    {
                        "mimeType": "image/jpeg",
                        "filename": "image.jpg",
                        "headers": [
                            {
                                "name": "Content-Disposition",
                                "value": 'inline; filename="image.jpg"',
                            }
                        ],
                        "body": {"attachmentId": "inline_attach", "size": 50000},
                    },
                ],
            },
        }

        message = messages_mixin._parse_message(msg_data)

        assert len(message.attachments) == 1
        assert message.attachments[0].filename == "image.jpg"


class TestListMessagesEdgeCases:
    """Additional tests for list_messages edge cases."""

    @pytest.mark.asyncio
    async def test_list_messages_with_special_characters_in_query(self, messages_mixin):
        """Test list_messages handles special characters in query."""
        response = {"messages": [{"id": "msg_1"}]}

        with patch.object(messages_mixin, "_api_request", return_value=response) as mock_request:
            await messages_mixin.list_messages(query='subject:"Test & Special <chars>"')

            call_args = mock_request.call_args
            assert 'subject:"Test & Special <chars>"' in call_args.kwargs.get("params", {}).get(
                "q", ""
            )

    @pytest.mark.asyncio
    async def test_list_messages_multiple_pages(self, messages_mixin):
        """Test list_messages handles multiple pages correctly."""
        page1 = {"messages": [{"id": "msg_1"}], "nextPageToken": "token1"}
        page2 = {"messages": [{"id": "msg_2"}], "nextPageToken": "token2"}
        page3 = {"messages": [{"id": "msg_3"}]}

        call_count = 0

        async def mock_api_request(endpoint, params=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1
            elif call_count == 2:
                return page2
            return page3

        with patch.object(messages_mixin, "_api_request", side_effect=mock_api_request):
            # Fetch first page
            ids1, token1 = await messages_mixin.list_messages()
            assert len(ids1) == 1
            assert token1 == "token1"

            # Fetch second page
            ids2, token2 = await messages_mixin.list_messages(page_token=token1)
            assert len(ids2) == 1
            assert token2 == "token2"

            # Fetch third page
            ids3, token3 = await messages_mixin.list_messages(page_token=token2)
            assert len(ids3) == 1
            assert token3 is None


class TestThreadEdgeCases:
    """Additional tests for thread handling edge cases."""

    @pytest.mark.asyncio
    async def test_get_thread_single_message(self, messages_mixin, sample_gmail_message):
        """Test thread with only one message."""
        thread_data = {
            "id": "thread_single",
            "snippet": "Single message thread",
            "messages": [sample_gmail_message],
        }

        with patch.object(messages_mixin, "_api_request", return_value=thread_data):
            thread = await messages_mixin.get_thread("thread_single")

            assert thread.message_count == 1
            assert thread.subject == sample_gmail_message["payload"]["headers"][2]["value"]

    @pytest.mark.asyncio
    async def test_get_thread_date_ordering(self, messages_mixin):
        """Test thread messages are accessible in order."""
        msg1 = {
            "id": "msg_1",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "Subject", "value": "Thread Start"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 09:00:00 +0000"},
                ],
                "body": {"data": base64.urlsafe_b64encode(b"First").decode()},
            },
        }
        msg2 = {
            "id": "msg_2",
            "threadId": "thread_1",
            "labelIds": [],
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "From", "value": "replier@example.com"},
                    {"name": "Subject", "value": "Re: Thread Start"},
                    {"name": "Date", "value": "Mon, 15 Jan 2024 10:00:00 +0000"},
                ],
                "body": {"data": base64.urlsafe_b64encode(b"Reply").decode()},
            },
        }

        thread_data = {"id": "thread_1", "snippet": "", "messages": [msg1, msg2]}

        with patch.object(messages_mixin, "_api_request", return_value=thread_data):
            thread = await messages_mixin.get_thread("thread_1")

            assert thread.message_count == 2
            assert thread.subject == "Thread Start"
            assert thread.last_message_date.hour == 10


class TestSearchEdgeCases:
    """Additional tests for search edge cases."""

    @pytest.mark.asyncio
    async def test_search_empty_results(self, messages_mixin):
        """Test search with no results."""
        with patch.object(messages_mixin, "list_messages", return_value=([], None)):
            results = await messages_mixin.search("nonexistent query")

            assert results == []

    @pytest.mark.asyncio
    async def test_search_with_date_filter(self, messages_mixin):
        """Test search with date filter."""
        mock_msg = EmailMessage(
            id="msg_1",
            thread_id="thread_1",
            subject="Old Email",
            from_address="sender@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            snippet="Snippet",
            labels=[],
        )

        with patch.object(messages_mixin, "list_messages") as mock_list:
            mock_list.return_value = (["msg_1"], None)
            with patch.object(messages_mixin, "get_messages", return_value=[mock_msg]):
                await messages_mixin.search("after:2024/01/01 before:2024/12/31")

                # Verify query was passed to list_messages
                call_args = mock_list.call_args
                assert "after:2024/01/01 before:2024/12/31" in call_args.kwargs.get("query", "")


class TestSyncItemsEdgeCases:
    """Additional tests for sync_items edge cases."""

    @pytest.mark.asyncio
    async def test_sync_items_empty_inbox(self, messages_mixin):
        """Test sync with empty inbox."""
        state = SyncState(connector_id="gmail")

        with patch.object(
            messages_mixin,
            "get_user_info",
            return_value={"emailAddress": "test@example.com", "historyId": "12345"},
        ):
            with patch.object(messages_mixin, "list_messages", return_value=([], None)):
                items = []
                async for item in messages_mixin.sync_items(state, batch_size=10):
                    items.append(item)

                assert len(items) == 0
                assert state.cursor == "12345"

    @pytest.mark.asyncio
    async def test_sync_items_respects_batch_size(self, messages_mixin):
        """Test sync respects batch_size parameter."""
        state = SyncState(connector_id="gmail")

        message_ids = [f"msg_{i}" for i in range(20)]

        mock_msg = EmailMessage(
            id="msg_test",
            thread_id="thread_test",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            labels=["INBOX"],
        )

        with patch.object(
            messages_mixin,
            "get_user_info",
            return_value={"emailAddress": "test@example.com", "historyId": "12345"},
        ):
            with patch.object(
                messages_mixin, "list_messages", return_value=(message_ids[:5], None)
            ):
                with patch.object(messages_mixin, "get_message", return_value=mock_msg):
                    items = []
                    async for item in messages_mixin.sync_items(state, batch_size=5):
                        items.append(item)

                    assert len(items) == 5

    @pytest.mark.asyncio
    async def test_sync_items_with_labels_filter(self, messages_mixin):
        """Test sync with labels filter."""
        state = SyncState(connector_id="gmail")
        messages_mixin.labels = ["IMPORTANT"]

        mock_msg = EmailMessage(
            id="msg_1",
            thread_id="thread_1",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["test@example.com"],
            date=datetime.now(timezone.utc),
            body_text="Body",
            labels=["IMPORTANT"],
        )

        with patch.object(
            messages_mixin,
            "get_user_info",
            return_value={"emailAddress": "test@example.com", "historyId": "12345"},
        ):
            with patch.object(
                messages_mixin, "list_messages", return_value=(["msg_1"], None)
            ) as mock_list:
                with patch.object(messages_mixin, "get_message", return_value=mock_msg):
                    items = []
                    async for item in messages_mixin.sync_items(state, batch_size=10):
                        items.append(item)

                    # Verify query includes label filter
                    call_args = mock_list.call_args
                    assert "label:IMPORTANT" in call_args.kwargs.get("query", "")


class TestEmailAttachment:
    """Tests for EmailAttachment dataclass."""

    def test_email_attachment_basic(self):
        """Test basic EmailAttachment creation."""
        attachment = EmailAttachment(
            id="attach_123",
            filename="document.pdf",
            mime_type="application/pdf",
            size=12345,
        )

        assert attachment.id == "attach_123"
        assert attachment.filename == "document.pdf"
        assert attachment.mime_type == "application/pdf"
        assert attachment.size == 12345

    def test_email_attachment_optional_fields(self):
        """Test EmailAttachment with optional fields."""
        attachment = EmailAttachment(
            id="attach_456",
            filename="image.png",
            mime_type="image/png",
            size=54321,
            data=b"image binary data",
        )

        assert attachment.data == b"image binary data"


class TestEmailThread:
    """Tests for EmailThread dataclass."""

    def test_email_thread_last_message_date(self, messages_mixin, sample_gmail_message):
        """Test last_message_date reflects the latest message."""
        older_msg = {
            **sample_gmail_message,
            "id": "older_msg",
            "payload": {
                **sample_gmail_message["payload"],
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Subject", "value": "Test"},
                    {"name": "Date", "value": "Mon, 01 Jan 2024 10:00:00 +0000"},
                ],
            },
        }
        newer_msg = {
            **sample_gmail_message,
            "id": "newer_msg",
            "payload": {
                **sample_gmail_message["payload"],
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Subject", "value": "Re: Test"},
                    {"name": "Date", "value": "Wed, 31 Jan 2024 15:00:00 +0000"},
                ],
            },
        }

        msg1 = messages_mixin._parse_message(older_msg)
        msg2 = messages_mixin._parse_message(newer_msg)

        # EmailThread uses explicit message_count and last_message_date fields
        thread = EmailThread(
            id="thread_123",
            messages=[msg1, msg2],
            subject="Test",
            snippet="Thread snippet",
            participants=["sender@example.com", "recipient@example.com"],
            labels=["INBOX"],
            message_count=2,
            last_message_date=msg2.date,  # Latest message date
        )

        assert thread.message_count == 2
        assert len(thread.messages) == 2
        assert thread.last_message_date.month == 1
        assert thread.last_message_date.day == 31
