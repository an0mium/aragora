"""
Tests for Email Debate Handler (aragora/server/handlers/email_debate.py).

Covers all endpoints and behavior:
- POST /api/v1/email/prioritize - Single email prioritization
- POST /api/v1/email/prioritize/batch - Batch email prioritization
- POST /api/v1/email/triage - Full inbox triage with sort/group
- GET returns 405 on all routes
- can_handle routing
- Rate limiting
- Input validation (missing fields, empty bodies, invalid JSON)
- Service errors (500 responses)
- Edge cases (large inputs, special characters, malformed dates)
- Security (path traversal, injection attempts)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.email_debate import EmailDebateHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract parsed body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("body", result)
    return json.loads(result.body.decode("utf-8"))


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    return result.status_code


# ---------------------------------------------------------------------------
# Mock data model classes matching EmailDebateService return types
# ---------------------------------------------------------------------------


class MockEmailPriority(str, Enum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    SPAM = "spam"


class MockEmailCategory(str, Enum):
    ACTION_REQUIRED = "action_required"
    REPLY_NEEDED = "reply_needed"
    FYI = "fyi"
    MEETING = "meeting"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    SOCIAL = "social"
    SPAM = "spam"
    PHISHING = "phishing"
    UNKNOWN = "unknown"


@dataclass
class MockEmailDebateResult:
    """Mock result matching EmailDebateResult interface."""

    message_id: str = "msg-123"
    priority: MockEmailPriority = MockEmailPriority.NORMAL
    category: MockEmailCategory = MockEmailCategory.FYI
    confidence: float = 0.85
    reasoning: str = "Test reasoning"
    action_items: list[str] = field(default_factory=list)
    suggested_labels: list[str] = field(default_factory=list)
    is_spam: bool = False
    is_phishing: bool = False
    sender_reputation: float | None = None
    debate_id: str | None = "debate-001"
    duration_seconds: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "priority": self.priority.value,
            "category": self.category.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "action_items": self.action_items,
            "suggested_labels": self.suggested_labels,
            "is_spam": self.is_spam,
            "is_phishing": self.is_phishing,
            "sender_reputation": self.sender_reputation,
            "debate_id": self.debate_id,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class MockBatchEmailResult:
    """Mock result matching BatchEmailResult interface."""

    results: list[MockEmailDebateResult] = field(default_factory=list)
    total_emails: int = 0
    processed_emails: int = 0
    duration_seconds: float = 2.0
    errors: list[str] = field(default_factory=list)

    @property
    def by_priority(self) -> dict[str, list[MockEmailDebateResult]]:
        grouped: dict[str, list[MockEmailDebateResult]] = {}
        for r in self.results:
            key = r.priority.value
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
        return grouped

    @property
    def urgent_count(self) -> int:
        return len([r for r in self.results if r.priority == MockEmailPriority.URGENT])

    @property
    def action_required_count(self) -> int:
        return len(
            [r for r in self.results if r.category == MockEmailCategory.ACTION_REQUIRED]
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_http_handler(body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with JSON body."""
    mock = MagicMock()
    mock.client_address = ("127.0.0.1", 12345)
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        mock.headers = {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "application/json",
        }
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = body_bytes
    else:
        mock.headers = {"Content-Length": "2", "Content-Type": "application/json"}
        mock.rfile = MagicMock()
        mock.rfile.read.return_value = b"{}"
    return mock


@pytest.fixture
def handler():
    """Create an EmailDebateHandler instance."""
    return EmailDebateHandler()


@pytest.fixture
def handler_with_ctx():
    """Create an EmailDebateHandler with a custom context."""
    return EmailDebateHandler(ctx={"user_store": MagicMock()})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the module-level rate limiter before each test."""
    import aragora.server.handlers.email_debate as mod

    mod._email_debate_limiter._requests = {}
    yield


def _single_email_body(**overrides) -> dict:
    """Build a typical single email request body."""
    body = {
        "subject": "Meeting tomorrow",
        "body": "Hi, can we meet tomorrow at 3pm?",
        "sender": "john@example.com",
        "received_at": "2024-01-15T10:30:00Z",
        "message_id": "msg-123",
        "user_id": "user-456",
    }
    body.update(overrides)
    return body


def _batch_email_body(count: int = 2, **overrides) -> dict:
    """Build a typical batch email request body."""
    emails = []
    for i in range(count):
        emails.append(
            {
                "subject": f"Email {i}",
                "body": f"Body of email {i}",
                "sender": f"sender{i}@example.com",
                "received_at": "2024-01-15T10:30:00Z",
                "message_id": f"msg-{i}",
            }
        )
    body: dict[str, Any] = {"emails": emails, "user_id": "user-456"}
    body.update(overrides)
    return body


def _make_single_result(**overrides) -> MockEmailDebateResult:
    """Create a single mock email debate result."""
    kwargs: dict[str, Any] = {
        "message_id": "msg-123",
        "priority": MockEmailPriority.NORMAL,
        "category": MockEmailCategory.FYI,
        "confidence": 0.85,
    }
    kwargs.update(overrides)
    return MockEmailDebateResult(**kwargs)


def _make_batch_result(
    count: int = 2, urgent: int = 0, action_required: int = 0
) -> MockBatchEmailResult:
    """Create a mock batch result with specified counts."""
    results = []
    for i in range(count):
        priority = MockEmailPriority.NORMAL
        category = MockEmailCategory.FYI
        if i < urgent:
            priority = MockEmailPriority.URGENT
        if i < action_required:
            category = MockEmailCategory.ACTION_REQUIRED
        results.append(
            MockEmailDebateResult(
                message_id=f"msg-{i}",
                priority=priority,
                category=category,
                confidence=0.85 - (i * 0.05),
            )
        )
    return MockBatchEmailResult(
        results=results,
        total_emails=count,
        processed_emails=count,
        duration_seconds=1.5,
    )


# ============================================================================
# Routing / can_handle Tests
# ============================================================================


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_can_handle_prioritize(self, handler):
        assert handler.can_handle("/api/v1/email/prioritize")

    def test_can_handle_prioritize_batch(self, handler):
        assert handler.can_handle("/api/v1/email/prioritize/batch")

    def test_can_handle_triage(self, handler):
        assert handler.can_handle("/api/v1/email/triage")

    def test_cannot_handle_unknown_path(self, handler):
        assert not handler.can_handle("/api/v1/email/unknown")

    def test_cannot_handle_wrong_prefix(self, handler):
        assert not handler.can_handle("/api/v1/other/prioritize")

    def test_cannot_handle_empty_path(self, handler):
        assert not handler.can_handle("")

    def test_cannot_handle_root(self, handler):
        assert not handler.can_handle("/")

    def test_cannot_handle_partial_match(self, handler):
        assert not handler.can_handle("/api/v1/email")

    def test_cannot_handle_extra_suffix(self, handler):
        assert not handler.can_handle("/api/v1/email/prioritize/extra")

    def test_cannot_handle_case_sensitive(self, handler):
        assert not handler.can_handle("/api/v1/email/Prioritize")

    def test_routes_list(self, handler):
        """ROUTES class attribute contains all expected paths."""
        assert "/api/v1/email/prioritize" in handler.ROUTES
        assert "/api/v1/email/prioritize/batch" in handler.ROUTES
        assert "/api/v1/email/triage" in handler.ROUTES
        assert len(handler.ROUTES) == 3


# ============================================================================
# GET (handle) Tests - Should Return 405
# ============================================================================


class TestGetMethodNotAllowed:
    """GET requests return 405 for all routes."""

    def test_get_prioritize_returns_405(self, handler):
        mock = _make_mock_http_handler()
        result = handler.handle("/api/v1/email/prioritize", {}, mock)
        assert _status(result) == 405
        assert "POST" in _body(result).get("error", "")

    def test_get_prioritize_batch_returns_405(self, handler):
        mock = _make_mock_http_handler()
        result = handler.handle("/api/v1/email/prioritize/batch", {}, mock)
        assert _status(result) == 405

    def test_get_triage_returns_405(self, handler):
        mock = _make_mock_http_handler()
        result = handler.handle("/api/v1/email/triage", {}, mock)
        assert _status(result) == 405

    def test_get_unknown_route_returns_405(self, handler):
        """Even unknown routes return 405 for GET (handle covers all)."""
        mock = _make_mock_http_handler()
        result = handler.handle("/api/v1/email/whatever", {}, mock)
        assert _status(result) == 405


# ============================================================================
# Initialization Tests
# ============================================================================


class TestInitialization:
    """Tests for handler initialization."""

    def test_default_context(self):
        h = EmailDebateHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        ctx = {"key": "value"}
        h = EmailDebateHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_none_context_becomes_empty(self):
        h = EmailDebateHandler(ctx=None)
        assert h.ctx == {}


# ============================================================================
# POST /api/v1/email/prioritize - Single Prioritization
# ============================================================================


class TestPrioritizeSingle:
    """Tests for single email prioritization."""

    @pytest.mark.asyncio
    async def test_prioritize_success(self, handler):
        """Happy path: single email prioritized successfully."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200
        data = _body(result)
        assert data["message_id"] == "msg-123"
        assert data["priority"] == "normal"
        assert data["category"] == "fyi"

    @pytest.mark.asyncio
    async def test_prioritize_with_only_subject(self, handler):
        """Email with only subject and no body should succeed."""
        body = {"subject": "Test subject"}
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_prioritize_with_only_body(self, handler):
        """Email with only body and no subject should succeed."""
        body = {"body": "Some email body text"}
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_prioritize_missing_subject_and_body(self, handler):
        """Missing both subject and body returns 400."""
        body = {"sender": "john@example.com"}
        mock_http = _make_mock_http_handler(body)

        result = await handler.handle_post(
            "/api/v1/email/prioritize", {}, mock_http
        )

        assert _status(result) == 400
        assert "subject or body" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_prioritize_empty_subject_and_body(self, handler):
        """Empty string subject and body is treated as missing."""
        body = {"subject": "", "body": ""}
        mock_http = _make_mock_http_handler(body)

        result = await handler.handle_post(
            "/api/v1/email/prioritize", {}, mock_http
        )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_prioritize_passes_fast_mode(self, handler):
        """fast_mode parameter is forwarded to service."""
        body = _single_email_body(fast_mode=False)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            await handler.handle_post("/api/v1/email/prioritize", {}, mock_http)

            MockService.assert_called_once_with(
                fast_mode=False, enable_pii_redaction=True
            )

    @pytest.mark.asyncio
    async def test_prioritize_passes_pii_redaction(self, handler):
        """enable_pii_redaction parameter is forwarded."""
        body = _single_email_body(enable_pii_redaction=False)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            await handler.handle_post("/api/v1/email/prioritize", {}, mock_http)

            MockService.assert_called_once_with(
                fast_mode=True, enable_pii_redaction=False
            )

    @pytest.mark.asyncio
    async def test_prioritize_default_user_id(self, handler):
        """User ID defaults to 'default' when not provided."""
        body = {"subject": "Test"}
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            await handler.handle_post("/api/v1/email/prioritize", {}, mock_http)

            call_args = instance.prioritize_email.call_args
            assert call_args[0][1] == "default"  # user_id arg

    @pytest.mark.asyncio
    async def test_prioritize_custom_user_id(self, handler):
        """Custom user_id is passed through."""
        body = _single_email_body(user_id="custom-user")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            await handler.handle_post("/api/v1/email/prioritize", {}, mock_http)

            call_args = instance.prioritize_email.call_args
            assert call_args[0][1] == "custom-user"

    @pytest.mark.asyncio
    async def test_prioritize_with_recipients_and_cc(self, handler):
        """Recipients and CC fields are passed through."""
        body = _single_email_body(
            recipients=["alice@example.com"],
            cc=["bob@example.com"],
        )
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_prioritize_with_attachments(self, handler):
        """Attachments list is passed through."""
        body = _single_email_body(attachments=["report.pdf", "data.csv"])
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_prioritize_invalid_date_uses_now(self, handler):
        """Invalid received_at falls back to current time (no error)."""
        body = _single_email_body(received_at="not-a-date")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_prioritize_no_received_at_uses_now(self, handler):
        """Missing received_at defaults to now."""
        body = {"subject": "Test", "body": "Test body"}
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_prioritize_service_connection_error(self, handler):
        """ConnectionError from service returns 500."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(
                side_effect=ConnectionError("API unreachable")
            )

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 500
        assert "failed" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_prioritize_service_timeout_error(self, handler):
        """TimeoutError from service returns 500."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(
                side_effect=TimeoutError("Request timed out")
            )

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_prioritize_service_value_error(self, handler):
        """ValueError from service returns 500."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(
                side_effect=ValueError("Invalid config")
            )

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_prioritize_service_runtime_error(self, handler):
        """RuntimeError from service returns 500."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(
                side_effect=RuntimeError("Unexpected error")
            )

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_prioritize_service_os_error(self, handler):
        """OSError from service returns 500."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(
                side_effect=OSError("Disk error")
            )

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_prioritize_result_fields(self, handler):
        """Response contains all expected fields from to_dict()."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)
        mock_result = MockEmailDebateResult(
            message_id="msg-xyz",
            priority=MockEmailPriority.URGENT,
            category=MockEmailCategory.ACTION_REQUIRED,
            confidence=0.95,
            reasoning="Very urgent meeting",
            action_items=["reply", "schedule"],
            suggested_labels=["urgent", "meeting"],
            is_spam=False,
            is_phishing=False,
            sender_reputation=0.9,
            debate_id="debate-xyz",
            duration_seconds=2.1,
        )

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        data = _body(result)
        assert data["message_id"] == "msg-xyz"
        assert data["priority"] == "urgent"
        assert data["category"] == "action_required"
        assert data["confidence"] == 0.95
        assert data["reasoning"] == "Very urgent meeting"
        assert data["action_items"] == ["reply", "schedule"]
        assert data["suggested_labels"] == ["urgent", "meeting"]
        assert data["is_spam"] is False
        assert data["is_phishing"] is False
        assert data["sender_reputation"] == 0.9
        assert data["debate_id"] == "debate-xyz"
        assert data["duration_seconds"] == 2.1


# ============================================================================
# POST /api/v1/email/prioritize/batch - Batch Prioritization
# ============================================================================


class TestPrioritizeBatch:
    """Tests for batch email prioritization."""

    @pytest.mark.asyncio
    async def test_batch_success(self, handler):
        """Happy path: batch of emails prioritized."""
        body = _batch_email_body(count=3)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=3, urgent=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

        assert _status(result) == 200
        data = _body(result)
        assert data["total_emails"] == 3
        assert data["processed_emails"] == 3
        assert len(data["results"]) == 3
        assert data["urgent_count"] == 1

    @pytest.mark.asyncio
    async def test_batch_missing_emails(self, handler):
        """Missing emails field returns 400."""
        body = {"user_id": "user-456"}
        mock_http = _make_mock_http_handler(body)

        result = await handler.handle_post(
            "/api/v1/email/prioritize/batch", {}, mock_http
        )

        assert _status(result) == 400
        assert "emails" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_batch_empty_emails_list(self, handler):
        """Empty emails list returns 400."""
        body = {"emails": []}
        mock_http = _make_mock_http_handler(body)

        result = await handler.handle_post(
            "/api/v1/email/prioritize/batch", {}, mock_http
        )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_batch_passes_max_concurrent(self, handler):
        """max_concurrent parameter is forwarded."""
        body = _batch_email_body(count=2, max_concurrent=10)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=2)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

            call_args = instance.prioritize_batch.call_args
            assert call_args[0][2] == 10  # max_concurrent

    @pytest.mark.asyncio
    async def test_batch_default_max_concurrent(self, handler):
        """max_concurrent defaults to 5."""
        body = _batch_email_body(count=1)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

            call_args = instance.prioritize_batch.call_args
            assert call_args[0][2] == 5

    @pytest.mark.asyncio
    async def test_batch_with_errors_in_result(self, handler):
        """Batch result includes errors list."""
        body = _batch_email_body(count=2)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=1)
        mock_result.errors = ["msg-1: timeout"]
        mock_result.total_emails = 2
        mock_result.processed_emails = 1

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

        data = _body(result)
        assert data["errors"] == ["msg-1: timeout"]
        assert data["processed_emails"] == 1
        assert data["total_emails"] == 2

    @pytest.mark.asyncio
    async def test_batch_service_connection_error(self, handler):
        """ConnectionError from service returns 500."""
        body = _batch_email_body(count=1)
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(
                side_effect=ConnectionError("API unreachable")
            )

            result = await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

        assert _status(result) == 500
        assert "batch" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_batch_service_timeout_error(self, handler):
        """TimeoutError from service returns 500."""
        body = _batch_email_body(count=1)
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(
                side_effect=TimeoutError("Timed out")
            )

            result = await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_batch_response_fields(self, handler):
        """Response contains all expected batch fields."""
        body = _batch_email_body(count=2)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=2, urgent=1, action_required=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

        data = _body(result)
        assert "results" in data
        assert "total_emails" in data
        assert "processed_emails" in data
        assert "duration_seconds" in data
        assert "urgent_count" in data
        assert "action_required_count" in data
        assert "errors" in data

    @pytest.mark.asyncio
    async def test_batch_email_with_invalid_date(self, handler):
        """Individual emails with invalid dates still succeed."""
        body = {
            "emails": [
                {
                    "subject": "Test",
                    "body": "Body",
                    "received_at": "invalid-date",
                }
            ],
            "user_id": "user-1",
        }
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_batch_passes_fast_mode(self, handler):
        """fast_mode parameter forwarded in batch mode."""
        body = _batch_email_body(count=1, fast_mode=False)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

            MockService.assert_called_once_with(
                fast_mode=False, enable_pii_redaction=True
            )


# ============================================================================
# POST /api/v1/email/triage - Inbox Triage
# ============================================================================


class TestTriageInbox:
    """Tests for full inbox triage."""

    @pytest.mark.asyncio
    async def test_triage_success_default_sort(self, handler):
        """Happy path: triage without grouping returns sorted results."""
        body = _batch_email_body(count=3)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=3, urgent=1, action_required=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        assert _status(result) == 200
        data = _body(result)
        assert "results" in data
        assert data["total_emails"] == 3
        assert data["urgent_count"] == 1

    @pytest.mark.asyncio
    async def test_triage_missing_emails(self, handler):
        """Missing emails field returns 400."""
        body = {"user_id": "user-1"}
        mock_http = _make_mock_http_handler(body)

        result = await handler.handle_post("/api/v1/email/triage", {}, mock_http)

        assert _status(result) == 400
        assert "emails" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_triage_empty_emails(self, handler):
        """Empty emails list returns 400."""
        body = {"emails": []}
        mock_http = _make_mock_http_handler(body)

        result = await handler.handle_post("/api/v1/email/triage", {}, mock_http)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_triage_group_by_category(self, handler):
        """group_by=category returns grouped response."""
        body = _batch_email_body(count=2, group_by="category")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=2, action_required=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        assert _status(result) == 200
        data = _body(result)
        assert "grouped" in data
        assert "total_emails" in data
        # Results are grouped by category value
        assert isinstance(data["grouped"], dict)

    @pytest.mark.asyncio
    async def test_triage_group_by_priority(self, handler):
        """group_by=priority returns grouped by priority response."""
        body = _batch_email_body(count=3, group_by="priority")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=3, urgent=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        assert _status(result) == 200
        data = _body(result)
        assert "grouped" in data
        # Should have priority groups
        assert isinstance(data["grouped"], dict)

    @pytest.mark.asyncio
    async def test_triage_no_group_by(self, handler):
        """No group_by returns flat results list."""
        body = _batch_email_body(count=2)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=2)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        data = _body(result)
        assert "results" in data
        assert "grouped" not in data

    @pytest.mark.asyncio
    async def test_triage_sort_by_priority(self, handler):
        """Results are sorted by priority order (urgent first)."""
        body = _batch_email_body(count=3, sort_by="priority")
        mock_http = _make_mock_http_handler(body)

        # Create results with mixed priorities
        results = [
            MockEmailDebateResult(
                message_id="low",
                priority=MockEmailPriority.LOW,
                confidence=0.9,
            ),
            MockEmailDebateResult(
                message_id="urgent",
                priority=MockEmailPriority.URGENT,
                confidence=0.8,
            ),
            MockEmailDebateResult(
                message_id="normal",
                priority=MockEmailPriority.NORMAL,
                confidence=0.7,
            ),
        ]
        mock_result = MockBatchEmailResult(
            results=results,
            total_emails=3,
            processed_emails=3,
            duration_seconds=1.5,
        )

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        data = _body(result)
        result_ids = [r["message_id"] for r in data["results"]]
        # Urgent (0) should come before normal (2) which should come before low (3)
        assert result_ids.index("urgent") < result_ids.index("normal")
        assert result_ids.index("normal") < result_ids.index("low")

    @pytest.mark.asyncio
    async def test_triage_service_error(self, handler):
        """Service error returns 500."""
        body = _batch_email_body(count=1)
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(
                side_effect=RuntimeError("Service down")
            )

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        assert _status(result) == 500
        assert "triage" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_triage_unknown_group_by_returns_flat(self, handler):
        """Unknown group_by value falls through to flat results."""
        body = _batch_email_body(count=2, group_by="unknown")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=2)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        data = _body(result)
        assert "results" in data
        assert "grouped" not in data

    @pytest.mark.asyncio
    async def test_triage_response_fields_no_group(self, handler):
        """Flat triage response contains all expected fields."""
        body = _batch_email_body(count=2)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=2, urgent=1, action_required=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        data = _body(result)
        assert "results" in data
        assert "total_emails" in data
        assert "urgent_count" in data
        assert "action_required_count" in data
        assert "duration_seconds" in data

    @pytest.mark.asyncio
    async def test_triage_grouped_response_fields(self, handler):
        """Grouped triage response contains all expected fields."""
        body = _batch_email_body(count=2, group_by="category")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=2, urgent=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        data = _body(result)
        assert "grouped" in data
        assert "total_emails" in data
        assert "urgent_count" in data
        assert "action_required_count" in data
        assert "duration_seconds" in data


# ============================================================================
# Rate Limiting Tests
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, handler):
        """Exceeding rate limit returns 429."""
        body = _single_email_body()

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            mock_http = _make_mock_http_handler(body)
            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 429
        assert "rate limit" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_rate_limit_not_exceeded(self, handler):
        """Requests within rate limit proceed normally."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter, patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            mock_limiter.is_allowed.return_value = True
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rate_limit_applies_to_batch(self, handler):
        """Rate limit also applies to batch endpoint."""
        body = _batch_email_body(count=1)

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            mock_http = _make_mock_http_handler(body)
            result = await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

        assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_rate_limit_applies_to_triage(self, handler):
        """Rate limit also applies to triage endpoint."""
        body = _batch_email_body(count=1)

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False

            mock_http = _make_mock_http_handler(body)
            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        assert _status(result) == 429

    @pytest.mark.asyncio
    async def test_rate_limit_uses_client_ip(self, handler):
        """Rate limiter is called with client IP."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)
        mock_http.client_address = ("10.0.0.1", 8080)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter, patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            mock_limiter.is_allowed.return_value = True
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            await handler.handle_post("/api/v1/email/prioritize", {}, mock_http)

            mock_limiter.is_allowed.assert_called_once()


# ============================================================================
# Unknown Route Tests
# ============================================================================


class TestUnknownRoute:
    """Tests for unknown route handling."""

    @pytest.mark.asyncio
    async def test_unknown_route_returns_none(self, handler):
        """Unknown path returns None from handle_post."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = await handler.handle_post(
                "/api/v1/email/unknown", {}, mock_http
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_empty_path_returns_none(self, handler):
        """Empty path returns None."""
        mock_http = _make_mock_http_handler(_single_email_body())

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = await handler.handle_post("", {}, mock_http)

        assert result is None


# ============================================================================
# Input Validation / Edge Case Tests
# ============================================================================


class TestInputValidation:
    """Tests for input validation edge cases."""

    @pytest.mark.asyncio
    async def test_empty_json_body_prioritize(self, handler):
        """Empty JSON body for prioritize returns 400 (no subject/body)."""
        mock_http = _make_mock_http_handler({})

        result = await handler.handle_post(
            "/api/v1/email/prioritize", {}, mock_http
        )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_invalid_json_body(self, handler):
        """Invalid JSON body returns 400."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        mock_http.headers = {
            "Content-Length": "11",
            "Content-Type": "application/json",
        }
        mock_http.rfile = MagicMock()
        mock_http.rfile.read.return_value = b"not-json!!!"

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_iso_date_with_z_suffix(self, handler):
        """ISO date with Z suffix is parsed correctly."""
        body = _single_email_body(received_at="2024-06-15T14:30:00Z")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_iso_date_with_offset(self, handler):
        """ISO date with timezone offset is parsed correctly."""
        body = _single_email_body(received_at="2024-06-15T14:30:00+05:00")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_batch_email_minimal_fields(self, handler):
        """Batch emails with minimal fields still succeed."""
        body = {
            "emails": [
                {"subject": "Minimal email"},
            ]
        }
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_batch_result(count=1)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_batch_emails_null_value(self, handler):
        """emails=null returns 400."""
        body = {"emails": None}
        mock_http = _make_mock_http_handler(body)

        result = await handler.handle_post(
            "/api/v1/email/prioritize/batch", {}, mock_http
        )

        assert _status(result) == 400


# ============================================================================
# Security Tests
# ============================================================================


class TestSecurity:
    """Tests for security concerns."""

    @pytest.mark.asyncio
    async def test_subject_with_script_injection(self, handler):
        """Subject containing script tags does not cause issues."""
        body = _single_email_body(subject="<script>alert('xss')</script>")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_body_with_sql_injection(self, handler):
        """Body containing SQL injection does not cause issues."""
        body = _single_email_body(body="'; DROP TABLE emails; --")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_sender_with_path_traversal(self, handler):
        """Sender field with path traversal does not cause issues."""
        body = _single_email_body(sender="../../../etc/passwd")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_very_long_subject(self, handler):
        """Very long subject does not crash the handler."""
        body = _single_email_body(subject="A" * 10000)
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_unicode_in_subject(self, handler):
        """Unicode characters in subject are handled correctly."""
        body = _single_email_body(subject="Reunion demain a 15h")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_null_bytes_in_body(self, handler):
        """Null bytes in email body are handled."""
        body = _single_email_body(body="Normal text\x00hidden")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_message_id_with_special_chars(self, handler):
        """Message IDs with special characters are passed through."""
        body = _single_email_body(message_id="<msg-123@example.com>")
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 200


# ============================================================================
# Content-Type Validation Tests
# ============================================================================


class TestContentTypeValidation:
    """Tests for Content-Type header validation."""

    @pytest.mark.asyncio
    async def test_missing_content_type_with_body(self, handler):
        """Missing Content-Type header with body returns 415."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        body_bytes = json.dumps(_single_email_body()).encode()
        mock_http.headers = {"Content-Length": str(len(body_bytes))}
        mock_http.rfile = MagicMock()
        mock_http.rfile.read.return_value = body_bytes

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 415

    @pytest.mark.asyncio
    async def test_wrong_content_type(self, handler):
        """Wrong Content-Type returns 415."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 12345)
        body_bytes = json.dumps(_single_email_body()).encode()
        mock_http.headers = {
            "Content-Length": str(len(body_bytes)),
            "Content-Type": "text/plain",
        }
        mock_http.rfile = MagicMock()
        mock_http.rfile.read.return_value = body_bytes

        with patch(
            "aragora.server.handlers.email_debate._email_debate_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = True

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        assert _status(result) == 415


# ============================================================================
# EmailInput Construction Tests
# ============================================================================


class TestEmailInputConstruction:
    """Tests that EmailInput objects are properly constructed from body data."""

    @pytest.mark.asyncio
    async def test_email_input_all_fields(self, handler):
        """All email fields are passed to EmailInput."""
        body = _single_email_body(
            recipients=["r1@example.com", "r2@example.com"],
            cc=["cc1@example.com"],
            attachments=["file.pdf"],
        )
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService, patch(
            "aragora.server.handlers.email_debate.EmailInput"
        ) as MockEmailInput:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            await handler.handle_post("/api/v1/email/prioritize", {}, mock_http)

            MockEmailInput.assert_called_once()
            call_kwargs = MockEmailInput.call_args[1]
            assert call_kwargs["subject"] == "Meeting tomorrow"
            assert call_kwargs["body"] == "Hi, can we meet tomorrow at 3pm?"
            assert call_kwargs["sender"] == "john@example.com"
            assert call_kwargs["message_id"] == "msg-123"
            assert call_kwargs["recipients"] == ["r1@example.com", "r2@example.com"]
            assert call_kwargs["cc"] == ["cc1@example.com"]
            assert call_kwargs["attachments"] == ["file.pdf"]

    @pytest.mark.asyncio
    async def test_email_input_defaults(self, handler):
        """Missing optional fields use empty defaults."""
        body = {"subject": "Test"}
        mock_http = _make_mock_http_handler(body)
        mock_result = _make_single_result()

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService, patch(
            "aragora.server.handlers.email_debate.EmailInput"
        ) as MockEmailInput:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(return_value=mock_result)

            await handler.handle_post("/api/v1/email/prioritize", {}, mock_http)

            call_kwargs = MockEmailInput.call_args[1]
            assert call_kwargs["sender"] == ""
            assert call_kwargs["recipients"] == []
            assert call_kwargs["cc"] == []
            assert call_kwargs["attachments"] == []
            assert call_kwargs["message_id"] is None


# ============================================================================
# Error Message Tests
# ============================================================================


class TestErrorMessages:
    """Tests that error messages are appropriate and not leaking internals."""

    @pytest.mark.asyncio
    async def test_prioritize_error_does_not_leak_details(self, handler):
        """Internal error details are not leaked in 500 response."""
        body = _single_email_body()
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_email = AsyncMock(
                side_effect=ConnectionError("Secret internal DB host at 10.0.0.5:5432")
            )

            result = await handler.handle_post(
                "/api/v1/email/prioritize", {}, mock_http
            )

        error_msg = _body(result).get("error", "")
        assert "10.0.0.5" not in error_msg
        assert "5432" not in error_msg
        assert "Prioritization failed" == error_msg

    @pytest.mark.asyncio
    async def test_batch_error_does_not_leak_details(self, handler):
        """Internal error details are not leaked in batch 500 response."""
        body = _batch_email_body(count=1)
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(
                side_effect=ValueError("API key abc123xyz")
            )

            result = await handler.handle_post(
                "/api/v1/email/prioritize/batch", {}, mock_http
            )

        error_msg = _body(result).get("error", "")
        assert "abc123xyz" not in error_msg
        assert "Batch prioritization failed" == error_msg

    @pytest.mark.asyncio
    async def test_triage_error_does_not_leak_details(self, handler):
        """Internal error details are not leaked in triage 500 response."""
        body = _batch_email_body(count=1)
        mock_http = _make_mock_http_handler(body)

        with patch(
            "aragora.server.handlers.email_debate.EmailDebateService"
        ) as MockService:
            instance = MockService.return_value
            instance.prioritize_batch = AsyncMock(
                side_effect=KeyError("sensitive_table_name")
            )

            result = await handler.handle_post(
                "/api/v1/email/triage", {}, mock_http
            )

        error_msg = _body(result).get("error", "")
        assert "sensitive_table_name" not in error_msg
        assert "Inbox triage failed" == error_msg


# ============================================================================
# Handler with Context Tests
# ============================================================================


class TestHandlerWithContext:
    """Tests for handler with custom context."""

    def test_handler_with_context(self, handler_with_ctx):
        """Handler with context stores it."""
        assert "user_store" in handler_with_ctx.ctx

    def test_handler_can_handle_with_context(self, handler_with_ctx):
        """Routing works with context."""
        assert handler_with_ctx.can_handle("/api/v1/email/prioritize")

    def test_handler_get_with_context(self, handler_with_ctx):
        """GET returns 405 even with context."""
        mock = _make_mock_http_handler()
        result = handler_with_ctx.handle("/api/v1/email/prioritize", {}, mock)
        assert _status(result) == 405


# ============================================================================
# Module-level Exports Tests
# ============================================================================


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports(self):
        """__all__ contains EmailDebateHandler."""
        from aragora.server.handlers import email_debate

        assert "EmailDebateHandler" in email_debate.__all__

    def test_handler_is_importable(self):
        """Handler class is importable."""
        from aragora.server.handlers.email_debate import EmailDebateHandler as H

        assert H is not None
        assert callable(H)
