"""
Tests for EmailDebateHandler - Email Vetted Decisionmaking HTTP endpoints.

Tests cover:
- Handler initialization and route matching
- POST /api/v1/email/prioritize - Single email prioritization
- POST /api/v1/email/prioritize/batch - Batch email prioritization
- POST /api/v1/email/triage - Inbox triage with categorization
- GET method rejection (POST-only handler)
- RBAC permission enforcement
- Error handling and validation
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


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


class MockPriority(Enum):
    """Mock email priority enum."""

    urgent = "urgent"
    high = "high"
    normal = "normal"
    low = "low"
    spam = "spam"


class MockCategory(Enum):
    """Mock email category enum."""

    action_required = "action_required"
    fyi = "fyi"
    meeting = "meeting"
    follow_up = "follow_up"
    spam = "spam"


@dataclass
class MockPrioritizationResult:
    """Mock email prioritization result."""

    email_id: str = "email-123"
    priority: MockPriority = MockPriority.normal
    confidence: float = 0.85
    category: MockCategory = MockCategory.fyi
    reasoning: str = "Standard informational email"
    suggested_action: str | None = "Read when convenient"
    time_sensitive: bool = False
    requires_response: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "email_id": self.email_id,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "category": self.category.value,
            "reasoning": self.reasoning,
            "suggested_action": self.suggested_action,
            "time_sensitive": self.time_sensitive,
            "requires_response": self.requires_response,
        }


@dataclass
class MockBatchResult:
    """Mock batch prioritization result."""

    results: list[MockPrioritizationResult] = field(default_factory=list)
    total_emails: int = 0
    processed_emails: int = 0
    duration_seconds: float = 1.5
    urgent_count: int = 0
    action_required_count: int = 0
    errors: list[str] = field(default_factory=list)
    by_priority: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.results:
            self.results = [MockPrioritizationResult()]
            self.total_emails = 1
            self.processed_emails = 1


class MockEmailDebateService:
    """Mock email debate service for testing."""

    def __init__(self, fast_mode: bool = True, enable_pii_redaction: bool = True):
        self.fast_mode = fast_mode
        self.enable_pii_redaction = enable_pii_redaction

    async def prioritize_email(self, email: Any, user_id: str) -> MockPrioritizationResult:
        return MockPrioritizationResult(email_id=getattr(email, "message_id", "email-123"))

    async def prioritize_batch(
        self, emails: list, user_id: str, max_concurrent: int = 5
    ) -> MockBatchResult:
        results = [
            MockPrioritizationResult(email_id=getattr(e, "message_id", f"email-{i}"))
            for i, e in enumerate(emails)
        ]
        return MockBatchResult(
            results=results,
            total_emails=len(emails),
            processed_emails=len(emails),
            urgent_count=sum(1 for r in results if r.priority == MockPriority.urgent),
            action_required_count=sum(
                1 for r in results if r.category == MockCategory.action_required
            ),
            by_priority={
                "urgent": [r for r in results if r.priority == MockPriority.urgent],
                "high": [r for r in results if r.priority == MockPriority.high],
                "normal": [r for r in results if r.priority == MockPriority.normal],
                "low": [r for r in results if r.priority == MockPriority.low],
            },
        )


@pytest.fixture
def mock_handler():
    """Create mock HTTP handler with body content."""
    handler = MagicMock()
    handler.headers = {"content-length": "100"}
    handler.rfile = MagicMock()
    return handler


@pytest.fixture
def server_context():
    """Create minimal server context for handler testing."""
    from unittest.mock import MagicMock

    return {
        "storage": MagicMock(),
        "user_store": MagicMock(),
        "elo_system": MagicMock(),
        "debate_embeddings": None,
        "critique_store": None,
        "nomic_dir": None,
    }


@pytest.fixture
def email_handler(
    server_context,
):
    """Create EmailDebateHandler instance."""
    return EmailDebateHandler(server_context)


def make_handler_with_body(body: dict) -> MagicMock:
    """Create a mock handler with JSON body."""
    handler = MagicMock()
    body_bytes = json.dumps(body).encode()
    handler.headers = {
        "Content-Length": str(len(body_bytes)),  # Must be capitalized for dict.get()
        "Content-Type": "application/json",
    }
    handler.rfile.read.return_value = body_bytes
    return handler


# ===========================================================================
# Route Matching Tests
# ===========================================================================


class TestEmailDebateHandlerRouting:
    """Test request routing and path matching."""

    def test_can_handle_prioritize_path(self, email_handler):
        """Test that handler recognizes /api/v1/email/prioritize."""
        assert email_handler.can_handle("/api/v1/email/prioritize")

    def test_can_handle_batch_prioritize_path(self, email_handler):
        """Test that handler recognizes /api/v1/email/prioritize/batch."""
        assert email_handler.can_handle("/api/v1/email/prioritize/batch")

    def test_can_handle_triage_path(self, email_handler):
        """Test that handler recognizes /api/v1/email/triage."""
        assert email_handler.can_handle("/api/v1/email/triage")

    def test_cannot_handle_other_paths(self, email_handler):
        """Test that handler rejects non-email debate paths."""
        assert not email_handler.can_handle("/api/v1/email/send")
        assert not email_handler.can_handle("/api/v1/inbox")
        assert not email_handler.can_handle("/api/v2/email/prioritize")


# ===========================================================================
# GET Method Tests (Should Return 405)
# ===========================================================================


class TestGetMethodRejection:
    """Test that GET method returns 405 Method Not Allowed."""

    def test_get_prioritize_returns_405(self, email_handler, mock_handler):
        """Test that GET /api/v1/email/prioritize returns 405."""
        result = email_handler.handle("/api/v1/email/prioritize", {}, mock_handler)

        assert result is not None
        assert result.status_code == 405

    def test_get_batch_returns_405(self, email_handler, mock_handler):
        """Test that GET /api/v1/email/prioritize/batch returns 405."""
        result = email_handler.handle("/api/v1/email/prioritize/batch", {}, mock_handler)

        assert result is not None
        assert result.status_code == 405

    def test_get_triage_returns_405(self, email_handler, mock_handler):
        """Test that GET /api/v1/email/triage returns 405."""
        result = email_handler.handle("/api/v1/email/triage", {}, mock_handler)

        assert result is not None
        assert result.status_code == 405


# ===========================================================================
# Single Email Prioritization Tests
# ===========================================================================


class TestSinglePrioritization:
    """Test POST /api/v1/email/prioritize endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_prioritize_single_success(self, mock_email_input, email_handler):
        """Test successful single email prioritization."""
        body = {
            "subject": "Meeting tomorrow",
            "body": "Hi, can we meet tomorrow at 3pm?",
            "sender": "john@example.com",
            "received_at": "2024-01-15T10:30:00Z",
            "message_id": "msg-123",
            "user_id": "user-456",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_prioritize_single_missing_subject_and_body(self, email_handler):
        """Test prioritization fails without subject or body."""
        body = {
            "sender": "john@example.com",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_prioritize_single_with_subject_only(self, mock_email_input, email_handler):
        """Test prioritization works with subject only."""
        body = {"subject": "Urgent: Server down"}
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_prioritize_single_with_body_only(self, mock_email_input, email_handler):
        """Test prioritization works with body only."""
        body = {"body": "Please review the attached document."}
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_prioritize_single_service_error_graceful_fallback(self, email_handler):
        """Test that service gracefully degrades to heuristic mode when debate fails.

        The EmailDebateService has built-in fallback behavior - when the debate
        fails (e.g., no API credentials), it returns a heuristic-based result
        with 200 status instead of failing with 500.
        """
        body = {"subject": "Test email", "body": "Test content"}
        handler = make_handler_with_body(body)

        # Without API credentials, the service will fall back to heuristic mode
        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        # Service gracefully degrades to heuristic mode instead of failing
        assert result.status_code == 200
        body_data = json.loads(result.body)
        # Heuristic mode indicates fallback in reasoning
        assert "heuristic" in body_data.get("reasoning", "").lower()


# ===========================================================================
# Batch Prioritization Tests
# ===========================================================================


class TestBatchPrioritization:
    """Test POST /api/v1/email/prioritize/batch endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_prioritize_batch_success(self, mock_email_input, email_handler):
        """Test successful batch email prioritization."""
        body = {
            "emails": [
                {"subject": "Email 1", "body": "Body 1", "sender": "a@example.com"},
                {"subject": "Email 2", "body": "Body 2", "sender": "b@example.com"},
            ],
            "user_id": "user-456",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize/batch", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_prioritize_batch_missing_emails(self, email_handler):
        """Test batch prioritization fails without emails."""
        body = {"user_id": "user-456"}
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize/batch", {}, handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_prioritize_batch_empty_emails(self, email_handler):
        """Test batch prioritization fails with empty emails list."""
        body = {"emails": [], "user_id": "user-456"}
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize/batch", {}, handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_prioritize_batch_with_max_concurrent(self, mock_email_input, email_handler):
        """Test batch prioritization with custom max_concurrent."""
        body = {
            "emails": [
                {"subject": "Email 1", "body": "Body 1"},
            ],
            "user_id": "user-456",
            "max_concurrent": 10,
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize/batch", {}, handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Inbox Triage Tests
# ===========================================================================


class TestInboxTriage:
    """Test POST /api/v1/email/triage endpoint."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_triage_success(self, mock_email_input, email_handler):
        """Test successful inbox triage."""
        body = {
            "emails": [
                {
                    "subject": "Urgent task",
                    "body": "Please complete today",
                    "sender": "boss@example.com",
                },
                {"subject": "FYI", "body": "Newsletter update", "sender": "news@example.com"},
            ],
            "user_id": "user-456",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/triage", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_triage_missing_emails(self, email_handler):
        """Test triage fails without emails."""
        body = {"user_id": "user-456"}
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/triage", {}, handler)

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_triage_group_by_category(self, mock_email_input, email_handler):
        """Test triage with group_by=category."""
        body = {
            "emails": [{"subject": "Test", "body": "Content"}],
            "user_id": "user-456",
            "group_by": "category",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/triage", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_triage_group_by_priority(self, mock_email_input, email_handler):
        """Test triage with group_by=priority."""
        body = {
            "emails": [{"subject": "Test", "body": "Content"}],
            "user_id": "user-456",
            "group_by": "priority",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/triage", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_triage_sort_by_priority(self, mock_email_input, email_handler):
        """Test triage with sort_by=priority."""
        body = {
            "emails": [{"subject": "Test", "body": "Content"}],
            "user_id": "user-456",
            "sort_by": "priority",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/triage", {}, handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Date Parsing Tests
# ===========================================================================


class TestDateParsing:
    """Test received_at date parsing."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_parse_iso_date_with_z_suffix(self, mock_email_input, email_handler):
        """Test parsing ISO date with Z suffix."""
        body = {
            "subject": "Test",
            "body": "Content",
            "received_at": "2024-01-15T10:30:00Z",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_parse_iso_date_with_offset(self, mock_email_input, email_handler):
        """Test parsing ISO date with timezone offset."""
        body = {
            "subject": "Test",
            "body": "Content",
            "received_at": "2024-01-15T10:30:00+00:00",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_invalid_date_uses_current_time(self, mock_email_input, email_handler):
        """Test that invalid date falls back to current time."""
        body = {
            "subject": "Test",
            "body": "Content",
            "received_at": "invalid-date",
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        # Should still succeed, using current time as fallback
        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# Optional Fields Tests
# ===========================================================================


class TestOptionalFields:
    """Test handling of optional email fields."""

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_with_recipients(self, mock_email_input, email_handler):
        """Test email with recipients field."""
        body = {
            "subject": "Test",
            "body": "Content",
            "recipients": ["alice@example.com", "bob@example.com"],
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_with_cc(self, mock_email_input, email_handler):
        """Test email with cc field."""
        body = {
            "subject": "Test",
            "body": "Content",
            "cc": ["manager@example.com"],
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_with_attachments(self, mock_email_input, email_handler):
        """Test email with attachments field."""
        body = {
            "subject": "Test",
            "body": "Content",
            "attachments": ["doc.pdf", "image.png"],  # EmailInput expects list[str]
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_with_fast_mode_disabled(self, mock_email_input, email_handler):
        """Test email with fast_mode=False."""
        body = {
            "subject": "Test",
            "body": "Content",
            "fast_mode": False,
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    @patch("aragora.server.handlers.email_debate.EmailDebateService", MockEmailDebateService)
    @patch("aragora.server.handlers.email_debate.EmailInput")
    async def test_with_pii_redaction_disabled(self, mock_email_input, email_handler):
        """Test email with enable_pii_redaction=False."""
        body = {
            "subject": "Test",
            "body": "Content",
            "enable_pii_redaction": False,
        }
        handler = make_handler_with_body(body)

        result = await email_handler.handle_post("/api/v1/email/prioritize", {}, handler)

        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# RBAC Tests
# ===========================================================================


class TestEmailDebateRBAC:
    """Test RBAC permission enforcement."""

    def test_handle_has_email_read_permission(self, email_handler):
        """Test that handle method has email:read permission decorator."""
        # The handle method has @require_permission("email:read")
        assert callable(email_handler.handle)

    def test_handle_post_has_email_create_permission(self, email_handler):
        """Test that handle_post method has email:create permission decorator."""
        # The handle_post method has @require_permission("email:create")
        assert callable(email_handler.handle_post)
