"""
Tests for aragora.server.handlers.email_debate - Email Debate HTTP Handler.

Tests cover:
- EmailDebateHandler: instantiation, ROUTES, can_handle
- handle (GET): returns 405 for all paths
- handle_post routing: prioritize, batch, triage, unmatched
- _prioritize_single: success, missing body fields, service error
- _prioritize_batch: success, missing emails, service error
- _triage_inbox: success, grouping by category, grouping by priority, missing emails
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.email_debate import EmailDebateHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "POST",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Mock Email Objects
# ===========================================================================


class MockEmailDebateResult:
    """Mock EmailDebateResult."""

    def __init__(self, message_id: str = "msg-001", priority_val: str = "normal", category_val: str = "fyi"):
        self.message_id = message_id
        self.priority = MagicMock(value=priority_val)
        self.category = MagicMock(value=category_val)
        self.confidence = 0.85

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "priority": self.priority.value,
            "category": self.category.value,
            "confidence": self.confidence,
        }


class MockBatchResult:
    """Mock BatchEmailResult."""

    def __init__(self, results: list | None = None):
        self.results = results or [MockEmailDebateResult()]
        self.total_emails = len(self.results)
        self.processed_emails = len(self.results)
        self.duration_seconds = 1.5
        self.urgent_count = 0
        self.action_required_count = 0
        self.errors: list[str] = []

    @property
    def by_priority(self) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for r in self.results:
            key = r.priority.value
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
        return grouped


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler():
    """Create an EmailDebateHandler."""
    h = EmailDebateHandler(ctx={})
    return h


# ===========================================================================
# Test Instantiation and Basics
# ===========================================================================


class TestEmailDebateHandlerBasics:
    """Basic instantiation and attribute tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, EmailDebateHandler)

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) == 3

    def test_can_handle_prioritize(self, handler):
        assert handler.can_handle("/api/v1/email/prioritize") is True

    def test_can_handle_batch(self, handler):
        assert handler.can_handle("/api/v1/email/prioritize/batch") is True

    def test_can_handle_triage(self, handler):
        assert handler.can_handle("/api/v1/email/triage") is True

    def test_cannot_handle_other(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_cannot_handle_partial_match(self, handler):
        assert handler.can_handle("/api/v1/email") is False


# ===========================================================================
# Test GET (405)
# ===========================================================================


class TestHandleGet:
    """Tests for the GET handler (should return 405)."""

    def test_get_returns_405(self, handler):
        mock_handler = _make_mock_handler("GET")
        # The handle method has @require_permission which needs to be bypassed
        with patch(
            "aragora.server.handlers.email_debate.require_permission",
            lambda perm: lambda fn: fn,
        ):
            # Re-create handler to pick up undecorated version
            h = EmailDebateHandler.__new__(EmailDebateHandler)
            h.ctx = {}
            result = EmailDebateHandler.handle.__wrapped__(h, "/api/v1/email/prioritize", {}, mock_handler)
            assert result.status_code == 405


# ===========================================================================
# Test POST /api/v1/email/prioritize
# ===========================================================================


class TestPrioritizeSingle:
    """Tests for the single email prioritize endpoint."""

    @pytest.mark.asyncio
    async def test_prioritize_success(self, handler):
        mock_handler = _make_mock_handler()
        mock_result = MockEmailDebateResult()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"subject": "Test", "body": "Hello", "sender": "a@b.com"}, None
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_email = AsyncMock(return_value=mock_result)

                result = await handler._prioritize_single(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["message_id"] == "msg-001"

    @pytest.mark.asyncio
    async def test_prioritize_missing_subject_and_body(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"sender": "a@b.com"}, None
        )):
            result = await handler._prioritize_single(mock_handler)
            assert result.status_code == 400
            data = _parse_body(result)
            assert "subject or body" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_prioritize_invalid_body(self, handler):
        mock_handler = _make_mock_handler()
        err_result = HandlerResult(
            status_code=400,
            content_type="application/json",
            body=b'{"error":"Invalid JSON"}',
        )

        with patch.object(handler, "read_json_body_validated", return_value=(None, err_result)):
            result = await handler._prioritize_single(mock_handler)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_prioritize_service_error(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"subject": "Test", "body": "Hello"}, None
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_email = AsyncMock(
                    side_effect=RuntimeError("LLM down")
                )

                result = await handler._prioritize_single(mock_handler)
                assert result.status_code == 500
                data = _parse_body(result)
                assert "failed" in data.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_prioritize_with_received_at(self, handler):
        """Ensure received_at ISO string is parsed."""
        mock_handler = _make_mock_handler()
        mock_result = MockEmailDebateResult()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {
                "subject": "Test",
                "body": "Hello",
                "received_at": "2024-01-15T10:30:00Z",
            },
            None,
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_email = AsyncMock(return_value=mock_result)

                result = await handler._prioritize_single(mock_handler)
                assert result.status_code == 200


# ===========================================================================
# Test POST /api/v1/email/prioritize/batch
# ===========================================================================


class TestPrioritizeBatch:
    """Tests for batch email prioritization."""

    @pytest.mark.asyncio
    async def test_batch_success(self, handler):
        mock_handler = _make_mock_handler()
        mock_batch = MockBatchResult()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {
                "emails": [
                    {"subject": "Test 1", "body": "Hello", "sender": "a@b.com"},
                    {"subject": "Test 2", "body": "Hi", "sender": "c@d.com"},
                ],
                "user_id": "user-1",
            },
            None,
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_batch = AsyncMock(return_value=mock_batch)

                result = await handler._prioritize_batch(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert "results" in data
                assert "total_emails" in data

    @pytest.mark.asyncio
    async def test_batch_missing_emails(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"user_id": "user-1"}, None
        )):
            result = await handler._prioritize_batch(mock_handler)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_empty_emails(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"emails": []}, None
        )):
            result = await handler._prioritize_batch(mock_handler)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_batch_service_error(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"emails": [{"subject": "Test", "body": "Hi"}]}, None
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_batch = AsyncMock(
                    side_effect=RuntimeError("Batch failed")
                )

                result = await handler._prioritize_batch(mock_handler)
                assert result.status_code == 500


# ===========================================================================
# Test POST /api/v1/email/triage
# ===========================================================================


class TestTriageInbox:
    """Tests for full inbox triage."""

    @pytest.mark.asyncio
    async def test_triage_success(self, handler):
        mock_handler = _make_mock_handler()
        mock_batch = MockBatchResult()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"emails": [{"subject": "Test", "body": "Hi"}]}, None
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_batch = AsyncMock(return_value=mock_batch)

                result = await handler._triage_inbox(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert "results" in data
                assert "total_emails" in data

    @pytest.mark.asyncio
    async def test_triage_missing_emails(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"sort_by": "priority"}, None
        )):
            result = await handler._triage_inbox(mock_handler)
            assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_triage_group_by_category(self, handler):
        mock_handler = _make_mock_handler()
        results = [
            MockEmailDebateResult("msg-1", "urgent", "action_required"),
            MockEmailDebateResult("msg-2", "normal", "fyi"),
        ]
        mock_batch = MockBatchResult(results=results)

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"emails": [{"subject": "A"}, {"subject": "B"}], "group_by": "category"}, None
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_batch = AsyncMock(return_value=mock_batch)

                result = await handler._triage_inbox(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert "grouped" in data

    @pytest.mark.asyncio
    async def test_triage_group_by_priority(self, handler):
        mock_handler = _make_mock_handler()
        results = [
            MockEmailDebateResult("msg-1", "urgent", "action_required"),
            MockEmailDebateResult("msg-2", "normal", "fyi"),
        ]
        mock_batch = MockBatchResult(results=results)

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"emails": [{"subject": "A"}, {"subject": "B"}], "group_by": "priority"}, None
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_batch = AsyncMock(return_value=mock_batch)

                result = await handler._triage_inbox(mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert "grouped" in data

    @pytest.mark.asyncio
    async def test_triage_service_error(self, handler):
        mock_handler = _make_mock_handler()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"emails": [{"subject": "Test"}]}, None
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_batch = AsyncMock(
                    side_effect=RuntimeError("Triage failed")
                )

                result = await handler._triage_inbox(mock_handler)
                assert result.status_code == 500


# ===========================================================================
# Test handle_post Routing
# ===========================================================================


class TestHandlePostRouting:
    """Tests for the top-level handle_post() routing."""

    @pytest.mark.asyncio
    async def test_route_to_prioritize(self, handler):
        mock_handler = _make_mock_handler()
        mock_result = MockEmailDebateResult()

        with patch.object(handler, "read_json_body_validated", return_value=(
            {"subject": "Test", "body": "Hello"}, None
        )):
            with patch(
                "aragora.server.handlers.email_debate.EmailDebateService"
            ) as MockService:
                service_instance = MockService.return_value
                service_instance.prioritize_email = AsyncMock(return_value=mock_result)

                # Call the underlying method directly, bypassing the decorator
                result = await handler.handle_post.__wrapped__(
                    handler, "/api/v1/email/prioritize", {}, mock_handler
                )
                assert result is not None
                assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_route_unmatched_returns_none(self, handler):
        mock_handler = _make_mock_handler()
        result = await handler.handle_post.__wrapped__(
            handler, "/api/v1/email/unknown", {}, mock_handler
        )
        assert result is None
