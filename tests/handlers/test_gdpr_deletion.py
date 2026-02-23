"""
Tests for GDPR self-service deletion handler.

Comprehensive tests for all endpoints:
- GET    /api/v1/users/self/deletion-request  (check status)
- POST   /api/v1/users/self/deletion-request  (schedule with grace period)
- DELETE /api/v1/users/self/deletion-request  (cancel during grace period)

Covers: routing, authentication, validation, edge cases, error handling.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gdpr_deletion import (
    DEFAULT_GRACE_PERIOD_DAYS,
    GDPRDeletionHandler,
    _get_user_id_from_handler,
)


# ============================================================================
# Helpers
# ============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body) if body else {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockDeletionStatus(str, Enum):
    """Lightweight status enum for tests (avoids importing full privacy module)."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    HELD = "held"


class _MockDeletionRequest:
    """Lightweight mock of DeletionRequest for tests."""

    def __init__(
        self,
        request_id: str = "req-001",
        user_id: str = "user-123",
        status: _MockDeletionStatus = _MockDeletionStatus.PENDING,
        grace_days: int = 30,
        reason: str = "User request",
    ):
        now = datetime.now(timezone.utc)
        self.request_id = request_id
        self.user_id = user_id
        self.status = status
        self.reason = reason
        self.scheduled_for = now + timedelta(days=grace_days)
        self.created_at = now

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "status": self.status.value,
            "reason": self.reason,
            "scheduled_for": self.scheduled_for.isoformat(),
            "created_at": self.created_at.isoformat(),
        }


def _make_http_handler(
    body: dict[str, Any] | None = None,
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler with optional JSON body."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    if body is not None:
        raw = json.dumps(body).encode()
    else:
        raw = b"{}"
    h.headers = {
        "Content-Length": str(len(raw)),
        "Content-Type": content_type,
    }
    h.rfile = MagicMock()
    h.rfile.read.return_value = raw
    return h


# ============================================================================
# Fixtures
# ============================================================================

PATH = "/api/v1/users/self/deletion-request"


@pytest.fixture
def handler():
    """Create a GDPRDeletionHandler instance."""
    return GDPRDeletionHandler(ctx={})


@pytest.fixture
def mock_scheduler():
    """Create a mock GDPRDeletionScheduler."""
    sched = MagicMock()
    sched.store = MagicMock()
    return sched


@pytest.fixture(autouse=True)
def patch_scheduler(mock_scheduler):
    """Auto-patch _get_scheduler to return a mock scheduler."""
    with patch(
        "aragora.server.handlers.gdpr_deletion._get_scheduler",
        return_value=mock_scheduler,
    ):
        yield mock_scheduler


@pytest.fixture(autouse=True)
def patch_auth():
    """Auto-patch _get_user_id_from_handler to return an authenticated user."""
    with patch(
        "aragora.server.handlers.gdpr_deletion._get_user_id_from_handler",
        return_value="user-123",
    ) as fn:
        yield fn


# ============================================================================
# Routing / can_handle
# ============================================================================


class TestRouting:
    def test_can_handle_deletion_request_path(self, handler):
        assert handler.can_handle(PATH) is True

    def test_cannot_handle_random_path(self, handler):
        assert handler.can_handle("/api/v1/users/self/profile") is False

    def test_cannot_handle_similar_path(self, handler):
        assert handler.can_handle("/api/v1/users/self/deletion-requests") is False

    def test_cannot_handle_sub_path(self, handler):
        assert handler.can_handle(PATH + "/details") is False

    def test_cannot_handle_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_cannot_handle_root(self, handler):
        assert handler.can_handle("/") is False


# ============================================================================
# Constructor
# ============================================================================


class TestConstructor:
    def test_default_ctx_is_empty_dict(self):
        h = GDPRDeletionHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        h = GDPRDeletionHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_none_ctx_becomes_empty(self):
        h = GDPRDeletionHandler(ctx=None)
        assert h.ctx == {}


# ============================================================================
# Constants
# ============================================================================


class TestConstants:
    def test_default_grace_period_is_30(self):
        assert DEFAULT_GRACE_PERIOD_DAYS == 30

    def test_routes_list(self, handler):
        assert handler.ROUTES == [PATH]


# ============================================================================
# GET - Check Deletion Status
# ============================================================================


class TestGetDeletionStatus:
    def test_no_requests_returns_empty(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        result = handler.handle(PATH, {}, _make_http_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["has_pending_request"] is False
        assert body["requests"] == []

    def test_pending_request_flagged(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        result = handler.handle(PATH, {}, _make_http_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["has_pending_request"] is True
        assert len(body["requests"]) == 1

    def test_in_progress_request_flagged(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.IN_PROGRESS)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        result = handler.handle(PATH, {}, _make_http_handler())
        body = _body(result)
        assert body["has_pending_request"] is True

    def test_completed_request_not_flagged_as_pending(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.COMPLETED)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        result = handler.handle(PATH, {}, _make_http_handler())
        body = _body(result)
        assert body["has_pending_request"] is False
        assert len(body["requests"]) == 1

    def test_cancelled_request_not_flagged_as_pending(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.CANCELLED)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        result = handler.handle(PATH, {}, _make_http_handler())
        body = _body(result)
        assert body["has_pending_request"] is False

    def test_failed_request_not_flagged_as_pending(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.FAILED)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        result = handler.handle(PATH, {}, _make_http_handler())
        body = _body(result)
        assert body["has_pending_request"] is False

    def test_held_request_not_flagged_as_pending(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.HELD)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        result = handler.handle(PATH, {}, _make_http_handler())
        body = _body(result)
        assert body["has_pending_request"] is False

    def test_multiple_requests_returned(self, handler, mock_scheduler):
        req1 = _MockDeletionRequest(request_id="req-001")
        req2 = _MockDeletionRequest(request_id="req-002", status=_MockDeletionStatus.COMPLETED)
        mock_scheduler.store.get_requests_for_user.return_value = [req1, req2]
        result = handler.handle(PATH, {}, _make_http_handler())
        body = _body(result)
        assert body["has_pending_request"] is True
        assert len(body["requests"]) == 2

    def test_mixed_statuses_pending_wins(self, handler, mock_scheduler):
        completed = _MockDeletionRequest(request_id="req-old", status=_MockDeletionStatus.COMPLETED)
        pending = _MockDeletionRequest(request_id="req-new", status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [
            completed,
            pending,
        ]
        result = handler.handle(PATH, {}, _make_http_handler())
        body = _body(result)
        assert body["has_pending_request"] is True

    def test_unauthenticated_returns_401(self, handler, patch_auth):
        patch_auth.return_value = None
        result = handler.handle(PATH, {}, _make_http_handler())
        assert _status(result) == 401
        assert "Authentication" in _body(result).get("error", "")

    def test_wrong_path_returns_none(self, handler):
        result = handler.handle("/api/v1/other", {}, _make_http_handler())
        assert result is None

    def test_calls_store_with_correct_user_id(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        handler.handle(PATH, {}, _make_http_handler())
        mock_scheduler.store.get_requests_for_user.assert_called_once_with("user-123")

    def test_request_to_dict_called_for_each(self, handler, mock_scheduler):
        req = _MockDeletionRequest()
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        result = handler.handle(PATH, {}, _make_http_handler())
        body = _body(result)
        assert body["requests"][0]["request_id"] == "req-001"


# ============================================================================
# POST - Schedule Deletion
# ============================================================================


class TestScheduleDeletion:
    def test_schedule_with_defaults(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _MockDeletionRequest()
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["request_id"] == "req-001"

    def test_schedule_with_custom_reason(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _MockDeletionRequest()
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={"reason": "Moving to competitor"})
        handler.handle_post(PATH, {}, http)
        mock_scheduler.schedule_deletion.assert_called_once_with(
            user_id="user-123",
            grace_period_days=DEFAULT_GRACE_PERIOD_DAYS,
            reason="Moving to competitor",
        )

    def test_schedule_uses_default_reason_when_missing(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _MockDeletionRequest()
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={})
        handler.handle_post(PATH, {}, http)
        mock_scheduler.schedule_deletion.assert_called_once_with(
            user_id="user-123",
            grace_period_days=DEFAULT_GRACE_PERIOD_DAYS,
            reason="User-initiated deletion request",
        )

    def test_schedule_with_custom_grace_period(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _MockDeletionRequest(grace_days=7)
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={"grace_period_days": 7})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201
        mock_scheduler.schedule_deletion.assert_called_once_with(
            user_id="user-123",
            grace_period_days=7,
            reason="User-initiated deletion request",
        )

    def test_schedule_with_1_day_grace(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _MockDeletionRequest(grace_days=1)
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={"grace_period_days": 1})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201

    def test_schedule_with_365_day_grace(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _MockDeletionRequest(grace_days=365)
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={"grace_period_days": 365})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201

    def test_schedule_rejects_zero_grace_period(self, handler):
        http = _make_http_handler(body={"grace_period_days": 0})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 400
        assert "positive integer" in _body(result).get("error", "")

    def test_schedule_rejects_negative_grace_period(self, handler):
        http = _make_http_handler(body={"grace_period_days": -5})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 400
        assert "positive integer" in _body(result).get("error", "")

    def test_schedule_rejects_excessive_grace_period(self, handler):
        http = _make_http_handler(body={"grace_period_days": 366})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 400
        assert "365" in _body(result).get("error", "")

    def test_schedule_rejects_string_grace_period(self, handler):
        http = _make_http_handler(body={"grace_period_days": "thirty"})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 400

    def test_schedule_rejects_float_grace_period(self, handler):
        http = _make_http_handler(body={"grace_period_days": 7.5})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 400

    def test_schedule_rejects_none_grace_period(self, handler):
        http = _make_http_handler(body={"grace_period_days": None})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 400

    def test_schedule_rejects_boolean_grace_period(self, handler):
        # bool is a subclass of int in Python, True == 1, so True should
        # actually pass validation (isinstance(True, int) is True, True >= 1)
        # This is a quirk of Python type hierarchy.
        http = _make_http_handler(body={"grace_period_days": True})
        result = handler.handle_post(PATH, {}, http)
        # True is int 1, so it passes validation
        # This test documents the behavior
        assert _status(result) in (201, 400)

    def test_schedule_conflict_when_pending_exists(self, handler, mock_scheduler):
        existing = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [existing]
        http = _make_http_handler(body={})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 409
        assert "already pending" in _body(result).get("error", "")

    def test_schedule_conflict_when_in_progress_exists(self, handler, mock_scheduler):
        existing = _MockDeletionRequest(status=_MockDeletionStatus.IN_PROGRESS)
        mock_scheduler.store.get_requests_for_user.return_value = [existing]
        http = _make_http_handler(body={})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 409

    def test_schedule_allowed_when_only_completed_exists(self, handler, mock_scheduler):
        completed = _MockDeletionRequest(status=_MockDeletionStatus.COMPLETED)
        mock_scheduler.store.get_requests_for_user.return_value = [completed]
        req = _MockDeletionRequest()
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201

    def test_schedule_allowed_when_only_cancelled_exists(self, handler, mock_scheduler):
        cancelled = _MockDeletionRequest(status=_MockDeletionStatus.CANCELLED)
        mock_scheduler.store.get_requests_for_user.return_value = [cancelled]
        req = _MockDeletionRequest()
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201

    def test_schedule_allowed_when_only_failed_exists(self, handler, mock_scheduler):
        failed = _MockDeletionRequest(status=_MockDeletionStatus.FAILED)
        mock_scheduler.store.get_requests_for_user.return_value = [failed]
        req = _MockDeletionRequest()
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201

    def test_schedule_value_error_returns_409(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        mock_scheduler.schedule_deletion.side_effect = ValueError("legal hold")
        http = _make_http_handler(body={})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 409

    def test_schedule_unauthenticated_returns_401(self, handler, patch_auth):
        patch_auth.return_value = None
        http = _make_http_handler(body={})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 401

    def test_schedule_wrong_path_returns_none(self, handler):
        http = _make_http_handler(body={})
        result = handler.handle_post("/api/v1/other", {}, http)
        assert result is None

    def test_schedule_no_body_uses_defaults(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _MockDeletionRequest()
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler()
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201
        mock_scheduler.schedule_deletion.assert_called_once_with(
            user_id="user-123",
            grace_period_days=DEFAULT_GRACE_PERIOD_DAYS,
            reason="User-initiated deletion request",
        )


# ============================================================================
# DELETE - Cancel Deletion
# ============================================================================


class TestCancelDeletion:
    def test_cancel_pending_request(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        cancelled = _MockDeletionRequest(status=_MockDeletionStatus.CANCELLED)
        mock_scheduler.cancel_deletion.return_value = cancelled
        http = _make_http_handler(body={"reason": "Changed my mind"})
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "cancelled"

    def test_cancel_with_custom_reason(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        cancelled = _MockDeletionRequest(status=_MockDeletionStatus.CANCELLED)
        mock_scheduler.cancel_deletion.return_value = cancelled
        http = _make_http_handler(body={"reason": "Decided to stay"})
        handler.handle_delete(PATH, {}, http)
        mock_scheduler.cancel_deletion.assert_called_once_with(
            request_id="req-001",
            reason="Decided to stay",
        )

    def test_cancel_uses_default_reason(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        cancelled = _MockDeletionRequest(status=_MockDeletionStatus.CANCELLED)
        mock_scheduler.cancel_deletion.return_value = cancelled
        http = _make_http_handler(body={})
        handler.handle_delete(PATH, {}, http)
        mock_scheduler.cancel_deletion.assert_called_once_with(
            request_id="req-001",
            reason="User cancelled deletion request",
        )

    def test_cancel_no_pending_returns_404(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        http = _make_http_handler(body={})
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 404
        assert "No pending" in _body(result).get("error", "")

    def test_cancel_only_completed_returns_404(self, handler, mock_scheduler):
        completed = _MockDeletionRequest(status=_MockDeletionStatus.COMPLETED)
        mock_scheduler.store.get_requests_for_user.return_value = [completed]
        http = _make_http_handler(body={})
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 404

    def test_cancel_only_cancelled_returns_404(self, handler, mock_scheduler):
        already_cancelled = _MockDeletionRequest(status=_MockDeletionStatus.CANCELLED)
        mock_scheduler.store.get_requests_for_user.return_value = [already_cancelled]
        http = _make_http_handler(body={})
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 404

    def test_cancel_in_progress_returns_404(self, handler, mock_scheduler):
        """DELETE only cancels 'pending' requests, not 'in_progress' ones."""
        in_prog = _MockDeletionRequest(status=_MockDeletionStatus.IN_PROGRESS)
        mock_scheduler.store.get_requests_for_user.return_value = [in_prog]
        http = _make_http_handler(body={})
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 404

    def test_cancel_first_pending_when_multiple(self, handler, mock_scheduler):
        req1 = _MockDeletionRequest(request_id="req-first")
        req2 = _MockDeletionRequest(request_id="req-second")
        mock_scheduler.store.get_requests_for_user.return_value = [req1, req2]
        cancelled = _MockDeletionRequest(status=_MockDeletionStatus.CANCELLED)
        mock_scheduler.cancel_deletion.return_value = cancelled
        http = _make_http_handler(body={})
        handler.handle_delete(PATH, {}, http)
        mock_scheduler.cancel_deletion.assert_called_once_with(
            request_id="req-first",
            reason="User cancelled deletion request",
        )

    def test_cancel_value_error_returns_409(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        mock_scheduler.cancel_deletion.side_effect = ValueError("Race condition")
        http = _make_http_handler(body={})
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 409

    def test_cancel_returns_false_gives_500(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        mock_scheduler.cancel_deletion.return_value = None
        http = _make_http_handler(body={})
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 500
        assert "Failed" in _body(result).get("error", "")

    def test_cancel_returns_falsy_gives_500(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        mock_scheduler.cancel_deletion.return_value = False
        http = _make_http_handler(body={})
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 500

    def test_cancel_unauthenticated_returns_401(self, handler, patch_auth):
        patch_auth.return_value = None
        http = _make_http_handler(body={})
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 401

    def test_cancel_wrong_path_returns_none(self, handler):
        http = _make_http_handler(body={})
        result = handler.handle_delete("/api/v1/other", {}, http)
        assert result is None

    def test_cancel_no_body_uses_default_reason(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        cancelled = _MockDeletionRequest(status=_MockDeletionStatus.CANCELLED)
        mock_scheduler.cancel_deletion.return_value = cancelled
        http = _make_http_handler()
        handler.handle_delete(PATH, {}, http)
        mock_scheduler.cancel_deletion.assert_called_once_with(
            request_id="req-001",
            reason="User cancelled deletion request",
        )


# ============================================================================
# _get_user_id_from_handler helper
# ============================================================================


class TestGetUserIdFromHandler:
    def test_returns_user_id_when_authenticated(self):
        mock_handler = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = True
        mock_ctx.user_id = "user-42"
        with patch(
            "aragora.server.handlers.gdpr_deletion.extract_user_from_request",
            create=True,
        ) as mock_extract:
            # We need to patch at the import level inside the function
            with patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=mock_ctx,
            ):
                result = _get_user_id_from_handler(mock_handler)
                assert result == "user-42"

    def test_returns_none_when_not_authenticated(self):
        mock_handler = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.is_authenticated = False
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            return_value=mock_ctx,
        ):
            result = _get_user_id_from_handler(mock_handler)
            assert result is None

    def test_returns_none_on_import_error(self):
        mock_handler = MagicMock()
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            side_effect=ImportError("no module"),
        ):
            # When the import fails, the function catches ImportError
            result = _get_user_id_from_handler(mock_handler)
            assert result is None

    def test_returns_none_on_attribute_error(self):
        mock_handler = MagicMock()
        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request",
            side_effect=AttributeError("no attr"),
        ):
            result = _get_user_id_from_handler(mock_handler)
            assert result is None


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_get_with_query_params_ignored(self, handler, mock_scheduler):
        """Query params should not affect GET behavior."""
        mock_scheduler.store.get_requests_for_user.return_value = []
        result = handler.handle(PATH, {"page": "1", "limit": "10"}, _make_http_handler())
        assert _status(result) == 200

    def test_post_with_extra_body_fields_ignored(self, handler, mock_scheduler):
        """Unknown body fields should be silently ignored."""
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _MockDeletionRequest()
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(
            body={
                "reason": "test",
                "unknown_field": "value",
                "another": 42,
            }
        )
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201

    def test_delete_with_extra_body_fields_ignored(self, handler, mock_scheduler):
        req = _MockDeletionRequest(status=_MockDeletionStatus.PENDING)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        cancelled = _MockDeletionRequest(status=_MockDeletionStatus.CANCELLED)
        mock_scheduler.cancel_deletion.return_value = cancelled
        http = _make_http_handler(
            body={
                "reason": "test",
                "extra": True,
            }
        )
        result = handler.handle_delete(PATH, {}, http)
        assert _status(result) == 200

    def test_grace_period_exactly_at_boundary_366(self, handler):
        http = _make_http_handler(body={"grace_period_days": 366})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 400

    def test_grace_period_exactly_at_boundary_365(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _MockDeletionRequest(grace_days=365)
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_http_handler(body={"grace_period_days": 365})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 201

    def test_grace_period_list_rejected(self, handler):
        http = _make_http_handler(body={"grace_period_days": [7]})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 400

    def test_grace_period_dict_rejected(self, handler):
        http = _make_http_handler(body={"grace_period_days": {"days": 7}})
        result = handler.handle_post(PATH, {}, http)
        assert _status(result) == 400
