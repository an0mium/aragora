"""
Tests for GDPR self-service deletion handler.

Covers:
- POST /api/v1/users/self/deletion-request (schedule deletion)
- GET /api/v1/users/self/deletion-request (check status)
- DELETE /api/v1/users/self/deletion-request (cancel deletion)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.privacy.deletion import (
    DeletionRequest,
    DeletionStatus,
    DeletionStore,
    GDPRDeletionScheduler,
)
from aragora.server.handlers.gdpr_deletion import GDPRDeletionHandler


# ============================================================================
# Fixtures
# ============================================================================


def _make_mock_handler(method="GET", body=None):
    """Create a mock HTTP handler."""
    mock = MagicMock()
    mock.command = method
    if body is not None:
        body_bytes = json.dumps(body).encode()
    else:
        body_bytes = b"{}"
    mock.rfile = MagicMock()
    mock.rfile.read = MagicMock(return_value=body_bytes)
    mock.headers = {"Content-Length": str(len(body_bytes))}
    return mock


def _make_deletion_request(
    user_id="user-123",
    status=DeletionStatus.PENDING,
    grace_days=30,
) -> DeletionRequest:
    now = datetime.now(timezone.utc)
    return DeletionRequest(
        request_id="req-001",
        user_id=user_id,
        scheduled_for=now + timedelta(days=grace_days),
        reason="User request",
        created_at=now,
        status=status,
    )


def _parse(result) -> dict[str, Any]:
    if result is None:
        return {}
    body = result.body
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body) if body else {}


@pytest.fixture
def handler():
    return GDPRDeletionHandler(ctx={})


@pytest.fixture
def mock_scheduler():
    scheduler = MagicMock(spec=GDPRDeletionScheduler)
    scheduler.store = MagicMock(spec=DeletionStore)
    return scheduler


@pytest.fixture(autouse=True)
def patch_scheduler(mock_scheduler):
    """Patch _get_scheduler to return a mock scheduler."""
    with patch(
        "aragora.server.handlers.gdpr_deletion._get_scheduler",
        return_value=mock_scheduler,
    ):
        yield mock_scheduler


@pytest.fixture(autouse=True)
def patch_auth():
    """Patch _get_user_id_from_handler to return an authenticated user ID."""
    with patch(
        "aragora.server.handlers.gdpr_deletion._get_user_id_from_handler",
        return_value="user-123",
    ) as mock_fn:
        yield mock_fn


# ============================================================================
# Route Tests
# ============================================================================


class TestRouting:
    def test_can_handle_deletion_request_path(self, handler):
        assert handler.can_handle("/api/v1/users/self/deletion-request") is True

    def test_cannot_handle_other_paths(self, handler):
        assert handler.can_handle("/api/v1/users/self/profile") is False
        assert handler.can_handle("/api/v1/privacy/export") is False


# ============================================================================
# GET - Check Deletion Status
# ============================================================================


class TestGetDeletionStatus:
    def test_no_pending_requests(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        http = _make_mock_handler()
        result = handler.handle("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert body["has_pending_request"] is False
        assert body["requests"] == []

    def test_has_pending_request(self, handler, mock_scheduler):
        req = _make_deletion_request()
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        http = _make_mock_handler()
        result = handler.handle("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert body["has_pending_request"] is True
        assert len(body["requests"]) == 1
        assert body["requests"][0]["request_id"] == "req-001"

    def test_completed_request_not_pending(self, handler, mock_scheduler):
        req = _make_deletion_request(status=DeletionStatus.COMPLETED)
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        http = _make_mock_handler()
        result = handler.handle("/api/v1/users/self/deletion-request", {}, http)
        body = _parse(result)
        assert body["has_pending_request"] is False

    def test_unauthenticated_returns_401(self, handler, patch_auth):
        patch_auth.return_value = None
        http = _make_mock_handler()
        result = handler.handle("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 401

    def test_returns_none_for_unhandled_path(self, handler):
        http = _make_mock_handler()
        result = handler.handle("/api/v1/other", {}, http)
        assert result is None

    def test_multiple_requests_returned(self, handler, mock_scheduler):
        req1 = _make_deletion_request()
        req2 = _make_deletion_request(status=DeletionStatus.CANCELLED)
        req2.request_id = "req-002"
        mock_scheduler.store.get_requests_for_user.return_value = [req1, req2]
        http = _make_mock_handler()
        result = handler.handle("/api/v1/users/self/deletion-request", {}, http)
        body = _parse(result)
        assert len(body["requests"]) == 2


# ============================================================================
# POST - Schedule Deletion
# ============================================================================


class TestScheduleDeletion:
    def test_schedule_with_defaults(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _make_deletion_request()
        mock_scheduler.schedule_deletion.return_value = req

        http = _make_mock_handler(method="POST", body={"reason": "GDPR request"})
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 201
        body = _parse(result)
        assert body["request_id"] == "req-001"
        assert body["status"] == "pending"

    def test_schedule_with_custom_grace_period(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        req = _make_deletion_request(grace_days=7)
        mock_scheduler.schedule_deletion.return_value = req

        http = _make_mock_handler(
            method="POST",
            body={"grace_period_days": 7},
        )
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 201
        mock_scheduler.schedule_deletion.assert_called_once_with(
            user_id="user-123",
            grace_period_days=7,
            reason="User-initiated deletion request",
        )

    def test_schedule_rejects_invalid_grace_period(self, handler):
        http = _make_mock_handler(method="POST", body={"grace_period_days": 0})
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 400
        body = _parse(result)
        assert "positive integer" in body.get("error", "")

    def test_schedule_rejects_excessive_grace_period(self, handler):
        http = _make_mock_handler(method="POST", body={"grace_period_days": 400})
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 400
        assert "365" in _parse(result).get("error", "")

    def test_schedule_conflict_when_pending_exists(self, handler, mock_scheduler):
        existing = _make_deletion_request()
        mock_scheduler.store.get_requests_for_user.return_value = [existing]
        http = _make_mock_handler(method="POST", body={})
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 409
        assert "already pending" in _parse(result).get("error", "")

    def test_schedule_allowed_when_only_completed_exists(self, handler, mock_scheduler):
        completed = _make_deletion_request(status=DeletionStatus.COMPLETED)
        mock_scheduler.store.get_requests_for_user.return_value = [completed]
        req = _make_deletion_request()
        mock_scheduler.schedule_deletion.return_value = req
        http = _make_mock_handler(method="POST", body={})
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 201

    def test_schedule_legal_hold_returns_409(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        mock_scheduler.schedule_deletion.side_effect = ValueError("legal hold")
        http = _make_mock_handler(method="POST", body={})
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 409

    def test_schedule_unauthenticated_returns_401(self, handler, patch_auth):
        patch_auth.return_value = None
        http = _make_mock_handler(method="POST", body={})
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 401

    def test_schedule_returns_none_for_unhandled_path(self, handler):
        http = _make_mock_handler(method="POST", body={})
        result = handler.handle_post("/api/v1/other", {}, http)
        assert result is None

    def test_schedule_negative_grace_period(self, handler):
        http = _make_mock_handler(method="POST", body={"grace_period_days": -5})
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 400

    def test_schedule_string_grace_period(self, handler):
        http = _make_mock_handler(method="POST", body={"grace_period_days": "seven"})
        result = handler.handle_post("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 400


# ============================================================================
# DELETE - Cancel Deletion
# ============================================================================


class TestCancelDeletion:
    def test_cancel_pending_request(self, handler, mock_scheduler):
        req = _make_deletion_request()
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        cancelled_req = _make_deletion_request(status=DeletionStatus.CANCELLED)
        mock_scheduler.cancel_deletion.return_value = cancelled_req

        http = _make_mock_handler(method="DELETE", body={"reason": "Changed my mind"})
        result = handler.handle_delete("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 200
        body = _parse(result)
        assert body["status"] == "cancelled"

    def test_cancel_no_pending_request(self, handler, mock_scheduler):
        mock_scheduler.store.get_requests_for_user.return_value = []
        http = _make_mock_handler(method="DELETE")
        result = handler.handle_delete("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 404

    def test_cancel_completed_not_found(self, handler, mock_scheduler):
        completed = _make_deletion_request(status=DeletionStatus.COMPLETED)
        mock_scheduler.store.get_requests_for_user.return_value = [completed]
        http = _make_mock_handler(method="DELETE")
        result = handler.handle_delete("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 404

    def test_cancel_unauthenticated_returns_401(self, handler, patch_auth):
        patch_auth.return_value = None
        http = _make_mock_handler(method="DELETE")
        result = handler.handle_delete("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 401

    def test_cancel_returns_none_for_unhandled_path(self, handler):
        http = _make_mock_handler(method="DELETE")
        result = handler.handle_delete("/api/v1/other", {}, http)
        assert result is None

    def test_cancel_value_error_returns_409(self, handler, mock_scheduler):
        req = _make_deletion_request()
        mock_scheduler.store.get_requests_for_user.return_value = [req]
        mock_scheduler.cancel_deletion.side_effect = ValueError("Cannot cancel")
        http = _make_mock_handler(method="DELETE")
        result = handler.handle_delete("/api/v1/users/self/deletion-request", {}, http)
        assert result.status_code == 409

    def test_cancel_returns_first_pending(self, handler, mock_scheduler):
        req1 = _make_deletion_request()
        req1.request_id = "req-first"
        req2 = _make_deletion_request()
        req2.request_id = "req-second"
        mock_scheduler.store.get_requests_for_user.return_value = [req1, req2]
        cancelled = _make_deletion_request(status=DeletionStatus.CANCELLED)
        mock_scheduler.cancel_deletion.return_value = cancelled

        http = _make_mock_handler(method="DELETE")
        handler.handle_delete("/api/v1/users/self/deletion-request", {}, http)
        mock_scheduler.cancel_deletion.assert_called_once_with(
            request_id="req-first",
            reason="User cancelled deletion request",
        )
