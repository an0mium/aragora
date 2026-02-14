"""
Tests for aragora.server.handlers.plans - Decision Plan Handlers.

Tests cover:
- PlansHandler: instantiation, can_handle routing
- GET /api/v1/plans: list plans with pagination and filters
- GET /api/v1/plans/{id}: get plan by ID
- POST /api/v1/plans: create plan with validation
- POST /api/v1/plans/{id}/approve: approve plan workflow
- POST /api/v1/plans/{id}/reject: reject plan workflow
- POST /api/v1/plans/{id}/execute: execute approved plan
- Serialization: _plan_summary, _plan_detail
- Error paths: plan not found, invalid status, missing fields
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.plans import PlansHandler
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


class MockPlanStatus(str, Enum):
    CREATED = "created"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


class MockApprovalMode(str, Enum):
    ALWAYS = "always"
    RISK_BASED = "risk_based"
    NEVER = "never"


class MockPlan:
    """Mock DecisionPlan."""

    def __init__(
        self,
        plan_id: str = "plan-001",
        debate_id: str = "debate-001",
        task: str = "Implement feature X",
        status: str = "awaiting_approval",
        approval_mode: str = "risk_based",
    ):
        self.id = plan_id
        self.debate_id = debate_id
        self.task = task
        self.status = MockPlanStatus(status)
        self.approval_mode = MockApprovalMode(approval_mode)
        self.created_at = datetime.now(timezone.utc)
        self.has_critical_risks = False
        self.requires_human_approval = True
        self.metadata = {"action_items": ["Item 1"], "estimated_duration": "2h"}
        self.budget = MagicMock()
        self.budget.limit_usd = 100.0

    def approve(self, approver_id: str, reason: str = "", conditions: list | None = None):
        self.status = MockPlanStatus.APPROVED

    def reject(self, rejecter_id: str, reason: str = ""):
        self.status = MockPlanStatus.REJECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "debate_id": self.debate_id,
            "task": self.task,
            "status": self.status.value,
            "approval_mode": self.approval_mode.value,
            "created_at": self.created_at.isoformat(),
            "has_critical_risks": self.has_critical_risks,
            "requires_human_approval": self.requires_human_approval,
            "metadata": self.metadata,
        }


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    user_id: str = "user-001",
) -> MagicMock:
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    # Simulate auth
    mock_user = MagicMock()
    mock_user.user_id = user_id
    handler._user = mock_user
    return handler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_store():
    """Create a mock plan store."""
    store = MagicMock()
    store.list.return_value = [MockPlan(), MockPlan("plan-002", task="Another task")]
    store.count.return_value = 2
    store.get.return_value = MockPlan()
    store.create.return_value = None
    store.update_status.return_value = None
    return store


@pytest.fixture
def handler(mock_store):
    """Create PlansHandler with mocked dependencies."""
    with patch(
        "aragora.server.handlers.plans._get_plan_store",
        return_value=mock_store,
    ):
        h = PlansHandler(ctx={})
        # Mock auth methods
        mock_user = MagicMock()
        mock_user.user_id = "user-001"
        h.require_auth_or_error = MagicMock(return_value=(mock_user, None))
        h.require_permission_or_error = MagicMock(return_value=(mock_user, None))
        yield h


# ===========================================================================
# Test Basics
# ===========================================================================


class TestPlansHandlerBasics:
    """Basic instantiation and routing tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, PlansHandler)

    def test_can_handle_list(self, handler):
        assert handler.can_handle("/api/v1/plans") is True

    def test_can_handle_unversioned(self, handler):
        assert handler.can_handle("/api/plans") is True

    def test_can_handle_specific_plan(self, handler):
        assert handler.can_handle("/api/v1/plans/plan-001") is True

    def test_can_handle_approve(self, handler):
        assert handler.can_handle("/api/v1/plans/plan-001/approve") is True

    def test_can_handle_reject(self, handler):
        assert handler.can_handle("/api/v1/plans/plan-001/reject") is True

    def test_can_handle_execute(self, handler):
        assert handler.can_handle("/api/v1/plans/plan-001/execute") is True

    def test_cannot_handle_other(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_cannot_handle_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/planning") is False


# ===========================================================================
# Test _list_plans
# ===========================================================================


class TestListPlans:
    """Tests for GET /api/v1/plans."""

    def test_list_plans_success(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ):
            result = handler._list_plans({})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["total"] == 2
            assert len(data["plans"]) == 2

    def test_list_plans_with_status_filter(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ):
            result = handler._list_plans({"status": "approved"})
            assert result.status_code == 200

    def test_list_plans_invalid_status(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ):
            result = handler._list_plans({"status": "nonexistent_status"})
            assert result.status_code == 400

    def test_list_plans_with_pagination(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ):
            result = handler._list_plans({"limit": "10", "offset": "5"})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["limit"] == 10
            assert data["offset"] == 5


# ===========================================================================
# Test _get_plan
# ===========================================================================


class TestGetPlan:
    """Tests for GET /api/v1/plans/{id}."""

    def test_get_plan_success(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ):
            result = handler._get_plan({"plan_id": "plan-001"}, {})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["id"] == "plan-001"

    def test_get_plan_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ):
            result = handler._get_plan({"plan_id": "nonexistent"}, {})
            assert result.status_code == 404


# ===========================================================================
# Test _create_plan
# ===========================================================================


class TestCreatePlan:
    """Tests for POST /api/v1/plans."""

    def test_create_plan_success(self, handler, mock_store):
        body = {"debate_id": "debate-001", "task": "Implement feature"}
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch.object(handler, "get_json_body", return_value=body), patch(
            "aragora.server.handlers.plans.DecisionPlan", MockPlan,
        ), patch(
            "aragora.server.handlers.plans.ApprovalMode", MockApprovalMode,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ), patch(
            "aragora.server.handlers.plans._fire_plan_notification",
        ):
            result = handler._create_plan({})
            assert result.status_code == 201

    def test_create_plan_missing_body(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch.object(handler, "get_json_body", return_value=None):
            result = handler._create_plan({})
            assert result.status_code == 400

    def test_create_plan_missing_debate_id(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch.object(handler, "get_json_body", return_value={"task": "Do something"}):
            result = handler._create_plan({})
            assert result.status_code == 400

    def test_create_plan_missing_task(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch.object(handler, "get_json_body", return_value={"debate_id": "d-1"}):
            result = handler._create_plan({})
            assert result.status_code == 400


# ===========================================================================
# Test _approve_plan
# ===========================================================================


class TestApprovePlan:
    """Tests for POST /api/v1/plans/{id}/approve."""

    def test_approve_plan_success(self, handler, mock_store):
        plan = MockPlan(status="awaiting_approval")
        mock_store.get.return_value = plan

        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ), patch.object(handler, "get_json_body", return_value={}), patch(
            "aragora.server.handlers.plans._fire_plan_notification",
        ):
            result = handler._approve_plan({"plan_id": "plan-001"}, {})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "approved"

    def test_approve_plan_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ):
            result = handler._approve_plan({"plan_id": "nonexistent"}, {})
            assert result.status_code == 404

    def test_approve_already_approved(self, handler, mock_store):
        plan = MockPlan(status="approved")
        mock_store.get.return_value = plan
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ):
            result = handler._approve_plan({"plan_id": "plan-001"}, {})
            assert result.status_code == 409


# ===========================================================================
# Test _reject_plan
# ===========================================================================


class TestRejectPlan:
    """Tests for POST /api/v1/plans/{id}/reject."""

    def test_reject_plan_success(self, handler, mock_store):
        plan = MockPlan(status="awaiting_approval")
        mock_store.get.return_value = plan

        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ), patch.object(
            handler, "get_json_body", return_value={"reason": "Not ready"}
        ), patch(
            "aragora.server.handlers.plans._fire_plan_notification",
        ):
            result = handler._reject_plan({"plan_id": "plan-001"}, {})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["status"] == "rejected"
            assert data["reason"] == "Not ready"

    def test_reject_plan_missing_reason(self, handler, mock_store):
        plan = MockPlan(status="awaiting_approval")
        mock_store.get.return_value = plan
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ), patch.object(
            handler, "get_json_body", return_value={}
        ):
            result = handler._reject_plan({"plan_id": "plan-001"}, {})
            assert result.status_code == 400

    def test_reject_plan_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ):
            result = handler._reject_plan({"plan_id": "nonexistent"}, {})
            assert result.status_code == 404


# ===========================================================================
# Test _execute_plan
# ===========================================================================


class TestExecutePlan:
    """Tests for POST /api/v1/plans/{id}/execute."""

    def test_execute_approved_plan(self, handler, mock_store):
        plan = MockPlan(status="approved")
        mock_store.get.return_value = plan

        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ), patch.object(
            handler, "get_json_body", return_value={}
        ), patch(
            "aragora.server.handlers.plans.get_execution_bridge",
            return_value=MagicMock(),
        ), patch(
            "aragora.server.handlers.plans._fire_plan_notification",
        ):
            result = handler._execute_plan({"plan_id": "plan-001"}, {})
            assert result.status_code == 202

    def test_execute_unapproved_plan(self, handler, mock_store):
        plan = MockPlan(status="awaiting_approval")
        mock_store.get.return_value = plan
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ), patch.object(handler, "get_json_body", return_value={}):
            result = handler._execute_plan({"plan_id": "plan-001"}, {})
            assert result.status_code == 409

    def test_execute_already_executing(self, handler, mock_store):
        plan = MockPlan(status="executing")
        mock_store.get.return_value = plan
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ), patch.object(handler, "get_json_body", return_value={}):
            result = handler._execute_plan({"plan_id": "plan-001"}, {})
            assert result.status_code == 409

    def test_execute_completed_plan(self, handler, mock_store):
        plan = MockPlan(status="completed")
        mock_store.get.return_value = plan
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ), patch.object(handler, "get_json_body", return_value={}):
            result = handler._execute_plan({"plan_id": "plan-001"}, {})
            assert result.status_code == 409

    def test_execute_not_found(self, handler, mock_store):
        mock_store.get.return_value = None
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ), patch.object(handler, "get_json_body", return_value={}):
            result = handler._execute_plan({"plan_id": "nonexistent"}, {})
            assert result.status_code == 404


# ===========================================================================
# Test Serialization
# ===========================================================================


class TestSerialization:
    """Tests for plan serialization helpers."""

    def test_plan_summary(self):
        plan = MockPlan()
        summary = PlansHandler._plan_summary(plan)
        assert summary["id"] == "plan-001"
        assert summary["status"] == "awaiting_approval"
        assert "created_at" in summary
        assert len(summary["task"]) <= 200

    def test_plan_detail(self):
        plan = MockPlan()
        detail = PlansHandler._plan_detail(plan)
        assert detail["id"] == "plan-001"
        assert "action_items" in detail
        assert detail["estimated_duration"] == "2h"

    def test_plan_summary_truncates_long_task(self):
        plan = MockPlan(task="x" * 300)
        summary = PlansHandler._plan_summary(plan)
        assert len(summary["task"]) == 200


# ===========================================================================
# Test handle() routing
# ===========================================================================


class TestHandleRouting:
    """Tests for top-level handle() method routing."""

    def test_handle_list_plans(self, handler, mock_store):
        mock_handler = _make_mock_handler()
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ), patch(
            "aragora.server.handlers.plans.PlanStatus", MockPlanStatus,
        ):
            handler.set_request_context(mock_handler, {})
            result = handler._get_dispatcher.dispatch("/api/v1/plans", {})
            assert result is not None
            assert result.status_code == 200

    def test_handle_get_by_id_fallback(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ):
            result = handler._try_get_by_id("/api/v1/plans/plan-001", {})
            assert result is not None
            assert result.status_code == 200

    def test_handle_unversioned_path(self, handler, mock_store):
        with patch(
            "aragora.server.handlers.plans._get_plan_store",
            return_value=mock_store,
        ):
            result = handler._try_get_by_id("/api/plans/plan-001", {})
            assert result is not None
            assert result.status_code == 200
