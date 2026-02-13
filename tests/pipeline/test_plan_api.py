"""Tests for the Plans HTTP handler API endpoints.

Tests:
- POST /api/v1/plans - create plan
- GET /api/v1/plans - list plans
- GET /api/v1/plans/{id} - get plan
- POST /api/v1/plans/{id}/approve - approve (RBAC gated)
- POST /api/v1/plans/{id}/reject - reject (RBAC gated)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipeline.decision_plan.core import (
    ApprovalMode,
    DecisionPlan,
    PlanStatus,
)
from aragora.pipeline.plan_store import PlanStore
from aragora.server.handlers.plans import PlansHandler


@pytest.fixture
def tmp_store(tmp_path: Path) -> PlanStore:
    """Create a PlanStore with a temp database."""
    return PlanStore(db_path=str(tmp_path / "plans_api.db"))


@pytest.fixture
def handler(tmp_store: PlanStore) -> PlansHandler:
    """Create a PlansHandler with a temp store."""
    ctx: dict[str, Any] = {}
    h = PlansHandler(ctx)
    return h


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler with auth context."""
    mock = MagicMock()
    mock.command = "GET"
    mock.headers = {}
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = b"{}"
    # Add auth context
    from aragora.rbac.models import AuthorizationContext

    mock._auth_context = AuthorizationContext(
        user_id="test-user",
        user_email="test@example.com",
        roles={"admin"},
        permissions=set(),
    )
    return mock


def _make_mock_handler_with_body(body: dict[str, Any]) -> MagicMock:
    """Create a mock handler with a JSON body."""
    from aragora.rbac.models import AuthorizationContext

    mock = MagicMock()
    mock.command = "POST"
    encoded = json.dumps(body).encode()
    mock.headers = {"Content-Length": str(len(encoded))}
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = encoded
    mock._auth_context = AuthorizationContext(
        user_id="test-user",
        user_email="test@example.com",
        roles={"admin"},
        permissions=set(),
    )
    return mock


class TestCanHandle:
    """Tests for route matching."""

    def test_handles_plans_root(self, handler: PlansHandler) -> None:
        assert handler.can_handle("/api/v1/plans") is True
        assert handler.can_handle("/api/plans") is True

    def test_handles_plan_by_id(self, handler: PlansHandler) -> None:
        assert handler.can_handle("/api/v1/plans/dp-abc123") is True
        assert handler.can_handle("/api/plans/dp-abc123") is True

    def test_handles_plan_actions(self, handler: PlansHandler) -> None:
        assert handler.can_handle("/api/v1/plans/dp-abc123/approve") is True
        assert handler.can_handle("/api/v1/plans/dp-abc123/reject") is True

    def test_does_not_handle_other_paths(self, handler: PlansHandler) -> None:
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/workflows") is False


class TestListPlans:
    """Tests for GET /api/v1/plans."""

    def test_list_empty(self, handler: PlansHandler, tmp_store: PlanStore, mock_handler: MagicMock) -> None:
        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle("/api/v1/plans", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["plans"] == []
        assert data["total"] == 0

    def test_list_with_plans(self, handler: PlansHandler, tmp_store: PlanStore, mock_handler: MagicMock) -> None:
        tmp_store.create(DecisionPlan(id="dp-1", debate_id="d1", task="Task 1"))
        tmp_store.create(DecisionPlan(id="dp-2", debate_id="d2", task="Task 2"))

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle("/api/v1/plans", {}, mock_handler)

        assert result is not None
        data = json.loads(result.body)
        assert data["total"] == 2
        assert len(data["plans"]) == 2

    def test_list_filter_by_status(self, handler: PlansHandler, tmp_store: PlanStore, mock_handler: MagicMock) -> None:
        tmp_store.create(DecisionPlan(id="dp-a", debate_id="d1", task="T", status=PlanStatus.APPROVED))
        tmp_store.create(DecisionPlan(id="dp-b", debate_id="d2", task="T", status=PlanStatus.AWAITING_APPROVAL))

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle("/api/v1/plans", {"status": "approved"}, mock_handler)

        data = json.loads(result.body)
        assert data["total"] == 1
        assert data["plans"][0]["id"] == "dp-a"

    def test_list_invalid_status(self, handler: PlansHandler, tmp_store: PlanStore, mock_handler: MagicMock) -> None:
        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle("/api/v1/plans", {"status": "bogus"}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_list_filter_by_debate_id(self, handler: PlansHandler, tmp_store: PlanStore, mock_handler: MagicMock) -> None:
        tmp_store.create(DecisionPlan(id="dp-x", debate_id="target", task="T"))
        tmp_store.create(DecisionPlan(id="dp-y", debate_id="other", task="T"))

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle("/api/v1/plans", {"debate_id": "target"}, mock_handler)

        data = json.loads(result.body)
        assert data["total"] == 1
        assert data["plans"][0]["debate_id"] == "target"


class TestGetPlan:
    """Tests for GET /api/v1/plans/{id}."""

    def test_get_existing(self, handler: PlansHandler, tmp_store: PlanStore, mock_handler: MagicMock) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-detail",
            debate_id="d-detail",
            task="Detailed plan",
            status=PlanStatus.AWAITING_APPROVAL,
            metadata={"action_items": [{"description": "Step 1"}]},
        ))

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle("/api/v1/plans/dp-detail", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["id"] == "dp-detail"
        assert data["task"] == "Detailed plan"
        assert data["status"] == "awaiting_approval"
        assert len(data["action_items"]) == 1

    def test_get_nonexistent(self, handler: PlansHandler, tmp_store: PlanStore, mock_handler: MagicMock) -> None:
        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle("/api/v1/plans/dp-ghost", {}, mock_handler)

        assert result is not None
        assert result.status_code == 404


class TestCreatePlan:
    """Tests for POST /api/v1/plans."""

    def test_create_plan(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        body = {
            "debate_id": "debate-123",
            "task": "Build the feature",
            "estimated_budget": 25.0,
            "estimated_duration": "2 weeks",
            "action_items": [
                {"description": "Write code", "assignee": "dev-1"},
                {"description": "Write tests", "assignee": "dev-2"},
            ],
        }
        mock = _make_mock_handler_with_body(body)

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle_post("/api/v1/plans", {}, mock)

        assert result is not None
        assert result.status_code == 201
        data = json.loads(result.body)
        assert data["debate_id"] == "debate-123"
        assert data["task"] == "Build the feature"
        assert data["status"] == "awaiting_approval"
        assert len(data["action_items"]) == 2

        # Verify persisted
        stored = tmp_store.get(data["id"])
        assert stored is not None
        assert stored.debate_id == "debate-123"

    def test_create_plan_missing_debate_id(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        mock = _make_mock_handler_with_body({"task": "Something"})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle_post("/api/v1/plans", {}, mock)

        assert result is not None
        assert result.status_code == 400
        assert b"debate_id" in result.body

    def test_create_plan_missing_task(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        mock = _make_mock_handler_with_body({"debate_id": "d-1"})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle_post("/api/v1/plans", {}, mock)

        assert result is not None
        assert result.status_code == 400
        assert b"task" in result.body

    def test_create_plan_auto_approve_never_mode(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        mock = _make_mock_handler_with_body({
            "debate_id": "d-auto",
            "task": "Auto approved",
            "approval_mode": "never",
        })

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store):
            result = handler.handle_post("/api/v1/plans", {}, mock)

        data = json.loads(result.body)
        assert data["status"] == "approved"


class TestApprovePlan:
    """Tests for POST /api/v1/plans/{id}/approve."""

    def test_approve_plan(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-approve-me",
            debate_id="d1",
            task="T",
            status=PlanStatus.AWAITING_APPROVAL,
        ))

        mock = _make_mock_handler_with_body({"reason": "Looks good"})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-approve-me/approve", {}, mock)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "approved"
        assert data["approved_by"] == "admin-1"

        # Verify persisted
        stored = tmp_store.get("dp-approve-me")
        assert stored is not None
        assert stored.status == PlanStatus.APPROVED

    def test_approve_nonexistent(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        mock = _make_mock_handler_with_body({})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-ghost/approve", {}, mock)

        assert result is not None
        assert result.status_code == 404

    def test_approve_already_approved(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-done",
            debate_id="d1",
            task="T",
            status=PlanStatus.APPROVED,
        ))

        mock = _make_mock_handler_with_body({})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-done/approve", {}, mock)

        assert result is not None
        assert result.status_code == 409

    def test_approve_requires_permission(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        from aragora.server.handlers.base import error_response

        tmp_store.create(DecisionPlan(
            id="dp-perm", debate_id="d1", task="T", status=PlanStatus.AWAITING_APPROVAL,
        ))

        mock = _make_mock_handler_with_body({})
        perm_error = error_response("Permission denied", 403)

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(None, perm_error)):
            result = handler.handle_post("/api/v1/plans/dp-perm/approve", {}, mock)

        assert result is not None
        assert result.status_code == 403


class TestRejectPlan:
    """Tests for POST /api/v1/plans/{id}/reject."""

    def test_reject_plan(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-reject-me",
            debate_id="d1",
            task="T",
            status=PlanStatus.AWAITING_APPROVAL,
        ))

        mock = _make_mock_handler_with_body({"reason": "Too risky"})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-reject-me/reject", {}, mock)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["status"] == "rejected"
        assert data["reason"] == "Too risky"

        # Verify persisted
        stored = tmp_store.get("dp-reject-me")
        assert stored is not None
        assert stored.status == PlanStatus.REJECTED

    def test_reject_requires_reason(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-no-reason", debate_id="d1", task="T", status=PlanStatus.AWAITING_APPROVAL,
        ))

        mock = _make_mock_handler_with_body({})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-no-reason/reject", {}, mock)

        assert result is not None
        assert result.status_code == 400
        assert b"reason" in result.body

    def test_reject_requires_permission(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        from aragora.server.handlers.base import error_response

        mock = _make_mock_handler_with_body({"reason": "some reason"})
        perm_error = error_response("Permission denied", 403)

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(None, perm_error)):
            result = handler.handle_post("/api/v1/plans/dp-any/reject", {}, mock)

        assert result is not None
        assert result.status_code == 403


class TestExecuteEndpoint:
    """Tests for POST /api/v1/plans/{id}/execute."""

    def test_handles_execute_path(self, handler: PlansHandler) -> None:
        assert handler.can_handle("/api/v1/plans/dp-abc123/execute") is True

    def test_execute_approved_plan(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-exec",
            debate_id="d1",
            task="T",
            status=PlanStatus.APPROVED,
        ))

        mock = _make_mock_handler_with_body({})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)), \
             patch("aragora.pipeline.execution_bridge.get_execution_bridge") as mock_bridge:
            mock_bridge.return_value = MagicMock()
            result = handler.handle_post("/api/v1/plans/dp-exec/execute", {}, mock)

        assert result is not None
        assert result.status_code == 202
        data = json.loads(result.body)
        assert data["status"] == "executing"
        assert data["plan_id"] == "dp-exec"
        mock_bridge.return_value.schedule_execution.assert_called_once()

    def test_execute_nonexistent_plan(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        mock = _make_mock_handler_with_body({})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-ghost/execute", {}, mock)

        assert result is not None
        assert result.status_code == 404

    def test_execute_unapproved_plan(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-pending",
            debate_id="d1",
            task="T",
            status=PlanStatus.AWAITING_APPROVAL,
        ))

        mock = _make_mock_handler_with_body({})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-pending/execute", {}, mock)

        assert result is not None
        assert result.status_code == 409

    def test_execute_already_executing(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-running",
            debate_id="d1",
            task="T",
            status=PlanStatus.EXECUTING,
        ))

        mock = _make_mock_handler_with_body({})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-running/execute", {}, mock)

        assert result is not None
        assert result.status_code == 409

    def test_execute_completed_plan(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-done",
            debate_id="d1",
            task="T",
            status=PlanStatus.COMPLETED,
        ))

        mock = _make_mock_handler_with_body({})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-done/execute", {}, mock)

        assert result is not None
        assert result.status_code == 409

    def test_execute_with_custom_mode(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-hybrid",
            debate_id="d1",
            task="T",
            status=PlanStatus.APPROVED,
        ))

        mock = _make_mock_handler_with_body({"execution_mode": "hybrid"})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)), \
             patch("aragora.pipeline.execution_bridge.get_execution_bridge") as mock_bridge:
            mock_bridge.return_value = MagicMock()
            result = handler.handle_post("/api/v1/plans/dp-hybrid/execute", {}, mock)

        assert result is not None
        assert result.status_code == 202
        mock_bridge.return_value.schedule_execution.assert_called_once_with(
            "dp-hybrid",
            execution_mode="hybrid",
        )

    def test_execute_requires_permission(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        from aragora.server.handlers.base import error_response

        mock = _make_mock_handler_with_body({})
        perm_error = error_response("Permission denied", 403)

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(None, perm_error)):
            result = handler.handle_post("/api/v1/plans/dp-any/execute", {}, mock)

        assert result is not None
        assert result.status_code == 403


class TestApproveAutoExecute:
    """Tests for auto_execute flag on approval."""

    def test_approve_with_auto_execute(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-auto-exec",
            debate_id="d1",
            task="T",
            status=PlanStatus.AWAITING_APPROVAL,
        ))

        mock = _make_mock_handler_with_body({"auto_execute": True})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)), \
             patch("aragora.pipeline.execution_bridge.get_execution_bridge") as mock_bridge:
            mock_bridge.return_value = MagicMock()
            result = handler.handle_post("/api/v1/plans/dp-auto-exec/approve", {}, mock)

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["execution_scheduled"] is True
        mock_bridge.return_value.schedule_execution.assert_called_once_with("dp-auto-exec")

    def test_approve_without_auto_execute(self, handler: PlansHandler, tmp_store: PlanStore) -> None:
        tmp_store.create(DecisionPlan(
            id="dp-no-exec",
            debate_id="d1",
            task="T",
            status=PlanStatus.AWAITING_APPROVAL,
        ))

        mock = _make_mock_handler_with_body({"reason": "approved"})

        with patch("aragora.server.handlers.plans._get_plan_store", return_value=tmp_store), \
             patch.object(handler, "require_permission_or_error", return_value=(MagicMock(user_id="admin-1"), None)):
            result = handler.handle_post("/api/v1/plans/dp-no-exec/approve", {}, mock)

        assert result is not None
        data = json.loads(result.body)
        assert data["execution_scheduled"] is False


class TestRBACPermissions:
    """Tests verifying RBAC permission definitions."""

    def test_plan_permissions_exist(self) -> None:
        from aragora.rbac.defaults.permissions.debate import (
            PERM_PLAN_APPROVE,
            PERM_PLAN_CREATE,
            PERM_PLAN_READ,
            PERM_PLAN_REJECT,
        )

        assert PERM_PLAN_CREATE.key == "plans.create"
        assert PERM_PLAN_READ.key == "plans.read"
        assert PERM_PLAN_APPROVE.key == "plans.approve"
        assert PERM_PLAN_REJECT.key == "plans.deny"

    def test_admin_role_has_plan_approve(self) -> None:
        from aragora.rbac.defaults.helpers import get_role_permissions

        admin_perms = get_role_permissions("admin", include_inherited=True)
        assert "plans.create" in admin_perms
        assert "plans.read" in admin_perms
        assert "plans.approve" in admin_perms
        assert "plans.deny" in admin_perms

    def test_member_role_has_plan_create_read(self) -> None:
        from aragora.rbac.defaults.helpers import get_role_permissions

        member_perms = get_role_permissions("member", include_inherited=True)
        assert "plans.create" in member_perms
        assert "plans.read" in member_perms
        # Members should NOT have approve/reject
        assert "plans.approve" not in member_perms
        assert "plans.deny" not in member_perms

    def test_resource_type_plans_exists(self) -> None:
        from aragora.rbac.models import ResourceType

        assert ResourceType.PLANS.value == "plans"
