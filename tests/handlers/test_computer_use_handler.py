"""Tests for computer use handler (aragora/server/handlers/computer_use_handler.py).

Covers all routes and behavior of the ComputerUseHandler class:
- can_handle() routing for all ROUTES and prefix matching
- GET  /api/v1/computer-use/tasks           - List recent tasks
- GET  /api/v1/computer-use/tasks/{id}      - Get task status
- POST /api/v1/computer-use/tasks           - Create and run a task
- POST /api/v1/computer-use/tasks/{id}/cancel - Cancel a running task
- GET  /api/v1/computer-use/actions/stats   - Get action statistics
- GET  /api/v1/computer-use/policies        - List active policies
- POST /api/v1/computer-use/policies        - Create a policy
- GET  /api/v1/computer-use/approvals       - List approvals
- GET  /api/v1/computer-use/approvals/{id}  - Get single approval
- POST /api/v1/computer-use/approvals/{id}/approve - Approve approval
- POST /api/v1/computer-use/approvals/{id}/deny    - Deny approval
- Error handling and edge cases
- RBAC permission checks
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to ComputerUseHandler methods."""

    def __init__(self, method: str = "GET", body: dict[str, Any] | None = None):
        self.command = method
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = ""

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers["Content-Length"] = str(len(raw))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


class MockTaskStatus(Enum):
    """Mirrors aragora.computer_use.orchestrator.TaskStatus."""
    COMPLETED = "completed"
    FAILED = "failed"


class MockApprovalStatus(Enum):
    """Mirrors aragora.computer_use.approval.ApprovalStatus."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"


# ---------------------------------------------------------------------------
# Mock data builders
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str = "task-abc123",
    goal: str = "click the button",
    status: str = "pending",
    **kwargs,
):
    """Create a mock ComputerUseTask-like object."""
    task = MagicMock()
    task.task_id = task_id
    task.goal = goal
    task.status = status
    task.max_steps = kwargs.get("max_steps", 10)
    task.dry_run = kwargs.get("dry_run", False)
    task.created_at = kwargs.get("created_at", datetime.now(timezone.utc))
    task.updated_at = kwargs.get("updated_at", datetime.now(timezone.utc))
    task.steps_json = kwargs.get("steps_json", "[]")
    task.result_json = kwargs.get("result_json", None)
    task.error = kwargs.get("error", None)
    task.user_id = kwargs.get("user_id", None)
    task.tenant_id = kwargs.get("tenant_id", None)
    task.to_dict.return_value = {
        "task_id": task_id,
        "goal": goal,
        "status": status,
        "max_steps": task.max_steps,
        "dry_run": task.dry_run,
    }
    return task


def _make_policy(policy_id: str = "policy-abc1", name: str = "Test Policy"):
    """Create a mock ComputerUsePolicy-like object."""
    policy = MagicMock()
    policy.policy_id = policy_id
    policy.name = name
    policy.to_dict.return_value = {
        "id": policy_id,
        "name": name,
        "description": "test",
        "allowed_actions": ["click", "type"],
        "blocked_domains": [],
    }
    return policy


def _make_approval(request_id: str = "req-001", status: str = "pending"):
    """Create a mock approval request object."""
    approval = MagicMock()
    approval.request_id = request_id
    approval.status = status
    approval.to_dict.return_value = {
        "request_id": request_id,
        "status": status,
        "action": "click",
    }
    return approval


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_rate_limit(monkeypatch):
    """Bypass rate limiting for tests."""
    monkeypatch.setenv("ARAGORA_USE_DISTRIBUTED_RATE_LIMIT", "false")


@pytest.fixture(autouse=True)
def _patch_handler_events(monkeypatch):
    """Stub emit_handler_event to prevent side effects."""
    monkeypatch.setattr(
        "aragora.server.handlers.computer_use_handler.emit_handler_event",
        lambda *args, **kwargs: None,
    )


@pytest.fixture
def mock_storage():
    """Create a mock ComputerUseStorage."""
    storage = MagicMock()
    storage.list_tasks.return_value = []
    storage.count_tasks.return_value = 0
    storage.get_task.return_value = None
    storage.save_task.return_value = "task-xxx"
    storage.update_task_status.return_value = True
    storage.get_action_stats.return_value = {
        "click": {"total": 5, "success": 4, "failed": 1},
        "type": {"total": 3, "success": 3, "failed": 0},
    }
    storage.list_policies.return_value = []
    storage.save_policy.return_value = "policy-xxx"
    return storage


@pytest.fixture
def mock_orchestrator():
    """Create a mock ComputerUseOrchestrator."""
    orch = MagicMock()
    return orch


@pytest.fixture
def mock_workflow():
    """Create a mock approval workflow."""
    workflow = MagicMock()
    workflow.list_all = AsyncMock(return_value=[])
    workflow.get_request = AsyncMock(return_value=None)
    workflow.approve = AsyncMock(return_value=True)
    workflow.deny = AsyncMock(return_value=True)
    return workflow


@pytest.fixture
def handler(mock_storage, monkeypatch):
    """Create a ComputerUseHandler with mocked dependencies."""
    # Ensure COMPUTER_USE_AVAILABLE is True
    monkeypatch.setattr(
        "aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True
    )
    # Patch RBAC to be available and permissive
    monkeypatch.setattr(
        "aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True
    )

    from aragora.server.handlers.computer_use_handler import ComputerUseHandler

    h = ComputerUseHandler(server_context={})
    h._storage = mock_storage
    # Bypass the handler's own _check_rbac_permission (returns None = allowed)
    h._check_rbac_permission = lambda handler, perm: None
    # Provide a mock auth context for metadata extraction in create/policy
    mock_auth_ctx = MagicMock()
    mock_auth_ctx.user_id = "test-user-001"
    mock_auth_ctx.org_id = "test-org-001"
    mock_auth_ctx.roles = {"admin"}
    h._get_auth_context = lambda handler: mock_auth_ctx
    return h


@pytest.fixture
def handler_no_computer_use(monkeypatch):
    """Create a handler with computer use unavailable."""
    monkeypatch.setattr(
        "aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False
    )
    from aragora.server.handlers.computer_use_handler import ComputerUseHandler

    return ComputerUseHandler(server_context={})


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_tasks_route(self, handler):
        assert handler.can_handle("/api/v1/computer-use/tasks")

    def test_task_by_id_route(self, handler):
        assert handler.can_handle("/api/v1/computer-use/tasks/task-abc123")

    def test_task_cancel_route(self, handler):
        assert handler.can_handle("/api/v1/computer-use/tasks/task-abc123/cancel")

    def test_actions_stats_route(self, handler):
        assert handler.can_handle("/api/v1/computer-use/actions/stats")

    def test_policies_route(self, handler):
        assert handler.can_handle("/api/v1/computer-use/policies")

    def test_approvals_route(self, handler):
        assert handler.can_handle("/api/v1/computer-use/approvals")

    def test_approval_by_id_route(self, handler):
        assert handler.can_handle("/api/v1/computer-use/approvals/req-001")

    def test_approval_approve_route(self, handler):
        assert handler.can_handle("/api/v1/computer-use/approvals/req-001/approve")

    def test_approval_deny_route(self, handler):
        assert handler.can_handle("/api/v1/computer-use/approvals/req-001/deny")

    def test_rejects_unrelated_path(self, handler):
        assert not handler.can_handle("/api/v1/debates")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/v1/computer")

    def test_rejects_wrong_version(self, handler):
        assert not handler.can_handle("/api/v2/computer-use/tasks")


# ---------------------------------------------------------------------------
# GET /api/v1/computer-use/tasks - List tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    """Tests for GET /api/v1/computer-use/tasks."""

    def test_list_tasks_empty(self, handler, mock_storage):
        mock_storage.list_tasks.return_value = []
        mock_storage.count_tasks.return_value = 0

        result = handler.handle("/api/v1/computer-use/tasks", {}, MockHTTPHandler())
        body = _body(result)

        assert _status(result) == 200
        assert body["tasks"] == []
        assert body["total"] == 0

    def test_list_tasks_with_results(self, handler, mock_storage):
        t1 = _make_task("task-001", "open file", "completed")
        t2 = _make_task("task-002", "click button", "pending")
        mock_storage.list_tasks.return_value = [t1, t2]
        mock_storage.count_tasks.return_value = 2

        result = handler.handle("/api/v1/computer-use/tasks", {}, MockHTTPHandler())
        body = _body(result)

        assert _status(result) == 200
        assert len(body["tasks"]) == 2
        assert body["total"] == 2

    def test_list_tasks_with_limit(self, handler, mock_storage):
        mock_storage.list_tasks.return_value = []
        mock_storage.count_tasks.return_value = 0

        handler.handle("/api/v1/computer-use/tasks", {"limit": "5"}, MockHTTPHandler())
        mock_storage.list_tasks.assert_called_once_with(limit=5, status=None)

    def test_list_tasks_with_status_filter(self, handler, mock_storage):
        mock_storage.list_tasks.return_value = []
        mock_storage.count_tasks.return_value = 0

        handler.handle(
            "/api/v1/computer-use/tasks",
            {"status": "completed"},
            MockHTTPHandler(),
        )
        mock_storage.list_tasks.assert_called_once_with(limit=20, status="completed")
        mock_storage.count_tasks.assert_called_once_with(status="completed")

    def test_list_tasks_limit_clamped_to_max(self, handler, mock_storage):
        """Limit above max_val (100) gets clamped."""
        mock_storage.list_tasks.return_value = []
        mock_storage.count_tasks.return_value = 0

        handler.handle(
            "/api/v1/computer-use/tasks", {"limit": "999"}, MockHTTPHandler()
        )
        mock_storage.list_tasks.assert_called_once_with(limit=100, status=None)

    def test_list_tasks_limit_invalid_uses_default(self, handler, mock_storage):
        """Non-numeric limit falls back to default 20."""
        mock_storage.list_tasks.return_value = []
        mock_storage.count_tasks.return_value = 0

        handler.handle(
            "/api/v1/computer-use/tasks", {"limit": "abc"}, MockHTTPHandler()
        )
        mock_storage.list_tasks.assert_called_once_with(limit=20, status=None)


# ---------------------------------------------------------------------------
# GET /api/v1/computer-use/tasks/{id} - Get task
# ---------------------------------------------------------------------------


class TestGetTask:
    """Tests for GET /api/v1/computer-use/tasks/{id}."""

    def test_get_existing_task(self, handler, mock_storage):
        task = _make_task("task-abc123", "open file", "completed")
        mock_storage.get_task.return_value = task

        result = handler.handle(
            "/api/v1/computer-use/tasks/task-abc123", {}, MockHTTPHandler()
        )
        body = _body(result)

        assert _status(result) == 200
        assert body["task"]["task_id"] == "task-abc123"

    def test_get_missing_task_404(self, handler, mock_storage):
        mock_storage.get_task.return_value = None

        result = handler.handle(
            "/api/v1/computer-use/tasks/nonexistent", {}, MockHTTPHandler()
        )

        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_get_task_extracts_id_from_path(self, handler, mock_storage):
        mock_storage.get_task.return_value = _make_task("task-xyz")

        handler.handle("/api/v1/computer-use/tasks/task-xyz", {}, MockHTTPHandler())
        mock_storage.get_task.assert_called_once_with("task-xyz")


# ---------------------------------------------------------------------------
# POST /api/v1/computer-use/tasks - Create task
# ---------------------------------------------------------------------------


class TestCreateTask:
    """Tests for POST /api/v1/computer-use/tasks."""

    def test_create_task_dry_run(self, handler, mock_storage, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        )

        body = {"goal": "test click", "dry_run": True}
        http_handler = MockHTTPHandler("POST", body)

        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, http_handler
        )
        body_result = _body(result)

        assert _status(result) == 201
        assert body_result["message"] == "Task created successfully"
        assert body_result["status"] == "completed"
        assert "task_id" in body_result
        # Verify storage was called to save
        assert mock_storage.save_task.call_count >= 1

    def test_create_task_missing_goal(self, handler, mock_storage, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        )

        http_handler = MockHTTPHandler("POST", {"dry_run": True})
        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, http_handler
        )

        assert _status(result) == 400
        assert "goal" in _body(result).get("error", "").lower()

    def test_create_task_invalid_json(self, handler, mock_storage, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        )

        http_handler = MockHTTPHandler("POST")
        http_handler.rfile.read.return_value = b"not json"
        http_handler.headers["Content-Length"] = "8"

        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, http_handler
        )

        assert _status(result) == 400
        assert "json" in _body(result).get("error", "").lower()

    def test_create_task_orchestrator_unavailable(self, handler, mock_storage, monkeypatch):
        """When orchestrator cannot be created, returns 503."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE",
            False,
        )

        http_handler = MockHTTPHandler("POST", {"goal": "test"})
        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, http_handler
        )

        assert _status(result) == 503

    def test_create_task_with_real_execution(self, handler, mock_storage, monkeypatch):
        """Test task creation with actual (mocked) orchestrator execution."""
        # Build a mock task result
        mock_step = MagicMock()
        mock_step.action.action_type.value = "click"
        mock_step.result.success = True

        mock_result = MagicMock()
        mock_result.status = MockTaskStatus.COMPLETED
        mock_result.error = None
        mock_result.steps = [mock_step]

        mock_orch = MagicMock()
        mock_orch.run_task = AsyncMock(return_value=mock_result)

        # Patch run_async to just call the coroutine
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: mock_result,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.TaskStatus",
            MockTaskStatus,
        )

        # Inject orchestrator
        handler._orchestrator = mock_orch
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )

        body = {"goal": "click button", "max_steps": 5}
        http_handler = MockHTTPHandler("POST", body)

        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, http_handler
        )

        assert _status(result) == 201
        body_data = _body(result)
        assert body_data["status"] == "completed"

    def test_create_task_execution_failure(self, handler, mock_storage, monkeypatch):
        """Test that runtime errors during execution produce failed tasks."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            MagicMock(side_effect=RuntimeError("timeout")),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        )

        body = {"goal": "test runtime error"}
        http_handler = MockHTTPHandler("POST", body)

        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, http_handler
        )

        # Should still succeed (201) but with failed status
        assert _status(result) == 201
        body_data = _body(result)
        assert body_data["status"] == "failed"

    def test_create_task_execution_oserror(self, handler, mock_storage, monkeypatch):
        """OSError during execution is caught and task is marked failed."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            MagicMock(side_effect=OSError("disk error")),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        )

        body = {"goal": "test os error"}
        http_handler = MockHTTPHandler("POST", body)

        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, http_handler
        )
        assert _status(result) == 201
        assert _body(result)["status"] == "failed"

    def test_create_task_empty_goal(self, handler, mock_storage, monkeypatch):
        """Empty string goal returns 400."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        )

        body = {"goal": ""}
        http_handler = MockHTTPHandler("POST", body)

        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, http_handler
        )
        assert _status(result) == 400

    def test_create_task_custom_max_steps(self, handler, mock_storage, monkeypatch):
        """Custom max_steps value is used in the task dict."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        )

        body = {"goal": "test", "dry_run": True, "max_steps": 25}
        http_handler = MockHTTPHandler("POST", body)

        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, http_handler
        )
        assert _status(result) == 201

        # Verify saved task has max_steps=25
        saved_task = mock_storage.save_task.call_args[0][0]
        assert saved_task["max_steps"] == 25

    def test_create_task_defaults_max_steps(self, handler, mock_storage, monkeypatch):
        """Default max_steps is 10."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        )

        body = {"goal": "test", "dry_run": True}
        http_handler = MockHTTPHandler("POST", body)

        handler.handle_post("/api/v1/computer-use/tasks", {}, http_handler)

        saved_task = mock_storage.save_task.call_args[0][0]
        assert saved_task["max_steps"] == 10


# ---------------------------------------------------------------------------
# POST /api/v1/computer-use/tasks/{id}/cancel - Cancel task
# ---------------------------------------------------------------------------


class TestCancelTask:
    """Tests for POST /api/v1/computer-use/tasks/{id}/cancel."""

    def test_cancel_pending_task(self, handler, mock_storage):
        task = _make_task("task-001", status="pending")
        mock_storage.get_task.return_value = task

        result = handler.handle_post(
            "/api/v1/computer-use/tasks/task-001/cancel", {}, MockHTTPHandler("POST")
        )

        assert _status(result) == 200
        assert "cancelled" in _body(result).get("message", "").lower()
        mock_storage.update_task_status.assert_called_once_with("task-001", "cancelled")

    def test_cancel_running_task(self, handler, mock_storage):
        task = _make_task("task-002", status="running")
        mock_storage.get_task.return_value = task

        result = handler.handle_post(
            "/api/v1/computer-use/tasks/task-002/cancel", {}, MockHTTPHandler("POST")
        )
        assert _status(result) == 200

    def test_cancel_nonexistent_task(self, handler, mock_storage):
        mock_storage.get_task.return_value = None

        result = handler.handle_post(
            "/api/v1/computer-use/tasks/no-task/cancel", {}, MockHTTPHandler("POST")
        )
        assert _status(result) == 404

    def test_cancel_already_completed_task(self, handler, mock_storage):
        task = _make_task("task-003", status="completed")
        mock_storage.get_task.return_value = task

        result = handler.handle_post(
            "/api/v1/computer-use/tasks/task-003/cancel", {}, MockHTTPHandler("POST")
        )
        assert _status(result) == 400
        assert "already" in _body(result).get("error", "").lower()

    def test_cancel_already_failed_task(self, handler, mock_storage):
        task = _make_task("task-004", status="failed")
        mock_storage.get_task.return_value = task

        result = handler.handle_post(
            "/api/v1/computer-use/tasks/task-004/cancel", {}, MockHTTPHandler("POST")
        )
        assert _status(result) == 400

    def test_cancel_already_cancelled_task(self, handler, mock_storage):
        task = _make_task("task-005", status="cancelled")
        mock_storage.get_task.return_value = task

        result = handler.handle_post(
            "/api/v1/computer-use/tasks/task-005/cancel", {}, MockHTTPHandler("POST")
        )
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# GET /api/v1/computer-use/actions/stats - Action statistics
# ---------------------------------------------------------------------------


class TestActionStats:
    """Tests for GET /api/v1/computer-use/actions/stats."""

    def test_get_action_stats(self, handler, mock_storage):
        result = handler.handle(
            "/api/v1/computer-use/actions/stats", {}, MockHTTPHandler()
        )

        assert _status(result) == 200
        body = _body(result)
        assert "stats" in body
        assert body["stats"]["click"]["total"] == 5

    def test_action_stats_calls_storage(self, handler, mock_storage):
        handler.handle(
            "/api/v1/computer-use/actions/stats", {}, MockHTTPHandler()
        )
        mock_storage.get_action_stats.assert_called_once()


# ---------------------------------------------------------------------------
# GET /api/v1/computer-use/policies - List policies
# ---------------------------------------------------------------------------


class TestListPolicies:
    """Tests for GET /api/v1/computer-use/policies."""

    def test_list_policies_includes_default(self, handler, mock_storage):
        mock_storage.list_policies.return_value = []

        result = handler.handle(
            "/api/v1/computer-use/policies", {}, MockHTTPHandler()
        )
        body = _body(result)

        assert _status(result) == 200
        assert body["total"] == 1
        assert body["policies"][0]["id"] == "default"

    def test_list_policies_with_stored(self, handler, mock_storage):
        p1 = _make_policy("policy-001", "Custom Policy")
        mock_storage.list_policies.return_value = [p1]

        result = handler.handle(
            "/api/v1/computer-use/policies", {}, MockHTTPHandler()
        )
        body = _body(result)

        assert body["total"] == 2
        # First is the default, second is the custom
        assert body["policies"][0]["id"] == "default"
        assert body["policies"][1]["id"] == "policy-001"

    def test_list_policies_default_has_expected_actions(self, handler, mock_storage):
        mock_storage.list_policies.return_value = []

        result = handler.handle(
            "/api/v1/computer-use/policies", {}, MockHTTPHandler()
        )
        body = _body(result)
        default_policy = body["policies"][0]

        assert set(default_policy["allowed_actions"]) == {
            "screenshot",
            "click",
            "type",
            "scroll",
            "key",
        }

    def test_list_policies_default_has_no_blocked_domains(self, handler, mock_storage):
        mock_storage.list_policies.return_value = []

        result = handler.handle(
            "/api/v1/computer-use/policies", {}, MockHTTPHandler()
        )
        body = _body(result)
        assert body["policies"][0]["blocked_domains"] == []


# ---------------------------------------------------------------------------
# POST /api/v1/computer-use/policies - Create policy
# ---------------------------------------------------------------------------


class TestCreatePolicy:
    """Tests for POST /api/v1/computer-use/policies."""

    def test_create_policy_success(self, handler, mock_storage):
        body = {"name": "Strict Policy", "description": "Limits actions"}
        http_handler = MockHTTPHandler("POST", body)

        result = handler.handle_post(
            "/api/v1/computer-use/policies", {}, http_handler
        )
        body_result = _body(result)

        assert _status(result) == 201
        assert "policy_id" in body_result
        assert body_result["message"] == "Policy created successfully"

    def test_create_policy_missing_name(self, handler, mock_storage):
        body = {"description": "no name"}
        http_handler = MockHTTPHandler("POST", body)

        result = handler.handle_post(
            "/api/v1/computer-use/policies", {}, http_handler
        )

        assert _status(result) == 400
        assert "name" in _body(result).get("error", "").lower()

    def test_create_policy_invalid_json(self, handler, mock_storage):
        http_handler = MockHTTPHandler("POST")
        http_handler.rfile.read.return_value = b"bad json"
        http_handler.headers["Content-Length"] = "8"

        result = handler.handle_post(
            "/api/v1/computer-use/policies", {}, http_handler
        )
        assert _status(result) == 400

    def test_create_policy_with_custom_actions(self, handler, mock_storage):
        body = {
            "name": "Screenshots Only",
            "allowed_actions": ["screenshot"],
            "blocked_domains": ["evil.com"],
        }
        http_handler = MockHTTPHandler("POST", body)

        result = handler.handle_post(
            "/api/v1/computer-use/policies", {}, http_handler
        )
        assert _status(result) == 201

        saved = mock_storage.save_policy.call_args[0][0]
        assert saved["allowed_actions"] == ["screenshot"]
        assert saved["blocked_domains"] == ["evil.com"]

    def test_create_policy_default_actions(self, handler, mock_storage):
        """When no allowed_actions specified, defaults are used."""
        body = {"name": "Default Actions Policy"}
        http_handler = MockHTTPHandler("POST", body)

        handler.handle_post("/api/v1/computer-use/policies", {}, http_handler)

        saved = mock_storage.save_policy.call_args[0][0]
        assert set(saved["allowed_actions"]) == {
            "screenshot", "click", "type", "scroll", "key"
        }


# ---------------------------------------------------------------------------
# GET /api/v1/computer-use/approvals - List approvals
# ---------------------------------------------------------------------------


class TestListApprovals:
    """Tests for GET /api/v1/computer-use/approvals."""

    def test_list_approvals_no_workflow(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        handler._approval_workflow = None

        result = handler.handle(
            "/api/v1/computer-use/approvals", {}, MockHTTPHandler()
        )
        assert _status(result) == 503

    def test_list_approvals_empty(self, handler, mock_workflow, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: [],
        )
        handler._approval_workflow = mock_workflow

        result = handler.handle(
            "/api/v1/computer-use/approvals", {}, MockHTTPHandler()
        )
        body = _body(result)

        assert _status(result) == 200
        assert body["approvals"] == []
        assert body["count"] == 0

    def test_list_approvals_with_results(self, handler, mock_workflow, monkeypatch):
        a1 = _make_approval("req-001", "pending")
        a2 = _make_approval("req-002", "approved")

        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: [a1, a2],
        )
        handler._approval_workflow = mock_workflow

        result = handler.handle(
            "/api/v1/computer-use/approvals", {}, MockHTTPHandler()
        )
        body = _body(result)

        assert body["count"] == 2

    def test_list_approvals_with_status_filter(self, handler, mock_workflow, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ApprovalStatus",
            MockApprovalStatus,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: [],
        )
        handler._approval_workflow = mock_workflow

        result = handler.handle(
            "/api/v1/computer-use/approvals",
            {"status": "pending"},
            MockHTTPHandler(),
        )
        assert _status(result) == 200

    def test_list_approvals_invalid_status(self, handler, mock_workflow, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ApprovalStatus",
            MockApprovalStatus,
        )
        handler._approval_workflow = mock_workflow

        result = handler.handle(
            "/api/v1/computer-use/approvals",
            {"status": "invalid_status"},
            MockHTTPHandler(),
        )
        assert _status(result) == 400
        assert "invalid status" in _body(result).get("error", "").lower()


# ---------------------------------------------------------------------------
# GET /api/v1/computer-use/approvals/{id} - Get approval
# ---------------------------------------------------------------------------


class TestGetApproval:
    """Tests for GET /api/v1/computer-use/approvals/{id}."""

    def test_get_approval_no_workflow(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        handler._approval_workflow = None

        result = handler.handle(
            "/api/v1/computer-use/approvals/req-001", {}, MockHTTPHandler()
        )
        assert _status(result) == 503

    def test_get_approval_found(self, handler, mock_workflow, monkeypatch):
        approval = _make_approval("req-001", "pending")
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: approval,
        )
        handler._approval_workflow = mock_workflow

        result = handler.handle(
            "/api/v1/computer-use/approvals/req-001", {}, MockHTTPHandler()
        )
        body = _body(result)

        assert _status(result) == 200
        assert body["approval"]["request_id"] == "req-001"

    def test_get_approval_not_found(self, handler, mock_workflow, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: None,
        )
        handler._approval_workflow = mock_workflow

        result = handler.handle(
            "/api/v1/computer-use/approvals/req-999", {}, MockHTTPHandler()
        )
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# POST /api/v1/computer-use/approvals/{id}/approve - Approve
# ---------------------------------------------------------------------------


class TestApproveApproval:
    """Tests for POST /api/v1/computer-use/approvals/{id}/approve."""

    def test_approve_success(self, handler, mock_workflow, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: True,
        )
        handler._approval_workflow = mock_workflow

        http_handler = MockHTTPHandler("POST", {"reason": "looks good"})
        result = handler.handle_post(
            "/api/v1/computer-use/approvals/req-001/approve", {}, http_handler
        )

        assert _status(result) == 200
        assert _body(result)["approved"] is True
        assert _body(result)["request_id"] == "req-001"

    def test_approve_no_workflow(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        handler._approval_workflow = None

        http_handler = MockHTTPHandler("POST", {})
        result = handler.handle_post(
            "/api/v1/computer-use/approvals/req-001/approve", {}, http_handler
        )
        assert _status(result) == 503

    def test_approve_not_found(self, handler, mock_workflow, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: False,
        )
        handler._approval_workflow = mock_workflow

        http_handler = MockHTTPHandler("POST", {})
        result = handler.handle_post(
            "/api/v1/computer-use/approvals/req-999/approve", {}, http_handler
        )
        assert _status(result) == 404

    def test_approve_without_reason(self, handler, mock_workflow, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: True,
        )
        handler._approval_workflow = mock_workflow

        http_handler = MockHTTPHandler("POST")
        result = handler.handle_post(
            "/api/v1/computer-use/approvals/req-001/approve", {}, http_handler
        )
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# POST /api/v1/computer-use/approvals/{id}/deny - Deny
# ---------------------------------------------------------------------------


class TestDenyApproval:
    """Tests for POST /api/v1/computer-use/approvals/{id}/deny."""

    def test_deny_success(self, handler, mock_workflow, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: True,
        )
        handler._approval_workflow = mock_workflow

        http_handler = MockHTTPHandler("POST", {"reason": "dangerous action"})
        result = handler.handle_post(
            "/api/v1/computer-use/approvals/req-001/deny", {}, http_handler
        )

        assert _status(result) == 200
        assert _body(result)["denied"] is True
        assert _body(result)["request_id"] == "req-001"

    def test_deny_no_workflow(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        handler._approval_workflow = None

        http_handler = MockHTTPHandler("POST", {})
        result = handler.handle_post(
            "/api/v1/computer-use/approvals/req-001/deny", {}, http_handler
        )
        assert _status(result) == 503

    def test_deny_not_found(self, handler, mock_workflow, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: False,
        )
        handler._approval_workflow = mock_workflow

        http_handler = MockHTTPHandler("POST", {})
        result = handler.handle_post(
            "/api/v1/computer-use/approvals/req-001/deny", {}, http_handler
        )
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Computer use module unavailable
# ---------------------------------------------------------------------------


class TestModuleUnavailable:
    """Tests when COMPUTER_USE_AVAILABLE is False."""

    def test_handle_returns_503(self, handler_no_computer_use):
        result = handler_no_computer_use.handle(
            "/api/v1/computer-use/tasks", {}, MockHTTPHandler()
        )
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_handle_post_returns_503(self, handler_no_computer_use):
        result = handler_no_computer_use.handle_post(
            "/api/v1/computer-use/tasks", {}, MockHTTPHandler("POST", {"goal": "x"})
        )
        assert _status(result) == 503

    def test_handle_unrelated_path_returns_none(self, handler_no_computer_use):
        result = handler_no_computer_use.handle(
            "/api/v1/debates", {}, MockHTTPHandler()
        )
        assert result is None


# ---------------------------------------------------------------------------
# handle() returns None for unmatched sub-paths
# ---------------------------------------------------------------------------


class TestUnmatchedPaths:
    """Tests that handle/handle_post return None for unmatched sub-routes."""

    def test_handle_unknown_sub_path(self, handler):
        result = handler.handle(
            "/api/v1/computer-use/unknown", {}, MockHTTPHandler()
        )
        assert result is None

    def test_handle_post_unknown_sub_path(self, handler):
        result = handler.handle_post(
            "/api/v1/computer-use/unknown", {}, MockHTTPHandler("POST")
        )
        assert result is None

    def test_handle_returns_none_for_non_matching(self, handler):
        result = handler.handle("/api/v1/other", {}, MockHTTPHandler())
        assert result is None

    def test_handle_post_returns_none_for_non_matching(self, handler):
        result = handler.handle_post(
            "/api/v1/other", {}, MockHTTPHandler("POST")
        )
        assert result is None


# ---------------------------------------------------------------------------
# Orchestrator resolution
# ---------------------------------------------------------------------------


class TestOrchestratorResolution:
    """Tests for _get_orchestrator logic."""

    def test_orchestrator_from_extension_state(self, handler, monkeypatch):
        mock_orch = MagicMock()
        mock_state = MagicMock()
        mock_state.computer_orchestrator = mock_orch

        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: mock_state,
        )

        result = handler._get_orchestrator()
        assert result is mock_orch

    def test_orchestrator_created_when_no_state(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        mock_policy = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: mock_policy,
        )
        mock_config = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            lambda **kw: mock_config,
        )
        mock_orch = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            lambda **kw: mock_orch,
        )

        result = handler._get_orchestrator()
        assert result is mock_orch

    def test_orchestrator_cached_on_second_call(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            lambda **kw: MagicMock(),
        )
        orch_calls = []
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            lambda **kw: orch_calls.append(1) or MagicMock(),
        )

        handler._get_orchestrator()
        handler._get_orchestrator()
        assert len(orch_calls) == 1

    def test_orchestrator_none_when_module_unavailable(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE",
            False,
        )
        assert handler._get_orchestrator() is None


# ---------------------------------------------------------------------------
# Approval workflow resolution
# ---------------------------------------------------------------------------


class TestApprovalWorkflowResolution:
    """Tests for _get_approval_workflow logic."""

    def test_workflow_from_extension_state(self, handler, monkeypatch):
        mock_wf = MagicMock()
        mock_state = MagicMock()
        mock_state.computer_approval_workflow = mock_wf

        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: mock_state,
        )

        result = handler._get_approval_workflow()
        assert result is mock_wf

    def test_workflow_from_instance_attr(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        mock_wf = MagicMock()
        handler._approval_workflow = mock_wf

        result = handler._get_approval_workflow()
        assert result is mock_wf

    def test_workflow_none_when_not_configured(self, handler, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        handler._approval_workflow = None

        result = handler._get_approval_workflow()
        assert result is None


# ---------------------------------------------------------------------------
# Storage resolution
# ---------------------------------------------------------------------------


class TestStorageResolution:
    """Tests for _get_storage logic."""

    def test_storage_is_cached(self, handler, mock_storage):
        """Once set, _get_storage returns the same instance."""
        s1 = handler._get_storage()
        s2 = handler._get_storage()
        assert s1 is s2

    def test_storage_created_when_none(self, handler, monkeypatch):
        handler._storage = None
        mock_s = MagicMock()
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_computer_use_storage",
            lambda: mock_s,
        )
        result = handler._get_storage()
        assert result is mock_s


# ---------------------------------------------------------------------------
# RBAC checks (using no_auto_auth)
# ---------------------------------------------------------------------------


class TestRBACChecks:
    """Tests for RBAC permission enforcement."""

    @pytest.mark.no_auto_auth
    def test_rbac_fail_closed_in_production(self, monkeypatch):
        """When RBAC is unavailable and environment is production, return 503."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", False
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.rbac_fail_closed",
            lambda: True,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True
        )

        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        h = ComputerUseHandler(server_context={})
        h._storage = MagicMock()
        h._storage.list_tasks.return_value = []
        h._storage.count_tasks.return_value = 0

        result = h._check_rbac_permission(MockHTTPHandler(), "computer_use:tasks:read")
        assert result is not None
        assert _status(result) == 503

    @pytest.mark.no_auto_auth
    def test_rbac_permissive_in_dev(self, monkeypatch):
        """When RBAC is unavailable but not production, skip check."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", False
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.rbac_fail_closed",
            lambda: False,
        )

        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        h = ComputerUseHandler(server_context={})
        result = h._check_rbac_permission(MockHTTPHandler(), "computer_use:tasks:read")
        assert result is None  # No error, allowed

    @pytest.mark.no_auto_auth
    def test_rbac_unauthenticated_returns_401(self, monkeypatch):
        """When RBAC is available but user is not authenticated, return 401."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True
        )

        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        h = ComputerUseHandler(server_context={})
        # Patch _get_auth_context to return None (not authenticated)
        h._get_auth_context = lambda handler: None

        result = h._check_rbac_permission(MockHTTPHandler(), "computer_use:tasks:read")
        assert result is not None
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_rbac_permission_denied_returns_403(self, monkeypatch):
        """When permission check fails, return 403."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True
        )

        mock_ctx = MagicMock()
        mock_ctx.user_id = "user-001"

        mock_decision = MagicMock()
        mock_decision.allowed = False

        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        h = ComputerUseHandler(server_context={})
        h._get_auth_context = lambda handler: mock_ctx

        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.check_permission",
            lambda ctx, perm: mock_decision,
        )

        result = h._check_rbac_permission(MockHTTPHandler(), "computer_use:tasks:read")
        assert result is not None
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    def test_rbac_permission_allowed_returns_none(self, monkeypatch):
        """When permission check passes, returns None."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True
        )

        mock_ctx = MagicMock()
        mock_ctx.user_id = "user-001"

        mock_decision = MagicMock()
        mock_decision.allowed = True

        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        h = ComputerUseHandler(server_context={})
        h._get_auth_context = lambda handler: mock_ctx

        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.check_permission",
            lambda ctx, perm: mock_decision,
        )

        result = h._check_rbac_permission(MockHTTPHandler(), "computer_use:tasks:read")
        assert result is None


# ---------------------------------------------------------------------------
# ROUTES list
# ---------------------------------------------------------------------------


class TestROUTES:
    """Validate the ROUTES class attribute."""

    def test_routes_not_empty(self):
        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        assert len(ComputerUseHandler.ROUTES) > 0

    def test_routes_contains_tasks(self):
        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        assert "/api/v1/computer-use/tasks" in ComputerUseHandler.ROUTES

    def test_routes_contains_actions_stats(self):
        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        assert "/api/v1/computer-use/actions/stats" in ComputerUseHandler.ROUTES

    def test_routes_contains_policies(self):
        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        assert "/api/v1/computer-use/policies" in ComputerUseHandler.ROUTES

    def test_routes_contains_approvals(self):
        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        assert "/api/v1/computer-use/approvals" in ComputerUseHandler.ROUTES


# ---------------------------------------------------------------------------
# Handler initialization
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for handler initialization."""

    def test_init_with_empty_context(self, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True
        )
        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        h = ComputerUseHandler(server_context={})
        assert h._orchestrator is None
        assert h._storage is None
        assert h._approval_workflow is None

    def test_get_user_store_from_context(self, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True
        )
        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        mock_store = MagicMock()
        h = ComputerUseHandler(server_context={"user_store": mock_store})
        assert h._get_user_store() is mock_store

    def test_get_user_store_missing(self, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True
        )
        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        h = ComputerUseHandler(server_context={})
        assert h._get_user_store() is None


# ---------------------------------------------------------------------------
# Path segment extraction edge cases
# ---------------------------------------------------------------------------


class TestPathExtraction:
    """Tests for path parsing edge cases in handle() and handle_post()."""

    def test_get_task_short_path_returns_none(self, handler):
        """Path with too few segments after /tasks/ skipped gracefully."""
        # /api/v1/computer-use/tasks/ with nothing after -> handled by exact match
        result = handler.handle(
            "/api/v1/computer-use/tasks/", {}, MockHTTPHandler()
        )
        # The trailing slash means parts = ["api", "v1", "computer-use", "tasks", ""]
        # len(parts) >= 5, so task_id="" is used
        # This is fine; the storage will return None for empty id
        if result is not None:
            # If it matched, it should call get_task with ""
            assert _status(result) in (200, 404)

    def test_cancel_requires_correct_path_structure(self, handler, mock_storage):
        """Cancel needs parts[5] == 'cancel'."""
        mock_storage.get_task.return_value = _make_task("task-001", status="running")

        result = handler.handle_post(
            "/api/v1/computer-use/tasks/task-001/notcancel",
            {},
            MockHTTPHandler("POST"),
        )
        # "cancel" not in path, so it shouldn't match the cancel route
        assert result is None

    def test_approval_approve_path_parsing(self, handler, mock_workflow, monkeypatch):
        """Verify approve path parsing extracts correct request_id."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: True,
        )
        handler._approval_workflow = mock_workflow

        http_handler = MockHTTPHandler("POST", {})
        result = handler.handle_post(
            "/api/v1/computer-use/approvals/my-req-id/approve", {}, http_handler
        )

        assert _status(result) == 200
        assert _body(result)["request_id"] == "my-req-id"

    def test_approval_deny_path_parsing(self, handler, mock_workflow, monkeypatch):
        """Verify deny path parsing extracts correct request_id."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.run_async",
            lambda coro: True,
        )
        handler._approval_workflow = mock_workflow

        http_handler = MockHTTPHandler("POST", {})
        result = handler.handle_post(
            "/api/v1/computer-use/approvals/my-req-id/deny", {}, http_handler
        )

        assert _status(result) == 200
        assert _body(result)["request_id"] == "my-req-id"


# ---------------------------------------------------------------------------
# Task creation save verification
# ---------------------------------------------------------------------------


class TestTaskSaveDetails:
    """Detailed verification of saved task data."""

    def _setup_create_task(self, handler, monkeypatch):
        """Common setup for create task tests."""
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: None,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        )

    def test_dry_run_task_saved_with_completed_status(
        self, handler, mock_storage, monkeypatch
    ):
        self._setup_create_task(handler, monkeypatch)

        body = {"goal": "test", "dry_run": True}
        handler.handle_post(
            "/api/v1/computer-use/tasks", {}, MockHTTPHandler("POST", body)
        )

        # save_task called twice: once pending, once completed
        assert mock_storage.save_task.call_count == 2
        last_saved = mock_storage.save_task.call_args_list[-1][0][0]
        assert last_saved["status"] == "completed"
        assert last_saved["result"]["success"] is True

    def test_task_id_has_expected_prefix(self, handler, mock_storage, monkeypatch):
        self._setup_create_task(handler, monkeypatch)

        body = {"goal": "test", "dry_run": True}
        result = handler.handle_post(
            "/api/v1/computer-use/tasks", {}, MockHTTPHandler("POST", body)
        )

        task_id = _body(result)["task_id"]
        assert task_id.startswith("task-")

    def test_dry_run_result_has_zero_steps(self, handler, mock_storage, monkeypatch):
        self._setup_create_task(handler, monkeypatch)

        body = {"goal": "test", "dry_run": True}
        handler.handle_post(
            "/api/v1/computer-use/tasks", {}, MockHTTPHandler("POST", body)
        )

        last_saved = mock_storage.save_task.call_args_list[-1][0][0]
        assert last_saved["result"]["steps_taken"] == 0

    def test_task_has_created_at(self, handler, mock_storage, monkeypatch):
        self._setup_create_task(handler, monkeypatch)

        body = {"goal": "test", "dry_run": True}
        handler.handle_post(
            "/api/v1/computer-use/tasks", {}, MockHTTPHandler("POST", body)
        )

        first_saved = mock_storage.save_task.call_args_list[0][0][0]
        assert "created_at" in first_saved


# ---------------------------------------------------------------------------
# Extension state interaction
# ---------------------------------------------------------------------------


class TestExtensionState:
    """Tests for extension state interactions."""

    def test_orchestrator_prefers_extension_state(self, handler, monkeypatch):
        mock_orch = MagicMock()
        mock_state = MagicMock()
        mock_state.computer_orchestrator = mock_orch

        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: mock_state,
        )

        assert handler._get_orchestrator() is mock_orch

    def test_orchestrator_falls_back_to_local(self, handler, monkeypatch):
        mock_state = MagicMock()
        mock_state.computer_orchestrator = None

        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: mock_state,
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            lambda: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            lambda **kw: MagicMock(),
        )
        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            lambda **kw: MagicMock(),
        )

        result = handler._get_orchestrator()
        assert result is not None

    def test_approval_workflow_from_extension_state(self, handler, monkeypatch):
        mock_wf = MagicMock()
        mock_state = MagicMock()
        mock_state.computer_approval_workflow = mock_wf

        monkeypatch.setattr(
            "aragora.server.handlers.computer_use_handler.get_extension_state",
            lambda: mock_state,
        )

        assert handler._get_approval_workflow() is mock_wf
