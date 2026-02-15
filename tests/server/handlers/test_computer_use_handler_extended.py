"""
Extended tests for ComputerUseHandler - comprehensive coverage of all functionality.

Tests cover:
- ComputerUseHandler initialization and orchestrator creation
- GET /api/v1/computer-use/tasks (list tasks with various filters and limits)
- GET /api/v1/computer-use/tasks/{id} (get single task, not found)
- POST /api/v1/computer-use/tasks (create task, dry run, missing goal, invalid JSON,
  execution success/failure paths)
- POST /api/v1/computer-use/tasks/{id}/cancel (cancel running/completed/failed/cancelled)
- GET /api/v1/computer-use/actions/stats (statistics aggregation from task steps)
- GET /api/v1/computer-use/policies (list policies, with custom policies)
- POST /api/v1/computer-use/policies (create policy, missing name, invalid JSON)
- RBAC permission checks for all endpoints (401, 403)
- can_handle() method with various paths
- Rate limiting on create/policy endpoints
- Computer use module unavailable (503) scenarios for GET and POST
- Task execution success and failure paths (via run_async)
- Edge cases: empty task stores, unknown paths, malformed paths
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.base import HandlerResult, error_response, json_response


# ===========================================================================
# Test Mocks and Fixtures
# ===========================================================================


class MockRequestHandler:
    """Mock HTTP request handler for tests."""

    def __init__(self, body: dict | None = None, headers: dict | None = None):
        self._body = body
        self.headers = headers or {}
        self.client_address = ("127.0.0.1", 12345)

    def read_body(self):
        return json.dumps(self._body).encode() if self._body else b"{}"


@dataclass
class MockTaskStatus:
    """Mock TaskStatus enum values."""

    COMPLETED = "completed"
    FAILED = "failed"
    RUNNING = "running"


@dataclass
class MockActionType:
    """Mock action type for testing."""

    value: str = "click"


@dataclass
class MockAction:
    """Mock action for testing."""

    action_type: MockActionType = field(default_factory=MockActionType)


@dataclass
class MockStepResult:
    """Mock step result for testing."""

    success: bool = True


@dataclass
class MockStep:
    """Mock step for testing."""

    action: MockAction = field(default_factory=MockAction)
    result: MockStepResult = field(default_factory=MockStepResult)


@dataclass
class MockTaskResult:
    """Mock task result for testing."""

    status: str = "completed"
    error: str | None = None
    steps: list = field(default_factory=list)


class MockComputerPolicy:
    """Mock computer policy for testing."""

    name: str = "Test Policy"
    description: str = "A test policy"
    allowed_actions: list = ["screenshot", "click", "type"]


@pytest.fixture(autouse=True)
def clear_rate_limiters():
    """Clear all rate limiters between tests to avoid interference."""
    try:
        from aragora.server.handlers.utils.rate_limit import clear_all_limiters

        clear_all_limiters()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.handlers.utils.rate_limit import clear_all_limiters

        clear_all_limiters()
    except ImportError:
        pass


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    ctx = MagicMock()
    ctx.get = MagicMock(return_value=None)
    return ctx


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_computer_use.db")


@pytest.fixture
def handler(mock_server_context, temp_db_path):
    """Create handler with COMPUTER_USE_AVAILABLE=True and mocked dependencies."""
    from aragora.computer_use.storage import ComputerUseStorage

    with (
        patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
        patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", False),
        patch(
            "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
            return_value=MockComputerPolicy(),
        ),
        patch(
            "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
            MagicMock(),
        ),
        patch(
            "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
            MagicMock(),
        ),
    ):
        from aragora.server.handlers.computer_use_handler import ComputerUseHandler

        h = ComputerUseHandler(mock_server_context)
        h._orchestrator = MagicMock()
        # Use temp database for storage
        h._storage = ComputerUseStorage(db_path=temp_db_path, backend="sqlite")
        return h


@pytest.fixture
def handler_with_tasks(handler):
    """Create handler pre-seeded with tasks."""
    storage = handler._get_storage()
    storage.save_task(
        {
            "task_id": "task-aaa",
            "goal": "Open browser",
            "max_steps": 10,
            "dry_run": False,
            "status": "completed",
            "created_at": "2025-01-29T10:00:00Z",
            "steps": [
                {"action": "click", "success": True},
                {"action": "type", "success": True},
                {"action": "screenshot", "success": False},
            ],
            "result": {"success": True, "message": "Done", "steps_taken": 3},
        }
    )
    storage.save_task(
        {
            "task_id": "task-bbb",
            "goal": "Navigate to settings",
            "max_steps": 5,
            "dry_run": False,
            "status": "running",
            "created_at": "2025-01-29T11:00:00Z",
            "steps": [],
            "result": None,
        }
    )
    storage.save_task(
        {
            "task_id": "task-ccc",
            "goal": "Click button",
            "max_steps": 3,
            "dry_run": False,
            "status": "failed",
            "created_at": "2025-01-29T09:00:00Z",
            "steps": [{"action": "click", "success": False}],
            "result": {"success": False, "message": "Element not found", "steps_taken": 1},
        }
    )
    storage.save_task(
        {
            "task_id": "task-ddd",
            "goal": "Scroll page",
            "max_steps": 2,
            "dry_run": False,
            "status": "pending",
            "created_at": "2025-01-29T12:00:00Z",
            "steps": [],
            "result": None,
        }
    )
    storage.save_task(
        {
            "task_id": "task-eee",
            "goal": "Already cancelled",
            "max_steps": 2,
            "dry_run": False,
            "status": "cancelled",
            "created_at": "2025-01-29T08:00:00Z",
            "steps": [],
            "result": None,
        }
    )
    return handler


# ===========================================================================
# Test: Initialization
# ===========================================================================


class TestComputerUseHandlerInit:
    """Test handler initialization."""

    def test_handler_initializes_with_empty_state(self, mock_server_context):
        """Test that handler initializes with no storage and orchestrator."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            # Storage is lazy-loaded, so initially None
            assert h._storage is None
            assert h._orchestrator is None

    def test_handler_stores_server_context(self, mock_server_context):
        """Test that handler stores the server context."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            assert h.ctx is mock_server_context

    def test_handler_has_routes(self, mock_server_context):
        """Test that handler has expected routes defined."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            assert len(ComputerUseHandler.ROUTES) >= 4
            assert "/api/v1/computer-use/tasks" in ComputerUseHandler.ROUTES
            assert "/api/v1/computer-use/policies" in ComputerUseHandler.ROUTES


# ===========================================================================
# Test: Orchestrator Creation
# ===========================================================================


class TestOrchestratorCreation:
    """Test _get_orchestrator method."""

    def test_get_orchestrator_when_unavailable(self, mock_server_context):
        """Test that _get_orchestrator returns None when module unavailable."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h._get_orchestrator()
            assert result is None

    def test_get_orchestrator_creates_once(self, mock_server_context):
        """Test that _get_orchestrator creates and caches orchestrator."""
        mock_policy = MockComputerPolicy()
        mock_config = MagicMock()
        mock_orch = MagicMock()

        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch(
                "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
                return_value=mock_policy,
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.ComputerUseConfig",
                return_value=mock_config,
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.ComputerUseOrchestrator",
                return_value=mock_orch,
            ),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            orch1 = h._get_orchestrator()
            orch2 = h._get_orchestrator()
            assert orch1 is orch2
            assert orch1 is mock_orch

    def test_get_orchestrator_returns_cached(self, handler):
        """Test that _get_orchestrator returns pre-set orchestrator."""
        mock_orch = MagicMock()
        handler._orchestrator = mock_orch
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True):
            result = handler._get_orchestrator()
            assert result is mock_orch


# ===========================================================================
# Test: can_handle()
# ===========================================================================


class TestCanHandle:
    """Test can_handle method with various paths."""

    def test_handles_tasks_path(self, handler):
        assert handler.can_handle("/api/v1/computer-use/tasks")

    def test_handles_tasks_id_path(self, handler):
        assert handler.can_handle("/api/v1/computer-use/tasks/task-123")

    def test_handles_cancel_path(self, handler):
        assert handler.can_handle("/api/v1/computer-use/tasks/task-123/cancel")

    def test_handles_actions_stats_path(self, handler):
        assert handler.can_handle("/api/v1/computer-use/actions/stats")

    def test_handles_policies_path(self, handler):
        assert handler.can_handle("/api/v1/computer-use/policies")

    def test_handles_policies_id_path(self, handler):
        assert handler.can_handle("/api/v1/computer-use/policies/policy-abc")

    def test_rejects_root_path(self, handler):
        assert not handler.can_handle("/")

    def test_rejects_api_root(self, handler):
        assert not handler.can_handle("/api/v1/")

    def test_rejects_other_api_paths(self, handler):
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/agents")
        assert not handler.can_handle("/api/v1/backups")

    def test_rejects_missing_v1(self, handler):
        assert not handler.can_handle("/api/computer-use/tasks")

    def test_rejects_partial_prefix(self, handler):
        assert not handler.can_handle("/api/v1/computer-us")

    def test_handles_deeply_nested_path(self, handler):
        assert handler.can_handle("/api/v1/computer-use/something/deeply/nested")


# ===========================================================================
# Test: GET /api/v1/computer-use/tasks (List Tasks)
# ===========================================================================


class TestListTasks:
    """Test list tasks endpoint."""

    def test_list_tasks_empty(self, handler):
        """Test listing tasks when no tasks exist."""
        mock_handler = MockRequestHandler()
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["tasks"] == []
        assert body["total"] == 0

    def test_list_tasks_returns_all(self, handler_with_tasks):
        """Test listing all tasks returns correct count."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 5

    def test_list_tasks_with_status_filter_completed(self, handler_with_tasks):
        """Test listing tasks filtered by completed status."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks",
                {"status": "completed"},
                mock_handler,
            )

        body = json.loads(result.body)
        assert body["total"] == 1
        assert all(t["status"] == "completed" for t in body["tasks"])

    def test_list_tasks_with_status_filter_running(self, handler_with_tasks):
        """Test listing tasks filtered by running status."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks",
                {"status": "running"},
                mock_handler,
            )

        body = json.loads(result.body)
        assert body["total"] == 1
        assert body["tasks"][0]["task_id"] == "task-bbb"

    def test_list_tasks_with_status_filter_failed(self, handler_with_tasks):
        """Test listing tasks filtered by failed status."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks",
                {"status": "failed"},
                mock_handler,
            )

        body = json.loads(result.body)
        assert body["total"] == 1
        assert body["tasks"][0]["task_id"] == "task-ccc"

    def test_list_tasks_with_status_filter_no_match(self, handler_with_tasks):
        """Test listing tasks with a status filter that matches nothing."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks",
                {"status": "nonexistent_status"},
                mock_handler,
            )

        body = json.loads(result.body)
        assert body["total"] == 0

    def test_list_tasks_with_limit(self, handler_with_tasks):
        """Test listing tasks with limit parameter."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks",
                {"limit": "2"},
                mock_handler,
            )

        body = json.loads(result.body)
        assert len(body["tasks"]) == 2

    def test_list_tasks_with_limit_one(self, handler_with_tasks):
        """Test listing tasks with limit=1 returns just one."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks",
                {"limit": "1"},
                mock_handler,
            )

        body = json.loads(result.body)
        assert len(body["tasks"]) == 1

    def test_list_tasks_sorted_by_created_at_descending(self, handler_with_tasks):
        """Test that tasks are sorted by created_at in descending order."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle("/api/v1/computer-use/tasks", {}, mock_handler)

        body = json.loads(result.body)
        created_times = [t["created_at"] for t in body["tasks"]]
        assert created_times == sorted(created_times, reverse=True)

    def test_list_tasks_default_limit(self, handler_with_tasks):
        """Test that default limit is applied (20)."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle("/api/v1/computer-use/tasks", {}, mock_handler)

        # Default limit is 20, and we have only 5 tasks
        body = json.loads(result.body)
        assert len(body["tasks"]) == 5


# ===========================================================================
# Test: GET /api/v1/computer-use/tasks/{id} (Get Single Task)
# ===========================================================================


class TestGetTask:
    """Test get single task endpoint."""

    def test_get_existing_task(self, handler_with_tasks):
        """Test getting an existing task returns full details."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks/task-aaa", {}, mock_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["task"]["task_id"] == "task-aaa"
        assert body["task"]["goal"] == "Open browser"
        assert body["task"]["status"] == "completed"

    def test_get_running_task(self, handler_with_tasks):
        """Test getting a running task."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks/task-bbb", {}, mock_handler
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["task"]["status"] == "running"

    def test_get_task_not_found(self, handler_with_tasks):
        """Test getting non-existent task returns 404."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks/task-nonexistent", {}, mock_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_get_task_response_structure(self, handler_with_tasks):
        """Test that get task response has correct structure."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks/task-aaa", {}, mock_handler
            )

        body = json.loads(result.body)
        task = body["task"]
        assert "task_id" in task
        assert "goal" in task
        assert "max_steps" in task
        assert "status" in task
        assert "created_at" in task
        assert "steps" in task
        assert "result" in task


# ===========================================================================
# Test: POST /api/v1/computer-use/tasks (Create Task)
# ===========================================================================


class TestCreateTask:
    """Test create task endpoint."""

    def test_create_task_dry_run(self, handler):
        """Test creating a task in dry run mode."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "Test dry run task",
                    "dry_run": True,
                },
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["status"] == "completed"
        assert "task_id" in body
        assert body["task_id"].startswith("task-")

    def test_create_task_dry_run_sets_correct_result(self, handler):
        """Test that dry run task has correct result in store."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "Test dry run",
                    "dry_run": True,
                },
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        body = json.loads(result.body)
        task = handler._get_storage().get_task(body["task_id"])
        assert task is not None
        result_dict = task.to_dict()["result"]
        assert result_dict["success"] is True
        assert result_dict["message"] == "Dry run completed"
        assert result_dict["steps_taken"] == 0

    def test_create_task_execution_success(self, handler):
        """Test creating a task that executes successfully."""
        mock_handler = MockRequestHandler()

        mock_step = MockStep(
            action=MockAction(action_type=MockActionType(value="click")),
            result=MockStepResult(success=True),
        )
        mock_result = MockTaskResult(
            status="completed",
            error=None,
            steps=[mock_step],
        )

        # TaskStatus needs to be compared as attribute
        mock_task_status = MagicMock()
        mock_task_status.COMPLETED = "completed"

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "Execute task",
                    "max_steps": 5,
                },
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=mock_result,
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.TaskStatus",
                mock_task_status,
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["status"] == "completed"

    def test_create_task_execution_failure(self, handler):
        """Test creating a task that fails during execution."""
        mock_handler = MockRequestHandler()

        mock_result = MockTaskResult(
            status="failed",
            error="Element not found",
            steps=[],
        )

        mock_task_status = MagicMock()
        mock_task_status.COMPLETED = "completed"

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "Failing task",
                },
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=mock_result,
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.TaskStatus",
                mock_task_status,
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["status"] == "failed"

    def test_create_task_execution_exception(self, handler):
        """Test creating a task that throws an exception during execution."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "Exception task",
                },
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                side_effect=RuntimeError("Connection timeout"),
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result.status_code == 201
        body = json.loads(result.body)
        assert body["status"] == "failed"

        # Verify the task result contains the error
        task_id = body["task_id"]
        task = handler._get_storage().get_task(task_id)
        assert task is not None
        result_dict = task.to_dict()["result"]
        assert result_dict["success"] is False
        assert result_dict["message"] == "Task execution failed"

    def test_create_task_missing_goal(self, handler):
        """Test creating task without goal returns 400."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "read_json_body", return_value={}),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result.status_code == 400

    def test_create_task_empty_goal(self, handler):
        """Test creating task with empty string goal returns 400."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "read_json_body", return_value={"goal": ""}),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result.status_code == 400

    def test_create_task_invalid_json_body(self, handler):
        """Test creating task with invalid JSON body returns 400."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "read_json_body", return_value=None),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result.status_code == 400

    def test_create_task_with_default_max_steps(self, handler):
        """Test that default max_steps is 10."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "Test default max_steps",
                    "dry_run": True,
                },
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        body = json.loads(result.body)
        task = handler._get_storage().get_task(body["task_id"])
        assert task is not None
        assert task.max_steps == 10

    def test_create_task_with_custom_max_steps(self, handler):
        """Test creating task with custom max_steps."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "Custom max steps",
                    "max_steps": 25,
                    "dry_run": True,
                },
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        body = json.loads(result.body)
        task = handler._get_storage().get_task(body["task_id"])
        assert task is not None
        assert task.max_steps == 25

    def test_create_task_generates_unique_ids(self, handler):
        """Test that multiple task creations generate unique IDs."""
        mock_handler = MockRequestHandler()
        task_ids = set()

        for i in range(5):
            with (
                patch.object(handler, "_check_rbac_permission", return_value=None),
                patch.object(
                    handler,
                    "read_json_body",
                    return_value={
                        "goal": f"Task {i}",
                        "dry_run": True,
                    },
                ),
            ):
                result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)
            body = json.loads(result.body)
            task_ids.add(body["task_id"])

        assert len(task_ids) == 5

    def test_create_task_stores_in_tasks_dict(self, handler):
        """Test that created task is stored in _tasks dict."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "Stored task",
                    "dry_run": True,
                },
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        body = json.loads(result.body)
        task = handler._get_storage().get_task(body["task_id"])
        assert task is not None

    def test_create_task_orchestrator_unavailable(self, handler):
        """Test creating task when orchestrator returns None."""
        mock_handler = MockRequestHandler()
        handler._orchestrator = None

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "No orchestrator",
                },
            ),
            patch.object(handler, "_get_orchestrator", return_value=None),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        assert result.status_code == 503

    def test_create_task_with_multiple_steps_success(self, handler):
        """Test task with multiple successful steps."""
        mock_handler = MockRequestHandler()

        steps = [
            MockStep(
                action=MockAction(action_type=MockActionType(value="screenshot")),
                result=MockStepResult(success=True),
            ),
            MockStep(
                action=MockAction(action_type=MockActionType(value="click")),
                result=MockStepResult(success=True),
            ),
            MockStep(
                action=MockAction(action_type=MockActionType(value="type")),
                result=MockStepResult(success=True),
            ),
        ]
        mock_result = MockTaskResult(status="completed", error=None, steps=steps)

        mock_task_status = MagicMock()
        mock_task_status.COMPLETED = "completed"

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "Multi-step task",
                },
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=mock_result,
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.TaskStatus",
                mock_task_status,
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        body = json.loads(result.body)
        task = handler._get_storage().get_task(body["task_id"])
        assert task is not None
        steps = task.to_dict()["steps"]
        assert len(steps) == 3
        assert steps[0]["action"] == "screenshot"
        assert steps[1]["action"] == "click"
        assert steps[2]["action"] == "type"


# ===========================================================================
# Test: POST /api/v1/computer-use/tasks/{id}/cancel (Cancel Task)
# ===========================================================================


class TestCancelTask:
    """Test cancel task endpoint."""

    def test_cancel_running_task(self, handler_with_tasks):
        """Test cancelling a running task succeeds."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle_post(
                "/api/v1/computer-use/tasks/task-bbb/cancel", {}, mock_handler
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["message"] == "Task cancelled"
        task = handler_with_tasks._get_storage().get_task("task-bbb")
        assert task is not None
        assert task.status == "cancelled"

    def test_cancel_pending_task(self, handler_with_tasks):
        """Test cancelling a pending task succeeds."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle_post(
                "/api/v1/computer-use/tasks/task-ddd/cancel", {}, mock_handler
            )

        assert result.status_code == 200
        task = handler_with_tasks._get_storage().get_task("task-ddd")
        assert task is not None
        assert task.status == "cancelled"

    def test_cancel_completed_task_fails(self, handler_with_tasks):
        """Test cancelling completed task returns 400."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle_post(
                "/api/v1/computer-use/tasks/task-aaa/cancel", {}, mock_handler
            )

        assert result.status_code == 400

    def test_cancel_failed_task_fails(self, handler_with_tasks):
        """Test cancelling failed task returns 400."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle_post(
                "/api/v1/computer-use/tasks/task-ccc/cancel", {}, mock_handler
            )

        assert result.status_code == 400

    def test_cancel_already_cancelled_task_fails(self, handler_with_tasks):
        """Test cancelling already cancelled task returns 400."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle_post(
                "/api/v1/computer-use/tasks/task-eee/cancel", {}, mock_handler
            )

        assert result.status_code == 400

    def test_cancel_nonexistent_task(self, handler_with_tasks):
        """Test cancelling non-existent task returns 404."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle_post(
                "/api/v1/computer-use/tasks/task-zzz/cancel", {}, mock_handler
            )

        assert result.status_code == 404

    def test_cancel_task_sets_cancelled_at(self, handler_with_tasks):
        """Test cancelling a task sets the cancelled_at timestamp."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle_post(
                "/api/v1/computer-use/tasks/task-bbb/cancel", {}, mock_handler
            )

        assert result.status_code == 200
        task = handler_with_tasks._get_storage().get_task("task-bbb")
        assert task is not None
        task_dict = task.to_dict()
        assert "cancelled_at" in task_dict
        assert task_dict["cancelled_at"] is not None
        # Verify it's an ISO format timestamp
        assert "T" in task_dict["cancelled_at"]


# ===========================================================================
# Test: GET /api/v1/computer-use/actions/stats (Action Statistics)
# ===========================================================================


class TestActionStats:
    """Test action statistics endpoint."""

    def test_action_stats_empty(self, handler):
        """Test action stats with no tasks returns zero stats."""
        mock_handler = MockRequestHandler()
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/actions/stats", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "stats" in body
        for action_type in ["click", "type", "screenshot", "scroll", "key"]:
            assert body["stats"][action_type]["total"] == 0
            assert body["stats"][action_type]["success"] == 0
            assert body["stats"][action_type]["failed"] == 0

    def test_action_stats_with_tasks(self, handler_with_tasks):
        """Test action stats aggregates from task steps."""
        mock_handler = MockRequestHandler()
        with patch.object(handler_with_tasks, "_check_rbac_permission", return_value=None):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/actions/stats", {}, mock_handler
            )

        body = json.loads(result.body)
        stats = body["stats"]
        # From task-aaa: click(success), type(success), screenshot(failed)
        # From task-ccc: click(failed)
        assert stats["click"]["total"] == 2
        assert stats["click"]["success"] == 1
        assert stats["click"]["failed"] == 1
        assert stats["type"]["total"] == 1
        assert stats["type"]["success"] == 1
        assert stats["screenshot"]["total"] == 1
        assert stats["screenshot"]["failed"] == 1
        assert stats["scroll"]["total"] == 0
        assert stats["key"]["total"] == 0

    def test_action_stats_response_structure(self, handler):
        """Test that stats response has expected structure."""
        mock_handler = MockRequestHandler()
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/actions/stats", {}, mock_handler)

        body = json.loads(result.body)
        for action_type in ["click", "type", "screenshot", "scroll", "key"]:
            assert action_type in body["stats"]
            stat = body["stats"][action_type]
            assert "total" in stat
            assert "success" in stat
            assert "failed" in stat

    def test_action_stats_ignores_unknown_actions(self, handler):
        """Test that stats ignores action types not in the predefined list."""
        storage = handler._get_storage()
        storage.save_task(
            {
                "task_id": "task-x",
                "goal": "Unknown action test",
                "status": "completed",
                "steps": [
                    {"action": "unknown_action", "success": True},
                    {"action": "click", "success": True},
                ],
            }
        )
        mock_handler = MockRequestHandler()
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/actions/stats", {}, mock_handler)

        body = json.loads(result.body)
        assert body["stats"]["click"]["total"] == 1
        assert "unknown_action" not in body["stats"]


# ===========================================================================
# Test: GET /api/v1/computer-use/policies (List Policies)
# ===========================================================================


class TestListPolicies:
    """Test list policies endpoint."""

    def test_list_policies_default_only(self, handler):
        """Test listing policies returns at least the default policy."""
        mock_handler = MockRequestHandler()
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/policies", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] >= 1
        default_policy = body["policies"][0]
        assert default_policy["id"] == "default"
        assert default_policy["name"] == "Default Policy"

    def test_list_policies_with_custom_policies(self, handler):
        """Test listing policies includes custom policies."""
        storage = handler._get_storage()
        storage.save_policy(
            {
                "policy_id": "policy-custom",
                "name": "Custom Policy",
                "description": "My custom policy",
                "allowed_actions": ["screenshot"],
            }
        )

        mock_handler = MockRequestHandler()
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/policies", {}, mock_handler)

        body = json.loads(result.body)
        assert body["total"] == 2
        policy_ids = [p["id"] for p in body["policies"]]
        assert "default" in policy_ids
        assert "policy-custom" in policy_ids

    def test_list_policies_custom_policy_attributes(self, handler):
        """Test that custom policy attributes are correctly serialized."""
        storage = handler._get_storage()
        storage.save_policy(
            {
                "policy_id": "policy-restricted",
                "name": "Restricted",
                "description": "Limited actions",
                "allowed_actions": ["screenshot", "scroll"],
            }
        )

        mock_handler = MockRequestHandler()
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/policies", {}, mock_handler)

        body = json.loads(result.body)
        custom = [p for p in body["policies"] if p["id"] == "policy-restricted"][0]
        assert custom["name"] == "Restricted"
        assert custom["description"] == "Limited actions"
        assert custom["allowed_actions"] == ["screenshot", "scroll"]

    def test_list_policies_response_structure(self, handler):
        """Test that policies response has correct structure."""
        mock_handler = MockRequestHandler()
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/policies", {}, mock_handler)

        body = json.loads(result.body)
        assert "policies" in body
        assert "total" in body
        assert isinstance(body["policies"], list)
        assert isinstance(body["total"], int)

    def test_list_policies_default_policy_has_all_action_types(self, handler):
        """Test that default policy includes all standard action types."""
        mock_handler = MockRequestHandler()
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/policies", {}, mock_handler)

        body = json.loads(result.body)
        default = body["policies"][0]
        for action in ["screenshot", "click", "type", "scroll", "key"]:
            assert action in default["allowed_actions"]


# ===========================================================================
# Test: POST /api/v1/computer-use/policies (Create Policy)
# ===========================================================================


class TestCreatePolicy:
    """Test create policy endpoint."""

    def test_create_policy_success(self, handler):
        """Test creating a new policy succeeds."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "name": "New Policy",
                    "allowed_actions": ["screenshot", "click"],
                },
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
                return_value=MockComputerPolicy(),
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/policies", {}, mock_handler)

        assert result.status_code == 201
        body = json.loads(result.body)
        assert "policy_id" in body
        assert body["policy_id"].startswith("policy-")
        assert "message" in body

    def test_create_policy_missing_name(self, handler):
        """Test creating policy without name returns 400."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "read_json_body", return_value={}),
        ):
            result = handler.handle_post("/api/v1/computer-use/policies", {}, mock_handler)

        assert result.status_code == 400

    def test_create_policy_empty_name(self, handler):
        """Test creating policy with empty name returns 400."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "read_json_body", return_value={"name": ""}),
        ):
            result = handler.handle_post("/api/v1/computer-use/policies", {}, mock_handler)

        assert result.status_code == 400

    def test_create_policy_invalid_json(self, handler):
        """Test creating policy with invalid JSON body returns 400."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "read_json_body", return_value=None),
        ):
            result = handler.handle_post("/api/v1/computer-use/policies", {}, mock_handler)

        assert result.status_code == 400

    def test_create_policy_stores_in_storage(self, handler):
        """Test that created policy is stored in storage."""
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "name": "Stored Policy",
                },
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
                return_value=MockComputerPolicy(),
            ),
        ):
            result = handler.handle_post("/api/v1/computer-use/policies", {}, mock_handler)

        body = json.loads(result.body)
        policy = handler._get_storage().get_policy(body["policy_id"])
        assert policy is not None

    def test_create_multiple_policies(self, handler):
        """Test creating multiple policies generates unique IDs."""
        mock_handler = MockRequestHandler()
        policy_ids = set()

        for i in range(3):
            with (
                patch.object(handler, "_check_rbac_permission", return_value=None),
                patch.object(
                    handler,
                    "read_json_body",
                    return_value={
                        "name": f"Policy {i}",
                    },
                ),
                patch(
                    "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
                    return_value=MockComputerPolicy(),
                ),
            ):
                result = handler.handle_post("/api/v1/computer-use/policies", {}, mock_handler)
            body = json.loads(result.body)
            policy_ids.add(body["policy_id"])

        assert len(policy_ids) == 3


# ===========================================================================
# Test: Computer-use approvals endpoints
# ===========================================================================


class TestApprovalEndpoints:
    """Test approval workflow endpoints."""

    def test_list_approvals_unavailable(self, handler):
        """Return 503 when no approval workflow is configured."""
        handler._approval_workflow = None
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/approvals", {}, mock_handler)

        assert result.status_code == 503

    def test_list_approvals_success(self, handler):
        """List approvals returns serialized approvals."""
        approval = MagicMock()
        approval.to_dict.return_value = {"id": "approval-1", "status": "pending"}

        handler._approval_workflow = MagicMock()
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=[approval],
            ),
        ):
            result = handler.handle("/api/v1/computer-use/approvals", {}, mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["approvals"][0]["id"] == "approval-1"

    def test_list_approvals_invalid_status(self, handler):
        """Invalid status filter returns 400."""
        handler._approval_workflow = MagicMock()
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle(
                "/api/v1/computer-use/approvals",
                {"status": "not-a-status"},
                mock_handler,
            )

        assert result.status_code == 400

    def test_get_approval_not_found(self, handler):
        """Return 404 when approval request is missing."""
        handler._approval_workflow = MagicMock()
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=None,
            ),
        ):
            result = handler.handle(
                "/api/v1/computer-use/approvals/approval-missing",
                {},
                mock_handler,
            )

        assert result.status_code == 404

    def test_get_approval_success(self, handler):
        """Return approval payload when found."""
        approval = MagicMock()
        approval.to_dict.return_value = {"id": "approval-2", "status": "approved"}

        handler._approval_workflow = MagicMock()
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=approval,
            ),
        ):
            result = handler.handle(
                "/api/v1/computer-use/approvals/approval-2",
                {},
                mock_handler,
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["approval"]["id"] == "approval-2"

    def test_approve_approval_success(self, handler):
        """Approve endpoint returns success."""
        handler._approval_workflow = MagicMock()
        mock_handler = MockRequestHandler(body={"reason": "ok"})

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "_get_auth_context", return_value=MagicMock(user_id="admin")),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=True,
            ),
        ):
            result = handler.handle_post(
                "/api/v1/computer-use/approvals/approval-3/approve",
                {},
                mock_handler,
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["approved"] is True

    def test_approve_approval_missing(self, handler):
        """Approve endpoint returns 404 when not found."""
        handler._approval_workflow = MagicMock()
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "_get_auth_context", return_value=MagicMock(user_id="admin")),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=False,
            ),
        ):
            result = handler.handle_post(
                "/api/v1/computer-use/approvals/approval-missing/approve",
                {},
                mock_handler,
            )

        assert result.status_code == 404

    def test_deny_approval_success(self, handler):
        """Deny endpoint returns success."""
        handler._approval_workflow = MagicMock()
        mock_handler = MockRequestHandler(body={"reason": "no"})

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "_get_auth_context", return_value=MagicMock(user_id="admin")),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=True,
            ),
        ):
            result = handler.handle_post(
                "/api/v1/computer-use/approvals/approval-4/deny",
                {},
                mock_handler,
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["denied"] is True

    def test_deny_approval_missing(self, handler):
        """Deny endpoint returns 404 when not found."""
        handler._approval_workflow = MagicMock()
        mock_handler = MockRequestHandler()

        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(handler, "_get_auth_context", return_value=MagicMock(user_id="admin")),
            patch(
                "aragora.server.handlers.computer_use_handler.run_async",
                return_value=False,
            ),
        ):
            result = handler.handle_post(
                "/api/v1/computer-use/approvals/approval-missing/deny",
                {},
                mock_handler,
            )

        assert result.status_code == 404


# ===========================================================================
# Test: RBAC Permission Checks
# ===========================================================================


class TestRBACPermissions:
    """Test RBAC permission enforcement across all endpoints."""

    def test_list_tasks_rbac_denied_401(self, handler):
        """Test list tasks returns 401 when not authenticated."""
        mock_handler = MockRequestHandler()
        with patch.object(
            handler,
            "_check_rbac_permission",
            return_value=error_response("Not authenticated", 401),
        ):
            result = handler.handle("/api/v1/computer-use/tasks", {}, mock_handler)
        assert result.status_code == 401

    def test_list_tasks_rbac_denied_403(self, handler):
        """Test list tasks returns 403 when not authorized."""
        mock_handler = MockRequestHandler()
        with patch.object(
            handler,
            "_check_rbac_permission",
            return_value=error_response("Permission denied", 403),
        ):
            result = handler.handle("/api/v1/computer-use/tasks", {}, mock_handler)
        assert result.status_code == 403

    def test_get_task_rbac_denied(self, handler_with_tasks):
        """Test get task returns 401 when not authenticated."""
        mock_handler = MockRequestHandler()
        with patch.object(
            handler_with_tasks,
            "_check_rbac_permission",
            return_value=error_response("Not authenticated", 401),
        ):
            result = handler_with_tasks.handle(
                "/api/v1/computer-use/tasks/task-aaa", {}, mock_handler
            )
        assert result.status_code == 401

    def test_create_task_rbac_denied(self, handler):
        """Test create task returns 403 when not authorized."""
        mock_handler = MockRequestHandler()
        with patch.object(
            handler,
            "_check_rbac_permission",
            return_value=error_response("Permission denied: insufficient privileges", 403),
        ):
            result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)
        assert result.status_code == 403

    def test_cancel_task_rbac_denied(self, handler_with_tasks):
        """Test cancel task returns 401 when not authenticated."""
        mock_handler = MockRequestHandler()
        with patch.object(
            handler_with_tasks,
            "_check_rbac_permission",
            return_value=error_response("Not authenticated", 401),
        ):
            result = handler_with_tasks.handle_post(
                "/api/v1/computer-use/tasks/task-bbb/cancel", {}, mock_handler
            )
        assert result.status_code == 401

    def test_action_stats_rbac_denied(self, handler):
        """Test action stats returns 403 when not authorized."""
        mock_handler = MockRequestHandler()
        with patch.object(
            handler,
            "_check_rbac_permission",
            return_value=error_response("Permission denied", 403),
        ):
            result = handler.handle("/api/v1/computer-use/actions/stats", {}, mock_handler)
        assert result.status_code == 403

    def test_list_policies_rbac_denied(self, handler):
        """Test list policies returns 401 when not authenticated."""
        mock_handler = MockRequestHandler()
        with patch.object(
            handler,
            "_check_rbac_permission",
            return_value=error_response("Not authenticated", 401),
        ):
            result = handler.handle("/api/v1/computer-use/policies", {}, mock_handler)
        assert result.status_code == 401

    def test_create_policy_rbac_denied(self, handler):
        """Test create policy returns 403 when not authorized."""
        mock_handler = MockRequestHandler()
        with patch.object(
            handler,
            "_check_rbac_permission",
            return_value=error_response("Permission denied", 403),
        ):
            result = handler.handle_post("/api/v1/computer-use/policies", {}, mock_handler)
        assert result.status_code == 403

    def test_rbac_bypassed_when_unavailable(self, mock_server_context):
        """Test that RBAC is bypassed when RBAC_AVAILABLE is False."""
        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", False),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            # _check_rbac_permission should return None when RBAC unavailable
            result = h._check_rbac_permission(MockRequestHandler(), "any:permission")
            assert result is None


# ===========================================================================
# Test: RBAC Integration (with RBAC_AVAILABLE=True)
# ===========================================================================


class TestRBACIntegration:
    """Test _check_rbac_permission and _get_auth_context with RBAC available."""

    def test_check_rbac_permission_no_auth_context(self, mock_server_context):
        """Test that permission check returns 401 when auth context is None."""
        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            with patch.object(h, "_get_auth_context", return_value=None):
                result = h._check_rbac_permission(MockRequestHandler(), "some:permission")
            assert result is not None
            assert result.status_code == 401

    def test_check_rbac_permission_denied(self, mock_server_context):
        """Test that permission check returns 403 when check_permission denies."""
        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-1"

        mock_decision = MagicMock()
        mock_decision.allowed = False
        mock_decision.reason = "No admin role"

        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True),
            patch(
                "aragora.server.handlers.computer_use_handler.check_permission",
                return_value=mock_decision,
            ),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            with patch.object(h, "_get_auth_context", return_value=mock_auth_ctx):
                result = h._check_rbac_permission(MockRequestHandler(), "some:perm")
            assert result is not None
            assert result.status_code == 403

    def test_check_rbac_permission_allowed(self, mock_server_context):
        """Test that permission check returns None when permission granted."""
        mock_auth_ctx = MagicMock()
        mock_auth_ctx.user_id = "user-1"

        mock_decision = MagicMock()
        mock_decision.allowed = True

        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True),
            patch(
                "aragora.server.handlers.computer_use_handler.check_permission",
                return_value=mock_decision,
            ),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            with patch.object(h, "_get_auth_context", return_value=mock_auth_ctx):
                result = h._check_rbac_permission(MockRequestHandler(), "some:perm")
            assert result is None


# ===========================================================================
# Test: Computer Use Module Unavailable (503)
# ===========================================================================


class TestModuleUnavailable:
    """Test behavior when computer use module is not available."""

    def test_get_tasks_returns_503(self, mock_server_context):
        """Test GET tasks returns 503 when module unavailable."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h.handle("/api/v1/computer-use/tasks", {}, MockRequestHandler())
            assert result is not None
            assert result.status_code == 503

    def test_get_single_task_returns_503(self, mock_server_context):
        """Test GET single task returns 503 when module unavailable."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h.handle("/api/v1/computer-use/tasks/task-123", {}, MockRequestHandler())
            assert result is not None
            assert result.status_code == 503

    def test_action_stats_returns_503(self, mock_server_context):
        """Test GET action stats returns 503 when module unavailable."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h.handle("/api/v1/computer-use/actions/stats", {}, MockRequestHandler())
            assert result.status_code == 503

    def test_list_policies_returns_503(self, mock_server_context):
        """Test GET policies returns 503 when module unavailable."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h.handle("/api/v1/computer-use/policies", {}, MockRequestHandler())
            assert result.status_code == 503

    def test_post_create_task_returns_503(self, mock_server_context):
        """Test POST create task returns 503 when module unavailable."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h.handle_post("/api/v1/computer-use/tasks", {}, MockRequestHandler())
            assert result.status_code == 503

    def test_post_cancel_task_returns_503(self, mock_server_context):
        """Test POST cancel task returns 503 when module unavailable."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h.handle_post(
                "/api/v1/computer-use/tasks/task-123/cancel", {}, MockRequestHandler()
            )
            assert result.status_code == 503

    def test_post_create_policy_returns_503(self, mock_server_context):
        """Test POST create policy returns 503 when module unavailable."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h.handle_post("/api/v1/computer-use/policies", {}, MockRequestHandler())
            assert result.status_code == 503

    def test_503_error_message(self, mock_server_context):
        """Test that 503 responses contain correct error message."""
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", False):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h.handle("/api/v1/computer-use/tasks", {}, MockRequestHandler())
            body = json.loads(result.body)
            assert "error" in body
            assert (
                "not available" in body["error"].lower()
                or "not available" in str(body["error"]).lower()
            )


# ===========================================================================
# Test: Routing Edge Cases
# ===========================================================================


class TestRoutingEdgeCases:
    """Test edge cases in request routing."""

    def test_handle_returns_none_for_non_matching_path(self, handler):
        """Test that handle returns None for paths it cannot handle."""
        mock_handler = MockRequestHandler()
        result = handler.handle("/api/v1/debates", {}, mock_handler)
        assert result is None

    def test_handle_post_returns_none_for_non_matching_path(self, handler):
        """Test that handle_post returns None for paths it cannot handle."""
        mock_handler = MockRequestHandler()
        result = handler.handle_post("/api/v1/debates", {}, mock_handler)
        assert result is None

    def test_handle_returns_none_for_unknown_computer_use_path(self, handler):
        """Test that handle returns None for unrecognized computer-use sub-paths."""
        mock_handler = MockRequestHandler()
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True):
            result = handler.handle("/api/v1/computer-use/unknown-endpoint", {}, mock_handler)
        assert result is None

    def test_handle_post_returns_none_for_unknown_computer_use_post_path(self, handler):
        """Test that handle_post returns None for unrecognized POST sub-paths."""
        mock_handler = MockRequestHandler()
        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True):
            result = handler.handle_post("/api/v1/computer-use/unknown", {}, mock_handler)
        assert result is None

    def test_handle_get_task_with_short_path(self, handler):
        """Test that handle correctly identifies task ID from path."""
        mock_handler = MockRequestHandler()
        storage = handler._get_storage()
        storage.save_task(
            {
                "task_id": "abc",
                "goal": "test",
                "status": "pending",
                "created_at": "2025-01-01T00:00:00Z",
            }
        )
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/tasks/abc", {}, mock_handler)
        assert result.status_code == 200

    def test_cancel_path_requires_correct_structure(self, handler):
        """Test that cancel path parsing is correct with short paths."""
        mock_handler = MockRequestHandler()
        # Path with /cancel but not enough parts should still match "/cancel" in path
        # but parts length check should prevent execution
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle_post("/api/v1/computer-use/cancel", {}, mock_handler)
        # "/cancel" is in path, but parts won't have enough segments
        # so it falls through to None
        assert result is None


# ===========================================================================
# Test: Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting on create endpoints."""

    def test_create_task_rate_limited(self, handler):
        """Test that create task endpoint is rate limited after many requests."""
        mock_handler = MockRequestHandler()
        results = []

        for i in range(15):
            with (
                patch.object(handler, "_check_rbac_permission", return_value=None),
                patch.object(
                    handler,
                    "read_json_body",
                    return_value={
                        "goal": f"Task {i}",
                        "dry_run": True,
                    },
                ),
            ):
                result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)
                results.append(result)

        # At least one should be rate limited (429)
        status_codes = [r.status_code for r in results]
        assert 429 in status_codes, f"Expected rate limiting but got statuses: {status_codes}"

    def test_create_policy_rate_limited(self, handler):
        """Test that create policy endpoint is rate limited after many requests."""
        mock_handler = MockRequestHandler()
        results = []

        for i in range(15):
            with (
                patch.object(handler, "_check_rbac_permission", return_value=None),
                patch.object(
                    handler,
                    "read_json_body",
                    return_value={
                        "name": f"Policy {i}",
                    },
                ),
                patch(
                    "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
                    return_value=MockComputerPolicy(),
                ),
            ):
                result = handler.handle_post("/api/v1/computer-use/policies", {}, mock_handler)
                results.append(result)

        status_codes = [r.status_code for r in results]
        assert 429 in status_codes, f"Expected rate limiting but got statuses: {status_codes}"


# ===========================================================================
# Test: _get_user_store
# ===========================================================================


class TestGetUserStore:
    """Test _get_user_store method."""

    def test_get_user_store_returns_from_context(self, mock_server_context):
        """Test that _get_user_store returns user_store from context."""
        mock_store = MagicMock()
        mock_server_context.get = MagicMock(return_value=mock_store)

        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h._get_user_store()
            assert result is mock_store

    def test_get_user_store_returns_none_when_missing(self, mock_server_context):
        """Test that _get_user_store returns None when user_store is not in context."""
        mock_server_context.get = MagicMock(return_value=None)

        with patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h._get_user_store()
            assert result is None


# ===========================================================================
# Test: _get_auth_context
# ===========================================================================


class TestGetAuthContext:
    """Test _get_auth_context method."""

    def test_get_auth_context_when_rbac_unavailable(self, mock_server_context):
        """Test returns None when RBAC is not available."""
        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", False),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h._get_auth_context(MockRequestHandler())
            assert result is None

    def test_get_auth_context_unauthenticated(self, mock_server_context):
        """Test returns None when user is not authenticated."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = False

        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True),
            patch(
                "aragora.server.handlers.computer_use_handler.extract_user_from_request",
                return_value=mock_auth,
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.AuthorizationContext",
                MagicMock(),
            ),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            result = h._get_auth_context(MockRequestHandler())
            assert result is None

    def test_get_auth_context_authenticated_with_user(self, mock_server_context):
        """Test returns AuthorizationContext when user is authenticated."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.user_id = "user-123"
        mock_auth.org_id = "org-456"

        mock_user = MagicMock()
        mock_user.role = "admin"

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id = MagicMock(return_value=mock_user)

        mock_auth_context_class = MagicMock()
        mock_auth_context_instance = MagicMock()
        mock_auth_context_class.return_value = mock_auth_context_instance

        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True),
            patch(
                "aragora.server.handlers.computer_use_handler.extract_user_from_request",
                return_value=mock_auth,
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.AuthorizationContext",
                mock_auth_context_class,
            ),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            h._get_user_store = MagicMock(return_value=mock_user_store)
            result = h._get_auth_context(MockRequestHandler())

            assert result is mock_auth_context_instance
            mock_auth_context_class.assert_called_once_with(
                user_id="user-123",
                roles={"admin"},
                org_id="org-456",
            )

    def test_get_auth_context_authenticated_no_user_store(self, mock_server_context):
        """Test returns AuthorizationContext with empty roles when user_store is None."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.user_id = "user-123"
        mock_auth.org_id = "org-456"

        mock_auth_context_class = MagicMock()
        mock_auth_context_instance = MagicMock()
        mock_auth_context_class.return_value = mock_auth_context_instance

        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True),
            patch(
                "aragora.server.handlers.computer_use_handler.extract_user_from_request",
                return_value=mock_auth,
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.AuthorizationContext",
                mock_auth_context_class,
            ),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            h._get_user_store = MagicMock(return_value=None)
            result = h._get_auth_context(MockRequestHandler())

            assert result is mock_auth_context_instance
            mock_auth_context_class.assert_called_once_with(
                user_id="user-123",
                roles=set(),
                org_id="org-456",
            )

    def test_get_auth_context_authenticated_user_no_role(self, mock_server_context):
        """Test returns AuthorizationContext with empty roles when user has no role."""
        mock_auth = MagicMock()
        mock_auth.is_authenticated = True
        mock_auth.user_id = "user-123"
        mock_auth.org_id = "org-456"

        mock_user = MagicMock()
        mock_user.role = None

        mock_user_store = MagicMock()
        mock_user_store.get_user_by_id = MagicMock(return_value=mock_user)

        mock_auth_context_class = MagicMock()
        mock_auth_context_instance = MagicMock()
        mock_auth_context_class.return_value = mock_auth_context_instance

        with (
            patch("aragora.server.handlers.computer_use_handler.COMPUTER_USE_AVAILABLE", True),
            patch("aragora.server.handlers.computer_use_handler.RBAC_AVAILABLE", True),
            patch(
                "aragora.server.handlers.computer_use_handler.extract_user_from_request",
                return_value=mock_auth,
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.AuthorizationContext",
                mock_auth_context_class,
            ),
        ):
            from aragora.server.handlers.computer_use_handler import ComputerUseHandler

            h = ComputerUseHandler(mock_server_context)
            h._get_user_store = MagicMock(return_value=mock_user_store)
            result = h._get_auth_context(MockRequestHandler())

            assert result is mock_auth_context_instance
            mock_auth_context_class.assert_called_once_with(
                user_id="user-123",
                roles=set(),
                org_id="org-456",
            )


# ===========================================================================
# Test: End-to-end Flows
# ===========================================================================


class TestEndToEndFlows:
    """Test complete request flows."""

    def test_create_then_list_tasks(self, handler):
        """Test creating a task then listing includes it."""
        mock_handler = MockRequestHandler()

        # Create task
        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "E2E test",
                    "dry_run": True,
                },
            ),
        ):
            create_result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        assert create_result.status_code == 201
        task_id = json.loads(create_result.body)["task_id"]

        # List tasks
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            list_result = handler.handle("/api/v1/computer-use/tasks", {}, mock_handler)

        list_body = json.loads(list_result.body)
        assert list_body["total"] == 1
        assert list_body["tasks"][0]["task_id"] == task_id

    def test_create_then_get_task(self, handler):
        """Test creating a task then getting it by ID."""
        mock_handler = MockRequestHandler()

        # Create task
        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "goal": "E2E get test",
                    "dry_run": True,
                },
            ),
        ):
            create_result = handler.handle_post("/api/v1/computer-use/tasks", {}, mock_handler)

        task_id = json.loads(create_result.body)["task_id"]

        # Get task
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            get_result = handler.handle(f"/api/v1/computer-use/tasks/{task_id}", {}, mock_handler)

        get_body = json.loads(get_result.body)
        assert get_body["task"]["task_id"] == task_id
        assert get_body["task"]["goal"] == "E2E get test"
        assert get_body["task"]["status"] == "completed"

    def test_create_then_cancel_task(self, handler):
        """Test creating a pending task and then cancelling it."""
        mock_handler = MockRequestHandler()

        # Create a task that stays pending (non dry_run but execution fails quickly)
        storage = handler._get_storage()
        storage.save_task(
            {
                "task_id": "task-pending-e2e",
                "goal": "Pending task",
                "max_steps": 5,
                "dry_run": False,
                "status": "running",
                "created_at": "2025-01-30T00:00:00Z",
                "steps": [],
                "result": None,
            }
        )

        # Cancel it
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            cancel_result = handler.handle_post(
                "/api/v1/computer-use/tasks/task-pending-e2e/cancel",
                {},
                mock_handler,
            )

        assert cancel_result.status_code == 200
        task = storage.get_task("task-pending-e2e")
        assert task is not None
        assert task.status == "cancelled"

    def test_create_policy_then_list(self, handler):
        """Test creating a policy then listing includes it."""
        mock_handler = MockRequestHandler()

        # Create policy
        with (
            patch.object(handler, "_check_rbac_permission", return_value=None),
            patch.object(
                handler,
                "read_json_body",
                return_value={
                    "name": "E2E Policy",
                },
            ),
            patch(
                "aragora.server.handlers.computer_use_handler.create_default_computer_policy",
                return_value=MockComputerPolicy(),
            ),
        ):
            create_result = handler.handle_post("/api/v1/computer-use/policies", {}, mock_handler)

        assert create_result.status_code == 201
        policy_id = json.loads(create_result.body)["policy_id"]

        # List policies
        with patch.object(handler, "_check_rbac_permission", return_value=None):
            list_result = handler.handle("/api/v1/computer-use/policies", {}, mock_handler)

        list_body = json.loads(list_result.body)
        assert list_body["total"] == 2  # default + created
        policy_ids = [p["id"] for p in list_body["policies"]]
        assert policy_id in policy_ids

    def test_action_stats_reflect_task_steps(self, handler):
        """Test that action stats accurately reflect task step data."""
        mock_handler = MockRequestHandler()

        # Add a task with known steps
        storage = handler._get_storage()
        storage.save_task(
            {
                "task_id": "task-stats",
                "goal": "Stats test",
                "status": "completed",
                "steps": [
                    {"action": "click", "success": True},
                    {"action": "click", "success": False},
                    {"action": "scroll", "success": True},
                    {"action": "key", "success": True},
                    {"action": "key", "success": False},
                    {"action": "key", "success": True},
                ],
            }
        )

        with patch.object(handler, "_check_rbac_permission", return_value=None):
            result = handler.handle("/api/v1/computer-use/actions/stats", {}, mock_handler)

        body = json.loads(result.body)
        stats = body["stats"]
        assert stats["click"]["total"] == 2
        assert stats["click"]["success"] == 1
        assert stats["click"]["failed"] == 1
        assert stats["scroll"]["total"] == 1
        assert stats["scroll"]["success"] == 1
        assert stats["key"]["total"] == 3
        assert stats["key"]["success"] == 2
        assert stats["key"]["failed"] == 1
