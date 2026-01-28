"""
Tests for control plane handler.

Tests:
- ControlPlaneHandler initialization
- Route matching (can_handle)
- Authentication requirements
- Agent registration/unregistration
- Task submission and lifecycle
- Health and status endpoints
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json

from aragora.server.handlers.control_plane import ControlPlaneHandler
from aragora.server.handlers.base import HandlerResult


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_class_coordinator():
    """Reset class-level coordinator before each test for isolation."""
    ControlPlaneHandler.coordinator = None
    yield
    ControlPlaneHandler.coordinator = None


@pytest.fixture
def handler():
    """Create a ControlPlaneHandler instance."""
    return ControlPlaneHandler({})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    mock = MagicMock()
    mock.headers = {"Content-Type": "application/json", "Content-Length": "100"}
    mock.command = "GET"
    mock.rfile = MagicMock()
    mock.rfile.read.return_value = b"{}"
    return mock


@pytest.fixture
def mock_admin_user():
    """Create a mock admin user for permission checks."""
    mock_user = MagicMock()
    mock_user.user_id = "user-1"
    mock_user.role = "admin"
    return mock_user


@pytest.fixture
def mock_coordinator():
    """Create a mock control plane coordinator."""
    coordinator = MagicMock()
    coordinator.register_agent = AsyncMock(
        return_value=MagicMock(
            id="agent-1",
            status="idle",
            capabilities=["debate", "analysis"],
            to_dict=lambda: {"id": "agent-1", "status": "idle"},
        )
    )
    coordinator.unregister_agent = AsyncMock(return_value=True)
    coordinator.get_agent_status = AsyncMock(
        return_value=MagicMock(to_dict=lambda: {"id": "agent-1", "status": "idle"})
    )
    coordinator.get_all_agents = AsyncMock(return_value=[])
    coordinator.submit_task = AsyncMock(
        return_value=MagicMock(
            id="task-1", status="pending", to_dict=lambda: {"id": "task-1", "status": "pending"}
        )
    )
    coordinator.get_task = AsyncMock(
        return_value=MagicMock(to_dict=lambda: {"id": "task-1", "status": "pending"})
    )
    coordinator.claim_task = AsyncMock(
        return_value=MagicMock(to_dict=lambda: {"id": "task-1", "status": "running"})
    )
    coordinator.complete_task = AsyncMock(return_value=True)
    coordinator.fail_task = AsyncMock(return_value=True)
    coordinator.cancel_task = AsyncMock(return_value=True)
    coordinator.heartbeat = AsyncMock(return_value=True)
    return coordinator


def get_status(result: HandlerResult) -> int:
    """Extract status code from handler result."""
    return result.status_code


def get_body(result: HandlerResult) -> dict:
    """Extract JSON body from handler result."""
    return json.loads(result.body.decode("utf-8"))


# ===========================================================================
# Test ControlPlaneHandler Initialization
# ===========================================================================


class TestControlPlaneHandlerInit:
    """Tests for ControlPlaneHandler initialization."""

    def test_init_with_empty_context(self):
        """Should initialize with empty context."""
        handler = ControlPlaneHandler({})
        assert handler is not None

    def test_init_with_coordinator(self):
        """Should accept coordinator in context."""
        mock_coord = MagicMock()
        handler = ControlPlaneHandler({"control_plane_coordinator": mock_coord})
        assert handler._get_coordinator() is mock_coord


# ===========================================================================
# Test Route Matching (can_handle)
# ===========================================================================


class TestControlPlaneHandlerCanHandle:
    """Tests for can_handle routing."""

    def test_can_handle_agents_endpoint(self, handler):
        """Should handle /api/control-plane/agents."""
        assert handler.can_handle("/api/v1/control-plane/agents") is True

    def test_can_handle_agents_by_id(self, handler):
        """Should handle /api/control-plane/agents/:id."""
        assert handler.can_handle("/api/v1/control-plane/agents/agent-1") is True

    def test_can_handle_tasks_endpoint(self, handler):
        """Should handle /api/control-plane/tasks."""
        assert handler.can_handle("/api/v1/control-plane/tasks") is True

    def test_can_handle_tasks_by_id(self, handler):
        """Should handle /api/control-plane/tasks/:id."""
        assert handler.can_handle("/api/v1/control-plane/tasks/task-1") is True

    def test_can_handle_health_endpoint(self, handler):
        """Should handle /api/control-plane/health."""
        assert handler.can_handle("/api/v1/control-plane/health") is True

    def test_can_handle_stats_endpoint(self, handler):
        """Should handle /api/control-plane/stats."""
        assert handler.can_handle("/api/v1/control-plane/stats") is True

    def test_cannot_handle_unknown_path(self, handler):
        """Should not handle unknown paths."""
        assert handler.can_handle("/api/v1/control-plane") is False
        assert handler.can_handle("/api/v1/unknown") is False
        assert handler.can_handle("/api/v1/other/path") is False


# ===========================================================================
# Test Health Endpoint
# ===========================================================================


class TestControlPlaneHealth:
    """Tests for GET /api/control-plane/health endpoint."""

    def test_health_returns_ok_with_coordinator(self, handler, mock_http_handler, mock_coordinator):
        """Should return health status when coordinator is available."""
        from enum import Enum

        class HealthStatus(Enum):
            HEALTHY = "healthy"

        mock_coordinator.get_system_health.return_value = HealthStatus.HEALTHY
        mock_coordinator._health_monitor = MagicMock()
        mock_coordinator._health_monitor.get_all_health.return_value = {}

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            result = handler.handle("/api/v1/control-plane/health", {}, mock_http_handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert "status" in body

    def test_health_returns_503_without_coordinator(self, handler, mock_http_handler):
        """Should return 503 when coordinator not initialized."""
        result = handler.handle("/api/v1/control-plane/health", {}, mock_http_handler)
        assert get_status(result) == 503
        assert "not initialized" in get_body(result)["error"]

    def test_health_no_auth_required(self, handler, mock_http_handler, mock_coordinator):
        """Health endpoint should not require auth."""
        from enum import Enum

        class HealthStatus(Enum):
            HEALTHY = "healthy"

        mock_coordinator.get_system_health.return_value = HealthStatus.HEALTHY
        mock_coordinator._health_monitor = MagicMock()
        mock_coordinator._health_monitor.get_all_health.return_value = {}

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            result = handler.handle("/api/v1/control-plane/health", {}, mock_http_handler)

        # Health endpoints don't require auth, should return 200 not 401
        assert get_status(result) != 401


# ===========================================================================
# Test Agent Registration (with Auth)
# ===========================================================================


class TestAgentRegistration:
    """Tests for POST /api/control-plane/agents endpoint."""

    @pytest.mark.asyncio
    async def test_register_requires_auth(self, handler, mock_http_handler, mock_coordinator):
        """Should require authentication for agent registration."""
        mock_http_handler.command = "POST"
        mock_http_handler.rfile = MagicMock()
        mock_http_handler.rfile.read.return_value = json.dumps(
            {"agent_id": "agent-1", "capabilities": ["debate"]}
        ).encode()

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler,
                "require_auth_or_error",
                return_value=(
                    None,
                    HandlerResult(401, "application/json", b'{"error": "Unauthorized"}', {}),
                ),
            ):
                result = await handler.handle_post(
                    "/api/v1/control-plane/agents", {}, mock_http_handler
                )

        assert get_status(result) == 401

    def test_register_agent_success(
        self, handler, mock_http_handler, mock_coordinator, mock_admin_user
    ):
        """Should register agent successfully with auth."""
        mock_http_handler.command = "POST"
        mock_http_handler.rfile.read.return_value = json.dumps(
            {
                "agent_id": "agent-1",
                "capabilities": ["debate", "analysis"],
                "model": "gpt-4",
                "provider": "openai",
            }
        ).encode()

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_admin_user, None)
            ):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"agent_id": "agent-1", "capabilities": ["debate"]}, None),
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/agents", {}, mock_http_handler
                    )

        assert get_status(result) == 201  # Created
        body = get_body(result)
        assert body["id"] == "agent-1"

    def test_register_agent_missing_id(
        self, handler, mock_http_handler, mock_coordinator, mock_admin_user
    ):
        """Should reject registration without agent_id."""
        mock_http_handler.command = "POST"

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_admin_user, None)
            ):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"capabilities": ["debate"]}, None),  # Missing agent_id
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/agents", {}, mock_http_handler
                    )

        assert get_status(result) == 400
        assert "agent_id" in get_body(result)["error"]


# ===========================================================================
# Test Task Submission (with Auth)
# ===========================================================================


class TestTaskSubmission:
    """Tests for POST /api/control-plane/tasks endpoint."""

    def test_submit_requires_auth(self, handler, mock_http_handler, mock_coordinator):
        """Should require authentication for task submission."""
        mock_http_handler.command = "POST"
        mock_http_handler.rfile.read.return_value = json.dumps({"task_type": "debate"}).encode()

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "read_json_body_validated", return_value=({"task_type": "debate"}, None)
            ):
                with patch.object(
                    handler,
                    "require_auth_or_error",
                    return_value=(
                        None,
                        HandlerResult(401, "application/json", b'{"error": "Unauthorized"}', {}),
                    ),
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/tasks", {}, mock_http_handler
                    )

        assert get_status(result) == 401

    def test_submit_task_success(
        self, handler, mock_http_handler, mock_coordinator, mock_admin_user
    ):
        """Should submit task successfully with auth."""
        mock_http_handler.command = "POST"

        # Configure mock to return actual task_id
        mock_coordinator.submit_task = AsyncMock(return_value="task-123")

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_admin_user, None)
            ):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"task_type": "debate", "payload": {"topic": "AI"}}, None),
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/tasks", {}, mock_http_handler
                    )

        assert get_status(result) == 201  # Created
        body = get_body(result)
        assert "task_id" in body

    def test_submit_task_missing_type(
        self, handler, mock_http_handler, mock_coordinator, mock_admin_user
    ):
        """Should reject task without task_type."""
        mock_http_handler.command = "POST"

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_admin_user, None)
            ):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"payload": {}}, None),  # Missing task_type
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/tasks", {}, mock_http_handler
                    )

        assert get_status(result) == 400
        assert "task_type" in get_body(result)["error"]


# ===========================================================================
# Test Agent Unregistration (with Auth)
# ===========================================================================


class TestAgentUnregistration:
    """Tests for DELETE /api/control-plane/agents/:id endpoint."""

    def test_unregister_requires_auth(self, handler, mock_http_handler, mock_coordinator):
        """Should require authentication for agent unregistration."""
        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler,
                "require_auth_or_error",
                return_value=(
                    None,
                    HandlerResult(401, "application/json", b'{"error": "Unauthorized"}', {}),
                ),
            ):
                result = handler.handle_delete(
                    "/api/v1/control-plane/agents/agent-1", {}, mock_http_handler
                )

        assert get_status(result) == 401

    def test_unregister_agent_success(
        self, handler, mock_http_handler, mock_coordinator, mock_admin_user
    ):
        """Should unregister agent successfully with auth."""
        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_admin_user, None)
            ):
                result = handler.handle_delete(
                    "/api/v1/control-plane/agents/agent-1", {}, mock_http_handler
                )

        assert get_status(result) == 200
        body = get_body(result)
        assert body["unregistered"] is True

    def test_unregister_nonexistent_agent(
        self, handler, mock_http_handler, mock_coordinator, mock_admin_user
    ):
        """Should return 404 for nonexistent agent."""
        mock_coordinator.unregister_agent = AsyncMock(return_value=False)

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_admin_user, None)
            ):
                result = handler.handle_delete(
                    "/api/v1/control-plane/agents/nonexistent", {}, mock_http_handler
                )

        assert get_status(result) == 404


# ===========================================================================
# Test Coordinator Not Initialized
# ===========================================================================


class TestCoordinatorNotInitialized:
    """Tests for when coordinator is not available."""

    def test_register_returns_503_without_coordinator(
        self, handler, mock_http_handler, mock_admin_user
    ):
        """Should return 503 when coordinator not initialized."""
        mock_http_handler.command = "POST"

        with patch.object(handler, "_get_coordinator", return_value=None):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_admin_user, None)
            ):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"agent_id": "agent-1"}, None),
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/agents", {}, mock_http_handler
                    )

        assert get_status(result) == 503
        assert "not initialized" in get_body(result)["error"]

    @pytest.mark.asyncio
    async def test_submit_returns_503_without_coordinator(
        self, handler, mock_http_handler, mock_admin_user
    ):
        """Should return 503 when coordinator not initialized."""
        mock_http_handler.command = "POST"

        with patch.object(handler, "_get_coordinator", return_value=None):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_admin_user, None)
            ):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"task_type": "debate"}, None),
                ):
                    result = await handler.handle_post(
                        "/api/v1/control-plane/tasks", {}, mock_http_handler
                    )

        assert get_status(result) == 503


# ===========================================================================
# Test Queue Endpoint
# ===========================================================================


class TestQueueEndpoint:
    """Tests for GET /api/control-plane/queue endpoint."""

    def test_can_handle_queue_endpoint(self, handler):
        """Should handle /api/control-plane/queue."""
        assert handler.can_handle("/api/v1/control-plane/queue") is True

    def test_queue_returns_jobs(self, handler, mock_http_handler, mock_coordinator):
        """Should return pending and running tasks as jobs."""
        import time

        # Create mock tasks
        mock_task_pending = MagicMock()
        mock_task_pending.id = "task-1"
        mock_task_pending.task_type = "audit"
        mock_task_pending.status = MagicMock(value="pending")
        mock_task_pending.priority = MagicMock(name="normal")
        mock_task_pending.metadata = {"name": "Test Audit"}
        mock_task_pending.payload = {"document_count": 5}
        mock_task_pending.started_at = None
        mock_task_pending.created_at = time.time()
        mock_task_pending.assigned_agent = None

        mock_task_running = MagicMock()
        mock_task_running.id = "task-2"
        mock_task_running.task_type = "analysis"
        mock_task_running.status = MagicMock(value="running")
        mock_task_running.priority = MagicMock(name="high")
        mock_task_running.metadata = {"name": "Data Analysis", "progress": 0.6}
        mock_task_running.payload = {"document_count": 10}
        mock_task_running.started_at = time.time()
        mock_task_running.created_at = time.time() - 60
        mock_task_running.assigned_agent = "agent-1"

        mock_coordinator._scheduler = MagicMock()
        mock_coordinator._scheduler.list_by_status = AsyncMock(
            side_effect=lambda status, limit: (
                [mock_task_pending] if status.value == "pending" else [mock_task_running]
            )
        )

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            result = handler.handle("/api/v1/control-plane/queue", {}, mock_http_handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert "jobs" in body
        assert len(body["jobs"]) == 2

    def test_queue_returns_503_without_coordinator(self, handler, mock_http_handler):
        """Should return 503 when coordinator not initialized."""
        result = handler.handle("/api/v1/control-plane/queue", {}, mock_http_handler)
        assert get_status(result) == 503


# ===========================================================================
# Test Metrics Endpoint
# ===========================================================================


class TestMetricsEndpoint:
    """Tests for GET /api/control-plane/metrics endpoint."""

    def test_can_handle_metrics_endpoint(self, handler):
        """Should handle /api/control-plane/metrics."""
        assert handler.can_handle("/api/v1/control-plane/metrics") is True

    def test_metrics_returns_dashboard_data(self, handler, mock_http_handler, mock_coordinator):
        """Should return metrics for dashboard."""
        mock_coordinator.get_stats = AsyncMock(
            return_value={
                "scheduler": {
                    "by_status": {"running": 2, "pending": 5, "completed": 10},
                    "by_type": {"audit": 3, "document_processing": 4},
                },
                "registry": {
                    "total_agents": 4,
                    "available_agents": 3,
                    "by_status": {"ready": 3, "busy": 1},
                },
            }
        )

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            result = handler.handle("/api/v1/control-plane/metrics", {}, mock_http_handler)

        assert get_status(result) == 200
        body = get_body(result)
        assert "active_jobs" in body
        assert "queued_jobs" in body
        assert "agents_available" in body
        assert body["active_jobs"] == 2
        assert body["queued_jobs"] == 5

    def test_metrics_returns_503_without_coordinator(self, handler, mock_http_handler):
        """Should return 503 when coordinator not initialized."""
        result = handler.handle("/api/v1/control-plane/metrics", {}, mock_http_handler)
        assert get_status(result) == 503


# ===========================================================================
# Test RBAC Permission Enforcement
# ===========================================================================


@pytest.fixture
def mock_member_user():
    """Create a mock member user (without admin permissions)."""
    mock_user = MagicMock()
    mock_user.user_id = "user-member"
    mock_user.role = "member"
    return mock_user


class TestRBACPermissionDenied:
    """Tests for RBAC permission enforcement - users without correct role should get 403."""

    def test_register_agent_denied_for_member(
        self, handler, mock_http_handler, mock_coordinator, mock_member_user
    ):
        """Member users should not be able to register agents."""
        mock_http_handler.command = "POST"

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_member_user, None)
            ):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"agent_id": "agent-1", "capabilities": ["debate"]}, None),
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/agents", {}, mock_http_handler
                    )

        assert get_status(result) == 403
        assert "controlplane:agents" in get_body(result)["error"]

    def test_submit_task_denied_for_member(
        self, handler, mock_http_handler, mock_coordinator, mock_member_user
    ):
        """Member users should not be able to submit tasks."""
        mock_http_handler.command = "POST"

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_member_user, None)
            ):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"task_type": "debate", "payload": {}}, None),
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/tasks", {}, mock_http_handler
                    )

        assert get_status(result) == 403
        assert "controlplane:tasks" in get_body(result)["error"]

    def test_unregister_agent_denied_for_member(
        self, handler, mock_http_handler, mock_coordinator, mock_member_user
    ):
        """Member users should not be able to unregister agents."""
        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_member_user, None)
            ):
                result = handler.handle_delete(
                    "/api/v1/control-plane/agents/agent-1", {}, mock_http_handler
                )

        assert get_status(result) == 403
        assert "controlplane:agents" in get_body(result)["error"]

    def test_user_without_role_denied(self, handler, mock_http_handler, mock_coordinator):
        """Users without a role attribute should be denied."""
        mock_http_handler.command = "POST"
        mock_user_no_role = MagicMock(spec=["user_id"])  # No role attribute
        mock_user_no_role.user_id = "user-no-role"

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(
                handler, "require_auth_or_error", return_value=(mock_user_no_role, None)
            ):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"agent_id": "agent-1"}, None),
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/agents", {}, mock_http_handler
                    )

        assert get_status(result) == 403

    def test_owner_role_allowed(self, handler, mock_http_handler, mock_coordinator):
        """Owner role should have access to all operations."""
        mock_http_handler.command = "POST"
        mock_owner = MagicMock()
        mock_owner.user_id = "user-owner"
        mock_owner.role = "owner"

        with patch.object(handler, "_get_coordinator", return_value=mock_coordinator):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_owner, None)):
                with patch.object(
                    handler,
                    "read_json_body_validated",
                    return_value=({"agent_id": "agent-1", "capabilities": ["debate"]}, None),
                ):
                    result = handler.handle_post(
                        "/api/v1/control-plane/agents", {}, mock_http_handler
                    )

        assert get_status(result) == 201  # Success - owner is allowed
