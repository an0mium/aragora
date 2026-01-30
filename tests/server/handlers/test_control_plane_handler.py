"""Tests for Control Plane handler endpoints.

Validates the REST API endpoints for control plane operations including:
- Agent registration and discovery
- Task submission and status
- Health monitoring
- Control plane statistics and metrics
- Policy violations management
- Notifications and audit logs

NOTE: The control_plane handler has a known issue where the `@track_handler`
decorator is applied to sync methods (`handle`, `handle_delete`, `handle_patch`).
The decorator creates an async wrapper that awaits the function, but sync functions
return values directly, causing TypeError. Tests for sync methods use direct method
calls to the underlying `_handle_*` methods to avoid this issue.

NOTE: The handler uses `_run_async` to bridge sync/async code. When running
in pytest-asyncio, we must patch `_run_async` to directly await coroutines
instead of using its internal event loop detection which fails in async contexts.
Tests that call POST/PATCH/DELETE endpoints through handle_post/handle_patch/handle_delete
use the `patch_run_async` fixture to enable async coordinator calls.
"""

import asyncio
import inspect
import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.control_plane import ControlPlaneHandler


def _test_run_async(coro):
    """Test-compatible run_async that works in async pytest contexts.

    Uses a new event loop in a thread to execute the coroutine,
    avoiding conflicts with pytest-asyncio's running loop.
    """
    import concurrent.futures

    if not inspect.iscoroutine(coro):
        return coro

    def run_in_new_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_new_loop)
        return future.result(timeout=30.0)


@pytest.fixture
def patch_run_async():
    """Patch _run_async to work in async test contexts.

    The handler uses _run_async to call async coordinator methods from sync code.
    In pytest-asyncio, we need to run the coroutine in a separate thread's event loop.
    """
    with patch("aragora.server.handlers.control_plane._run_async", side_effect=_test_run_async):
        yield


@pytest.fixture
def control_plane_handler():
    """Create a control plane handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = ControlPlaneHandler(ctx)
    # Clear class-level coordinator to ensure clean test state
    ControlPlaneHandler.coordinator = None
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    return handler


def create_request_body(data: dict) -> MagicMock:
    """Create a mock HTTP handler with a JSON body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    body = json.dumps(data).encode("utf-8")
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }
    handler.rfile = BytesIO(body)
    handler.command = "POST"
    return handler


def create_auth_request_body(data: dict) -> MagicMock:
    """Create a mock HTTP handler with a JSON body and auth token."""
    handler = create_request_body(data)
    handler.headers["Authorization"] = "Bearer test-token"
    return handler


def create_admin_user() -> MagicMock:
    """Create a mock admin user with proper role for RBAC."""
    user = MagicMock()
    user.user_id = "test-admin"
    user.id = "test-admin"
    user.role = "admin"  # Required for controlplane:* permissions
    return user


def create_regular_user() -> MagicMock:
    """Create a mock regular user without admin role."""
    user = MagicMock()
    user.user_id = "test-user"
    user.id = "test-user"
    user.role = "viewer"  # No controlplane:* permissions
    return user


@pytest.fixture
def mock_coordinator():
    """Create a mock control plane coordinator."""
    coordinator = MagicMock()

    # Mock agent data
    mock_agent = MagicMock()
    mock_agent.to_dict.return_value = {
        "agent_id": "test-agent",
        "capabilities": ["debate", "coding"],
        "model": "claude-3",
        "provider": "anthropic",
        "status": "ready",
    }

    # Mock task data
    mock_task = MagicMock()
    mock_task.id = "task-123"
    mock_task.task_type = "debate"
    mock_task.status = MagicMock(value="running")
    mock_task.priority = MagicMock(name="NORMAL")
    mock_task.started_at = 1704067200.0
    mock_task.created_at = 1704067100.0
    mock_task.metadata = {"name": "Test Task", "progress": 0.5}
    mock_task.payload = {"document_count": 5}
    mock_task.assigned_agent = "test-agent"
    mock_task.to_dict.return_value = {
        "id": "task-123",
        "task_type": "debate",
        "status": "pending",
        "payload": {},
    }

    # Mock health data
    mock_health = MagicMock()
    mock_health.to_dict.return_value = {
        "status": "healthy",
        "last_heartbeat": "2024-01-01T00:00:00Z",
    }
    mock_health.value = "healthy"

    # Set up async methods
    coordinator.list_agents = AsyncMock(return_value=[mock_agent])
    coordinator.get_agent = AsyncMock(return_value=mock_agent)
    coordinator.register_agent = AsyncMock(return_value=mock_agent)
    coordinator.unregister_agent = AsyncMock(return_value=True)
    coordinator.heartbeat = AsyncMock(return_value=True)

    coordinator.get_task = AsyncMock(return_value=mock_task)
    coordinator.submit_task = AsyncMock(return_value="task-123")
    coordinator.claim_task = AsyncMock(return_value=mock_task)
    coordinator.complete_task = AsyncMock(return_value=True)
    coordinator.fail_task = AsyncMock(return_value=True)
    coordinator.cancel_task = AsyncMock(return_value=True)

    coordinator.get_stats = AsyncMock(
        return_value={
            "scheduler": {"by_status": {"running": 5, "pending": 10, "completed": 100}},
            "registry": {"total_agents": 10, "available_agents": 7, "by_status": {"busy": 3}},
        }
    )

    # Sync methods
    coordinator.get_system_health.return_value = mock_health
    coordinator.get_agent_health.return_value = mock_health
    coordinator._health_monitor = MagicMock()
    coordinator._health_monitor.get_all_health.return_value = {"test-agent": mock_health}

    # Mock scheduler for queue operations
    mock_scheduler = MagicMock()
    mock_scheduler.list_by_status = AsyncMock(return_value=[mock_task])
    coordinator._scheduler = mock_scheduler

    return coordinator


@pytest.fixture
def mock_policy_store():
    """Create a mock policy store for violation tests."""
    store = MagicMock()
    store.list_violations.return_value = [
        {
            "id": "violation-1",
            "policy_id": "policy-1",
            "policy_name": "Test Policy",
            "violation_type": "agent_blocklist",
            "description": "Agent is blocklisted",
            "task_id": "task-1",
            "task_type": "debate",
            "agent_id": "blocked-agent",
            "region": "us-east-1",
            "workspace_id": "ws-1",
            "enforcement_level": "hard",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "open",
            "resolved_at": None,
            "resolved_by": None,
            "resolution_notes": None,
            "metadata": {},
        }
    ]
    store.count_violations.return_value = {"agent_blocklist": 5, "region_constraint": 3}
    store.update_violation_status.return_value = True
    return store


class TestControlPlaneHandlerCanHandle:
    """Test ControlPlaneHandler.can_handle method."""

    def test_can_handle_agents(self, control_plane_handler):
        """Test can_handle returns True for agents endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/agents")

    def test_can_handle_agent_by_id(self, control_plane_handler):
        """Test can_handle returns True for agent by ID."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/agents/test-agent")

    def test_can_handle_agent_heartbeat(self, control_plane_handler):
        """Test can_handle returns True for heartbeat endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/agents/test-agent/heartbeat")

    def test_can_handle_tasks(self, control_plane_handler):
        """Test can_handle returns True for tasks endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/tasks")

    def test_can_handle_task_by_id(self, control_plane_handler):
        """Test can_handle returns True for task by ID."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/tasks/task-123")

    def test_can_handle_health(self, control_plane_handler):
        """Test can_handle returns True for health endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/health")

    def test_can_handle_stats(self, control_plane_handler):
        """Test can_handle returns True for stats endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/stats")

    def test_can_handle_queue(self, control_plane_handler):
        """Test can_handle returns True for queue endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/queue")

    def test_can_handle_metrics(self, control_plane_handler):
        """Test can_handle returns True for metrics endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/metrics")

    def test_can_handle_notifications(self, control_plane_handler):
        """Test can_handle returns True for notifications endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/notifications")

    def test_can_handle_audit(self, control_plane_handler):
        """Test can_handle returns True for audit endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/audit")

    def test_can_handle_violations(self, control_plane_handler):
        """Test can_handle returns True for violations endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/policies/violations")

    def test_can_handle_breakers(self, control_plane_handler):
        """Test can_handle returns True for circuit breakers endpoint."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/breakers")

    def test_cannot_handle_unknown(self, control_plane_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not control_plane_handler.can_handle("/api/v1/unknown")
        assert not control_plane_handler.can_handle("/api/v1/debates")


class TestControlPlanePathNormalization:
    """Test path normalization for versioned endpoints."""

    def test_normalize_path_v1_to_legacy(self, control_plane_handler):
        """Test path normalization converts v1 to legacy format."""
        result = control_plane_handler._normalize_path("/api/v1/control-plane/agents")
        assert result == "/api/control-plane/agents"

    def test_normalize_path_v1_base(self, control_plane_handler):
        """Test path normalization for v1 base path."""
        result = control_plane_handler._normalize_path("/api/v1/control-plane")
        assert result == "/api/control-plane"

    def test_normalize_path_legacy_unchanged(self, control_plane_handler):
        """Test legacy paths remain unchanged."""
        result = control_plane_handler._normalize_path("/api/control-plane/agents")
        assert result == "/api/control-plane/agents"


class TestControlPlaneHandlerListAgents:
    """Test _handle_list_agents method."""

    def test_list_agents_no_coordinator(self, control_plane_handler):
        """Test list agents returns 503 when coordinator not initialized."""
        result = control_plane_handler._handle_list_agents({})

        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body
        assert "not initialized" in body["error"]

    def test_list_agents_success(self, control_plane_handler, mock_coordinator):
        """Test listing agents with coordinator."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_list_agents({})

        assert result is not None
        body = json.loads(result.body)
        assert "agents" in body
        assert "total" in body
        assert body["total"] >= 0

    def test_list_agents_with_capability_filter(self, control_plane_handler, mock_coordinator):
        """Test listing agents filtered by capability."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_list_agents({"capability": "debate"})

        assert result is not None
        body = json.loads(result.body)
        assert "agents" in body

    def test_list_agents_with_available_filter(self, control_plane_handler, mock_coordinator):
        """Test listing agents with available filter set to false."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_list_agents({"available": "false"})

        assert result is not None
        body = json.loads(result.body)
        assert "agents" in body

    def test_list_agents_error_handling(self, control_plane_handler, mock_coordinator):
        """Test listing agents handles errors gracefully."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.list_agents = AsyncMock(side_effect=ValueError("Invalid capability"))

        result = control_plane_handler._handle_list_agents({})

        assert result is not None
        assert result.status_code == 400


class TestControlPlaneHandlerGetAgent:
    """Test _handle_get_agent method."""

    def test_get_agent_no_coordinator(self, control_plane_handler):
        """Test get agent returns 503 when coordinator not initialized."""
        result = control_plane_handler._handle_get_agent("test-agent")

        assert result is not None
        assert result.status_code == 503

    def test_get_agent_success(self, control_plane_handler, mock_coordinator):
        """Test getting agent by ID."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_get_agent("test-agent")

        assert result is not None
        body = json.loads(result.body)
        assert "agent_id" in body

    def test_get_agent_not_found(self, control_plane_handler, mock_coordinator):
        """Test getting non-existent agent returns 404."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.get_agent = AsyncMock(return_value=None)

        result = control_plane_handler._handle_get_agent("nonexistent")

        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerGetTask:
    """Test _handle_get_task method."""

    def test_get_task_no_coordinator(self, control_plane_handler):
        """Test get task returns 503 when coordinator not initialized."""
        result = control_plane_handler._handle_get_task("task-123")

        assert result is not None
        assert result.status_code == 503

    def test_get_task_success(self, control_plane_handler, mock_coordinator):
        """Test getting task by ID."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_get_task("task-123")

        assert result is not None
        body = json.loads(result.body)
        assert "id" in body

    def test_get_task_not_found(self, control_plane_handler, mock_coordinator):
        """Test getting non-existent task returns 404."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.get_task = AsyncMock(return_value=None)

        result = control_plane_handler._handle_get_task("nonexistent")

        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerHealth:
    """Test health endpoint methods."""

    def test_get_system_health_no_coordinator(self, control_plane_handler):
        """Test get health returns 503 when coordinator not initialized."""
        result = control_plane_handler._handle_system_health()

        assert result is not None
        assert result.status_code == 503

    def test_get_system_health(self, control_plane_handler, mock_coordinator):
        """Test getting system health status."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_system_health()

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body
        assert "agents" in body

    def test_get_agent_health(self, control_plane_handler, mock_coordinator):
        """Test getting specific agent health."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_agent_health("test-agent")

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body

    def test_get_agent_health_not_found(self, control_plane_handler, mock_coordinator):
        """Test getting health for non-existent agent returns 404."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.get_agent_health.return_value = None

        result = control_plane_handler._handle_agent_health("nonexistent")

        assert result is not None
        assert result.status_code == 404

    def test_get_detailed_health_no_coordinator(self, control_plane_handler):
        """Test detailed health without coordinator."""
        result = control_plane_handler._handle_detailed_health()

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body
        assert "components" in body
        assert "uptime_seconds" in body
        assert "version" in body

    def test_get_detailed_health_with_coordinator(self, control_plane_handler, mock_coordinator):
        """Test detailed health with coordinator returns components."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_detailed_health()

        assert result is not None
        body = json.loads(result.body)
        assert "status" in body
        assert "components" in body
        # Should include Coordinator and Scheduler components
        component_names = [c["name"] for c in body["components"]]
        assert "Coordinator" in component_names


class TestControlPlaneHandlerCircuitBreakers:
    """Test circuit breaker endpoint."""

    def test_get_circuit_breakers_empty(self, control_plane_handler):
        """Test getting circuit breakers when none configured."""
        result = control_plane_handler._handle_circuit_breakers()

        assert result is not None
        body = json.loads(result.body)
        assert "breakers" in body
        assert isinstance(body["breakers"], list)

    def test_get_circuit_breakers_with_breakers(self, control_plane_handler):
        """Test getting circuit breakers with configured breakers."""
        mock_breaker = MagicMock()
        mock_breaker.state = MagicMock(value="closed")
        mock_breaker.failure_count = 2
        mock_breaker.success_count = 10
        mock_breaker.last_failure_time = None
        mock_breaker.reset_timeout = 30

        with patch(
            "aragora.resilience.get_circuit_breakers",
            return_value={"agent_calls": mock_breaker},
        ):
            result = control_plane_handler._handle_circuit_breakers()

        assert result is not None
        body = json.loads(result.body)
        assert "breakers" in body
        assert len(body["breakers"]) == 1
        assert body["breakers"][0]["name"] == "agent_calls"
        assert body["breakers"][0]["state"] == "closed"


class TestControlPlaneHandlerStats:
    """Test _handle_stats method."""

    def test_get_stats_no_coordinator(self, control_plane_handler):
        """Test get stats returns 503 when coordinator not initialized."""
        result = control_plane_handler._handle_stats()

        assert result is not None
        assert result.status_code == 503

    def test_get_stats(self, control_plane_handler, mock_coordinator):
        """Test getting control plane statistics."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_stats()

        assert result is not None
        body = json.loads(result.body)
        assert "scheduler" in body or "registry" in body


class TestControlPlaneHandlerMetrics:
    """Test _handle_get_metrics method."""

    def test_get_metrics_no_coordinator(self, control_plane_handler):
        """Test get metrics returns 503 when coordinator not initialized."""
        result = control_plane_handler._handle_get_metrics()

        assert result is not None
        assert result.status_code == 503

    def test_get_metrics(self, control_plane_handler, mock_coordinator):
        """Test getting dashboard metrics."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_get_metrics()

        assert result is not None
        body = json.loads(result.body)
        assert "active_jobs" in body
        assert "queued_jobs" in body
        assert "completed_jobs" in body
        assert "agents_available" in body
        assert "total_agents" in body


class TestControlPlaneHandlerQueue:
    """Test queue endpoint methods."""

    def test_get_queue_no_coordinator(self, control_plane_handler):
        """Test get queue returns 503 when coordinator not initialized."""
        result = control_plane_handler._handle_get_queue({})

        assert result is not None
        assert result.status_code == 503

    def test_get_queue(self, control_plane_handler, mock_coordinator):
        """Test getting current job queue."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_get_queue({})

        assert result is not None
        body = json.loads(result.body)
        assert "jobs" in body
        assert "total" in body

    def test_get_queue_with_limit(self, control_plane_handler, mock_coordinator):
        """Test getting queue with custom limit."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_get_queue({"limit": "10"})

        assert result is not None
        body = json.loads(result.body)
        assert "jobs" in body

    def test_get_queue_metrics(self, control_plane_handler, mock_coordinator):
        """Test getting queue performance metrics."""
        ControlPlaneHandler.coordinator = mock_coordinator

        result = control_plane_handler._handle_queue_metrics()

        assert result is not None
        body = json.loads(result.body)
        assert "pending" in body
        assert "running" in body
        assert "completed_today" in body

    def test_get_queue_metrics_no_coordinator(self, control_plane_handler):
        """Test queue metrics without coordinator returns defaults."""
        result = control_plane_handler._handle_queue_metrics()

        assert result is not None
        body = json.loads(result.body)
        assert body["pending"] == 0
        assert body["running"] == 0


class TestControlPlaneHandlerRegisterAgent:
    """Test POST /api/control-plane/agents endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_register_agent_requires_auth(self, control_plane_handler, mock_coordinator):
        """Test registering agent requires authentication."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_request_body(
            {
                "agent_id": "new-agent",
                "capabilities": ["debate"],
            }
        )

        result = await control_plane_handler.handle_post(
            "/api/v1/control-plane/agents", {}, handler
        )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_register_agent_missing_id(self, control_plane_handler, mock_coordinator):
        """Test registering agent without ID returns error."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "capabilities": ["debate"],
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/agents", {}, handler
            )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_register_agent_success(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test successfully registering an agent."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "agent_id": "new-agent",
                "capabilities": ["debate", "coding"],
                "model": "claude-3",
                "provider": "anthropic",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/agents", {}, handler
            )

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert "agent_id" in body

    @pytest.mark.asyncio
    async def test_register_agent_permission_denied(self, control_plane_handler, mock_coordinator):
        """Test registering agent without proper permission."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "agent_id": "new-agent",
                "capabilities": ["debate"],
            }
        )

        with patch.object(
            control_plane_handler,
            "require_auth_or_error",
            return_value=(create_regular_user(), None),
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/agents", {}, handler
            )

        assert result is not None
        assert result.status_code == 403


class TestControlPlaneHandlerHeartbeat:
    """Test POST /api/control-plane/agents/:id/heartbeat endpoint."""

    @pytest.mark.asyncio
    async def test_heartbeat_success(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test sending heartbeat."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "status": "ready",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/agents/test-agent/heartbeat", {}, handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert body.get("acknowledged") is True

    @pytest.mark.asyncio
    async def test_heartbeat_agent_not_found(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test heartbeat for non-existent agent."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.heartbeat = AsyncMock(return_value=False)
        handler = create_auth_request_body(
            {
                "status": "ready",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/agents/nonexistent/heartbeat", {}, handler
            )

        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerSubmitTask:
    """Test POST /api/control-plane/tasks endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_submit_task_requires_auth(self, control_plane_handler, mock_coordinator):
        """Test submitting task requires authentication."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_request_body(
            {
                "task_type": "debate",
                "payload": {},
            }
        )

        result = await control_plane_handler.handle_post("/api/v1/control-plane/tasks", {}, handler)

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_submit_task_missing_type(self, control_plane_handler, mock_coordinator):
        """Test submitting task without type returns error."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "payload": {},
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks", {}, handler
            )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_task_success(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test successfully submitting a task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "task_type": "debate",
                "payload": {"topic": "test"},
                "priority": "normal",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks", {}, handler
            )

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert "task_id" in body

    @pytest.mark.asyncio
    async def test_submit_task_invalid_priority(self, control_plane_handler, mock_coordinator):
        """Test submitting task with invalid priority."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "task_type": "debate",
                "priority": "invalid_priority",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks", {}, handler
            )

        assert result is not None
        assert result.status_code == 400


class TestControlPlaneHandlerClaimTask:
    """Test POST /api/control-plane/tasks/claim endpoint."""

    @pytest.mark.asyncio
    async def test_claim_task_success(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test claiming a task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "agent_id": "test-agent",
                "capabilities": ["debate"],
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/task-123/claim", {}, handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert "task" in body

    @pytest.mark.asyncio
    async def test_claim_task_missing_agent_id(self, control_plane_handler, mock_coordinator):
        """Test claiming task without agent_id returns error."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "capabilities": ["debate"],
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/task-123/claim", {}, handler
            )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_claim_task_no_available_tasks(self, control_plane_handler, mock_coordinator):
        """Test claiming when no tasks available."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.claim_task = AsyncMock(return_value=None)
        handler = create_auth_request_body(
            {
                "agent_id": "test-agent",
                "capabilities": ["debate"],
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/task-123/claim", {}, handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert body.get("task") is None


class TestControlPlaneHandlerDeliberations:
    """Test deliberation endpoints."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_submit_deliberation_requires_auth(
        self, control_plane_handler, mock_coordinator, monkeypatch
    ):
        """Test submitting a deliberation requires authentication."""
        monkeypatch.setenv("ARAGORA_TEST_REAL_AUTH", "true")
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_request_body({"content": "Test deliberation"})

        result = await control_plane_handler.handle_post(
            "/api/v1/control-plane/deliberations", {}, handler
        )

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_submit_deliberation_missing_content(
        self, control_plane_handler, mock_coordinator
    ):
        """Test submitting a deliberation without content returns error."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({"decision_type": "debate"})

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/deliberations", {}, handler
            )

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_submit_deliberation_async_success(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test submitting an async deliberation succeeds."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "request_id": "req-123",
                "content": "Test deliberation",
                "decision_type": "debate",
                "async": True,
            }
        )
        auth_ctx = MagicMock(authenticated=True, user_id="user-1", org_id="org-1")

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            with patch("aragora.billing.auth.extract_user_from_request", return_value=auth_ctx):
                result = await control_plane_handler.handle_post(
                    "/api/v1/control-plane/deliberations", {}, handler
                )

        assert result is not None
        assert result.status_code == 202
        body = json.loads(result.body)
        assert body.get("request_id") == "req-123"
        assert body.get("status") == "queued"
        assert body.get("task_id") == "task-123"

    @pytest.mark.asyncio
    async def test_submit_deliberation_sync_success(self, control_plane_handler, mock_coordinator):
        """Test submitting a sync deliberation succeeds."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "request_id": "req-sync-1",
                "content": "Test deliberation",
                "decision_type": "debate",
            }
        )
        auth_ctx = MagicMock(authenticated=True, user_id="user-1", org_id="org-1")
        result_payload = MagicMock(
            success=True,
            decision_type=MagicMock(value="debate"),
            answer="Approved",
            confidence=0.82,
            consensus_reached=True,
            reasoning="Consensus achieved.",
            evidence_used=["doc-1"],
            duration_seconds=2.1,
            error=None,
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            with patch("aragora.billing.auth.extract_user_from_request", return_value=auth_ctx):
                with patch(
                    "aragora.control_plane.deliberation.run_deliberation",
                    new=AsyncMock(return_value=result_payload),
                ):
                    result = await control_plane_handler.handle_post(
                        "/api/v1/control-plane/deliberations", {}, handler
                    )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("status") == "completed"
        assert body.get("decision_type") == "debate"

    def test_get_deliberation_success(self, control_plane_handler, mock_http_handler):
        """Test fetching deliberation results."""
        mock_http_handler.headers = {"Authorization": "Bearer test-token"}

        with patch(
            "aragora.core.decision_results.get_decision_result",
            return_value={"request_id": "req-123", "status": "completed"},
        ):
            result = control_plane_handler._handle_get_deliberation("req-123", mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("request_id") == "req-123"

    def test_get_deliberation_not_found(self, control_plane_handler, mock_http_handler):
        """Test fetching non-existent deliberation returns 404."""
        mock_http_handler.headers = {"Authorization": "Bearer test-token"}

        with patch(
            "aragora.core.decision_results.get_decision_result",
            return_value=None,
        ):
            result = control_plane_handler._handle_get_deliberation(
                "nonexistent", mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_get_deliberation_status_success(self, control_plane_handler, mock_http_handler):
        """Test fetching deliberation status."""
        mock_http_handler.headers = {"Authorization": "Bearer test-token"}

        with patch(
            "aragora.core.decision_results.get_decision_status",
            return_value={"request_id": "req-123", "status": "queued"},
        ):
            result = control_plane_handler._handle_get_deliberation_status(
                "req-123", mock_http_handler
            )

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body.get("status") == "queued"


class TestControlPlaneHandlerCompleteTask:
    """Test POST /api/control-plane/tasks/:id/complete endpoint."""

    @pytest.mark.asyncio
    async def test_complete_task_success(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test completing a task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "result": {"output": "completed"},
                "agent_id": "test-agent",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/task-123/complete", {}, handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert body.get("completed") is True

    @pytest.mark.asyncio
    async def test_complete_task_not_found(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test completing non-existent task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.complete_task = AsyncMock(return_value=False)
        handler = create_auth_request_body(
            {
                "result": {},
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/nonexistent/complete", {}, handler
            )

        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerFailTask:
    """Test POST /api/control-plane/tasks/:id/fail endpoint."""

    @pytest.mark.asyncio
    async def test_fail_task_success(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test failing a task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "error": "Task failed due to timeout",
                "agent_id": "test-agent",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/task-123/fail", {}, handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert body.get("failed") is True

    @pytest.mark.asyncio
    async def test_fail_task_not_found(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test failing non-existent task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.fail_task = AsyncMock(return_value=False)
        handler = create_auth_request_body(
            {
                "error": "Some error",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/nonexistent/fail", {}, handler
            )

        assert result is not None
        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_fail_task_with_requeue(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test failing a task with requeue option."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body(
            {
                "error": "Temporary failure",
                "requeue": True,
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/task-123/fail", {}, handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert body.get("failed") is True


class TestControlPlaneHandlerCancelTask:
    """Test POST /api/control-plane/tasks/:id/cancel endpoint."""

    @pytest.mark.asyncio
    async def test_cancel_task_success(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test canceling a task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({})

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/task-123/cancel", {}, handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert body.get("cancelled") is True

    @pytest.mark.asyncio
    async def test_cancel_task_not_found(
        self, control_plane_handler, mock_coordinator, patch_run_async
    ):
        """Test canceling non-existent task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.cancel_task = AsyncMock(return_value=False)
        handler = create_auth_request_body({})

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks/nonexistent/cancel", {}, handler
            )

        assert result is not None
        assert result.status_code == 404


@pytest.mark.usefixtures("patch_run_async")
class TestControlPlaneHandlerUnregisterAgent:
    """Test _handle_unregister_agent method."""

    def test_unregister_agent_no_coordinator(self, control_plane_handler, mock_http_handler):
        """Test unregistering agent without coordinator returns 503."""
        mock_http_handler.headers = {"Authorization": "Bearer test-token"}

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = control_plane_handler._handle_unregister_agent("test-agent", mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_unregister_agent_success(
        self, control_plane_handler, mock_coordinator, mock_http_handler
    ):
        """Test successfully unregistering an agent."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_http_handler.headers = {"Authorization": "Bearer test-token"}

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = control_plane_handler._handle_unregister_agent("test-agent", mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert body.get("unregistered") is True

    def test_unregister_agent_not_found(
        self, control_plane_handler, mock_coordinator, mock_http_handler
    ):
        """Test unregistering non-existent agent."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.unregister_agent = AsyncMock(return_value=False)
        mock_http_handler.headers = {"Authorization": "Bearer test-token"}

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = control_plane_handler._handle_unregister_agent(
                "nonexistent", mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerNotifications:
    """Test notification endpoint methods."""

    def test_get_notifications_no_manager(self, control_plane_handler):
        """Test getting notifications when manager not configured."""
        result = control_plane_handler._handle_get_notifications({})

        assert result is not None
        body = json.loads(result.body)
        assert "notifications" in body
        assert "message" in body
        assert "not configured" in body["message"]

    def test_get_notifications_with_manager(self, control_plane_handler):
        """Test getting notifications with manager configured."""
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {"total_sent": 100, "successful": 95}
        control_plane_handler.ctx["notification_manager"] = mock_manager

        result = control_plane_handler._handle_get_notifications({})

        assert result is not None
        body = json.loads(result.body)
        assert "stats" in body

    def test_get_notification_stats_no_manager(self, control_plane_handler):
        """Test getting notification stats without manager."""
        result = control_plane_handler._handle_get_notification_stats()

        assert result is not None
        body = json.loads(result.body)
        assert body["total_sent"] == 0
        assert body["successful"] == 0

    def test_get_notification_stats_with_manager(self, control_plane_handler):
        """Test getting notification stats with manager."""
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "total_sent": 100,
            "successful": 95,
            "failed": 5,
            "by_channel": {"slack": 50, "email": 50},
        }
        control_plane_handler.ctx["notification_manager"] = mock_manager

        result = control_plane_handler._handle_get_notification_stats()

        assert result is not None
        body = json.loads(result.body)
        assert body["total_sent"] == 100
        assert body["successful"] == 95


@pytest.mark.usefixtures("patch_run_async")
class TestControlPlaneHandlerAuditLogs:
    """Test audit log endpoint methods."""

    def test_get_audit_logs_no_log(self, control_plane_handler, mock_http_handler):
        """Test getting audit logs when log not configured."""
        mock_http_handler.headers = {"Authorization": "Bearer test-token"}

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = control_plane_handler._handle_get_audit_logs({}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "entries" in body
        assert body["total"] == 0
        assert "not configured" in body.get("message", "")

    def test_get_audit_logs_with_log(self, control_plane_handler, mock_http_handler):
        """Test getting audit logs with log configured."""
        mock_entry = MagicMock()
        mock_entry.to_dict.return_value = {
            "id": "entry-1",
            "action": "task_created",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        mock_audit_log = MagicMock()
        mock_audit_log.query = AsyncMock(return_value=[mock_entry])
        control_plane_handler.ctx["audit_log"] = mock_audit_log
        mock_http_handler.headers = {"Authorization": "Bearer test-token"}

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = control_plane_handler._handle_get_audit_logs({}, mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "entries" in body
        assert body["total"] == 1

    def test_get_audit_stats_no_log(self, control_plane_handler):
        """Test getting audit stats without log configured."""
        result = control_plane_handler._handle_get_audit_stats()

        assert result is not None
        body = json.loads(result.body)
        assert body["total_entries"] == 0
        assert body["storage_backend"] == "none"

    def test_verify_audit_integrity(self, control_plane_handler, mock_http_handler):
        """Test verifying audit log integrity."""
        mock_audit_log = MagicMock()
        mock_audit_log.verify_integrity = AsyncMock(return_value=True)
        control_plane_handler.ctx["audit_log"] = mock_audit_log
        mock_http_handler.headers = {"Authorization": "Bearer test-token"}

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = control_plane_handler._handle_verify_audit_integrity(
                    {"start_seq": "0"}, mock_http_handler
                )

        assert result is not None
        body = json.loads(result.body)
        assert body["valid"] is True


@pytest.mark.usefixtures("patch_run_async")
class TestControlPlaneHandlerPolicyViolations:
    """Test policy violation endpoint methods."""

    def test_list_violations_no_store(self, control_plane_handler, mock_http_handler):
        """Test listing violations when store not available."""
        with patch.object(control_plane_handler, "_get_policy_store", return_value=None):
            result = control_plane_handler._handle_list_policy_violations({}, mock_http_handler)

        assert result is not None
        assert result.status_code == 503

    def test_list_violations_success(
        self, control_plane_handler, mock_http_handler, mock_policy_store
    ):
        """Test listing violations successfully."""
        with patch.object(
            control_plane_handler, "_get_policy_store", return_value=mock_policy_store
        ):
            result = control_plane_handler._handle_list_policy_violations({}, mock_http_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "violations" in body
        assert body["total"] == 1

    def test_list_violations_with_filters(
        self, control_plane_handler, mock_http_handler, mock_policy_store
    ):
        """Test listing violations with query filters."""
        with patch.object(
            control_plane_handler, "_get_policy_store", return_value=mock_policy_store
        ):
            result = control_plane_handler._handle_list_policy_violations(
                {"status": "open", "policy_id": "policy-1"},
                mock_http_handler,
            )

        assert result is not None
        body = json.loads(result.body)
        assert "violations" in body

    def test_get_violation_success(
        self, control_plane_handler, mock_http_handler, mock_policy_store
    ):
        """Test getting a specific violation."""
        with patch.object(
            control_plane_handler, "_get_policy_store", return_value=mock_policy_store
        ):
            result = control_plane_handler._handle_get_policy_violation(
                "violation-1", mock_http_handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert "violation" in body

    def test_get_violation_not_found(
        self, control_plane_handler, mock_http_handler, mock_policy_store
    ):
        """Test getting non-existent violation returns 404."""
        mock_policy_store.list_violations.return_value = []

        with patch.object(
            control_plane_handler, "_get_policy_store", return_value=mock_policy_store
        ):
            result = control_plane_handler._handle_get_policy_violation(
                "nonexistent", mock_http_handler
            )

        assert result is not None
        assert result.status_code == 404

    def test_get_violation_stats(self, control_plane_handler, mock_http_handler, mock_policy_store):
        """Test getting violation statistics."""
        with patch.object(
            control_plane_handler, "_get_policy_store", return_value=mock_policy_store
        ):
            result = control_plane_handler._handle_get_policy_violation_stats(mock_http_handler)

        assert result is not None
        body = json.loads(result.body)
        assert "total" in body
        assert "open" in body
        assert "by_type" in body


@pytest.mark.usefixtures("patch_run_async")
class TestControlPlaneHandlerIntegration:
    """Integration tests for control plane handler."""

    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(
        self, control_plane_handler, mock_coordinator, mock_http_handler
    ):
        """Test full agent registration, heartbeat, unregistration lifecycle."""
        ControlPlaneHandler.coordinator = mock_coordinator

        # Step 1: Register agent
        register_handler = create_auth_request_body(
            {
                "agent_id": "lifecycle-agent",
                "capabilities": ["debate"],
                "model": "claude-3",
                "provider": "anthropic",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/agents", {}, register_handler
            )

        assert result is not None
        assert result.status_code == 201

        # Step 2: Get agent (using internal method to avoid decorator issue)
        result = control_plane_handler._handle_get_agent("lifecycle-agent")
        assert result is not None

        # Step 3: Send heartbeat
        heartbeat_handler = create_auth_request_body({"status": "ready"})

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/agents/lifecycle-agent/heartbeat", {}, heartbeat_handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert body.get("acknowledged") is True

        # Step 4: Unregister agent (using internal method to avoid decorator issue)
        result = control_plane_handler._handle_unregister_agent("lifecycle-agent")

        assert result is not None
        body = json.loads(result.body)
        assert body.get("unregistered") is True

    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self, control_plane_handler, mock_coordinator):
        """Test full task submission, claim, completion lifecycle."""
        ControlPlaneHandler.coordinator = mock_coordinator

        # Step 1: Submit task
        submit_handler = create_auth_request_body(
            {
                "task_type": "debate",
                "payload": {"topic": "AI safety"},
                "priority": "high",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                "/api/v1/control-plane/tasks", {}, submit_handler
            )

        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        task_id = body.get("task_id")
        assert task_id is not None

        # Step 2: Complete task
        complete_handler = create_auth_request_body(
            {
                "result": {"consensus": "AI safety is important"},
                "agent_id": "test-agent",
            }
        )

        with patch.object(
            control_plane_handler, "require_auth_or_error", return_value=(create_admin_user(), None)
        ):
            result = await control_plane_handler.handle_post(
                f"/api/v1/control-plane/tasks/{task_id}/complete", {}, complete_handler
            )

        assert result is not None
        body = json.loads(result.body)
        assert body.get("completed") is True


class TestControlPlaneHandlerEmitEvent:
    """Test _emit_event retry logic."""

    def test_emit_event_success_first_try(self, control_plane_handler):
        """Test successful emission on first try."""
        mock_stream = MagicMock()
        mock_stream.emit_task_created = AsyncMock()
        control_plane_handler.ctx["control_plane_stream"] = mock_stream

        control_plane_handler._emit_event("emit_task_created", {"task_id": "123"})

        mock_stream.emit_task_created.assert_called_once_with({"task_id": "123"})

    def test_emit_event_no_stream(self, control_plane_handler):
        """Test graceful handling when no stream configured."""
        control_plane_handler.ctx.pop("control_plane_stream", None)

        # Should not raise
        control_plane_handler._emit_event("emit_task_created", {"task_id": "123"})

    def test_emit_event_method_not_found(self, control_plane_handler):
        """Test graceful handling when emit method doesn't exist."""
        mock_stream = MagicMock(spec=[])  # No methods
        control_plane_handler.ctx["control_plane_stream"] = mock_stream

        # Should not raise
        control_plane_handler._emit_event("nonexistent_method", {"data": "test"})


class TestControlPlaneStartupWiring:
    """Test Control Plane coordinator startup wiring."""

    @pytest.mark.asyncio
    async def test_init_control_plane_coordinator(self):
        """Test init_control_plane_coordinator function."""
        from aragora.server.startup import init_control_plane_coordinator

        # This may return None if Redis is not available
        coordinator = await init_control_plane_coordinator()

        # Either we get a coordinator or None (Redis not available)
        # In CI/local dev without Redis, this should gracefully return None
        assert coordinator is None or hasattr(coordinator, "register_agent")

    @pytest.mark.asyncio
    async def test_startup_sequence_includes_coordinator(self):
        """Test that run_startup_sequence includes control_plane_coordinator."""
        from aragora.server.startup import run_startup_sequence

        with patch(
            "aragora.server.startup.validate_backend_connectivity",
            new=AsyncMock(
                return_value={
                    "valid": True,
                    "redis": {"connected": True, "message": "mock"},
                    "database": {"connected": True, "message": "mock"},
                    "errors": [],
                }
            ),
        ):
            status = await run_startup_sequence(nomic_dir=None, stream_emitter=None)

        # Should include control_plane_coordinator key
        assert "control_plane_coordinator" in status

    def test_handler_class_level_coordinator(self):
        """Test that handler can access class-level coordinator."""
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        # Verify class has coordinator attribute
        assert hasattr(ControlPlaneHandler, "coordinator")

        # Test setting coordinator at class level
        mock_coordinator = MagicMock()
        ControlPlaneHandler.coordinator = mock_coordinator

        ctx = {"storage": None}
        handler = ControlPlaneHandler(ctx)

        # _get_coordinator should return the class-level coordinator
        result = handler._get_coordinator()
        assert result is mock_coordinator

        # Clean up
        ControlPlaneHandler.coordinator = None


class TestControlPlaneHandlerErrorHandling:
    """Test error handling scenarios."""

    def test_runtime_error_handling(self, control_plane_handler, mock_coordinator):
        """Test runtime error returns 503."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.list_agents = AsyncMock(side_effect=RuntimeError("Connection lost"))

        result = control_plane_handler._handle_list_agents({})

        assert result is not None
        assert result.status_code == 503

    def test_timeout_error_handling(self, control_plane_handler, mock_coordinator):
        """Test timeout error returns 503."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.list_agents = AsyncMock(side_effect=TimeoutError("Request timed out"))

        result = control_plane_handler._handle_list_agents({})

        assert result is not None
        assert result.status_code == 503

    def test_generic_error_handling(self, control_plane_handler, mock_coordinator):
        """Test generic exception returns 500."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.list_agents = AsyncMock(side_effect=Exception("Unexpected error"))

        result = control_plane_handler._handle_list_agents({})

        assert result is not None
        assert result.status_code == 500
