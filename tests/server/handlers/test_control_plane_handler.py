"""Tests for Control Plane handler endpoints.

Validates the REST API endpoints for control plane operations including:
- Agent registration and discovery
- Task submission and status
- Health monitoring
- Control plane statistics and metrics
"""

import json
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.control_plane import ControlPlaneHandler


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
    
    coordinator.get_stats = AsyncMock(return_value={
        "scheduler": {"by_status": {"running": 5, "pending": 10, "completed": 100}},
        "registry": {"total_agents": 10, "available_agents": 7, "by_status": {"busy": 3}},
    })
    
    # Sync methods
    coordinator.get_system_health.return_value = mock_health
    coordinator.get_agent_health.return_value = mock_health
    coordinator._health_monitor = MagicMock()
    coordinator._health_monitor.get_all_health.return_value = {"test-agent": mock_health}
    
    return coordinator


class TestControlPlaneHandlerCanHandle:
    """Test ControlPlaneHandler.can_handle method."""

    def test_can_handle_agents(self, control_plane_handler):
        """Test can_handle returns True for agents endpoint."""
        assert control_plane_handler.can_handle("/api/control-plane/agents")

    def test_can_handle_agent_by_id(self, control_plane_handler):
        """Test can_handle returns True for agent by ID."""
        assert control_plane_handler.can_handle("/api/control-plane/agents/test-agent")

    def test_can_handle_agent_heartbeat(self, control_plane_handler):
        """Test can_handle returns True for heartbeat endpoint."""
        assert control_plane_handler.can_handle("/api/control-plane/agents/test-agent/heartbeat")

    def test_can_handle_tasks(self, control_plane_handler):
        """Test can_handle returns True for tasks endpoint."""
        assert control_plane_handler.can_handle("/api/control-plane/tasks")

    def test_can_handle_task_by_id(self, control_plane_handler):
        """Test can_handle returns True for task by ID."""
        assert control_plane_handler.can_handle("/api/control-plane/tasks/task-123")

    def test_can_handle_health(self, control_plane_handler):
        """Test can_handle returns True for health endpoint."""
        assert control_plane_handler.can_handle("/api/control-plane/health")

    def test_can_handle_stats(self, control_plane_handler):
        """Test can_handle returns True for stats endpoint."""
        assert control_plane_handler.can_handle("/api/control-plane/stats")

    def test_can_handle_queue(self, control_plane_handler):
        """Test can_handle returns True for queue endpoint."""
        assert control_plane_handler.can_handle("/api/control-plane/queue")

    def test_can_handle_metrics(self, control_plane_handler):
        """Test can_handle returns True for metrics endpoint."""
        assert control_plane_handler.can_handle("/api/control-plane/metrics")

    def test_cannot_handle_unknown(self, control_plane_handler):
        """Test can_handle returns False for unknown endpoint."""
        assert not control_plane_handler.can_handle("/api/unknown")
        assert not control_plane_handler.can_handle("/api/debates")


class TestControlPlaneHandlerNoCoordinator:
    """Test handlers when coordinator is not initialized."""

    def test_list_agents_no_coordinator(self, control_plane_handler, mock_http_handler):
        """Test list agents returns 503 when coordinator not initialized."""
        result = control_plane_handler.handle(
            "/api/control-plane/agents", {}, mock_http_handler
        )
        
        assert result is not None
        assert result.status_code == 503
        body = json.loads(result.body)
        assert "error" in body
        assert "not initialized" in body["error"]

    def test_get_agent_no_coordinator(self, control_plane_handler, mock_http_handler):
        """Test get agent returns 503 when coordinator not initialized."""
        result = control_plane_handler.handle(
            "/api/control-plane/agents/test-agent", {}, mock_http_handler
        )
        
        assert result is not None
        assert result.status_code == 503

    def test_get_health_no_coordinator(self, control_plane_handler, mock_http_handler):
        """Test get health returns 503 when coordinator not initialized."""
        result = control_plane_handler.handle(
            "/api/control-plane/health", {}, mock_http_handler
        )
        
        assert result is not None
        assert result.status_code == 503

    def test_get_stats_no_coordinator(self, control_plane_handler, mock_http_handler):
        """Test get stats returns 503 when coordinator not initialized."""
        result = control_plane_handler.handle(
            "/api/control-plane/stats", {}, mock_http_handler
        )
        
        assert result is not None
        assert result.status_code == 503


class TestControlPlaneHandlerListAgents:
    """Test GET /api/control-plane/agents endpoint."""

    def test_list_agents_success(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test listing agents with coordinator."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        result = control_plane_handler.handle(
            "/api/control-plane/agents", {}, mock_http_handler
        )
        
        assert result is not None
        body = json.loads(result.body)
        assert "agents" in body
        assert "total" in body
        assert body["total"] >= 0

    def test_list_agents_with_capability_filter(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test listing agents filtered by capability."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        result = control_plane_handler.handle(
            "/api/control-plane/agents", {"capability": "debate"}, mock_http_handler
        )
        
        assert result is not None
        body = json.loads(result.body)
        assert "agents" in body


class TestControlPlaneHandlerGetAgent:
    """Test GET /api/control-plane/agents/:id endpoint."""

    def test_get_agent_success(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test getting agent by ID."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        result = control_plane_handler.handle(
            "/api/control-plane/agents/test-agent", {}, mock_http_handler
        )
        
        assert result is not None
        body = json.loads(result.body)
        assert "agent_id" in body

    def test_get_agent_not_found(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test getting non-existent agent returns 404."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.get_agent = AsyncMock(return_value=None)
        
        result = control_plane_handler.handle(
            "/api/control-plane/agents/nonexistent", {}, mock_http_handler
        )
        
        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerGetTask:
    """Test GET /api/control-plane/tasks/:id endpoint."""

    def test_get_task_success(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test getting task by ID."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        result = control_plane_handler.handle(
            "/api/control-plane/tasks/task-123", {}, mock_http_handler
        )
        
        assert result is not None
        body = json.loads(result.body)
        assert "id" in body

    def test_get_task_not_found(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test getting non-existent task returns 404."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.get_task = AsyncMock(return_value=None)
        
        result = control_plane_handler.handle(
            "/api/control-plane/tasks/nonexistent", {}, mock_http_handler
        )
        
        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerHealth:
    """Test health endpoints."""

    def test_get_system_health(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test getting system health status."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        result = control_plane_handler.handle(
            "/api/control-plane/health", {}, mock_http_handler
        )
        
        assert result is not None
        body = json.loads(result.body)
        assert "status" in body
        assert "agents" in body

    def test_get_agent_health(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test getting specific agent health."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        result = control_plane_handler.handle(
            "/api/control-plane/health/test-agent", {}, mock_http_handler
        )
        
        assert result is not None
        body = json.loads(result.body)
        assert "status" in body

    def test_get_agent_health_not_found(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test getting health for non-existent agent returns 404."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.get_agent_health.return_value = None
        
        result = control_plane_handler.handle(
            "/api/control-plane/health/nonexistent", {}, mock_http_handler
        )
        
        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerStats:
    """Test GET /api/control-plane/stats endpoint."""

    def test_get_stats(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test getting control plane statistics."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        result = control_plane_handler.handle(
            "/api/control-plane/stats", {}, mock_http_handler
        )
        
        assert result is not None
        body = json.loads(result.body)
        assert "scheduler" in body or "registry" in body


class TestControlPlaneHandlerMetrics:
    """Test GET /api/control-plane/metrics endpoint."""

    def test_get_metrics(self, control_plane_handler, mock_http_handler, mock_coordinator):
        """Test getting dashboard metrics."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        result = control_plane_handler.handle(
            "/api/control-plane/metrics", {}, mock_http_handler
        )
        
        assert result is not None
        body = json.loads(result.body)
        assert "active_jobs" in body
        assert "queued_jobs" in body
        assert "completed_jobs" in body
        assert "agents_available" in body
        assert "total_agents" in body


class TestControlPlaneHandlerRegisterAgent:
    """Test POST /api/control-plane/agents endpoint."""

    def test_register_agent_requires_auth(self, control_plane_handler, mock_coordinator):
        """Test registering agent requires authentication."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_request_body({
            "agent_id": "new-agent",
            "capabilities": ["debate"],
        })
        
        result = control_plane_handler.handle_post(
            "/api/control-plane/agents", {}, handler
        )
        
        assert result is not None
        assert result.status_code == 401

    def test_register_agent_missing_id(self, control_plane_handler, mock_coordinator):
        """Test registering agent without ID returns error."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({
            "capabilities": ["debate"],
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/agents", {}, handler
            )
        
        assert result is not None
        assert result.status_code == 400

    def test_register_agent_success(self, control_plane_handler, mock_coordinator):
        """Test successfully registering an agent."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({
            "agent_id": "new-agent",
            "capabilities": ["debate", "coding"],
            "model": "claude-3",
            "provider": "anthropic",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/agents", {}, handler
            )
        
        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert "agent_id" in body


class TestControlPlaneHandlerHeartbeat:
    """Test POST /api/control-plane/agents/:id/heartbeat endpoint."""

    def test_heartbeat_success(self, control_plane_handler, mock_coordinator):
        """Test sending heartbeat."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({
            "status": "ready",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/agents/test-agent/heartbeat", {}, handler
            )
        
        assert result is not None
        body = json.loads(result.body)
        assert body.get("acknowledged") is True

    def test_heartbeat_agent_not_found(self, control_plane_handler, mock_coordinator):
        """Test heartbeat for non-existent agent."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.heartbeat = AsyncMock(return_value=False)
        handler = create_auth_request_body({
            "status": "ready",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/agents/nonexistent/heartbeat", {}, handler
            )
        
        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerSubmitTask:
    """Test POST /api/control-plane/tasks endpoint."""

    def test_submit_task_requires_auth(self, control_plane_handler, mock_coordinator):
        """Test submitting task requires authentication."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_request_body({
            "task_type": "debate",
            "payload": {},
        })
        
        result = control_plane_handler.handle_post(
            "/api/control-plane/tasks", {}, handler
        )
        
        assert result is not None
        assert result.status_code == 401

    def test_submit_task_missing_type(self, control_plane_handler, mock_coordinator):
        """Test submitting task without type returns error."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({
            "payload": {},
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks", {}, handler
            )
        
        assert result is not None
        assert result.status_code == 400

    def test_submit_task_success(self, control_plane_handler, mock_coordinator):
        """Test successfully submitting a task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({
            "task_type": "debate",
            "payload": {"topic": "test"},
            "priority": "normal",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks", {}, handler
            )
        
        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        assert "task_id" in body

    def test_submit_task_invalid_priority(self, control_plane_handler, mock_coordinator):
        """Test submitting task with invalid priority."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({
            "task_type": "debate",
            "priority": "invalid_priority",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks", {}, handler
            )
        
        assert result is not None
        assert result.status_code == 400


class TestControlPlaneHandlerCompleteTask:
    """Test POST /api/control-plane/tasks/:id/complete endpoint."""

    def test_complete_task_success(self, control_plane_handler, mock_coordinator):
        """Test completing a task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({
            "result": {"output": "completed"},
            "agent_id": "test-agent",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks/task-123/complete", {}, handler
            )
        
        assert result is not None
        body = json.loads(result.body)
        assert body.get("completed") is True

    def test_complete_task_not_found(self, control_plane_handler, mock_coordinator):
        """Test completing non-existent task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.complete_task = AsyncMock(return_value=False)
        handler = create_auth_request_body({
            "result": {},
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks/nonexistent/complete", {}, handler
            )
        
        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerFailTask:
    """Test POST /api/control-plane/tasks/:id/fail endpoint."""

    def test_fail_task_success(self, control_plane_handler, mock_coordinator):
        """Test failing a task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({
            "error": "Task failed due to timeout",
            "agent_id": "test-agent",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks/task-123/fail", {}, handler
            )
        
        assert result is not None
        body = json.loads(result.body)
        assert body.get("failed") is True

    def test_fail_task_not_found(self, control_plane_handler, mock_coordinator):
        """Test failing non-existent task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.fail_task = AsyncMock(return_value=False)
        handler = create_auth_request_body({
            "error": "Some error",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks/nonexistent/fail", {}, handler
            )
        
        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerCancelTask:
    """Test POST /api/control-plane/tasks/:id/cancel endpoint."""

    def test_cancel_task_success(self, control_plane_handler, mock_coordinator):
        """Test canceling a task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = create_auth_request_body({})
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks/task-123/cancel", {}, handler
            )
        
        assert result is not None
        body = json.loads(result.body)
        assert body.get("cancelled") is True

    def test_cancel_task_not_found(self, control_plane_handler, mock_coordinator):
        """Test canceling non-existent task."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.cancel_task = AsyncMock(return_value=False)
        handler = create_auth_request_body({})
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks/nonexistent/cancel", {}, handler
            )
        
        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerUnregisterAgent:
    """Test DELETE /api/control-plane/agents/:id endpoint."""

    def test_unregister_agent_requires_auth(self, control_plane_handler, mock_coordinator, mock_http_handler):
        """Test unregistering agent requires authentication."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        result = control_plane_handler.handle_delete(
            "/api/control-plane/agents/test-agent", {}, mock_http_handler
        )
        
        assert result is not None
        assert result.status_code == 401

    def test_unregister_agent_success(self, control_plane_handler, mock_coordinator, mock_http_handler):
        """Test successfully unregistering an agent."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_delete(
                "/api/control-plane/agents/test-agent", {}, mock_http_handler
            )
        
        assert result is not None
        body = json.loads(result.body)
        assert body.get("unregistered") is True

    def test_unregister_agent_not_found(self, control_plane_handler, mock_coordinator, mock_http_handler):
        """Test unregistering non-existent agent."""
        ControlPlaneHandler.coordinator = mock_coordinator
        mock_coordinator.unregister_agent = AsyncMock(return_value=False)
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_delete(
                "/api/control-plane/agents/nonexistent", {}, mock_http_handler
            )
        
        assert result is not None
        assert result.status_code == 404


class TestControlPlaneHandlerIntegration:
    """Integration tests for control plane handler."""

    def test_full_agent_lifecycle(self, control_plane_handler, mock_coordinator, mock_http_handler):
        """Test full agent registration, heartbeat, unregistration lifecycle."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        # Step 1: Register agent
        register_handler = create_auth_request_body({
            "agent_id": "lifecycle-agent",
            "capabilities": ["debate"],
            "model": "claude-3",
            "provider": "anthropic",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/agents", {}, register_handler
            )
        
        assert result is not None
        assert result.status_code == 201
        
        # Step 2: Get agent
        result = control_plane_handler.handle(
            "/api/control-plane/agents/lifecycle-agent", {}, mock_http_handler
        )
        assert result is not None
        
        # Step 3: Send heartbeat
        heartbeat_handler = create_auth_request_body({"status": "ready"})
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/agents/lifecycle-agent/heartbeat", {}, heartbeat_handler
            )
        
        assert result is not None
        body = json.loads(result.body)
        assert body.get("acknowledged") is True
        
        # Step 4: Unregister agent
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_delete(
                "/api/control-plane/agents/lifecycle-agent", {}, mock_http_handler
            )
        
        assert result is not None
        body = json.loads(result.body)
        assert body.get("unregistered") is True

    def test_full_task_lifecycle(self, control_plane_handler, mock_coordinator):
        """Test full task submission, claim, completion lifecycle."""
        ControlPlaneHandler.coordinator = mock_coordinator
        
        # Step 1: Submit task
        submit_handler = create_auth_request_body({
            "task_type": "debate",
            "payload": {"topic": "AI safety"},
            "priority": "high",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                "/api/control-plane/tasks", {}, submit_handler
            )
        
        assert result is not None
        assert result.status_code == 201
        body = json.loads(result.body)
        task_id = body.get("task_id")
        assert task_id is not None
        
        # Step 2: Complete task
        complete_handler = create_auth_request_body({
            "result": {"consensus": "AI safety is important"},
            "agent_id": "test-agent",
        })
        
        with patch.object(control_plane_handler, 'require_auth_or_error', return_value=(MagicMock(), None)):
            result = control_plane_handler.handle_post(
                f"/api/control-plane/tasks/{task_id}/complete", {}, complete_handler
            )
        
        assert result is not None
        body = json.loads(result.body)
        assert body.get("completed") is True
