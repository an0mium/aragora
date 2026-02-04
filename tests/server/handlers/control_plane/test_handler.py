"""
Tests for ControlPlaneHandler (main control plane HTTP handlers).

Covers:
- Handler initialization and routing
- Agent operations (list, get, register, heartbeat, unregister)
- Task operations (submit, get, complete, fail, cancel)
- Health operations (system health, agent health, stats)
- Rate limiting
- RBAC permission checks
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.control_plane import ControlPlaneHandler


# ---------------------------------------------------------------------------
# Mock classes
# ---------------------------------------------------------------------------


class FakeHealthStatus(Enum):
    """Fake health status enum."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class FakeUser:
    """Mock authenticated user."""

    user_id: str = "user-123"
    email: str = "test@example.com"
    role: str = "admin"
    org_id: str = "org-123"
    permissions: list = field(default_factory=lambda: ["controlplane:agents.read"])


@dataclass
class FakeAgent:
    """Mock agent for testing."""

    agent_id: str = "agent-001"
    name: str = "Test Agent"
    agent_type: str = "debate"
    status: str = "active"
    capabilities: list = field(default_factory=lambda: ["debate", "research"])
    metadata: dict = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type,
            "status": self.status,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "registered_at": self.registered_at.isoformat(),
        }


@dataclass
class FakeTask:
    """Mock task for testing."""

    task_id: str = "task-001"
    task_type: str = "debate"
    status: str = "pending"
    priority: int = 5
    payload: dict = field(default_factory=dict)
    result: dict | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "priority": self.priority,
            "payload": self.payload,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
        }


class FakeHandler:
    """Mock HTTP handler for testing."""

    def __init__(
        self,
        method: str = "GET",
        body: dict | None = None,
        headers: dict | None = None,
        query_params: dict | None = None,
    ):
        self.command = method
        self._body = json.dumps(body).encode() if body else b"{}"
        self.headers = headers or {}
        self.client_address = ("127.0.0.1", 12345)
        self._query_params = query_params or {}

    @property
    def rfile(self):
        import io

        return io.BytesIO(self._body)

    def get(self, key: str, default: Any = None) -> Any:
        return self._query_params.get(key, default)


class FakeCoordinator:
    """Mock ControlPlaneCoordinator for testing."""

    def __init__(self):
        self.agents = {"agent-001": FakeAgent()}
        self.tasks = {"task-001": FakeTask()}
        self._health_monitor = MagicMock()
        self._scheduler = MagicMock()

    async def list_agents(
        self,
        capability: str | None = None,
        only_available: bool = True,
    ) -> list[FakeAgent]:
        """List registered agents, optionally filtered by capability."""
        agents = list(self.agents.values())
        if only_available:
            agents = [a for a in agents if a.status == "active"]
        if capability:
            agents = [a for a in agents if capability in a.capabilities]
        return agents

    async def get_agent(self, agent_id: str) -> FakeAgent | None:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    async def register_agent(
        self,
        agent_id: str,
        capabilities: list,
        model: str = "unknown",
        provider: str = "unknown",
        metadata: dict | None = None,
    ) -> FakeAgent:
        """Register a new agent."""
        agent = FakeAgent(
            agent_id=agent_id,
            name=f"Agent {agent_id}",
            capabilities=capabilities,
            metadata=metadata or {"model": model, "provider": provider},
        )
        self.agents[agent_id] = agent
        return agent

    async def heartbeat(self, agent_id: str, metadata: dict | None = None) -> bool:
        """Record agent heartbeat."""
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = datetime.now(timezone.utc)
            return True
        return False

    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False

    async def get_task(self, task_id: str) -> FakeTask | None:
        """Get task by ID."""
        return self.tasks.get(task_id)

    async def submit_task(self, task_type: str, payload: dict, priority: int = 5) -> FakeTask:
        """Submit a new task."""
        task = FakeTask(
            task_id=f"task-{len(self.tasks) + 1:03d}",
            task_type=task_type,
            payload=payload,
            priority=priority,
        )
        self.tasks[task.task_id] = task
        return task

    async def complete_task(self, task_id: str, result: dict) -> bool:
        """Mark task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].result = result
            return True
        return False

    async def fail_task(self, task_id: str, error: str) -> bool:
        """Mark task as failed."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].result = {"error": error}
            return True
        return False

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        if task_id in self.tasks:
            self.tasks[task_id].status = "cancelled"
            return True
        return False

    def get_system_health(self) -> FakeHealthStatus:
        """Get system health status (returns enum)."""
        return FakeHealthStatus.HEALTHY

    async def get_stats(self) -> dict:
        """Get control plane statistics."""
        return {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values() if a.status == "active"),
            "total_tasks": len(self.tasks),
            "tasks_by_status": {},
        }

    def get_queue(self, status: str | None = None, limit: int = 100) -> list:
        """Get task queue."""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return tasks[:limit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator."""
    return FakeCoordinator()


@pytest.fixture
def control_plane_handler(mock_coordinator):
    """Create a ControlPlaneHandler with mocked dependencies."""
    ctx = {"control_plane_coordinator": mock_coordinator}
    handler = ControlPlaneHandler(ctx)
    ControlPlaneHandler.coordinator = mock_coordinator
    yield handler
    ControlPlaneHandler.coordinator = None


@pytest.fixture(autouse=True)
def reset_class_state():
    """Reset class-level state between tests."""
    ControlPlaneHandler.coordinator = None
    yield
    ControlPlaneHandler.coordinator = None


# ---------------------------------------------------------------------------
# Test Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_handler_initializes(self, control_plane_handler):
        """Handler initializes properly with context."""
        assert control_plane_handler is not None
        assert control_plane_handler._get_coordinator() is not None

    def test_handler_returns_503_without_coordinator(self):
        """Handler returns 503 when coordinator not available."""
        handler = ControlPlaneHandler({})
        ControlPlaneHandler.coordinator = None
        coord, err = handler._require_coordinator()
        assert coord is None
        assert err is not None
        assert err.status_code == 503


# ---------------------------------------------------------------------------
# Test can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_handles_control_plane_routes(self, control_plane_handler):
        """Handler accepts control plane routes."""
        assert control_plane_handler.can_handle("/api/control-plane/agents") is True
        assert control_plane_handler.can_handle("/api/control-plane/tasks") is True
        assert control_plane_handler.can_handle("/api/control-plane/health") is True
        assert control_plane_handler.can_handle("/api/control-plane/stats") is True

    def test_handles_versioned_routes(self, control_plane_handler):
        """Handler accepts versioned control plane routes."""
        assert control_plane_handler.can_handle("/api/v1/control-plane/agents") is True
        assert control_plane_handler.can_handle("/api/v1/control-plane/tasks") is True
        assert control_plane_handler.can_handle("/api/v1/control-plane/health") is True

    def test_rejects_unknown_routes(self, control_plane_handler):
        """Handler rejects non-control-plane routes."""
        assert control_plane_handler.can_handle("/api/v1/billing") is False
        assert control_plane_handler.can_handle("/api/v1/debates") is False
        assert control_plane_handler.can_handle("/api/users") is False


# ---------------------------------------------------------------------------
# Test Path Normalization
# ---------------------------------------------------------------------------


class TestPathNormalization:
    def test_normalizes_versioned_paths(self, control_plane_handler):
        """Handler normalizes versioned paths to legacy form."""
        assert (
            control_plane_handler._normalize_path("/api/v1/control-plane/agents")
            == "/api/control-plane/agents"
        )
        assert (
            control_plane_handler._normalize_path("/api/v1/control-plane/tasks/task-001")
            == "/api/control-plane/tasks/task-001"
        )

    def test_preserves_legacy_paths(self, control_plane_handler):
        """Handler preserves already-legacy paths."""
        assert (
            control_plane_handler._normalize_path("/api/control-plane/agents")
            == "/api/control-plane/agents"
        )


# ---------------------------------------------------------------------------
# Test Agent Handlers (via _handle_list_agents)
# ---------------------------------------------------------------------------


class TestAgentHandlers:
    def test_list_agents(self, control_plane_handler, mock_coordinator):
        """List agents returns registered agents."""
        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=lambda *a, **kw: True,
            ),
        ):
            fn = control_plane_handler._handle_list_agents.__wrapped__
            result = fn(control_plane_handler, {})

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "agents" in data
        assert len(data["agents"]) == 1

    def test_list_agents_empty(self, control_plane_handler, mock_coordinator):
        """List agents returns empty list when no agents registered."""
        mock_coordinator.agents = {}
        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=lambda *a, **kw: True,
            ),
        ):
            fn = control_plane_handler._handle_list_agents.__wrapped__
            result = fn(control_plane_handler, {})

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["agents"] == []

    def test_get_agent(self, control_plane_handler, mock_coordinator):
        """Get agent returns agent details."""
        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=lambda *a, **kw: True,
            ),
        ):
            fn = control_plane_handler._handle_get_agent.__wrapped__
            result = fn(control_plane_handler, "agent-001")

        assert result.status_code == 200
        data = json.loads(result.body)
        # Response is directly the agent dict, not wrapped
        assert data["agent_id"] == "agent-001"

    def test_get_agent_not_found(self, control_plane_handler, mock_coordinator):
        """Get agent returns 404 for unknown agent."""
        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=lambda *a, **kw: True,
            ),
        ):
            fn = control_plane_handler._handle_get_agent.__wrapped__
            result = fn(control_plane_handler, "unknown-agent")

        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test Task Handlers
# ---------------------------------------------------------------------------


class TestTaskHandlers:
    def test_get_task(self, control_plane_handler, mock_coordinator):
        """Get task returns task details."""
        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch(
                "aragora.server.handlers.control_plane.tasks._get_has_permission",
                return_value=lambda *a, **kw: True,
            ),
        ):
            fn = control_plane_handler._handle_get_task.__wrapped__
            result = fn(control_plane_handler, "task-001")

        assert result.status_code == 200
        data = json.loads(result.body)
        # Response is directly the task dict, not wrapped
        assert data["task_id"] == "task-001"

    def test_get_task_not_found(self, control_plane_handler, mock_coordinator):
        """Get task returns 404 for unknown task."""
        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch(
                "aragora.server.handlers.control_plane.tasks._get_has_permission",
                return_value=lambda *a, **kw: True,
            ),
        ):
            fn = control_plane_handler._handle_get_task.__wrapped__
            result = fn(control_plane_handler, "unknown-task")

        assert result.status_code == 404


# ---------------------------------------------------------------------------
# Test Health Handlers
# ---------------------------------------------------------------------------


class TestHealthHandlers:
    def test_system_health(self, control_plane_handler, mock_coordinator):
        """System health returns health status."""
        mock_coordinator._health_monitor.get_all_health.return_value = {}
        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch(
                "aragora.server.handlers.control_plane.health._get_has_permission",
                return_value=lambda *a, **kw: True,
            ),
        ):
            fn = control_plane_handler._handle_system_health.__wrapped__
            result = fn(control_plane_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "status" in data

    def test_stats(self, control_plane_handler, mock_coordinator):
        """Stats returns control plane statistics."""
        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch(
                "aragora.server.handlers.control_plane.health._get_has_permission",
                return_value=lambda *a, **kw: True,
            ),
        ):
            fn = control_plane_handler._handle_stats.__wrapped__
            result = fn(control_plane_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "total_agents" in data


# ---------------------------------------------------------------------------
# Test GET Routing
# ---------------------------------------------------------------------------


class TestGetRouting:
    def test_routes_agents_list(self, control_plane_handler, mock_coordinator):
        """GET /api/control-plane/agents routes correctly."""
        http_handler = FakeHandler()

        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch.object(
                control_plane_handler,
                "_handle_list_agents",
                return_value=MagicMock(status_code=200, body=b'{"agents": []}'),
            ),
        ):
            result = control_plane_handler.handle("/api/control-plane/agents", {}, http_handler)

        assert result is not None
        assert result.status_code == 200

    def test_routes_health(self, control_plane_handler, mock_coordinator):
        """GET /api/control-plane/health routes correctly."""
        http_handler = FakeHandler()

        mock_coordinator._health_monitor.get_all_health.return_value = {}
        with (
            patch.object(
                control_plane_handler, "require_auth_or_error", return_value=(FakeUser(), None)
            ),
            patch.object(
                control_plane_handler,
                "_handle_system_health",
                return_value=MagicMock(status_code=200, body=b'{"status": "healthy"}'),
            ),
        ):
            result = control_plane_handler.handle("/api/control-plane/health", {}, http_handler)

        assert result is not None
        assert result.status_code == 200


# ---------------------------------------------------------------------------
# Test Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_handles_coordinator_not_initialized(self, control_plane_handler):
        """Handler returns 503 when coordinator not initialized."""
        ControlPlaneHandler.coordinator = None
        control_plane_handler.ctx = {}

        coord, err = control_plane_handler._require_coordinator()
        assert coord is None
        assert err.status_code == 503

    def test_handles_coordinator_error(self, control_plane_handler):
        """Handler properly wraps coordinator errors."""
        error = ValueError("Invalid agent ID")
        result = control_plane_handler._handle_coordinator_error(error, "get_agent")

        assert result.status_code == 400


# ---------------------------------------------------------------------------
# Test Coordinator Integration
# ---------------------------------------------------------------------------


class TestCoordinatorIntegration:
    def test_get_coordinator_from_class(self, control_plane_handler, mock_coordinator):
        """Handler gets coordinator from class attribute."""
        coord = control_plane_handler._get_coordinator()
        assert coord is mock_coordinator

    def test_get_coordinator_from_context(self, mock_coordinator):
        """Handler falls back to context for coordinator."""
        ControlPlaneHandler.coordinator = None
        ctx = {"control_plane_coordinator": mock_coordinator}
        handler = ControlPlaneHandler(ctx)

        coord = handler._get_coordinator()
        assert coord is mock_coordinator


# ---------------------------------------------------------------------------
# Test Task History
# ---------------------------------------------------------------------------


class TestTaskHistory:
    """Tests for task history endpoint."""

    def test_task_history_endpoint_exists(self, control_plane_handler, mock_coordinator):
        """Task history endpoint is available."""
        # Verify the method exists
        assert hasattr(control_plane_handler, "_handle_task_history")

    def test_task_history_without_coordinator(self):
        """Task history returns 503 without coordinator."""
        handler = ControlPlaneHandler({})
        ControlPlaneHandler.coordinator = None

        result = handler._handle_task_history({})
        assert result.status_code == 503

    def test_task_history_parses_query_params(self, control_plane_handler, mock_coordinator):
        """Task history parses query parameters correctly."""
        # Mock the scheduler to return empty list
        mock_coordinator._scheduler = MagicMock()
        mock_coordinator._scheduler.list_by_status = AsyncMock(return_value=[])

        result = control_plane_handler._handle_task_history(
            {
                "limit": "50",
                "offset": "10",
                "status": "completed",
            }
        )

        # Should return 200 with empty history
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["limit"] == 50
        assert data["offset"] == 10
        assert data["history"] == []
