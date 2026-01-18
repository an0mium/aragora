"""
Tests for ControlPlaneHandler.

Tests the REST API endpoints for the enterprise control plane:
- Agent registration and discovery
- Task submission and management
- Health monitoring
- Statistics
"""

import json
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.control_plane import ControlPlaneHandler


class MockCoordinator:
    """Mock ControlPlaneCoordinator for testing."""

    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._tasks: Dict[str, Any] = {}
        self._health_monitor = MagicMock()
        self._health_monitor.get_all_health.return_value = {}

    async def list_agents(
        self, capability: Optional[str] = None, only_available: bool = True
    ) -> list:
        agents = list(self._agents.values())
        if capability:
            agents = [a for a in agents if capability in a.capabilities]
        return agents

    async def get_agent(self, agent_id: str) -> Optional[Any]:
        return self._agents.get(agent_id)

    async def register_agent(
        self,
        agent_id: str,
        capabilities: list,
        model: str = "unknown",
        provider: str = "unknown",
        metadata: dict = None,
    ) -> Any:
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.capabilities = capabilities
        agent.model = model
        agent.provider = provider
        agent.metadata = metadata or {}
        agent.to_dict.return_value = {
            "agent_id": agent_id,
            "capabilities": capabilities,
            "model": model,
            "provider": provider,
            "metadata": metadata or {},
        }
        self._agents[agent_id] = agent
        return agent

    async def unregister_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    async def heartbeat(self, agent_id: str, status: Any = None) -> bool:
        return agent_id in self._agents

    async def submit_task(
        self,
        task_type: str,
        payload: dict,
        required_capabilities: list = None,
        priority: Any = None,
        timeout_seconds: Optional[int] = None,
        metadata: dict = None,
    ) -> str:
        task_id = f"task-{len(self._tasks) + 1}"
        self._tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "payload": payload,
            "required_capabilities": required_capabilities or [],
            "priority": priority,
            "timeout_seconds": timeout_seconds,
            "metadata": metadata or {},
            "status": "pending",
        }
        return task_id

    async def get_task(self, task_id: str) -> Optional[Any]:
        task_data = self._tasks.get(task_id)
        if not task_data:
            return None
        task = MagicMock()
        task.to_dict.return_value = task_data
        return task

    async def claim_task(
        self, agent_id: str, capabilities: list = None, block_ms: int = 5000
    ) -> Optional[Any]:
        for task_id, task_data in self._tasks.items():
            if task_data["status"] == "pending":
                task_data["status"] = "running"
                task_data["assigned_agent"] = agent_id
                task = MagicMock()
                task.to_dict.return_value = task_data
                return task
        return None

    async def complete_task(
        self,
        task_id: str,
        result: Any = None,
        agent_id: str = None,
        latency_ms: int = None,
    ) -> bool:
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = "completed"
            self._tasks[task_id]["result"] = result
            return True
        return False

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_id: str = None,
        latency_ms: int = None,
        requeue: bool = True,
    ) -> bool:
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = "failed"
            self._tasks[task_id]["error"] = error
            return True
        return False

    async def cancel_task(self, task_id: str) -> bool:
        if task_id in self._tasks:
            if self._tasks[task_id]["status"] not in ["completed", "failed"]:
                self._tasks[task_id]["status"] = "cancelled"
                return True
        return False

    def get_system_health(self) -> Any:
        health = MagicMock()
        health.value = "healthy"
        return health

    def get_agent_health(self, agent_id: str) -> Optional[Any]:
        if agent_id in self._agents:
            health = MagicMock()
            health.to_dict.return_value = {
                "agent_id": agent_id,
                "status": "healthy",
                "last_seen": "2024-01-01T00:00:00Z",
            }
            return health
        return None

    async def get_stats(self) -> Dict[str, Any]:
        return {
            "total_agents": len(self._agents),
            "total_tasks": len(self._tasks),
            "pending_tasks": sum(
                1 for t in self._tasks.values() if t["status"] == "pending"
            ),
            "completed_tasks": sum(
                1 for t in self._tasks.values() if t["status"] == "completed"
            ),
        }


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator."""
    return MockCoordinator()


@pytest.fixture
def handler(mock_coordinator):
    """Create a handler with mock coordinator."""
    handler = ControlPlaneHandler({})
    handler.__class__.coordinator = mock_coordinator
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for reading body."""
    handler = MagicMock()
    handler.rfile = MagicMock()
    return handler


class TestControlPlaneHandlerInit:
    """Test handler initialization."""

    def test_init_with_context(self):
        """Test handler initializes with context."""
        context = {"key": "value"}
        handler = ControlPlaneHandler(context)
        assert handler.ctx == context

    def test_get_coordinator_from_class(self, mock_coordinator):
        """Test getting coordinator from class attribute."""
        ControlPlaneHandler.coordinator = mock_coordinator
        handler = ControlPlaneHandler({})
        assert handler._get_coordinator() == mock_coordinator
        ControlPlaneHandler.coordinator = None

    def test_get_coordinator_from_context(self, mock_coordinator):
        """Test getting coordinator from context."""
        ControlPlaneHandler.coordinator = None
        handler = ControlPlaneHandler({"control_plane_coordinator": mock_coordinator})
        assert handler._get_coordinator() == mock_coordinator


class TestListAgents:
    """Test GET /api/control-plane/agents."""

    def test_list_agents_empty(self, handler, mock_http_handler):
        """Test listing agents when none registered."""
        result = handler.handle("/api/control-plane/agents", {}, mock_http_handler)

        assert result is not None
        data = json.loads(result.body)
        assert data["agents"] == []
        assert data["total"] == 0

    def test_list_agents_with_agents(self, handler, mock_coordinator, mock_http_handler):
        """Test listing agents with registered agents."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.register_agent("agent-1", ["debate", "critique"])
        )
        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.register_agent("agent-2", ["debate"])
        )

        result = handler.handle("/api/control-plane/agents", {}, mock_http_handler)

        data = json.loads(result.body)
        assert data["total"] == 2

    def test_list_agents_filter_by_capability(
        self, handler, mock_coordinator, mock_http_handler
    ):
        """Test filtering agents by capability."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.register_agent("agent-1", ["debate", "critique"])
        )
        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.register_agent("agent-2", ["debate"])
        )

        result = handler.handle(
            "/api/control-plane/agents", {"capability": "critique"}, mock_http_handler
        )

        data = json.loads(result.body)
        assert data["total"] == 1

    def test_list_agents_no_coordinator(self, mock_http_handler):
        """Test listing agents without coordinator."""
        ControlPlaneHandler.coordinator = None
        handler = ControlPlaneHandler({})

        result = handler.handle("/api/control-plane/agents", {}, mock_http_handler)

        assert result.status_code == 503
        data = json.loads(result.body)
        assert "not initialized" in data["error"]


class TestGetAgent:
    """Test GET /api/control-plane/agents/:id."""

    def test_get_agent_success(self, handler, mock_coordinator, mock_http_handler):
        """Test getting agent by ID."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.register_agent("agent-1", ["debate"])
        )

        result = handler.handle(
            "/api/control-plane/agents/agent-1", {}, mock_http_handler
        )

        assert result is not None
        data = json.loads(result.body)
        assert data["agent_id"] == "agent-1"

    def test_get_agent_not_found(self, handler, mock_http_handler):
        """Test getting non-existent agent."""
        result = handler.handle(
            "/api/control-plane/agents/nonexistent", {}, mock_http_handler
        )

        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data["error"]


class TestRegisterAgent:
    """Test POST /api/control-plane/agents."""

    def test_register_agent_success(self, handler, mock_http_handler):
        """Test registering a new agent."""
        body = {
            "agent_id": "agent-new",
            "capabilities": ["debate", "critique"],
            "model": "gpt-4",
            "provider": "openai",
        }

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            result = handler.handle_post(
                "/api/control-plane/agents", {}, mock_http_handler
            )

        assert result.status_code == 201
        data = json.loads(result.body)
        assert data["agent_id"] == "agent-new"

    def test_register_agent_missing_id(self, handler, mock_http_handler):
        """Test registering agent without ID."""
        body = {"capabilities": ["debate"]}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            result = handler.handle_post(
                "/api/control-plane/agents", {}, mock_http_handler
            )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "agent_id" in data["error"]


class TestUnregisterAgent:
    """Test DELETE /api/control-plane/agents/:id."""

    def test_unregister_agent_success(
        self, handler, mock_coordinator, mock_http_handler
    ):
        """Test unregistering an agent."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.register_agent("agent-1", ["debate"])
        )

        result = handler.handle_delete(
            "/api/control-plane/agents/agent-1", {}, mock_http_handler
        )

        assert result is not None
        data = json.loads(result.body)
        assert data["unregistered"] is True

    def test_unregister_agent_not_found(self, handler, mock_http_handler):
        """Test unregistering non-existent agent."""
        result = handler.handle_delete(
            "/api/control-plane/agents/nonexistent", {}, mock_http_handler
        )

        assert result.status_code == 404


class TestAgentHeartbeat:
    """Test POST /api/control-plane/agents/:id/heartbeat."""

    def test_heartbeat_success(self, handler, mock_coordinator, mock_http_handler):
        """Test sending heartbeat."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.register_agent("agent-1", ["debate"])
        )

        body = {"status": "available"}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch(
                "aragora.control_plane.registry.AgentStatus", MagicMock(return_value="available")
            ):
                result = handler.handle_post(
                    "/api/control-plane/agents/agent-1/heartbeat", {}, mock_http_handler
                )

        assert result is not None
        data = json.loads(result.body)
        assert data["acknowledged"] is True

    def test_heartbeat_agent_not_found(self, handler, mock_http_handler):
        """Test heartbeat for non-existent agent."""
        body = {"status": "available"}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch(
                "aragora.control_plane.registry.AgentStatus", MagicMock(return_value="available")
            ):
                result = handler.handle_post(
                    "/api/control-plane/agents/nonexistent/heartbeat",
                    {},
                    mock_http_handler,
                )

        assert result.status_code == 404


class TestSubmitTask:
    """Test POST /api/control-plane/tasks."""

    def test_submit_task_success(self, handler, mock_http_handler):
        """Test submitting a new task."""
        body = {
            "task_type": "debate",
            "payload": {"topic": "AI safety"},
            "priority": "normal",
        }

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch(
                "aragora.control_plane.scheduler.TaskPriority",
                MagicMock(NORMAL="normal"),
            ):
                result = handler.handle_post(
                    "/api/control-plane/tasks", {}, mock_http_handler
                )

        assert result.status_code == 201
        data = json.loads(result.body)
        assert "task_id" in data

    def test_submit_task_missing_type(self, handler, mock_http_handler):
        """Test submitting task without type."""
        body = {"payload": {"topic": "AI"}}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            result = handler.handle_post(
                "/api/control-plane/tasks", {}, mock_http_handler
            )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "task_type" in data["error"]


class TestGetTask:
    """Test GET /api/control-plane/tasks/:id."""

    def test_get_task_success(self, handler, mock_coordinator, mock_http_handler):
        """Test getting task by ID."""
        import asyncio

        task_id = asyncio.get_event_loop().run_until_complete(
            mock_coordinator.submit_task("debate", {"topic": "AI"})
        )

        result = handler.handle(f"/api/control-plane/tasks/{task_id}", {}, mock_http_handler)

        assert result is not None
        data = json.loads(result.body)
        assert data["task_type"] == "debate"

    def test_get_task_not_found(self, handler, mock_http_handler):
        """Test getting non-existent task."""
        result = handler.handle(
            "/api/control-plane/tasks/nonexistent", {}, mock_http_handler
        )

        assert result.status_code == 404


class TestCompleteTask:
    """Test POST /api/control-plane/tasks/:id/complete."""

    def test_complete_task_success(self, handler, mock_coordinator, mock_http_handler):
        """Test completing a task."""
        import asyncio

        task_id = asyncio.get_event_loop().run_until_complete(
            mock_coordinator.submit_task("debate", {"topic": "AI"})
        )

        body = {"result": {"consensus": "reached"}, "latency_ms": 1500}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            result = handler.handle_post(
                f"/api/control-plane/tasks/{task_id}/complete", {}, mock_http_handler
            )

        assert result is not None
        data = json.loads(result.body)
        assert data["completed"] is True

    def test_complete_task_not_found(self, handler, mock_http_handler):
        """Test completing non-existent task."""
        body = {"result": {"consensus": "reached"}}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            result = handler.handle_post(
                "/api/control-plane/tasks/nonexistent/complete", {}, mock_http_handler
            )

        assert result.status_code == 404


class TestFailTask:
    """Test POST /api/control-plane/tasks/:id/fail."""

    def test_fail_task_success(self, handler, mock_coordinator, mock_http_handler):
        """Test failing a task."""
        import asyncio

        task_id = asyncio.get_event_loop().run_until_complete(
            mock_coordinator.submit_task("debate", {"topic": "AI"})
        )

        body = {"error": "Agent timeout", "requeue": True}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            result = handler.handle_post(
                f"/api/control-plane/tasks/{task_id}/fail", {}, mock_http_handler
            )

        assert result is not None
        data = json.loads(result.body)
        assert data["failed"] is True


class TestCancelTask:
    """Test POST /api/control-plane/tasks/:id/cancel."""

    def test_cancel_task_success(self, handler, mock_coordinator, mock_http_handler):
        """Test cancelling a task."""
        import asyncio

        task_id = asyncio.get_event_loop().run_until_complete(
            mock_coordinator.submit_task("debate", {"topic": "AI"})
        )

        result = handler.handle_post(
            f"/api/control-plane/tasks/{task_id}/cancel", {}, mock_http_handler
        )

        assert result is not None
        data = json.loads(result.body)
        assert data["cancelled"] is True


class TestSystemHealth:
    """Test GET /api/control-plane/health."""

    def test_system_health(self, handler, mock_http_handler):
        """Test getting system health."""
        result = handler.handle("/api/control-plane/health", {}, mock_http_handler)

        assert result is not None
        data = json.loads(result.body)
        assert data["status"] == "healthy"


class TestAgentHealth:
    """Test GET /api/control-plane/health/:agent_id."""

    def test_agent_health_success(self, handler, mock_coordinator, mock_http_handler):
        """Test getting agent health."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.register_agent("agent-1", ["debate"])
        )

        result = handler.handle(
            "/api/control-plane/health/agent-1", {}, mock_http_handler
        )

        assert result is not None
        data = json.loads(result.body)
        assert data["agent_id"] == "agent-1"

    def test_agent_health_not_found(self, handler, mock_http_handler):
        """Test getting health for non-existent agent."""
        result = handler.handle(
            "/api/control-plane/health/nonexistent", {}, mock_http_handler
        )

        assert result.status_code == 404


class TestStats:
    """Test GET /api/control-plane/stats."""

    def test_get_stats(self, handler, mock_coordinator, mock_http_handler):
        """Test getting control plane statistics."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.register_agent("agent-1", ["debate"])
        )
        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.submit_task("debate", {"topic": "AI"})
        )

        result = handler.handle("/api/control-plane/stats", {}, mock_http_handler)

        assert result is not None
        data = json.loads(result.body)
        assert data["total_agents"] == 1
        assert data["total_tasks"] == 1


class TestClaimTask:
    """Test POST /api/control-plane/tasks/:id/claim."""

    def test_claim_task_success(self, handler, mock_coordinator, mock_http_handler):
        """Test claiming a task."""
        import asyncio

        asyncio.get_event_loop().run_until_complete(
            mock_coordinator.submit_task("debate", {"topic": "AI"})
        )

        body = {"agent_id": "agent-1", "capabilities": ["debate"]}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            result = handler.handle_post(
                "/api/control-plane/tasks/any/claim", {}, mock_http_handler
            )

        assert result is not None
        data = json.loads(result.body)
        assert "task" in data
        assert data["task"] is not None

    def test_claim_task_no_tasks_available(self, handler, mock_http_handler):
        """Test claiming when no tasks available."""
        body = {"agent_id": "agent-1", "capabilities": ["debate"]}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            result = handler.handle_post(
                "/api/control-plane/tasks/any/claim", {}, mock_http_handler
            )

        data = json.loads(result.body)
        assert data["task"] is None

    def test_claim_task_missing_agent_id(self, handler, mock_http_handler):
        """Test claiming without agent_id."""
        body = {"capabilities": ["debate"]}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            result = handler.handle_post(
                "/api/control-plane/tasks/any/claim", {}, mock_http_handler
            )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "agent_id" in data["error"]
