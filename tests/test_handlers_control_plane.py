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
            "pending_tasks": sum(1 for t in self._tasks.values() if t["status"] == "pending"),
            "completed_tasks": sum(1 for t in self._tasks.values() if t["status"] == "completed"),
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
    yield handler
    # Cleanup: Remove class attribute to prevent test pollution
    handler.__class__.coordinator = None


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for reading body."""
    handler = MagicMock()
    handler.rfile = MagicMock()
    return handler


@pytest.fixture
def mock_user():
    """Create a mock authenticated user context."""
    user = MagicMock()
    user.user_id = "test-user"
    user.email = "test@example.com"
    user.role = "user"
    return user


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

        asyncio.run(mock_coordinator.register_agent("agent-1", ["debate", "critique"]))
        asyncio.run(mock_coordinator.register_agent("agent-2", ["debate"]))

        result = handler.handle("/api/control-plane/agents", {}, mock_http_handler)

        data = json.loads(result.body)
        assert data["total"] == 2

    def test_list_agents_filter_by_capability(self, handler, mock_coordinator, mock_http_handler):
        """Test filtering agents by capability."""
        import asyncio

        asyncio.run(mock_coordinator.register_agent("agent-1", ["debate", "critique"]))
        asyncio.run(mock_coordinator.register_agent("agent-2", ["debate"]))

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

        asyncio.run(mock_coordinator.register_agent("agent-1", ["debate"]))

        result = handler.handle("/api/control-plane/agents/agent-1", {}, mock_http_handler)

        assert result is not None
        data = json.loads(result.body)
        assert data["agent_id"] == "agent-1"

    def test_get_agent_not_found(self, handler, mock_http_handler):
        """Test getting non-existent agent."""
        result = handler.handle("/api/control-plane/agents/nonexistent", {}, mock_http_handler)

        assert result.status_code == 404
        data = json.loads(result.body)
        assert "not found" in data["error"]


class TestRegisterAgent:
    """Test POST /api/control-plane/agents."""

    def test_register_agent_success(self, handler, mock_http_handler, mock_user):
        """Test registering a new agent."""
        body = {
            "agent_id": "agent-new",
            "capabilities": ["debate", "critique"],
            "model": "gpt-4",
            "provider": "openai",
        }

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle_post("/api/control-plane/agents", {}, mock_http_handler)

        assert result.status_code == 201
        data = json.loads(result.body)
        assert data["agent_id"] == "agent-new"

    def test_register_agent_missing_id(self, handler, mock_http_handler, mock_user):
        """Test registering agent without ID."""
        body = {"capabilities": ["debate"]}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle_post("/api/control-plane/agents", {}, mock_http_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "agent_id" in data["error"]


class TestUnregisterAgent:
    """Test DELETE /api/control-plane/agents/:id."""

    def test_unregister_agent_success(
        self, handler, mock_coordinator, mock_http_handler, mock_user
    ):
        """Test unregistering an agent."""
        import asyncio

        asyncio.run(mock_coordinator.register_agent("agent-1", ["debate"]))

        with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = handler.handle_delete(
                    "/api/control-plane/agents/agent-1", {}, mock_http_handler
                )

        assert result is not None
        data = json.loads(result.body)
        assert data["unregistered"] is True

    def test_unregister_agent_not_found(self, handler, mock_http_handler, mock_user):
        """Test unregistering non-existent agent."""
        with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = handler.handle_delete(
                    "/api/control-plane/agents/nonexistent", {}, mock_http_handler
                )

        assert result.status_code == 404


class TestAgentHeartbeat:
    """Test POST /api/control-plane/agents/:id/heartbeat."""

    def test_heartbeat_success(self, handler, mock_coordinator, mock_http_handler, mock_user):
        """Test sending heartbeat."""
        import asyncio

        asyncio.run(mock_coordinator.register_agent("agent-1", ["debate"]))

        body = {"status": "available"}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    with patch(
                        "aragora.control_plane.registry.AgentStatus",
                        MagicMock(return_value="available"),
                    ):
                        result = handler.handle_post(
                            "/api/control-plane/agents/agent-1/heartbeat", {}, mock_http_handler
                        )

        assert result is not None
        data = json.loads(result.body)
        assert data["acknowledged"] is True

    def test_heartbeat_agent_not_found(self, handler, mock_http_handler, mock_user):
        """Test heartbeat for non-existent agent."""
        body = {"status": "available"}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    with patch(
                        "aragora.control_plane.registry.AgentStatus",
                        MagicMock(return_value="available"),
                    ):
                        result = handler.handle_post(
                            "/api/control-plane/agents/nonexistent/heartbeat",
                            {},
                            mock_http_handler,
                        )

        assert result.status_code == 404


class TestSubmitTask:
    """Test POST /api/control-plane/tasks."""

    def test_submit_task_success(self, handler, mock_http_handler, mock_user):
        """Test submitting a new task."""
        body = {
            "task_type": "debate",
            "payload": {"topic": "AI safety"},
            "priority": "normal",
        }

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
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

    def test_submit_task_missing_type(self, handler, mock_http_handler, mock_user):
        """Test submitting task without type."""
        body = {"payload": {"topic": "AI"}}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle_post("/api/control-plane/tasks", {}, mock_http_handler)

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "task_type" in data["error"]


class TestGetTask:
    """Test GET /api/control-plane/tasks/:id."""

    def test_get_task_success(self, handler, mock_coordinator, mock_http_handler):
        """Test getting task by ID."""
        import asyncio

        task_id = asyncio.run(mock_coordinator.submit_task("debate", {"topic": "AI"}))

        result = handler.handle(f"/api/control-plane/tasks/{task_id}", {}, mock_http_handler)

        assert result is not None
        data = json.loads(result.body)
        assert data["task_type"] == "debate"

    def test_get_task_not_found(self, handler, mock_http_handler):
        """Test getting non-existent task."""
        result = handler.handle("/api/control-plane/tasks/nonexistent", {}, mock_http_handler)

        assert result.status_code == 404


class TestCompleteTask:
    """Test POST /api/control-plane/tasks/:id/complete."""

    def test_complete_task_success(self, handler, mock_coordinator, mock_http_handler, mock_user):
        """Test completing a task."""
        import asyncio

        task_id = asyncio.run(mock_coordinator.submit_task("debate", {"topic": "AI"}))

        body = {"result": {"consensus": "reached"}, "latency_ms": 1500}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle_post(
                        f"/api/control-plane/tasks/{task_id}/complete", {}, mock_http_handler
                    )

        assert result is not None
        data = json.loads(result.body)
        assert data["completed"] is True

    def test_complete_task_not_found(self, handler, mock_http_handler, mock_user):
        """Test completing non-existent task."""
        body = {"result": {"consensus": "reached"}}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle_post(
                        "/api/control-plane/tasks/nonexistent/complete", {}, mock_http_handler
                    )

        assert result.status_code == 404


class TestFailTask:
    """Test POST /api/control-plane/tasks/:id/fail."""

    def test_fail_task_success(self, handler, mock_coordinator, mock_http_handler, mock_user):
        """Test failing a task."""
        import asyncio

        task_id = asyncio.run(mock_coordinator.submit_task("debate", {"topic": "AI"}))

        body = {"error": "Agent timeout", "requeue": True}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle_post(
                        f"/api/control-plane/tasks/{task_id}/fail", {}, mock_http_handler
                    )

        assert result is not None
        data = json.loads(result.body)
        assert data["failed"] is True


class TestCancelTask:
    """Test POST /api/control-plane/tasks/:id/cancel."""

    def test_cancel_task_success(self, handler, mock_coordinator, mock_http_handler, mock_user):
        """Test cancelling a task."""
        import asyncio

        task_id = asyncio.run(mock_coordinator.submit_task("debate", {"topic": "AI"}))

        with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
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

        asyncio.run(mock_coordinator.register_agent("agent-1", ["debate"]))

        result = handler.handle("/api/control-plane/health/agent-1", {}, mock_http_handler)

        assert result is not None
        data = json.loads(result.body)
        assert data["agent_id"] == "agent-1"

    def test_agent_health_not_found(self, handler, mock_http_handler):
        """Test getting health for non-existent agent."""
        result = handler.handle("/api/control-plane/health/nonexistent", {}, mock_http_handler)

        assert result.status_code == 404


class TestStats:
    """Test GET /api/control-plane/stats."""

    def test_get_stats(self, handler, mock_coordinator, mock_http_handler):
        """Test getting control plane statistics."""
        import asyncio

        asyncio.run(mock_coordinator.register_agent("agent-1", ["debate"]))
        asyncio.run(mock_coordinator.submit_task("debate", {"topic": "AI"}))

        result = handler.handle("/api/control-plane/stats", {}, mock_http_handler)

        assert result is not None
        data = json.loads(result.body)
        assert data["total_agents"] == 1
        assert data["total_tasks"] == 1


class TestClaimTask:
    """Test POST /api/control-plane/tasks/:id/claim."""

    def test_claim_task_success(self, handler, mock_coordinator, mock_http_handler, mock_user):
        """Test claiming a task."""
        import asyncio

        asyncio.run(mock_coordinator.submit_task("debate", {"topic": "AI"}))

        body = {"agent_id": "agent-1", "capabilities": ["debate"]}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle_post(
                        "/api/control-plane/tasks/any/claim", {}, mock_http_handler
                    )

        assert result is not None
        data = json.loads(result.body)
        assert "task" in data
        assert data["task"] is not None

    def test_claim_task_no_tasks_available(self, handler, mock_http_handler, mock_user):
        """Test claiming when no tasks available."""
        body = {"agent_id": "agent-1", "capabilities": ["debate"]}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle_post(
                        "/api/control-plane/tasks/any/claim", {}, mock_http_handler
                    )

        data = json.loads(result.body)
        assert data["task"] is None

    def test_claim_task_missing_agent_id(self, handler, mock_http_handler, mock_user):
        """Test claiming without agent_id."""
        body = {"capabilities": ["debate"]}

        with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
            with patch.object(handler, "require_auth_or_error", return_value=(mock_user, None)):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle_post(
                        "/api/control-plane/tasks/any/claim", {}, mock_http_handler
                    )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "agent_id" in data["error"]


# =============================================================================
# Policy Violation Tests
# =============================================================================


class MockPolicyStore:
    """Mock policy store for testing."""

    def __init__(self):
        self._violations = []

    def list_violations(
        self,
        policy_id=None,
        violation_type=None,
        status=None,
        workspace_id=None,
        limit=100,
        offset=0,
    ):
        """List violations with filtering."""
        results = self._violations.copy()
        if policy_id:
            results = [v for v in results if v.get("policy_id") == policy_id]
        if violation_type:
            results = [v for v in results if v.get("violation_type") == violation_type]
        if status:
            results = [v for v in results if v.get("status") == status]
        if workspace_id:
            results = [v for v in results if v.get("workspace_id") == workspace_id]
        return results[offset : offset + limit]

    def count_violations(self, status=None, policy_id=None):
        """Count violations by type."""
        results = self._violations.copy()
        if status:
            results = [v for v in results if v.get("status") == status]
        if policy_id:
            results = [v for v in results if v.get("policy_id") == policy_id]
        counts = {}
        for v in results:
            vtype = v.get("violation_type", "unknown")
            counts[vtype] = counts.get(vtype, 0) + 1
        return counts

    def update_violation_status(
        self, violation_id, status, resolved_by=None, resolution_notes=None
    ):
        """Update a violation's status."""
        for v in self._violations:
            if v.get("id") == violation_id:
                v["status"] = status
                if resolved_by:
                    v["resolved_by"] = resolved_by
                if resolution_notes:
                    v["resolution_notes"] = resolution_notes
                return True
        return False

    def add_violation(self, violation):
        """Add a test violation."""
        self._violations.append(violation)


@pytest.fixture
def mock_policy_store():
    """Create a mock policy store with test violations."""
    store = MockPolicyStore()
    store.add_violation(
        {
            "id": "viol-1",
            "policy_id": "policy-1",
            "policy_name": "Agent Allowlist",
            "violation_type": "agent_blocked",
            "description": "Agent not in allowlist",
            "task_id": "task-123",
            "task_type": "debate",
            "agent_id": "agent-blocked",
            "region": None,
            "workspace_id": "ws-1",
            "enforcement_level": "hard",
            "timestamp": "2024-01-01T00:00:00Z",
            "status": "open",
            "resolved_at": None,
            "resolved_by": None,
            "resolution_notes": None,
            "metadata": {},
        }
    )
    store.add_violation(
        {
            "id": "viol-2",
            "policy_id": "policy-2",
            "policy_name": "Region Constraint",
            "violation_type": "region_violation",
            "description": "Task executed in restricted region",
            "task_id": "task-456",
            "task_type": "analysis",
            "agent_id": "agent-1",
            "region": "eu-west-1",
            "workspace_id": "ws-1",
            "enforcement_level": "soft",
            "timestamp": "2024-01-02T00:00:00Z",
            "status": "resolved",
            "resolved_at": "2024-01-02T01:00:00Z",
            "resolved_by": "admin",
            "resolution_notes": "Approved exception",
            "metadata": {},
        }
    )
    return store


@pytest.fixture
def admin_user():
    """Create a mock admin user with policy permissions."""
    user = MagicMock()
    user.user_id = "admin-user"
    user.id = "admin-user"
    user.email = "admin@example.com"
    user.role = "admin"
    return user


class TestListPolicyViolations:
    """Test GET /api/control-plane/policies/violations."""

    def test_list_violations_success(
        self, handler, mock_http_handler, mock_policy_store, admin_user
    ):
        """Test listing policy violations."""
        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch.object(handler, "_get_policy_store", return_value=mock_policy_store):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle(
                        "/api/control-plane/policies/violations", {}, mock_http_handler
                    )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "violations" in data
        assert data["total"] == 2

    def test_list_violations_filter_by_status(
        self, handler, mock_http_handler, mock_policy_store, admin_user
    ):
        """Test filtering violations by status."""
        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch.object(handler, "_get_policy_store", return_value=mock_policy_store):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle(
                        "/api/control-plane/policies/violations",
                        {"status": ["open"]},
                        mock_http_handler,
                    )

        data = json.loads(result.body)
        assert data["total"] == 1
        assert data["violations"][0]["status"] == "open"

    def test_list_violations_permission_denied(self, handler, mock_http_handler, admin_user):
        """Test listing violations without permission."""
        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=False):
                result = handler.handle(
                    "/api/control-plane/policies/violations", {}, mock_http_handler
                )

        assert result.status_code == 403
        data = json.loads(result.body)
        assert "Permission denied" in data["error"]


class TestGetPolicyViolation:
    """Test GET /api/control-plane/policies/violations/:id."""

    def test_get_violation_success(self, handler, mock_http_handler, mock_policy_store, admin_user):
        """Test getting a specific violation."""
        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch.object(handler, "_get_policy_store", return_value=mock_policy_store):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle(
                        "/api/control-plane/policies/violations/viol-1", {}, mock_http_handler
                    )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["violation"]["id"] == "viol-1"

    def test_get_violation_not_found(
        self, handler, mock_http_handler, mock_policy_store, admin_user
    ):
        """Test getting non-existent violation."""
        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch.object(handler, "_get_policy_store", return_value=mock_policy_store):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle(
                        "/api/control-plane/policies/violations/nonexistent", {}, mock_http_handler
                    )

        assert result.status_code == 404


class TestGetPolicyViolationStats:
    """Test GET /api/control-plane/policies/violations/stats."""

    def test_get_violation_stats(self, handler, mock_http_handler, mock_policy_store, admin_user):
        """Test getting violation statistics."""
        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch.object(handler, "_get_policy_store", return_value=mock_policy_store):
                with patch(
                    "aragora.server.handlers.control_plane.has_permission", return_value=True
                ):
                    result = handler.handle(
                        "/api/control-plane/policies/violations/stats", {}, mock_http_handler
                    )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert "total" in data
        assert "open" in data
        assert "by_type" in data


class TestUpdatePolicyViolation:
    """Test PATCH /api/control-plane/policies/violations/:id."""

    def test_update_violation_status(
        self, handler, mock_http_handler, mock_policy_store, admin_user
    ):
        """Test updating a violation status."""
        body = {"status": "resolved", "resolution_notes": "Verified and closed"}

        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
                with patch.object(handler, "_get_policy_store", return_value=mock_policy_store):
                    with patch(
                        "aragora.server.handlers.control_plane.has_permission", return_value=True
                    ):
                        result = handler.handle_patch(
                            "/api/control-plane/policies/violations/viol-1", {}, mock_http_handler
                        )

        assert result is not None
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["updated"] is True
        assert data["status"] == "resolved"

    def test_update_violation_invalid_status(
        self, handler, mock_http_handler, mock_policy_store, admin_user
    ):
        """Test updating with invalid status."""
        body = {"status": "invalid_status"}

        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
                with patch.object(handler, "_get_policy_store", return_value=mock_policy_store):
                    with patch(
                        "aragora.server.handlers.control_plane.has_permission", return_value=True
                    ):
                        result = handler.handle_patch(
                            "/api/control-plane/policies/violations/viol-1", {}, mock_http_handler
                        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "Invalid status" in data["error"]

    def test_update_violation_missing_status(
        self, handler, mock_http_handler, mock_policy_store, admin_user
    ):
        """Test updating without status field."""
        body = {"resolution_notes": "Notes without status"}

        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
                with patch.object(handler, "_get_policy_store", return_value=mock_policy_store):
                    with patch(
                        "aragora.server.handlers.control_plane.has_permission", return_value=True
                    ):
                        result = handler.handle_patch(
                            "/api/control-plane/policies/violations/viol-1", {}, mock_http_handler
                        )

        assert result.status_code == 400
        data = json.loads(result.body)
        assert "status" in data["error"]

    def test_update_violation_not_found(
        self, handler, mock_http_handler, mock_policy_store, admin_user
    ):
        """Test updating non-existent violation."""
        body = {"status": "resolved"}

        with patch.object(handler, "require_auth_or_error", return_value=(admin_user, None)):
            with patch.object(handler, "read_json_body_validated", return_value=(body, None)):
                with patch.object(handler, "_get_policy_store", return_value=mock_policy_store):
                    with patch(
                        "aragora.server.handlers.control_plane.has_permission", return_value=True
                    ):
                        result = handler.handle_patch(
                            "/api/control-plane/policies/violations/nonexistent",
                            {},
                            mock_http_handler,
                        )

        assert result.status_code == 404
