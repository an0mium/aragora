"""
Tests for Control Plane / Agent Dashboard Handler.

Tests cover:
- can_handle routing
- Permission checks (auth, RBAC)
- List/get agents
- Pause/resume agents
- Agent metrics
- Task queue management (list, prioritize)
- System metrics
- Health check
- Streaming updates
- Default agents and sample queue generation
- Module-level state cleanup
- Utility methods
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from aragora.server.handlers.features.control_plane import (
    AgentDashboardHandler,
    ControlPlaneHandler,
    _agents,
    _task_queue,
    _stream_clients,
    _metrics,
    CONTROL_PLANE_READ_PERMISSION,
    CONTROL_PLANE_WRITE_PERMISSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_result(result):
    """Parse a handler response dict into (body_dict, status_code)."""
    body = json.loads(result.get("body", "{}")) if result.get("body") else {}
    return body, result.get("status", 500)


def _make_request(
    *,
    method="GET",
    path="/api/v1/control-plane/agents",
    query=None,
    json_body=None,
    headers=None,
):
    """Build a fake request object."""
    req = SimpleNamespace()
    req.method = method
    req.path = path
    req.query = query or {}
    req.headers = headers or {}

    if json_body is not None:

        async def _json():
            return json_body

        req.json = _json

        async def _body():
            return json.dumps(json_body).encode()

        req.body = _body
    else:

        async def _json():
            return {}

        req.json = _json

        async def _body():
            return b"{}"

        req.body = _body

    return req


def _make_agent(
    id="agent-001",
    name="Test Agent",
    agent_type="scanner",
    model="claude-3.5-sonnet",
    status="active",
    role="Full analysis",
    tasks_completed=10,
    findings_generated=5,
    avg_response_time=150,
    error_rate=0.01,
    created_at=None,
    last_active=None,
):
    """Create an agent dict for testing."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": id,
        "name": name,
        "type": agent_type,
        "model": model,
        "status": status,
        "role": role,
        "capabilities": ["analysis"],
        "tasks_completed": tasks_completed,
        "findings_generated": findings_generated,
        "avg_response_time": avg_response_time,
        "error_rate": error_rate,
        "created_at": created_at or now,
        "last_active": last_active or now,
    }


def _make_task(
    id="task-001",
    task_type="document_audit",
    priority="normal",
    status="pending",
    document_id="doc-001",
    created_at=None,
):
    """Create a task dict for testing."""
    return {
        "id": id,
        "type": task_type,
        "priority": priority,
        "status": status,
        "document_id": document_id,
        "audit_type": "security",
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_module_state():
    """Clear module-level state between tests."""
    _agents.clear()
    _task_queue.clear()
    _stream_clients.clear()
    _metrics["total_tasks_processed"] = 0
    _metrics["total_findings_generated"] = 0
    _metrics["active_sessions"] = 0
    _metrics["agent_uptime"] = {}
    yield
    _agents.clear()
    _task_queue.clear()
    _stream_clients.clear()


@pytest.fixture()
def handler():
    """Create an AgentDashboardHandler with mocked auth that always succeeds."""
    h = AgentDashboardHandler(server_context={})
    auth_ctx = MagicMock()
    auth_ctx.is_authenticated = True
    auth_ctx.user_id = "test-user"

    async def _get_auth(request, require_auth=True):
        return auth_ctx

    h.get_auth_context = _get_auth
    h.check_permission = MagicMock(return_value=True)
    return h


@pytest.fixture()
def mock_shared_state():
    """Create a mock shared state object."""
    state = AsyncMock()
    state.is_persistent = True
    state.list_agents = AsyncMock(return_value=[])
    state.get_agent = AsyncMock(return_value=None)
    state.register_agent = AsyncMock()
    state.update_agent_status = AsyncMock()
    state.list_tasks = AsyncMock(return_value=[])
    state.add_task = AsyncMock()
    state.update_task_priority = AsyncMock()
    state.get_metrics = AsyncMock(return_value={})
    return state


# ---------------------------------------------------------------------------
# Tests: can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_control_plane_agents_path(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/agents") is True

    def test_control_plane_agents_with_id(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/agents/agent-123") is True

    def test_control_plane_agents_pause(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/agents/agent-123/pause") is True

    def test_control_plane_agents_resume(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/agents/agent-123/resume") is True

    def test_control_plane_agents_metrics(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/agents/agent-123/metrics") is True

    def test_control_plane_queue(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/queue") is True

    def test_control_plane_queue_prioritize(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/queue/prioritize") is True

    def test_control_plane_metrics(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/metrics") is True

    def test_control_plane_stream(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/stream") is True

    def test_control_plane_health(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/control-plane/health") is True

    def test_unrelated_path(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/api/v1/debates") is False

    def test_unrelated_root(self):
        h = AgentDashboardHandler(server_context={})
        assert h.can_handle("/health") is False


class TestRoutes:
    def test_routes_defined(self):
        assert len(AgentDashboardHandler.ROUTES) > 0

    def test_expected_routes_present(self):
        expected = [
            "/api/v1/agent-dashboard/agents",
            "/api/v1/agent-dashboard/queue",
            "/api/v1/agent-dashboard/metrics",
            "/api/v1/agent-dashboard/stream",
            "/api/v1/agent-dashboard/health",
        ]
        for route in expected:
            assert route in AgentDashboardHandler.ROUTES, f"Expected route: {route}"


class TestBackwardCompatibility:
    def test_control_plane_handler_alias(self):
        """Test that ControlPlaneHandler is an alias for AgentDashboardHandler."""
        assert ControlPlaneHandler is AgentDashboardHandler


# ---------------------------------------------------------------------------
# Tests: Authentication & Permissions
# ---------------------------------------------------------------------------


class TestAuthPermissions:
    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        h = AgentDashboardHandler(server_context={})

        async def _fail_auth(request, require_auth=True):
            raise UnauthorizedError("no token")

        h.get_auth_context = _fail_auth
        req = _make_request()
        result = await h.handle_request(req)
        _, status = _parse_result(result)
        assert status == 401

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        h = AgentDashboardHandler(server_context={})
        auth_ctx = MagicMock()

        async def _ok_auth(request, require_auth=True):
            return auth_ctx

        h.get_auth_context = _ok_auth

        def _deny(ctx, perm, resource_id=None):
            raise ForbiddenError(f"Permission denied: {perm}", permission=perm)

        h.check_permission = _deny
        req = _make_request()
        result = await h.handle_request(req)
        _, status = _parse_result(result)
        assert status == 403

    @pytest.mark.asyncio
    async def test_get_requires_read_permission(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        h = AgentDashboardHandler(server_context={})
        auth_ctx = MagicMock()

        async def _ok_auth(request, require_auth=True):
            return auth_ctx

        h.get_auth_context = _ok_auth

        checked_permissions = []

        def _check(ctx, perm, resource_id=None):
            checked_permissions.append(perm)
            raise ForbiddenError(f"denied: {perm}", permission=perm)

        h.check_permission = _check
        req = _make_request(method="GET")
        await h.handle_request(req)
        assert CONTROL_PLANE_READ_PERMISSION in checked_permissions

    @pytest.mark.asyncio
    async def test_post_requires_write_permission(self):
        from aragora.server.handlers.utils.auth import ForbiddenError

        h = AgentDashboardHandler(server_context={})
        auth_ctx = MagicMock()

        async def _ok_auth(request, require_auth=True):
            return auth_ctx

        h.get_auth_context = _ok_auth

        checked_permissions = []

        def _check(ctx, perm, resource_id=None):
            checked_permissions.append(perm)
            raise ForbiddenError(f"denied: {perm}", permission=perm)

        h.check_permission = _check
        req = _make_request(method="POST", path="/api/v1/control-plane/agents/a1/pause")
        await h.handle_request(req)
        assert CONTROL_PLANE_WRITE_PERMISSION in checked_permissions


# ---------------------------------------------------------------------------
# Tests: List Agents
# ---------------------------------------------------------------------------


class TestListAgents:
    @pytest.mark.asyncio
    async def test_list_agents_empty_returns_defaults(self, handler):
        """When no agents exist, default agents are returned."""
        req = _make_request(path="/api/v1/control-plane/agents")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] > 0
        assert len(body["agents"]) > 0
        # Check for default agents
        agent_ids = [a["id"] for a in body["agents"]]
        assert "agent-gemini-scanner" in agent_ids

    @pytest.mark.asyncio
    async def test_list_agents_with_existing_agents(self, handler):
        """List agents when agents are pre-populated."""
        _agents["agent-001"] = _make_agent(id="agent-001", status="active")
        _agents["agent-002"] = _make_agent(id="agent-002", status="paused")

        req = _make_request(path="/api/v1/control-plane/agents")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] == 2
        assert body["active"] == 1
        assert body["paused"] == 1

    @pytest.mark.asyncio
    async def test_list_agents_status_filter(self, handler):
        """Filter agents by status."""
        _agents["agent-001"] = _make_agent(id="agent-001", status="active")
        _agents["agent-002"] = _make_agent(id="agent-002", status="paused")
        _agents["agent-003"] = _make_agent(id="agent-003", status="idle")

        req = _make_request(path="/api/v1/control-plane/agents", query={"status": "active"})

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] == 1
        assert body["agents"][0]["id"] == "agent-001"

    @pytest.mark.asyncio
    async def test_list_agents_type_filter(self, handler):
        """Filter agents by type."""
        _agents["agent-001"] = _make_agent(id="agent-001", agent_type="scanner")
        _agents["agent-002"] = _make_agent(id="agent-002", agent_type="reasoner")

        req = _make_request(path="/api/v1/control-plane/agents", query={"type": "scanner"})

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] == 1
        assert body["agents"][0]["type"] == "scanner"

    @pytest.mark.asyncio
    async def test_list_agents_with_shared_state(self, handler, mock_shared_state):
        """List agents using shared state backend."""
        agents = [_make_agent(id="shared-agent-001")]
        mock_shared_state.list_agents = AsyncMock(return_value=agents)

        req = _make_request(path="/api/v1/control-plane/agents")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=mock_shared_state,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] == 1
        mock_shared_state.list_agents.assert_called()


# ---------------------------------------------------------------------------
# Tests: Get Agent
# ---------------------------------------------------------------------------


class TestGetAgent:
    @pytest.mark.asyncio
    async def test_get_agent_success(self, handler):
        """Get an existing agent by ID."""
        _agents["agent-001"] = _make_agent(id="agent-001", name="My Agent")

        req = _make_request(path="/api/v1/control-plane/agents/agent-001")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["id"] == "agent-001"
        assert body["name"] == "My Agent"

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, handler):
        """Get non-existent agent returns 404."""
        req = _make_request(path="/api/v1/control-plane/agents/nonexistent")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 404
        assert "not found" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_get_agent_with_shared_state(self, handler, mock_shared_state):
        """Get agent using shared state backend."""
        agent = _make_agent(id="shared-agent-001")
        mock_shared_state.get_agent = AsyncMock(return_value=agent)

        req = _make_request(path="/api/v1/control-plane/agents/shared-agent-001")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=mock_shared_state,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        mock_shared_state.get_agent.assert_called_with("shared-agent-001")


# ---------------------------------------------------------------------------
# Tests: Pause Agent
# ---------------------------------------------------------------------------


class TestPauseAgent:
    @pytest.mark.asyncio
    async def test_pause_agent_success(self, handler):
        """Pause an active agent."""
        _agents["agent-001"] = _make_agent(id="agent-001", status="active")

        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/agents/agent-001/pause",
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["status"] == "paused"
        assert _agents["agent-001"]["status"] == "paused"
        assert "paused_at" in _agents["agent-001"]

    @pytest.mark.asyncio
    async def test_pause_agent_not_found(self, handler):
        """Pause non-existent agent returns 404."""
        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/agents/nonexistent/pause",
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_pause_agent_not_active(self, handler):
        """Pause already paused agent returns 400."""
        _agents["agent-001"] = _make_agent(id="agent-001", status="paused")

        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/agents/agent-001/pause",
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 400
        assert "not active" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_pause_agent_with_shared_state(self, handler, mock_shared_state):
        """Pause agent using shared state backend."""
        agent = _make_agent(id="agent-001", status="active")
        paused_agent = _make_agent(id="agent-001", status="paused")
        mock_shared_state.get_agent = AsyncMock(return_value=agent)
        mock_shared_state.update_agent_status = AsyncMock(return_value=paused_agent)

        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/agents/agent-001/pause",
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=mock_shared_state,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        mock_shared_state.update_agent_status.assert_called_with("agent-001", "paused")


# ---------------------------------------------------------------------------
# Tests: Resume Agent
# ---------------------------------------------------------------------------


class TestResumeAgent:
    @pytest.mark.asyncio
    async def test_resume_agent_success(self, handler):
        """Resume a paused agent."""
        _agents["agent-001"] = _make_agent(id="agent-001", status="paused")

        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/agents/agent-001/resume",
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["status"] == "active"
        assert _agents["agent-001"]["status"] == "active"
        assert "resumed_at" in _agents["agent-001"]

    @pytest.mark.asyncio
    async def test_resume_agent_not_found(self, handler):
        """Resume non-existent agent returns 404."""
        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/agents/nonexistent/resume",
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_resume_agent_not_paused(self, handler):
        """Resume active agent returns 400."""
        _agents["agent-001"] = _make_agent(id="agent-001", status="active")

        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/agents/agent-001/resume",
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 400
        assert "not paused" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_resume_agent_with_shared_state(self, handler, mock_shared_state):
        """Resume agent using shared state backend."""
        agent = _make_agent(id="agent-001", status="paused")
        resumed_agent = _make_agent(id="agent-001", status="active")
        mock_shared_state.get_agent = AsyncMock(return_value=agent)
        mock_shared_state.update_agent_status = AsyncMock(return_value=resumed_agent)

        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/agents/agent-001/resume",
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=mock_shared_state,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        mock_shared_state.update_agent_status.assert_called_with("agent-001", "active")


# ---------------------------------------------------------------------------
# Tests: Agent Metrics
# ---------------------------------------------------------------------------


class TestAgentMetrics:
    @pytest.mark.asyncio
    async def test_get_agent_metrics_success(self, handler):
        """Get metrics for a specific agent."""
        _agents["agent-001"] = _make_agent(
            id="agent-001",
            tasks_completed=50,
            findings_generated=25,
            avg_response_time=200,
            error_rate=0.02,
        )

        req = _make_request(path="/api/v1/control-plane/agents/agent-001/metrics")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["agent_id"] == "agent-001"
        assert body["tasks_completed"] == 50
        assert body["findings_generated"] == 25
        assert body["average_response_time_ms"] == 200
        assert body["error_rate"] == 0.02

    @pytest.mark.asyncio
    async def test_get_agent_metrics_not_found(self, handler):
        """Get metrics for non-existent agent returns 404."""
        req = _make_request(path="/api/v1/control-plane/agents/nonexistent/metrics")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 404


# ---------------------------------------------------------------------------
# Tests: Task Queue
# ---------------------------------------------------------------------------


class TestGetQueue:
    @pytest.mark.asyncio
    async def test_get_queue_empty_returns_sample(self, handler):
        """When queue is empty, sample tasks are returned."""
        req = _make_request(path="/api/v1/control-plane/queue")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] > 0
        assert len(body["tasks"]) > 0

    @pytest.mark.asyncio
    async def test_get_queue_with_tasks(self, handler):
        """Get queue with pre-populated tasks."""
        _task_queue.append(_make_task(id="task-001", priority="high", status="pending"))
        _task_queue.append(_make_task(id="task-002", priority="normal", status="processing"))
        _task_queue.append(_make_task(id="task-003", priority="low", status="pending"))

        req = _make_request(path="/api/v1/control-plane/queue")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["total"] == 3
        assert body["by_priority"]["high"] == 1
        assert body["by_priority"]["normal"] == 1
        assert body["by_priority"]["low"] == 1
        assert body["by_status"]["pending"] == 2
        assert body["by_status"]["processing"] == 1

    @pytest.mark.asyncio
    async def test_get_queue_with_shared_state(self, handler, mock_shared_state):
        """Get queue using shared state backend."""
        tasks = [_make_task(id="shared-task-001")]
        mock_shared_state.list_tasks = AsyncMock(return_value=tasks)

        req = _make_request(path="/api/v1/control-plane/queue")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=mock_shared_state,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        mock_shared_state.list_tasks.assert_called()


class TestPrioritizeQueue:
    @pytest.mark.asyncio
    async def test_prioritize_task_success(self, handler):
        """Prioritize a task in the queue."""
        _task_queue.append(_make_task(id="task-001", priority="normal"))

        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/queue/prioritize",
            json_body={"task_id": "task-001", "priority": "high"},
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["success"] is True
        assert _task_queue[0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_prioritize_task_with_position(self, handler):
        """Prioritize a task and move it to a specific position."""
        _task_queue.append(_make_task(id="task-001", priority="low"))
        _task_queue.append(_make_task(id="task-002", priority="normal"))
        _task_queue.append(_make_task(id="task-003", priority="high"))

        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/queue/prioritize",
            json_body={"task_id": "task-003", "position": 0},
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert _task_queue[0]["id"] == "task-003"

    @pytest.mark.asyncio
    async def test_prioritize_task_missing_task_id(self, handler):
        """Prioritize without task_id returns 400."""
        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/queue/prioritize",
            json_body={"priority": "high"},
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 400
        assert "task_id" in body.get("error", "").lower()

    @pytest.mark.asyncio
    async def test_prioritize_task_not_found(self, handler):
        """Prioritize non-existent task returns 404."""
        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/queue/prioritize",
            json_body={"task_id": "nonexistent", "priority": "high"},
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 404

    @pytest.mark.asyncio
    async def test_prioritize_invalid_json(self, handler):
        """Prioritize with invalid JSON returns 400."""
        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/queue/prioritize",
        )
        # Override json to raise an error
        req.json = AsyncMock(side_effect=json.JSONDecodeError("test", "doc", 0))

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 400

    @pytest.mark.asyncio
    async def test_prioritize_with_shared_state(self, handler, mock_shared_state):
        """Prioritize task using shared state backend."""
        task = _make_task(id="task-001", priority="high")
        mock_shared_state.update_task_priority = AsyncMock(return_value=task)

        req = _make_request(
            method="POST",
            path="/api/v1/control-plane/queue/prioritize",
            json_body={"task_id": "task-001", "priority": "high"},
        )

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=mock_shared_state,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        mock_shared_state.update_task_priority.assert_called()


# ---------------------------------------------------------------------------
# Tests: System Metrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    @pytest.mark.asyncio
    async def test_get_metrics_success(self, handler):
        """Get system-wide metrics."""
        _agents["agent-001"] = _make_agent(status="active", avg_response_time=100)
        _agents["agent-002"] = _make_agent(status="paused", avg_response_time=200)
        _task_queue.append(_make_task(status="pending"))
        _task_queue.append(_make_task(status="processing"))
        _metrics["total_tasks_processed"] = 100
        _metrics["total_findings_generated"] = 50

        req = _make_request(path="/api/v1/control-plane/metrics")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert "timestamp" in body
        assert body["agents"]["total"] == 2
        assert body["agents"]["active"] == 1
        assert body["agents"]["paused"] == 1
        assert body["queue"]["total_tasks"] == 2
        assert body["queue"]["pending"] == 1
        assert body["queue"]["processing"] == 1
        assert body["processing"]["total_tasks_processed"] == 100
        assert body["processing"]["total_findings_generated"] == 50

    @pytest.mark.asyncio
    async def test_get_metrics_with_shared_state(self, handler, mock_shared_state):
        """Get metrics using shared state backend."""
        metrics = {"timestamp": datetime.now(timezone.utc).isoformat()}
        mock_shared_state.get_metrics = AsyncMock(return_value=metrics)

        req = _make_request(path="/api/v1/control-plane/metrics")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=mock_shared_state,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        mock_shared_state.get_metrics.assert_called()


# ---------------------------------------------------------------------------
# Tests: Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, handler):
        """Health check returns healthy when agents are active."""
        _agents["agent-001"] = _make_agent(status="active")

        req = _make_request(path="/api/v1/control-plane/health")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["status"] == "healthy"
        assert body["persistence"]["enabled"] is False
        assert body["persistence"]["backend"] == "in_memory"
        assert body["components"]["agents"]["status"] == "healthy"
        assert body["components"]["api"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, handler):
        """Health check returns degraded when no agents are active."""
        _agents["agent-001"] = _make_agent(status="paused")

        req = _make_request(path="/api/v1/control-plane/health")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["status"] == "degraded"
        assert body["components"]["agents"]["status"] == "no_active_agents"

    @pytest.mark.asyncio
    async def test_health_check_with_shared_state(self, handler, mock_shared_state):
        """Health check using shared state backend."""
        mock_shared_state.list_agents = AsyncMock(
            return_value=[_make_agent(status="active")]
        )
        mock_shared_state.list_tasks = AsyncMock(return_value=[])

        req = _make_request(path="/api/v1/control-plane/health")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=mock_shared_state,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 200
        assert body["persistence"]["enabled"] is True
        assert body["persistence"]["backend"] == "redis"


# ---------------------------------------------------------------------------
# Tests: Not Found
# ---------------------------------------------------------------------------


class TestNotFound:
    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_404(self, handler):
        """Unknown endpoint returns 404."""
        req = _make_request(path="/api/v1/control-plane/unknown")

        with patch(
            "aragora.server.handlers.features.control_plane._get_shared_state",
            return_value=None,
        ):
            result = await handler.handle_request(req)

        body, status = _parse_result(result)
        assert status == 404


# ---------------------------------------------------------------------------
# Tests: Utility Methods
# ---------------------------------------------------------------------------


class TestUtilityMethods:
    def test_get_default_agents(self, handler):
        """Test default agents generator."""
        agents = handler._get_default_agents()
        assert len(agents) > 0
        for agent in agents:
            assert "id" in agent
            assert "name" in agent
            assert "type" in agent
            assert "model" in agent
            assert "status" in agent

    def test_get_sample_queue(self, handler):
        """Test sample queue generator."""
        tasks = handler._get_sample_queue()
        assert len(tasks) > 0
        for task in tasks:
            assert "id" in task
            assert "type" in task
            assert "priority" in task
            assert "status" in task

    def test_calculate_avg_task_duration_empty(self, handler):
        """Calculate avg task duration with no agents."""
        result = handler._calculate_avg_task_duration([])
        assert result == 0.0

    def test_calculate_avg_task_duration(self, handler):
        """Calculate avg task duration with agents."""
        agents = [
            {"avg_response_time": 100},
            {"avg_response_time": 200},
            {"avg_response_time": 300},
        ]
        result = handler._calculate_avg_task_duration(agents)
        assert result == 200.0

    def test_calculate_avg_task_duration_with_zeros(self, handler):
        """Calculate avg task duration ignoring zero values."""
        agents = [
            {"avg_response_time": 100},
            {"avg_response_time": 0},
            {"avg_response_time": 200},
        ]
        result = handler._calculate_avg_task_duration(agents)
        assert result == 150.0

    def test_calculate_throughput(self, handler):
        """Calculate throughput."""
        agents = [
            {"tasks_completed": 60},
            {"tasks_completed": 120},
        ]
        result = handler._calculate_throughput(agents)
        assert result == 3.0  # 180 / 60

    def test_calculate_error_rate_empty(self, handler):
        """Calculate error rate with no agents."""
        result = handler._calculate_error_rate([])
        assert result == 0.0

    def test_calculate_error_rate(self, handler):
        """Calculate error rate."""
        agents = [
            {"error_rate": 0.01},
            {"error_rate": 0.03},
        ]
        result = handler._calculate_error_rate(agents)
        assert result == 0.02

    def test_json_response(self, handler):
        """Test JSON response helper."""
        result = handler._json_response(200, {"key": "value"})
        assert result["status"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        assert '"key": "value"' in result["body"]

    def test_error_response(self, handler):
        """Test error response helper."""
        result = handler._error_response(400, "Bad request")
        assert result["status"] == 400
        body = json.loads(result["body"])
        assert body["error"] == "Bad request"


# ---------------------------------------------------------------------------
# Tests: Broadcast Updates
# ---------------------------------------------------------------------------


class TestBroadcastUpdates:
    @pytest.mark.asyncio
    async def test_broadcast_update(self, handler):
        """Test broadcasting updates to stream clients."""
        import asyncio

        queue1 = asyncio.Queue()
        queue2 = asyncio.Queue()
        _stream_clients.append(queue1)
        _stream_clients.append(queue2)

        await handler._broadcast_update({"type": "test", "data": "value"})

        event1 = await asyncio.wait_for(queue1.get(), timeout=1.0)
        event2 = await asyncio.wait_for(queue2.get(), timeout=1.0)
        assert event1["type"] == "test"
        assert event2["type"] == "test"

    @pytest.mark.asyncio
    async def test_broadcast_update_full_queue(self, handler):
        """Test broadcasting to full queues doesn't raise."""
        import asyncio

        # Create a queue with maxsize=1 and fill it
        queue = asyncio.Queue(maxsize=1)
        await queue.put({"type": "blocking"})
        _stream_clients.append(queue)

        # This should not raise even though queue is full
        await handler._broadcast_update({"type": "test"})

        # The blocking event should still be there
        event = await asyncio.wait_for(queue.get(), timeout=1.0)
        assert event["type"] == "blocking"


# ---------------------------------------------------------------------------
# Tests: Parse JSON Body
# ---------------------------------------------------------------------------


class TestParseJsonBody:
    @pytest.mark.asyncio
    async def test_parse_json_body_callable(self, handler):
        """Parse JSON body from callable."""

        async def _json():
            return {"key": "value"}

        req = SimpleNamespace(json=_json)
        result = await handler._parse_json_body(req)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_parse_json_body_from_body(self, handler):
        """Parse JSON body from body attribute."""

        async def _body():
            return b'{"key": "value"}'

        # Create a SimpleNamespace without json attribute
        req = SimpleNamespace(body=_body)
        # The handler's _parse_json_body falls back to body when json is not available
        result = await handler._parse_json_body(req)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_parse_json_body_empty(self, handler):
        """Parse empty JSON body."""
        req = SimpleNamespace()
        result = await handler._parse_json_body(req)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: Handler Creation
# ---------------------------------------------------------------------------


class TestHandlerCreation:
    def test_handler_creation(self):
        """Test handler can be created."""
        h = AgentDashboardHandler(server_context={})
        assert h is not None

    def test_handler_with_context(self):
        """Test handler creation with server context."""
        ctx = {"storage": MagicMock(), "user_store": MagicMock()}
        h = AgentDashboardHandler(server_context=ctx)
        assert h is not None
        assert h.ctx == ctx


# ---------------------------------------------------------------------------
# Tests: SSE Response
# ---------------------------------------------------------------------------


class TestSSEResponse:
    def test_sse_response(self, handler):
        """Test SSE response helper."""

        async def generator():
            yield "data: test\n\n"

        result = handler._sse_response(generator())
        assert result["status"] == 200
        assert result["headers"]["Content-Type"] == "text/event-stream"
        assert result["headers"]["Cache-Control"] == "no-cache"
        assert result["headers"]["Connection"] == "keep-alive"
