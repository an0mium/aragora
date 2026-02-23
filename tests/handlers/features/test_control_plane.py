"""
Tests for AgentDashboardHandler (control_plane.py).

Comprehensive test coverage for the agent dashboard / control plane endpoints:

Routing (can_handle):
- /api/v1/control-plane/agents              GET (list)
- /api/v1/control-plane/agents/{id}         GET (detail)
- /api/v1/control-plane/agents/{id}/pause   POST
- /api/v1/control-plane/agents/{id}/resume  POST
- /api/v1/control-plane/agents/{id}/metrics GET
- /api/v1/control-plane/queue               GET
- /api/v1/control-plane/queue/prioritize    POST
- /api/v1/control-plane/metrics             GET
- /api/v1/control-plane/stream              GET (SSE)
- /api/v1/control-plane/health              GET

Legacy routes:
- /api/v1/agent-dashboard/*                 backward compat
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.control_plane import (
    AgentDashboardHandler,
    ControlPlaneHandler,
    _agents,
    _task_queue,
    _stream_clients,
    _metrics,
)

# =============================================================================
# Module path for patching
# =============================================================================

MODULE = "aragora.server.handlers.features.control_plane"

# =============================================================================
# Mock Request
# =============================================================================


@dataclass
class MockHTTPRequest:
    """Mock HTTP request for handler tests."""

    path: str = "/api/v1/control-plane/agents"
    method: str = "GET"
    body: dict[str, Any] | None = None
    query: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self._raw = json.dumps(self.body).encode() if self.body is not None else b"{}"

    async def read(self) -> bytes:
        return self._raw

    async def json(self) -> dict[str, Any]:
        return self.body if self.body is not None else {}


# =============================================================================
# Helpers
# =============================================================================


def _status(result: dict[str, Any]) -> int:
    """Extract status code from handler response dict."""
    return result.get("status", 0)


def _body(result: dict[str, Any]) -> dict[str, Any]:
    """Parse the JSON body from a handler response dict."""
    raw = result.get("body", "{}")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _make_agent(
    agent_id: str = "agent-1",
    status: str = "active",
    agent_type: str = "scanner",
    **overrides: Any,
) -> dict[str, Any]:
    """Create an agent dict for seeding _agents."""
    now = datetime.now(timezone.utc).isoformat()
    agent: dict[str, Any] = {
        "id": agent_id,
        "name": f"Agent {agent_id}",
        "type": agent_type,
        "model": "test-model",
        "status": status,
        "role": "Test role",
        "capabilities": ["test"],
        "tasks_completed": 5,
        "findings_generated": 2,
        "avg_response_time": 150,
        "error_rate": 0.01,
        "created_at": now,
        "last_active": now,
        "uptime_seconds": 3600,
    }
    agent.update(overrides)
    return agent


def _make_task(
    task_id: str = "task-1",
    priority: str = "normal",
    status: str = "pending",
    **overrides: Any,
) -> dict[str, Any]:
    """Create a task dict for seeding _task_queue."""
    now = datetime.now(timezone.utc).isoformat()
    task: dict[str, Any] = {
        "id": task_id,
        "type": "document_audit",
        "priority": priority,
        "status": status,
        "document_id": "doc-001",
        "audit_type": "security",
        "created_at": now,
    }
    task.update(overrides)
    return task


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create an AgentDashboardHandler instance."""
    return AgentDashboardHandler(server_context={})


@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear module-level mutable state before each test."""
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
    _metrics["total_tasks_processed"] = 0
    _metrics["total_findings_generated"] = 0
    _metrics["active_sessions"] = 0
    _metrics["agent_uptime"] = {}


@pytest.fixture
def seeded_agents():
    """Seed three agents: 2 active, 1 idle."""
    agents = [
        _make_agent("agent-a", status="active", agent_type="scanner"),
        _make_agent("agent-b", status="active", agent_type="reasoner"),
        _make_agent("agent-c", status="idle", agent_type="verifier"),
    ]
    for a in agents:
        _agents[a["id"]] = a
    return agents


@pytest.fixture
def seeded_tasks():
    """Seed three tasks with mixed priorities."""
    tasks = [
        _make_task("task-1", priority="high", status="pending"),
        _make_task("task-2", priority="normal", status="pending"),
        _make_task("task-3", priority="low", status="processing"),
    ]
    _task_queue.extend(tasks)
    return tasks


# =============================================================================
# 1. can_handle() Tests
# =============================================================================


class TestCanHandle:
    """Test can_handle routes correctly."""

    def test_handles_control_plane_agents(self, handler):
        assert handler.can_handle("/api/v1/control-plane/agents") is True

    def test_handles_control_plane_agents_id(self, handler):
        assert handler.can_handle("/api/v1/control-plane/agents/abc-123") is True

    def test_handles_control_plane_pause(self, handler):
        assert handler.can_handle("/api/v1/control-plane/agents/abc/pause") is True

    def test_handles_control_plane_resume(self, handler):
        assert handler.can_handle("/api/v1/control-plane/agents/abc/resume") is True

    def test_handles_control_plane_agent_metrics(self, handler):
        assert handler.can_handle("/api/v1/control-plane/agents/abc/metrics") is True

    def test_handles_control_plane_tasks(self, handler):
        assert handler.can_handle("/api/v1/control-plane/tasks") is True

    def test_handles_control_plane_queue(self, handler):
        assert handler.can_handle("/api/v1/control-plane/queue") is True

    def test_handles_control_plane_metrics(self, handler):
        assert handler.can_handle("/api/v1/control-plane/metrics") is True

    def test_handles_control_plane_health(self, handler):
        assert handler.can_handle("/api/v1/control-plane/health") is True

    def test_handles_control_plane_stream(self, handler):
        assert handler.can_handle("/api/v1/control-plane/stream") is True

    def test_does_not_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_does_not_handle_agent_dashboard_path(self, handler):
        # can_handle only checks /api/v1/control-plane/ prefix
        assert handler.can_handle("/api/v1/agent-dashboard/agents") is False

    def test_does_not_handle_root(self, handler):
        assert handler.can_handle("/") is False

    def test_routes_class_attribute_count(self):
        assert len(AgentDashboardHandler.ROUTES) == 37


# =============================================================================
# 2. Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Test that ControlPlaneHandler is an alias."""

    def test_alias_is_same_class(self):
        assert ControlPlaneHandler is AgentDashboardHandler

    def test_alias_instantiation(self):
        h = ControlPlaneHandler(server_context={})
        assert isinstance(h, AgentDashboardHandler)


# =============================================================================
# 3. List Agents Tests
# =============================================================================


class TestListAgents:
    """Test GET /api/v1/control-plane/agents."""

    @pytest.mark.asyncio
    async def test_list_agents_returns_default_when_empty(self, handler):
        """When no agents are seeded, defaults are populated."""
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        assert body["active"] == 2
        assert body["idle"] == 1
        assert len(body["agents"]) == 3

    @pytest.mark.asyncio
    async def test_list_agents_returns_seeded_agents(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_list_agents_filter_by_status(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents",
            method="GET",
            query={"status": "idle"},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["total"] == 1
        assert body["agents"][0]["id"] == "agent-c"

    @pytest.mark.asyncio
    async def test_list_agents_filter_by_type(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents",
            method="GET",
            query={"type": "scanner"},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["total"] == 1
        assert body["agents"][0]["type"] == "scanner"

    @pytest.mark.asyncio
    async def test_list_agents_filter_returns_empty(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents",
            method="GET",
            query={"status": "paused"},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["total"] == 0
        assert body["paused"] == 0

    @pytest.mark.asyncio
    async def test_list_agents_with_shared_state(self, handler):
        """When shared state is available, it delegates to shared state."""
        mock_shared = AsyncMock()
        mock_shared.list_agents = AsyncMock(
            return_value=[
                _make_agent("shared-1", status="active"),
            ]
        )

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_agents_shared_state_populates_defaults(self, handler):
        """Shared state with empty agents populates defaults."""
        mock_shared = AsyncMock()
        mock_shared.list_agents = AsyncMock(
            side_effect=[
                [],  # first call returns empty
                [_make_agent("default-1")],  # after register
            ]
        )
        mock_shared.register_agent = AsyncMock()

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        assert mock_shared.register_agent.called

    @pytest.mark.asyncio
    async def test_list_agents_response_counts(self, handler, seeded_agents):
        # Add a paused agent
        _agents["agent-d"] = _make_agent("agent-d", status="paused")

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["total"] == 4
        assert body["active"] == 2
        assert body["paused"] == 1
        assert body["idle"] == 1


# =============================================================================
# 4. Get Agent Tests
# =============================================================================


class TestGetAgent:
    """Test GET /api/v1/control-plane/agents/{id}."""

    @pytest.mark.asyncio
    async def test_get_agent_found(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-a",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "agent-a"

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/nonexistent",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"]

    @pytest.mark.asyncio
    async def test_get_agent_with_shared_state(self, handler):
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=_make_agent("shared-x"))

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/shared-x",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "shared-x"

    @pytest.mark.asyncio
    async def test_get_agent_shared_state_not_found(self, handler):
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=None)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/no-such",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 404


# =============================================================================
# 5. Pause Agent Tests
# =============================================================================


class TestPauseAgent:
    """Test POST /api/v1/control-plane/agents/{id}/pause."""

    @pytest.mark.asyncio
    async def test_pause_active_agent(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-a/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "paused"
        assert body["paused_at"] is not None
        assert _agents["agent-a"]["status"] == "paused"

    @pytest.mark.asyncio
    async def test_pause_already_paused_agent(self, handler):
        _agents["agent-p"] = _make_agent("agent-p", status="paused")

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-p/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 400
        body = _body(result)
        assert "not active" in body["error"]

    @pytest.mark.asyncio
    async def test_pause_idle_agent(self, handler):
        _agents["agent-i"] = _make_agent("agent-i", status="idle")

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-i/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_pause_nonexistent_agent(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/no-such/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_pause_broadcasts_update(self, handler, seeded_agents):
        queue = asyncio.Queue()
        _stream_clients.append(queue)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-a/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        event = queue.get_nowait()
        assert event["type"] == "agent_paused"
        assert event["agent_id"] == "agent-a"

    @pytest.mark.asyncio
    async def test_pause_with_shared_state(self, handler):
        agent_data = _make_agent("shared-1", status="active")
        updated_data = {**agent_data, "status": "paused"}
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=agent_data)
        mock_shared.update_agent_status = AsyncMock(return_value=updated_data)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/shared-1/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        mock_shared.update_agent_status.assert_awaited_once_with("shared-1", "paused")

    @pytest.mark.asyncio
    async def test_pause_shared_state_not_active(self, handler):
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=_make_agent("s-1", status="idle"))

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/s-1/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_pause_shared_state_not_found(self, handler):
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=None)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/missing/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 404


# =============================================================================
# 6. Resume Agent Tests
# =============================================================================


class TestResumeAgent:
    """Test POST /api/v1/control-plane/agents/{id}/resume."""

    @pytest.mark.asyncio
    async def test_resume_paused_agent(self, handler):
        _agents["agent-p"] = _make_agent("agent-p", status="paused")

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-p/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "active"
        assert body["resumed_at"] is not None
        assert body["paused_at"] is None

    @pytest.mark.asyncio
    async def test_resume_active_agent_fails(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-a/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 400
        body = _body(result)
        assert "not paused" in body["error"]

    @pytest.mark.asyncio
    async def test_resume_idle_agent_fails(self, handler):
        _agents["agent-i"] = _make_agent("agent-i", status="idle")

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-i/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_resume_nonexistent_agent(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/no-such/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_resume_broadcasts_update(self, handler):
        _agents["agent-p"] = _make_agent("agent-p", status="paused")
        queue = asyncio.Queue()
        _stream_clients.append(queue)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-p/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        event = queue.get_nowait()
        assert event["type"] == "agent_resumed"
        assert event["agent_id"] == "agent-p"

    @pytest.mark.asyncio
    async def test_resume_with_shared_state(self, handler):
        agent_data = _make_agent("shared-2", status="paused")
        updated_data = {**agent_data, "status": "active"}
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=agent_data)
        mock_shared.update_agent_status = AsyncMock(return_value=updated_data)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/shared-2/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        mock_shared.update_agent_status.assert_awaited_once_with("shared-2", "active")

    @pytest.mark.asyncio
    async def test_resume_shared_state_not_paused(self, handler):
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=_make_agent("s-2", status="active"))

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/s-2/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_resume_shared_state_not_found(self, handler):
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=None)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/missing/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 404


# =============================================================================
# 7. Agent Metrics Tests
# =============================================================================


class TestGetAgentMetrics:
    """Test GET /api/v1/control-plane/agents/{id}/metrics."""

    @pytest.mark.asyncio
    async def test_get_agent_metrics(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-a/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == "agent-a"
        assert body["tasks_completed"] == 5
        assert body["findings_generated"] == 2
        assert body["average_response_time_ms"] == 150
        assert body["error_rate"] == 0.01
        assert body["uptime_seconds"] == 3600

    @pytest.mark.asyncio
    async def test_get_agent_metrics_not_found(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/no-such/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_agent_metrics_defaults(self, handler):
        """Agent with no metrics fields returns zeros."""
        _agents["agent-bare"] = {"id": "agent-bare", "status": "active"}

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-bare/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["tasks_completed"] == 0
        assert body["findings_generated"] == 0
        assert body["average_response_time_ms"] == 0
        assert body["error_rate"] == 0.0
        assert body["last_active"] is None
        assert body["uptime_seconds"] == 0

    @pytest.mark.asyncio
    async def test_get_agent_metrics_with_shared_state(self, handler):
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=_make_agent("sm-1"))

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/sm-1/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_agent_metrics_shared_not_found(self, handler):
        mock_shared = AsyncMock()
        mock_shared.get_agent = AsyncMock(return_value=None)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/nope/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 404


# =============================================================================
# 8. Queue Tests
# =============================================================================


class TestGetQueue:
    """Test GET /api/v1/control-plane/queue."""

    @pytest.mark.asyncio
    async def test_get_queue_returns_sample_when_empty(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["tasks"]) == 2

    @pytest.mark.asyncio
    async def test_get_queue_returns_seeded_tasks(self, handler, seeded_tasks):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["total"] == 3
        assert body["by_priority"]["high"] == 1
        assert body["by_priority"]["normal"] == 1
        assert body["by_priority"]["low"] == 1
        assert body["by_status"]["pending"] == 2
        assert body["by_status"]["processing"] == 1

    @pytest.mark.asyncio
    async def test_get_queue_with_shared_state(self, handler):
        mock_shared = AsyncMock()
        mock_shared.list_tasks = AsyncMock(
            return_value=[
                _make_task("st-1", priority="high"),
            ]
        )

        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_get_queue_shared_state_populates_sample(self, handler):
        mock_shared = AsyncMock()
        mock_shared.list_tasks = AsyncMock(
            side_effect=[
                [],  # first call empty
                [_make_task("st-demo")],  # after add_task
            ]
        )
        mock_shared.add_task = AsyncMock()

        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        assert mock_shared.add_task.called


# =============================================================================
# 9. Prioritize Queue Tests
# =============================================================================


class TestPrioritizeQueue:
    """Test POST /api/v1/control-plane/queue/prioritize."""

    @pytest.mark.asyncio
    async def test_prioritize_task(self, handler, seeded_tasks):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
            body={"task_id": "task-2", "priority": "high"},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["task"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_prioritize_task_with_position(self, handler, seeded_tasks):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
            body={"task_id": "task-3", "priority": "high", "position": 0},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        assert _task_queue[0]["id"] == "task-3"
        assert _task_queue[0]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_prioritize_task_not_found(self, handler, seeded_tasks):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
            body={"task_id": "no-such", "priority": "high"},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_prioritize_missing_task_id(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
            body={"priority": "high"},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 400
        body = _body(result)
        assert "task_id" in body["error"]

    @pytest.mark.asyncio
    async def test_prioritize_invalid_json(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
        )

        # Force _parse_json_body to raise
        async def bad_json():
            raise json.JSONDecodeError("bad", "", 0)

        req.json = bad_json

        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 400
        body = _body(result)
        assert "Invalid JSON" in body["error"]

    @pytest.mark.asyncio
    async def test_prioritize_broadcasts_update(self, handler, seeded_tasks):
        queue = asyncio.Queue()
        _stream_clients.append(queue)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
            body={"task_id": "task-1", "priority": "low"},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        event = queue.get_nowait()
        assert event["type"] == "queue_updated"
        assert event["task_id"] == "task-1"

    @pytest.mark.asyncio
    async def test_prioritize_only_position_no_priority(self, handler, seeded_tasks):
        """Position can be updated without changing priority."""
        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
            body={"task_id": "task-3", "position": 0},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        assert _task_queue[0]["id"] == "task-3"
        # Priority unchanged
        assert _task_queue[0]["priority"] == "low"

    @pytest.mark.asyncio
    async def test_prioritize_position_beyond_end(self, handler, seeded_tasks):
        """Position beyond queue length inserts at end."""
        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
            body={"task_id": "task-1", "position": 999},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        assert _task_queue[-1]["id"] == "task-1"

    @pytest.mark.asyncio
    async def test_prioritize_with_shared_state(self, handler):
        mock_shared = AsyncMock()
        mock_shared.update_task_priority = AsyncMock(
            return_value=_make_task("st-1", priority="high")
        )

        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
            body={"task_id": "st-1", "priority": "high"},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        mock_shared.update_task_priority.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prioritize_shared_state_not_found(self, handler):
        mock_shared = AsyncMock()
        mock_shared.update_task_priority = AsyncMock(return_value=None)

        req = MockHTTPRequest(
            path="/api/v1/control-plane/queue/prioritize",
            method="POST",
            body={"task_id": "no-such", "priority": "high"},
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 404


# =============================================================================
# 10. System Metrics Tests
# =============================================================================


class TestGetMetrics:
    """Test GET /api/v1/control-plane/metrics."""

    @pytest.mark.asyncio
    async def test_get_metrics_empty_state(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["total"] == 0
        assert body["queue"]["total_tasks"] == 0
        assert body["processing"]["total_tasks_processed"] == 0
        assert body["performance"]["avg_task_duration_ms"] == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_with_agents(self, handler, seeded_agents):
        _metrics["total_tasks_processed"] = 42
        _metrics["total_findings_generated"] = 10
        _metrics["active_sessions"] = 3

        req = MockHTTPRequest(
            path="/api/v1/control-plane/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["agents"]["total"] == 3
        assert body["agents"]["active"] == 2
        assert body["agents"]["idle"] == 1
        assert body["processing"]["total_tasks_processed"] == 42
        assert body["processing"]["total_findings_generated"] == 10
        assert body["processing"]["active_sessions"] == 3

    @pytest.mark.asyncio
    async def test_get_metrics_with_queue(self, handler, seeded_tasks):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["queue"]["total_tasks"] == 3
        assert body["queue"]["pending"] == 2
        assert body["queue"]["processing"] == 1

    @pytest.mark.asyncio
    async def test_get_metrics_performance_calculations(self, handler):
        """Performance metrics are correctly calculated."""
        _agents["a1"] = _make_agent(
            "a1", avg_response_time=100, tasks_completed=60, error_rate=0.05
        )
        _agents["a2"] = _make_agent(
            "a2", avg_response_time=200, tasks_completed=120, error_rate=0.10
        )

        req = MockHTTPRequest(
            path="/api/v1/control-plane/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        # avg_task_duration_ms = (100 + 200) / 2 = 150
        assert body["performance"]["avg_task_duration_ms"] == 150.0
        # tasks_per_minute = (60 + 120) / 60 = 3.0
        assert body["performance"]["tasks_per_minute"] == 3.0
        # error_rate = (0.05 + 0.10) / 2 = 0.075
        assert abs(body["performance"]["error_rate"] - 0.075) < 0.001

    @pytest.mark.asyncio
    async def test_get_metrics_with_shared_state(self, handler):
        mock_shared = AsyncMock()
        mock_shared.get_metrics = AsyncMock(
            return_value={
                "agents": {"total": 5},
                "queue": {"total_tasks": 10},
            }
        )

        req = MockHTTPRequest(
            path="/api/v1/control-plane/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["total"] == 5

    @pytest.mark.asyncio
    async def test_get_metrics_timestamp_present(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert "timestamp" in body


# =============================================================================
# 11. Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test GET /api/v1/control-plane/health."""

    @pytest.mark.asyncio
    async def test_health_empty_healthy(self, handler):
        """No agents means healthy (empty list evaluates correctly)."""
        req = MockHTTPRequest(
            path="/api/v1/control-plane/health",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        # With no agents, the condition `not agents` is True, so status is "healthy"
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_with_active_agents(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/health",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["status"] == "healthy"
        assert body["components"]["agents"]["active"] == 2
        assert body["components"]["agents"]["total"] == 3

    @pytest.mark.asyncio
    async def test_health_degraded_no_active_agents(self, handler):
        """Agents exist but none are active."""
        _agents["idle-1"] = _make_agent("idle-1", status="idle")
        _agents["paused-1"] = _make_agent("paused-1", status="paused")

        req = MockHTTPRequest(
            path="/api/v1/control-plane/health",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["status"] == "degraded"
        assert body["components"]["agents"]["status"] == "no_active_agents"

    @pytest.mark.asyncio
    async def test_health_persistence_in_memory(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/health",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["persistence"]["enabled"] is False
        assert body["persistence"]["backend"] == "in_memory"

    @pytest.mark.asyncio
    async def test_health_persistence_redis(self, handler):
        mock_shared = AsyncMock()
        mock_shared.list_agents = AsyncMock(return_value=[_make_agent("a1")])
        mock_shared.list_tasks = AsyncMock(return_value=[])
        mock_shared.is_persistent = True

        req = MockHTTPRequest(
            path="/api/v1/control-plane/health",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=mock_shared):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["persistence"]["enabled"] is True
        assert body["persistence"]["backend"] == "redis"

    @pytest.mark.asyncio
    async def test_health_queue_component(self, handler, seeded_tasks):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/health",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["components"]["queue"]["status"] == "healthy"
        assert body["components"]["queue"]["tasks"] == 3

    @pytest.mark.asyncio
    async def test_health_api_component(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/health",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["components"]["api"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_timestamp_present(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/health",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert "timestamp" in body


# =============================================================================
# 12. Stream Updates Tests
# =============================================================================


class TestStreamUpdates:
    """Test GET /api/v1/control-plane/stream (SSE)."""

    @pytest.mark.asyncio
    async def test_stream_returns_sse_response(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/stream",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        assert result["headers"]["Content-Type"] == "text/event-stream"
        assert result["headers"]["Cache-Control"] == "no-cache"
        assert result["headers"]["Connection"] == "keep-alive"

    @pytest.mark.asyncio
    async def test_stream_registers_client(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/stream",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert len(_stream_clients) == 1

    @pytest.mark.asyncio
    async def test_stream_body_is_generator(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/stream",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        # body should be an async generator
        assert hasattr(result["body"], "__aiter__") or hasattr(result["body"], "__anext__")


# =============================================================================
# 13. Routing Tests
# =============================================================================


class TestRouting:
    """Test that handle_request routes to the correct handler."""

    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_404(self, handler):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/unknown",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 404
        body = _body(result)
        assert "not found" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_routing_agents_list_get(self, handler, seeded_agents):
        req = MockHTTPRequest(path="/api/v1/control-plane/agents", method="GET")
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert "agents" in body

    @pytest.mark.asyncio
    async def test_routing_agent_detail_get(self, handler, seeded_agents):
        req = MockHTTPRequest(path="/api/v1/control-plane/agents/agent-a", method="GET")
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "agent-a"

    @pytest.mark.asyncio
    async def test_routing_pause_post(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-a/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "paused"

    @pytest.mark.asyncio
    async def test_routing_resume_post(self, handler):
        _agents["agent-p"] = _make_agent("agent-p", status="paused")
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-p/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "active"

    @pytest.mark.asyncio
    async def test_routing_agent_metrics_get(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-a/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == "agent-a"


# =============================================================================
# 14. Broadcast / Stream Client Tests
# =============================================================================


class TestBroadcastUpdate:
    """Test the _broadcast_update mechanism."""

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_clients(self, handler):
        q1 = asyncio.Queue()
        q2 = asyncio.Queue()
        _stream_clients.extend([q1, q2])

        await handler._broadcast_update({"type": "test_event"})

        assert q1.get_nowait() == {"type": "test_event"}
        assert q2.get_nowait() == {"type": "test_event"}

    @pytest.mark.asyncio
    async def test_broadcast_handles_full_queue(self, handler):
        q = asyncio.Queue(maxsize=1)
        q.put_nowait({"type": "old"})
        _stream_clients.append(q)

        # Should not raise when queue is full
        await handler._broadcast_update({"type": "new"})

        # Only old event remains
        event = q.get_nowait()
        assert event["type"] == "old"

    @pytest.mark.asyncio
    async def test_broadcast_empty_clients(self, handler):
        # No clients, should not raise
        await handler._broadcast_update({"type": "no_one_listening"})


# =============================================================================
# 15. Default Agents / Sample Queue Tests
# =============================================================================


class TestDefaults:
    """Test default agent and sample queue generation."""

    def test_default_agents_count(self, handler):
        agents = handler._get_default_agents()
        assert len(agents) == 3

    def test_default_agents_have_required_fields(self, handler):
        for agent in handler._get_default_agents():
            assert "id" in agent
            assert "name" in agent
            assert "type" in agent
            assert "model" in agent
            assert "status" in agent
            assert "capabilities" in agent
            assert "created_at" in agent

    def test_default_agents_statuses(self, handler):
        agents = handler._get_default_agents()
        statuses = [a["status"] for a in agents]
        assert statuses.count("active") == 2
        assert statuses.count("idle") == 1

    def test_sample_queue_count(self, handler):
        queue = handler._get_sample_queue()
        assert len(queue) == 2

    def test_sample_queue_has_required_fields(self, handler):
        for task in handler._get_sample_queue():
            assert "id" in task
            assert "type" in task
            assert "priority" in task
            assert "status" in task
            assert "created_at" in task

    def test_sample_queue_unique_ids(self, handler):
        tasks = handler._get_sample_queue()
        ids = [t["id"] for t in tasks]
        assert len(set(ids)) == 2


# =============================================================================
# 16. Calculation Helper Tests
# =============================================================================


class TestCalculationHelpers:
    """Test the internal metric calculation methods."""

    def test_avg_task_duration_empty(self, handler):
        assert handler._calculate_avg_task_duration([]) == 0.0

    def test_avg_task_duration_with_agents(self, handler):
        agents = [
            {"avg_response_time": 100},
            {"avg_response_time": 200},
        ]
        assert handler._calculate_avg_task_duration(agents) == 150.0

    def test_avg_task_duration_some_zero(self, handler):
        agents = [
            {"avg_response_time": 0},
            {"avg_response_time": 200},
        ]
        # Only agents with non-zero avg_response_time are included
        assert handler._calculate_avg_task_duration(agents) == 200.0

    def test_avg_task_duration_all_zero(self, handler):
        agents = [
            {"avg_response_time": 0},
            {"avg_response_time": 0},
        ]
        assert handler._calculate_avg_task_duration(agents) == 0.0

    def test_throughput_empty(self, handler):
        assert handler._calculate_throughput([]) == 0.0

    def test_throughput_with_agents(self, handler):
        agents = [
            {"tasks_completed": 60},
            {"tasks_completed": 120},
        ]
        assert handler._calculate_throughput(agents) == 3.0

    def test_error_rate_empty(self, handler):
        assert handler._calculate_error_rate([]) == 0.0

    def test_error_rate_with_agents(self, handler):
        agents = [
            {"error_rate": 0.1},
            {"error_rate": 0.3},
        ]
        assert handler._calculate_error_rate(agents) == 0.2

    def test_error_rate_all_zero(self, handler):
        agents = [
            {"error_rate": 0},
            {"error_rate": 0},
        ]
        assert handler._calculate_error_rate(agents) == 0.0


# =============================================================================
# 17. JSON Response / Error Response Tests
# =============================================================================


class TestResponseHelpers:
    """Test _json_response and _error_response formatting."""

    def test_json_response_format(self, handler):
        result = handler._json_response(200, {"key": "value"})
        assert result["status"] == 200
        assert result["headers"]["Content-Type"] == "application/json"
        body = json.loads(result["body"])
        assert body["key"] == "value"

    def test_error_response_format(self, handler):
        result = handler._error_response(404, "Not found")
        assert result["status"] == 404
        body = json.loads(result["body"])
        assert body["error"] == "Not found"

    def test_json_response_serializes_datetime(self, handler):
        """datetime objects are serialized via default=str."""
        now = datetime.now(timezone.utc)
        result = handler._json_response(200, {"ts": now})
        body = json.loads(result["body"])
        assert str(now) in body["ts"]

    def test_sse_response_format(self, handler):
        async def gen():
            yield "data: test\n\n"

        result = handler._sse_response(gen())
        assert result["status"] == 200
        assert result["headers"]["Content-Type"] == "text/event-stream"
        assert result["headers"]["Cache-Control"] == "no-cache"
        assert result["headers"]["Connection"] == "keep-alive"


# =============================================================================
# 18. Parse JSON Body Tests
# =============================================================================


class TestParseJsonBody:
    """Test _parse_json_body for different request types."""

    @pytest.mark.asyncio
    async def test_parse_json_body_with_json_method(self, handler):
        req = MockHTTPRequest(body={"foo": "bar"})
        result = await handler._parse_json_body(req)
        assert result == {"foo": "bar"}

    @pytest.mark.asyncio
    async def test_parse_json_body_with_body_method(self, handler):
        """Falls back to request.body() when .json not available."""

        class BodyOnlyRequest:
            async def body(self):
                return json.dumps({"baz": 42}).encode()

        req = BodyOnlyRequest()
        result = await handler._parse_json_body(req)
        assert result == {"baz": 42}

    @pytest.mark.asyncio
    async def test_parse_json_body_no_methods(self, handler):
        """Returns empty dict when neither .json nor .body available."""

        class EmptyRequest:
            pass

        req = EmptyRequest()
        result = await handler._parse_json_body(req)
        assert result == {}


# =============================================================================
# 19. Constructor Tests
# =============================================================================


class TestConstructor:
    """Test AgentDashboardHandler initialization."""

    def test_init_with_server_context(self):
        h = AgentDashboardHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_with_ctx(self):
        h = AgentDashboardHandler(ctx={"old": "param"})
        assert h.ctx == {"old": "param"}

    def test_init_with_both_prefers_server_context(self):
        h = AgentDashboardHandler(ctx={"old": 1}, server_context={"new": 2})
        assert h.ctx == {"new": 2}

    def test_init_with_none(self):
        h = AgentDashboardHandler()
        assert h.ctx == {}


# =============================================================================
# 20. RBAC Permission Check Tests
# =============================================================================


class TestRBACPermissions:
    """Test RBAC permission routing in handle_request."""

    @pytest.mark.asyncio
    async def test_get_requires_read_permission(self, handler, seeded_agents):
        """GET requests succeed with auto-patched auth (read permission)."""
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_post_requires_write_permission(self, handler, seeded_agents):
        """POST requests succeed with auto-patched auth (write permission)."""
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-a/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200


# =============================================================================
# 21. Agent ID Parsing Tests
# =============================================================================


class TestAgentIdParsing:
    """Test that agent_id is correctly parsed from paths."""

    @pytest.mark.asyncio
    async def test_agent_id_extracted_for_detail(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-b",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["id"] == "agent-b"

    @pytest.mark.asyncio
    async def test_agent_id_extracted_for_pause(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-b/pause",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_agent_id_extracted_for_resume(self, handler):
        _agents["agent-r"] = _make_agent("agent-r", status="paused")
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-r/resume",
            method="POST",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_agent_id_extracted_for_metrics(self, handler, seeded_agents):
        req = MockHTTPRequest(
            path="/api/v1/control-plane/agents/agent-c/metrics",
            method="GET",
        )
        with patch(f"{MODULE}._get_shared_state", return_value=None):
            result = await handler.handle_request(req)

        body = _body(result)
        assert body["agent_id"] == "agent-c"
