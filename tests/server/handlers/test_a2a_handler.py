"""Tests for A2A Protocol handler endpoints.

Validates the REST API endpoints for A2A (Agent-to-Agent) protocol including:
- Agent discovery
- Agent listing
- Task submission and status
- OpenAPI spec endpoint
"""

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.a2a import A2AHandler


@pytest.fixture
def a2a_handler():
    """Create an A2A handler with mocked dependencies."""
    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    handler = A2AHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    handler.command = "GET"
    return handler


def create_post_request(data: dict) -> MagicMock:
    """Create a mock HTTP handler with a JSON body for POST requests."""
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


class TestA2AHandlerRoutes:
    """Test A2A handler route configuration."""

    def test_routes_defined(self, a2a_handler):
        """Verify ROUTES are defined."""
        assert hasattr(a2a_handler, "ROUTES")
        assert len(a2a_handler.ROUTES) > 0

    def test_well_known_route_in_routes(self, a2a_handler):
        """Verify well-known discovery route exists."""
        assert "/.well-known/agent.json" in a2a_handler.ROUTES

    def test_agents_route_in_routes(self, a2a_handler):
        """Verify agents route exists."""
        assert "/api/v1/a2a/agents" in a2a_handler.ROUTES

    def test_tasks_route_in_routes(self, a2a_handler):
        """Verify tasks route exists."""
        assert "/api/v1/a2a/tasks" in a2a_handler.ROUTES


class TestA2AHandlerCanHandle:
    """Test can_handle method."""

    def test_can_handle_a2a_prefix(self, a2a_handler):
        """Handler should match /api/v1/a2a/ paths."""
        assert a2a_handler.can_handle("/api/v1/a2a/agents") is True
        assert a2a_handler.can_handle("/api/v1/a2a/tasks") is True
        assert a2a_handler.can_handle("/api/v1/a2a/.well-known/agent.json") is True

    def test_can_handle_well_known(self, a2a_handler):
        """Handler should match well-known discovery path."""
        assert a2a_handler.can_handle("/.well-known/agent.json") is True

    def test_cannot_handle_other_paths(self, a2a_handler):
        """Handler should not match unrelated paths."""
        assert a2a_handler.can_handle("/api/debates") is False
        assert a2a_handler.can_handle("/api/agents") is False


class TestA2AHandlerDiscovery:
    """Test agent discovery endpoint."""

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_discovery_with_agents(self, mock_get_server, a2a_handler, mock_http_handler):
        """Discovery returns primary agent card when agents exist."""
        mock_agent = MagicMock()
        mock_agent.to_dict.return_value = {
            "name": "test-agent",
            "version": "1.0.0",
            "capabilities": ["debate"],
        }
        mock_server = MagicMock()
        mock_server.list_agents.return_value = [mock_agent]
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_discovery()

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "test-agent"

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_discovery_without_agents(self, mock_get_server, a2a_handler, mock_http_handler):
        """Discovery returns default card when no agents registered."""
        mock_server = MagicMock()
        mock_server.list_agents.return_value = []
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_discovery()

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "aragora"
        assert "capabilities" in body


class TestA2AHandlerListAgents:
    """Test agent listing endpoint."""

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_list_agents_success(self, mock_get_server, a2a_handler, mock_http_handler):
        """List agents returns all registered agents."""
        mock_agent = MagicMock()
        mock_agent.to_dict.return_value = {"name": "test-agent"}
        mock_server = MagicMock()
        mock_server.list_agents.return_value = [mock_agent]
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_list_agents()

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert "agents" in body
        assert len(body["agents"]) == 1

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_list_agents_empty(self, mock_get_server, a2a_handler, mock_http_handler):
        """List agents returns empty list when no agents."""
        mock_server = MagicMock()
        mock_server.list_agents.return_value = []
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_list_agents()

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["agents"] == []


class TestA2AHandlerGetAgent:
    """Test get agent by name endpoint."""

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_agent_success(self, mock_get_server, a2a_handler, mock_http_handler):
        """Get agent returns agent card when found."""
        mock_agent = MagicMock()
        mock_agent.to_dict.return_value = {"name": "test-agent", "version": "1.0.0"}
        mock_server = MagicMock()
        mock_server.get_agent.return_value = mock_agent
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_get_agent("test-agent")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["name"] == "test-agent"

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_agent_not_found(self, mock_get_server, a2a_handler, mock_http_handler):
        """Get agent returns 404 when agent not found."""
        mock_server = MagicMock()
        mock_server.get_agent.return_value = None
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_get_agent("nonexistent")

        assert result is not None
        assert result.status_code == 404


class TestA2AHandlerTasks:
    """Test task-related endpoints."""

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_task_success(self, mock_get_server, a2a_handler, mock_http_handler):
        """Get task returns task status when found."""
        mock_task = MagicMock()
        mock_task.to_dict.return_value = {
            "id": "task-123",
            "status": "completed",
            "result": {"answer": "test"},
        }
        mock_server = MagicMock()
        mock_server.get_task_status.return_value = mock_task
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_get_task("task-123")

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["id"] == "task-123"
        assert body["status"] == "completed"

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_get_task_not_found(self, mock_get_server, a2a_handler, mock_http_handler):
        """Get task returns 404 when task not found."""
        mock_server = MagicMock()
        mock_server.get_task_status.return_value = None
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_get_task("nonexistent")

        assert result is not None
        assert result.status_code == 404

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_cancel_task_success(self, mock_get_server, a2a_handler, mock_http_handler):
        """Cancel task returns success when task exists."""
        import asyncio

        async def mock_cancel(task_id):
            return True

        mock_server = MagicMock()
        mock_server.cancel_task = mock_cancel
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_cancel_task("task-123")

        assert result is not None
        assert result.status_code == 204

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_cancel_task_not_found(self, mock_get_server, a2a_handler, mock_http_handler):
        """Cancel task returns 404 when task not found."""
        import asyncio

        async def mock_cancel(task_id):
            return False

        mock_server = MagicMock()
        mock_server.cancel_task = mock_cancel
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_cancel_task("nonexistent")

        assert result is not None
        assert result.status_code == 404


class TestA2AHandlerOpenAPI:
    """Test OpenAPI spec endpoint."""

    @patch("aragora.server.handlers.a2a.get_a2a_server")
    def test_openapi_spec(self, mock_get_server, a2a_handler, mock_http_handler):
        """OpenAPI endpoint returns specification."""
        mock_server = MagicMock()
        mock_server.get_openapi_spec.return_value = {
            "openapi": "3.1.0",
            "info": {"title": "A2A API"},
        }
        mock_get_server.return_value = mock_server

        result = a2a_handler._handle_openapi()

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["openapi"] == "3.1.0"
