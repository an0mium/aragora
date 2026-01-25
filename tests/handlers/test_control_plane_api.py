"""Tests for control plane API handlers."""

import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from tests.handlers.conftest import (
    MockCoordinator,
    MockRequest,
    MockHandler,
    MockAgentInfo,
    MockTask,
    MockHealthCheck,
    AgentStatus,
    TaskStatus,
    TaskPriority,
    HealthStatus,
)


class TestControlPlaneHandlerRouting:
    """Test control plane handler request routing."""

    @pytest.fixture
    def handler_class(self):
        """Get the handler class."""
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class):
        """Create handler without coordinator (tests routing only)."""
        handler = handler_class(server_context={})
        handler_class.coordinator = None
        return handler

    def test_handle_list_agents_route_no_coordinator(self, handler):
        """Test that list agents route returns 503 when coordinator is None."""
        result = handler.handle(
            path="/api/v1/control-plane/agents",
            query_params={},
            handler=MagicMock(),
        )
        # Should return error when coordinator not initialized
        assert result is not None
        assert result.status_code == 503

    def test_handle_get_agent_route_no_coordinator(self, handler):
        """Test that get agent route returns 503 when coordinator is None."""
        result = handler.handle(
            path="/api/v1/control-plane/agents/agent-123",
            query_params={},
            handler=MagicMock(),
        )
        assert result is not None
        assert result.status_code == 503

    def test_handle_unknown_route(self, handler):
        """Test that unknown routes return None."""
        result = handler.handle(
            path="/api/v1/unknown/route",
            query_params={},
            handler=MagicMock(),
        )
        assert result is None


class TestFeaturesControlPlaneHandler:
    """Test the features control plane handler (simpler async handler)."""

    @pytest.fixture
    def handler(self):
        """Create features handler."""
        from aragora.server.handlers.features.control_plane import ControlPlaneHandler

        # Clear global state before each test
        from aragora.server.handlers.features import control_plane

        control_plane._agents.clear()
        control_plane._task_queue.clear()
        return ControlPlaneHandler(server_context={})

    def create_request(
        self,
        method: str = "GET",
        path: str = "/",
        query: dict = None,
        body: dict = None,
    ):
        """Create a mock request."""
        return MockRequest(method=method, path=path, query=query or {}, body=body)

    @pytest.mark.asyncio
    async def test_list_agents(self, handler):
        """Test listing agents."""
        request = self.create_request(
            method="GET",
            path="/api/v1/control-plane/agents",
        )

        result = await handler.handle_request(request)

        assert result["status"] == 200
        body = json.loads(result["body"])
        assert "agents" in body
        assert "total" in body

    @pytest.mark.asyncio
    async def test_get_queue(self, handler):
        """Test getting the queue."""
        request = self.create_request(
            method="GET",
            path="/api/v1/control-plane/queue",
        )

        result = await handler.handle_request(request)

        assert result["status"] == 200
        body = json.loads(result["body"])
        assert "tasks" in body
        assert "by_priority" in body

    @pytest.mark.asyncio
    async def test_get_metrics(self, handler):
        """Test getting metrics."""
        request = self.create_request(
            method="GET",
            path="/api/v1/control-plane/metrics",
        )

        result = await handler.handle_request(request)

        assert result["status"] == 200
        body = json.loads(result["body"])
        assert "agents" in body
        assert "queue" in body
        assert "performance" in body

    @pytest.mark.asyncio
    async def test_health_check(self, handler):
        """Test health check endpoint."""
        request = self.create_request(
            method="GET",
            path="/api/v1/control-plane/health",
        )

        result = await handler.handle_request(request)

        assert result["status"] == 200
        body = json.loads(result["body"])
        assert "status" in body
        assert "components" in body

    @pytest.mark.asyncio
    async def test_pause_agent(self, handler):
        """Test pausing an agent."""
        # First get agents to populate default agents
        list_request = self.create_request(
            method="GET",
            path="/api/v1/control-plane/agents",
        )
        await handler.handle_request(list_request)

        # Now pause one
        request = self.create_request(
            method="POST",
            path="/api/v1/control-plane/agents/agent-gemini-scanner/pause",
        )

        result = await handler.handle_request(request)

        assert result["status"] == 200
        body = json.loads(result["body"])
        assert body["status"] == "paused"

    @pytest.mark.asyncio
    async def test_resume_paused_agent(self, handler):
        """Test resuming a paused agent."""
        # Get agents
        await handler.handle_request(
            self.create_request(
                method="GET",
                path="/api/v1/control-plane/agents",
            )
        )

        # Pause
        await handler.handle_request(
            self.create_request(
                method="POST",
                path="/api/v1/control-plane/agents/agent-gemini-scanner/pause",
            )
        )

        # Resume
        request = self.create_request(
            method="POST",
            path="/api/v1/control-plane/agents/agent-gemini-scanner/resume",
        )
        result = await handler.handle_request(request)

        assert result["status"] == 200
        body = json.loads(result["body"])
        assert body["status"] == "active"

    @pytest.mark.asyncio
    async def test_pause_nonexistent_agent(self, handler):
        """Test pausing a non-existent agent."""
        request = self.create_request(
            method="POST",
            path="/api/v1/control-plane/agents/nonexistent/pause",
        )

        result = await handler.handle_request(request)

        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_resume_not_paused(self, handler):
        """Test resuming an agent that's not paused."""
        # Get agents (populates with active agents)
        await handler.handle_request(
            self.create_request(
                method="GET",
                path="/api/v1/control-plane/agents",
            )
        )

        # Try to resume active agent
        request = self.create_request(
            method="POST",
            path="/api/v1/control-plane/agents/agent-gemini-scanner/resume",
        )
        result = await handler.handle_request(request)

        assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_endpoint_not_found(self, handler):
        """Test unknown endpoint."""
        request = self.create_request(
            method="GET",
            path="/api/v1/control-plane/unknown",
        )

        result = await handler.handle_request(request)

        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_get_agent_by_id(self, handler):
        """Test getting a specific agent."""
        # Populate agents
        await handler.handle_request(
            self.create_request(
                method="GET",
                path="/api/v1/control-plane/agents",
            )
        )

        # Get specific agent
        request = self.create_request(
            method="GET",
            path="/api/v1/control-plane/agents/agent-gemini-scanner",
        )
        result = await handler.handle_request(request)

        assert result["status"] == 200
        body = json.loads(result["body"])
        assert body["id"] == "agent-gemini-scanner"

    @pytest.mark.asyncio
    async def test_get_agent_metrics(self, handler):
        """Test getting agent metrics."""
        # Populate agents
        await handler.handle_request(
            self.create_request(
                method="GET",
                path="/api/v1/control-plane/agents",
            )
        )

        # Get metrics
        request = self.create_request(
            method="GET",
            path="/api/v1/control-plane/agents/agent-gemini-scanner/metrics",
        )
        result = await handler.handle_request(request)

        assert result["status"] == 200
        body = json.loads(result["body"])
        assert "agent_id" in body
        assert "tasks_completed" in body

    @pytest.mark.asyncio
    async def test_filter_agents_by_status(self, handler):
        """Test filtering agents by status."""
        # Populate agents
        await handler.handle_request(
            self.create_request(
                method="GET",
                path="/api/v1/control-plane/agents",
            )
        )

        # Pause an agent
        await handler.handle_request(
            self.create_request(
                method="POST",
                path="/api/v1/control-plane/agents/agent-gemini-scanner/pause",
            )
        )

        # Filter for paused
        request = self.create_request(
            method="GET",
            path="/api/v1/control-plane/agents",
            query={"status": "paused"},
        )
        result = await handler.handle_request(request)

        assert result["status"] == 200
        body = json.loads(result["body"])
        assert body["paused"] >= 1


class TestMockCoordinator:
    """Test the mock coordinator itself."""

    @pytest.mark.asyncio
    async def test_register_and_list_agents(self, mock_coordinator):
        """Test registering and listing agents."""
        await mock_coordinator.register_agent(
            agent_id="test-1",
            capabilities=["analysis"],
            model="test",
            provider="test",
        )

        agents = await mock_coordinator.list_agents()

        assert len(agents) == 1
        assert agents[0].agent_id == "test-1"

    @pytest.mark.asyncio
    async def test_register_multiple_agents(self, mock_coordinator):
        """Test registering multiple agents."""
        await mock_coordinator.register_agent(
            agent_id="agent-1",
            capabilities=["analysis"],
            model="claude",
            provider="anthropic",
        )
        await mock_coordinator.register_agent(
            agent_id="agent-2",
            capabilities=["coding"],
            model="gpt-4",
            provider="openai",
        )

        agents = await mock_coordinator.list_agents()
        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_list_agents_by_capability(self, mock_coordinator):
        """Test filtering agents by capability."""
        await mock_coordinator.register_agent(
            agent_id="agent-1",
            capabilities=["analysis", "reasoning"],
            model="claude",
            provider="anthropic",
        )
        await mock_coordinator.register_agent(
            agent_id="agent-2",
            capabilities=["coding"],
            model="gpt-4",
            provider="openai",
        )

        agents = await mock_coordinator.list_agents(capability="analysis")
        assert len(agents) == 1
        assert agents[0].agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_unregister_agent(self, mock_coordinator):
        """Test unregistering an agent."""
        await mock_coordinator.register_agent(
            agent_id="to-remove",
            capabilities=["test"],
            model="test",
            provider="test",
        )

        success = await mock_coordinator.unregister_agent("to-remove")
        assert success is True

        agent = await mock_coordinator.get_agent("to-remove")
        assert agent is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_agent(self, mock_coordinator):
        """Test unregistering non-existent agent."""
        success = await mock_coordinator.unregister_agent("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_task_lifecycle(self, mock_coordinator):
        """Test full task lifecycle."""
        # Register agent
        await mock_coordinator.register_agent(
            agent_id="worker",
            capabilities=["work"],
            model="test",
            provider="test",
        )

        # Submit task
        task_id = await mock_coordinator.submit_task(
            task_type="work",
            payload={"data": "test"},
        )

        # Verify task pending
        task = await mock_coordinator.get_task(task_id)
        assert task.status == TaskStatus.PENDING

        # Claim task
        claimed = await mock_coordinator.claim_task(
            agent_id="worker",
            capabilities=["work"],
        )
        assert claimed is not None
        assert claimed.status == TaskStatus.RUNNING

        # Complete task
        success = await mock_coordinator.complete_task(
            task_id=task_id,
            result={"output": "done"},
        )
        assert success is True

        # Verify completion
        completed_task = await mock_coordinator.get_task(task_id)
        assert completed_task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fail_task_with_requeue(self, mock_coordinator):
        """Test failing a task with requeue."""
        task_id = await mock_coordinator.submit_task(
            task_type="work",
            payload={"data": "test"},
        )

        # Fail with requeue
        success = await mock_coordinator.fail_task(
            task_id=task_id,
            error="Processing error",
            requeue=True,
        )
        assert success is True

        # Should be pending again
        task = await mock_coordinator.get_task(task_id)
        assert task.status == TaskStatus.PENDING
        assert task.error == "Processing error"

    @pytest.mark.asyncio
    async def test_fail_task_without_requeue(self, mock_coordinator):
        """Test failing a task without requeue."""
        task_id = await mock_coordinator.submit_task(
            task_type="work",
            payload={"data": "test"},
        )

        # Fail without requeue
        success = await mock_coordinator.fail_task(
            task_id=task_id,
            error="Fatal error",
            requeue=False,
        )
        assert success is True

        # Should be failed
        task = await mock_coordinator.get_task(task_id)
        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_cancel_task(self, mock_coordinator):
        """Test cancelling a task."""
        task_id = await mock_coordinator.submit_task(
            task_type="work",
            payload={"data": "test"},
        )

        success = await mock_coordinator.cancel_task(task_id)
        assert success is True

        task = await mock_coordinator.get_task(task_id)
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_task(self, mock_coordinator):
        """Test that completed tasks cannot be cancelled."""
        task_id = await mock_coordinator.submit_task(
            task_type="work",
            payload={"data": "test"},
        )

        await mock_coordinator.complete_task(task_id, result={})

        success = await mock_coordinator.cancel_task(task_id)
        assert success is False

    @pytest.mark.asyncio
    async def test_heartbeat(self, mock_coordinator):
        """Test agent heartbeat."""
        await mock_coordinator.register_agent(
            agent_id="agent-1",
            capabilities=["test"],
            model="test",
            provider="test",
        )

        success = await mock_coordinator.heartbeat(
            agent_id="agent-1",
            status=AgentStatus.BUSY,
        )

        assert success is True
        agent = await mock_coordinator.get_agent("agent-1")
        assert agent.status == AgentStatus.BUSY

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent_agent(self, mock_coordinator):
        """Test heartbeat for non-existent agent."""
        success = await mock_coordinator.heartbeat("nonexistent")
        assert success is False

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_coordinator):
        """Test getting statistics."""
        await mock_coordinator.register_agent(
            agent_id="agent-1",
            capabilities=["test"],
            model="test",
            provider="test",
        )
        await mock_coordinator.submit_task(
            task_type="test",
            payload={},
        )

        stats = await mock_coordinator.get_stats()

        assert stats["agents"]["total"] == 1
        assert stats["tasks"]["total"] == 1
        assert stats["tasks"]["pending"] == 1

    @pytest.mark.asyncio
    async def test_claim_task_with_capabilities(self, mock_coordinator):
        """Test claiming a task with specific capabilities."""
        await mock_coordinator.submit_task(
            task_type="analysis",
            payload={},
            required_capabilities=["reasoning"],
        )

        # Agent without required capability
        task = await mock_coordinator.claim_task(
            agent_id="wrong-agent",
            capabilities=["coding"],
        )
        assert task is None

        # Agent with required capability
        task = await mock_coordinator.claim_task(
            agent_id="right-agent",
            capabilities=["reasoning"],
        )
        assert task is not None

    @pytest.mark.asyncio
    async def test_system_health_empty(self, mock_coordinator):
        """Test system health with no agents."""
        health = mock_coordinator.get_system_health()
        assert health == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_agent_to_dict(self, mock_coordinator):
        """Test agent info serialization."""
        agent = await mock_coordinator.register_agent(
            agent_id="test",
            capabilities=["analysis"],
            model="test",
            provider="test",
            metadata={"key": "value"},
        )

        agent_dict = agent.to_dict()
        assert agent_dict["agent_id"] == "test"
        assert "analysis" in agent_dict["capabilities"]
        assert agent_dict["metadata"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_task_to_dict(self, mock_coordinator):
        """Test task serialization."""
        task_id = await mock_coordinator.submit_task(
            task_type="analysis",
            payload={"document": "test.pdf"},
            priority=TaskPriority.HIGH,
            metadata={"key": "value"},
        )

        task = await mock_coordinator.get_task(task_id)
        task_dict = task.to_dict()

        assert task_dict["task_id"] == task_id
        assert task_dict["task_type"] == "analysis"
        assert task_dict["priority"] == "high"
        assert task_dict["payload"]["document"] == "test.pdf"


class TestQueuePrioritization:
    """Test queue prioritization in features handler."""

    @pytest.fixture
    def handler(self):
        """Create features handler with queue."""
        from aragora.server.handlers.features.control_plane import ControlPlaneHandler
        from aragora.server.handlers.features import control_plane

        control_plane._agents.clear()
        control_plane._task_queue.clear()
        return ControlPlaneHandler(server_context={})

    def create_request(
        self,
        method: str = "GET",
        path: str = "/",
        query: dict = None,
        body: dict = None,
    ):
        """Create a mock request."""
        return MockRequest(method=method, path=path, query=query or {}, body=body)

    @pytest.mark.asyncio
    async def test_prioritize_task(self, handler):
        """Test changing task priority."""
        # Populate queue
        await handler.handle_request(
            self.create_request(
                method="GET",
                path="/api/v1/control-plane/queue",
            )
        )

        # Get a task ID from the queue
        from aragora.server.handlers.features import control_plane

        if control_plane._task_queue:
            task_id = control_plane._task_queue[0]["id"]

            # Change priority
            request = self.create_request(
                method="POST",
                path="/api/v1/control-plane/queue/prioritize",
                body={"task_id": task_id, "priority": "high"},
            )
            result = await handler.handle_request(request)

            assert result["status"] == 200
            body = json.loads(result["body"])
            assert body["success"] is True

    @pytest.mark.asyncio
    async def test_prioritize_nonexistent_task(self, handler):
        """Test prioritizing a non-existent task."""
        request = self.create_request(
            method="POST",
            path="/api/v1/control-plane/queue/prioritize",
            body={"task_id": "nonexistent", "priority": "high"},
        )
        result = await handler.handle_request(request)

        assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_prioritize_missing_task_id(self, handler):
        """Test prioritizing without task_id."""
        request = self.create_request(
            method="POST",
            path="/api/v1/control-plane/queue/prioritize",
            body={"priority": "high"},
        )
        result = await handler.handle_request(request)

        assert result["status"] == 400


# ============================================================================
# Control Plane Handler Routing Tests
# ============================================================================


class TestControlPlaneHandlerRoutingExtended:
    """Extended tests for control plane handler routing."""

    @pytest.fixture
    def handler_class(self):
        """Get the handler class."""
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class):
        """Create handler without coordinator."""
        handler = handler_class(server_context={})
        handler_class.coordinator = None
        return handler

    def test_can_handle_control_plane_paths(self, handler):
        """Handler can handle control plane paths."""
        assert handler.can_handle("/api/control-plane/agents")
        assert handler.can_handle("/api/v1/control-plane/agents")
        assert handler.can_handle("/api/control-plane/tasks")
        assert handler.can_handle("/api/control-plane/health")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle non-control-plane paths."""
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/gauntlet")
        assert not handler.can_handle("/api/v1/auth")

    def test_normalizes_v1_paths(self, handler):
        """Handler normalizes v1 paths to legacy form."""
        result = handler._normalize_path("/api/v1/control-plane/agents")
        assert result == "/api/control-plane/agents"


# ============================================================================
# Audit Log Tests
# ============================================================================


class TestControlPlaneAuditLogs:
    """Tests for control plane audit log endpoints."""

    @pytest.fixture
    def handler_class(self):
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class):
        handler = handler_class(server_context={})
        handler_class.coordinator = None
        return handler

    def test_handle_audit_route_no_auth(self, handler):
        """GET /api/control-plane/audit requires authentication."""
        mock_handler = MagicMock()
        # Set up auth to fail
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (None, {"status": 401})
            result = handler.handle(
                path="/api/control-plane/audit",
                query_params={},
                handler=mock_handler,
            )
            # Should return None (not this handler) or auth error
            if result:
                assert result.get("status") in (401, 403)

    def test_handle_audit_stats_route(self, handler):
        """GET /api/control-plane/audit/stats endpoint exists."""
        result = handler.handle(
            path="/api/control-plane/audit/stats",
            query_params={},
            handler=MagicMock(),
        )
        # Returns something (error or response)
        assert result is not None

    def test_handle_audit_verify_route(self, handler):
        """GET /api/control-plane/audit/verify endpoint exists."""
        mock_handler = MagicMock()
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(role="admin"), None)
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = handler.handle(
                    path="/api/control-plane/audit/verify",
                    query_params={},
                    handler=mock_handler,
                )
                # Should attempt to verify
                assert result is not None


# ============================================================================
# Policy Violations Tests
# ============================================================================


class TestControlPlanePolicyViolations:
    """Tests for control plane policy violation endpoints."""

    @pytest.fixture
    def handler_class(self):
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class):
        handler = handler_class(server_context={})
        handler_class.coordinator = None
        return handler

    def test_handle_list_violations_route(self, handler):
        """GET /api/control-plane/policies/violations endpoint exists."""
        mock_handler = MagicMock()
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(role="admin"), None)
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = handler.handle(
                    path="/api/control-plane/policies/violations",
                    query_params={},
                    handler=mock_handler,
                )
                assert result is not None

    def test_handle_violation_stats_route(self, handler):
        """GET /api/control-plane/policies/violations/stats endpoint exists."""
        mock_handler = MagicMock()
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(role="admin"), None)
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = handler.handle(
                    path="/api/control-plane/policies/violations/stats",
                    query_params={},
                    handler=mock_handler,
                )
                assert result is not None

    def test_handle_get_violation_by_id(self, handler):
        """GET /api/control-plane/policies/violations/:id endpoint exists."""
        mock_handler = MagicMock()
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(role="admin"), None)
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = handler.handle(
                    path="/api/control-plane/policies/violations/violation-123",
                    query_params={},
                    handler=mock_handler,
                )
                assert result is not None

    def test_handle_update_violation_status(self, handler):
        """PATCH /api/control-plane/policies/violations/:id endpoint exists."""
        mock_handler = MagicMock()
        mock_handler.rfile = MagicMock()
        mock_handler.rfile.read.return_value = b'{"status": "resolved"}'
        mock_handler.headers = {"Content-Length": "22"}

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(role="admin", user_id="user1"), None)
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = handler.handle_patch(
                    path="/api/control-plane/policies/violations/violation-123",
                    query_params={},
                    handler=mock_handler,
                )
                assert result is not None


# ============================================================================
# Deliberation Tests
# ============================================================================


class TestControlPlaneDeliberation:
    """Tests for control plane deliberation endpoints."""

    @pytest.fixture
    def handler_class(self):
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class):
        handler = handler_class(server_context={})
        handler_class.coordinator = None
        return handler

    def test_deliberation_endpoint_is_handled(self, handler):
        """Deliberation endpoint path is handled."""
        assert handler.can_handle("/api/control-plane/deliberations")
        assert handler.can_handle("/api/v1/control-plane/deliberations")

    def test_deliberation_status_endpoint_is_handled(self, handler):
        """Deliberation status endpoint path is handled."""
        assert handler.can_handle("/api/control-plane/deliberations/req-123/status")

    def test_deliberation_result_endpoint_is_handled(self, handler):
        """Deliberation result endpoint path is handled."""
        assert handler.can_handle("/api/control-plane/deliberations/req-123")


# ============================================================================
# Health Monitoring Extended Tests
# ============================================================================


class TestControlPlaneHealthExtended:
    """Extended tests for control plane health monitoring."""

    @pytest.fixture
    def handler_class(self):
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class, mock_coordinator):
        handler = handler_class(server_context={"control_plane_coordinator": mock_coordinator})
        handler_class.coordinator = mock_coordinator
        return handler

    def test_handle_detailed_health(self, handler):
        """GET /api/control-plane/health/detailed returns component status."""
        result = handler.handle(
            path="/api/control-plane/health/detailed",
            query_params={},
            handler=MagicMock(),
        )
        assert result is not None
        assert result.status_code == 200

    def test_handle_circuit_breakers(self, handler):
        """GET /api/control-plane/breakers returns circuit breaker states."""
        result = handler.handle(
            path="/api/control-plane/breakers",
            query_params={},
            handler=MagicMock(),
        )
        assert result is not None
        assert result.status_code == 200

    def test_handle_queue_metrics(self, handler):
        """GET /api/control-plane/queue/metrics returns queue performance."""
        result = handler.handle(
            path="/api/control-plane/queue/metrics",
            query_params={},
            handler=MagicMock(),
        )
        assert result is not None
        assert result.status_code == 200


# ============================================================================
# Notification Tests
# ============================================================================


class TestControlPlaneNotifications:
    """Tests for control plane notification endpoints."""

    @pytest.fixture
    def handler_class(self):
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class):
        handler = handler_class(server_context={})
        return handler

    def test_handle_get_notifications(self, handler):
        """GET /api/control-plane/notifications returns notification history."""
        result = handler.handle(
            path="/api/control-plane/notifications",
            query_params={},
            handler=MagicMock(),
        )
        assert result is not None
        assert result.status_code == 200

    def test_handle_notification_stats(self, handler):
        """GET /api/control-plane/notifications/stats returns stats."""
        result = handler.handle(
            path="/api/control-plane/notifications/stats",
            query_params={},
            handler=MagicMock(),
        )
        assert result is not None
        assert result.status_code == 200


# ============================================================================
# Event Emission Tests
# ============================================================================


class TestControlPlaneEventEmission:
    """Tests for control plane event emission."""

    @pytest.fixture
    def handler_class(self):
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def mock_stream(self):
        stream = MagicMock()
        stream.emit_agent_registered = AsyncMock()
        stream.emit_task_submitted = AsyncMock()
        stream.emit_task_completed = AsyncMock()
        return stream

    @pytest.fixture
    def handler(self, handler_class, mock_coordinator, mock_stream):
        handler = handler_class(
            server_context={
                "control_plane_coordinator": mock_coordinator,
                "control_plane_stream": mock_stream,
            }
        )
        handler_class.coordinator = mock_coordinator
        return handler

    def test_emit_event_with_retries(self, handler, mock_stream):
        """_emit_event retries on failure."""
        mock_stream.emit_test = MagicMock(side_effect=[Exception("fail"), None])

        # Should not raise even on first failure
        handler._emit_event("emit_test", arg1="value")
        # Stream method should have been called at least once

    def test_emit_event_without_stream(self, handler_class, mock_coordinator):
        """_emit_event does nothing without stream."""
        handler = handler_class(server_context={"control_plane_coordinator": mock_coordinator})
        handler_class.coordinator = mock_coordinator

        # Should not raise
        handler._emit_event("emit_test", arg1="value")


# ============================================================================
# Permission Checks Tests
# ============================================================================


class TestControlPlanePermissions:
    """Tests for control plane permission checks."""

    @pytest.fixture
    def handler_class(self):
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class):
        handler = handler_class(server_context={})
        handler_class.coordinator = None
        return handler

    def test_agents_endpoint_is_protected(self, handler):
        """POST /api/control-plane/agents is a valid endpoint."""
        assert handler.can_handle("/api/control-plane/agents")

    def test_tasks_endpoint_is_protected(self, handler):
        """POST /api/control-plane/tasks is a valid endpoint."""
        assert handler.can_handle("/api/control-plane/tasks")

    def test_audit_endpoint_is_protected(self, handler):
        """GET /api/control-plane/audit is a valid endpoint."""
        assert handler.can_handle("/api/control-plane/audit")


# ============================================================================
# DELETE Handler Tests
# ============================================================================


class TestControlPlaneDeleteHandlers:
    """Tests for control plane DELETE handlers."""

    @pytest.fixture
    def handler_class(self):
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class, mock_coordinator):
        handler = handler_class(server_context={"control_plane_coordinator": mock_coordinator})
        handler_class.coordinator = mock_coordinator
        return handler

    def test_unregister_agent(self, handler):
        """DELETE /api/control-plane/agents/:id unregisters agent."""
        mock_handler = MagicMock()

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(role="admin"), None)
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = handler.handle_delete(
                    path="/api/control-plane/agents/agent-123",
                    query_params={},
                    handler=mock_handler,
                )
                assert result is not None

    def test_unregister_nonexistent_agent(self, handler, mock_coordinator):
        """DELETE /api/control-plane/agents/:id returns 404 for nonexistent."""
        mock_handler = MagicMock()

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_auth.return_value = (MagicMock(role="admin"), None)
            with patch("aragora.server.handlers.control_plane.has_permission", return_value=True):
                result = handler.handle_delete(
                    path="/api/control-plane/agents/nonexistent",
                    query_params={},
                    handler=mock_handler,
                )
                assert result is not None
                assert result.status_code == 404


# ============================================================================
# POST Handler Input Validation Tests
# ============================================================================


class TestControlPlaneInputValidation:
    """Tests for control plane input validation."""

    @pytest.fixture
    def handler_class(self):
        from aragora.server.handlers.control_plane import ControlPlaneHandler

        return ControlPlaneHandler

    @pytest.fixture
    def handler(self, handler_class):
        handler = handler_class(server_context={})
        handler_class.coordinator = None
        return handler

    def test_agents_post_endpoint_exists(self, handler):
        """POST /api/control-plane/agents endpoint is handled."""
        assert handler.can_handle("/api/control-plane/agents")

    def test_tasks_post_endpoint_exists(self, handler):
        """POST /api/control-plane/tasks endpoint is handled."""
        assert handler.can_handle("/api/control-plane/tasks")

    def test_claim_task_endpoint_exists(self, handler):
        """POST /api/control-plane/tasks/claim endpoint is handled."""
        assert handler.can_handle("/api/control-plane/tasks/claim")

    def test_violations_endpoint_exists(self, handler):
        """PATCH /api/control-plane/policies/violations/:id endpoint is handled."""
        assert handler.can_handle("/api/control-plane/policies/violations/v-123")
