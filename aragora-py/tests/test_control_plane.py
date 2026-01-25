"""Tests for the Control Plane API."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from aragora_client.control_plane import (
    AgentHealth,
    ControlPlaneStatus,
    RegisteredAgent,
    ResourceUtilization,
    Task,
)


class TestControlPlaneAPI:
    """Tests for ControlPlaneAPI methods."""

    @pytest.mark.asyncio
    async def test_register_agent(self, mock_client, mock_response):
        """Test registering an agent."""
        response_data = {
            "agent": {
                "agent_id": "my-agent",
                "capabilities": ["debate", "code"],
                "status": "ready",
                "model": "claude-3-opus",
                "provider": "anthropic",
                "is_available": True,
                "tasks_completed": 0,
                "tasks_failed": 0,
            }
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.register_agent(
            agent_id="my-agent",
            capabilities=["debate", "code"],
            model="claude-3-opus",
            provider="anthropic",
        )

        assert isinstance(result, RegisteredAgent)
        assert result.agent_id == "my-agent"
        assert "debate" in result.capabilities

    @pytest.mark.asyncio
    async def test_unregister_agent(self, mock_client, mock_response):
        """Test unregistering an agent."""
        response_data = {"success": True}
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.unregister_agent("my-agent")

        assert result is True

    @pytest.mark.asyncio
    async def test_list_agents(self, mock_client, mock_response):
        """Test listing agents."""
        response_data = {
            "agents": [
                {
                    "agent_id": "agent-1",
                    "capabilities": ["debate"],
                    "status": "ready",
                    "is_available": True,
                },
                {
                    "agent_id": "agent-2",
                    "capabilities": ["code"],
                    "status": "busy",
                    "is_available": False,
                },
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.list_agents()

        assert len(result) == 2
        assert isinstance(result[0], RegisteredAgent)
        assert result[0].agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_list_agents_with_capability_filter(self, mock_client, mock_response):
        """Test listing agents with capability filter."""
        response_data = {
            "agents": [
                {
                    "agent_id": "agent-1",
                    "capabilities": ["debate"],
                    "status": "ready",
                    "is_available": True,
                }
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.list_agents(
            capability="debate", only_available=True
        )

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_agent_health(self, mock_client, mock_response):
        """Test getting agent health."""
        response_data = {
            "agent_id": "my-agent",
            "status": "ready",
            "is_available": True,
            "last_heartbeat": time.time(),
            "heartbeat_age_seconds": 5.0,
            "tasks_completed": 100,
            "tasks_failed": 2,
            "avg_latency_ms": 150.5,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.get_agent_health("my-agent")

        assert isinstance(result, AgentHealth)
        assert result.agent_id == "my-agent"
        assert result.tasks_completed == 100

    @pytest.mark.asyncio
    async def test_submit_task(self, mock_client, mock_response):
        """Test submitting a task."""
        response_data = {"task_id": "task-123"}
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.submit_task(
            task_type="debate",
            payload={"question": "Should we use microservices?"},
            required_capabilities=["debate"],
            priority="high",
        )

        assert result == "task-123"

    @pytest.mark.asyncio
    async def test_get_task_status(self, mock_client, mock_response):
        """Test getting task status."""
        response_data = {
            "task_id": "task-123",
            "task_type": "debate",
            "status": "running",
            "priority": "high",
            "created_at": time.time() - 60,
            "assigned_at": time.time() - 30,
            "started_at": time.time() - 25,
            "assigned_agent": "agent-1",
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.get_task_status("task-123")

        assert isinstance(result, Task)
        assert result.task_id == "task-123"
        assert result.status == "running"

    @pytest.mark.asyncio
    async def test_cancel_task(self, mock_client, mock_response):
        """Test cancelling a task."""
        response_data = {"success": True}
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.cancel_task("task-123")

        assert result is True

    @pytest.mark.asyncio
    async def test_list_pending_tasks(self, mock_client, mock_response):
        """Test listing pending tasks."""
        response_data = {
            "tasks": [
                {
                    "task_id": "task-1",
                    "task_type": "debate",
                    "status": "pending",
                    "priority": "normal",
                    "created_at": time.time(),
                },
                {
                    "task_id": "task-2",
                    "task_type": "code_review",
                    "status": "pending",
                    "priority": "high",
                    "created_at": time.time(),
                },
            ]
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.list_pending_tasks()

        assert len(result) == 2
        assert isinstance(result[0], Task)

    @pytest.mark.asyncio
    async def test_get_status(self, mock_client, mock_response):
        """Test getting overall control plane status."""
        response_data = {
            "status": "healthy",
            "registry": {"total_agents": 5, "available_agents": 4},
            "scheduler": {"pending_tasks": 10, "running_tasks": 3},
            "health_monitor": {"last_check": "2026-01-01T00:00:00Z"},
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.get_status()

        assert isinstance(result, ControlPlaneStatus)
        assert result.status == "healthy"
        assert result.registry["total_agents"] == 5

    @pytest.mark.asyncio
    async def test_trigger_health_check_all(self, mock_client, mock_response):
        """Test triggering health check for all agents."""
        response_data = {
            "checked": 5,
            "healthy": 4,
            "unhealthy": 1,
            "results": {"agent-1": "healthy", "agent-2": "unhealthy"},
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.trigger_health_check()

        assert result["checked"] == 5

    @pytest.mark.asyncio
    async def test_trigger_health_check_specific_agent(
        self, mock_client, mock_response
    ):
        """Test triggering health check for specific agent."""
        response_data = {
            "agent_id": "my-agent",
            "status": "healthy",
            "latency_ms": 50,
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.trigger_health_check("my-agent")

        assert result["agent_id"] == "my-agent"
        assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_resource_utilization(self, mock_client, mock_response):
        """Test getting resource utilization."""
        response_data = {
            "queue_depths": {"debate": 5, "code_review": 2},
            "agents": {"total": 10, "available": 8, "busy": 2},
            "tasks_by_type": {"debate": 15, "code_review": 8},
            "tasks_by_priority": {"low": 3, "normal": 15, "high": 5},
        }
        mock_client._client.request = AsyncMock(
            return_value=mock_response(200, response_data)
        )

        result = await mock_client.control_plane.get_resource_utilization()

        assert isinstance(result, ResourceUtilization)
        assert result.queue_depths["debate"] == 5
        assert result.tasks_by_type["debate"] == 15
