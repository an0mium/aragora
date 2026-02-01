"""Tests for MCP control plane tools execution logic."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.mcp.tools_module.control_plane import (
    cancel_task_tool,
    get_agent_health_tool,
    get_control_plane_status_tool,
    get_resource_utilization_tool,
    get_task_status_tool,
    list_pending_tasks_tool,
    list_registered_agents_tool,
    register_agent_tool,
    submit_task_tool,
    trigger_health_check_tool,
    unregister_agent_tool,
)


@pytest.fixture(autouse=True)
def reset_coordinator():
    """Reset global coordinator between tests."""
    import aragora.mcp.tools_module.control_plane as cp_mod

    cp_mod._coordinator = None
    yield
    cp_mod._coordinator = None


# =============================================================================
# Agent Operations
# =============================================================================


class TestRegisterAgentTool:
    """Tests for register_agent_tool."""

    @pytest.mark.asyncio
    async def test_register_agent_empty_id(self):
        """Test register with empty agent_id."""
        result = await register_agent_tool(agent_id="")
        assert "error" in result
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_register_agent_success(self):
        """Test successful agent registration."""
        mock_coordinator = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.agent_id = "claude-test"
        mock_agent.capabilities = {"debate", "code"}
        mock_agent.status = MagicMock()
        mock_agent.status.value = "available"
        mock_agent.model = "claude-3-opus"
        mock_agent.provider = "anthropic"
        mock_agent.registered_at = 1700000000.0
        mock_coordinator.register_agent.return_value = mock_agent

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await register_agent_tool(
                agent_id="claude-test",
                capabilities="debate,code",
                model="claude-3-opus",
                provider="anthropic",
            )

        assert result["success"] is True
        assert result["agent"]["agent_id"] == "claude-test"
        assert "debate" in result["agent"]["capabilities"]

    @pytest.mark.asyncio
    async def test_register_agent_no_coordinator(self):
        """Test register when control plane unavailable."""
        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=None,
        ):
            result = await register_agent_tool(agent_id="test-agent")

        assert "error" in result
        assert "not available" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_register_agent_default_capabilities(self):
        """Test register with empty capabilities defaults to debate."""
        mock_coordinator = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.agent_id = "test"
        mock_agent.capabilities = {"debate"}
        mock_agent.status = MagicMock(value="available")
        mock_agent.model = "unknown"
        mock_agent.provider = "unknown"
        mock_agent.registered_at = 1700000000.0
        mock_coordinator.register_agent.return_value = mock_agent

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await register_agent_tool(agent_id="test", capabilities="")

        assert result["success"] is True


class TestUnregisterAgentTool:
    """Tests for unregister_agent_tool."""

    @pytest.mark.asyncio
    async def test_unregister_empty_id(self):
        """Test unregister with empty agent_id."""
        result = await unregister_agent_tool(agent_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unregister_success(self):
        """Test successful unregistration."""
        mock_coordinator = AsyncMock()
        mock_coordinator.unregister_agent.return_value = True

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await unregister_agent_tool(agent_id="claude-test")

        assert result["success"] is True
        assert result["agent_id"] == "claude-test"

    @pytest.mark.asyncio
    async def test_unregister_not_found(self):
        """Test unregister agent not found."""
        mock_coordinator = AsyncMock()
        mock_coordinator.unregister_agent.return_value = False

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await unregister_agent_tool(agent_id="nonexistent")

        assert result["success"] is False
        assert "not found" in result["message"].lower()


class TestListRegisteredAgentsTool:
    """Tests for list_registered_agents_tool."""

    @pytest.mark.asyncio
    async def test_list_agents_no_coordinator_returns_fallback(self):
        """Test fallback list when coordinator unavailable."""
        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=None,
        ):
            result = await list_registered_agents_tool()

        assert "agents" in result
        assert result["count"] == 3
        assert "note" in result
        assert "fallback" in result["note"].lower()

    @pytest.mark.asyncio
    async def test_list_agents_success(self):
        """Test successful agent listing."""
        mock_coordinator = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.agent_id = "claude"
        mock_agent.status = MagicMock(value="available")
        mock_agent.capabilities = {"debate"}
        mock_agent.model = "claude-3"
        mock_agent.provider = "anthropic"
        mock_agent.is_available.return_value = True
        mock_agent.tasks_completed = 10
        mock_agent.tasks_failed = 1
        mock_agent.avg_latency_ms = 150.5
        mock_agent.region_id = "us-east-1"
        mock_coordinator.list_agents.return_value = [mock_agent]

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await list_registered_agents_tool()

        assert result["count"] == 1
        assert result["agents"][0]["agent_id"] == "claude"
        assert result["agents"][0]["is_available"] is True

    @pytest.mark.asyncio
    async def test_list_agents_with_capability_filter(self):
        """Test agent listing with capability filter."""
        mock_coordinator = AsyncMock()
        mock_coordinator.list_agents.return_value = []

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await list_registered_agents_tool(capability="code")

        assert result["count"] == 0
        assert result["filter"]["capability"] == "code"


class TestGetAgentHealthTool:
    """Tests for get_agent_health_tool."""

    @pytest.mark.asyncio
    async def test_get_health_empty_id(self):
        """Test get health with empty agent_id."""
        result = await get_agent_health_tool(agent_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_health_success(self):
        """Test successful health check."""
        mock_coordinator = AsyncMock()
        mock_agent = MagicMock()
        mock_agent.status = MagicMock(value="available")
        mock_agent.last_heartbeat = 1700000000.0
        mock_agent.tasks_completed = 5
        mock_agent.tasks_failed = 0
        mock_agent.avg_latency_ms = 100.0
        mock_agent.current_task_id = None
        mock_coordinator.get_agent.return_value = mock_agent
        mock_coordinator.is_agent_available.return_value = True
        mock_coordinator.get_agent_health.return_value = None

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await get_agent_health_tool(agent_id="claude")

        assert result["agent_id"] == "claude"
        assert result["is_available"] is True
        assert result["tasks_completed"] == 5

    @pytest.mark.asyncio
    async def test_get_health_agent_not_found(self):
        """Test health check for non-existent agent."""
        mock_coordinator = AsyncMock()
        mock_coordinator.get_agent.return_value = None

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await get_agent_health_tool(agent_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()


# =============================================================================
# Task Operations
# =============================================================================


class TestSubmitTaskTool:
    """Tests for submit_task_tool."""

    @pytest.mark.asyncio
    async def test_submit_empty_task_type(self):
        """Test submit with empty task_type."""
        result = await submit_task_tool(task_type="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_submit_invalid_json_payload(self):
        """Test submit with invalid JSON payload."""
        mock_coordinator = AsyncMock()

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await submit_task_tool(task_type="debate", payload="not json")

        assert "error" in result
        assert "Invalid JSON" in result["error"]

    @pytest.mark.asyncio
    async def test_submit_non_dict_payload(self):
        """Test submit with non-dict JSON payload."""
        mock_coordinator = AsyncMock()

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await submit_task_tool(task_type="debate", payload='["a","b"]')

        assert "error" in result
        assert "JSON object" in result["error"]

    @pytest.mark.asyncio
    async def test_submit_success(self):
        """Test successful task submission."""
        mock_coordinator = AsyncMock()
        mock_coordinator.submit_task.return_value = "task-abc123"

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ), patch(
            "aragora.mcp.tools_module.control_plane.TaskPriority",
            create=True,
        ) as mock_priority:
            mock_priority.LOW = "low"
            mock_priority.NORMAL = "normal"
            mock_priority.HIGH = "high"
            mock_priority.URGENT = "urgent"

            result = await submit_task_tool(
                task_type="debate",
                payload='{"question": "test"}',
                priority="high",
            )

        assert result["success"] is True
        assert result["task_id"] == "task-abc123"
        assert result["priority"] == "high"


class TestGetTaskStatusTool:
    """Tests for get_task_status_tool."""

    @pytest.mark.asyncio
    async def test_get_status_empty_id(self):
        """Test get status with empty task_id."""
        result = await get_task_status_tool(task_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_status_not_found(self):
        """Test get status for non-existent task."""
        mock_coordinator = AsyncMock()
        mock_coordinator.get_task.return_value = None

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await get_task_status_tool(task_id="nonexistent")

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_status_success(self):
        """Test successful task status retrieval."""
        mock_coordinator = AsyncMock()
        mock_task = MagicMock()
        mock_task.id = "task-123"
        mock_task.task_type = "debate"
        mock_task.status = MagicMock(value="completed")
        mock_task.priority = MagicMock()
        mock_task.priority.name = "HIGH"
        mock_task.created_at = 1700000000.0
        mock_task.assigned_at = 1700000001.0
        mock_task.started_at = 1700000002.0
        mock_task.completed_at = 1700000010.0
        mock_task.assigned_agent = "claude"
        mock_task.retries = 0
        mock_task.max_retries = 3
        mock_task.timeout_seconds = 300.0
        mock_task.result = {"answer": "yes"}
        mock_task.error = None
        mock_task.is_timed_out.return_value = False
        mock_coordinator.get_task.return_value = mock_task

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await get_task_status_tool(task_id="task-123")

        assert result["task_id"] == "task-123"
        assert result["status"] == "completed"
        assert result["assigned_agent"] == "claude"
        assert result["is_timed_out"] is False


class TestCancelTaskTool:
    """Tests for cancel_task_tool."""

    @pytest.mark.asyncio
    async def test_cancel_empty_id(self):
        """Test cancel with empty task_id."""
        result = await cancel_task_tool(task_id="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_cancel_success(self):
        """Test successful task cancellation."""
        mock_coordinator = AsyncMock()
        mock_coordinator.cancel_task.return_value = True

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await cancel_task_tool(task_id="task-123")

        assert result["success"] is True
        assert "cancelled" in result["message"].lower()


class TestListPendingTasksTool:
    """Tests for list_pending_tasks_tool."""

    @pytest.mark.asyncio
    async def test_list_pending_no_coordinator(self):
        """Test list pending when coordinator unavailable."""
        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=None,
        ):
            result = await list_pending_tasks_tool()

        assert "error" in result


# =============================================================================
# Health & Status
# =============================================================================


class TestGetControlPlaneStatusTool:
    """Tests for get_control_plane_status_tool."""

    @pytest.mark.asyncio
    async def test_status_no_coordinator(self):
        """Test status when coordinator unavailable."""
        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=None,
        ):
            result = await get_control_plane_status_tool()

        assert result["status"] == "unavailable"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_status_success(self):
        """Test successful status retrieval."""
        mock_coordinator = AsyncMock()
        mock_coordinator.get_system_health.return_value = MagicMock(value="healthy")
        mock_coordinator.get_stats.return_value = {
            "registry": {"total_agents": 5},
            "scheduler": {"total_tasks": 100},
            "health": {},
            "config": {},
            "knowledge_mound": None,
        }

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await get_control_plane_status_tool()

        assert result["status"] == "healthy"
        assert "registry" in result


class TestTriggerHealthCheckTool:
    """Tests for trigger_health_check_tool."""

    @pytest.mark.asyncio
    async def test_health_check_specific_agent(self):
        """Test health check for specific agent."""
        mock_coordinator = AsyncMock()
        mock_coordinator.is_agent_available.return_value = True
        mock_coordinator.get_agent_health.return_value = None

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await trigger_health_check_tool(agent_id="claude")

        assert result["agent_id"] == "claude"
        assert result["is_available"] is True

    @pytest.mark.asyncio
    async def test_health_check_all_agents(self):
        """Test health check for all agents."""
        mock_coordinator = AsyncMock()
        mock_coordinator.get_system_health.return_value = MagicMock(value="healthy")
        mock_agent1 = MagicMock()
        mock_agent1.is_available.return_value = True
        mock_agent2 = MagicMock()
        mock_agent2.is_available.return_value = False
        mock_coordinator.list_agents.return_value = [mock_agent1, mock_agent2]

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await trigger_health_check_tool(agent_id="")

        assert result["system_health"] == "healthy"
        assert result["agents_checked"] == 2
        assert result["agents_available"] == 1
        assert result["agents_offline"] == 1


class TestGetResourceUtilizationTool:
    """Tests for get_resource_utilization_tool."""

    @pytest.mark.asyncio
    async def test_utilization_no_coordinator(self):
        """Test utilization when coordinator unavailable."""
        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=None,
        ):
            result = await get_resource_utilization_tool()

        assert "error" in result

    @pytest.mark.asyncio
    async def test_utilization_success(self):
        """Test successful utilization retrieval."""
        mock_coordinator = AsyncMock()
        mock_coordinator.get_stats.return_value = {
            "scheduler": {
                "by_status": {"pending": 5, "running": 2, "completed": 100, "failed": 3},
                "by_type": {"debate": 80, "analysis": 20},
                "by_priority": {"high": 10, "normal": 90},
            },
            "registry": {
                "total_agents": 5,
                "available_agents": 4,
                "by_status": {"available": 4, "busy": 1},
            },
        }

        with patch(
            "aragora.mcp.tools_module.control_plane._get_coordinator",
            return_value=mock_coordinator,
        ):
            result = await get_resource_utilization_tool()

        assert result["queue_depths"]["pending"] == 5
        assert result["queue_depths"]["running"] == 2
        assert result["agents"]["total"] == 5
        assert result["agents"]["available"] == 4
