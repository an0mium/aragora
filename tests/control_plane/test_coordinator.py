"""
Tests for the ControlPlaneCoordinator.

Tests cover:
- Agent registration and lifecycle
- Task submission and execution
- Health monitoring integration
- Statistics and metrics
"""

import asyncio
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.control_plane.coordinator import (
    ControlPlaneConfig,
    ControlPlaneCoordinator,
    create_control_plane,
)
from aragora.control_plane.health import HealthCheck, HealthMonitor, HealthStatus
from aragora.control_plane.registry import AgentCapability, AgentInfo, AgentRegistry, AgentStatus
from aragora.control_plane.scheduler import Task, TaskPriority, TaskScheduler, TaskStatus


# ============================================================================
# Configuration Tests
# ============================================================================


class TestControlPlaneConfig:
    """Tests for ControlPlaneConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ControlPlaneConfig()

        assert config.redis_url == "redis://localhost:6379"
        assert config.key_prefix == "aragora:cp:"
        assert config.heartbeat_timeout == 30.0
        assert config.task_timeout == 300.0
        assert config.max_task_retries == 3

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("REDIS_URL", "redis://custom:6380")
        monkeypatch.setenv("CONTROL_PLANE_PREFIX", "custom:cp:")
        monkeypatch.setenv("HEARTBEAT_TIMEOUT", "60")
        monkeypatch.setenv("TASK_TIMEOUT", "600")

        config = ControlPlaneConfig.from_env()

        assert config.redis_url == "redis://custom:6380"
        assert config.key_prefix == "custom:cp:"
        assert config.heartbeat_timeout == 60.0
        assert config.task_timeout == 600.0


# ============================================================================
# Agent Operations Tests
# ============================================================================


class TestAgentOperations:
    """Tests for agent registration and management."""

    @pytest.mark.asyncio
    async def test_register_agent(self, coordinator: ControlPlaneCoordinator, sample_agent: Dict[str, Any]):
        """Test registering an agent."""
        agent = await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
            model=sample_agent["model"],
            provider=sample_agent["provider"],
            metadata=sample_agent["metadata"],
        )

        assert agent.agent_id == sample_agent["agent_id"]
        assert "debate" in agent.capabilities
        assert "code" in agent.capabilities
        assert agent.model == sample_agent["model"]
        assert agent.status == AgentStatus.READY

    @pytest.mark.asyncio
    async def test_register_agent_with_enum_capabilities(self, coordinator: ControlPlaneCoordinator):
        """Test registering with AgentCapability enum values."""
        agent = await coordinator.register_agent(
            agent_id="enum-agent",
            capabilities=[AgentCapability.DEBATE, AgentCapability.CODE],
            model="test-model",
        )

        assert "debate" in agent.capabilities
        assert "code" in agent.capabilities

    @pytest.mark.asyncio
    async def test_unregister_agent(self, coordinator: ControlPlaneCoordinator, sample_agent: Dict[str, Any]):
        """Test unregistering an agent."""
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        result = await coordinator.unregister_agent(sample_agent["agent_id"])
        assert result is True

        agent = await coordinator.get_agent(sample_agent["agent_id"])
        assert agent is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_agent(self, coordinator: ControlPlaneCoordinator):
        """Test unregistering an agent that doesn't exist."""
        result = await coordinator.unregister_agent("nonexistent-agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat(self, coordinator: ControlPlaneCoordinator, sample_agent: Dict[str, Any]):
        """Test agent heartbeat."""
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        result = await coordinator.heartbeat(sample_agent["agent_id"])
        assert result is True

        agent = await coordinator.get_agent(sample_agent["agent_id"])
        assert agent is not None
        assert agent.last_heartbeat > 0

    @pytest.mark.asyncio
    async def test_heartbeat_with_status_update(self, coordinator: ControlPlaneCoordinator, sample_agent: Dict[str, Any]):
        """Test heartbeat with status update."""
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        await coordinator.heartbeat(sample_agent["agent_id"], status=AgentStatus.BUSY)

        agent = await coordinator.get_agent(sample_agent["agent_id"])
        assert agent.status == AgentStatus.BUSY

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent_agent(self, coordinator: ControlPlaneCoordinator):
        """Test heartbeat for nonexistent agent."""
        result = await coordinator.heartbeat("nonexistent-agent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_agents(self, coordinator: ControlPlaneCoordinator, sample_agents):
        """Test listing all agents."""
        for agent_data in sample_agents:
            await coordinator.register_agent(
                agent_id=agent_data["agent_id"],
                capabilities=agent_data["capabilities"],
                model=agent_data["model"],
                provider=agent_data["provider"],
            )

        agents = await coordinator.list_agents()
        assert len(agents) == 3

    @pytest.mark.asyncio
    async def test_list_agents_by_capability(self, coordinator: ControlPlaneCoordinator, sample_agents):
        """Test listing agents filtered by capability."""
        for agent_data in sample_agents:
            await coordinator.register_agent(
                agent_id=agent_data["agent_id"],
                capabilities=agent_data["capabilities"],
            )

        # Filter by debate capability
        debate_agents = await coordinator.list_agents(capability="debate")
        assert len(debate_agents) == 2

        # Filter by code capability
        code_agents = await coordinator.list_agents(capability="code")
        assert len(code_agents) == 2

        # Filter by research capability
        research_agents = await coordinator.list_agents(capability="research")
        assert len(research_agents) == 1

    @pytest.mark.asyncio
    async def test_select_agent(self, coordinator: ControlPlaneCoordinator, sample_agents):
        """Test selecting an agent for a task."""
        for agent_data in sample_agents:
            await coordinator.register_agent(
                agent_id=agent_data["agent_id"],
                capabilities=agent_data["capabilities"],
            )

        # Select agent with debate capability
        agent = await coordinator.select_agent(capabilities=["debate"])
        assert agent is not None
        assert "debate" in agent.capabilities

    @pytest.mark.asyncio
    async def test_select_agent_with_exclusion(self, coordinator: ControlPlaneCoordinator, sample_agents):
        """Test selecting an agent with exclusions."""
        for agent_data in sample_agents:
            await coordinator.register_agent(
                agent_id=agent_data["agent_id"],
                capabilities=agent_data["capabilities"],
            )

        # Exclude claude, should get gpt
        agent = await coordinator.select_agent(
            capabilities=["debate"],
            exclude=["agent-claude"],
        )
        assert agent is not None
        assert agent.agent_id == "agent-gpt"

    @pytest.mark.asyncio
    async def test_select_agent_no_match(self, coordinator: ControlPlaneCoordinator, sample_agents):
        """Test selecting agent with no matching capabilities."""
        for agent_data in sample_agents:
            await coordinator.register_agent(
                agent_id=agent_data["agent_id"],
                capabilities=agent_data["capabilities"],
            )

        # No agent has all these capabilities
        agent = await coordinator.select_agent(capabilities=["debate", "research"])
        assert agent is None


# ============================================================================
# Task Operations Tests
# ============================================================================


class TestTaskOperations:
    """Tests for task submission and execution."""

    @pytest.mark.asyncio
    async def test_submit_task(self, coordinator: ControlPlaneCoordinator, sample_task: Dict[str, Any]):
        """Test submitting a task."""
        task_id = await coordinator.submit_task(
            task_type=sample_task["task_type"],
            payload=sample_task["payload"],
            required_capabilities=sample_task["required_capabilities"],
        )

        assert task_id is not None
        assert len(task_id) > 0

        task = await coordinator.get_task(task_id)
        assert task is not None
        assert task.task_type == sample_task["task_type"]
        assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_submit_task_with_priority(self, coordinator: ControlPlaneCoordinator):
        """Test submitting a task with priority."""
        task_id = await coordinator.submit_task(
            task_type="urgent",
            payload={"data": "test"},
            priority=TaskPriority.HIGH,
        )

        task = await coordinator.get_task(task_id)
        assert task.priority == TaskPriority.HIGH

    @pytest.mark.asyncio
    async def test_claim_task(self, coordinator: ControlPlaneCoordinator, sample_agent, sample_task):
        """Test claiming a task."""
        # Register agent
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        # Submit task
        task_id = await coordinator.submit_task(
            task_type=sample_task["task_type"],
            payload=sample_task["payload"],
            required_capabilities=sample_task["required_capabilities"],
        )

        # Claim task
        claimed_task = await coordinator.claim_task(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
            block_ms=0,
        )

        assert claimed_task is not None
        assert claimed_task.id == task_id
        assert claimed_task.status == TaskStatus.RUNNING
        assert claimed_task.assigned_agent == sample_agent["agent_id"]

    @pytest.mark.asyncio
    async def test_complete_task(self, coordinator: ControlPlaneCoordinator, sample_agent, sample_task):
        """Test completing a task."""
        # Register agent and submit task
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        task_id = await coordinator.submit_task(
            task_type=sample_task["task_type"],
            payload=sample_task["payload"],
            required_capabilities=sample_task["required_capabilities"],
        )

        # Claim and complete
        await coordinator.claim_task(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
            block_ms=0,
        )

        result = {"conclusion": "Test completed successfully"}
        success = await coordinator.complete_task(
            task_id=task_id,
            result=result,
            agent_id=sample_agent["agent_id"],
        )

        assert success is True

        task = await coordinator.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result

    @pytest.mark.asyncio
    async def test_fail_task_with_retry(self, coordinator: ControlPlaneCoordinator, sample_agent, sample_task):
        """Test failing a task with retry."""
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        task_id = await coordinator.submit_task(
            task_type=sample_task["task_type"],
            payload=sample_task["payload"],
        )

        await coordinator.claim_task(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
            block_ms=0,
        )

        # Fail with requeue
        success = await coordinator.fail_task(
            task_id=task_id,
            error="Test error",
            agent_id=sample_agent["agent_id"],
            requeue=True,
        )

        assert success is True

        task = await coordinator.get_task(task_id)
        assert task.status == TaskStatus.PENDING  # Requeued
        assert task.retries == 1
        assert task.error == "Test error"

    @pytest.mark.asyncio
    async def test_fail_task_permanent(self, coordinator: ControlPlaneCoordinator, sample_agent, sample_task):
        """Test permanently failing a task."""
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        task_id = await coordinator.submit_task(
            task_type=sample_task["task_type"],
            payload=sample_task["payload"],
        )

        await coordinator.claim_task(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
            block_ms=0,
        )

        # Fail without requeue
        success = await coordinator.fail_task(
            task_id=task_id,
            error="Permanent failure",
            requeue=False,
        )

        assert success is True

        task = await coordinator.get_task(task_id)
        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_cancel_task(self, coordinator: ControlPlaneCoordinator, sample_task):
        """Test cancelling a task."""
        task_id = await coordinator.submit_task(
            task_type=sample_task["task_type"],
            payload=sample_task["payload"],
        )

        success = await coordinator.cancel_task(task_id)
        assert success is True

        task = await coordinator.get_task(task_id)
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_task(self, coordinator: ControlPlaneCoordinator, sample_agent, sample_task):
        """Test cancelling an already completed task."""
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        task_id = await coordinator.submit_task(
            task_type=sample_task["task_type"],
            payload=sample_task["payload"],
        )

        await coordinator.claim_task(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
            block_ms=0,
        )

        await coordinator.complete_task(task_id=task_id, result={})

        # Try to cancel completed task
        success = await coordinator.cancel_task(task_id)
        assert success is False

    @pytest.mark.asyncio
    async def test_wait_for_result_already_completed(self, coordinator: ControlPlaneCoordinator, sample_agent, sample_task):
        """Test waiting for task result that's already completed."""
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        task_id = await coordinator.submit_task(
            task_type=sample_task["task_type"],
            payload=sample_task["payload"],
        )

        # Claim and complete the task immediately
        await coordinator.claim_task(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
            block_ms=0,
        )
        await coordinator.complete_task(task_id=task_id, result={"done": True})

        # Wait for result (should return immediately since already completed)
        task = await coordinator.wait_for_result(task_id, timeout=5.0)

        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"done": True}

    @pytest.mark.asyncio
    async def test_wait_for_result_timeout(self, coordinator: ControlPlaneCoordinator, sample_task):
        """Test waiting for task result with timeout."""
        task_id = await coordinator.submit_task(
            task_type=sample_task["task_type"],
            payload=sample_task["payload"],
        )

        # Wait with very short timeout (task won't complete)
        task = await coordinator.wait_for_result(task_id, timeout=0.1)

        # Should return None due to timeout
        assert task is None


# ============================================================================
# Health Operations Tests
# ============================================================================


class TestHealthOperations:
    """Tests for health monitoring."""

    @pytest.mark.asyncio
    async def test_get_system_health_healthy(self, coordinator: ControlPlaneCoordinator):
        """Test getting system health when all healthy."""
        status = coordinator.get_system_health()
        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_is_agent_available(self, coordinator: ControlPlaneCoordinator, sample_agent):
        """Test checking agent availability."""
        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
        )

        # Agent should be available by default (no health checks registered)
        available = coordinator.is_agent_available(sample_agent["agent_id"])
        assert available is True

    @pytest.mark.asyncio
    async def test_register_with_health_probe(self, coordinator: ControlPlaneCoordinator, sample_agent):
        """Test registering agent with health probe."""
        probe_called = False

        def health_probe() -> bool:
            nonlocal probe_called
            probe_called = True
            return True

        await coordinator.register_agent(
            agent_id=sample_agent["agent_id"],
            capabilities=sample_agent["capabilities"],
            health_probe=health_probe,
        )

        # Probe should be registered with health monitor
        assert sample_agent["agent_id"] in coordinator._health_monitor._probes


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Tests for control plane statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, coordinator: ControlPlaneCoordinator, sample_agents, sample_task):
        """Test getting comprehensive statistics."""
        # Register some agents
        for agent_data in sample_agents:
            await coordinator.register_agent(
                agent_id=agent_data["agent_id"],
                capabilities=agent_data["capabilities"],
            )

        # Submit some tasks
        for _ in range(3):
            await coordinator.submit_task(
                task_type=sample_task["task_type"],
                payload=sample_task["payload"],
            )

        stats = await coordinator.get_stats()

        assert "registry" in stats
        assert "scheduler" in stats
        assert "health" in stats
        assert "config" in stats

        assert stats["registry"]["total_agents"] == 3
        assert stats["scheduler"]["total"] == 3


# ============================================================================
# Connection Lifecycle Tests
# ============================================================================


class TestConnectionLifecycle:
    """Tests for coordinator connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_and_shutdown(self, mock_registry, mock_scheduler, mock_health_monitor):
        """Test connect and shutdown sequence."""
        coordinator = ControlPlaneCoordinator(
            registry=mock_registry,
            scheduler=mock_scheduler,
            health_monitor=mock_health_monitor,
        )

        # Mock the connect methods
        mock_registry.connect = AsyncMock()
        mock_scheduler.connect = AsyncMock()
        mock_health_monitor.start = AsyncMock()
        mock_health_monitor.stop = AsyncMock()
        mock_registry.close = AsyncMock()
        mock_scheduler.close = AsyncMock()

        await coordinator.connect()
        assert coordinator._connected is True

        await coordinator.shutdown()
        assert coordinator._connected is False

    @pytest.mark.asyncio
    async def test_multiple_connect_calls(self, mock_registry, mock_scheduler, mock_health_monitor):
        """Test that multiple connect calls are idempotent."""
        coordinator = ControlPlaneCoordinator(
            registry=mock_registry,
            scheduler=mock_scheduler,
            health_monitor=mock_health_monitor,
        )

        mock_registry.connect = AsyncMock()
        mock_scheduler.connect = AsyncMock()
        mock_health_monitor.start = AsyncMock()

        await coordinator.connect()
        await coordinator.connect()  # Should not raise

        # Connect methods should only be called once
        assert mock_registry.connect.call_count == 1


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunction:
    """Tests for create_control_plane factory function."""

    @pytest.mark.asyncio
    async def test_create_control_plane(self, mock_redis):
        """Test create_control_plane factory function."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            # This would require Redis, so we patch it
            # For unit tests, we just verify the function exists
            assert callable(create_control_plane)
