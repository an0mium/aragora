"""
End-to-end tests for Control Plane workflow.

Tests the full lifecycle of:
- Agent registration and discovery
- Task submission, claiming, and completion
- Error handling and retry logic
- Health monitoring integration
- Multi-agent coordination
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.control_plane.coordinator import (
    ControlPlaneConfig,
    ControlPlaneCoordinator,
)
from aragora.control_plane.health import HealthMonitor, HealthStatus
from aragora.control_plane.registry import AgentCapability, AgentInfo, AgentRegistry, AgentStatus
from aragora.control_plane.scheduler import Task, TaskPriority, TaskScheduler, TaskStatus


# ============================================================================
# E2E Test Fixtures
# ============================================================================


@pytest.fixture
async def e2e_coordinator():
    """Create a full ControlPlaneCoordinator with in-memory backends."""
    config = ControlPlaneConfig(
        redis_url="redis://localhost:6379",
        key_prefix="e2e:cp:",
        heartbeat_timeout=5.0,  # Short timeout for testing
        task_timeout=10.0,
        max_task_retries=2,
    )

    # Create components with in-memory fallback
    registry = AgentRegistry(
        redis_url="redis://localhost:6379",
        key_prefix="e2e:agents:",
        heartbeat_timeout=5.0,
    )
    registry._redis = None  # Force in-memory mode

    scheduler = TaskScheduler(
        redis_url="redis://localhost:6379",
        key_prefix="e2e:tasks:",
        stream_prefix="e2e:stream:",
    )
    scheduler._redis = None  # Force in-memory mode

    health_monitor = HealthMonitor(
        registry=registry,
        probe_interval=1.0,
        probe_timeout=1.0,
    )

    coordinator = ControlPlaneCoordinator(
        config=config,
        registry=registry,
        scheduler=scheduler,
        health_monitor=health_monitor,
    )
    coordinator._connected = True

    yield coordinator

    # Cleanup
    coordinator._connected = False


# ============================================================================
# Basic Lifecycle E2E Tests
# ============================================================================


class TestBasicLifecycle:
    """Tests for basic agent and task lifecycle."""

    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test complete flow: register agent -> submit task -> claim -> complete."""
        # 1. Register an agent
        agent = await e2e_coordinator.register_agent(
            agent_id="e2e-agent-1",
            capabilities=["debate", "code"],
            model="test-model",
            provider="test-provider",
        )
        assert agent.agent_id == "e2e-agent-1"
        assert agent.status == AgentStatus.READY

        # 2. Submit a task
        task_id = await e2e_coordinator.submit_task(
            task_type="debate",
            payload={"question": "What is the best testing strategy?"},
            required_capabilities=["debate"],
            priority=TaskPriority.NORMAL,
        )
        assert task_id is not None

        # 3. Claim the task as the agent
        task = await e2e_coordinator.claim_task(
            agent_id="e2e-agent-1",
            capabilities=["debate", "code"],
            block_ms=100,
        )
        assert task is not None
        assert task.id == task_id
        assert task.status == TaskStatus.RUNNING
        assert task.assigned_agent == "e2e-agent-1"

        # 4. Complete the task
        result = await e2e_coordinator.complete_task(
            task_id=task_id,
            result={"answer": "Unit tests plus integration tests"},
            agent_id="e2e-agent-1",
            latency_ms=100.0,
        )
        assert result is True

        # 5. Verify task is completed
        completed_task = await e2e_coordinator.get_task(task_id)
        assert completed_task is not None
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result is not None

    @pytest.mark.asyncio
    async def test_task_failure_and_retry(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test task failure and automatic requeue for retry."""
        # Register agent
        await e2e_coordinator.register_agent(
            agent_id="retry-agent",
            capabilities=["analysis"],
            model="test-model",
        )

        # Submit task
        task_id = await e2e_coordinator.submit_task(
            task_type="analysis",
            payload={"data": "analyze this"},
            required_capabilities=["analysis"],
        )

        # Claim task
        task = await e2e_coordinator.claim_task(
            agent_id="retry-agent",
            capabilities=["analysis"],
            block_ms=100,
        )
        assert task is not None
        assert task.retries == 0

        # Fail the task (should requeue)
        await e2e_coordinator.fail_task(
            task_id=task_id,
            error="Simulated failure",
            agent_id="retry-agent",
            requeue=True,
        )

        # Claim again - should get the same task with retry count incremented
        task2 = await e2e_coordinator.claim_task(
            agent_id="retry-agent",
            capabilities=["analysis"],
            block_ms=100,
        )
        assert task2 is not None
        assert task2.id == task_id
        assert task2.retries == 1

        # Complete successfully this time
        await e2e_coordinator.complete_task(
            task_id=task_id,
            result={"analysis": "complete"},
            agent_id="retry-agent",
        )

        final_task = await e2e_coordinator.get_task(task_id)
        assert final_task.status == TaskStatus.COMPLETED


class TestMultiAgentCoordination:
    """Tests for multi-agent task coordination."""

    @pytest.mark.asyncio
    async def test_capability_based_task_routing(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test that tasks are routed to agents with matching capabilities."""
        # Register agents with different capabilities
        await e2e_coordinator.register_agent(
            agent_id="debate-specialist",
            capabilities=["debate"],
            model="debate-model",
        )
        await e2e_coordinator.register_agent(
            agent_id="code-specialist",
            capabilities=["code"],
            model="code-model",
        )

        # Submit code task
        code_task_id = await e2e_coordinator.submit_task(
            task_type="implement",
            payload={"code": "function test()"},
            required_capabilities=["code"],
        )

        # Debate specialist should NOT be able to claim code task
        debate_claim = await e2e_coordinator.claim_task(
            agent_id="debate-specialist",
            capabilities=["debate"],
            block_ms=50,
        )
        assert debate_claim is None

        # Code specialist SHOULD be able to claim code task
        code_claim = await e2e_coordinator.claim_task(
            agent_id="code-specialist",
            capabilities=["code"],
            block_ms=50,
        )
        assert code_claim is not None
        assert code_claim.id == code_task_id

    @pytest.mark.asyncio
    async def test_priority_based_task_ordering(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test that high priority tasks are claimed before normal priority."""
        # Register agent
        await e2e_coordinator.register_agent(
            agent_id="priority-agent",
            capabilities=["general"],
            model="test-model",
        )

        # Submit normal priority task first
        normal_task_id = await e2e_coordinator.submit_task(
            task_type="work",
            payload={"type": "normal"},
            required_capabilities=["general"],
            priority=TaskPriority.NORMAL,
        )

        # Submit high priority task second
        high_task_id = await e2e_coordinator.submit_task(
            task_type="work",
            payload={"type": "urgent"},
            required_capabilities=["general"],
            priority=TaskPriority.HIGH,
        )

        # Claim should return high priority task first
        task = await e2e_coordinator.claim_task(
            agent_id="priority-agent",
            capabilities=["general"],
            block_ms=50,
        )
        assert task is not None
        assert task.id == high_task_id
        assert task.priority == TaskPriority.HIGH

        # Complete and claim again - should get normal priority
        await e2e_coordinator.complete_task(task_id=high_task_id, agent_id="priority-agent")

        task2 = await e2e_coordinator.claim_task(
            agent_id="priority-agent",
            capabilities=["general"],
            block_ms=50,
        )
        assert task2 is not None
        assert task2.id == normal_task_id


class TestAgentManagement:
    """Tests for agent lifecycle management."""

    @pytest.mark.asyncio
    async def test_agent_discovery(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test listing agents by capability."""
        # Register multiple agents
        await e2e_coordinator.register_agent(
            agent_id="discovery-1",
            capabilities=["debate", "code"],
            model="model-1",
        )
        await e2e_coordinator.register_agent(
            agent_id="discovery-2",
            capabilities=["debate"],
            model="model-2",
        )
        await e2e_coordinator.register_agent(
            agent_id="discovery-3",
            capabilities=["code"],
            model="model-3",
        )

        # Find debate-capable agents
        debate_agents = await e2e_coordinator.list_agents(capability="debate")
        agent_ids = [a.agent_id for a in debate_agents]
        assert "discovery-1" in agent_ids
        assert "discovery-2" in agent_ids
        assert "discovery-3" not in agent_ids

        # Find code-capable agents
        code_agents = await e2e_coordinator.list_agents(capability="code")
        agent_ids = [a.agent_id for a in code_agents]
        assert "discovery-1" in agent_ids
        assert "discovery-3" in agent_ids
        assert "discovery-2" not in agent_ids

    @pytest.mark.asyncio
    async def test_agent_unregistration(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test agent unregistration."""
        # Register agent
        await e2e_coordinator.register_agent(
            agent_id="unregister-test",
            capabilities=["test"],
            model="test-model",
        )

        # Verify registered
        agent = await e2e_coordinator.get_agent("unregister-test")
        assert agent is not None

        # Unregister
        result = await e2e_coordinator.unregister_agent("unregister-test")
        assert result is True

        # Verify unregistered
        agent = await e2e_coordinator.get_agent("unregister-test")
        assert agent is None

    @pytest.mark.asyncio
    async def test_heartbeat_updates_status(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test that heartbeats update agent status."""
        # Register agent
        await e2e_coordinator.register_agent(
            agent_id="heartbeat-test",
            capabilities=["test"],
            model="test-model",
        )

        # Update status via heartbeat
        result = await e2e_coordinator.heartbeat(
            agent_id="heartbeat-test",
            status=AgentStatus.BUSY,
        )
        assert result is True

        # Verify status updated
        agent = await e2e_coordinator.get_agent("heartbeat-test")
        assert agent is not None
        assert agent.status == AgentStatus.BUSY


class TestStatistics:
    """Tests for control plane statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test statistics gathering."""
        # Register agents and submit tasks
        await e2e_coordinator.register_agent(
            agent_id="stats-agent",
            capabilities=["test"],
            model="test-model",
        )

        await e2e_coordinator.submit_task(
            task_type="test",
            payload={},
            required_capabilities=["test"],
        )

        # Get stats
        stats = await e2e_coordinator.get_stats()

        assert "registry" in stats
        assert "scheduler" in stats
        assert "health" in stats
        assert "config" in stats


class TestTaskCancellation:
    """Tests for task cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_pending_task(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test cancelling a pending task."""
        await e2e_coordinator.register_agent(
            agent_id="cancel-agent",
            capabilities=["test"],
            model="test-model",
        )

        task_id = await e2e_coordinator.submit_task(
            task_type="test",
            payload={},
            required_capabilities=["test"],
        )

        # Cancel before claiming
        result = await e2e_coordinator.cancel_task(task_id)
        assert result is True

        # Verify cancelled
        task = await e2e_coordinator.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cannot_cancel_completed_task(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test that completed tasks cannot be cancelled."""
        await e2e_coordinator.register_agent(
            agent_id="complete-cancel-agent",
            capabilities=["test"],
            model="test-model",
        )

        task_id = await e2e_coordinator.submit_task(
            task_type="test",
            payload={},
            required_capabilities=["test"],
        )

        # Claim and complete
        task = await e2e_coordinator.claim_task(
            agent_id="complete-cancel-agent",
            capabilities=["test"],
            block_ms=50,
        )
        await e2e_coordinator.complete_task(task_id=task_id, agent_id="complete-cancel-agent")

        # Try to cancel - should fail
        result = await e2e_coordinator.cancel_task(task_id)
        assert result is False


class TestWaitForResult:
    """Tests for waiting on task results."""

    @pytest.mark.asyncio
    async def test_wait_for_completed_task(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test waiting for a task that completes."""
        await e2e_coordinator.register_agent(
            agent_id="wait-agent",
            capabilities=["test"],
            model="test-model",
        )

        task_id = await e2e_coordinator.submit_task(
            task_type="test",
            payload={},
            required_capabilities=["test"],
        )

        # Start a background task to complete the work
        async def worker():
            await asyncio.sleep(0.05)
            task = await e2e_coordinator.claim_task(
                agent_id="wait-agent",
                capabilities=["test"],
                block_ms=50,
            )
            if task:
                await e2e_coordinator.complete_task(
                    task_id=task.id,
                    result={"done": True},
                    agent_id="wait-agent",
                )

        asyncio.create_task(worker())

        # Wait for result
        result = await e2e_coordinator.wait_for_result(task_id, timeout=2.0)
        assert result is not None
        assert result.status == TaskStatus.COMPLETED


class TestHealthIntegration:
    """Tests for health monitoring integration."""

    @pytest.mark.asyncio
    async def test_system_health_status(self, e2e_coordinator: ControlPlaneCoordinator):
        """Test system health reporting."""
        # Initially no agents
        health = e2e_coordinator.get_system_health()
        assert health in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

        # Register agents
        await e2e_coordinator.register_agent(
            agent_id="health-agent-1",
            capabilities=["test"],
            model="test-model",
        )
        await e2e_coordinator.register_agent(
            agent_id="health-agent-2",
            capabilities=["test"],
            model="test-model",
        )

        health = e2e_coordinator.get_system_health()
        assert health in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
