"""
Tests for state divergence scenarios in the control plane.

Tests cover:
- Agent heartbeat loss detection
- Task status desync recovery
- Health status mismatch handling
- Capability requirement mismatches
- Multi-instance event broadcasting

Run with: pytest tests/control_plane/test_state_divergence.py -v
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================================================================
# Mock Components
# ============================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(
        self,
        agent_id: str,
        capabilities: Optional[Set[str]] = None,
        status: str = "ready",
    ):
        self.agent_id = agent_id
        self.capabilities = capabilities or {"debate", "code"}
        self.status = status
        self.last_heartbeat = time.time()
        self.task_count = 0


class MockAgentRegistry:
    """Mock agent registry for testing state divergence."""

    def __init__(self):
        self._agents: Dict[str, MockAgent] = {}
        self._offline_threshold = 30.0
        self._cleanup_interval = 10.0
        self._status_updates: List[Dict[str, Any]] = []

    async def register(
        self,
        agent_id: str,
        capabilities: Optional[Set[str]] = None,
        **kwargs,
    ) -> MockAgent:
        """Register an agent."""
        agent = MockAgent(agent_id, capabilities)
        self._agents[agent_id] = agent
        return agent

    async def heartbeat(self, agent_id: str) -> bool:
        """Update agent heartbeat."""
        if agent_id in self._agents:
            self._agents[agent_id].last_heartbeat = time.time()
            return True
        return False

    async def mark_offline(self, agent_id: str) -> None:
        """Mark agent as offline."""
        if agent_id in self._agents:
            self._agents[agent_id].status = "offline"
            self._status_updates.append(
                {
                    "agent_id": agent_id,
                    "old_status": "ready",
                    "new_status": "offline",
                    "time": time.time(),
                }
            )

    async def cleanup_stale_agents(self) -> List[str]:
        """Clean up agents that haven't sent heartbeats."""
        now = time.time()
        stale = []
        for agent_id, agent in self._agents.items():
            if (now - agent.last_heartbeat) > self._offline_threshold:
                if agent.status != "offline":
                    stale.append(agent_id)
                    await self.mark_offline(agent_id)
        return stale

    def get_agent(self, agent_id: str) -> Optional[MockAgent]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def get_agents_by_status(self, status: str) -> List[MockAgent]:
        """Get agents by status."""
        return [a for a in self._agents.values() if a.status == status]

    def get_agents_by_capability(self, capability: str) -> List[MockAgent]:
        """Get agents with a specific capability."""
        return [
            a for a in self._agents.values() if capability in a.capabilities and a.status == "ready"
        ]


class MockTask:
    """Mock task for testing."""

    def __init__(
        self,
        task_id: str,
        required_capabilities: Optional[Set[str]] = None,
        priority: str = "normal",
    ):
        self.task_id = task_id
        self.required_capabilities = required_capabilities or {"debate"}
        self.priority = priority
        self.status = "pending"
        self.assigned_agent: Optional[str] = None
        self.retry_count = 0
        self.max_retries = 3
        self.created_at = time.time()
        self.claimed_at: Optional[float] = None


class MockTaskScheduler:
    """Mock task scheduler for testing."""

    def __init__(self, registry: MockAgentRegistry):
        self._registry = registry
        self._tasks: Dict[str, MockTask] = {}
        self._dead_letter_queue: List[MockTask] = []
        self._claim_timeout = 60.0

    async def submit(
        self,
        task_id: str,
        required_capabilities: Optional[Set[str]] = None,
        priority: str = "normal",
    ) -> MockTask:
        """Submit a new task."""
        task = MockTask(task_id, required_capabilities, priority)
        self._tasks[task_id] = task
        return task

    async def claim(self, task_id: str, agent_id: str) -> bool:
        """Claim a task for an agent."""
        task = self._tasks.get(task_id)
        if not task or task.status != "pending":
            return False

        # Check agent has required capabilities
        agent = self._registry.get_agent(agent_id)
        if not agent or agent.status != "ready":
            return False

        if not task.required_capabilities.issubset(agent.capabilities):
            return False

        task.status = "running"
        task.assigned_agent = agent_id
        task.claimed_at = time.time()
        agent.task_count += 1
        return True

    async def complete(self, task_id: str) -> bool:
        """Complete a task."""
        task = self._tasks.get(task_id)
        if not task or task.status != "running":
            return False

        task.status = "completed"
        agent = self._registry.get_agent(task.assigned_agent)
        if agent:
            agent.task_count -= 1
        return True

    async def fail(self, task_id: str, requeue: bool = True) -> bool:
        """Fail a task, optionally requeuing it."""
        task = self._tasks.get(task_id)
        if not task:
            return False

        agent = self._registry.get_agent(task.assigned_agent)
        if agent:
            agent.task_count -= 1

        if requeue and task.retry_count < task.max_retries:
            task.status = "pending"
            task.assigned_agent = None
            task.retry_count += 1
            return True
        else:
            task.status = "failed"
            self._dead_letter_queue.append(task)
            return True

    async def claim_stale_tasks(self) -> List[str]:
        """Reclaim tasks that have been running too long."""
        now = time.time()
        reclaimed = []

        for task_id, task in self._tasks.items():
            if task.status == "running" and task.claimed_at:
                if (now - task.claimed_at) > self._claim_timeout:
                    # Agent may have crashed - requeue task
                    task.status = "pending"
                    task.assigned_agent = None
                    task.retry_count += 1
                    reclaimed.append(task_id)

        return reclaimed

    def get_pending_tasks(self) -> List[MockTask]:
        """Get all pending tasks."""
        return [t for t in self._tasks.values() if t.status == "pending"]

    def get_task(self, task_id: str) -> Optional[MockTask]:
        """Get task by ID."""
        return self._tasks.get(task_id)


class MockHealthMonitor:
    """Mock health monitor for testing."""

    def __init__(self, registry: MockAgentRegistry):
        self._registry = registry
        self._health_checks: Dict[str, Dict[str, Any]] = {}
        self._failure_threshold = 3
        self._recovery_threshold = 2

    async def check_agent(self, agent_id: str) -> bool:
        """Check agent health (mock implementation)."""
        agent = self._registry.get_agent(agent_id)
        if not agent:
            return False
        return agent.status == "ready"

    async def record_health(self, agent_id: str, healthy: bool) -> None:
        """Record a health check result."""
        if agent_id not in self._health_checks:
            self._health_checks[agent_id] = {
                "consecutive_failures": 0,
                "consecutive_successes": 0,
                "status": "healthy",
            }

        check = self._health_checks[agent_id]

        if healthy:
            check["consecutive_successes"] += 1
            check["consecutive_failures"] = 0
            if check["consecutive_successes"] >= self._recovery_threshold:
                check["status"] = "healthy"
        else:
            check["consecutive_failures"] += 1
            check["consecutive_successes"] = 0
            if check["consecutive_failures"] >= self._failure_threshold:
                check["status"] = "unhealthy"
                # Sync with registry
                await self._registry.mark_offline(agent_id)

    def get_health_status(self, agent_id: str) -> str:
        """Get agent health status."""
        if agent_id not in self._health_checks:
            return "unknown"
        return self._health_checks[agent_id]["status"]


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def registry():
    """Create a mock registry."""
    return MockAgentRegistry()


@pytest.fixture
def scheduler(registry):
    """Create a mock scheduler."""
    return MockTaskScheduler(registry)


@pytest.fixture
def health_monitor(registry):
    """Create a mock health monitor."""
    return MockHealthMonitor(registry)


# ============================================================================
# Agent Heartbeat Tests
# ============================================================================


class TestAgentHeartbeatLoss:
    """Tests for agent heartbeat loss detection."""

    @pytest.mark.asyncio
    async def test_agent_marked_offline_after_timeout(self, registry):
        """Test that agents are marked offline after heartbeat timeout."""
        # Register agent
        agent = await registry.register("agent-1")
        assert agent.status == "ready"

        # Simulate time passing without heartbeat
        agent.last_heartbeat = time.time() - 60  # 60 seconds ago

        # Run cleanup
        stale = await registry.cleanup_stale_agents()

        assert "agent-1" in stale
        assert registry.get_agent("agent-1").status == "offline"

    @pytest.mark.asyncio
    async def test_heartbeat_keeps_agent_alive(self, registry):
        """Test that heartbeats keep agents alive."""
        agent = await registry.register("agent-1")

        # Simulate heartbeat
        await registry.heartbeat("agent-1")

        # Should still be ready
        assert registry.get_agent("agent-1").status == "ready"

    @pytest.mark.asyncio
    async def test_multiple_agents_cleanup(self, registry):
        """Test cleanup of multiple stale agents."""
        # Register multiple agents
        for i in range(5):
            await registry.register(f"agent-{i}")

        # Make some stale
        registry.get_agent("agent-1").last_heartbeat = time.time() - 60
        registry.get_agent("agent-3").last_heartbeat = time.time() - 60

        # Run cleanup
        stale = await registry.cleanup_stale_agents()

        assert len(stale) == 2
        assert "agent-1" in stale
        assert "agent-3" in stale
        assert registry.get_agent("agent-0").status == "ready"
        assert registry.get_agent("agent-2").status == "ready"


# ============================================================================
# Task Status Desync Tests
# ============================================================================


class TestTaskStatusDesync:
    """Tests for task status desync recovery."""

    @pytest.mark.asyncio
    async def test_stale_task_reclaimed(self, registry, scheduler):
        """Test that stale running tasks are reclaimed."""
        await registry.register("agent-1")
        task = await scheduler.submit("task-1")

        # Claim task
        await scheduler.claim("task-1", "agent-1")
        assert task.status == "running"

        # Simulate task running too long
        task.claimed_at = time.time() - 120  # 2 minutes ago

        # Run stale task recovery
        reclaimed = await scheduler.claim_stale_tasks()

        assert "task-1" in reclaimed
        assert task.status == "pending"
        assert task.retry_count == 1

    @pytest.mark.asyncio
    async def test_crashed_agent_task_recovery(self, registry, scheduler):
        """Test task recovery when agent crashes."""
        await registry.register("agent-1")
        task = await scheduler.submit("task-1")

        # Claim task
        await scheduler.claim("task-1", "agent-1")

        # Agent crashes (mark offline)
        await registry.mark_offline("agent-1")

        # Task should be reclaimable after requeue
        await scheduler.fail("task-1", requeue=True)

        assert task.status == "pending"
        assert task.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_exhaustion_sends_to_dlq(self, registry, scheduler):
        """Test that exhausted retries send task to DLQ."""
        await registry.register("agent-1")
        task = await scheduler.submit("task-1")
        task.max_retries = 2
        task.retry_count = 2  # Already at max

        await scheduler.claim("task-1", "agent-1")
        await scheduler.fail("task-1", requeue=True)

        assert task.status == "failed"
        assert len(scheduler._dead_letter_queue) == 1


# ============================================================================
# Health Status Mismatch Tests
# ============================================================================


class TestHealthStatusMismatch:
    """Tests for health status mismatch handling."""

    @pytest.mark.asyncio
    async def test_health_failure_marks_agent_offline(self, registry, health_monitor):
        """Test that health failures mark agent as offline."""
        await registry.register("agent-1")

        # Record consecutive failures
        for _ in range(3):
            await health_monitor.record_health("agent-1", healthy=False)

        assert health_monitor.get_health_status("agent-1") == "unhealthy"
        assert registry.get_agent("agent-1").status == "offline"

    @pytest.mark.asyncio
    async def test_health_recovery(self, registry, health_monitor):
        """Test that healthy checks allow recovery."""
        await registry.register("agent-1")

        # First make unhealthy
        for _ in range(3):
            await health_monitor.record_health("agent-1", healthy=False)
        assert health_monitor.get_health_status("agent-1") == "unhealthy"

        # Now recover
        for _ in range(2):
            await health_monitor.record_health("agent-1", healthy=True)

        assert health_monitor.get_health_status("agent-1") == "healthy"

    @pytest.mark.asyncio
    async def test_flapping_detection(self, registry, health_monitor):
        """Test detection of flapping health status."""
        await registry.register("agent-1")

        # Alternating health
        for i in range(10):
            await health_monitor.record_health("agent-1", healthy=(i % 2 == 0))

        # Should not be marked unhealthy (no consecutive failures)
        check = health_monitor._health_checks["agent-1"]
        assert check["consecutive_failures"] <= 1
        assert check["consecutive_successes"] <= 1


# ============================================================================
# Capability Mismatch Tests
# ============================================================================


class TestCapabilityMismatch:
    """Tests for capability requirement mismatches."""

    @pytest.mark.asyncio
    async def test_task_requires_missing_capability(self, registry, scheduler):
        """Test that tasks can't be claimed by agents without required capabilities."""
        await registry.register("agent-1", capabilities={"debate"})
        task = await scheduler.submit("task-1", required_capabilities={"code", "math"})

        # Should fail to claim
        result = await scheduler.claim("task-1", "agent-1")

        assert result is False
        assert task.status == "pending"

    @pytest.mark.asyncio
    async def test_task_matches_agent_capabilities(self, registry, scheduler):
        """Test successful claim when capabilities match."""
        await registry.register("agent-1", capabilities={"debate", "code", "math"})
        task = await scheduler.submit("task-1", required_capabilities={"code", "math"})

        result = await scheduler.claim("task-1", "agent-1")

        assert result is True
        assert task.status == "running"
        assert task.assigned_agent == "agent-1"

    @pytest.mark.asyncio
    async def test_find_agent_by_capability(self, registry):
        """Test finding agents by capability."""
        await registry.register("agent-1", capabilities={"debate", "code"})
        await registry.register("agent-2", capabilities={"debate"})
        await registry.register("agent-3", capabilities={"code", "math"})

        code_agents = registry.get_agents_by_capability("code")

        assert len(code_agents) == 2
        ids = {a.agent_id for a in code_agents}
        assert "agent-1" in ids
        assert "agent-3" in ids

    @pytest.mark.asyncio
    async def test_offline_agent_excluded_from_capability_search(self, registry):
        """Test that offline agents are excluded from capability searches."""
        await registry.register("agent-1", capabilities={"debate", "code"})
        await registry.register("agent-2", capabilities={"debate", "code"})

        # Mark one offline
        await registry.mark_offline("agent-1")

        code_agents = registry.get_agents_by_capability("code")

        assert len(code_agents) == 1
        assert code_agents[0].agent_id == "agent-2"


# ============================================================================
# Concurrent Access Tests
# ============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_task_claims(self, registry, scheduler):
        """Test that concurrent task claims don't cause issues."""
        await registry.register("agent-1")
        await registry.register("agent-2")
        task = await scheduler.submit("task-1")

        # Simulate concurrent claims
        results = await asyncio.gather(
            scheduler.claim("task-1", "agent-1"),
            scheduler.claim("task-1", "agent-2"),
        )

        # Only one should succeed
        assert results.count(True) == 1
        assert task.status == "running"

    @pytest.mark.asyncio
    async def test_concurrent_heartbeats(self, registry):
        """Test concurrent heartbeats don't cause issues."""
        await registry.register("agent-1")

        # Simulate concurrent heartbeats
        tasks = [registry.heartbeat("agent-1") for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)

    @pytest.mark.asyncio
    async def test_concurrent_registrations(self, registry):
        """Test concurrent agent registrations."""
        tasks = [registry.register(f"agent-{i}") for i in range(20)]
        agents = await asyncio.gather(*tasks)

        # All should be registered
        assert len(agents) == 20
        assert len(registry._agents) == 20


# ============================================================================
# Event Broadcasting Tests
# ============================================================================


class TestEventBroadcasting:
    """Tests for event broadcasting scenarios."""

    @pytest.mark.asyncio
    async def test_status_update_logged(self, registry):
        """Test that status updates are logged."""
        await registry.register("agent-1")
        await registry.mark_offline("agent-1")

        assert len(registry._status_updates) == 1
        assert registry._status_updates[0]["agent_id"] == "agent-1"
        assert registry._status_updates[0]["new_status"] == "offline"

    @pytest.mark.asyncio
    async def test_multiple_status_changes_tracked(self, registry):
        """Test tracking of multiple status changes."""
        for i in range(5):
            await registry.register(f"agent-{i}")

        # Mark some offline
        await registry.mark_offline("agent-1")
        await registry.mark_offline("agent-3")

        assert len(registry._status_updates) == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestControlPlaneIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self, registry, scheduler, health_monitor):
        """Test complete task lifecycle with health monitoring."""
        # Register agents
        await registry.register("agent-1")
        await registry.register("agent-2")

        # Submit task
        task = await scheduler.submit("task-1")

        # Health check before claiming
        await health_monitor.record_health("agent-1", healthy=True)
        await health_monitor.record_health("agent-2", healthy=True)

        # Claim and complete
        await scheduler.claim("task-1", "agent-1")
        assert task.status == "running"

        await scheduler.complete("task-1")
        assert task.status == "completed"

    @pytest.mark.asyncio
    async def test_failover_scenario(self, registry, scheduler, health_monitor):
        """Test task failover when agent becomes unhealthy."""
        await registry.register("agent-1")
        await registry.register("agent-2")
        task = await scheduler.submit("task-1")

        # Claim task with agent-1
        await scheduler.claim("task-1", "agent-1")

        # Agent-1 becomes unhealthy
        for _ in range(3):
            await health_monitor.record_health("agent-1", healthy=False)

        # Fail the task (simulating detection)
        await scheduler.fail("task-1", requeue=True)

        # Task should be reclaimable by agent-2
        result = await scheduler.claim("task-1", "agent-2")

        assert result is True
        assert task.assigned_agent == "agent-2"
        assert task.retry_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
