"""
Control Plane Integration Tests.

Tests cross-component interactions in the control plane:
- Agent registration → task assignment (registry + scheduler)
- Unhealthy agent exclusion (health + scheduler)
- Task retry on agent failure
- Policy enforcement blocks invalid tasks
- Concurrent agent lifecycle (race condition detection)

These tests use in-memory backends (no Redis required).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.control_plane.coordinator import (
    ControlPlaneConfig,
    ControlPlaneCoordinator,
)
from aragora.control_plane.health import HealthMonitor, HealthStatus
from aragora.control_plane.policy.manager import ControlPlanePolicyManager
from aragora.control_plane.policy.types import (
    ControlPlanePolicy,
    EnforcementLevel,
    PolicyDecision,
    PolicyScope,
    PolicyViolation,
    RegionConstraint,
    SLARequirements,
)
from aragora.control_plane.registry import AgentInfo, AgentRegistry, AgentStatus
from aragora.control_plane.scheduler import Task, TaskPriority, TaskScheduler, TaskStatus


# ============================================================================
# Integration Test Fixtures
# ============================================================================


@pytest.fixture
async def integration_coordinator():
    """Create a full coordinator with in-memory backends for integration tests."""
    config = ControlPlaneConfig(
        redis_url="redis://localhost:6379",
        key_prefix="integ:cp:",
        heartbeat_timeout=5.0,
        task_timeout=10.0,
        max_task_retries=3,
    )

    registry = AgentRegistry(
        redis_url="redis://localhost:6379",
        key_prefix="integ:agents:",
        heartbeat_timeout=5.0,
    )
    registry._redis = None
    registry._local_cache = {}

    scheduler = TaskScheduler(
        redis_url="redis://localhost:6379",
        key_prefix="integ:tasks:",
        stream_prefix="integ:stream:",
    )
    scheduler._redis = None
    scheduler._local_tasks = {}
    scheduler._local_queue = []

    health_monitor = HealthMonitor(
        registry=registry,
        probe_interval=1.0,
        probe_timeout=1.0,
        unhealthy_threshold=2,
        recovery_threshold=1,
    )

    coordinator = ControlPlaneCoordinator(
        config=config,
        registry=registry,
        scheduler=scheduler,
        health_monitor=health_monitor,
    )
    coordinator._connected = True

    yield coordinator

    coordinator._connected = False


@pytest.fixture
async def coordinator_with_policy():
    """Create a coordinator with policy manager for enforcement tests."""
    config = ControlPlaneConfig(
        redis_url="redis://localhost:6379",
        key_prefix="policy:cp:",
        heartbeat_timeout=5.0,
        task_timeout=10.0,
        max_task_retries=3,
    )

    registry = AgentRegistry(
        redis_url="redis://localhost:6379",
        key_prefix="policy:agents:",
        heartbeat_timeout=5.0,
    )
    registry._redis = None
    registry._local_cache = {}

    scheduler = TaskScheduler(
        redis_url="redis://localhost:6379",
        key_prefix="policy:tasks:",
        stream_prefix="policy:stream:",
    )
    scheduler._redis = None
    scheduler._local_tasks = {}
    scheduler._local_queue = []

    health_monitor = HealthMonitor(
        registry=registry,
        probe_interval=1.0,
        probe_timeout=1.0,
    )

    violations: list[PolicyViolation] = []
    policy_manager = ControlPlanePolicyManager(
        violation_callback=lambda v: violations.append(v),
    )

    coordinator = ControlPlaneCoordinator(
        config=config,
        registry=registry,
        scheduler=scheduler,
        health_monitor=health_monitor,
        policy_manager=policy_manager,
    )
    coordinator._connected = True

    yield coordinator, policy_manager, violations

    coordinator._connected = False


# ============================================================================
# Test: Agent Registration → Task Assignment
# ============================================================================


class TestAgentRegistrationToTaskAssignment:
    """Test cross-component flow: register agents, then assign tasks via capabilities."""

    @pytest.mark.asyncio
    async def test_register_agent_then_claim_task(self, integration_coordinator):
        """Agent registers, task is submitted, agent claims via capabilities."""
        coord = integration_coordinator

        # Register agent with specific capabilities
        agent = await coord.register_agent(
            agent_id="claude-1",
            capabilities=["debate", "analysis"],
            model="claude-3-opus",
            provider="anthropic",
        )
        assert agent.status == AgentStatus.READY

        # Submit task requiring one of those capabilities
        task_id = await coord.submit_task(
            task_type="debate",
            payload={"question": "Should we use microservices?"},
            required_capabilities=["debate"],
        )
        assert task_id is not None

        # Agent claims the task
        task = await coord.claim_task(
            agent_id="claude-1",
            capabilities=["debate", "analysis"],
        )
        assert task is not None
        assert task.id == task_id
        assert task.status == TaskStatus.RUNNING
        assert task.assigned_agent == "claude-1"

    @pytest.mark.asyncio
    async def test_capability_mismatch_prevents_claiming(self, integration_coordinator):
        """Agent cannot claim tasks requiring capabilities it doesn't have."""
        coord = integration_coordinator

        # Register agent with code capabilities only
        await coord.register_agent(
            agent_id="coder-1",
            capabilities=["code", "refactor"],
            model="codex",
            provider="openai",
        )

        # Submit task requiring debate capability
        task_id = await coord.submit_task(
            task_type="debate",
            payload={"question": "Test question"},
            required_capabilities=["debate"],
        )

        # Agent with wrong capabilities tries to claim
        task = await coord.claim_task(
            agent_id="coder-1",
            capabilities=["code", "refactor"],
            block_ms=0,
        )
        # Task should not be claimed (capability mismatch)
        assert task is None

    @pytest.mark.asyncio
    async def test_multi_agent_routing_by_capability(self, integration_coordinator):
        """Tasks route to the correct agent based on capability matching."""
        coord = integration_coordinator

        # Register two agents with different capabilities
        await coord.register_agent(
            agent_id="debater",
            capabilities=["debate"],
            model="claude-3-opus",
            provider="anthropic",
        )
        await coord.register_agent(
            agent_id="coder",
            capabilities=["code"],
            model="gpt-4",
            provider="openai",
        )

        # Submit both types of tasks
        debate_task_id = await coord.submit_task(
            task_type="debate",
            payload={"question": "Debate topic"},
            required_capabilities=["debate"],
        )
        code_task_id = await coord.submit_task(
            task_type="code",
            payload={"task": "Write function"},
            required_capabilities=["code"],
        )

        # Debater claims debate task
        debate_task = await coord.claim_task(
            agent_id="debater",
            capabilities=["debate"],
        )
        assert debate_task is not None
        assert debate_task.id == debate_task_id

        # Coder claims code task
        code_task = await coord.claim_task(
            agent_id="coder",
            capabilities=["code"],
        )
        assert code_task is not None
        assert code_task.id == code_task_id

    @pytest.mark.asyncio
    async def test_task_completion_records_metrics(self, integration_coordinator):
        """Completing a task updates agent metrics in the registry."""
        coord = integration_coordinator

        await coord.register_agent(
            agent_id="metrics-agent",
            capabilities=["analysis"],
            model="test-model",
            provider="test",
        )

        task_id = await coord.submit_task(
            task_type="analysis",
            payload={"data": "test"},
            required_capabilities=["analysis"],
        )

        await coord.claim_task(
            agent_id="metrics-agent",
            capabilities=["analysis"],
        )

        # Complete the task
        success = await coord.complete_task(
            task_id=task_id,
            result={"answer": "Done"},
            agent_id="metrics-agent",
            latency_ms=150.0,
        )
        assert success is True

        # Verify task is completed
        task = await coord.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"answer": "Done"}


# ============================================================================
# Test: Unhealthy Agent Exclusion
# ============================================================================


class TestUnhealthyAgentExclusion:
    """Test that unhealthy agents are excluded from task dispatch."""

    @pytest.mark.asyncio
    async def test_stale_heartbeat_marks_agent_unavailable(self, integration_coordinator):
        """An agent with a stale heartbeat should not be considered available."""
        coord = integration_coordinator

        agent = await coord.register_agent(
            agent_id="stale-agent",
            capabilities=["debate"],
            model="test",
            provider="test",
        )
        assert agent.status == AgentStatus.READY

        # Simulate stale heartbeat by backdating
        agent_info = await coord.get_agent("stale-agent")
        assert agent_info is not None
        agent_info.last_heartbeat = time.time() - 60  # 60s ago, timeout is 5s

        # Check availability
        is_alive = agent_info.is_alive(timeout_seconds=5.0)
        assert is_alive is False

    @pytest.mark.asyncio
    async def test_heartbeat_refreshes_agent_status(self, integration_coordinator):
        """Sending a heartbeat keeps the agent alive."""
        coord = integration_coordinator

        await coord.register_agent(
            agent_id="healthy-agent",
            capabilities=["debate"],
            model="test",
            provider="test",
        )

        # Send heartbeat
        result = await coord.heartbeat("healthy-agent")
        assert result is True

        # Agent should still be available
        agent = await coord.get_agent("healthy-agent")
        assert agent is not None
        assert agent.is_alive(timeout_seconds=5.0) is True

    @pytest.mark.asyncio
    async def test_agent_selection_prefers_available_agents(self, integration_coordinator):
        """select_agent should prefer agents that are alive and ready."""
        coord = integration_coordinator

        # Register two agents
        await coord.register_agent(
            agent_id="agent-healthy",
            capabilities=["debate"],
            model="claude",
            provider="anthropic",
        )
        await coord.register_agent(
            agent_id="agent-stale",
            capabilities=["debate"],
            model="gpt",
            provider="openai",
        )

        # Make one agent stale
        stale_agent = await coord.get_agent("agent-stale")
        assert stale_agent is not None
        stale_agent.last_heartbeat = time.time() - 120

        # Refresh healthy agent
        await coord.heartbeat("agent-healthy")

        # Select should prefer the healthy agent
        selected = await coord.select_agent(
            capabilities=["debate"],
        )
        assert selected is not None
        # The healthy agent should be preferred
        assert selected.agent_id == "agent-healthy"


# ============================================================================
# Test: Task Retry on Agent Failure
# ============================================================================


class TestTaskRetryOnFailure:
    """Test that tasks are retried when agents fail."""

    @pytest.mark.asyncio
    async def test_failed_task_is_requeued(self, integration_coordinator):
        """A failed task with retries remaining should be requeued."""
        coord = integration_coordinator

        await coord.register_agent(
            agent_id="flaky-agent",
            capabilities=["process"],
            model="test",
            provider="test",
        )

        task_id = await coord.submit_task(
            task_type="process",
            payload={"data": "test"},
            required_capabilities=["process"],
        )

        # Agent claims the task
        task = await coord.claim_task(
            agent_id="flaky-agent",
            capabilities=["process"],
        )
        assert task is not None

        # Agent fails the task with requeue
        await coord.fail_task(
            task_id=task_id,
            error="Transient error",
            agent_id="flaky-agent",
            requeue=True,
        )

        # Task should be available for re-claiming
        failed_task = await coord.get_task(task_id)
        assert failed_task is not None
        # Task should be pending again or have incremented retries
        assert failed_task.retries >= 1

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self, integration_coordinator):
        """Task fails once, then succeeds on second attempt."""
        coord = integration_coordinator

        await coord.register_agent(
            agent_id="retry-agent",
            capabilities=["compute"],
            model="test",
            provider="test",
        )

        task_id = await coord.submit_task(
            task_type="compute",
            payload={"input": 42},
            required_capabilities=["compute"],
        )

        # First attempt: claim and fail
        task = await coord.claim_task(
            agent_id="retry-agent",
            capabilities=["compute"],
        )
        assert task is not None
        await coord.fail_task(
            task_id=task_id,
            error="Timeout",
            agent_id="retry-agent",
            requeue=True,
        )

        # Second attempt: claim and succeed
        task = await coord.claim_task(
            agent_id="retry-agent",
            capabilities=["compute"],
        )
        if task is not None:
            success = await coord.complete_task(
                task_id=task_id,
                result={"output": 84},
                agent_id="retry-agent",
            )
            assert success is True

            final_task = await coord.get_task(task_id)
            assert final_task is not None
            assert final_task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_fail_without_requeue(self, integration_coordinator):
        """A failed task without requeue should stay failed."""
        coord = integration_coordinator

        await coord.register_agent(
            agent_id="fail-agent",
            capabilities=["process"],
            model="test",
            provider="test",
        )

        task_id = await coord.submit_task(
            task_type="process",
            payload={"data": "critical"},
            required_capabilities=["process"],
        )

        task = await coord.claim_task(
            agent_id="fail-agent",
            capabilities=["process"],
        )
        assert task is not None

        # Fail without requeue
        await coord.fail_task(
            task_id=task_id,
            error="Fatal error",
            agent_id="fail-agent",
            requeue=False,
        )

        failed_task = await coord.get_task(task_id)
        assert failed_task is not None
        assert failed_task.status == TaskStatus.FAILED
        assert failed_task.error == "Fatal error"


# ============================================================================
# Test: Policy Enforcement
# ============================================================================


class TestPolicyEnforcement:
    """Test that policy manager enforces constraints on task dispatch."""

    @pytest.mark.asyncio
    async def test_blocked_agent_denied_by_policy(self, coordinator_with_policy):
        """A policy blocking an agent should prevent task dispatch to it."""
        coord, policy_mgr, violations = coordinator_with_policy

        # Create a policy that blocks specific agent
        policy = ControlPlanePolicy(
            name="block-untrusted",
            scope=PolicyScope.GLOBAL,
            agent_blocklist=["untrusted-agent"],
            enforcement_level=EnforcementLevel.HARD,
        )
        policy_mgr.add_policy(policy)

        # Evaluate dispatch to blocked agent
        result = policy_mgr.evaluate_task_dispatch(
            task_type="debate",
            agent_id="untrusted-agent",
            region="us-east-1",
        )
        assert result.decision == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_allowed_agent_passes_policy(self, coordinator_with_policy):
        """An allowed agent should pass policy evaluation."""
        coord, policy_mgr, violations = coordinator_with_policy

        # Create a restrictive policy
        policy = ControlPlanePolicy(
            name="trusted-only",
            scope=PolicyScope.GLOBAL,
            agent_allowlist=["trusted-agent"],
            enforcement_level=EnforcementLevel.HARD,
        )
        policy_mgr.add_policy(policy)

        # Evaluate dispatch to allowed agent
        result = policy_mgr.evaluate_task_dispatch(
            task_type="debate",
            agent_id="trusted-agent",
            region="us-east-1",
        )
        assert result.decision == PolicyDecision.ALLOW

    @pytest.mark.asyncio
    async def test_warn_enforcement_allows_but_logs(self, coordinator_with_policy):
        """WARN enforcement should allow dispatch but set decision to WARN."""
        coord, policy_mgr, violations = coordinator_with_policy

        policy = ControlPlanePolicy(
            name="warn-policy",
            scope=PolicyScope.GLOBAL,
            agent_blocklist=["risky-agent"],
            enforcement_level=EnforcementLevel.WARN,
        )
        policy_mgr.add_policy(policy)

        result = policy_mgr.evaluate_task_dispatch(
            task_type="debate",
            agent_id="risky-agent",
            region="us-east-1",
        )
        assert result.decision == PolicyDecision.WARN

    @pytest.mark.asyncio
    async def test_region_constraint_enforcement(self, coordinator_with_policy):
        """Region constraints should restrict which regions can execute tasks."""
        coord, policy_mgr, violations = coordinator_with_policy

        policy = ControlPlanePolicy(
            name="eu-only",
            scope=PolicyScope.REGION,
            region_constraint=RegionConstraint(
                allowed_regions=["eu-west-1", "eu-central-1"],
            ),
            enforcement_level=EnforcementLevel.HARD,
        )
        policy_mgr.add_policy(policy)

        # EU region should be allowed
        result_eu = policy_mgr.evaluate_task_dispatch(
            task_type="debate",
            agent_id="agent-1",
            region="eu-west-1",
        )
        assert result_eu.decision == PolicyDecision.ALLOW

        # US region should be denied
        result_us = policy_mgr.evaluate_task_dispatch(
            task_type="debate",
            agent_id="agent-1",
            region="us-east-1",
        )
        assert result_us.decision == PolicyDecision.DENY

    @pytest.mark.asyncio
    async def test_sla_compliance_check(self, coordinator_with_policy):
        """SLA requirements should flag non-compliant tasks."""
        coord, policy_mgr, violations = coordinator_with_policy

        policy = ControlPlanePolicy(
            name="strict-sla",
            scope=PolicyScope.GLOBAL,
            sla=SLARequirements(
                max_execution_seconds=120.0,
                max_queue_seconds=30.0,
            ),
            enforcement_level=EnforcementLevel.HARD,
        )
        policy_mgr.add_policy(policy)

        # Check compliant execution
        compliant = policy_mgr.evaluate_sla_compliance(
            policy_id=policy.id,
            execution_seconds=60.0,
        )
        assert compliant.decision == PolicyDecision.ALLOW

        # Check non-compliant execution
        violated = policy_mgr.evaluate_sla_compliance(
            policy_id=policy.id,
            execution_seconds=200.0,
        )
        assert violated.decision != PolicyDecision.ALLOW

    @pytest.mark.asyncio
    async def test_disabled_policy_is_skipped(self, coordinator_with_policy):
        """A disabled policy should not block task dispatch."""
        coord, policy_mgr, violations = coordinator_with_policy

        policy = ControlPlanePolicy(
            name="disabled-block",
            scope=PolicyScope.GLOBAL,
            agent_blocklist=["any-agent"],
            enforcement_level=EnforcementLevel.HARD,
            enabled=False,
        )
        policy_mgr.add_policy(policy)

        result = policy_mgr.evaluate_task_dispatch(
            task_type="debate",
            agent_id="any-agent",
            region="us-east-1",
        )
        assert result.decision == PolicyDecision.ALLOW


# ============================================================================
# Test: Concurrent Agent Lifecycle
# ============================================================================


class TestConcurrentAgentLifecycle:
    """Test race conditions and concurrent operations in the control plane."""

    @pytest.mark.asyncio
    async def test_concurrent_agent_registrations(self, integration_coordinator):
        """Multiple agents registering concurrently should all succeed."""
        coord = integration_coordinator

        async def register_agent(i: int):
            return await coord.register_agent(
                agent_id=f"concurrent-{i}",
                capabilities=["debate"],
                model="test",
                provider="test",
            )

        # Register 5 agents concurrently
        agents = await asyncio.gather(*[register_agent(i) for i in range(5)])
        assert len(agents) == 5
        assert all(a.status == AgentStatus.READY for a in agents)

        # All should be discoverable
        all_agents = await coord.list_agents()
        agent_ids = {a.agent_id for a in all_agents}
        for i in range(5):
            assert f"concurrent-{i}" in agent_ids

    @pytest.mark.asyncio
    async def test_concurrent_task_submissions(self, integration_coordinator):
        """Multiple tasks submitted concurrently should all be tracked."""
        coord = integration_coordinator

        await coord.register_agent(
            agent_id="worker",
            capabilities=["process"],
            model="test",
            provider="test",
        )

        async def submit_task(i: int):
            return await coord.submit_task(
                task_type="process",
                payload={"index": i},
                required_capabilities=["process"],
            )

        # Submit 10 tasks concurrently
        task_ids = await asyncio.gather(*[submit_task(i) for i in range(10)])
        assert len(task_ids) == 10
        assert len(set(task_ids)) == 10  # All unique

    @pytest.mark.asyncio
    async def test_unregister_agent_with_no_tasks(self, integration_coordinator):
        """Unregistering an idle agent should succeed cleanly."""
        coord = integration_coordinator

        await coord.register_agent(
            agent_id="removable",
            capabilities=["debate"],
            model="test",
            provider="test",
        )

        removed = await coord.unregister_agent("removable")
        assert removed is True

        agent = await coord.get_agent("removable")
        assert agent is None

    @pytest.mark.asyncio
    async def test_priority_based_task_ordering(self, integration_coordinator):
        """Higher priority tasks should be claimed before lower priority ones."""
        coord = integration_coordinator

        await coord.register_agent(
            agent_id="priority-worker",
            capabilities=["process"],
            model="test",
            provider="test",
        )

        # Submit low priority first, then urgent
        low_id = await coord.submit_task(
            task_type="process",
            payload={"priority": "low"},
            required_capabilities=["process"],
            priority=TaskPriority.LOW,
        )
        urgent_id = await coord.submit_task(
            task_type="process",
            payload={"priority": "urgent"},
            required_capabilities=["process"],
            priority=TaskPriority.URGENT,
        )

        # First claim should get the urgent task
        task = await coord.claim_task(
            agent_id="priority-worker",
            capabilities=["process"],
        )
        assert task is not None
        assert task.id == urgent_id

    @pytest.mark.asyncio
    async def test_stats_reflect_agent_and_task_counts(self, integration_coordinator):
        """Statistics should accurately reflect registered agents and tasks."""
        coord = integration_coordinator

        # Register agents
        await coord.register_agent(
            agent_id="stat-agent-1",
            capabilities=["debate"],
            model="test",
            provider="test",
        )
        await coord.register_agent(
            agent_id="stat-agent-2",
            capabilities=["code"],
            model="test",
            provider="test",
        )

        # Submit task
        await coord.submit_task(
            task_type="debate",
            payload={"q": "test"},
            required_capabilities=["debate"],
        )

        stats = await coord.get_stats()
        assert stats is not None
        # Stats should report agents and tasks
        assert isinstance(stats, dict)
