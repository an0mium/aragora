"""
Tests for the Agent Fabric.

Tests the core functionality of task scheduling, agent lifecycle,
policy enforcement, and budget management.
"""

import pytest
import asyncio
from datetime import datetime

from aragora.fabric import (
    AgentFabric,
    AgentConfig,
    BudgetConfig,
    IsolationConfig,
    Policy,
    PolicyContext,
    PolicyRule,
    Priority,
    Task,
    TaskStatus,
    Usage,
)
from aragora.fabric.models import PolicyEffect, HealthStatus


@pytest.fixture
async def fabric():
    """Create and start an Agent Fabric instance."""
    fabric = AgentFabric()
    await fabric.start()
    yield fabric
    await fabric.stop()


class TestAgentLifecycle:
    """Tests for agent spawn and terminate."""

    @pytest.mark.asyncio
    async def test_spawn_agent(self, fabric: AgentFabric):
        """Test spawning a new agent."""
        config = AgentConfig(
            id="test-agent-1",
            model="claude-3-opus",
            tools=["shell", "browser"],
        )

        handle = await fabric.spawn(config)

        assert handle.agent_id == "test-agent-1"
        assert handle.config.model == "claude-3-opus"
        assert handle.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_spawn_duplicate_fails(self, fabric: AgentFabric):
        """Test that spawning duplicate agent fails."""
        config = AgentConfig(id="dup-agent", model="claude-3-opus")
        await fabric.spawn(config)

        with pytest.raises(ValueError, match="already exists"):
            await fabric.spawn(config)

    @pytest.mark.asyncio
    async def test_terminate_agent(self, fabric: AgentFabric):
        """Test terminating an agent."""
        config = AgentConfig(id="term-agent", model="claude-3-opus")
        await fabric.spawn(config)

        result = await fabric.terminate("term-agent")
        assert result is True

        agent = await fabric.get_agent("term-agent")
        assert agent is None

    @pytest.mark.asyncio
    async def test_list_agents(self, fabric: AgentFabric):
        """Test listing agents."""
        await fabric.spawn(AgentConfig(id="agent-a", model="claude-3-opus"))
        await fabric.spawn(AgentConfig(id="agent-b", model="gpt-4"))

        agents = await fabric.list_agents()
        assert len(agents) == 2

        opus_agents = await fabric.list_agents(model="claude-3-opus")
        assert len(opus_agents) == 1
        assert opus_agents[0].model == "claude-3-opus"

    @pytest.mark.asyncio
    async def test_heartbeat(self, fabric: AgentFabric):
        """Test agent heartbeat."""
        await fabric.spawn(AgentConfig(id="hb-agent", model="claude-3-opus"))

        result = await fabric.heartbeat("hb-agent")
        assert result is True

        result = await fabric.heartbeat("nonexistent")
        assert result is False


class TestTaskScheduling:
    """Tests for task scheduling."""

    @pytest.mark.asyncio
    async def test_schedule_task(self, fabric: AgentFabric):
        """Test scheduling a task."""
        await fabric.spawn(AgentConfig(id="sched-agent", model="claude-3-opus"))

        task = Task(
            id="task-1",
            type="debate",
            payload={"topic": "AI safety"},
        )

        handle = await fabric.schedule(task, "sched-agent")

        assert handle.task_id == "task-1"
        assert handle.agent_id == "sched-agent"
        assert handle.status == TaskStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_schedule_with_priority(self, fabric: AgentFabric):
        """Test scheduling tasks with different priorities."""
        await fabric.spawn(AgentConfig(id="prio-agent", model="claude-3-opus"))

        task_low = Task(id="low", type="test", payload={})
        task_high = Task(id="high", type="test", payload={})

        await fabric.schedule(task_low, "prio-agent", priority=Priority.LOW)
        await fabric.schedule(task_high, "prio-agent", priority=Priority.HIGH)

        # High priority should come first
        next_task = await fabric.pop_next_task("prio-agent")
        assert next_task.id == "high"

    @pytest.mark.asyncio
    async def test_complete_task(self, fabric: AgentFabric):
        """Test completing a task."""
        await fabric.spawn(AgentConfig(id="comp-agent", model="claude-3-opus"))

        task = Task(id="comp-task", type="test", payload={})
        await fabric.schedule(task, "comp-agent")
        await fabric.pop_next_task("comp-agent")

        await fabric.complete_task("comp-task", result={"answer": 42})

        handle = await fabric.get_task("comp-task")
        assert handle.status == TaskStatus.COMPLETED
        assert handle.result == {"answer": 42}

    @pytest.mark.asyncio
    async def test_cancel_task(self, fabric: AgentFabric):
        """Test canceling a task."""
        await fabric.spawn(AgentConfig(id="canc-agent", model="claude-3-opus"))

        task = Task(id="canc-task", type="test", payload={})
        await fabric.schedule(task, "canc-agent")

        result = await fabric.cancel_task("canc-task")
        assert result is True

        handle = await fabric.get_task("canc-task")
        assert handle.status == TaskStatus.CANCELLED


class TestPolicyEnforcement:
    """Tests for policy enforcement."""

    @pytest.mark.asyncio
    async def test_allow_policy(self, fabric: AgentFabric):
        """Test allowing an action by policy."""
        policy = Policy(
            id="allow-shell",
            name="Allow shell commands",
            rules=[
                PolicyRule(
                    action_pattern="tool.shell.*",
                    effect=PolicyEffect.ALLOW,
                ),
            ],
        )
        await fabric.add_policy(policy)

        context = PolicyContext(agent_id="test-agent")
        decision = await fabric.check_policy("tool.shell.execute", context)

        assert decision.allowed is True

    @pytest.mark.asyncio
    async def test_deny_policy(self, fabric: AgentFabric):
        """Test denying an action by policy."""
        policy = Policy(
            id="deny-network",
            name="Deny network access",
            rules=[
                PolicyRule(
                    action_pattern="tool.network.*",
                    effect=PolicyEffect.DENY,
                ),
            ],
        )
        await fabric.add_policy(policy)

        context = PolicyContext(agent_id="test-agent")
        decision = await fabric.check_policy("tool.network.fetch", context)

        assert decision.allowed is False

    @pytest.mark.asyncio
    async def test_policy_priority(self, fabric: AgentFabric):
        """Test policy priority ordering."""
        # Add low-priority allow-all policy
        await fabric.add_policy(
            Policy(
                id="allow-all",
                name="Allow all",
                priority=0,
                rules=[PolicyRule(action_pattern="*", effect=PolicyEffect.ALLOW)],
            )
        )

        # Add high-priority deny policy
        await fabric.add_policy(
            Policy(
                id="deny-dangerous",
                name="Deny dangerous",
                priority=10,
                rules=[PolicyRule(action_pattern="tool.shell.rm", effect=PolicyEffect.DENY)],
            )
        )

        context = PolicyContext(agent_id="test-agent")

        # Regular command should be allowed
        decision = await fabric.check_policy("tool.shell.ls", context)
        assert decision.allowed is True

        # Dangerous command should be denied
        decision = await fabric.check_policy("tool.shell.rm", context)
        assert decision.allowed is False


class TestBudgetManagement:
    """Tests for budget management."""

    @pytest.mark.asyncio
    async def test_track_usage(self, fabric: AgentFabric):
        """Test tracking usage."""
        await fabric.spawn(AgentConfig(id="budget-agent", model="claude-3-opus"))
        await fabric.set_budget(
            "budget-agent",
            BudgetConfig(
                max_tokens_per_day=10000,
            ),
        )

        usage = Usage(
            agent_id="budget-agent",
            tokens_input=1000,
            tokens_output=500,
        )
        status = await fabric.track_usage(usage)

        assert status.tokens_used == 1500
        assert status.usage_percent == 15.0

    @pytest.mark.asyncio
    async def test_budget_check(self, fabric: AgentFabric):
        """Test budget checking."""
        await fabric.spawn(AgentConfig(id="check-agent", model="claude-3-opus"))
        await fabric.set_budget(
            "check-agent",
            BudgetConfig(
                max_tokens_per_day=1000,
                hard_limit=True,
            ),
        )

        # Should be allowed
        allowed, status = await fabric.check_budget("check-agent", estimated_tokens=500)
        assert allowed is True

        # Track some usage
        await fabric.track_usage(
            Usage(
                agent_id="check-agent",
                tokens_input=800,
            )
        )

        # Should be denied (would exceed limit)
        allowed, status = await fabric.check_budget("check-agent", estimated_tokens=500)
        assert allowed is False

    @pytest.mark.asyncio
    async def test_usage_report(self, fabric: AgentFabric):
        """Test usage report generation."""
        await fabric.spawn(AgentConfig(id="report-agent", model="claude-3-opus"))

        await fabric.track_usage(
            Usage(
                agent_id="report-agent",
                tokens_input=1000,
                model="claude-3-opus",
                cost_usd=0.01,
            )
        )
        await fabric.track_usage(
            Usage(
                agent_id="report-agent",
                tokens_input=500,
                model="claude-3-opus",
                cost_usd=0.005,
            )
        )

        report = await fabric.get_usage_report("report-agent")

        assert report.total_tokens == 1500
        assert report.total_cost_usd == 0.015


class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, fabric: AgentFabric):
        """Test a complete workflow: spawn, schedule, execute, complete."""
        # Set up policy
        await fabric.add_policy(
            Policy(
                id="allow-debate",
                name="Allow debate",
                rules=[PolicyRule(action_pattern="debate.*", effect=PolicyEffect.ALLOW)],
            )
        )

        # Spawn agent with budget
        config = AgentConfig(
            id="workflow-agent",
            model="claude-3-opus",
            budget=BudgetConfig(max_tokens_per_day=100000),
        )
        agent = await fabric.spawn(config)

        # Check policy before scheduling
        context = PolicyContext(agent_id=agent.agent_id)
        decision = await fabric.check_policy("debate.run", context)
        assert decision.allowed is True

        # Check budget before scheduling
        allowed, _ = await fabric.check_budget(agent.agent_id, estimated_tokens=5000)
        assert allowed is True

        # Schedule task
        task = Task(id="debate-1", type="debate", payload={"topic": "Climate"})
        handle = await fabric.schedule(task, agent.agent_id)
        assert handle.status == TaskStatus.SCHEDULED

        # Pop and execute task
        next_task = await fabric.pop_next_task(agent.agent_id)
        assert next_task.id == "debate-1"

        # Track usage during execution
        await fabric.track_usage(
            Usage(
                agent_id=agent.agent_id,
                tokens_input=3000,
                tokens_output=2000,
                task_id="debate-1",
            )
        )

        # Complete task
        await fabric.complete_task("debate-1", result={"consensus": True})

        # Verify final state
        handle = await fabric.get_task("debate-1")
        assert handle.status == TaskStatus.COMPLETED

        report = await fabric.get_usage_report(agent.agent_id)
        assert report.total_tokens == 5000

    @pytest.mark.asyncio
    async def test_stats(self, fabric: AgentFabric):
        """Test getting comprehensive stats."""
        await fabric.spawn(AgentConfig(id="stats-agent", model="claude-3-opus"))

        task = Task(id="stats-task", type="test", payload={})
        await fabric.schedule(task, "stats-agent")

        stats = await fabric.get_stats()

        assert "scheduler" in stats
        assert "lifecycle" in stats
        assert "policy" in stats
        assert "budget" in stats
        assert stats["scheduler"]["tasks_scheduled"] >= 1
        assert stats["lifecycle"]["agents_active"] >= 1
