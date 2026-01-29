"""Tests for AgentFabric unified facade."""

from __future__ import annotations

import pytest

from aragora.fabric.fabric import AgentFabric
from aragora.fabric.models import (
    AgentConfig,
    BudgetConfig,
    HealthStatus,
    Policy,
    PolicyContext,
    PolicyEffect,
    PolicyRule,
    Priority,
    Task,
    TaskStatus,
    Usage,
)


@pytest.fixture
def fabric():
    return AgentFabric()


def make_config(id: str = "a1", model: str = "claude-3-opus") -> AgentConfig:
    return AgentConfig(id=id, model=model)


def make_task(id: str = "t1", type: str = "debate") -> Task:
    return Task(id=id, type=type, payload={"data": "test"})


class TestFabricLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self, fabric):
        await fabric.start()
        assert fabric._started
        await fabric.stop()
        assert not fabric._started

    @pytest.mark.asyncio
    async def test_start_idempotent(self, fabric):
        await fabric.start()
        await fabric.start()
        assert fabric._started

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, fabric):
        await fabric.start()
        await fabric.stop()
        await fabric.stop()
        assert not fabric._started


class TestFabricAgentManagement:
    @pytest.mark.asyncio
    async def test_spawn_agent(self, fabric):
        await fabric.start()
        handle = await fabric.spawn(make_config())
        assert handle.agent_id == "a1"
        assert handle.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_spawn_with_budget(self, fabric):
        await fabric.start()
        config = AgentConfig(
            id="a1",
            model="claude-3-opus",
            budget=BudgetConfig(max_tokens_per_day=10000),
        )
        await fabric.spawn(config)
        budget = await fabric.budget.get_budget("a1")
        assert budget is not None
        assert budget.max_tokens_per_day == 10000

    @pytest.mark.asyncio
    async def test_terminate_agent(self, fabric):
        await fabric.start()
        await fabric.spawn(make_config())
        result = await fabric.terminate("a1")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_agent(self, fabric):
        await fabric.start()
        await fabric.spawn(make_config())
        agent = await fabric.get_agent("a1")
        assert agent is not None
        assert agent.agent_id == "a1"

    @pytest.mark.asyncio
    async def test_get_agent_nonexistent(self, fabric):
        agent = await fabric.get_agent("nonexistent")
        assert agent is None

    @pytest.mark.asyncio
    async def test_list_agents(self, fabric):
        await fabric.start()
        for i in range(3):
            await fabric.spawn(make_config(id=f"a{i}"))
        agents = await fabric.list_agents()
        assert len(agents) == 3

    @pytest.mark.asyncio
    async def test_heartbeat(self, fabric):
        await fabric.start()
        await fabric.spawn(make_config())
        result = await fabric.heartbeat("a1")
        assert result is True


class TestFabricScheduling:
    @pytest.mark.asyncio
    async def test_schedule_task(self, fabric):
        await fabric.start()
        await fabric.spawn(make_config())
        handle = await fabric.schedule(make_task(), "a1")
        assert handle.task_id == "t1"
        assert handle.status == TaskStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_schedule_nonexistent_agent(self, fabric):
        with pytest.raises(ValueError, match="not found"):
            await fabric.schedule(make_task(), "nonexistent")

    @pytest.mark.asyncio
    async def test_pop_next_task(self, fabric):
        await fabric.start()
        await fabric.spawn(make_config())
        await fabric.schedule(make_task(), "a1")
        task = await fabric.pop_next_task("a1")
        assert task is not None
        assert task.id == "t1"

    @pytest.mark.asyncio
    async def test_complete_task(self, fabric):
        await fabric.start()
        await fabric.spawn(make_config())
        await fabric.schedule(make_task(), "a1")
        await fabric.pop_next_task("a1")
        await fabric.complete_task("t1", result={"output": "done"})

        handle = await fabric.get_task("t1")
        assert handle.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_task(self, fabric):
        await fabric.start()
        await fabric.spawn(make_config())
        await fabric.schedule(make_task(), "a1")
        result = await fabric.cancel_task("t1")
        assert result is True


class TestFabricPolicy:
    @pytest.mark.asyncio
    async def test_add_check_policy(self, fabric):
        await fabric.add_policy(
            Policy(
                id="deny-shell",
                name="Deny Shell",
                rules=[
                    PolicyRule(
                        action_pattern="tool:shell:*",
                        effect=PolicyEffect.DENY,
                    )
                ],
            )
        )
        ctx = PolicyContext(agent_id="a1", action="tool:shell:execute")
        decision = await fabric.check_policy("tool:shell:execute", ctx)
        assert not decision.allowed

    @pytest.mark.asyncio
    async def test_remove_policy(self, fabric):
        await fabric.add_policy(Policy(id="test", name="Test", rules=[]))
        result = await fabric.remove_policy("test")
        assert result is True


class TestFabricBudget:
    @pytest.mark.asyncio
    async def test_set_and_check_budget(self, fabric):
        await fabric.set_budget("a1", BudgetConfig(max_tokens_per_day=10000))
        allowed, status = await fabric.check_budget("a1", estimated_tokens=100)
        assert allowed

    @pytest.mark.asyncio
    async def test_track_usage(self, fabric):
        await fabric.set_budget("a1", BudgetConfig(max_tokens_per_day=10000))
        status = await fabric.track_usage(
            Usage(
                agent_id="a1",
                tokens_input=500,
                tokens_output=200,
                cost_usd=0.01,
            )
        )
        assert status.tokens_used == 700

    @pytest.mark.asyncio
    async def test_usage_report(self, fabric):
        await fabric.track_usage(
            Usage(agent_id="a1", tokens_input=100, tokens_output=50, cost_usd=0.01)
        )
        report = await fabric.get_usage_report("a1")
        assert report.total_tokens == 150


class TestFabricStats:
    @pytest.mark.asyncio
    async def test_stats(self, fabric):
        await fabric.start()
        await fabric.spawn(make_config())
        stats = await fabric.get_stats()
        assert "scheduler" in stats
        assert "lifecycle" in stats
        assert "policy" in stats
        assert "budget" in stats
        assert stats["lifecycle"]["agents_active"] == 1


class TestComponentAccessors:
    def test_scheduler_accessor(self, fabric):
        assert fabric.scheduler is not None

    def test_lifecycle_accessor(self, fabric):
        assert fabric.lifecycle is not None

    def test_policy_accessor(self, fabric):
        assert fabric.policy is not None

    def test_budget_accessor(self, fabric):
        assert fabric.budget is not None
