"""Tests for Agent Fabric LifecycleManager."""

from __future__ import annotations

import pytest
from datetime import datetime

from aragora.fabric.lifecycle import LifecycleManager
from aragora.fabric.models import AgentConfig, HealthStatus


@pytest.fixture
def manager():
    return LifecycleManager()


@pytest.fixture
def agent_config():
    return AgentConfig(id="test-agent", model="claude-3-opus")


class TestSpawn:
    @pytest.mark.asyncio
    async def test_spawn_agent(self, manager, agent_config):
        handle = await manager.spawn(agent_config)
        assert handle.agent_id == "test-agent"
        assert handle.config.model == "claude-3-opus"
        assert handle.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_spawn_duplicate_raises(self, manager, agent_config):
        await manager.spawn(agent_config)
        with pytest.raises(ValueError, match="already exists"):
            await manager.spawn(agent_config)

    @pytest.mark.asyncio
    async def test_spawn_sets_timestamps(self, manager, agent_config):
        handle = await manager.spawn(agent_config)
        assert isinstance(handle.spawned_at, datetime)
        assert isinstance(handle.last_heartbeat, datetime)

    @pytest.mark.asyncio
    async def test_spawn_multiple_agents(self, manager):
        for i in range(5):
            config = AgentConfig(id=f"agent-{i}", model="gpt-4")
            await manager.spawn(config)
        agents = await manager.list_agents()
        assert len(agents) == 5


class TestTerminate:
    @pytest.mark.asyncio
    async def test_terminate_existing(self, manager, agent_config):
        await manager.spawn(agent_config)
        result = await manager.terminate("test-agent")
        assert result is True

    @pytest.mark.asyncio
    async def test_terminate_nonexistent(self, manager):
        result = await manager.terminate("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_terminate_graceful(self, manager, agent_config):
        await manager.spawn(agent_config)
        result = await manager.terminate("test-agent", graceful=True)
        assert result is True

    @pytest.mark.asyncio
    async def test_terminate_force(self, manager, agent_config):
        await manager.spawn(agent_config)
        result = await manager.terminate("test-agent", graceful=False)
        assert result is True

    @pytest.mark.asyncio
    async def test_terminate_removes_from_list(self, manager, agent_config):
        config = AgentConfig(id="no-pool", model="gpt-4")
        await manager.spawn(config)
        await manager.terminate("no-pool", graceful=False)
        agents = await manager.list_agents()
        assert len(agents) == 0


class TestPooling:
    @pytest.mark.asyncio
    async def test_pool_reuse(self, manager):
        config1 = AgentConfig(id="a1", model="claude-3-opus", pool_id="pool-1")
        await manager.spawn(config1)
        await manager.terminate("a1", graceful=False)

        config2 = AgentConfig(id="a2", model="claude-3-opus", pool_id="pool-1")
        handle = await manager.spawn(config2)
        assert handle.agent_id == "a2"


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_heartbeat_success(self, manager, agent_config):
        await manager.spawn(agent_config)
        result = await manager.heartbeat("test-agent")
        assert result is True

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent(self, manager):
        result = await manager.heartbeat("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_heartbeat_recovers_degraded(self, manager, agent_config):
        await manager.spawn(agent_config)
        handle = manager._agents["test-agent"]
        handle.status = HealthStatus.DEGRADED
        await manager.heartbeat("test-agent")
        assert handle.status == HealthStatus.HEALTHY


class TestHealthStatus:
    @pytest.mark.asyncio
    async def test_get_health(self, manager, agent_config):
        await manager.spawn(agent_config)
        status = await manager.get_health("test-agent")
        assert status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_get_health_nonexistent(self, manager):
        status = await manager.get_health("nonexistent")
        assert status is None


class TestListAgents:
    @pytest.mark.asyncio
    async def test_list_all(self, manager):
        for i in range(3):
            await manager.spawn(AgentConfig(id=f"a{i}", model="gpt-4"))
        agents = await manager.list_agents()
        assert len(agents) == 3

    @pytest.mark.asyncio
    async def test_filter_by_status(self, manager):
        await manager.spawn(AgentConfig(id="a1", model="gpt-4"))
        await manager.spawn(AgentConfig(id="a2", model="gpt-4"))
        manager._agents["a2"].status = HealthStatus.UNHEALTHY

        healthy = await manager.list_agents(status=HealthStatus.HEALTHY)
        assert len(healthy) == 1
        assert healthy[0].agent_id == "a1"

    @pytest.mark.asyncio
    async def test_filter_by_model(self, manager):
        await manager.spawn(AgentConfig(id="a1", model="claude-3-opus"))
        await manager.spawn(AgentConfig(id="a2", model="gpt-4"))

        claude_agents = await manager.list_agents(model="claude-3-opus")
        assert len(claude_agents) == 1
        assert claude_agents[0].model == "claude-3-opus"

    @pytest.mark.asyncio
    async def test_list_empty(self, manager):
        agents = await manager.list_agents()
        assert agents == []


class TestTaskStats:
    @pytest.mark.asyncio
    async def test_update_completed(self, manager, agent_config):
        await manager.spawn(agent_config)
        await manager.update_task_stats("test-agent", completed=5)
        handle = manager._agents["test-agent"]
        assert handle.tasks_completed == 5

    @pytest.mark.asyncio
    async def test_update_failed(self, manager, agent_config):
        await manager.spawn(agent_config)
        await manager.update_task_stats("test-agent", failed=2)
        handle = manager._agents["test-agent"]
        assert handle.tasks_failed == 2

    @pytest.mark.asyncio
    async def test_update_nonexistent_no_error(self, manager):
        await manager.update_task_stats("nonexistent", completed=1)


class TestStats:
    @pytest.mark.asyncio
    async def test_stats(self, manager, agent_config):
        await manager.spawn(agent_config)
        stats = await manager.get_stats()
        assert stats["agents_spawned"] == 1
        assert stats["agents_active"] == 1
        assert stats["agents_healthy"] == 1
        assert stats["agents_degraded"] == 0
        assert stats["agents_unhealthy"] == 0
