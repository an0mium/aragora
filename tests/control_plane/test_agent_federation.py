"""
Tests for federated agent pool.
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.control_plane.agent_federation import (
    FederatedAgentPool,
    FederatedAgentConfig,
    FederatedAgent,
    LoadBalanceStrategy,
    FederationMode,
)
from aragora.control_plane.registry import AgentInfo


@pytest.fixture
def mock_agent_info():
    """Create mock agent info."""

    def _create(agent_id: str, capabilities: list = None, status: str = "available"):
        info = MagicMock(spec=AgentInfo)
        info.agent_id = agent_id
        info.model = "test-model"
        info.provider = "test-provider"
        info.capabilities = capabilities or ["debate"]
        info.status = status
        info.tasks_completed = 0
        info.current_task_id = None
        info.is_available.return_value = status == "available"
        info.has_capability.side_effect = lambda c: c in info.capabilities
        info.has_all_capabilities.side_effect = lambda caps: all(
            c in info.capabilities for c in caps
        )
        info.is_available_in_region.return_value = True
        return info

    return _create


@pytest.fixture
def mock_registry(mock_agent_info):
    """Create mock agent registry."""
    registry = MagicMock()
    registry.list_all = AsyncMock(
        return_value=[
            mock_agent_info("claude-3", ["debate", "analysis"]),
            mock_agent_info("gpt-4", ["debate"]),
        ]
    )
    registry.register = AsyncMock(
        side_effect=lambda **kwargs: mock_agent_info(
            kwargs["agent_id"], kwargs.get("capabilities", [])
        )
    )
    registry.unregister = AsyncMock(return_value=True)
    return registry


@pytest.fixture
def mock_event_bus():
    """Create mock event bus."""
    bus = MagicMock()
    bus.publish = AsyncMock(return_value=True)
    bus.subscribe = AsyncMock()
    return bus


@pytest.fixture
def pool(mock_registry, mock_event_bus):
    """Create a federated agent pool."""
    return FederatedAgentPool(
        local_registry=mock_registry,
        event_bus=mock_event_bus,
        instance_id="test-instance",
    )


class TestFederatedAgentPoolInit:
    """Tests for pool initialization."""

    def test_pool_creation(self, mock_registry, mock_event_bus):
        """Test creating a federated pool."""
        pool = FederatedAgentPool(
            local_registry=mock_registry,
            event_bus=mock_event_bus,
            instance_id="my-instance",
        )

        assert pool._instance_id == "my-instance"
        assert pool._connected is False

    def test_pool_without_event_bus(self, mock_registry):
        """Test pool works without event bus."""
        pool = FederatedAgentPool(
            local_registry=mock_registry,
            instance_id="local-only",
        )

        assert pool._event_bus is None

    def test_pool_generates_instance_id(self, mock_registry):
        """Test pool generates instance ID."""
        pool = FederatedAgentPool(local_registry=mock_registry)

        assert pool._instance_id is not None
        assert len(pool._instance_id) > 0

    def test_pool_with_custom_config(self, mock_registry):
        """Test pool with custom configuration."""
        config = FederatedAgentConfig(
            mode=FederationMode.READONLY,
            load_balance_strategy=LoadBalanceStrategy.ROUND_ROBIN,
            prefer_local=False,
        )

        pool = FederatedAgentPool(
            local_registry=mock_registry,
            config=config,
        )

        assert pool._config.mode == FederationMode.READONLY
        assert pool._config.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN


class TestFederatedAgentPoolLifecycle:
    """Tests for pool lifecycle."""

    @pytest.mark.asyncio
    async def test_connect(self, pool, mock_registry, mock_event_bus):
        """Test connecting the pool."""
        await pool.connect()

        assert pool._connected is True
        mock_registry.list_all.assert_called_once()
        mock_event_bus.subscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_loads_local_agents(self, pool, mock_registry):
        """Test connecting loads local agents."""
        await pool.connect()

        # Should have loaded 2 agents from mock registry
        local_agents = pool.list_local_agents()
        assert len(local_agents) == 2

    @pytest.mark.asyncio
    async def test_connect_idempotent(self, pool, mock_event_bus):
        """Test connecting twice is safe."""
        await pool.connect()
        await pool.connect()

        # Should only subscribe once
        assert mock_event_bus.subscribe.call_count == 1

    @pytest.mark.asyncio
    async def test_close(self, pool):
        """Test closing the pool."""
        await pool.connect()
        await pool.close()

        assert pool._connected is False


class TestFederatedAgentPoolFindAgents:
    """Tests for finding agents."""

    @pytest.mark.asyncio
    async def test_find_all_agents(self, pool):
        """Test finding all agents."""
        await pool.connect()

        agents = pool.find_agents()

        assert len(agents) == 2

    @pytest.mark.asyncio
    async def test_find_by_capability(self, pool):
        """Test finding agents by capability."""
        await pool.connect()

        agents = pool.find_agents(capability="analysis")

        # Only claude-3 has "analysis" capability
        assert len(agents) == 1
        assert agents[0].agent_id == "claude-3"

    @pytest.mark.asyncio
    async def test_find_by_multiple_capabilities(self, pool):
        """Test finding agents by multiple capabilities."""
        await pool.connect()

        agents = pool.find_agents(capabilities=["debate", "analysis"])

        # Only claude-3 has both
        assert len(agents) == 1
        assert agents[0].agent_id == "claude-3"

    @pytest.mark.asyncio
    async def test_find_exclude_remote(self, pool):
        """Test finding only local agents."""
        await pool.connect()

        # Add a remote agent
        pool._agents["remote-agent"] = FederatedAgent(
            info=MagicMock(is_available=lambda: True, has_capability=lambda c: True),
            instance_id="other-instance",
            is_local=False,
        )

        agents = pool.find_agents(include_remote=False)

        # Should only return local agents
        assert all(a.is_local for a in agents)

    @pytest.mark.asyncio
    async def test_find_excludes_unhealthy(self, pool):
        """Test unhealthy agents are excluded."""
        await pool.connect()

        # Mark an agent as unhealthy
        pool._agents["claude-3"].consecutive_failures = 5

        agents = pool.find_agents()

        # claude-3 should be excluded (unhealthy)
        assert len(agents) == 1
        assert agents[0].agent_id == "gpt-4"


class TestFederatedAgentPoolSelectAgent:
    """Tests for agent selection strategies."""

    @pytest.mark.asyncio
    async def test_select_random(self, pool):
        """Test random selection."""
        await pool.connect()

        agents = pool.find_agents()
        selected = pool.select_agent(agents, LoadBalanceStrategy.RANDOM)

        assert selected is not None
        assert selected in agents

    @pytest.mark.asyncio
    async def test_select_round_robin(self, pool):
        """Test round-robin selection."""
        await pool.connect()

        agents = pool.find_agents()

        # Select multiple times
        selections = [pool.select_agent(agents, LoadBalanceStrategy.ROUND_ROBIN) for _ in range(4)]

        # Should alternate (approximately)
        assert len(set(s.agent_id for s in selections)) == 2

    @pytest.mark.asyncio
    async def test_select_lowest_latency(self, pool):
        """Test lowest latency selection."""
        await pool.connect()

        # Set different latencies
        pool._agents["claude-3"].estimated_latency_ms = 100
        pool._agents["gpt-4"].estimated_latency_ms = 50

        agents = pool.find_agents()
        selected = pool.select_agent(agents, LoadBalanceStrategy.LOWEST_LATENCY)

        assert selected.agent_id == "gpt-4"

    @pytest.mark.asyncio
    async def test_select_prefer_local(self, pool):
        """Test prefer local selection."""
        await pool.connect()

        # Add remote agents
        for i in range(5):
            remote_info = MagicMock()
            remote_info.is_available.return_value = True
            remote_info.agent_id = f"remote-{i}"
            pool._agents[f"remote-{i}"] = FederatedAgent(
                info=remote_info,
                instance_id="other",
                is_local=False,
            )

        agents = pool.find_agents()

        # With high local_bias, should usually select local
        local_count = 0
        for _ in range(20):
            selected = pool.select_agent(agents, LoadBalanceStrategy.PREFER_LOCAL)
            if selected.is_local:
                local_count += 1

        # Should prefer local agents (with 0.7 bias, expect ~70% local)
        assert local_count >= 10

    @pytest.mark.asyncio
    async def test_select_empty_list(self, pool):
        """Test selecting from empty list."""
        await pool.connect()

        selected = pool.select_agent([], LoadBalanceStrategy.RANDOM)

        assert selected is None


class TestFederatedAgentPoolRegistration:
    """Tests for agent registration."""

    @pytest.mark.asyncio
    async def test_register_agent(self, pool, mock_registry, mock_event_bus):
        """Test registering an agent."""
        await pool.connect()

        agent = await pool.register_agent(
            agent_id="new-agent",
            capabilities=["debate"],
            model="test-model",
            provider="test-provider",
        )

        assert agent.agent_id == "new-agent"
        assert agent.is_local is True
        mock_registry.register.assert_called()

    @pytest.mark.asyncio
    async def test_register_broadcasts_event(self, pool, mock_event_bus):
        """Test registration broadcasts event."""
        pool._config.mode = FederationMode.FULL
        await pool.connect()

        await pool.register_agent(
            agent_id="new-agent",
            capabilities=["debate"],
        )

        # Should publish registration event
        assert mock_event_bus.publish.called

    @pytest.mark.asyncio
    async def test_unregister_agent(self, pool, mock_registry):
        """Test unregistering an agent."""
        await pool.connect()

        result = await pool.unregister_agent("claude-3")

        assert result is True
        assert "claude-3" not in pool._agents

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, pool):
        """Test unregistering nonexistent agent."""
        await pool.connect()

        result = await pool.unregister_agent("nonexistent")

        assert result is False


class TestFederatedAgentPoolEventHandling:
    """Tests for remote event handling."""

    @pytest.mark.asyncio
    async def test_handle_agent_registered_event(self, pool):
        """Test handling agent registration event."""
        await pool.connect()

        # Simulate receiving a registration event from another instance
        from aragora.control_plane.regional_sync import RegionalEvent, RegionalEventType

        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region="other-instance",
            entity_id="remote-claude",
            data={
                "agent_id": "remote-claude",
                "model": "claude-3",
                "provider": "anthropic",
                "capabilities": ["debate"],
                "status": "available",
                "instance_id": "other-instance",
            },
        )

        await pool._handle_remote_event(event)

        # Should have added the remote agent
        assert "remote-claude" in pool._agents
        assert pool._agents["remote-claude"].is_local is False

    @pytest.mark.asyncio
    async def test_handle_agent_unregistered_event(self, pool):
        """Test handling agent unregistration event."""
        await pool.connect()

        # Add a remote agent first
        pool._agents["remote-agent"] = FederatedAgent(
            info=MagicMock(agent_id="remote-agent"),
            instance_id="other-instance",
            is_local=False,
        )

        from aragora.control_plane.regional_sync import RegionalEvent, RegionalEventType

        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_UNREGISTERED,
            source_region="other-instance",
            entity_id="remote-agent",
            data={"agent_id": "remote-agent"},
        )

        await pool._handle_remote_event(event)

        # Should have removed the remote agent
        assert "remote-agent" not in pool._agents

    @pytest.mark.asyncio
    async def test_ignores_own_events(self, pool):
        """Test pool ignores its own events."""
        await pool.connect()
        initial_count = len(pool._agents)

        from aragora.control_plane.regional_sync import RegionalEvent, RegionalEventType

        event = RegionalEvent(
            event_type=RegionalEventType.AGENT_REGISTERED,
            source_region=pool._instance_id,  # Same instance
            entity_id="self-agent",
            data={"agent_id": "self-agent"},
        )

        await pool._handle_remote_event(event)

        # Should not add the agent (from self)
        assert len(pool._agents) == initial_count


class TestFederatedAgentPoolStats:
    """Tests for pool statistics."""

    @pytest.mark.asyncio
    async def test_get_instance_stats(self, pool):
        """Test getting pool statistics."""
        await pool.connect()

        stats = pool.get_instance_stats()

        assert stats["instance_id"] == "test-instance"
        assert stats["total_agents"] == 2
        assert stats["local_agents"] == 2
        assert stats["remote_agents"] == 0
        assert stats["healthy_agents"] == 2

    @pytest.mark.asyncio
    async def test_stats_with_remote_agents(self, pool):
        """Test stats include remote agents."""
        await pool.connect()

        # Add remote agent
        pool._agents["remote-1"] = FederatedAgent(
            info=MagicMock(is_available=lambda: True),
            instance_id="other",
            is_local=False,
        )
        pool._remote_instances["other"] = {"last_seen": time.time()}

        stats = pool.get_instance_stats()

        assert stats["total_agents"] == 3
        assert stats["local_agents"] == 2
        assert stats["remote_agents"] == 1
        assert stats["remote_instances"] == 1


class TestFederatedAgent:
    """Tests for FederatedAgent dataclass."""

    def test_agent_creation(self, mock_agent_info):
        """Test creating a federated agent."""
        info = mock_agent_info("test-agent")
        agent = FederatedAgent(
            info=info,
            instance_id="test-instance",
            is_local=True,
        )

        assert agent.agent_id == "test-agent"
        assert agent.is_local is True
        assert agent.is_healthy is True

    def test_agent_health_tracking(self, mock_agent_info):
        """Test agent health tracking."""
        info = mock_agent_info("test-agent")
        agent = FederatedAgent(info=info, instance_id="i1", is_local=True)

        # Record successes
        agent.record_success(100.0)
        assert agent.consecutive_failures == 0
        assert agent.last_success_at is not None

        # Record failures
        agent.record_failure()
        agent.record_failure()
        agent.record_failure()
        assert agent.consecutive_failures == 3
        assert agent.is_healthy is False

    def test_agent_latency_tracking(self, mock_agent_info):
        """Test latency estimation."""
        info = mock_agent_info("test-agent")
        agent = FederatedAgent(info=info, instance_id="i1", is_local=True)

        # Initial latency
        assert agent.estimated_latency_ms == 0.0

        # Record with EMA
        agent.record_success(100.0)
        assert agent.estimated_latency_ms > 0

        agent.record_success(200.0)
        # EMA with alpha=0.3 starting from 0: 0.3*100 + 0.7*0 = 30, then 0.3*200 + 0.7*30 = 81
        # Value should be higher than after first sample but still influenced by initial 0
        assert 30 < agent.estimated_latency_ms < 200
