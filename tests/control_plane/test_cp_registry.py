"""
Tests for the Control Plane AgentRegistry.

These tests verify the agent registration, heartbeat, and capability-based
selection functionality.
"""

import asyncio
import time

import pytest

from aragora.control_plane.registry import (
    AgentCapability,
    AgentInfo,
    AgentRegistry,
    AgentStatus,
)


class TestAgentInfo:
    """Tests for AgentInfo dataclass."""

    def test_creation(self):
        """Test basic AgentInfo creation."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate", "code"},
            model="claude-3-opus",
            provider="anthropic",
        )

        assert agent.agent_id == "test-agent"
        assert agent.capabilities == {"debate", "code"}
        assert agent.model == "claude-3-opus"
        assert agent.provider == "anthropic"
        assert agent.status == AgentStatus.STARTING

    def test_is_available(self):
        """Test availability check."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities=set(),
            status=AgentStatus.READY,
        )
        assert agent.is_available()

        agent.status = AgentStatus.BUSY
        assert not agent.is_available()

        agent.status = AgentStatus.OFFLINE
        assert not agent.is_available()

    def test_is_alive(self):
        """Test liveness check."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities=set(),
            last_heartbeat=time.time(),
        )
        assert agent.is_alive(timeout_seconds=30.0)

        # Simulate old heartbeat
        agent.last_heartbeat = time.time() - 60
        assert not agent.is_alive(timeout_seconds=30.0)

    def test_has_capability(self):
        """Test capability checking."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate", "code"},
        )

        assert agent.has_capability("debate")
        assert agent.has_capability(AgentCapability.DEBATE)
        assert not agent.has_capability("research")

    def test_has_all_capabilities(self):
        """Test multiple capability checking."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate", "code", "analysis"},
        )

        assert agent.has_all_capabilities(["debate", "code"])
        assert agent.has_all_capabilities([AgentCapability.DEBATE])
        assert not agent.has_all_capabilities(["debate", "research"])

    def test_serialization(self):
        """Test to_dict and from_dict."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate", "code"},
            model="claude-3-opus",
            provider="anthropic",
            status=AgentStatus.READY,
            metadata={"version": "1.0"},
            tags={"fast", "reliable"},
        )

        data = agent.to_dict()
        restored = AgentInfo.from_dict(data)

        assert restored.agent_id == agent.agent_id
        assert restored.capabilities == agent.capabilities
        assert restored.model == agent.model
        assert restored.provider == agent.provider
        assert restored.status == agent.status
        assert restored.metadata == agent.metadata
        assert restored.tags == agent.tags


class TestAgentRegistry:
    """Tests for AgentRegistry with in-memory fallback."""

    @pytest.fixture
    def registry(self):
        """Create a registry using in-memory fallback."""
        return AgentRegistry(redis_url="redis://nonexistent:6379")

    @pytest.mark.asyncio
    async def test_register_agent(self, registry):
        """Test agent registration."""
        agent = await registry.register(
            agent_id="claude-3",
            capabilities=["debate", "code"],
            model="claude-3-opus",
            provider="anthropic",
            metadata={"tier": "premium"},
        )

        assert agent.agent_id == "claude-3"
        assert agent.capabilities == {"debate", "code"}
        assert agent.model == "claude-3-opus"
        assert agent.status == AgentStatus.READY

    @pytest.mark.asyncio
    async def test_get_agent(self, registry):
        """Test getting agent by ID."""
        await registry.register(
            agent_id="test-agent",
            capabilities=["debate"],
        )

        agent = await registry.get("test-agent")
        assert agent is not None
        assert agent.agent_id == "test-agent"

        # Non-existent agent
        missing = await registry.get("nonexistent")
        assert missing is None

    @pytest.mark.asyncio
    async def test_unregister_agent(self, registry):
        """Test agent unregistration."""
        await registry.register(
            agent_id="test-agent",
            capabilities=["debate"],
        )

        success = await registry.unregister("test-agent")
        assert success

        agent = await registry.get("test-agent")
        assert agent is None

        # Unregister non-existent
        success = await registry.unregister("nonexistent")
        assert not success

    @pytest.mark.asyncio
    async def test_heartbeat(self, registry):
        """Test heartbeat updates."""
        await registry.register(
            agent_id="test-agent",
            capabilities=["debate"],
        )

        initial_agent = await registry.get("test-agent")
        initial_heartbeat = initial_agent.last_heartbeat

        # Wait a bit and send heartbeat
        await asyncio.sleep(0.1)
        success = await registry.heartbeat("test-agent")
        assert success

        updated_agent = await registry.get("test-agent")
        assert updated_agent.last_heartbeat > initial_heartbeat

        # Heartbeat with status update
        success = await registry.heartbeat("test-agent", status=AgentStatus.BUSY)
        assert success

        busy_agent = await registry.get("test-agent")
        assert busy_agent.status == AgentStatus.BUSY

    @pytest.mark.asyncio
    async def test_list_all(self, registry):
        """Test listing all agents."""
        await registry.register(agent_id="agent-1", capabilities=["debate"])
        await registry.register(agent_id="agent-2", capabilities=["code"])

        agents = await registry.list_all()
        assert len(agents) == 2

        agent_ids = {a.agent_id for a in agents}
        assert agent_ids == {"agent-1", "agent-2"}

    @pytest.mark.asyncio
    async def test_find_by_capability(self, registry):
        """Test finding agents by capability."""
        await registry.register(agent_id="agent-1", capabilities=["debate", "code"])
        await registry.register(agent_id="agent-2", capabilities=["code"])
        await registry.register(agent_id="agent-3", capabilities=["research"])

        debate_agents = await registry.find_by_capability("debate")
        assert len(debate_agents) == 1
        assert debate_agents[0].agent_id == "agent-1"

        code_agents = await registry.find_by_capability("code")
        assert len(code_agents) == 2

    @pytest.mark.asyncio
    async def test_find_by_capabilities(self, registry):
        """Test finding agents by multiple capabilities."""
        await registry.register(agent_id="agent-1", capabilities=["debate", "code"])
        await registry.register(agent_id="agent-2", capabilities=["code"])

        agents = await registry.find_by_capabilities(["debate", "code"])
        assert len(agents) == 1
        assert agents[0].agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_select_agent(self, registry):
        """Test agent selection with load balancing."""
        await registry.register(agent_id="agent-1", capabilities=["debate"])
        await registry.register(agent_id="agent-2", capabilities=["debate"])

        # Select should return one of the agents
        selected = await registry.select_agent(capabilities=["debate"])
        assert selected is not None
        assert selected.agent_id in {"agent-1", "agent-2"}

        # No agent with required capability
        selected = await registry.select_agent(capabilities=["research"])
        assert selected is None

        # Exclude specific agents
        selected = await registry.select_agent(
            capabilities=["debate"],
            exclude=["agent-1"],
        )
        assert selected is not None
        assert selected.agent_id == "agent-2"

    @pytest.mark.asyncio
    async def test_update_status(self, registry):
        """Test status updates."""
        await registry.register(agent_id="test-agent", capabilities=["debate"])

        success = await registry.update_status("test-agent", AgentStatus.BUSY)
        assert success

        agent = await registry.get("test-agent")
        assert agent.status == AgentStatus.BUSY

    @pytest.mark.asyncio
    async def test_record_task_completion(self, registry):
        """Test task completion recording."""
        await registry.register(agent_id="test-agent", capabilities=["debate"])

        # Record successful task
        success = await registry.record_task_completion(
            agent_id="test-agent",
            success=True,
            latency_ms=1500.0,
        )
        assert success

        agent = await registry.get("test-agent")
        assert agent.tasks_completed == 1
        assert agent.tasks_failed == 0
        assert agent.avg_latency_ms == 1500.0

        # Record failed task
        await registry.record_task_completion(
            agent_id="test-agent",
            success=False,
            latency_ms=500.0,
        )

        agent = await registry.get("test-agent")
        assert agent.tasks_completed == 1
        assert agent.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_get_stats(self, registry):
        """Test statistics retrieval."""
        await registry.register(
            agent_id="agent-1",
            capabilities=["debate"],
            provider="anthropic",
        )
        await registry.register(
            agent_id="agent-2",
            capabilities=["code"],
            provider="openai",
        )

        stats = await registry.get_stats()

        assert stats["total_agents"] == 2
        assert stats["by_capability"]["debate"] == 1
        assert stats["by_capability"]["code"] == 1
        assert stats["by_provider"]["anthropic"] == 1
        assert stats["by_provider"]["openai"] == 1


class TestAgentInfoRegional:
    """Tests for AgentInfo regional functionality."""

    def test_default_region(self):
        """Test default region is 'default'."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate"},
        )
        assert agent.region_id == "default"
        assert "default" in agent.available_regions

    def test_region_assignment(self):
        """Test explicit region assignment."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate"},
            region_id="us-west-2",
            available_regions={"us-west-2", "us-east-1"},
        )
        assert agent.region_id == "us-west-2"
        assert agent.available_regions == {"us-west-2", "us-east-1"}

    def test_is_available_in_region(self):
        """Test regional availability check."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate"},
            region_id="us-west-2",
            available_regions={"us-west-2", "us-east-1"},
        )
        assert agent.is_available_in_region("us-west-2")
        assert agent.is_available_in_region("us-east-1")
        assert not agent.is_available_in_region("eu-west-1")

    def test_get_latency_for_region(self):
        """Test regional latency retrieval."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate"},
            region_id="us-west-2",
            available_regions={"us-west-2", "us-east-1"},
            avg_latency_ms=100.0,
            region_latency_ms={"us-west-2": 50.0, "us-east-1": 150.0},
        )
        assert agent.get_latency_for_region("us-west-2") == 50.0
        assert agent.get_latency_for_region("us-east-1") == 150.0
        # Unknown region in available_regions uses avg_latency_ms
        agent.available_regions.add("eu-west-1")
        assert agent.get_latency_for_region("eu-west-1") == 100.0
        # Region not available returns inf
        assert agent.get_latency_for_region("ap-southeast-1") == float("inf")

    def test_update_region_heartbeat(self):
        """Test regional heartbeat update."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate"},
            region_id="us-west-2",
        )
        old_time = agent.last_heartbeat
        time.sleep(0.01)
        agent.update_region_heartbeat("us-east-1")

        assert agent.last_heartbeat > old_time
        assert "us-east-1" in agent.last_heartbeat_by_region
        assert agent.last_heartbeat_by_region["us-east-1"] == agent.last_heartbeat

    def test_serialization_with_regions(self):
        """Test serialization includes regional fields."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate"},
            region_id="us-west-2",
            available_regions={"us-west-2", "us-east-1"},
            region_latency_ms={"us-west-2": 50.0},
        )
        data = agent.to_dict()

        assert data["region_id"] == "us-west-2"
        assert set(data["available_regions"]) == {"us-west-2", "us-east-1"}
        assert data["region_latency_ms"] == {"us-west-2": 50.0}

    def test_deserialization_with_regions(self):
        """Test deserialization includes regional fields."""
        data = {
            "agent_id": "test-agent",
            "capabilities": ["debate"],
            "status": "ready",
            "region_id": "us-west-2",
            "available_regions": ["us-west-2", "us-east-1"],
            "region_latency_ms": {"us-west-2": 50.0},
        }
        agent = AgentInfo.from_dict(data)

        assert agent.region_id == "us-west-2"
        assert agent.available_regions == {"us-west-2", "us-east-1"}
        assert agent.region_latency_ms == {"us-west-2": 50.0}


class TestAgentRegistryRegional:
    """Tests for AgentRegistry regional functionality."""

    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return AgentRegistry(redis_url="redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_register_with_region(self, registry):
        """Test agent registration with regional info."""
        agent = await registry.register(
            agent_id="regional-agent",
            capabilities=["debate"],
            region_id="us-west-2",
            available_regions=["us-west-2", "us-east-1"],
        )

        assert agent.region_id == "us-west-2"
        assert "us-west-2" in agent.available_regions
        assert "us-east-1" in agent.available_regions

    @pytest.mark.asyncio
    async def test_find_by_region(self, registry):
        """Test finding agents by region."""
        await registry.register(
            agent_id="west-agent",
            capabilities=["debate"],
            region_id="us-west-2",
        )
        await registry.register(
            agent_id="east-agent",
            capabilities=["debate"],
            region_id="us-east-1",
        )
        await registry.register(
            agent_id="multi-agent",
            capabilities=["debate"],
            region_id="us-west-2",
            available_regions=["us-west-2", "us-east-1", "eu-west-1"],
        )

        west_agents = await registry.find_by_region("us-west-2")
        assert len(west_agents) == 2
        agent_ids = {a.agent_id for a in west_agents}
        assert "west-agent" in agent_ids
        assert "multi-agent" in agent_ids

        eu_agents = await registry.find_by_region("eu-west-1")
        assert len(eu_agents) == 1
        assert eu_agents[0].agent_id == "multi-agent"

    @pytest.mark.asyncio
    async def test_find_by_capability_and_region(self, registry):
        """Test finding agents by capability and region."""
        await registry.register(
            agent_id="west-debater",
            capabilities=["debate"],
            region_id="us-west-2",
        )
        await registry.register(
            agent_id="west-coder",
            capabilities=["code"],
            region_id="us-west-2",
        )
        await registry.register(
            agent_id="east-debater",
            capabilities=["debate"],
            region_id="us-east-1",
        )

        agents = await registry.find_by_capability_and_region("debate", "us-west-2")
        assert len(agents) == 1
        assert agents[0].agent_id == "west-debater"

    @pytest.mark.asyncio
    async def test_select_agent_in_region(self, registry):
        """Test regional agent selection."""
        await registry.register(
            agent_id="west-agent",
            capabilities=["debate"],
            region_id="us-west-2",
        )
        await registry.register(
            agent_id="east-agent",
            capabilities=["debate"],
            region_id="us-east-1",
        )

        # Select preferring us-west-2
        agent = await registry.select_agent_in_region(
            capabilities=["debate"],
            target_region="us-west-2",
        )
        assert agent is not None
        assert agent.agent_id == "west-agent"

        # Select preferring unknown region with fallback
        agent = await registry.select_agent_in_region(
            capabilities=["debate"],
            target_region="eu-west-1",
            fallback_regions=["us-east-1"],
        )
        assert agent is not None
        assert agent.agent_id == "east-agent"

    @pytest.mark.asyncio
    async def test_list_regions(self, registry):
        """Test listing all regions."""
        await registry.register(
            agent_id="agent-1",
            capabilities=["debate"],
            region_id="us-west-2",
        )
        await registry.register(
            agent_id="agent-2",
            capabilities=["debate"],
            region_id="us-east-1",
            available_regions=["us-east-1", "eu-west-1"],
        )

        regions = await registry.list_regions()
        assert "us-west-2" in regions
        assert "us-east-1" in regions
        assert "eu-west-1" in regions

    @pytest.mark.asyncio
    async def test_get_agents_by_region(self, registry):
        """Test grouping agents by region."""
        await registry.register(
            agent_id="west-1",
            capabilities=["debate"],
            region_id="us-west-2",
        )
        await registry.register(
            agent_id="west-2",
            capabilities=["code"],
            region_id="us-west-2",
        )
        await registry.register(
            agent_id="east-1",
            capabilities=["debate"],
            region_id="us-east-1",
        )

        by_region = await registry.get_agents_by_region()

        assert "us-west-2" in by_region
        assert len(by_region["us-west-2"]) == 2
        assert "us-east-1" in by_region
        assert len(by_region["us-east-1"]) == 1
