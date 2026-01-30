"""
Tests for Agent Registry.

Tests cover:
- AgentStatus enum
- AgentCapability enum
- AgentInfo dataclass (properties, methods, serialization)
- AgentRegistry (in-memory mode):
  - Registration and unregistration
  - Heartbeat tracking
  - Capability-based lookups
  - Region-based lookups
  - Agent selection strategies
  - Task completion recording
  - Statistics
  - Cleanup of stale agents
  - Status updates

Run with:
    pytest tests/control_plane/test_registry.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest

from aragora.control_plane.registry import (
    AgentCapability,
    AgentInfo,
    AgentRegistry,
    AgentStatus,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_all_statuses(self):
        """Test all expected status values."""
        assert AgentStatus.STARTING.value == "starting"
        assert AgentStatus.READY.value == "ready"
        assert AgentStatus.BUSY.value == "busy"
        assert AgentStatus.DRAINING.value == "draining"
        assert AgentStatus.OFFLINE.value == "offline"
        assert AgentStatus.FAILED.value == "failed"

    def test_status_count(self):
        """Test expected number of statuses."""
        assert len(AgentStatus) == 6


class TestAgentCapability:
    """Tests for AgentCapability enum."""

    def test_all_capabilities(self):
        """Test all expected capability values."""
        assert AgentCapability.DEBATE.value == "debate"
        assert AgentCapability.CODE.value == "code"
        assert AgentCapability.ANALYSIS.value == "analysis"
        assert AgentCapability.CRITIQUE.value == "critique"
        assert AgentCapability.JUDGE.value == "judge"
        assert AgentCapability.IMPLEMENT.value == "implement"
        assert AgentCapability.DESIGN.value == "design"
        assert AgentCapability.RESEARCH.value == "research"
        assert AgentCapability.AUDIT.value == "audit"
        assert AgentCapability.SUMMARIZE.value == "summarize"

    def test_capability_count(self):
        """Test expected number of capabilities."""
        assert len(AgentCapability) == 10


# =============================================================================
# AgentInfo Tests
# =============================================================================


class TestAgentInfoCreation:
    """Tests for AgentInfo creation."""

    def test_basic_creation(self):
        """Test creating a basic AgentInfo."""
        agent = AgentInfo(
            agent_id="test-agent",
            capabilities={"debate", "code"},
        )
        assert agent.agent_id == "test-agent"
        assert agent.capabilities == {"debate", "code"}
        assert agent.status == AgentStatus.STARTING
        assert agent.model == "unknown"
        assert agent.provider == "unknown"

    def test_full_creation(self):
        """Test creating an AgentInfo with all fields."""
        agent = AgentInfo(
            agent_id="claude-3",
            capabilities={"debate", "critique"},
            status=AgentStatus.READY,
            model="claude-3-opus",
            provider="anthropic",
            metadata={"tier": "premium"},
            tasks_completed=10,
            tasks_failed=1,
            avg_latency_ms=500.0,
            tags={"production", "fast"},
            region_id="us-west-2",
            available_regions={"us-west-2", "us-east-1"},
        )
        assert agent.model == "claude-3-opus"
        assert agent.provider == "anthropic"
        assert agent.tasks_completed == 10
        assert agent.tasks_failed == 1
        assert agent.avg_latency_ms == 500.0
        assert "production" in agent.tags
        assert agent.region_id == "us-west-2"

    def test_default_region(self):
        """Test default region values."""
        agent = AgentInfo(agent_id="a", capabilities=set())
        assert agent.region_id == "default"
        assert "default" in agent.available_regions


class TestAgentInfoMethods:
    """Tests for AgentInfo methods."""

    def test_is_available_ready(self):
        """Test is_available when READY."""
        agent = AgentInfo(agent_id="a", capabilities=set(), status=AgentStatus.READY)
        assert agent.is_available() is True

    def test_is_available_busy(self):
        """Test is_available when BUSY."""
        agent = AgentInfo(agent_id="a", capabilities=set(), status=AgentStatus.BUSY)
        assert agent.is_available() is False

    def test_is_available_offline(self):
        """Test is_available when OFFLINE."""
        agent = AgentInfo(agent_id="a", capabilities=set(), status=AgentStatus.OFFLINE)
        assert agent.is_available() is False

    def test_is_alive_recent_heartbeat(self):
        """Test is_alive with recent heartbeat."""
        agent = AgentInfo(
            agent_id="a",
            capabilities=set(),
            last_heartbeat=time.time(),
        )
        assert agent.is_alive(timeout_seconds=30.0) is True

    def test_is_alive_stale_heartbeat(self):
        """Test is_alive with stale heartbeat."""
        agent = AgentInfo(
            agent_id="a",
            capabilities=set(),
            last_heartbeat=time.time() - 60.0,
        )
        assert agent.is_alive(timeout_seconds=30.0) is False

    def test_has_capability_string(self):
        """Test has_capability with string."""
        agent = AgentInfo(agent_id="a", capabilities={"debate", "code"})
        assert agent.has_capability("debate") is True
        assert agent.has_capability("judge") is False

    def test_has_capability_enum(self):
        """Test has_capability with enum."""
        agent = AgentInfo(agent_id="a", capabilities={"debate"})
        assert agent.has_capability(AgentCapability.DEBATE) is True
        assert agent.has_capability(AgentCapability.CODE) is False

    def test_has_all_capabilities(self):
        """Test has_all_capabilities."""
        agent = AgentInfo(agent_id="a", capabilities={"debate", "code", "critique"})
        assert agent.has_all_capabilities(["debate", "code"]) is True
        assert agent.has_all_capabilities(["debate", "judge"]) is False

    def test_has_all_capabilities_with_enums(self):
        """Test has_all_capabilities with enum values."""
        agent = AgentInfo(agent_id="a", capabilities={"debate", "code"})
        assert agent.has_all_capabilities([AgentCapability.DEBATE, AgentCapability.CODE]) is True

    def test_is_available_in_region(self):
        """Test is_available_in_region."""
        agent = AgentInfo(
            agent_id="a",
            capabilities=set(),
            available_regions={"us-west-2", "us-east-1"},
        )
        assert agent.is_available_in_region("us-west-2") is True
        assert agent.is_available_in_region("eu-west-1") is False

    def test_get_latency_for_region_known(self):
        """Test get_latency_for_region with known region."""
        agent = AgentInfo(
            agent_id="a",
            capabilities=set(),
            available_regions={"us-west-2"},
            region_latency_ms={"us-west-2": 50.0},
        )
        assert agent.get_latency_for_region("us-west-2") == 50.0

    def test_get_latency_for_region_unknown(self):
        """Test get_latency_for_region with unavailable region."""
        agent = AgentInfo(
            agent_id="a",
            capabilities=set(),
            available_regions={"us-west-2"},
        )
        assert agent.get_latency_for_region("eu-west-1") == float("inf")

    def test_get_latency_for_region_fallback_to_avg(self):
        """Test get_latency_for_region falls back to avg_latency_ms."""
        agent = AgentInfo(
            agent_id="a",
            capabilities=set(),
            available_regions={"us-west-2"},
            avg_latency_ms=100.0,
        )
        assert agent.get_latency_for_region("us-west-2") == 100.0

    def test_is_alive_in_region(self):
        """Test is_alive_in_region."""
        now = time.time()
        agent = AgentInfo(
            agent_id="a",
            capabilities=set(),
            last_heartbeat_by_region={"us-west-2": now, "eu-west-1": now - 60},
        )
        assert agent.is_alive_in_region("us-west-2") is True
        assert agent.is_alive_in_region("eu-west-1", timeout_seconds=30.0) is False
        assert agent.is_alive_in_region("unknown-region") is False

    def test_update_region_heartbeat(self):
        """Test update_region_heartbeat."""
        agent = AgentInfo(agent_id="a", capabilities=set())
        old_hb = agent.last_heartbeat

        time.sleep(0.01)
        agent.update_region_heartbeat("us-west-2")

        assert agent.last_heartbeat > old_hb
        assert "us-west-2" in agent.last_heartbeat_by_region
        assert agent.last_heartbeat_by_region["us-west-2"] >= old_hb


class TestAgentInfoSerialization:
    """Tests for AgentInfo serialization."""

    def test_to_dict(self):
        """Test serialization to dict."""
        agent = AgentInfo(
            agent_id="claude-3",
            capabilities={"debate", "code"},
            status=AgentStatus.READY,
            model="claude-3-opus",
            provider="anthropic",
            region_id="us-west-2",
            available_regions={"us-west-2", "us-east-1"},
            tags={"production"},
        )
        d = agent.to_dict()

        assert d["agent_id"] == "claude-3"
        assert set(d["capabilities"]) == {"debate", "code"}
        assert d["status"] == "ready"
        assert d["model"] == "claude-3-opus"
        assert d["provider"] == "anthropic"
        assert d["region_id"] == "us-west-2"
        assert set(d["available_regions"]) == {"us-west-2", "us-east-1"}
        assert "production" in d["tags"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "agent_id": "gpt-4",
            "capabilities": ["debate", "analysis"],
            "status": "ready",
            "model": "gpt-4",
            "provider": "openai",
            "tasks_completed": 5,
            "region_id": "eu-west-1",
            "available_regions": ["eu-west-1"],
        }
        agent = AgentInfo.from_dict(data)

        assert agent.agent_id == "gpt-4"
        assert agent.capabilities == {"debate", "analysis"}
        assert agent.status == AgentStatus.READY
        assert agent.model == "gpt-4"
        assert agent.tasks_completed == 5
        assert agent.region_id == "eu-west-1"

    def test_round_trip(self):
        """Test to_dict then from_dict preserves data."""
        original = AgentInfo(
            agent_id="test",
            capabilities={"debate", "code"},
            status=AgentStatus.BUSY,
            model="claude-3-opus",
            provider="anthropic",
            tasks_completed=42,
            tasks_failed=3,
            avg_latency_ms=250.0,
            region_id="us-west-2",
            available_regions={"us-west-2", "us-east-1"},
            tags={"fast"},
            metadata={"tier": "premium"},
        )

        restored = AgentInfo.from_dict(original.to_dict())

        assert restored.agent_id == original.agent_id
        assert restored.capabilities == original.capabilities
        assert restored.status == original.status
        assert restored.model == original.model
        assert restored.provider == original.provider
        assert restored.tasks_completed == original.tasks_completed
        assert restored.tasks_failed == original.tasks_failed
        assert restored.avg_latency_ms == original.avg_latency_ms
        assert restored.region_id == original.region_id
        assert restored.available_regions == original.available_regions


# =============================================================================
# AgentRegistry Tests (In-Memory Mode)
# =============================================================================


@pytest.fixture
def registry() -> AgentRegistry:
    """Create a registry in in-memory mode."""
    return AgentRegistry(redis_url="memory://")


@pytest.fixture
async def connected_registry(registry: AgentRegistry) -> AgentRegistry:
    """Create and connect a registry."""
    await registry.connect()
    yield registry
    await registry.close()


class TestRegistryConnect:
    """Tests for registry connection."""

    @pytest.mark.asyncio
    async def test_connect_memory_mode(self):
        """Test connecting in memory mode."""
        registry = AgentRegistry(redis_url="memory://")
        await registry.connect()
        assert registry._redis is None
        await registry.close()

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        """Test that close is safe to call multiple times."""
        registry = AgentRegistry(redis_url="memory://")
        await registry.connect()
        await registry.close()
        await registry.close()  # Should not raise


class TestRegistryRegistration:
    """Tests for agent registration."""

    @pytest.mark.asyncio
    async def test_register_agent(self, connected_registry: AgentRegistry):
        """Test basic agent registration."""
        agent = await connected_registry.register(
            agent_id="claude-3",
            capabilities=["debate", "code"],
            model="claude-3-opus",
            provider="anthropic",
        )

        assert agent.agent_id == "claude-3"
        assert agent.capabilities == {"debate", "code"}
        assert agent.model == "claude-3-opus"
        assert agent.provider == "anthropic"
        assert agent.status == AgentStatus.READY

    @pytest.mark.asyncio
    async def test_register_with_enums(self, connected_registry: AgentRegistry):
        """Test registration with AgentCapability enums."""
        agent = await connected_registry.register(
            agent_id="enum-agent",
            capabilities=[AgentCapability.DEBATE, AgentCapability.CRITIQUE],
        )
        assert agent.capabilities == {"debate", "critique"}

    @pytest.mark.asyncio
    async def test_register_with_tags(self, connected_registry: AgentRegistry):
        """Test registration with tags."""
        agent = await connected_registry.register(
            agent_id="tagged-agent",
            capabilities=["debate"],
            tags=["production", "fast"],
        )
        assert "production" in agent.tags
        assert "fast" in agent.tags

    @pytest.mark.asyncio
    async def test_register_with_region(self, connected_registry: AgentRegistry):
        """Test registration with region info."""
        agent = await connected_registry.register(
            agent_id="regional-agent",
            capabilities=["debate"],
            region_id="us-west-2",
            available_regions=["us-west-2", "us-east-1"],
        )
        assert agent.region_id == "us-west-2"
        assert agent.available_regions == {"us-west-2", "us-east-1"}

    @pytest.mark.asyncio
    async def test_register_region_auto_includes_primary(self, connected_registry: AgentRegistry):
        """Test that primary region is always in available_regions."""
        agent = await connected_registry.register(
            agent_id="primary-region",
            capabilities=["debate"],
            region_id="us-west-2",
            available_regions=["us-east-1"],  # Doesn't include primary
        )
        assert "us-west-2" in agent.available_regions  # Auto-added
        assert "us-east-1" in agent.available_regions

    @pytest.mark.asyncio
    async def test_register_default_region(self, connected_registry: AgentRegistry):
        """Test default region assignment."""
        agent = await connected_registry.register(
            agent_id="default-region",
            capabilities=["debate"],
        )
        assert agent.region_id == "default"
        assert "default" in agent.available_regions

    @pytest.mark.asyncio
    async def test_register_and_get(self, connected_registry: AgentRegistry):
        """Test register then get."""
        await connected_registry.register(
            agent_id="get-test",
            capabilities=["debate"],
            model="gpt-4",
        )
        agent = await connected_registry.get("get-test")

        assert agent is not None
        assert agent.agent_id == "get-test"
        assert agent.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, connected_registry: AgentRegistry):
        """Test getting a non-existent agent."""
        agent = await connected_registry.get("nonexistent")
        assert agent is None


class TestRegistryUnregister:
    """Tests for agent unregistration."""

    @pytest.mark.asyncio
    async def test_unregister_existing(self, connected_registry: AgentRegistry):
        """Test unregistering an existing agent."""
        await connected_registry.register(agent_id="unreg-test", capabilities=["debate"])
        result = await connected_registry.unregister("unreg-test")
        assert result is True

        agent = await connected_registry.get("unreg-test")
        assert agent is None

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self, connected_registry: AgentRegistry):
        """Test unregistering a non-existent agent."""
        result = await connected_registry.unregister("nonexistent")
        assert result is False


class TestRegistryHeartbeat:
    """Tests for agent heartbeats."""

    @pytest.mark.asyncio
    async def test_heartbeat_updates_timestamp(self, connected_registry: AgentRegistry):
        """Test that heartbeat updates last_heartbeat."""
        await connected_registry.register(agent_id="hb-test", capabilities=["debate"])

        before = (await connected_registry.get("hb-test")).last_heartbeat
        await asyncio.sleep(0.01)
        result = await connected_registry.heartbeat("hb-test")

        assert result is True
        after = (await connected_registry.get("hb-test")).last_heartbeat
        assert after > before

    @pytest.mark.asyncio
    async def test_heartbeat_with_status_update(self, connected_registry: AgentRegistry):
        """Test heartbeat with status change."""
        await connected_registry.register(agent_id="status-hb", capabilities=["debate"])
        await connected_registry.heartbeat("status-hb", status=AgentStatus.BUSY)

        agent = await connected_registry.get("status-hb")
        assert agent.status == AgentStatus.BUSY

    @pytest.mark.asyncio
    async def test_heartbeat_with_task_id(self, connected_registry: AgentRegistry):
        """Test heartbeat with current task ID."""
        await connected_registry.register(agent_id="task-hb", capabilities=["debate"])
        await connected_registry.heartbeat("task-hb", current_task_id="task-123")

        agent = await connected_registry.get("task-hb")
        assert agent.current_task_id == "task-123"

    @pytest.mark.asyncio
    async def test_heartbeat_nonexistent(self, connected_registry: AgentRegistry):
        """Test heartbeat for non-existent agent."""
        result = await connected_registry.heartbeat("nonexistent")
        assert result is False


class TestRegistryListAll:
    """Tests for listing agents."""

    @pytest.mark.asyncio
    async def test_list_empty(self, connected_registry: AgentRegistry):
        """Test listing from empty registry."""
        agents = await connected_registry.list_all()
        assert agents == []

    @pytest.mark.asyncio
    async def test_list_all_agents(self, connected_registry: AgentRegistry):
        """Test listing all registered agents."""
        await connected_registry.register(agent_id="a1", capabilities=["debate"])
        await connected_registry.register(agent_id="a2", capabilities=["code"])

        agents = await connected_registry.list_all()
        assert len(agents) == 2
        ids = {a.agent_id for a in agents}
        assert ids == {"a1", "a2"}

    @pytest.mark.asyncio
    async def test_list_excludes_stale_by_default(self, connected_registry: AgentRegistry):
        """Test that stale agents are excluded by default."""
        # Register agent with old heartbeat
        agent = AgentInfo(
            agent_id="stale",
            capabilities=set(),
            status=AgentStatus.READY,
            last_heartbeat=time.time() - 120,  # Way past timeout
        )
        connected_registry._local_cache["stale"] = agent

        agents = await connected_registry.list_all(include_offline=False)
        assert len(agents) == 0

    @pytest.mark.asyncio
    async def test_list_includes_stale_when_requested(self, connected_registry: AgentRegistry):
        """Test listing includes stale agents when requested."""
        agent = AgentInfo(
            agent_id="stale",
            capabilities=set(),
            status=AgentStatus.READY,
            last_heartbeat=time.time() - 120,
        )
        connected_registry._local_cache["stale"] = agent

        agents = await connected_registry.list_all(include_offline=True)
        assert len(agents) == 1


class TestRegistryFindByCapability:
    """Tests for capability-based lookups."""

    @pytest.mark.asyncio
    async def _setup_agents(self, registry: AgentRegistry):
        """Helper to register test agents."""
        await registry.register(agent_id="claude", capabilities=["debate", "code", "critique"])
        await registry.register(agent_id="gpt4", capabilities=["debate", "analysis"])
        await registry.register(agent_id="gemini", capabilities=["debate", "code"])

    @pytest.mark.asyncio
    async def test_find_by_single_capability(self, connected_registry: AgentRegistry):
        """Test finding agents by single capability."""
        await self._setup_agents(connected_registry)

        debate_agents = await connected_registry.find_by_capability("debate")
        assert len(debate_agents) == 3

        code_agents = await connected_registry.find_by_capability("code")
        assert len(code_agents) == 2

        critique_agents = await connected_registry.find_by_capability("critique")
        assert len(critique_agents) == 1

    @pytest.mark.asyncio
    async def test_find_by_capability_enum(self, connected_registry: AgentRegistry):
        """Test finding agents by capability enum."""
        await self._setup_agents(connected_registry)

        agents = await connected_registry.find_by_capability(AgentCapability.DEBATE)
        assert len(agents) == 3

    @pytest.mark.asyncio
    async def test_find_by_capability_only_available(self, connected_registry: AgentRegistry):
        """Test that only_available filters BUSY agents."""
        await self._setup_agents(connected_registry)
        await connected_registry.update_status("claude", AgentStatus.BUSY)

        available = await connected_registry.find_by_capability("debate", only_available=True)
        assert len(available) == 2

        all_agents = await connected_registry.find_by_capability("debate", only_available=False)
        assert len(all_agents) == 3

    @pytest.mark.asyncio
    async def test_find_by_capabilities_multiple(self, connected_registry: AgentRegistry):
        """Test finding agents with multiple capabilities."""
        await self._setup_agents(connected_registry)

        agents = await connected_registry.find_by_capabilities(["debate", "code"])
        assert len(agents) == 2
        ids = {a.agent_id for a in agents}
        assert ids == {"claude", "gemini"}

    @pytest.mark.asyncio
    async def test_find_by_nonexistent_capability(self, connected_registry: AgentRegistry):
        """Test finding agents with capability no one has."""
        await self._setup_agents(connected_registry)

        agents = await connected_registry.find_by_capability("teleportation")
        assert len(agents) == 0


class TestRegistryFindByRegion:
    """Tests for region-based lookups."""

    @pytest.mark.asyncio
    async def _setup_regional_agents(self, registry: AgentRegistry):
        """Helper to register agents in different regions."""
        await registry.register(
            agent_id="west-1",
            capabilities=["debate"],
            region_id="us-west-2",
            available_regions=["us-west-2"],
        )
        await registry.register(
            agent_id="east-1",
            capabilities=["debate", "code"],
            region_id="us-east-1",
            available_regions=["us-east-1"],
        )
        await registry.register(
            agent_id="multi",
            capabilities=["debate"],
            region_id="us-west-2",
            available_regions=["us-west-2", "us-east-1", "eu-west-1"],
        )

    @pytest.mark.asyncio
    async def test_find_by_region(self, connected_registry: AgentRegistry):
        """Test finding agents by region."""
        await self._setup_regional_agents(connected_registry)

        west_agents = await connected_registry.find_by_region("us-west-2")
        assert len(west_agents) == 2  # west-1 and multi

        east_agents = await connected_registry.find_by_region("us-east-1")
        assert len(east_agents) == 2  # east-1 and multi

        eu_agents = await connected_registry.find_by_region("eu-west-1")
        assert len(eu_agents) == 1  # only multi

    @pytest.mark.asyncio
    async def test_find_by_capability_and_region(self, connected_registry: AgentRegistry):
        """Test finding agents by both capability and region."""
        await self._setup_regional_agents(connected_registry)

        agents = await connected_registry.find_by_capability_and_region("code", "us-east-1")
        assert len(agents) == 1
        assert agents[0].agent_id == "east-1"

    @pytest.mark.asyncio
    async def test_find_by_capabilities_and_regions(self, connected_registry: AgentRegistry):
        """Test finding agents with capabilities across regions."""
        await self._setup_regional_agents(connected_registry)

        agents = await connected_registry.find_by_capabilities_and_regions(
            ["debate"],
            ["us-west-2", "us-east-1"],
        )
        assert len(agents) == 3

    @pytest.mark.asyncio
    async def test_list_regions(self, connected_registry: AgentRegistry):
        """Test listing all regions."""
        await self._setup_regional_agents(connected_registry)

        regions = await connected_registry.list_regions()
        assert "us-west-2" in regions
        assert "us-east-1" in regions
        assert "eu-west-1" in regions

    @pytest.mark.asyncio
    async def test_get_agents_by_region(self, connected_registry: AgentRegistry):
        """Test getting agents grouped by region."""
        await self._setup_regional_agents(connected_registry)

        by_region = await connected_registry.get_agents_by_region()
        assert "us-west-2" in by_region
        assert len(by_region["us-west-2"]) == 2  # west-1 and multi
        assert "us-east-1" in by_region
        assert len(by_region["us-east-1"]) == 1  # east-1 only (primary)


class TestRegistrySelectAgent:
    """Tests for agent selection strategies."""

    @pytest.mark.asyncio
    async def _setup_load_agents(self, registry: AgentRegistry):
        """Helper to register agents with varying loads."""
        a1 = await registry.register(
            agent_id="light",
            capabilities=["debate"],
        )
        a1.tasks_completed = 5
        await registry._save_agent(a1)

        a2 = await registry.register(
            agent_id="heavy",
            capabilities=["debate"],
        )
        a2.tasks_completed = 50
        await registry._save_agent(a2)

        a3 = await registry.register(
            agent_id="medium",
            capabilities=["debate"],
        )
        a3.tasks_completed = 25
        await registry._save_agent(a3)

    @pytest.mark.asyncio
    async def test_select_least_loaded(self, connected_registry: AgentRegistry):
        """Test least loaded selection strategy."""
        await self._setup_load_agents(connected_registry)

        agent = await connected_registry.select_agent(["debate"], strategy="least_loaded")
        assert agent is not None
        assert agent.agent_id == "light"

    @pytest.mark.asyncio
    async def test_select_with_exclude(self, connected_registry: AgentRegistry):
        """Test selection with exclusions."""
        await self._setup_load_agents(connected_registry)

        agent = await connected_registry.select_agent(
            ["debate"], strategy="least_loaded", exclude=["light"]
        )
        assert agent is not None
        assert agent.agent_id == "medium"

    @pytest.mark.asyncio
    async def test_select_no_candidates(self, connected_registry: AgentRegistry):
        """Test selection with no matching agents."""
        agent = await connected_registry.select_agent(["teleportation"])
        assert agent is None

    @pytest.mark.asyncio
    async def test_select_random(self, connected_registry: AgentRegistry):
        """Test random selection strategy."""
        await self._setup_load_agents(connected_registry)

        agent = await connected_registry.select_agent(["debate"], strategy="random")
        assert agent is not None
        assert agent.agent_id in {"light", "heavy", "medium"}

    @pytest.mark.asyncio
    async def test_select_round_robin(self, connected_registry: AgentRegistry):
        """Test round robin selection strategy."""
        await self._setup_load_agents(connected_registry)

        agent = await connected_registry.select_agent(["debate"], strategy="round_robin")
        assert agent is not None

    @pytest.mark.asyncio
    async def test_select_default_strategy(self, connected_registry: AgentRegistry):
        """Test default (unknown) strategy returns first candidate."""
        await self._setup_load_agents(connected_registry)

        agent = await connected_registry.select_agent(["debate"], strategy="unknown_strategy")
        assert agent is not None


class TestRegistrySelectAgentInRegion:
    """Tests for region-aware agent selection."""

    @pytest.mark.asyncio
    async def _setup_regional_load_agents(self, registry: AgentRegistry):
        """Helper for regional selection tests."""
        a1 = await registry.register(
            agent_id="west-light",
            capabilities=["debate"],
            region_id="us-west-2",
        )
        a1.tasks_completed = 5
        await registry._save_agent(a1)

        a2 = await registry.register(
            agent_id="east-heavy",
            capabilities=["debate"],
            region_id="us-east-1",
        )
        a2.tasks_completed = 50
        await registry._save_agent(a2)

        a3 = await registry.register(
            agent_id="multi-medium",
            capabilities=["debate"],
            region_id="us-west-2",
            available_regions=["us-west-2", "us-east-1"],
        )
        a3.tasks_completed = 25
        a3.region_latency_ms = {"us-west-2": 10.0, "us-east-1": 50.0}
        await registry._save_agent(a3)

    @pytest.mark.asyncio
    async def test_select_prefers_target_region(self, connected_registry: AgentRegistry):
        """Test that selection prefers target region."""
        await self._setup_regional_load_agents(connected_registry)

        agent = await connected_registry.select_agent_in_region(
            capabilities=["debate"],
            target_region="us-west-2",
        )
        assert agent is not None
        assert agent.is_available_in_region("us-west-2")

    @pytest.mark.asyncio
    async def test_select_fallback_region(self, connected_registry: AgentRegistry):
        """Test fallback to other regions."""
        await self._setup_regional_load_agents(connected_registry)

        agent = await connected_registry.select_agent_in_region(
            capabilities=["debate"],
            target_region="eu-west-1",  # No agents here
            fallback_regions=["us-west-2"],
        )
        assert agent is not None

    @pytest.mark.asyncio
    async def test_select_no_region_constraint(self, connected_registry: AgentRegistry):
        """Test selection without region constraint."""
        await self._setup_regional_load_agents(connected_registry)

        agent = await connected_registry.select_agent_in_region(
            capabilities=["debate"],
        )
        assert agent is not None


class TestRegistryUpdateStatus:
    """Tests for updating agent status."""

    @pytest.mark.asyncio
    async def test_update_status(self, connected_registry: AgentRegistry):
        """Test updating agent status."""
        await connected_registry.register(agent_id="status-test", capabilities=["debate"])
        result = await connected_registry.update_status("status-test", AgentStatus.BUSY)
        assert result is True

        agent = await connected_registry.get("status-test")
        assert agent.status == AgentStatus.BUSY

    @pytest.mark.asyncio
    async def test_update_status_nonexistent(self, connected_registry: AgentRegistry):
        """Test updating status of non-existent agent."""
        result = await connected_registry.update_status("nonexistent", AgentStatus.BUSY)
        assert result is False


class TestRegistryRecordTaskCompletion:
    """Tests for recording task completion."""

    @pytest.mark.asyncio
    async def test_record_success(self, connected_registry: AgentRegistry):
        """Test recording successful task completion."""
        await connected_registry.register(agent_id="task-agent", capabilities=["debate"])
        await connected_registry.heartbeat(
            "task-agent",
            status=AgentStatus.BUSY,
            current_task_id="task-1",
        )

        result = await connected_registry.record_task_completion(
            "task-agent", success=True, latency_ms=500.0
        )
        assert result is True

        agent = await connected_registry.get("task-agent")
        assert agent.tasks_completed == 1
        assert agent.tasks_failed == 0
        assert agent.avg_latency_ms == 500.0
        assert agent.current_task_id is None
        assert agent.status == AgentStatus.READY

    @pytest.mark.asyncio
    async def test_record_failure(self, connected_registry: AgentRegistry):
        """Test recording failed task completion."""
        await connected_registry.register(agent_id="fail-agent", capabilities=["debate"])

        result = await connected_registry.record_task_completion(
            "fail-agent", success=False, latency_ms=1000.0
        )
        assert result is True

        agent = await connected_registry.get("fail-agent")
        assert agent.tasks_completed == 0
        assert agent.tasks_failed == 1

    @pytest.mark.asyncio
    async def test_record_rolling_average(self, connected_registry: AgentRegistry):
        """Test that average latency is computed correctly."""
        await connected_registry.register(agent_id="avg-agent", capabilities=["debate"])

        await connected_registry.record_task_completion("avg-agent", success=True, latency_ms=100.0)
        await connected_registry.record_task_completion("avg-agent", success=True, latency_ms=300.0)

        agent = await connected_registry.get("avg-agent")
        assert agent.tasks_completed == 2
        assert agent.avg_latency_ms == pytest.approx(200.0, abs=1.0)

    @pytest.mark.asyncio
    async def test_record_nonexistent_agent(self, connected_registry: AgentRegistry):
        """Test recording for non-existent agent."""
        result = await connected_registry.record_task_completion(
            "nonexistent", success=True, latency_ms=100.0
        )
        assert result is False


class TestRegistryStats:
    """Tests for registry statistics."""

    @pytest.mark.asyncio
    async def test_empty_stats(self, connected_registry: AgentRegistry):
        """Test stats for empty registry."""
        stats = await connected_registry.get_stats()
        assert stats["total_agents"] == 0
        assert stats["available_agents"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_agents(self, connected_registry: AgentRegistry):
        """Test stats with registered agents."""
        await connected_registry.register(
            agent_id="a1",
            capabilities=["debate", "code"],
            model="claude-3",
            provider="anthropic",
        )
        await connected_registry.register(
            agent_id="a2",
            capabilities=["debate"],
            model="gpt-4",
            provider="openai",
        )

        # Make one busy
        await connected_registry.update_status("a2", AgentStatus.BUSY)

        stats = await connected_registry.get_stats()
        assert stats["total_agents"] == 2
        assert stats["available_agents"] == 1
        assert stats["by_status"]["ready"] == 1
        assert stats["by_status"]["busy"] == 1
        assert stats["by_capability"]["debate"] == 2
        assert stats["by_capability"]["code"] == 1
        assert stats["by_provider"]["anthropic"] == 1
        assert stats["by_provider"]["openai"] == 1


class TestRegistryCleanup:
    """Tests for stale agent cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_agents(self, connected_registry: AgentRegistry):
        """Test that stale agents are marked offline."""
        # Register agent with old heartbeat
        agent = AgentInfo(
            agent_id="stale-agent",
            capabilities={"debate"},
            status=AgentStatus.READY,
            last_heartbeat=time.time() - 120.0,
        )
        connected_registry._local_cache["stale-agent"] = agent

        count = await connected_registry._cleanup_stale_agents()
        assert count == 1

        cleaned = await connected_registry.get("stale-agent")
        assert cleaned.status == AgentStatus.OFFLINE

    @pytest.mark.asyncio
    async def test_cleanup_skips_already_offline(self, connected_registry: AgentRegistry):
        """Test that already offline agents aren't re-marked."""
        agent = AgentInfo(
            agent_id="already-offline",
            capabilities={"debate"},
            status=AgentStatus.OFFLINE,
            last_heartbeat=time.time() - 120.0,
        )
        connected_registry._local_cache["already-offline"] = agent

        count = await connected_registry._cleanup_stale_agents()
        assert count == 0

    @pytest.mark.asyncio
    async def test_cleanup_keeps_alive_agents(self, connected_registry: AgentRegistry):
        """Test that alive agents are not marked offline."""
        await connected_registry.register(agent_id="alive", capabilities=["debate"])

        count = await connected_registry._cleanup_stale_agents()
        assert count == 0

        agent = await connected_registry.get("alive")
        assert agent.status == AgentStatus.READY


# =============================================================================
# Integration Tests
# =============================================================================


class TestRegistryIntegration:
    """Integration tests combining multiple operations."""

    @pytest.mark.asyncio
    async def test_full_agent_lifecycle(self, connected_registry: AgentRegistry):
        """Test complete agent lifecycle."""
        # Register
        await connected_registry.register(
            agent_id="lifecycle",
            capabilities=["debate", "code"],
            model="claude-3-opus",
            provider="anthropic",
        )

        # Heartbeat
        await connected_registry.heartbeat("lifecycle")

        # Assign task
        await connected_registry.heartbeat(
            "lifecycle",
            status=AgentStatus.BUSY,
            current_task_id="task-1",
        )

        agent = await connected_registry.get("lifecycle")
        assert agent.status == AgentStatus.BUSY
        assert agent.current_task_id == "task-1"

        # Complete task
        await connected_registry.record_task_completion("lifecycle", success=True, latency_ms=300.0)

        agent = await connected_registry.get("lifecycle")
        assert agent.status == AgentStatus.READY
        assert agent.current_task_id is None
        assert agent.tasks_completed == 1

        # Unregister
        result = await connected_registry.unregister("lifecycle")
        assert result is True
        assert await connected_registry.get("lifecycle") is None

    @pytest.mark.asyncio
    async def test_multi_agent_selection(self, connected_registry: AgentRegistry):
        """Test selecting from multiple agents."""
        for i in range(5):
            agent = await connected_registry.register(
                agent_id=f"agent-{i}",
                capabilities=["debate"],
            )
            # Simulate varying load
            for _ in range(i):
                await connected_registry.record_task_completion(
                    f"agent-{i}", success=True, latency_ms=100.0
                )

        # Least loaded should be agent-0
        selected = await connected_registry.select_agent(["debate"], strategy="least_loaded")
        assert selected.agent_id == "agent-0"

        # Excluding agent-0, should get agent-1
        selected = await connected_registry.select_agent(
            ["debate"],
            strategy="least_loaded",
            exclude=["agent-0"],
        )
        assert selected.agent_id == "agent-1"
