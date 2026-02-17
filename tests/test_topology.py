"""
Tests for sparse communication topologies in debates.
"""

import pytest
from unittest.mock import Mock

from aragora.core import Agent, Environment
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.debate.agent_pool import AgentPool, AgentPoolConfig


def _make_arena(agents, topology, critic_count=None):
    """Create an Arena with minimal setup for topology tests."""
    protocol = DebateProtocol(topology=topology)
    arena = Arena.__new__(Arena)
    arena.protocol = protocol
    arena.agents = agents
    pool_config = AgentPoolConfig(
        topology=topology,
        critic_count=critic_count if critic_count is not None else len(agents),
    )
    arena.agent_pool = AgentPool(agents, config=pool_config)
    return arena


class TestTopology:
    """Test different debate topologies reduce communication."""

    def test_all_to_all_topology(self):
        """Test all-to-all topology includes all possible critiques."""
        agents = []
        for i in range(4):
            agent = Mock(spec=Agent)
            agent.name = f"agent{i}"
            agents.append(agent)

        arena = _make_arena(agents, "all-to-all")

        # Test selection for agent0
        critics = arena._select_critics_for_proposal("agent0", agents)
        assert len(critics) == 3  # all except self
        assert all(c.name != "agent0" for c in critics)

    def test_ring_topology(self):
        """Test ring topology connects neighbors."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(4)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        arena = _make_arena(agents, "ring", critic_count=2)

        # In a ring of 4, agent0 should be critiqued by agent3 and agent1
        critics = arena._select_critics_for_proposal("agent0", agents)
        critic_names = {c.name for c in critics}
        assert critic_names == {"agent3", "agent1"}

    def test_star_topology(self):
        """Test star topology with hub agent (first agent is hub)."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(4)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        arena = _make_arena(agents, "star")

        # Hub's proposal gets critiqued by all others
        critics = arena._select_critics_for_proposal("agent0", agents)
        assert len(critics) == 3
        assert all(c.name != "agent0" for c in critics)

        # Other proposals get critiqued only by hub
        critics = arena._select_critics_for_proposal("agent1", agents)
        assert len(critics) == 1
        assert critics[0].name == "agent0"

    def test_sparse_topology(self):
        """Test sparse topology (full_mesh with limited critic_count) selects subset."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(6)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        # Use full_mesh with critic_count=2 to simulate sparse
        arena = _make_arena(agents, "full_mesh", critic_count=2)

        critics = arena._select_critics_for_proposal("agent0", agents)
        assert 1 <= len(critics) <= 2
        assert all(c.name != "agent0" for c in critics)

    def test_round_robin_topology(self):
        """Test round-robin topology assigns one critic per proposal."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(4)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        # full_mesh with critic_count=1 for round-robin effect
        arena = _make_arena(agents, "full_mesh", critic_count=1)

        critics = arena._select_critics_for_proposal("agent0", agents)
        assert len(critics) == 1
        assert critics[0].name != "agent0"

    def test_topology_reduces_communication(self):
        """Verify limited critic_count reduces total critiques vs all-to-all."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(5)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        arena_all = _make_arena(agents, "full_mesh", critic_count=10)
        total_all = sum(len(arena_all._select_critics_for_proposal(a.name, agents)) for a in agents)

        arena_sparse = _make_arena(agents, "full_mesh", critic_count=2)
        total_sparse = sum(
            len(arena_sparse._select_critics_for_proposal(a.name, agents)) for a in agents
        )

        assert total_sparse < total_all
        print(f"All-to-all: {total_all} critiques, Sparse: {total_sparse} critiques")
