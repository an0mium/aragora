"""
Tests for sparse communication topologies in debates.
"""

import pytest
from unittest.mock import Mock

from aragora.core import Agent, Environment
from aragora.debate.orchestrator import Arena, DebateProtocol


class TestTopology:
    """Test different debate topologies reduce communication."""

    def test_all_to_all_topology(self):
        """Test all-to-all topology includes all possible critiques."""
        agents = []
        for i in range(4):
            agent = Mock(spec=Agent)
            agent.name = f"agent{i}"
            agents.append(agent)

        protocol = DebateProtocol(topology="all-to-all")
        env = Environment(task="test")

        # Create arena with minimal mocking
        arena = Arena.__new__(Arena)  # Create without __init__
        arena.protocol = protocol
        arena.agents = agents

        # Test selection for agent0
        critics = arena._select_critics_for_proposal("agent0", agents)
        assert len(critics) == 3  # all except self
        assert all(c.name != "agent0" for c in critics)

    def test_ring_topology(self):
        """Test ring topology connects neighbors."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(4)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        protocol = DebateProtocol(topology="ring")
        env = Environment(task="test")
        arena = Arena.__new__(Arena)
        arena.protocol = protocol
        arena.agents = agents

        # In a ring of 4, agent0 should be critiqued by agent3 and agent1
        critics = arena._select_critics_for_proposal("agent0", agents)
        critic_names = {c.name for c in critics}
        assert critic_names == {"agent3", "agent1"}

    def test_star_topology(self):
        """Test star topology with hub agent."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(4)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        protocol = DebateProtocol(topology="star", topology_hub_agent="agent0")
        env = Environment(task="test")
        arena = Arena.__new__(Arena)
        arena.protocol = protocol
        arena.agents = agents

        # Hub's proposal gets critiqued by all others
        critics = arena._select_critics_for_proposal("agent0", agents)
        assert len(critics) == 3
        assert all(c.name != "agent0" for c in critics)

        # Other proposals get critiqued only by hub
        critics = arena._select_critics_for_proposal("agent1", agents)
        assert len(critics) == 1
        assert critics[0].name == "agent0"

    def test_sparse_topology(self):
        """Test sparse topology selects subset."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(6)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        protocol = DebateProtocol(topology="sparse", topology_sparsity=0.5)
        env = Environment(task="test")
        arena = Arena.__new__(Arena)
        arena.protocol = protocol
        arena.agents = agents

        critics = arena._select_critics_for_proposal("agent0", agents)
        # With 5 available critics, 50% sparsity should select 2-3 (max(1, int(5*0.5))=2)
        assert 1 <= len(critics) <= 3
        assert all(c.name != "agent0" for c in critics)

    def test_round_robin_topology(self):
        """Test round-robin topology assigns one critic per proposal."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(4)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        protocol = DebateProtocol(topology="round-robin")
        env = Environment(task="test")
        arena = Arena.__new__(Arena)
        arena.protocol = protocol
        arena.agents = agents

        critics = arena._select_critics_for_proposal("agent0", agents)
        assert len(critics) == 1
        assert critics[0].name != "agent0"

    def test_topology_reduces_communication(self):
        """Verify sparse topologies reduce total critiques vs all-to-all."""
        agents = [Mock(spec=Agent, name=f"agent{i}") for i in range(5)]
        for i, agent in enumerate(agents):
            agent.name = f"agent{i}"

        env = Environment(task="test")

        # All-to-all
        protocol_all = DebateProtocol(topology="all-to-all")
        arena_all = Arena.__new__(Arena)
        arena_all.protocol = protocol_all
        arena_all.agents = agents
        total_all = sum(len(arena_all._select_critics_for_proposal(a.name, agents))
                       for a in agents)

        # Sparse
        protocol_sparse = DebateProtocol(topology="sparse", topology_sparsity=0.4)
        arena_sparse = Arena.__new__(Arena)
        arena_sparse.protocol = protocol_sparse
        arena_sparse.agents = agents
        total_sparse = sum(len(arena_sparse._select_critics_for_proposal(a.name, agents))
                          for a in agents)

        assert total_sparse < total_all
        print(f"All-to-all: {total_all} critiques, Sparse: {total_sparse} critiques")