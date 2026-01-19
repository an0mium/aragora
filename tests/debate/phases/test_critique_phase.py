"""
Tests for the CritiquePhase module.

Tests cover:
- CritiquePhase initialization
- Critic selection for different topologies (all-to-all, round-robin, ring, star, sparse)
- Edge cases with empty or single agent lists
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from aragora.debate.phases.critique import CritiquePhase
from aragora.debate.protocol import DebateProtocol


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    role: str = "critic"
    model: str = "mock-model"


class TestCritiquePhaseInit:
    """Tests for CritiquePhase initialization."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority", topology="all-to-all")

    @pytest.fixture
    def agents(self):
        return [MockAgent("agent1"), MockAgent("agent2"), MockAgent("agent3")]

    def test_init_stores_protocol(self, protocol, agents):
        """CritiquePhase stores protocol."""
        phase = CritiquePhase(protocol, agents)

        assert phase.protocol is protocol

    def test_init_stores_agents(self, protocol, agents):
        """CritiquePhase stores agents."""
        phase = CritiquePhase(protocol, agents)

        assert phase.agents == agents


class TestCritiquePhaseAllToAll:
    """Tests for all-to-all topology."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority", topology="all-to-all")

    @pytest.fixture
    def agents(self):
        return [MockAgent("agent1"), MockAgent("agent2"), MockAgent("agent3")]

    def test_all_to_all_excludes_proposer(self, protocol, agents):
        """All-to-all excludes the proposer from critics."""
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        assert MockAgent("agent1") not in critics
        assert len(critics) == 2

    def test_all_to_all_includes_all_others(self, protocol, agents):
        """All-to-all includes all non-proposer agents."""
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)
        critic_names = {c.name for c in critics}

        assert "agent2" in critic_names
        assert "agent3" in critic_names

    def test_all_to_all_with_single_agent(self, protocol):
        """All-to-all with single agent returns empty list."""
        agents = [MockAgent("agent1")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        assert critics == []


class TestCritiquePhaseRoundRobin:
    """Tests for round-robin topology."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority", topology="round-robin")

    @pytest.fixture
    def agents(self):
        return [MockAgent("agent1"), MockAgent("agent2"), MockAgent("agent3")]

    def test_round_robin_returns_single_critic(self, protocol, agents):
        """Round-robin returns exactly one critic."""
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        assert len(critics) == 1

    def test_round_robin_excludes_proposer(self, protocol, agents):
        """Round-robin excludes the proposer."""
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        assert all(c.name != "agent1" for c in critics)

    def test_round_robin_deterministic(self, protocol, agents):
        """Round-robin selection is deterministic for same proposer."""
        phase = CritiquePhase(protocol, agents)

        critics1 = phase.select_critics_for_proposal("agent1", agents)
        critics2 = phase.select_critics_for_proposal("agent1", agents)

        assert critics1 == critics2

    def test_round_robin_different_for_different_proposers(self, protocol, agents):
        """Round-robin may select different critics for different proposers."""
        phase = CritiquePhase(protocol, agents)

        # With 3 agents, different proposers get different critics
        critics1 = phase.select_critics_for_proposal("agent1", agents)
        critics2 = phase.select_critics_for_proposal("agent2", agents)
        critics3 = phase.select_critics_for_proposal("agent3", agents)

        # Each should get exactly one critic
        assert len(critics1) == 1
        assert len(critics2) == 1
        assert len(critics3) == 1


class TestCritiquePhaseRing:
    """Tests for ring topology."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority", topology="ring")

    @pytest.fixture
    def agents(self):
        return [MockAgent("agent1"), MockAgent("agent2"), MockAgent("agent3")]

    def test_ring_excludes_proposer(self, protocol, agents):
        """Ring topology excludes the proposer."""
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        assert all(c.name != "agent1" for c in critics)

    def test_ring_with_single_critic(self, protocol):
        """Ring with single agent may return that agent as critic."""
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        # Ring returns neighbors, so agent2 should be a critic
        assert len(critics) >= 1


class TestCritiquePhaseStar:
    """Tests for star topology."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority", topology="star")

    @pytest.fixture
    def agents(self):
        return [MockAgent("hub"), MockAgent("spoke1"), MockAgent("spoke2")]

    def test_star_excludes_proposer(self, protocol, agents):
        """Star topology excludes the proposer."""
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("hub", agents)

        assert all(c.name != "hub" for c in critics)


class TestCritiquePhaseSparse:
    """Tests for sparse/random-graph topology."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority", topology="sparse")

    @pytest.fixture
    def agents(self):
        return [
            MockAgent("agent1"),
            MockAgent("agent2"),
            MockAgent("agent3"),
            MockAgent("agent4"),
        ]

    def test_sparse_excludes_proposer(self, protocol, agents):
        """Sparse topology excludes the proposer."""
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        assert all(c.name != "agent1" for c in critics)

    def test_sparse_returns_subset(self, protocol, agents):
        """Sparse topology returns a subset of critics."""
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        # Should return at most all minus proposer
        assert len(critics) <= len(agents) - 1


class TestCritiquePhaseUnknownTopology:
    """Tests for unknown topology fallback."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority", topology="unknown-type")

    @pytest.fixture
    def agents(self):
        return [MockAgent("agent1"), MockAgent("agent2"), MockAgent("agent3")]

    def test_unknown_falls_back_to_all_to_all(self, protocol, agents):
        """Unknown topology falls back to all-to-all."""
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        # Should behave like all-to-all
        assert len(critics) == 2
        assert all(c.name != "agent1" for c in critics)


class TestCritiquePhaseEdgeCases:
    """Edge case tests for CritiquePhase."""

    @pytest.fixture
    def protocol(self):
        return DebateProtocol(rounds=2, consensus="majority", topology="all-to-all")

    def test_empty_critics_list(self, protocol):
        """Empty critics list returns empty."""
        phase = CritiquePhase(protocol, [])

        critics = phase.select_critics_for_proposal("agent1", [])

        assert critics == []

    def test_proposer_not_in_critics(self, protocol):
        """Proposer not in critics list still works."""
        agents = [MockAgent("agent2"), MockAgent("agent3")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent1", agents)

        # Should return all critics since proposer isn't in list
        assert len(critics) == 2

    def test_many_agents(self, protocol):
        """Works with many agents."""
        agents = [MockAgent(f"agent{i}") for i in range(20)]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("agent0", agents)

        assert len(critics) == 19  # All except proposer
