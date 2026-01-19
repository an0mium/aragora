"""
Tests for the CritiquePhase class.

Tests critic selection logic for various debate topologies.
"""

from dataclasses import dataclass
from typing import Optional

import pytest


@dataclass
class MockAgent:
    """Mock Agent for testing."""

    name: str
    role: str = "debater"


@dataclass
class MockCritique:
    """Mock Critique for testing."""

    agent: str
    target: str
    content: str
    summary: str = ""


@dataclass
class MockProtocol:
    """Mock DebateProtocol for testing."""

    topology: str = "all-to-all"
    topology_hub_agent: Optional[str] = None
    topology_sparsity: float = 0.5


class TestCritiquePhaseTopologies:
    """Tests for critic selection based on topology."""

    def test_all_to_all_selects_all_other_critics(self):
        """All-to-all topology should select all critics except the proposer."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="all-to-all")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal(
            "alice", [MockAgent("bob"), MockAgent("charlie")]
        )

        assert len(critics) == 2
        assert all(c.name != "alice" for c in critics)

    def test_all_to_all_excludes_self(self):
        """All-to-all should not include the proposal agent as a critic."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="all-to-all")
        agents = [MockAgent("alice"), MockAgent("bob")]
        phase = CritiquePhase(protocol, agents)

        # Include proposer in critics list
        critics = phase.select_critics_for_proposal(
            "alice", [MockAgent("alice"), MockAgent("bob")]
        )

        assert len(critics) == 1
        assert critics[0].name == "bob"

    def test_round_robin_selects_single_critic(self):
        """Round-robin should select exactly one critic."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="round-robin")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal(
            "alice", [MockAgent("bob"), MockAgent("charlie")]
        )

        assert len(critics) == 1
        assert critics[0].name != "alice"

    def test_round_robin_is_deterministic(self):
        """Same proposer should always get same critic in round-robin."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="round-robin")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = CritiquePhase(protocol, agents)

        critics1 = phase.select_critics_for_proposal(
            "alice", [MockAgent("bob"), MockAgent("charlie")]
        )
        critics2 = phase.select_critics_for_proposal(
            "alice", [MockAgent("bob"), MockAgent("charlie")]
        )

        assert critics1[0].name == critics2[0].name

    def test_ring_selects_neighbors(self):
        """Ring topology should select left and right neighbors."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="ring")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal(
            "bob", [MockAgent("alice"), MockAgent("charlie")]
        )

        # bob's neighbors should be alice and charlie
        critic_names = {c.name for c in critics}
        assert "alice" in critic_names or "charlie" in critic_names

    def test_star_hub_critiqued_by_all(self):
        """In star topology, hub's proposal is critiqued by all others."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="star", topology_hub_agent="alice")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal(
            "alice", [MockAgent("bob"), MockAgent("charlie")]
        )

        assert len(critics) == 2
        critic_names = {c.name for c in critics}
        assert "bob" in critic_names
        assert "charlie" in critic_names

    def test_star_spoke_critiqued_only_by_hub(self):
        """In star topology, spoke's proposal is critiqued only by hub."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="star", topology_hub_agent="alice")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal(
            "bob", [MockAgent("alice"), MockAgent("charlie")]
        )

        assert len(critics) == 1
        assert critics[0].name == "alice"

    def test_sparse_selects_subset(self):
        """Sparse topology should select a random subset based on sparsity."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="sparse", topology_sparsity=0.5)
        agents = [
            MockAgent("alice"),
            MockAgent("bob"),
            MockAgent("charlie"),
            MockAgent("diana"),
        ]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal(
            "alice", [MockAgent("bob"), MockAgent("charlie"), MockAgent("diana")]
        )

        # With sparsity 0.5 and 3 eligible critics, should select ~1-2
        assert 1 <= len(critics) <= 3
        assert all(c.name != "alice" for c in critics)

    def test_sparse_is_deterministic(self):
        """Same proposer should get consistent critics in sparse topology."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="sparse", topology_sparsity=0.5)
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = CritiquePhase(protocol, agents)

        critics1 = phase.select_critics_for_proposal(
            "alice", [MockAgent("bob"), MockAgent("charlie")]
        )
        critics2 = phase.select_critics_for_proposal(
            "alice", [MockAgent("bob"), MockAgent("charlie")]
        )

        assert {c.name for c in critics1} == {c.name for c in critics2}

    def test_unknown_topology_defaults_to_all_to_all(self):
        """Unknown topology should default to all-to-all behavior."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="unknown-topology")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal(
            "alice", [MockAgent("bob"), MockAgent("charlie")]
        )

        assert len(critics) == 2

    def test_empty_critics_returns_empty(self):
        """No available critics should return empty list."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol(topology="all-to-all")
        agents = [MockAgent("alice")]
        phase = CritiquePhase(protocol, agents)

        critics = phase.select_critics_for_proposal("alice", [])

        assert critics == []


class TestCritiqueAggregation:
    """Tests for critique aggregation methods."""

    def test_aggregate_by_target(self):
        """Should group critiques by target agent."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol()
        phase = CritiquePhase(protocol, [])

        critiques = [
            MockCritique("bob", "alice", "Good point"),
            MockCritique("charlie", "alice", "Needs more evidence"),
            MockCritique("alice", "bob", "Interesting approach"),
        ]

        result = phase.aggregate_critiques(critiques, by_target=True)

        assert "alice" in result
        assert len(result["alice"]) == 2
        assert "bob" in result
        assert len(result["bob"]) == 1

    def test_aggregate_by_critic(self):
        """Should group critiques by critic agent."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol()
        phase = CritiquePhase(protocol, [])

        critiques = [
            MockCritique("bob", "alice", "Good point"),
            MockCritique("bob", "charlie", "Solid reasoning"),
            MockCritique("alice", "bob", "Interesting approach"),
        ]

        result = phase.aggregate_critiques(critiques, by_target=False)

        assert "bob" in result
        assert len(result["bob"]) == 2
        assert "alice" in result
        assert len(result["alice"]) == 1

    def test_aggregate_empty_critiques(self):
        """Empty critiques list should return empty dict."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol()
        phase = CritiquePhase(protocol, [])

        result = phase.aggregate_critiques([], by_target=True)

        assert result == {}


class TestCritiqueStats:
    """Tests for critique statistics."""

    def test_stats_count(self):
        """Should correctly count critiques."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol()
        phase = CritiquePhase(protocol, [])

        critiques = [
            MockCritique("bob", "alice", "Good point"),
            MockCritique("charlie", "alice", "Needs evidence"),
        ]

        stats = phase.get_critique_stats(critiques)

        assert stats["count"] == 2

    def test_stats_avg_length(self):
        """Should calculate average critique length."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol()
        phase = CritiquePhase(protocol, [])

        critiques = [
            MockCritique("bob", "alice", "12345"),  # 5 chars
            MockCritique("charlie", "alice", "123456789012345"),  # 15 chars
        ]

        stats = phase.get_critique_stats(critiques)

        assert stats["avg_length"] == 10.0

    def test_stats_critics_and_targets(self):
        """Should list unique critics and targets."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol()
        phase = CritiquePhase(protocol, [])

        critiques = [
            MockCritique("bob", "alice", "Point 1"),
            MockCritique("bob", "charlie", "Point 2"),
            MockCritique("diana", "alice", "Point 3"),
        ]

        stats = phase.get_critique_stats(critiques)

        assert set(stats["critics"]) == {"bob", "diana"}
        assert set(stats["targets"]) == {"alice", "charlie"}

    def test_stats_empty_critiques(self):
        """Empty critiques should return zeroed stats."""
        from aragora.debate.phases.critique import CritiquePhase

        protocol = MockProtocol()
        phase = CritiquePhase(protocol, [])

        stats = phase.get_critique_stats([])

        assert stats["count"] == 0
        assert stats["avg_length"] == 0
        assert stats["critics"] == []
        assert stats["targets"] == []
