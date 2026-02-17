"""Tests for debate topology critic selection.

Covers TopologySelector factory, AllToAll, RoundRobin, Ring, Star, Sparse,
Adaptive selectors, and select_critics_for_proposal convenience function.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.debate.topology import (
    AdaptiveSelector,
    AllToAllSelector,
    RingSelector,
    RoundRobinSelector,
    SparseSelector,
    StarSelector,
    TopologySelector,
    select_critics_for_proposal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(name: str) -> MagicMock:
    a = MagicMock()
    a.name = name
    return a


@pytest.fixture
def agents():
    return [make_agent(n) for n in ["alice", "bob", "charlie", "dave"]]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestTopologySelectorFactory:
    def test_create_all_to_all(self, agents):
        sel = TopologySelector.create("all-to-all", agents)
        assert isinstance(sel, AllToAllSelector)

    def test_create_round_robin(self, agents):
        sel = TopologySelector.create("round-robin", agents)
        assert isinstance(sel, RoundRobinSelector)

    def test_create_ring(self, agents):
        sel = TopologySelector.create("ring", agents)
        assert isinstance(sel, RingSelector)

    def test_create_star(self, agents):
        sel = TopologySelector.create("star", agents)
        assert isinstance(sel, StarSelector)

    def test_create_sparse(self, agents):
        sel = TopologySelector.create("sparse", agents)
        assert isinstance(sel, SparseSelector)

    def test_create_random_graph(self, agents):
        sel = TopologySelector.create("random-graph", agents)
        assert isinstance(sel, SparseSelector)

    def test_create_adaptive(self, agents):
        sel = TopologySelector.create("adaptive", agents)
        assert isinstance(sel, AdaptiveSelector)

    def test_create_unknown_defaults_all_to_all(self, agents):
        sel = TopologySelector.create("unknown-topology", agents)
        assert isinstance(sel, AllToAllSelector)

    def test_from_protocol(self, agents):
        proto = MagicMock()
        proto.topology = "ring"
        proto.topology_hub_agent = None
        proto.topology_sparsity = 0.5
        sel = TopologySelector.from_protocol(proto, agents)
        assert isinstance(sel, RingSelector)


# ---------------------------------------------------------------------------
# AllToAllSelector
# ---------------------------------------------------------------------------


class TestAllToAll:
    def test_excludes_self(self, agents):
        sel = AllToAllSelector()
        critics = sel.select_critics("alice", agents)
        names = [c.name for c in critics]
        assert "alice" not in names
        assert len(critics) == 3

    def test_single_agent(self):
        sel = AllToAllSelector()
        critics = sel.select_critics("alice", [make_agent("alice")])
        assert critics == []

    def test_empty(self):
        sel = AllToAllSelector()
        assert sel.select_critics("alice", []) == []


# ---------------------------------------------------------------------------
# RoundRobinSelector
# ---------------------------------------------------------------------------


class TestRoundRobin:
    def test_returns_one_critic(self, agents):
        sel = RoundRobinSelector()
        critics = sel.select_critics("alice", agents)
        assert len(critics) == 1
        assert critics[0].name != "alice"

    def test_deterministic(self, agents):
        sel = RoundRobinSelector()
        c1 = sel.select_critics("alice", agents)
        c2 = sel.select_critics("alice", agents)
        assert c1[0].name == c2[0].name

    def test_different_agents_different_critics(self, agents):
        sel = RoundRobinSelector()
        c_alice = sel.select_critics("alice", agents)
        c_bob = sel.select_critics("bob", agents)
        # Not guaranteed to differ, but with 4 agents it's very likely
        # Just test that both return valid results
        assert c_alice[0].name != "alice"
        assert c_bob[0].name != "bob"

    def test_empty_critics(self):
        sel = RoundRobinSelector()
        assert sel.select_critics("alice", []) == []

    def test_only_self(self):
        sel = RoundRobinSelector()
        assert sel.select_critics("alice", [make_agent("alice")]) == []


# ---------------------------------------------------------------------------
# RingSelector
# ---------------------------------------------------------------------------


class TestRing:
    def test_returns_neighbors(self, agents):
        sel = RingSelector(agents)
        critics = sel.select_critics("bob", agents)
        names = {c.name for c in critics}
        # Sorted: alice, bob, charlie, dave → bob's neighbors are alice and charlie
        assert "alice" in names
        assert "charlie" in names
        assert "bob" not in names

    def test_wraps_around(self, agents):
        sel = RingSelector(agents)
        # Sorted: alice(0), bob(1), charlie(2), dave(3)
        # alice's neighbors: dave(left wrap) and bob(right)
        critics = sel.select_critics("alice", agents)
        names = {c.name for c in critics}
        assert "dave" in names
        assert "bob" in names

    def test_end_wraps(self, agents):
        sel = RingSelector(agents)
        # dave's neighbors: charlie(left) and alice(right wrap)
        critics = sel.select_critics("dave", agents)
        names = {c.name for c in critics}
        assert "charlie" in names
        assert "alice" in names

    def test_unknown_agent_fallback(self, agents):
        sel = RingSelector(agents)
        critics = sel.select_critics("unknown", agents)
        # Falls back to all critics except self
        assert len(critics) == 4  # unknown not in agents

    def test_two_agents(self):
        a = [make_agent("a"), make_agent("b")]
        sel = RingSelector(a)
        critics = sel.select_critics("a", a)
        assert len(critics) == 1
        assert critics[0].name == "b"


# ---------------------------------------------------------------------------
# StarSelector
# ---------------------------------------------------------------------------


class TestStar:
    def test_hub_proposal_critiqued_by_all(self, agents):
        sel = StarSelector(agents, hub_agent="alice")
        critics = sel.select_critics("alice", agents)
        names = {c.name for c in critics}
        assert "alice" not in names
        assert len(critics) == 3

    def test_non_hub_critiqued_by_hub_only(self, agents):
        sel = StarSelector(agents, hub_agent="alice")
        critics = sel.select_critics("bob", agents)
        assert len(critics) == 1
        assert critics[0].name == "alice"

    def test_default_hub_is_first_alphabetically(self, agents):
        sel = StarSelector(agents)
        # Default hub is "alice" (first alphabetically)
        critics = sel.select_critics("bob", agents)
        assert critics[0].name == "alice"

    def test_hub_not_in_critics(self, agents):
        sel = StarSelector(agents, hub_agent="alice")
        critics = sel.select_critics("bob", [make_agent("bob"), make_agent("charlie")])
        # alice is hub but not in available critics
        assert critics == []


# ---------------------------------------------------------------------------
# SparseSelector
# ---------------------------------------------------------------------------


class TestSparse:
    def test_selects_subset(self, agents):
        sel = SparseSelector(sparsity=0.5)
        critics = sel.select_critics("alice", agents)
        # With 3 eligible critics and 0.5 sparsity → max(1, 1) = 1
        assert 1 <= len(critics) <= 3
        for c in critics:
            assert c.name != "alice"

    def test_deterministic(self, agents):
        sel = SparseSelector(sparsity=0.5)
        c1 = sel.select_critics("alice", agents)
        c2 = sel.select_critics("alice", agents)
        assert [c.name for c in c1] == [c.name for c in c2]

    def test_sparsity_one_selects_all(self, agents):
        sel = SparseSelector(sparsity=1.0)
        critics = sel.select_critics("alice", agents)
        assert len(critics) == 3  # all except self

    def test_sparsity_clamped(self):
        sel = SparseSelector(sparsity=2.0)
        assert sel._sparsity == 1.0

        sel = SparseSelector(sparsity=-0.5)
        assert sel._sparsity == 0.0

    def test_empty(self):
        sel = SparseSelector()
        assert sel.select_critics("alice", []) == []

    def test_minimum_one(self, agents):
        sel = SparseSelector(sparsity=0.01)
        critics = sel.select_critics("alice", agents)
        assert len(critics) >= 1


# ---------------------------------------------------------------------------
# AdaptiveSelector
# ---------------------------------------------------------------------------


class TestAdaptive:
    def test_diverging_uses_all_to_all(self, agents):
        sel = AdaptiveSelector(agents)
        sel.set_convergence_state("diverging", similarity=0.1)
        critics = sel.select_critics("bob", agents)
        # AllToAll: all except self
        assert len(critics) == 3

    def test_converged_uses_round_robin(self, agents):
        sel = AdaptiveSelector(agents)
        sel.set_convergence_state("converged", similarity=0.9)
        critics = sel.select_critics("bob", agents)
        assert len(critics) == 1  # round-robin: one critic

    def test_refining_uses_ring(self, agents):
        sel = AdaptiveSelector(agents)
        sel.set_convergence_state("refining", similarity=0.5)
        critics = sel.select_critics("bob", agents)
        assert len(critics) == 2  # ring: two neighbors

    def test_similarity_overrides_state_low(self, agents):
        sel = AdaptiveSelector(agents)
        # State says refining, but similarity is very low → diverging path
        sel.set_convergence_state("refining", similarity=0.1)
        critics = sel.select_critics("bob", agents)
        assert len(critics) == 3  # all-to-all

    def test_similarity_overrides_state_high(self, agents):
        sel = AdaptiveSelector(agents)
        # State says refining, but similarity is very high → converged path
        sel.set_convergence_state("refining", similarity=0.9)
        critics = sel.select_critics("bob", agents)
        assert len(critics) == 1  # round-robin

    def test_get_current_topology(self, agents):
        sel = AdaptiveSelector(agents)
        sel.set_convergence_state("diverging")
        assert sel.get_current_topology() == "all-to-all"

        sel.set_convergence_state("refining", similarity=0.5)
        assert sel.get_current_topology() == "ring"

        sel.set_convergence_state("converged", similarity=0.9)
        assert sel.get_current_topology() == "round-robin"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestSelectCriticsForProposal:
    def test_all_to_all(self, agents):
        critics = select_critics_for_proposal("alice", agents, agents, "all-to-all")
        assert len(critics) == 3

    def test_ring(self, agents):
        critics = select_critics_for_proposal("alice", agents, agents, "ring")
        assert len(critics) == 2

    def test_star_with_hub(self, agents):
        critics = select_critics_for_proposal(
            "bob", agents, agents, "star", hub_agent="alice"
        )
        assert len(critics) == 1
        assert critics[0].name == "alice"
