"""Tests for delegation strategies.

Covers DelegationStrategy, ContentBasedDelegation, LoadBalancedDelegation,
ExpertiseDelegation, RoundRobinDelegation, HybridDelegation, and
create_default_delegation factory.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.debate.delegation import (
    ContentBasedDelegation,
    ExpertiseDelegation,
    HybridDelegation,
    LoadBalancedDelegation,
    RoundRobinDelegation,
    create_default_delegation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(name: str, expertise_domains=None, capabilities=None) -> MagicMock:
    agent = MagicMock()
    agent.name = name
    # For ExpertiseDelegation
    if expertise_domains is not None:
        config = MagicMock()
        config.expertise_domains = expertise_domains
        config.capabilities = capabilities or []
        config.name = name
        agent._config = config
    else:
        agent._config = None
    return agent


@pytest.fixture
def agents():
    return [make_agent("alice"), make_agent("bob"), make_agent("charlie")]


# ---------------------------------------------------------------------------
# ContentBasedDelegation
# ---------------------------------------------------------------------------


class TestContentBasedDelegation:
    def test_default_score_without_keywords(self, agents):
        d = ContentBasedDelegation()
        scores = [d.score_agent(a, "some task") for a in agents]
        assert all(s == 1.0 for s in scores)

    def test_keyword_match_boosts_score(self):
        d = ContentBasedDelegation()
        d.add_expertise("security", ["alice"])
        alice = make_agent("alice")
        bob = make_agent("bob")
        assert d.score_agent(alice, "security audit") > d.score_agent(bob, "security audit")

    def test_case_insensitive_by_default(self):
        d = ContentBasedDelegation()
        d.add_expertise("security", ["alice"])
        alice = make_agent("alice")
        assert d.score_agent(alice, "SECURITY review") > 1.0

    def test_case_sensitive_mode(self):
        d = ContentBasedDelegation(case_sensitive=True)
        d.add_expertise("Security", ["alice"])
        alice = make_agent("alice")
        assert d.score_agent(alice, "security review") == 1.0  # no match (lowercase)
        assert d.score_agent(alice, "Security review") > 1.0

    def test_select_agents_ordered_by_score(self):
        d = ContentBasedDelegation()
        d.add_expertise("security", ["alice"])
        d.add_expertise("compliance", ["alice"])
        agents = [make_agent("bob"), make_agent("alice")]
        selected = d.select_agents("security compliance review", agents)
        assert selected[0].name == "alice"

    def test_select_agents_max_agents(self, agents):
        d = ContentBasedDelegation()
        selected = d.select_agents("task", agents, max_agents=2)
        assert len(selected) == 2

    def test_add_from_config(self):
        d = ContentBasedDelegation()
        config = MagicMock()
        config.expertise_domains = ["security", "compliance"]
        config.capabilities = ["code_review"]
        config.name = "alice"
        d.add_from_config(config)
        assert "security" in d.keyword_mapping
        assert "alice" in d.keyword_mapping["security"]
        assert "alice" in d.keyword_mapping["code_review"]

    def test_extract_keywords(self):
        d = ContentBasedDelegation()
        keywords = d._extract_keywords("Find SQL injection vulnerabilities")
        assert "find" in keywords
        assert "sql" in keywords
        assert "injection" in keywords

    def test_multiple_keywords_for_agent(self):
        d = ContentBasedDelegation()
        d.add_expertise("security", ["alice"])
        d.add_expertise("vulnerability", ["alice"])
        alice = make_agent("alice")
        score = d.score_agent(alice, "fix security vulnerability")
        assert score >= 3.0  # base 1.0 + 2 keyword matches


# ---------------------------------------------------------------------------
# LoadBalancedDelegation
# ---------------------------------------------------------------------------


class TestLoadBalancedDelegation:
    def test_default_all_agents_equal(self, agents):
        d = LoadBalancedDelegation()
        scores = [d.score_agent(a, "task") for a in agents]
        assert all(s == 1.0 for s in scores)

    def test_loaded_agent_gets_lower_score(self):
        d = LoadBalancedDelegation()
        d.record_task("alice")
        d.record_task("alice")
        alice = make_agent("alice")
        bob = make_agent("bob")
        assert d.score_agent(alice, "task") < d.score_agent(bob, "task")

    def test_overloaded_agent_gets_zero(self):
        d = LoadBalancedDelegation(max_concurrent_per_agent=2)
        d.record_task("alice")
        d.record_task("alice")
        alice = make_agent("alice")
        assert d.score_agent(alice, "task") == 0.0

    def test_complete_task_reduces_load(self):
        d = LoadBalancedDelegation()
        d.record_task("alice")
        d.record_task("alice")
        d.complete_task("alice")
        assert d.agent_load["alice"] == 1

    def test_complete_task_no_negative(self):
        d = LoadBalancedDelegation()
        d.complete_task("alice")
        assert d.agent_load.get("alice", 0) == 0

    def test_select_agents_excludes_overloaded(self):
        d = LoadBalancedDelegation(max_concurrent_per_agent=1)
        d.record_task("alice")
        agents = [make_agent("alice"), make_agent("bob")]
        selected = d.select_agents("task", agents)
        names = [a.name for a in selected]
        assert "alice" not in names
        assert "bob" in names

    def test_context_workload_override(self):
        d = LoadBalancedDelegation()
        ctx = MagicMock()
        ctx.agent_workloads = {"alice": 5}
        alice = make_agent("alice")
        load = d.get_load("alice", ctx)
        assert load == 5

    def test_select_with_max_agents(self, agents):
        d = LoadBalancedDelegation()
        selected = d.select_agents("task", agents, max_agents=1)
        assert len(selected) == 1


# ---------------------------------------------------------------------------
# ExpertiseDelegation
# ---------------------------------------------------------------------------


class TestExpertiseDelegation:
    def test_default_domain_keywords(self):
        d = ExpertiseDelegation()
        assert "security" in d.domain_keywords
        assert "compliance" in d.domain_keywords
        assert "performance" in d.domain_keywords

    def test_security_match(self):
        d = ExpertiseDelegation()
        matched = d._match_domains("Find SQL injection vulnerabilities")
        assert "security" in matched

    def test_no_match_returns_default(self):
        d = ExpertiseDelegation()
        agent = make_agent("alice")
        score = d.score_agent(agent, "unrelated topic about cooking")
        assert score == 1.0

    def test_expertise_match_boosts_score(self):
        d = ExpertiseDelegation()
        alice = make_agent("alice", expertise_domains=["security"])
        score = d.score_agent(alice, "Fix authentication vulnerability")
        assert score > 1.0

    def test_multiple_domain_overlap(self):
        d = ExpertiseDelegation()
        alice = make_agent("alice", expertise_domains=["security", "compliance"])
        score = d.score_agent(alice, "security audit for regulatory compliance")
        assert score >= 3.0  # base 1.0 + 2 domain matches

    def test_select_agents_ordered(self):
        d = ExpertiseDelegation()
        agents = [
            make_agent("generic"),
            make_agent("expert", expertise_domains=["security"]),
        ]
        selected = d.select_agents("Fix security issue", agents)
        assert selected[0].name == "expert"

    def test_agent_without_config(self):
        d = ExpertiseDelegation()
        agent = make_agent("no-config")
        domains = d._get_agent_domains(agent)
        assert domains == []


# ---------------------------------------------------------------------------
# RoundRobinDelegation
# ---------------------------------------------------------------------------


class TestRoundRobinDelegation:
    def test_selects_all_agents_by_default(self, agents):
        d = RoundRobinDelegation()
        selected = d.select_agents("task", agents)
        assert len(selected) == 3

    def test_max_agents_limits(self, agents):
        d = RoundRobinDelegation()
        selected = d.select_agents("task", agents, max_agents=1)
        assert len(selected) == 1

    def test_cursor_advances(self, agents):
        d = RoundRobinDelegation()
        s1 = d.select_agents("task", agents, max_agents=1)
        s2 = d.select_agents("task", agents, max_agents=1)
        assert s1[0].name != s2[0].name

    def test_cursor_wraps(self):
        agents = [make_agent("a"), make_agent("b")]
        d = RoundRobinDelegation()
        # Select all twice to wrap
        d.select_agents("task", agents, max_agents=2)
        d.select_agents("task", agents, max_agents=2)
        assert d.cursor == 0  # wraps back

    def test_empty_agents(self):
        d = RoundRobinDelegation()
        assert d.select_agents("task", []) == []

    def test_score_agent_equal(self, agents):
        d = RoundRobinDelegation()
        for a in agents:
            assert d.score_agent(a, "task") == 1.0


# ---------------------------------------------------------------------------
# HybridDelegation
# ---------------------------------------------------------------------------


class TestHybridDelegation:
    def test_normalizes_weights(self):
        s1 = ContentBasedDelegation()
        s2 = LoadBalancedDelegation()
        h = HybridDelegation(strategies=[(s1, 2.0), (s2, 2.0)])
        # Weights should be normalized to sum to 1.0
        total = sum(w for _, w in h.strategies)
        assert abs(total - 1.0) < 0.001

    def test_empty_strategies_returns_default_score(self):
        h = HybridDelegation()
        agent = make_agent("alice")
        assert h.score_agent(agent, "task") == 1.0

    def test_weighted_scoring(self):
        s1 = ContentBasedDelegation()
        s1.add_expertise("security", ["alice"])
        h = HybridDelegation(strategies=[(s1, 1.0)])
        alice = make_agent("alice")
        bob = make_agent("bob")
        assert h.score_agent(alice, "security") > h.score_agent(bob, "security")

    def test_add_strategy(self):
        h = HybridDelegation()
        h.add_strategy(ContentBasedDelegation(), 0.5)
        h.add_strategy(LoadBalancedDelegation(), 0.5)
        assert len(h.strategies) == 2

    def test_select_agents_ordered(self):
        s1 = ContentBasedDelegation()
        s1.add_expertise("security", ["alice"])
        h = HybridDelegation(strategies=[(s1, 1.0)])
        agents = [make_agent("bob"), make_agent("alice")]
        selected = h.select_agents("security audit", agents)
        assert selected[0].name == "alice"

    def test_select_with_max_agents(self, agents):
        h = HybridDelegation(strategies=[(ContentBasedDelegation(), 1.0)])
        selected = h.select_agents("task", agents, max_agents=2)
        assert len(selected) == 2


# ---------------------------------------------------------------------------
# create_default_delegation
# ---------------------------------------------------------------------------


class TestCreateDefaultDelegation:
    def test_creates_hybrid(self):
        h = create_default_delegation()
        assert isinstance(h, HybridDelegation)
        assert len(h.strategies) == 3  # content + load + expertise

    def test_exclude_content(self):
        h = create_default_delegation(include_content=False)
        assert len(h.strategies) == 2

    def test_exclude_all(self):
        h = create_default_delegation(
            include_content=False, include_load=False, include_expertise=False
        )
        assert len(h.strategies) == 0

    def test_custom_weights(self):
        h = create_default_delegation(content_weight=0.6, load_weight=0.2, expertise_weight=0.2)
        assert len(h.strategies) == 3
        # Weights are normalized, so check relative ordering
        total = sum(w for _, w in h.strategies)
        assert abs(total - 1.0) < 0.001

    def test_default_has_security_keywords(self):
        h = create_default_delegation()
        # The content-based strategy should have security keywords
        content_strategy = h.strategies[0][0]
        assert isinstance(content_strategy, ContentBasedDelegation)
        assert "security" in content_strategy.keyword_mapping

    def test_select_agents_works(self):
        h = create_default_delegation()
        agents = [make_agent("alice"), make_agent("bob")]
        selected = h.select_agents("security review", agents)
        assert len(selected) == 2
