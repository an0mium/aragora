"""Comprehensive tests for selection plugin strategies.

Tests the built-in selection strategy implementations:
- ELOWeightedScorer - Agent scoring algorithm
- DiverseTeamSelector - Diversity-aware team selection
- GreedyTeamSelector - Simple greedy selection
- RandomTeamSelector - Random weighted selection
- DomainBasedRoleAssigner - Domain-expertise role assignment
- SimpleRoleAssigner - Basic role assignment
"""

import random
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.plugins.selection.protocols import SelectionContext
from aragora.plugins.selection.strategies import (
    DiverseTeamSelector,
    DomainBasedRoleAssigner,
    ELOWeightedScorer,
    GreedyTeamSelector,
    RandomTeamSelector,
    SimpleRoleAssigner,
)
from aragora.routing.selection import AgentProfile, TaskRequirements


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent_pool():
    """Create a diverse pool of agent profiles for testing."""
    return [
        AgentProfile(
            name="claude",
            agent_type="claude",
            elo_rating=1700,
            expertise={"security": 0.95, "architecture": 0.9, "api": 0.85, "general": 0.85},
            traits=["thorough", "security", "analytical"],
            domain_ratings={"security": 1750, "architecture": 1680},
            success_rate=0.9,
            latency_ms=1200,
            cost_factor=1.5,
            probe_score=0.95,
            calibration_score=0.9,
        ),
        AgentProfile(
            name="codex",
            agent_type="codex",
            elo_rating=1600,
            expertise={"performance": 0.95, "debugging": 0.9, "testing": 0.85, "general": 0.9},
            traits=["fast", "pragmatic", "performance"],
            domain_ratings={"performance": 1650, "debugging": 1620},
            success_rate=0.85,
            latency_ms=800,
            cost_factor=1.0,
            probe_score=0.9,
            calibration_score=0.85,
        ),
        AgentProfile(
            name="gemini",
            agent_type="gemini",
            elo_rating=1650,
            expertise={"architecture": 0.95, "data_analysis": 0.9, "api": 0.85, "general": 0.85},
            traits=["creative", "analytical"],
            domain_ratings={"architecture": 1700, "data_analysis": 1660},
            success_rate=0.88,
            latency_ms=1000,
            cost_factor=1.2,
            probe_score=0.92,
            calibration_score=0.88,
        ),
        AgentProfile(
            name="grok",
            agent_type="grok",
            elo_rating=1550,
            expertise={"debugging": 0.9, "security": 0.85, "testing": 0.85, "general": 0.8},
            traits=["lateral", "pragmatic", "thorough"],
            domain_ratings={"debugging": 1580, "security": 1560},
            success_rate=0.8,
            latency_ms=900,
            cost_factor=0.8,
            probe_score=0.85,
            calibration_score=0.8,
        ),
        AgentProfile(
            name="deepseek",
            agent_type="deepseek",
            elo_rating=1580,
            expertise={
                "architecture": 0.9,
                "performance": 0.9,
                "data_analysis": 0.9,
                "general": 0.85,
            },
            traits=["analytical", "rigorous"],
            domain_ratings={"architecture": 1600, "data_analysis": 1590},
            success_rate=0.82,
            latency_ms=1100,
            cost_factor=0.9,
            probe_score=0.88,
            calibration_score=0.85,
        ),
    ]


@pytest.fixture
def security_requirements():
    """Create security-focused task requirements."""
    return TaskRequirements(
        task_id="security-task",
        description="Implement secure authentication",
        primary_domain="security",
        secondary_domains=["api", "architecture"],
        required_traits=["thorough", "security"],
        min_agents=2,
        max_agents=4,
        quality_priority=0.8,
        diversity_preference=0.5,
    )


@pytest.fixture
def performance_requirements():
    """Create performance-focused task requirements."""
    return TaskRequirements(
        task_id="perf-task",
        description="Optimize database queries",
        primary_domain="performance",
        secondary_domains=["debugging"],
        required_traits=["pragmatic", "fast"],
        min_agents=2,
        max_agents=3,
        quality_priority=0.4,  # Prioritize speed
        diversity_preference=0.3,
    )


@pytest.fixture
def basic_context(agent_pool):
    """Create a basic selection context."""
    return SelectionContext(
        agent_pool={a.name: a for a in agent_pool},
        bench=[],
    )


@pytest.fixture
def context_with_probe_filter(basic_context):
    """Create context with mock probe filter."""
    mock_filter = MagicMock()

    # Set up probe profiles for each agent
    def get_agent_profile(name):
        profile = MagicMock()
        profile.total_probes = 10
        profile.probe_score = 0.9 if name == "claude" else 0.7
        profile.has_critical_issues = lambda: name == "grok"
        return profile

    mock_filter.get_agent_profile = get_agent_profile
    basic_context.probe_filter = mock_filter
    return basic_context


@pytest.fixture
def context_with_calibration(basic_context):
    """Create context with mock calibration tracker."""
    mock_tracker = MagicMock()

    def get_calibration_summary(name):
        summary = MagicMock()
        summary.total_predictions = 20
        summary.ece = 0.1 if name == "claude" else 0.3
        summary.is_overconfident = name == "grok"
        return summary

    mock_tracker.get_calibration_summary = get_calibration_summary
    basic_context.calibration_tracker = mock_tracker
    return basic_context


@pytest.fixture
def context_with_performance(basic_context):
    """Create context with performance insights."""
    basic_context.performance_insights = {
        "agent_stats": {
            "claude": {"success_rate": 95, "timeout_rate": 2, "failure_rate": 3},
            "codex": {"success_rate": 85, "timeout_rate": 8, "failure_rate": 7},
            "gemini": {"success_rate": 88, "timeout_rate": 5, "failure_rate": 7},
            "grok": {"success_rate": 65, "timeout_rate": 25, "failure_rate": 35},
            "deepseek": {"success_rate": 82, "timeout_rate": 12, "failure_rate": 6},
        }
    }
    return basic_context


# =============================================================================
# ELOWeightedScorer Tests
# =============================================================================


class TestELOWeightedScorer:
    """Tests for ELO-weighted scoring algorithm."""

    def test_scorer_properties(self):
        """Scorer has correct name and description."""
        scorer = ELOWeightedScorer()
        assert scorer.name == "elo-weighted"
        assert "ELO" in scorer.description or "expertise" in scorer.description.lower()

    def test_scores_in_valid_range(self, agent_pool, security_requirements, basic_context):
        """All scores are between 0 and 1."""
        scorer = ELOWeightedScorer()

        for agent in agent_pool:
            score = scorer.score_agent(agent, security_requirements, basic_context)
            assert 0.0 <= score <= 1.0, f"Score {score} for {agent.name} out of range"

    def test_domain_expert_scores_higher(self, agent_pool, security_requirements, basic_context):
        """Agent with primary domain expertise scores highest."""
        scorer = ELOWeightedScorer()

        scores = {
            agent.name: scorer.score_agent(agent, security_requirements, basic_context)
            for agent in agent_pool
        }

        # Claude has highest security expertise (0.95), should score highest
        assert scores["claude"] >= max(scores[a] for a in scores if a != "claude")

    def test_trait_matching_increases_score(self, agent_pool, basic_context):
        """Agents matching required traits score higher."""
        scorer = ELOWeightedScorer()

        # Create requirements with traits that match claude
        requirements = TaskRequirements(
            task_id="trait-test",
            description="Test",
            primary_domain="general",
            required_traits=["thorough", "security"],
            min_agents=2,
            max_agents=3,
        )

        scores = {
            agent.name: scorer.score_agent(agent, requirements, basic_context)
            for agent in agent_pool
        }

        # Claude has both thorough and security traits
        # Grok has thorough and pragmatic (only 1 match)
        assert scores["claude"] >= scores["grok"]

    def test_quality_priority_affects_scoring(self, agent_pool, basic_context):
        """Quality priority changes scoring emphasis."""
        scorer = ELOWeightedScorer()

        # High quality priority
        quality_req = TaskRequirements(
            task_id="q-test",
            description="Test",
            primary_domain="general",
            quality_priority=0.9,
            min_agents=2,
            max_agents=3,
        )

        # Low quality priority (speed/cost)
        speed_req = TaskRequirements(
            task_id="s-test",
            description="Test",
            primary_domain="general",
            quality_priority=0.1,
            min_agents=2,
            max_agents=3,
        )

        # Codex is fast and cheap, Claude is expensive but high quality
        claude = next(a for a in agent_pool if a.name == "claude")
        codex = next(a for a in agent_pool if a.name == "codex")

        quality_scores = {
            "claude": scorer.score_agent(claude, quality_req, basic_context),
            "codex": scorer.score_agent(codex, quality_req, basic_context),
        }

        speed_scores = {
            "claude": scorer.score_agent(claude, speed_req, basic_context),
            "codex": scorer.score_agent(codex, speed_req, basic_context),
        }

        # With quality priority, success rate matters more
        # With speed priority, latency and cost matter more
        # The gap should be different
        quality_gap = quality_scores["claude"] - quality_scores["codex"]
        speed_gap = speed_scores["claude"] - speed_scores["codex"]

        # Claude should be more favored in quality mode than in speed mode
        assert quality_gap > speed_gap

    def test_probe_filter_adjustment(
        self, agent_pool, security_requirements, context_with_probe_filter
    ):
        """Probe filter affects scoring."""
        scorer = ELOWeightedScorer()

        # Grok has critical probe issues in our mock
        grok = next(a for a in agent_pool if a.name == "grok")
        claude = next(a for a in agent_pool if a.name == "claude")

        grok_score = scorer.score_agent(grok, security_requirements, context_with_probe_filter)
        claude_score = scorer.score_agent(claude, security_requirements, context_with_probe_filter)

        # Claude should score higher due to better probe score
        assert claude_score > grok_score

    def test_calibration_adjustment(
        self, agent_pool, security_requirements, context_with_calibration
    ):
        """Calibration tracker affects scoring."""
        scorer = ELOWeightedScorer()

        # Grok is overconfident in our mock
        grok = next(a for a in agent_pool if a.name == "grok")
        claude = next(a for a in agent_pool if a.name == "claude")

        grok_score = scorer.score_agent(grok, security_requirements, context_with_calibration)
        claude_score = scorer.score_agent(claude, security_requirements, context_with_calibration)

        # Claude has better calibration
        assert claude_score > grok_score

    def test_performance_adjustment(
        self, agent_pool, security_requirements, context_with_performance
    ):
        """Performance insights affect scoring."""
        scorer = ELOWeightedScorer()

        # Grok has poor performance metrics in our mock
        grok = next(a for a in agent_pool if a.name == "grok")
        claude = next(a for a in agent_pool if a.name == "claude")

        grok_score = scorer.score_agent(grok, security_requirements, context_with_performance)
        claude_score = scorer.score_agent(claude, security_requirements, context_with_performance)

        # Claude has much better performance metrics
        assert claude_score > grok_score

    def test_persona_manager_integration(self, agent_pool, basic_context):
        """PersonaManager expertise overrides static expertise."""
        scorer = ELOWeightedScorer()

        # Set up mock persona manager
        mock_persona_manager = MagicMock()
        mock_persona = MagicMock()
        mock_persona.expertise = {"security": 0.99}  # Override claude's security
        mock_persona.traits = ["super_secure"]
        mock_persona_manager.get_persona.return_value = mock_persona
        basic_context.persona_manager = mock_persona_manager

        requirements = TaskRequirements(
            task_id="persona-test",
            description="Test",
            primary_domain="security",
            required_traits=["super_secure"],
            min_agents=2,
            max_agents=3,
        )

        claude = next(a for a in agent_pool if a.name == "claude")
        score = scorer.score_agent(claude, requirements, basic_context)

        # Persona manager should have been called
        mock_persona_manager.get_persona.assert_called()
        # Score should be valid
        assert 0.0 <= score <= 1.0


# =============================================================================
# DiverseTeamSelector Tests
# =============================================================================


class TestDiverseTeamSelector:
    """Tests for diversity-aware team selection."""

    def test_selector_properties(self):
        """Selector has correct name and description."""
        selector = DiverseTeamSelector()
        assert selector.name == "diverse"
        assert "diversity" in selector.description.lower()

    def test_respects_min_agents(self, agent_pool, security_requirements, basic_context):
        """Selected team has at least min_agents."""
        selector = DiverseTeamSelector()
        scored = [(a, 1.0 - i * 0.1) for i, a in enumerate(agent_pool)]

        team = selector.select_team(scored, security_requirements, basic_context)

        assert len(team) >= security_requirements.min_agents

    def test_respects_max_agents(self, agent_pool, basic_context):
        """Selected team has at most max_agents."""
        selector = DiverseTeamSelector()
        requirements = TaskRequirements(
            task_id="max-test",
            description="Test",
            primary_domain="general",
            min_agents=1,
            max_agents=2,  # Limit to 2
        )
        scored = [(a, 1.0 - i * 0.1) for i, a in enumerate(agent_pool)]

        team = selector.select_team(scored, requirements, basic_context)

        assert len(team) <= requirements.max_agents

    def test_returns_all_when_fewer_candidates(self, agent_pool, basic_context):
        """Returns all candidates when fewer than min_agents."""
        selector = DiverseTeamSelector()
        requirements = TaskRequirements(
            task_id="few-test",
            description="Test",
            primary_domain="general",
            min_agents=10,  # More than available
            max_agents=15,
        )
        scored = [(a, 0.5) for a in agent_pool]

        team = selector.select_team(scored, requirements, basic_context)

        assert len(team) == len(agent_pool)

    def test_high_diversity_preference(self, agent_pool, basic_context):
        """High diversity preference selects different agent types."""
        selector = DiverseTeamSelector()
        requirements = TaskRequirements(
            task_id="div-test",
            description="Test",
            primary_domain="general",
            min_agents=2,
            max_agents=4,
            diversity_preference=0.9,  # Very high diversity
        )

        # Give all agents similar scores
        scored = [(a, 0.8) for a in agent_pool]

        # Run multiple times to account for randomness
        random.seed(42)
        team = selector.select_team(scored, requirements, basic_context)

        # With high diversity, should have different agent types
        agent_types = set(a.agent_type for a in team)
        assert len(agent_types) >= min(2, len(team))


# =============================================================================
# GreedyTeamSelector Tests
# =============================================================================


class TestGreedyTeamSelector:
    """Tests for simple greedy team selection."""

    def test_selector_properties(self):
        """Selector has correct name and description."""
        selector = GreedyTeamSelector()
        assert selector.name == "greedy"
        assert "greedy" in selector.description.lower()

    def test_selects_highest_scored(self, agent_pool, security_requirements, basic_context):
        """Greedy selector picks highest scored agents."""
        selector = GreedyTeamSelector()

        # Assign known scores - claude highest, then gemini, etc.
        scored = [
            (agent_pool[0], 1.0),  # claude
            (agent_pool[2], 0.9),  # gemini
            (agent_pool[1], 0.8),  # codex
            (agent_pool[4], 0.7),  # deepseek
            (agent_pool[3], 0.6),  # grok
        ]

        team = selector.select_team(scored, security_requirements, basic_context)

        # Should pick claude and gemini (top 2+ based on max_agents)
        names = [a.name for a in team]
        assert "claude" in names  # Highest score
        assert team[0].name == "claude"  # First should be highest

    def test_respects_max_agents(self, agent_pool, basic_context):
        """Greedy selector respects max_agents limit."""
        selector = GreedyTeamSelector()
        requirements = TaskRequirements(
            task_id="max-test",
            description="Test",
            primary_domain="general",
            min_agents=1,
            max_agents=2,
        )
        scored = [(a, 1.0 - i * 0.1) for i, a in enumerate(agent_pool)]

        team = selector.select_team(scored, requirements, basic_context)

        assert len(team) == 2  # Exactly max_agents


# =============================================================================
# RandomTeamSelector Tests
# =============================================================================


class TestRandomTeamSelector:
    """Tests for random weighted team selection."""

    def test_selector_properties(self):
        """Selector has correct name and description."""
        selector = RandomTeamSelector()
        assert selector.name == "random"
        assert "random" in selector.description.lower()

    def test_respects_team_size(self, agent_pool, security_requirements, basic_context):
        """Random selector respects team size constraints."""
        selector = RandomTeamSelector()
        scored = [(a, 0.5) for a in agent_pool]

        team = selector.select_team(scored, security_requirements, basic_context)

        assert security_requirements.min_agents <= len(team) <= security_requirements.max_agents

    def test_returns_all_when_requested(self, agent_pool, basic_context):
        """Returns all when max_agents >= pool size."""
        selector = RandomTeamSelector()
        requirements = TaskRequirements(
            task_id="all-test",
            description="Test",
            primary_domain="general",
            min_agents=5,
            max_agents=10,  # More than available
        )
        scored = [(a, 0.5) for a in agent_pool]

        team = selector.select_team(scored, requirements, basic_context)

        assert len(team) == len(agent_pool)

    def test_weighted_selection(self, agent_pool, basic_context):
        """Higher scored agents more likely to be selected."""
        selector = RandomTeamSelector()
        requirements = TaskRequirements(
            task_id="weighted-test",
            description="Test",
            primary_domain="general",
            min_agents=1,
            max_agents=1,  # Select just one
        )

        # Give one agent very high score
        scored = [(a, 0.01) for a in agent_pool]
        high_agent = agent_pool[0]
        scored[0] = (high_agent, 0.99)

        # Run many times and count selections
        selection_counts = {a.name: 0 for a in agent_pool}
        for i in range(100):
            random.seed(i)
            team = selector.select_team(scored.copy(), requirements, basic_context)
            selection_counts[team[0].name] += 1

        # High-scored agent should be selected most often
        assert selection_counts[high_agent.name] > 50  # More than half the time


# =============================================================================
# DomainBasedRoleAssigner Tests
# =============================================================================


class TestDomainBasedRoleAssigner:
    """Tests for domain-expertise role assignment."""

    def test_assigner_properties(self):
        """Assigner has correct name and description."""
        assigner = DomainBasedRoleAssigner()
        assert assigner.name == "domain-based"
        assert "domain" in assigner.description.lower()

    def test_assigns_proposer_to_expert(self, agent_pool, security_requirements, basic_context):
        """Proposer role goes to domain expert."""
        assigner = DomainBasedRoleAssigner()
        team = agent_pool[:3]  # claude, codex, gemini

        roles = assigner.assign_roles(team, security_requirements, basic_context)

        # Claude has highest security expertise, should be proposer
        assert roles["claude"] == "proposer"

    def test_assigns_synthesizer(self, agent_pool, security_requirements, basic_context):
        """Synthesizer role is assigned when team > 1."""
        assigner = DomainBasedRoleAssigner()
        team = agent_pool[:3]

        roles = assigner.assign_roles(team, security_requirements, basic_context)

        assert "synthesizer" in roles.values()

    def test_assigns_critics(self, agent_pool, security_requirements, basic_context):
        """Remaining agents get critic roles."""
        assigner = DomainBasedRoleAssigner()
        team = agent_pool[:4]

        roles = assigner.assign_roles(team, security_requirements, basic_context)

        critic_roles = [r for r in roles.values() if "critic" in r]
        assert len(critic_roles) >= 1

    def test_security_critic_for_security_trait(self, agent_pool, basic_context):
        """Agent with security trait gets security_critic role."""
        assigner = DomainBasedRoleAssigner()

        # Create a team where non-proposer has security trait
        requirements = TaskRequirements(
            task_id="sec-critic-test",
            description="Test",
            primary_domain="performance",  # So claude isn't proposer
            min_agents=2,
            max_agents=3,
        )
        team = agent_pool[:3]  # claude, codex, gemini

        roles = assigner.assign_roles(team, requirements, basic_context)

        # Claude has security trait and shouldn't be proposer (not a performance expert)
        # Should get security_critic
        if roles.get("claude") not in ("proposer", "synthesizer"):
            assert roles["claude"] == "security_critic"

    def test_empty_team(self, basic_context):
        """Empty team returns empty roles."""
        assigner = DomainBasedRoleAssigner()
        requirements = TaskRequirements(
            task_id="empty-test",
            description="Test",
            primary_domain="general",
            min_agents=1,
            max_agents=3,
        )

        roles = assigner.assign_roles([], requirements, basic_context)

        assert roles == {}

    def test_single_agent_is_proposer(self, agent_pool, security_requirements, basic_context):
        """Single agent gets proposer role."""
        assigner = DomainBasedRoleAssigner()
        team = [agent_pool[0]]  # Just claude

        roles = assigner.assign_roles(team, security_requirements, basic_context)

        assert roles["claude"] == "proposer"

    def test_phase_roles_debate(self, agent_pool, security_requirements, basic_context):
        """Debate phase assigns all as proposers."""
        assigner = DomainBasedRoleAssigner()
        team = agent_pool[:3]

        roles = assigner.assign_roles(team, security_requirements, basic_context, phase="debate")

        # In debate phase, all get proposer or the default fallback
        for role in roles.values():
            assert role in ("proposer", "critic")

    def test_phase_roles_design(self, agent_pool, security_requirements, basic_context):
        """Design phase assigns gemini as lead."""
        assigner = DomainBasedRoleAssigner()
        team = agent_pool[:4]  # claude, codex, gemini, grok

        roles = assigner.assign_roles(team, security_requirements, basic_context, phase="design")

        assert roles["gemini"] == "design_lead"

    def test_phase_roles_implement(self, agent_pool, security_requirements, basic_context):
        """Implement phase assigns claude as implementer."""
        assigner = DomainBasedRoleAssigner()
        team = agent_pool[:3]  # claude, codex, gemini

        roles = assigner.assign_roles(team, security_requirements, basic_context, phase="implement")

        assert roles["claude"] == "implementer"

    def test_phase_roles_verify(self, agent_pool, security_requirements, basic_context):
        """Verify phase assigns codex as verification lead."""
        assigner = DomainBasedRoleAssigner()
        team = agent_pool[:3]  # claude, codex, gemini

        roles = assigner.assign_roles(team, security_requirements, basic_context, phase="verify")

        assert roles["codex"] == "verification_lead"

    def test_phase_roles_commit(self, agent_pool, security_requirements, basic_context):
        """Commit phase assigns all as reviewers."""
        assigner = DomainBasedRoleAssigner()
        team = agent_pool[:3]

        roles = assigner.assign_roles(team, security_requirements, basic_context, phase="commit")

        for role in roles.values():
            assert role == "reviewer"

    def test_unknown_phase_uses_fallback(self, agent_pool, security_requirements, basic_context):
        """Unknown phase uses participant as fallback."""
        assigner = DomainBasedRoleAssigner()
        team = agent_pool[:3]

        roles = assigner.assign_roles(
            team, security_requirements, basic_context, phase="unknown_phase"
        )

        for role in roles.values():
            assert role == "participant"


# =============================================================================
# SimpleRoleAssigner Tests
# =============================================================================


class TestSimpleRoleAssigner:
    """Tests for simple role assignment."""

    def test_assigner_properties(self):
        """Assigner has correct name and description."""
        assigner = SimpleRoleAssigner()
        assert assigner.name == "simple"
        assert "simple" in assigner.description.lower()

    def test_first_is_proposer(self, agent_pool, security_requirements, basic_context):
        """First agent is always proposer."""
        assigner = SimpleRoleAssigner()
        team = agent_pool[:3]

        roles = assigner.assign_roles(team, security_requirements, basic_context)

        assert roles[team[0].name] == "proposer"

    def test_last_is_synthesizer(self, agent_pool, security_requirements, basic_context):
        """Last agent is synthesizer when team > 1."""
        assigner = SimpleRoleAssigner()
        team = agent_pool[:3]

        roles = assigner.assign_roles(team, security_requirements, basic_context)

        assert roles[team[-1].name] == "synthesizer"

    def test_middle_are_critics(self, agent_pool, security_requirements, basic_context):
        """Middle agents are critics."""
        assigner = SimpleRoleAssigner()
        team = agent_pool[:4]

        roles = assigner.assign_roles(team, security_requirements, basic_context)

        # First is proposer, last is synthesizer, middle should be critics
        for agent in team[1:-1]:
            assert roles[agent.name] == "critic"

    def test_empty_team(self, basic_context):
        """Empty team returns empty roles."""
        assigner = SimpleRoleAssigner()
        requirements = TaskRequirements(
            task_id="empty-test",
            description="Test",
            primary_domain="general",
            min_agents=1,
            max_agents=3,
        )

        roles = assigner.assign_roles([], requirements, basic_context)

        assert roles == {}

    def test_single_agent(self, agent_pool, security_requirements, basic_context):
        """Single agent is proposer only."""
        assigner = SimpleRoleAssigner()
        team = [agent_pool[0]]

        roles = assigner.assign_roles(team, security_requirements, basic_context)

        assert len(roles) == 1
        assert roles[team[0].name] == "proposer"

    def test_two_agents(self, agent_pool, security_requirements, basic_context):
        """Two agents: proposer and synthesizer."""
        assigner = SimpleRoleAssigner()
        team = agent_pool[:2]

        roles = assigner.assign_roles(team, security_requirements, basic_context)

        assert roles[team[0].name] == "proposer"
        assert roles[team[1].name] == "synthesizer"

    def test_phase_parameter_ignored(self, agent_pool, security_requirements, basic_context):
        """Phase parameter is ignored in simple assigner."""
        assigner = SimpleRoleAssigner()
        team = agent_pool[:3]

        roles_no_phase = assigner.assign_roles(team, security_requirements, basic_context)
        roles_with_phase = assigner.assign_roles(
            team, security_requirements, basic_context, phase="verify"
        )

        # Should be the same regardless of phase
        assert roles_no_phase == roles_with_phase
