"""
Tests for TeamBuilder class in aragora.routing.team_builder.

Tests cover:
- Team selection with diversity optimization
- Role assignment for standard debates
- Hybrid role assignment for different phases
- Diversity calculation
- Selection history tracking
- Rationale generation
- Edge cases (empty teams, single agent)
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
import random

import pytest

from aragora.routing.selection import AgentProfile, TaskRequirements, TeamComposition
from aragora.routing.team_builder import PHASE_ROLES, TeamBuilder


# =============================================================================
# TestTeamBuilderInit - Initialization Tests
# =============================================================================


class TestTeamBuilderInit:
    """Tests for TeamBuilder initialization."""

    def test_default_initialization(self):
        """Should initialize with empty selection history."""
        builder = TeamBuilder()
        assert builder._selection_history == []

    def test_separate_instances(self):
        """Different instances should have separate histories."""
        builder1 = TeamBuilder()
        builder2 = TeamBuilder()

        builder1._selection_history.append({"task_id": "test"})

        assert len(builder1._selection_history) == 1
        assert len(builder2._selection_history) == 0


# =============================================================================
# TestTeamBuilderSelectDiverseTeam - Diversity-Aware Selection
# =============================================================================


class TestTeamBuilderSelectDiverseTeam:
    """Tests for select_diverse_team() method."""

    def test_returns_all_if_fewer_than_min(self):
        """Should return all agents if fewer than min_size."""
        builder = TeamBuilder()

        agents = [
            AgentProfile(name="a1", agent_type="claude"),
            AgentProfile(name="a2", agent_type="gemini"),
        ]
        scored = [(a, 0.8) for a in agents]

        result = builder.select_diverse_team(scored, min_size=5, max_size=10, diversity_pref=0.5)

        assert len(result) == 2
        assert all(a in result for a in agents)

    def test_respects_min_size(self):
        """Should select at least min_size agents."""
        builder = TeamBuilder()

        agents = [
            AgentProfile(name=f"a{i}", agent_type="claude", elo_rating=1500 - i * 50)
            for i in range(10)
        ]
        scored = [(a, 1.0 - i * 0.05) for i, a in enumerate(agents)]

        result = builder.select_diverse_team(scored, min_size=3, max_size=5, diversity_pref=0.5)

        assert len(result) >= 3

    def test_respects_max_size(self):
        """Should select at most max_size agents."""
        builder = TeamBuilder()

        agents = [AgentProfile(name=f"a{i}", agent_type="claude") for i in range(10)]
        scored = [(a, 0.8) for a in agents]

        result = builder.select_diverse_team(scored, min_size=2, max_size=4, diversity_pref=0.5)

        assert len(result) <= 4

    def test_low_diversity_prefers_top_scored(self):
        """With low diversity, should prefer highest scored."""
        builder = TeamBuilder()

        # Fix random seed for deterministic test
        random.seed(42)

        agents = [
            AgentProfile(name="best", agent_type="claude", elo_rating=1900),
            AgentProfile(name="good", agent_type="claude", elo_rating=1700),
            AgentProfile(name="ok", agent_type="claude", elo_rating=1500),
            AgentProfile(name="bad", agent_type="claude", elo_rating=1300),
        ]
        # Scores in descending order
        scored = [(agents[0], 0.95), (agents[1], 0.8), (agents[2], 0.6), (agents[3], 0.4)]

        result = builder.select_diverse_team(scored, min_size=2, max_size=2, diversity_pref=0.0)

        # Should pick the top 2 by score
        names = [a.name for a in result]
        assert "best" in names

    def test_high_diversity_prefers_different_types(self):
        """With high diversity, should prefer different agent types."""
        builder = TeamBuilder()

        # Fix random seed for deterministic test
        random.seed(42)

        agents = [
            AgentProfile(name="c1", agent_type="claude", elo_rating=1800),
            AgentProfile(name="c2", agent_type="claude", elo_rating=1750),
            AgentProfile(name="g1", agent_type="gemini", elo_rating=1700),
            AgentProfile(name="x1", agent_type="codex", elo_rating=1650),
        ]
        scored = [(a, 0.9 - i * 0.05) for i, a in enumerate(agents)]

        result = builder.select_diverse_team(scored, min_size=3, max_size=3, diversity_pref=1.0)

        # Should have multiple types
        types = set(a.agent_type for a in result)
        assert len(types) >= 2

    def test_considers_trait_diversity(self):
        """Should consider trait diversity when selecting."""
        builder = TeamBuilder()

        random.seed(42)

        agents = [
            AgentProfile(name="a1", agent_type="claude", traits=["thorough", "security"]),
            AgentProfile(name="a2", agent_type="claude", traits=["thorough", "security"]),
            AgentProfile(name="a3", agent_type="claude", traits=["creative", "fast"]),
        ]
        scored = [(a, 0.8) for a in agents]

        # Multiple calls should sometimes pick different agents
        # when diversity is high
        result = builder.select_diverse_team(scored, min_size=2, max_size=2, diversity_pref=1.0)

        assert len(result) == 2

    def test_empty_scored_list(self):
        """Should return empty list for empty input."""
        builder = TeamBuilder()

        result = builder.select_diverse_team([], min_size=2, max_size=5, diversity_pref=0.5)

        assert result == []


# =============================================================================
# TestTeamBuilderCalculateDiversity - Diversity Score Calculation
# =============================================================================


class TestTeamBuilderCalculateDiversity:
    """Tests for calculate_diversity() method."""

    def test_single_agent_returns_zero(self):
        """Single agent team should have zero diversity."""
        builder = TeamBuilder()

        team = [AgentProfile(name="solo", agent_type="claude")]

        result = builder.calculate_diversity(team)

        assert result == 0.0

    def test_empty_team_returns_zero(self):
        """Empty team should have zero diversity."""
        builder = TeamBuilder()

        result = builder.calculate_diversity([])

        assert result == 0.0

    def test_same_type_low_diversity(self):
        """Same-type team should have low diversity."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="c1", agent_type="claude", traits=[], elo_rating=1500),
            AgentProfile(name="c2", agent_type="claude", traits=[], elo_rating=1500),
        ]

        result = builder.calculate_diversity(team)

        # Should be low (type diversity = 0.5, trait = 0, elo = 0)
        assert result < 0.3

    def test_different_types_high_diversity(self):
        """Different-type team should have higher diversity."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="c1", agent_type="claude"),
            AgentProfile(name="g1", agent_type="gemini"),
            AgentProfile(name="x1", agent_type="codex"),
        ]

        result = builder.calculate_diversity(team)

        # Type diversity = 1.0 (3 types / 3 agents)
        assert result > 0.3

    def test_elo_spread_increases_diversity(self):
        """ELO spread should increase diversity score."""
        builder = TeamBuilder()

        # Same type, same traits, but different ELO
        narrow_team = [
            AgentProfile(name="a1", agent_type="claude", elo_rating=1500),
            AgentProfile(name="a2", agent_type="claude", elo_rating=1510),
        ]
        wide_team = [
            AgentProfile(name="a1", agent_type="claude", elo_rating=1200),
            AgentProfile(name="a2", agent_type="claude", elo_rating=1800),
        ]

        narrow_div = builder.calculate_diversity(narrow_team)
        wide_div = builder.calculate_diversity(wide_team)

        assert wide_div > narrow_div

    def test_traits_increase_diversity(self):
        """Different traits should increase diversity."""
        builder = TeamBuilder()

        same_traits = [
            AgentProfile(name="a1", agent_type="claude", traits=["a"], elo_rating=1500),
            AgentProfile(name="a2", agent_type="claude", traits=["a"], elo_rating=1500),
        ]
        diff_traits = [
            AgentProfile(name="a1", agent_type="claude", traits=["a", "b"], elo_rating=1500),
            AgentProfile(name="a2", agent_type="claude", traits=["c", "d"], elo_rating=1500),
        ]

        same_div = builder.calculate_diversity(same_traits)
        diff_div = builder.calculate_diversity(diff_traits)

        assert diff_div > same_div


# =============================================================================
# TestTeamBuilderAssignRoles - Standard Role Assignment
# =============================================================================


class TestTeamBuilderAssignRoles:
    """Tests for assign_roles() method."""

    def test_assigns_proposer_to_domain_expert(self):
        """Should assign proposer role to highest domain expert."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="expert", agent_type="claude", expertise={"backend": 0.9}),
            AgentProfile(name="novice", agent_type="gemini", expertise={"backend": 0.3}),
        ]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")

        roles = builder.assign_roles(team, req)

        assert roles["expert"] == "proposer"

    def test_assigns_synthesizer(self):
        """Should assign synthesizer to balanced agent."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="high", agent_type="claude", expertise={"backend": 0.9}),
            AgentProfile(name="mid", agent_type="gemini", expertise={"backend": 0.5}),
            AgentProfile(name="low", agent_type="codex", expertise={"backend": 0.2}),
        ]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")

        roles = builder.assign_roles(team, req)

        assert "synthesizer" in roles.values()

    def test_assigns_security_critic(self):
        """Should assign security_critic to agent with security trait that's not proposer/synthesizer."""
        builder = TeamBuilder()

        # Need 3+ agents so security agent isn't taken for proposer or synthesizer
        team = [
            AgentProfile(
                name="proposer", agent_type="claude", traits=[], expertise={"backend": 0.9}
            ),
            AgentProfile(
                name="sec", agent_type="gemini", traits=["security"], expertise={"backend": 0.3}
            ),
            AgentProfile(name="synth", agent_type="codex", traits=[], expertise={"backend": 0.5}),
        ]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")

        roles = builder.assign_roles(team, req)

        # Security trait agent should get security_critic (if not assigned proposer/synthesizer)
        assert roles["sec"] == "security_critic"

    def test_assigns_performance_critic(self):
        """Should assign performance_critic to agent with performance trait that's not proposer/synthesizer."""
        builder = TeamBuilder()

        # Need 3+ agents so performance agent isn't taken for proposer or synthesizer
        team = [
            AgentProfile(
                name="proposer", agent_type="claude", traits=[], expertise={"backend": 0.9}
            ),
            AgentProfile(
                name="perf", agent_type="gemini", traits=["performance"], expertise={"backend": 0.3}
            ),
            AgentProfile(name="synth", agent_type="codex", traits=[], expertise={"backend": 0.5}),
        ]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")

        roles = builder.assign_roles(team, req)

        # Performance trait agent should get performance_critic
        assert roles["perf"] == "performance_critic"

    def test_assigns_generic_critic(self):
        """Should assign generic critic to agents without specific traits."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="a", agent_type="claude", traits=[], expertise={"backend": 0.9}),
            AgentProfile(name="b", agent_type="gemini", traits=[], expertise={"backend": 0.7}),
            AgentProfile(name="c", agent_type="codex", traits=[], expertise={"backend": 0.5}),
        ]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")

        roles = builder.assign_roles(team, req)

        # At least one should be generic critic
        assert "critic" in roles.values()

    def test_empty_team_returns_empty_roles(self):
        """Should return empty dict for empty team."""
        builder = TeamBuilder()

        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")

        roles = builder.assign_roles([], req)

        assert roles == {}

    def test_all_agents_get_roles(self):
        """All agents should get a role assigned."""
        builder = TeamBuilder()

        team = [AgentProfile(name=f"a{i}", agent_type="claude") for i in range(5)]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")

        roles = builder.assign_roles(team, req)

        assert len(roles) == 5
        for agent in team:
            assert agent.name in roles


# =============================================================================
# TestTeamBuilderAssignHybridRoles - Phase-Specific Role Assignment
# =============================================================================


class TestTeamBuilderAssignHybridRoles:
    """Tests for assign_hybrid_roles() method."""

    def test_debate_phase_all_proposers(self):
        """In debate phase, all agents should be proposers."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="claude", agent_type="claude"),
            AgentProfile(name="gemini", agent_type="gemini"),
            AgentProfile(name="codex", agent_type="codex"),
        ]

        roles = builder.assign_hybrid_roles(team, "debate")

        for agent in team:
            assert roles[agent.name] == "proposer"

    def test_design_phase_roles(self):
        """In design phase, should assign specific roles."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="gemini", agent_type="gemini"),
            AgentProfile(name="claude", agent_type="claude"),
            AgentProfile(name="codex", agent_type="codex"),
            AgentProfile(name="grok", agent_type="grok"),
            AgentProfile(name="deepseek", agent_type="deepseek"),
        ]

        roles = builder.assign_hybrid_roles(team, "design")

        assert roles["gemini"] == "design_lead"
        assert roles["claude"] == "architecture_critic"
        assert roles["codex"] == "implementation_critic"
        assert roles["grok"] == "devil_advocate"
        assert roles["deepseek"] == "logic_validator"

    def test_implement_phase_roles(self):
        """In implement phase, Claude should be implementer."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="claude", agent_type="claude"),
            AgentProfile(name="gemini", agent_type="gemini"),
        ]

        roles = builder.assign_hybrid_roles(team, "implement")

        assert roles["claude"] == "implementer"
        assert roles["gemini"] == "advisor"

    def test_verify_phase_roles(self):
        """In verify phase, should assign verification roles."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="codex", agent_type="codex"),
            AgentProfile(name="grok", agent_type="grok"),
            AgentProfile(name="gemini", agent_type="gemini"),
            AgentProfile(name="claude", agent_type="claude"),
            AgentProfile(name="deepseek", agent_type="deepseek"),
        ]

        roles = builder.assign_hybrid_roles(team, "verify")

        assert roles["codex"] == "verification_lead"
        assert roles["grok"] == "quality_auditor"
        assert roles["gemini"] == "design_validator"
        assert roles["claude"] == "implementation_reviewer"
        assert roles["deepseek"] == "formal_verifier"

    def test_commit_phase_all_reviewers(self):
        """In commit phase, all agents should be reviewers."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="claude", agent_type="claude"),
            AgentProfile(name="gemini", agent_type="gemini"),
        ]

        roles = builder.assign_hybrid_roles(team, "commit")

        for agent in team:
            assert roles[agent.name] == "reviewer"

    def test_unknown_phase_uses_participant(self):
        """Unknown phase should use 'participant' as fallback."""
        builder = TeamBuilder()

        team = [AgentProfile(name="agent", agent_type="claude")]

        roles = builder.assign_hybrid_roles(team, "unknown_phase")

        assert roles["agent"] == "participant"

    def test_matches_agent_by_type(self):
        """Should match agents by agent_type field."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="my_claude_instance", agent_type="claude"),
        ]

        roles = builder.assign_hybrid_roles(team, "implement")

        assert roles["my_claude_instance"] == "implementer"

    def test_matches_agent_by_name_contains(self):
        """Should match agents if name contains agent type."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="claude-sonnet-3.5", agent_type="unknown"),
        ]

        roles = builder.assign_hybrid_roles(team, "implement")

        assert roles["claude-sonnet-3.5"] == "implementer"


# =============================================================================
# TestTeamBuilderGenerateRationale - Rationale Generation
# =============================================================================


class TestTeamBuilderGenerateRationale:
    """Tests for _generate_rationale() method."""

    def test_includes_team_size(self):
        """Rationale should mention team size."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="a1", agent_type="claude"),
            AgentProfile(name="a2", agent_type="gemini"),
        ]
        req = TaskRequirements(
            task_id="test-123", description="Test task", primary_domain="backend"
        )
        scored = [(a, 0.8) for a in team]

        rationale = builder._generate_rationale(team, req, scored)

        assert "2 agents" in rationale

    def test_includes_task_id(self):
        """Rationale should mention task ID."""
        builder = TeamBuilder()

        team = [AgentProfile(name="a1", agent_type="claude")]
        req = TaskRequirements(task_id="task-456", description="Test", primary_domain="backend")
        scored = [(a, 0.8) for a in team]

        rationale = builder._generate_rationale(team, req, scored)

        assert "task-456" in rationale

    def test_includes_primary_domain(self):
        """Rationale should mention primary domain."""
        builder = TeamBuilder()

        team = [AgentProfile(name="a1", agent_type="claude")]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="security")
        scored = [(a, 0.8) for a in team]

        rationale = builder._generate_rationale(team, req, scored)

        assert "security" in rationale

    def test_includes_agent_details(self):
        """Rationale should include agent details."""
        builder = TeamBuilder()

        team = [
            AgentProfile(
                name="expert",
                agent_type="claude",
                elo_rating=1800,
                expertise={"backend": 0.9},
            )
        ]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")
        scored = [(team[0], 0.9)]

        rationale = builder._generate_rationale(team, req, scored)

        assert "expert" in rationale
        assert "1800" in rationale

    def test_indicates_total_candidates(self):
        """Rationale should indicate total candidates considered."""
        builder = TeamBuilder()

        team = [AgentProfile(name="a1", agent_type="claude")]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")
        all_scored = [(AgentProfile(name=f"a{i}", agent_type="claude"), 0.8) for i in range(5)]

        rationale = builder._generate_rationale(team, req, all_scored)

        assert "5 candidates" in rationale


# =============================================================================
# TestTeamBuilderSelectTeam - Full Team Selection
# =============================================================================


class TestTeamBuilderSelectTeam:
    """Tests for select_team() method."""

    def test_returns_team_composition(self):
        """Should return a TeamComposition object."""
        builder = TeamBuilder()

        def score_fn(agent, req):
            return 0.8

        agents = [AgentProfile(name=f"a{i}", agent_type="claude") for i in range(5)]
        scored = [(a, 0.8) for a in agents]
        req = TaskRequirements(
            task_id="test-123",
            description="Test task",
            primary_domain="backend",
            min_agents=2,
            max_agents=3,
        )

        team = builder.select_team(scored, req, score_fn)

        assert isinstance(team, TeamComposition)
        assert team.task_id == "test-123"
        assert team.team_id == "team-test-123"
        assert len(team.agents) >= 2
        assert len(team.agents) <= 3

    def test_raises_on_empty_scored(self):
        """Should raise ValueError for empty scored list."""
        builder = TeamBuilder()

        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")

        with pytest.raises(ValueError, match="No available agents"):
            builder.select_team([], req, lambda a, r: 0.5)

    def test_records_to_history(self):
        """Should record selection to history."""
        builder = TeamBuilder()

        def score_fn(agent, req):
            return 0.8

        agents = [AgentProfile(name="a1", agent_type="claude")]
        scored = [(agents[0], 0.8)]
        req = TaskRequirements(
            task_id="history-test",
            description="Test",
            primary_domain="backend",
            min_agents=1,
        )

        builder.select_team(scored, req, score_fn)

        history = builder.get_selection_history()
        assert len(history) == 1
        assert history[0]["task_id"] == "history-test"

    def test_calculates_expected_quality(self):
        """Should calculate expected quality."""
        builder = TeamBuilder()

        def score_fn(agent, req):
            return 0.75

        agents = [AgentProfile(name="a1", agent_type="claude")]
        scored = [(agents[0], 0.75)]
        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
            min_agents=1,
        )

        team = builder.select_team(scored, req, score_fn)

        assert team.expected_quality == pytest.approx(0.75, rel=0.1)

    def test_calculates_expected_cost(self):
        """Should sum cost factors."""
        builder = TeamBuilder()

        agents = [
            AgentProfile(name="a1", agent_type="claude", cost_factor=1.5),
            AgentProfile(name="a2", agent_type="gemini", cost_factor=2.0),
        ]
        scored = [(a, 0.8) for a in agents]
        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
            min_agents=2,
            max_agents=2,
        )

        team = builder.select_team(scored, req, lambda a, r: 0.8)

        assert team.expected_cost == pytest.approx(3.5, rel=0.1)


# =============================================================================
# TestTeamBuilderSelectionHistory - History Tracking
# =============================================================================


class TestTeamBuilderSelectionHistory:
    """Tests for selection history functionality."""

    def test_record_selection(self):
        """Should record selection with all fields."""
        builder = TeamBuilder()

        builder.record_selection(
            task_id="task-001",
            selected=["a1", "a2"],
            result="success",
            confidence=0.85,
        )

        history = builder.get_selection_history()
        assert len(history) == 1
        assert history[0]["task_id"] == "task-001"
        assert history[0]["selected"] == ["a1", "a2"]
        assert history[0]["result"] == "success"
        assert history[0]["confidence"] == 0.85

    def test_record_selection_without_result(self):
        """Should record selection without result."""
        builder = TeamBuilder()

        builder.record_selection(task_id="task-001", selected=["a1"])

        history = builder.get_selection_history()
        assert "result" not in history[0]

    def test_get_selection_history_limit(self):
        """Should respect limit parameter."""
        builder = TeamBuilder()

        for i in range(10):
            builder.record_selection(task_id=f"task-{i:03d}", selected=["a1"])

        history = builder.get_selection_history(limit=3)

        assert len(history) == 3

    def test_get_selection_history_sorted_by_timestamp(self):
        """Should return history sorted by timestamp descending."""
        builder = TeamBuilder()

        builder._selection_history = [
            {"task_id": "old", "timestamp": "2024-01-01T00:00:00"},
            {"task_id": "new", "timestamp": "2024-12-01T00:00:00"},
            {"task_id": "mid", "timestamp": "2024-06-01T00:00:00"},
        ]

        history = builder.get_selection_history()

        assert history[0]["task_id"] == "new"
        assert history[1]["task_id"] == "mid"
        assert history[2]["task_id"] == "old"

    def test_get_selection_history_empty(self):
        """Should return empty list for no history."""
        builder = TeamBuilder()

        history = builder.get_selection_history()

        assert history == []


# =============================================================================
# TestPhaseRolesConstant - PHASE_ROLES Configuration
# =============================================================================


class TestPhaseRolesConstant:
    """Tests for PHASE_ROLES configuration constant."""

    def test_has_required_phases(self):
        """Should have all required phases."""
        required_phases = ["debate", "design", "implement", "verify", "commit"]
        for phase in required_phases:
            assert phase in PHASE_ROLES

    def test_phase_structure(self):
        """Each phase should have (role_map, fallback) structure."""
        for phase, config in PHASE_ROLES.items():
            assert isinstance(config, tuple), f"{phase} should be a tuple"
            assert len(config) == 2, f"{phase} should have 2 elements"
            role_map, fallback = config
            assert isinstance(role_map, dict), f"{phase} role_map should be dict"
            assert isinstance(fallback, str), f"{phase} fallback should be str"

    def test_design_phase_has_lead_role(self):
        """Design phase should have a design_lead role."""
        role_map, _ = PHASE_ROLES["design"]
        assert "design_lead" in role_map.values()

    def test_verify_phase_has_verification_lead(self):
        """Verify phase should have a verification_lead role."""
        role_map, _ = PHASE_ROLES["verify"]
        assert "verification_lead" in role_map.values()

    def test_implement_phase_has_implementer(self):
        """Implement phase should have an implementer role."""
        role_map, _ = PHASE_ROLES["implement"]
        assert "implementer" in role_map.values()


# =============================================================================
# TestTeamBuilderEdgeCases - Edge Cases
# =============================================================================


class TestTeamBuilderEdgeCases:
    """Tests for edge cases in TeamBuilder."""

    def test_single_agent_team(self):
        """Should handle single agent team."""
        builder = TeamBuilder()

        team = [AgentProfile(name="solo", agent_type="claude", expertise={"backend": 0.9})]
        req = TaskRequirements(task_id="test", description="Test", primary_domain="backend")

        roles = builder.assign_roles(team, req)

        assert roles["solo"] == "proposer"

    def test_agents_with_no_traits(self):
        """Should handle agents without traits."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="a1", agent_type="claude", traits=[]),
            AgentProfile(name="a2", agent_type="gemini", traits=[]),
        ]

        diversity = builder.calculate_diversity(team)

        # Should still calculate diversity
        assert isinstance(diversity, float)

    def test_agents_with_same_elo(self):
        """Should handle agents with identical ELO."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="a1", agent_type="claude", elo_rating=1500),
            AgentProfile(name="a2", agent_type="gemini", elo_rating=1500),
        ]

        diversity = builder.calculate_diversity(team)

        # ELO diversity component should be 0
        assert isinstance(diversity, float)

    def test_unknown_agent_type_in_hybrid_roles(self):
        """Should handle unknown agent types in hybrid role assignment."""
        builder = TeamBuilder()

        team = [
            AgentProfile(name="custom_agent", agent_type="unknown_type"),
        ]

        roles = builder.assign_hybrid_roles(team, "design")

        # Should get the fallback role for design phase
        assert roles["custom_agent"] == "critic"
