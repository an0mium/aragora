"""
Tests for routing/selection module.

Tests the AgentSelector, AgentProfile, TaskRequirements, and TeamComposition
classes that handle agent selection for debates.
"""

import pytest
from dataclasses import asdict
from datetime import datetime
from unittest.mock import MagicMock, patch

from aragora.routing.selection import (
    AgentProfile,
    TaskRequirements,
    TeamComposition,
    AgentSelector,
)


# =============================================================================
# AgentProfile Tests
# =============================================================================


class TestAgentProfile:
    """Tests for AgentProfile dataclass."""

    def test_default_values(self):
        """AgentProfile should have sensible defaults."""
        profile = AgentProfile(name="test-agent", agent_type="claude")

        assert profile.name == "test-agent"
        assert profile.agent_type == "claude"
        assert profile.elo_rating == 1500
        assert profile.domain_ratings == {}
        assert profile.expertise == {}
        assert profile.traits == []
        assert profile.availability == 1.0
        assert profile.cost_factor == 1.0
        assert profile.latency_ms == 1000
        assert profile.success_rate == 0.8
        assert profile.probe_score == 1.0
        assert profile.has_critical_probes is False
        assert profile.calibration_score == 1.0
        assert profile.brier_score == 0.0
        assert profile.is_overconfident is False

    def test_overall_score_calculation(self):
        """overall_score should combine metrics correctly."""
        profile = AgentProfile(
            name="test",
            agent_type="claude",
            elo_rating=1600,  # Above average
            success_rate=0.9,  # High
            probe_score=0.8,  # Good reliability
            calibration_score=0.7,  # Decent calibration
            latency_ms=500,  # Fast
            cost_factor=0.5,  # Cheap
        )

        score = profile.overall_score
        assert 0 < score < 1
        # Higher values should yield higher scores
        assert score > 0.5

    def test_overall_score_penalties(self):
        """overall_score should apply penalties for critical issues."""
        # Base profile
        base = AgentProfile(
            name="base",
            agent_type="claude",
            elo_rating=1600,
            success_rate=0.9,
            probe_score=0.8,
        )
        base_score = base.overall_score

        # With critical probes - 30% penalty
        critical_probes = AgentProfile(
            name="critical",
            agent_type="claude",
            elo_rating=1600,
            success_rate=0.9,
            probe_score=0.8,
            has_critical_probes=True,
        )
        assert critical_probes.overall_score < base_score
        assert critical_probes.overall_score == pytest.approx(base_score * 0.7, rel=0.01)

        # With overconfidence - 10% penalty
        overconfident = AgentProfile(
            name="overconfident",
            agent_type="claude",
            elo_rating=1600,
            success_rate=0.9,
            probe_score=0.8,
            is_overconfident=True,
        )
        assert overconfident.overall_score < base_score
        assert overconfident.overall_score == pytest.approx(base_score * 0.9, rel=0.01)

    def test_overall_score_combined_penalties(self):
        """overall_score should stack penalties correctly."""
        base = AgentProfile(name="base", agent_type="claude")
        base_score = base.overall_score

        combined = AgentProfile(
            name="combined",
            agent_type="claude",
            has_critical_probes=True,
            is_overconfident=True,
        )
        # 0.7 * 0.9 = 0.63
        expected_penalty = 0.7 * 0.9
        assert combined.overall_score == pytest.approx(base_score * expected_penalty, rel=0.01)

    def test_overall_score_edge_cases(self):
        """overall_score should handle edge case values."""
        # Minimum values
        low = AgentProfile(
            name="low",
            agent_type="claude",
            elo_rating=0,
            success_rate=0,
            probe_score=0,
            calibration_score=0,
            latency_ms=10000,  # Very slow
            cost_factor=5.0,  # Very expensive
        )
        assert low.overall_score >= 0

        # Maximum values - note: overall_score is not clamped,
        # so extremely high ELO can result in scores > 1.0
        high = AgentProfile(
            name="high",
            agent_type="claude",
            elo_rating=3000,
            success_rate=1.0,
            probe_score=1.0,
            calibration_score=1.0,
            latency_ms=0,
            cost_factor=0,
        )
        # With ELO 3000, score can exceed 1.0 due to ELO normalization
        # (3000/2000 * 0.30 = 0.45 just from ELO component)
        assert high.overall_score > 1.0  # Actually exceeds 1.0 with extreme values

        # Normal high values stay within reasonable range
        normal_high = AgentProfile(
            name="normal_high",
            agent_type="claude",
            elo_rating=2000,
            success_rate=1.0,
            probe_score=1.0,
            calibration_score=1.0,
            latency_ms=100,
            cost_factor=0.5,
        )
        # Normal high ELO agent scores close to 1.0
        assert 0.9 <= normal_high.overall_score <= 1.1


# =============================================================================
# TaskRequirements Tests
# =============================================================================


class TestTaskRequirements:
    """Tests for TaskRequirements dataclass."""

    def test_default_values(self):
        """TaskRequirements should have sensible defaults."""
        req = TaskRequirements(
            task_id="task-1",
            description="Test task",
            primary_domain="backend",
        )

        assert req.task_id == "task-1"
        assert req.description == "Test task"
        assert req.primary_domain == "backend"
        assert req.secondary_domains == []
        assert req.required_traits == []
        assert req.min_agents == 2
        assert req.max_agents == 5
        assert req.quality_priority == 0.5
        assert req.diversity_preference == 0.5

    def test_custom_values(self):
        """TaskRequirements should accept custom values."""
        req = TaskRequirements(
            task_id="task-2",
            description="Complex security task",
            primary_domain="security",
            secondary_domains=["backend", "devops"],
            required_traits=["thorough", "security-focused"],
            min_agents=3,
            max_agents=7,
            quality_priority=0.9,
            diversity_preference=0.3,
        )

        assert req.primary_domain == "security"
        assert len(req.secondary_domains) == 2
        assert "thorough" in req.required_traits
        assert req.min_agents == 3
        assert req.quality_priority == 0.9


# =============================================================================
# AgentSelector - Pool Management Tests
# =============================================================================


class TestAgentSelectorPoolManagement:
    """Tests for AgentSelector agent pool management."""

    def test_register_agent(self):
        """register_agent should add agent to pool."""
        selector = AgentSelector()
        profile = AgentProfile(name="claude-1", agent_type="claude")

        selector.register_agent(profile)

        assert "claude-1" in selector.agent_pool
        assert selector.agent_pool["claude-1"] == profile

    def test_register_multiple_agents(self):
        """register_agent should handle multiple agents."""
        selector = AgentSelector()

        for i in range(5):
            selector.register_agent(AgentProfile(name=f"agent-{i}", agent_type="claude"))

        assert len(selector.agent_pool) == 5

    def test_register_agent_overwrites(self):
        """register_agent should overwrite existing agent."""
        selector = AgentSelector()

        old = AgentProfile(name="agent", agent_type="claude", elo_rating=1500)
        new = AgentProfile(name="agent", agent_type="claude", elo_rating=1800)

        selector.register_agent(old)
        selector.register_agent(new)

        assert selector.agent_pool["agent"].elo_rating == 1800

    def test_remove_agent(self):
        """remove_agent should remove agent from pool."""
        selector = AgentSelector()
        selector.register_agent(AgentProfile(name="agent", agent_type="claude"))

        selector.remove_agent("agent")

        assert "agent" not in selector.agent_pool

    def test_remove_agent_not_found(self):
        """remove_agent should handle missing agent gracefully."""
        selector = AgentSelector()

        # Should not raise
        selector.remove_agent("nonexistent")

    def test_remove_agent_also_removes_from_bench(self):
        """remove_agent should also remove from bench."""
        selector = AgentSelector()
        selector.register_agent(AgentProfile(name="agent", agent_type="claude"))
        selector.move_to_bench("agent")

        selector.remove_agent("agent")

        assert "agent" not in selector.agent_pool
        assert "agent" not in selector.bench


class TestAgentSelectorBenchManagement:
    """Tests for AgentSelector bench (probation) system."""

    def test_move_to_bench(self):
        """move_to_bench should add agent to bench."""
        selector = AgentSelector()
        selector.register_agent(AgentProfile(name="agent", agent_type="claude"))

        selector.move_to_bench("agent")

        assert "agent" in selector.bench

    def test_move_to_bench_not_in_pool(self):
        """move_to_bench should not add agent not in pool."""
        selector = AgentSelector()

        selector.move_to_bench("nonexistent")

        assert "nonexistent" not in selector.bench

    def test_move_to_bench_idempotent(self):
        """move_to_bench should not duplicate entries."""
        selector = AgentSelector()
        selector.register_agent(AgentProfile(name="agent", agent_type="claude"))

        selector.move_to_bench("agent")
        selector.move_to_bench("agent")

        assert selector.bench.count("agent") == 1

    def test_promote_from_bench(self):
        """promote_from_bench should remove from bench."""
        selector = AgentSelector()
        selector.register_agent(AgentProfile(name="agent", agent_type="claude"))
        selector.move_to_bench("agent")

        selector.promote_from_bench("agent")

        assert "agent" not in selector.bench
        assert "agent" in selector.agent_pool

    def test_promote_from_bench_not_on_bench(self):
        """promote_from_bench should handle agent not on bench."""
        selector = AgentSelector()
        selector.register_agent(AgentProfile(name="agent", agent_type="claude"))

        # Should not raise
        selector.promote_from_bench("agent")

    def test_bench_agents_excluded_from_selection(self):
        """Agents on bench should not be selected for teams."""
        selector = AgentSelector()

        # Add 3 agents
        for name in ["active-1", "active-2", "benched"]:
            selector.register_agent(AgentProfile(name=name, agent_type="claude"))

        selector.move_to_bench("benched")

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
            min_agents=2,
            max_agents=3,
        )

        team = selector.select_team(req)

        assert "benched" not in [a.name for a in team.agents]


# =============================================================================
# AgentSelector - Score Calculation Tests
# =============================================================================


class TestAgentSelectorScoring:
    """Tests for AgentSelector scoring logic."""

    def test_score_for_task_basic(self):
        """_score_for_task should return score between 0 and 1."""
        selector = AgentSelector()

        agent = AgentProfile(
            name="test",
            agent_type="claude",
            elo_rating=1600,
            expertise={"backend": 0.8},
        )

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        score = selector._score_for_task(agent, req)

        assert 0 <= score <= 1

    def test_score_for_task_domain_expertise(self):
        """_score_for_task should prefer domain experts."""
        selector = AgentSelector()

        expert = AgentProfile(
            name="expert",
            agent_type="claude",
            expertise={"backend": 0.9},
        )
        generalist = AgentProfile(
            name="generalist",
            agent_type="claude",
            expertise={"backend": 0.3},
        )

        req = TaskRequirements(
            task_id="test",
            description="Backend task",
            primary_domain="backend",
        )

        expert_score = selector._score_for_task(expert, req)
        generalist_score = selector._score_for_task(generalist, req)

        assert expert_score > generalist_score

    def test_score_for_task_elo_impact(self):
        """_score_for_task should consider ELO rating."""
        selector = AgentSelector()

        high_elo = AgentProfile(name="high", agent_type="claude", elo_rating=1800)
        low_elo = AgentProfile(name="low", agent_type="claude", elo_rating=1200)

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        high_score = selector._score_for_task(high_elo, req)
        low_score = selector._score_for_task(low_elo, req)

        assert high_score > low_score

    def test_score_for_task_trait_matching(self):
        """_score_for_task should reward matching traits."""
        selector = AgentSelector()

        with_traits = AgentProfile(
            name="with",
            agent_type="claude",
            traits=["security", "thorough"],
        )
        without_traits = AgentProfile(
            name="without",
            agent_type="claude",
            traits=[],
        )

        req = TaskRequirements(
            task_id="test",
            description="Security audit",
            primary_domain="security",
            required_traits=["security", "thorough"],
        )

        with_score = selector._score_for_task(with_traits, req)
        without_score = selector._score_for_task(without_traits, req)

        assert with_score > without_score

    def test_score_for_task_quality_priority(self):
        """_score_for_task should adjust for quality priority."""
        selector = AgentSelector()

        fast_cheap = AgentProfile(
            name="fast",
            agent_type="claude",
            success_rate=0.6,
            latency_ms=200,
            cost_factor=0.3,
        )

        quality_req = TaskRequirements(
            task_id="quality",
            description="Critical task",
            primary_domain="backend",
            quality_priority=0.9,
        )

        speed_req = TaskRequirements(
            task_id="speed",
            description="Quick task",
            primary_domain="backend",
            quality_priority=0.1,
        )

        quality_score = selector._score_for_task(fast_cheap, quality_req)
        speed_score = selector._score_for_task(fast_cheap, speed_req)

        # Fast/cheap agent should score better on speed-priority task
        assert speed_score > quality_score

    def test_score_for_task_secondary_domains(self):
        """_score_for_task should consider secondary domains."""
        selector = AgentSelector()

        multi_domain = AgentProfile(
            name="multi",
            agent_type="claude",
            expertise={"backend": 0.5, "devops": 0.8, "security": 0.7},
        )
        single_domain = AgentProfile(
            name="single",
            agent_type="claude",
            expertise={"backend": 0.5},
        )

        req = TaskRequirements(
            task_id="test",
            description="Complex task",
            primary_domain="backend",
            secondary_domains=["devops", "security"],
        )

        multi_score = selector._score_for_task(multi_domain, req)
        single_score = selector._score_for_task(single_domain, req)

        assert multi_score > single_score


class TestAgentSelectorProbeAdjustment:
    """Tests for probe-based score adjustments."""

    def test_get_probe_adjusted_score_default(self):
        """get_probe_adjusted_score should return base score with no probe data."""
        selector = AgentSelector()

        adjusted = selector.get_probe_adjusted_score("unknown", 0.8)

        # Should apply default adjustment (0.5 + 1.0 * 0.5 = 1.0)
        assert adjusted == pytest.approx(0.8, rel=0.1)

    def test_get_probe_adjusted_score_from_profile(self):
        """get_probe_adjusted_score should use agent profile data."""
        selector = AgentSelector()
        selector.register_agent(
            AgentProfile(
                name="vulnerable",
                agent_type="claude",
                probe_score=0.5,
            )
        )

        adjusted = selector.get_probe_adjusted_score("vulnerable", 0.8)

        # 0.5 + 0.5 * 0.5 = 0.75 adjustment factor
        expected = 0.8 * 0.75
        assert adjusted == pytest.approx(expected, rel=0.1)

    def test_get_probe_adjusted_score_with_critical(self):
        """get_probe_adjusted_score should apply extra penalty for critical issues."""
        selector = AgentSelector()
        selector.register_agent(
            AgentProfile(
                name="critical",
                agent_type="claude",
                probe_score=0.5,
                has_critical_probes=True,
            )
        )

        base_adjusted = selector.get_probe_adjusted_score("critical", 0.8)

        # 0.5 + 0.5 * 0.5 = 0.75, then * 0.8 for critical = 0.6
        expected = 0.8 * 0.75 * 0.8
        assert base_adjusted == pytest.approx(expected, rel=0.1)

    def test_get_probe_adjusted_score_from_filter(self):
        """get_probe_adjusted_score should prefer probe filter data."""
        mock_filter = MagicMock()
        mock_profile = MagicMock()
        mock_profile.probe_score = 0.3
        mock_profile.total_probes = 10
        mock_profile.has_critical_issues.return_value = False
        mock_filter.get_agent_profile.return_value = mock_profile

        selector = AgentSelector(probe_filter=mock_filter)

        adjusted = selector.get_probe_adjusted_score("test", 0.8)

        # 0.5 + 0.3 * 0.5 = 0.65 adjustment factor
        expected = 0.8 * 0.65
        assert adjusted == pytest.approx(expected, rel=0.1)


class TestAgentSelectorCalibrationAdjustment:
    """Tests for calibration-based score adjustments."""

    def test_get_calibration_adjusted_score_default(self):
        """get_calibration_adjusted_score should return near-base with no data."""
        selector = AgentSelector()

        adjusted = selector.get_calibration_adjusted_score("unknown", 0.8)

        # Default calibration_score = 1.0, adjustment = 0.7 + 1.0 * 0.3 = 1.0
        assert adjusted == pytest.approx(0.8, rel=0.1)

    def test_get_calibration_adjusted_score_from_profile(self):
        """get_calibration_adjusted_score should use agent profile data."""
        selector = AgentSelector()
        selector.register_agent(
            AgentProfile(
                name="poorly_calibrated",
                agent_type="claude",
                calibration_score=0.5,
            )
        )

        adjusted = selector.get_calibration_adjusted_score("poorly_calibrated", 0.8)

        # 0.7 + 0.5 * 0.3 = 0.85 adjustment factor
        expected = 0.8 * 0.85
        assert adjusted == pytest.approx(expected, rel=0.1)

    def test_get_calibration_adjusted_score_overconfident(self):
        """get_calibration_adjusted_score should penalize overconfidence."""
        selector = AgentSelector()
        selector.register_agent(
            AgentProfile(
                name="overconfident",
                agent_type="claude",
                calibration_score=0.8,
                is_overconfident=True,
            )
        )

        adjusted = selector.get_calibration_adjusted_score("overconfident", 0.8)

        # 0.7 + 0.8 * 0.3 = 0.94, then * 0.9 for overconfident
        expected = 0.8 * 0.94 * 0.9
        assert adjusted == pytest.approx(expected, rel=0.1)


# =============================================================================
# AgentSelector - Team Selection Tests
# =============================================================================


class TestAgentSelectorTeamSelection:
    """Tests for team selection logic."""

    def test_select_team_minimum_agents(self):
        """select_team should select at least min_agents."""
        selector = AgentSelector()
        for i in range(5):
            selector.register_agent(AgentProfile(name=f"agent-{i}", agent_type="claude"))

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
            min_agents=3,
            max_agents=5,
        )

        team = selector.select_team(req)

        assert len(team.agents) >= 3

    def test_select_team_maximum_agents(self):
        """select_team should not exceed max_agents."""
        selector = AgentSelector()
        for i in range(10):
            selector.register_agent(AgentProfile(name=f"agent-{i}", agent_type="claude"))

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
            min_agents=2,
            max_agents=4,
        )

        team = selector.select_team(req)

        assert len(team.agents) <= 4

    def test_select_team_no_agents_raises(self):
        """select_team should raise if no agents available."""
        selector = AgentSelector()

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        with pytest.raises(ValueError, match="No available agents"):
            selector.select_team(req)

    def test_select_team_excludes_specified(self):
        """select_team should exclude specified agents."""
        selector = AgentSelector()
        for name in ["a", "b", "c", "d"]:
            selector.register_agent(AgentProfile(name=name, agent_type="claude"))

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
            min_agents=2,
        )

        team = selector.select_team(req, exclude=["a", "b"])

        agent_names = [a.name for a in team.agents]
        assert "a" not in agent_names
        assert "b" not in agent_names

    def test_select_team_returns_team_composition(self):
        """select_team should return proper TeamComposition."""
        selector = AgentSelector()
        for i in range(3):
            selector.register_agent(AgentProfile(name=f"agent-{i}", agent_type="claude"))

        req = TaskRequirements(
            task_id="test-123",
            description="Test task",
            primary_domain="backend",
            min_agents=2,
        )

        team = selector.select_team(req)

        assert isinstance(team, TeamComposition)
        assert team.task_id == "test-123"
        assert team.team_id == "team-test-123"
        assert len(team.agents) >= 2
        assert len(team.roles) == len(team.agents)
        assert isinstance(team.expected_quality, float)
        assert isinstance(team.expected_cost, float)
        assert isinstance(team.diversity_score, float)
        assert isinstance(team.rationale, str)

    def test_select_team_records_history(self):
        """select_team should record selection in history."""
        selector = AgentSelector()
        for i in range(3):
            selector.register_agent(AgentProfile(name=f"agent-{i}", agent_type="claude"))

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        selector.select_team(req)

        history = selector.get_selection_history()
        assert len(history) == 1
        assert history[0]["task_id"] == "test"

    def test_select_team_prefers_higher_scored(self):
        """select_team should prefer higher-scored agents."""
        selector = AgentSelector()

        # Low ELO agent
        selector.register_agent(
            AgentProfile(
                name="low",
                agent_type="claude",
                elo_rating=1200,
            )
        )
        # High ELO agent
        selector.register_agent(
            AgentProfile(
                name="high",
                agent_type="claude",
                elo_rating=1800,
            )
        )
        # Medium ELO agent
        selector.register_agent(
            AgentProfile(
                name="medium",
                agent_type="claude",
                elo_rating=1500,
            )
        )

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
            min_agents=2,
            max_agents=2,
            diversity_preference=0.0,  # Pure quality selection
        )

        team = selector.select_team(req)
        agent_names = [a.name for a in team.agents]

        # High and medium should be selected
        assert "high" in agent_names
        assert "low" not in agent_names


class TestAgentSelectorDiverseTeamSelection:
    """Tests for diversity-aware team selection."""

    def test_diversity_prefers_different_types(self):
        """_select_diverse_team should prefer type diversity."""
        selector = AgentSelector()

        selector.register_agent(AgentProfile(name="claude-1", agent_type="claude", elo_rating=1600))
        selector.register_agent(AgentProfile(name="claude-2", agent_type="claude", elo_rating=1550))
        selector.register_agent(AgentProfile(name="gemini", agent_type="gemini", elo_rating=1500))
        selector.register_agent(AgentProfile(name="codex", agent_type="codex", elo_rating=1450))

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
            min_agents=3,
            max_agents=3,
            diversity_preference=0.9,  # High diversity
        )

        team = selector.select_team(req)
        agent_types = [a.agent_type for a in team.agents]

        # Should have at least 2 different types
        assert len(set(agent_types)) >= 2

    def test_low_diversity_prefers_quality(self):
        """With low diversity preference, should pick highest quality."""
        selector = AgentSelector()

        # All same type but different ratings
        selector.register_agent(AgentProfile(name="best", agent_type="claude", elo_rating=1900))
        selector.register_agent(AgentProfile(name="good", agent_type="claude", elo_rating=1700))
        selector.register_agent(AgentProfile(name="ok", agent_type="claude", elo_rating=1500))
        selector.register_agent(AgentProfile(name="bad", agent_type="claude", elo_rating=1300))

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
            min_agents=2,
            max_agents=2,
            diversity_preference=0.0,  # Pure quality
        )

        team = selector.select_team(req)
        agent_names = [a.name for a in team.agents]

        assert "best" in agent_names
        assert "bad" not in agent_names


# =============================================================================
# AgentSelector - Role Assignment Tests
# =============================================================================


class TestAgentSelectorRoleAssignment:
    """Tests for role assignment logic."""

    def test_assign_roles_proposer(self):
        """_assign_roles should assign proposer to domain expert."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="expert", agent_type="claude", expertise={"backend": 0.9}),
            AgentProfile(name="novice", agent_type="gemini", expertise={"backend": 0.3}),
        ]

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        roles = selector._assign_roles(team, req)

        assert roles["expert"] == "proposer"

    def test_assign_roles_synthesizer(self):
        """_assign_roles should assign synthesizer to balanced agent."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="high", agent_type="claude", expertise={"backend": 0.9}),
            AgentProfile(name="mid", agent_type="gemini", expertise={"backend": 0.5}),
            AgentProfile(name="low", agent_type="codex", expertise={"backend": 0.2}),
        ]

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        roles = selector._assign_roles(team, req)

        # "mid" has overall_score closest to 0.5
        assert "synthesizer" in roles.values()

    def test_assign_roles_critics(self):
        """_assign_roles should assign critic roles."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="a", agent_type="claude", expertise={"backend": 0.9}),
            AgentProfile(name="b", agent_type="gemini", expertise={"backend": 0.5}),
            AgentProfile(name="c", agent_type="codex", traits=["security"]),
            AgentProfile(name="d", agent_type="grok", traits=["performance"]),
        ]

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        roles = selector._assign_roles(team, req)

        # All agents should have roles
        assert len(roles) == 4
        # Security trait should get security_critic
        assert roles["c"] == "security_critic"
        # Performance trait should get performance_critic
        assert roles["d"] == "performance_critic"

    def test_assign_roles_empty_team(self):
        """_assign_roles should handle empty team."""
        selector = AgentSelector()

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        roles = selector._assign_roles([], req)

        assert roles == {}


class TestAgentSelectorHybridRoles:
    """Tests for hybrid architecture role assignment."""

    def test_assign_hybrid_roles_debate_phase(self):
        """assign_hybrid_roles should make all agents proposers in debate."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="claude", agent_type="claude"),
            AgentProfile(name="gemini", agent_type="gemini"),
            AgentProfile(name="codex", agent_type="codex"),
        ]

        roles = selector.assign_hybrid_roles(team, "debate")

        for agent in team:
            assert roles[agent.name] == "proposer"

    def test_assign_hybrid_roles_design_phase(self):
        """assign_hybrid_roles should make gemini design lead."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="claude", agent_type="claude"),
            AgentProfile(name="gemini", agent_type="gemini"),
            AgentProfile(name="codex", agent_type="codex"),
        ]

        roles = selector.assign_hybrid_roles(team, "design")

        assert roles["gemini"] == "design_lead"
        assert roles["claude"] == "architecture_critic"
        assert roles["codex"] == "implementation_critic"

    def test_assign_hybrid_roles_implement_phase(self):
        """assign_hybrid_roles should make claude implementer."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="claude", agent_type="claude"),
            AgentProfile(name="gemini", agent_type="gemini"),
        ]

        roles = selector.assign_hybrid_roles(team, "implement")

        assert roles["claude"] == "implementer"
        assert roles["gemini"] == "advisor"

    def test_assign_hybrid_roles_verify_phase(self):
        """assign_hybrid_roles should make codex verification lead."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="claude", agent_type="claude"),
            AgentProfile(name="codex", agent_type="codex"),
            AgentProfile(name="grok", agent_type="grok"),
        ]

        roles = selector.assign_hybrid_roles(team, "verify")

        assert roles["codex"] == "verification_lead"
        assert roles["grok"] == "quality_auditor"
        assert roles["claude"] == "implementation_reviewer"

    def test_assign_hybrid_roles_unknown_phase(self):
        """assign_hybrid_roles should handle unknown phase."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="agent", agent_type="claude"),
        ]

        roles = selector.assign_hybrid_roles(team, "unknown_phase")

        assert roles["agent"] == "participant"


# =============================================================================
# AgentSelector - Diversity Calculation Tests
# =============================================================================


class TestAgentSelectorDiversity:
    """Tests for diversity calculation."""

    def test_calculate_diversity_single_agent(self):
        """_calculate_diversity should return 0 for single agent."""
        selector = AgentSelector()

        team = [AgentProfile(name="solo", agent_type="claude")]

        diversity = selector._calculate_diversity(team)

        assert diversity == 0.0

    def test_calculate_diversity_same_type(self):
        """_calculate_diversity should be low for same-type team."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="c1", agent_type="claude", traits=["a"], elo_rating=1500),
            AgentProfile(name="c2", agent_type="claude", traits=["a"], elo_rating=1500),
        ]

        diversity = selector._calculate_diversity(team)

        # All same - low diversity
        assert diversity < 0.3

    def test_calculate_diversity_different_types(self):
        """_calculate_diversity should be high for diverse team."""
        selector = AgentSelector()

        team = [
            AgentProfile(name="c", agent_type="claude", traits=["thorough"], elo_rating=1800),
            AgentProfile(name="g", agent_type="gemini", traits=["creative"], elo_rating=1500),
            AgentProfile(name="x", agent_type="codex", traits=["fast"], elo_rating=1200),
        ]

        diversity = selector._calculate_diversity(team)

        # Different types, traits, and ELOs - high diversity
        assert diversity > 0.5


# =============================================================================
# AgentSelector - History and Analytics Tests
# =============================================================================


class TestAgentSelectorHistory:
    """Tests for selection history tracking."""

    def test_get_selection_history_empty(self):
        """get_selection_history should return empty list initially."""
        selector = AgentSelector()

        history = selector.get_selection_history()

        assert history == []

    def test_get_selection_history_limit(self):
        """get_selection_history should respect limit."""
        selector = AgentSelector()

        # Add history entries manually
        for i in range(10):
            selector._selection_history.append(
                {
                    "task_id": f"task-{i}",
                    "selected": ["agent"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

        history = selector.get_selection_history(limit=5)

        assert len(history) == 5

    def test_get_selection_history_sorted(self):
        """get_selection_history should be sorted by timestamp descending."""
        selector = AgentSelector()

        selector._selection_history = [
            {"task_id": "old", "timestamp": "2024-01-01T00:00:00"},
            {"task_id": "new", "timestamp": "2024-12-01T00:00:00"},
            {"task_id": "mid", "timestamp": "2024-06-01T00:00:00"},
        ]

        history = selector.get_selection_history()

        assert history[0]["task_id"] == "new"
        assert history[1]["task_id"] == "mid"
        assert history[2]["task_id"] == "old"


class TestAgentSelectorTeamAnalytics:
    """Tests for team combination analytics."""

    def test_get_best_team_combinations_empty(self):
        """get_best_team_combinations should return empty for no history."""
        selector = AgentSelector()

        results = selector.get_best_team_combinations()

        assert results == []

    def test_get_best_team_combinations_min_debates(self):
        """get_best_team_combinations should respect min_debates."""
        selector = AgentSelector()

        # Team with 2 debates (below threshold)
        selector._selection_history = [
            {"selected": ["a", "b"], "result": "success"},
            {"selected": ["a", "b"], "result": "success"},
        ]

        results = selector.get_best_team_combinations(min_debates=3)

        assert results == []

    def test_get_best_team_combinations_success_rate(self):
        """get_best_team_combinations should calculate success rate."""
        selector = AgentSelector()

        # Team with 75% success rate
        selector._selection_history = [
            {"selected": ["a", "b"], "result": "success"},
            {"selected": ["a", "b"], "result": "success"},
            {"selected": ["a", "b"], "result": "success"},
            {"selected": ["a", "b"], "result": "no_consensus"},
        ]

        results = selector.get_best_team_combinations(min_debates=3)

        assert len(results) == 1
        assert results[0]["success_rate"] == 0.75
        assert results[0]["total_debates"] == 4
        assert results[0]["wins"] == 3


class TestAgentSelectorUpdateFromResult:
    """Tests for updating profiles from debate results."""

    def test_update_from_result_success_rate(self):
        """update_from_result should update agent success rates."""
        selector = AgentSelector()

        agent = AgentProfile(name="test", agent_type="claude", success_rate=0.5)
        selector.register_agent(agent)

        team = TeamComposition(
            team_id="team-1",
            task_id="task-1",
            agents=[agent],
            roles={"test": "proposer"},
            expected_quality=0.8,
            expected_cost=1.0,
            diversity_score=0.5,
            rationale="Test",
        )

        result = MagicMock()
        result.scores = {"test": 0.9}  # High score

        selector.update_from_result(team, result)

        # Success rate should increase (EMA: 0.1 * 1.0 + 0.9 * 0.5 = 0.55)
        assert selector.agent_pool["test"].success_rate > 0.5

    def test_update_from_result_records_history(self):
        """update_from_result should record to history."""
        selector = AgentSelector()

        agent = AgentProfile(name="test", agent_type="claude")
        selector.register_agent(agent)

        team = TeamComposition(
            team_id="team-1",
            task_id="task-1",
            agents=[agent],
            roles={"test": "proposer"},
            expected_quality=0.8,
            expected_cost=1.0,
            diversity_score=0.5,
            rationale="Test",
        )

        result = MagicMock()
        result.scores = {"test": 0.9}
        result.consensus_reached = True
        result.confidence = 0.85

        selector.update_from_result(team, result)

        history = selector.get_selection_history()
        assert len(history) >= 1
        # Most recent should have result info
        latest = history[0]
        assert latest["result"] == "success"
        assert latest["confidence"] == 0.85


# =============================================================================
# AgentSelector - Leaderboard and Recommendations Tests
# =============================================================================


class TestAgentSelectorLeaderboard:
    """Tests for leaderboard functionality."""

    def test_get_leaderboard_overall(self):
        """get_leaderboard should rank by overall score."""
        selector = AgentSelector()

        selector.register_agent(AgentProfile(name="best", agent_type="claude", elo_rating=1900))
        selector.register_agent(AgentProfile(name="mid", agent_type="gemini", elo_rating=1500))
        selector.register_agent(AgentProfile(name="low", agent_type="codex", elo_rating=1100))

        leaderboard = selector.get_leaderboard()

        assert len(leaderboard) == 3
        assert leaderboard[0]["name"] == "best"
        assert leaderboard[2]["name"] == "low"

    def test_get_leaderboard_by_domain(self):
        """get_leaderboard should rank by domain ELO."""
        selector = AgentSelector()

        selector.register_agent(
            AgentProfile(
                name="backend-expert",
                agent_type="claude",
                elo_rating=1500,
                domain_ratings={"backend": 1900, "frontend": 1200},
            )
        )
        selector.register_agent(
            AgentProfile(
                name="frontend-expert",
                agent_type="gemini",
                elo_rating=1600,
                domain_ratings={"backend": 1200, "frontend": 1900},
            )
        )

        backend_lb = selector.get_leaderboard(domain="backend")
        frontend_lb = selector.get_leaderboard(domain="frontend")

        assert backend_lb[0]["name"] == "backend-expert"
        assert frontend_lb[0]["name"] == "frontend-expert"

    def test_get_leaderboard_limit(self):
        """get_leaderboard should respect limit."""
        selector = AgentSelector()

        for i in range(20):
            selector.register_agent(AgentProfile(name=f"agent-{i}", agent_type="claude"))

        leaderboard = selector.get_leaderboard(limit=5)

        assert len(leaderboard) == 5

    def test_get_leaderboard_includes_bench_status(self):
        """get_leaderboard should indicate bench status."""
        selector = AgentSelector()

        selector.register_agent(AgentProfile(name="active", agent_type="claude"))
        selector.register_agent(AgentProfile(name="benched", agent_type="claude"))
        selector.move_to_bench("benched")

        leaderboard = selector.get_leaderboard()

        active_entry = next(e for e in leaderboard if e["name"] == "active")
        benched_entry = next(e for e in leaderboard if e["name"] == "benched")

        assert active_entry["on_bench"] is False
        assert benched_entry["on_bench"] is True


class TestAgentSelectorRecommendations:
    """Tests for agent recommendations."""

    def test_get_recommendations_basic(self):
        """get_recommendations should return scored recommendations."""
        selector = AgentSelector()

        selector.register_agent(
            AgentProfile(
                name="expert",
                agent_type="claude",
                expertise={"backend": 0.9},
            )
        )
        selector.register_agent(
            AgentProfile(
                name="novice",
                agent_type="gemini",
                expertise={"backend": 0.2},
            )
        )

        req = TaskRequirements(
            task_id="test",
            description="Backend task",
            primary_domain="backend",
        )

        recs = selector.get_recommendations(req)

        assert len(recs) == 2
        assert recs[0]["name"] == "expert"
        assert recs[0]["match_score"] > recs[1]["match_score"]
        assert "reasoning" in recs[0]

    def test_get_recommendations_limit(self):
        """get_recommendations should respect limit."""
        selector = AgentSelector()

        for i in range(10):
            selector.register_agent(AgentProfile(name=f"agent-{i}", agent_type="claude"))

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        recs = selector.get_recommendations(req, limit=3)

        assert len(recs) == 3

    def test_explain_match_strong_expertise(self):
        """_explain_match should mention strong expertise."""
        selector = AgentSelector()

        agent = AgentProfile(
            name="expert",
            agent_type="claude",
            expertise={"backend": 0.9},
        )

        req = TaskRequirements(
            task_id="test",
            description="Test",
            primary_domain="backend",
        )

        explanation = selector._explain_match(agent, req)

        assert "Strong" in explanation
        assert "backend" in explanation

    def test_explain_match_traits(self):
        """_explain_match should mention matching traits."""
        selector = AgentSelector()

        agent = AgentProfile(
            name="secure",
            agent_type="claude",
            traits=["security", "thorough"],
        )

        req = TaskRequirements(
            task_id="test",
            description="Security audit",
            primary_domain="security",
            required_traits=["security"],
        )

        explanation = selector._explain_match(agent, req)

        assert "traits" in explanation.lower()
        assert "security" in explanation


# =============================================================================
# AgentSelector - Refresh Integration Tests
# =============================================================================


class TestAgentSelectorRefresh:
    """Tests for refreshing data from external systems."""

    def test_refresh_from_elo_system(self):
        """refresh_from_elo_system should update ratings."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1700
        mock_rating.domain_elos = {"backend": 1800}
        mock_rating.win_rate = 0.75
        # Now uses get_ratings_batch which returns a dict
        mock_elo.get_ratings_batch.return_value = {"test": mock_rating}

        selector = AgentSelector(elo_system=mock_elo)
        selector.register_agent(
            AgentProfile(
                name="test",
                agent_type="claude",
                elo_rating=1500,
            )
        )

        selector.refresh_from_elo_system()

        assert selector.agent_pool["test"].elo_rating == 1700
        assert selector.agent_pool["test"].domain_ratings == {"backend": 1800}
        assert selector.agent_pool["test"].success_rate == 0.75

    def test_refresh_probe_scores(self):
        """refresh_probe_scores should update probe data."""
        mock_filter = MagicMock()
        mock_profile = MagicMock()
        mock_profile.probe_score = 0.6
        mock_profile.has_critical_issues.return_value = True
        mock_filter.get_agent_profile.return_value = mock_profile

        selector = AgentSelector(probe_filter=mock_filter)
        selector.register_agent(AgentProfile(name="test", agent_type="claude"))

        selector.refresh_probe_scores()

        assert selector.agent_pool["test"].probe_score == 0.6
        assert selector.agent_pool["test"].has_critical_probes is True

    def test_refresh_calibration_scores(self):
        """refresh_calibration_scores should update calibration data."""
        mock_tracker = MagicMock()
        mock_summary = MagicMock()
        mock_summary.total_predictions = 10
        mock_summary.ece = 0.2
        mock_summary.brier_score = 0.15
        mock_summary.is_overconfident = True
        mock_tracker.get_calibration_summary.return_value = mock_summary

        selector = AgentSelector(calibration_tracker=mock_tracker)
        selector.register_agent(AgentProfile(name="test", agent_type="claude"))

        selector.refresh_calibration_scores()

        assert selector.agent_pool["test"].calibration_score == pytest.approx(0.8, rel=0.01)
        assert selector.agent_pool["test"].brier_score == 0.15
        assert selector.agent_pool["test"].is_overconfident is True

    def test_set_probe_filter_refreshes(self):
        """set_probe_filter should trigger immediate refresh."""
        mock_filter = MagicMock()
        mock_profile = MagicMock()
        mock_profile.probe_score = 0.5
        mock_profile.has_critical_issues.return_value = False
        mock_filter.get_agent_profile.return_value = mock_profile

        selector = AgentSelector()
        selector.register_agent(AgentProfile(name="test", agent_type="claude"))

        selector.set_probe_filter(mock_filter)

        assert selector.agent_pool["test"].probe_score == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
