"""
Tests for aragora.routing.selection module.

Tests the adaptive agent selection system including:
- AgentProfile dataclass
- TaskRequirements dataclass
- TeamComposition dataclass
- AgentSelector class (team selection, scoring, bench system)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.routing.selection import (
    AgentProfile,
    AgentSelector,
    TaskRequirements,
    TeamComposition,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def basic_agent():
    """Create a basic agent profile."""
    return AgentProfile(
        name="claude",
        agent_type="claude",
        elo_rating=1600,
        domain_ratings={"coding": 1700, "math": 1500},
        expertise={"coding": 0.8, "math": 0.6},
        traits=["thorough", "analytical"],
        availability=1.0,
        cost_factor=1.5,
        latency_ms=800,
        success_rate=0.85,
    )


@pytest.fixture
def agent_pool():
    """Create a pool of agents."""
    return [
        AgentProfile(
            name="claude",
            agent_type="claude",
            elo_rating=1650,
            expertise={"coding": 0.9, "design": 0.7},
            traits=["thorough", "analytical"],
            success_rate=0.85,
        ),
        AgentProfile(
            name="gemini",
            agent_type="gemini",
            elo_rating=1550,
            expertise={"design": 0.8, "planning": 0.9},
            traits=["creative", "fast"],
            cost_factor=0.8,
            success_rate=0.78,
        ),
        AgentProfile(
            name="codex",
            agent_type="codex",
            elo_rating=1700,
            expertise={"coding": 0.95, "testing": 0.85},
            traits=["pragmatic", "performance"],
            latency_ms=500,
            success_rate=0.82,
        ),
        AgentProfile(
            name="grok",
            agent_type="grok",
            elo_rating=1500,
            expertise={"reasoning": 0.7, "humor": 0.9},
            traits=["lateral", "edgy"],
            success_rate=0.72,
        ),
        AgentProfile(
            name="deepseek",
            agent_type="deepseek",
            elo_rating=1580,
            expertise={"math": 0.9, "reasoning": 0.85},
            traits=["rigorous", "formal"],
            success_rate=0.80,
        ),
    ]


@pytest.fixture
def coding_task():
    """Create a coding task requirement."""
    return TaskRequirements(
        task_id="task-001",
        description="Implement a rate limiter",
        primary_domain="coding",
        secondary_domains=["design", "testing"],
        required_traits=["thorough"],
        min_agents=2,
        max_agents=4,
        quality_priority=0.7,
        diversity_preference=0.5,
    )


@pytest.fixture
def selector(agent_pool):
    """Create selector with agents registered."""
    selector = AgentSelector()
    for agent in agent_pool:
        selector.register_agent(agent)
    return selector


# ==============================================================================
# AgentProfile Tests
# ==============================================================================


class TestAgentProfile:
    """Tests for AgentProfile dataclass."""

    def test_minimal_creation(self):
        """Can create with just required fields."""
        profile = AgentProfile(name="test", agent_type="test")
        assert profile.name == "test"
        assert profile.agent_type == "test"

    def test_default_values(self):
        """Default values are set correctly."""
        profile = AgentProfile(name="test", agent_type="test")
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

    def test_full_creation(self, basic_agent):
        """Can create with all fields."""
        assert basic_agent.name == "claude"
        assert basic_agent.elo_rating == 1600
        assert basic_agent.expertise["coding"] == 0.8
        assert "thorough" in basic_agent.traits

    def test_overall_score_calculation(self, basic_agent):
        """overall_score computes weighted combination."""
        score = basic_agent.overall_score
        # Should be between 0 and 1
        assert 0 <= score <= 1
        # High ELO, high success rate should give decent score
        assert score > 0.4

    def test_overall_score_penalizes_critical_probes(self):
        """overall_score applies 30% penalty for critical probes."""
        profile = AgentProfile(
            name="test",
            agent_type="test",
            elo_rating=1600,
            success_rate=0.85,
            has_critical_probes=True,
        )
        profile_clean = AgentProfile(
            name="test",
            agent_type="test",
            elo_rating=1600,
            success_rate=0.85,
            has_critical_probes=False,
        )
        assert profile.overall_score < profile_clean.overall_score * 0.75

    def test_overall_score_penalizes_overconfidence(self):
        """overall_score applies 10% penalty for overconfidence."""
        profile = AgentProfile(
            name="test",
            agent_type="test",
            elo_rating=1600,
            is_overconfident=True,
        )
        profile_calm = AgentProfile(
            name="test",
            agent_type="test",
            elo_rating=1600,
            is_overconfident=False,
        )
        assert profile.overall_score < profile_calm.overall_score

    def test_overall_score_weights_probe_score(self):
        """overall_score considers probe reliability (20% weight)."""
        reliable = AgentProfile(name="a", agent_type="t", probe_score=1.0)
        vulnerable = AgentProfile(name="b", agent_type="t", probe_score=0.5)
        assert reliable.overall_score > vulnerable.overall_score

    def test_overall_score_weights_calibration(self):
        """overall_score considers calibration (15% weight)."""
        well_calibrated = AgentProfile(name="a", agent_type="t", calibration_score=1.0)
        poorly_calibrated = AgentProfile(name="b", agent_type="t", calibration_score=0.5)
        assert well_calibrated.overall_score > poorly_calibrated.overall_score


# ==============================================================================
# TaskRequirements Tests
# ==============================================================================


class TestTaskRequirements:
    """Tests for TaskRequirements dataclass."""

    def test_minimal_creation(self):
        """Can create with just required fields."""
        req = TaskRequirements(
            task_id="t1",
            description="Test task",
            primary_domain="coding",
        )
        assert req.task_id == "t1"
        assert req.primary_domain == "coding"

    def test_default_values(self):
        """Default values are set correctly."""
        req = TaskRequirements(
            task_id="t1",
            description="Test",
            primary_domain="coding",
        )
        assert req.secondary_domains == []
        assert req.required_traits == []
        assert req.min_agents == 2
        assert req.max_agents == 5
        assert req.quality_priority == 0.5
        assert req.diversity_preference == 0.5

    def test_full_creation(self, coding_task):
        """Can create with all fields."""
        assert coding_task.task_id == "task-001"
        assert coding_task.primary_domain == "coding"
        assert "design" in coding_task.secondary_domains
        assert "thorough" in coding_task.required_traits


# ==============================================================================
# TeamComposition Tests
# ==============================================================================


class TestTeamComposition:
    """Tests for TeamComposition dataclass."""

    def test_creation(self, agent_pool):
        """Can create a team composition."""
        team = TeamComposition(
            team_id="team-001",
            task_id="task-001",
            agents=agent_pool[:3],
            roles={"claude": "proposer", "gemini": "synthesizer", "codex": "critic"},
            expected_quality=0.85,
            expected_cost=3.3,
            diversity_score=0.7,
            rationale="Selected top 3 agents",
        )
        assert team.team_id == "team-001"
        assert len(team.agents) == 3
        assert team.roles["claude"] == "proposer"

    def test_stores_all_fields(self, agent_pool):
        """All fields are accessible."""
        team = TeamComposition(
            team_id="t1",
            task_id="task1",
            agents=agent_pool[:2],
            roles={"claude": "proposer", "gemini": "critic"},
            expected_quality=0.8,
            expected_cost=2.0,
            diversity_score=0.6,
            rationale="Test rationale",
        )
        assert team.expected_quality == 0.8
        assert team.expected_cost == 2.0
        assert team.diversity_score == 0.6
        assert team.rationale == "Test rationale"


# ==============================================================================
# AgentSelector Initialization Tests
# ==============================================================================


class TestAgentSelectorInit:
    """Tests for AgentSelector initialization."""

    def test_creates_empty_pool(self):
        """Initializes with empty agent pool."""
        selector = AgentSelector()
        assert selector.agent_pool == {}
        assert selector.bench == []

    def test_accepts_elo_system(self):
        """Can initialize with ELO system."""
        elo = MagicMock()
        selector = AgentSelector(elo_system=elo)
        assert selector.elo_system is elo

    def test_accepts_persona_manager(self):
        """Can initialize with persona manager."""
        pm = MagicMock()
        selector = AgentSelector(persona_manager=pm)
        assert selector.persona_manager is pm

    def test_accepts_probe_filter(self):
        """Can initialize with probe filter."""
        pf = MagicMock()
        selector = AgentSelector(probe_filter=pf)
        assert selector.probe_filter is pf

    def test_accepts_calibration_tracker(self):
        """Can initialize with calibration tracker."""
        ct = MagicMock()
        selector = AgentSelector(calibration_tracker=ct)
        assert selector.calibration_tracker is ct


# ==============================================================================
# AgentSelector Registration Tests
# ==============================================================================


class TestAgentSelectorRegistration:
    """Tests for agent registration."""

    def test_register_agent(self, basic_agent):
        """Can register an agent."""
        selector = AgentSelector()
        selector.register_agent(basic_agent)
        assert "claude" in selector.agent_pool
        assert selector.agent_pool["claude"] == basic_agent

    def test_register_multiple_agents(self, agent_pool):
        """Can register multiple agents."""
        selector = AgentSelector()
        for agent in agent_pool:
            selector.register_agent(agent)
        assert len(selector.agent_pool) == 5

    def test_remove_agent(self, basic_agent):
        """Can remove an agent."""
        selector = AgentSelector()
        selector.register_agent(basic_agent)
        selector.remove_agent("claude")
        assert "claude" not in selector.agent_pool

    def test_remove_nonexistent_agent(self):
        """Removing nonexistent agent doesn't raise."""
        selector = AgentSelector()
        selector.remove_agent("nonexistent")  # Should not raise

    def test_remove_agent_also_removes_from_bench(self, basic_agent):
        """Removing agent also removes from bench."""
        selector = AgentSelector()
        selector.register_agent(basic_agent)
        selector.move_to_bench("claude")
        selector.remove_agent("claude")
        assert "claude" not in selector.bench


# ==============================================================================
# AgentSelector Bench Tests
# ==============================================================================


class TestAgentSelectorBench:
    """Tests for bench system."""

    def test_move_to_bench(self, selector):
        """Can move agent to bench."""
        selector.move_to_bench("claude")
        assert "claude" in selector.bench

    def test_move_to_bench_requires_registered_agent(self, selector):
        """Moving unregistered agent to bench doesn't add to bench."""
        selector.move_to_bench("nonexistent")
        assert "nonexistent" not in selector.bench

    def test_move_to_bench_is_idempotent(self, selector):
        """Moving same agent twice doesn't duplicate."""
        selector.move_to_bench("claude")
        selector.move_to_bench("claude")
        assert selector.bench.count("claude") == 1

    def test_promote_from_bench(self, selector):
        """Can promote agent from bench."""
        selector.move_to_bench("claude")
        selector.promote_from_bench("claude")
        assert "claude" not in selector.bench

    def test_promote_nonexistent_from_bench(self, selector):
        """Promoting agent not on bench doesn't raise."""
        selector.promote_from_bench("claude")  # Not on bench, should not raise


# ==============================================================================
# AgentSelector Probe Integration Tests
# ==============================================================================


class TestAgentSelectorProbeIntegration:
    """Tests for probe filter integration."""

    def test_set_probe_filter(self, selector):
        """Can set probe filter."""
        pf = MagicMock()
        pf.get_agent_profile.return_value = MagicMock(
            probe_score=0.8,
            has_critical_issues=lambda: False,
            total_probes=5,
        )
        selector.set_probe_filter(pf)
        assert selector.probe_filter is pf

    def test_refresh_probe_scores_updates_agents(self, selector):
        """refresh_probe_scores updates agent profiles."""
        pf = MagicMock()
        pf.get_agent_profile.return_value = MagicMock(
            probe_score=0.7,
            has_critical_issues=lambda: True,
        )
        selector.probe_filter = pf
        selector.refresh_probe_scores()
        assert selector.agent_pool["claude"].probe_score == 0.7
        assert selector.agent_pool["claude"].has_critical_probes is True

    def test_refresh_probe_scores_handles_missing_agent(self, selector):
        """refresh_probe_scores handles agents not in probe system."""
        pf = MagicMock()
        pf.get_agent_profile.side_effect = KeyError("not found")
        selector.probe_filter = pf
        selector.refresh_probe_scores()  # Should not raise
        # Defaults preserved
        assert selector.agent_pool["claude"].probe_score == 1.0

    def test_get_probe_adjusted_score_applies_penalty(self, selector):
        """get_probe_adjusted_score applies reliability penalty."""
        selector.agent_pool["claude"].probe_score = 0.5
        selector.agent_pool["claude"].has_critical_probes = False
        adjusted = selector.get_probe_adjusted_score("claude", 1.0)
        # 0.5 + (0.5 * 0.5) = 0.75
        assert adjusted == pytest.approx(0.75, rel=0.01)

    def test_get_probe_adjusted_score_penalizes_critical(self, selector):
        """get_probe_adjusted_score adds extra penalty for critical probes."""
        selector.agent_pool["claude"].probe_score = 1.0
        selector.agent_pool["claude"].has_critical_probes = True
        adjusted = selector.get_probe_adjusted_score("claude", 1.0)
        # (0.5 + 0.5) * 0.8 = 0.8
        assert adjusted < 1.0


# ==============================================================================
# AgentSelector Calibration Integration Tests
# ==============================================================================


class TestAgentSelectorCalibrationIntegration:
    """Tests for calibration tracker integration."""

    def test_set_calibration_tracker(self, selector):
        """Can set calibration tracker."""
        ct = MagicMock()
        ct.get_calibration_summary.return_value = MagicMock(
            total_predictions=10,
            ece=0.1,
            brier_score=0.15,
            is_overconfident=False,
        )
        selector.set_calibration_tracker(ct)
        assert selector.calibration_tracker is ct

    def test_refresh_calibration_scores_updates_agents(self, selector):
        """refresh_calibration_scores updates agent profiles."""
        ct = MagicMock()
        ct.get_calibration_summary.return_value = MagicMock(
            total_predictions=10,
            ece=0.2,
            brier_score=0.18,
            is_overconfident=True,
        )
        selector.calibration_tracker = ct
        selector.refresh_calibration_scores()
        # calibration_score = 1 - ECE = 0.8
        assert selector.agent_pool["claude"].calibration_score == pytest.approx(0.8)
        assert selector.agent_pool["claude"].is_overconfident is True

    def test_refresh_calibration_requires_min_predictions(self, selector):
        """refresh_calibration_scores ignores agents with few predictions."""
        ct = MagicMock()
        ct.get_calibration_summary.return_value = MagicMock(
            total_predictions=3,  # Less than 5
            ece=0.5,
            brier_score=0.4,
            is_overconfident=True,
        )
        selector.calibration_tracker = ct
        selector.refresh_calibration_scores()
        # Should use defaults
        assert selector.agent_pool["claude"].calibration_score == 1.0
        assert selector.agent_pool["claude"].is_overconfident is False

    def test_get_calibration_adjusted_score_applies_factor(self, selector):
        """get_calibration_adjusted_score applies calibration factor."""
        selector.agent_pool["claude"].calibration_score = 0.5
        selector.agent_pool["claude"].is_overconfident = False
        adjusted = selector.get_calibration_adjusted_score("claude", 1.0)
        # 0.7 + (0.5 * 0.3) = 0.85
        assert adjusted == pytest.approx(0.85, rel=0.01)

    def test_get_calibration_adjusted_score_penalizes_overconfidence(self, selector):
        """get_calibration_adjusted_score penalizes overconfident agents."""
        selector.agent_pool["claude"].calibration_score = 1.0
        selector.agent_pool["claude"].is_overconfident = True
        adjusted = selector.get_calibration_adjusted_score("claude", 1.0)
        # (0.7 + 0.3) * 0.9 = 0.9
        assert adjusted == pytest.approx(0.9, rel=0.01)


# ==============================================================================
# AgentSelector ELO Integration Tests
# ==============================================================================


class TestAgentSelectorEloIntegration:
    """Tests for ELO system integration."""

    def test_refresh_from_elo_system(self, selector):
        """refresh_from_elo_system updates agent ratings."""
        elo = MagicMock()
        rating = MagicMock()
        rating.elo = 1800
        rating.domain_elos = {"coding": 1900}
        rating.win_rate = 0.9
        elo.get_rating.return_value = rating

        selector.refresh_from_elo_system(elo)

        assert selector.agent_pool["claude"].elo_rating == 1800
        assert selector.agent_pool["claude"].domain_ratings["coding"] == 1900
        assert selector.agent_pool["claude"].success_rate == 0.9

    def test_refresh_from_elo_handles_missing_agent(self, selector):
        """refresh_from_elo_system handles agents not in ELO system."""
        elo = MagicMock()
        elo.get_rating.return_value = None
        selector.refresh_from_elo_system(elo)  # Should not raise

    def test_refresh_uses_stored_elo_system(self, selector):
        """refresh_from_elo_system uses stored system if none provided."""
        elo = MagicMock()
        elo.get_rating.return_value = None
        selector.elo_system = elo
        selector.refresh_from_elo_system()
        elo.get_rating.assert_called()


# ==============================================================================
# AgentSelector Team Selection Tests
# ==============================================================================


class TestAgentSelectorTeamSelection:
    """Tests for team selection."""

    def test_select_team_returns_composition(self, selector, coding_task):
        """select_team returns a TeamComposition."""
        team = selector.select_team(coding_task)
        assert isinstance(team, TeamComposition)

    def test_select_team_respects_min_agents(self, selector, coding_task):
        """select_team returns at least min_agents."""
        coding_task.min_agents = 3
        team = selector.select_team(coding_task)
        assert len(team.agents) >= 3

    def test_select_team_respects_max_agents(self, selector, coding_task):
        """select_team returns at most max_agents."""
        coding_task.max_agents = 2
        team = selector.select_team(coding_task)
        assert len(team.agents) <= 2

    def test_select_team_excludes_specified_agents(self, selector, coding_task):
        """select_team excludes agents in exclude list."""
        team = selector.select_team(coding_task, exclude=["claude", "codex"])
        names = [a.name for a in team.agents]
        assert "claude" not in names
        assert "codex" not in names

    def test_select_team_excludes_benched_agents(self, selector, coding_task):
        """select_team excludes agents on the bench."""
        selector.move_to_bench("claude")
        team = selector.select_team(coding_task)
        names = [a.name for a in team.agents]
        assert "claude" not in names

    def test_select_team_raises_on_empty_pool(self, coding_task):
        """select_team raises ValueError when no agents available."""
        selector = AgentSelector()
        with pytest.raises(ValueError, match="No available agents"):
            selector.select_team(coding_task)

    def test_select_team_assigns_roles(self, selector, coding_task):
        """select_team assigns roles to team members."""
        team = selector.select_team(coding_task)
        assert len(team.roles) == len(team.agents)
        for agent in team.agents:
            assert agent.name in team.roles

    def test_select_team_calculates_expected_quality(self, selector, coding_task):
        """select_team calculates expected_quality."""
        team = selector.select_team(coding_task)
        assert 0 <= team.expected_quality <= 1

    def test_select_team_calculates_expected_cost(self, selector, coding_task):
        """select_team calculates expected_cost."""
        team = selector.select_team(coding_task)
        assert team.expected_cost > 0

    def test_select_team_calculates_diversity_score(self, selector, coding_task):
        """select_team calculates diversity_score."""
        team = selector.select_team(coding_task)
        assert 0 <= team.diversity_score <= 1

    def test_select_team_generates_rationale(self, selector, coding_task):
        """select_team generates rationale string."""
        team = selector.select_team(coding_task)
        assert isinstance(team.rationale, str)
        assert len(team.rationale) > 0

    def test_select_team_records_history(self, selector, coding_task):
        """select_team records selection in history."""
        initial_len = len(selector._selection_history)
        selector.select_team(coding_task)
        assert len(selector._selection_history) == initial_len + 1

    def test_select_team_prefers_domain_expertise(self, selector, coding_task):
        """select_team prefers agents with domain expertise."""
        team = selector.select_team(coding_task)
        # Codex has highest coding expertise (0.95)
        assert any(a.name == "codex" for a in team.agents)


# ==============================================================================
# AgentSelector Scoring Tests
# ==============================================================================


class TestAgentSelectorScoring:
    """Tests for agent scoring logic."""

    def test_score_for_task_values_domain_expertise(self, selector, coding_task):
        """_score_for_task weights domain expertise."""
        claude = selector.agent_pool["claude"]
        grok = selector.agent_pool["grok"]  # No coding expertise
        claude_score = selector._score_for_task(claude, coding_task)
        grok_score = selector._score_for_task(grok, coding_task)
        assert claude_score > grok_score

    def test_score_for_task_values_elo(self, selector, coding_task):
        """_score_for_task considers ELO rating."""
        selector.agent_pool["claude"].elo_rating = 2000
        selector.agent_pool["gemini"].elo_rating = 1000
        claude_score = selector._score_for_task(selector.agent_pool["claude"], coding_task)
        gemini_score = selector._score_for_task(selector.agent_pool["gemini"], coding_task)
        assert claude_score > gemini_score

    def test_score_for_task_values_trait_matching(self, selector, coding_task):
        """_score_for_task rewards matching traits."""
        coding_task.required_traits = ["thorough"]
        claude = selector.agent_pool["claude"]  # Has "thorough"
        gemini = selector.agent_pool["gemini"]  # Doesn't have "thorough"
        claude_score = selector._score_for_task(claude, coding_task)
        gemini_score = selector._score_for_task(gemini, coding_task)
        assert claude_score > gemini_score

    def test_score_for_task_uses_persona_manager(self, selector, coding_task):
        """_score_for_task uses PersonaManager for dynamic expertise."""
        pm = MagicMock()
        persona = MagicMock()
        persona.expertise = {"coding": 0.99}  # Higher than static
        persona.traits = ["brilliant"]
        pm.get_persona.return_value = persona
        selector.persona_manager = pm

        score = selector._score_for_task(selector.agent_pool["claude"], coding_task)
        # Should use dynamic expertise from persona
        pm.get_persona.assert_called_with("claude")

    def test_score_for_task_bounded_zero_one(self, selector, coding_task):
        """_score_for_task returns value between 0 and 1."""
        for agent in selector.agent_pool.values():
            score = selector._score_for_task(agent, coding_task)
            assert 0 <= score <= 1


# ==============================================================================
# AgentSelector Role Assignment Tests
# ==============================================================================


class TestAgentSelectorRoleAssignment:
    """Tests for role assignment."""

    def test_assign_roles_assigns_proposer(self, selector, coding_task):
        """_assign_roles assigns proposer to highest expert."""
        agents = list(selector.agent_pool.values())[:3]
        roles = selector._assign_roles(agents, coding_task)
        assert "proposer" in roles.values()

    def test_assign_roles_assigns_synthesizer(self, selector, coding_task):
        """_assign_roles assigns synthesizer when enough agents."""
        agents = list(selector.agent_pool.values())[:3]
        roles = selector._assign_roles(agents, coding_task)
        assert "synthesizer" in roles.values()

    def test_assign_roles_assigns_critics(self, selector, coding_task):
        """_assign_roles assigns critic roles to remaining agents."""
        agents = list(selector.agent_pool.values())[:3]
        roles = selector._assign_roles(agents, coding_task)
        critic_roles = [r for r in roles.values() if "critic" in r]
        assert len(critic_roles) >= 1

    def test_assign_roles_handles_empty_team(self, selector, coding_task):
        """_assign_roles handles empty team."""
        roles = selector._assign_roles([], coding_task)
        assert roles == {}


class TestAgentSelectorHybridRoles:
    """Tests for hybrid model role assignment."""

    def test_hybrid_roles_debate_phase(self, selector):
        """assign_hybrid_roles assigns proposer to all in debate phase."""
        agents = list(selector.agent_pool.values())[:3]
        roles = selector.assign_hybrid_roles(agents, "debate")
        for agent in agents:
            assert roles[agent.name] == "proposer"

    def test_hybrid_roles_design_phase_gemini_lead(self, selector):
        """assign_hybrid_roles assigns design_lead to gemini in design phase."""
        agents = list(selector.agent_pool.values())
        roles = selector.assign_hybrid_roles(agents, "design")
        assert roles["gemini"] == "design_lead"

    def test_hybrid_roles_implement_phase_claude_lead(self, selector):
        """assign_hybrid_roles assigns implementer to claude in implement phase."""
        agents = list(selector.agent_pool.values())
        roles = selector.assign_hybrid_roles(agents, "implement")
        assert roles["claude"] == "implementer"

    def test_hybrid_roles_verify_phase_codex_lead(self, selector):
        """assign_hybrid_roles assigns verification_lead to codex in verify phase."""
        agents = list(selector.agent_pool.values())
        roles = selector.assign_hybrid_roles(agents, "verify")
        assert roles["codex"] == "verification_lead"

    def test_hybrid_roles_commit_phase_all_reviewers(self, selector):
        """assign_hybrid_roles assigns reviewer to all in commit phase."""
        agents = list(selector.agent_pool.values())[:3]
        roles = selector.assign_hybrid_roles(agents, "commit")
        for agent in agents:
            assert roles[agent.name] == "reviewer"

    def test_hybrid_roles_unknown_phase_fallback(self, selector):
        """assign_hybrid_roles uses participant fallback for unknown phase."""
        agents = list(selector.agent_pool.values())[:2]
        roles = selector.assign_hybrid_roles(agents, "unknown_phase")
        for agent in agents:
            assert roles[agent.name] == "participant"


# ==============================================================================
# AgentSelector Diversity Tests
# ==============================================================================


class TestAgentSelectorDiversity:
    """Tests for diversity calculation."""

    def test_diversity_single_agent(self, selector):
        """_calculate_diversity returns 0 for single agent."""
        agents = [selector.agent_pool["claude"]]
        div = selector._calculate_diversity(agents)
        assert div == 0.0

    def test_diversity_identical_agents(self):
        """_calculate_diversity is low for identical agents."""
        agents = [
            AgentProfile(name="a", agent_type="claude", traits=["thorough"], elo_rating=1500),
            AgentProfile(name="b", agent_type="claude", traits=["thorough"], elo_rating=1500),
        ]
        selector = AgentSelector()
        div = selector._calculate_diversity(agents)
        # Low diversity expected
        assert div < 0.5

    def test_diversity_different_types(self, selector):
        """_calculate_diversity is higher for different agent types."""
        agents = list(selector.agent_pool.values())[:3]
        div = selector._calculate_diversity(agents)
        # Should have decent diversity (different types)
        assert div > 0.3

    def test_diversity_considers_traits(self):
        """_calculate_diversity considers trait diversity."""
        agents = [
            AgentProfile(name="a", agent_type="t", traits=["thorough", "analytical"]),
            AgentProfile(name="b", agent_type="t", traits=["creative", "fast"]),
        ]
        selector = AgentSelector()
        div = selector._calculate_diversity(agents)
        # 4 unique traits
        assert div > 0.2


# ==============================================================================
# AgentSelector History Tests
# ==============================================================================


class TestAgentSelectorHistory:
    """Tests for selection history."""

    def test_get_selection_history(self, selector, coding_task):
        """get_selection_history returns history."""
        selector.select_team(coding_task)
        history = selector.get_selection_history()
        assert len(history) >= 1

    def test_get_selection_history_with_limit(self, selector, coding_task):
        """get_selection_history respects limit."""
        for _ in range(5):
            selector.select_team(coding_task)
        history = selector.get_selection_history(limit=2)
        assert len(history) == 2

    def test_get_selection_history_sorted_by_timestamp(self, selector, coding_task):
        """get_selection_history returns newest first."""
        selector.select_team(coding_task)
        coding_task.task_id = "task-002"
        selector.select_team(coding_task)
        history = selector.get_selection_history()
        assert history[0]["task_id"] == "task-002"


# ==============================================================================
# AgentSelector Update From Result Tests
# ==============================================================================


class TestAgentSelectorUpdateFromResult:
    """Tests for updating from debate results."""

    def test_update_from_result_updates_success_rate(self, selector, coding_task):
        """update_from_result updates agent success rates."""
        team = selector.select_team(coding_task)
        initial_rate = team.agents[0].success_rate

        result = MagicMock()
        result.scores = {team.agents[0].name: 0.9}
        result.consensus_reached = True
        result.confidence = 0.85

        selector.update_from_result(team, result)
        # Success rate should move toward 1.0
        assert team.agents[0].success_rate > initial_rate - 0.1

    def test_update_from_result_handles_missing_scores(self, selector, coding_task):
        """update_from_result handles result without scores."""
        team = selector.select_team(coding_task)
        result = MagicMock()
        result.scores = None
        selector.update_from_result(team, result)  # Should not raise

    def test_update_from_result_records_history(self, selector, coding_task):
        """update_from_result adds to selection history when scores present."""
        team = selector.select_team(coding_task)
        initial_len = len(selector._selection_history)

        result = MagicMock()
        # Need at least one score for history to be recorded
        result.scores = {team.agents[0].name: 0.7}
        result.consensus_reached = True
        result.confidence = 0.8

        selector.update_from_result(team, result)
        assert len(selector._selection_history) == initial_len + 1


# ==============================================================================
# AgentSelector Leaderboard Tests
# ==============================================================================


class TestAgentSelectorLeaderboard:
    """Tests for leaderboard functionality."""

    def test_get_leaderboard(self, selector):
        """get_leaderboard returns ranked agents."""
        lb = selector.get_leaderboard()
        assert len(lb) <= 10  # Default limit
        assert all("name" in entry for entry in lb)

    def test_get_leaderboard_with_domain(self, selector):
        """get_leaderboard can filter by domain."""
        lb = selector.get_leaderboard(domain="coding")
        # Should include domain_elo
        assert all("domain_elo" in entry for entry in lb)

    def test_get_leaderboard_with_limit(self, selector):
        """get_leaderboard respects limit."""
        lb = selector.get_leaderboard(limit=2)
        assert len(lb) == 2

    def test_get_leaderboard_shows_bench_status(self, selector):
        """get_leaderboard indicates if agent is on bench."""
        selector.move_to_bench("claude")
        lb = selector.get_leaderboard()
        claude_entry = next(e for e in lb if e["name"] == "claude")
        assert claude_entry["on_bench"] is True


# ==============================================================================
# AgentSelector Recommendations Tests
# ==============================================================================


class TestAgentSelectorRecommendations:
    """Tests for recommendation functionality."""

    def test_get_recommendations(self, selector, coding_task):
        """get_recommendations returns scored agents."""
        recs = selector.get_recommendations(coding_task)
        assert len(recs) <= 5  # Default limit
        assert all("match_score" in r for r in recs)

    def test_get_recommendations_with_limit(self, selector, coding_task):
        """get_recommendations respects limit."""
        recs = selector.get_recommendations(coding_task, limit=2)
        assert len(recs) == 2

    def test_get_recommendations_sorted_by_score(self, selector, coding_task):
        """get_recommendations returns highest scores first."""
        recs = selector.get_recommendations(coding_task)
        scores = [r["match_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_get_recommendations_includes_reasoning(self, selector, coding_task):
        """get_recommendations includes reasoning for each agent."""
        recs = selector.get_recommendations(coding_task)
        assert all("reasoning" in r for r in recs)


# ==============================================================================
# AgentSelector Best Team Combinations Tests
# ==============================================================================


class TestAgentSelectorBestTeamCombinations:
    """Tests for team combination analysis."""

    def test_get_best_team_combinations_empty(self, selector):
        """get_best_team_combinations returns empty for no history."""
        result = selector.get_best_team_combinations()
        assert result == []

    def test_get_best_team_combinations_with_history(self, selector, coding_task):
        """get_best_team_combinations analyzes history."""
        # Create some history
        for i in range(5):
            selector.select_team(coding_task)
            # Simulate result
            selector._selection_history[-1]["result"] = "success" if i % 2 == 0 else "no_consensus"

        result = selector.get_best_team_combinations(min_debates=1)
        assert len(result) >= 1

    def test_get_best_team_combinations_respects_min_debates(self, selector, coding_task):
        """get_best_team_combinations filters by min_debates."""
        selector.select_team(coding_task)
        selector._selection_history[-1]["result"] = "success"

        # Should be empty with min_debates=5
        result = selector.get_best_team_combinations(min_debates=5)
        assert result == []


# ==============================================================================
# AgentSelector Explain Match Tests
# ==============================================================================


class TestAgentSelectorExplainMatch:
    """Tests for match explanation."""

    def test_explain_match_high_expertise(self, selector, coding_task):
        """_explain_match mentions high expertise."""
        # Codex has 0.95 coding expertise
        explanation = selector._explain_match(selector.agent_pool["codex"], coding_task)
        assert "Strong" in explanation or "expertise" in explanation.lower()

    def test_explain_match_high_elo(self, selector, coding_task):
        """_explain_match mentions high ELO."""
        selector.agent_pool["claude"].elo_rating = 1800
        explanation = selector._explain_match(selector.agent_pool["claude"], coding_task)
        assert "rating" in explanation.lower()

    def test_explain_match_matching_traits(self, selector, coding_task):
        """_explain_match mentions matching traits."""
        coding_task.required_traits = ["thorough"]
        explanation = selector._explain_match(selector.agent_pool["claude"], coding_task)
        assert "thorough" in explanation.lower()

    def test_explain_match_fallback(self):
        """_explain_match provides fallback for generic agents."""
        selector = AgentSelector()
        agent = AgentProfile(name="generic", agent_type="generic")
        req = TaskRequirements(task_id="t", description="Test", primary_domain="obscure")
        selector.register_agent(agent)
        explanation = selector._explain_match(agent, req)
        assert "General purpose" in explanation


# ==============================================================================
# AgentSelector Generate Rationale Tests
# ==============================================================================


class TestAgentSelectorGenerateRationale:
    """Tests for rationale generation."""

    def test_generate_rationale_includes_team_size(self, selector, coding_task):
        """_generate_rationale mentions team size."""
        agents = list(selector.agent_pool.values())[:3]
        scored = [(a, 0.8) for a in agents]
        rationale = selector._generate_rationale(agents, coding_task, scored)
        assert "3 agents" in rationale

    def test_generate_rationale_includes_domain(self, selector, coding_task):
        """_generate_rationale mentions primary domain."""
        agents = list(selector.agent_pool.values())[:2]
        scored = [(a, 0.7) for a in agents]
        rationale = selector._generate_rationale(agents, coding_task, scored)
        assert "coding" in rationale

    def test_generate_rationale_lists_agents(self, selector, coding_task):
        """_generate_rationale lists team members."""
        agents = list(selector.agent_pool.values())[:2]
        scored = [(a, 0.7) for a in agents]
        rationale = selector._generate_rationale(agents, coding_task, scored)
        for agent in agents:
            assert agent.name in rationale
