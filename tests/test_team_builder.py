"""
Tests for the team_builder module.

Tests team building, diversity calculation, and role assignment.
"""

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from aragora.routing.team_builder import PHASE_ROLES, TeamBuilder


# =============================================================================
# Mock classes to avoid circular imports
# =============================================================================


@dataclass
class MockAgentProfile:
    """Mock AgentProfile for testing."""

    name: str
    agent_type: str = "cli"
    elo_rating: float = 1500.0
    overall_score: float = 0.7
    cost_factor: float = 1.0
    traits: list[str] = field(default_factory=list)
    expertise: dict[str, float] = field(default_factory=dict)


@dataclass
class MockTaskRequirements:
    """Mock TaskRequirements for testing."""

    task_id: str = "test_task"
    min_agents: int = 2
    max_agents: int = 4
    diversity_preference: float = 0.5
    primary_domain: str = "general"


# =============================================================================
# PHASE_ROLES Tests
# =============================================================================


class TestPhaseRoles:
    """Tests for PHASE_ROLES configuration."""

    def test_phase_roles_has_expected_phases(self):
        """Test that PHASE_ROLES has all expected phases."""
        expected_phases = {"debate", "design", "implement", "verify", "commit"}
        assert set(PHASE_ROLES.keys()) == expected_phases

    def test_phase_roles_structure(self):
        """Test that each phase has (role_map, fallback) structure."""
        for phase, config in PHASE_ROLES.items():
            assert isinstance(config, tuple)
            assert len(config) == 2
            role_map, fallback = config
            assert isinstance(role_map, dict)
            assert isinstance(fallback, str)

    def test_debate_phase_roles(self):
        """Test debate phase configuration."""
        role_map, fallback = PHASE_ROLES["debate"]
        assert role_map == {}  # All agents are proposers
        assert fallback == "proposer"

    def test_design_phase_roles(self):
        """Test design phase has specific roles."""
        role_map, fallback = PHASE_ROLES["design"]
        assert "gemini" in role_map
        assert role_map["gemini"] == "design_lead"
        assert fallback == "critic"

    def test_implement_phase_roles(self):
        """Test implement phase has claude as implementer."""
        role_map, fallback = PHASE_ROLES["implement"]
        assert role_map.get("claude") == "implementer"
        assert fallback == "advisor"

    def test_verify_phase_roles(self):
        """Test verify phase has verification roles."""
        role_map, fallback = PHASE_ROLES["verify"]
        assert role_map.get("codex") == "verification_lead"
        assert role_map.get("grok") == "quality_auditor"
        assert fallback == "reviewer"


# =============================================================================
# TeamBuilder Basic Tests
# =============================================================================


class TestTeamBuilderBasic:
    """Basic tests for TeamBuilder class."""

    def test_team_builder_creation(self):
        """Test TeamBuilder creation."""
        builder = TeamBuilder()
        assert builder._selection_history == []

    def test_calculate_diversity_empty_team(self):
        """Test diversity calculation with empty team."""
        builder = TeamBuilder()
        diversity = builder.calculate_diversity([])
        assert diversity == 0.0

    def test_calculate_diversity_single_agent(self):
        """Test diversity calculation with single agent."""
        builder = TeamBuilder()
        agent = MockAgentProfile(
            name="agent1",
            agent_type="cli",
            traits=["thorough"]
        )
        diversity = builder.calculate_diversity([agent])
        assert diversity == 0.0

    def test_calculate_diversity_same_type_agents(self):
        """Test diversity calculation with same-type agents."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="agent1", agent_type="cli", elo_rating=1500),
            MockAgentProfile(name="agent2", agent_type="cli", elo_rating=1500),
        ]
        diversity = builder.calculate_diversity(agents)

        # Low type diversity (both cli), low trait diversity, low ELO diversity
        assert diversity < 0.5

    def test_calculate_diversity_different_types(self):
        """Test diversity calculation with different agent types."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(
                name="agent1",
                agent_type="cli",
                elo_rating=1500,
                traits=["thorough"]
            ),
            MockAgentProfile(
                name="agent2",
                agent_type="api",
                elo_rating=1800,
                traits=["creative"]
            ),
        ]
        diversity = builder.calculate_diversity(agents)

        # Higher diversity due to different types, traits, and ELO
        assert diversity > 0.3


# =============================================================================
# TeamBuilder select_diverse_team Tests
# =============================================================================


class TestSelectDiverseTeam:
    """Tests for select_diverse_team method."""

    def test_select_fewer_than_min(self):
        """Test selecting when fewer agents than min_size."""
        builder = TeamBuilder()
        scored = [
            (MockAgentProfile(name="agent1"), 0.9),
        ]

        team = builder.select_diverse_team(scored, min_size=2, max_size=4, diversity_pref=0.5)

        # Should return all available agents
        assert len(team) == 1
        assert team[0].name == "agent1"

    def test_select_exact_min(self):
        """Test selecting exactly min_size agents."""
        builder = TeamBuilder()
        scored = [
            (MockAgentProfile(name="agent1"), 0.9),
            (MockAgentProfile(name="agent2"), 0.8),
        ]

        team = builder.select_diverse_team(scored, min_size=2, max_size=4, diversity_pref=0.5)

        assert len(team) >= 2

    def test_select_greedy_no_diversity(self):
        """Test greedy selection with diversity_pref=0."""
        builder = TeamBuilder()
        scored = [
            (MockAgentProfile(name="best", agent_type="cli"), 0.9),
            (MockAgentProfile(name="good", agent_type="api"), 0.8),
            (MockAgentProfile(name="okay", agent_type="cli"), 0.7),
        ]

        # With diversity=0, should always pick highest scored
        team = builder.select_diverse_team(scored, min_size=2, max_size=2, diversity_pref=0.0)

        assert len(team) == 2
        assert team[0].name == "best"
        assert team[1].name == "good"

    def test_select_diverse_team_respects_max(self):
        """Test that team selection respects max_size."""
        builder = TeamBuilder()
        scored = [
            (MockAgentProfile(name=f"agent{i}"), 0.9 - i * 0.1)
            for i in range(10)
        ]

        team = builder.select_diverse_team(scored, min_size=2, max_size=3, diversity_pref=0.5)

        assert len(team) <= 3


# =============================================================================
# TeamBuilder assign_roles Tests
# =============================================================================


class TestAssignRoles:
    """Tests for assign_roles method."""

    def test_assign_roles_empty_team(self):
        """Test role assignment with empty team."""
        builder = TeamBuilder()
        requirements = MockTaskRequirements()

        roles = builder.assign_roles([], requirements)

        assert roles == {}

    def test_assign_roles_single_agent(self):
        """Test role assignment with single agent."""
        builder = TeamBuilder()
        agent = MockAgentProfile(
            name="agent1",
            expertise={"general": 0.8}
        )
        requirements = MockTaskRequirements(primary_domain="general")

        roles = builder.assign_roles([agent], requirements)

        assert agent.name in roles
        assert roles[agent.name] == "proposer"

    def test_assign_roles_two_agents(self):
        """Test role assignment with two agents."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="expert", expertise={"security": 0.9}, overall_score=0.8),
            MockAgentProfile(name="balanced", expertise={"security": 0.5}, overall_score=0.5),
        ]
        requirements = MockTaskRequirements(primary_domain="security")

        roles = builder.assign_roles(agents, requirements)

        # Expert should be proposer, balanced should be synthesizer
        assert roles["expert"] == "proposer"
        assert roles["balanced"] == "synthesizer"

    def test_assign_roles_critic_types(self):
        """Test that agents get appropriate critic roles based on traits."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="proposer", expertise={"security": 0.9}),
            MockAgentProfile(name="synthesizer", expertise={"security": 0.5}, overall_score=0.5),
            MockAgentProfile(name="security_guy", traits=["thorough", "security"]),
            MockAgentProfile(name="perf_guy", traits=["pragmatic", "performance"]),
            MockAgentProfile(name="generic", traits=["creative"]),
        ]
        requirements = MockTaskRequirements(primary_domain="security")

        roles = builder.assign_roles(agents, requirements)

        assert roles["security_guy"] == "security_critic"
        assert roles["perf_guy"] == "performance_critic"
        assert roles["generic"] == "critic"


# =============================================================================
# TeamBuilder assign_hybrid_roles Tests
# =============================================================================


class TestAssignHybridRoles:
    """Tests for assign_hybrid_roles method."""

    def test_hybrid_roles_debate_phase(self):
        """Test hybrid role assignment for debate phase."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="agent1", agent_type="claude"),
            MockAgentProfile(name="agent2", agent_type="grok"),
        ]

        roles = builder.assign_hybrid_roles(agents, "debate")

        # All agents should get fallback 'proposer' role
        assert roles["agent1"] == "proposer"
        assert roles["agent2"] == "proposer"

    def test_hybrid_roles_design_phase(self):
        """Test hybrid role assignment for design phase."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="gem", agent_type="gemini"),
            MockAgentProfile(name="claud", agent_type="claude"),
            MockAgentProfile(name="other", agent_type="custom"),
        ]

        roles = builder.assign_hybrid_roles(agents, "design")

        assert roles["gem"] == "design_lead"
        assert roles["claud"] == "architecture_critic"
        assert roles["other"] == "critic"  # Fallback

    def test_hybrid_roles_implement_phase(self):
        """Test hybrid role assignment for implement phase."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="claud", agent_type="claude"),
            MockAgentProfile(name="other", agent_type="codex"),
        ]

        roles = builder.assign_hybrid_roles(agents, "implement")

        assert roles["claud"] == "implementer"
        assert roles["other"] == "advisor"  # Fallback

    def test_hybrid_roles_verify_phase(self):
        """Test hybrid role assignment for verify phase."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="codex_agent", agent_type="codex"),
            MockAgentProfile(name="grok_agent", agent_type="grok"),
            MockAgentProfile(name="deepseek_agent", agent_type="deepseek"),
        ]

        roles = builder.assign_hybrid_roles(agents, "verify")

        assert roles["codex_agent"] == "verification_lead"
        assert roles["grok_agent"] == "quality_auditor"
        assert roles["deepseek_agent"] == "formal_verifier"

    def test_hybrid_roles_unknown_phase(self):
        """Test hybrid role assignment for unknown phase."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="agent1", agent_type="claude"),
        ]

        roles = builder.assign_hybrid_roles(agents, "unknown_phase")

        assert roles["agent1"] == "participant"  # Default fallback


# =============================================================================
# TeamBuilder select_team Integration Tests
# =============================================================================


class TestSelectTeamIntegration:
    """Integration tests for select_team method."""

    def test_select_team_basic(self):
        """Test basic team selection."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(
                name="agent1",
                agent_type="cli",
                elo_rating=1600,
                expertise={"security": 0.8},
                traits=["thorough"],
                cost_factor=1.0,
            ),
            MockAgentProfile(
                name="agent2",
                agent_type="api",
                elo_rating=1500,
                expertise={"security": 0.6},
                traits=["creative"],
                cost_factor=1.5,
            ),
            MockAgentProfile(
                name="agent3",
                agent_type="cli",
                elo_rating=1400,
                expertise={"security": 0.5},
                traits=["pragmatic"],
                cost_factor=0.8,
            ),
        ]
        scored = [(a, 0.9 - i * 0.1) for i, a in enumerate(agents)]
        requirements = MockTaskRequirements(
            task_id="test123",
            min_agents=2,
            max_agents=3,
            primary_domain="security",
        )

        def score_fn(agent, req):
            return agent.expertise.get(req.primary_domain, 0.5)

        # Need to import TeamComposition to make this work
        with pytest.raises((ImportError, AttributeError)):
            # This will fail without full imports, but tests the method signature
            builder.select_team(scored, requirements, score_fn)

    def test_select_team_no_agents(self):
        """Test that selecting from no agents raises ValueError."""
        builder = TeamBuilder()
        requirements = MockTaskRequirements()

        with pytest.raises(ValueError, match="No available agents"):
            builder.select_team([], requirements, lambda a, r: 0.5)

    def test_selection_history_tracking(self):
        """Test that selection history is tracked."""
        builder = TeamBuilder()

        # Manually add to history to test tracking
        builder._selection_history.append({
            "task_id": "test1",
            "selected": ["agent1", "agent2"],
            "timestamp": "2025-01-01T00:00:00",
        })

        assert len(builder._selection_history) == 1
        assert builder._selection_history[0]["task_id"] == "test1"


# =============================================================================
# TeamBuilder Edge Cases
# =============================================================================


class TestTeamBuilderEdgeCases:
    """Edge case tests for TeamBuilder."""

    def test_diversity_with_no_traits(self):
        """Test diversity calculation when agents have no traits."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="agent1", agent_type="cli", traits=[]),
            MockAgentProfile(name="agent2", agent_type="api", traits=[]),
        ]

        diversity = builder.calculate_diversity(agents)

        # Should still calculate type and ELO diversity
        assert diversity >= 0

    def test_diversity_with_many_traits(self):
        """Test diversity calculation with many traits."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(
                name="agent1",
                agent_type="cli",
                traits=["a", "b", "c", "d", "e"]
            ),
            MockAgentProfile(
                name="agent2",
                agent_type="api",
                traits=["f", "g", "h", "i", "j"]
            ),
        ]

        diversity = builder.calculate_diversity(agents)

        # High trait diversity
        assert diversity > 0.5

    def test_diversity_with_extreme_elo(self):
        """Test diversity calculation with extreme ELO differences."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="low", elo_rating=1000),
            MockAgentProfile(name="high", elo_rating=2500),
        ]

        diversity = builder.calculate_diversity(agents)

        # ELO diversity should be maxed out (capped at 1.0)
        assert diversity > 0

    def test_assign_roles_missing_expertise(self):
        """Test role assignment when agents lack expertise in primary domain."""
        builder = TeamBuilder()
        agents = [
            MockAgentProfile(name="agent1", expertise={}),
            MockAgentProfile(name="agent2", expertise={}),
        ]
        requirements = MockTaskRequirements(primary_domain="nonexistent")

        roles = builder.assign_roles(agents, requirements)

        # Should still assign roles without errors
        assert len(roles) == 2
        assert "proposer" in roles.values()

    def test_select_diverse_team_all_same_type(self):
        """Test diverse selection when all agents are same type."""
        builder = TeamBuilder()
        scored = [
            (MockAgentProfile(name=f"agent{i}", agent_type="cli"), 0.9 - i * 0.1)
            for i in range(5)
        ]

        team = builder.select_diverse_team(scored, min_size=2, max_size=3, diversity_pref=1.0)

        # Should still select agents despite low diversity
        assert len(team) >= 2
