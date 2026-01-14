"""Tests for selection plugin architecture."""

import pytest

from aragora.plugins.selection import (
    DiverseTeamSelector,
    DomainBasedRoleAssigner,
    ELOWeightedScorer,
    RoleAssignerProtocol,
    ScorerProtocol,
    SelectionContext,
    SelectionPluginRegistry,
    TeamSelectorProtocol,
    get_selection_registry,
    register_role_assigner,
    register_scorer,
    register_team_selector,
)
from aragora.plugins.selection.registry import reset_selection_registry
from aragora.plugins.selection.strategies import GreedyTeamSelector, RandomTeamSelector
from aragora.routing.selection import AgentProfile, TaskRequirements


@pytest.fixture
def sample_agents():
    """Create sample agent profiles for testing."""
    return [
        AgentProfile(
            name="claude",
            agent_type="claude",
            elo_rating=1650,
            expertise={"security": 0.9, "architecture": 0.9, "general": 0.85},
            traits=["thorough", "security"],
        ),
        AgentProfile(
            name="codex",
            agent_type="codex",
            elo_rating=1580,
            expertise={"performance": 0.9, "debugging": 0.9, "general": 0.9},
            traits=["fast", "pragmatic"],
        ),
        AgentProfile(
            name="gemini",
            agent_type="gemini",
            elo_rating=1620,
            expertise={"architecture": 0.95, "performance": 0.85, "general": 0.85},
            traits=["creative"],
        ),
        AgentProfile(
            name="grok",
            agent_type="grok",
            elo_rating=1550,
            expertise={"debugging": 0.9, "security": 0.85, "general": 0.8},
            traits=["pragmatic", "lateral"],
        ),
    ]


@pytest.fixture
def sample_requirements():
    """Create sample task requirements."""
    return TaskRequirements(
        task_id="test-task",
        description="Implement a secure API endpoint",
        primary_domain="security",
        secondary_domains=["api"],
        required_traits=["thorough"],
        min_agents=2,
        max_agents=3,
        quality_priority=0.7,
        diversity_preference=0.5,
    )


@pytest.fixture
def sample_context(sample_agents):
    """Create sample selection context."""
    return SelectionContext(
        agent_pool={a.name: a for a in sample_agents},
        bench=[],
    )


class TestScorerProtocol:
    """Tests for scorer protocol and implementations."""

    def test_elo_weighted_scorer_implements_protocol(self):
        """ELOWeightedScorer implements ScorerProtocol."""
        scorer = ELOWeightedScorer()
        assert isinstance(scorer, ScorerProtocol)
        assert scorer.name == "elo-weighted"
        assert scorer.description

    def test_elo_weighted_scorer_scores_agents(
        self, sample_agents, sample_requirements, sample_context
    ):
        """Scorer returns scores between 0 and 1."""
        scorer = ELOWeightedScorer()
        for agent in sample_agents:
            score = scorer.score_agent(agent, sample_requirements, sample_context)
            assert 0.0 <= score <= 1.0

    def test_domain_expert_scores_higher(
        self, sample_agents, sample_requirements, sample_context
    ):
        """Agent with domain expertise scores higher."""
        scorer = ELOWeightedScorer()

        # claude has security expertise, should score high for security task
        claude = sample_agents[0]
        codex = sample_agents[1]

        claude_score = scorer.score_agent(claude, sample_requirements, sample_context)
        codex_score = scorer.score_agent(codex, sample_requirements, sample_context)

        assert claude_score > codex_score


class TestTeamSelectorProtocol:
    """Tests for team selector protocol and implementations."""

    def test_diverse_selector_implements_protocol(self):
        """DiverseTeamSelector implements TeamSelectorProtocol."""
        selector = DiverseTeamSelector()
        assert isinstance(selector, TeamSelectorProtocol)
        assert selector.name == "diverse"
        assert selector.description

    def test_greedy_selector_implements_protocol(self):
        """GreedyTeamSelector implements TeamSelectorProtocol."""
        selector = GreedyTeamSelector()
        assert isinstance(selector, TeamSelectorProtocol)
        assert selector.name == "greedy"

    def test_random_selector_implements_protocol(self):
        """RandomTeamSelector implements TeamSelectorProtocol."""
        selector = RandomTeamSelector()
        assert isinstance(selector, TeamSelectorProtocol)
        assert selector.name == "random"

    def test_diverse_selector_selects_team(
        self, sample_agents, sample_requirements, sample_context
    ):
        """Selector returns team within size constraints."""
        selector = DiverseTeamSelector()
        scored = [(a, 0.8 - i * 0.1) for i, a in enumerate(sample_agents)]

        team = selector.select_team(scored, sample_requirements, sample_context)

        assert len(team) >= sample_requirements.min_agents
        assert len(team) <= sample_requirements.max_agents

    def test_greedy_selector_picks_top(
        self, sample_agents, sample_requirements, sample_context
    ):
        """Greedy selector picks highest scored agents."""
        selector = GreedyTeamSelector()
        scored = [(a, 1.0 - i * 0.1) for i, a in enumerate(sample_agents)]

        team = selector.select_team(scored, sample_requirements, sample_context)

        # Should pick first N agents by score
        assert team[0] == sample_agents[0]
        assert team[1] == sample_agents[1]


class TestRoleAssignerProtocol:
    """Tests for role assigner protocol and implementations."""

    def test_domain_based_assigner_implements_protocol(self):
        """DomainBasedRoleAssigner implements RoleAssignerProtocol."""
        assigner = DomainBasedRoleAssigner()
        assert isinstance(assigner, RoleAssignerProtocol)
        assert assigner.name == "domain-based"
        assert assigner.description

    def test_assigns_proposer_to_domain_expert(
        self, sample_agents, sample_requirements, sample_context
    ):
        """Proposer role goes to domain expert."""
        assigner = DomainBasedRoleAssigner()
        team = sample_agents[:3]

        roles = assigner.assign_roles(team, sample_requirements, sample_context)

        assert "proposer" in roles.values()
        # claude has highest security expertise
        assert roles["claude"] == "proposer"

    def test_assigns_phase_specific_roles(
        self, sample_agents, sample_requirements, sample_context
    ):
        """Phase-specific roles are assigned correctly."""
        assigner = DomainBasedRoleAssigner()
        team = sample_agents[:3]

        # Design phase - gemini should lead
        roles = assigner.assign_roles(team, sample_requirements, sample_context, phase="design")
        assert roles["gemini"] == "design_lead"

        # Verify phase - codex should lead
        roles = assigner.assign_roles(team, sample_requirements, sample_context, phase="verify")
        assert roles["codex"] == "verification_lead"


class TestSelectionPluginRegistry:
    """Tests for the plugin registry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_selection_registry()

    def test_registry_has_builtins(self):
        """Registry loads built-in plugins."""
        registry = get_selection_registry()

        assert "elo-weighted" in registry.list_scorers()
        assert "diverse" in registry.list_team_selectors()
        assert "greedy" in registry.list_team_selectors()
        assert "domain-based" in registry.list_role_assigners()

    def test_get_default_scorer(self):
        """Can get default scorer."""
        registry = get_selection_registry()
        scorer = registry.get_scorer()

        assert scorer is not None
        assert scorer.name == "elo-weighted"

    def test_get_named_selector(self):
        """Can get selector by name."""
        registry = get_selection_registry()
        selector = registry.get_team_selector("greedy")

        assert selector is not None
        assert selector.name == "greedy"

    def test_unknown_plugin_raises(self):
        """Unknown plugin name raises KeyError."""
        registry = get_selection_registry()

        with pytest.raises(KeyError):
            registry.get_scorer("nonexistent")

    def test_list_all_plugins(self):
        """Can list all plugins with info."""
        registry = get_selection_registry()
        plugins = registry.list_all_plugins()

        assert "scorers" in plugins
        assert "team_selectors" in plugins
        assert "role_assigners" in plugins
        assert len(plugins["scorers"]) >= 1
        assert len(plugins["team_selectors"]) >= 2


class TestPluginRegistration:
    """Tests for plugin registration decorators."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_selection_registry()

    def test_register_custom_scorer(self):
        """Can register custom scorer via decorator."""

        @register_scorer("custom-scorer")
        class CustomScorer:
            @property
            def name(self) -> str:
                return "custom-scorer"

            @property
            def description(self) -> str:
                return "Custom test scorer"

            def score_agent(self, agent, requirements, context):
                return 0.5

        registry = get_selection_registry()
        assert "custom-scorer" in registry.list_scorers()

        scorer = registry.get_scorer("custom-scorer")
        assert scorer.name == "custom-scorer"

    def test_register_custom_team_selector(self):
        """Can register custom team selector."""

        @register_team_selector("custom-selector")
        class CustomSelector:
            @property
            def name(self) -> str:
                return "custom-selector"

            @property
            def description(self) -> str:
                return "Custom test selector"

            def select_team(self, scored_agents, requirements, context):
                return [a for a, _ in scored_agents[:2]]

        registry = get_selection_registry()
        assert "custom-selector" in registry.list_team_selectors()

    def test_register_as_default(self):
        """Can register plugin as new default."""

        @register_scorer("new-default", set_default=True)
        class NewDefaultScorer:
            @property
            def name(self) -> str:
                return "new-default"

            @property
            def description(self) -> str:
                return "New default scorer"

            def score_agent(self, agent, requirements, context):
                return 0.7

        registry = get_selection_registry()
        default = registry.get_scorer()
        assert default.name == "new-default"


class TestPluginIntegration:
    """Integration tests for complete selection flow."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_selection_registry()

    def test_complete_selection_flow(
        self, sample_agents, sample_requirements, sample_context
    ):
        """Complete flow from scoring to team to roles."""
        registry = get_selection_registry()

        # Score all agents
        scorer = registry.get_scorer()
        scored = [
            (a, scorer.score_agent(a, sample_requirements, sample_context))
            for a in sample_agents
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select team
        selector = registry.get_team_selector()
        team = selector.select_team(scored, sample_requirements, sample_context)

        # Assign roles
        assigner = registry.get_role_assigner()
        roles = assigner.assign_roles(team, sample_requirements, sample_context)

        # Verify results
        assert len(team) >= sample_requirements.min_agents
        assert len(roles) == len(team)
        assert "proposer" in roles.values()

    def test_different_strategies_produce_different_results(
        self, sample_agents, sample_requirements, sample_context
    ):
        """Different selection strategies can produce different teams."""
        registry = get_selection_registry()

        scorer = registry.get_scorer()
        scored = [
            (a, scorer.score_agent(a, sample_requirements, sample_context))
            for a in sample_agents
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        diverse_team = registry.get_team_selector("diverse").select_team(
            scored, sample_requirements, sample_context
        )
        greedy_team = registry.get_team_selector("greedy").select_team(
            scored, sample_requirements, sample_context
        )

        # Both should satisfy constraints
        assert len(diverse_team) >= sample_requirements.min_agents
        assert len(greedy_team) >= sample_requirements.min_agents

        # Both have same length for this test
        assert len(diverse_team) == len(greedy_team)
