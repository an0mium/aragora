"""Comprehensive tests for selection plugin registry.

Tests the plugin registry functionality including:
- SelectionPluginRegistry core operations
- Plugin registration and retrieval
- Default plugin management
- Plugin info and listing
- Global registry functions
- Registration decorators
- Built-in plugin integration
"""

import pytest

from aragora.plugins.selection.protocols import (
    RoleAssignerProtocol,
    ScorerProtocol,
    SelectionContext,
    TeamSelectorProtocol,
)
from aragora.plugins.selection.registry import (
    SelectionPluginRegistry,
    get_selection_registry,
    register_role_assigner,
    register_scorer,
    register_team_selector,
    reset_selection_registry,
    _register_builtins,
)
from aragora.plugins.selection.strategies import (
    DiverseTeamSelector,
    DomainBasedRoleAssigner,
    ELOWeightedScorer,
    GreedyTeamSelector,
    RandomTeamSelector,
    SimpleRoleAssigner,
)


# =============================================================================
# SelectionPluginRegistry Core Tests
# =============================================================================


class TestSelectionPluginRegistryCore:
    """Core functionality tests for SelectionPluginRegistry."""

    def test_empty_registry_initialization(self):
        """Fresh registry is empty."""
        registry = SelectionPluginRegistry()

        assert len(registry._scorers) == 0
        assert len(registry._team_selectors) == 0
        assert len(registry._role_assigners) == 0

    def test_default_plugin_names(self):
        """Registry has default plugin name preferences."""
        registry = SelectionPluginRegistry()

        assert registry._default_scorer == "elo-weighted"
        assert registry._default_team_selector == "diverse"
        assert registry._default_role_assigner == "domain-based"


# =============================================================================
# Scorer Registration Tests
# =============================================================================


class TestScorerRegistration:
    """Tests for scorer plugin registration."""

    def test_register_scorer(self):
        """Can register a scorer plugin."""
        registry = SelectionPluginRegistry()

        class TestScorer:
            @property
            def name(self) -> str:
                return "test-scorer"

            @property
            def description(self) -> str:
                return "Test scorer"

            def score_agent(self, agent, requirements, context):
                return 0.5

        registry.register_scorer("test-scorer", TestScorer)

        assert "test-scorer" in registry._scorers
        assert registry._scorers["test-scorer"] is TestScorer

    def test_register_scorer_as_default(self):
        """Can register scorer as default."""
        registry = SelectionPluginRegistry()

        class DefaultScorer:
            @property
            def name(self) -> str:
                return "my-default"

            @property
            def description(self) -> str:
                return "My default scorer"

            def score_agent(self, agent, requirements, context):
                return 0.5

        registry.register_scorer("my-default", DefaultScorer, set_default=True)

        assert registry._default_scorer == "my-default"

    def test_register_scorer_overwrites_existing(self):
        """Registering scorer with same name overwrites."""
        registry = SelectionPluginRegistry()

        class OldScorer:
            @property
            def name(self) -> str:
                return "same-name"

            @property
            def description(self) -> str:
                return "Old"

            def score_agent(self, agent, requirements, context):
                return 0.1

        class NewScorer:
            @property
            def name(self) -> str:
                return "same-name"

            @property
            def description(self) -> str:
                return "New"

            def score_agent(self, agent, requirements, context):
                return 0.9

        registry.register_scorer("same-name", OldScorer)
        registry.register_scorer("same-name", NewScorer)

        assert registry._scorers["same-name"] is NewScorer


# =============================================================================
# Team Selector Registration Tests
# =============================================================================


class TestTeamSelectorRegistration:
    """Tests for team selector plugin registration."""

    def test_register_team_selector(self):
        """Can register a team selector plugin."""
        registry = SelectionPluginRegistry()

        class TestSelector:
            @property
            def name(self) -> str:
                return "test-selector"

            @property
            def description(self) -> str:
                return "Test selector"

            def select_team(self, scored_agents, requirements, context):
                return [a for a, _ in scored_agents[:2]]

        registry.register_team_selector("test-selector", TestSelector)

        assert "test-selector" in registry._team_selectors
        assert registry._team_selectors["test-selector"] is TestSelector

    def test_register_team_selector_as_default(self):
        """Can register team selector as default."""
        registry = SelectionPluginRegistry()

        class DefaultSelector:
            @property
            def name(self) -> str:
                return "my-default"

            @property
            def description(self) -> str:
                return "My default selector"

            def select_team(self, scored_agents, requirements, context):
                return []

        registry.register_team_selector("my-default", DefaultSelector, set_default=True)

        assert registry._default_team_selector == "my-default"


# =============================================================================
# Role Assigner Registration Tests
# =============================================================================


class TestRoleAssignerRegistration:
    """Tests for role assigner plugin registration."""

    def test_register_role_assigner(self):
        """Can register a role assigner plugin."""
        registry = SelectionPluginRegistry()

        class TestAssigner:
            @property
            def name(self) -> str:
                return "test-assigner"

            @property
            def description(self) -> str:
                return "Test assigner"

            def assign_roles(self, team, requirements, context, phase=None):
                return {a.name: "participant" for a in team}

        registry.register_role_assigner("test-assigner", TestAssigner)

        assert "test-assigner" in registry._role_assigners
        assert registry._role_assigners["test-assigner"] is TestAssigner

    def test_register_role_assigner_as_default(self):
        """Can register role assigner as default."""
        registry = SelectionPluginRegistry()

        class DefaultAssigner:
            @property
            def name(self) -> str:
                return "my-default"

            @property
            def description(self) -> str:
                return "My default assigner"

            def assign_roles(self, team, requirements, context, phase=None):
                return {}

        registry.register_role_assigner("my-default", DefaultAssigner, set_default=True)

        assert registry._default_role_assigner == "my-default"


# =============================================================================
# Plugin Retrieval Tests
# =============================================================================


class TestPluginRetrieval:
    """Tests for retrieving plugins from registry."""

    def setup_method(self):
        """Reset registry and load builtins."""
        reset_selection_registry()

    def test_get_scorer_by_name(self):
        """Can get scorer by name."""
        registry = get_selection_registry()

        scorer = registry.get_scorer("elo-weighted")

        assert scorer is not None
        assert isinstance(scorer, ScorerProtocol)
        assert scorer.name == "elo-weighted"

    def test_get_scorer_default(self):
        """get_scorer() returns default when no name given."""
        registry = get_selection_registry()

        scorer = registry.get_scorer()

        assert scorer is not None
        assert scorer.name == "elo-weighted"

    def test_get_scorer_unknown_raises(self):
        """get_scorer() raises KeyError for unknown name."""
        registry = get_selection_registry()

        with pytest.raises(KeyError, match="Unknown scorer"):
            registry.get_scorer("nonexistent-scorer")

    def test_get_team_selector_by_name(self):
        """Can get team selector by name."""
        registry = get_selection_registry()

        selector = registry.get_team_selector("greedy")

        assert selector is not None
        assert isinstance(selector, TeamSelectorProtocol)
        assert selector.name == "greedy"

    def test_get_team_selector_default(self):
        """get_team_selector() returns default when no name given."""
        registry = get_selection_registry()

        selector = registry.get_team_selector()

        assert selector is not None
        assert selector.name == "diverse"

    def test_get_team_selector_unknown_raises(self):
        """get_team_selector() raises KeyError for unknown name."""
        registry = get_selection_registry()

        with pytest.raises(KeyError, match="Unknown team selector"):
            registry.get_team_selector("nonexistent-selector")

    def test_get_role_assigner_by_name(self):
        """Can get role assigner by name."""
        registry = get_selection_registry()

        assigner = registry.get_role_assigner("simple")

        assert assigner is not None
        assert isinstance(assigner, RoleAssignerProtocol)
        assert assigner.name == "simple"

    def test_get_role_assigner_default(self):
        """get_role_assigner() returns default when no name given."""
        registry = get_selection_registry()

        assigner = registry.get_role_assigner()

        assert assigner is not None
        assert assigner.name == "domain-based"

    def test_get_role_assigner_unknown_raises(self):
        """get_role_assigner() raises KeyError for unknown name."""
        registry = get_selection_registry()

        with pytest.raises(KeyError, match="Unknown role assigner"):
            registry.get_role_assigner("nonexistent-assigner")


# =============================================================================
# Plugin Listing Tests
# =============================================================================


class TestPluginListing:
    """Tests for listing plugins."""

    def setup_method(self):
        """Reset registry and load builtins."""
        reset_selection_registry()

    def test_list_scorers(self):
        """list_scorers() returns scorer names."""
        registry = get_selection_registry()

        scorers = registry.list_scorers()

        assert "elo-weighted" in scorers
        assert isinstance(scorers, list)

    def test_list_team_selectors(self):
        """list_team_selectors() returns selector names."""
        registry = get_selection_registry()

        selectors = registry.list_team_selectors()

        assert "diverse" in selectors
        assert "greedy" in selectors
        assert "random" in selectors
        assert isinstance(selectors, list)

    def test_list_role_assigners(self):
        """list_role_assigners() returns assigner names."""
        registry = get_selection_registry()

        assigners = registry.list_role_assigners()

        assert "domain-based" in assigners
        assert "simple" in assigners
        assert isinstance(assigners, list)


# =============================================================================
# Plugin Info Tests
# =============================================================================


class TestPluginInfo:
    """Tests for getting plugin info."""

    def setup_method(self):
        """Reset registry and load builtins."""
        reset_selection_registry()

    def test_get_scorer_info(self):
        """get_scorer_info() returns scorer details."""
        registry = get_selection_registry()

        info = registry.get_scorer_info("elo-weighted")

        assert info["name"] == "elo-weighted"
        assert "description" in info
        assert "is_default" in info
        assert info["is_default"] is True

    def test_get_scorer_info_unknown_raises(self):
        """get_scorer_info() raises for unknown scorer."""
        registry = get_selection_registry()

        with pytest.raises(KeyError, match="Unknown scorer"):
            registry.get_scorer_info("nonexistent")

    def test_get_team_selector_info(self):
        """get_team_selector_info() returns selector details."""
        registry = get_selection_registry()

        info = registry.get_team_selector_info("diverse")

        assert info["name"] == "diverse"
        assert "description" in info
        assert info["is_default"] is True

    def test_get_team_selector_info_non_default(self):
        """Non-default selector has is_default=False."""
        registry = get_selection_registry()

        info = registry.get_team_selector_info("greedy")

        assert info["is_default"] is False

    def test_get_team_selector_info_unknown_raises(self):
        """get_team_selector_info() raises for unknown selector."""
        registry = get_selection_registry()

        with pytest.raises(KeyError, match="Unknown team selector"):
            registry.get_team_selector_info("nonexistent")

    def test_get_role_assigner_info(self):
        """get_role_assigner_info() returns assigner details."""
        registry = get_selection_registry()

        info = registry.get_role_assigner_info("domain-based")

        assert info["name"] == "domain-based"
        assert "description" in info
        assert info["is_default"] is True

    def test_get_role_assigner_info_unknown_raises(self):
        """get_role_assigner_info() raises for unknown assigner."""
        registry = get_selection_registry()

        with pytest.raises(KeyError, match="Unknown role assigner"):
            registry.get_role_assigner_info("nonexistent")

    def test_list_all_plugins(self):
        """list_all_plugins() returns all plugin info."""
        registry = get_selection_registry()

        all_plugins = registry.list_all_plugins()

        assert "scorers" in all_plugins
        assert "team_selectors" in all_plugins
        assert "role_assigners" in all_plugins

        assert len(all_plugins["scorers"]) >= 1
        assert len(all_plugins["team_selectors"]) >= 3
        assert len(all_plugins["role_assigners"]) >= 2

        # Each entry should have info dict structure
        for scorer_info in all_plugins["scorers"]:
            assert "name" in scorer_info
            assert "description" in scorer_info
            assert "is_default" in scorer_info


# =============================================================================
# Global Registry Functions Tests
# =============================================================================


class TestGlobalRegistryFunctions:
    """Tests for global registry functions."""

    def test_get_selection_registry_singleton(self):
        """get_selection_registry() returns singleton."""
        reset_selection_registry()

        registry1 = get_selection_registry()
        registry2 = get_selection_registry()

        assert registry1 is registry2

    def test_reset_selection_registry(self):
        """reset_selection_registry() clears the singleton."""
        reset_selection_registry()
        registry1 = get_selection_registry()

        reset_selection_registry()
        registry2 = get_selection_registry()

        # Should be different instances after reset
        assert registry1 is not registry2

    def test_get_selection_registry_loads_builtins(self):
        """get_selection_registry() loads builtins automatically."""
        reset_selection_registry()

        registry = get_selection_registry()

        assert "elo-weighted" in registry.list_scorers()
        assert "diverse" in registry.list_team_selectors()
        assert "domain-based" in registry.list_role_assigners()


# =============================================================================
# Registration Decorators Tests
# =============================================================================


class TestRegistrationDecorators:
    """Tests for registration decorator functions."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_selection_registry()

    def test_register_scorer_decorator(self):
        """@register_scorer decorator registers scorer."""

        @register_scorer("decorator-scorer")
        class DecoratorScorer:
            @property
            def name(self) -> str:
                return "decorator-scorer"

            @property
            def description(self) -> str:
                return "Registered via decorator"

            def score_agent(self, agent, requirements, context):
                return 0.5

        registry = get_selection_registry()
        assert "decorator-scorer" in registry.list_scorers()

        scorer = registry.get_scorer("decorator-scorer")
        assert scorer.name == "decorator-scorer"

    def test_register_scorer_decorator_as_default(self):
        """@register_scorer decorator can set as default."""

        @register_scorer("new-default-scorer", set_default=True)
        class NewDefaultScorer:
            @property
            def name(self) -> str:
                return "new-default-scorer"

            @property
            def description(self) -> str:
                return "New default"

            def score_agent(self, agent, requirements, context):
                return 0.5

        registry = get_selection_registry()
        default = registry.get_scorer()

        assert default.name == "new-default-scorer"

    def test_register_team_selector_decorator(self):
        """@register_team_selector decorator registers selector."""

        @register_team_selector("decorator-selector")
        class DecoratorSelector:
            @property
            def name(self) -> str:
                return "decorator-selector"

            @property
            def description(self) -> str:
                return "Registered via decorator"

            def select_team(self, scored_agents, requirements, context):
                return []

        registry = get_selection_registry()
        assert "decorator-selector" in registry.list_team_selectors()

        selector = registry.get_team_selector("decorator-selector")
        assert selector.name == "decorator-selector"

    def test_register_team_selector_decorator_as_default(self):
        """@register_team_selector decorator can set as default."""

        @register_team_selector("new-default-selector", set_default=True)
        class NewDefaultSelector:
            @property
            def name(self) -> str:
                return "new-default-selector"

            @property
            def description(self) -> str:
                return "New default"

            def select_team(self, scored_agents, requirements, context):
                return []

        registry = get_selection_registry()
        default = registry.get_team_selector()

        assert default.name == "new-default-selector"

    def test_register_role_assigner_decorator(self):
        """@register_role_assigner decorator registers assigner."""

        @register_role_assigner("decorator-assigner")
        class DecoratorAssigner:
            @property
            def name(self) -> str:
                return "decorator-assigner"

            @property
            def description(self) -> str:
                return "Registered via decorator"

            def assign_roles(self, team, requirements, context, phase=None):
                return {}

        registry = get_selection_registry()
        assert "decorator-assigner" in registry.list_role_assigners()

        assigner = registry.get_role_assigner("decorator-assigner")
        assert assigner.name == "decorator-assigner"

    def test_register_role_assigner_decorator_as_default(self):
        """@register_role_assigner decorator can set as default."""

        @register_role_assigner("new-default-assigner", set_default=True)
        class NewDefaultAssigner:
            @property
            def name(self) -> str:
                return "new-default-assigner"

            @property
            def description(self) -> str:
                return "New default"

            def assign_roles(self, team, requirements, context, phase=None):
                return {}

        registry = get_selection_registry()
        default = registry.get_role_assigner()

        assert default.name == "new-default-assigner"

    def test_decorator_returns_original_class(self):
        """Decorator returns the original class unchanged."""

        @register_scorer("unchanged-scorer")
        class UnchangedScorer:
            @property
            def name(self) -> str:
                return "unchanged-scorer"

            @property
            def description(self) -> str:
                return "Test"

            def score_agent(self, agent, requirements, context):
                return 0.5

        # Class should be the original, not wrapped
        assert UnchangedScorer.__name__ == "UnchangedScorer"
        instance = UnchangedScorer()
        assert instance.name == "unchanged-scorer"


# =============================================================================
# Built-in Registration Tests
# =============================================================================


class TestBuiltinRegistration:
    """Tests for built-in plugin registration."""

    def test_register_builtins_function(self):
        """_register_builtins() registers all built-ins."""
        registry = SelectionPluginRegistry()

        _register_builtins(registry)

        # Scorers
        assert "elo-weighted" in registry._scorers

        # Team selectors
        assert "diverse" in registry._team_selectors
        assert "greedy" in registry._team_selectors
        assert "random" in registry._team_selectors

        # Role assigners
        assert "domain-based" in registry._role_assigners
        assert "ahmad" in registry._role_assigners
        assert "simple" in registry._role_assigners

    def test_builtin_scorers_implement_protocol(self):
        """Built-in scorers implement ScorerProtocol."""
        reset_selection_registry()
        registry = get_selection_registry()

        scorer = registry.get_scorer("elo-weighted")

        assert isinstance(scorer, ScorerProtocol)
        assert hasattr(scorer, "name")
        assert hasattr(scorer, "description")
        assert hasattr(scorer, "score_agent")

    def test_builtin_selectors_implement_protocol(self):
        """Built-in team selectors implement TeamSelectorProtocol."""
        reset_selection_registry()
        registry = get_selection_registry()

        for name in ["diverse", "greedy", "random"]:
            selector = registry.get_team_selector(name)

            assert isinstance(selector, TeamSelectorProtocol)
            assert hasattr(selector, "name")
            assert hasattr(selector, "description")
            assert hasattr(selector, "select_team")

    def test_builtin_assigners_implement_protocol(self):
        """Built-in role assigners implement RoleAssignerProtocol."""
        reset_selection_registry()
        registry = get_selection_registry()

        for name in ["domain-based", "ahmad", "simple"]:
            assigner = registry.get_role_assigner(name)

            assert isinstance(assigner, RoleAssignerProtocol)
            assert hasattr(assigner, "name")
            assert hasattr(assigner, "description")
            assert hasattr(assigner, "assign_roles")


# =============================================================================
# Plugin Instance Creation Tests
# =============================================================================


class TestPluginInstanceCreation:
    """Tests for plugin instance creation."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_selection_registry()

    def test_get_scorer_creates_new_instance(self):
        """get_scorer() creates new instance each time."""
        registry = get_selection_registry()

        scorer1 = registry.get_scorer("elo-weighted")
        scorer2 = registry.get_scorer("elo-weighted")

        # Should be different instances (new instance per call)
        assert scorer1 is not scorer2

    def test_get_team_selector_creates_new_instance(self):
        """get_team_selector() creates new instance each time."""
        registry = get_selection_registry()

        selector1 = registry.get_team_selector("diverse")
        selector2 = registry.get_team_selector("diverse")

        assert selector1 is not selector2

    def test_get_role_assigner_creates_new_instance(self):
        """get_role_assigner() creates new instance each time."""
        registry = get_selection_registry()

        assigner1 = registry.get_role_assigner("domain-based")
        assigner2 = registry.get_role_assigner("domain-based")

        assert assigner1 is not assigner2


# =============================================================================
# Error Message Tests
# =============================================================================


class TestErrorMessages:
    """Tests for error message clarity."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_selection_registry()

    def test_unknown_scorer_error_includes_available(self):
        """Unknown scorer error includes available scorers."""
        registry = get_selection_registry()

        try:
            registry.get_scorer("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            error_msg = str(e)
            assert "elo-weighted" in error_msg

    def test_unknown_selector_error_includes_available(self):
        """Unknown selector error includes available selectors."""
        registry = get_selection_registry()

        try:
            registry.get_team_selector("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            error_msg = str(e)
            assert "diverse" in error_msg
            assert "greedy" in error_msg

    def test_unknown_assigner_error_includes_available(self):
        """Unknown assigner error includes available assigners."""
        registry = get_selection_registry()

        try:
            registry.get_role_assigner("nonexistent")
            assert False, "Should have raised KeyError"
        except KeyError as e:
            error_msg = str(e)
            assert "domain-based" in error_msg
            assert "simple" in error_msg
