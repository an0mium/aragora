"""
Selection Plugin Registry - Central registry for selection algorithms.

Provides:
- Plugin registration via decorators
- Plugin discovery and listing
- Default plugin configuration
- Type-safe plugin retrieval
"""

import logging
from typing import Callable, Optional, Type, TypeVar

from aragora.plugins.selection.protocols import (
    RoleAssignerProtocol,
    ScorerProtocol,
    TeamSelectorProtocol,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class SelectionPluginRegistry:
    """
    Central registry for selection algorithm plugins.

    Supports three plugin types:
    - Scorers: Agent scoring algorithms
    - TeamSelectors: Team composition algorithms
    - RoleAssigners: Role assignment algorithms

    Usage:
        registry = get_selection_registry()

        # Get default scorer
        scorer = registry.get_scorer()

        # Get custom scorer
        scorer = registry.get_scorer("ml-based")

        # List available scorers
        for name in registry.list_scorers():
            print(name)
    """

    def __init__(self):
        self._scorers: dict[str, Type[ScorerProtocol]] = {}
        self._team_selectors: dict[str, Type[TeamSelectorProtocol]] = {}
        self._role_assigners: dict[str, Type[RoleAssignerProtocol]] = {}

        # Default plugin names
        self._default_scorer: str = "elo-weighted"
        self._default_team_selector: str = "diverse"
        self._default_role_assigner: str = "domain-based"

    def register_scorer(
        self,
        name: str,
        scorer_class: Type[ScorerProtocol],
        set_default: bool = False,
    ) -> None:
        """Register a scorer plugin."""
        if name in self._scorers:
            logger.warning(f"Overwriting existing scorer: {name}")
        self._scorers[name] = scorer_class
        if set_default:
            self._default_scorer = name
        logger.debug(f"Registered scorer: {name}")

    def register_team_selector(
        self,
        name: str,
        selector_class: Type[TeamSelectorProtocol],
        set_default: bool = False,
    ) -> None:
        """Register a team selector plugin."""
        if name in self._team_selectors:
            logger.warning(f"Overwriting existing team selector: {name}")
        self._team_selectors[name] = selector_class
        if set_default:
            self._default_team_selector = name
        logger.debug(f"Registered team selector: {name}")

    def register_role_assigner(
        self,
        name: str,
        assigner_class: Type[RoleAssignerProtocol],
        set_default: bool = False,
    ) -> None:
        """Register a role assigner plugin."""
        if name in self._role_assigners:
            logger.warning(f"Overwriting existing role assigner: {name}")
        self._role_assigners[name] = assigner_class
        if set_default:
            self._default_role_assigner = name
        logger.debug(f"Registered role assigner: {name}")

    def get_scorer(self, name: Optional[str] = None) -> ScorerProtocol:
        """Get a scorer instance by name (or default)."""
        name = name or self._default_scorer
        if name not in self._scorers:
            raise KeyError(f"Unknown scorer: {name}. Available: {list(self._scorers.keys())}")
        return self._scorers[name]()

    def get_team_selector(self, name: Optional[str] = None) -> TeamSelectorProtocol:
        """Get a team selector instance by name (or default)."""
        name = name or self._default_team_selector
        if name not in self._team_selectors:
            raise KeyError(
                f"Unknown team selector: {name}. Available: {list(self._team_selectors.keys())}"
            )
        return self._team_selectors[name]()

    def get_role_assigner(self, name: Optional[str] = None) -> RoleAssignerProtocol:
        """Get a role assigner instance by name (or default)."""
        name = name or self._default_role_assigner
        if name not in self._role_assigners:
            raise KeyError(
                f"Unknown role assigner: {name}. Available: {list(self._role_assigners.keys())}"
            )
        return self._role_assigners[name]()

    def list_scorers(self) -> list[str]:
        """List available scorer names."""
        return list(self._scorers.keys())

    def list_team_selectors(self) -> list[str]:
        """List available team selector names."""
        return list(self._team_selectors.keys())

    def list_role_assigners(self) -> list[str]:
        """List available role assigner names."""
        return list(self._role_assigners.keys())

    def get_scorer_info(self, name: str) -> dict:
        """Get info about a scorer."""
        if name not in self._scorers:
            raise KeyError(f"Unknown scorer: {name}")
        instance = self._scorers[name]()
        return {
            "name": name,
            "description": instance.description,
            "is_default": name == self._default_scorer,
        }

    def get_team_selector_info(self, name: str) -> dict:
        """Get info about a team selector."""
        if name not in self._team_selectors:
            raise KeyError(f"Unknown team selector: {name}")
        instance = self._team_selectors[name]()
        return {
            "name": name,
            "description": instance.description,
            "is_default": name == self._default_team_selector,
        }

    def get_role_assigner_info(self, name: str) -> dict:
        """Get info about a role assigner."""
        if name not in self._role_assigners:
            raise KeyError(f"Unknown role assigner: {name}")
        instance = self._role_assigners[name]()
        return {
            "name": name,
            "description": instance.description,
            "is_default": name == self._default_role_assigner,
        }

    def list_all_plugins(self) -> dict[str, list[dict]]:
        """List all plugins with their info."""
        return {
            "scorers": [self.get_scorer_info(name) for name in self._scorers],
            "team_selectors": [self.get_team_selector_info(name) for name in self._team_selectors],
            "role_assigners": [self.get_role_assigner_info(name) for name in self._role_assigners],
        }


# Global registry instance
_registry: Optional[SelectionPluginRegistry] = None


def get_selection_registry() -> SelectionPluginRegistry:
    """Get the global selection plugin registry."""
    global _registry
    if _registry is None:
        _registry = SelectionPluginRegistry()
        # Register built-in strategies
        _register_builtins(_registry)
    return _registry


def reset_selection_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None


def _register_builtins(registry: SelectionPluginRegistry) -> None:
    """Register built-in selection strategies."""
    from aragora.plugins.selection.strategies import (
        DiverseTeamSelector,
        DomainBasedRoleAssigner,
        ELOWeightedScorer,
        GreedyTeamSelector,
        RandomTeamSelector,
        SimpleRoleAssigner,
    )

    # Scorers
    registry.register_scorer("elo-weighted", ELOWeightedScorer, set_default=True)

    # Team selectors
    registry.register_team_selector("diverse", DiverseTeamSelector, set_default=True)
    registry.register_team_selector("greedy", GreedyTeamSelector)
    registry.register_team_selector("random", RandomTeamSelector)

    # Role assigners
    registry.register_role_assigner("domain-based", DomainBasedRoleAssigner, set_default=True)
    registry.register_role_assigner("simple", SimpleRoleAssigner)


def register_scorer(
    name: str,
    set_default: bool = False,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a scorer plugin.

    Usage:
        @register_scorer("my-scorer")
        class MyScorer:
            def score_agent(self, agent, requirements, context):
                return agent.elo_rating / 2000

            @property
            def name(self) -> str:
                return "my-scorer"

            @property
            def description(self) -> str:
                return "Simple ELO-only scoring"
    """

    def decorator(cls: Type[T]) -> Type[T]:
        get_selection_registry().register_scorer(name, cls, set_default)
        return cls

    return decorator


def register_team_selector(
    name: str,
    set_default: bool = False,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a team selector plugin.

    Usage:
        @register_team_selector("my-selector")
        class MySelector:
            def select_team(self, scored_agents, requirements, context):
                return [a for a, s in scored_agents[:requirements.max_agents]]
            ...
    """

    def decorator(cls: Type[T]) -> Type[T]:
        get_selection_registry().register_team_selector(name, cls, set_default)
        return cls

    return decorator


def register_role_assigner(
    name: str,
    set_default: bool = False,
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a role assigner plugin.

    Usage:
        @register_role_assigner("my-assigner")
        class MyAssigner:
            def assign_roles(self, team, requirements, context, phase):
                return {a.name: "participant" for a in team}
            ...
    """

    def decorator(cls: Type[T]) -> Type[T]:
        get_selection_registry().register_role_assigner(name, cls, set_default)
        return cls

    return decorator
