"""
Selection Plugins - Extensible agent selection algorithms.

Provides a Protocol-based plugin architecture for customizing:
- Agent scoring algorithms
- Team composition strategies
- Role assignment logic
- Domain detection

Built-in strategies:
- ELOWeightedScorer: Default ELO + expertise weighted scoring
- DiverseTeamSelector: Diversity-aware team selection
- DomainBasedRoleAssigner: Domain expertise-based role assignment

Example custom plugin:
    @register_scorer("my-scorer")
    class MyScorer(ScorerProtocol):
        def score_agent(self, agent, requirements, context):
            return agent.elo_rating / 2000  # Simple ELO-only scoring
"""

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
)
from aragora.plugins.selection.strategies import (
    DiverseTeamSelector,
    DomainBasedRoleAssigner,
    ELOWeightedScorer,
)

__all__ = [
    # Protocols
    "ScorerProtocol",
    "TeamSelectorProtocol",
    "RoleAssignerProtocol",
    "SelectionContext",
    # Registry
    "SelectionPluginRegistry",
    "get_selection_registry",
    "register_scorer",
    "register_team_selector",
    "register_role_assigner",
    # Built-in strategies
    "ELOWeightedScorer",
    "DiverseTeamSelector",
    "DomainBasedRoleAssigner",
]
