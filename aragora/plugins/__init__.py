"""
Plugin System - Extensible tools for aragora debates.

Provides a manifest-based plugin architecture with sandboxed execution:
- PluginManifest: Schema for declaring plugin capabilities
- PluginRunner: Sandboxed execution environment
- Built-in plugins: lint, security-scan, test-runner

Selection Plugins (aragora.plugins.selection):
- ScorerProtocol: Agent scoring algorithms
- TeamSelectorProtocol: Team composition algorithms
- RoleAssignerProtocol: Role assignment algorithms

Key design decisions:
- Python-first (not WASM): Uses ProofExecutor's restricted namespace
- Manifest validation: Plugins must declare their requirements
- Curated set: Ship with core distribution, user plugins later
"""

from aragora.plugins.manifest import PluginCapability, PluginManifest, PluginRequirement
from aragora.plugins.runner import PluginContext, PluginResult, PluginRunner

# Selection plugins (Protocol-based)
from aragora.plugins.selection import (
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

__all__ = [
    # Execution plugins
    "PluginManifest",
    "PluginCapability",
    "PluginRequirement",
    "PluginRunner",
    "PluginResult",
    "PluginContext",
    # Selection plugins
    "ScorerProtocol",
    "TeamSelectorProtocol",
    "RoleAssignerProtocol",
    "SelectionContext",
    "SelectionPluginRegistry",
    "get_selection_registry",
    "register_scorer",
    "register_team_selector",
    "register_role_assigner",
]
