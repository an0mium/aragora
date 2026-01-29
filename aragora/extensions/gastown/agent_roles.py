"""
Gastown Agent Roles Adapter — re-exports from nomic agent roles.

Provides the gastown extension entry point for agent hierarchy and
role management. Dashboard handlers import from here rather than
reaching into aragora.nomic.agent_roles directly.
"""

from __future__ import annotations

from aragora.nomic.agent_roles import (
    AgentHierarchy,
    AgentRole,
    RoleAssignment,
    RoleBasedRouter,
    RoleCapability,
)

__all__ = [
    "AgentHierarchy",
    "AgentRole",
    "RoleAssignment",
    "RoleBasedRouter",
    "RoleCapability",
]
