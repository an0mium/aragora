"""
Shared type definitions for the agents module.

This module provides common type aliases and TypeVars used across
the agents package to avoid duplication and ensure consistency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, TypeAlias, TypeVar

# Generic type variable for decorators and generic functions
T = TypeVar("T")

# Covariant type variable for return types
T_co = TypeVar("T_co", covariant=True)

# Type variable bound to Agent for agent-specific generics
if TYPE_CHECKING:
    from aragora.agents.api_agents import APIAgent
    from aragora.agents.api_agents.openrouter import OpenRouterAgent
    from aragora.agents.cli_agents import CLIAgent

    # Union type for any agent instance
    Agent: TypeAlias = APIAgent | CLIAgent

    # Type alias for fallback agent references
    FallbackAgent: TypeAlias = OpenRouterAgent | None

    # Callback type for agent operations
    AgentCallback: TypeAlias = Callable[[Agent, Any], Any]


__all__ = [
    "T",
    "T_co",
    "Agent",
    "FallbackAgent",
    "AgentCallback",
]
