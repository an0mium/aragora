"""
Agent Introspection API.

Provides self-awareness data injection for agents during debates.
Agents receive their own track record (reputation, performance history)
as context, enabling meta-cognitive reasoning.

Active introspection extends this with per-round performance tracking
and reflective prompt generation via MetaReasoningEngine.

Usage:
    from aragora.introspection import IntrospectionCache, IntrospectionSnapshot

    # In Arena initialization (when enable_introspection=True):
    cache = IntrospectionCache()
    cache.warm(agents=agents, memory=critique_store)

    # In prompt building:
    snapshot = cache.get("claude")
    prompt_section = snapshot.to_prompt_section()

    # Active introspection (per-round updates):
    from aragora.introspection.active import (
        ActiveIntrospectionTracker,
        MetaReasoningEngine,
        IntrospectionGoals,
        RoundMetrics,
    )
"""

from .active import (
    ActiveIntrospectionTracker,
    AgentPerformanceSummary,
    IntrospectionGoals,
    MetaReasoningEngine,
    RoundMetrics,
)
from .api import format_introspection_section, get_agent_introspection
from .cache import IntrospectionCache
from .types import IntrospectionSnapshot

__all__ = [
    "ActiveIntrospectionTracker",
    "AgentPerformanceSummary",
    "IntrospectionCache",
    "IntrospectionGoals",
    "IntrospectionSnapshot",
    "MetaReasoningEngine",
    "RoundMetrics",
    "format_introspection_section",
    "get_agent_introspection",
]
