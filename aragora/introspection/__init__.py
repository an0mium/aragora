"""
Agent Introspection API.

Provides self-awareness data injection for agents during debates.
Agents receive their own track record (reputation, performance history)
as context, enabling meta-cognitive reasoning.

Usage:
    from aragora.introspection import IntrospectionCache, IntrospectionSnapshot

    # In Arena initialization (when enable_introspection=True):
    cache = IntrospectionCache()
    cache.warm(agents=agents, memory=critique_store)

    # In prompt building:
    snapshot = cache.get("claude")
    prompt_section = snapshot.to_prompt_section()
"""

from .api import format_introspection_section, get_agent_introspection
from .cache import IntrospectionCache
from .types import IntrospectionSnapshot

__all__ = [
    "IntrospectionSnapshot",
    "IntrospectionCache",
    "get_agent_introspection",
    "format_introspection_section",
]
