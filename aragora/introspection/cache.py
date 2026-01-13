"""
Caching layer for Agent Introspection API.

Provides in-memory caching for introspection data to minimize
performance impact during debates. Data is loaded once at Arena
initialization and reused throughout the debate.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

from .types import IntrospectionSnapshot

if TYPE_CHECKING:
    from aragora.agents.personas import PersonaManager
    from aragora.core import Agent
    from aragora.memory.store import CritiqueStore


class IntrospectionCache:
    """
    In-memory cache for introspection data.

    Follows the same pattern as Arena._historical_context_cache:
    - Data is loaded once at debate start via warm()
    - Subsequent get() calls return cached data
    - No mid-debate queries to avoid latency

    Usage:
        cache = IntrospectionCache()
        cache.warm(agents=agents, memory=critique_store)
        snapshot = cache.get("claude")
    """

    def __init__(self):
        self._cache: dict[str, IntrospectionSnapshot] = {}
        self._loaded_at: Optional[datetime] = None

    def warm(
        self,
        agents: list["Agent"],
        memory: Optional["CritiqueStore"] = None,
        persona_manager: Optional["PersonaManager"] = None,
    ) -> None:
        """
        Pre-load introspection data for all agents.

        Called once at Arena initialization to populate the cache.
        Each agent's data is aggregated from available sources.

        Args:
            agents: List of agents participating in the debate
            memory: Optional CritiqueStore for reputation data
            persona_manager: Optional PersonaManager for traits/expertise
        """
        from .api import get_agent_introspection

        self._cache.clear()
        self._loaded_at = datetime.now()

        for agent in agents:
            agent_name = agent.name if hasattr(agent, "name") else str(agent)
            snapshot = get_agent_introspection(
                agent_name=agent_name,
                memory=memory,
                persona_manager=persona_manager,
            )
            self._cache[agent_name] = snapshot

    def get(self, agent_name: str) -> Optional[IntrospectionSnapshot]:
        """
        Get cached introspection data for an agent.

        Returns None if the agent isn't in the cache.
        Does not query any data sources - only returns cached data.

        Args:
            agent_name: Name of the agent

        Returns:
            IntrospectionSnapshot if cached, None otherwise
        """
        return self._cache.get(agent_name)

    def invalidate(self) -> None:
        """
        Clear the cache.

        Call between debates if needed, or when data should be refreshed.
        """
        self._cache.clear()
        self._loaded_at = None

    @property
    def is_warm(self) -> bool:
        """Check if cache has been warmed."""
        return self._loaded_at is not None and len(self._cache) > 0

    @property
    def agent_count(self) -> int:
        """Number of agents in cache."""
        return len(self._cache)

    def get_all(self) -> dict[str, IntrospectionSnapshot]:
        """Get all cached snapshots."""
        return self._cache.copy()
