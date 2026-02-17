"""
Caching layer for Agent Introspection API.

Provides in-memory caching for introspection data to minimize
performance impact during debates. Data is loaded once at Arena
initialization and reused throughout the debate.

Supports round-by-round incremental updates via update_round(),
which feeds metrics to an ActiveIntrospectionTracker for dynamic
introspection alongside the static historical snapshots.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from .types import IntrospectionSnapshot

if TYPE_CHECKING:
    from aragora.agents.personas import PersonaManager
    from aragora.core import Agent
    from aragora.introspection.active import (
        ActiveIntrospectionTracker,
        AgentPerformanceSummary,
        RoundMetrics,
    )
    from aragora.memory.store import CritiqueStore

logger = logging.getLogger(__name__)


class IntrospectionCache:
    """
    In-memory cache for introspection data.

    Follows the same pattern as Arena._historical_context_cache:
    - Data is loaded once at debate start via warm()
    - Subsequent get() calls return cached data
    - No mid-debate queries to avoid latency

    Additionally supports round-by-round updates via update_round(),
    which tracks live performance metrics through an
    ActiveIntrospectionTracker. Use get_active_summary() to retrieve
    dynamic performance data alongside the static historical snapshot.

    Usage:
        cache = IntrospectionCache()
        cache.warm(agents=agents, memory=critique_store)
        snapshot = cache.get("claude")

        # After each round:
        cache.update_round("claude", round_num=1, metrics=round_metrics)
        summary = cache.get_active_summary("claude")
    """

    def __init__(self) -> None:
        self._cache: dict[str, IntrospectionSnapshot] = {}
        self._loaded_at: datetime | None = None
        self._active_tracker: ActiveIntrospectionTracker | None = None
        self._last_round_updated: dict[str, int] = {}

    def warm(
        self,
        agents: list[Agent],
        memory: CritiqueStore | None = None,
        persona_manager: PersonaManager | None = None,
    ) -> None:
        """
        Pre-load introspection data for all agents.

        Called once at Arena initialization to populate the cache.
        Each agent's data is aggregated from available sources.
        Also initializes the active tracker for round-by-round updates.

        Args:
            agents: List of agents participating in the debate
            memory: Optional CritiqueStore for reputation data
            persona_manager: Optional PersonaManager for traits/expertise
        """
        from .api import get_agent_introspection

        self._cache.clear()
        self._loaded_at = datetime.now()
        self._last_round_updated.clear()

        # Initialize active tracker
        try:
            from .active import ActiveIntrospectionTracker

            self._active_tracker = ActiveIntrospectionTracker()
        except ImportError:
            logger.debug("[introspection] Active tracker not available")
            self._active_tracker = None

        for agent in agents:
            agent_name = agent.name if hasattr(agent, "name") else str(agent)
            snapshot = get_agent_introspection(
                agent_name=agent_name,
                memory=memory,
                persona_manager=persona_manager,
            )
            self._cache[agent_name] = snapshot

    def get(self, agent_name: str) -> IntrospectionSnapshot | None:
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

    def update_round(
        self,
        agent_name: str,
        round_num: int,
        metrics: RoundMetrics,
    ) -> None:
        """Record per-round performance metrics for an agent.

        Delegates to the internal ActiveIntrospectionTracker.
        Safe to call even if the active tracker is not available
        (silently no-ops in that case).

        Args:
            agent_name: Name of the agent
            round_num: Round number (1-indexed)
            metrics: Metrics collected for this round
        """
        if self._active_tracker is None:
            return

        try:
            self._active_tracker.update_round(agent_name, round_num, metrics)
            self._last_round_updated[agent_name] = round_num
        except (TypeError, ValueError, AttributeError) as e:
            logger.debug(
                "[introspection] Failed to update round %d for %s: %s",
                round_num,
                agent_name,
                e,
            )

    def get_active_summary(
        self,
        agent_name: str,
    ) -> AgentPerformanceSummary | None:
        """Get the active (live) performance summary for an agent.

        Returns None if no active data has been recorded.

        Args:
            agent_name: Name of the agent

        Returns:
            AgentPerformanceSummary if available, None otherwise
        """
        if self._active_tracker is None:
            return None
        return self._active_tracker.get_summary(agent_name)

    def get_last_round_updated(self, agent_name: str) -> int | None:
        """Get the last round number that was updated for an agent.

        Returns None if no rounds have been recorded.
        """
        return self._last_round_updated.get(agent_name)

    @property
    def has_active_tracker(self) -> bool:
        """Check if active tracking is available."""
        return self._active_tracker is not None

    def invalidate(self) -> None:
        """
        Clear the cache.

        Call between debates if needed, or when data should be refreshed.
        Also resets the active tracker.
        """
        self._cache.clear()
        self._loaded_at = None
        self._last_round_updated.clear()
        if self._active_tracker is not None:
            self._active_tracker.reset()

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
