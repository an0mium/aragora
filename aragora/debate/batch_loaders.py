"""
Debate-Specific DataLoaders.

Provides batch loading infrastructure for debate operations to prevent N+1 queries
when fetching agents, ELO ratings, and related data.

Usage:
    from aragora.debate.batch_loaders import DebateLoaders, get_debate_loaders

    # Get request-scoped loaders
    loaders = get_debate_loaders()

    # Load agents in batches
    agents = await loaders.agents.load_many(agent_ids)

    # Load ELO ratings for agents
    ratings = await loaders.elo.load_many(agent_names)
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional

from aragora.performance import DataLoader, BatchResolver

logger = logging.getLogger(__name__)

# Context variable for request-scoped loaders
_loaders_context: contextvars.ContextVar[Optional["DebateLoaders"]] = contextvars.ContextVar(
    "debate_loaders", default=None
)


@dataclass
class ELORating:
    """ELO rating data for an agent."""

    agent_name: str
    rating: float
    games_played: int
    wins: int
    losses: int
    last_updated: Optional[str] = None


@dataclass
class AgentStats:
    """Aggregated statistics for an agent."""

    agent_name: str
    debate_count: int
    win_rate: float
    avg_confidence: float
    avg_response_time_ms: float
    domains: List[str]


class DebateLoaders:
    """
    Request-scoped DataLoaders for debate operations.

    Provides batched loading for:
    - Agent configurations
    - ELO ratings
    - Agent statistics
    - Debate histories
    """

    def __init__(
        self,
        elo_system: Optional[Any] = None,
        agent_registry: Optional[Any] = None,
        debate_store: Optional[Any] = None,
        max_batch_size: int = 50,
    ):
        """
        Initialize debate loaders.

        Args:
            elo_system: Optional ELO system for rating lookups
            agent_registry: Optional agent registry for config lookups
            debate_store: Optional debate store for history lookups
            max_batch_size: Maximum items per batch
        """
        self._elo_system = elo_system
        self._agent_registry = agent_registry
        self._debate_store = debate_store
        self._max_batch_size = max_batch_size
        self._token: Optional[contextvars.Token] = None

        # Initialize loaders
        self._elo_loader: Optional[DataLoader[str, Optional[ELORating]]] = None
        self._stats_loader: Optional[DataLoader[str, Optional[AgentStats]]] = None
        self._resolver = BatchResolver()

    @property
    def elo(self) -> DataLoader[str, Optional[ELORating]]:
        """Get ELO rating loader."""
        if self._elo_loader is None:
            self._elo_loader = DataLoader(
                self._batch_load_elo,
                max_batch_size=self._max_batch_size,
                name="elo_loader",
            )
        return self._elo_loader

    @property
    def stats(self) -> DataLoader[str, Optional[AgentStats]]:
        """Get agent stats loader."""
        if self._stats_loader is None:
            self._stats_loader = DataLoader(
                self._batch_load_stats,
                max_batch_size=self._max_batch_size,
                name="stats_loader",
            )
        return self._stats_loader

    async def _batch_load_elo(self, agent_names: List[str]) -> List[Optional[ELORating]]:
        """
        Batch load ELO ratings for multiple agents.

        Args:
            agent_names: List of agent names to load ratings for

        Returns:
            List of ELORating objects (or None for missing agents)
        """
        if not self._elo_system:
            logger.debug("No ELO system configured, returning empty ratings")
            return [None] * len(agent_names)

        try:
            # Try to use batch method if available
            if hasattr(self._elo_system, "get_ratings_batch"):
                ratings_dict = await self._maybe_await(
                    self._elo_system.get_ratings_batch(agent_names)
                )
                return [
                    self._elo_to_rating(agent, ratings_dict.get(agent)) for agent in agent_names
                ]

            # Fall back to individual lookups (still batched at DataLoader level)
            results: List[Optional[ELORating]] = []
            for name in agent_names:
                try:
                    if hasattr(self._elo_system, "get_rating"):
                        rating = await self._maybe_await(self._elo_system.get_rating(name))
                        results.append(self._elo_to_rating(name, rating))
                    else:
                        results.append(None)
                except Exception as e:
                    logger.debug(f"Failed to load ELO for {name}: {e}")
                    results.append(None)
            return results

        except Exception as e:
            logger.warning(f"Batch ELO load failed: {e}")
            return [None] * len(agent_names)

    async def _batch_load_stats(self, agent_names: List[str]) -> List[Optional[AgentStats]]:
        """
        Batch load statistics for multiple agents.

        Args:
            agent_names: List of agent names to load stats for

        Returns:
            List of AgentStats objects (or None for missing agents)
        """
        if not self._debate_store:
            logger.debug("No debate store configured, returning empty stats")
            return [None] * len(agent_names)

        try:
            # Try to use batch method if available
            if hasattr(self._debate_store, "get_agent_stats_batch"):
                stats_dict = await self._maybe_await(
                    self._debate_store.get_agent_stats_batch(agent_names)
                )
                return [self._dict_to_stats(agent, stats_dict.get(agent)) for agent in agent_names]

            # Fall back to individual lookups
            results: List[Optional[AgentStats]] = []
            for name in agent_names:
                try:
                    if hasattr(self._debate_store, "get_agent_stats"):
                        stats = await self._maybe_await(self._debate_store.get_agent_stats(name))
                        results.append(self._dict_to_stats(name, stats))
                    else:
                        results.append(None)
                except Exception as e:
                    logger.debug(f"Failed to load stats for {name}: {e}")
                    results.append(None)
            return results

        except Exception as e:
            logger.warning(f"Batch stats load failed: {e}")
            return [None] * len(agent_names)

    def _elo_to_rating(self, agent_name: str, data: Optional[Any]) -> Optional[ELORating]:
        """Convert raw ELO data to ELORating dataclass."""
        if data is None:
            return None

        if isinstance(data, ELORating):
            return data

        if isinstance(data, dict):
            return ELORating(
                agent_name=agent_name,
                rating=data.get("rating", 1000.0),
                games_played=data.get("games_played", 0),
                wins=data.get("wins", 0),
                losses=data.get("losses", 0),
                last_updated=data.get("last_updated"),
            )

        # Handle numeric rating
        if isinstance(data, (int, float)):
            return ELORating(
                agent_name=agent_name,
                rating=float(data),
                games_played=0,
                wins=0,
                losses=0,
            )

        return None

    def _dict_to_stats(self, agent_name: str, data: Optional[Dict]) -> Optional[AgentStats]:
        """Convert raw stats dict to AgentStats dataclass."""
        if data is None:
            return None

        if isinstance(data, AgentStats):
            return data

        if isinstance(data, dict):
            return AgentStats(
                agent_name=agent_name,
                debate_count=data.get("debate_count", 0),
                win_rate=data.get("win_rate", 0.0),
                avg_confidence=data.get("avg_confidence", 0.0),
                avg_response_time_ms=data.get("avg_response_time_ms", 0.0),
                domains=data.get("domains", []),
            )

        return None

    async def _maybe_await(self, result: Any) -> Any:
        """Await result if it's a coroutine."""
        if asyncio.iscoroutine(result):
            return await result
        return result

    def clear(self) -> None:
        """Clear all loader caches."""
        if self._elo_loader:
            self._elo_loader.clear()
        if self._stats_loader:
            self._stats_loader.clear()
        self._resolver.clear_all()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all loaders."""
        stats = {}
        if self._elo_loader:
            stats["elo"] = self._elo_loader.stats.to_dict()
        if self._stats_loader:
            stats["stats"] = self._stats_loader.stats.to_dict()
        stats["resolver"] = self._resolver.stats()
        return stats

    def __enter__(self) -> "DebateLoaders":
        """Enter context - set as current loaders."""
        self._token = _loaders_context.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context - clear and reset."""
        if self._token is not None:
            _loaders_context.reset(self._token)
            self._token = None
        self.clear()


def get_debate_loaders() -> Optional[DebateLoaders]:
    """Get the current request-scoped debate loaders."""
    return _loaders_context.get()


@contextmanager
def debate_loader_context(
    elo_system: Optional[Any] = None,
    agent_registry: Optional[Any] = None,
    debate_store: Optional[Any] = None,
    max_batch_size: int = 50,
) -> Generator[DebateLoaders, None, None]:
    """
    Context manager for request-scoped debate loaders.

    Args:
        elo_system: Optional ELO system for rating lookups
        agent_registry: Optional agent registry for config lookups
        debate_store: Optional debate store for history lookups
        max_batch_size: Maximum items per batch

    Yields:
        DebateLoaders instance

    Example:
        with debate_loader_context(elo_system=elo) as loaders:
            ratings = await loaders.elo.load_many(["claude", "gemini"])
    """
    loaders = DebateLoaders(
        elo_system=elo_system,
        agent_registry=agent_registry,
        debate_store=debate_store,
        max_batch_size=max_batch_size,
    )
    with loaders:
        yield loaders


__all__ = [
    "DebateLoaders",
    "ELORating",
    "AgentStats",
    "get_debate_loaders",
    "debate_loader_context",
]
