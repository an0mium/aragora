"""
Caching utilities for context gathering.

Provides cache management with task-keyed isolation and size limits
to prevent memory leaks and cross-debate context contamination.
"""

import hashlib
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Cache size limits to prevent unbounded memory growth
# These can be configured via environment variables for different deployment scenarios
MAX_EVIDENCE_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_EVIDENCE_CACHE", "100"))
MAX_CONTEXT_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_CONTEXT_CACHE", "100"))
MAX_CONTINUUM_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_CONTINUUM_CACHE", "100"))
MAX_TRENDING_CACHE_SIZE = int(os.getenv("ARAGORA_MAX_TRENDING_CACHE", "50"))


class ContextCache:
    """
    Manages caching for context gathering operations.

    Uses task-keyed isolation to prevent context leakage between debates.
    Implements FIFO eviction when caches exceed configured size limits.

    IMPORTANT: Each ContextGatherer should have its own ContextCache instance.
    Do not share a ContextCache across multiple debates.
    """

    def __init__(
        self,
        max_evidence_size: int | None = None,
        max_context_size: int | None = None,
        max_continuum_size: int | None = None,
        max_trending_size: int | None = None,
    ):
        """
        Initialize the context cache.

        Args:
            max_evidence_size: Maximum evidence pack cache entries (default: from env)
            max_context_size: Maximum context cache entries (default: from env)
            max_continuum_size: Maximum continuum cache entries (default: from env)
            max_trending_size: Maximum trending topics to cache (default: from env)
        """
        self._max_evidence_size = max_evidence_size or MAX_EVIDENCE_CACHE_SIZE
        self._max_context_size = max_context_size or MAX_CONTEXT_CACHE_SIZE
        self._max_continuum_size = max_continuum_size or MAX_CONTINUUM_CACHE_SIZE
        self._max_trending_size = max_trending_size or MAX_TRENDING_CACHE_SIZE

        # Cache for evidence pack (keyed by task hash to prevent leaks between debates)
        self._research_evidence_pack: dict[str, Any] = {}

        # Cache for research context (keyed by task hash to prevent leaks between debates)
        self._research_context_cache: dict[str, str] = {}

        # Cache for continuum memory context (keyed by task hash to prevent leaks)
        self._continuum_context_cache: dict[str, str] = {}

        # Cache for trending topics (TrendingTopic objects, not just formatted string)
        self._trending_topics_cache: list[Any] = []

    @staticmethod
    def get_task_hash(task: str) -> str:
        """Generate a cache key from task to prevent cache leaks between debates.

        Args:
            task: The debate task description.

        Returns:
            A 16-character hex hash of the task.
        """
        return hashlib.sha256(task.encode()).hexdigest()[:16]

    def get_evidence_pack(self, task: str) -> Any | None:
        """Get the cached evidence pack for a specific task.

        Args:
            task: The debate task description.

        Returns:
            Cached evidence pack or None if not found.
        """
        task_hash = self.get_task_hash(task)
        return self._research_evidence_pack.get(task_hash)

    def set_evidence_pack(self, task: str, pack: Any) -> None:
        """Cache an evidence pack for a specific task.

        Args:
            task: The debate task description.
            pack: The evidence pack to cache.
        """
        task_hash = self.get_task_hash(task)
        self._enforce_cache_limit(self._research_evidence_pack, self._max_evidence_size)
        self._research_evidence_pack[task_hash] = pack

    def get_latest_evidence_pack(self) -> Any | None:
        """Get the most recent cached evidence pack.

        For task-specific evidence, use get_evidence_pack(task) instead.

        Returns:
            Most recently cached evidence pack or None if cache is empty.
        """
        if not self._research_evidence_pack:
            return None
        # Return last added pack (dict preserves insertion order in Python 3.7+)
        return list(self._research_evidence_pack.values())[-1]

    def get_context(self, task: str) -> str | None:
        """Get cached research context for a specific task.

        Args:
            task: The debate task description.

        Returns:
            Cached context string or None if not found.
        """
        task_hash = self.get_task_hash(task)
        return self._research_context_cache.get(task_hash)

    def set_context(self, task: str, context: str) -> None:
        """Cache research context for a specific task.

        Args:
            task: The debate task description.
            context: The formatted context string.
        """
        task_hash = self.get_task_hash(task)
        self._enforce_cache_limit(self._research_context_cache, self._max_context_size)
        self._research_context_cache[task_hash] = context

    def get_continuum_context(self, task: str) -> str | None:
        """Get cached continuum memory context for a specific task.

        Args:
            task: The debate task description.

        Returns:
            Cached continuum context or None if not found.
        """
        task_hash = self.get_task_hash(task)
        return self._continuum_context_cache.get(task_hash)

    def set_continuum_context(self, task: str, context: str) -> None:
        """Cache continuum memory context for a specific task.

        Args:
            task: The debate task description.
            context: The formatted continuum context string.
        """
        task_hash = self.get_task_hash(task)
        self._enforce_cache_limit(self._continuum_context_cache, self._max_continuum_size)
        self._continuum_context_cache[task_hash] = context

    def get_trending_topics(self) -> list[Any]:
        """Get cached trending topics.

        Returns:
            List of TrendingTopic objects from the last gather_trending_context call.
        """
        return self._trending_topics_cache

    def set_trending_topics(self, topics: list[Any]) -> None:
        """Cache trending topics.

        Args:
            topics: List of TrendingTopic objects.
        """
        self._trending_topics_cache = list(topics)[: self._max_trending_size]

    def clear(self, task: str | None = None) -> None:
        """Clear cached context, optionally for a specific task.

        Args:
            task: If provided, only clear cache for this specific task.
                  If None, clear all cached context.
        """
        if task is None:
            self._research_context_cache.clear()
            self._research_evidence_pack.clear()
            self._continuum_context_cache.clear()
            self._trending_topics_cache = []
        else:
            task_hash = self.get_task_hash(task)
            self._research_context_cache.pop(task_hash, None)
            self._research_evidence_pack.pop(task_hash, None)
            self._continuum_context_cache.pop(task_hash, None)

    def merge_evidence_pack(self, task: str, new_pack: Any) -> Any | None:
        """Merge a new evidence pack with existing cached evidence.

        Avoids duplicates by checking snippet IDs.

        Args:
            task: The debate task description.
            new_pack: New evidence pack to merge.

        Returns:
            Merged evidence pack or None if no snippets.
        """
        task_hash = self.get_task_hash(task)
        existing_pack = self._research_evidence_pack.get(task_hash)

        if existing_pack:
            existing_ids = {s.id for s in existing_pack.snippets}
            new_snippets = [s for s in new_pack.snippets if s.id not in existing_ids]
            existing_pack.snippets.extend(new_snippets)
            existing_pack.total_searched += new_pack.total_searched
        else:
            self._enforce_cache_limit(self._research_evidence_pack, self._max_evidence_size)
            self._research_evidence_pack[task_hash] = new_pack

        return self._research_evidence_pack.get(task_hash)

    def _enforce_cache_limit(self, cache: dict, max_size: int) -> None:
        """Enforce maximum cache size using FIFO eviction.

        When the cache exceeds max_size, removes the oldest entries
        (first-inserted) to bring it back under the limit.

        Args:
            cache: The cache dict to enforce limits on
            max_size: Maximum number of entries allowed
        """
        while len(cache) >= max_size:
            # Remove oldest entry (first key in dict - Python 3.7+ maintains order)
            oldest_key = next(iter(cache))
            del cache[oldest_key]


# Re-export constants for backwards compatibility
__all__ = [
    "ContextCache",
    "MAX_EVIDENCE_CACHE_SIZE",
    "MAX_CONTEXT_CACHE_SIZE",
    "MAX_CONTINUUM_CACHE_SIZE",
    "MAX_TRENDING_CACHE_SIZE",
]
