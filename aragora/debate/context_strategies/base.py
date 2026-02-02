"""
Base class for context gathering strategies.

Each strategy is responsible for gathering a specific type of context
(e.g., web search, evidence, trending topics, knowledge mound).

Strategies are designed to:
- Be independently testable
- Have clear timeout handling
- Return None on failure (never raise in gather methods)
- Log appropriately for debugging
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class ContextStrategy(ABC):
    """Abstract base class for context gathering strategies."""

    # Strategy name for logging
    name: str = "base"

    # Default timeout in seconds
    default_timeout: float = 10.0

    @abstractmethod
    async def gather(self, task: str, **kwargs: Any) -> str | None:
        """
        Gather context for the given task.

        Args:
            task: The debate task/question
            **kwargs: Strategy-specific options

        Returns:
            Context string or None if gathering failed/unavailable
        """
        pass

    async def gather_with_timeout(
        self,
        task: str,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> str | None:
        """
        Gather context with timeout protection.

        Args:
            task: The debate task/question
            timeout: Timeout in seconds (uses default_timeout if None)
            **kwargs: Strategy-specific options

        Returns:
            Context string or None if timeout/failure
        """
        effective_timeout = timeout or self.default_timeout

        try:
            return await asyncio.wait_for(
                self.gather(task, **kwargs),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[context:%s] Timeout after %.1fs gathering context",
                self.name,
                effective_timeout,
            )
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[context:%s] Error gathering context: %s",
                self.name,
                e,
            )
            return None

    def is_available(self) -> bool:
        """Check if this strategy is available (dependencies installed, etc.)."""
        return True


class CachingStrategy(ContextStrategy):
    """
    Base class for strategies that support caching.

    Provides a simple in-memory cache with size limits.
    """

    max_cache_size: int = 100

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def _get_cache_key(self, task: str, **kwargs: Any) -> str:
        """Generate cache key from task and options."""
        import hashlib

        key_data = f"{task}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def get_cached(self, task: str, **kwargs: Any) -> str | None:
        """Get cached result if available."""
        key = self._get_cache_key(task, **kwargs)
        return self._cache.get(key)

    def set_cached(self, task: str, result: str, **kwargs: Any) -> None:
        """Cache a result, enforcing size limits."""
        key = self._get_cache_key(task, **kwargs)

        # Enforce cache size limit (FIFO eviction)
        while len(self._cache) >= self.max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result

    def clear_cache(self, task: str | None = None) -> None:
        """Clear cache, optionally for a specific task."""
        if task is None:
            self._cache.clear()
        else:
            key = self._get_cache_key(task)
            self._cache.pop(key, None)
