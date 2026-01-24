"""
DataLoader: Batch Query Resolution with Deduplication.

Implements Facebook's DataLoader pattern for batching and caching
database queries within a single request/event loop tick.

Features:
- Automatic batching of individual queries
- Request-level caching (deduplication)
- Configurable batch size limits
- Async-first design
- Prometheus metrics integration

Usage:
    # Create a data loader for users
    async def batch_load_users(ids: List[str]) -> List[User]:
        users = await db.users.find({"_id": {"$in": ids}})
        return [users.get(id) for id in ids]

    user_loader = DataLoader(batch_load_users)

    # Individual loads are automatically batched
    user1 = await user_loader.load("user_1")
    user2 = await user_loader.load("user_2")  # Batched with user_1
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
)
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)

K = TypeVar("K")  # Key type
V = TypeVar("V")  # Value type


@dataclass
class LoaderStats:
    """Statistics for a DataLoader instance."""

    loads: int = 0
    batches: int = 0
    cache_hits: int = 0
    batch_sizes: List[int] = field(default_factory=list)
    total_load_time_ms: float = 0.0

    @property
    def avg_batch_size(self) -> float:
        """Average batch size."""
        if not self.batch_sizes:
            return 0.0
        return sum(self.batch_sizes) / len(self.batch_sizes)

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        if self.loads == 0:
            return 0.0
        return (self.cache_hits / self.loads) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "loads": self.loads,
            "batches": self.batches,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{self.cache_hit_rate:.1f}%",
            "avg_batch_size": round(self.avg_batch_size, 1),
            "total_load_time_ms": round(self.total_load_time_ms, 2),
        }


@dataclass
class BatchRequest(Generic[K]):
    """A pending batch request."""

    key: K
    future: asyncio.Future[Any]


class DataLoader(Generic[K, V]):
    """
    DataLoader batches and caches individual loads.

    Collects individual load calls within the same event loop tick
    and batches them into a single batch function call.
    """

    def __init__(
        self,
        batch_fn: Callable[[List[K]], Awaitable[List[V]]],
        *,
        max_batch_size: int = 100,
        cache: bool = True,
        cache_key_fn: Optional[Callable[[K], str]] = None,
        name: str = "DataLoader",
    ):
        """
        Initialize DataLoader.

        Args:
            batch_fn: Async function that loads multiple items by keys
            max_batch_size: Maximum items per batch (default 100)
            cache: Whether to cache results (default True)
            cache_key_fn: Function to convert key to cache key string
            name: Name for logging and metrics
        """
        self._batch_fn = batch_fn
        self._max_batch_size = max_batch_size
        self._cache_enabled = cache
        self._cache_key_fn = cache_key_fn or str
        self._name = name

        # Pending batch requests
        self._queue: List[BatchRequest[K]] = []
        self._batch_scheduled = False

        # Request-level cache
        self._cache: Dict[str, V] = {}

        # Statistics
        self._stats = LoaderStats()

    @property
    def stats(self) -> LoaderStats:
        """Get loader statistics."""
        return self._stats

    async def load(self, key: K) -> V:
        """
        Load a single item by key.

        Multiple calls within the same event loop tick are batched.

        Args:
            key: The key to load

        Returns:
            The loaded value
        """
        self._stats.loads += 1

        # Check cache first
        if self._cache_enabled:
            cache_key = self._cache_key_fn(key)
            if cache_key in self._cache:
                self._stats.cache_hits += 1
                return self._cache[cache_key]

        # Create future for this request
        loop = asyncio.get_event_loop()
        future: asyncio.Future[V] = loop.create_future()

        # Add to queue
        self._queue.append(BatchRequest(key=key, future=future))

        # Schedule batch execution if not already scheduled
        if not self._batch_scheduled:
            self._batch_scheduled = True
            loop.call_soon(lambda: asyncio.create_task(self._dispatch_batch()))

        return await future

    async def load_many(self, keys: List[K]) -> List[V]:
        """
        Load multiple items by keys.

        Args:
            keys: List of keys to load

        Returns:
            List of loaded values in same order as keys
        """
        return await asyncio.gather(*[self.load(key) for key in keys])

    async def _dispatch_batch(self) -> None:
        """Dispatch queued requests as a batch."""
        self._batch_scheduled = False

        if not self._queue:
            return

        # Take all pending requests
        batch = self._queue[: self._max_batch_size]
        self._queue = self._queue[self._max_batch_size :]

        # If more items remain, schedule another batch
        if self._queue:
            loop = asyncio.get_event_loop()
            self._batch_scheduled = True
            loop.call_soon(lambda: asyncio.create_task(self._dispatch_batch()))

        # Execute batch
        keys = [req.key for req in batch]
        self._stats.batches += 1
        self._stats.batch_sizes.append(len(keys))

        start = time.perf_counter()
        try:
            values = await self._batch_fn(keys)
            self._stats.total_load_time_ms += (time.perf_counter() - start) * 1000

            # Validate response length
            if len(values) != len(keys):
                error = ValueError(
                    f"Batch function returned {len(values)} values for {len(keys)} keys"
                )
                for req in batch:
                    if not req.future.done():
                        req.future.set_exception(error)
                return

            # Resolve futures and update cache
            for req, value in zip(batch, values):
                if self._cache_enabled:
                    cache_key = self._cache_key_fn(req.key)
                    self._cache[cache_key] = value

                if not req.future.done():
                    req.future.set_result(value)

        except Exception as e:
            logger.exception(f"DataLoader {self._name} batch failed")
            self._stats.total_load_time_ms += (time.perf_counter() - start) * 1000
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)

    def prime(self, key: K, value: V) -> None:
        """
        Prime the cache with a known value.

        Args:
            key: The key
            value: The value to cache
        """
        if self._cache_enabled:
            cache_key = self._cache_key_fn(key)
            self._cache[cache_key] = value

    def clear(self, key: Optional[K] = None) -> None:
        """
        Clear the cache.

        Args:
            key: Specific key to clear, or None to clear all
        """
        if key is not None:
            cache_key = self._cache_key_fn(key)
            self._cache.pop(cache_key, None)
        else:
            self._cache.clear()

    def clear_all(self) -> None:
        """Clear entire cache and reset stats."""
        self._cache.clear()
        self._stats = LoaderStats()


class BatchResolver:
    """
    Manages multiple DataLoaders for different entity types.

    Provides a centralized way to create and access data loaders
    within a request context.
    """

    def __init__(self):
        """Initialize BatchResolver."""
        self._loaders: Dict[str, DataLoader[Any, Any]] = {}

    def register(
        self,
        name: str,
        batch_fn: Callable[[List[Any]], Awaitable[List[Any]]],
        **kwargs: Any,
    ) -> DataLoader[Any, Any]:
        """
        Register a new DataLoader.

        Args:
            name: Unique name for the loader
            batch_fn: Batch loading function
            **kwargs: Additional DataLoader options

        Returns:
            The created DataLoader
        """
        if name in self._loaders:
            return self._loaders[name]

        loader = DataLoader(batch_fn, name=name, **kwargs)
        self._loaders[name] = loader
        return loader

    def get(self, name: str) -> Optional[DataLoader[Any, Any]]:
        """Get a registered DataLoader by name."""
        return self._loaders.get(name)

    def clear_all(self) -> None:
        """Clear all loader caches."""
        for loader in self._loaders.values():
            loader.clear_all()

    def stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all loaders."""
        return {name: loader.stats.to_dict() for name, loader in self._loaders.items()}


# Request-scoped loader storage
_request_loaders: WeakValueDictionary[int, BatchResolver] = WeakValueDictionary()


def create_data_loader(
    batch_fn: Callable[[List[K]], Awaitable[List[V]]],
    **kwargs: Any,
) -> DataLoader[K, V]:
    """
    Create a DataLoader with the given batch function.

    Convenience function for creating standalone loaders.

    Args:
        batch_fn: Async function that loads items by keys
        **kwargs: Additional DataLoader options

    Returns:
        A new DataLoader instance
    """
    return DataLoader(batch_fn, **kwargs)


# Pre-built batch functions for common patterns


async def batch_by_id(
    ids: List[str],
    fetch_fn: Callable[[List[str]], Awaitable[Dict[str, Any]]],
) -> List[Optional[Any]]:
    """
    Standard batch function that fetches by ID and returns in order.

    Args:
        ids: List of IDs to fetch
        fetch_fn: Function that returns dict mapping id -> value

    Returns:
        List of values in same order as ids (None for missing)
    """
    results = await fetch_fn(ids)
    return [results.get(id) for id in ids]


__all__ = [
    "DataLoader",
    "LoaderStats",
    "BatchResolver",
    "create_data_loader",
    "batch_by_id",
]
