"""
Lazy Loading Framework for N+1 Query Prevention.

Provides decorators and utilities for lazy loading model relationships
while detecting and preventing N+1 query patterns.

Features:
- @lazy_property decorator for deferred loading
- Automatic N+1 detection with warnings
- Prefetch hints for bulk loading
- Integration with DataLoader for batching

Usage:
    class User:
        def __init__(self, id: str):
            self.id = id

        @lazy_property
        async def posts(self) -> List[Post]:
            return await db.posts.find({"user_id": self.id})

        @lazy_property(prefetch_key="posts")
        async def recent_posts(self) -> List[Post]:
            return await db.posts.find(
                {"user_id": self.id},
                limit=5,
                sort="-created_at"
            )

    # Prefetch for multiple users
    users = await get_users()
    await prefetch(users, "posts")  # Single batch query
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
import weakref
from collections import defaultdict
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
    Union,
    overload,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class LazyLoadStats:
    """Statistics for lazy loading operations."""

    total_loads: int = 0
    n_plus_one_detections: int = 0
    prefetch_hits: int = 0
    load_times_ms: List[float] = field(default_factory=list)

    @property
    def avg_load_time_ms(self) -> float:
        """Average load time in milliseconds."""
        if not self.load_times_ms:
            return 0.0
        return sum(self.load_times_ms) / len(self.load_times_ms)


# Global stats
_lazy_load_stats = LazyLoadStats()

# N+1 detection state (property_name -> (timestamp, count))
_n_plus_one_tracker: Dict[str, List[float]] = defaultdict(list)
_n_plus_one_lock = threading.Lock()
N_PLUS_ONE_THRESHOLD = 5  # Detections within window
N_PLUS_ONE_WINDOW_MS = 100  # Time window in milliseconds


def _detect_n_plus_one(property_name: str) -> bool:
    """
    Detect potential N+1 query pattern.

    Returns True if N+1 pattern detected.
    """
    now = time.perf_counter() * 1000  # Convert to ms

    with _n_plus_one_lock:
        # Clean old entries
        timestamps = _n_plus_one_tracker[property_name]
        cutoff = now - N_PLUS_ONE_WINDOW_MS
        _n_plus_one_tracker[property_name] = [t for t in timestamps if t > cutoff]

        # Add current
        _n_plus_one_tracker[property_name].append(now)

        # Check threshold
        if len(_n_plus_one_tracker[property_name]) >= N_PLUS_ONE_THRESHOLD:
            _lazy_load_stats.n_plus_one_detections += 1
            return True

    return False


class LazyValue(Generic[T]):
    """
    Container for a lazily loaded value.

    Caches the result after first load.
    """

    def __init__(
        self,
        loader: Callable[[], Awaitable[T]],
        property_name: str,
    ):
        """Initialize lazy value."""
        self._loader = loader
        self._property_name = property_name
        self._value: Optional[T] = None
        self._loaded = False
        self._loading = False
        self._load_future: Optional[asyncio.Future[T]] = None

    @property
    def is_loaded(self) -> bool:
        """Check if value has been loaded."""
        return self._loaded

    async def get(self) -> T:
        """Get the value, loading if necessary."""
        if self._loaded:
            _lazy_load_stats.prefetch_hits += 1
            return self._value  # type: ignore

        # Handle concurrent loads
        if self._loading:
            if self._load_future:
                return await self._load_future
            raise RuntimeError("Loader in invalid state")

        self._loading = True
        self._load_future = asyncio.get_event_loop().create_future()

        try:
            # Detect N+1
            if _detect_n_plus_one(self._property_name):
                logger.warning(
                    f"Potential N+1 query detected for '{self._property_name}'. "
                    f"Consider using prefetch() to batch load."
                )

            # Load value
            start = time.perf_counter()
            _lazy_load_stats.total_loads += 1
            self._value = await self._loader()
            _lazy_load_stats.load_times_ms.append((time.perf_counter() - start) * 1000)

            self._loaded = True
            self._load_future.set_result(self._value)
            return self._value

        except Exception as e:
            if self._load_future and not self._load_future.done():
                self._load_future.set_exception(e)
            raise
        finally:
            self._loading = False

    def set(self, value: T) -> None:
        """Set the value directly (for prefetching)."""
        self._value = value
        self._loaded = True


class LazyDescriptor(Generic[T]):
    """
    Descriptor for lazy property access.

    Manages LazyValue instances per object.
    """

    def __init__(
        self,
        func: Callable[[Any], Awaitable[T]],
        prefetch_key: Optional[str] = None,
    ):
        """Initialize descriptor."""
        self._func = func
        self._name = func.__name__
        self._prefetch_key = prefetch_key or self._name
        self._cache: weakref.WeakKeyDictionary[Any, LazyValue[T]] = weakref.WeakKeyDictionary()
        functools.update_wrapper(self, func)  # type: ignore[arg-type]

    def __get__(self, obj: Any, objtype: Any = None) -> Union[LazyValue[T], "LazyDescriptor[T]"]:
        """Get LazyValue for the object."""
        if obj is None:
            return self

        if obj not in self._cache:
            self._cache[obj] = LazyValue(
                loader=lambda: self._func(obj),
                property_name=f"{objtype.__name__}.{self._name}" if objtype else self._name,
            )

        return self._cache[obj]

    def __set__(self, obj: Any, value: T) -> None:
        """Set value directly."""
        if obj not in self._cache:

            async def _immediate_loader() -> T:
                return value

            self._cache[obj] = LazyValue(
                loader=_immediate_loader,
                property_name=self._name,
            )
        self._cache[obj].set(value)

    @property
    def prefetch_key(self) -> str:
        """Get the prefetch key for this property."""
        return self._prefetch_key


@overload
def lazy_property(func: Callable[[Any], Awaitable[T]]) -> LazyDescriptor[T]: ...


@overload
def lazy_property(
    *,
    prefetch_key: Optional[str] = None,
) -> Callable[[Callable[[Any], Awaitable[T]]], LazyDescriptor[T]]: ...


def lazy_property(
    func: Optional[Callable[[Any], Awaitable[T]]] = None,
    *,
    prefetch_key: Optional[str] = None,
) -> Union[LazyDescriptor[T], Callable[[Callable[[Any], Awaitable[T]]], LazyDescriptor[T]]]:
    """
    Decorator for lazy loading properties.

    Can be used with or without arguments:

        @lazy_property
        async def posts(self) -> List[Post]:
            ...

        @lazy_property(prefetch_key="user_posts")
        async def posts(self) -> List[Post]:
            ...

    Args:
        func: The async method to decorate
        prefetch_key: Key for grouping prefetch operations

    Returns:
        LazyDescriptor that manages lazy loading
    """
    if func is not None:
        return LazyDescriptor(func, prefetch_key=prefetch_key)

    def decorator(f: Callable[[Any], Awaitable[T]]) -> LazyDescriptor[T]:
        return LazyDescriptor(f, prefetch_key=prefetch_key)

    return decorator


class LazyLoader:
    """
    Utility class for bulk lazy loading operations.

    Coordinates prefetching across multiple objects.
    """

    def __init__(self):
        """Initialize LazyLoader."""
        self._prefetch_fns: Dict[str, Callable[[List[Any]], Awaitable[Dict[Any, Any]]]] = {}

    def register_prefetch(
        self,
        key: str,
        fn: Callable[[List[Any]], Awaitable[Dict[Any, Any]]],
    ) -> None:
        """
        Register a prefetch function for a property.

        Args:
            key: The prefetch key (matches lazy_property prefetch_key)
            fn: Async function that loads values for multiple objects
                Returns dict mapping object -> value
        """
        self._prefetch_fns[key] = fn

    async def prefetch(
        self,
        objects: List[Any],
        *property_names: str,
    ) -> None:
        """
        Prefetch properties for multiple objects.

        Args:
            objects: List of objects to prefetch for
            *property_names: Property names to prefetch
        """
        if not objects:
            return

        for prop_name in property_names:
            # Get the descriptor
            obj_type = type(objects[0])
            descriptor = getattr(obj_type, prop_name, None)

            if not isinstance(descriptor, LazyDescriptor):
                logger.warning(
                    f"Property '{prop_name}' is not a lazy_property on {obj_type.__name__}"
                )
                continue

            prefetch_key = descriptor.prefetch_key

            # Check for registered prefetch function
            if prefetch_key in self._prefetch_fns:
                prefetch_fn = self._prefetch_fns[prefetch_key]
                results = await prefetch_fn(objects)

                for obj in objects:
                    if obj in results:
                        lazy_value = descriptor.__get__(obj, obj_type)
                        if isinstance(lazy_value, LazyValue):
                            lazy_value.set(results[obj])
            else:
                # Fall back to individual loads (still benefits from N+1 warning)
                logger.debug(f"No prefetch function for '{prefetch_key}', " f"loading individually")
                lazy_values = [
                    lv
                    for obj in objects
                    if isinstance(lv := descriptor.__get__(obj, obj_type), LazyValue)
                ]
                await asyncio.gather(*[lv.get() for lv in lazy_values])


# Global lazy loader instance
_lazy_loader = LazyLoader()


async def prefetch(objects: List[Any], *property_names: str) -> None:
    """
    Prefetch lazy properties for multiple objects.

    Convenience function using the global LazyLoader.

    Args:
        objects: List of objects
        *property_names: Property names to prefetch

    Example:
        users = await get_users()
        await prefetch(users, "posts", "followers")
        # Now accessing user.posts won't trigger individual queries
    """
    await _lazy_loader.prefetch(objects, *property_names)


def register_prefetch(
    key: str,
    fn: Callable[[List[Any]], Awaitable[Dict[Any, Any]]],
) -> None:
    """
    Register a prefetch function.

    Args:
        key: Prefetch key matching lazy_property prefetch_key
        fn: Batch loading function
    """
    _lazy_loader.register_prefetch(key, fn)


def get_lazy_load_stats() -> Dict[str, Any]:
    """Get lazy loading statistics."""
    return {
        "total_loads": _lazy_load_stats.total_loads,
        "n_plus_one_detections": _lazy_load_stats.n_plus_one_detections,
        "prefetch_hits": _lazy_load_stats.prefetch_hits,
        "avg_load_time_ms": round(_lazy_load_stats.avg_load_time_ms, 2),
    }


def reset_lazy_load_stats() -> None:
    """Reset lazy loading statistics."""
    global _lazy_load_stats
    _lazy_load_stats = LazyLoadStats()


__all__ = [
    "lazy_property",
    "LazyValue",
    "LazyDescriptor",
    "LazyLoader",
    "prefetch",
    "register_prefetch",
    "get_lazy_load_stats",
    "reset_lazy_load_stats",
]
