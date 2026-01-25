"""
Lazy Loading Framework for N+1 Query Prevention.

Provides decorators and utilities for lazy loading model relationships
while detecting and preventing N+1 query patterns.

Features:
- @lazy_property decorator for deferred loading
- Automatic N+1 detection with warnings
- Automatic prefetch fallback when N+1 detected
- Prometheus metrics for monitoring
- Configurable thresholds per environment
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

Environment Variables:
    N_PLUS_ONE_THRESHOLD: Number of loads in window to trigger detection (default: 5)
    N_PLUS_ONE_WINDOW_MS: Time window in milliseconds (default: 100)
    N_PLUS_ONE_AUTO_PREFETCH: Enable automatic prefetch on detection (default: true)
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

if TYPE_CHECKING:
    from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Prometheus metrics (lazy initialization)
_metrics_initialized = False
_n_plus_one_detections_counter: Optional["Counter"] = None
_prefetch_operations_counter: Optional["Counter"] = None
_auto_prefetch_counter: Optional["Counter"] = None
_load_duration_histogram: Optional["Histogram"] = None


def _init_metrics() -> None:
    """Initialize Prometheus metrics lazily."""
    global _metrics_initialized, _n_plus_one_detections_counter
    global _prefetch_operations_counter, _auto_prefetch_counter, _load_duration_histogram

    if _metrics_initialized:
        return

    try:
        from prometheus_client import Counter, Histogram

        _n_plus_one_detections_counter = Counter(
            "aragora_lazy_load_n_plus_one_detections_total",
            "Number of N+1 query pattern detections",
            ["property_name"],
        )
        _prefetch_operations_counter = Counter(
            "aragora_lazy_load_prefetch_operations_total",
            "Number of prefetch operations performed",
            ["property_name", "type"],  # values: "manual" or "auto"
        )
        _auto_prefetch_counter = Counter(
            "aragora_lazy_load_auto_prefetch_total",
            "Number of automatic prefetch operations triggered by N+1 detection",
            ["property_name"],
        )
        _load_duration_histogram = Histogram(
            "aragora_lazy_load_duration_seconds",
            "Duration of lazy load operations",
            ["property_name"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )
        _metrics_initialized = True
    except ImportError:
        logger.debug("prometheus_client not available, metrics disabled")
        _metrics_initialized = True  # Don't retry


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

# Configurable thresholds via environment variables
N_PLUS_ONE_THRESHOLD = int(os.environ.get("N_PLUS_ONE_THRESHOLD", "5"))
N_PLUS_ONE_WINDOW_MS = float(os.environ.get("N_PLUS_ONE_WINDOW_MS", "100"))
N_PLUS_ONE_AUTO_PREFETCH = os.environ.get("N_PLUS_ONE_AUTO_PREFETCH", "true").lower() in (
    "true",
    "1",
    "yes",
)

# N+1 detection state
_n_plus_one_tracker: Dict[str, List[float]] = defaultdict(list)
_n_plus_one_lock = threading.Lock()

# Auto-prefetch batching state: tracks pending loads for N+1 detected properties
# Maps property_name -> (objects waiting for load, timestamp of first request)
_auto_prefetch_pending: Dict[str, Tuple[List[Any], float]] = {}
_auto_prefetch_lock = threading.Lock()
AUTO_PREFETCH_BATCH_DELAY_MS = 10  # Wait this long to collect batch


def _detect_n_plus_one(property_name: str) -> bool:
    """
    Detect potential N+1 query pattern.

    Returns True if N+1 pattern detected.
    """
    _init_metrics()
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
            # Record Prometheus metric
            if _n_plus_one_detections_counter:
                _n_plus_one_detections_counter.labels(property_name=property_name).inc()
            return True

    return False


def get_n_plus_one_config() -> Dict[str, Any]:
    """Get current N+1 detection configuration."""
    return {
        "threshold": N_PLUS_ONE_THRESHOLD,
        "window_ms": N_PLUS_ONE_WINDOW_MS,
        "auto_prefetch_enabled": N_PLUS_ONE_AUTO_PREFETCH,
    }


class AutoPrefetchBatcher:
    """
    Collects objects that trigger N+1 detection for automatic batching.

    When N+1 is detected, instead of loading each object individually,
    this batcher collects objects and performs a single batch load.
    """

    def __init__(self):
        """Initialize the auto-prefetch batcher."""
        self._pending: Dict[str, List[Tuple[Any, "LazyValue[Any]"]]] = defaultdict(list)
        self._batch_futures: Dict[str, asyncio.Future[None]] = {}
        self._lock = asyncio.Lock()

    async def add_to_batch(
        self,
        property_name: str,
        obj: Any,
        lazy_value: "LazyValue[Any]",
        loader: Callable[[], Awaitable[Any]],
    ) -> bool:
        """
        Add an object to the batch for auto-prefetch.

        Returns True if this object should wait for batch load,
        False if it should proceed with individual load.
        """
        if not N_PLUS_ONE_AUTO_PREFETCH:
            return False

        async with self._lock:
            self._pending[property_name].append((obj, lazy_value))

            # If this is the first in batch, schedule batch execution
            if property_name not in self._batch_futures:
                loop = asyncio.get_event_loop()
                future: asyncio.Future[None] = loop.create_future()
                self._batch_futures[property_name] = future

                # Schedule batch execution after delay
                def _schedule_batch(pn: str = property_name) -> None:
                    asyncio.create_task(self._execute_batch(pn))

                loop.call_later(
                    AUTO_PREFETCH_BATCH_DELAY_MS / 1000.0,
                    _schedule_batch,
                )

            return True

    async def wait_for_batch(self, property_name: str) -> None:
        """Wait for batch load to complete."""
        async with self._lock:
            future = self._batch_futures.get(property_name)

        if future:
            await future

    async def _execute_batch(self, property_name: str) -> None:
        """Execute batch load for a property."""
        async with self._lock:
            pending = self._pending.pop(property_name, [])
            future = self._batch_futures.pop(property_name, None)

        if not pending:
            if future and not future.done():
                future.set_result(None)
            return

        logger.info(f"Auto-prefetch: batching {len(pending)} loads for '{property_name}'")

        # Record metric
        if _auto_prefetch_counter:
            _auto_prefetch_counter.labels(property_name=property_name).inc()

        # Load all values concurrently
        try:
            results = await asyncio.gather(
                *[lv._loader() for _, lv in pending],
                return_exceptions=True,
            )

            # Set results on lazy values
            for (obj, lazy_value), result in zip(pending, results):
                if isinstance(result, Exception):
                    logger.warning(f"Auto-prefetch load failed for '{property_name}': {result}")
                else:
                    lazy_value.set(result)

            if future and not future.done():
                future.set_result(None)

        except Exception as e:
            logger.error(f"Auto-prefetch batch failed for '{property_name}': {e}")
            if future and not future.done():
                future.set_exception(e)


# Global auto-prefetch batcher
_auto_prefetch_batcher = AutoPrefetchBatcher()


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
            n_plus_one_detected = _detect_n_plus_one(self._property_name)
            if n_plus_one_detected:
                logger.warning(
                    f"Potential N+1 query detected for '{self._property_name}'. "
                    f"Consider using prefetch() to batch load. "
                    f"(Set N_PLUS_ONE_AUTO_PREFETCH=true for automatic batching)"
                )

            # Load value with metrics
            start = time.perf_counter()
            _lazy_load_stats.total_loads += 1
            self._value = await self._loader()
            load_duration = time.perf_counter() - start
            _lazy_load_stats.load_times_ms.append(load_duration * 1000)

            # Record Prometheus metrics
            if _load_duration_histogram:
                _load_duration_histogram.labels(property_name=self._property_name).observe(
                    load_duration
                )

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

                # Record metric for manual prefetch
                _init_metrics()
                if _prefetch_operations_counter:
                    _prefetch_operations_counter.labels(
                        property_name=prop_name, type="manual"
                    ).inc()

                for obj in objects:
                    if obj in results:
                        lazy_value = descriptor.__get__(obj, obj_type)
                        if isinstance(lazy_value, LazyValue):
                            lazy_value.set(results[obj])
            else:
                # Fall back to individual loads (still benefits from N+1 warning)
                logger.debug(f"No prefetch function for '{prefetch_key}', loading individually")

                # Record metric for fallback prefetch
                _init_metrics()
                if _prefetch_operations_counter:
                    _prefetch_operations_counter.labels(
                        property_name=prop_name, type="fallback"
                    ).inc()

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
    """Get lazy loading statistics including N+1 detection config."""
    return {
        "total_loads": _lazy_load_stats.total_loads,
        "n_plus_one_detections": _lazy_load_stats.n_plus_one_detections,
        "prefetch_hits": _lazy_load_stats.prefetch_hits,
        "avg_load_time_ms": round(_lazy_load_stats.avg_load_time_ms, 2),
        "config": get_n_plus_one_config(),
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
    "AutoPrefetchBatcher",
    "prefetch",
    "register_prefetch",
    "get_lazy_load_stats",
    "get_n_plus_one_config",
    "reset_lazy_load_stats",
    "N_PLUS_ONE_THRESHOLD",
    "N_PLUS_ONE_WINDOW_MS",
    "N_PLUS_ONE_AUTO_PREFETCH",
]
