"""
LRU Cache Registry for periodic cleanup.

Provides a centralized registry for module-level @lru_cache decorated functions,
enabling periodic clearing to prevent memory accumulation across long-running
Arena instances.

Usage:
    from aragora.utils.cache_registry import register_lru_cache

    @register_lru_cache
    @lru_cache(maxsize=256)
    def expensive_computation(arg: str) -> str:
        ...

    # Later, to clear all registered caches:
    from aragora.utils.cache_registry import clear_all_lru_caches
    cleared = clear_all_lru_caches()
"""

import logging
import threading
from typing import Callable, Dict, List, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

# Thread-safe registry of LRU cache functions
_registered_caches: List[Callable] = []
_registry_lock = threading.Lock()


def register_lru_cache(func: F) -> F:
    """Register an LRU cache function for periodic cleanup.

    This decorator should be applied BEFORE @lru_cache to ensure the
    cache_info() and cache_clear() methods are accessible.

    Args:
        func: The LRU-cached function to register

    Returns:
        The function unchanged (for decorator chaining)

    Example:
        @register_lru_cache
        @lru_cache(maxsize=256)
        def compute_domain(task: str) -> str:
            ...
    """
    with _registry_lock:
        if func not in _registered_caches:
            _registered_caches.append(func)
            logger.debug(f"Registered LRU cache: {getattr(func, '__name__', repr(func))}")
    return func


def unregister_lru_cache(func: Callable) -> bool:
    """Unregister an LRU cache function.

    Args:
        func: The function to unregister

    Returns:
        True if the function was found and removed, False otherwise
    """
    with _registry_lock:
        if func in _registered_caches:
            _registered_caches.remove(func)
            return True
        return False


def clear_all_lru_caches() -> int:
    """Clear all registered LRU caches.

    Returns:
        Total number of entries cleared across all caches
    """
    total_cleared = 0
    with _registry_lock:
        for func in _registered_caches:
            if hasattr(func, "cache_clear") and hasattr(func, "cache_info"):
                try:
                    info = func.cache_info()
                    entries = info.currsize
                    func.cache_clear()
                    total_cleared += entries
                    logger.debug(
                        f"Cleared {entries} entries from {getattr(func, '__name__', 'unknown')}"
                    )
                except (TypeError, AttributeError) as e:
                    logger.warning(f"Failed to clear cache for {func}: {e}")

    if total_cleared > 0:
        logger.info(f"Cleared {total_cleared} total LRU cache entries")

    return total_cleared


def get_lru_cache_stats() -> Dict[str, dict]:
    """Get statistics for all registered LRU caches.

    Returns:
        Dictionary mapping function names to their cache stats (hits, misses,
        maxsize, currsize)
    """
    stats: Dict[str, dict] = {}
    with _registry_lock:
        for func in _registered_caches:
            name = getattr(func, "__name__", repr(func))
            if hasattr(func, "cache_info"):
                try:
                    info = func.cache_info()
                    stats[name] = {
                        "hits": info.hits,
                        "misses": info.misses,
                        "maxsize": info.maxsize,
                        "currsize": info.currsize,
                        "hit_rate": (
                            info.hits / (info.hits + info.misses)
                            if (info.hits + info.misses) > 0
                            else 0.0
                        ),
                    }
                except (TypeError, AttributeError):
                    stats[name] = {"error": "Unable to get cache info"}
            else:
                stats[name] = {"error": "No cache_info method"}

    return stats


def get_total_cache_entries() -> int:
    """Get total number of entries across all registered caches.

    Returns:
        Total entry count
    """
    total = 0
    with _registry_lock:
        for func in _registered_caches:
            if hasattr(func, "cache_info"):
                try:
                    total += func.cache_info().currsize
                except (TypeError, AttributeError):
                    pass
    return total


def get_registered_cache_count() -> int:
    """Get the number of registered LRU caches.

    Returns:
        Number of registered cache functions
    """
    with _registry_lock:
        return len(_registered_caches)


__all__ = [
    "register_lru_cache",
    "unregister_lru_cache",
    "clear_all_lru_caches",
    "get_lru_cache_stats",
    "get_total_cache_entries",
    "get_registered_cache_count",
]
