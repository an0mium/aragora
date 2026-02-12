"""
Analytics Dashboard Cache Module.

Provides workspace-scoped TTL caching for expensive analytics dashboard queries.
Implements cache invalidation hooks for data freshness.

Cached Endpoints:
- /api/analytics/summary (overview) - 60s TTL (fast refresh for dashboard)
- /api/analytics/trends/findings (summary) - 300s TTL
- /api/analytics/agents (performance) - 300s TTL
- /api/analytics/remediation (memory usage proxy) - 300s TTL
- /api/analytics/cost (costs breakdown) - 300s TTL

Usage:
    from aragora.server.handlers.analytics.cache import (
        AnalyticsDashboardCache,
        get_analytics_dashboard_cache,
        invalidate_analytics_cache,
    )

    # Using the decorator
    @cached_analytics("summary", workspace_key="workspace_id")
    def _get_summary(self, query_params, handler, user):
        ...

    # Manual cache access
    cache = get_analytics_dashboard_cache()
    cache.get_summary("workspace-123", "30d")
    cache.invalidate_workspace("workspace-123")
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, TypeVar

from aragora.cache import TTLCache, make_cache_key
from aragora.config import (
    CACHE_TTL_ANALYTICS_AGENTS,
    CACHE_TTL_ANALYTICS_COSTS,
    CACHE_TTL_ANALYTICS_MEMORY,
    CACHE_TTL_ANALYTICS_OVERVIEW,
    CACHE_TTL_ANALYTICS_SUMMARY,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CacheConfig:
    """Configuration for a cache endpoint."""

    ttl_seconds: float
    key_prefix: str
    maxsize: int = 200


# Cache configurations per endpoint type
CACHE_CONFIGS: dict[str, CacheConfig] = {
    "summary": CacheConfig(
        ttl_seconds=CACHE_TTL_ANALYTICS_OVERVIEW,
        key_prefix="analytics_dashboard_summary",
        maxsize=200,
    ),
    "trends": CacheConfig(
        ttl_seconds=CACHE_TTL_ANALYTICS_SUMMARY,
        key_prefix="analytics_dashboard_trends",
        maxsize=300,
    ),
    "agents": CacheConfig(
        ttl_seconds=CACHE_TTL_ANALYTICS_AGENTS,
        key_prefix="analytics_dashboard_agents",
        maxsize=200,
    ),
    "remediation": CacheConfig(
        ttl_seconds=CACHE_TTL_ANALYTICS_MEMORY,
        key_prefix="analytics_dashboard_remediation",
        maxsize=150,
    ),
    "cost": CacheConfig(
        ttl_seconds=CACHE_TTL_ANALYTICS_COSTS,
        key_prefix="analytics_dashboard_cost",
        maxsize=200,
    ),
    # Token usage endpoints
    "tokens": CacheConfig(
        ttl_seconds=CACHE_TTL_ANALYTICS_SUMMARY,
        key_prefix="analytics_dashboard_tokens",
        maxsize=150,
    ),
    # Deliberation analytics
    "deliberations": CacheConfig(
        ttl_seconds=CACHE_TTL_ANALYTICS_SUMMARY,
        key_prefix="analytics_dashboard_deliberations",
        maxsize=200,
    ),
}


class AnalyticsDashboardCache:
    """
    Workspace-scoped cache for analytics dashboard queries.

    Provides separate caches for different endpoint types with appropriate TTLs,
    and supports bulk invalidation by workspace.
    """

    _instance: AnalyticsDashboardCache | None = None
    _lock = threading.Lock()

    def __init__(self):
        self._caches: dict[str, TTLCache[Any]] = {}
        self._init_lock = threading.Lock()
        self._initialized = False

    @classmethod
    def get_instance(cls) -> AnalyticsDashboardCache:
        """Get the singleton cache instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_initialized(self) -> None:
        """Lazily initialize caches."""
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            for name, config in CACHE_CONFIGS.items():
                self._caches[name] = TTLCache(
                    maxsize=config.maxsize,
                    ttl_seconds=config.ttl_seconds,
                )
            self._initialized = True
            logger.debug(f"AnalyticsDashboardCache initialized with {len(self._caches)} caches")

    def _get_cache(self, cache_type: str) -> TTLCache[Any]:
        """Get or create a cache by type."""
        self._ensure_initialized()
        if cache_type not in self._caches:
            # Default cache for unknown types
            self._caches[cache_type] = TTLCache(
                maxsize=100,
                ttl_seconds=CACHE_TTL_ANALYTICS_SUMMARY,
            )
        return self._caches[cache_type]

    def _make_key(self, cache_type: str, workspace_id: str, *args: Any) -> str:
        """Generate a cache key including workspace scope."""
        config = CACHE_CONFIGS.get(cache_type)
        prefix = config.key_prefix if config else f"analytics_{cache_type}"
        test_id = _get_pytest_cache_tag()
        key_args = [str(a) for a in args]
        if test_id:
            key_args.append(test_id)
        return make_cache_key(prefix, workspace_id, *key_args)

    def get(self, cache_type: str, workspace_id: str, *args: Any) -> Any | None:
        """Get a cached value."""
        cache = self._get_cache(cache_type)
        key = self._make_key(cache_type, workspace_id, *args)
        return cache.get(key)

    def set(self, cache_type: str, workspace_id: str, value: Any, *args: Any) -> None:
        """Set a cached value."""
        cache = self._get_cache(cache_type)
        key = self._make_key(cache_type, workspace_id, *args)
        cache.set(key, value)
        logger.debug(f"Cached {cache_type} for workspace {workspace_id}")

    def invalidate(self, cache_type: str, workspace_id: str, *args: Any) -> bool:
        """Invalidate a specific cache entry."""
        cache = self._get_cache(cache_type)
        key = self._make_key(cache_type, workspace_id, *args)
        return cache.invalidate(key)

    def invalidate_workspace(self, workspace_id: str) -> int:
        """Invalidate all cache entries for a workspace."""
        self._ensure_initialized()
        total_cleared = 0
        for cache_type, cache in self._caches.items():
            config = CACHE_CONFIGS.get(cache_type)
            prefix = config.key_prefix if config else f"analytics_{cache_type}"
            workspace_prefix = f"{prefix}:{workspace_id}"
            cleared = cache.clear_prefix(workspace_prefix)
            total_cleared += cleared
        if total_cleared > 0:
            logger.info(f"Invalidated {total_cleared} cache entries for workspace {workspace_id}")
        return total_cleared

    def invalidate_all(self) -> int:
        """Invalidate all analytics dashboard caches."""
        self._ensure_initialized()
        total_cleared = 0
        for cache in self._caches.values():
            total_cleared += cache.clear()
        if total_cleared > 0:
            logger.info(f"Invalidated all analytics dashboard caches: {total_cleared} entries")
        return total_cleared

    def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all caches."""
        self._ensure_initialized()
        return {name: cache.stats for name, cache in self._caches.items()}

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics across all caches."""
        self._ensure_initialized()
        total_size = 0
        total_hits = 0
        total_misses = 0
        for cache in self._caches.values():
            stats = cache.stats
            total_size += stats.get("size", 0)
            total_hits += stats.get("hits", 0)
            total_misses += stats.get("misses", 0)
        total_requests = total_hits + total_misses
        return {
            "cache_count": len(self._caches),
            "total_size": total_size,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "overall_hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
        }


# Global accessor
def get_analytics_dashboard_cache() -> AnalyticsDashboardCache:
    """Get the global analytics dashboard cache instance."""
    return AnalyticsDashboardCache.get_instance()


def invalidate_analytics_cache(workspace_id: str | None = None) -> int:
    """
    Invalidate analytics dashboard cache.

    Args:
        workspace_id: If provided, only invalidate for this workspace.
                     If None, invalidate all.

    Returns:
        Number of entries cleared.
    """
    cache = get_analytics_dashboard_cache()
    if workspace_id:
        return cache.invalidate_workspace(workspace_id)
    return cache.invalidate_all()


def cached_analytics(
    cache_type: str,
    workspace_key: str = "workspace_id",
    time_range_key: str = "time_range",
    extra_keys: list[str] | None = None,
):
    """
    Decorator for caching analytics dashboard endpoint results.

    The decorator extracts workspace_id and time_range from query_params
    to build a cache key.

    Args:
        cache_type: Type of cache to use (summary, trends, agents, etc.)
        workspace_key: Query param key for workspace ID
        time_range_key: Query param key for time range
        extra_keys: Additional query param keys to include in cache key

    Example:
        @cached_analytics("summary", workspace_key="workspace_id")
        def _get_summary(self, query_params, handler, user):
            ...
    """
    extra_keys = extra_keys or []

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, query_params: dict, *args, **kwargs) -> T:
            # Extract workspace ID
            workspace_id = query_params.get(workspace_key)
            if not workspace_id:
                # No caching without workspace scope
                return func(self, query_params, *args, **kwargs)

            # Build cache key components
            time_range = query_params.get(time_range_key, "30d")
            extra_values = [query_params.get(k, "") for k in extra_keys]

            cache = get_analytics_dashboard_cache()

            # Try cache
            cached_result = cache.get(cache_type, workspace_id, time_range, *extra_values)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_type} workspace={workspace_id}")
                return cached_result

            # Cache miss - compute result
            result = func(self, query_params, *args, **kwargs)

            # Only cache successful responses
            if result is not None and hasattr(result, "status_code"):
                if result.status_code == 200:
                    cache.set(cache_type, workspace_id, result, time_range, *extra_values)

            return result

        return wrapper

    return decorator


def cached_analytics_org(
    cache_type: str,
    org_key: str = "org_id",
    days_key: str = "days",
    extra_keys: list[str] | None = None,
):
    """
    Decorator for caching analytics endpoints scoped by org_id.

    Similar to cached_analytics but uses org_id instead of workspace_id.

    Args:
        cache_type: Type of cache to use
        org_key: Query param key for organization ID
        days_key: Query param key for days lookback
        extra_keys: Additional query param keys to include in cache key
    """
    extra_keys = extra_keys or []

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(self, query_params: dict, *args, **kwargs) -> T:
            org_id = query_params.get(org_key)
            if not org_id:
                return func(self, query_params, *args, **kwargs)

            days = query_params.get(days_key, "30")
            extra_values = [query_params.get(k, "") for k in extra_keys]

            cache = get_analytics_dashboard_cache()

            cached_result = cache.get(cache_type, org_id, days, *extra_values)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_type} org={org_id}")
                return cached_result

            result = func(self, query_params, *args, **kwargs)

            if result is not None and hasattr(result, "status_code"):
                if result.status_code == 200:
                    cache.set(cache_type, org_id, result, days, *extra_values)

            return result

        return wrapper

    return decorator


def _get_pytest_cache_tag() -> str | None:
    """Return a stable per-test cache tag when running under pytest."""
    current = os.environ.get("PYTEST_CURRENT_TEST")
    if not current:
        return None
    return current.split(" (", 1)[0]


# Cache invalidation hooks - to be called from data mutation handlers
def on_debate_completed(workspace_id: str) -> None:
    """Invalidate relevant caches when a debate completes."""
    cache = get_analytics_dashboard_cache()
    # Invalidate summary and trends caches
    for cache_type in ["summary", "trends", "deliberations"]:
        cache._get_cache(cache_type).clear_prefix(
            f"{CACHE_CONFIGS[cache_type].key_prefix}:{workspace_id}"
        )
    logger.debug(f"Invalidated analytics caches for completed debate in {workspace_id}")


def on_agent_performance_update(workspace_id: str) -> None:
    """Invalidate agent performance cache when ELO/rankings update."""
    cache = get_analytics_dashboard_cache()
    cache._get_cache("agents").clear_prefix(f"{CACHE_CONFIGS['agents'].key_prefix}:{workspace_id}")
    logger.debug(f"Invalidated agent analytics cache for {workspace_id}")


def on_cost_event(org_id: str) -> None:
    """Invalidate cost caches when billing events occur."""
    cache = get_analytics_dashboard_cache()
    for cache_type in ["cost", "tokens"]:
        cache._get_cache(cache_type).clear_prefix(
            f"{CACHE_CONFIGS[cache_type].key_prefix}:{org_id}"
        )
    logger.debug(f"Invalidated cost analytics caches for {org_id}")


__all__ = [
    # Main cache class
    "AnalyticsDashboardCache",
    "get_analytics_dashboard_cache",
    # Invalidation
    "invalidate_analytics_cache",
    # Decorators
    "cached_analytics",
    "cached_analytics_org",
    # Cache configs
    "CACHE_CONFIGS",
    "CacheConfig",
    # Invalidation hooks
    "on_debate_completed",
    "on_agent_performance_update",
    "on_cost_event",
]
