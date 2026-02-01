"""
Analytics handlers subpackage.

Consolidated analytics handlers for metrics and reporting.

This package contains:
- core: AnalyticsHandler and AnalyticsMetricsHandler re-exports
- cache: Workspace-scoped caching for analytics dashboard endpoints

Note: AnalyticsDashboardHandler remains in analytics_dashboard.py as a
standalone module due to its size.

Migration Notes:
- AnalyticsHandler from analytics.py
- AnalyticsMetricsHandler from analytics_metrics.py

All exports are maintained for backward compatibility.
"""

from .core import (
    # Handlers
    AnalyticsHandler,
    AnalyticsMetricsHandler,
    # Permissions
    ANALYTICS_PERMISSION,
    ANALYTICS_METRICS_PERMISSION,
    # Rate limiters
    _analytics_limiter,
    _analytics_metrics_limiter,
    # Utility functions
    _group_by_time,
    _parse_time_range,
    # Constants
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
)

from .cache import (
    # Cache class
    AnalyticsDashboardCache,
    get_analytics_dashboard_cache,
    # Invalidation
    invalidate_analytics_cache,
    # Decorators
    cached_analytics,
    cached_analytics_org,
    # Cache configs
    CACHE_CONFIGS,
    CacheConfig,
    # Invalidation hooks
    on_debate_completed,
    on_agent_performance_update,
    on_cost_event,
)

__all__ = [
    # Handlers
    "AnalyticsHandler",
    "AnalyticsMetricsHandler",
    # Permissions
    "ANALYTICS_PERMISSION",
    "ANALYTICS_METRICS_PERMISSION",
    # Rate limiters
    "_analytics_limiter",
    "_analytics_metrics_limiter",
    # Utility functions
    "_group_by_time",
    "_parse_time_range",
    # Constants
    "VALID_GRANULARITIES",
    "VALID_TIME_RANGES",
    # Cache
    "AnalyticsDashboardCache",
    "get_analytics_dashboard_cache",
    "invalidate_analytics_cache",
    "cached_analytics",
    "cached_analytics_org",
    "CACHE_CONFIGS",
    "CacheConfig",
    # Invalidation hooks
    "on_debate_completed",
    "on_agent_performance_update",
    "on_cost_event",
]
