"""
Analytics Core Module - Consolidated Analytics Handlers.

This module consolidates analytics handlers for a unified domain structure.
The original handler implementations remain in their respective files for
maintainability, while this module provides a unified import point.

Handlers consolidated:
- AnalyticsHandler (from analytics.py): Core analytics endpoints
- AnalyticsMetricsHandler (from analytics_metrics.py): Debate/agent metrics

Note: AnalyticsDashboardHandler remains in analytics_dashboard.py as a separate
large module per the consolidation plan.

Migrated as part of handler consolidation Tier 1.
"""

from __future__ import annotations

# Re-export from original implementation files for backward compatibility
# The actual implementations remain in the parent directory for maintainability
from ..analytics import (
    ANALYTICS_PERMISSION,
    AnalyticsHandler,
    _analytics_limiter,
)

from ..analytics_metrics import (
    ANALYTICS_METRICS_PERMISSION,
    AnalyticsMetricsHandler,
    VALID_GRANULARITIES,
    VALID_TIME_RANGES,
    _analytics_metrics_limiter,
    _group_by_time,
    _parse_time_range,
)

__all__ = [
    # Core analytics handler
    "AnalyticsHandler",
    "ANALYTICS_PERMISSION",
    "_analytics_limiter",
    # Analytics metrics handler
    "AnalyticsMetricsHandler",
    "ANALYTICS_METRICS_PERMISSION",
    "_analytics_metrics_limiter",
    # Utility functions
    "_parse_time_range",
    "_group_by_time",
    # Constants
    "VALID_GRANULARITIES",
    "VALID_TIME_RANGES",
]
