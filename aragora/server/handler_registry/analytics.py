"""
Analytics and metrics handler imports and registry entries.

This module contains imports and registry entries for:
- Analytics handlers (AnalyticsHandler, AnalyticsDashboardHandler, etc.)
- Metrics handlers (MetricsHandler, UnifiedMetricsHandler, etc.)
- SLO handlers
- Memory analytics
- Pulse and trending handlers
"""

from __future__ import annotations

from .core import _safe_import

# =============================================================================
# Analytics Handler Imports
# =============================================================================

# Core analytics handlers
AnalyticsHandler = _safe_import("aragora.server.handlers", "AnalyticsHandler")
AnalyticsDashboardHandler = _safe_import("aragora.server.handlers", "AnalyticsDashboardHandler")
EndpointAnalyticsHandler = _safe_import("aragora.server.handlers", "EndpointAnalyticsHandler")
AnalyticsMetricsHandler = _safe_import("aragora.server.handlers", "AnalyticsMetricsHandler")

# Memory analytics
MemoryAnalyticsHandler = _safe_import("aragora.server.handlers", "MemoryAnalyticsHandler")

# Cross-platform analytics
CrossPlatformAnalyticsHandler = _safe_import(
    "aragora.server.handlers.features.cross_platform_analytics", "CrossPlatformAnalyticsHandler"
)

# Analytics platforms integration
AnalyticsPlatformsHandler = _safe_import(
    "aragora.server.handlers.features", "AnalyticsPlatformsHandler"
)

# =============================================================================
# Metrics Handler Imports
# =============================================================================

MetricsHandler = _safe_import("aragora.server.handlers", "MetricsHandler")
UnifiedMetricsHandler = _safe_import(
    "aragora.server.handlers.metrics_endpoint", "UnifiedMetricsHandler"
)
SLOHandler = _safe_import("aragora.server.handlers", "SLOHandler")

# =============================================================================
# Pulse & Trending Handler Imports
# =============================================================================

PulseHandler = _safe_import("aragora.server.handlers", "PulseHandler")

# =============================================================================
# Cost Handler
# =============================================================================

CostHandler = _safe_import("aragora.server.handlers.costs", "CostHandler")

# =============================================================================
# Usage Metering
# =============================================================================

UsageMeteringHandler = _safe_import(
    "aragora.server.handlers.usage_metering", "UsageMeteringHandler"
)

# =============================================================================
# Analytics Handler Registry Entries
# =============================================================================

ANALYTICS_HANDLER_REGISTRY: list[tuple[str, object]] = [
    ("_pulse_handler", PulseHandler),
    ("_analytics_handler", AnalyticsHandler),
    ("_analytics_dashboard_handler", AnalyticsDashboardHandler),
    ("_endpoint_analytics_handler", EndpointAnalyticsHandler),
    ("_analytics_metrics_handler", AnalyticsMetricsHandler),
    ("_metrics_handler", MetricsHandler),
    ("_unified_metrics_handler", UnifiedMetricsHandler),
    ("_slo_handler", SLOHandler),
    ("_memory_analytics_handler", MemoryAnalyticsHandler),
    ("_cross_platform_analytics_handler", CrossPlatformAnalyticsHandler),
    ("_analytics_platforms_handler", AnalyticsPlatformsHandler),
    ("_cost_handler", CostHandler),
    ("_usage_metering_handler", UsageMeteringHandler),
]

__all__ = [
    # Analytics handlers
    "AnalyticsHandler",
    "AnalyticsDashboardHandler",
    "EndpointAnalyticsHandler",
    "AnalyticsMetricsHandler",
    "MemoryAnalyticsHandler",
    "CrossPlatformAnalyticsHandler",
    "AnalyticsPlatformsHandler",
    # Metrics handlers
    "MetricsHandler",
    "UnifiedMetricsHandler",
    "SLOHandler",
    # Pulse handlers
    "PulseHandler",
    # Cost handlers
    "CostHandler",
    "UsageMeteringHandler",
    # Registry
    "ANALYTICS_HANDLER_REGISTRY",
]
