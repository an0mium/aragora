"""Compatibility shim for analytics metrics handler.

Historically tests and integrations import:
    aragora.server.handlers.analytics_metrics.AnalyticsMetricsHandler

The implementation lives in `_analytics_metrics_impl.py`. This module
re-exports the handler to preserve that import path.
"""

from __future__ import annotations

from ._analytics_metrics_impl import AnalyticsMetricsHandler

__all__ = ["AnalyticsMetricsHandler"]
