"""
Operational metrics module.

This module provides:
- MetricsHandler for serving metrics endpoints
- Tracking functions for requests and verification
- Formatting utilities for display

Endpoints served:
- GET /api/metrics - Comprehensive operational metrics
- GET /api/metrics/health - Detailed health check
- GET /api/metrics/cache - Cache statistics
- GET /api/metrics/verification - Z3 verification statistics
- GET /api/metrics/system - System information
- GET /api/metrics/background - Background task statistics
- GET /api/metrics/debate - Debate performance statistics
- GET /metrics - Prometheus-format metrics
"""

from .formatters import format_size, format_uptime
from .handler import MetricsHandler
from .tracking import (
    _error_counts,
    _request_counts,
    _start_time,
    get_request_stats,
    get_start_time,
    get_verification_stats,
    track_request,
    track_verification,
)

__all__ = [
    # Handler
    "MetricsHandler",
    # Tracking
    "track_request",
    "track_verification",
    "get_verification_stats",
    "get_request_stats",
    "get_start_time",
    "_request_counts",
    "_error_counts",
    "_start_time",
    # Formatters
    "format_uptime",
    "format_size",
]
