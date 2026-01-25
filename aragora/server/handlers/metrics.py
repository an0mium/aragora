"""
Operational metrics endpoint handlers.

DEPRECATED: This module has been refactored into the metrics/ package.
Import from aragora.server.handlers.metrics instead.

This file re-exports for backward compatibility.
"""

# Re-export everything from the new module for backward compatibility
from .metrics import (
    MetricsHandler,
    format_size,
    format_uptime,
    get_request_stats,
    get_start_time,
    get_verification_stats,
    track_request,
    track_verification,
)

__all__ = [
    "MetricsHandler",
    "track_request",
    "track_verification",
    "get_verification_stats",
    "get_request_stats",
    "get_start_time",
    "format_uptime",
    "format_size",
]
