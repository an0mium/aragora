"""
Health handler - main entry point.

Re-exports HealthHandler from the implementation module for backward compatibility.
The modular structure is in place for future incremental migration of methods.
"""

# Re-export HealthHandler from the implementation module
from .._health_impl import (
    HealthHandler,
    _get_cached_health,
    _set_cached_health,
    _SERVER_START_TIME,
    _HEALTH_CACHE,
    _HEALTH_CACHE_TTL,
    _HEALTH_CACHE_TIMESTAMPS,
)

__all__ = [
    "HealthHandler",
    "_get_cached_health",
    "_set_cached_health",
    "_SERVER_START_TIME",
    "_HEALTH_CACHE",
    "_HEALTH_CACHE_TTL",
    "_HEALTH_CACHE_TIMESTAMPS",
]
