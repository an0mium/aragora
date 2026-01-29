"""
Health handler - main entry point.

Re-exports HealthHandler from the package __init__ for backward compatibility.
The modular structure splits the implementation across multiple modules.
"""

# Re-export HealthHandler from the package
from . import (
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
