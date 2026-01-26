"""
Health handler package.

Provides health and readiness endpoints for Kubernetes deployments.
The handler is split into logical modules for better maintainability:

- handler.py: Main HealthHandler class (re-exported from _health_impl.py)
- probes.py: K8s liveness/readiness probes mixin
- checks/: Health check implementations (basic, detailed, deep, database)
- status/: Status endpoints (sync, circuits, debates)
- features/: Feature-specific health (cross_pollination, knowledge_mound, encryption)
- platform.py: Platform health
- diagnostics.py: Deployment diagnostics

For backward compatibility, import HealthHandler from this package:

    from aragora.server.handlers.admin.health import HealthHandler
"""

from .handler import (
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
