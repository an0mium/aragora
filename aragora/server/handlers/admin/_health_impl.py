"""
Health and readiness endpoint handlers.

DEPRECATED: This module is a backward-compatibility shim.
The implementation has been split into the health/ package:
  - health/kubernetes.py: Liveness/readiness probes
  - health/database.py: Schema and stores health checks
  - health/detailed.py: Detailed, deep, and comprehensive health checks
  - health/knowledge_mound.py: Knowledge Mound and confidence decay health
  - health/cross_pollination.py: Cross-pollination feature health
  - health/platform.py: Platform resilience, encryption, and startup checks
  - health/diagnostics.py: Deployment diagnostics and production readiness checklist
  - health/helpers.py: Sync, circuits, slow debates, component health

Import from aragora.server.handlers.admin.health instead:
    from aragora.server.handlers.admin.health import HealthHandler
"""

from __future__ import annotations

# Re-export everything from the health package for backward compatibility
from .health import (
    HealthHandler,
    _get_cached_health,
    _set_cached_health,
    _SERVER_START_TIME,
    _HEALTH_CACHE,
    _HEALTH_CACHE_TTL,
    _HEALTH_CACHE_TTL_DETAILED,
    _HEALTH_CACHE_TIMESTAMPS,
)

__all__ = [
    "HealthHandler",
    "_get_cached_health",
    "_set_cached_health",
    "_SERVER_START_TIME",
    "_HEALTH_CACHE",
    "_HEALTH_CACHE_TTL",
    "_HEALTH_CACHE_TTL_DETAILED",
    "_HEALTH_CACHE_TIMESTAMPS",
]
