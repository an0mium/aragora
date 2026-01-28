"""
Health handler package.

Provides health and readiness endpoints for Kubernetes deployments.
The handler is split into logical modules for better maintainability:

- handler.py: Main HealthHandler class (re-exported from _health_impl.py)
- probes.py: K8s liveness/readiness probes mixin (ProbesMixin)
- knowledge.py: Knowledge Mound health checks (KnowledgeMixin)
- platform.py: Platform health and deployment diagnostics (PlatformMixin)

The mixins provide extracted implementations that can be composed into handlers.
For backward compatibility, import HealthHandler from this package:

    from aragora.server.handlers.admin.health import HealthHandler

Mixins for custom handlers:

    from aragora.server.handlers.admin.health import (
        ProbesMixin,
        KnowledgeMixin,
        PlatformMixin,
    )
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
from .probes import ProbesMixin
from .knowledge import KnowledgeMixin
from .platform import PlatformMixin

__all__ = [
    # Main handler
    "HealthHandler",
    # Mixins
    "ProbesMixin",
    "KnowledgeMixin",
    "PlatformMixin",
    # Cache utilities
    "_get_cached_health",
    "_set_cached_health",
    "_SERVER_START_TIME",
    "_HEALTH_CACHE",
    "_HEALTH_CACHE_TTL",
    "_HEALTH_CACHE_TIMESTAMPS",
]
