# mypy: ignore-errors
"""
Health handler package.

Provides health and readiness endpoints for Kubernetes deployments.
The handler is split into logical modules for better maintainability:

- kubernetes.py: K8s liveness/readiness probes
- database.py: Schema and stores health checks
- detailed.py: Detailed, deep, and comprehensive health checks
- knowledge_mound.py: Knowledge Mound and confidence decay health
- cross_pollination.py: Cross-pollination feature health
- platform.py: Platform resilience, encryption, and startup checks
- diagnostics.py: Deployment diagnostics and production readiness checklist
- helpers.py: Sync, circuits, slow debates, component health

For backward compatibility, import HealthHandler from this package:

    from aragora.server.handlers.admin.health import HealthHandler

Or from the legacy path:

    from aragora.server.handlers.admin._health_impl import HealthHandler
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from ...base import (
    HandlerResult,
    error_response,
)
from ...secure import SecureHandler, ForbiddenError, UnauthorizedError

logger = logging.getLogger(__name__)

# Server start time for uptime tracking
_SERVER_START_TIME = time.time()

# Health check cache for performance
# K8s probes need 5s TTL to ensure fast responses; detailed checks use 2s
_HEALTH_CACHE: dict[str, Any] = {}
_HEALTH_CACHE_TTL = 5.0  # seconds for K8s probes (liveness/readiness)
_HEALTH_CACHE_TTL_DETAILED = 2.0  # seconds for detailed health checks
_HEALTH_CACHE_TIMESTAMPS: dict[str, float] = {}


def _get_cached_health(key: str) -> Optional[dict[str, Any]]:
    """Get cached health result if still valid."""
    if key in _HEALTH_CACHE:
        cached_time = _HEALTH_CACHE_TIMESTAMPS.get(key, 0)
        if time.time() - cached_time < _HEALTH_CACHE_TTL:
            return _HEALTH_CACHE[key]
    return None


def _set_cached_health(key: str, value: dict[str, Any]) -> None:
    """Cache health check result."""
    _HEALTH_CACHE[key] = value
    _HEALTH_CACHE_TIMESTAMPS[key] = time.time()


# Import module functions
from .kubernetes import liveness_probe, readiness_probe_fast, readiness_dependencies
from .database import database_schema_health, database_stores_health
from .detailed import health_check, websocket_health, detailed_health_check, deep_health_check
from .knowledge_mound import knowledge_mound_health, decay_health
from .cross_pollination import cross_pollination_health
from .platform import startup_health, encryption_health, platform_health
from .diagnostics import deployment_diagnostics
from .helpers import (
    sync_status,
    slow_debates_status,
    circuit_breakers_status,
    component_health_status,
)

# Keep mixin imports for backward compatibility
from .probes import ProbesMixin
from .knowledge import KnowledgeMixin
from .stores import StoresMixin


class HealthHandler(SecureHandler):
    """Handler for health and readiness endpoints.

    RBAC Policy:
    - /healthz, /readyz: Public (K8s probes, no auth required)
    - All other endpoints: Require authentication and system.health.read permission
    """

    ROUTES = [
        "/healthz",
        "/readyz",
        "/readyz/dependencies",  # Full dependency validation (slow)
        # v1 routes
        "/api/v1/health",
        "/api/v1/health/detailed",
        "/api/v1/health/deep",
        "/api/v1/health/stores",
        "/api/v1/health/sync",
        "/api/v1/health/circuits",
        "/api/v1/health/components",  # Component health from HealthRegistry
        "/api/v1/health/slow-debates",
        "/api/v1/health/cross-pollination",
        "/api/v1/health/knowledge-mound",
        "/api/v1/health/decay",  # Confidence decay scheduler status
        "/api/v1/health/startup",  # Startup report and SLO status
        "/api/v1/health/encryption",
        "/api/v1/health/database",
        "/api/v1/health/platform",
        "/api/v1/platform/health",
        "/api/v1/diagnostics",
        "/api/v1/diagnostics/deployment",
        # Non-v1 routes (for backward compatibility)
        "/api/health",
        "/api/health/detailed",
        "/api/health/deep",
        "/api/health/stores",
        "/api/health/components",  # Component health from HealthRegistry
        "/api/diagnostics",
        "/api/diagnostics/deployment",
    ]

    # Routes that are public (no auth required)
    # SECURITY: Only K8s probes should be public to minimize information exposure.
    # All other health endpoints require authentication via system.health.read permission.
    PUBLIC_ROUTES = {
        "/healthz",  # K8s liveness probe
        "/readyz",  # K8s readiness probe
        "/readyz/dependencies",  # K8s extended readiness
    }
    # Note: /api/health and /api/v1/health now require authentication.
    # Load balancers should use /healthz or configure proper auth headers.

    # Permission required for protected health endpoints
    HEALTH_PERMISSION = "system.health.read"
    RESOURCE_TYPE = "health"

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route health endpoint requests with RBAC for non-public routes."""
        # K8s probes are public - no auth required
        if path not in self.PUBLIC_ROUTES:
            # All other health endpoints require authentication and permission
            try:
                auth_context = await self.get_auth_context(handler, require_auth=True)
                self.check_permission(auth_context, self.HEALTH_PERMISSION)
            except UnauthorizedError:
                return error_response("Authentication required", 401)
            except ForbiddenError as e:
                logger.warning(f"Health endpoint access denied: {e}")
                return error_response(str(e), 403)

        # Normalize path for routing (support both v1 and non-v1)
        normalized = path.replace("/api/v1/", "/api/")

        handlers = {
            "/healthz": self._liveness_probe,
            "/readyz": self._readiness_probe_fast,  # Fast check for K8s (<10ms)
            "/readyz/dependencies": self._readiness_dependencies,  # Full validation (slow)
            "/api/health": self._health_check,
            "/api/health/detailed": self._detailed_health_check,
            "/api/health/deep": self._deep_health_check,
            "/api/health/stores": self._database_stores_health,
            "/api/health/sync": self._sync_status,
            "/api/health/circuits": self._circuit_breakers_status,
            "/api/health/components": self._component_health_status,
            "/api/health/slow-debates": self._slow_debates_status,
            "/api/health/cross-pollination": self._cross_pollination_health,
            "/api/health/knowledge-mound": self._knowledge_mound_health,
            "/api/health/decay": self._decay_health,  # Confidence decay status
            "/api/health/startup": self._startup_health,  # Startup report
            "/api/health/database": self._database_schema_health,
            "/api/health/platform": self._platform_health,
            "/api/platform/health": self._platform_health,
            "/api/diagnostics": self._deployment_diagnostics,
            "/api/diagnostics/deployment": self._deployment_diagnostics,
        }

        endpoint_handler = handlers.get(normalized)
        if endpoint_handler:
            return endpoint_handler()
        return None

    # Delegate to module functions
    def _liveness_probe(self) -> HandlerResult:
        return liveness_probe(self)

    def _readiness_probe_fast(self) -> HandlerResult:
        return readiness_probe_fast(self)

    def _readiness_dependencies(self) -> HandlerResult:
        return readiness_dependencies(self)

    def _health_check(self) -> HandlerResult:
        return health_check(self)

    def _websocket_health(self) -> HandlerResult:
        return websocket_health(self)

    def _detailed_health_check(self) -> HandlerResult:
        return detailed_health_check(self)

    def _deep_health_check(self) -> HandlerResult:
        return deep_health_check(self)

    def _database_schema_health(self) -> HandlerResult:
        return database_schema_health(self)

    def _database_stores_health(self) -> HandlerResult:
        return database_stores_health(self)

    def _knowledge_mound_health(self) -> HandlerResult:
        return knowledge_mound_health(self)

    def _decay_health(self) -> HandlerResult:
        return decay_health(self)

    def _cross_pollination_health(self) -> HandlerResult:
        return cross_pollination_health(self)

    def _startup_health(self) -> HandlerResult:
        return startup_health(self)

    def _encryption_health(self) -> HandlerResult:
        return encryption_health(self)

    def _platform_health(self) -> HandlerResult:
        return platform_health(self)

    def _deployment_diagnostics(self) -> HandlerResult:
        return deployment_diagnostics(self)

    def _generate_checklist(self, result) -> dict[str, Any]:
        from .diagnostics import _generate_checklist

        return _generate_checklist(result)

    def _sync_status(self) -> HandlerResult:
        return sync_status(self)

    def _slow_debates_status(self) -> HandlerResult:
        return slow_debates_status(self)

    def _circuit_breakers_status(self) -> HandlerResult:
        return circuit_breakers_status(self)

    def _component_health_status(self) -> HandlerResult:
        return component_health_status(self)

    def _check_filesystem_health(self) -> dict[str, Any]:
        """Check filesystem write access to data directory."""
        from ..health_utils import check_filesystem_health

        nomic_dir = self.get_nomic_dir()
        return check_filesystem_health(nomic_dir)

    def _check_redis_health(self) -> dict[str, Any]:
        """Check Redis connectivity if configured."""
        from ..health_utils import check_redis_health

        return check_redis_health()

    def _check_ai_providers_health(self) -> dict[str, Any]:
        """Check AI provider API key availability."""
        from ..health_utils import check_ai_providers_health

        return check_ai_providers_health()

    def _check_security_services(self) -> dict[str, Any]:
        """Check security services health."""
        from ..health_utils import check_security_services

        return check_security_services()


__all__ = [
    # Main handler
    "HealthHandler",
    # Mixins (backward compat)
    "ProbesMixin",
    "KnowledgeMixin",
    "StoresMixin",
    # Cache utilities
    "_get_cached_health",
    "_set_cached_health",
    "_SERVER_START_TIME",
    "_HEALTH_CACHE",
    "_HEALTH_CACHE_TTL",
    "_HEALTH_CACHE_TTL_DETAILED",
    "_HEALTH_CACHE_TIMESTAMPS",
]
