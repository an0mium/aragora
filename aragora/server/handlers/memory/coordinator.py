"""
Memory Coordinator endpoint handlers.

Endpoints:
- GET /api/memory/coordinator/metrics - Get coordinator metrics (success rate, rollbacks)
- GET /api/memory/coordinator/config - Get current coordinator configuration
"""

from __future__ import annotations

import logging
from typing import Optional


from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from ..secure import ForbiddenError, SecureHandler, UnauthorizedError
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# RBAC permission for coordinator endpoints
COORDINATOR_PERMISSION = "memory:admin"

# Rate limiter for coordinator endpoints (30 requests per minute)
_coordinator_limiter = RateLimiter(requests_per_minute=30)

# Optional import for coordinator functionality
try:
    from aragora.memory.coordinator import MemoryCoordinator, CoordinatorOptions

    COORDINATOR_AVAILABLE = True
except ImportError:
    COORDINATOR_AVAILABLE = False
    MemoryCoordinator = None  # type: ignore[misc,assignment]
    CoordinatorOptions = None  # type: ignore[misc,assignment]


class CoordinatorHandler(SecureHandler):
    """Handler for memory coordinator endpoints.

    Requires authentication and memory:read permission (RBAC).
    """

    ROUTES = [
        "/api/v1/memory/coordinator/metrics",
        "/api/v1/memory/coordinator/config",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        return path in self.ROUTES

    async def handle(  # type: ignore[override]
        self, path: str, query_params: dict, handler=None
    ) -> Optional[HandlerResult]:
        """Route coordinator requests with RBAC."""
        client_ip = get_client_ip(handler)

        # Rate limit all coordinator endpoints
        if not _coordinator_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for coordinator endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        # RBAC: Require authentication and memory:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, COORDINATOR_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required to access coordinator data", 401)
        except ForbiddenError as e:
            logger.warning(f"Coordinator access denied: {e}")
            return error_response(str(e), 403)

        if not COORDINATOR_AVAILABLE:
            return error_response(
                "Memory coordinator not available. Install aragora[memory] for full functionality.",
                501,
            )

        # Debug: Log path matching for troubleshooting
        logger.debug(f"Coordinator handler path: {path!r}")

        if path == "/api/v1/memory/coordinator/metrics":
            return self._get_metrics()
        if path == "/api/v1/memory/coordinator/config":
            return self._get_config()

        return None

    def _get_coordinator(self) -> Optional["MemoryCoordinator"]:
        """Get coordinator from context."""
        coordinator = self.ctx.get("memory_coordinator")
        if coordinator is None:
            return None
        return coordinator  # type: ignore[return-value]

    @handle_errors("coordinator metrics")
    def _get_metrics(self) -> HandlerResult:
        """Get coordinator metrics including success rate and rollback stats."""
        coordinator = self._get_coordinator()

        if not coordinator:
            # Return default metrics when no coordinator is configured
            return json_response(
                {
                    "configured": False,
                    "metrics": {
                        "total_transactions": 0,
                        "successful_transactions": 0,
                        "partial_failures": 0,
                        "rollbacks_performed": 0,
                        "success_rate": 0.0,
                    },
                    "memory_systems": {
                        "continuum": False,
                        "consensus": False,
                        "critique": False,
                        "mound": False,
                    },
                }
            )

        metrics = coordinator.get_metrics()

        return json_response(
            {
                "configured": True,
                "metrics": {
                    "total_transactions": metrics.get("total_transactions", 0),
                    "successful_transactions": metrics.get("successful_transactions", 0),
                    "partial_failures": metrics.get("partial_failures", 0),
                    "rollbacks_performed": metrics.get("rollbacks_performed", 0),
                    "success_rate": metrics.get("success_rate", 0.0),
                },
                "memory_systems": {
                    "continuum": coordinator.continuum_memory is not None,
                    "consensus": coordinator.consensus_memory is not None,
                    "critique": coordinator.critique_store is not None,
                    "mound": coordinator.knowledge_mound is not None,
                },
                "rollback_handlers": list(coordinator._rollback_handlers.keys()),
            }
        )

    @handle_errors("coordinator config")
    def _get_config(self) -> HandlerResult:
        """Get current coordinator configuration."""
        coordinator = self._get_coordinator()

        if not coordinator:
            # Return default config when no coordinator is configured
            return json_response(
                {
                    "configured": False,
                    "options": CoordinatorOptions().__dict__ if CoordinatorOptions else {},
                }
            )

        options = coordinator.options if hasattr(coordinator, "options") else CoordinatorOptions()

        return json_response(
            {
                "configured": True,
                "options": {
                    "write_continuum": options.write_continuum,
                    "write_consensus": options.write_consensus,
                    "write_critique": options.write_critique,
                    "write_mound": options.write_mound,
                    "rollback_on_failure": options.rollback_on_failure,
                    "parallel_writes": options.parallel_writes,
                    "min_confidence_for_mound": options.min_confidence_for_mound,
                },
            }
        )
