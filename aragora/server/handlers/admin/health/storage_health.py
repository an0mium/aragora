"""
Storage health handler - standalone handler for database and store health endpoints.

Provides health checks for storage subsystems:
- /api/v1/health/stores (and /api/health/stores): All database stores health
- /api/v1/health/database: Database schema health

Usage:
    from aragora.server.handlers.admin.health.storage_health import StorageHealthHandler

    handler = StorageHealthHandler(ctx)
    if handler.can_handle("/api/v1/health/stores"):
        result = await handler.handle("/api/v1/health/stores", {}, http_handler)
"""

from __future__ import annotations

import logging
from typing import Any, cast, TYPE_CHECKING

from ...base import HandlerResult, error_response
from ...secure import SecureHandler
from ...utils.auth import ForbiddenError, UnauthorizedError
from .database import database_schema_health, database_stores_health

if TYPE_CHECKING:
    from .database import _HealthHandlerProtocol as _DatabaseHandlerProtocol

logger = logging.getLogger(__name__)


class StorageHealthHandler(SecureHandler):
    """Handler for storage and database health check endpoints.

    Provides health checks for database schema validation and all
    database stores (debate storage, ELO, insights, consensus, etc.).

    RBAC Policy:
    - All endpoints require authentication and system.health.read permission.
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/api/v1/health/stores",
        "/api/v1/health/database",
        # Non-v1 backward compatibility
        "/api/health/stores",
    ]

    # Permission required for protected health endpoints
    HEALTH_PERMISSION = "system.health.read"
    RESOURCE_TYPE = "health"

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route storage health endpoint requests with RBAC.

        All storage health endpoints require authentication and
        system.health.read permission.
        """
        # All storage health endpoints require authentication
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, self.HEALTH_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning("Storage health endpoint access denied: %s", e)
            return error_response("Permission denied", 403)

        # Normalize path for routing (support both v1 and non-v1)
        normalized = path.replace("/api/v1/", "/api/")

        if normalized == "/api/health/stores":
            return self._database_stores_health()
        elif normalized == "/api/health/database":
            return self._database_schema_health()

        return None

    def _database_stores_health(self) -> HandlerResult:
        """Check health of all database stores."""
        return database_stores_health(cast("_DatabaseHandlerProtocol", self))

    def _database_schema_health(self) -> HandlerResult:
        """Check health of consolidated database schema."""
        return database_schema_health(cast("_DatabaseHandlerProtocol", self))

    # Provide the same context accessors as HealthHandler for the database module Protocol
    def get_storage(self) -> Any:
        """Get debate storage instance."""
        return self.ctx.get("storage")

    def get_elo_system(self) -> Any:
        """Get ELO ranking system instance."""
        if hasattr(self.__class__, "elo_system") and self.__class__.elo_system is not None:
            return self.__class__.elo_system
        return self.ctx.get("elo_system")

    def get_nomic_dir(self) -> Any:
        """Get nomic directory path."""
        return self.ctx.get("nomic_dir")


__all__ = ["StorageHealthHandler"]
