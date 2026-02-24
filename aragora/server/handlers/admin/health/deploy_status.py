"""
Deploy status endpoint.

Provides /api/v1/deploy/status endpoint that returns:
- Current deployed SHA
- Deploy timestamp
- Health status of backend
- Uptime information
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from ...base import HandlerResult, json_response, error_response
from ...secure import SecureHandler
from ...utils.auth import UnauthorizedError, ForbiddenError

logger = logging.getLogger(__name__)


class DeployStatusHandler(SecureHandler):
    """Handler for the deploy status endpoint.

    Shows current deployment information including SHA, health, and uptime.

    RBAC Policy:
    - Requires authentication and system.health.read permission.
    """

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    ROUTES = [
        "/api/deploy/status",
        "/api/v1/deploy/status",
    ]

    DEPLOY_PERMISSION = "system.health.read"

    def can_handle(self, path: str) -> bool:
        return path in self.ROUTES

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        if path not in self.ROUTES:
            return None

        # Require authentication
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, self.DEPLOY_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError:
            return error_response("Permission denied", 403)

        return self._deploy_status()

    def _deploy_status(self) -> HandlerResult:
        """Return comprehensive deploy status."""
        from aragora.server.build_info import get_build_info
        from . import _SERVER_START_TIME

        build = get_build_info()
        now = time.time()
        uptime_seconds = now - _SERVER_START_TIME

        # Check backend health
        backend_healthy = True
        try:
            from aragora.server.degraded_mode import is_degraded
            if is_degraded():
                backend_healthy = False
        except ImportError:
            pass

        # Check server readiness
        try:
            from aragora.server.unified_server import is_server_ready
            server_ready = is_server_ready()
        except ImportError:
            server_ready = True

        return json_response({
            "deploy": {
                "sha": build["sha"],
                "sha_short": build["sha_short"],
                "build_time": build["build_time"],
                "version": build["version"],
            },
            "health": {
                "backend": "healthy" if backend_healthy else "degraded",
                "server_ready": server_ready,
            },
            "uptime": {
                "seconds": round(uptime_seconds, 1),
                "started_at": datetime.fromtimestamp(
                    _SERVER_START_TIME, tz=timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })


__all__ = ["DeployStatusHandler"]
