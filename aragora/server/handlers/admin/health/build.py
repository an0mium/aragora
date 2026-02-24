"""
Build health endpoint.

Provides /health/build and /api/v1/health/build endpoints that return
the current git SHA, build time, and deploy version.

Used by CI/CD pipelines to verify that the correct code is running
after a deployment.
"""

from __future__ import annotations

import logging
from typing import Any

from ...base import HandlerResult, json_response
from ...secure import SecureHandler

logger = logging.getLogger(__name__)


class BuildInfoHandler(SecureHandler):
    """Handler for the /health/build endpoint.

    Returns build information including git SHA, build time, and version.
    This endpoint is public so CI/CD pipelines can verify deployments
    without authentication.

    RBAC Policy:
    - /health/build, /api/health/build, /api/v1/health/build: Public
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/health/build",
        "/api/health/build",
        "/api/v1/health/build",
    ]

    PUBLIC_ROUTES = {
        "/health/build",
        "/api/health/build",
        "/api/v1/health/build",
    }

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Return build information."""
        if path in self.ROUTES:
            return self._build_info(query_params)
        return None

    def _build_info(self, query_params: dict[str, Any]) -> HandlerResult:
        """Return build SHA, time, and version.

        If ?verify=<sha> is provided, also checks whether the running
        build matches the expected SHA.
        """
        from aragora.server.build_info import get_build_info, verify_sha

        info = get_build_info()
        response: dict[str, Any] = {
            "sha": info["sha"],
            "sha_short": info["sha_short"],
            "build_time": info["build_time"],
            "version": info["version"],
        }

        expected = query_params.get("verify", "")
        if expected:
            verification = verify_sha(expected)
            response["verification"] = verification

        return json_response(response)


__all__ = ["BuildInfoHandler"]
