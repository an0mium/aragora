"""
Liveness handler - standalone handler for /healthz endpoint.

Provides a lightweight liveness probe suitable for K8s deployments.
The liveness probe checks if the server process is alive and can respond.
It should never check external dependencies or perform heavy operations.

Usage:
    from aragora.server.handlers.admin.health.liveness import LivenessHandler

    handler = LivenessHandler(ctx)
    if handler.can_handle("/healthz"):
        result = await handler.handle("/healthz", {}, http_handler)
"""

from __future__ import annotations

import logging
from typing import Any

from ...base import HandlerResult
from ...secure import SecureHandler
from .kubernetes import liveness_probe

logger = logging.getLogger(__name__)


class LivenessHandler(SecureHandler):
    """Handler for the /healthz liveness probe endpoint.

    This handler provides a simple alive check for Kubernetes liveness probes.
    It returns 200 if the server process is running and can respond, even in
    degraded mode (the container is alive and should not be restarted).

    RBAC Policy:
    - /healthz: Public (K8s probe, no auth required)
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = ["/healthz"]

    PUBLIC_ROUTES = {"/healthz"}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route liveness probe request."""
        if path == "/healthz":
            return self._liveness_probe()
        return None

    def _liveness_probe(self) -> HandlerResult:
        """Delegate to the kubernetes module liveness probe."""
        return liveness_probe(self)


__all__ = ["LivenessHandler"]
