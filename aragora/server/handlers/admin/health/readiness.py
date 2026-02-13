"""
Readiness handler - standalone handler for /readyz endpoints.

Provides readiness probes for K8s deployments:
- /readyz: Fast readiness probe (<10ms, in-memory only)
- /readyz/dependencies: Full dependency validation (slow, 2-5s)

Usage:
    from aragora.server.handlers.admin.health.readiness import ReadinessHandler

    handler = ReadinessHandler(ctx)
    if handler.can_handle("/readyz"):
        result = await handler.handle("/readyz", {}, http_handler)
"""

from __future__ import annotations

import logging
from typing import Any

from ...base import HandlerResult
from ...secure import SecureHandler
from .kubernetes import readiness_probe_fast, readiness_dependencies

logger = logging.getLogger(__name__)


class ReadinessHandler(SecureHandler):
    """Handler for /readyz readiness probe endpoints.

    This handler provides readiness checks for Kubernetes readiness probes.
    Two modes are available:
    - Fast probe (/readyz): In-memory only checks, <10ms target latency.
    - Full probe (/readyz/dependencies): Validates all external connections (2-5s).

    RBAC Policy:
    - /readyz: Public (K8s probe, no auth required)
    - /readyz/dependencies: Public (K8s probe, no auth required)
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    ROUTES = [
        "/readyz",
        "/readyz/dependencies",
    ]

    PUBLIC_ROUTES = {
        "/readyz",
        "/readyz/dependencies",
    }

    def can_handle(self, path: str) -> bool:
        """Check if this handler can handle the given path."""
        return path in self.ROUTES

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Route readiness probe requests."""
        if path == "/readyz":
            return self._readiness_probe_fast()
        elif path == "/readyz/dependencies":
            return self._readiness_dependencies()
        return None

    def _readiness_probe_fast(self) -> HandlerResult:
        """Fast readiness probe - in-memory checks only."""
        return readiness_probe_fast(self)

    def _readiness_dependencies(self) -> HandlerResult:
        """Full dependency validation probe."""
        return readiness_dependencies(self)


__all__ = ["ReadinessHandler"]
