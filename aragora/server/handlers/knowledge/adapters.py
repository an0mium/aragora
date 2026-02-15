"""
Knowledge Mound adapter status endpoint handler.

Endpoints:
- GET /api/v1/knowledge/adapters - List all KM adapters with status, priority,
  circuit breaker state, and sync metrics.
"""

from __future__ import annotations

__all__ = ["KMAdapterStatusHandler"]

import logging
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from aragora.rbac.decorators import require_permission
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Lazy imports for KM components
try:
    from aragora.knowledge.mound.adapters.factory import ADAPTER_SPECS

    KM_AVAILABLE = True
except ImportError:
    KM_AVAILABLE = False
    ADAPTER_SPECS = {}  # type: ignore[misc]


class KMAdapterStatusHandler(BaseHandler):
    """Handler for KM adapter status endpoints."""

    ROUTES: list[str] = [
        "/api/v1/knowledge/adapters",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        path = strip_version_prefix(path)
        return path == "/api/knowledge/adapters"

    @handle_errors("KM adapter status GET")
    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle GET requests for adapter status."""
        path = strip_version_prefix(path)

        if path != "/api/knowledge/adapters":
            return None

        return self._list_adapters(handler)

    @require_permission("knowledge:read")
    @rate_limit(requests_per_minute=30, limiter_name="km_adapter_status")
    def _list_adapters(
        self, handler: Any = None, user: Any = None
    ) -> HandlerResult:
        """List all KM adapters with status information.

        Returns adapter specs from the factory registry and, if a coordinator
        is available in server context, live status from the coordinator.
        """
        if not KM_AVAILABLE:
            return error_response("Knowledge Mound system not available", 503)

        adapters: list[dict[str, Any]] = []

        # Get live coordinator status if available
        coordinator_status: dict[str, Any] = {}
        coordinator = self.ctx.get("km_coordinator")
        if coordinator and hasattr(coordinator, "get_status"):
            try:
                status = coordinator.get_status()
                coordinator_status = status.get("adapters", {})
            except (AttributeError, TypeError, ValueError, RuntimeError):
                logger.debug("Could not get coordinator status", exc_info=True)

        # Build adapter list from specs
        for name, spec in sorted(ADAPTER_SPECS.items()):
            adapter_info: dict[str, Any] = {
                "name": name,
                "priority": spec.priority,
                "enabled_by_default": spec.enabled_by_default,
                "required_deps": spec.required_deps,
                "forward_method": spec.forward_method,
                "reverse_method": spec.reverse_method,
            }

            # Merge live status if available
            live = coordinator_status.get(name)
            if live:
                adapter_info["enabled"] = live.get("enabled", spec.enabled_by_default)
                adapter_info["has_reverse"] = live.get("has_reverse", spec.reverse_method is not None)
                adapter_info["forward_errors"] = live.get("forward_errors", 0)
                adapter_info["reverse_errors"] = live.get("reverse_errors", 0)
                adapter_info["last_forward_sync"] = live.get("last_forward_sync")
                adapter_info["last_reverse_sync"] = live.get("last_reverse_sync")
                adapter_info["status"] = "active"
            else:
                adapter_info["status"] = "registered"

            adapters.append(adapter_info)

        return json_response(
            {
                "adapters": adapters,
                "total": len(adapters),
                "coordinator_available": coordinator is not None,
            }
        )
