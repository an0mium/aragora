"""
Public Status Page endpoint handler.

Stability: STABLE

Exposes the public status page at /api/v1/status (no authentication required).
This endpoint is designed for external monitoring tools, status page widgets,
and customers checking service health.

Endpoints:
- GET /api/v1/status - Public service status (no auth required)

Rate Limiting:
- 60 requests per minute per client IP

Usage:
    curl http://localhost:8080/api/v1/status
"""

from __future__ import annotations

import logging
from typing import Any

from aragora.ops.status_page import get_status_page
from aragora.server.versioning.compat import strip_version_prefix

from .base import BaseHandler, HandlerResult, error_response, json_response
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for status endpoint (60 requests per minute - generous for monitoring)
_status_limiter = RateLimiter(requests_per_minute=60)


class StatusPageHandler(BaseHandler):
    """Handler for the public status page endpoint.

    Stability: STABLE

    Provides an unauthenticated status endpoint suitable for external
    monitoring services, customer-facing status pages, and health checks.

    No RBAC or authentication is required -- this is intentionally public.

    Example:
        handler = StatusPageHandler(ctx)
        if handler.can_handle("/api/v1/status"):
            result = handler.handle("/api/v1/status", {}, http_handler)
    """

    ROUTES = [
        "/api/status",
    ]

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context.

        Args:
            ctx: Server context dictionary. Optional for this handler
                 as it reads from the StatusPage singleton.
        """
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = strip_version_prefix(path)
        return normalized in self.ROUTES

    def handle(self, path: str, query_params: dict, handler: Any) -> HandlerResult | None:
        """Handle GET /api/v1/status -- public, no auth required.

        Args:
            path: Request path
            query_params: Parsed query parameters
            handler: HTTP request handler instance

        Returns:
            JSON response with full status page data
        """
        normalized = strip_version_prefix(path)

        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _status_limiter.is_allowed(client_ip):
            logger.warning("Rate limit exceeded for status endpoint: %s", client_ip)
            return error_response(
                "Rate limit exceeded. Please try again later.", 429, code="RATE_LIMITED"
            )

        try:
            if normalized == "/api/status":
                return self._handle_status()
            return error_response(f"Unknown status endpoint: {path}", 404, code="NOT_FOUND")
        except (KeyError, ValueError, AttributeError, TypeError, RuntimeError, OSError) as e:
            logger.exception("Error handling status request: %s", e)
            return error_response("Internal server error", 500, code="INTERNAL_ERROR")

    def _handle_status(self) -> HandlerResult:
        """GET /api/v1/status - Full public status page."""
        try:
            page = get_status_page()
            return json_response({"data": page.to_dict()})
        except (KeyError, ValueError, AttributeError, TypeError, RuntimeError, OSError) as e:
            logger.exception("Failed to build status page: %s", e)
            return error_response("Internal server error", 500, code="STATUS_ERROR")


__all__ = ["StatusPageHandler"]
