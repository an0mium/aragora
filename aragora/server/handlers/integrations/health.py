"""
Integration health check endpoint handler.

Endpoints:
- GET /api/v1/integrations/health - Returns health status for each configured
  integration (Slack, Email, Discord, Teams, Zapier).
"""

from __future__ import annotations

__all__ = ["IntegrationHealthHandler"]

import logging
import os
from typing import Any

from aragora.server.versioning.compat import strip_version_prefix

from ..base import (
    BaseHandler,
    HandlerResult,
    handle_errors,
    json_response,
)
from aragora.rbac.decorators import require_permission
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


# Integration definitions: name -> (env_vars_that_indicate_configured, module_path)
_INTEGRATIONS: list[dict[str, Any]] = [
    {
        "name": "slack",
        "env_vars": ["SLACK_WEBHOOK_URL", "SLACK_BOT_TOKEN"],
        "module": "aragora.integrations.slack",
    },
    {
        "name": "email",
        "env_vars": ["SMTP_HOST", "SMTP_SERVER", "SENDGRID_API_KEY"],
        "module": "aragora.integrations.email",
    },
    {
        "name": "discord",
        "env_vars": ["DISCORD_WEBHOOK_URL", "DISCORD_BOT_TOKEN"],
        "module": "aragora.integrations.discord",
    },
    {
        "name": "teams",
        "env_vars": ["TEAMS_WEBHOOK_URL", "MS_TEAMS_WEBHOOK"],
        "module": "aragora.integrations.teams",
    },
    {
        "name": "zapier",
        "env_vars": ["ZAPIER_WEBHOOK_URL", "ZAPIER_API_KEY"],
        "module": "aragora.integrations.zapier",
    },
]


class IntegrationHealthHandler(BaseHandler):
    """Handler for integration health check endpoint."""

    ROUTES: list[str] = [
        "/api/v1/integrations/health",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        path = strip_version_prefix(path)
        return path == "/api/integrations/health"

    @handle_errors("integration health GET")
    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any
    ) -> HandlerResult | None:
        """Handle GET requests for integration health."""
        path = strip_version_prefix(path)

        if path != "/api/integrations/health":
            return None

        return self._get_health(handler)

    @require_permission("integrations:read")
    @rate_limit(requests_per_minute=30, limiter_name="integration_health")
    def _get_health(
        self, handler: Any = None, user: Any = None
    ) -> HandlerResult:
        """Get health status for all integrations."""
        results: list[dict[str, Any]] = []

        for integration in _INTEGRATIONS:
            name = integration["name"]
            env_vars = integration["env_vars"]
            module_path = integration["module"]

            # Check if configured (any relevant env var is set)
            configured = any(os.environ.get(var) for var in env_vars)

            # Check if module is importable
            module_available = False
            try:
                __import__(module_path)
                module_available = True
            except ImportError:
                pass

            # Check connector state from server context
            healthy = False
            last_check = None
            connectors = self.ctx.get("connectors", {})
            connector = connectors.get(name) if isinstance(connectors, dict) else None

            if connector:
                healthy = getattr(connector, "healthy", False)
                last_check_val = getattr(connector, "last_check", None)
                if last_check_val:
                    last_check = (
                        last_check_val.isoformat()
                        if hasattr(last_check_val, "isoformat")
                        else str(last_check_val)
                    )

            results.append(
                {
                    "name": name,
                    "configured": configured,
                    "module_available": module_available,
                    "healthy": healthy,
                    "last_check": last_check,
                }
            )

        configured_count = sum(1 for r in results if r["configured"])
        healthy_count = sum(1 for r in results if r["healthy"])

        return json_response(
            {
                "integrations": results,
                "total": len(results),
                "configured": configured_count,
                "healthy": healthy_count,
            }
        )
