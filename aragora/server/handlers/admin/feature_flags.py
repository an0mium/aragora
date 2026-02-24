"""
Feature flag administration endpoint handlers.

Endpoints:
- GET /api/v1/admin/feature-flags - List all flags with values, categories, statuses
- GET /api/v1/admin/feature-flags/:name - Get flag value + usage stats
- PUT /api/v1/admin/feature-flags/:name - Toggle/set flag value (admin RBAC)
"""

from __future__ import annotations

__all__ = ["FeatureFlagAdminHandler"]

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
from aragora.server.middleware.mfa import enforce_admin_mfa_policy
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# Lazy imports for feature flag system
try:
    from aragora.config.feature_flags import (
        FlagCategory,
        FlagStatus,
        get_flag_registry,
    )

    FLAGS_AVAILABLE = True
except ImportError:
    FLAGS_AVAILABLE = False
    get_flag_registry = None  # type: ignore[assignment]
    FlagCategory = None  # type: ignore[assignment,misc]
    FlagStatus = None  # type: ignore[assignment,misc]


class FeatureFlagAdminHandler(BaseHandler):
    """Handler for feature flag administration endpoints."""

    ROUTES: list[str] = [
        "/api/v1/admin/feature-flags",
        "/api/v1/admin/feature-flags/*",
    ]

    def __init__(self, ctx: dict[str, Any] | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        path = strip_version_prefix(path)
        return path == "/api/admin/feature-flags" or path.startswith("/api/admin/feature-flags/")

    @handle_errors("feature flags GET")
    @require_permission("admin:feature_flags:read")
    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any, user: Any = None
    ) -> HandlerResult | None:
        """Handle GET requests for feature flag endpoints."""
        path = strip_version_prefix(path)

        if not FLAGS_AVAILABLE:
            return error_response("Feature flag system not available", 503)

        if path == "/api/admin/feature-flags":
            return self._list_flags(query_params, handler)

        if path.startswith("/api/admin/feature-flags/"):
            name = path.split("/api/admin/feature-flags/", 1)[1]
            if not name:
                return error_response("Flag name is required", 400)
            return self._get_flag(name, handler)

        return None

    @handle_errors("feature flags PUT")
    @require_permission("admin:feature_flags:write")
    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: Any, user: Any = None
    ) -> HandlerResult | None:
        """Handle PUT requests for feature flag endpoints."""
        path = strip_version_prefix(path)

        if not FLAGS_AVAILABLE:
            return error_response("Feature flag system not available", 503)

        if path.startswith("/api/admin/feature-flags/"):
            name = path.split("/api/admin/feature-flags/", 1)[1]
            if not name:
                return error_response("Flag name is required", 400)
            return self._set_flag(name, handler)

        return None

    @require_permission("admin:feature_flags")
    @rate_limit(requests_per_minute=60, limiter_name="feature_flags_list")
    def _list_flags(
        self,
        query_params: dict[str, Any],
        handler: Any = None,
        user: Any = None,
    ) -> HandlerResult:
        """List all feature flags with values, categories, and statuses.

        Query params:
            category: Optional filter by category (core, knowledge, etc.)
            status: Optional filter by status (active, beta, deprecated, etc.)
        """
        registry = get_flag_registry()

        # Parse optional filters
        category_filter = None
        status_filter = None

        category_str = query_params.get("category")
        if category_str:
            try:
                category_filter = FlagCategory(category_str)
            except ValueError:
                valid = [c.value for c in FlagCategory]
                return error_response(f"Invalid category. Valid: {', '.join(valid)}", 400)

        status_str = query_params.get("status")
        if status_str:
            try:
                status_filter = FlagStatus(status_str)
            except ValueError:
                valid = [s.value for s in FlagStatus]
                return error_response(f"Invalid status. Valid: {', '.join(valid)}", 400)

        flags = registry.get_all_flags(category=category_filter, status=status_filter)

        flag_list = []
        for flag in flags:
            current_value = registry.get_value(flag.name, flag.default)
            flag_list.append(
                {
                    "name": flag.name,
                    "value": current_value,
                    "default": flag.default,
                    "type": flag.flag_type.__name__,
                    "description": flag.description,
                    "category": flag.category.value,
                    "status": flag.status.value,
                    "env_var": flag.env_var,
                }
            )

        stats = registry.get_stats()

        return json_response(
            {
                "flags": flag_list,
                "total": len(flag_list),
                "stats": stats.to_dict(),
            }
        )

    @require_permission("admin:feature_flags")
    @rate_limit(requests_per_minute=60, limiter_name="feature_flags_get")
    def _get_flag(self, name: str, handler: Any = None, user: Any = None) -> HandlerResult:
        """Get a specific flag with its value and usage stats."""
        registry = get_flag_registry()

        definition = registry.get_definition(name)
        if not definition:
            return error_response(f"Flag not found: {name}", 404)

        current_value = registry.get_value(name, definition.default)
        usage = registry.get_usage(name)

        result: dict[str, Any] = {
            "name": definition.name,
            "value": current_value,
            "default": definition.default,
            "type": definition.flag_type.__name__,
            "description": definition.description,
            "category": definition.category.value,
            "status": definition.status.value,
            "env_var": definition.env_var,
        }

        if definition.deprecated_since:
            result["deprecated_since"] = definition.deprecated_since
        if definition.removed_in:
            result["removed_in"] = definition.removed_in
        if definition.replacement:
            result["replacement"] = definition.replacement

        if usage:
            result["usage"] = {
                "access_count": usage.access_count,
                "last_accessed": usage.last_accessed,
                "access_locations": dict(usage.access_locations),
            }

        return json_response(result)

    @require_permission("admin:feature_flags")
    @rate_limit(requests_per_minute=30, limiter_name="feature_flags_set")
    def _set_flag(self, name: str, handler: Any = None, user: Any = None) -> HandlerResult:
        """Set a feature flag value.

        Request body:
            value: The new value for the flag (must match flag type)
        """
        registry = get_flag_registry()

        definition = registry.get_definition(name)
        if not definition:
            return error_response(f"Flag not found: {name}", 404)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid JSON body", 400)

        if "value" not in body:
            return error_response("'value' field is required", 400)

        new_value = body["value"]

        # Validate type
        if not isinstance(new_value, definition.flag_type):
            return error_response(
                f"Flag '{name}' expects {definition.flag_type.__name__}, "
                f"got {type(new_value).__name__}",
                400,
            )

        # Set via environment variable override
        if definition.env_var:
            import os

            os.environ[definition.env_var] = str(new_value)
            logger.info("Feature flag '%s' set to %s via admin API", name, new_value)

        return json_response(
            {
                "name": name,
                "value": new_value,
                "previous_default": definition.default,
                "updated": True,
            }
        )
