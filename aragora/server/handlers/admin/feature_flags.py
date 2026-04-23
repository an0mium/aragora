"""
Feature flag administration endpoint handlers.

Endpoints:
- GET /api/v1/admin/feature-flags - List all flags with values, categories, statuses
- PUT /api/v1/admin/feature-flags - Update multiple flags in one request
- GET /api/v1/admin/feature-flags/:name - Get flag value + usage stats
- PUT /api/v1/admin/feature-flags/:name - Toggle/set flag value (admin RBAC)
"""

from __future__ import annotations

__all__ = ["FeatureFlagAdminHandler"]

import json
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


def _value_matches_flag_type(value: Any, flag_type: type) -> bool:
    """isinstance check that rejects ``bool`` values against ``int`` flags.

    ``bool`` is a subclass of ``int`` in Python, so ``isinstance(True, int)``
    is ``True``. That would let a batch update set an int-typed flag to a
    boolean and silently coerce to ``0``/``1`` at use time. Treating ``bool``
    as a distinct type across the admin API surfaces the mistake instead.
    """
    if flag_type is int and isinstance(value, bool):
        return False
    return isinstance(value, flag_type)


class FeatureFlagAdminHandler(BaseHandler):
    """Handler for feature flag administration endpoints."""

    ROUTES: list[str] = [
        "/api/v1/admin/feature-flags",
        "/api/v1/admin/feature-flags/*",
    ]
    ROUTE_PREFIXES: list[str] = ["/api/admin/feature-flags"]

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
        # Enforce MFA for admin users (SOC 2 CC5-01)
        if user is not None:
            user_store = self.ctx.get("user_store") if hasattr(self, "ctx") else None
            if user_store:
                mfa_result = enforce_admin_mfa_policy(user, user_store)
                if mfa_result and mfa_result.get("enforced"):
                    return error_response(
                        "Administrative access requires MFA. Please enable MFA at /api/auth/mfa/setup",
                        403,
                    )

        path = strip_version_prefix(path)

        if not FLAGS_AVAILABLE:
            return error_response("Feature flag system not available", 503)

        if path == "/api/admin/feature-flags":
            return self._set_flags(handler)

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
    @rate_limit(requests_per_minute=30, limiter_name="feature_flags_batch_set")
    def _set_flags(self, handler: Any = None, user: Any = None) -> HandlerResult:
        """Set multiple feature flags in a single request.

        Accepts either a plain JSON object mapping ``flag_name -> value`` or an
        object with a top-level ``flags`` dictionary for compatibility with
        older collection-level clients.

        Atomicity: all updates are validated up-front in a first pass; if any
        update is invalid the handler returns an error without mutating any
        feature flag. Only after every update passes validation do the
        ``os.environ`` writes happen. This avoids leaving the feature-flag
        runtime in a partially-applied state when a batch contains an
        invalid entry.

        Type check: ``bool`` is a subclass of ``int`` in Python, so a naive
        ``isinstance(value, int)`` check would accept booleans for integer
        flags. ``_value_matches_flag_type`` rejects that case explicitly to
        keep the numeric/boolean types separate across the admin API.
        """
        registry = get_flag_registry()

        body, parsed = self._read_json_body_value(handler)
        if not parsed:
            return error_response("Invalid JSON body", 400)
        if not isinstance(body, dict):
            return error_response("JSON body must deserialize to an object", 400)

        updates = body.get("flags", body)
        if not isinstance(updates, dict) or not updates:
            return error_response("Expected at least one feature flag update", 400)

        validated: list[tuple[str, Any, Any]] = []
        for name, new_value in updates.items():
            if not isinstance(name, str) or not name:
                return error_response("Feature flag names must be non-empty strings", 400)

            definition = registry.get_definition(name)
            if not definition:
                return error_response(f"Flag not found: {name}", 404)

            if not _value_matches_flag_type(new_value, definition.flag_type):
                return error_response(
                    f"Flag '{name}' expects {definition.flag_type.__name__}, "
                    f"got {type(new_value).__name__}",
                    400,
                )

            validated.append((name, new_value, definition))

        import os

        results: dict[str, dict[str, Any]] = {}
        for applied_name, applied_value, applied_def in validated:
            if applied_def.env_var:
                os.environ[applied_def.env_var] = str(applied_value)
                logger.info(
                    "Feature flag '%s' set to %s via admin API", applied_name, applied_value
                )

            results[applied_name] = {
                "name": applied_name,
                "value": applied_value,
                "previous_default": applied_def.default,
                "updated": True,
            }

        return json_response(results)

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

        body, parsed = self._read_json_body_value(handler)
        if not parsed:
            return error_response("Invalid JSON body", 400)
        if not isinstance(body, dict):
            return error_response("JSON body must deserialize to an object", 400)

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

    def _read_json_body_value(
        self,
        handler: Any,
        max_size: int | None = None,
    ) -> tuple[Any | None, bool]:
        """Read JSON while distinguishing parse failures from non-object payloads."""
        max_size = max_size or self.MAX_BODY_SIZE
        try:
            for raw_body in (
                getattr(handler, "body", None),
                getattr(getattr(handler, "request", None), "body", None),
            ):
                if isinstance(raw_body, str):
                    raw_body = raw_body.encode("utf-8")
                if isinstance(raw_body, (bytes, bytearray)):
                    if len(raw_body) > max_size:
                        return None, False
                    if not raw_body:
                        return {}, True
                    return json.loads(bytes(raw_body)), True

            content_length = int(handler.headers.get("Content-Length", 0))
            is_chunked = "chunked" in (handler.headers.get("Transfer-Encoding", "") or "").lower()

            if content_length > max_size:
                return None, False

            if content_length > 0:
                body = handler.rfile.read(content_length)
            elif is_chunked or content_length == 0:
                body = handler.rfile.read(max_size)
            else:
                return {}, True

            if not body:
                return {}, True
            if len(body) > max_size:
                return None, False
            return json.loads(body), True
        except (AttributeError, json.JSONDecodeError, TypeError, ValueError):
            return None, False
