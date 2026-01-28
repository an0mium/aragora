"""
Bindings endpoint handlers.

Endpoints:
- GET /api/bindings - List all message bindings
- GET /api/bindings/:provider - List bindings for a provider
- POST /api/bindings - Create a new binding
- PUT /api/bindings/:id - Update a binding
- DELETE /api/bindings/:id - Remove a binding
- POST /api/bindings/resolve - Resolve binding for a message
- GET /api/bindings/stats - Get router statistics
"""

from __future__ import annotations

__all__ = [
    "BindingsHandler",
]

import json
import logging
from typing import Any, Optional

from aragora.server.versioning.compat import strip_version_prefix

from .base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from aragora.rbac.decorators import require_permission
from .utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# Rate limiter for bindings endpoints (60 requests per minute)
_bindings_limiter = RateLimiter(requests_per_minute=60)

# Lazy imports for bindings system
try:
    from aragora.server.bindings import (
        BindingRouter,
        BindingType,
        MessageBinding,
        get_binding_router,
    )

    BINDINGS_AVAILABLE = True
except ImportError:
    BINDINGS_AVAILABLE = False
    get_binding_router = None  # type: ignore[assignment, misc]
    BindingRouter = None  # type: ignore[assignment, misc]
    BindingType = None  # type: ignore[assignment, misc]
    MessageBinding = None  # type: ignore[assignment, misc]


class BindingsHandler(BaseHandler):
    """Handler for message bindings management endpoints."""

    ROUTES: list[str] = [
        "/api/bindings",
        "/api/bindings/resolve",
        "/api/bindings/stats",
        "/api/bindings/*",  # Must be last due to wildcard
    ]

    def __init__(self, server_context: dict):
        """Initialize with server context."""
        super().__init__(server_context)  # type: ignore[arg-type]
        self._router: Optional["BindingRouter"] = None

    def _get_router(self) -> Optional["BindingRouter"]:
        """Get or create the binding router singleton."""
        if self._router is None and BINDINGS_AVAILABLE and get_binding_router:
            self._router = get_binding_router()
        return self._router

    @handle_errors("bindings GET request")
    async def handle_get(self, path: str, request: Any) -> HandlerResult:
        """Handle GET requests for bindings endpoints."""
        path = strip_version_prefix(path)

        if not BINDINGS_AVAILABLE:
            return error_response(
                "Bindings system not available",
                503,
                code="BINDINGS_UNAVAILABLE",
            )

        # Rate limit check
        client_ip = get_client_ip(request)
        if not _bindings_limiter.is_allowed(client_ip):
            return error_response(
                "Rate limit exceeded for bindings endpoints",
                429,
                code="RATE_LIMITED",
            )

        # GET /api/bindings - List all bindings
        if path == "/api/bindings":
            return await self._list_bindings(request)

        # GET /api/bindings/stats - Get router statistics
        if path == "/api/bindings/stats":
            return await self._get_stats(request)

        # GET /api/bindings/:provider - List bindings for provider
        parts = path.split("/")
        if len(parts) >= 4:
            provider = parts[3]
            return await self._list_bindings_by_provider(provider, request)

        return error_response(f"Unknown bindings endpoint: {path}", 404)

    @handle_errors("bindings POST request")
    async def handle_post(self, path: str, request: Any) -> HandlerResult:
        """Handle POST requests for bindings endpoints."""
        path = strip_version_prefix(path)

        if not BINDINGS_AVAILABLE:
            return error_response(
                "Bindings system not available",
                503,
                code="BINDINGS_UNAVAILABLE",
            )

        # Rate limit check
        client_ip = get_client_ip(request)
        if not _bindings_limiter.is_allowed(client_ip):
            return error_response(
                "Rate limit exceeded for bindings endpoints",
                429,
                code="RATE_LIMITED",
            )

        # POST /api/bindings/resolve
        if path == "/api/bindings/resolve":
            return await self._resolve_binding(request)

        # POST /api/bindings - Create binding
        if path == "/api/bindings":
            return await self._create_binding(request)

        return error_response(f"Unknown bindings endpoint: {path}", 404)

    @handle_errors("bindings DELETE request")
    async def handle_delete(self, path: str, request: Any) -> HandlerResult:
        """Handle DELETE requests for bindings endpoints."""
        path = strip_version_prefix(path)

        if not BINDINGS_AVAILABLE:
            return error_response(
                "Bindings system not available",
                503,
                code="BINDINGS_UNAVAILABLE",
            )

        # Rate limit check
        client_ip = get_client_ip(request)
        if not _bindings_limiter.is_allowed(client_ip):
            return error_response(
                "Rate limit exceeded for bindings endpoints",
                429,
                code="RATE_LIMITED",
            )

        # DELETE /api/bindings/:provider/:account/:pattern
        return await self._delete_binding(path, request)

    @require_permission("bindings.read")
    async def _list_bindings(self, request: Any) -> HandlerResult:
        """List all registered bindings."""
        router = self._get_router()
        if not router:
            return error_response(
                "Binding router not available",
                503,
                code="ROUTER_UNAVAILABLE",
            )

        bindings = router.list_bindings()
        return json_response(
            {
                "bindings": [b.to_dict() for b in bindings],
                "total": len(bindings),
            }
        )

    @require_permission("bindings.read")
    async def _list_bindings_by_provider(self, provider: str, request: Any) -> HandlerResult:
        """List bindings for a specific provider."""
        router = self._get_router()
        if not router:
            return error_response(
                "Binding router not available",
                503,
                code="ROUTER_UNAVAILABLE",
            )

        bindings = router.list_bindings(provider=provider)
        return json_response(
            {
                "provider": provider,
                "bindings": [b.to_dict() for b in bindings],
                "total": len(bindings),
            }
        )

    @require_permission("bindings.read")
    async def _get_stats(self, request: Any) -> HandlerResult:
        """Get router statistics."""
        router = self._get_router()
        if not router:
            return error_response(
                "Binding router not available",
                503,
                code="ROUTER_UNAVAILABLE",
            )

        stats = router.get_stats()
        return json_response(stats)

    @require_permission("bindings.create")
    async def _create_binding(self, request: Any) -> HandlerResult:
        """Create a new message binding."""
        router = self._get_router()
        if not router:
            return error_response(
                "Binding router not available",
                503,
                code="ROUTER_UNAVAILABLE",
            )

        # Parse request body
        try:
            if hasattr(request, "json"):
                body = await request.json()
            else:
                body = request.get("body", {})
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"Invalid JSON body in create binding: {e}")
            return error_response("Invalid JSON body", 400)

        # Validate required fields
        required = ["provider", "account_id", "peer_pattern", "agent_binding"]
        missing = [f for f in required if f not in body]
        if missing:
            return error_response(f"Missing required fields: {', '.join(missing)}", 400)

        # Create binding
        try:
            binding_type = BindingType(body.get("binding_type", "default"))
        except ValueError:
            return error_response(
                f"Invalid binding_type: {body.get('binding_type')}",
                400,
            )

        binding = MessageBinding(
            provider=body["provider"],
            account_id=body["account_id"],
            peer_pattern=body["peer_pattern"],
            agent_binding=body["agent_binding"],
            binding_type=binding_type,
            priority=body.get("priority", 0),
            time_window_start=body.get("time_window_start"),
            time_window_end=body.get("time_window_end"),
            allowed_users=set(body["allowed_users"]) if body.get("allowed_users") else None,
            blocked_users=set(body["blocked_users"]) if body.get("blocked_users") else None,
            config_overrides=body.get("config_overrides", {}),
            name=body.get("name"),
            description=body.get("description"),
            enabled=body.get("enabled", True),
        )

        router.add_binding(binding)

        logger.info(
            f"Created binding: {binding.name or binding.peer_pattern} "
            f"for {binding.provider}/{binding.account_id}"
        )

        return json_response(
            {
                "status": "created",
                "binding": binding.to_dict(),
            },
            status=201,
        )

    @require_permission("bindings.read")
    async def _resolve_binding(self, request: Any) -> HandlerResult:
        """Resolve which binding applies to a message."""
        router = self._get_router()
        if not router:
            return error_response(
                "Binding router not available",
                503,
                code="ROUTER_UNAVAILABLE",
            )

        # Parse request body
        try:
            if hasattr(request, "json"):
                body = await request.json()
            else:
                body = request.get("body", {})
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"Invalid JSON body in resolve binding: {e}")
            return error_response("Invalid JSON body", 400)

        # Validate required fields
        required = ["provider", "account_id", "peer_id"]
        missing = [f for f in required if f not in body]
        if missing:
            return error_response(f"Missing required fields: {', '.join(missing)}", 400)

        # Resolve binding
        resolution = router.resolve(
            provider=body["provider"],
            account_id=body["account_id"],
            peer_id=body["peer_id"],
            user_id=body.get("user_id"),
            hour=body.get("hour"),
        )

        return json_response(
            {
                "matched": resolution.matched,
                "agent_binding": resolution.agent_binding,
                "binding_type": resolution.binding_type.value if resolution.binding_type else None,
                "config_overrides": resolution.config_overrides,
                "match_reason": resolution.match_reason,
                "candidates_checked": resolution.candidates_checked,
                "binding": resolution.binding.to_dict() if resolution.binding else None,
            }
        )

    @require_permission("bindings.delete")
    async def _delete_binding(self, path: str, request: Any) -> HandlerResult:
        """Delete a message binding."""
        router = self._get_router()
        if not router:
            return error_response(
                "Binding router not available",
                503,
                code="ROUTER_UNAVAILABLE",
            )

        # Parse path: /api/bindings/:provider/:account/:pattern
        parts = path.split("/")
        if len(parts) < 6:
            return error_response(
                "Delete path must be /api/bindings/:provider/:account/:pattern",
                400,
            )

        provider = parts[3]
        account_id = parts[4]
        peer_pattern = "/".join(parts[5:])  # Pattern may contain /

        removed = router.remove_binding(provider, account_id, peer_pattern)
        if removed:
            logger.info(f"Removed binding: {provider}/{account_id}/{peer_pattern}")
            return json_response({"status": "deleted"})
        else:
            return error_response(
                f"Binding not found: {provider}/{account_id}/{peer_pattern}",
                404,
                code="BINDING_NOT_FOUND",
            )
