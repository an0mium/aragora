"""
Typed handler base classes with explicit type annotations.

This module provides handler base classes for improved type safety and IDE support:
- TypedHandler: Base handler with explicit HTTPRequestHandler typing
- AuthenticatedHandler: Requires authentication for all endpoints
- PermissionHandler: Fine-grained RBAC permission checking
- AdminHandler: Requires admin privileges
- AsyncTypedHandler: For async handler methods
- ResourceHandler: RESTful resource endpoint pattern

These classes extend BaseHandler with proper type annotations, better IDE autocomplete,
and consistent method signatures across all handlers.

Usage:
    from aragora.server.handlers.typed_handlers import (
        TypedHandler,
        AuthenticatedHandler,
        PermissionHandler,
        AdminHandler,
        AsyncTypedHandler,
        ResourceHandler,
    )

    class MyHandler(TypedHandler):
        def handle(self, path: str, query_params: dict, handler: HTTPRequestHandler):
            # IDE knows handler.headers, handler.path, etc.
            return self.json_response({"status": "ok"})
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, TypeAlias
from collections.abc import Awaitable, Callable

from aragora.billing.auth.context import UserAuthContext
from aragora.protocols import HTTPRequestHandler
from aragora.server.handlers.base import BaseHandler
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.handlers.utils.responses import HandlerResult, error_response

if TYPE_CHECKING:
    from aragora.server.handlers.base import ServerContext
    from aragora.server.storage import DebateStorage
    from aragora.users.store import UserStore

logger = logging.getLogger("aragora.server.handlers.base")

# Type alias for handlers that may be sync or async
MaybeAsyncHandlerResult: TypeAlias = HandlerResult | None | Awaitable[HandlerResult | None]


class TypedHandler(BaseHandler):
    """
    Typed base handler with explicit type annotations for all methods.

    This class provides:
    - Explicit HTTPRequestHandler type for handler parameters
    - Generic type support for response types
    - Better IDE autocomplete and type checking
    - Consistent method signatures across all handlers
    - Dependency injection support for testing

    Usage:
        class MyHandler(TypedHandler):
            def handle(
                self, path: str, query_params: dict, handler: HTTPRequestHandler
            ) -> HandlerResult | None:
                # IDE knows handler.headers, handler.path, etc.
                return self.json_response({"status": "ok"})

    Note: This is a drop-in replacement for BaseHandler that adds type safety.
    Existing handlers can be gradually migrated to use TypedHandler.
    """

    # Server context containing shared resources (narrowed from BaseHandler.ctx: dict[str, Any])
    ctx: ServerContext  # type: ignore[assignment]  # Subclass narrows dict to ServerContext TypedDict

    # Class-level dependency injection points for testing
    # These can be overridden in test fixtures
    _user_store_factory: Callable[[], Any] | None = None
    _storage_factory: Callable[[], Any] | None = None

    def __init__(self, server_context: ServerContext):
        """Initialize with server context.

        Args:
            server_context: ServerContext containing shared server resources
        """
        super().__init__(server_context)

    def read_json_body(
        self, handler: HTTPRequestHandler, max_size: int | None = None
    ) -> dict[str, Any] | None:
        """Read and parse JSON body from request.

        Args:
            handler: HTTP request handler
            max_size: Maximum body size in bytes (default: 1MB)

        Returns:
            Parsed JSON as dict, or None if parsing fails
        """
        import json

        try:
            content_length = int(handler.headers.get("Content-Length", "0"))
            if max_size and content_length > max_size:
                return None
            if content_length == 0:
                return None
            body = handler.rfile.read(content_length)
            return json.loads(body.decode("utf-8"))
        except (ValueError, json.JSONDecodeError, OSError):
            return None

    def error_response(self, message: str, status: int = 400) -> HandlerResult:
        """Create an error response.

        Args:
            message: Error message
            status: HTTP status code

        Returns:
            HandlerResult with error details
        """
        return error_response(message, status)

    def handle(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """
        Handle a GET request with proper typing.

        Args:
            path: The request path (e.g., "/api/v1/debates/123")
            query_params: Parsed query parameters as dict
            handler: HTTP request handler with typed access to headers, path, etc.

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None

    @require_permission("debates:write")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """Handle a POST request with proper typing."""
        return None

    @require_permission("debates:delete")
    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """Handle a DELETE request with proper typing."""
        return None

    @require_permission("debates:read")
    def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """Handle a PATCH request with proper typing."""
        return None

    @require_permission("debates:write")
    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """Handle a PUT request with proper typing."""
        return None

    # Type-safe dependency access

    def get_user_store(self) -> UserStore | None:
        """Get user store instance with type safety.

        Returns:
            UserStore instance if available, None otherwise
        """
        if self._user_store_factory:
            return self._user_store_factory()
        return self.ctx.get("user_store")

    def get_storage(self) -> DebateStorage | None:
        """Get debate storage instance."""
        if self._storage_factory:
            return self._storage_factory()
        return self.ctx.get("storage")

    @classmethod
    def with_dependencies(
        cls,
        server_context: ServerContext,
        user_store: UserStore | None = None,
        storage: DebateStorage | None = None,
    ) -> TypedHandler:
        """
        Factory method for creating handlers with injected dependencies.

        This is primarily useful for testing, allowing mock dependencies
        to be injected without modifying the server context.

        Args:
            server_context: Server context dict
            user_store: Optional user store to inject
            storage: Optional debate storage to inject

        Returns:
            Handler instance with injected dependencies

        Example:
            # In tests:
            mock_store = Mock(spec=UserStore)
            handler = MyHandler.with_dependencies(ctx, user_store=mock_store)
            result = handler.handle("/api/test", {}, mock_request)
        """
        instance = cls(server_context)
        if user_store:
            instance._user_store_factory = lambda: user_store
        if storage:
            instance._storage_factory = lambda: storage
        return instance

    # Helper methods from BaseHandler that TypedHandler also needs

    def require_auth_or_error(
        self, handler: HTTPRequestHandler
    ) -> tuple[UserAuthContext | None, HandlerResult | None]:
        """Require authentication and return user or error response.

        Args:
            handler: HTTP request handler with headers

        Returns:
            Tuple of (UserAuthContext, None) if authenticated,
            or (None, HandlerResult) with 401 error if not
        """
        user = self.get_current_user(handler)
        if user is None:
            return None, error_response("Authentication required", 401)
        return user, None

    def get_current_user(self, handler: HTTPRequestHandler) -> UserAuthContext | None:
        """Get authenticated user from request, if any.

        Args:
            handler: HTTP request handler with headers

        Returns:
            UserAuthContext if authenticated, None otherwise
        """
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = None
        if hasattr(handler, "user_store"):
            user_store = handler.user_store
        elif hasattr(self.__class__, "user_store"):
            user_store = self.__class__.user_store
        elif self._user_store_factory:
            user_store = self._user_store_factory()

        user_ctx = extract_user_from_request(handler, user_store)
        return user_ctx if user_ctx.is_authenticated else None

    def require_admin_or_error(
        self, handler: HTTPRequestHandler
    ) -> tuple[UserAuthContext | None, HandlerResult | None]:
        """Require admin authentication and return user or error response.

        Args:
            handler: HTTP request handler with headers

        Returns:
            Tuple of (UserAuthContext, None) if authenticated as admin,
            or (None, HandlerResult) with 401/403 error if not
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return None, err

        # Check for admin role or permission
        roles = getattr(user, "roles", []) or []
        permissions = getattr(user, "permissions", []) or []

        is_admin = "admin" in roles or "admin" in permissions or getattr(user, "is_admin", False)

        if not is_admin:
            return None, error_response("Admin access required", 403)

        return user, None

    def require_permission_or_error(
        self, handler: HTTPRequestHandler, permission: str
    ) -> tuple[UserAuthContext, None] | tuple[None, HandlerResult]:
        """Require authentication and specific permission.

        Args:
            handler: HTTP request handler with headers
            permission: Required permission string (e.g., "knowledge.read")

        Returns:
            Tuple of (UserAuthContext, None) if authenticated with permission,
            or (None, HandlerResult) with 401/403 error if not
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            return None, err

        # Check permission using role and permissions
        roles = getattr(user, "roles", []) or []
        permissions = getattr(user, "permissions", []) or []
        role = getattr(user, "role", None)

        # Admin role has all permissions
        if "admin" in roles or "admin" in permissions or role == "admin":
            return user, None

        # Owner role has all permissions
        if "owner" in roles or role == "owner":
            return user, None

        # Check specific permission
        if permission in permissions:
            return user, None

        # Check using PERMISSION_MATRIX from decorators
        try:
            from aragora.server.handlers.utils.decorators import has_permission

            if role and has_permission(role, permission):
                return user, None
        except ImportError:
            pass

        return None, error_response(f"Permission denied: requires '{permission}'", 403)


class AuthenticatedHandler(TypedHandler):
    """
    Handler base class that requires authentication for all endpoints.

    All handler methods automatically verify authentication before
    proceeding. Use this for handlers where every endpoint requires
    a logged-in user.

    The authenticated user context is available via self.current_user
    after calling _ensure_authenticated().

    Usage:
        class UserSettingsHandler(AuthenticatedHandler):
            def handle(self, path, query_params, handler):
                user, err = self._ensure_authenticated(handler)
                if err:
                    return err
                # user is guaranteed to be authenticated here
                return self.json_response({"user_id": user.user_id})
    """

    _current_user: UserAuthContext | None = None

    @property
    def current_user(self) -> UserAuthContext | None:
        """Get the current authenticated user context.

        Returns None if _ensure_authenticated() hasn't been called yet
        or if authentication failed.
        """
        return self._current_user

    def _ensure_authenticated(
        self, handler: HTTPRequestHandler
    ) -> tuple[UserAuthContext, None] | tuple[None, HandlerResult]:
        """
        Ensure the request is authenticated.

        This method verifies authentication and caches the result in
        self._current_user for subsequent access.

        Args:
            handler: HTTP request handler

        Returns:
            Tuple of (UserAuthContext, None) if authenticated,
            or (None, error_response) if not authenticated

        Example:
            def handle(self, path, query_params, handler):
                user, err = self._ensure_authenticated(handler)
                if err:
                    return err
                # Use user.user_id, user.email, etc.
        """
        user, err = self.require_auth_or_error(handler)
        if err:
            self._current_user = None
            return None, err
        self._current_user = user
        return user, None

    def _ensure_admin(
        self, handler: HTTPRequestHandler
    ) -> tuple[UserAuthContext, None] | tuple[None, HandlerResult]:
        """
        Ensure the request is authenticated with admin privileges.

        Args:
            handler: HTTP request handler

        Returns:
            Tuple of (UserAuthContext, None) if authenticated as admin,
            or (None, error_response) if not authenticated or not admin
        """
        user, err = self.require_admin_or_error(handler)
        if err:
            self._current_user = None
            return None, err
        self._current_user = user
        return user, None


class PermissionHandler(AuthenticatedHandler):
    """
    Handler base class with fine-grained permission checking.

    Extends AuthenticatedHandler with RBAC permission enforcement.
    Use this for handlers that need to check specific permissions
    beyond simple authentication.

    Class Attributes:
        REQUIRED_PERMISSIONS: Dict mapping HTTP methods to required permissions.
                             Override in subclasses to define permissions.

    Usage:
        class KnowledgeHandler(PermissionHandler):
            REQUIRED_PERMISSIONS = {
                "GET": "knowledge:read",
                "POST": "knowledge:write",
                "DELETE": "knowledge:delete",
            }

            def handle(self, path, query_params, handler):
                user, err = self._ensure_permission(handler, "GET")
                if err:
                    return err
                # User has knowledge:read permission
                return self.json_response({"data": "..."})
    """

    # Override in subclasses to define method -> permission mapping
    REQUIRED_PERMISSIONS: dict[str, str | None] = {
        "GET": None,  # None means no permission required (just auth)
        "POST": None,
        "PUT": None,
        "PATCH": None,
        "DELETE": None,
    }

    def _ensure_permission(
        self, handler: HTTPRequestHandler, method: str | None = None
    ) -> tuple[UserAuthContext, None] | tuple[None, HandlerResult]:
        """
        Ensure the request has the required permission for the given method.

        Args:
            handler: HTTP request handler
            method: HTTP method (GET, POST, etc.). If None, extracted from handler.

        Returns:
            Tuple of (UserAuthContext, None) if permission granted,
            or (None, error_response) if permission denied
        """
        # First ensure authentication
        user, err = self._ensure_authenticated(handler)
        if err:
            return None, err

        # Determine the required permission
        if method is None:
            method = getattr(handler, "command", "GET")

        permission = self.REQUIRED_PERMISSIONS.get(method)
        if permission is None:
            # No specific permission required for this method
            return user, None

        # Check the permission
        user_with_perm, perm_err = self.require_permission_or_error(handler, permission)
        if perm_err:
            return None, perm_err

        return user_with_perm, None

    def _check_custom_permission(
        self, handler: HTTPRequestHandler, permission: str
    ) -> tuple[UserAuthContext, None] | tuple[None, HandlerResult]:
        """
        Check a specific custom permission (not from REQUIRED_PERMISSIONS).

        Use this when the permission depends on the specific endpoint
        or resource being accessed, not just the HTTP method.

        Args:
            handler: HTTP request handler
            permission: Permission string to check (e.g., "debates:fork")

        Returns:
            Tuple of (UserAuthContext, None) if permission granted,
            or (None, error_response) if permission denied
        """
        # First ensure authentication
        user, err = self._ensure_authenticated(handler)
        if err:
            return None, err

        # Check the specific permission
        return self.require_permission_or_error(handler, permission)


class AdminHandler(AuthenticatedHandler):
    """
    Handler base class that requires admin privileges for all endpoints.

    All handler methods automatically verify the user has admin role
    before proceeding. Use this for administrative endpoints.

    Usage:
        class SystemConfigHandler(AdminHandler):
            def handle(self, path, query_params, handler):
                user, err = self._ensure_admin(handler)
                if err:
                    return err
                # User is guaranteed to be admin here
                return self.json_response({"config": system_config})
    """

    # Admin handlers should always audit sensitive actions
    AUDIT_ACTIONS: bool = True

    def _log_admin_action(
        self,
        action: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Log an admin action for audit trail.

        Args:
            action: Action performed (e.g., "update_config", "delete_user")
            resource_id: ID of the affected resource
            details: Additional details to log
        """
        if not self.AUDIT_ACTIONS:
            return

        user = self.current_user
        user_id = user.user_id if user else "unknown"

        logger.info(
            f"Admin action: user={user_id} action={action} resource={resource_id} details={details}"
        )

        # Try to use unified audit logging if available
        try:
            from aragora.audit.unified import audit_admin

            if audit_admin:
                audit_admin(
                    admin_id=user_id,
                    action=action,
                    resource_id=resource_id,
                    details=details or {},
                )
        except ImportError:
            pass  # Audit module not available


class AsyncTypedHandler(TypedHandler):
    """
    Typed handler base class for async handler methods.

    Use this when your handler methods need to be async (e.g., for
    database operations, external API calls, or other I/O operations).

    All handle methods are defined as async and should be awaited.

    Usage:
        class AsyncDataHandler(AsyncTypedHandler):
            async def handle(self, path, query_params, handler):
                data = await self._fetch_data_async()
                return self.json_response({"data": data})
    """

    async def handle(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle a GET request asynchronously."""
        return None

    @require_permission("debates:write")
    async def handle_post(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle a POST request asynchronously."""
        return None

    @require_permission("debates:delete")
    async def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle a DELETE request asynchronously."""
        return None

    @require_permission("debates:read")
    async def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle a PATCH request asynchronously."""
        return None

    @require_permission("debates:write")
    async def handle_put(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult | None:
        """Handle a PUT request asynchronously."""
        return None


class ResourceHandler(PermissionHandler):
    """
    Handler base class for RESTful resource endpoints.

    Provides a structured approach to handling CRUD operations on a
    single resource type. Subclasses define the resource name and
    implement the standard operations.

    Class Attributes:
        RESOURCE_NAME: Name of the resource (e.g., "debate", "user")
        RESOURCE_ID_PARAM: URL parameter name for resource ID (default: "id")
        ROUTES: List of route patterns handled by this handler

    Automatically generates REQUIRED_PERMISSIONS from RESOURCE_NAME:
        - GET -> {resource}:read
        - POST -> {resource}:create
        - PUT/PATCH -> {resource}:update
        - DELETE -> {resource}:delete

    Usage:
        class DocumentHandler(ResourceHandler):
            RESOURCE_NAME = "document"
            ROUTES = ["/api/v1/documents", "/api/v1/documents/*"]

            def _get_resource(self, resource_id: str, handler) -> HandlerResult:
                doc = self.storage.get_document(resource_id)
                return self.json_response(doc)

            def _create_resource(self, handler) -> HandlerResult:
                body = self.read_json_body(handler)
                doc = self.storage.create_document(body)
                return self.json_response(doc, status=201)
    """

    RESOURCE_NAME: str = "resource"
    RESOURCE_ID_PARAM: str = "id"
    ROUTES: list[str] = []

    @classmethod
    def _get_resource_permissions(cls) -> dict[str, str | None]:
        """Generate permission mapping from resource name."""
        name = cls.RESOURCE_NAME
        return {
            "GET": f"{name}:read",
            "POST": f"{name}:create",
            "PUT": f"{name}:update",
            "PATCH": f"{name}:update",
            "DELETE": f"{name}:delete",
        }

    def __init__(self, server_context: ServerContext):
        super().__init__(server_context)
        # Override REQUIRED_PERMISSIONS with resource-specific ones
        self.__class__.REQUIRED_PERMISSIONS = self._get_resource_permissions()

    def handle(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """Route GET requests to list or get operations."""
        user, err = self._ensure_permission(handler, "GET")
        if err:
            return err

        resource_id = self._extract_resource_id(path)
        if resource_id:
            return self._get_resource(resource_id, handler)
        return self._list_resources(query_params, handler)

    @require_permission("debates:write")
    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """Route POST requests to create operation."""
        user, err = self._ensure_permission(handler, "POST")
        if err:
            return err
        return self._create_resource(handler)

    @require_permission("debates:write")
    def handle_put(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """Route PUT requests to update operation."""
        user, err = self._ensure_permission(handler, "PUT")
        if err:
            return err

        resource_id = self._extract_resource_id(path)
        if not resource_id:
            return error_response(f"{self.RESOURCE_NAME} ID required", 400)
        return self._update_resource(resource_id, handler)

    @require_permission("debates:read")
    def handle_patch(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """Route PATCH requests to partial update operation."""
        user, err = self._ensure_permission(handler, "PATCH")
        if err:
            return err

        resource_id = self._extract_resource_id(path)
        if not resource_id:
            return error_response(f"{self.RESOURCE_NAME} ID required", 400)
        return self._patch_resource(resource_id, handler)

    @require_permission("debates:delete")
    def handle_delete(
        self, path: str, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> MaybeAsyncHandlerResult:
        """Route DELETE requests to delete operation."""
        user, err = self._ensure_permission(handler, "DELETE")
        if err:
            return err

        resource_id = self._extract_resource_id(path)
        if not resource_id:
            return error_response(f"{self.RESOURCE_NAME} ID required", 400)
        return self._delete_resource(resource_id, handler)

    def _extract_resource_id(self, path: str) -> str | None:
        """Extract resource ID from path.

        Override this method if your URL structure is different.
        Default assumes /{resource}s/{id} pattern.
        """
        parts = path.rstrip("/").split("/")
        # Check if last segment looks like an ID (not the resource name plural)
        if len(parts) >= 2:
            last = parts[-1]
            # Skip if it's the resource collection endpoint
            if last == f"{self.RESOURCE_NAME}s" or last == self.RESOURCE_NAME:
                return None
            # Return the ID
            if last and last not in ("", "list", "search"):
                return last
        return None

    # Override these methods in subclasses

    def _list_resources(
        self, query_params: dict[str, Any], handler: HTTPRequestHandler
    ) -> HandlerResult:
        """List resources. Override in subclass."""
        return error_response("Not implemented", 501)

    def _get_resource(self, resource_id: str, handler: HTTPRequestHandler) -> HandlerResult:
        """Get a single resource. Override in subclass."""
        return error_response("Not implemented", 501)

    def _create_resource(self, handler: HTTPRequestHandler) -> HandlerResult:
        """Create a new resource. Override in subclass."""
        return error_response("Not implemented", 501)

    def _update_resource(self, resource_id: str, handler: HTTPRequestHandler) -> HandlerResult:
        """Update a resource (full replacement). Override in subclass."""
        return error_response("Not implemented", 501)

    def _patch_resource(self, resource_id: str, handler: HTTPRequestHandler) -> HandlerResult:
        """Partially update a resource. Override in subclass."""
        # Default to full update if not overridden
        return self._update_resource(resource_id, handler)

    def _delete_resource(self, resource_id: str, handler: HTTPRequestHandler) -> HandlerResult:
        """Delete a resource. Override in subclass."""
        return error_response("Not implemented", 501)


__all__ = [
    "TypedHandler",
    "AuthenticatedHandler",
    "PermissionHandler",
    "AdminHandler",
    "AsyncTypedHandler",
    "ResourceHandler",
    "MaybeAsyncHandlerResult",
]
