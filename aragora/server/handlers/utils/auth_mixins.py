"""
Authentication Mixins for consistent RBAC handling across handlers.

This module provides mixins that simplify authentication and permission
checking patterns, eliminating repetitive try-except blocks.

Usage:
    class MyHandler(SecureEndpointMixin, SecureHandler):
        async def handle_get(self, request):
            auth, err = await self.require_permission_or_error(
                request, "resource:read"
            )
            if err:
                return err

            # Proceed with authenticated request
            user_id = auth.user_id
            ...
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple, Type

if TYPE_CHECKING:
    from aragora.rbac.models import AuthorizationContext
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)

# Exception types with proper typing for fallback
_ForbiddenError: Type[Exception]
_UnauthorizedError: Type[Exception]

# Pre-declare module-level names for type checking
ForbiddenError: Type[Exception]  # noqa: N816
UnauthorizedError: Type[Exception]  # noqa: N816

try:
    from aragora.server.handlers.utils.auth import ForbiddenError, UnauthorizedError

    _ForbiddenError = ForbiddenError
    _UnauthorizedError = UnauthorizedError
    _AUTH_EXCEPTIONS_AVAILABLE = True
except ImportError:
    _AUTH_EXCEPTIONS_AVAILABLE = False
    _ForbiddenError = Exception
    _UnauthorizedError = Exception
    # Create module-level aliases for decorator compatibility
    ForbiddenError = Exception  # noqa: N816
    UnauthorizedError = Exception  # noqa: N816

# Track if auth module is available
_AUTH_AVAILABLE = _AUTH_EXCEPTIONS_AVAILABLE


def _get_error_response() -> Callable[[str, int], "HandlerResult"]:
    """Lazy import to avoid circular dependency with base module."""
    from aragora.server.handlers.base import error_response

    return error_response


def error_response(message: str, status: int) -> "HandlerResult":
    """Wrapper for lazy-loaded error_response to avoid repeated imports."""
    return _get_error_response()(message, status)


class SecureEndpointMixin:
    """
    Mixin providing simplified authentication and permission checking.

    This mixin is designed to work with SecureHandler and provides
    streamlined methods for common auth patterns.

    Methods:
        require_auth_or_error: Check authentication only
        require_permission_or_error: Check auth and specific permission
        require_any_permission_or_error: Check auth and any of multiple permissions
        require_all_permissions_or_error: Check auth and all permissions

    Note:
        This mixin expects to be combined with a class implementing
        get_auth_context and check_permission methods (e.g., SecureHandler).
    """

    # Type hints for methods provided by SecureHandler (mixin pattern)
    get_auth_context: Any  # Provided by SecureHandler
    check_permission: Any  # Provided by SecureHandler

    async def require_auth_or_error(
        self,
        request: Any,
    ) -> Tuple[Optional["AuthorizationContext"], Optional["HandlerResult"]]:
        """
        Require authentication, returning context or error response.

        Args:
            request: The request/handler object

        Returns:
            Tuple of (AuthorizationContext, None) on success,
            or (None, HandlerResult) on failure
        """
        if not _AUTH_AVAILABLE:
            logger.error("Auth module not available")
            return None, _get_error_response()("Auth not configured", 500)

        try:
            auth_context = await self.get_auth_context(request, require_auth=True)
            return auth_context, None
        except _UnauthorizedError:
            return None, _get_error_response()("Authentication required", 401)
        except Exception as e:
            logger.exception(f"Unexpected auth error: {e}")
            return None, _get_error_response()("Authentication failed", 500)

    async def require_permission_or_error(
        self,
        request: Any,
        permission: str,
        resource_id: Optional[str] = None,
    ) -> Tuple[Optional["AuthorizationContext"], Optional["HandlerResult"]]:
        """
        Require authentication and a specific permission.

        Args:
            request: The request/handler object
            permission: Permission to check (e.g., "resource:read")
            resource_id: Optional resource ID for resource-level checks

        Returns:
            Tuple of (AuthorizationContext, None) on success,
            or (None, HandlerResult) on failure
        """
        if not _AUTH_AVAILABLE:
            logger.error("Auth module not available")
            return None, _get_error_response()("Auth not configured", 500)

        try:
            auth_context = await self.get_auth_context(request, require_auth=True)
            self.check_permission(auth_context, permission, resource_id)
            return auth_context, None
        except _UnauthorizedError:
            return None, _get_error_response()("Authentication required", 401)
        except _ForbiddenError as e:
            return None, _get_error_response()(str(e), 403)
        except Exception as e:
            logger.exception(f"Unexpected auth error: {e}")
            return None, _get_error_response()("Authorization failed", 500)

    async def require_any_permission_or_error(
        self,
        request: Any,
        permissions: list[str],
        resource_id: Optional[str] = None,
    ) -> Tuple[Optional["AuthorizationContext"], Optional["HandlerResult"]]:
        """
        Require authentication and ANY of the specified permissions.

        Args:
            request: The request/handler object
            permissions: List of permissions (user needs at least one)
            resource_id: Optional resource ID for resource-level checks

        Returns:
            Tuple of (AuthorizationContext, None) on success,
            or (None, HandlerResult) on failure
        """
        if not _AUTH_AVAILABLE:
            logger.error("Auth module not available")
            return None, _get_error_response()("Auth not configured", 500)

        if not permissions:
            return None, _get_error_response()("No permissions specified", 400)

        try:
            auth_context = await self.get_auth_context(request, require_auth=True)

            # Check each permission, succeed if any passes
            errors = []
            for perm in permissions:
                try:
                    self.check_permission(auth_context, perm, resource_id)
                    return auth_context, None  # Success on first match
                except _ForbiddenError as e:
                    errors.append(str(e))

            # All permissions denied
            return None, _get_error_response()(
                f"Permission denied: requires one of {permissions}", 403
            )

        except _UnauthorizedError:
            return None, _get_error_response()("Authentication required", 401)
        except Exception as e:
            logger.exception(f"Unexpected auth error: {e}")
            return None, _get_error_response()("Authorization failed", 500)

    async def require_all_permissions_or_error(
        self,
        request: Any,
        permissions: list[str],
        resource_id: Optional[str] = None,
    ) -> Tuple[Optional["AuthorizationContext"], Optional["HandlerResult"]]:
        """
        Require authentication and ALL of the specified permissions.

        Args:
            request: The request/handler object
            permissions: List of permissions (user needs all of them)
            resource_id: Optional resource ID for resource-level checks

        Returns:
            Tuple of (AuthorizationContext, None) on success,
            or (None, HandlerResult) on failure
        """
        if not _AUTH_AVAILABLE:
            logger.error("Auth module not available")
            return None, _get_error_response()("Auth not configured", 500)

        if not permissions:
            return None, _get_error_response()("No permissions specified", 400)

        try:
            auth_context = await self.get_auth_context(request, require_auth=True)

            # Check all permissions
            for perm in permissions:
                self.check_permission(auth_context, perm, resource_id)

            return auth_context, None

        except _UnauthorizedError:
            return None, _get_error_response()("Authentication required", 401)
        except _ForbiddenError as e:
            return None, _get_error_response()(str(e), 403)
        except Exception as e:
            logger.exception(f"Unexpected auth error: {e}")
            return None, _get_error_response()("Authorization failed", 500)

    async def require_admin_or_error(
        self,
        request: Any,
    ) -> Tuple[Optional["AuthorizationContext"], Optional["HandlerResult"]]:
        """
        Require authentication and admin role.

        Args:
            request: The request/handler object

        Returns:
            Tuple of (AuthorizationContext, None) on success,
            or (None, HandlerResult) on failure
        """
        return await self.require_permission_or_error(request, "admin:*")


class AuthenticatedHandlerMixin:
    """
    Mixin that provides authentication context management for handlers.

    This is a simpler mixin for handlers that just need to track
    the current user without complex permission checking.

    Attributes:
        current_user: The authenticated user context (set after auth check)
    """

    _current_auth: Optional["AuthorizationContext"] = None

    @property
    def current_user(self) -> Optional["AuthorizationContext"]:
        """Get the current authenticated user context."""
        return self._current_auth

    @property
    def user_id(self) -> Optional[str]:
        """Get the current user's ID, or None if not authenticated."""
        if self._current_auth:
            return self._current_auth.user_id
        return None

    @property
    def org_id(self) -> Optional[str]:
        """Get the current user's organization ID, or None."""
        if self._current_auth:
            return getattr(self._current_auth, "org_id", None)
        return None

    def set_auth_context(self, auth: "AuthorizationContext") -> None:
        """Set the authentication context for this request."""
        self._current_auth = auth

    def clear_auth_context(self) -> None:
        """Clear the authentication context."""
        self._current_auth = None


# Convenience decorators for permission checking
def require_permission(permission: str, handler_arg: int = 0):
    """
    Decorator that requires a specific permission.

    Args:
        permission: The permission string required (e.g., "resource:read")
        handler_arg: Index (0-based) of the argument containing the HTTP handler.
            For handlers with signature (self, request) use handler_arg=0 (default).
            For handlers with signature (self, path, query_params, handler) use handler_arg=2.

    Usage:
        @require_permission("resource:read")
        async def my_handler(self, request):
            # Only reaches here if authenticated with permission
            pass

        @require_permission("nomic:read", handler_arg=2)
        async def handle(self, path, query_params, handler):
            # Handler is the third argument (index 2)
            pass
    """

    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Lazy import to avoid circular dependency
            from aragora.server.handlers.base import error_response

            # Extract the handler from the specified argument position
            if handler_arg < len(args):
                request = args[handler_arg]
            else:
                # Fallback: try to get from kwargs or first positional arg
                request = (
                    kwargs.get("handler") or kwargs.get("request") or (args[0] if args else None)
                )

            # Check auth using SecureHandler's get_auth_context if available
            if hasattr(self, "get_auth_context") and hasattr(self, "check_permission"):
                try:
                    auth_context = await self.get_auth_context(request, require_auth=True)
                    self.check_permission(auth_context, permission)
                    # Store auth context if handler supports it
                    if hasattr(self, "set_auth_context"):
                        self.set_auth_context(auth_context)
                except _UnauthorizedError:
                    return error_response("Authentication required", 401)
                except _ForbiddenError as e:
                    return error_response(str(e), 403)
                except Exception as e:
                    logger.exception(f"Auth error in @require_permission: {e}")
                    return error_response("Authorization failed", 500)

            return await func(self, *args, **kwargs)

        return wrapper

    return decorator


def require_any_permission(*permissions: str):
    """
    Decorator that requires any of the specified permissions.

    Usage:
        @require_any_permission("resource:read", "resource:admin")
        async def my_handler(self, request):
            pass
    """

    def decorator(func):
        async def wrapper(self, request, *args, **kwargs):
            if hasattr(self, "require_any_permission_or_error"):
                auth, err = await self.require_any_permission_or_error(request, list(permissions))
                if err:
                    return err
                if hasattr(self, "set_auth_context"):
                    self.set_auth_context(auth)
            return await func(self, request, *args, **kwargs)

        return wrapper

    return decorator


def require_all_permissions(*permissions: str):
    """
    Decorator that requires all of the specified permissions.

    Usage:
        @require_all_permissions("resource:read", "resource:write")
        async def my_handler(self, request):
            pass
    """

    def decorator(func):
        async def wrapper(self, request, *args, **kwargs):
            if hasattr(self, "require_all_permissions_or_error"):
                auth, err = await self.require_all_permissions_or_error(request, list(permissions))
                if err:
                    return err
                if hasattr(self, "set_auth_context"):
                    self.set_auth_context(auth)
            return await func(self, request, *args, **kwargs)

        return wrapper

    return decorator
