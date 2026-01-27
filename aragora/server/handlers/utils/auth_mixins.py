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
from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from aragora.rbac.models import AuthorizationContext

logger = logging.getLogger(__name__)

# Import error types and response builders
try:
    from ..base import HandlerResult, error_response
    from .auth import ForbiddenError, UnauthorizedError

    _AUTH_AVAILABLE = True
except ImportError:
    _AUTH_AVAILABLE = False
    HandlerResult = Any  # type: ignore
    error_response = None  # type: ignore
    ForbiddenError = Exception  # type: ignore
    UnauthorizedError = Exception  # type: ignore


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
    """

    async def require_auth_or_error(
        self,
        request: Any,
    ) -> Tuple[Optional["AuthorizationContext"], Optional[HandlerResult]]:
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
            return None, error_response("Auth not configured", 500)  # type: ignore

        try:
            auth_context = await self.get_auth_context(request, require_auth=True)  # type: ignore
            return auth_context, None
        except UnauthorizedError:
            return None, error_response("Authentication required", 401)  # type: ignore
        except Exception as e:
            logger.exception(f"Unexpected auth error: {e}")
            return None, error_response("Authentication failed", 500)  # type: ignore

    async def require_permission_or_error(
        self,
        request: Any,
        permission: str,
        resource_id: Optional[str] = None,
    ) -> Tuple[Optional["AuthorizationContext"], Optional[HandlerResult]]:
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
            return None, error_response("Auth not configured", 500)  # type: ignore

        try:
            auth_context = await self.get_auth_context(request, require_auth=True)  # type: ignore
            self.check_permission(auth_context, permission, resource_id)  # type: ignore
            return auth_context, None
        except UnauthorizedError:
            return None, error_response("Authentication required", 401)  # type: ignore
        except ForbiddenError as e:
            return None, error_response(str(e), 403)  # type: ignore
        except Exception as e:
            logger.exception(f"Unexpected auth error: {e}")
            return None, error_response("Authorization failed", 500)  # type: ignore

    async def require_any_permission_or_error(
        self,
        request: Any,
        permissions: list[str],
        resource_id: Optional[str] = None,
    ) -> Tuple[Optional["AuthorizationContext"], Optional[HandlerResult]]:
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
            return None, error_response("Auth not configured", 500)  # type: ignore

        if not permissions:
            return None, error_response("No permissions specified", 400)  # type: ignore

        try:
            auth_context = await self.get_auth_context(request, require_auth=True)  # type: ignore

            # Check each permission, succeed if any passes
            errors = []
            for perm in permissions:
                try:
                    self.check_permission(auth_context, perm, resource_id)  # type: ignore
                    return auth_context, None  # Success on first match
                except ForbiddenError as e:
                    errors.append(str(e))

            # All permissions denied
            return None, error_response(f"Permission denied: requires one of {permissions}", 403)  # type: ignore

        except UnauthorizedError:
            return None, error_response("Authentication required", 401)  # type: ignore
        except Exception as e:
            logger.exception(f"Unexpected auth error: {e}")
            return None, error_response("Authorization failed", 500)  # type: ignore

    async def require_all_permissions_or_error(
        self,
        request: Any,
        permissions: list[str],
        resource_id: Optional[str] = None,
    ) -> Tuple[Optional["AuthorizationContext"], Optional[HandlerResult]]:
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
            return None, error_response("Auth not configured", 500)  # type: ignore

        if not permissions:
            return None, error_response("No permissions specified", 400)  # type: ignore

        try:
            auth_context = await self.get_auth_context(request, require_auth=True)  # type: ignore

            # Check all permissions
            for perm in permissions:
                self.check_permission(auth_context, perm, resource_id)  # type: ignore

            return auth_context, None

        except UnauthorizedError:
            return None, error_response("Authentication required", 401)  # type: ignore
        except ForbiddenError as e:
            return None, error_response(str(e), 403)  # type: ignore
        except Exception as e:
            logger.exception(f"Unexpected auth error: {e}")
            return None, error_response("Authorization failed", 500)  # type: ignore

    async def require_admin_or_error(
        self,
        request: Any,
    ) -> Tuple[Optional["AuthorizationContext"], Optional[HandlerResult]]:
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
def require_permission(permission: str):
    """
    Decorator that requires a specific permission.

    Usage:
        @require_permission("resource:read")
        async def my_handler(self, request):
            # Only reaches here if authenticated with permission
            pass
    """

    def decorator(func):
        async def wrapper(self, request, *args, **kwargs):
            if hasattr(self, "require_permission_or_error"):
                auth, err = await self.require_permission_or_error(request, permission)
                if err:
                    return err
                # Store auth context if handler supports it
                if hasattr(self, "set_auth_context"):
                    self.set_auth_context(auth)
            return await func(self, request, *args, **kwargs)

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
