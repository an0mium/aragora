"""
Unified Authentication Utilities for Handlers.

Provides secure authentication extraction from requests,
replacing header-trust patterns with JWT verification.

Usage:
    from aragora.server.handlers.utils.auth import (
        get_auth_context,
        require_authenticated,
        UnauthorizedError,
    )

    # In a handler method:
    auth_ctx = await get_auth_context(request)
    if not auth_ctx.is_authenticated:
        return error_response("Not authenticated", 401)
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec

from aragora.rbac.models import AuthorizationContext

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class UnauthorizedError(Exception):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message)
        self.message = message


class ForbiddenError(Exception):
    """Raised when authorization fails."""

    def __init__(self, message: str = "Access denied", permission: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.permission = permission


async def get_auth_context(
    request: Any,
    require_auth: bool = False,
) -> AuthorizationContext:
    """
    Extract and verify authentication context from request.

    This function replaces the pattern of trusting X-User-ID headers
    with proper JWT verification.

    Args:
        request: HTTP request object (aiohttp request or similar)
        require_auth: If True, raises UnauthorizedError when not authenticated

    Returns:
        AuthorizationContext with user information

    Raises:
        UnauthorizedError: If require_auth=True and authentication fails
    """
    from aragora.billing.jwt_auth import extract_user_from_request, UserAuthContext

    try:
        # Try to get user store from request app context
        user_store = None
        if hasattr(request, "app") and hasattr(request.app, "get"):
            user_store = request.app.get("user_store")

        # Extract user from JWT token
        user_ctx: UserAuthContext = extract_user_from_request(request, user_store)

        if not user_ctx.is_authenticated:
            if require_auth:
                raise UnauthorizedError("Valid authentication token required")
            # Return anonymous context
            return AuthorizationContext(
                user_id="anonymous",
                org_id=None,
                workspace_id=None,
                roles=set(),
                permissions=set(),
            )

        # Build authorization context from authenticated user
        return AuthorizationContext(
            user_id=user_ctx.user_id,
            user_email=user_ctx.email,
            org_id=user_ctx.org_id,
            workspace_id=_extract_workspace_id(request),
            roles={user_ctx.role} if user_ctx.role else {"member"},
            permissions=_get_user_permissions(user_ctx),
        )

    except UnauthorizedError:
        raise
    except Exception as e:
        logger.warning(f"Error extracting auth context: {e}")
        if require_auth:
            raise UnauthorizedError("Authentication failed")
        return AuthorizationContext(
            user_id="anonymous",
            org_id=None,
            workspace_id=None,
            roles=set(),
            permissions=set(),
        )


def _extract_workspace_id(request: Any) -> Optional[str]:
    """Extract workspace ID from request headers."""
    if hasattr(request, "headers"):
        return request.headers.get("X-Workspace-ID")
    return None


def _get_user_permissions(user_ctx: Any) -> set[str]:
    """Get permissions for a user based on their roles."""
    from aragora.rbac.checker import get_permission_checker

    try:
        checker = get_permission_checker()
        # Get all permissions for user's roles
        permissions = set()
        roles = {user_ctx.role} if user_ctx.role else {"member"}
        for role in roles:
            role_perms = checker.get_role_permissions(role)
            permissions.update(role_perms)
        return permissions
    except Exception as e:
        logger.warning(f"Error getting user permissions: {e}")
        return set()


def require_authenticated(
    func: Optional[Callable[P, T]] = None,
    *,
    on_failure: Optional[Callable[[Exception], Any]] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to require authentication for a handler method.

    Can be used with or without arguments:

        @require_authenticated
        async def my_handler(self, request):
            ...

        @require_authenticated(on_failure=custom_handler)
        async def my_handler(self, request):
            ...

    Args:
        func: The function to wrap (when used without parentheses)
        on_failure: Optional callback when authentication fails

    Returns:
        Decorated function
    """

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        @wraps(fn)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Find request in args/kwargs
            request = _find_request(args, kwargs)
            if request is None:
                raise UnauthorizedError("No request found for authentication")

            try:
                auth_ctx = await get_auth_context(request, require_auth=True)
                # Add auth context to kwargs for use by the handler
                kwargs["auth_context"] = auth_ctx
                return await fn(*args, **kwargs)  # type: ignore
            except UnauthorizedError as e:
                if on_failure:
                    return on_failure(e)
                raise

        @wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # For sync functions, we can't do async auth
            # This is a fallback - prefer async handlers
            raise NotImplementedError(
                "require_authenticated requires async handlers. "
                "Use 'async def' for your handler method."
            )

        import asyncio

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    if func is not None:
        # Called without arguments: @require_authenticated
        return decorator(func)  # type: ignore[return-value]
    # Called with arguments: @require_authenticated(...)
    return decorator


def _find_request(args: tuple, kwargs: dict) -> Optional[Any]:
    """Find the request object in function arguments."""
    # Check kwargs first
    if "request" in kwargs:
        return kwargs["request"]

    # Check positional args - typically request is first or second arg
    # (first if standalone function, second if method with self)
    for arg in args:
        if hasattr(arg, "headers") and hasattr(arg, "method"):
            return arg

    return None


def get_user_from_handler(handler: Any) -> tuple[str, str]:
    """
    Extract user ID and name from handler request.

    This is a compatibility function for existing handlers that
    use the old pattern. New code should use get_auth_context().

    Args:
        handler: HTTP request handler with headers

    Returns:
        Tuple of (user_id, user_name)
    """
    import asyncio

    # Try to get auth context asynchronously
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Can't await in sync context with running loop
            # Fall back to header extraction with warning
            logger.warning(
                "get_user_from_handler called in async context - "
                "consider using await get_auth_context() instead"
            )
            return _extract_user_from_headers(handler)
    except RuntimeError:
        pass

    # Sync fallback - extract from headers with validation
    return _extract_user_from_headers(handler)


def _extract_user_from_headers(handler: Any) -> tuple[str, str]:
    """
    Extract user from JWT token only.

    SECURITY: Header-based authentication fallback has been removed to prevent
    identity spoofing and privilege escalation attacks. Only JWT-verified
    identities are returned.

    Args:
        handler: Request handler with headers

    Returns:
        Tuple of (user_id, user_name) from JWT, or ("anonymous", "Anonymous User")
        if no valid JWT token is present.
    """
    from aragora.billing.jwt_auth import extract_user_from_request

    try:
        # Extract user from JWT token - this is the ONLY trusted source
        user_ctx = extract_user_from_request(handler, None)
        if user_ctx.is_authenticated:
            return user_ctx.user_id, user_ctx.email or user_ctx.user_id
    except Exception as e:
        logger.debug(f"JWT extraction failed: {e}")

    # SECURITY: Do NOT fall back to X-User-ID headers - they can be spoofed
    # Return anonymous identity instead
    logger.debug(
        "_extract_user_from_headers: No valid JWT token. "
        "Returning anonymous. X-User-ID headers are NOT trusted."
    )
    return "anonymous", "Anonymous User"
