"""
Authentication Middleware.

Provides decorators and utilities for authenticating API requests.

Usage:
    from aragora.server.middleware import require_auth, optional_auth

    @require_auth
    def sensitive_endpoint(self, handler):
        # Only executed if authenticated
        ...

    @optional_auth
    def public_endpoint(self, handler, auth_context):
        if auth_context.authenticated:
            # Show personalized data
        else:
            # Show public data
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)


@dataclass
class AuthContext:
    """
    Authentication context passed to handlers.

    Contains information about the authenticated user/client.
    """

    authenticated: bool = False
    token: Optional[str] = None
    client_ip: Optional[str] = None
    user_id: Optional[str] = None  # Future: user identification

    @property
    def is_authenticated(self) -> bool:
        """Alias for authenticated."""
        return self.authenticated


# Import utilities from auth_v2 to avoid duplication
from .auth_v2 import extract_client_ip, extract_token


def validate_token(token: str) -> bool:
    """
    Validate an authentication token.

    Checks both token validity and revocation status.

    Args:
        token: Token to validate.

    Returns:
        True if valid and not revoked, False otherwise.
    """
    from aragora.server.auth import auth_config
    from aragora.server.middleware.token_revocation import is_token_revoked

    if not token:
        return False

    # Check if token is valid
    if not auth_config.validate_token(token):
        return False

    # Check if token has been revoked
    if is_token_revoked(token):
        logger.debug("Token validation failed: token has been revoked")
        return False

    return True


def _extract_handler(*args, **kwargs) -> Any:
    """Extract handler from function arguments."""
    handler = kwargs.get("handler")
    if handler is None:
        for arg in args:
            if hasattr(arg, "headers"):
                handler = arg
                break
    return handler


def _error_response(message: str, status: int = 401) -> "HandlerResult":
    """Create an error response."""
    from aragora.server.handlers.base import error_response

    return error_response(message, status)


def require_auth(func: Callable) -> Callable:
    """
    Decorator that ALWAYS requires authentication.

    Use this for sensitive endpoints that must never run without
    authentication, even in development/testing environments.

    Examples of sensitive endpoints:
    - Plugin execution
    - Capability probing
    - Laboratory experiments
    - Admin operations

    Usage:
        @require_auth
        def sensitive_operation(self, handler):
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.server.auth import auth_config

        handler = _extract_handler(*args, **kwargs)

        if handler is None:
            logger.warning("require_auth: No handler provided, denying access")
            return _error_response("Authentication required", 401)

        # Check that API token is configured
        if not auth_config.api_token:
            logger.warning(
                "require_auth: No API token configured, denying access to sensitive endpoint"
            )
            return _error_response(
                "Authentication required. Set ARAGORA_API_TOKEN environment variable.",
                401,
            )

        # Extract and validate token
        token = extract_token(handler)
        if not token or not validate_token(token):
            return _error_response("Invalid or missing authentication token", 401)

        return func(*args, **kwargs)

    return wrapper


def optional_auth(func: Callable) -> Callable:
    """
    Decorator that provides optional authentication context.

    Unlike require_auth, this allows unauthenticated requests but
    provides an AuthContext to the handler indicating auth status.

    The AuthContext is injected as the 'auth_context' keyword argument.

    Usage:
        @optional_auth
        def public_endpoint(self, handler, auth_context: AuthContext):
            if auth_context.authenticated:
                return personalized_response()
            else:
                return public_response()
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        handler = _extract_handler(*args, **kwargs)

        # Build auth context
        token = extract_token(handler)
        authenticated = validate_token(token) if token else False
        client_ip = extract_client_ip(handler)

        auth_context = AuthContext(
            authenticated=authenticated,
            token=token if authenticated else None,
            client_ip=client_ip,
        )

        # Inject auth context
        kwargs["auth_context"] = auth_context

        return func(*args, **kwargs)

    return wrapper


def require_auth_or_localhost(func: Callable) -> Callable:
    """
    Decorator that requires auth OR allows localhost connections.

    Useful for endpoints that should be protected in production but
    accessible for local development without token setup.

    Usage:
        @require_auth_or_localhost
        def dev_friendly_endpoint(self, handler):
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        handler = _extract_handler(*args, **kwargs)

        if handler is None:
            return _error_response("Authentication required", 401)

        # Check if localhost
        client_ip = extract_client_ip(handler)
        if client_ip in ("127.0.0.1", "::1", "localhost"):
            logger.debug(f"Allowing localhost access from {client_ip}")
            return func(*args, **kwargs)

        # Not localhost - require auth
        from aragora.server.auth import auth_config

        if not auth_config.api_token:
            return _error_response(
                "Authentication required for non-localhost requests",
                401,
            )

        token = extract_token(handler)
        if not token or not validate_token(token):
            return _error_response("Invalid or missing authentication token", 401)

        return func(*args, **kwargs)

    return wrapper
