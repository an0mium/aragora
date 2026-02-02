"""
OAuth utility functions.

Provides the ``_impl()`` helper that all mixin modules use to resolve config
functions and utility names from the ``_oauth_impl`` backward-compatibility
shim at runtime.  This ensures that ``unittest.mock.patch`` calls targeting
``_oauth_impl.<name>`` are visible to the actual mixin code.

Also holds the shared ``_oauth_limiter`` instance with security audit logging
and exponential backoff for brute force protection.
"""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

from aragora.server.middleware.rate_limit.oauth_limiter import (
    OAuthRateLimiter,
    get_oauth_limiter,
)

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult
    from aragora.server.handlers.oauth.models import OAuthUserInfo

logger = logging.getLogger(__name__)


class OAuthHandlerProtocol(Protocol):
    """Protocol defining methods that OAuth mixins expect from OAuthHandler.

    This protocol enables mypy to type-check mixin classes that reference
    methods defined on the main OAuthHandler or its parent classes.
    """

    def _get_user_store(self) -> Any:
        """Get user store from context."""
        ...

    def _redirect_with_error(self, error: str) -> "HandlerResult":
        """Redirect to error page with error message."""
        ...

    def _redirect_with_tokens(self, redirect_url: str, tokens: Any) -> "HandlerResult":
        """Redirect to frontend with tokens in URL query parameters."""
        ...

    def _check_permission(
        self, handler: Any, permission_key: str, resource_id: str | None = None
    ) -> "HandlerResult | None":
        """Check RBAC permission. Returns error response if denied, None if allowed."""
        ...

    def _complete_oauth_flow(
        self, user_info: "OAuthUserInfo", state_data: dict[str, Any]
    ) -> "HandlerResult":
        """Complete OAuth flow - create/login user and redirect with tokens."""
        ...

    def _find_user_by_oauth(self, user_store: Any, user_info: "OAuthUserInfo") -> Any:
        """Find user by OAuth provider ID."""
        ...

    def _link_oauth_to_user(
        self, user_store: Any, user_id: str, user_info: "OAuthUserInfo"
    ) -> bool:
        """Link OAuth provider to existing user."""
        ...

    def _create_oauth_user(self, user_store: Any, user_info: "OAuthUserInfo") -> Any:
        """Create a new user from OAuth info."""
        ...

    def _handle_account_linking(
        self,
        user_store: Any,
        user_id: str,
        user_info: "OAuthUserInfo",
        state_data: dict[str, Any],
    ) -> "HandlerResult":
        """Handle linking OAuth account to existing user."""
        ...

    def read_json_body(self, handler: Any, max_size: int | None = None) -> dict[str, Any] | None:
        """Read and parse JSON body from request handler."""
        ...

    # Provider-specific auth start methods (used by AccountManagementMixin)
    def _handle_google_auth_start(
        self, handler: Any, query_params: dict[str, Any]
    ) -> "HandlerResult":
        """Redirect user to Google OAuth consent screen."""
        ...

    def _handle_github_auth_start(
        self, handler: Any, query_params: dict[str, Any]
    ) -> "HandlerResult":
        """Redirect user to GitHub OAuth consent screen."""
        ...

    def _handle_microsoft_auth_start(
        self, handler: Any, query_params: dict[str, Any]
    ) -> "HandlerResult":
        """Redirect user to Microsoft OAuth consent screen."""
        ...

    def _handle_apple_auth_start(
        self, handler: Any, query_params: dict[str, Any]
    ) -> "HandlerResult":
        """Redirect user to Apple OAuth consent screen."""
        ...

    def _handle_oidc_auth_start(
        self, handler: Any, query_params: dict[str, Any]
    ) -> "HandlerResult":
        """Redirect user to OIDC OAuth consent screen."""
        ...

    # Provider-specific callback methods (used by AccountManagementMixin)
    def _handle_google_callback(
        self, handler: Any, query_params: dict[str, Any]
    ) -> "HandlerResult":
        """Handle Google OAuth callback."""
        ...

    def _handle_github_callback(
        self, handler: Any, query_params: dict[str, Any]
    ) -> "HandlerResult":
        """Handle GitHub OAuth callback."""
        ...

    def _handle_microsoft_callback(
        self, handler: Any, query_params: dict[str, Any]
    ) -> "HandlerResult":
        """Handle Microsoft OAuth callback."""
        ...

    def _handle_apple_callback(self, handler: Any, query_params: dict[str, Any]) -> "HandlerResult":
        """Handle Apple OAuth callback."""
        ...

    def _handle_oidc_callback(self, handler: Any, query_params: dict[str, Any]) -> "HandlerResult":
        """Handle OIDC OAuth callback."""
        ...


class OAuthRateLimiterWrapper:
    """Wrapper around the new OAuthRateLimiter for backward compatibility.

    This wrapper provides the old `is_allowed(key)` interface expected by
    existing code while using the new OAuth rate limiting infrastructure
    with exponential backoff and stricter limits.
    """

    def __init__(self):
        """Initialize wrapper with the global OAuth limiter."""
        self._limiter = get_oauth_limiter()

    def is_allowed(self, key: str, endpoint_type: str = "auth_start") -> bool:
        """Check if request is allowed.

        Args:
            key: Client IP address
            endpoint_type: Type of endpoint for limit selection

        Returns:
            True if request is allowed, False if rate limited
        """
        result = self._limiter.check(key, endpoint_type)
        return result.allowed

    @property
    def rpm(self) -> int:
        """Get approximate requests per minute (for logging compatibility)."""
        # Return auth_start limit as RPM equivalent (15 per 15 min = 1/min)
        return self._limiter.config.auth_start_limit


# Rate limiter for OAuth endpoints - now uses the new OAuthRateLimiter with
# exponential backoff and stricter limits:
# - Token endpoints: 5 requests per 15 minutes
# - Callback handlers: 10 requests per 15 minutes
# - Auth start (redirect): 15 requests per 15 minutes
_oauth_limiter = OAuthRateLimiterWrapper()

# Module path constant for the backward-compat shim that tests patch against.
_IMPL_MODULE = "aragora.server.handlers._oauth_impl"


def _impl() -> ModuleType:
    """Return the ``_oauth_impl`` module lazily to avoid circular imports.

    All mixin modules call ``_impl().<name>`` instead of importing config
    functions directly so that ``unittest.mock.patch`` applied to
    ``_oauth_impl.<name>`` is visible to the running code.
    """
    return sys.modules[_IMPL_MODULE]
