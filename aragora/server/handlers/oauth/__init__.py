"""
OAuth handler package.

Provides OAuth2/OIDC authentication endpoints for various providers:
- Google OAuth 2.0
- GitHub OAuth
- Microsoft OAuth (Azure AD)
- Apple Sign-In
- Generic OIDC

The handler is split into logical modules for better maintainability:

- handler.py: Main OAuthHandler class (re-exported from _oauth_impl.py)
- providers/: Provider-specific OAuth implementations
- flow/: Common OAuth flow utilities
- account/: Account linking and provider management

For backward compatibility, import OAuthHandler from this package:

    from aragora.server.handlers.oauth import OAuthHandler
"""

from .handler import (
    OAuthHandler,
    OAuthUserInfo,
    validate_oauth_config,
    _oauth_limiter,
    _validate_redirect_url,
    _validate_state,
    _cleanup_expired_states,
    OAUTH_SUCCESS_URL,
    OAUTH_ERROR_URL,
)

__all__ = [
    "OAuthHandler",
    "OAuthUserInfo",
    "validate_oauth_config",
    "_oauth_limiter",
    "_validate_redirect_url",
    "_validate_state",
    "_cleanup_expired_states",
    "OAUTH_SUCCESS_URL",
    "OAUTH_ERROR_URL",
]
