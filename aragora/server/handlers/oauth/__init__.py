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
    _generate_state,
    _get_google_client_id,
    _get_google_client_secret,
    _get_google_redirect_uri,
    _get_oauth_success_url,
    _get_oauth_error_url,
    _get_allowed_redirect_hosts,
    _IS_PRODUCTION,
    GOOGLE_CLIENT_ID,
    ALLOWED_OAUTH_REDIRECT_HOSTS,
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
    "_generate_state",
    "_get_google_client_id",
    "_get_google_client_secret",
    "_get_google_redirect_uri",
    "_get_oauth_success_url",
    "_get_oauth_error_url",
    "_get_allowed_redirect_hosts",
    "_IS_PRODUCTION",
    "GOOGLE_CLIENT_ID",
    "ALLOWED_OAUTH_REDIRECT_HOSTS",
    "OAUTH_SUCCESS_URL",
    "OAUTH_ERROR_URL",
]
