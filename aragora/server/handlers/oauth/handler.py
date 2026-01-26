"""
OAuth handler - main entry point.

Re-exports OAuthHandler from the implementation module for backward compatibility.
The modular structure is in place for future incremental migration of methods.
"""

# Re-export OAuthHandler from the implementation module
from .._oauth_impl import (
    OAuthHandler,
    OAuthUserInfo,
    validate_oauth_config,
    _oauth_limiter,
    _validate_redirect_url,
    _validate_state,
    _cleanup_expired_states,
    _generate_state,
    _get_google_client_id,
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
    "_get_allowed_redirect_hosts",
    "_IS_PRODUCTION",
    "GOOGLE_CLIENT_ID",
    "ALLOWED_OAUTH_REDIRECT_HOSTS",
    "OAUTH_SUCCESS_URL",
    "OAUTH_ERROR_URL",
]
