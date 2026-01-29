"""
OAuth Authentication Handlers - backward compatibility shim.

This module previously contained the monolithic OAuthHandler class (~1,800 LOC).
It has been refactored into the `_oauth/` package with provider-specific mixins:

- _oauth/base.py: OAuthHandler (combines all mixins)
- _oauth/google.py: GoogleOAuthMixin
- _oauth/github.py: GitHubOAuthMixin
- _oauth/microsoft.py: MicrosoftOAuthMixin
- _oauth/apple.py: AppleOAuthMixin
- _oauth/oidc.py: OIDCOAuthMixin
- _oauth/account.py: AccountManagementMixin
- _oauth/utils.py: Shared utility functions

All public names are re-exported here for backward compatibility.
"""

from __future__ import annotations

# Re-export OAuthHandler from the new package
from ._oauth.base import OAuthHandler  # noqa: F401

# Re-export utility functions from the new package
from ._oauth.utils import (  # noqa: F401
    _oauth_limiter,
    _validate_redirect_url,
    _validate_state,
)

# Re-export tracing (used by tests that patch at this module path)
from aragora.observability.tracing import create_span, add_span_attributes  # noqa: F401

# Re-export everything from oauth.config (tests patch these at _oauth_impl level)
from .oauth.config import (  # noqa: F401
    _get_secret,
    _is_production,
    _get_google_client_id,
    _get_google_client_secret,
    _get_github_client_id,
    _get_github_client_secret,
    _get_microsoft_client_id,
    _get_microsoft_client_secret,
    _get_microsoft_tenant,
    _get_apple_client_id,
    _get_apple_team_id,
    _get_apple_key_id,
    _get_apple_private_key,
    _get_oidc_issuer,
    _get_oidc_client_id,
    _get_oidc_client_secret,
    _get_google_redirect_uri,
    _get_github_redirect_uri,
    _get_microsoft_redirect_uri,
    _get_apple_redirect_uri,
    _get_oidc_redirect_uri,
    _get_oauth_success_url,
    _get_oauth_error_url,
    _get_allowed_redirect_hosts,
    _IS_PRODUCTION,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GITHUB_CLIENT_ID,
    GITHUB_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
    GITHUB_REDIRECT_URI,
    OAUTH_SUCCESS_URL,
    OAUTH_ERROR_URL,
    ALLOWED_OAUTH_REDIRECT_HOSTS,
    validate_oauth_config,
    GOOGLE_AUTH_URL,
    GOOGLE_TOKEN_URL,
    GOOGLE_USERINFO_URL,
    GITHUB_AUTH_URL,
    GITHUB_TOKEN_URL,
    GITHUB_USERINFO_URL,
    GITHUB_EMAILS_URL,
    MICROSOFT_AUTH_URL_TEMPLATE,
    MICROSOFT_TOKEN_URL_TEMPLATE,
    MICROSOFT_USERINFO_URL,
    APPLE_AUTH_URL,
    APPLE_TOKEN_URL,
    APPLE_KEYS_URL,
)

# Re-export models
from .oauth.models import OAuthUserInfo, _get_param  # noqa: F401

# Re-export state management
from .oauth.state import (  # noqa: F401
    _OAuthStatesView,
    _OAUTH_STATES,
    _STATE_TTL_SECONDS,
    MAX_OAUTH_STATES,
    _cleanup_expired_states,
)

# Re-export state store functions (tests patch these at _oauth_impl level)
from aragora.server.oauth_state_store import (  # noqa: F401
    generate_oauth_state as _generate_state,
    validate_oauth_state as _validate_state_internal,
)

__all__ = ["OAuthHandler", "validate_oauth_config"]
