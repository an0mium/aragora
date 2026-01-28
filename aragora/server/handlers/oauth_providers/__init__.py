"""
OAuth Provider Implementations.

This module provides OAuth provider implementations that can be used
with the OAuthHandler. Each provider handles its specific OAuth flow.

Available providers:
- GoogleOAuthProvider - Google OAuth 2.0
- GitHubOAuthProvider - GitHub OAuth 2.0
- MicrosoftOAuthProvider - Microsoft/Azure AD OAuth 2.0
- AppleOAuthProvider - Sign in with Apple
- OIDCProvider - Generic OpenID Connect

Usage:
    from aragora.server.handlers.oauth_providers import GoogleOAuthProvider

    provider = GoogleOAuthProvider()
    auth_url = provider.get_authorization_url(state, redirect_uri)
    user_info = provider.exchange_and_get_user(code, redirect_uri)
"""

from aragora.server.handlers.oauth_providers.base import (
    OAuthProvider,
    OAuthProviderConfig,
    OAuthTokens,
    OAuthUserInfo,
)
from aragora.server.handlers.oauth_providers.google import GoogleOAuthProvider
from aragora.server.handlers.oauth_providers.github import GitHubOAuthProvider
from aragora.server.handlers.oauth_providers.microsoft import MicrosoftOAuthProvider
from aragora.server.handlers.oauth_providers.apple import AppleOAuthProvider
from aragora.server.handlers.oauth_providers.oidc import OIDCProvider

__all__ = [
    # Base classes
    "OAuthProvider",
    "OAuthProviderConfig",
    "OAuthTokens",
    "OAuthUserInfo",
    # Providers
    "GoogleOAuthProvider",
    "GitHubOAuthProvider",
    "MicrosoftOAuthProvider",
    "AppleOAuthProvider",
    "OIDCProvider",
]
