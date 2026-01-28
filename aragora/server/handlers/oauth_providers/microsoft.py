"""
Microsoft OAuth Provider - Microsoft/Azure AD OAuth 2.0 implementation.

Handles:
- Authorization URL generation for Microsoft consent screen
- Token exchange with Microsoft identity platform
- User info retrieval from Microsoft Graph API
"""

from __future__ import annotations

import logging
from typing import List, Optional

from aragora.server.handlers.oauth_providers.base import (
    OAuthProvider,
    OAuthProviderConfig,
    OAuthTokens,
    OAuthUserInfo,
    _get_secret,
    _is_production,
)

logger = logging.getLogger(__name__)

# Microsoft OAuth endpoints (Azure AD v2.0)
# Note: {tenant} is replaced at runtime
MICROSOFT_AUTH_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
MICROSOFT_TOKEN_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
MICROSOFT_USERINFO_URL = "https://graph.microsoft.com/v1.0/me"
MICROSOFT_LOGOUT_URL_TEMPLATE = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/logout"


class MicrosoftOAuthProvider(OAuthProvider):
    """
    Microsoft OAuth 2.0 provider (Azure AD v2.0).

    Supports:
    - OAuth 2.0 authorization code flow
    - Multi-tenant and single-tenant configurations
    - Microsoft Graph API for user info
    - Token refresh

    Configuration:
    - MICROSOFT_OAUTH_CLIENT_ID: Application (client) ID
    - MICROSOFT_OAUTH_CLIENT_SECRET: Client secret
    - MICROSOFT_OAUTH_TENANT: Tenant ID or "common" for multi-tenant
    - MICROSOFT_OAUTH_REDIRECT_URI: Callback URL
    """

    PROVIDER_NAME = "microsoft"

    def _load_config_from_env(self) -> OAuthProviderConfig:
        """Load Microsoft OAuth configuration from environment."""
        redirect_uri = _get_secret("MICROSOFT_OAUTH_REDIRECT_URI", "")
        if not redirect_uri and not _is_production():
            redirect_uri = "http://localhost:8080/api/auth/oauth/microsoft/callback"

        tenant = _get_secret("MICROSOFT_OAUTH_TENANT", "common")

        return OAuthProviderConfig(
            client_id=_get_secret("MICROSOFT_OAUTH_CLIENT_ID", ""),
            client_secret=_get_secret("MICROSOFT_OAUTH_CLIENT_SECRET", ""),
            redirect_uri=redirect_uri,
            scopes=["openid", "email", "profile", "User.Read"],
            authorization_endpoint=MICROSOFT_AUTH_URL_TEMPLATE.format(tenant=tenant),
            token_endpoint=MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant=tenant),
            userinfo_endpoint=MICROSOFT_USERINFO_URL,
            tenant=tenant,
        )

    @property
    def tenant(self) -> str:
        """Get the configured tenant."""
        return self._config.tenant or "common"

    def get_authorization_url(
        self,
        state: str,
        redirect_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate Microsoft OAuth authorization URL.

        Args:
            state: CSRF protection state parameter
            redirect_uri: Override redirect URI
            scopes: Override scopes
            **kwargs: Additional parameters:
                - response_mode: "query" or "fragment" (default "query")
                - prompt: "login", "consent", "select_account", or "none"
                - login_hint: Pre-fill email/username
                - domain_hint: Hint for Azure AD tenant

        Returns:
            Authorization URL to redirect user to
        """
        auth_url = MICROSOFT_AUTH_URL_TEMPLATE.format(tenant=self.tenant)

        params = {
            "client_id": self._config.client_id,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes or self._config.scopes),
            "state": state,
            "response_mode": kwargs.get("response_mode", "query"),
        }

        # Optional Microsoft-specific parameters
        if "prompt" in kwargs:
            params["prompt"] = kwargs["prompt"]
        if "login_hint" in kwargs:
            params["login_hint"] = kwargs["login_hint"]
        if "domain_hint" in kwargs:
            params["domain_hint"] = kwargs["domain_hint"]

        return self._build_authorization_url(auth_url, params)

    def exchange_code(
        self,
        code: str,
        redirect_uri: Optional[str] = None,
    ) -> OAuthTokens:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            redirect_uri: Redirect URI used in authorization

        Returns:
            OAuth tokens including access_token and refresh_token
        """
        token_url = MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant=self.tenant)

        data = {
            "code": code,
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
            "grant_type": "authorization_code",
        }

        return self._request_tokens(token_url, data)

    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from Microsoft Graph API.

        Args:
            access_token: Access token from exchange

        Returns:
            User information including email, name
        """
        user_data = self._request_user_info(MICROSOFT_USERINFO_URL, access_token)

        # Microsoft Graph uses "mail" or "userPrincipalName" for email
        email = user_data.get("mail") or user_data.get("userPrincipalName", "")
        if not email or "@" not in email:
            raise ValueError("Could not retrieve email from Microsoft")

        return OAuthUserInfo(
            provider=self.PROVIDER_NAME,
            provider_user_id=user_data["id"],
            email=email,
            email_verified=True,  # Microsoft validates all emails
            name=user_data.get("displayName"),
            given_name=user_data.get("givenName"),
            family_name=user_data.get("surname"),
            picture=None,  # Microsoft Graph requires separate call for photo
            raw_data=user_data,
        )

    def refresh_access_token(self, refresh_token: str) -> OAuthTokens:
        """
        Refresh the access token using a refresh token.

        Args:
            refresh_token: Refresh token from previous exchange

        Returns:
            New OAuth tokens
        """
        token_url = MICROSOFT_TOKEN_URL_TEMPLATE.format(tenant=self.tenant)

        data = {
            "refresh_token": refresh_token,
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "grant_type": "refresh_token",
        }

        return self._request_tokens(token_url, data)

    def get_logout_url(self, post_logout_redirect_uri: Optional[str] = None) -> str:
        """
        Get the Microsoft logout URL.

        Args:
            post_logout_redirect_uri: URL to redirect after logout

        Returns:
            Logout URL
        """
        logout_url = MICROSOFT_LOGOUT_URL_TEMPLATE.format(tenant=self.tenant)

        if post_logout_redirect_uri:
            from urllib.parse import urlencode

            params = {"post_logout_redirect_uri": post_logout_redirect_uri}
            return f"{logout_url}?{urlencode(params)}"

        return logout_url

    def get_user_photo(self, access_token: str) -> Optional[bytes]:
        """
        Get user's profile photo from Microsoft Graph.

        Args:
            access_token: Access token with User.Read scope

        Returns:
            Photo bytes or None if not available
        """
        try:
            client = self._get_http_client()
            response = client.get(
                "https://graph.microsoft.com/v1.0/me/photo/$value",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            logger.warning(f"[{self.PROVIDER_NAME}] Failed to fetch photo: {e}")
            return None


__all__ = ["MicrosoftOAuthProvider"]
