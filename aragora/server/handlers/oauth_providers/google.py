"""
Google OAuth Provider - Google OAuth 2.0 implementation.

Handles:
- Authorization URL generation for Google consent screen
- Token exchange with Google
- User info retrieval from Google API
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aragora.server.handlers.oauth_providers.base import (
    OAuthProvider,
    OAuthProviderConfig,
    OAuthTokens,
    OAuthUserInfo,
    _get_secret,
    _is_production,
)

logger = logging.getLogger(__name__)

# Google OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_REVOCATION_URL = "https://oauth2.googleapis.com/revoke"


class GoogleOAuthProvider(OAuthProvider):
    """
    Google OAuth 2.0 provider.

    Supports:
    - OAuth 2.0 authorization code flow
    - OpenID Connect for user info
    - Offline access for refresh tokens
    - Token refresh and revocation
    """

    PROVIDER_NAME = "google"

    def _load_config_from_env(self) -> OAuthProviderConfig:
        """Load Google OAuth configuration from environment."""
        redirect_uri = _get_secret("GOOGLE_OAUTH_REDIRECT_URI", "")
        if not redirect_uri and not _is_production():
            redirect_uri = "http://localhost:8080/api/auth/oauth/google/callback"

        return OAuthProviderConfig(
            client_id=_get_secret("GOOGLE_OAUTH_CLIENT_ID", ""),
            client_secret=_get_secret("GOOGLE_OAUTH_CLIENT_SECRET", ""),
            redirect_uri=redirect_uri,
            scopes=["openid", "email", "profile"],
            authorization_endpoint=GOOGLE_AUTH_URL,
            token_endpoint=GOOGLE_TOKEN_URL,
            userinfo_endpoint=GOOGLE_USERINFO_URL,
            revocation_endpoint=GOOGLE_REVOCATION_URL,
        )

    def get_authorization_url(
        self,
        state: str,
        redirect_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate Google OAuth authorization URL.

        Args:
            state: CSRF protection state parameter
            redirect_uri: Override redirect URI
            scopes: Override scopes
            **kwargs: Additional parameters:
                - access_type: "offline" for refresh tokens (default)
                - prompt: "consent" to force consent screen (default)
                - login_hint: Pre-fill email address
                - hd: Restrict to Google Workspace domain

        Returns:
            Authorization URL to redirect user to
        """
        params = {
            "client_id": self._config.client_id,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes or self._config.scopes),
            "state": state,
            "access_type": kwargs.get("access_type", "offline"),
            "prompt": kwargs.get("prompt", "consent"),
        }

        # Optional Google-specific parameters
        if "login_hint" in kwargs:
            params["login_hint"] = kwargs["login_hint"]
        if "hd" in kwargs:
            params["hd"] = kwargs["hd"]

        return self._build_authorization_url(GOOGLE_AUTH_URL, params)

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
            OAuth tokens including access_token and optionally refresh_token
        """
        data = {
            "code": code,
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
            "grant_type": "authorization_code",
        }

        return self._request_tokens(GOOGLE_TOKEN_URL, data)

    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from Google API.

        Args:
            access_token: Access token from exchange

        Returns:
            User information including email, name, picture
        """
        data = self._request_user_info(GOOGLE_USERINFO_URL, access_token)

        return OAuthUserInfo(
            provider=self.PROVIDER_NAME,
            provider_user_id=data["id"],
            email=data.get("email"),
            email_verified=data.get("verified_email", False),
            name=data.get("name"),
            given_name=data.get("given_name"),
            family_name=data.get("family_name"),
            picture=data.get("picture"),
            locale=data.get("locale"),
            raw_data=data,
        )

    def refresh_access_token(self, refresh_token: str) -> OAuthTokens:
        """
        Refresh the access token using a refresh token.

        Args:
            refresh_token: Refresh token from previous exchange

        Returns:
            New OAuth tokens (may not include new refresh_token)
        """
        data = {
            "refresh_token": refresh_token,
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "grant_type": "refresh_token",
        }

        return self._request_tokens(GOOGLE_TOKEN_URL, data)

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token (access or refresh).

        Args:
            token: Token to revoke

        Returns:
            True if revocation succeeded
        """
        try:
            client = self._get_http_client()
            response = client.post(
                GOOGLE_REVOCATION_URL,
                data={"token": token},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"[{self.PROVIDER_NAME}] Token revocation failed: {e}")
            return False

    def get_id_token_info(self, id_token: str) -> Dict[str, Any]:
        """
        Parse and return claims from an ID token.

        Note: This performs basic parsing without signature verification.
        For production, use Google's tokeninfo endpoint or verify JWT signature.

        Args:
            id_token: ID token from token exchange

        Returns:
            Dictionary of ID token claims
        """
        import base64
        import json

        parts = id_token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid ID token format")

        # Decode payload (middle part)
        payload_b64 = parts[1]
        # Add padding if needed
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding

        payload = base64.urlsafe_b64decode(payload_b64)
        return json.loads(payload)


__all__ = ["GoogleOAuthProvider"]
