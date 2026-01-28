"""
Generic OIDC Provider - OpenID Connect implementation.

Handles:
- Discovery document fetching
- Authorization URL generation
- Token exchange
- User info retrieval from userinfo endpoint or ID token
"""

from __future__ import annotations

import base64
import json
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


class OIDCProvider(OAuthProvider):
    """
    Generic OpenID Connect provider.

    Supports any OIDC-compliant identity provider through automatic
    discovery of endpoints via the .well-known/openid-configuration document.

    Configuration:
    - OIDC_ISSUER: Issuer URL (e.g., https://idp.example.com)
    - OIDC_CLIENT_ID: Client ID
    - OIDC_CLIENT_SECRET: Client secret
    - OIDC_REDIRECT_URI: Callback URL
    """

    PROVIDER_NAME = "oidc"

    def __init__(self, config: Optional[OAuthProviderConfig] = None):
        """Initialize with optional discovery."""
        super().__init__(config)
        self._discovery: Optional[Dict[str, Any]] = None

    def _load_config_from_env(self) -> OAuthProviderConfig:
        """Load OIDC configuration from environment."""
        redirect_uri = _get_secret("OIDC_REDIRECT_URI", "")
        if not redirect_uri and not _is_production():
            redirect_uri = "http://localhost:8080/api/auth/oauth/oidc/callback"

        return OAuthProviderConfig(
            client_id=_get_secret("OIDC_CLIENT_ID", ""),
            client_secret=_get_secret("OIDC_CLIENT_SECRET", ""),
            redirect_uri=redirect_uri,
            scopes=["openid", "email", "profile"],
            # Endpoints discovered dynamically from issuer
            authorization_endpoint="",
            token_endpoint="",
            userinfo_endpoint="",
        )

    @property
    def issuer(self) -> str:
        """Get the OIDC issuer URL."""
        return _get_secret("OIDC_ISSUER", "")

    @property
    def is_configured(self) -> bool:
        """Check if OIDC has required configuration."""
        return bool(self.issuer and self._config.client_id and self._config.client_secret)

    def _get_discovery(self) -> Dict[str, Any]:
        """
        Fetch and cache OIDC discovery document.

        Returns:
            Discovery document with endpoints
        """
        if self._discovery is not None:
            return self._discovery

        issuer = self.issuer
        if not issuer:
            raise ValueError("OIDC_ISSUER not configured")

        discovery_url = f"{issuer.rstrip('/')}/.well-known/openid-configuration"

        try:
            client = self._get_http_client()
            response = client.get(discovery_url)
            response.raise_for_status()
            self._discovery = response.json()
            return self._discovery
        except Exception as e:
            logger.error(f"[{self.PROVIDER_NAME}] Discovery failed: {e}")
            raise ValueError(f"Failed to fetch OIDC discovery: {e}") from e

    def get_authorization_url(
        self,
        state: str,
        redirect_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate OIDC authorization URL.

        Args:
            state: CSRF protection state parameter
            redirect_uri: Override redirect URI
            scopes: Override scopes
            **kwargs: Additional parameters:
                - nonce: For ID token validation
                - prompt: "none", "login", "consent", "select_account"
                - login_hint: Pre-fill username/email
                - acr_values: Authentication context class reference

        Returns:
            Authorization URL to redirect user to
        """
        discovery = self._get_discovery()
        auth_endpoint = discovery.get("authorization_endpoint")
        if not auth_endpoint:
            raise ValueError("No authorization_endpoint in discovery")

        params = {
            "client_id": self._config.client_id,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes or self._config.scopes),
            "state": state,
        }

        # Optional OIDC parameters
        if "nonce" in kwargs:
            params["nonce"] = kwargs["nonce"]
        if "prompt" in kwargs:
            params["prompt"] = kwargs["prompt"]
        if "login_hint" in kwargs:
            params["login_hint"] = kwargs["login_hint"]
        if "acr_values" in kwargs:
            params["acr_values"] = kwargs["acr_values"]

        return self._build_authorization_url(auth_endpoint, params)

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
            OAuth tokens including access_token, id_token, and optionally refresh_token
        """
        discovery = self._get_discovery()
        token_endpoint = discovery.get("token_endpoint")
        if not token_endpoint:
            raise ValueError("No token_endpoint in discovery")

        data = {
            "code": code,
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
            "grant_type": "authorization_code",
        }

        return self._request_tokens(token_endpoint, data)

    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from OIDC userinfo endpoint.

        Args:
            access_token: Access token from exchange

        Returns:
            User information
        """
        discovery = self._get_discovery()
        userinfo_endpoint = discovery.get("userinfo_endpoint")

        if not userinfo_endpoint:
            raise ValueError(
                "No userinfo_endpoint in discovery. Use get_user_info_from_id_token() instead."
            )

        user_data = self._request_user_info(userinfo_endpoint, access_token)
        return self._parse_user_data(user_data)

    def get_user_info_from_id_token(self, id_token: str) -> OAuthUserInfo:
        """
        Extract user info from ID token claims.

        Args:
            id_token: ID token from token exchange

        Returns:
            User information
        """
        claims = self._decode_id_token(id_token)
        return self._parse_user_data(claims)

    def get_user_info_combined(
        self,
        access_token: str,
        id_token: Optional[str] = None,
    ) -> OAuthUserInfo:
        """
        Get user info from userinfo endpoint with ID token fallback.

        Args:
            access_token: Access token
            id_token: Optional ID token for fallback

        Returns:
            User information
        """
        discovery = self._get_discovery()
        user_data = {}

        # Try userinfo endpoint first
        userinfo_endpoint = discovery.get("userinfo_endpoint")
        if userinfo_endpoint:
            try:
                user_data = self._request_user_info(userinfo_endpoint, access_token)
            except Exception as e:
                logger.warning(f"[{self.PROVIDER_NAME}] Userinfo failed: {e}")

        # Fall back to ID token claims
        if not user_data and id_token:
            user_data = self._decode_id_token(id_token)

        if not user_data:
            raise ValueError("Could not retrieve user info from OIDC provider")

        return self._parse_user_data(user_data)

    def _parse_user_data(self, data: Dict[str, Any]) -> OAuthUserInfo:
        """Parse user data from userinfo or ID token claims."""
        sub = data.get("sub")
        if not sub:
            raise ValueError("No subject (sub) in OIDC response")

        email = data.get("email")
        if not email:
            raise ValueError("No email in OIDC response")

        return OAuthUserInfo(
            provider=self.PROVIDER_NAME,
            provider_user_id=sub,
            email=email,
            email_verified=data.get("email_verified", False),
            name=data.get("name"),
            given_name=data.get("given_name"),
            family_name=data.get("family_name"),
            picture=data.get("picture"),
            locale=data.get("locale"),
            raw_data=data,
        )

    def _decode_id_token(self, id_token: str) -> Dict[str, Any]:
        """
        Decode ID token claims without verification.

        For production, verify the token signature using the provider's
        JWKS from the jwks_uri in the discovery document.

        Args:
            id_token: ID token to decode

        Returns:
            Token claims
        """
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

    def refresh_access_token(self, refresh_token: str) -> OAuthTokens:
        """
        Refresh the access token using a refresh token.

        Args:
            refresh_token: Refresh token from previous exchange

        Returns:
            New OAuth tokens
        """
        discovery = self._get_discovery()
        token_endpoint = discovery.get("token_endpoint")
        if not token_endpoint:
            raise ValueError("No token_endpoint in discovery")

        data = {
            "refresh_token": refresh_token,
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "grant_type": "refresh_token",
        }

        return self._request_tokens(token_endpoint, data)

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token if revocation endpoint is available.

        Args:
            token: Token to revoke

        Returns:
            True if revocation succeeded
        """
        discovery = self._get_discovery()
        revocation_endpoint = discovery.get("revocation_endpoint")

        if not revocation_endpoint:
            logger.warning(f"[{self.PROVIDER_NAME}] No revocation endpoint available")
            return False

        try:
            client = self._get_http_client()
            response = client.post(
                revocation_endpoint,
                data={
                    "token": token,
                    "client_id": self._config.client_id,
                    "client_secret": self._config.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"[{self.PROVIDER_NAME}] Token revocation failed: {e}")
            return False

    def get_end_session_url(
        self,
        id_token_hint: Optional[str] = None,
        post_logout_redirect_uri: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Optional[str]:
        """
        Get the OIDC end session (logout) URL if available.

        Args:
            id_token_hint: ID token for logout
            post_logout_redirect_uri: Where to redirect after logout
            state: State parameter for callback

        Returns:
            Logout URL or None if not supported
        """
        discovery = self._get_discovery()
        end_session_endpoint = discovery.get("end_session_endpoint")

        if not end_session_endpoint:
            return None

        from urllib.parse import urlencode

        params = {}
        if id_token_hint:
            params["id_token_hint"] = id_token_hint
        if post_logout_redirect_uri:
            params["post_logout_redirect_uri"] = post_logout_redirect_uri
        if state:
            params["state"] = state

        if params:
            return f"{end_session_endpoint}?{urlencode(params)}"
        return end_session_endpoint


__all__ = ["OIDCProvider"]
