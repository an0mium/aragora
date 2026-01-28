"""
Apple OAuth Provider - Sign in with Apple implementation.

Handles:
- Authorization URL generation for Apple consent screen
- Token exchange with Apple (using JWT client authentication)
- User info extraction from ID token
"""

from __future__ import annotations

import base64
import json
import logging
import time
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

# Apple OAuth endpoints
APPLE_AUTH_URL = "https://appleid.apple.com/auth/authorize"
APPLE_TOKEN_URL = "https://appleid.apple.com/auth/token"
APPLE_KEYS_URL = "https://appleid.apple.com/auth/keys"
APPLE_REVOCATION_URL = "https://appleid.apple.com/auth/revoke"


class AppleOAuthProvider(OAuthProvider):
    """
    Apple OAuth provider (Sign in with Apple).

    Supports:
    - OAuth 2.0 authorization code flow
    - JWT-based client authentication
    - ID token parsing for user info
    - Token revocation

    Configuration:
    - APPLE_OAUTH_CLIENT_ID: Services ID (com.example.app.web)
    - APPLE_TEAM_ID: Apple Developer Team ID
    - APPLE_KEY_ID: Key ID from Apple Developer Portal
    - APPLE_PRIVATE_KEY: Private key (PEM format)

    Note: Apple only provides user info (name, email) on first authorization.
    Store it immediately as it won't be available on subsequent sign-ins.
    """

    PROVIDER_NAME = "apple"

    def _load_config_from_env(self) -> OAuthProviderConfig:
        """Load Apple OAuth configuration from environment."""
        redirect_uri = _get_secret("APPLE_OAUTH_REDIRECT_URI", "")
        if not redirect_uri and not _is_production():
            redirect_uri = "http://localhost:8080/api/auth/oauth/apple/callback"

        return OAuthProviderConfig(
            client_id=_get_secret("APPLE_OAUTH_CLIENT_ID", ""),
            client_secret="",  # Apple uses JWT instead of static secret
            redirect_uri=redirect_uri,
            scopes=["name", "email"],
            authorization_endpoint=APPLE_AUTH_URL,
            token_endpoint=APPLE_TOKEN_URL,
            revocation_endpoint=APPLE_REVOCATION_URL,
            team_id=_get_secret("APPLE_TEAM_ID", ""),
            key_id=_get_secret("APPLE_KEY_ID", ""),
            private_key=_get_secret("APPLE_PRIVATE_KEY", ""),
        )

    @property
    def is_configured(self) -> bool:
        """Check if Apple Sign In has required configuration."""
        return bool(
            self._config.client_id
            and self._config.team_id
            and self._config.key_id
            and self._config.private_key
        )

    def _generate_client_secret(self) -> str:
        """
        Generate JWT client secret for Apple token exchange.

        Apple requires a signed JWT instead of a static client secret.
        The JWT is signed with your private key from Apple Developer Portal.

        Returns:
            JWT client secret
        """
        try:
            import jwt
        except ImportError:
            raise ImportError("PyJWT is required for Apple Sign In: pip install pyjwt")

        now = int(time.time())

        headers = {
            "alg": "ES256",
            "kid": self._config.key_id,
        }

        payload = {
            "iss": self._config.team_id,
            "iat": now,
            "exp": now + 86400 * 180,  # 180 days (Apple max)
            "aud": "https://appleid.apple.com",
            "sub": self._config.client_id,
        }

        return jwt.encode(
            payload,
            self._config.private_key,
            algorithm="ES256",
            headers=headers,
        )

    def get_authorization_url(
        self,
        state: str,
        redirect_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate Apple OAuth authorization URL.

        Args:
            state: CSRF protection state parameter
            redirect_uri: Override redirect URI
            scopes: Override scopes (name, email)
            **kwargs: Additional parameters:
                - response_mode: "query", "fragment", or "form_post" (default "form_post")
                - nonce: For ID token validation

        Returns:
            Authorization URL to redirect user to
        """
        params = {
            "client_id": self._config.client_id,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
            "response_type": "code",
            "scope": " ".join(scopes or self._config.scopes),
            "state": state,
            "response_mode": kwargs.get("response_mode", "form_post"),
        }

        # Optional nonce for ID token validation
        if "nonce" in kwargs:
            params["nonce"] = kwargs["nonce"]

        return self._build_authorization_url(APPLE_AUTH_URL, params)

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
            OAuth tokens including access_token, id_token, and refresh_token
        """
        client_secret = self._generate_client_secret()

        data = {
            "code": code,
            "client_id": self._config.client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
            "grant_type": "authorization_code",
        }

        return self._request_tokens(APPLE_TOKEN_URL, data)

    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from Apple.

        Apple doesn't have a userinfo endpoint. User info comes from:
        1. The ID token (always includes 'sub' identifier)
        2. The 'user' parameter in the callback (only on first authorization)

        For this basic method, we extract from ID token only.
        Use get_user_info_from_callback for full user data.

        Args:
            access_token: Access token (unused - Apple uses ID token)

        Returns:
            User information (limited - only sub from ID token)
        """
        raise NotImplementedError(
            "Apple doesn't provide a userinfo endpoint. "
            "Use get_user_info_from_id_token() or get_user_info_from_callback() instead."
        )

    def get_user_info_from_id_token(
        self,
        id_token: str,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> OAuthUserInfo:
        """
        Extract user info from ID token and optional user data.

        Args:
            id_token: ID token from token exchange
            user_data: Optional user data from callback (first auth only)

        Returns:
            User information
        """
        claims = self._decode_id_token(id_token)

        # Extract from ID token
        sub = claims.get("sub")
        if not sub:
            raise ValueError("No subject in Apple ID token")

        email = claims.get("email")
        email_verified = claims.get("email_verified", False)

        # Extract name from user data (only available on first auth)
        name = None
        given_name = None
        family_name = None

        if user_data and "name" in user_data:
            name_data = user_data["name"]
            given_name = name_data.get("firstName")
            family_name = name_data.get("lastName")
            if given_name and family_name:
                name = f"{given_name} {family_name}"
            elif given_name:
                name = given_name
            elif family_name:
                name = family_name

        return OAuthUserInfo(
            provider=self.PROVIDER_NAME,
            provider_user_id=sub,
            email=email,
            email_verified=email_verified
            if isinstance(email_verified, bool)
            else email_verified == "true",
            name=name,
            given_name=given_name,
            family_name=family_name,
            raw_data={"id_token_claims": claims, "user_data": user_data},
        )

    def get_user_info_from_callback(
        self,
        tokens: OAuthTokens,
        user_json: Optional[str] = None,
    ) -> OAuthUserInfo:
        """
        Extract user info from callback response.

        Apple sends user info as a JSON string in the 'user' POST parameter
        on the first authorization only.

        Args:
            tokens: OAuth tokens from exchange
            user_json: Optional 'user' JSON string from callback

        Returns:
            User information
        """
        user_data = None
        if user_json:
            try:
                user_data = json.loads(user_json)
            except json.JSONDecodeError:
                logger.warning("[apple] Failed to parse user JSON from callback")

        if not tokens.id_token:
            raise ValueError("No ID token in Apple token response")

        return self.get_user_info_from_id_token(tokens.id_token, user_data)

    def _decode_id_token(self, id_token: str) -> Dict[str, Any]:
        """
        Decode Apple ID token claims without verification.

        For production, you should verify the token signature using
        Apple's public keys from APPLE_KEYS_URL.

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
        client_secret = self._generate_client_secret()

        data = {
            "refresh_token": refresh_token,
            "client_id": self._config.client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
        }

        return self._request_tokens(APPLE_TOKEN_URL, data)

    def revoke_token(self, token: str, token_type: str = "access_token") -> bool:
        """
        Revoke a token.

        Args:
            token: Token to revoke
            token_type: "access_token" or "refresh_token"

        Returns:
            True if revocation succeeded
        """
        try:
            client_secret = self._generate_client_secret()
            client = self._get_http_client()

            response = client.post(
                APPLE_REVOCATION_URL,
                data={
                    "token": token,
                    "token_type_hint": token_type,
                    "client_id": self._config.client_id,
                    "client_secret": client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"[{self.PROVIDER_NAME}] Token revocation failed: {e}")
            return False


__all__ = ["AppleOAuthProvider"]
