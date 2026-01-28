"""
Apple OAuth Provider - Sign in with Apple implementation.

Handles:
- Authorization URL generation for Apple consent screen
- Token exchange with Apple (using JWT client authentication)
- User info extraction from ID token
- ID token signature verification using Apple's JWKS
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from aragora.server.handlers.oauth_providers.base import (
    OAuthProvider,
    OAuthProviderConfig,
    OAuthTokens,
    OAuthUserInfo,
    _get_secret,
    _is_production,
)

logger = logging.getLogger(__name__)

# JWKS cache TTL (24 hours)
JWKS_CACHE_TTL = 86400

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

    # Class-level JWKS cache (shared across instances)
    _jwks_cache: ClassVar[Optional[Dict[str, Any]]] = None
    _jwks_cache_expiry: ClassVar[float] = 0
    _jwks_lock: ClassVar[threading.Lock] = threading.Lock()

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

    def _fetch_apple_jwks(self) -> Dict[str, Any]:
        """
        Fetch Apple's public keys from JWKS endpoint with caching.

        Returns:
            JWKS dictionary with public keys

        Raises:
            RuntimeError: If unable to fetch keys
        """
        now = time.time()

        # Check cache with thread safety
        with AppleOAuthProvider._jwks_lock:
            if (
                AppleOAuthProvider._jwks_cache is not None
                and now < AppleOAuthProvider._jwks_cache_expiry
            ):
                return AppleOAuthProvider._jwks_cache

        # Fetch fresh keys outside the lock
        try:
            client = self._get_http_client()
            response = client.get(APPLE_KEYS_URL)
            response.raise_for_status()
            jwks = response.json()

            # Update cache with lock
            with AppleOAuthProvider._jwks_lock:
                AppleOAuthProvider._jwks_cache = jwks
                AppleOAuthProvider._jwks_cache_expiry = now + JWKS_CACHE_TTL

            logger.debug("[apple] Refreshed JWKS cache with %d keys", len(jwks.get("keys", [])))
            return jwks

        except Exception as e:
            logger.error("[apple] Failed to fetch JWKS: %s", e)
            # Return cached keys if available, even if expired
            if AppleOAuthProvider._jwks_cache is not None:
                logger.warning("[apple] Using expired JWKS cache as fallback")
                return AppleOAuthProvider._jwks_cache
            raise RuntimeError(f"Unable to fetch Apple JWKS: {e}") from e

    def _get_signing_key(self, id_token: str) -> Tuple[Any, str]:
        """
        Get the signing key for verifying an ID token.

        Args:
            id_token: The ID token to get the key for

        Returns:
            Tuple of (key, algorithm) for verification

        Raises:
            ValueError: If key not found or token invalid
        """
        try:
            import jwt
        except ImportError:
            raise ImportError("PyJWT is required for Apple Sign In: pip install pyjwt")

        # Get the key ID from token header
        try:
            unverified_header = jwt.get_unverified_header(id_token)
        except jwt.exceptions.DecodeError as e:
            raise ValueError(f"Invalid ID token format: {e}") from e

        kid = unverified_header.get("kid")
        if not kid:
            raise ValueError("No 'kid' in ID token header")

        alg = unverified_header.get("alg", "RS256")

        # Find matching key in JWKS
        jwks = self._fetch_apple_jwks()
        keys = jwks.get("keys", [])

        for key_data in keys:
            if key_data.get("kid") == kid:
                # Convert JWK to key object
                _ = APPLE_KEYS_URL  # URL used for key lookup
                # Use cached JWKS data to create key
                from jwt import PyJWK

                signing_key = PyJWK.from_dict(key_data)
                return signing_key.key, alg

        raise ValueError(f"No matching key found for kid: {kid}")

    def _verify_id_token(
        self,
        id_token: str,
        nonce: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Verify Apple ID token signature and validate claims.

        This method:
        1. Fetches Apple's public keys (JWKS) with caching
        2. Verifies the token signature
        3. Validates required claims (iss, aud, exp, iat)
        4. Optionally validates nonce

        Args:
            id_token: The ID token to verify
            nonce: Optional nonce to validate (if provided during authorization)

        Returns:
            Verified token claims

        Raises:
            ValueError: If verification fails
        """
        try:
            import jwt
        except ImportError:
            raise ImportError("PyJWT is required for Apple Sign In: pip install pyjwt")

        # Get the signing key
        key, algorithm = self._get_signing_key(id_token)

        # Verify and decode
        try:
            claims = jwt.decode(
                id_token,
                key,
                algorithms=[algorithm],
                audience=self._config.client_id,
                issuer="https://appleid.apple.com",
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                    "require": ["sub", "iss", "aud", "exp", "iat"],
                },
            )
        except jwt.ExpiredSignatureError:
            raise ValueError("ID token has expired")
        except jwt.InvalidAudienceError:
            raise ValueError("ID token audience does not match client_id")
        except jwt.InvalidIssuerError:
            raise ValueError("ID token issuer is not Apple")
        except jwt.InvalidSignatureError:
            raise ValueError("ID token signature verification failed")
        except jwt.DecodeError as e:
            raise ValueError(f"Failed to decode ID token: {e}")

        # Validate nonce if provided
        if nonce is not None:
            token_nonce = claims.get("nonce")
            if token_nonce != nonce:
                raise ValueError("ID token nonce does not match")

        return claims

    def _decode_id_token(
        self,
        id_token: str,
        verify: bool = True,
        nonce: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Decode and optionally verify Apple ID token claims.

        By default, verifies the token signature using Apple's public keys
        from APPLE_KEYS_URL with caching.

        Args:
            id_token: ID token to decode
            verify: Whether to verify signature (default True, recommended)
            nonce: Optional nonce to validate

        Returns:
            Token claims

        Raises:
            ValueError: If token is invalid or verification fails
        """
        if verify:
            return self._verify_id_token(id_token, nonce)

        # Fallback to unverified decode (for testing or edge cases)
        logger.warning("[apple] Decoding ID token without verification - not recommended!")
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
