"""
JWT Verification for Chat Platform Webhooks.

Provides JWT token verification for platforms that use OAuth bearer tokens:
- Microsoft Teams (Bot Framework)
- Google Chat

Uses PyJWK to fetch signing keys from platform endpoints and validate tokens.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)

# Optional: PyJWT for token validation
try:
    import jwt
    from jwt import PyJWKClient as _PyJWKClient
    from jwt.exceptions import PyJWTError as _PyJWTError

    HAS_JWT = True
    PyJWKClient: Optional[Type[Any]] = _PyJWKClient
    PyJWTError: Type[Exception] = _PyJWTError
except ImportError:
    jwt = None
    PyJWKClient = None
    PyJWTError = Exception
    HAS_JWT = False

# Microsoft Bot Framework OpenID configuration
MICROSOFT_OPENID_METADATA_URL = "https://login.botframework.com/v1/.well-known/openidconfiguration"
MICROSOFT_JWKS_URI = "https://login.botframework.com/v1/.well-known/keys"
MICROSOFT_VALID_ISSUERS = [
    "https://api.botframework.com",
    "https://sts.windows.net/d6d49420-f39b-4df7-a1dc-d59a935871db/",  # Bot Framework tenant
    "https://login.microsoftonline.com/d6d49420-f39b-4df7-a1dc-d59a935871db/v2.0",
]

# Google Chat OpenID configuration
GOOGLE_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_VALID_ISSUERS = [
    "chat@system.gserviceaccount.com",
    "https://accounts.google.com",
]


@dataclass
class JWTVerificationResult:
    """Result of JWT verification."""

    valid: bool
    claims: Dict[str, Any]
    error: Optional[str] = None


class JWTVerifier:
    """
    Verifies JWT tokens from chat platform webhooks.

    Caches JWKS clients to avoid repeated key fetches.
    """

    def __init__(self):
        """Initialize the verifier."""
        self._microsoft_jwks_client: Optional[Any] = None
        self._google_jwks_client: Optional[Any] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 3600  # Refresh keys every hour

    def _get_microsoft_jwks_client(self) -> Optional[Any]:
        """Get or create Microsoft JWKS client."""
        if not HAS_JWT:
            return None

        now = time.time()
        if self._microsoft_jwks_client is None or now - self._cache_time > self._cache_ttl:
            try:
                self._microsoft_jwks_client = PyJWKClient(MICROSOFT_JWKS_URI)
                self._cache_time = now
            except Exception as e:
                logger.warning(f"Failed to create Microsoft JWKS client: {e}")
                return None

        return self._microsoft_jwks_client

    def _get_google_jwks_client(self) -> Optional[Any]:
        """Get or create Google JWKS client."""
        if not HAS_JWT:
            return None

        now = time.time()
        if self._google_jwks_client is None or now - self._cache_time > self._cache_ttl:
            try:
                self._google_jwks_client = PyJWKClient(GOOGLE_JWKS_URI)
                self._cache_time = now
            except Exception as e:
                logger.warning(f"Failed to create Google JWKS client: {e}")
                return None

        return self._google_jwks_client

    def verify_microsoft_token(
        self,
        token: str,
        app_id: str,
    ) -> JWTVerificationResult:
        """
        Verify a Microsoft Bot Framework JWT token.

        Args:
            token: JWT token from Authorization header (without 'Bearer ' prefix)
            app_id: Expected audience (Bot application ID)

        Returns:
            JWTVerificationResult with validation status and claims
        """
        if not HAS_JWT:
            logger.error("PyJWT not available - cannot verify token (fail-closed)")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error="PyJWT library not available - install pyjwt for token verification",
            )

        jwks_client = self._get_microsoft_jwks_client()
        if jwks_client is None:
            logger.error("Could not create JWKS client - cannot verify token (fail-closed)")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error="JWKS client unavailable - cannot verify token signature",
            )

        try:
            # Get signing key from JWKS
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Decode and verify the token
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=app_id,
                issuer=MICROSOFT_VALID_ISSUERS,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                },
            )

            logger.debug(f"Microsoft token verified: iss={claims.get('iss')}")
            return JWTVerificationResult(valid=True, claims=claims)

        except PyJWTError as e:
            logger.warning(f"Microsoft token verification failed: {e}")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error verifying Microsoft token: {e}")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error=str(e),
            )

    def verify_google_token(
        self,
        token: str,
        project_id: Optional[str] = None,
    ) -> JWTVerificationResult:
        """
        Verify a Google Chat JWT token.

        Args:
            token: JWT token from Authorization header (without 'Bearer ' prefix)
            project_id: Expected audience (Google Cloud project number)

        Returns:
            JWTVerificationResult with validation status and claims
        """
        if not HAS_JWT:
            logger.error("PyJWT not available - cannot verify token (fail-closed)")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error="PyJWT library not available - install pyjwt for token verification",
            )

        jwks_client = self._get_google_jwks_client()
        if jwks_client is None:
            logger.error("Could not create JWKS client - cannot verify token (fail-closed)")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error="JWKS client unavailable - cannot verify token signature",
            )

        try:
            # Get signing key from JWKS
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Build decode options
            options = {
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
            }

            # Only verify audience if project_id provided
            decode_kwargs: Dict[str, Any] = {
                "algorithms": ["RS256"],
                "issuer": GOOGLE_VALID_ISSUERS,
                "options": options,
            }

            if project_id:
                options["verify_aud"] = True
                decode_kwargs["audience"] = project_id
            else:
                # SECURITY WARNING: Skipping audience validation allows tokens
                # intended for other applications to be accepted. This should
                # only be used in development. In production, always provide project_id.
                logger.warning(
                    "SECURITY: Skipping JWT audience validation - "
                    "provide project_id for secure token verification"
                )
                options["verify_aud"] = False

            # Decode and verify the token
            claims = jwt.decode(token, signing_key.key, **decode_kwargs)

            logger.debug(f"Google token verified: iss={claims.get('iss')}")
            return JWTVerificationResult(valid=True, claims=claims)

        except PyJWTError as e:
            logger.warning(f"Google token verification failed: {e}")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error=str(e),
            )
        except Exception as e:
            logger.error(f"Unexpected error verifying Google token: {e}")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error=str(e),
            )


# Singleton instance
_verifier: Optional[JWTVerifier] = None


def get_jwt_verifier() -> JWTVerifier:
    """Get or create the JWT verifier singleton."""
    global _verifier
    if _verifier is None:
        _verifier = JWTVerifier()
    return _verifier


def verify_teams_webhook(
    auth_header: str,
    app_id: str,
) -> bool:
    """
    Verify a Microsoft Teams webhook Authorization header.

    Args:
        auth_header: Full Authorization header value (e.g., "Bearer eyJ...")
        app_id: Bot application ID

    Returns:
        True if token is valid, False otherwise
    """
    if not auth_header.startswith("Bearer "):
        logger.warning("Invalid Authorization header format")
        return False

    token = auth_header[7:]  # Remove "Bearer " prefix
    verifier = get_jwt_verifier()
    result = verifier.verify_microsoft_token(token, app_id)

    if not result.valid and result.error:
        logger.warning(f"Teams webhook verification failed: {result.error}")

    return result.valid


def verify_google_chat_webhook(
    auth_header: str,
    project_id: Optional[str] = None,
) -> bool:
    """
    Verify a Google Chat webhook Authorization header.

    Args:
        auth_header: Full Authorization header value (e.g., "Bearer eyJ...")
        project_id: Google Cloud project number (optional)

    Returns:
        True if token is valid, False otherwise
    """
    if not auth_header.startswith("Bearer "):
        logger.warning("Invalid Authorization header format")
        return False

    token = auth_header[7:]  # Remove "Bearer " prefix
    verifier = get_jwt_verifier()
    result = verifier.verify_google_token(token, project_id)

    if not result.valid and result.error:
        logger.warning(f"Google Chat webhook verification failed: {result.error}")

    return result.valid


# Flag for checking JWT availability
__all__ = [
    "JWTVerifier",
    "JWTVerificationResult",
    "get_jwt_verifier",
    "verify_teams_webhook",
    "verify_google_chat_webhook",
    "HAS_JWT",
]
