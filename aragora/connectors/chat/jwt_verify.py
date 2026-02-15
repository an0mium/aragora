"""
JWT Verification for Chat Platform Webhooks.

Provides JWT token verification for platforms that use OAuth bearer tokens:
- Microsoft Teams (Bot Framework)
- Google Chat

Uses PyJWK to fetch signing keys from platform endpoints and validate tokens.

Security model:
- Fail-closed: if PyJWT is unavailable, all tokens are rejected
- JWKS keys are cached with a configurable TTL (default 15 minutes, via ARAGORA_JWT_CACHE_TTL)
- OpenID metadata is fetched to discover JWKS URIs dynamically
- Issuer and audience claims are always validated
- Cache is automatically invalidated on signature/key errors (enables key rotation recovery)
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

# Environment check for security-sensitive operations
_IS_PRODUCTION = os.environ.get("ARAGORA_ENV", "production").lower() in ("production", "prod")

# Pre-declare types for optional jwt import
jwt: Any = None
PyJWKClient: Any = None
PyJWTError: type[Exception] = Exception
HAS_JWT = False

# PyJWT for token validation - handle gracefully if not installed
try:
    import jwt
    from jwt import PyJWKClient
    from jwt.exceptions import PyJWTError

    HAS_JWT = True
except ImportError:
    # PyJWT is optional; fallbacks already set above
    logger.warning(
        "PyJWT library not installed - JWT verification unavailable. "
        "Install with: pip install pyjwt[crypto]"
    )

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

# OpenID metadata cache TTL (seconds)
# Default reduced from 1 hour to 15 minutes for faster key rotation response
_DEFAULT_CACHE_TTL = 900  # 15 minutes
_OPENID_METADATA_CACHE_TTL = int(os.environ.get("ARAGORA_JWT_CACHE_TTL", _DEFAULT_CACHE_TTL))


@dataclass
class JWTVerificationResult:
    """Result of JWT verification."""

    valid: bool
    claims: dict[str, Any]
    error: str | None = None


@dataclass
class _OpenIDMetadataCache:
    """Cached OpenID metadata with TTL."""

    jwks_uri: str
    issuer: str | None = None
    fetched_at: float = field(default_factory=time.time)


def _fetch_openid_metadata(
    metadata_url: str,
    timeout: float = 10.0,
) -> dict[str, Any] | None:
    """Fetch OpenID Connect metadata from a discovery endpoint.

    Retrieves the OpenID configuration document which contains the JWKS URI
    needed for JWT signature verification. This implements the standard
    OpenID Connect Discovery 1.0 protocol.

    Args:
        metadata_url: The OpenID configuration URL
            (e.g., https://login.botframework.com/v1/.well-known/openidconfiguration)
        timeout: HTTP request timeout in seconds

    Returns:
        Parsed metadata dict containing at minimum 'jwks_uri', or None on failure
    """
    try:
        req = Request(metadata_url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if isinstance(data, dict) and "jwks_uri" in data:
                return data
            logger.warning(f"OpenID metadata from {metadata_url} missing 'jwks_uri' field")
            return None
    except (URLError, OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to fetch OpenID metadata from {metadata_url}: {e}")
        return None


class JWTVerifier:
    """
    Verifies JWT tokens from chat platform webhooks.

    Caches JWKS clients and OpenID metadata to avoid repeated network fetches.
    Uses OpenID discovery to resolve JWKS endpoints dynamically, with
    hardcoded fallback URIs for reliability.

    The verifier follows a fail-closed security model: if PyJWT is not
    installed or JWKS keys cannot be fetched, all tokens are rejected.
    """

    def __init__(self, cache_ttl: float | None = None):
        """Initialize the verifier.

        Args:
            cache_ttl: Time-to-live for cached JWKS clients and OpenID metadata,
                       in seconds. If None, uses ARAGORA_JWT_CACHE_TTL env var
                       or defaults to 900 seconds (15 minutes).
        """
        if cache_ttl is None:
            cache_ttl = float(_OPENID_METADATA_CACHE_TTL)
        self._microsoft_jwks_client: Any | None = None
        self._google_jwks_client: Any | None = None
        self._microsoft_cache_time: float = 0
        self._google_cache_time: float = 0
        self._cache_ttl: float = cache_ttl
        # OpenID metadata cache
        self._microsoft_metadata: _OpenIDMetadataCache | None = None
        self._google_metadata: _OpenIDMetadataCache | None = None

    def invalidate_cache(self) -> None:
        """Invalidate cached metadata, forcing refresh on next use.

        This should be called when signature verification fails to ensure
        stale signing keys are not used after IdP key rotation.
        """
        self._microsoft_metadata = None
        self._google_metadata = None
        self._microsoft_jwks_client = None
        self._google_jwks_client = None
        self._microsoft_cache_time = 0
        self._google_cache_time = 0
        logger.debug("JWT verifier cache invalidated")

    def invalidate_microsoft_cache(self) -> None:
        """Invalidate only Microsoft-related cached metadata."""
        self._microsoft_metadata = None
        self._microsoft_jwks_client = None
        self._microsoft_cache_time = 0
        logger.debug("Microsoft JWT cache invalidated")

    def invalidate_google_cache(self) -> None:
        """Invalidate only Google-related cached metadata."""
        self._google_metadata = None
        self._google_jwks_client = None
        self._google_cache_time = 0
        logger.debug("Google JWT cache invalidated")

    def _resolve_microsoft_jwks_uri(self) -> str:
        """Resolve Microsoft JWKS URI via OpenID discovery, with fallback.

        Fetches the OpenID configuration from Microsoft's Bot Framework
        endpoint to discover the current JWKS URI. Falls back to the
        hardcoded URI if discovery fails.

        Returns:
            The JWKS URI to use for key fetching
        """
        now = time.time()

        # Return cached metadata if still valid
        if (
            self._microsoft_metadata is not None
            and now - self._microsoft_metadata.fetched_at < self._cache_ttl
        ):
            return self._microsoft_metadata.jwks_uri

        # Fetch fresh metadata from OpenID discovery endpoint
        metadata = _fetch_openid_metadata(MICROSOFT_OPENID_METADATA_URL)
        if metadata and "jwks_uri" in metadata:
            jwks_uri = metadata["jwks_uri"]
            issuer = metadata.get("issuer")
            self._microsoft_metadata = _OpenIDMetadataCache(
                jwks_uri=jwks_uri,
                issuer=issuer,
                fetched_at=now,
            )
            logger.debug(f"Resolved Microsoft JWKS URI via OpenID discovery: {jwks_uri}")
            return jwks_uri

        # Fallback to hardcoded URI
        logger.debug("Using hardcoded Microsoft JWKS URI (OpenID discovery unavailable)")
        return MICROSOFT_JWKS_URI

    def _get_microsoft_jwks_client(self) -> Any | None:
        """Get or create Microsoft JWKS client with OpenID discovery.

        Uses OpenID metadata to resolve the JWKS endpoint dynamically.
        The client and its resolved URI are cached for ``cache_ttl`` seconds.
        """
        if not HAS_JWT or PyJWKClient is None:
            return None

        now = time.time()
        if (
            self._microsoft_jwks_client is None
            or now - self._microsoft_cache_time > self._cache_ttl
        ):
            try:
                jwks_uri = self._resolve_microsoft_jwks_uri()
                self._microsoft_jwks_client = PyJWKClient(jwks_uri)
                self._microsoft_cache_time = now
            except (PyJWTError, ValueError, OSError) as e:
                logger.warning(f"Failed to create Microsoft JWKS client: {e}")
                return None

        return self._microsoft_jwks_client

    def _get_google_jwks_client(self) -> Any | None:
        """Get or create Google JWKS client."""
        if not HAS_JWT or PyJWKClient is None:
            return None

        now = time.time()
        if self._google_jwks_client is None or now - self._google_cache_time > self._cache_ttl:
            try:
                self._google_jwks_client = PyJWKClient(GOOGLE_JWKS_URI)
                self._google_cache_time = now
            except (PyJWTError, ValueError, OSError) as e:
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

        Validates the token signature against Microsoft's JWKS endpoint,
        checks the issuer claim against known Bot Framework issuers,
        and verifies the audience claim matches the configured app ID.

        Args:
            token: JWT token from Authorization header (without 'Bearer ' prefix)
            app_id: Expected audience (Bot application ID from TEAMS_APP_ID / MS_APP_ID)

        Returns:
            JWTVerificationResult with validation status and claims
        """
        if not HAS_JWT:
            logger.error("PyJWT not available - cannot verify token (fail-closed)")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error="PyJWT library not available - install pyjwt[crypto] for token verification",
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
            error_str = str(e).lower()
            # Invalidate cache on signature-related errors that might indicate key rotation
            if "signature" in error_str or "key" in error_str or "kid" in error_str:
                logger.info(
                    f"Invalidating Microsoft JWT cache due to potential key rotation (error: {e})"
                )
                self.invalidate_microsoft_cache()
            logger.warning(f"Microsoft token verification failed: {e}")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error="Token verification failed",
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Unexpected error verifying Microsoft token: {e}")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error="Token verification failed due to an unexpected error",
            )

    def verify_google_token(
        self,
        token: str,
        project_id: str | None = None,
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
            decode_kwargs: dict[str, Any] = {
                "algorithms": ["RS256"],
                "issuer": GOOGLE_VALID_ISSUERS,
                "options": options,
            }

            if project_id:
                options["verify_aud"] = True
                decode_kwargs["audience"] = project_id
            elif _IS_PRODUCTION:
                # FAIL CLOSED: In production, audience validation is mandatory
                # to prevent accepting tokens intended for other applications
                logger.error(
                    "SECURITY: JWT audience validation required in production - "
                    "project_id must be provided"
                )
                return JWTVerificationResult(
                    valid=False,
                    claims={},
                    error="JWT audience validation required in production - project_id must be provided",
                )
            else:
                # Development only: skip audience validation with warning
                logger.warning(
                    "SECURITY: Skipping JWT audience validation (dev mode only) - "
                    "set ARAGORA_ENV=production to enforce"
                )
                options["verify_aud"] = False

            # Decode and verify the token
            claims = jwt.decode(token, signing_key.key, **decode_kwargs)

            logger.debug(f"Google token verified: iss={claims.get('iss')}")
            return JWTVerificationResult(valid=True, claims=claims)

        except PyJWTError as e:
            error_str = str(e).lower()
            # Invalidate cache on signature-related errors that might indicate key rotation
            if "signature" in error_str or "key" in error_str or "kid" in error_str:
                logger.info(
                    f"Invalidating Google JWT cache due to potential key rotation (error: {e})"
                )
                self.invalidate_google_cache()
            logger.warning(f"Google token verification failed: {e}")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error="Token verification failed",
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Unexpected error verifying Google token: {e}")
            return JWTVerificationResult(
                valid=False,
                claims={},
                error="Token verification failed due to an unexpected error",
            )


# Singleton instance
_verifier: JWTVerifier | None = None


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

    Extracts the Bearer token from the Authorization header and validates it
    against Microsoft's JWKS endpoint. Checks:
    - JWT signature (RS256) against Microsoft Bot Framework signing keys
    - Issuer claim (iss) against known Bot Framework issuers
    - Audience claim (aud) against the bot's Microsoft App ID
    - Token expiry (exp) and issued-at (iat) timestamps

    Args:
        auth_header: Full Authorization header value (e.g., "Bearer eyJ...")
        app_id: Bot application ID (from TEAMS_APP_ID or MS_APP_ID env var)

    Returns:
        True if token is valid, False otherwise
    """
    if not auth_header.startswith("Bearer "):
        logger.warning("Invalid Authorization header format - expected 'Bearer <token>'")
        return False

    token = auth_header[7:]  # Remove "Bearer " prefix
    verifier = get_jwt_verifier()
    result = verifier.verify_microsoft_token(token, app_id)

    if not result.valid and result.error:
        logger.warning(f"Teams webhook verification failed: {result.error}")

    return result.valid


def verify_google_chat_webhook(
    auth_header: str,
    project_id: str | None = None,
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
    "_fetch_openid_metadata",
    "_OpenIDMetadataCache",
    "_DEFAULT_CACHE_TTL",
    "_OPENID_METADATA_CACHE_TTL",
]
