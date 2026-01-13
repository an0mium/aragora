"""
JWT Token Creation, Encoding, and Validation.

Provides JWT token handling including:
- JWTPayload dataclass
- Token encoding and decoding
- Access and refresh token creation
- Token validation with blacklist checking
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from .config import (
    ALLOWED_ALGORITHMS,
    JWT_ALGORITHM,
    JWT_EXPIRY_HOURS,
    MAX_ACCESS_TOKEN_HOURS,
    MAX_REFRESH_TOKEN_DAYS,
    REFRESH_TOKEN_EXPIRY_DAYS,
    get_previous_secret,
    get_secret,
)

logger = logging.getLogger(__name__)


def _base64url_encode(data: bytes) -> str:
    """Base64 URL-safe encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("utf-8")


def _base64url_decode(data: str) -> bytes:
    """Base64 URL-safe decode with padding restoration."""
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += "=" * padding
    return base64.urlsafe_b64decode(data)


@dataclass
class JWTPayload:
    """JWT token payload."""

    sub: str  # Subject (user ID)
    email: str
    org_id: Optional[str]
    role: str
    iat: int  # Issued at (Unix timestamp)
    exp: int  # Expiration (Unix timestamp)
    type: str = "access"  # access or refresh
    tv: int = 1  # Token version - for logout-all functionality

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sub": self.sub,
            "email": self.email,
            "org_id": self.org_id,
            "role": self.role,
            "iat": self.iat,
            "exp": self.exp,
            "type": self.type,
            "tv": self.tv,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JWTPayload":
        """Create from dictionary."""
        return cls(
            sub=data.get("sub", ""),
            email=data.get("email", ""),
            org_id=data.get("org_id"),
            role=data.get("role", "member"),
            iat=data.get("iat", 0),
            exp=data.get("exp", 0),
            type=data.get("type", "access"),
            tv=data.get("tv", 1),
        )

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.exp

    @property
    def user_id(self) -> str:
        """Alias for sub (subject = user ID)."""
        return self.sub


def _encode_jwt(payload: JWTPayload) -> str:
    """
    Encode a JWT token.

    Args:
        payload: Token payload

    Returns:
        Encoded JWT string
    """
    # Header
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    header_b64 = _base64url_encode(json.dumps(header).encode("utf-8"))

    # Payload
    payload_b64 = _base64url_encode(json.dumps(payload.to_dict()).encode("utf-8"))

    # Signature
    message = f"{header_b64}.{payload_b64}"
    signature = hmac.new(
        get_secret(),
        message.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature_b64 = _base64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_jwt(token: str) -> Optional[JWTPayload]:
    """
    Decode and validate a JWT token.

    Validates:
    1. Token format (3 parts)
    2. Algorithm is in ALLOWED_ALGORITHMS (prevents algorithm confusion attacks)
    3. Signature using current secret (with fallback to previous for rotation)
    4. Token expiration

    Args:
        token: JWT token string

    Returns:
        JWTPayload if valid, None otherwise
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            logger.debug("jwt_decode_failed: invalid format")
            return None

        header_b64, payload_b64, signature_b64 = parts

        # SECURITY: Validate algorithm before signature verification
        try:
            header_json = _base64url_decode(header_b64).decode("utf-8")
            header = json.loads(header_json)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.debug("jwt_decode_failed: invalid header encoding: %s", e)
            return None

        token_alg = header.get("alg", "")

        # Reject 'none' algorithm attack and other disallowed algorithms
        if token_alg not in ALLOWED_ALGORITHMS:
            logger.warning(f"jwt_decode_failed: disallowed algorithm '{token_alg}'")
            return None

        if token_alg != JWT_ALGORITHM:
            logger.warning(
                f"jwt_decode_failed: algorithm mismatch "
                f"(expected {JWT_ALGORITHM}, got {token_alg})"
            )
            return None

        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        actual_signature = _base64url_decode(signature_b64)

        # Try current secret first
        expected_signature = hmac.new(
            get_secret(),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        signature_valid = hmac.compare_digest(expected_signature, actual_signature)

        # If current secret fails, try previous secret (for rotation)
        if not signature_valid:
            prev_secret = get_previous_secret()
            if prev_secret:
                expected_signature_prev = hmac.new(
                    prev_secret,
                    message.encode("utf-8"),
                    hashlib.sha256,
                ).digest()
                signature_valid = hmac.compare_digest(expected_signature_prev, actual_signature)
                if signature_valid:
                    logger.debug("jwt_decode: validated with previous secret (rotation)")

        if not signature_valid:
            logger.debug("jwt_decode_failed: invalid signature")
            return None

        # Decode payload
        try:
            payload_json = _base64url_decode(payload_b64).decode("utf-8")
            payload_data = json.loads(payload_json)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(f"jwt_decode_failed: malformed payload - {type(e).__name__}")
            return None

        try:
            payload = JWTPayload.from_dict(payload_data)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"jwt_decode_failed: invalid payload structure - {type(e).__name__}")
            return None

        # Check expiration
        if payload.is_expired:
            logger.info("jwt_decode_failed: token expired")
            return None

        return payload

    except Exception as e:
        # Catch-all for unexpected errors - log at warning level for visibility
        logger.warning(f"jwt_decode_failed: unexpected error - {type(e).__name__}: {e}")
        return None


def create_access_token(
    user_id: str,
    email: str,
    org_id: Optional[str] = None,
    role: str = "member",
    expiry_hours: Optional[int] = None,
    token_version: int = 1,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User ID (sub claim)
        email: User email
        org_id: Organization ID
        role: User role in organization
        expiry_hours: Token expiry in hours (default from config, max 168h/7 days)
        token_version: User's token version for logout-all support

    Returns:
        JWT token string
    """
    if expiry_hours is None:
        expiry_hours = JWT_EXPIRY_HOURS

    # Enforce expiry bounds
    if expiry_hours > MAX_ACCESS_TOKEN_HOURS:
        logger.warning(
            f"Token expiry {expiry_hours}h exceeds max {MAX_ACCESS_TOKEN_HOURS}h, capping"
        )
        expiry_hours = MAX_ACCESS_TOKEN_HOURS
    if expiry_hours < 1:
        expiry_hours = 1  # Minimum 1 hour

    now = int(time.time())
    exp = now + (expiry_hours * 3600)

    payload = JWTPayload(
        sub=user_id,
        email=email,
        org_id=org_id,
        role=role,
        iat=now,
        exp=exp,
        type="access",
        tv=token_version,
    )

    return _encode_jwt(payload)


def create_refresh_token(
    user_id: str,
    expiry_days: Optional[int] = None,
    token_version: int = 1,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        user_id: User ID
        expiry_days: Token expiry in days (default from config, max 90 days)
        token_version: User's token version for logout-all support

    Returns:
        JWT refresh token string
    """
    if expiry_days is None:
        expiry_days = REFRESH_TOKEN_EXPIRY_DAYS

    # Enforce expiry bounds
    if expiry_days > MAX_REFRESH_TOKEN_DAYS:
        logger.warning(
            f"Refresh token expiry {expiry_days}d exceeds max {MAX_REFRESH_TOKEN_DAYS}d, capping"
        )
        expiry_days = MAX_REFRESH_TOKEN_DAYS
    if expiry_days < 1:
        expiry_days = 1  # Minimum 1 day

    now = int(time.time())
    exp = now + (expiry_days * 86400)

    payload = JWTPayload(
        sub=user_id,
        email="",
        org_id=None,
        role="",
        iat=now,
        exp=exp,
        type="refresh",
        tv=token_version,
    )

    return _encode_jwt(payload)


def validate_access_token(
    token: str,
    use_persistent_blacklist: bool = True,
    user_store: Optional[Any] = None,
) -> Optional[JWTPayload]:
    """
    Validate an access token.

    Checks:
    1. Token structure and signature
    2. Token expiration
    3. Token type is "access"
    4. Token is not blacklisted (revoked)
    5. Token version matches user's current version (if user_store provided)

    Args:
        token: JWT token string
        use_persistent_blacklist: If True, also check persistent blacklist (default)
        user_store: Optional UserStore for token version validation (logout-all support)

    Returns:
        JWTPayload if valid access token, None otherwise
    """
    from .blacklist import (
        get_token_blacklist,
        is_token_revoked_persistent,
    )

    payload = decode_jwt(token)
    if payload is None:
        return None
    if payload.type != "access":
        logger.debug("jwt_validate_failed: not an access token")
        return None

    # Check if token has been revoked (in-memory cache)
    blacklist = get_token_blacklist()
    if blacklist.is_revoked(token):
        logger.debug("jwt_validate_failed: token revoked (memory)")
        return None

    # Also check persistent blacklist for multi-instance consistency
    if use_persistent_blacklist and is_token_revoked_persistent(token):
        logger.debug("jwt_validate_failed: token revoked (persistent)")
        # Add to in-memory cache for faster subsequent checks
        import hashlib

        token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
        blacklist.revoke(token_jti, payload.exp)
        return None

    # Check token version against user's current version (logout-all support)
    if user_store is not None:
        try:
            user = user_store.get_user_by_id(payload.user_id)
            if user is not None:
                user_token_version = getattr(user, "token_version", 1)
                if payload.tv < user_token_version:
                    logger.debug(
                        f"jwt_validate_failed: token version mismatch "
                        f"(token={payload.tv}, user={user_token_version})"
                    )
                    return None
        except Exception as e:
            logger.warning(f"jwt_validate_failed: error checking token version - {e}")
            # Continue validation - don't block on store errors

    return payload


def validate_refresh_token(
    token: str,
    use_persistent_blacklist: bool = True,
    user_store: Optional[Any] = None,
) -> Optional[JWTPayload]:
    """
    Validate a refresh token.

    Checks:
    1. Token structure and signature
    2. Token expiration
    3. Token type is "refresh"
    4. Token is not blacklisted (revoked)
    5. Token version matches user's current version (if user_store provided)

    Args:
        token: JWT refresh token string
        use_persistent_blacklist: If True, also check persistent blacklist (default)
        user_store: Optional UserStore for token version validation (logout-all support)

    Returns:
        JWTPayload if valid refresh token, None otherwise
    """
    from .blacklist import (
        get_token_blacklist,
        is_token_revoked_persistent,
    )

    payload = decode_jwt(token)
    if payload is None:
        return None
    if payload.type != "refresh":
        logger.debug("jwt_validate_failed: not a refresh token")
        return None

    # Check if token has been revoked (in-memory cache)
    blacklist = get_token_blacklist()
    if blacklist.is_revoked(token):
        logger.debug("jwt_validate_failed: refresh token revoked (memory)")
        return None

    # Also check persistent blacklist for multi-instance consistency
    if use_persistent_blacklist and is_token_revoked_persistent(token):
        logger.debug("jwt_validate_failed: refresh token revoked (persistent)")
        import hashlib

        token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
        blacklist.revoke(token_jti, payload.exp)
        return None

    # Check token version against user's current version (logout-all support)
    if user_store is not None:
        try:
            user = user_store.get_user_by_id(payload.user_id)
            if user is not None:
                user_token_version = getattr(user, "token_version", 1)
                if payload.tv < user_token_version:
                    logger.debug(
                        f"jwt_validate_failed: refresh token version mismatch "
                        f"(token={payload.tv}, user={user_token_version})"
                    )
                    return None
        except Exception as e:
            logger.warning(f"jwt_validate_failed: error checking token version - {e}")

    return payload


class TokenPair:
    """Access and refresh token pair."""

    def __init__(self, access_token: str, refresh_token: str):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_type = "Bearer"
        self.expires_in = JWT_EXPIRY_HOURS * 3600

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
        }


def create_token_pair(
    user_id: str,
    email: str,
    org_id: Optional[str] = None,
    role: str = "member",
    token_version: int = 1,
) -> TokenPair:
    """
    Create a new access/refresh token pair.

    Args:
        user_id: User ID
        email: User email
        org_id: Organization ID
        role: User role
        token_version: User's token version for logout-all support

    Returns:
        TokenPair with access and refresh tokens
    """
    access = create_access_token(user_id, email, org_id, role, token_version=token_version)
    refresh = create_refresh_token(user_id, token_version=token_version)
    return TokenPair(access, refresh)


# MFA Pending Token (short-lived token for MFA flow)
MFA_PENDING_TOKEN_EXPIRY_MINUTES = 5  # Short-lived for security


def create_mfa_pending_token(user_id: str, email: str) -> str:
    """
    Create a short-lived token for MFA pending state.

    This token is issued after password verification when MFA is enabled.
    It can only be exchanged for real tokens after MFA verification.

    Args:
        user_id: User ID
        email: User email

    Returns:
        JWT token string with type="mfa_pending"
    """
    now = int(time.time())
    exp = now + (MFA_PENDING_TOKEN_EXPIRY_MINUTES * 60)

    payload = JWTPayload(
        sub=user_id,
        email=email,
        org_id=None,
        role="",
        iat=now,
        exp=exp,
        type="mfa_pending",
    )

    return _encode_jwt(payload)


def validate_mfa_pending_token(token: str) -> Optional[JWTPayload]:
    """
    Validate an MFA pending token.

    Args:
        token: JWT token string

    Returns:
        JWTPayload if valid and type is mfa_pending, None otherwise
    """
    from .blacklist import get_token_blacklist

    # Check blacklist first to prevent replay attacks
    blacklist = get_token_blacklist()
    if blacklist.is_revoked(token):
        logger.debug("mfa_pending_token_already_used")
        return None

    payload = decode_jwt(token)
    if payload is None:
        return None

    if payload.type != "mfa_pending":
        logger.warning(f"mfa_pending_token_invalid_type: got {payload.type}")
        return None

    return payload


__all__ = [
    "JWTPayload",
    "TokenPair",
    "decode_jwt",
    "create_access_token",
    "create_refresh_token",
    "validate_access_token",
    "validate_refresh_token",
    "create_token_pair",
    "create_mfa_pending_token",
    "validate_mfa_pending_token",
    # Internal utilities (exposed for testing)
    "_base64url_encode",
    "_base64url_decode",
]
