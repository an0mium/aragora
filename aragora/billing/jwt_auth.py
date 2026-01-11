"""
JWT Authentication for User Sessions.

Provides JWT token generation, validation, and middleware for user authentication.
Integrates with the billing User model.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Token Blacklist (for revocation)
# =============================================================================


class TokenBlacklist:
    """
    Thread-safe in-memory token blacklist with TTL cleanup.

    Tokens are stored with their expiry time and automatically cleaned up
    when they would have expired anyway. This ensures the blacklist doesn't
    grow unbounded.

    For production deployments with multiple instances, consider using Redis
    or a shared database for the blacklist.
    """

    _instance: Optional["TokenBlacklist"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TokenBlacklist":
        """Singleton pattern for global blacklist."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, cleanup_interval: int = 300):
        """
        Initialize the blacklist.

        Args:
            cleanup_interval: Seconds between automatic cleanups (default 5 min)
        """
        if self._initialized:
            return
        self._blacklist: dict[str, float] = {}  # token_jti -> expiry_timestamp
        self._data_lock = threading.Lock()
        self._cleanup_interval = cleanup_interval
        self._last_cleanup = time.time()
        self._initialized = True
        logger.info("token_blacklist_initialized")

    def revoke(self, token_jti: str, expires_at: float) -> None:
        """
        Add a token to the blacklist.

        Args:
            token_jti: Token's unique identifier (jti claim or hash of token)
            expires_at: When the token would naturally expire (Unix timestamp)
        """
        with self._data_lock:
            self._blacklist[token_jti] = expires_at
            logger.info(f"token_revoked jti={token_jti[:16]}...")
            self._maybe_cleanup()

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a token by decoding and blacklisting it.

        Args:
            token: The JWT token string

        Returns:
            True if token was valid and revoked, False otherwise
        """
        payload = decode_jwt(token)
        if payload is None:
            return False
        # Use a hash of the token as the JTI if no jti claim
        token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
        self.revoke(token_jti, payload.exp)
        return True

    def is_revoked(self, token: str) -> bool:
        """
        Check if a token has been revoked.

        Args:
            token: The JWT token string

        Returns:
            True if token is in blacklist, False otherwise
        """
        token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
        with self._data_lock:
            return token_jti in self._blacklist

    def cleanup_expired(self) -> int:
        """
        Remove expired tokens from the blacklist.

        Returns:
            Number of tokens removed
        """
        now = time.time()
        with self._data_lock:
            expired = [k for k, v in self._blacklist.items() if v < now]
            for k in expired:
                del self._blacklist[k]
            if expired:
                logger.debug(f"token_blacklist_cleanup removed={len(expired)}")
            self._last_cleanup = now
            return len(expired)

    def _maybe_cleanup(self) -> None:
        """Run cleanup if enough time has passed."""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self.cleanup_expired()

    def size(self) -> int:
        """Get current blacklist size."""
        with self._data_lock:
            return len(self._blacklist)

    def clear(self) -> None:
        """Clear all revoked tokens (for testing)."""
        with self._data_lock:
            self._blacklist.clear()
            logger.info("token_blacklist_cleared")


# Global blacklist instance
_token_blacklist: Optional[TokenBlacklist] = None


def get_token_blacklist() -> TokenBlacklist:
    """Get the global token blacklist instance."""
    global _token_blacklist
    if _token_blacklist is None:
        _token_blacklist = TokenBlacklist()
    return _token_blacklist


def get_persistent_blacklist():
    """
    Get the persistent blacklist backend.

    This is the preferred method for production. Uses SQLite by default,
    or Redis for multi-instance deployments.

    Returns:
        BlacklistBackend instance (SQLite, Redis, or in-memory)
    """
    from aragora.storage.token_blacklist_store import get_blacklist_backend
    return get_blacklist_backend()


def revoke_token_persistent(token: str) -> bool:
    """
    Revoke a token using the persistent blacklist backend.

    Args:
        token: The JWT token string

    Returns:
        True if token was valid and revoked, False otherwise
    """
    payload = decode_jwt(token)
    if payload is None:
        return False
    token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
    backend = get_persistent_blacklist()
    backend.add(token_jti, payload.exp)
    logger.info(f"token_revoked_persistent jti={token_jti[:16]}...")
    return True


def is_token_revoked_persistent(token: str) -> bool:
    """
    Check if a token has been revoked using persistent backend.

    Args:
        token: The JWT token string

    Returns:
        True if token is revoked
    """
    token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
    backend = get_persistent_blacklist()
    return backend.contains(token_jti)

# Configuration
ARAGORA_ENVIRONMENT = os.environ.get("ARAGORA_ENVIRONMENT", "development")
JWT_SECRET = os.environ.get("ARAGORA_JWT_SECRET", "")
JWT_SECRET_PREVIOUS = os.environ.get("ARAGORA_JWT_SECRET_PREVIOUS", "")
# Unix timestamp when secret was rotated (for limiting previous secret validity)
JWT_SECRET_ROTATED_AT = os.environ.get("ARAGORA_JWT_SECRET_ROTATED_AT", "")
# How long previous secret remains valid after rotation (default: 24 hours)
JWT_ROTATION_GRACE_HOURS = int(os.environ.get("ARAGORA_JWT_ROTATION_GRACE_HOURS", "24"))
JWT_ALGORITHM = "HS256"
ALLOWED_ALGORITHMS = frozenset(["HS256"])  # Explicitly allowed algorithms
JWT_EXPIRY_HOURS = int(os.environ.get("ARAGORA_JWT_EXPIRY_HOURS", "24"))
REFRESH_TOKEN_EXPIRY_DAYS = int(os.environ.get("ARAGORA_REFRESH_TOKEN_EXPIRY_DAYS", "30"))

# Security constraints
MIN_SECRET_LENGTH = 32
MAX_ACCESS_TOKEN_HOURS = 168  # 7 days max
MAX_REFRESH_TOKEN_DAYS = 90  # 90 days max


def _is_production() -> bool:
    """Check if running in production environment.

    Conservative detection - treats any production-like environment as production
    to prevent security misconfigurations.
    """
    env = ARAGORA_ENVIRONMENT.lower()
    production_indicators = ["production", "prod", "live", "prd"]
    return any(indicator in env for indicator in production_indicators)


def _validate_security_config() -> None:
    """Validate security configuration at module load.

    Placeholder for future security configuration validation.
    Format-only API key validation has been removed for security.
    """
    # All API key validation now requires a user store lookup
    pass


def _validate_secret_strength(secret: str) -> bool:
    """Validate JWT secret meets minimum entropy requirements."""
    return len(secret) >= MIN_SECRET_LENGTH


def _get_secret() -> bytes:
    """
    Get JWT secret with strict validation.

    ARAGORA_JWT_SECRET must be set in all environments except pytest.
    This prevents issues with:
    - Load balancing (different instances need same secret)
    - Server restarts invalidating all tokens

    Raises:
        RuntimeError: If secret is missing or weak (except in pytest).
    """
    global JWT_SECRET
    running_under_pytest = "pytest" in sys.modules

    if not JWT_SECRET:
        if running_under_pytest:
            # Allow ephemeral secret only in test environments
            JWT_SECRET = base64.b64encode(os.urandom(32)).decode("utf-8")
            logger.debug("TEST MODE: Using ephemeral JWT secret")
        else:
            raise RuntimeError(
                "ARAGORA_JWT_SECRET must be set. "
                "Generate with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )

    if not _validate_secret_strength(JWT_SECRET):
        if running_under_pytest:
            logger.debug(f"TEST MODE: JWT secret is weak (< {MIN_SECRET_LENGTH} chars)")
        else:
            raise RuntimeError(
                f"ARAGORA_JWT_SECRET must be at least {MIN_SECRET_LENGTH} characters. "
                f"Current length: {len(JWT_SECRET)}"
            )

    return JWT_SECRET.encode("utf-8")


def _get_previous_secret() -> Optional[bytes]:
    """
    Get previous JWT secret for rotation support.

    Returns the previous secret only if:
    1. It meets minimum length requirements
    2. The rotation timestamp is within the grace period

    This prevents leaked old secrets from being exploitable indefinitely.
    """
    if not JWT_SECRET_PREVIOUS or len(JWT_SECRET_PREVIOUS) < MIN_SECRET_LENGTH:
        return None

    # Check rotation timestamp if set
    if JWT_SECRET_ROTATED_AT:
        try:
            rotated_at = int(JWT_SECRET_ROTATED_AT)
            grace_seconds = JWT_ROTATION_GRACE_HOURS * 3600
            if time.time() - rotated_at > grace_seconds:
                logger.debug(
                    f"jwt_previous_secret_expired: rotated {JWT_ROTATION_GRACE_HOURS}+ hours ago"
                )
                return None
        except ValueError:
            logger.warning(
                "jwt_previous_secret: invalid ARAGORA_JWT_SECRET_ROTATED_AT format, "
                "expected Unix timestamp"
            )
            # In production, reject previous secret if timestamp is invalid
            if _is_production():
                return None

    return JWT_SECRET_PREVIOUS.encode("utf-8")


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
        )

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.exp

    @property
    def user_id(self) -> str:
        """Alias for sub (subject = user ID)."""
        return self.sub


def create_access_token(
    user_id: str,
    email: str,
    org_id: Optional[str] = None,
    role: str = "member",
    expiry_hours: Optional[int] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        user_id: User ID (sub claim)
        email: User email
        org_id: Organization ID
        role: User role in organization
        expiry_hours: Token expiry in hours (default from config, max 168h/7 days)

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
    )

    return _encode_jwt(payload)


def create_refresh_token(
    user_id: str,
    expiry_days: Optional[int] = None,
) -> str:
    """
    Create a JWT refresh token.

    Args:
        user_id: User ID
        expiry_days: Token expiry in days (default from config, max 90 days)

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
    )

    return _encode_jwt(payload)


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
        _get_secret(),
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
        except Exception:
            logger.debug("jwt_decode_failed: invalid header encoding")
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
            _get_secret(),
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        signature_valid = hmac.compare_digest(expected_signature, actual_signature)

        # If current secret fails, try previous secret (for rotation)
        if not signature_valid:
            prev_secret = _get_previous_secret()
            if prev_secret:
                expected_signature_prev = hmac.new(
                    prev_secret,
                    message.encode("utf-8"),
                    hashlib.sha256,
                ).digest()
                signature_valid = hmac.compare_digest(
                    expected_signature_prev, actual_signature
                )
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


def validate_access_token(token: str, use_persistent_blacklist: bool = True) -> Optional[JWTPayload]:
    """
    Validate an access token.

    Checks:
    1. Token structure and signature
    2. Token expiration
    3. Token type is "access"
    4. Token is not blacklisted (revoked)

    Args:
        token: JWT token string
        use_persistent_blacklist: If True, also check persistent blacklist (default)

    Returns:
        JWTPayload if valid access token, None otherwise
    """
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
        token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
        blacklist.revoke(token_jti, payload.exp)
        return None

    return payload


def validate_refresh_token(token: str, use_persistent_blacklist: bool = True) -> Optional[JWTPayload]:
    """
    Validate a refresh token.

    Checks:
    1. Token structure and signature
    2. Token expiration
    3. Token type is "refresh"
    4. Token is not blacklisted (revoked)

    Args:
        token: JWT refresh token string
        use_persistent_blacklist: If True, also check persistent blacklist (default)

    Returns:
        JWTPayload if valid refresh token, None otherwise
    """
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
        token_jti = hashlib.sha256(token.encode()).hexdigest()[:32]
        blacklist.revoke(token_jti, payload.exp)
        return None

    return payload


@dataclass
class UserAuthContext:
    """
    User authentication context for handlers.

    Extended version of AuthContext with user-specific information.
    """

    authenticated: bool = False
    user_id: Optional[str] = None
    email: Optional[str] = None
    org_id: Optional[str] = None
    role: str = "member"
    token_type: str = "none"  # none, access, api_key
    client_ip: Optional[str] = None
    error_reason: Optional[str] = None  # Set when auth fails with a specific reason

    @property
    def is_authenticated(self) -> bool:
        """Alias for authenticated."""
        return self.authenticated

    @property
    def is_owner(self) -> bool:
        """Check if user is org owner."""
        return self.role == "owner"

    @property
    def is_admin(self) -> bool:
        """Check if user is admin or owner."""
        return self.role in ("owner", "admin")


def extract_user_from_request(handler: Any, user_store=None) -> UserAuthContext:
    """
    Extract user authentication from a request.

    Checks for:
    1. Bearer token (JWT)
    2. API key (ara_xxx)

    Args:
        handler: HTTP request handler
        user_store: Optional user store for API key validation against database.
                    API keys require a store unless ARAGORA_ALLOW_FORMAT_ONLY_API_KEYS=1.

    Returns:
        UserAuthContext with authentication info
    """
    from aragora.server.middleware.auth import extract_client_ip

    context = UserAuthContext(
        authenticated=False,
        client_ip=extract_client_ip(handler),
    )

    if handler is None:
        return context

    # Get authorization header
    auth_header = ""
    if hasattr(handler, "headers"):
        auth_header = handler.headers.get("Authorization", "")

    if not auth_header:
        return context

    # Check for Bearer token (JWT)
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]

        # Check if it's an API key
        if token.startswith("ara_"):
            return _validate_api_key(token, context, user_store)

        # Validate as JWT
        payload = validate_access_token(token)
        if payload:
            context.authenticated = True
            context.user_id = payload.user_id
            context.email = payload.email
            context.org_id = payload.org_id
            context.role = payload.role
            context.token_type = "access"

    return context


def _validate_api_key(api_key: str, context: UserAuthContext, user_store=None) -> UserAuthContext:
    """
    Validate an API key and populate context.

    Performs format validation and optional database lookup if user_store is provided.

    API key format: ara_<random_string> (minimum 15 chars total)

    Args:
        api_key: API key string (expected format: ara_xxxxx)
        context: Context to populate
        user_store: Optional user store for database validation

    Returns:
        Updated context with authentication status
    """
    # Format validation (basic security check)
    if not api_key.startswith("ara_") or len(api_key) < 15:
        logger.warning(f"api_key_invalid_format key_prefix={api_key[:8]}...")
        context.authenticated = False
        return context

    # Database lookup if store available
    if user_store is not None:
        try:
            # Look up user by API key
            user = user_store.get_user_by_api_key(api_key)
            if user is None:
                logger.warning(f"api_key_not_found key_prefix={api_key[:8]}...")
                context.authenticated = False
                return context

            # Check if user is active
            if not user.is_active:
                logger.warning(f"api_key_user_inactive user_id={user.id}")
                context.authenticated = False
                return context

            # Populate context with user info
            context.authenticated = True
            context.user_id = user.id
            context.email = user.email
            context.org_id = user.org_id
            context.role = user.role
            context.token_type = "api_key"
            logger.debug(f"api_key_validated user_id={user.id}")
            return context

        except Exception as e:
            logger.error(f"api_key_validation_error: {e}")
            context.authenticated = False
            return context

    # No user store available - API key auth requires database validation
    # This is an intentional security requirement - format-only validation was removed
    logger.error(
        "api_key_validation_failed: user_store is required for API key authentication. "
        "Ensure UserStore is initialized before processing API key auth requests."
    )
    context.authenticated = False
    context.error_reason = "API key authentication requires server configuration"
    return context


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
) -> TokenPair:
    """
    Create a new access/refresh token pair.

    Args:
        user_id: User ID
        email: User email
        org_id: Organization ID
        role: User role

    Returns:
        TokenPair with access and refresh tokens
    """
    access = create_access_token(user_id, email, org_id, role)
    refresh = create_refresh_token(user_id)
    return TokenPair(access, refresh)


# =============================================================================
# MFA Pending Token (short-lived token for MFA flow)
# =============================================================================

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
    "UserAuthContext",
    "TokenPair",
    "TokenBlacklist",
    "get_token_blacklist",
    "get_persistent_blacklist",
    "revoke_token_persistent",
    "is_token_revoked_persistent",
    "create_access_token",
    "create_refresh_token",
    "decode_jwt",
    "validate_access_token",
    "validate_refresh_token",
    "extract_user_from_request",
    "create_token_pair",
    # MFA pending token
    "create_mfa_pending_token",
    "validate_mfa_pending_token",
    # Security configuration
    "ARAGORA_ENVIRONMENT",
    "MIN_SECRET_LENGTH",
    "ALLOWED_ALGORITHMS",
]

# Validate security configuration at module load
_validate_security_config()
