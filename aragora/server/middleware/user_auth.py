"""
Supabase JWT Authentication Middleware - Core Implementation.

Provides comprehensive user authentication via Supabase Auth JWTs.
Supports both session tokens (browser) and API keys (programmatic).

This is the primary authentication implementation. The companion auth.py
module provides simplified decorator wrappers (@require_auth, @optional_auth)
for basic token validation. Both modules are exported through __init__.py.

Module Structure:
- auth_v2.py (this file): Full implementation (User, Workspace, JWT, Supabase)
- auth.py: Simple decorators for basic auth flows

Key exports:
- User, Workspace, APIKey: Data classes for authenticated entities
- require_user, require_admin: Decorators requiring user authentication
- authenticate_request: Function to authenticate a request
- get_current_user: Get the authenticated user from context

Usage:
    from aragora.server.middleware import require_user, get_current_user, User

    @require_user
    async def protected_endpoint(request, user: User):
        return {"user_id": user.id}
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Protocol

if TYPE_CHECKING:
    pass


# Stub exception classes for when PyJWT is not installed
# Defined unconditionally so type checker sees them
class _ExpiredSignatureError(Exception):
    """Stub for jwt.exceptions.ExpiredSignatureError."""

    pass


class _InvalidSignatureError(Exception):
    """Stub for jwt.exceptions.InvalidSignatureError."""

    pass


class _DecodeError(Exception):
    """Stub for jwt.exceptions.DecodeError."""

    pass


class _InvalidTokenError(Exception):
    """Stub for jwt.exceptions.InvalidTokenError."""

    pass


class _InvalidAudienceError(Exception):
    """Stub for jwt.exceptions.InvalidAudienceError."""

    pass


class _JWTModuleProtocol(Protocol):
    """Protocol for the jwt module to satisfy type checker."""

    def decode(
        self,
        jwt: str,
        key: str,
        algorithms: List[str],
        audience: str,
    ) -> Dict[str, Any]: ...


# JWT validation (using PyJWT if available, fallback to manual)
_jwt_module: Optional[_JWTModuleProtocol] = None
HAS_JWT = False

# Exception types - will be real jwt exceptions or our stubs
ExpiredSignatureError: type[Exception] = _ExpiredSignatureError
InvalidSignatureError: type[Exception] = _InvalidSignatureError
DecodeError: type[Exception] = _DecodeError
InvalidTokenError: type[Exception] = _InvalidTokenError
InvalidAudienceError: type[Exception] = _InvalidAudienceError

try:
    import jwt

    _jwt_module = jwt
    HAS_JWT = True
    # Use real exception classes from jwt
    from jwt.exceptions import DecodeError as _RealDecodeError
    from jwt.exceptions import ExpiredSignatureError as _RealExpiredSignatureError
    from jwt.exceptions import InvalidAudienceError as _RealInvalidAudienceError
    from jwt.exceptions import InvalidSignatureError as _RealInvalidSignatureError
    from jwt.exceptions import InvalidTokenError as _RealInvalidTokenError

    ExpiredSignatureError = _RealExpiredSignatureError
    InvalidSignatureError = _RealInvalidSignatureError
    DecodeError = _RealDecodeError
    InvalidTokenError = _RealInvalidTokenError
    InvalidAudienceError = _RealInvalidAudienceError
except ImportError:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class User:
    """Authenticated user from Supabase."""

    id: str
    email: str
    role: str = "user"  # user, admin, service
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Subscription info
    plan: str = "free"  # free, pro, team, enterprise
    workspace_id: Optional[str] = None

    # Timestamps
    created_at: Optional[str] = None
    last_sign_in: Optional[str] = None

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def is_pro(self) -> bool:
        return self.plan in ("pro", "team", "enterprise")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "plan": self.plan,
            "workspace_id": self.workspace_id,
            "is_admin": self.is_admin,
            "is_pro": self.is_pro,
        }


@dataclass
class Workspace:
    """Multi-tenant workspace."""

    id: str
    name: str
    owner_id: str
    plan: str = "free"

    # Limits based on plan
    max_debates: int = 50
    max_agents: int = 2
    max_members: int = 1

    # Members
    member_ids: List[str] = field(default_factory=list)

    # Settings
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "owner_id": self.owner_id,
            "plan": self.plan,
            "max_debates": self.max_debates,
            "max_agents": self.max_agents,
            "max_members": self.max_members,
            "member_count": len(self.member_ids) + 1,  # +1 for owner
        }


@dataclass
class APIKey:
    """API key for programmatic access."""

    id: str
    user_id: str
    workspace_id: str
    name: str
    key_hash: str  # Store hash, not the key itself
    prefix: str  # First 8 chars for identification (e.g., "ara_xxxx")

    # Permissions
    scopes: List[str] = field(default_factory=lambda: ["read", "write"])

    # Metadata
    created_at: Optional[str] = None
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "prefix": self.prefix,
            "scopes": self.scopes,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
        }


# =============================================================================
# Supabase JWT Validator
# =============================================================================


class SupabaseAuthValidator:
    """
    Validates Supabase Auth JWTs.

    Supports:
    - JWT signature verification (HS256)
    - Token expiration checking
    - Role/claims extraction
    """

    def __init__(self, jwt_secret: Optional[str] = None, supabase_url: Optional[str] = None):
        self.jwt_secret = jwt_secret or os.getenv("SUPABASE_JWT_SECRET")
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")

        # Cache for validated tokens (short TTL)
        # Stores: (user, cached_at, token_exp)
        self._cache: Dict[str, tuple[User, float, float]] = {}
        self._cache_ttl = 60  # 1 minute

    def validate_jwt(self, token: str) -> Optional[User]:
        """
        Validate a Supabase JWT and return user info.

        Args:
            token: JWT from Supabase Auth

        Returns:
            User object if valid, None otherwise
        """
        if not token:
            return None

        # Check cache first
        if token in self._cache:
            user, cached_at, token_exp = self._cache[token]
            now = time.time()
            # Must pass BOTH cache freshness AND token expiration
            if now - cached_at < self._cache_ttl and now < token_exp:
                return user
            else:
                # Cache stale or token expired - remove it
                del self._cache[token]
                if now >= token_exp:
                    logger.debug("Cached token expired, re-validating")
                    return None  # Don't re-validate expired tokens

        try:
            if HAS_JWT and self.jwt_secret and _jwt_module is not None:
                # Use PyJWT for proper validation
                payload = _jwt_module.decode(
                    token,
                    self.jwt_secret,
                    algorithms=["HS256"],
                    audience="authenticated",
                )
            else:
                # SECURITY: In production, reject tokens if proper validation unavailable
                env = os.getenv("ARAGORA_ENVIRONMENT", "development").lower()
                if env == "production":
                    if not HAS_JWT:
                        logger.error(
                            "JWT validation unavailable in production. "
                            "Install PyJWT: pip install pyjwt"
                        )
                    if not self.jwt_secret:
                        logger.error(
                            "SUPABASE_JWT_SECRET not set in production. "
                            "JWT validation requires this secret."
                        )
                    return None

                # Fallback: decode without verification (dev only!)
                logger.warning(
                    "INSECURE: Decoding JWT without signature verification. "
                    "This is only acceptable in development!"
                )
                payload = self._decode_jwt_unsafe(token)
                if not payload:
                    return None

            # Extract user info
            user = self._payload_to_user(payload)

            # Cache the result with token expiration time
            # Default to cache_ttl from now if no exp claim (shouldn't happen with valid JWTs)
            token_exp = payload.get("exp", time.time() + self._cache_ttl)
            self._cache[token] = (user, time.time(), token_exp)

            return user

        except ExpiredSignatureError:
            # Token has expired - this is expected for old sessions
            logger.debug("JWT token expired")
            return None
        except InvalidAudienceError:
            # Token was issued for a different audience
            logger.warning("JWT has invalid audience claim")
            return None
        except (InvalidSignatureError, DecodeError) as e:
            # Token signature invalid or malformed - potential tampering
            logger.warning(f"JWT signature/decode error: {e}")
            return None
        except InvalidTokenError as e:
            # Catch-all for other JWT validation errors
            logger.warning(f"JWT validation error: {e}")
            return None
        except (KeyError, ValueError, TypeError) as e:
            # Malformed token structure or payload
            logger.warning(f"JWT structure error: {e}")
            return None
        except Exception as e:
            # Unexpected system error - fail closed in production
            env = os.getenv("ARAGORA_ENVIRONMENT", "development").lower()
            if env == "production":
                logger.error(f"JWT validation system error (failing closed): {e}")
                raise  # Re-raise to trigger 500 error, don't silently allow
            else:
                logger.warning(f"JWT validation system error (dev mode): {e}")
                return None

    def _decode_jwt_unsafe(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode JWT without signature verification.
        WARNING: Only use in development!
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            # Decode payload (second part)
            payload_b64 = parts[1]
            # Add padding if needed
            padding = 4 - len(payload_b64) % 4
            if padding != 4:
                payload_b64 += "=" * padding

            payload_json = base64.urlsafe_b64decode(payload_b64)
            payload = json.loads(payload_json)

            # Check expiration
            if "exp" in payload and payload["exp"] < time.time():
                logger.debug("JWT expired")
                return None

            return payload

        except Exception as e:
            logger.warning(f"JWT decode failed: {e}")
            return None

    def _payload_to_user(self, payload: Dict[str, Any]) -> User:
        """Convert JWT payload to User object."""
        # Supabase JWT structure
        user_meta = payload.get("user_metadata", {})
        app_meta = payload.get("app_metadata", {})

        return User(
            id=payload.get("sub", ""),
            email=payload.get("email", ""),
            role=payload.get("role", "user"),
            metadata=user_meta,
            plan=app_meta.get("plan", "free"),
            workspace_id=app_meta.get("workspace_id"),
            last_sign_in=payload.get("iat"),
        )

    def clear_cache(self):
        """Clear the token cache."""
        self._cache.clear()


# =============================================================================
# API Key Validator
# =============================================================================


class APIKeyValidator:
    """
    Validates API keys for programmatic access.

    API keys are stored as hashes in the database.
    Format: ara_<random>
    """

    def __init__(self, storage: Optional[Any] = None):
        self._storage = storage
        self._cache: Dict[str, tuple[User, float]] = {}
        self._cache_ttl = 300  # 5 minutes

    async def validate_key(self, key: str) -> Optional[User]:
        """
        Validate an API key and return associated user.

        Args:
            key: API key (format: ara_xxxx...)

        Returns:
            User object if valid, None otherwise
        """
        if not key or not key.startswith("ara_"):
            return None

        # Check cache
        if key in self._cache:
            user, cached_at = self._cache[key]
            if time.time() - cached_at < self._cache_ttl:
                return user
            else:
                del self._cache[key]

        # Look up key in storage
        try:
            import hashlib

            key_hash = hashlib.sha256(key.encode()).hexdigest()

            # Query storage for key
            if self._storage:
                api_key_record = await self._storage.get_api_key_by_hash(key_hash)
                if api_key_record:
                    user = await self._storage.get_user(api_key_record["user_id"])
                    if user:
                        # Update last_used
                        await self._storage.update_api_key_usage(api_key_record["id"])
                        self._cache[key] = (user, time.time())
                        return user

        except Exception as e:
            logger.warning(f"API key validation failed: {e}")

        return None


# =============================================================================
# Global Instances
# =============================================================================

# Singleton validators
_jwt_validator: Optional[SupabaseAuthValidator] = None
_api_key_validator: Optional[APIKeyValidator] = None


def get_jwt_validator() -> SupabaseAuthValidator:
    """Get the global JWT validator."""
    global _jwt_validator
    if _jwt_validator is None:
        _jwt_validator = SupabaseAuthValidator()
    return _jwt_validator


def get_api_key_validator() -> APIKeyValidator:
    """Get the global API key validator."""
    global _api_key_validator
    if _api_key_validator is None:
        _api_key_validator = APIKeyValidator()
    return _api_key_validator


# =============================================================================
# Authentication Functions
# =============================================================================


def extract_auth_token(handler: Any) -> Optional[str]:
    """Extract Bearer token or API key from request."""
    if handler is None:
        return None

    auth_header = None
    if hasattr(handler, "headers"):
        auth_header = handler.headers.get("Authorization", "")

    if not auth_header:
        return None

    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    elif auth_header.startswith("ApiKey "):
        return auth_header[7:]

    return auth_header


def extract_token(handler: Any) -> Optional[str]:
    """
    Extract Bearer token from request handler.

    Only extracts Bearer tokens. For broader token extraction
    (including ApiKey), use extract_auth_token.

    Args:
        handler: HTTP request handler with headers attribute.

    Returns:
        Token string or None if not present or not Bearer type.
    """
    if handler is None:
        return None

    auth_header = None
    if hasattr(handler, "headers"):
        auth_header = handler.headers.get("Authorization", "")

    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[7:]

    return None


def extract_client_ip(handler: Any) -> Optional[str]:
    """
    Extract client IP from request handler.

    Checks X-Forwarded-For for proxied requests, then client_address.

    Args:
        handler: HTTP request handler.

    Returns:
        Client IP string or None.
    """
    if handler is None:
        return None

    # Check for forwarded IP (behind proxy)
    if hasattr(handler, "headers"):
        forwarded = handler.headers.get("X-Forwarded-For", "")
        if forwarded:
            # Take first IP in chain (original client)
            return forwarded.split(",")[0].strip()

    # Check for direct connection
    if hasattr(handler, "client_address"):
        addr = handler.client_address
        if isinstance(addr, tuple) and len(addr) >= 1:
            return str(addr[0])

    return None


async def authenticate_request(handler: Any) -> Optional[User]:
    """
    Authenticate a request and return user.

    Tries JWT first, then API key.

    Args:
        handler: HTTP request handler

    Returns:
        User if authenticated, None otherwise
    """
    token = extract_auth_token(handler)
    if not token:
        return None

    # Try JWT first
    jwt_validator = get_jwt_validator()
    user = jwt_validator.validate_jwt(token)
    if user:
        return user

    # Try API key
    if token.startswith("ara_"):
        api_validator = get_api_key_validator()
        user = await api_validator.validate_key(token)
        if user:
            return user

    return None


def get_current_user(handler: Any) -> Optional[User]:
    """
    Get the current authenticated user (sync version).

    For async handlers, use authenticate_request instead.
    """
    token = extract_auth_token(handler)
    if not token:
        return None

    jwt_validator = get_jwt_validator()
    return jwt_validator.validate_jwt(token)


# =============================================================================
# Decorators
# =============================================================================


def require_user(func: Callable) -> Callable:
    """
    Decorator that requires authenticated user.

    Injects 'user' keyword argument to the handler.

    Usage:
        @require_user
        def endpoint(self, handler, user: User):
            return {"user_id": user.id}
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.server.handlers.base import error_response

        # Extract handler
        handler = kwargs.get("handler")
        if handler is None:
            for arg in args:
                if hasattr(arg, "headers"):
                    handler = arg
                    break

        if handler is None:
            return error_response("No request handler", 500)

        # Authenticate
        user = get_current_user(handler)
        if not user:
            return error_response("Authentication required", 401)

        # Inject user
        kwargs["user"] = user
        return func(*args, **kwargs)

    return wrapper


def require_admin(func: Callable) -> Callable:
    """
    Decorator that requires admin user.

    Usage:
        @require_admin
        def admin_endpoint(self, handler, user: User):
            ...
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.server.handlers.base import error_response

        # Extract handler
        handler = kwargs.get("handler")
        if handler is None:
            for arg in args:
                if hasattr(arg, "headers"):
                    handler = arg
                    break

        if handler is None:
            return error_response("No request handler", 500)

        # Authenticate
        user = get_current_user(handler)
        if not user:
            return error_response("Authentication required", 401)

        if not user.is_admin:
            return error_response("Admin access required", 403)

        kwargs["user"] = user
        return func(*args, **kwargs)

    return wrapper


def require_plan(min_plan: str) -> Callable:
    """
    Decorator that requires a minimum subscription plan.

    Args:
        min_plan: Minimum plan required (free, pro, team, enterprise)

    Usage:
        @require_plan("pro")
        def pro_endpoint(self, handler, user: User):
            ...
    """
    plan_levels = {"free": 0, "pro": 1, "team": 2, "enterprise": 3}
    min_level = plan_levels.get(min_plan, 0)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from aragora.server.handlers.base import error_response

            handler = kwargs.get("handler")
            if handler is None:
                for arg in args:
                    if hasattr(arg, "headers"):
                        handler = arg
                        break

            user = get_current_user(handler)
            if not user:
                return error_response("Authentication required", 401)

            user_level = plan_levels.get(user.plan, 0)
            if user_level < min_level:
                return error_response(
                    f"This feature requires {min_plan} plan or higher",
                    403,
                )

            kwargs["user"] = user
            return func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    # Models
    "User",
    "Workspace",
    "APIKey",
    # Validators
    "SupabaseAuthValidator",
    "APIKeyValidator",
    "get_jwt_validator",
    "get_api_key_validator",
    # Functions
    "authenticate_request",
    "get_current_user",
    "extract_auth_token",
    "extract_token",  # Alias for backward compatibility
    "extract_client_ip",
    # Decorators
    "require_user",
    "require_admin",
    "require_plan",
]
