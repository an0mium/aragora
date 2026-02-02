"""
CSRF (Cross-Site Request Forgery) Protection Middleware.

Implements the double-submit cookie pattern for CSRF protection:
1. Generates a CSRF token on GET requests and sets it as a cookie
2. Validates the CSRF token on state-changing requests (POST, PUT, DELETE, PATCH)
3. Expects the token in both the cookie and the X-CSRF-Token header

The double-submit cookie pattern works because:
- Attackers cannot read the CSRF cookie value due to same-origin policy
- Attackers cannot set custom headers in cross-origin requests
- Both cookie and header must match for the request to be valid

Configuration:
- ARAGORA_CSRF_ENABLED: Enable/disable CSRF protection (default: true in production)
- ARAGORA_CSRF_SECRET: Secret key for HMAC-signed tokens (default: auto-generated)
- ARAGORA_CSRF_COOKIE_NAME: Cookie name (default: _csrf_token)
- ARAGORA_CSRF_HEADER_NAME: Header name (default: X-CSRF-Token)
- ARAGORA_CSRF_COOKIE_SECURE: Require HTTPS for cookie (default: true in production)
- ARAGORA_CSRF_COOKIE_SAMESITE: SameSite cookie attribute (default: Strict)

Usage:
    from aragora.server.middleware.csrf import (
        CSRFConfig,
        CSRFMiddleware,
        generate_csrf_token,
        validate_csrf_token,
        csrf_protect,
    )

    # As a decorator
    @csrf_protect
    def handle_form_submit(self, handler):
        ...

    # As middleware
    middleware = CSRFMiddleware()
    if not middleware.validate_request(handler):
        return error_response("Invalid CSRF token", 403)

    # Manual token generation
    token = generate_csrf_token()
    handler.send_header("Set-Cookie", f"_csrf_token={token}; HttpOnly; Secure; SameSite=Strict")
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Default cookie and header names
DEFAULT_COOKIE_NAME = "_csrf_token"
DEFAULT_HEADER_NAME = "X-CSRF-Token"

# Alternative header names to check (for compatibility)
ALTERNATIVE_HEADER_NAMES = (
    "X-CSRF-Token",
    "x-csrf-token",
    "X-XSRF-Token",
    "x-xsrf-token",
)

# HTTP methods that require CSRF validation
STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

# HTTP methods that should receive a CSRF token
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# Token validity period (default: 24 hours)
DEFAULT_TOKEN_MAX_AGE = 24 * 60 * 60  # seconds

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class CSRFConfig:
    """Configuration for CSRF middleware."""

    # Enable/disable CSRF protection
    enabled: bool = field(
        default_factory=lambda: os.getenv(
            "ARAGORA_CSRF_ENABLED",
            # Default: enabled in production, disabled in development
            "true" if os.getenv("ARAGORA_ENV", "development").lower() == "production" else "false",
        ).lower()
        in ("true", "1", "yes")
    )

    # Secret key for HMAC-signed tokens
    secret: str = field(
        default_factory=lambda: os.getenv(
            "ARAGORA_CSRF_SECRET",
            # Auto-generate if not configured (will change on restart)
            secrets.token_hex(32),
        )
    )

    # Cookie configuration
    cookie_name: str = field(
        default_factory=lambda: os.getenv("ARAGORA_CSRF_COOKIE_NAME", DEFAULT_COOKIE_NAME)
    )

    # Header configuration
    header_name: str = field(
        default_factory=lambda: os.getenv("ARAGORA_CSRF_HEADER_NAME", DEFAULT_HEADER_NAME)
    )

    # Cookie security settings
    cookie_secure: bool = field(
        default_factory=lambda: os.getenv(
            "ARAGORA_CSRF_COOKIE_SECURE",
            "true" if os.getenv("ARAGORA_ENV", "development").lower() == "production" else "false",
        ).lower()
        in ("true", "1", "yes")
    )

    cookie_httponly: bool = True  # Always HttpOnly to prevent XSS access

    cookie_samesite: str = field(
        default_factory=lambda: os.getenv("ARAGORA_CSRF_COOKIE_SAMESITE", "Strict")
    )

    # Token settings
    token_max_age: int = field(
        default_factory=lambda: int(
            os.getenv("ARAGORA_CSRF_TOKEN_MAX_AGE", str(DEFAULT_TOKEN_MAX_AGE))
        )
    )

    # Paths to exclude from CSRF protection
    # These paths will not require CSRF validation (useful for webhooks, API endpoints with API key auth)
    excluded_paths: frozenset[str] = field(
        default_factory=lambda: frozenset(
            p.strip()
            for p in os.getenv(
                "ARAGORA_CSRF_EXCLUDED_PATHS",
                # Default exclusions: webhooks and API endpoints that use API key auth
                "/api/webhooks/,/api/v1/webhooks/,/api/v2/webhooks/,"
                "/webhooks/,/api/oauth/,/api/v1/oauth/,/api/v2/oauth/",
            ).split(",")
            if p.strip()
        )
    )

    # Path prefixes to exclude (more flexible than exact paths)
    excluded_prefixes: frozenset[str] = field(
        default_factory=lambda: frozenset(
            p.strip()
            for p in os.getenv(
                "ARAGORA_CSRF_EXCLUDED_PREFIXES",
                # Webhooks and OAuth typically use their own auth mechanisms
                "/webhooks/,/api/webhooks/",
            ).split(",")
            if p.strip()
        )
    )

    def is_path_excluded(self, path: str) -> bool:
        """Check if a path is excluded from CSRF protection.

        Args:
            path: Request path to check

        Returns:
            True if the path should be excluded from CSRF validation
        """
        # Check exact path matches
        if path in self.excluded_paths:
            return True

        # Check prefix matches
        for prefix in self.excluded_prefixes:
            if path.startswith(prefix):
                return True

        return False


# =============================================================================
# Token Generation and Validation
# =============================================================================


def _generate_token_value(secret: str, timestamp: int | None = None) -> str:
    """Generate a CSRF token value with HMAC signature.

    The token format is: base64(timestamp:random_bytes:hmac_signature)

    Args:
        secret: Secret key for HMAC
        timestamp: Optional timestamp (defaults to current time)

    Returns:
        Base64-encoded token string
    """
    if timestamp is None:
        timestamp = int(time.time())

    # Generate random bytes
    random_bytes = secrets.token_bytes(16)

    # Create message to sign
    message = f"{timestamp}:{random_bytes.hex()}".encode()

    # Generate HMAC signature
    signature = hmac.new(
        secret.encode(),
        message,
        hashlib.sha256,
    ).digest()

    # Combine all parts
    token_data = f"{timestamp}:{random_bytes.hex()}:{signature.hex()}"

    # Base64 encode for transport
    return base64.urlsafe_b64encode(token_data.encode()).decode()


def _validate_token_value(
    token: str,
    secret: str,
    max_age: int = DEFAULT_TOKEN_MAX_AGE,
) -> tuple[bool, str]:
    """Validate a CSRF token value.

    Args:
        token: The token to validate
        secret: Secret key for HMAC verification
        max_age: Maximum token age in seconds

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not token:
        return False, "Empty token"

    try:
        # Decode base64
        token_data = base64.urlsafe_b64decode(token.encode()).decode()

        # Parse components
        parts = token_data.split(":")
        if len(parts) != 3:
            return False, "Invalid token format"

        timestamp_str, random_hex, signature_hex = parts

        # Parse timestamp
        try:
            timestamp = int(timestamp_str)
        except ValueError:
            return False, "Invalid timestamp"

        # Check token age
        current_time = int(time.time())
        if current_time - timestamp > max_age:
            return False, "Token expired"

        # Check for future timestamps (clock skew tolerance of 60 seconds)
        if timestamp > current_time + 60:
            return False, "Token timestamp in future"

        # Reconstruct message and verify signature
        message = f"{timestamp}:{random_hex}".encode()
        expected_signature = hmac.new(
            secret.encode(),
            message,
            hashlib.sha256,
        ).digest()

        # Constant-time comparison
        actual_signature = bytes.fromhex(signature_hex)
        if not hmac.compare_digest(expected_signature, actual_signature):
            return False, "Invalid signature"

        return True, ""

    except (ValueError, UnicodeDecodeError, binascii_Error) as e:
        return False, f"Token decode error: {e}"


# Handle potential import error for binascii.Error
binascii_Error: type[Exception] = ValueError
try:
    from binascii import Error as _binascii_Error_impl

    binascii_Error = _binascii_Error_impl
except ImportError:
    pass


# =============================================================================
# Public API Functions
# =============================================================================


def generate_csrf_token(config: CSRFConfig | None = None) -> str:
    """Generate a new CSRF token.

    Args:
        config: Optional CSRF configuration (uses default if not provided)

    Returns:
        New CSRF token string
    """
    if config is None:
        config = CSRFConfig()

    return _generate_token_value(config.secret)


def validate_csrf_token(
    token: str,
    config: CSRFConfig | None = None,
) -> tuple[bool, str]:
    """Validate a CSRF token.

    Args:
        token: Token to validate
        config: Optional CSRF configuration

    Returns:
        Tuple of (is_valid, error_message)
    """
    if config is None:
        config = CSRFConfig()

    return _validate_token_value(token, config.secret, config.token_max_age)


def get_csrf_cookie_value(
    token: str,
    config: CSRFConfig | None = None,
) -> str:
    """Build the Set-Cookie header value for a CSRF token.

    Args:
        token: CSRF token to set
        config: Optional CSRF configuration

    Returns:
        Cookie header value string
    """
    if config is None:
        config = CSRFConfig()

    parts = [
        f"{config.cookie_name}={token}",
        f"Max-Age={config.token_max_age}",
        "Path=/",
    ]

    if config.cookie_httponly:
        parts.append("HttpOnly")

    if config.cookie_secure:
        parts.append("Secure")

    if config.cookie_samesite:
        parts.append(f"SameSite={config.cookie_samesite}")

    return "; ".join(parts)


# =============================================================================
# Middleware Class
# =============================================================================


class CSRFMiddleware:
    """CSRF protection middleware using double-submit cookie pattern.

    Usage:
        middleware = CSRFMiddleware()

        # In request handler:
        if request.method in ("POST", "PUT", "DELETE"):
            result = middleware.validate_request(handler)
            if not result.valid:
                return error_response(result.error, 403)

        # For GET requests, set token:
        if request.method == "GET":
            middleware.set_token_cookie(handler)
    """

    def __init__(self, config: CSRFConfig | None = None):
        """Initialize CSRF middleware.

        Args:
            config: Optional configuration (uses environment-based defaults if not provided)
        """
        self.config = config or CSRFConfig()

    @property
    def enabled(self) -> bool:
        """Check if CSRF protection is enabled."""
        return self.config.enabled

    def should_validate(self, method: str, path: str) -> bool:
        """Check if a request should be validated for CSRF.

        Args:
            method: HTTP method
            path: Request path

        Returns:
            True if CSRF validation should be performed
        """
        if not self.enabled:
            return False

        if method.upper() not in STATE_CHANGING_METHODS:
            return False

        if self.config.is_path_excluded(path):
            return False

        return True

    def extract_cookie_token(self, handler: Any) -> str | None:
        """Extract CSRF token from cookie.

        Args:
            handler: HTTP request handler with headers attribute

        Returns:
            Token string or None if not found
        """
        # Get Cookie header
        cookie_header = ""
        if hasattr(handler, "headers"):
            cookie_header = handler.headers.get("Cookie", "")
        elif isinstance(handler, dict):
            cookie_header = handler.get("Cookie", handler.get("cookie", ""))

        if not cookie_header:
            return None

        # Parse cookies
        for cookie in cookie_header.split(";"):
            cookie = cookie.strip()
            if cookie.startswith(f"{self.config.cookie_name}="):
                return cookie[len(self.config.cookie_name) + 1 :]

        return None

    def extract_header_token(self, handler: Any) -> str | None:
        """Extract CSRF token from header.

        Args:
            handler: HTTP request handler with headers attribute

        Returns:
            Token string or None if not found
        """
        if hasattr(handler, "headers"):
            # Check primary header name
            token = handler.headers.get(self.config.header_name)
            if token:
                return token

            # Check alternative header names
            for alt_name in ALTERNATIVE_HEADER_NAMES:
                token = handler.headers.get(alt_name)
                if token:
                    return token

        elif isinstance(handler, dict):
            # Dictionary-style headers
            token = handler.get(self.config.header_name)
            if token:
                return token

            for alt_name in ALTERNATIVE_HEADER_NAMES:
                token = handler.get(alt_name)
                if token:
                    return token

        return None

    def validate_request(self, handler: Any, path: str | None = None) -> "CSRFValidationResult":
        """Validate a request for CSRF protection.

        Args:
            handler: HTTP request handler
            path: Optional request path (extracted from handler if not provided)

        Returns:
            CSRFValidationResult with validation status
        """
        # Extract path if not provided
        if path is None:
            if hasattr(handler, "path"):
                path = handler.path.split("?")[0]  # Remove query string
            else:
                path = "/"

        # Extract method
        method = "GET"
        if hasattr(handler, "command"):
            method = handler.command
        elif hasattr(handler, "method"):
            method = handler.method

        # Check if validation is needed
        if not self.should_validate(method, path):
            return CSRFValidationResult(valid=True, reason="Validation not required")

        # Extract tokens
        cookie_token = self.extract_cookie_token(handler)
        header_token = self.extract_header_token(handler)

        # Validate presence of both tokens
        if not cookie_token:
            logger.warning(f"CSRF validation failed: Missing cookie token for {method} {path}")
            return CSRFValidationResult(
                valid=False,
                reason="Missing CSRF cookie",
                error="CSRF token cookie not found",
            )

        if not header_token:
            logger.warning(f"CSRF validation failed: Missing header token for {method} {path}")
            return CSRFValidationResult(
                valid=False,
                reason="Missing CSRF header",
                error=f"CSRF token header ({self.config.header_name}) not found",
            )

        # Validate cookie token
        is_valid, error = validate_csrf_token(cookie_token, self.config)
        if not is_valid:
            logger.warning(
                f"CSRF validation failed: Invalid cookie token for {method} {path}: {error}"
            )
            return CSRFValidationResult(
                valid=False,
                reason="Invalid cookie token",
                error=f"CSRF cookie token invalid: {error}",
            )

        # Compare tokens (double-submit validation)
        if not hmac.compare_digest(cookie_token, header_token):
            logger.warning(f"CSRF validation failed: Token mismatch for {method} {path}")
            return CSRFValidationResult(
                valid=False,
                reason="Token mismatch",
                error="CSRF token mismatch between cookie and header",
            )

        return CSRFValidationResult(valid=True, reason="Tokens validated successfully")

    def generate_token(self) -> str:
        """Generate a new CSRF token.

        Returns:
            New CSRF token string
        """
        return generate_csrf_token(self.config)

    def get_cookie_header(self, token: str | None = None) -> str:
        """Get the Set-Cookie header value for CSRF token.

        Args:
            token: Optional token (generates new one if not provided)

        Returns:
            Set-Cookie header value
        """
        if token is None:
            token = self.generate_token()

        return get_csrf_cookie_value(token, self.config)

    def set_token_cookie(self, handler: Any, token: str | None = None) -> str:
        """Set CSRF token cookie on handler response.

        Args:
            handler: HTTP request handler with send_header method
            token: Optional token (generates new one if not provided)

        Returns:
            The token that was set
        """
        if token is None:
            token = self.generate_token()

        cookie_value = self.get_cookie_header(token)

        if hasattr(handler, "send_header"):
            handler.send_header("Set-Cookie", cookie_value)

        return token


@dataclass
class CSRFValidationResult:
    """Result of CSRF validation."""

    valid: bool
    reason: str = ""
    error: str = ""

    @property
    def is_valid(self) -> bool:
        """Alias for valid."""
        return self.valid


# =============================================================================
# Decorator
# =============================================================================


def csrf_protect(func: Callable | None = None, *, config: CSRFConfig | None = None) -> Callable:
    """Decorator that requires CSRF validation for state-changing requests.

    Can be used with or without arguments:

        @csrf_protect
        def handler(self, request):
            ...

        @csrf_protect(config=custom_config)
        def handler(self, request):
            ...

    Args:
        func: Function to decorate
        config: Optional CSRF configuration

    Returns:
        Decorated function
    """

    def decorator(fn: Callable) -> Callable:
        middleware = CSRFMiddleware(config)

        @wraps(fn)
        def wrapper(*args, **kwargs) -> "HandlerResult":
            from aragora.server.handlers.base import error_response

            # Extract handler from args
            handler = kwargs.get("handler")
            if handler is None:
                for arg in args:
                    if hasattr(arg, "headers"):
                        handler = arg
                        break

            if handler is None:
                logger.warning("csrf_protect: No handler found in arguments")
                return error_response("Internal server error", 500)

            # Validate CSRF
            result = middleware.validate_request(handler)
            if not result.valid:
                return error_response(result.error or "CSRF validation failed", 403)

            return fn(*args, **kwargs)

        return wrapper

    if func is not None:
        # Called without arguments: @csrf_protect
        return decorator(func)

    # Called with arguments: @csrf_protect(config=...)
    return decorator


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Configuration
    "CSRFConfig",
    "CSRFValidationResult",
    # Middleware class
    "CSRFMiddleware",
    # Functions
    "generate_csrf_token",
    "validate_csrf_token",
    "get_csrf_cookie_value",
    # Decorator
    "csrf_protect",
    # Constants
    "DEFAULT_COOKIE_NAME",
    "DEFAULT_HEADER_NAME",
    "STATE_CHANGING_METHODS",
    "SAFE_METHODS",
    "DEFAULT_TOKEN_MAX_AGE",
]
