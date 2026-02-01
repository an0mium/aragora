"""
Security Headers Middleware.

Adds HTTP security headers to all responses to protect against common web vulnerabilities:
- Clickjacking (X-Frame-Options)
- MIME sniffing (X-Content-Type-Options)
- XSS attacks (X-XSS-Protection)
- Referrer leakage (Referrer-Policy)
- Content injection (Content-Security-Policy)
- Protocol downgrade (Strict-Transport-Security)

Configuration:
- ARAGORA_SECURITY_HEADERS_ENABLED: Enable/disable security headers (default: true)
- ARAGORA_HSTS_ENABLED: Enable HSTS header (default: true in production)
- ARAGORA_ENV: Environment mode - "production" or "development" (default: development)

Usage:
    from aragora.server.middleware.security_headers import (
        SecurityHeadersConfig,
        SecurityHeadersMiddleware,
        get_security_response_headers,
    )

    # As middleware
    middleware = SecurityHeadersMiddleware()
    middleware.apply_headers(handler)

    # Get headers as dict
    headers = get_security_response_headers()
    for name, value in headers.items():
        handler.send_header(name, value)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Default security header values
DEFAULT_X_FRAME_OPTIONS = "DENY"
DEFAULT_X_CONTENT_TYPE_OPTIONS = "nosniff"
DEFAULT_X_XSS_PROTECTION = "1; mode=block"
DEFAULT_REFERRER_POLICY = "strict-origin-when-cross-origin"
DEFAULT_CSP = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"
)
DEFAULT_HSTS = "max-age=31536000; includeSubDomains"

# HSTS max-age value (1 year in seconds)
HSTS_MAX_AGE = 31536000

# =============================================================================
# Configuration
# =============================================================================


def _is_production() -> bool:
    """Check if running in production environment."""
    return os.getenv("ARAGORA_ENV", "development").lower() == "production"


@dataclass
class SecurityHeadersConfig:
    """Configuration for security headers middleware."""

    # Enable/disable security headers entirely
    enabled: bool = field(
        default_factory=lambda: os.getenv("ARAGORA_SECURITY_HEADERS_ENABLED", "true").lower()
        in ("true", "1", "yes")
    )

    # HSTS configuration (only applies in production by default)
    hsts_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "ARAGORA_HSTS_ENABLED",
            # Default: enabled in production, disabled in development
            "true" if _is_production() else "false",
        ).lower()
        in ("true", "1", "yes")
    )

    # Individual header values (can be overridden)
    x_frame_options: str = field(
        default_factory=lambda: os.getenv("ARAGORA_X_FRAME_OPTIONS", DEFAULT_X_FRAME_OPTIONS)
    )

    x_content_type_options: str = field(
        default_factory=lambda: os.getenv(
            "ARAGORA_X_CONTENT_TYPE_OPTIONS", DEFAULT_X_CONTENT_TYPE_OPTIONS
        )
    )

    x_xss_protection: str = field(
        default_factory=lambda: os.getenv("ARAGORA_X_XSS_PROTECTION", DEFAULT_X_XSS_PROTECTION)
    )

    referrer_policy: str = field(
        default_factory=lambda: os.getenv("ARAGORA_REFERRER_POLICY", DEFAULT_REFERRER_POLICY)
    )

    content_security_policy: str = field(
        default_factory=lambda: os.getenv("ARAGORA_CONTENT_SECURITY_POLICY", DEFAULT_CSP)
    )

    strict_transport_security: str = field(
        default_factory=lambda: os.getenv("ARAGORA_STRICT_TRANSPORT_SECURITY", DEFAULT_HSTS)
    )


# =============================================================================
# Header Generation
# =============================================================================


def get_security_response_headers(
    config: SecurityHeadersConfig | None = None,
) -> dict[str, str]:
    """Get security headers to add to HTTP responses.

    Args:
        config: Optional configuration (uses environment-based defaults if not provided)

    Returns:
        Dict of header name -> header value
    """
    if config is None:
        config = SecurityHeadersConfig()

    if not config.enabled:
        return {}

    headers: dict[str, str] = {
        "X-Frame-Options": config.x_frame_options,
        "X-Content-Type-Options": config.x_content_type_options,
        "X-XSS-Protection": config.x_xss_protection,
        "Referrer-Policy": config.referrer_policy,
        "Content-Security-Policy": config.content_security_policy,
    }

    # Only add HSTS if enabled (typically production only)
    if config.hsts_enabled:
        headers["Strict-Transport-Security"] = config.strict_transport_security

    return headers


# =============================================================================
# Middleware Class
# =============================================================================


class SecurityHeadersMiddleware:
    """Middleware for adding security headers to HTTP responses.

    Adds standard security headers to protect against common web vulnerabilities.
    HSTS is only added when explicitly enabled (typically in production).

    Usage:
        middleware = SecurityHeadersMiddleware()

        # In response handler (before sending body):
        middleware.apply_headers(handler)

        # Or get headers as dict:
        headers = middleware.get_headers()
        for name, value in headers.items():
            handler.send_header(name, value)
    """

    def __init__(self, config: SecurityHeadersConfig | None = None):
        """Initialize security headers middleware.

        Args:
            config: Optional configuration (uses environment-based defaults if not provided)
        """
        self.config = config or SecurityHeadersConfig()

    @property
    def enabled(self) -> bool:
        """Check if security headers are enabled."""
        return self.config.enabled

    @property
    def hsts_enabled(self) -> bool:
        """Check if HSTS header is enabled."""
        return self.config.hsts_enabled

    def get_headers(self) -> dict[str, str]:
        """Get all security headers as a dictionary.

        Returns:
            Dict of header name -> header value
        """
        return get_security_response_headers(self.config)

    def apply_headers(self, handler: Any) -> None:
        """Apply security headers to an HTTP response handler.

        This is designed to work with http.server.BaseHTTPRequestHandler
        or any handler with a send_header method.

        Args:
            handler: HTTP request handler with send_header method
        """
        if not self.enabled:
            return

        headers = self.get_headers()

        if hasattr(handler, "send_header"):
            for name, value in headers.items():
                handler.send_header(name, value)
        else:
            logger.warning("SecurityHeadersMiddleware: handler does not have send_header method")

    def apply_to_response_dict(self, response_headers: dict[str, str]) -> None:
        """Apply security headers to a response headers dictionary.

        Useful for frameworks that use dict-style headers instead of send_header.

        Args:
            response_headers: Mutable dict of response headers to update
        """
        if not self.enabled:
            return

        headers = self.get_headers()
        response_headers.update(headers)


# =============================================================================
# Convenience Functions
# =============================================================================


def apply_security_headers_to_handler(
    handler: Any,
    config: SecurityHeadersConfig | None = None,
) -> None:
    """Apply security headers to an HTTP handler.

    Convenience function that creates a middleware instance and applies headers.

    Args:
        handler: HTTP request handler with send_header method
        config: Optional configuration
    """
    middleware = SecurityHeadersMiddleware(config)
    middleware.apply_headers(handler)


def create_security_headers_config(
    enabled: bool = True,
    hsts_enabled: bool | None = None,
    x_frame_options: str = DEFAULT_X_FRAME_OPTIONS,
    x_content_type_options: str = DEFAULT_X_CONTENT_TYPE_OPTIONS,
    x_xss_protection: str = DEFAULT_X_XSS_PROTECTION,
    referrer_policy: str = DEFAULT_REFERRER_POLICY,
    content_security_policy: str = DEFAULT_CSP,
    strict_transport_security: str = DEFAULT_HSTS,
) -> SecurityHeadersConfig:
    """Create a security headers configuration with explicit values.

    This bypasses environment variable lookups for programmatic configuration.

    Args:
        enabled: Whether to enable security headers
        hsts_enabled: Whether to enable HSTS (defaults to production check)
        x_frame_options: X-Frame-Options header value
        x_content_type_options: X-Content-Type-Options header value
        x_xss_protection: X-XSS-Protection header value
        referrer_policy: Referrer-Policy header value
        content_security_policy: Content-Security-Policy header value
        strict_transport_security: Strict-Transport-Security header value

    Returns:
        SecurityHeadersConfig instance
    """
    if hsts_enabled is None:
        hsts_enabled = _is_production()

    # Create config with explicit values (bypassing default_factory)
    config = SecurityHeadersConfig.__new__(SecurityHeadersConfig)
    config.enabled = enabled
    config.hsts_enabled = hsts_enabled
    config.x_frame_options = x_frame_options
    config.x_content_type_options = x_content_type_options
    config.x_xss_protection = x_xss_protection
    config.referrer_policy = referrer_policy
    config.content_security_policy = content_security_policy
    config.strict_transport_security = strict_transport_security

    return config


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Constants
    "DEFAULT_X_FRAME_OPTIONS",
    "DEFAULT_X_CONTENT_TYPE_OPTIONS",
    "DEFAULT_X_XSS_PROTECTION",
    "DEFAULT_REFERRER_POLICY",
    "DEFAULT_CSP",
    "DEFAULT_HSTS",
    "HSTS_MAX_AGE",
    # Configuration
    "SecurityHeadersConfig",
    # Middleware class
    "SecurityHeadersMiddleware",
    # Functions
    "get_security_response_headers",
    "apply_security_headers_to_handler",
    "create_security_headers_config",
]
