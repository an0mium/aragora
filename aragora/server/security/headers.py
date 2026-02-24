"""
Security Headers Middleware for the Aragora HTTP server.

Adds standard security headers to all responses:
- Strict-Transport-Security (HSTS)
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Content-Security-Policy
- Referrer-Policy
- Permissions-Policy

Opt-in via the ``ARAGORA_SECURITY_HEADERS`` environment variable (default ``true``
in production, ``true`` elsewhere unless explicitly disabled).

Usage::

    from aragora.server.security.headers import SecurityHeadersMiddleware

    middleware = SecurityHeadersMiddleware()
    headers = middleware.get_headers()
    for name, value in headers.items():
        handler.send_header(name, value)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HSTS = "max-age=63072000; includeSubDomains; preload"
DEFAULT_X_CONTENT_TYPE_OPTIONS = "nosniff"
DEFAULT_X_FRAME_OPTIONS = "DENY"
DEFAULT_X_XSS_PROTECTION = "1; mode=block"
DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "connect-src 'self'; "
    "font-src 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self'; "
    "base-uri 'self'"
)
DEFAULT_REFERRER_POLICY = "strict-origin-when-cross-origin"
DEFAULT_PERMISSIONS_POLICY = (
    "camera=(), microphone=(), geolocation=(), "
    "payment=(), usb=(), magnetometer=(), gyroscope=(), accelerometer=()"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: str = "true") -> bool:
    """Read an environment variable as a boolean."""
    return os.getenv(name, default).lower() in ("true", "1", "yes")


def _is_production() -> bool:
    return os.getenv("ARAGORA_ENV", "development").lower() == "production"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SecurityHeadersConfig:
    """Configuration for the security headers middleware."""

    enabled: bool = field(default_factory=lambda: _env_bool("ARAGORA_SECURITY_HEADERS", "true"))
    hsts_enabled: bool = field(
        default_factory=lambda: _env_bool(
            "ARAGORA_HSTS_ENABLED", "true" if _is_production() else "false"
        )
    )

    # Individual header values
    hsts: str = DEFAULT_HSTS
    x_content_type_options: str = DEFAULT_X_CONTENT_TYPE_OPTIONS
    x_frame_options: str = DEFAULT_X_FRAME_OPTIONS
    x_xss_protection: str = DEFAULT_X_XSS_PROTECTION
    content_security_policy: str = DEFAULT_CSP
    referrer_policy: str = DEFAULT_REFERRER_POLICY
    permissions_policy: str = DEFAULT_PERMISSIONS_POLICY

    # Paths to exclude from security headers (e.g. health checks)
    exclude_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Header generation
# ---------------------------------------------------------------------------


def get_default_security_headers(
    config: SecurityHeadersConfig | None = None,
) -> dict[str, str]:
    """Return the dict of security headers to attach to every response.

    Args:
        config: Optional explicit config. Falls back to environment defaults.

    Returns:
        Mapping of header-name to header-value.
    """
    if config is None:
        config = SecurityHeadersConfig()

    if not config.enabled:
        return {}

    headers: dict[str, str] = {
        "X-Content-Type-Options": config.x_content_type_options,
        "X-Frame-Options": config.x_frame_options,
        "X-XSS-Protection": config.x_xss_protection,
        "Content-Security-Policy": config.content_security_policy,
        "Referrer-Policy": config.referrer_policy,
        "Permissions-Policy": config.permissions_policy,
    }

    if config.hsts_enabled:
        headers["Strict-Transport-Security"] = config.hsts

    return headers


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class SecurityHeadersMiddleware:
    """Middleware that injects security headers into HTTP responses.

    Works with ``http.server.BaseHTTPRequestHandler`` or any object exposing
    a ``send_header(name, value)`` method.
    """

    def __init__(self, config: SecurityHeadersConfig | None = None) -> None:
        self.config = config or SecurityHeadersConfig()
        self._cached_headers: dict[str, str] | None = None

    # -- public API --------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def get_headers(self) -> dict[str, str]:
        """Return the security headers dict (cached after first call)."""
        if self._cached_headers is None:
            self._cached_headers = get_default_security_headers(self.config)
        return dict(self._cached_headers)

    def should_apply(self, path: str) -> bool:
        """Check whether security headers should be applied to *path*.

        Excluded paths (e.g. ``/healthz``) skip header injection.
        """
        if not self.enabled:
            return False
        for excluded in self.config.exclude_paths:
            if path.startswith(excluded):
                return False
        return True

    def apply_headers(self, handler: Any, path: str = "") -> None:
        """Apply security headers to an HTTP handler via ``send_header``.

        Args:
            handler: HTTP request handler with a ``send_header`` method.
            path: Request path for exclusion check.
        """
        if not self.should_apply(path):
            return

        headers = self.get_headers()
        if hasattr(handler, "send_header"):
            for name, value in headers.items():
                handler.send_header(name, value)
        else:
            logger.warning("SecurityHeadersMiddleware: handler missing send_header method")

    def apply_to_dict(self, response_headers: dict[str, str], path: str = "") -> None:
        """Merge security headers into an existing headers dict.

        Args:
            response_headers: Mutable mapping to update.
            path: Request path for exclusion check.
        """
        if not self.should_apply(path):
            return
        response_headers.update(self.get_headers())

    def invalidate_cache(self) -> None:
        """Clear the cached headers (call after config change)."""
        self._cached_headers = None


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_HSTS",
    "DEFAULT_X_CONTENT_TYPE_OPTIONS",
    "DEFAULT_X_FRAME_OPTIONS",
    "DEFAULT_X_XSS_PROTECTION",
    "DEFAULT_CSP",
    "DEFAULT_REFERRER_POLICY",
    "DEFAULT_PERMISSIONS_POLICY",
    "SecurityHeadersConfig",
    "SecurityHeadersMiddleware",
    "get_default_security_headers",
]
