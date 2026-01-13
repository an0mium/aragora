"""
Security middleware for request validation and DoS protection.

Consolidates security checks that were scattered in unified_server.py:
- Content length validation
- Query parameter whitelisting
- JSON payload size limits
- Request rate limiting coordination
- Security headers for responses
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional, FrozenSet

logger = logging.getLogger(__name__)


# =============================================================================
# Security Headers
# =============================================================================

# Headers applied to all responses
SECURITY_HEADERS = {
    # Prevent MIME sniffing
    "X-Content-Type-Options": "nosniff",
    # Prevent clickjacking
    "X-Frame-Options": "DENY",
    # XSS protection (legacy but still useful)
    "X-XSS-Protection": "1; mode=block",
    # Don't leak referrer info
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # Control browser features
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}

# HSTS header (only in production over HTTPS)
HSTS_HEADER = "Strict-Transport-Security"
HSTS_VALUE = "max-age=31536000; includeSubDomains"

# CSP header (configurable per environment)
CSP_HEADER = "Content-Security-Policy"
CSP_DEFAULT = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: https:; "
    "font-src 'self' data:; "
    "connect-src 'self' wss: https:; "
    "frame-ancestors 'none';"
)


def get_security_headers(
    production: bool = False,
    enable_hsts: bool = True,
    enable_csp: bool = False,
    custom_csp: Optional[str] = None,
) -> dict[str, str]:
    """
    Get security headers to add to HTTP responses.

    Args:
        production: Whether running in production (enables stricter headers)
        enable_hsts: Whether to enable HSTS (only with production=True)
        enable_csp: Whether to enable Content-Security-Policy
        custom_csp: Custom CSP value (overrides default if provided)

    Returns:
        Dict of header name -> header value
    """
    headers = dict(SECURITY_HEADERS)

    # Add HSTS in production
    if production and enable_hsts:
        headers[HSTS_HEADER] = HSTS_VALUE

    # Add CSP if enabled
    if enable_csp:
        headers[CSP_HEADER] = custom_csp or CSP_DEFAULT

    return headers


def apply_security_headers(
    handler,
    production: bool = False,
    enable_hsts: bool = True,
    enable_csp: bool = False,
) -> None:
    """
    Apply security headers to an HTTP response handler.

    This is designed to work with http.server.BaseHTTPRequestHandler.

    Args:
        handler: HTTP request handler with send_header method
        production: Whether running in production
        enable_hsts: Whether to enable HSTS
        enable_csp: Whether to enable CSP
    """
    headers = get_security_headers(
        production=production,
        enable_hsts=enable_hsts,
        enable_csp=enable_csp,
    )

    for name, value in headers.items():
        handler.send_header(name, value)


@dataclass
class SecurityConfig:
    """Configuration for security middleware."""

    # Content limits (DoS protection)
    max_content_length: int = 100 * 1024 * 1024  # 100MB for uploads
    max_json_length: int = 10 * 1024 * 1024  # 10MB for JSON API
    max_multipart_parts: int = 10

    # Trusted proxies for X-Forwarded-For
    trusted_proxies: FrozenSet[str] = field(
        default_factory=lambda: frozenset(
            p.strip()
            for p in os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
        )
    )

    # Query parameter whitelist
    # Maps param name -> allowed values (None = any string, set = restricted)
    allowed_query_params: dict[str, Optional[set[str]]] = field(
        default_factory=lambda: {
            # Pagination
            "limit": None,
            "offset": None,
            # Filtering
            "domain": None,
            "loop_id": None,
            "topic": None,
            "query": None,
            # Export
            "table": {"summary", "debates", "proposals", "votes", "critiques", "messages"},
            # Agent queries
            "agent": None,
            "agent_a": None,
            "agent_b": None,
            "sections": {"identity", "performance", "relationships", "all"},
            # Calibration
            "buckets": None,
            # Memory
            "tiers": None,
            "min_importance": None,
            # Genesis
            "event_type": {"mutation", "crossover", "selection", "extinction", "speciation"},
            # Logs
            "lines": None,
        }
    )


@dataclass
class ValidationResult:
    """Result of request validation."""

    valid: bool
    error_message: str = ""
    error_code: int = 400

    @classmethod
    def ok(cls) -> "ValidationResult":
        """Return successful validation result."""
        return cls(valid=True)

    @classmethod
    def error(cls, message: str, code: int = 400) -> "ValidationResult":
        """Return failed validation result."""
        return cls(valid=False, error_message=message, error_code=code)


class SecurityMiddleware:
    """Middleware for security validation and DoS protection.

    Consolidates security checks from unified_server.py into a reusable class.

    Usage:
        middleware = SecurityMiddleware()

        # Validate content length
        result = middleware.validate_content_length(request_headers, max_size=None)
        if not result.valid:
            return error_response(result.error_message, result.error_code)

        # Validate query params
        result = middleware.validate_query_params(query_dict)
        if not result.valid:
            return error_response(result.error_message, result.error_code)
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security middleware.

        Args:
            config: Optional security configuration. Uses defaults if not provided.
        """
        self.config = config or SecurityConfig()

    def validate_content_length(
        self,
        headers: dict[str, str],
        max_size: Optional[int] = None,
        is_json: bool = True,
    ) -> ValidationResult:
        """Validate Content-Length header for DoS protection.

        Args:
            headers: Request headers dict
            max_size: Optional max size override (uses config defaults if None)
            is_json: If True, uses JSON limit; otherwise uses content limit

        Returns:
            ValidationResult indicating if request is valid
        """
        if max_size is None:
            max_size = self.config.max_json_length if is_json else self.config.max_content_length

        content_length_str = headers.get("Content-Length") or headers.get("content-length")

        if not content_length_str:
            # No Content-Length header - could be chunked encoding
            return ValidationResult.ok()

        try:
            content_length = int(content_length_str)
        except ValueError:
            return ValidationResult.error(
                f"Invalid Content-Length header: {content_length_str}",
                code=400,
            )

        if content_length > max_size:
            max_size_mb = max_size / (1024 * 1024)
            return ValidationResult.error(
                f"Content too large. Max {max_size_mb:.1f}MB allowed.",
                code=413,
            )

        if content_length < 0:
            return ValidationResult.error(
                "Invalid negative Content-Length",
                code=400,
            )

        return ValidationResult.ok()

    def validate_query_params(
        self,
        params: dict[str, list[str]],
    ) -> ValidationResult:
        """Validate query parameters against whitelist.

        Args:
            params: Query parameters dict (param -> list of values)

        Returns:
            ValidationResult indicating if params are valid
        """
        for param, values in params.items():
            if param not in self.config.allowed_query_params:
                return ValidationResult.error(
                    f"Unknown query parameter: {param}",
                    code=400,
                )

            allowed = self.config.allowed_query_params[param]
            if allowed is not None:
                for val in values:
                    if val not in allowed:
                        return ValidationResult.error(
                            f"Invalid value for {param}: {val}",
                            code=400,
                        )

        return ValidationResult.ok()

    def get_client_ip(
        self,
        remote_address: str,
        headers: dict[str, str],
    ) -> str:
        """Get client IP address, respecting X-Forwarded-For from trusted proxies.

        Args:
            remote_address: Direct remote address
            headers: Request headers

        Returns:
            Best guess at actual client IP
        """
        # Only trust X-Forwarded-For from trusted proxies
        if remote_address not in self.config.trusted_proxies:
            return remote_address

        forwarded_for = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
        if forwarded_for:
            # Take first IP (original client) from comma-separated list
            client_ip = forwarded_for.split(",")[0].strip()
            if client_ip:
                return client_ip

        return remote_address

    def validate_multipart_parts(self, part_count: int) -> ValidationResult:
        """Validate multipart request doesn't exceed max parts limit.

        Args:
            part_count: Number of parts in multipart request

        Returns:
            ValidationResult indicating if parts count is valid
        """
        if part_count > self.config.max_multipart_parts:
            return ValidationResult.error(
                f"Too many multipart parts. Max {self.config.max_multipart_parts} allowed.",
                code=400,
            )
        return ValidationResult.ok()

    def add_allowed_param(self, param: str, allowed_values: Optional[set[str]] = None) -> None:
        """Add a new allowed query parameter.

        Args:
            param: Parameter name
            allowed_values: Optional set of allowed values (None = any string)
        """
        self.config.allowed_query_params[param] = allowed_values

    def validate_json_body_size(self, body: bytes) -> ValidationResult:
        """Validate JSON body size.

        Args:
            body: Raw request body bytes

        Returns:
            ValidationResult indicating if body size is valid
        """
        if len(body) > self.config.max_json_length:
            max_mb = self.config.max_json_length / (1024 * 1024)
            return ValidationResult.error(
                f"JSON body too large. Max {max_mb:.1f}MB allowed.",
                code=413,
            )
        return ValidationResult.ok()
