"""
Response utilities for HTTP handlers.

This module provides helper methods for sending HTTP responses with
proper headers (CORS, security, rate limiting, tracing).
"""

import json
import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from aragora.server.middleware.rate_limit import RateLimitResult

logger = logging.getLogger(__name__)


class ResponseHelpersMixin:
    """Mixin providing HTTP response helper methods.

    This mixin expects the following methods from the parent class:
    - send_response(status: int)
    - send_header(name: str, value: str)
    - end_headers()
    - wfile.write(data: bytes)
    - headers (dict-like)

    And these class attributes:
    - _rate_limit_result: Optional[RateLimitResult]
    - _response_status: int
    """

    # Type stubs for expected attributes
    _rate_limit_result: Optional["RateLimitResult"]
    _response_status: int
    headers: Any
    wfile: Any

    def _send_json(self, data: Any, status: int = 200) -> None:
        """Send JSON response with all standard headers."""
        self._response_status = status
        content = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self._add_cors_headers()
        self._add_security_headers()
        self._add_rate_limit_headers()
        self._add_trace_headers()
        self.end_headers()
        self.wfile.write(content)

    def _add_trace_headers(self) -> None:
        """Add trace ID header to response for correlation."""
        from aragora.server.middleware.tracing import TRACE_ID_HEADER, get_trace_id

        trace_id = get_trace_id()
        if trace_id:
            self.send_header(TRACE_ID_HEADER, trace_id)

    def _add_rate_limit_headers(self) -> None:
        """Add rate limit headers to response.

        Includes X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset
        headers if a rate limit check was performed for this request.
        """
        result = getattr(self, "_rate_limit_result", None)
        if result is None:
            return

        from aragora.server.middleware.rate_limit import rate_limit_headers

        headers = rate_limit_headers(result)
        for name, value in headers.items():
            self.send_header(name, value)

    def _add_security_headers(self) -> None:
        """Add security headers to prevent common attacks."""
        # Prevent clickjacking
        self.send_header("X-Frame-Options", "DENY")
        # Prevent MIME type sniffing
        self.send_header("X-Content-Type-Options", "nosniff")
        # Enable XSS filter
        self.send_header("X-XSS-Protection", "1; mode=block")
        # Referrer policy - don't leak internal URLs
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        # Content Security Policy - prevent XSS and data injection
        # Note: 'unsafe-inline' for styles needed by CSS-in-JS frameworks
        # 'unsafe-eval' removed for security - blocks eval()/new Function()
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' wss: https:; "
            "font-src 'self' data:; "
            "frame-ancestors 'none'",
        )
        # HTTP Strict Transport Security - enforce HTTPS
        self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")

    def _add_cors_headers(self) -> None:
        """Add CORS headers with origin validation."""
        from aragora.server.cors_config import cors_config

        # Security: Validate origin against centralized allowlist
        request_origin = self.headers.get("Origin", "")

        if cors_config.is_origin_allowed(request_origin):
            self.send_header("Access-Control-Allow-Origin", request_origin)
            # Allow credentials (cookies, authorization headers) for authenticated requests
            self.send_header("Access-Control-Allow-Credentials", "true")
        elif not request_origin:
            # Same-origin requests don't have Origin header
            pass
        # else: no CORS header = browser blocks cross-origin request

        # Support all REST methods including DELETE for privacy endpoints
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, PATCH, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-Filename, Authorization, Accept, Origin, X-Requested-With",
        )
        # Cache preflight response for 1 hour to reduce OPTIONS requests
        self.send_header("Access-Control-Max-Age", "3600")


__all__ = ["ResponseHelpersMixin"]
