"""
Security Headers Middleware for FastAPI.

Adds HTTP security headers to all responses to protect against common web vulnerabilities:
- Clickjacking (X-Frame-Options)
- MIME sniffing (X-Content-Type-Options)
- XSS attacks (X-XSS-Protection)
- Referrer leakage (Referrer-Policy)
- Content injection (Content-Security-Policy)
- Protocol downgrade (Strict-Transport-Security)

This wraps the existing SecurityHeadersConfig for consistent header values
across both legacy HTTP and FastAPI servers.
"""

from __future__ import annotations

import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from aragora.server.middleware.security_headers import (
    SecurityHeadersConfig,
    get_security_response_headers,
)

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for adding security headers to all responses.

    Integrates with the existing SecurityHeadersConfig for consistent
    header values across both legacy and FastAPI servers.

    Usage:
        from aragora.server.fastapi.middleware.security_headers import SecurityHeadersMiddleware

        app.add_middleware(SecurityHeadersMiddleware)

        # Or with custom config:
        from aragora.server.middleware.security_headers import SecurityHeadersConfig

        config = SecurityHeadersConfig(hsts_enabled=True)
        app.add_middleware(SecurityHeadersMiddleware, config=config)
    """

    def __init__(self, app, config: SecurityHeadersConfig | None = None):
        """
        Initialize security headers middleware.

        Args:
            app: The FastAPI application
            config: Optional security headers configuration.
                    Uses environment-based defaults if not provided.
        """
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()
        self._headers = get_security_response_headers(self.config)

        if self._headers:
            logger.debug(
                f"SecurityHeadersMiddleware initialized with {len(self._headers)} headers"
            )
        else:
            logger.warning("SecurityHeadersMiddleware disabled (no headers configured)")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and add security headers to response.

        Args:
            request: The incoming request
            call_next: The next middleware/handler in the chain

        Returns:
            Response with security headers added
        """
        response = await call_next(request)

        # Add security headers to all responses
        for name, value in self._headers.items():
            response.headers[name] = value

        return response


__all__ = ["SecurityHeadersMiddleware"]
