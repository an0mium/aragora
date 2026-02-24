"""
Server security infrastructure.

Provides hardened security middleware for the Aragora HTTP server:
- Security headers (HSTS, CSP, X-Frame-Options, etc.)
- Per-endpoint rate limiting with token bucket algorithm
- Request size and depth limits
"""

from .headers import (
    SecurityHeadersConfig,
    SecurityHeadersMiddleware,
    get_default_security_headers,
)
from .endpoint_rate_limiter import (
    EndpointRateLimiter,
    RateTier,
    EndpointRateLimitConfig,
)
from .request_limits import (
    RequestLimitsConfig,
    RequestLimitsMiddleware,
    check_json_depth,
    check_query_params,
)

__all__ = [
    # Headers
    "SecurityHeadersConfig",
    "SecurityHeadersMiddleware",
    "get_default_security_headers",
    # Rate limiting
    "EndpointRateLimiter",
    "RateTier",
    "EndpointRateLimitConfig",
    # Request limits
    "RequestLimitsConfig",
    "RequestLimitsMiddleware",
    "check_json_depth",
    "check_query_params",
]
