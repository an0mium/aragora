"""
Rate limiting middleware for API endpoints.

This module re-exports the unified rate limiting implementation from
aragora.server.middleware.rate_limit for backward compatibility.

Provides unified rate limiting across HTTP handlers with configurable
limits per endpoint, IP, or token.

Usage:
    from aragora.server.rate_limit import RateLimiter, rate_limit

    limiter = RateLimiter()

    # Use as decorator
    @rate_limit(requests_per_minute=30)
    def handle_request(handler):
        ...

    # Or check directly
    if not limiter.allow(client_ip, endpoint="/api/debates"):
        return error_response("Rate limit exceeded", 429)
"""

# Re-export everything from the canonical middleware implementation
from aragora.server.middleware.rate_limit import (
    # Core classes
    TokenBucket,
    RedisTokenBucket,
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    RedisRateLimiter,
    RateLimiterRegistry,
    # Functions
    get_rate_limiter,
    cleanup_rate_limiters,
    reset_rate_limiters,
    rate_limit_headers,
    rate_limit,
    get_redis_client,
    reset_redis_client,
    # Constants
    DEFAULT_RATE_LIMIT,
    IP_RATE_LIMIT,
    BURST_MULTIPLIER,
    REDIS_AVAILABLE,
)


# Legacy API compatibility - these function names were in the old module
def get_limiter() -> RateLimiter:
    """Get or create the global rate limiter.

    Deprecated: Use get_rate_limiter() instead.
    """
    return get_rate_limiter("_default")


def set_limiter(limiter: RateLimiter) -> None:
    """Set the global rate limiter (for testing).

    Deprecated: Use reset_rate_limiters() and get_rate_limiter() instead.
    """
    # This is a no-op in the new implementation since we use ServiceRegistry
    pass


def check_rate_limit(
    client_ip: str,
    endpoint: str | None = None,
    token: str | None = None,
) -> RateLimitResult:
    """Check rate limit without consuming a token (read-only check).

    Note: This still consumes a token in the current implementation.
    For read-only checks, use the limiter's remaining property directly.
    """
    limiter = get_rate_limiter("_default")
    return limiter.allow(client_ip, endpoint, token)


def is_rate_limited(client_ip: str, endpoint: str | None = None) -> bool:
    """Quick check if a client is currently rate limited."""
    result = check_rate_limit(client_ip, endpoint)
    return not result.allowed


__all__ = [
    # Core classes
    "TokenBucket",
    "RedisTokenBucket",
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimiter",
    "RedisRateLimiter",
    "RateLimiterRegistry",
    # Functions
    "get_rate_limiter",
    "cleanup_rate_limiters",
    "reset_rate_limiters",
    "rate_limit_headers",
    "rate_limit",
    "get_redis_client",
    "reset_redis_client",
    # Legacy API compatibility
    "get_limiter",
    "set_limiter",
    "check_rate_limit",
    "is_rate_limited",
    # Constants
    "DEFAULT_RATE_LIMIT",
    "IP_RATE_LIMIT",
    "BURST_MULTIPLIER",
    "REDIS_AVAILABLE",
]
