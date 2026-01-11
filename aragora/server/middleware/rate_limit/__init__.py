"""
Rate Limiting Middleware Package.

Provides configurable rate limiting decorators and classes for API endpoints.
Supports per-IP, per-token, and per-endpoint rate limiting with automatic
cleanup of stale entries.

Usage:
    from aragora.server.middleware.rate_limit import rate_limit, RateLimiter

    # Use as decorator
    @rate_limit(rpm=30)
    def handle_request(self, handler):
        ...

    # Or use RateLimiter directly
    limiter = RateLimiter()
    if not limiter.allow(client_ip):
        return error_response("Rate limit exceeded", 429)
"""

# Re-export from base module
from .base import (
    DEFAULT_RATE_LIMIT,
    IP_RATE_LIMIT,
    BURST_MULTIPLIER,
    TRUSTED_PROXIES,
    _normalize_ip,
    _is_trusted_proxy,
    _extract_client_ip,
    sanitize_rate_limit_key_component,
    normalize_rate_limit_path,
)

# Re-export from bucket module
from .bucket import TokenBucket, RedisTokenBucket

# Import from the original module for backwards compatibility
# This will be refactored in future iterations
from aragora.server.middleware.rate_limit_legacy import (
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    RedisRateLimiter,
    RateLimiterRegistry,
    TierRateLimiter,
    UserRateLimiter,
    get_redis_client,
    reset_redis_client,
    get_rate_limiter,
    cleanup_rate_limiters,
    reset_rate_limiters,
    get_user_rate_limiter,
    check_user_rate_limit,
    user_rate_limit,
    get_tier_rate_limiter,
    check_tier_rate_limit,
    rate_limit,
    rate_limit_headers,
    REDIS_AVAILABLE,
    TIER_RATE_LIMITS,
)

__all__ = [
    # Base types
    "RateLimitConfig",
    "RateLimitResult",
    # Configuration
    "DEFAULT_RATE_LIMIT",
    "IP_RATE_LIMIT",
    "BURST_MULTIPLIER",
    "TRUSTED_PROXIES",
    "REDIS_AVAILABLE",
    # Helper functions
    "_normalize_ip",
    "_is_trusted_proxy",
    "_extract_client_ip",
    "sanitize_rate_limit_key_component",
    "normalize_rate_limit_path",
    "rate_limit_headers",
    # Token buckets
    "TokenBucket",
    "RedisTokenBucket",
    # Rate limiters
    "RateLimiter",
    "RedisRateLimiter",
    "TierRateLimiter",
    "UserRateLimiter",
    # Registry
    "RateLimiterRegistry",
    # Global functions
    "get_redis_client",
    "reset_redis_client",
    "get_rate_limiter",
    "cleanup_rate_limiters",
    "reset_rate_limiters",
    "get_user_rate_limiter",
    "check_user_rate_limit",
    "get_tier_rate_limiter",
    "check_tier_rate_limit",
    # Decorators
    "rate_limit",
    "user_rate_limit",
    # Tier configuration
    "TIER_RATE_LIMITS",
]
