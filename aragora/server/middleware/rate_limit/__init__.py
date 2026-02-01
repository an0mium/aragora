"""
Rate Limiting Middleware Package.

Provides configurable rate limiting decorators and classes for API endpoints.
Supports per-IP, per-token, and per-endpoint rate limiting with automatic
cleanup of stale entries.

Usage:
    from aragora.server.middleware.rate_limit import rate_limit, RateLimiter

    # Use as decorator
    @rate_limit(requests_per_minute=30)
    def handle_request(self, handler):
        ...

    # Or use RateLimiter directly
    limiter = RateLimiter()
    if not limiter.allow(client_ip):
        return error_response("Rate limit exceeded", 429)
"""

# Re-export from base module
from .base import (
    BURST_MULTIPLIER,
    DEFAULT_RATE_LIMIT,
    IP_RATE_LIMIT,
    TRUSTED_PROXIES,
    _extract_client_ip,
    _is_trusted_proxy,
    _normalize_ip,
    normalize_rate_limit_path,
    sanitize_rate_limit_key_component,
)

# Re-export from bucket module
from .bucket import RedisTokenBucket, TokenBucket

# Re-export from decorators module
from .decorators import (
    rate_limit,
    rate_limit_headers,
    user_rate_limit,
)

# Re-export from limiter module
from .limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitResult,
)

# Re-export from redis_limiter module
from .redis_limiter import (
    REDIS_AVAILABLE,
    RedisRateLimiter,
    get_redis_client,
    reset_redis_client,
)

# Re-export from registry module
from .registry import (
    RateLimiterRegistry,
    cleanup_rate_limiters,
    get_rate_limiter,
    reset_rate_limiters,
)

# Re-export from tier_limiter module
from .tier_limiter import (
    TIER_RATE_LIMITS,
    TierRateLimiter,
    check_tier_rate_limit,
    get_tier_rate_limiter,
)

# Re-export from user_limiter module
from .user_limiter import (
    USER_RATE_LIMITS,
    UserRateLimiter,
    check_user_rate_limit,
    get_user_rate_limiter,
)

# Re-export from platform_limiter module
from .platform_limiter import (
    PLATFORM_RATE_LIMITS,
    PlatformRateLimiter,
    PlatformRateLimitResult,
    check_platform_rate_limit,
    get_platform_rate_limiter,
    reset_platform_rate_limiters,
)

# Re-export from default_limiter module
from .default_limiter import (
    DEFAULT_RATE_LIMITS,
    DefaultRateLimiter,
    check_default_rate_limit,
    determine_auth_tier,
    get_default_rate_limiter,
    get_handler_default_rpm,
    handler_has_rate_limit_decorator,
    reset_default_rate_limiter,
    should_apply_default_rate_limit,
)

# Re-export from tenant_limiter module
from .tenant_limiter import (
    DEFAULT_TENANT_RATE_LIMITS,
    TenantRateLimiter,
    TenantRateLimitConfig,
    check_tenant_rate_limit,
    get_tenant_rate_limiter,
    reset_tenant_rate_limiter,
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
    "USER_RATE_LIMITS",
    # Platform rate limiting
    "PLATFORM_RATE_LIMITS",
    "PlatformRateLimiter",
    "PlatformRateLimitResult",
    "get_platform_rate_limiter",
    "check_platform_rate_limit",
    "reset_platform_rate_limiters",
    # Default rate limiting
    "DEFAULT_RATE_LIMITS",
    "DefaultRateLimiter",
    "get_default_rate_limiter",
    "reset_default_rate_limiter",
    "determine_auth_tier",
    "check_default_rate_limit",
    "handler_has_rate_limit_decorator",
    "should_apply_default_rate_limit",
    "get_handler_default_rpm",
    # Tenant rate limiting
    "DEFAULT_TENANT_RATE_LIMITS",
    "TenantRateLimiter",
    "TenantRateLimitConfig",
    "get_tenant_rate_limiter",
    "reset_tenant_rate_limiter",
    "check_tenant_rate_limit",
]
