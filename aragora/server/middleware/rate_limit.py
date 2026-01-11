"""
Rate Limiting Middleware.

Provides configurable rate limiting decorators and classes for API endpoints.
Supports per-IP, per-token, and per-endpoint rate limiting with automatic
cleanup of stale entries.

Usage:
    from aragora.server.middleware import rate_limit, RateLimiter

    # Use as decorator
    @rate_limit(requests_per_minute=30)
    def handle_request(self, handler):
        ...

    # Or use RateLimiter directly
    limiter = RateLimiter()
    if not limiter.allow(client_ip):
        return error_response("Rate limit exceeded", 429)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)

# Optional Redis support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False

# Configuration from environment
DEFAULT_RATE_LIMIT = int(os.environ.get("ARAGORA_RATE_LIMIT", "60"))
IP_RATE_LIMIT = int(os.environ.get("ARAGORA_IP_RATE_LIMIT", "120"))
BURST_MULTIPLIER = float(os.environ.get("ARAGORA_BURST_MULTIPLIER", "2.0"))

# Trusted proxies for X-Forwarded-For header (comma-separated IPs/CIDRs)
# Only trust XFF header when request comes from these addresses
# Example: "127.0.0.1,10.0.0.0/8,172.16.0.0/12"
TRUSTED_PROXIES_RAW = os.environ.get("ARAGORA_TRUSTED_PROXIES", "").strip()
TRUSTED_PROXIES: frozenset[str] = frozenset(
    p.strip() for p in TRUSTED_PROXIES_RAW.split(",") if p.strip()
)


def _is_trusted_proxy(ip: str) -> bool:
    """
    Check if an IP address is a trusted proxy.

    Supports exact IP matches. For production with CIDR ranges,
    consider using the ipaddress module.

    Args:
        ip: IP address to check.

    Returns:
        True if IP is in TRUSTED_PROXIES.
    """
    if not TRUSTED_PROXIES:
        return False
    return ip in TRUSTED_PROXIES


def _extract_client_ip(
    remote_ip: str,
    xff_header: str | None,
    trust_proxy: bool = False,
) -> str:
    """
    Extract the real client IP from request headers.

    Only trusts X-Forwarded-For when:
    1. trust_proxy is True, OR
    2. remote_ip is in TRUSTED_PROXIES

    Args:
        remote_ip: Direct connection IP address.
        xff_header: X-Forwarded-For header value (may be None).
        trust_proxy: Override to force trusting XFF header.

    Returns:
        Best guess at the real client IP.
    """
    # Only trust XFF from known proxies
    if xff_header and (trust_proxy or _is_trusted_proxy(remote_ip)):
        # XFF format: "client, proxy1, proxy2, ..."
        # First IP is the original client
        client_ip = xff_header.split(",")[0].strip()
        if client_ip:
            return client_ip

    return remote_ip


class TokenBucket:
    """
    Thread-safe token bucket rate limiter.

    Allows burst traffic up to burst_size, then limits to rate_per_minute.
    """

    def __init__(self, rate_per_minute: float, burst_size: int | None = None):
        """
        Initialize token bucket.

        Args:
            rate_per_minute: Token refill rate (tokens per minute).
            burst_size: Maximum tokens (defaults to 2x rate).
        """
        self.rate_per_minute = rate_per_minute
        self.burst_size = burst_size or int(rate_per_minute * BURST_MULTIPLIER)
        self.tokens = float(self.burst_size)  # Start full
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.

        Returns True if tokens were consumed, False if rate limited.
        """
        with self._lock:
            now = time.monotonic()
            elapsed_minutes = (now - self.last_refill) / 60.0
            refill_amount = elapsed_minutes * self.rate_per_minute
            self.tokens = min(self.burst_size, self.tokens + refill_amount)
            self.last_refill = now

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def get_retry_after(self) -> float:
        """Get seconds until next token is available."""
        if self.tokens >= 1:
            return 0
        tokens_needed = 1 - self.tokens
        minutes_needed = tokens_needed / self.rate_per_minute
        return minutes_needed * 60

    @property
    def remaining(self) -> int:
        """Get remaining tokens (approximate, no lock)."""
        return max(0, int(self.tokens))


class RedisTokenBucket:
    """
    Redis-backed token bucket rate limiter.

    Stores token state in Redis for persistence across restarts and
    horizontal scaling. Uses Lua scripts for atomic operations.
    """

    # Lua script for atomic consume operation
    CONSUME_SCRIPT = """
    local key = KEYS[1]
    local rate = tonumber(ARGV[1])
    local burst = tonumber(ARGV[2])
    local now = tonumber(ARGV[3])
    local tokens_requested = tonumber(ARGV[4])
    local ttl = tonumber(ARGV[5])

    -- Get current state
    local data = redis.call('HMGET', key, 'tokens', 'last_refill')
    local tokens = tonumber(data[1]) or burst
    local last_refill = tonumber(data[2]) or now

    -- Calculate refill
    local elapsed_minutes = (now - last_refill) / 60.0
    local refill_amount = elapsed_minutes * rate
    tokens = math.min(burst, tokens + refill_amount)

    -- Try to consume
    local allowed = 0
    if tokens >= tokens_requested then
        tokens = tokens - tokens_requested
        allowed = 1
    end

    -- Save state
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, ttl)

    return {allowed, tokens, burst}
    """

    def __init__(
        self,
        redis_client: "redis.Redis",
        key: str,
        rate_per_minute: float,
        burst_size: int | None = None,
        key_prefix: str = "aragora:ratelimit:",
        ttl_seconds: int = 120,
    ):
        """
        Initialize Redis token bucket.

        Args:
            redis_client: Redis client instance.
            key: Unique key for this bucket.
            rate_per_minute: Token refill rate (tokens per minute).
            burst_size: Maximum tokens (defaults to 2x rate).
            key_prefix: Redis key prefix.
            ttl_seconds: TTL for Redis keys.
        """
        self.redis = redis_client
        self.key = f"{key_prefix}{key}"
        self.rate_per_minute = rate_per_minute
        self.burst_size = burst_size or int(rate_per_minute * BURST_MULTIPLIER)
        self.ttl_seconds = ttl_seconds
        self._consume_sha: Optional[str] = None

    def _get_consume_script(self) -> str:
        """Get or register the consume Lua script."""
        if self._consume_sha is None:
            self._consume_sha = self.redis.script_load(self.CONSUME_SCRIPT)
        return self._consume_sha

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.

        Returns True if tokens were consumed, False if rate limited.
        """
        try:
            now = time.time()
            sha = self._get_consume_script()
            result = self.redis.evalsha(
                sha,
                1,  # number of keys
                self.key,  # KEYS[1]
                self.rate_per_minute,  # ARGV[1]
                self.burst_size,  # ARGV[2]
                now,  # ARGV[3]
                tokens,  # ARGV[4]
                self.ttl_seconds,  # ARGV[5]
            )
            return bool(result[0])
        except Exception as e:
            logger.warning(f"Redis rate limit error, allowing request: {e}")
            return True  # Fail open on Redis errors

    def get_retry_after(self) -> float:
        """Get seconds until next token is available."""
        try:
            data = self.redis.hmget(self.key, "tokens", "last_refill")
            tokens = float(data[0]) if data[0] else self.burst_size
            if tokens >= 1:
                return 0
            tokens_needed = 1 - tokens
            minutes_needed = tokens_needed / self.rate_per_minute
            return minutes_needed * 60
        except Exception as e:
            logger.debug(f"Error getting retry_after, defaulting to 0: {e}")
            return 0

    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        try:
            data = self.redis.hmget(self.key, "tokens", "last_refill")
            tokens = float(data[0]) if data[0] else self.burst_size
            last_refill = float(data[1]) if data[1] else time.time()

            # Calculate refill since last access
            elapsed_minutes = (time.time() - last_refill) / 60.0
            refill_amount = elapsed_minutes * self.rate_per_minute
            tokens = min(self.burst_size, tokens + refill_amount)

            return max(0, int(tokens))
        except Exception as e:
            logger.debug(f"Error getting remaining tokens, defaulting to burst_size: {e}")
            return self.burst_size


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule."""

    requests_per_minute: int = DEFAULT_RATE_LIMIT
    burst_size: int | None = None
    key_type: str = "ip"  # "ip", "token", "endpoint", "combined"
    enabled: bool = True


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int = 0
    limit: int = 0
    retry_after: float = 0
    key: str = ""


class RateLimiter:
    """
    Unified rate limiter for API endpoints.

    Supports per-IP, per-token, and per-endpoint rate limiting with
    automatic cleanup of stale entries via LRU eviction.
    """

    def __init__(
        self,
        default_limit: int = DEFAULT_RATE_LIMIT,
        ip_limit: int = IP_RATE_LIMIT,
        cleanup_interval: int = 300,  # 5 minutes
        max_entries: int = 10000,
    ):
        """
        Initialize rate limiter.

        Args:
            default_limit: Default requests per minute for unspecified endpoints.
            ip_limit: Requests per minute per IP address.
            cleanup_interval: Seconds between stats logging.
            max_entries: Maximum entries before LRU eviction.
        """
        self.default_limit = default_limit
        self.ip_limit = ip_limit
        self.cleanup_interval = cleanup_interval
        self.max_entries = max_entries

        # Buckets by key type (OrderedDict for LRU eviction)
        self._ip_buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._token_buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._endpoint_buckets: Dict[
            str, OrderedDict[str, TokenBucket]
        ] = {}

        # Per-endpoint configuration
        self._endpoint_configs: Dict[str, RateLimitConfig] = {}

        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def configure_endpoint(
        self,
        endpoint: str,
        requests_per_minute: int,
        burst_size: int | None = None,
        key_type: str = "ip",
    ) -> None:
        """
        Configure rate limit for a specific endpoint.

        Args:
            endpoint: API endpoint path (e.g., "/api/debates").
            requests_per_minute: Max requests per minute.
            burst_size: Max burst capacity (default: 2x rate).
            key_type: How to key the limit ("ip", "token", "endpoint", "combined").
        """
        self._endpoint_configs[endpoint] = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            key_type=key_type,
        )

    def get_config(self, endpoint: str) -> RateLimitConfig:
        """Get rate limit config for an endpoint."""
        if endpoint in self._endpoint_configs:
            return self._endpoint_configs[endpoint]

        # Check for prefix match (wildcard endpoints)
        for path, config in self._endpoint_configs.items():
            if path.endswith("*") and endpoint.startswith(path[:-1]):
                return config

        return RateLimitConfig(requests_per_minute=self.default_limit)

    def allow(
        self,
        client_ip: str,
        endpoint: str | None = None,
        token: str | None = None,
    ) -> RateLimitResult:
        """
        Check if a request should be allowed.

        Args:
            client_ip: Client IP address.
            endpoint: Optional endpoint for per-endpoint limits.
            token: Optional auth token for per-token limits.

        Returns:
            RateLimitResult with allowed status and metadata.
        """
        self._maybe_cleanup()

        config = self.get_config(endpoint) if endpoint else RateLimitConfig()
        if not config.enabled:
            return RateLimitResult(allowed=True, limit=0)

        # Determine the key based on config
        if config.key_type == "token" and token:
            key = f"token:{token}"
            bucket = self._get_or_create_token_bucket(token, config)
        elif config.key_type == "combined" and endpoint:
            key = f"ep:{endpoint}:ip:{client_ip}"
            bucket = self._get_or_create_endpoint_bucket(endpoint, client_ip, config)
        elif config.key_type == "endpoint" and endpoint:
            key = f"ep:{endpoint}"
            bucket = self._get_or_create_endpoint_bucket(endpoint, "_global", config)
        else:
            # Default to IP-based limiting
            key = f"ip:{client_ip}"
            bucket = self._get_or_create_ip_bucket(client_ip)

        allowed = bucket.consume(1)

        return RateLimitResult(
            allowed=allowed,
            remaining=bucket.remaining,
            limit=config.requests_per_minute,
            retry_after=bucket.get_retry_after() if not allowed else 0,
            key=key,
        )

    def _get_or_create_ip_bucket(self, ip: str) -> TokenBucket:
        """Get or create an IP-based bucket with LRU eviction."""
        with self._lock:
            if ip in self._ip_buckets:
                self._ip_buckets.move_to_end(ip)
                return self._ip_buckets[ip]

            # Evict oldest entries if at capacity
            max_ip_buckets = self.max_entries // 3
            while len(self._ip_buckets) >= max_ip_buckets:
                self._ip_buckets.popitem(last=False)

            self._ip_buckets[ip] = TokenBucket(self.ip_limit)
            return self._ip_buckets[ip]

    def _get_or_create_token_bucket(
        self,
        token: str,
        config: RateLimitConfig,
    ) -> TokenBucket:
        """Get or create a token-based bucket with LRU eviction."""
        with self._lock:
            if token in self._token_buckets:
                self._token_buckets.move_to_end(token)
                return self._token_buckets[token]

            max_token_buckets = self.max_entries // 3
            while len(self._token_buckets) >= max_token_buckets:
                self._token_buckets.popitem(last=False)

            self._token_buckets[token] = TokenBucket(
                config.requests_per_minute,
                config.burst_size,
            )
            return self._token_buckets[token]

    def _get_or_create_endpoint_bucket(
        self,
        endpoint: str,
        key: str,
        config: RateLimitConfig,
    ) -> TokenBucket:
        """Get or create an endpoint-specific bucket with LRU eviction."""
        with self._lock:
            if endpoint not in self._endpoint_buckets:
                self._endpoint_buckets[endpoint] = OrderedDict()

            buckets = self._endpoint_buckets[endpoint]
            if key in buckets:
                buckets.move_to_end(key)
                return buckets[key]

            max_endpoint_buckets = self.max_entries // 3
            total_endpoint_entries = sum(
                len(b) for b in self._endpoint_buckets.values()
            )
            while total_endpoint_entries >= max_endpoint_buckets and len(buckets) > 0:
                buckets.popitem(last=False)
                total_endpoint_entries -= 1

            buckets[key] = TokenBucket(
                config.requests_per_minute,
                config.burst_size,
            )
            return buckets[key]

    def _maybe_cleanup(self) -> None:
        """Periodic stats logging."""
        now = time.monotonic()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            self._last_cleanup = now
            total = (
                len(self._ip_buckets)
                + len(self._token_buckets)
                + sum(len(v) for v in self._endpoint_buckets.values())
            )

            if total > 0:
                logger.debug(
                    f"Rate limiter stats: {len(self._ip_buckets)} IP, "
                    f"{len(self._token_buckets)} token, "
                    f"{sum(len(v) for v in self._endpoint_buckets.values())} "
                    f"endpoint buckets"
                )

    def cleanup(self, max_age_seconds: int = 300) -> int:
        """
        Remove all stale entries older than max_age_seconds.

        This is more aggressive than _maybe_cleanup - it actually removes
        entries based on last activity time, not just LRU eviction.

        Args:
            max_age_seconds: Maximum age in seconds before entry is removed.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            now = time.monotonic()
            removed = 0

            # For simplicity with token buckets that use monotonic time,
            # we can check last_refill against now
            for bucket_dict in [self._ip_buckets, self._token_buckets]:
                stale_keys = [
                    key
                    for key, bucket in bucket_dict.items()
                    if now - bucket.last_refill > max_age_seconds
                ]
                for key in stale_keys:
                    del bucket_dict[key]
                    removed += 1

            # Clean endpoint buckets
            for endpoint, buckets in list(self._endpoint_buckets.items()):
                stale_keys = [
                    key
                    for key, bucket in buckets.items()
                    if now - bucket.last_refill > max_age_seconds
                ]
                for key in stale_keys:
                    del buckets[key]
                    removed += 1

                # Remove empty endpoint dicts
                if not buckets:
                    del self._endpoint_buckets[endpoint]

            if removed > 0:
                logger.debug(f"Rate limiter cleanup removed {removed} stale entries")

            return removed

    def reset(self) -> None:
        """Reset all rate limiter state. Primarily for testing."""
        with self._lock:
            self._ip_buckets.clear()
            self._token_buckets.clear()
            self._endpoint_buckets.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "ip_buckets": len(self._ip_buckets),
                "token_buckets": len(self._token_buckets),
                "endpoint_buckets": {
                    ep: len(buckets)
                    for ep, buckets in self._endpoint_buckets.items()
                },
                "configured_endpoints": list(self._endpoint_configs.keys()),
                "default_limit": self.default_limit,
                "ip_limit": self.ip_limit,
            }

    def get_client_key(self, handler: Any) -> str:
        """
        Extract client key from request handler.

        Only trusts X-Forwarded-For when request comes from a trusted proxy
        (configured via ARAGORA_TRUSTED_PROXIES environment variable).
        Falls back to 'anonymous' if neither available.

        Args:
            handler: HTTP request handler.

        Returns:
            Client identifier string.
        """
        if handler is None:
            return "anonymous"

        # Get direct connection IP
        remote_ip = "anonymous"
        if hasattr(handler, "client_address"):
            addr = handler.client_address
            if isinstance(addr, tuple) and len(addr) >= 1:
                remote_ip = str(addr[0])

        # Get X-Forwarded-For header
        xff_header = None
        if hasattr(handler, "headers"):
            xff_header = handler.headers.get("X-Forwarded-For", "")

        # Only trust XFF from configured trusted proxies
        return _extract_client_ip(remote_ip, xff_header)


# Use ServiceRegistry for rate limiter management
from aragora.services import ServiceRegistry


# Global Redis client (lazy-initialized)
_redis_client: Optional["redis.Redis"] = None
_redis_init_attempted: bool = False


def get_redis_client() -> Optional["redis.Redis"]:
    """
    Get Redis client if configured and available.

    Uses settings from aragora.config.settings for Redis URL.
    Returns None if Redis is not configured or unavailable.
    """
    global _redis_client, _redis_init_attempted

    if _redis_init_attempted:
        return _redis_client

    _redis_init_attempted = True

    if not REDIS_AVAILABLE:
        logger.debug("Redis package not installed, using in-memory rate limiting")
        return None

    try:
        from aragora.config.settings import get_settings
        settings = get_settings()

        redis_url = settings.rate_limit.redis_url
        if not redis_url:
            logger.debug("ARAGORA_REDIS_URL not set, using in-memory rate limiting")
            return None

        _redis_client = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        # Test connection
        _redis_client.ping()
        logger.info(f"Redis rate limiting enabled: {redis_url.split('@')[-1]}")
        return _redis_client

    except Exception as e:
        logger.warning(f"Redis connection failed, using in-memory rate limiting: {e}")
        _redis_client = None
        return None


def reset_redis_client() -> None:
    """Reset Redis client (for testing)."""
    global _redis_client, _redis_init_attempted
    if _redis_client is not None:
        try:
            _redis_client.close()
        except Exception as e:
            # Log but don't fail - we're resetting anyway
            logger.debug(f"Error closing Redis client during reset: {e}")
    _redis_client = None
    _redis_init_attempted = False


class RedisRateLimiter:
    """
    Rate limiter using Redis for persistent storage.

    Provides the same interface as RateLimiter but stores state in Redis.
    Falls back to in-memory on Redis errors.
    """

    def __init__(
        self,
        redis_client: "redis.Redis",
        default_limit: int = DEFAULT_RATE_LIMIT,
        ip_limit: int = IP_RATE_LIMIT,
        key_prefix: str = "aragora:ratelimit:",
        ttl_seconds: int = 120,
    ):
        """
        Initialize Redis rate limiter.

        Args:
            redis_client: Redis client instance.
            default_limit: Default requests per minute.
            ip_limit: Requests per minute per IP.
            key_prefix: Redis key prefix.
            ttl_seconds: TTL for Redis keys.
        """
        self.redis = redis_client
        self.default_limit = default_limit
        self.ip_limit = ip_limit
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

        # In-memory fallback for when Redis fails
        self._fallback = RateLimiter(default_limit, ip_limit)

        # Per-endpoint configuration (stored in memory, not Redis)
        self._endpoint_configs: Dict[str, RateLimitConfig] = {}

        # Cache of Redis buckets
        self._buckets: Dict[str, RedisTokenBucket] = {}
        self._lock = threading.Lock()

    def configure_endpoint(
        self,
        endpoint: str,
        requests_per_minute: int,
        burst_size: int | None = None,
        key_type: str = "ip",
    ) -> None:
        """Configure rate limit for a specific endpoint."""
        self._endpoint_configs[endpoint] = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            key_type=key_type,
        )
        # Also configure fallback
        self._fallback.configure_endpoint(endpoint, requests_per_minute, burst_size, key_type)

    def get_config(self, endpoint: str) -> RateLimitConfig:
        """Get rate limit config for an endpoint."""
        if endpoint in self._endpoint_configs:
            return self._endpoint_configs[endpoint]

        for path, config in self._endpoint_configs.items():
            if path.endswith("*") and endpoint.startswith(path[:-1]):
                return config

        return RateLimitConfig(requests_per_minute=self.default_limit)

    def _get_bucket(self, key: str, config: RateLimitConfig) -> RedisTokenBucket:
        """Get or create a Redis bucket."""
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = RedisTokenBucket(
                    self.redis,
                    key,
                    config.requests_per_minute,
                    config.burst_size,
                    self.key_prefix,
                    self.ttl_seconds,
                )
            return self._buckets[key]

    def allow(
        self,
        client_ip: str,
        endpoint: str | None = None,
        token: str | None = None,
    ) -> RateLimitResult:
        """Check if a request should be allowed."""
        config = self.get_config(endpoint) if endpoint else RateLimitConfig()
        if not config.enabled:
            return RateLimitResult(allowed=True, limit=0)

        # Determine the key based on config
        if config.key_type == "token" and token:
            key = f"token:{token}"
        elif config.key_type == "combined" and endpoint:
            key = f"ep:{endpoint}:ip:{client_ip}"
        elif config.key_type == "endpoint" and endpoint:
            key = f"ep:{endpoint}"
        else:
            key = f"ip:{client_ip}"

        try:
            bucket = self._get_bucket(key, config)
            allowed = bucket.consume(1)

            return RateLimitResult(
                allowed=allowed,
                remaining=bucket.remaining,
                limit=config.requests_per_minute,
                retry_after=bucket.get_retry_after() if not allowed else 0,
                key=key,
            )
        except Exception as e:
            logger.warning(f"Redis rate limit failed, using fallback: {e}")
            return self._fallback.allow(client_ip, endpoint, token)

    def get_client_key(self, handler: Any) -> str:
        """Extract client key from request handler."""
        return self._fallback.get_client_key(handler)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        try:
            # Count Redis keys with our prefix
            keys = list(self.redis.scan_iter(f"{self.key_prefix}*", count=1000))
            return {
                "backend": "redis",
                "redis_keys": len(keys),
                "configured_endpoints": list(self._endpoint_configs.keys()),
                "default_limit": self.default_limit,
                "ip_limit": self.ip_limit,
            }
        except Exception as e:
            return {
                "backend": "redis",
                "error": str(e),
                "fallback_stats": self._fallback.get_stats(),
            }

    def cleanup(self, max_age_seconds: int = 300) -> int:
        """Redis handles TTL-based cleanup automatically."""
        return 0

    def reset(self) -> None:
        """Reset all rate limiter state."""
        try:
            keys = list(self.redis.scan_iter(f"{self.key_prefix}*", count=10000))
            if keys:
                self.redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Redis reset failed: {e}")

        with self._lock:
            self._buckets.clear()
        self._fallback.reset()


class RateLimiterRegistry:
    """Container for named rate limiters, managed via ServiceRegistry."""

    def __init__(self):
        self._limiters: Dict[str, RateLimiter] = {}
        self._default_limiter: Optional[RateLimiter | RedisRateLimiter] = None
        self._use_redis: Optional[bool] = None

    def get_default(self) -> RateLimiter | RedisRateLimiter:
        """Get the default rate limiter with configured endpoints."""
        if self._default_limiter is None:
            # Check if Redis is available
            redis_client = get_redis_client()

            if redis_client is not None:
                # Use Redis-backed rate limiter
                try:
                    from aragora.config.settings import get_settings
                    settings = get_settings()
                    self._default_limiter = RedisRateLimiter(
                        redis_client,
                        key_prefix=settings.rate_limit.redis_key_prefix,
                        ttl_seconds=settings.rate_limit.redis_ttl_seconds,
                    )
                    self._use_redis = True
                    logger.info("Using Redis-backed rate limiter")
                except Exception as e:
                    logger.warning(f"Failed to create Redis rate limiter: {e}")
                    self._default_limiter = RateLimiter()
                    self._use_redis = False
            else:
                self._default_limiter = RateLimiter()
                self._use_redis = False

            # Configure default endpoint limits
            self._default_limiter.configure_endpoint("/api/debates", 30, key_type="ip")
            self._default_limiter.configure_endpoint("/api/debates/*", 60, key_type="ip")
            self._default_limiter.configure_endpoint(
                "/api/debates/*/fork", 5, key_type="ip"
            )
            self._default_limiter.configure_endpoint("/api/agent/*", 120, key_type="ip")
            self._default_limiter.configure_endpoint("/api/leaderboard*", 60, key_type="ip")
            self._default_limiter.configure_endpoint("/api/pulse/*", 30, key_type="ip")
            self._default_limiter.configure_endpoint(
                "/api/memory/continuum/cleanup", 2, key_type="ip"
            )
            self._default_limiter.configure_endpoint("/api/memory/*", 60, key_type="ip")

            # CPU-intensive endpoints (stricter limits)
            self._default_limiter.configure_endpoint(
                "/api/debates/*/broadcast", 3, key_type="ip"  # Audio generation
            )
            self._default_limiter.configure_endpoint(
                "/api/probes/*", 10, key_type="ip"  # Capability probes
            )
            self._default_limiter.configure_endpoint(
                "/api/verification/*", 10, key_type="ip"  # Proof verification
            )
            self._default_limiter.configure_endpoint(
                "/api/video/*", 2, key_type="ip"  # Video generation
            )
        return self._default_limiter

    @property
    def is_using_redis(self) -> bool:
        """Check if the rate limiter is using Redis backend."""
        if self._use_redis is None:
            # Trigger initialization
            self.get_default()
        return self._use_redis or False

    def get(
        self,
        name: str,
        requests_per_minute: int = DEFAULT_RATE_LIMIT,
        burst: int | None = None,
    ) -> RateLimiter:
        """Get or create a named rate limiter."""
        if name not in self._limiters:
            self._limiters[name] = RateLimiter(
                default_limit=requests_per_minute,
                ip_limit=requests_per_minute,
            )
        return self._limiters[name]

    def cleanup(self, max_age_seconds: int = 300) -> int:
        """Cleanup all rate limiters."""
        removed = 0
        if self._default_limiter is not None:
            removed += self._default_limiter.cleanup(max_age_seconds)
        for limiter in self._limiters.values():
            removed += limiter.cleanup(max_age_seconds)
        return removed

    def reset(self) -> None:
        """Reset all rate limiters, including their internal state."""
        # Reset internal state of all limiters (decorators hold references)
        if self._default_limiter is not None:
            self._default_limiter.reset()
        for limiter in self._limiters.values():
            limiter.reset()
        # Clear registry
        self._default_limiter = None
        self._limiters.clear()


def _get_limiter_registry() -> RateLimiterRegistry:
    """Get the RateLimiterRegistry from ServiceRegistry."""
    registry = ServiceRegistry.get()
    if not registry.has(RateLimiterRegistry):
        registry.register_factory(RateLimiterRegistry, RateLimiterRegistry)
    return registry.resolve(RateLimiterRegistry)


def get_rate_limiter(
    name: str = "_default",
    requests_per_minute: int = DEFAULT_RATE_LIMIT,
    burst: int | None = None,
) -> RateLimiter:
    """
    Get or create a named rate limiter.

    Args:
        name: Unique name for this limiter (e.g., "debate_create").
        requests_per_minute: Max requests per minute.
        burst: Burst capacity (default: 2x rate).

    Returns:
        RateLimiter instance.
    """
    limiter_registry = _get_limiter_registry()

    if name == "_default":
        return limiter_registry.get_default()

    return limiter_registry.get(name, requests_per_minute, burst)


def cleanup_rate_limiters(max_age_seconds: int = 300) -> int:
    """
    Cleanup all rate limiters, removing stale entries.

    Args:
        max_age_seconds: Maximum age in seconds before entry is removed.

    Returns:
        Total number of entries removed across all limiters.
    """
    return _get_limiter_registry().cleanup(max_age_seconds)


def reset_rate_limiters() -> None:
    """Reset all rate limiters. Primarily for testing."""
    registry = ServiceRegistry.get()
    if registry.has(RateLimiterRegistry):
        registry.resolve(RateLimiterRegistry).reset()
        registry.unregister(RateLimiterRegistry)

    # Also reset Redis client
    reset_redis_client()


def rate_limit_headers(result: RateLimitResult) -> Dict[str, str]:
    """Generate rate limit headers for HTTP response."""
    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
    }
    if result.retry_after > 0:
        headers["Retry-After"] = str(int(result.retry_after) + 1)
        headers["X-RateLimit-Reset"] = str(int(time.time() + result.retry_after))
    return headers


def _extract_handler(*args, **kwargs) -> Any:
    """Extract handler from function arguments."""
    handler = kwargs.get("handler")
    if handler is None:
        for arg in args:
            if hasattr(arg, "headers"):
                handler = arg
                break
    return handler


def _error_response(message: str, status: int, headers: Dict[str, str]) -> "HandlerResult":
    """Create an error response."""
    from aragora.server.handlers.base import error_response

    return error_response(message, status, headers=headers)


# Tier-based rate limits (requests per minute)
TIER_RATE_LIMITS: Dict[str, tuple[int, int]] = {
    "free": (10, 60),           # 10 req/min, 60 burst
    "starter": (50, 100),       # 50 req/min, 100 burst
    "professional": (200, 400), # 200 req/min, 400 burst
    "enterprise": (1000, 2000), # 1000 req/min, 2000 burst
}


class TierRateLimiter:
    """
    Tier-aware rate limiter that applies different limits based on subscription tier.

    Looks up user's organization tier and applies corresponding rate limits.
    Falls back to 'free' tier limits for unauthenticated requests.
    """

    def __init__(
        self,
        tier_limits: Optional[Dict[str, tuple[int, int]]] = None,
        max_entries: int = 10000,
    ):
        """
        Initialize tier rate limiter.

        Args:
            tier_limits: Dict mapping tier name to (requests_per_minute, burst_size).
            max_entries: Maximum bucket entries before LRU eviction.
        """
        self.tier_limits = tier_limits or TIER_RATE_LIMITS
        self.max_entries = max_entries

        # Separate buckets per tier for fair isolation
        self._tier_buckets: Dict[str, OrderedDict[str, TokenBucket]] = {
            tier: OrderedDict() for tier in self.tier_limits
        }
        self._lock = threading.Lock()

    def get_tier_limits(self, tier: str) -> tuple[int, int]:
        """Get (rate, burst) for a tier, defaulting to free."""
        return self.tier_limits.get(tier.lower(), self.tier_limits.get("free", (10, 60)))

    def allow(
        self,
        client_key: str,
        tier: str = "free",
    ) -> RateLimitResult:
        """
        Check if request is allowed for given tier.

        Args:
            client_key: Unique client identifier (user_id, org_id, or IP).
            tier: Subscription tier name.

        Returns:
            RateLimitResult with allowed status and metadata.
        """
        tier = tier.lower()
        rate, burst = self.get_tier_limits(tier)

        with self._lock:
            if tier not in self._tier_buckets:
                self._tier_buckets[tier] = OrderedDict()

            buckets = self._tier_buckets[tier]

            if client_key in buckets:
                buckets.move_to_end(client_key)
                bucket = buckets[client_key]
            else:
                # LRU eviction
                max_per_tier = self.max_entries // len(self.tier_limits)
                while len(buckets) >= max_per_tier:
                    buckets.popitem(last=False)

                bucket = TokenBucket(rate, burst)
                buckets[client_key] = bucket

        allowed = bucket.consume(1)

        return RateLimitResult(
            allowed=allowed,
            remaining=bucket.remaining,
            limit=rate,
            retry_after=bucket.get_retry_after() if not allowed else 0,
            key=f"tier:{tier}:{client_key}",
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get tier rate limiter statistics."""
        with self._lock:
            return {
                "tier_buckets": {
                    tier: len(buckets)
                    for tier, buckets in self._tier_buckets.items()
                },
                "tier_limits": self.tier_limits,
            }

    def reset(self) -> None:
        """Reset all rate limiter state."""
        with self._lock:
            for buckets in self._tier_buckets.values():
                buckets.clear()


# Per-user rate limits (requests per minute)
USER_RATE_LIMITS: Dict[str, int] = {
    "default": 60,          # Default for authenticated users
    "debate_create": 10,    # Creating new debates
    "vote": 30,             # Voting on proposals
    "agent_call": 120,      # Calling agent APIs
    "export": 5,            # Exporting data
    "admin": 300,           # Admin operations
}


class UserRateLimiter:
    """
    Per-authenticated-user rate limiter.

    Provides fine-grained rate limiting based on user_id rather than IP.
    This ensures that:
    1. Users behind shared IPs don't compete for rate limits
    2. Individual users can't abuse the system by changing IPs
    3. Different operations can have different limits per user

    Usage:
        limiter = UserRateLimiter()

        # Check if user can perform action
        result = limiter.allow(user_id="user-123", action="debate_create")
        if not result.allowed:
            return error_429(retry_after=result.retry_after)

        # Or use check_user_rate_limit helper
        result = check_user_rate_limit(handler, user_store, action="vote")
    """

    def __init__(
        self,
        action_limits: Optional[Dict[str, int]] = None,
        default_limit: int = 60,
        max_users: int = 10000,
    ):
        """
        Initialize per-user rate limiter.

        Args:
            action_limits: Dict mapping action name to requests_per_minute.
            default_limit: Default limit for unlisted actions.
            max_users: Maximum user entries before LRU eviction.
        """
        self.action_limits = action_limits or USER_RATE_LIMITS
        self.default_limit = default_limit
        self.max_users = max_users

        # Nested structure: action -> user_id -> TokenBucket
        self._user_buckets: Dict[str, OrderedDict[str, TokenBucket]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def get_action_limit(self, action: str) -> int:
        """Get rate limit for an action."""
        return self.action_limits.get(action, self.default_limit)

    def allow(
        self,
        user_id: str,
        action: str = "default",
    ) -> RateLimitResult:
        """
        Check if user can perform action.

        Args:
            user_id: Unique user identifier.
            action: Action name (maps to rate limit).

        Returns:
            RateLimitResult with allowed status and metadata.
        """
        limit = self.get_action_limit(action)

        with self._lock:
            if action not in self._user_buckets:
                self._user_buckets[action] = OrderedDict()

            buckets = self._user_buckets[action]

            if user_id in buckets:
                buckets.move_to_end(user_id)
                bucket = buckets[user_id]
            else:
                # LRU eviction per action
                max_per_action = self.max_users // max(1, len(self._user_buckets))
                while len(buckets) >= max_per_action:
                    buckets.popitem(last=False)

                bucket = TokenBucket(limit, burst_size=int(limit * BURST_MULTIPLIER))
                buckets[user_id] = bucket

        allowed = bucket.consume(1)

        return RateLimitResult(
            allowed=allowed,
            remaining=bucket.remaining,
            limit=limit,
            retry_after=bucket.get_retry_after() if not allowed else 0,
            key=f"user:{user_id}:{action}",
        )

    def cleanup(self, max_age_seconds: int = 600) -> int:
        """Remove stale user entries."""
        with self._lock:
            now = time.monotonic()
            removed = 0

            for action, buckets in list(self._user_buckets.items()):
                stale = [
                    uid for uid, bucket in buckets.items()
                    if now - bucket.last_refill > max_age_seconds
                ]
                for uid in stale:
                    del buckets[uid]
                    removed += 1

                if not buckets:
                    del self._user_buckets[action]

            return removed

    def get_user_status(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """Get rate limit status for a user across all actions."""
        with self._lock:
            status = {}
            for action, buckets in self._user_buckets.items():
                if user_id in buckets:
                    bucket = buckets[user_id]
                    status[action] = {
                        "remaining": bucket.remaining,
                        "limit": self.get_action_limit(action),
                        "retry_after": bucket.get_retry_after(),
                    }
            return status

    def get_stats(self) -> Dict[str, Any]:
        """Get user rate limiter statistics."""
        with self._lock:
            return {
                "action_buckets": {
                    action: len(buckets)
                    for action, buckets in self._user_buckets.items()
                },
                "action_limits": self.action_limits,
                "total_users": sum(
                    len(buckets) for buckets in self._user_buckets.values()
                ),
            }

    def reset(self) -> None:
        """Reset all rate limiter state."""
        with self._lock:
            self._user_buckets.clear()


# Global user rate limiter instance
_user_limiter: Optional[UserRateLimiter] = None


def get_user_rate_limiter() -> UserRateLimiter:
    """Get the global user rate limiter instance."""
    global _user_limiter
    if _user_limiter is None:
        _user_limiter = UserRateLimiter()
    return _user_limiter


def check_user_rate_limit(
    handler: Any,
    user_store: Any = None,
    action: str = "default",
) -> RateLimitResult:
    """
    Check rate limit for authenticated user.

    Falls back to IP-based limiting for unauthenticated requests.

    Args:
        handler: HTTP request handler.
        user_store: UserStore instance for auth extraction.
        action: Action being performed (maps to rate limit).

    Returns:
        RateLimitResult with allowed status.
    """
    limiter = get_user_rate_limiter()

    # Default to IP-based key for unauthenticated
    # Use secure IP extraction that respects TRUSTED_PROXIES
    remote_ip = "anon"
    if hasattr(handler, "client_address"):
        addr = handler.client_address
        if isinstance(addr, tuple) and len(addr) >= 1:
            remote_ip = str(addr[0])

    xff_header = None
    if hasattr(handler, "headers"):
        xff_header = handler.headers.get("X-Forwarded-For", "")

    client_ip = _extract_client_ip(remote_ip, xff_header)
    client_key = f"ip:{client_ip}" if client_ip != "anon" else "anon"

    # Try to extract authenticated user
    if user_store:
        try:
            from aragora.billing.jwt_auth import extract_user_from_request
            auth_ctx = extract_user_from_request(handler, user_store)

            if auth_ctx.is_authenticated and auth_ctx.user_id:
                client_key = auth_ctx.user_id
        except Exception as e:
            logger.debug(f"Could not extract user for rate limiting: {e}")

    return limiter.allow(client_key, action)


def user_rate_limit(
    action: str = "default",
    user_store_factory: Optional[Callable[[], Any]] = None,
):
    """
    Decorator for per-user rate limiting.

    Args:
        action: Action name for rate limit lookup.
        user_store_factory: Optional callable to get UserStore instance.

    Usage:
        @user_rate_limit(action="debate_create")
        def _create_debate(self, handler):
            ...

        @user_rate_limit(action="vote", user_store_factory=get_user_store)
        def _submit_vote(self, path, query_params, handler):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = _extract_handler(*args, **kwargs)
            user_store = user_store_factory() if user_store_factory else None

            result = check_user_rate_limit(handler, user_store, action)

            if not result.allowed:
                logger.warning(
                    f"User rate limit exceeded for {result.key} on {action}"
                )
                return _error_response(
                    f"Rate limit exceeded for {action}. Please try again later.",
                    429,
                    rate_limit_headers(result),
                )

            response = func(*args, **kwargs)

            # Add headers to response if possible
            if hasattr(response, "headers") and isinstance(response.headers, dict):
                response.headers.update(rate_limit_headers(result))

            return response

        return wrapper
    return decorator


# Global tier rate limiter instance
_tier_limiter: Optional[TierRateLimiter] = None


def get_tier_rate_limiter() -> TierRateLimiter:
    """Get the global tier rate limiter instance."""
    global _tier_limiter
    if _tier_limiter is None:
        _tier_limiter = TierRateLimiter()
    return _tier_limiter


def check_tier_rate_limit(
    handler: Any,
    user_store: Any = None,
) -> RateLimitResult:
    """
    Check rate limit based on user's subscription tier.

    Extracts user from request, looks up their org tier, and applies
    tier-appropriate rate limits.

    Args:
        handler: HTTP request handler.
        user_store: UserStore instance for looking up orgs.

    Returns:
        RateLimitResult with allowed status.
    """
    limiter = get_tier_rate_limiter()

    # Default to free tier for anonymous/unauthenticated
    tier = "free"

    # Use secure IP extraction that respects TRUSTED_PROXIES
    remote_ip = "anonymous"
    if hasattr(handler, "client_address"):
        addr = handler.client_address
        if isinstance(addr, tuple) and len(addr) >= 1:
            remote_ip = str(addr[0])

    xff_header = None
    if hasattr(handler, "headers"):
        xff_header = handler.headers.get("X-Forwarded-For", "")

    client_key = _extract_client_ip(remote_ip, xff_header)

    # Try to look up user tier
    if user_store:
        try:
            from aragora.billing.jwt_auth import extract_user_from_request
            auth_ctx = extract_user_from_request(handler, user_store)

            if auth_ctx.is_authenticated:
                # Use user_id as key for authenticated users (more stable)
                client_key = auth_ctx.user_id or client_key

                if auth_ctx.org_id:
                    org = user_store.get_organization_by_id(auth_ctx.org_id)
                    if org:
                        tier = org.tier.value
        except Exception as e:
            logger.debug(f"Could not extract user tier: {e}")

    return limiter.allow(client_key, tier)


def rate_limit(
    requests_per_minute: int = 30,
    burst: int | None = None,
    limiter_name: Optional[str] = None,
    key_type: str = "ip",
):
    """
    Decorator for rate limiting endpoint handlers.

    Applies token bucket rate limiting per client. Returns 429 Too Many Requests
    when limit exceeded.

    Args:
        requests_per_minute: Maximum requests per minute per client.
        burst: Additional burst capacity (default: 2x rate).
        limiter_name: Optional name to share limiter across handlers.
        key_type: How to key the limit ("ip", "token", "endpoint", "combined").

    Usage:
        @rate_limit(requests_per_minute=30)
        def _create_debate(self, handler):
            ...

        @rate_limit(requests_per_minute=10, burst=2, limiter_name="expensive")
        def _run_deep_analysis(self, path, query_params, handler):
            ...
    """

    def decorator(func: Callable) -> Callable:
        name = limiter_name or func.__name__
        limiter = get_rate_limiter(name, requests_per_minute, burst)

        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = _extract_handler(*args, **kwargs)

            # Get client key and check rate limit
            client_key = limiter.get_client_key(handler)

            # Extract endpoint path if available
            endpoint = None
            if args and len(args) > 1 and isinstance(args[1], str):
                endpoint = args[1]  # path is usually second arg

            result = limiter.allow(client_key, endpoint=endpoint)

            if not result.allowed:
                logger.warning(
                    f"Rate limit exceeded for {client_key} on {func.__name__}"
                )
                return _error_response(
                    "Rate limit exceeded. Please try again later.",
                    429,
                    rate_limit_headers(result),
                )

            # Call handler and add rate limit headers to response
            response = func(*args, **kwargs)

            # Add headers to response if possible
            if hasattr(response, "headers") and isinstance(response.headers, dict):
                response.headers.update(
                    {k: v for k, v in rate_limit_headers(result).items()}
                )

            return response

        return wrapper

    return decorator
