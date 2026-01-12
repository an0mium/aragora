"""
Redis-backed rate limiter implementation.

Provides RedisRateLimiter for distributed rate limiting across multiple
server instances.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, TYPE_CHECKING

from .base import (
    DEFAULT_RATE_LIMIT,
    IP_RATE_LIMIT,
    _normalize_ip,
    sanitize_rate_limit_key_component,
    normalize_rate_limit_path,
)
from .bucket import RedisTokenBucket
from .limiter import RateLimitConfig, RateLimitResult, RateLimiter

if TYPE_CHECKING:
    import redis

logger = logging.getLogger(__name__)

# Optional Redis support
try:
    import redis as redis_lib

    REDIS_AVAILABLE = True
except ImportError:
    redis_lib = None  # type: ignore
    REDIS_AVAILABLE = False

# Redis rate limiter fail-open policy (SECURITY: default to fail-closed)
# Set ARAGORA_RATE_LIMIT_FAIL_OPEN=true only in dev/testing
RATE_LIMIT_FAIL_OPEN = (
    os.environ.get("ARAGORA_RATE_LIMIT_FAIL_OPEN", "false").lower() == "true"
)
# Number of consecutive Redis failures before falling back to in-memory limiting
REDIS_FAILURE_THRESHOLD = int(
    os.environ.get("ARAGORA_REDIS_FAILURE_THRESHOLD", "3")
)

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

        _redis_client = redis_lib.from_url(
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

        # Observability metrics (in-memory, not persisted to Redis)
        self._requests_allowed: int = 0
        self._requests_rejected: int = 0
        self._rejections_by_endpoint: Dict[str, int] = {}

    def configure_endpoint(
        self,
        endpoint: str,
        requests_per_minute: int,
        burst_size: int | None = None,
        key_type: str = "ip",
    ) -> None:
        """Configure rate limit for a specific endpoint."""
        normalized_endpoint = normalize_rate_limit_path(endpoint)
        self._endpoint_configs[normalized_endpoint] = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            key_type=key_type,
        )
        # Also configure fallback
        self._fallback.configure_endpoint(
            normalized_endpoint, requests_per_minute, burst_size, key_type
        )

    def get_config(self, endpoint: str) -> RateLimitConfig:
        """Get rate limit config for an endpoint."""
        normalized_endpoint = normalize_rate_limit_path(endpoint)
        if normalized_endpoint in self._endpoint_configs:
            return self._endpoint_configs[normalized_endpoint]

        for path, config in self._endpoint_configs.items():
            if path.endswith("*") and normalized_endpoint.startswith(path[:-1]):
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
        normalized_endpoint = (
            normalize_rate_limit_path(endpoint) if endpoint else None
        )
        config = (
            self.get_config(normalized_endpoint)
            if normalized_endpoint
            else RateLimitConfig()
        )
        if not config.enabled:
            return RateLimitResult(allowed=True, limit=0)

        client_ip = _normalize_ip(client_ip or "anonymous")
        safe_ip = sanitize_rate_limit_key_component(client_ip)
        safe_token = sanitize_rate_limit_key_component(token) if token else None
        safe_endpoint = (
            sanitize_rate_limit_key_component(normalized_endpoint)
            if normalized_endpoint
            else None
        )

        # Determine the key based on config
        if config.key_type == "token" and safe_token:
            key = f"token:{safe_token}"
        elif config.key_type == "combined" and safe_endpoint:
            key = f"ep:{safe_endpoint}:ip:{safe_ip}"
        elif config.key_type == "endpoint" and safe_endpoint:
            key = f"ep:{safe_endpoint}"
        else:
            key = f"ip:{safe_ip}"

        try:
            bucket = self._get_bucket(key, config)
            allowed = bucket.consume(1)

            # Track metrics
            with self._lock:
                if allowed:
                    self._requests_allowed += 1
                else:
                    self._requests_rejected += 1
                    if safe_endpoint:
                        self._rejections_by_endpoint[safe_endpoint] = (
                            self._rejections_by_endpoint.get(safe_endpoint, 0) + 1
                        )

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
        """Get rate limiter statistics including observability metrics."""
        with self._lock:
            total_requests = self._requests_allowed + self._requests_rejected
            rejection_rate = (
                self._requests_rejected / total_requests
                if total_requests > 0
                else 0.0
            )
            base_stats = {
                "backend": "redis",
                "configured_endpoints": list(self._endpoint_configs.keys()),
                "default_limit": self.default_limit,
                "ip_limit": self.ip_limit,
                # Observability metrics
                "requests_allowed": self._requests_allowed,
                "requests_rejected": self._requests_rejected,
                "total_requests": total_requests,
                "rejection_rate": rejection_rate,
                "rejections_by_endpoint": dict(self._rejections_by_endpoint),
            }

        try:
            # Count Redis keys with our prefix
            keys = list(self.redis.scan_iter(f"{self.key_prefix}*", count=1000))
            base_stats["redis_keys"] = len(keys)
            return base_stats
        except Exception as e:
            base_stats["error"] = str(e)
            base_stats["fallback_stats"] = self._fallback.get_stats()
            return base_stats

    def reset_metrics(self) -> None:
        """Reset observability metrics (useful for testing)."""
        with self._lock:
            self._requests_allowed = 0
            self._requests_rejected = 0
            self._rejections_by_endpoint.clear()
        self._fallback.reset_metrics()

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


__all__ = [
    "REDIS_AVAILABLE",
    "RATE_LIMIT_FAIL_OPEN",
    "REDIS_FAILURE_THRESHOLD",
    "get_redis_client",
    "reset_redis_client",
    "RedisRateLimiter",
]
