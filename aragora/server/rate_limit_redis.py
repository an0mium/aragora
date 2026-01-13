"""
Distributed Rate Limiting with Redis.

Provides Redis-backed rate limiting for distributed deployments.
Compatible with the in-memory RateLimiter interface.

Usage:
    from aragora.server.rate_limit_redis import RedisRateLimiter

    # Initialize with Redis URL
    limiter = RedisRateLimiter(redis_url="redis://localhost:6379")

    # Check rate limit
    result = limiter.allow(client_ip="192.168.1.1", endpoint="/api/debates")
    if not result.allowed:
        return error_response("Rate limit exceeded", 429, headers=rate_limit_headers(result))

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379)
    REDIS_RATE_LIMIT_PREFIX: Key prefix (default: aragora:ratelimit)

Requirements:
    pip install redis

See docs/RATE_LIMITING.md for configuration guide.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Import RateLimitResult and RateLimitConfig for compatibility
from aragora.server.middleware.rate_limit import (
    BURST_MULTIPLIER,
    DEFAULT_RATE_LIMIT,
    IP_RATE_LIMIT,
    RateLimitConfig,
    RateLimitResult,
)


@dataclass
class RedisConfig:
    """Configuration for Redis rate limiter."""

    url: str = "redis://localhost:6379"
    prefix: str = "aragora:ratelimit"
    default_limit: int = DEFAULT_RATE_LIMIT
    ip_limit: int = IP_RATE_LIMIT
    burst_multiplier: float = BURST_MULTIPLIER
    socket_timeout: float = 1.0
    socket_connect_timeout: float = 1.0
    max_connections: int = 50
    retry_on_timeout: bool = True


def get_redis_config() -> RedisConfig:
    """Get Redis configuration from environment variables."""
    return RedisConfig(
        url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        prefix=os.getenv("REDIS_RATE_LIMIT_PREFIX", "aragora:ratelimit"),
        default_limit=int(os.getenv("ARAGORA_RATE_LIMIT", str(DEFAULT_RATE_LIMIT))),
        ip_limit=int(os.getenv("ARAGORA_IP_RATE_LIMIT", str(IP_RATE_LIMIT))),
        burst_multiplier=float(os.getenv("ARAGORA_BURST_MULTIPLIER", str(BURST_MULTIPLIER))),
    )


class RedisRateLimiter:
    """
    Distributed rate limiter using Redis as backend.

    Uses Redis sorted sets with sliding window algorithm for accurate
    rate limiting across multiple server instances.

    Features:
    - Sliding window rate limiting (more accurate than fixed windows)
    - Automatic key expiration (no manual cleanup needed)
    - Connection pooling for performance
    - Graceful fallback on Redis errors
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        config: Optional[RedisConfig] = None,
    ):
        """
        Initialize Redis rate limiter.

        Args:
            redis_url: Redis connection URL (overrides config.url)
            config: Full configuration (uses defaults if not provided)
        """
        self.config = config or get_redis_config()
        if redis_url:
            self.config.url = redis_url

        self._redis: Optional[Any] = None
        self._endpoint_configs: Dict[str, RateLimitConfig] = {}
        self._available = True

    def _get_redis(self) -> Any:
        """Get or create Redis connection pool."""
        if self._redis is not None:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(
                self.config.url,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                max_connections=self.config.max_connections,
                retry_on_timeout=self.config.retry_on_timeout,
                decode_responses=True,
            )

            # Test connection
            self._redis.ping()
            logger.info(f"Redis rate limiter connected: {self.config.url}")
            self._available = True
            return self._redis

        except ImportError:
            logger.warning("redis package not installed. Install with: pip install redis")
            self._available = False
            return None
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._available = False
            return None

    @property
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self._available

    def configure_endpoint(
        self,
        endpoint: str,
        requests_per_minute: int,
        burst_size: Optional[int] = None,
        key_type: str = "ip",
    ) -> None:
        """
        Configure rate limit for a specific endpoint.

        Args:
            endpoint: API endpoint path (e.g., "/api/debates")
            requests_per_minute: Max requests per minute
            burst_size: Max burst capacity (default: 2x rate)
            key_type: How to key the limit ("ip", "token", "endpoint", "combined")
        """
        self._endpoint_configs[endpoint] = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            key_type=key_type,
        )

    def get_config(self, endpoint: Optional[str]) -> RateLimitConfig:
        """Get rate limit config for an endpoint."""
        if endpoint and endpoint in self._endpoint_configs:
            return self._endpoint_configs[endpoint]

        # Check for prefix match (wildcard endpoints)
        if endpoint:
            for path, config in self._endpoint_configs.items():
                if path.endswith("*") and endpoint.startswith(path[:-1]):
                    return config

        return RateLimitConfig(requests_per_minute=self.config.default_limit)

    def allow(
        self,
        client_ip: str,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
    ) -> RateLimitResult:
        """
        Check if a request should be allowed using sliding window algorithm.

        Uses Redis sorted sets with score = timestamp for accurate sliding window.

        Args:
            client_ip: Client IP address
            endpoint: Optional endpoint for per-endpoint limits
            token: Optional auth token for per-token limits

        Returns:
            RateLimitResult with allowed status and metadata
        """
        redis_client = self._get_redis()
        if redis_client is None:
            # Fallback: allow request if Redis unavailable
            logger.debug("Redis unavailable, allowing request")
            return RateLimitResult(allowed=True, limit=0)

        config = self.get_config(endpoint)
        if not config.enabled:
            return RateLimitResult(allowed=True, limit=0)

        # Build key based on config
        if config.key_type == "token" and token:
            key = f"{self.config.prefix}:token:{token}"
        elif config.key_type == "combined" and endpoint:
            key = f"{self.config.prefix}:ep:{endpoint}:ip:{client_ip}"
        elif config.key_type == "endpoint" and endpoint:
            key = f"{self.config.prefix}:ep:{endpoint}"
        else:
            key = f"{self.config.prefix}:ip:{client_ip}"

        # Window size in seconds (1 minute)
        window_seconds = 60
        limit = config.requests_per_minute

        try:
            now = time.time()
            window_start = now - window_seconds

            # Use pipeline for atomic operation
            pipe = redis_client.pipeline()

            # Remove entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)

            # Count current entries in window
            pipe.zcard(key)

            # Add current request (will be committed only if under limit)
            pipe.zadd(key, {str(now): now})

            # Set expiry on the key (auto-cleanup)
            pipe.expire(key, window_seconds * 2)

            results = pipe.execute()
            current_count = results[1]  # zcard result

            # Check if under limit
            if current_count < limit:
                return RateLimitResult(
                    allowed=True,
                    remaining=limit - current_count - 1,
                    limit=limit,
                    retry_after=0,
                    key=key,
                )
            else:
                # Remove the entry we just added (request denied)
                redis_client.zrem(key, str(now))

                # Calculate retry_after from oldest entry
                oldest = redis_client.zrange(key, 0, 0, withscores=True)
                retry_after = 0.0
                if oldest:
                    oldest_time = oldest[0][1]
                    retry_after = max(0, window_seconds - (now - oldest_time))

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=limit,
                    retry_after=retry_after,
                    key=key,
                )

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fallback: allow request on error
            return RateLimitResult(allowed=True, limit=0)

    def get_remaining(
        self,
        client_ip: str,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
    ) -> int:
        """
        Get remaining requests without consuming a token.

        Args:
            client_ip: Client IP address
            endpoint: Optional endpoint for per-endpoint limits
            token: Optional auth token for per-token limits

        Returns:
            Number of remaining requests in current window
        """
        redis_client = self._get_redis()
        if redis_client is None:
            return 0

        config = self.get_config(endpoint)

        # Build key
        if config.key_type == "token" and token:
            key = f"{self.config.prefix}:token:{token}"
        elif config.key_type == "combined" and endpoint:
            key = f"{self.config.prefix}:ep:{endpoint}:ip:{client_ip}"
        elif config.key_type == "endpoint" and endpoint:
            key = f"{self.config.prefix}:ep:{endpoint}"
        else:
            key = f"{self.config.prefix}:ip:{client_ip}"

        try:
            now = time.time()
            window_start = now - 60

            # Clean and count in one pipeline
            pipe = redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            results = pipe.execute()

            current_count = results[1]
            return max(0, config.requests_per_minute - current_count)

        except Exception as e:
            logger.error(f"Redis get_remaining error: {e}")
            return 0

    def reset(self, pattern: Optional[str] = None) -> int:
        """
        Reset rate limiter state.

        Args:
            pattern: Optional pattern to match keys (default: all rate limit keys)

        Returns:
            Number of keys deleted
        """
        redis_client = self._get_redis()
        if redis_client is None:
            return 0

        try:
            if pattern:
                full_pattern = f"{self.config.prefix}:{pattern}"
            else:
                full_pattern = f"{self.config.prefix}:*"

            keys = list(redis_client.scan_iter(match=full_pattern, count=1000))
            if keys:
                deleted = redis_client.delete(*keys)
                logger.info(f"Reset {deleted} rate limit keys matching {full_pattern}")
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Redis reset error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        redis_client = self._get_redis()
        if redis_client is None:
            return {"available": False}

        try:
            # Count keys by type
            ip_keys = len(
                list(redis_client.scan_iter(match=f"{self.config.prefix}:ip:*", count=100))
            )
            token_keys = len(
                list(redis_client.scan_iter(match=f"{self.config.prefix}:token:*", count=100))
            )
            endpoint_keys = len(
                list(redis_client.scan_iter(match=f"{self.config.prefix}:ep:*", count=100))
            )

            return {
                "available": True,
                "backend": "redis",
                "url": self.config.url,
                "ip_keys": ip_keys,
                "token_keys": token_keys,
                "endpoint_keys": endpoint_keys,
                "configured_endpoints": list(self._endpoint_configs.keys()),
            }

        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"available": False, "error": str(e)}

    def get_client_key(self, handler: Any) -> str:
        """
        Extract client key from request handler.

        Uses X-Forwarded-For if behind proxy, otherwise client_address.

        Args:
            handler: HTTP request handler

        Returns:
            Client identifier string
        """
        if handler is None:
            return "anonymous"

        # Check for forwarded IP (behind proxy)
        if hasattr(handler, "headers"):
            forwarded = handler.headers.get("X-Forwarded-For", "")
            if forwarded:
                return forwarded.split(",")[0].strip()

        # Check for direct connection
        if hasattr(handler, "client_address"):
            addr = handler.client_address
            if isinstance(addr, tuple) and len(addr) >= 1:
                return str(addr[0])

        return "anonymous"

    def close(self) -> None:
        """Close Redis connection pool."""
        if self._redis is not None:
            try:
                self._redis.close()
                logger.info("Redis rate limiter connection closed")
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
            finally:
                self._redis = None


# Factory function for creating rate limiter based on environment
_rate_limiter_instance: Optional[Any] = None


def get_distributed_rate_limiter() -> Any:
    """
    Get rate limiter instance based on environment configuration.

    Returns Redis-backed limiter if REDIS_URL is set, otherwise returns
    in-memory limiter.

    Returns:
        RateLimiter instance (Redis or in-memory)
    """
    global _rate_limiter_instance

    if _rate_limiter_instance is not None:
        return _rate_limiter_instance

    redis_url = os.getenv("REDIS_URL")

    if redis_url:
        _rate_limiter_instance = RedisRateLimiter(redis_url=redis_url)
        if _rate_limiter_instance.is_available:
            logger.info("Using Redis-backed rate limiter")
            return _rate_limiter_instance
        else:
            logger.warning("Redis unavailable, falling back to in-memory limiter")

    # Fallback to in-memory
    from aragora.server.middleware.rate_limit import get_rate_limiter

    _rate_limiter_instance = get_rate_limiter("_default")
    logger.info("Using in-memory rate limiter")
    return _rate_limiter_instance


def reset_distributed_rate_limiter() -> None:
    """Reset the distributed rate limiter instance (for testing)."""
    global _rate_limiter_instance
    if _rate_limiter_instance is not None:
        if hasattr(_rate_limiter_instance, "close"):
            _rate_limiter_instance.close()
        _rate_limiter_instance = None


__all__ = [
    "RedisConfig",
    "RedisRateLimiter",
    "get_redis_config",
    "get_distributed_rate_limiter",
    "reset_distributed_rate_limiter",
]
