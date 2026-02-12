"""
Rate limiter registry and global functions.

Provides centralized management of rate limiters via ServiceRegistry,
with automatic Redis detection and configuration.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import DEFAULT_RATE_LIMIT
from .limiter import RateLimiter
from .redis_limiter import (
    RedisRateLimiter,
    get_redis_client,
    reset_redis_client,
)

logger = logging.getLogger(__name__)


class RateLimiterRegistry:
    """Container for named rate limiters, managed via ServiceRegistry."""

    def __init__(self):
        self._limiters: dict[str, RateLimiter | RedisRateLimiter] = {}
        self._default_limiter: RateLimiter | RedisRateLimiter | None = None
        self._use_redis: bool | None = None

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
                except (ImportError, OSError, ConnectionError, ValueError, AttributeError) as e:
                    logger.warning(f"Failed to create Redis rate limiter: {e}")
                    self._default_limiter = RateLimiter()
                    self._use_redis = False
            else:
                self._default_limiter = RateLimiter()
                self._use_redis = False

            # Configure default endpoint limits
            # Tenant-aware endpoints use "tenant" key_type for per-tenant isolation
            self._default_limiter.configure_endpoint("/api/debates", 30, key_type="tenant")
            self._default_limiter.configure_endpoint("/api/debates/*", 60, key_type="tenant")
            self._default_limiter.configure_endpoint("/api/debates/*/fork", 5, key_type="tenant")
            self._default_limiter.configure_endpoint("/api/agent/*", 120, key_type="tenant")
            self._default_limiter.configure_endpoint("/api/leaderboard*", 60, key_type="ip")
            self._default_limiter.configure_endpoint("/api/pulse/*", 30, key_type="tenant")
            self._default_limiter.configure_endpoint(
                "/api/memory/continuum/cleanup", 2, key_type="tenant"
            )
            self._default_limiter.configure_endpoint("/api/memory/*", 60, key_type="tenant")

            # CPU-intensive endpoints (stricter limits, tenant-aware)
            self._default_limiter.configure_endpoint(
                "/api/debates/*/broadcast",
                3,
                key_type="tenant",  # Audio generation
            )
            self._default_limiter.configure_endpoint(
                "/api/probes/*",
                10,
                key_type="tenant",  # Capability probes
            )
            self._default_limiter.configure_endpoint(
                "/api/verification/*",
                10,
                key_type="tenant",  # Proof verification
            )
            self._default_limiter.configure_endpoint(
                "/api/video/*",
                2,
                key_type="tenant",  # Video generation
            )

            # Gauntlet endpoints (stress testing - strict limits, tenant-aware)
            self._default_limiter.configure_endpoint(
                "/api/gauntlet/*",
                5,
                key_type="tenant",  # Adversarial stress testing
            )
            self._default_limiter.configure_endpoint(
                "/api/gauntlet/run",
                3,
                key_type="tenant",  # Gauntlet runs are expensive
            )

            # Billing endpoints (financial operations)
            self._default_limiter.configure_endpoint(
                "/api/billing/*",
                20,
                key_type="token",  # Rate limit by auth token
            )
            self._default_limiter.configure_endpoint(
                "/api/billing/checkout",
                5,
                key_type="token",  # Checkout creation
            )
            self._default_limiter.configure_endpoint(
                "/api/webhooks/stripe",
                100,
                key_type="ip",  # Stripe webhooks (higher limit)
            )

            # Admin endpoints (administrative operations)
            self._default_limiter.configure_endpoint(
                "/api/admin/*",
                30,
                key_type="token",  # Admin operations
            )
            self._default_limiter.configure_endpoint(
                "/api/admin/security/*",
                10,
                key_type="token",  # Security operations (stricter)
            )

            # Streaming endpoints (concurrent connections, tenant-aware)
            self._default_limiter.configure_endpoint(
                "/api/stream/*",
                10,
                key_type="tenant",  # WebSocket/SSE streams
            )
            self._default_limiter.configure_endpoint(
                "/api/v1/stream/*",
                10,
                key_type="tenant",  # Versioned streaming endpoints
            )

            # Knowledge mound endpoints (tenant-aware)
            self._default_limiter.configure_endpoint(
                "/api/knowledge/*",
                30,
                key_type="tenant",  # Knowledge operations
            )
            self._default_limiter.configure_endpoint(
                "/api/knowledge/search",
                20,
                key_type="tenant",  # Search is more expensive
            )

            # OAuth endpoints (auth flows)
            self._default_limiter.configure_endpoint(
                "/api/oauth/*",
                20,
                key_type="ip",  # OAuth operations
            )
            self._default_limiter.configure_endpoint(
                "/api/auth/*",
                30,
                key_type="ip",  # Auth operations
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
    ) -> RateLimiter | RedisRateLimiter:
        """Get or create a named rate limiter.

        When Redis is available, returns a Redis-backed limiter for distributed
        rate limiting across multiple server instances. Falls back to in-memory
        limiter if Redis is unavailable.

        Args:
            name: Unique name for this limiter (e.g., "debate_create").
            requests_per_minute: Max requests per minute.
            burst: Burst capacity (default: 2x rate).

        Returns:
            RateLimiter or RedisRateLimiter instance.
        """
        if name not in self._limiters:
            # Ensure default limiter is initialized (triggers Redis detection)
            self.get_default()

            if self._use_redis:
                # Create Redis-backed limiter for distributed rate limiting
                redis_client = get_redis_client()
                if redis_client is not None:
                    try:
                        from aragora.config.settings import get_settings

                        settings = get_settings()
                        limiter = RedisRateLimiter(
                            redis_client,
                            key_prefix=f"{settings.rate_limit.redis_key_prefix}:{name}",
                            ttl_seconds=settings.rate_limit.redis_ttl_seconds,
                        )
                        # Configure the default limit
                        limiter.configure_endpoint("*", requests_per_minute, key_type="ip")
                        self._limiters[name] = limiter
                        logger.debug(f"Created Redis-backed rate limiter: {name}")
                    except (ImportError, OSError, ConnectionError, ValueError, AttributeError) as e:
                        logger.warning(f"Failed to create Redis rate limiter for {name}: {e}")
                        self._limiters[name] = RateLimiter(
                            default_limit=requests_per_minute,
                            ip_limit=requests_per_minute,
                        )
                else:
                    self._limiters[name] = RateLimiter(
                        default_limit=requests_per_minute,
                        ip_limit=requests_per_minute,
                    )
            else:
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
    from aragora.services import ServiceRegistry

    registry = ServiceRegistry.get()
    if not registry.has(RateLimiterRegistry):
        registry.register_factory(RateLimiterRegistry, RateLimiterRegistry)
    return registry.resolve(RateLimiterRegistry)


def get_rate_limiter(
    name: str = "_default",
    requests_per_minute: int = DEFAULT_RATE_LIMIT,
    burst: int | None = None,
) -> RateLimiter | RedisRateLimiter:
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
    from aragora.services import ServiceRegistry

    registry = ServiceRegistry.get()
    if registry.has(RateLimiterRegistry):
        registry.resolve(RateLimiterRegistry).reset()
        registry.unregister(RateLimiterRegistry)

    # Also reset Redis client
    reset_redis_client()


__all__ = [
    "RateLimiterRegistry",
    "get_rate_limiter",
    "cleanup_rate_limiters",
    "reset_rate_limiters",
]
