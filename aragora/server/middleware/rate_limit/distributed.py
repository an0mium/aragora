"""
Distributed Rate Limiting Module.

Provides a unified interface for distributed rate limiting that:
- Enforces ARAGORA_RATE_LIMIT_STRICT mode in production
- Falls back to in-memory in dev mode when Redis is unavailable
- Integrates Prometheus metrics for all decisions
- Supports multi-instance coordination

Environment Variables:
    ARAGORA_RATE_LIMIT_STRICT: When "true", requires Redis for rate limiting
        (raises error if Redis unavailable in production mode)
    ARAGORA_ENV / ENVIRONMENT / NODE_ENV: Used to detect production mode
    REDIS_URL / ARAGORA_REDIS_URL: Redis connection URL
    ARAGORA_INSTANCE_ID: Unique identifier for this server instance

Usage:
    from aragora.server.middleware.rate_limit.distributed import (
        get_distributed_limiter,
        DistributedRateLimiter,
    )

    # Get the singleton distributed limiter
    limiter = get_distributed_limiter()

    # Check rate limit
    result = limiter.allow(
        client_ip="192.168.1.1",
        endpoint="/api/debates",
        tenant_id="tenant-123",
    )
    if not result.allowed:
        return error_response("Rate limited", 429)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Any

from .limiter import RateLimiter, RateLimitResult
from .metrics import (
    PROMETHEUS_AVAILABLE,
    record_backend_status,
    record_circuit_breaker_state,
    record_fallback_request,
    record_rate_limit_decision,
    record_redis_operation,
)
from .redis_limiter import (
    RateLimitCircuitBreaker,
    RedisRateLimiter,
    get_redis_client,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Strict mode: require Redis for rate limiting (raises error if unavailable)
STRICT_MODE = os.environ.get("ARAGORA_RATE_LIMIT_STRICT", "").lower() in (
    "1",
    "true",
    "yes",
)


def _is_production_mode() -> bool:
    """Check if running in production mode."""
    for env_var in ("ARAGORA_ENV", "ENVIRONMENT", "NODE_ENV"):
        value = os.environ.get(env_var, "").lower()
        if value in ("production", "prod"):
            return True
    return False


def _is_development_mode() -> bool:
    """Check if running in development mode."""
    for env_var in ("ARAGORA_ENV", "ENVIRONMENT", "NODE_ENV"):
        value = os.environ.get(env_var, "").lower()
        if value in ("development", "dev", "local"):
            return True
    # Default to dev mode if not explicitly set
    if not os.environ.get("ARAGORA_ENV") and not os.environ.get("ENVIRONMENT"):
        return True
    return False


# ============================================================================
# Distributed Rate Limiter
# ============================================================================


class DistributedRateLimiter:
    """
    Unified distributed rate limiter with Redis coordination.

    Provides:
    - Automatic Redis detection and fallback
    - Strict mode enforcement for production
    - Circuit breaker for fault tolerance
    - Prometheus metrics integration
    - Per-tenant rate limit tracking

    The limiter automatically uses Redis when available and falls back to
    in-memory rate limiting when Redis is unavailable (in dev mode) or when
    the circuit breaker is open.
    """

    def __init__(
        self,
        instance_id: str | None = None,
        strict_mode: bool | None = None,
        enable_metrics: bool = True,
    ):
        """
        Initialize distributed rate limiter.

        Args:
            instance_id: Unique identifier for this server instance
            strict_mode: Override strict mode setting (default: from env)
            enable_metrics: Whether to record Prometheus metrics

        Raises:
            RuntimeError: If strict_mode is True and Redis is unavailable
        """
        self.instance_id = instance_id or os.environ.get(
            "ARAGORA_INSTANCE_ID", f"instance-{os.getpid()}"
        )
        self.strict_mode = strict_mode if strict_mode is not None else STRICT_MODE
        self.enable_metrics = enable_metrics and PROMETHEUS_AVAILABLE

        self._redis_limiter: RedisRateLimiter | None = None
        self._memory_limiter: RateLimiter | None = None
        self._using_redis: bool = False
        self._initialized: bool = False
        self._lock = threading.Lock()

        # Track metrics
        self._total_requests = 0
        self._redis_requests = 0
        self._fallback_requests = 0
        self._last_circuit_state = "closed"

        # Eagerly initialize when strict mode is requested to enforce requirements.
        if self.strict_mode:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize the rate limiter backend."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Always create memory fallback
            self._memory_limiter = RateLimiter()

            # Try to create Redis limiter
            redis_client = get_redis_client()

            if redis_client is not None:
                try:
                    self._redis_limiter = RedisRateLimiter(
                        redis_client,
                        enable_circuit_breaker=True,
                        enable_distributed_metrics=True,
                        instance_id=self.instance_id,
                    )
                    self._using_redis = True
                    logger.info(
                        "Distributed rate limiter initialized with Redis backend (instance=%s)",
                        self.instance_id,
                    )
                except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
                    logger.warning("Failed to create Redis rate limiter: %s", e)
                    self._redis_limiter = None
                    self._using_redis = False
                except Exception as e:  # noqa: BLE001 - redis.exceptions.* don't inherit builtins.ConnectionError
                    if "redis" in type(e).__module__:
                        logger.warning("Failed to create Redis rate limiter: %s", e)
                        self._redis_limiter = None
                        self._using_redis = False
                    else:
                        raise
            else:
                self._using_redis = False

            # Enforce strict mode
            if self.strict_mode and not self._using_redis:
                if _is_production_mode():
                    raise RuntimeError(
                        "ARAGORA_RATE_LIMIT_STRICT is enabled but Redis is unavailable. "
                        "In production mode, Redis is required for distributed rate limiting. "
                        "Either configure REDIS_URL/ARAGORA_REDIS_URL or disable strict mode."
                    )
                elif _is_development_mode():
                    logger.warning(
                        "ARAGORA_RATE_LIMIT_STRICT is enabled but Redis is unavailable. "
                        "Falling back to in-memory rate limiting in development mode. "
                        "Rate limits will NOT be shared across server instances."
                    )
                else:
                    raise RuntimeError(
                        "ARAGORA_RATE_LIMIT_STRICT is enabled but Redis is unavailable. "
                        "Configure Redis or disable strict mode."
                    )

            if not self._using_redis:
                logger.info(
                    "Distributed rate limiter using in-memory backend (instance=%s)",
                    self.instance_id,
                )

            # Record initial backend status
            if self.enable_metrics:
                record_backend_status(self.instance_id, self._using_redis)

            self._initialized = True

    def configure_endpoint(
        self,
        endpoint: str,
        requests_per_minute: int,
        burst_size: int | None = None,
        key_type: str = "combined",
    ) -> None:
        """Configure rate limit for a specific endpoint.

        Configures both Redis and memory backends for consistency.

        Args:
            endpoint: API endpoint path (e.g., "/api/debates")
            requests_per_minute: Maximum requests per minute
            burst_size: Burst capacity (default: 2x rate)
            key_type: Key type ("ip", "token", "tenant", "combined")
        """
        self._initialize()

        if burst_size is None:
            burst_size = requests_per_minute

        if self._redis_limiter:
            self._redis_limiter.configure_endpoint(
                endpoint, requests_per_minute, burst_size, key_type
            )

        if self._memory_limiter:
            self._memory_limiter.configure_endpoint(
                endpoint, requests_per_minute, burst_size, key_type
            )

    def allow(
        self,
        client_ip: str,
        endpoint: str | None = None,
        token: str | None = None,
        tenant_id: str | None = None,
    ) -> RateLimitResult:
        """
        Check if a request should be allowed.

        Uses Redis if available, otherwise falls back to in-memory limiting.
        Records Prometheus metrics for all decisions.

        Args:
            client_ip: Client IP address
            endpoint: Optional API endpoint for per-endpoint limits
            token: Optional auth token for per-token limits
            tenant_id: Optional tenant ID for per-tenant limits

        Returns:
            RateLimitResult with allowed status and metadata
        """
        self._initialize()

        start_time = time.monotonic()
        result: RateLimitResult
        used_redis = False

        try:
            # Try Redis first if available
            if self._redis_limiter and self._using_redis:
                # Check circuit breaker state
                cb = self._redis_limiter._circuit_breaker
                if cb:
                    current_state = cb.state
                    if current_state != self._last_circuit_state:
                        self._last_circuit_state = current_state
                        if self.enable_metrics:
                            record_circuit_breaker_state(self.instance_id, current_state)

                    if current_state == RateLimitCircuitBreaker.OPEN:
                        # Circuit is open, use fallback
                        self._fallback_requests += 1
                        if self.enable_metrics:
                            record_fallback_request(self.instance_id)
                        result = self._memory_limiter.allow(client_ip, endpoint, token, tenant_id)
                    else:
                        # Try Redis
                        result = self._redis_limiter.allow(client_ip, endpoint, token, tenant_id)
                        used_redis = True
                        self._redis_requests += 1
                else:
                    result = self._redis_limiter.allow(client_ip, endpoint, token, tenant_id)
                    used_redis = True
                    self._redis_requests += 1
            else:
                # Use in-memory limiter
                result = self._memory_limiter.allow(client_ip, endpoint, token, tenant_id)

        except (ConnectionError, TimeoutError, OSError, ValueError, RuntimeError) as e:
            logger.warning("Rate limit check failed, using fallback: %s", e)
            self._fallback_requests += 1
            if self.enable_metrics:
                record_fallback_request(self.instance_id)
            result = self._memory_limiter.allow(client_ip, endpoint, token, tenant_id)
        except Exception as e:  # noqa: BLE001 - redis.exceptions.* don't inherit builtins.ConnectionError
            if "redis" in type(e).__module__:
                logger.warning("Rate limit check failed (redis), using fallback: %s", e)
                self._fallback_requests += 1
                if self.enable_metrics:
                    record_fallback_request(self.instance_id)
                result = self._memory_limiter.allow(client_ip, endpoint, token, tenant_id)
            else:
                raise

        # Record metrics
        self._total_requests += 1
        latency = time.monotonic() - start_time

        if self.enable_metrics:
            record_rate_limit_decision(
                endpoint=endpoint or "unknown",
                allowed=result.allowed,
                remaining=result.remaining,
                limit=result.limit,
                tenant_id=tenant_id,
            )
            if used_redis:
                record_redis_operation("check", success=True, latency_seconds=latency)

        # Also record tenant-specific rejection if applicable
        if not result.allowed and tenant_id:
            with self._lock:
                if not hasattr(self, "_tenant_rejections"):
                    self._tenant_rejections: dict[str, int] = {}
                key = f"{endpoint or 'unknown'}:{tenant_id}"
                self._tenant_rejections[key] = self._tenant_rejections.get(key, 0) + 1

        return result

    def get_client_key(self, handler: Any) -> str:
        """Extract client key from request handler.

        Args:
            handler: HTTP request handler

        Returns:
            Client identifier string
        """
        self._initialize()

        if self._redis_limiter:
            return self._redis_limiter.get_client_key(handler)
        return self._memory_limiter.get_client_key(handler)

    @property
    def is_using_redis(self) -> bool:
        """Check if Redis backend is active."""
        self._initialize()
        return self._using_redis

    @property
    def backend(self) -> str:
        """Get current backend type."""
        self._initialize()
        return "redis" if self._using_redis else "memory"

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive rate limiter statistics.

        Returns:
            Dictionary with backend status, metrics, and configuration
        """
        self._initialize()

        stats = {
            "instance_id": self.instance_id,
            "backend": self.backend,
            "strict_mode": self.strict_mode,
            "total_requests": self._total_requests,
            "redis_requests": self._redis_requests,
            "fallback_requests": self._fallback_requests,
            "tenant_rejections": getattr(self, "_tenant_rejections", {}),
        }

        if self._redis_limiter:
            stats["redis"] = self._redis_limiter.get_stats()
            if self._redis_limiter._enable_distributed_metrics:
                stats["distributed"] = self._redis_limiter.get_distributed_metrics()

        if self._memory_limiter:
            stats["memory"] = self._memory_limiter.get_stats()

        return stats

    def reset(self) -> None:
        """Reset all rate limiter state."""
        with self._lock:
            if self._redis_limiter:
                self._redis_limiter.reset()
            if self._memory_limiter:
                self._memory_limiter.reset()
            self._total_requests = 0
            self._redis_requests = 0
            self._fallback_requests = 0

    def cleanup(self, max_age_seconds: int = 300) -> int:
        """Cleanup stale entries.

        Args:
            max_age_seconds: Maximum age before cleanup

        Returns:
            Number of entries cleaned up
        """
        total = 0
        if self._redis_limiter:
            total += self._redis_limiter.cleanup(max_age_seconds)
        if self._memory_limiter:
            total += self._memory_limiter.cleanup(max_age_seconds)
        return total


# ============================================================================
# Module-level Singleton
# ============================================================================

_distributed_limiter: DistributedRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_distributed_limiter() -> DistributedRateLimiter:
    """
    Get the global distributed rate limiter instance.

    Creates a singleton instance on first call. Thread-safe.

    Returns:
        DistributedRateLimiter instance
    """
    global _distributed_limiter

    if _distributed_limiter is not None:
        return _distributed_limiter

    with _limiter_lock:
        if _distributed_limiter is None:
            _distributed_limiter = DistributedRateLimiter()
        return _distributed_limiter


def reset_distributed_limiter() -> None:
    """Reset the global distributed limiter (for testing).

    Only resets state (clears buckets); does NOT destroy the singleton.
    Destroying the singleton would orphan references captured in
    ``@rate_limit`` decorator closures at decoration time, causing those
    closures to accumulate state across tests without ever being reset.
    """
    global _distributed_limiter

    with _limiter_lock:
        if _distributed_limiter is not None:
            _distributed_limiter.reset()


def configure_distributed_endpoint(
    endpoint: str,
    requests_per_minute: int,
    burst_size: int | None = None,
    key_type: str = "ip",
) -> None:
    """Configure rate limit for an endpoint on the global limiter.

    Convenience function for configuring the singleton limiter.

    Args:
        endpoint: API endpoint path
        requests_per_minute: Maximum requests per minute
        burst_size: Burst capacity
        key_type: Key type for rate limiting
    """
    limiter = get_distributed_limiter()
    limiter.configure_endpoint(endpoint, requests_per_minute, burst_size, key_type)


__all__ = [
    "STRICT_MODE",
    "DistributedRateLimiter",
    "get_distributed_limiter",
    "reset_distributed_limiter",
    "configure_distributed_endpoint",
]
