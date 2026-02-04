"""
Distributed Email Rate Limiter.

Provides Redis-backed distributed rate limiting for email sending across
multiple application instances. Supports per-provider and per-tenant limits.

Algorithms:
- Token bucket for smooth rate limiting
- Sliding window for accurate hourly/daily counts

Provider Limits (defaults, configurable per-tenant):
- Gmail: 500/day (personal), 2000/day (Google Workspace)
- Microsoft 365: 10,000/day
- SendGrid: Varies by plan
- AWS SES: Varies by region and account

Usage:
    from aragora.integrations.email_rate_limiter import get_email_rate_limiter

    limiter = get_email_rate_limiter()

    # Check before sending
    if await limiter.acquire("tenant_123", "gmail"):
        # Send email
        await send_email(...)
    else:
        # Rate limited, queue for later
        await queue_email(...)

    # Get current usage
    usage = await limiter.get_usage("tenant_123", "gmail")
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RateLimitWindow(Enum):
    """Rate limit time windows."""

    MINUTE = 60
    HOUR = 3600
    DAY = 86400


@dataclass
class ProviderLimits:
    """Rate limits for an email provider."""

    per_minute: int = 60
    per_hour: int = 500
    per_day: int = 2000

    # Burst allowance (token bucket max)
    burst_size: int = 20

    # Refill rate (tokens per second)
    refill_rate: float = 1.0


# Default provider limits
DEFAULT_PROVIDER_LIMITS: dict[str, ProviderLimits] = {
    "gmail": ProviderLimits(
        per_minute=30,
        per_hour=100,
        per_day=500,
        burst_size=10,
        refill_rate=0.5,
    ),
    "gmail_workspace": ProviderLimits(
        per_minute=100,
        per_hour=500,
        per_day=2000,
        burst_size=30,
        refill_rate=2.0,
    ),
    "microsoft": ProviderLimits(
        per_minute=100,
        per_hour=1000,
        per_day=10000,
        burst_size=50,
        refill_rate=3.0,
    ),
    "sendgrid": ProviderLimits(
        per_minute=100,
        per_hour=2000,
        per_day=50000,
        burst_size=100,
        refill_rate=5.0,
    ),
    "ses": ProviderLimits(
        per_minute=100,
        per_hour=1000,
        per_day=10000,
        burst_size=50,
        refill_rate=3.0,
    ),
    "smtp": ProviderLimits(
        per_minute=30,
        per_hour=200,
        per_day=1000,
        burst_size=15,
        refill_rate=1.0,
    ),
}


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int = 0
    reset_at: datetime | None = None
    retry_after: float = 0.0
    limit_type: str = ""  # "minute", "hour", "day", "burst"

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_at": self.reset_at.isoformat() if self.reset_at else None,
            "retry_after": self.retry_after,
            "limit_type": self.limit_type,
        }


@dataclass
class UsageStats:
    """Current usage statistics."""

    tenant_id: str
    provider: str
    minute_count: int = 0
    hour_count: int = 0
    day_count: int = 0
    bucket_tokens: float = 0.0
    minute_reset: datetime | None = None
    hour_reset: datetime | None = None
    day_reset: datetime | None = None
    limits: ProviderLimits = field(default_factory=ProviderLimits)

    @property
    def minute_remaining(self) -> int:
        return max(0, self.limits.per_minute - self.minute_count)

    @property
    def hour_remaining(self) -> int:
        return max(0, self.limits.per_hour - self.hour_count)

    @property
    def day_remaining(self) -> int:
        return max(0, self.limits.per_day - self.day_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "provider": self.provider,
            "minute": {
                "count": self.minute_count,
                "limit": self.limits.per_minute,
                "remaining": self.minute_remaining,
                "reset_at": self.minute_reset.isoformat() if self.minute_reset else None,
            },
            "hour": {
                "count": self.hour_count,
                "limit": self.limits.per_hour,
                "remaining": self.hour_remaining,
                "reset_at": self.hour_reset.isoformat() if self.hour_reset else None,
            },
            "day": {
                "count": self.day_count,
                "limit": self.limits.per_day,
                "remaining": self.day_remaining,
                "reset_at": self.day_reset.isoformat() if self.day_reset else None,
            },
            "burst": {
                "tokens": self.bucket_tokens,
                "max": self.limits.burst_size,
            },
        }


class EmailRateLimiter:
    """
    Distributed rate limiter for email sending.

    Uses Redis for distributed coordination with in-memory fallback.
    Implements both token bucket (for burst control) and sliding window
    (for accurate hourly/daily counts).
    """

    REDIS_PREFIX = "aragora:email:ratelimit"

    def __init__(
        self,
        redis_url: str | None = None,
        default_limits: dict[str, ProviderLimits] | None = None,
    ):
        self._redis_url = redis_url or os.environ.get("ARAGORA_REDIS_URL", "redis://localhost:6379")
        self._redis: Any | None = None
        self._redis_checked = False
        self._default_limits = default_limits or DEFAULT_PROVIDER_LIMITS

        # In-memory fallback (per-instance only)
        self._local_counts: dict[str, dict[str, int]] = {}
        self._local_timestamps: dict[str, dict[str, float]] = {}
        self._local_buckets: dict[str, float] = {}
        self._local_bucket_times: dict[str, float] = {}
        self._lock = asyncio.Lock()

        # Custom tenant limits
        self._tenant_limits: dict[str, dict[str, ProviderLimits]] = {}

    def _get_redis(self) -> Any | None:
        """Get Redis client (lazy initialization)."""
        if self._redis_checked:
            return self._redis

        try:
            import redis

            self._redis = redis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)
            self._redis.ping()
            self._redis_checked = True
            logger.info("Redis connected for email rate limiter")
        except Exception as e:
            logger.debug(f"Redis not available for rate limiter: {e}")
            self._redis = None
            self._redis_checked = True

        return self._redis

    def _get_limits(self, tenant_id: str, provider: str) -> ProviderLimits:
        """Get rate limits for a tenant/provider combination."""
        # Check tenant-specific limits first
        if tenant_id in self._tenant_limits:
            if provider in self._tenant_limits[tenant_id]:
                return self._tenant_limits[tenant_id][provider]

        # Fall back to default provider limits
        return self._default_limits.get(provider, ProviderLimits())

    def set_tenant_limits(
        self,
        tenant_id: str,
        provider: str,
        limits: ProviderLimits,
    ) -> None:
        """Set custom rate limits for a tenant/provider."""
        if tenant_id not in self._tenant_limits:
            self._tenant_limits[tenant_id] = {}
        self._tenant_limits[tenant_id][provider] = limits
        logger.debug(f"Set custom rate limits for {tenant_id}/{provider}")

    def _key(self, tenant_id: str, provider: str, window: str) -> str:
        """Generate Redis key for rate limit counter."""
        return f"{self.REDIS_PREFIX}:{tenant_id}:{provider}:{window}"

    def _bucket_key(self, tenant_id: str, provider: str) -> str:
        """Generate Redis key for token bucket."""
        return f"{self.REDIS_PREFIX}:{tenant_id}:{provider}:bucket"

    async def acquire(
        self,
        tenant_id: str,
        provider: str,
        count: int = 1,
    ) -> RateLimitResult:
        """
        Attempt to acquire rate limit tokens.

        Args:
            tenant_id: The tenant ID
            provider: The email provider
            count: Number of tokens to acquire (default 1)

        Returns:
            RateLimitResult indicating if the request is allowed
        """
        limits = self._get_limits(tenant_id, provider)
        redis = self._get_redis()

        if redis:
            return await self._acquire_redis(tenant_id, provider, limits, count)
        else:
            return await self._acquire_local(tenant_id, provider, limits, count)

    async def _acquire_redis(
        self,
        tenant_id: str,
        provider: str,
        limits: ProviderLimits,
        count: int,
    ) -> RateLimitResult:
        """Acquire using Redis for distributed coordination."""
        redis = self._redis
        now = time.time()
        # now_dt could be used for logging in the future

        # Use Lua script for atomic operations
        lua_script = """
        local bucket_key = KEYS[1]
        local minute_key = KEYS[2]
        local hour_key = KEYS[3]
        local day_key = KEYS[4]

        local burst_size = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local minute_limit = tonumber(ARGV[3])
        local hour_limit = tonumber(ARGV[4])
        local day_limit = tonumber(ARGV[5])
        local count = tonumber(ARGV[6])
        local now = tonumber(ARGV[7])

        -- Token bucket refill
        local bucket_data = redis.call('HMGET', bucket_key, 'tokens', 'last_update')
        local tokens = tonumber(bucket_data[1]) or burst_size
        local last_update = tonumber(bucket_data[2]) or now

        local elapsed = now - last_update
        tokens = math.min(burst_size, tokens + elapsed * refill_rate)

        -- Check burst limit first
        if tokens < count then
            return {0, tokens, 0, 0, 0, 'burst'}
        end

        -- Check sliding window counters
        local minute_count = tonumber(redis.call('GET', minute_key)) or 0
        local hour_count = tonumber(redis.call('GET', hour_key)) or 0
        local day_count = tonumber(redis.call('GET', day_key)) or 0

        if minute_count + count > minute_limit then
            return {0, minute_limit - minute_count, minute_count, hour_count, day_count, 'minute'}
        end

        if hour_count + count > hour_limit then
            return {0, hour_limit - hour_count, minute_count, hour_count, day_count, 'hour'}
        end

        if day_count + count > day_limit then
            return {0, day_limit - day_count, minute_count, hour_count, day_count, 'day'}
        end

        -- All checks passed, consume tokens and increment counters
        tokens = tokens - count
        redis.call('HMSET', bucket_key, 'tokens', tokens, 'last_update', now)
        redis.call('EXPIRE', bucket_key, 86400)

        redis.call('INCRBY', minute_key, count)
        redis.call('EXPIRE', minute_key, 60)

        redis.call('INCRBY', hour_key, count)
        redis.call('EXPIRE', hour_key, 3600)

        redis.call('INCRBY', day_key, count)
        redis.call('EXPIRE', day_key, 86400)

        return {1, tokens, minute_count + count, hour_count + count, day_count + count, 'ok'}
        """

        try:
            result = redis.eval(
                lua_script,
                4,  # Number of keys
                self._bucket_key(tenant_id, provider),
                self._key(tenant_id, provider, "minute"),
                self._key(tenant_id, provider, "hour"),
                self._key(tenant_id, provider, "day"),
                limits.burst_size,
                limits.refill_rate,
                limits.per_minute,
                limits.per_hour,
                limits.per_day,
                count,
                now,
            )

            allowed = bool(result[0])
            remaining = int(result[1])
            limit_type = result[5].decode() if isinstance(result[5], bytes) else result[5]

            if not allowed:
                # Calculate retry_after based on limit type
                if limit_type == "burst":
                    retry_after = (count - remaining) / limits.refill_rate
                elif limit_type == "minute":
                    retry_after = 60 - (now % 60)
                elif limit_type == "hour":
                    retry_after = 3600 - (now % 3600)
                else:
                    retry_after = 86400 - (now % 86400)

                return RateLimitResult(
                    allowed=False,
                    remaining=remaining,
                    retry_after=retry_after,
                    limit_type=limit_type,
                )

            return RateLimitResult(
                allowed=True,
                remaining=remaining,
                limit_type="ok",
            )

        except Exception as e:
            logger.warning(f"Redis rate limit error, falling back to local: {e}")
            return await self._acquire_local(tenant_id, provider, limits, count)

    async def _acquire_local(
        self,
        tenant_id: str,
        provider: str,
        limits: ProviderLimits,
        count: int,
    ) -> RateLimitResult:
        """Acquire using local in-memory state (fallback)."""
        now = time.time()
        # now_dt could be used for logging in the future
        key = f"{tenant_id}:{provider}"

        async with self._lock:
            # Initialize if needed
            if key not in self._local_counts:
                self._local_counts[key] = {"minute": 0, "hour": 0, "day": 0}
                self._local_timestamps[key] = {
                    "minute": now,
                    "hour": now,
                    "day": now,
                }
                self._local_buckets[key] = float(limits.burst_size)
                self._local_bucket_times[key] = now

            counts = self._local_counts[key]
            timestamps = self._local_timestamps[key]

            # Reset expired windows
            if now - timestamps["minute"] >= 60:
                counts["minute"] = 0
                timestamps["minute"] = now
            if now - timestamps["hour"] >= 3600:
                counts["hour"] = 0
                timestamps["hour"] = now
            if now - timestamps["day"] >= 86400:
                counts["day"] = 0
                timestamps["day"] = now

            # Refill token bucket
            elapsed = now - self._local_bucket_times[key]
            self._local_buckets[key] = min(
                limits.burst_size,
                self._local_buckets[key] + elapsed * limits.refill_rate,
            )
            self._local_bucket_times[key] = now

            # Check burst limit
            if self._local_buckets[key] < count:
                retry_after = (count - self._local_buckets[key]) / limits.refill_rate
                return RateLimitResult(
                    allowed=False,
                    remaining=int(self._local_buckets[key]),
                    retry_after=retry_after,
                    limit_type="burst",
                )

            # Check window limits
            if counts["minute"] + count > limits.per_minute:
                return RateLimitResult(
                    allowed=False,
                    remaining=limits.per_minute - counts["minute"],
                    retry_after=60 - (now - timestamps["minute"]),
                    limit_type="minute",
                )

            if counts["hour"] + count > limits.per_hour:
                return RateLimitResult(
                    allowed=False,
                    remaining=limits.per_hour - counts["hour"],
                    retry_after=3600 - (now - timestamps["hour"]),
                    limit_type="hour",
                )

            if counts["day"] + count > limits.per_day:
                return RateLimitResult(
                    allowed=False,
                    remaining=limits.per_day - counts["day"],
                    retry_after=86400 - (now - timestamps["day"]),
                    limit_type="day",
                )

            # All checks passed, consume
            self._local_buckets[key] -= count
            counts["minute"] += count
            counts["hour"] += count
            counts["day"] += count

            return RateLimitResult(
                allowed=True,
                remaining=int(self._local_buckets[key]),
                limit_type="ok",
            )

    async def get_usage(self, tenant_id: str, provider: str) -> UsageStats:
        """Get current usage statistics."""
        limits = self._get_limits(tenant_id, provider)
        redis = self._get_redis()
        now = time.time()
        now_dt = datetime.now(timezone.utc)

        if redis:
            try:
                pipe = redis.pipeline()
                pipe.get(self._key(tenant_id, provider, "minute"))
                pipe.get(self._key(tenant_id, provider, "hour"))
                pipe.get(self._key(tenant_id, provider, "day"))
                pipe.hgetall(self._bucket_key(tenant_id, provider))
                results = pipe.execute()

                minute_count = int(results[0] or 0)
                hour_count = int(results[1] or 0)
                day_count = int(results[2] or 0)
                bucket_data = results[3] or {}
                tokens = float(bucket_data.get("tokens", limits.burst_size))
                last_update = float(bucket_data.get("last_update", now))

                # Refill tokens for display
                elapsed = now - last_update
                tokens = min(limits.burst_size, tokens + elapsed * limits.refill_rate)

                return UsageStats(
                    tenant_id=tenant_id,
                    provider=provider,
                    minute_count=minute_count,
                    hour_count=hour_count,
                    day_count=day_count,
                    bucket_tokens=tokens,
                    minute_reset=now_dt + timedelta(seconds=60 - (now % 60)),
                    hour_reset=now_dt + timedelta(seconds=3600 - (now % 3600)),
                    day_reset=now_dt + timedelta(seconds=86400 - (now % 86400)),
                    limits=limits,
                )
            except Exception as e:
                logger.warning(f"Redis usage fetch failed: {e}")

        # Local fallback
        key = f"{tenant_id}:{provider}"
        async with self._lock:
            counts = self._local_counts.get(key, {"minute": 0, "hour": 0, "day": 0})
            timestamps = self._local_timestamps.get(key, {"minute": now, "hour": now, "day": now})
            tokens = self._local_buckets.get(key, float(limits.burst_size))

            return UsageStats(
                tenant_id=tenant_id,
                provider=provider,
                minute_count=counts["minute"],
                hour_count=counts["hour"],
                day_count=counts["day"],
                bucket_tokens=tokens,
                minute_reset=now_dt + timedelta(seconds=60 - (now - timestamps["minute"])),
                hour_reset=now_dt + timedelta(seconds=3600 - (now - timestamps["hour"])),
                day_reset=now_dt + timedelta(seconds=86400 - (now - timestamps["day"])),
                limits=limits,
            )

    async def reset(self, tenant_id: str, provider: str) -> None:
        """Reset rate limits for a tenant/provider (admin use)."""
        redis = self._get_redis()

        if redis:
            try:
                pipe = redis.pipeline()
                pipe.delete(self._key(tenant_id, provider, "minute"))
                pipe.delete(self._key(tenant_id, provider, "hour"))
                pipe.delete(self._key(tenant_id, provider, "day"))
                pipe.delete(self._bucket_key(tenant_id, provider))
                pipe.execute()
            except Exception as e:
                logger.warning(f"Redis reset failed: {e}")

        # Also reset local state
        key = f"{tenant_id}:{provider}"
        async with self._lock:
            self._local_counts.pop(key, None)
            self._local_timestamps.pop(key, None)
            self._local_buckets.pop(key, None)
            self._local_bucket_times.pop(key, None)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            self._redis.close()
            self._redis = None
            self._redis_checked = False


# =============================================================================
# Global Rate Limiter Factory
# =============================================================================

_email_rate_limiter: EmailRateLimiter | None = None
_email_rate_limiter_lock = threading.Lock()


def get_email_rate_limiter() -> EmailRateLimiter:
    """Get or create the global email rate limiter."""
    global _email_rate_limiter

    if _email_rate_limiter is not None:
        return _email_rate_limiter

    with _email_rate_limiter_lock:
        if _email_rate_limiter is not None:
            return _email_rate_limiter

        _email_rate_limiter = EmailRateLimiter()
        return _email_rate_limiter


def set_email_rate_limiter(limiter: EmailRateLimiter) -> None:
    """Set custom email rate limiter (for testing)."""
    global _email_rate_limiter
    _email_rate_limiter = limiter


def reset_email_rate_limiter() -> None:
    """Reset the global email rate limiter (for testing)."""
    global _email_rate_limiter
    if _email_rate_limiter:
        _email_rate_limiter.close()
    _email_rate_limiter = None


__all__ = [
    "EmailRateLimiter",
    "ProviderLimits",
    "RateLimitResult",
    "RateLimitWindow",
    "UsageStats",
    "DEFAULT_PROVIDER_LIMITS",
    "get_email_rate_limiter",
    "set_email_rate_limiter",
    "reset_email_rate_limiter",
]
