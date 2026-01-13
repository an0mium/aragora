"""
Token bucket implementations for rate limiting.

Provides in-memory and Redis-backed token bucket rate limiters.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, TYPE_CHECKING

from .base import BURST_MULTIPLIER

if TYPE_CHECKING:
    import redis

logger = logging.getLogger(__name__)


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
        if tokens <= 0:
            return False
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
            self._consume_sha = self.redis.script_load(self.CONSUME_SCRIPT)  # type: ignore[assignment]
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
            return bool(result[0])  # type: ignore[index]
        except Exception as e:
            logger.warning(f"Redis rate limit error, allowing request: {e}")
            return True  # Fail open on Redis errors

    def get_retry_after(self) -> float:
        """Get seconds until next token is available."""
        try:
            data: list[bytes | None] = self.redis.hmget(self.key, "tokens", "last_refill")  # type: ignore[assignment, arg-type]
            tokens = float(data[0]) if data[0] else float(self.burst_size)
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
            data: list[bytes | None] = self.redis.hmget(self.key, "tokens", "last_refill")  # type: ignore[assignment, arg-type]
            tokens = float(data[0]) if data[0] else float(self.burst_size)
            last_refill = float(data[1]) if data[1] else time.time()

            # Calculate refill since last access
            elapsed_minutes = (time.time() - last_refill) / 60.0
            refill_amount = elapsed_minutes * self.rate_per_minute
            tokens = min(self.burst_size, tokens + refill_amount)

            return max(0, int(tokens))
        except Exception as e:
            logger.debug(f"Error getting remaining tokens, defaulting to burst_size: {e}")
            return self.burst_size


__all__ = ["TokenBucket", "RedisTokenBucket"]
