"""
Unified token bucket rate limiter.

Provides a reusable token bucket implementation that can be used by:
- API agent rate limiters (ProviderRateLimiter, OpenRouterRateLimiter)
- Event rate limiters (EventRateLimiter)
- Any other component needing rate limiting

Features:
- Thread-safe synchronous acquisition
- Async-friendly acquisition with asyncio.Lock
- Configurable rate and burst capacity
- Optional key-based buckets for multi-tenant limiting
- Statistics tracking
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenBucketStats:
    """Statistics for a token bucket."""

    acquired: int = 0
    rejected: int = 0
    total_wait_ms: float = 0.0

    def record_acquired(self, wait_ms: float = 0.0) -> None:
        """Record a successful token acquisition."""
        self.acquired += 1
        self.total_wait_ms += wait_ms

    def record_rejected(self) -> None:
        """Record a rejected acquisition (timeout)."""
        self.rejected += 1

    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "acquired": self.acquired,
            "rejected": self.rejected,
            "total_wait_ms": round(self.total_wait_ms, 2),
            "avg_wait_ms": (
                round(self.total_wait_ms / self.acquired, 2) if self.acquired > 0 else 0.0
            ),
        }


class TokenBucket:
    """
    Thread-safe token bucket rate limiter.

    Uses the standard token bucket algorithm where tokens are added at a fixed
    rate up to a maximum capacity (burst). Each request consumes one token.

    Features:
    - Synchronous `try_acquire()` for non-blocking checks
    - Synchronous `acquire()` with optional blocking and timeout
    - Async `acquire_async()` for non-blocking async contexts
    - API header updates for dynamic rate adjustment
    - Statistics tracking

    Example:
        # Create a rate limiter: 100 requests/minute, burst of 20
        bucket = TokenBucket(rate_per_minute=100, burst=20)

        # Non-blocking check
        if bucket.try_acquire():
            make_request()

        # Blocking with timeout (sync)
        if bucket.acquire(timeout=5.0):
            make_request()

        # Async acquisition
        if await bucket.acquire_async(timeout=5.0):
            await make_request()
    """

    def __init__(
        self,
        rate_per_minute: float = 60.0,
        burst: int = 10,
        name: str = "default",
    ):
        """
        Initialize token bucket.

        Args:
            rate_per_minute: Tokens (requests) allowed per minute
            burst: Maximum token capacity (allows burst traffic)
            name: Optional name for logging/debugging
        """
        self.rate_per_minute = rate_per_minute
        self.burst = burst
        self.name = name

        # Token state
        self._tokens = float(burst)
        self._last_refill = time.monotonic()

        # Thread safety
        self._sync_lock = threading.Lock()
        self._async_lock: Optional[asyncio.Lock] = None

        # API-reported limits (optional, from response headers)
        self._api_limit: Optional[int] = None
        self._api_remaining: Optional[int] = None
        self._api_reset: Optional[float] = None

        # Statistics
        self._stats = TokenBucketStats()

    def _get_async_lock(self) -> asyncio.Lock:
        """Get or create the async lock (lazy initialization per event loop)."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _refill(self) -> None:
        """Refill tokens based on elapsed time (must hold lock)."""
        now = time.monotonic()
        elapsed_minutes = (now - self._last_refill) / 60.0
        refill_amount = elapsed_minutes * self.rate_per_minute
        self._tokens = min(self.burst, self._tokens + refill_amount)
        self._last_refill = now

    def _consume_if_available(self) -> bool:
        """Try to consume a token (must hold lock). Returns True if successful."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _check_api_limit(self) -> Optional[float]:
        """Check API-reported limits and return wait time if needed (must hold lock)."""
        if self._api_remaining is not None and self._api_remaining <= 0:
            if self._api_reset:
                wait_time = self._api_reset - time.time()
                if wait_time > 0:
                    return min(wait_time, 1.0)  # Cap individual wait at 1s
        return None

    def _time_until_token(self) -> float:
        """Calculate time until next token is available (must hold lock)."""
        return 60.0 / self.rate_per_minute

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens without blocking.

        Args:
            tokens: Number of tokens to acquire (default 1)

        Returns:
            True if tokens were acquired, False if rate limited
        """
        with self._sync_lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.record_acquired()
                return True
            self._stats.record_rejected()
            return False

    def acquire(self, timeout: float = 0.0, tokens: int = 1) -> bool:
        """
        Acquire tokens with optional blocking.

        Args:
            timeout: Maximum time to wait (0 = non-blocking)
            tokens: Number of tokens to acquire (default 1)

        Returns:
            True if tokens were acquired, False if timed out

        Note:
            For async contexts, use acquire_async() instead to avoid
            blocking the event loop.
        """
        if timeout <= 0:
            return self.try_acquire(tokens)

        start = time.monotonic()
        deadline = start + timeout

        while True:
            with self._sync_lock:
                self._refill()

                # Check API-reported limits
                api_wait = self._check_api_limit()
                if api_wait:
                    wait_time = api_wait
                elif self._tokens >= tokens:
                    self._tokens -= tokens
                    self._stats.record_acquired(wait_ms=(time.monotonic() - start) * 1000)
                    return True
                else:
                    wait_time = self._time_until_token()

            # Check timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._sync_lock:
                    self._stats.record_rejected()
                return False

            # Wait and retry
            time.sleep(min(wait_time, remaining, 0.1))

    async def acquire_async(self, timeout: float = 30.0, tokens: int = 1) -> bool:
        """
        Acquire tokens asynchronously.

        Uses asyncio.Lock to avoid blocking the event loop while waiting.
        Other coroutines can run while this one waits for tokens.

        Args:
            timeout: Maximum time to wait in seconds
            tokens: Number of tokens to acquire (default 1)

        Returns:
            True if tokens were acquired, False if timed out
        """
        start = time.monotonic()
        deadline = start + timeout
        async_lock = self._get_async_lock()

        while True:
            # Check state inside lock, sleep outside
            wait_time: Optional[float] = None

            async with async_lock:
                self._refill()

                # Check API-reported limits
                api_wait = self._check_api_limit()
                if api_wait:
                    wait_time = api_wait
                elif self._tokens >= tokens:
                    self._tokens -= tokens
                    self._stats.record_acquired(wait_ms=(time.monotonic() - start) * 1000)
                    return True
                else:
                    wait_time = self._time_until_token()

            # Check timeout
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                async with async_lock:
                    self._stats.record_rejected()
                return False

            # Wait outside lock so other coroutines can acquire
            await asyncio.sleep(min(wait_time, remaining, 0.1))

    def release(self, tokens: int = 1) -> None:
        """
        Release tokens back to the bucket (e.g., on request error).

        Useful for retry scenarios where the request didn't actually use
        the rate limit slot.

        Args:
            tokens: Number of tokens to release (default 1)
        """
        with self._sync_lock:
            self._tokens = min(self.burst, self._tokens + tokens)

    def update_from_headers(self, headers: dict) -> None:
        """
        Update rate limit state from API response headers.

        Supports common header formats:
        - X-RateLimit-Limit / x-ratelimit-limit / RateLimit-Limit
        - X-RateLimit-Remaining / x-ratelimit-remaining / RateLimit-Remaining
        - X-RateLimit-Reset / x-ratelimit-reset / RateLimit-Reset

        Args:
            headers: Response headers dict
        """
        with self._sync_lock:
            # Try common header name variants
            limit_headers = ["X-RateLimit-Limit", "x-ratelimit-limit", "RateLimit-Limit"]
            remaining_headers = [
                "X-RateLimit-Remaining",
                "x-ratelimit-remaining",
                "RateLimit-Remaining",
            ]
            reset_headers = ["X-RateLimit-Reset", "x-ratelimit-reset", "RateLimit-Reset"]

            for h in limit_headers:
                if h in headers:
                    try:
                        self._api_limit = int(headers[h])
                        break
                    except (ValueError, TypeError):
                        pass

            for h in remaining_headers:
                if h in headers:
                    try:
                        self._api_remaining = int(headers[h])
                        break
                    except (ValueError, TypeError):
                        pass

            for h in reset_headers:
                if h in headers:
                    try:
                        self._api_reset = float(headers[h])
                        break
                    except (ValueError, TypeError):
                        pass

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (thread-safe, triggers refill)."""
        with self._sync_lock:
            self._refill()
            return self._tokens

    @property
    def stats(self) -> dict:
        """Get current statistics."""
        with self._sync_lock:
            return {
                "name": self.name,
                "rate_per_minute": self.rate_per_minute,
                "burst": self.burst,
                "tokens_available": round(self._tokens, 2),
                "api_limit": self._api_limit,
                "api_remaining": self._api_remaining,
                **self._stats.to_dict(),
            }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._sync_lock:
            self._stats = TokenBucketStats()


class KeyedTokenBucket:
    """
    Multi-key token bucket for per-key rate limiting.

    Maintains separate token buckets for each key, useful for:
    - Per-user rate limiting
    - Per-event-type limiting
    - Per-endpoint limiting

    Example:
        # Limit each user to 10 requests/minute
        limiter = KeyedTokenBucket(rate_per_minute=10, burst=5)

        if limiter.try_acquire("user-123"):
            process_request()
    """

    def __init__(
        self,
        rate_per_minute: float = 60.0,
        burst: int = 10,
        name: str = "keyed",
    ):
        """
        Initialize keyed token bucket.

        Args:
            rate_per_minute: Tokens (requests) allowed per minute per key
            burst: Maximum token capacity per key
            name: Optional name for logging/debugging
        """
        self.rate_per_minute = rate_per_minute
        self.burst = burst
        self.name = name

        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        self._stats = TokenBucketStats()

    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create a bucket for the given key."""
        if key not in self._buckets:
            with self._lock:
                if key not in self._buckets:
                    self._buckets[key] = TokenBucket(
                        rate_per_minute=self.rate_per_minute,
                        burst=self.burst,
                        name=f"{self.name}:{key}",
                    )
        return self._buckets[key]

    def try_acquire(self, key: str, tokens: int = 1) -> bool:
        """Try to acquire tokens for a key without blocking."""
        bucket = self._get_bucket(key)
        result = bucket.try_acquire(tokens)
        with self._lock:
            if result:
                self._stats.record_acquired()
            else:
                self._stats.record_rejected()
        return result

    def acquire(self, key: str, timeout: float = 0.0, tokens: int = 1) -> bool:
        """Acquire tokens for a key with optional blocking."""
        bucket = self._get_bucket(key)
        start = time.monotonic()
        result = bucket.acquire(timeout, tokens)
        with self._lock:
            if result:
                self._stats.record_acquired(wait_ms=(time.monotonic() - start) * 1000)
            else:
                self._stats.record_rejected()
        return result

    async def acquire_async(self, key: str, timeout: float = 30.0, tokens: int = 1) -> bool:
        """Acquire tokens for a key asynchronously."""
        bucket = self._get_bucket(key)
        start = time.monotonic()
        result = await bucket.acquire_async(timeout, tokens)
        with self._lock:
            if result:
                self._stats.record_acquired(wait_ms=(time.monotonic() - start) * 1000)
            else:
                self._stats.record_rejected()
        return result

    def release(self, key: str, tokens: int = 1) -> None:
        """Release tokens back to a key's bucket."""
        if key in self._buckets:
            self._buckets[key].release(tokens)

    @property
    def stats(self) -> dict:
        """Get aggregate statistics."""
        with self._lock:
            return {
                "name": self.name,
                "rate_per_minute": self.rate_per_minute,
                "burst": self.burst,
                "active_keys": len(self._buckets),
                **self._stats.to_dict(),
            }

    def get_key_stats(self, key: str) -> Optional[dict]:
        """Get statistics for a specific key."""
        if key in self._buckets:
            return self._buckets[key].stats
        return None

    def reset_stats(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._stats = TokenBucketStats()
            for bucket in self._buckets.values():
                bucket.reset_stats()

    def prune_inactive(self, max_age_seconds: float = 3600.0) -> int:
        """
        Remove buckets that haven't been used recently.

        Args:
            max_age_seconds: Remove buckets not accessed within this time

        Returns:
            Number of buckets removed
        """
        # Note: TokenBucket doesn't track last access time yet
        # This is a placeholder for future implementation
        return 0


__all__ = ["TokenBucket", "KeyedTokenBucket", "TokenBucketStats"]
