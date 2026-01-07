"""
OpenRouter rate limiting infrastructure.

Provides token bucket rate limiting for OpenRouter API calls,
with configurable tiers and thread-safe operation.
"""

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterTier:
    """Rate limit configuration for an OpenRouter pricing tier."""
    name: str
    requests_per_minute: int
    tokens_per_minute: int = 0  # 0 = unlimited
    burst_size: int = 10  # Allow short bursts


# OpenRouter tier configurations (based on their pricing)
OPENROUTER_TIERS = {
    "free": OpenRouterTier(name="free", requests_per_minute=20, burst_size=5),
    "basic": OpenRouterTier(name="basic", requests_per_minute=60, burst_size=15),
    "standard": OpenRouterTier(name="standard", requests_per_minute=200, burst_size=30),
    "premium": OpenRouterTier(name="premium", requests_per_minute=500, burst_size=50),
    "unlimited": OpenRouterTier(name="unlimited", requests_per_minute=1000, burst_size=100),
}


class OpenRouterRateLimiter:
    """Rate limiter for OpenRouter API calls.

    Uses token bucket algorithm with configurable tiers.
    Thread-safe for use across multiple agent instances.
    """

    def __init__(self, tier: str = "standard"):
        """
        Initialize rate limiter with specified tier.

        Tier can be set via OPENROUTER_TIER environment variable.
        """
        tier_name = os.environ.get("OPENROUTER_TIER", tier).lower()
        self.tier = OPENROUTER_TIERS.get(tier_name, OPENROUTER_TIERS["standard"])

        self._tokens = float(self.tier.burst_size)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

        # Track rate limit headers from API
        self._api_limit: Optional[int] = None
        self._api_remaining: Optional[int] = None
        self._api_reset: Optional[float] = None

        logger.debug(f"OpenRouter rate limiter initialized: tier={self.tier.name}, rpm={self.tier.requests_per_minute}")

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed_minutes = (now - self._last_refill) / 60.0
        refill_amount = elapsed_minutes * self.tier.requests_per_minute
        self._tokens = min(self.tier.burst_size, self._tokens + refill_amount)
        self._last_refill = now

    async def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire permission to make an API request.

        Blocks until a token is available or timeout is reached.
        Returns True if acquired, False if timed out.
        """
        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill()

                # Check API-reported limits if available
                if self._api_remaining is not None and self._api_remaining <= 0:
                    wait_time = (self._api_reset or 60) - time.time()
                    if wait_time > 0 and wait_time < timeout:
                        logger.debug(f"OpenRouter API limit reached, waiting {wait_time:.1f}s")
                        await asyncio.sleep(min(wait_time, 1.0))
                        continue

                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

            # Wait and retry
            if time.monotonic() >= deadline:
                logger.warning("OpenRouter rate limit timeout")
                return False

            wait_time = 60.0 / self.tier.requests_per_minute  # Time for 1 token
            await asyncio.sleep(min(wait_time, 1.0))

    def update_from_headers(self, headers: dict) -> None:
        """Update rate limit state from API response headers.

        OpenRouter returns standard rate limit headers:
        - X-RateLimit-Limit: Total requests allowed
        - X-RateLimit-Remaining: Requests remaining
        - X-RateLimit-Reset: Unix timestamp when limit resets
        """
        with self._lock:
            if "X-RateLimit-Limit" in headers:
                try:
                    self._api_limit = int(headers["X-RateLimit-Limit"])
                except ValueError as e:
                    logger.warning(f"Failed to parse X-RateLimit-Limit header: {headers.get('X-RateLimit-Limit')!r} - {e}")

            if "X-RateLimit-Remaining" in headers:
                try:
                    self._api_remaining = int(headers["X-RateLimit-Remaining"])
                except ValueError as e:
                    logger.warning(f"Failed to parse X-RateLimit-Remaining header: {headers.get('X-RateLimit-Remaining')!r} - {e}")

            if "X-RateLimit-Reset" in headers:
                try:
                    self._api_reset = float(headers["X-RateLimit-Reset"])
                except ValueError as e:
                    logger.warning(f"Failed to parse X-RateLimit-Reset header: {headers.get('X-RateLimit-Reset')!r} - {e}")

    def release_on_error(self) -> None:
        """Release a token back on request error (optional, for retries)."""
        with self._lock:
            self._tokens = min(self.tier.burst_size, self._tokens + 1.0)

    @property
    def stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self._lock:
            return {
                "tier": self.tier.name,
                "rpm_limit": self.tier.requests_per_minute,
                "tokens_available": int(self._tokens),
                "burst_size": self.tier.burst_size,
                "api_limit": self._api_limit,
                "api_remaining": self._api_remaining,
            }


# Global rate limiter instance (shared across all OpenRouterAgent instances)
_openrouter_limiter: Optional[OpenRouterRateLimiter] = None
_openrouter_limiter_lock = threading.Lock()


def get_openrouter_limiter() -> OpenRouterRateLimiter:
    """Get or create the global OpenRouter rate limiter."""
    global _openrouter_limiter
    with _openrouter_limiter_lock:
        if _openrouter_limiter is None:
            _openrouter_limiter = OpenRouterRateLimiter()
        return _openrouter_limiter


def set_openrouter_tier(tier: str) -> None:
    """Set the OpenRouter rate limit tier.

    Valid tiers: free, basic, standard, premium, unlimited
    """
    global _openrouter_limiter
    with _openrouter_limiter_lock:
        _openrouter_limiter = OpenRouterRateLimiter(tier=tier)


__all__ = [
    "OpenRouterTier",
    "OPENROUTER_TIERS",
    "OpenRouterRateLimiter",
    "get_openrouter_limiter",
    "set_openrouter_tier",
]
