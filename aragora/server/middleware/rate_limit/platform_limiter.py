"""
Platform-specific rate limiting for chat integrations.

Provides pre-configured rate limiters for each chat platform with
appropriate limits based on each platform's API constraints.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .bucket import TokenBucket

logger = logging.getLogger(__name__)

# Platform-specific rate limit configurations
# Based on each platform's documented API limits
PLATFORM_RATE_LIMITS: dict[str, dict[str, int]] = {
    # Slack: Web API has complex limits, conservative default
    # https://api.slack.com/docs/rate-limits
    "slack": {"rpm": 10, "burst": 5, "daily": 0},
    # Discord: 50 requests/second with bursting
    # https://discord.com/developers/docs/topics/rate-limits
    "discord": {"rpm": 30, "burst": 10, "daily": 0},
    # Microsoft Teams: ~60 requests/minute
    # https://docs.microsoft.com/en-us/graph/throttling
    "teams": {"rpm": 10, "burst": 5, "daily": 0},
    # Telegram: 30 messages/second to different users
    # https://core.telegram.org/bots/faq#my-bot-is-hitting-limits-how-do-i-avoid-this
    "telegram": {"rpm": 20, "burst": 5, "daily": 0},
    # WhatsApp Business API: varies by tier
    # https://developers.facebook.com/docs/whatsapp/api/rate-limits
    "whatsapp": {"rpm": 5, "burst": 2, "daily": 100},
    # Matrix: depends on homeserver, conservative default
    "matrix": {"rpm": 10, "burst": 5, "daily": 0},
    # Zoom: 10-20 requests/second
    # https://marketplace.zoom.us/docs/api-reference/rate-limits
    "zoom": {"rpm": 30, "burst": 10, "daily": 1000},
    # Email (SMTP): varies by provider, conservative
    "email": {"rpm": 10, "burst": 3, "daily": 500},
    # Google Chat: similar to other Google APIs
    "google_chat": {"rpm": 15, "burst": 5, "daily": 0},
}


@dataclass
class PlatformRateLimitResult:
    """Extended rate limit result with platform-specific info."""

    allowed: bool
    remaining: int
    limit: int
    reset_at: float
    retry_after: Optional[float] = None
    platform: str = ""
    daily_remaining: Optional[int] = None
    daily_reset_at: Optional[float] = None

    def to_headers(self) -> dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after is not None and self.retry_after > 0:
            headers["Retry-After"] = str(int(self.retry_after) + 1)
        if self.daily_remaining is not None:
            headers["X-RateLimit-Daily-Remaining"] = str(self.daily_remaining)
        return headers


@dataclass
class PlatformRateLimiter:
    """Rate limiter configured for a specific chat platform.

    Uses simple token bucket algorithm with per-key tracking.
    """

    platform: str
    requests_per_minute: int = 30
    burst_size: int = 10
    daily_limit: int = 0

    # Internal state
    _buckets: Dict[str, TokenBucket] = field(default_factory=dict, repr=False)
    _daily_counts: Dict[str, tuple[int, float]] = field(
        default_factory=dict, repr=False
    )  # key -> (count, reset_at)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def rpm(self) -> int:
        """Alias for requests_per_minute."""
        return self.requests_per_minute

    def check(self, key: str) -> PlatformRateLimitResult:
        """Check if request is allowed.

        Args:
            key: Rate limit key (channel ID, user ID, etc.)

        Returns:
            PlatformRateLimitResult with allowed status and metadata
        """
        now = time.time()

        with self._lock:
            # Get or create bucket for this key
            if key not in self._buckets:
                self._buckets[key] = TokenBucket(
                    rate_per_minute=self.requests_per_minute,
                    burst_size=self.requests_per_minute + self.burst_size,
                )

            bucket = self._buckets[key]

            # Check daily limit first if configured
            daily_remaining = None
            daily_reset_at = None

            if self.daily_limit > 0:
                daily_count, daily_reset = self._daily_counts.get(key, (0, 0))

                # Reset daily counter if new day
                if daily_reset == 0 or now >= daily_reset:
                    # Reset at midnight UTC
                    tomorrow = now + 86400  # Simple 24h window
                    self._daily_counts[key] = (0, tomorrow)
                    daily_count = 0
                    daily_reset = tomorrow

                daily_remaining = self.daily_limit - daily_count
                daily_reset_at = daily_reset

                if daily_count >= self.daily_limit:
                    return PlatformRateLimitResult(
                        allowed=False,
                        remaining=0,
                        limit=self.daily_limit,
                        reset_at=daily_reset,
                        retry_after=daily_reset - now,
                        platform=self.platform,
                        daily_remaining=0,
                        daily_reset_at=daily_reset,
                    )

            # Check per-minute limit
            if bucket.consume():
                # Success - increment daily counter if applicable
                if self.daily_limit > 0:
                    count, reset = self._daily_counts[key]
                    self._daily_counts[key] = (count + 1, reset)
                    daily_remaining = self.daily_limit - count - 1

                return PlatformRateLimitResult(
                    allowed=True,
                    remaining=int(bucket.tokens),
                    limit=self.requests_per_minute,
                    reset_at=now + 60,
                    platform=self.platform,
                    daily_remaining=daily_remaining,
                    daily_reset_at=daily_reset_at,
                )
            else:
                # Rate limited
                retry_after = bucket.get_retry_after()
                return PlatformRateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.requests_per_minute,
                    reset_at=now + retry_after,
                    retry_after=retry_after,
                    platform=self.platform,
                    daily_remaining=daily_remaining,
                    daily_reset_at=daily_reset_at,
                )

    def is_allowed(self, key: str) -> bool:
        """Simple check if request is allowed."""
        return self.check(key).allowed

    def reset(self, key: str) -> bool:
        """Reset rate limit for a key."""
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]
            if key in self._daily_counts:
                del self._daily_counts[key]
            return True

    def cleanup(self, max_age_seconds: float = 3600) -> int:
        """Remove stale buckets (not accessed in max_age_seconds)."""
        # TokenBucket doesn't track last access, so we just clear all
        # This is called periodically to prevent memory growth
        with self._lock:
            count = len(self._buckets)
            # Only clear if we have too many entries
            if count > 10000:
                self._buckets.clear()
                return count
            return 0


# Global platform limiter registry
_platform_limiters: dict[str, PlatformRateLimiter] = {}
_platform_limiters_lock = threading.Lock()


def get_platform_rate_limiter(
    platform: str,
    requests_per_minute: Optional[int] = None,
    burst_size: Optional[int] = None,
    daily_limit: Optional[int] = None,
) -> PlatformRateLimiter:
    """Get or create a platform-specific rate limiter.

    Args:
        platform: Platform name
        requests_per_minute: Optional override for RPM
        burst_size: Optional override for burst
        daily_limit: Optional override for daily limit

    Returns:
        PlatformRateLimiter instance
    """
    platform = platform.lower()

    # Get platform defaults
    defaults = PLATFORM_RATE_LIMITS.get(platform, {"rpm": 30, "burst": 10, "daily": 0})

    rpm = requests_per_minute or defaults["rpm"]
    burst = burst_size or defaults["burst"]
    daily = daily_limit if daily_limit is not None else defaults.get("daily", 0)

    # If custom config, create new limiter (don't cache)
    if requests_per_minute or burst_size or daily_limit:
        return PlatformRateLimiter(
            platform=platform,
            requests_per_minute=rpm,
            burst_size=burst,
            daily_limit=daily,
        )

    with _platform_limiters_lock:
        if platform not in _platform_limiters:
            _platform_limiters[platform] = PlatformRateLimiter(
                platform=platform,
                requests_per_minute=rpm,
                burst_size=burst,
                daily_limit=daily,
            )
        return _platform_limiters[platform]


def check_platform_rate_limit(platform: str, key: str) -> PlatformRateLimitResult:
    """Check rate limit for a platform+key combination.

    Convenience function that gets the platform limiter and checks the key.

    Args:
        platform: Platform name (slack, discord, etc.)
        key: Rate limit key (channel ID, user ID, etc.)

    Returns:
        PlatformRateLimitResult with allowed status
    """
    limiter = get_platform_rate_limiter(platform)
    return limiter.check(key)


def reset_platform_rate_limiters() -> int:
    """Reset all platform rate limiters (for testing).

    Returns:
        Number of limiters reset
    """
    with _platform_limiters_lock:
        count = len(_platform_limiters)
        _platform_limiters.clear()
        return count


__all__ = [
    "PLATFORM_RATE_LIMITS",
    "PlatformRateLimiter",
    "PlatformRateLimitResult",
    "get_platform_rate_limiter",
    "check_platform_rate_limit",
    "reset_platform_rate_limiters",
]
