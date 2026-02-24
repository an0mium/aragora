"""
Per-endpoint rate limiting with configurable tiers.

Uses an in-memory token bucket algorithm (no Redis dependency) to enforce
rate limits based on the request path and client identifier.

Default tiers::

    auth   ->  5 requests / minute   (login, register, password reset)
    write  -> 30 requests / minute   (POST, PUT, PATCH, DELETE)
    read   -> 120 requests / minute  (GET, HEAD, OPTIONS)

Usage::

    from aragora.server.security.endpoint_rate_limiter import EndpointRateLimiter

    limiter = EndpointRateLimiter()
    result = limiter.check("192.168.1.1", "/api/v1/auth/login", "POST")
    if not result.allowed:
        # Return 429 with Retry-After header
        ...
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tier definitions
# ---------------------------------------------------------------------------


class RateTier(Enum):
    """Rate-limit tiers, from most restrictive to least."""

    AUTH = "auth"
    WRITE = "write"
    READ = "read"


# Default rates per minute for each tier.
DEFAULT_TIER_RATES: dict[RateTier, int] = {
    RateTier.AUTH: 5,
    RateTier.WRITE: 30,
    RateTier.READ: 120,
}

# Default burst multiplier (burst_size = rate * multiplier).
DEFAULT_BURST_MULTIPLIER: float = 2.0

# Path prefixes that map to the AUTH tier.
AUTH_PATH_PREFIXES: tuple[str, ...] = (
    "/api/auth/",
    "/api/v1/auth/",
    "/api/login",
    "/api/v1/login",
    "/api/register",
    "/api/v1/register",
    "/api/password",
    "/api/v1/password",
    "/api/oauth/",
    "/api/v1/oauth/",
    "/api/mfa/",
    "/api/v1/mfa/",
)

# HTTP methods considered writes.
WRITE_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Stale-bucket cleanup interval (seconds).
CLEANUP_INTERVAL: float = 300.0  # 5 min

# Stale threshold: remove buckets unused for this many seconds.
STALE_THRESHOLD: float = 600.0  # 10 min

# Maximum number of tracked buckets before forced cleanup.
MAX_BUCKETS: int = 100_000


# ---------------------------------------------------------------------------
# Token bucket (minimal, self-contained)
# ---------------------------------------------------------------------------


class _TokenBucket:
    """Lightweight in-memory token bucket."""

    __slots__ = ("rate_per_second", "burst_size", "tokens", "last_refill")

    def __init__(self, rate_per_minute: float, burst_size: int) -> None:
        self.rate_per_second = rate_per_minute / 60.0
        self.burst_size = burst_size
        self.tokens: float = float(burst_size)
        self.last_refill: float = time.monotonic()

    def consume(self, now: float | None = None) -> bool:
        """Try to consume one token. Returns ``True`` if allowed."""
        now = now if now is not None else time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.burst_size, self.tokens + elapsed * self.rate_per_second)
        self.last_refill = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def retry_after(self) -> float:
        """Seconds until the next token becomes available."""
        if self.tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self.tokens
        if self.rate_per_second <= 0:
            return 0.0
        return deficit / self.rate_per_second


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EndpointRateLimitConfig:
    """Configuration for the per-endpoint rate limiter."""

    tier_rates: dict[RateTier, int] = field(default_factory=lambda: dict(DEFAULT_TIER_RATES))
    burst_multiplier: float = DEFAULT_BURST_MULTIPLIER

    # Per-path overrides: path-prefix -> requests/min.
    path_overrides: dict[str, int] = field(default_factory=dict)

    # Custom path-prefix -> tier mapping additions.
    auth_path_prefixes: tuple[str, ...] = AUTH_PATH_PREFIXES

    # Cleanup settings
    cleanup_interval: float = CLEANUP_INTERVAL
    stale_threshold: float = STALE_THRESHOLD
    max_buckets: int = MAX_BUCKETS


# ---------------------------------------------------------------------------
# Check result
# ---------------------------------------------------------------------------


@dataclass
class RateLimitCheckResult:
    """Result of a rate limit check."""

    allowed: bool
    tier: RateTier
    retry_after: float = 0.0
    limit: int = 0
    remaining: int = 0

    def headers(self) -> dict[str, str]:
        """Return rate-limit response headers (RFC draft style)."""
        h: dict[str, str] = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
        }
        if not self.allowed:
            h["Retry-After"] = str(int(self.retry_after) + 1)
        return h


# ---------------------------------------------------------------------------
# Endpoint rate limiter
# ---------------------------------------------------------------------------


class EndpointRateLimiter:
    """Per-endpoint, per-client rate limiter using in-memory token buckets.

    Thread-safe. Buckets are keyed on ``(client_id, tier)`` and lazily
    created on first request.
    """

    def __init__(self, config: EndpointRateLimitConfig | None = None) -> None:
        self.config = config or EndpointRateLimitConfig()
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()
        self._last_cleanup: float = time.monotonic()

    # -- public API --------------------------------------------------------

    def classify(self, path: str, method: str = "GET") -> RateTier:
        """Determine the rate tier for a given path and method."""
        path_lower = path.lower()
        for prefix in self.config.auth_path_prefixes:
            if path_lower.startswith(prefix.lower()):
                return RateTier.AUTH
        if method.upper() in WRITE_METHODS:
            return RateTier.WRITE
        return RateTier.READ

    def check(
        self,
        client_id: str,
        path: str,
        method: str = "GET",
    ) -> RateLimitCheckResult:
        """Check whether the request should be allowed.

        Args:
            client_id: Client identifier (IP, API key, etc.).
            path: Request path.
            method: HTTP method.

        Returns:
            RateLimitCheckResult with ``allowed``, headers, and retry info.
        """
        tier = self.classify(path, method)
        rate = self._rate_for(path, tier)
        burst = int(rate * self.config.burst_multiplier)
        key = f"{client_id}:{tier.value}"

        with self._lock:
            self._maybe_cleanup()
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _TokenBucket(rate_per_minute=rate, burst_size=burst)
                self._buckets[key] = bucket

            allowed = bucket.consume()
            remaining = max(0, int(bucket.tokens))
            retry_after = 0.0 if allowed else bucket.retry_after()

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "client_id": client_id,
                    "path": path,
                    "method": method,
                    "tier": tier.value,
                    "retry_after": retry_after,
                },
            )

        return RateLimitCheckResult(
            allowed=allowed,
            tier=tier,
            retry_after=retry_after,
            limit=rate,
            remaining=remaining,
        )

    def reset(self, client_id: str | None = None) -> None:
        """Reset buckets. If *client_id* given, only that client's buckets."""
        with self._lock:
            if client_id is None:
                self._buckets.clear()
            else:
                prefix = f"{client_id}:"
                keys = [k for k in self._buckets if k.startswith(prefix)]
                for k in keys:
                    del self._buckets[k]

    @property
    def bucket_count(self) -> int:
        """Number of active buckets (for monitoring)."""
        with self._lock:
            return len(self._buckets)

    # -- internals ---------------------------------------------------------

    def _rate_for(self, path: str, tier: RateTier) -> int:
        """Get the effective rate (requests/min) for a path and tier."""
        # Check explicit path overrides first.
        for prefix, rate in self.config.path_overrides.items():
            if path.startswith(prefix):
                return rate
        return self.config.tier_rates.get(tier, DEFAULT_TIER_RATES[RateTier.READ])

    def _maybe_cleanup(self) -> None:
        """Evict stale buckets if enough time has passed.

        Must be called while holding ``self._lock``.
        """
        now = time.monotonic()
        if (
            now - self._last_cleanup < self.config.cleanup_interval
            and len(self._buckets) < self.config.max_buckets
        ):
            return
        threshold = now - self.config.stale_threshold
        stale = [k for k, b in self._buckets.items() if b.last_refill < threshold]
        for k in stale:
            del self._buckets[k]
        if stale:
            logger.debug(
                "Cleaned up stale rate-limit buckets",
                extra={"removed": len(stale), "remaining": len(self._buckets)},
            )
        self._last_cleanup = now


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "RateTier",
    "DEFAULT_TIER_RATES",
    "AUTH_PATH_PREFIXES",
    "WRITE_METHODS",
    "EndpointRateLimitConfig",
    "RateLimitCheckResult",
    "EndpointRateLimiter",
]
