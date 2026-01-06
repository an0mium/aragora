"""
Rate limiting middleware for API endpoints.

Provides unified rate limiting across HTTP handlers with configurable
limits per endpoint, IP, or token.

Usage:
    from aragora.server.rate_limit import RateLimiter, rate_limit

    limiter = RateLimiter()

    # Use as decorator
    @rate_limit(requests_per_minute=30)
    def handle_request(handler):
        ...

    # Or check directly
    if not limiter.allow(client_ip, endpoint="/api/debates"):
        return error_response("Rate limit exceeded", 429)
"""

import logging
import os
import threading
import time
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Configuration from environment
DEFAULT_RATE_LIMIT = int(os.environ.get("ARAGORA_RATE_LIMIT", "60"))
IP_RATE_LIMIT = int(os.environ.get("ARAGORA_IP_RATE_LIMIT", "120"))
BURST_MULTIPLIER = float(os.environ.get("ARAGORA_BURST_MULTIPLIER", "2.0"))


class TokenBucket:
    """
    Thread-safe token bucket rate limiter.

    Allows burst traffic up to burst_size, then limits to rate_per_minute.
    """

    def __init__(self, rate_per_minute: float, burst_size: int | None = None):
        """
        Initialize token bucket.

        Args:
            rate_per_minute: Token refill rate (tokens per minute)
            burst_size: Maximum tokens (defaults to 2x rate)
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
        with self._lock:
            # Refill tokens based on elapsed time
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


@dataclass
class RateLimitConfig:
    """Configuration for a rate limit rule."""
    requests_per_minute: int = DEFAULT_RATE_LIMIT
    burst_size: int | None = None
    key_type: str = "ip"  # "ip", "token", "endpoint", "combined"
    enabled: bool = True


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int = 0
    limit: int = 0
    retry_after: float = 0
    key: str = ""


class RateLimiter:
    """
    Unified rate limiter for API endpoints.

    Supports per-IP, per-token, and per-endpoint rate limiting with
    automatic cleanup of stale entries.
    """

    def __init__(
        self,
        default_limit: int = DEFAULT_RATE_LIMIT,
        ip_limit: int = IP_RATE_LIMIT,
        cleanup_interval: int = 300,  # 5 minutes
        max_entries: int = 10000,
    ):
        self.default_limit = default_limit
        self.ip_limit = ip_limit
        self.cleanup_interval = cleanup_interval
        self.max_entries = max_entries

        # Buckets by key type (OrderedDict for LRU eviction)
        self._ip_buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._token_buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._endpoint_buckets: Dict[str, OrderedDict[str, TokenBucket]] = defaultdict(OrderedDict)

        # Per-endpoint configuration
        self._endpoint_configs: Dict[str, RateLimitConfig] = {}

        self._lock = threading.Lock()
        self._last_cleanup = time.monotonic()

    def configure_endpoint(
        self,
        endpoint: str,
        requests_per_minute: int,
        burst_size: int | None = None,
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

    def get_config(self, endpoint: str) -> RateLimitConfig:
        """Get rate limit config for an endpoint."""
        # Check for exact match
        if endpoint in self._endpoint_configs:
            return self._endpoint_configs[endpoint]

        # Check for prefix match
        for path, config in self._endpoint_configs.items():
            if path.endswith("*") and endpoint.startswith(path[:-1]):
                return config

        # Return default
        return RateLimitConfig(requests_per_minute=self.default_limit)

    def allow(
        self,
        client_ip: str,
        endpoint: str | None = None,
        token: str | None = None,
    ) -> RateLimitResult:
        """
        Check if a request should be allowed.

        Args:
            client_ip: Client IP address
            endpoint: Optional endpoint for per-endpoint limits
            token: Optional auth token for per-token limits

        Returns:
            RateLimitResult with allowed status and metadata
        """
        self._maybe_cleanup()

        config = self.get_config(endpoint) if endpoint else RateLimitConfig()
        if not config.enabled:
            return RateLimitResult(allowed=True, limit=0)

        # Determine the key based on config
        if config.key_type == "token" and token:
            key = f"token:{token}"
            bucket = self._get_or_create_token_bucket(token, config)
        elif config.key_type == "combined" and endpoint:
            key = f"ep:{endpoint}:ip:{client_ip}"
            bucket = self._get_or_create_endpoint_bucket(endpoint, client_ip, config)
        elif config.key_type == "endpoint" and endpoint:
            key = f"ep:{endpoint}"
            bucket = self._get_or_create_endpoint_bucket(endpoint, "_global", config)
        else:
            # Default to IP-based limiting
            key = f"ip:{client_ip}"
            bucket = self._get_or_create_ip_bucket(client_ip)

        allowed = bucket.consume(1)

        return RateLimitResult(
            allowed=allowed,
            remaining=bucket.remaining,
            limit=config.requests_per_minute,
            retry_after=bucket.get_retry_after() if not allowed else 0,
            key=key,
        )

    def _get_or_create_ip_bucket(self, ip: str) -> TokenBucket:
        """Get or create an IP-based bucket with LRU eviction."""
        with self._lock:
            if ip in self._ip_buckets:
                self._ip_buckets.move_to_end(ip)  # Mark as recently used
                return self._ip_buckets[ip]

            # Evict oldest entries if at capacity (1/3 of max for IP buckets)
            max_ip_buckets = self.max_entries // 3
            while len(self._ip_buckets) >= max_ip_buckets:
                self._ip_buckets.popitem(last=False)  # Remove oldest

            self._ip_buckets[ip] = TokenBucket(self.ip_limit)
            return self._ip_buckets[ip]

    def _get_or_create_token_bucket(
        self,
        token: str,
        config: RateLimitConfig,
    ) -> TokenBucket:
        """Get or create a token-based bucket with LRU eviction."""
        with self._lock:
            if token in self._token_buckets:
                self._token_buckets.move_to_end(token)  # Mark as recently used
                return self._token_buckets[token]

            # Evict oldest entries if at capacity (1/3 of max for token buckets)
            max_token_buckets = self.max_entries // 3
            while len(self._token_buckets) >= max_token_buckets:
                self._token_buckets.popitem(last=False)  # Remove oldest

            self._token_buckets[token] = TokenBucket(
                config.requests_per_minute,
                config.burst_size,
            )
            return self._token_buckets[token]

    def _get_or_create_endpoint_bucket(
        self,
        endpoint: str,
        key: str,
        config: RateLimitConfig,
    ) -> TokenBucket:
        """Get or create an endpoint-specific bucket with LRU eviction."""
        with self._lock:
            buckets = self._endpoint_buckets[endpoint]
            if key in buckets:
                buckets.move_to_end(key)  # Mark as recently used
                return buckets[key]

            # Evict oldest entries if at capacity (1/3 of max split across endpoints)
            max_endpoint_buckets = self.max_entries // 3
            total_endpoint_entries = sum(len(b) for b in self._endpoint_buckets.values())
            while total_endpoint_entries >= max_endpoint_buckets and len(buckets) > 0:
                buckets.popitem(last=False)  # Remove oldest from this endpoint
                total_endpoint_entries -= 1

            buckets[key] = TokenBucket(
                config.requests_per_minute,
                config.burst_size,
            )
            return buckets[key]

    def _maybe_cleanup(self) -> None:
        """Periodic stats logging (size is now enforced by LRU eviction)."""
        now = time.monotonic()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            self._last_cleanup = now

            # Log stats periodically for monitoring
            total = (
                len(self._ip_buckets) +
                len(self._token_buckets) +
                sum(len(v) for v in self._endpoint_buckets.values())
            )

            if total > 0:
                logger.debug(
                    f"Rate limiter stats: {len(self._ip_buckets)} IP, "
                    f"{len(self._token_buckets)} token, "
                    f"{sum(len(v) for v in self._endpoint_buckets.values())} endpoint buckets"
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "ip_buckets": len(self._ip_buckets),
                "token_buckets": len(self._token_buckets),
                "endpoint_buckets": {
                    ep: len(buckets)
                    for ep, buckets in self._endpoint_buckets.items()
                },
                "configured_endpoints": list(self._endpoint_configs.keys()),
                "default_limit": self.default_limit,
                "ip_limit": self.ip_limit,
            }


# Global rate limiter instance
_limiter: Optional[RateLimiter] = None


def get_limiter() -> RateLimiter:
    """Get or create the global rate limiter."""
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
        # Configure default endpoint limits
        _limiter.configure_endpoint("/api/debates", 30, key_type="ip")
        _limiter.configure_endpoint("/api/debates/*", 60, key_type="ip")
        _limiter.configure_endpoint("/api/debates/*/fork", 5, key_type="ip")  # Fork is expensive
        _limiter.configure_endpoint("/api/agent/*", 120, key_type="ip")
        _limiter.configure_endpoint("/api/leaderboard*", 60, key_type="ip")
        _limiter.configure_endpoint("/api/pulse/*", 30, key_type="ip")
        _limiter.configure_endpoint("/api/memory/continuum/cleanup", 2, key_type="ip")  # Cleanup is heavy
        _limiter.configure_endpoint("/api/memory/*", 60, key_type="ip")
    return _limiter


def set_limiter(limiter: RateLimiter) -> None:
    """Set the global rate limiter (for testing)."""
    global _limiter
    _limiter = limiter


def rate_limit_headers(result: RateLimitResult) -> Dict[str, str]:
    """Generate rate limit headers for HTTP response."""
    headers = {
        "X-RateLimit-Limit": str(result.limit),
        "X-RateLimit-Remaining": str(result.remaining),
    }
    if result.retry_after > 0:
        headers["Retry-After"] = str(int(result.retry_after) + 1)
        headers["X-RateLimit-Reset"] = str(int(time.time() + result.retry_after))
    return headers


def rate_limit(
    requests_per_minute: int | None = None,
    burst_size: int | None = None,
    key_type: str = "ip",
):
    """
    Decorator for rate limiting endpoint handlers.

    Usage:
        @rate_limit(requests_per_minute=30)
        def _handle_create_debate(self, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Extract client IP from handler context
            handler = kwargs.get('handler') or (args[2] if len(args) > 2 else None)
            client_ip = "unknown"
            if handler:
                client_ip = getattr(handler, 'client_address', ("unknown",))[0]

            # Extract path from args
            path = args[0] if args else kwargs.get('path', '')

            limiter = get_limiter()
            result = limiter.allow(client_ip, endpoint=path)

            if not result.allowed:
                from aragora.server.handlers.base import error_response
                response = error_response("Rate limit exceeded", 429)
                # Add rate limit headers
                response.headers.update(rate_limit_headers(result))
                return response

            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# Convenience functions for manual rate limiting


def check_rate_limit(
    client_ip: str,
    endpoint: str | None = None,
    token: str | None = None,
) -> RateLimitResult:
    """Check rate limit without consuming a token (read-only check)."""
    limiter = get_limiter()
    # Note: This still consumes a token. For read-only checks,
    # use the bucket's remaining property directly.
    return limiter.allow(client_ip, endpoint, token)


def is_rate_limited(client_ip: str, endpoint: str | None = None) -> bool:
    """Quick check if a client is currently rate limited."""
    result = check_rate_limit(client_ip, endpoint)
    return not result.allowed
