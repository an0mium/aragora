"""
Rate Limiting Middleware.

Provides configurable rate limiting decorators and classes for API endpoints.
Supports per-IP, per-token, and per-endpoint rate limiting with automatic
cleanup of stale entries.

Usage:
    from aragora.server.middleware import rate_limit, RateLimiter

    # Use as decorator
    @rate_limit(requests_per_minute=30)
    def handle_request(self, handler):
        ...

    # Or use RateLimiter directly
    limiter = RateLimiter()
    if not limiter.allow(client_ip):
        return error_response("Rate limit exceeded", 429)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult

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
    automatic cleanup of stale entries via LRU eviction.
    """

    def __init__(
        self,
        default_limit: int = DEFAULT_RATE_LIMIT,
        ip_limit: int = IP_RATE_LIMIT,
        cleanup_interval: int = 300,  # 5 minutes
        max_entries: int = 10000,
    ):
        """
        Initialize rate limiter.

        Args:
            default_limit: Default requests per minute for unspecified endpoints.
            ip_limit: Requests per minute per IP address.
            cleanup_interval: Seconds between stats logging.
            max_entries: Maximum entries before LRU eviction.
        """
        self.default_limit = default_limit
        self.ip_limit = ip_limit
        self.cleanup_interval = cleanup_interval
        self.max_entries = max_entries

        # Buckets by key type (OrderedDict for LRU eviction)
        self._ip_buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._token_buckets: OrderedDict[str, TokenBucket] = OrderedDict()
        self._endpoint_buckets: Dict[
            str, OrderedDict[str, TokenBucket]
        ] = {}

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
            endpoint: API endpoint path (e.g., "/api/debates").
            requests_per_minute: Max requests per minute.
            burst_size: Max burst capacity (default: 2x rate).
            key_type: How to key the limit ("ip", "token", "endpoint", "combined").
        """
        self._endpoint_configs[endpoint] = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            key_type=key_type,
        )

    def get_config(self, endpoint: str) -> RateLimitConfig:
        """Get rate limit config for an endpoint."""
        if endpoint in self._endpoint_configs:
            return self._endpoint_configs[endpoint]

        # Check for prefix match (wildcard endpoints)
        for path, config in self._endpoint_configs.items():
            if path.endswith("*") and endpoint.startswith(path[:-1]):
                return config

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
            client_ip: Client IP address.
            endpoint: Optional endpoint for per-endpoint limits.
            token: Optional auth token for per-token limits.

        Returns:
            RateLimitResult with allowed status and metadata.
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
                self._ip_buckets.move_to_end(ip)
                return self._ip_buckets[ip]

            # Evict oldest entries if at capacity
            max_ip_buckets = self.max_entries // 3
            while len(self._ip_buckets) >= max_ip_buckets:
                self._ip_buckets.popitem(last=False)

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
                self._token_buckets.move_to_end(token)
                return self._token_buckets[token]

            max_token_buckets = self.max_entries // 3
            while len(self._token_buckets) >= max_token_buckets:
                self._token_buckets.popitem(last=False)

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
            if endpoint not in self._endpoint_buckets:
                self._endpoint_buckets[endpoint] = OrderedDict()

            buckets = self._endpoint_buckets[endpoint]
            if key in buckets:
                buckets.move_to_end(key)
                return buckets[key]

            max_endpoint_buckets = self.max_entries // 3
            total_endpoint_entries = sum(
                len(b) for b in self._endpoint_buckets.values()
            )
            while total_endpoint_entries >= max_endpoint_buckets and len(buckets) > 0:
                buckets.popitem(last=False)
                total_endpoint_entries -= 1

            buckets[key] = TokenBucket(
                config.requests_per_minute,
                config.burst_size,
            )
            return buckets[key]

    def _maybe_cleanup(self) -> None:
        """Periodic stats logging."""
        now = time.monotonic()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        with self._lock:
            self._last_cleanup = now
            total = (
                len(self._ip_buckets)
                + len(self._token_buckets)
                + sum(len(v) for v in self._endpoint_buckets.values())
            )

            if total > 0:
                logger.debug(
                    f"Rate limiter stats: {len(self._ip_buckets)} IP, "
                    f"{len(self._token_buckets)} token, "
                    f"{sum(len(v) for v in self._endpoint_buckets.values())} "
                    f"endpoint buckets"
                )

    def cleanup(self, max_age_seconds: int = 300) -> int:
        """
        Remove all stale entries older than max_age_seconds.

        This is more aggressive than _maybe_cleanup - it actually removes
        entries based on last activity time, not just LRU eviction.

        Args:
            max_age_seconds: Maximum age in seconds before entry is removed.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            now = time.monotonic()
            removed = 0

            # For simplicity with token buckets that use monotonic time,
            # we can check last_refill against now
            for bucket_dict in [self._ip_buckets, self._token_buckets]:
                stale_keys = [
                    key
                    for key, bucket in bucket_dict.items()
                    if now - bucket.last_refill > max_age_seconds
                ]
                for key in stale_keys:
                    del bucket_dict[key]
                    removed += 1

            # Clean endpoint buckets
            for endpoint, buckets in list(self._endpoint_buckets.items()):
                stale_keys = [
                    key
                    for key, bucket in buckets.items()
                    if now - bucket.last_refill > max_age_seconds
                ]
                for key in stale_keys:
                    del buckets[key]
                    removed += 1

                # Remove empty endpoint dicts
                if not buckets:
                    del self._endpoint_buckets[endpoint]

            if removed > 0:
                logger.debug(f"Rate limiter cleanup removed {removed} stale entries")

            return removed

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

    def get_client_key(self, handler: Any) -> str:
        """
        Extract client key from request handler.

        Uses X-Forwarded-For if behind proxy, otherwise client_address.
        Falls back to 'anonymous' if neither available.

        Args:
            handler: HTTP request handler.

        Returns:
            Client identifier string.
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


# Global rate limiter instances
_rate_limiters: Dict[str, RateLimiter] = {}
_default_limiter: Optional[RateLimiter] = None


def get_rate_limiter(
    name: str = "_default",
    requests_per_minute: int = DEFAULT_RATE_LIMIT,
    burst: int | None = None,
) -> RateLimiter:
    """
    Get or create a named rate limiter.

    Args:
        name: Unique name for this limiter (e.g., "debate_create").
        requests_per_minute: Max requests per minute.
        burst: Burst capacity (default: 2x rate).

    Returns:
        RateLimiter instance.
    """
    global _default_limiter

    if name == "_default":
        if _default_limiter is None:
            _default_limiter = RateLimiter()
            # Configure default endpoint limits
            _default_limiter.configure_endpoint("/api/debates", 30, key_type="ip")
            _default_limiter.configure_endpoint("/api/debates/*", 60, key_type="ip")
            _default_limiter.configure_endpoint(
                "/api/debates/*/fork", 5, key_type="ip"
            )
            _default_limiter.configure_endpoint("/api/agent/*", 120, key_type="ip")
            _default_limiter.configure_endpoint("/api/leaderboard*", 60, key_type="ip")
            _default_limiter.configure_endpoint("/api/pulse/*", 30, key_type="ip")
            _default_limiter.configure_endpoint(
                "/api/memory/continuum/cleanup", 2, key_type="ip"
            )
            _default_limiter.configure_endpoint("/api/memory/*", 60, key_type="ip")
        return _default_limiter

    if name not in _rate_limiters:
        burst_size = burst or int(requests_per_minute * BURST_MULTIPLIER)
        _rate_limiters[name] = RateLimiter(
            default_limit=requests_per_minute,
            ip_limit=requests_per_minute,
        )
    return _rate_limiters[name]


def cleanup_rate_limiters(max_age_seconds: int = 300) -> int:
    """
    Cleanup all rate limiters, removing stale entries.

    Args:
        max_age_seconds: Maximum age in seconds before entry is removed.

    Returns:
        Total number of entries removed across all limiters.
    """
    removed = 0

    if _default_limiter is not None:
        removed += _default_limiter.cleanup(max_age_seconds)

    for limiter in _rate_limiters.values():
        removed += limiter.cleanup(max_age_seconds)

    return removed


def reset_rate_limiters() -> None:
    """Reset all rate limiters. Primarily for testing."""
    global _default_limiter, _rate_limiters
    _default_limiter = None
    _rate_limiters.clear()


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


def _extract_handler(*args, **kwargs) -> Any:
    """Extract handler from function arguments."""
    handler = kwargs.get("handler")
    if handler is None:
        for arg in args:
            if hasattr(arg, "headers"):
                handler = arg
                break
    return handler


def _error_response(message: str, status: int, headers: Dict[str, str]) -> "HandlerResult":
    """Create an error response."""
    from aragora.server.handlers.base import error_response

    return error_response(message, status, headers=headers)


def rate_limit(
    requests_per_minute: int = 30,
    burst: int | None = None,
    limiter_name: Optional[str] = None,
    key_type: str = "ip",
):
    """
    Decorator for rate limiting endpoint handlers.

    Applies token bucket rate limiting per client. Returns 429 Too Many Requests
    when limit exceeded.

    Args:
        requests_per_minute: Maximum requests per minute per client.
        burst: Additional burst capacity (default: 2x rate).
        limiter_name: Optional name to share limiter across handlers.
        key_type: How to key the limit ("ip", "token", "endpoint", "combined").

    Usage:
        @rate_limit(requests_per_minute=30)
        def _create_debate(self, handler):
            ...

        @rate_limit(requests_per_minute=10, burst=2, limiter_name="expensive")
        def _run_deep_analysis(self, path, query_params, handler):
            ...
    """

    def decorator(func: Callable) -> Callable:
        name = limiter_name or func.__name__
        limiter = get_rate_limiter(name, requests_per_minute, burst)

        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = _extract_handler(*args, **kwargs)

            # Get client key and check rate limit
            client_key = limiter.get_client_key(handler)

            # Extract endpoint path if available
            endpoint = None
            if args and len(args) > 1 and isinstance(args[1], str):
                endpoint = args[1]  # path is usually second arg

            result = limiter.allow(client_key, endpoint=endpoint)

            if not result.allowed:
                logger.warning(
                    f"Rate limit exceeded for {client_key} on {func.__name__}"
                )
                return _error_response(
                    "Rate limit exceeded. Please try again later.",
                    429,
                    rate_limit_headers(result),
                )

            # Call handler and add rate limit headers to response
            response = func(*args, **kwargs)

            # Add headers to response if possible
            if hasattr(response, "headers") and isinstance(response.headers, dict):
                response.headers.update(
                    {k: v for k, v in rate_limit_headers(result).items()}
                )

            return response

        return wrapper

    return decorator
