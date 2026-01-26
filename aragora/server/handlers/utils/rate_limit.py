"""
Rate limiting utilities for HTTP handlers.

This module provides a simplified rate limiting interface for HTTP handlers.
For the full-featured rate limiting implementation with Redis support, see
aragora.server.middleware.rate_limit.

The simplified interface here is optimized for handler decorators with
a simple `is_allowed(key)` API.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import threading
import time
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

from aragora.server.middleware.rate_limit import (
    RateLimitResult,
    rate_limit_headers,
    user_rate_limit,
)
from aragora.server.middleware.rate_limit import (
    get_rate_limiter as get_middleware_limiter,
)

# Re-export the middleware rate_limit decorator for handlers that want
# the full-featured version with burst support and rate limit headers
from aragora.server.middleware.rate_limit import (
    rate_limit as middleware_rate_limit,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Global rate limit disable for testing/load tests
# When set, ALL rate limiters (both local and Redis-based) are bypassed
RATE_LIMITING_DISABLED = os.environ.get("ARAGORA_DISABLE_ALL_RATE_LIMITS", "").lower() in (
    "1",
    "true",
    "yes",
)

TRUSTED_PROXIES = frozenset(
    p.strip()
    for p in os.getenv("ARAGORA_TRUSTED_PROXIES", "127.0.0.1,::1,localhost").split(",")
    if p.strip()
)


def _normalize_ip(ip_value: str) -> str:
    """Normalize IP address string for consistent keying."""
    if not ip_value:
        return ""
    ip_value = str(ip_value).strip()
    try:
        return str(ipaddress.ip_address(ip_value))
    except ValueError:
        return ip_value


def get_client_ip(handler) -> str:
    """Extract client IP from request handler.

    Only trusts X-Forwarded-For when the direct IP is a trusted proxy.

    Args:
        handler: HTTP request handler with headers

    Returns:
        Client IP address string
    """
    if handler is None:
        return "unknown"

    remote_ip = ""
    client_address = getattr(handler, "client_address", None)
    if client_address and isinstance(client_address, tuple):
        remote_ip = str(client_address[0])

    remote_ip = _normalize_ip(remote_ip)

    # Check for proxy headers
    headers = getattr(handler, "headers", None)
    if headers and hasattr(headers, "get"):
        try:
            if remote_ip in TRUSTED_PROXIES:
                # X-Forwarded-For can contain multiple IPs: "client, proxy1, proxy2"
                forwarded = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for") or ""
                if forwarded and isinstance(forwarded, str):
                    # Take the first (original client) IP
                    candidate = forwarded.split(",")[0].strip()
                    if candidate:
                        return _normalize_ip(candidate)

                # Also check X-Real-IP (used by nginx)
                real_ip = headers.get("X-Real-IP") or headers.get("x-real-ip") or ""
                if real_ip and isinstance(real_ip, str):
                    return _normalize_ip(real_ip.strip())
        except (TypeError, AttributeError):
            # Handle mock objects or unusual header types
            pass

    if remote_ip:
        return remote_ip
    return "unknown"


class RateLimiter:
    """Thread-safe token bucket rate limiter with simple API.

    Provides a simplified `is_allowed(key)` interface for handler code.
    For the full-featured implementation with Redis, burst support, and
    rate limit headers, use aragora.server.middleware.rate_limit.

    Example:
        limiter = RateLimiter(requests_per_minute=60)
        if limiter.is_allowed("192.168.1.1"):
            # Process request
        else:
            # Return 429 Too Many Requests
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        cleanup_interval: int = 300,
    ):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute per key
            cleanup_interval: Seconds between cleanup of expired buckets
        """
        self.rpm = requests_per_minute
        self.cleanup_interval = cleanup_interval
        self._buckets: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def is_allowed(self, key: str) -> bool:
        """Check if a request is allowed for the given key.

        Args:
            key: Identifier for rate limiting (e.g., client IP)

        Returns:
            True if request is allowed, False if rate limited
        """
        # Bypass rate limiting if globally disabled (for load tests)
        if RATE_LIMITING_DISABLED:
            return True

        now = time.time()

        with self._lock:
            # Periodic cleanup of old buckets
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired(now)

            bucket = self._buckets[key]

            # Remove timestamps older than 60 seconds
            cutoff = now - 60
            bucket[:] = [t for t in bucket if t > cutoff]

            # Check if under limit
            if len(bucket) >= self.rpm:
                logger.debug(
                    "Rate limit exceeded for %s: %d/%d requests",
                    key,
                    len(bucket),
                    self.rpm,
                )
                return False

            # Record this request
            bucket.append(now)
            return True

    def _cleanup_expired(self, now: float) -> None:
        """Remove empty or fully-expired buckets."""
        cutoff = now - 60
        expired_keys = [
            key
            for key, timestamps in self._buckets.items()
            if not timestamps or all(t <= cutoff for t in timestamps)
        ]
        for key in expired_keys:
            del self._buckets[key]

        self._last_cleanup = now
        if expired_keys:
            logger.debug("Cleaned up %d expired rate limit buckets", len(expired_keys))

    def get_remaining(self, key: str) -> int:
        """Get remaining requests allowed for a key."""
        now = time.time()
        cutoff = now - 60

        with self._lock:
            bucket = self._buckets.get(key, [])
            current = sum(1 for t in bucket if t > cutoff)
            return max(0, self.rpm - current)

    def reset(self, key: str) -> None:
        """Reset rate limit for a specific key."""
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]

    def clear(self) -> None:
        """Clear all rate limit buckets (for testing)."""
        with self._lock:
            self._buckets.clear()


# Global rate limiters for different endpoint categories
_limiters: dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()


def _get_limiter(name: str, rpm: int) -> RateLimiter:
    """Get or create a named rate limiter."""
    with _limiters_lock:
        if name not in _limiters:
            _limiters[name] = RateLimiter(requests_per_minute=rpm)
        return _limiters[name]


def clear_all_limiters() -> int:
    """Clear all rate limiters (for testing).

    Returns:
        Number of limiters cleared.
    """
    with _limiters_lock:
        count = 0
        for limiter in _limiters.values():
            limiter.clear()
            count += 1
        return count


def rate_limit(
    rpm: int = 60,
    key_func: Optional[Callable[[Any], str]] = None,
    limiter_name: Optional[str] = None,
    *,
    requests_per_minute: Optional[int] = None,
) -> Callable[[F], F]:
    """Decorator to rate limit handler methods.

    Applies token bucket rate limiting based on client IP (or custom key).
    Returns 429 Too Many Requests when limit is exceeded.

    Args:
        rpm: Maximum requests per minute (default: 60)
        requests_per_minute: Alias for rpm (for consistency with middleware)
        key_func: Optional function to extract rate limit key from handler
        limiter_name: Optional name to share limiter across decorators

    Returns:
        Decorated function

    Example:
        class MyHandler(BaseHandler):
            @rate_limit(rpm=30)
            def _list_items(self, handler) -> HandlerResult:
                ...

            @rate_limit(requests_per_minute=10, limiter_name="auth")
            def _login(self, handler) -> HandlerResult:
                ...
    """
    import asyncio

    # Support both rpm and requests_per_minute for compatibility
    effective_rpm = requests_per_minute if requests_per_minute is not None else rpm

    def decorator(func: F) -> F:
        name = limiter_name or f"{func.__module__}.{func.__qualname__}"
        limiter = _get_limiter(name, effective_rpm)

        def _get_key_from_args(args, kwargs) -> str:
            """Extract rate limit key from function arguments.

            Supports both old pattern (handler object) and new pattern (headers dict).

            Security note: For the new pattern with headers dict, we do NOT trust
            X-Forwarded-For or X-Real-IP headers because we cannot verify if the
            request came from a trusted proxy (no access to client_address).
            Use the old pattern with a handler object for proper proxy validation.
            """
            if key_func:
                # Try old pattern first (handler object as first arg)
                if args and hasattr(args[0], "headers"):
                    return key_func(args[0])
                # For new pattern, key_func needs to handle kwargs
                return key_func(kwargs)

            # Old pattern: handler object with headers attribute and client_address
            # This path uses get_client_ip() which properly validates trusted proxies
            if args and hasattr(args[0], "headers"):
                return get_client_ip(args[0])

            # New pattern: headers passed as kwarg
            # SECURITY: We cannot safely use X-Forwarded-For here because we don't
            # have access to client_address to verify the request came from a trusted
            # proxy. Only use the validated_client_ip kwarg if explicitly provided.
            validated_ip = kwargs.get("validated_client_ip")
            if validated_ip and isinstance(validated_ip, str):
                return _normalize_ip(validated_ip)

            # Fallback: Use a hash of request characteristics for some differentiation
            # This prevents all requests without IPs from sharing the same quota
            headers = kwargs.get("headers") or {}
            if isinstance(headers, dict):
                # Use User-Agent + Accept-Language as a weak differentiator
                # This is NOT secure against spoofing but better than "unknown" for all
                ua = headers.get("User-Agent") or headers.get("user-agent") or ""
                lang = headers.get("Accept-Language") or headers.get("accept-language") or ""
                if ua or lang:
                    import hashlib

                    key_data = f"{ua}:{lang}".encode()
                    return f"anon:{hashlib.sha256(key_data).hexdigest()[:16]}"

            return "unknown"

        def _check_rate_limit(key: str):
            """Check rate limit and return error response if exceeded."""
            if not limiter.is_allowed(key):
                from aragora.server.handlers.base import error_response

                remaining = limiter.get_remaining(key)
                logger.warning(
                    "Rate limit exceeded for %s on %s (remaining: %d)",
                    key,
                    func.__qualname__,
                    remaining,
                )
                return error_response(
                    "Rate limit exceeded. Please try again later.",
                    status=429,
                )
            return None

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                key = _get_key_from_args(args, kwargs)
                error = _check_rate_limit(key)
                if error:
                    return error
                return await func(self, *args, **kwargs)

            return cast(F, async_wrapper)
        else:

            @wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                key = _get_key_from_args(args, kwargs)
                error = _check_rate_limit(key)
                if error:
                    return error
                return func(self, *args, **kwargs)

            return cast(F, sync_wrapper)

    return decorator


__all__ = [
    "RateLimiter",
    "rate_limit",
    "user_rate_limit",
    "get_client_ip",
    "_get_limiter",
    "_limiters",
    # Re-exports from middleware for convenience
    "middleware_rate_limit",
    "get_middleware_limiter",
    "RateLimitResult",
    "rate_limit_headers",
]
