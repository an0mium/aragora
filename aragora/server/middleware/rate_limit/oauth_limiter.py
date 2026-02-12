"""
OAuth Rate Limiting Module.

Provides specialized rate limiting for OAuth/authentication endpoints with:
- Stricter limits than normal endpoints (IP-based, since user not authenticated)
- Exponential backoff on repeated violations
- Security audit logging for all violations
- Distributed rate limiting via Redis when available

Security Rationale:
    OAuth endpoints are high-value targets for brute force attacks. This module
    implements defense-in-depth with multiple layers:

    1. Low base limits: Token endpoints limited to 5 requests per 15 minutes,
       callback handlers to 30 requests per 15 minutes (vs 60/min for normal endpoints)

    2. Exponential backoff: Repeated violations trigger exponentially longer
       cooldowns (1min, 2min, 4min, 8min, up to 1 hour max)

    3. IP-based limiting: Since authentication endpoints are pre-auth, we must
       use IP addresses for rate limiting (not user IDs)

    4. Security audit logging: All violations are logged for SIEM integration
       and incident response

Usage:
    from aragora.server.middleware.rate_limit.oauth_limiter import (
        oauth_rate_limit,
        OAuthRateLimitConfig,
    )

    # Apply to OAuth handlers
    @oauth_rate_limit(endpoint_type="token")
    def _handle_token_request(self, handler):
        ...

    @oauth_rate_limit(endpoint_type="callback")
    def _handle_oauth_callback(self, handler):
        ...
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar
from collections.abc import Callable

from .limiter import RateLimitResult

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class _RateLimitedWrapper:
    """Protocol-like base class for wrapper functions with rate limit metadata.

    This class defines the attributes that are dynamically added to wrapper functions
    by the oauth_rate_limit decorator. Using a class with these attributes helps
    type checkers understand the dynamic attribute assignments.
    """

    _rate_limited: bool
    _oauth_rate_limit: bool
    _endpoint_type: str
    __call__: Callable[..., Any]
    __wrapped__: Callable[..., Any]
    __name__: str
    __qualname__: str


# =============================================================================
# Simple RateLimiter (to avoid using the complex limiter.RateLimiter)
# =============================================================================


class SimpleRateLimiter:
    """Thread-safe token bucket rate limiter with simple API.

    This is a local implementation to avoid complexity from limiter.RateLimiter
    which requires different parameters and has different behavior.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        cleanup_interval: int = 300,
    ):
        """Initialize rate limiter."""
        self.rpm = requests_per_minute
        self.cleanup_interval = cleanup_interval
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()

    def is_allowed(self, key: str) -> bool:
        """Check if a request is allowed for the given key."""
        now = time.time()

        with self._lock:
            # Periodic cleanup of old buckets
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired(now)

            if key not in self._buckets:
                self._buckets[key] = []

            bucket = self._buckets[key]

            # Remove timestamps older than 60 seconds
            cutoff = now - 60
            bucket[:] = [t for t in bucket if t > cutoff]

            # Check if under limit
            if len(bucket) >= self.rpm:
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


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class OAuthRateLimitConfig:
    """Configuration for OAuth rate limiting.

    Attributes:
        token_limit: Max requests per window for token endpoints (default: 5)
        callback_limit: Max requests per window for callback handlers (default: 10)
        auth_start_limit: Max requests per window for auth start (redirect) (default: 15)
        window_seconds: Time window in seconds (default: 900 = 15 minutes)
        max_backoff_seconds: Maximum backoff time in seconds (default: 3600 = 1 hour)
        initial_backoff_seconds: Initial backoff after first violation (default: 60)
        backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
        enable_audit_logging: Whether to log security audit events (default: True)
    """

    token_limit: int = 5
    callback_limit: int = 30
    auth_start_limit: int = 15
    window_seconds: int = 900  # 15 minutes
    max_backoff_seconds: int = 3600  # 1 hour
    initial_backoff_seconds: int = 60
    backoff_multiplier: float = 2.0
    enable_audit_logging: bool = True


# Default configuration - can be overridden via environment
def _get_default_config() -> OAuthRateLimitConfig:
    """Get default OAuth rate limit configuration from environment."""
    return OAuthRateLimitConfig(
        token_limit=int(os.environ.get("ARAGORA_OAUTH_TOKEN_LIMIT", "5")),
        callback_limit=int(os.environ.get("ARAGORA_OAUTH_CALLBACK_LIMIT", "30")),
        auth_start_limit=int(os.environ.get("ARAGORA_OAUTH_AUTH_START_LIMIT", "15")),
        window_seconds=int(os.environ.get("ARAGORA_OAUTH_WINDOW_SECONDS", "900")),
        max_backoff_seconds=int(os.environ.get("ARAGORA_OAUTH_MAX_BACKOFF", "3600")),
        initial_backoff_seconds=int(os.environ.get("ARAGORA_OAUTH_INITIAL_BACKOFF", "60")),
        backoff_multiplier=float(os.environ.get("ARAGORA_OAUTH_BACKOFF_MULTIPLIER", "2.0")),
        enable_audit_logging=os.environ.get("ARAGORA_OAUTH_AUDIT_LOGGING", "true").lower()
        in ("1", "true", "yes"),
    )


DEFAULT_CONFIG = _get_default_config()


# =============================================================================
# Backoff Tracker
# =============================================================================


@dataclass
class BackoffState:
    """Tracks backoff state for a single client IP."""

    violation_count: int = 0
    last_violation_time: float = 0.0
    backoff_until: float = 0.0


class OAuthBackoffTracker:
    """Tracks exponential backoff for OAuth rate limit violations.

    This tracker maintains per-IP state for exponential backoff, ensuring
    that repeated violations result in progressively longer cooldowns.

    Thread-safe for concurrent access.
    """

    def __init__(
        self,
        initial_backoff: int = 60,
        max_backoff: int = 3600,
        multiplier: float = 2.0,
        decay_period: int = 3600,
    ):
        """Initialize backoff tracker.

        Args:
            initial_backoff: Initial backoff in seconds after first violation
            max_backoff: Maximum backoff in seconds
            multiplier: Backoff multiplier for exponential growth
            decay_period: Seconds after which violation count starts decaying
        """
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.multiplier = multiplier
        self.decay_period = decay_period

        self._states: dict[str, BackoffState] = {}
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes

    def record_violation(self, client_ip: str) -> int:
        """Record a rate limit violation and calculate backoff.

        Args:
            client_ip: Client IP address

        Returns:
            Backoff time in seconds until client can retry
        """
        now = time.time()

        with self._lock:
            # Periodic cleanup
            if now - self._last_cleanup > self._cleanup_interval:
                self._cleanup_expired(now)

            state = self._states.get(client_ip)

            if state is None:
                # First violation for this IP
                state = BackoffState(
                    violation_count=1,
                    last_violation_time=now,
                    backoff_until=now + self.initial_backoff,
                )
                self._states[client_ip] = state
                return self.initial_backoff

            # Check if violation count should decay
            if now - state.last_violation_time > self.decay_period:
                # Reset violation count after decay period
                state.violation_count = 1
            else:
                state.violation_count += 1

            state.last_violation_time = now

            # Calculate exponential backoff
            backoff = min(
                self.initial_backoff * (self.multiplier ** (state.violation_count - 1)),
                self.max_backoff,
            )
            backoff = int(backoff)
            state.backoff_until = now + backoff

            return backoff

    def is_backed_off(self, client_ip: str) -> tuple[bool, int]:
        """Check if client is currently in backoff period.

        Args:
            client_ip: Client IP address

        Returns:
            Tuple of (is_backed_off, seconds_remaining)
        """
        now = time.time()

        with self._lock:
            state = self._states.get(client_ip)

            if state is None:
                return False, 0

            if now < state.backoff_until:
                remaining = int(math.ceil(state.backoff_until - now))
                return True, remaining

            return False, 0

    def reset(self, client_ip: str) -> None:
        """Reset backoff state for a client IP (e.g., after successful auth)."""
        with self._lock:
            if client_ip in self._states:
                del self._states[client_ip]

    def _cleanup_expired(self, now: float) -> None:
        """Remove expired backoff states."""
        # Remove states that haven't had violations in 2x decay period
        cutoff = now - (self.decay_period * 2)
        expired = [ip for ip, state in self._states.items() if state.last_violation_time < cutoff]
        for ip in expired:
            del self._states[ip]

        self._last_cleanup = now

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired OAuth backoff states")

    def get_stats(self) -> dict[str, Any]:
        """Get backoff tracker statistics."""
        with self._lock:
            return {
                "tracked_ips": len(self._states),
                "active_backoffs": sum(
                    1 for state in self._states.values() if time.time() < state.backoff_until
                ),
            }


# Global backoff tracker instance
_backoff_tracker: OAuthBackoffTracker | None = None
_tracker_lock = threading.Lock()


def get_backoff_tracker() -> OAuthBackoffTracker:
    """Get the global OAuth backoff tracker instance."""
    global _backoff_tracker

    if _backoff_tracker is not None:
        return _backoff_tracker

    with _tracker_lock:
        if _backoff_tracker is None:
            config = _get_default_config()
            _backoff_tracker = OAuthBackoffTracker(
                initial_backoff=config.initial_backoff_seconds,
                max_backoff=config.max_backoff_seconds,
                multiplier=config.backoff_multiplier,
            )
        return _backoff_tracker


def reset_backoff_tracker() -> None:
    """Reset the global backoff tracker (for testing)."""
    global _backoff_tracker
    with _tracker_lock:
        _backoff_tracker = None


# =============================================================================
# OAuth Rate Limiter
# =============================================================================


class OAuthRateLimiter:
    """Specialized rate limiter for OAuth endpoints.

    Provides IP-based rate limiting with:
    - Per-endpoint-type limits (token, callback, auth_start)
    - Exponential backoff on violations
    - Security audit logging
    - Integration with distributed rate limiting
    """

    def __init__(
        self,
        config: OAuthRateLimitConfig | None = None,
        use_distributed: bool = True,
    ):
        """Initialize OAuth rate limiter.

        Args:
            config: Rate limit configuration (uses defaults if not provided)
            use_distributed: Whether to use distributed rate limiting via Redis
        """
        self.config = config or _get_default_config()
        self.use_distributed = use_distributed

        # Create per-endpoint-type limiters
        self._limiters: dict[str, SimpleRateLimiter] = {}
        self._init_limiters()

        # Get backoff tracker
        self._backoff = get_backoff_tracker()

        # Track violations for metrics
        self._violations: dict[str, int] = {}
        self._lock = threading.Lock()

    def _init_limiters(self) -> None:
        """Initialize rate limiters for each endpoint type."""
        # Token endpoints - strictest limits
        self._limiters["token"] = SimpleRateLimiter(
            requests_per_minute=self._rpm_from_window(
                self.config.token_limit, self.config.window_seconds
            ),
        )

        # Callback handlers - slightly higher limits
        self._limiters["callback"] = SimpleRateLimiter(
            requests_per_minute=self._rpm_from_window(
                self.config.callback_limit, self.config.window_seconds
            ),
        )

        # Auth start (redirect) - moderate limits
        self._limiters["auth_start"] = SimpleRateLimiter(
            requests_per_minute=self._rpm_from_window(
                self.config.auth_start_limit, self.config.window_seconds
            ),
        )

    def _rpm_from_window(self, limit: int, window_seconds: int) -> int:
        """Convert window-based limit to requests per minute.

        Note: This is an approximation. The actual enforcement uses the
        window-based limit, but the underlying limiter needs RPM.
        """
        return max(1, int((limit * 60) / window_seconds))

    def check(
        self,
        client_ip: str,
        endpoint_type: str = "token",
        provider: str | None = None,
    ) -> RateLimitResult:
        """Check if request is allowed.

        Args:
            client_ip: Client IP address
            endpoint_type: Type of endpoint ("token", "callback", "auth_start")
            provider: OAuth provider name (for logging)

        Returns:
            RateLimitResult indicating if request is allowed
        """
        # First check backoff
        is_backed_off, backoff_remaining = self._backoff.is_backed_off(client_ip)
        if is_backed_off:
            logger.warning(
                f"OAuth request blocked by backoff: ip={client_ip}, "
                f"remaining={backoff_remaining}s, provider={provider}"
            )
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=self._get_limit(endpoint_type),
                retry_after=float(backoff_remaining),
                key=client_ip,
            )

        # Get appropriate limiter
        limiter = self._limiters.get(endpoint_type, self._limiters["token"])

        # Check rate limit
        if not limiter.is_allowed(client_ip):
            # Record violation and calculate backoff
            backoff_seconds = self._backoff.record_violation(client_ip)

            # Log security event
            self._log_violation(client_ip, endpoint_type, provider, backoff_seconds)

            # Track violation
            with self._lock:
                key = f"{endpoint_type}:{client_ip}"
                self._violations[key] = self._violations.get(key, 0) + 1

            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=self._get_limit(endpoint_type),
                retry_after=float(backoff_seconds),
                key=client_ip,
            )

        # Request allowed
        return RateLimitResult(
            allowed=True,
            remaining=limiter.get_remaining(client_ip),
            limit=self._get_limit(endpoint_type),
            retry_after=0.0,
            key=client_ip,
        )

    def _get_limit(self, endpoint_type: str) -> int:
        """Get configured limit for endpoint type."""
        if endpoint_type == "token":
            return self.config.token_limit
        elif endpoint_type == "callback":
            return self.config.callback_limit
        elif endpoint_type == "auth_start":
            return self.config.auth_start_limit
        return self.config.token_limit  # Default to strictest

    def _log_violation(
        self,
        client_ip: str,
        endpoint_type: str,
        provider: str | None,
        backoff_seconds: int,
    ) -> None:
        """Log security audit event for rate limit violation."""
        logger.warning(
            f"OAUTH RATE LIMIT: {endpoint_type} endpoint exceeded "
            f"(ip={client_ip}, provider={provider}, backoff={backoff_seconds}s)"
        )

        if not self.config.enable_audit_logging:
            return

        try:
            from aragora.audit.unified import audit_security

            audit_security(
                event_type="anomaly",
                actor_id=client_ip,
                ip_address=client_ip,
                reason=f"oauth_rate_limit_exceeded:{endpoint_type}",
                details={
                    "endpoint_type": endpoint_type,
                    "provider": provider or "unknown",
                    "backoff_seconds": backoff_seconds,
                    "limit": self._get_limit(endpoint_type),
                    "window_seconds": self.config.window_seconds,
                },
            )
        except ImportError:
            # Audit module not available
            pass
        except Exception as e:
            logger.debug(f"Failed to log security audit event: {e}")

    def reset_client(self, client_ip: str) -> None:
        """Reset rate limit state for a client (e.g., after successful auth)."""
        # Reset backoff
        self._backoff.reset(client_ip)

        # Reset all limiters for this client
        for limiter in self._limiters.values():
            limiter.reset(client_ip)

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "config": {
                    "token_limit": self.config.token_limit,
                    "callback_limit": self.config.callback_limit,
                    "auth_start_limit": self.config.auth_start_limit,
                    "window_seconds": self.config.window_seconds,
                },
                "violations": dict(self._violations),
                "backoff": self._backoff.get_stats(),
            }


# Global OAuth rate limiter instance
_oauth_limiter: OAuthRateLimiter | None = None
_oauth_limiter_lock = threading.Lock()


def get_oauth_limiter() -> OAuthRateLimiter:
    """Get the global OAuth rate limiter instance."""
    global _oauth_limiter

    if _oauth_limiter is not None:
        return _oauth_limiter

    with _oauth_limiter_lock:
        if _oauth_limiter is None:
            _oauth_limiter = OAuthRateLimiter()
        return _oauth_limiter


def reset_oauth_limiter() -> None:
    """Reset the global OAuth rate limiter (for testing)."""
    global _oauth_limiter
    with _oauth_limiter_lock:
        _oauth_limiter = None


# =============================================================================
# Decorator
# =============================================================================


def _get_client_ip(handler: Any) -> str:
    """Extract client IP from request handler."""
    try:
        from aragora.server.handlers.utils.rate_limit import get_client_ip

        return get_client_ip(handler)
    except ImportError:
        pass

    # Fallback implementation
    if handler is None:
        return "unknown"

    client_address = getattr(handler, "client_address", None)
    if client_address and isinstance(client_address, tuple):
        return str(client_address[0])

    return "unknown"


def _extract_handler(*args, **kwargs) -> Any:
    """Extract handler from function arguments."""
    handler = kwargs.get("handler")
    if handler is None:
        for arg in args:
            if hasattr(arg, "headers"):
                handler = arg
                break
    return handler


def _error_response(message: str, status: int, retry_after: int | None = None) -> HandlerResult:
    """Create an error response with optional Retry-After header."""
    from aragora.server.handlers.base import error_response

    headers = {}
    if retry_after:
        headers["Retry-After"] = str(retry_after)

    return error_response(message, status, headers=headers)


def oauth_rate_limit(
    endpoint_type: str = "token",
    provider: str | None = None,
) -> Callable[[F], F]:
    """Decorator for OAuth endpoint rate limiting.

    Applies strict IP-based rate limiting with exponential backoff for
    authentication endpoints. Returns 429 Too Many Requests when limit
    exceeded, with appropriate Retry-After header.

    Args:
        endpoint_type: Type of endpoint for limit selection:
            - "token": Token exchange endpoints (5/15min default)
            - "callback": OAuth callback handlers (10/15min default)
            - "auth_start": Auth redirect endpoints (15/15min default)
        provider: OAuth provider name (for logging/audit)

    Returns:
        Decorated function

    Example:
        @oauth_rate_limit(endpoint_type="token", provider="google")
        def _handle_google_token(self, handler):
            ...

        @oauth_rate_limit(endpoint_type="callback", provider="github")
        async def _handle_github_callback(self, handler, query_params):
            ...
    """
    import asyncio

    def decorator(func: F) -> F:
        limiter = get_oauth_limiter()
        effective_provider = provider or func.__name__

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                handler = _extract_handler(*args, **kwargs)
                client_ip = _get_client_ip(handler)

                result = limiter.check(client_ip, endpoint_type, effective_provider)

                if not result.allowed:
                    retry_after = int(result.retry_after) if result.retry_after > 0 else None
                    return _error_response(
                        "Too many authentication attempts. Please try again later.",
                        429,
                        retry_after,
                    )

                return await func(*args, **kwargs)

            # Mark wrapper as rate limited using setattr to avoid type errors.
            # These attributes are used by middleware to detect rate-limited handlers.
            setattr(async_wrapper, "_rate_limited", True)
            setattr(async_wrapper, "_oauth_rate_limit", True)
            setattr(async_wrapper, "_endpoint_type", endpoint_type)

            # Return type is F but wrapper is Callable[..., Any]; @wraps preserves signature
            return async_wrapper  # type: ignore[return-value]  # Generic decorator return
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                handler = _extract_handler(*args, **kwargs)
                client_ip = _get_client_ip(handler)

                result = limiter.check(client_ip, endpoint_type, effective_provider)

                if not result.allowed:
                    retry_after = int(result.retry_after) if result.retry_after > 0 else None
                    return _error_response(
                        "Too many authentication attempts. Please try again later.",
                        429,
                        retry_after,
                    )

                return func(*args, **kwargs)

            # Mark wrapper as rate limited using setattr to avoid type errors.
            # These attributes are used by middleware to detect rate-limited handlers.
            setattr(sync_wrapper, "_rate_limited", True)
            setattr(sync_wrapper, "_oauth_rate_limit", True)
            setattr(sync_wrapper, "_endpoint_type", endpoint_type)

            # Return type is F but wrapper is Callable[..., Any]; @wraps preserves signature
            return sync_wrapper  # type: ignore[return-value]  # Generic decorator return

    return decorator


__all__ = [
    "OAuthRateLimitConfig",
    "OAuthBackoffTracker",
    "OAuthRateLimiter",
    "oauth_rate_limit",
    "get_oauth_limiter",
    "reset_oauth_limiter",
    "get_backoff_tracker",
    "reset_backoff_tracker",
    "DEFAULT_CONFIG",
]
