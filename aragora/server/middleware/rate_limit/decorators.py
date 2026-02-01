"""
Rate limiting decorators.

Provides decorator functions for applying rate limits to handler methods.

By default, decorators use the distributed rate limiter which automatically
coordinates across server instances via Redis when available.
"""

from __future__ import annotations

import logging
import os
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional

from .limiter import RateLimiter, RateLimitResult
from .registry import get_rate_limiter, RedisRateLimiter
from .user_limiter import check_user_rate_limit

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)

# Use distributed limiter by default (can be disabled for testing)
USE_DISTRIBUTED_LIMITER = os.environ.get("ARAGORA_USE_DISTRIBUTED_RATE_LIMIT", "true").lower() in (
    "1",
    "true",
    "yes",
)


def rate_limit_headers(result: RateLimitResult) -> dict[str, str]:
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


def _error_response(message: str, status: int, headers: dict[str, str]) -> "HandlerResult":
    """Create an error response."""
    from aragora.server.handlers.base import error_response

    return error_response(message, status, headers=headers)


def rate_limit(
    requests_per_minute: int | None = None,
    burst: int | None = None,
    limiter_name: str | None = None,
    key_type: str = "ip",
    *,
    rpm: int | None = None,
    use_distributed: bool | None = None,
    tenant_aware: bool = False,
):
    """
    Decorator for rate limiting endpoint handlers.

    Applies token bucket rate limiting per client. Returns 429 Too Many Requests
    when limit exceeded.

    By default, uses the distributed rate limiter which coordinates across
    server instances via Redis when available.

    Args:
        requests_per_minute: Maximum requests per minute per client.
        rpm: Alias for requests_per_minute (for compatibility with handlers utils).
        burst: Additional burst capacity (default: 2x rate).
        limiter_name: Optional name to share limiter across handlers.
        key_type: How to key the limit ("ip", "token", "endpoint", "combined", "tenant").
        use_distributed: Whether to use distributed limiter (default: True).
        tenant_aware: If True, extracts tenant_id from request for per-tenant limits.

    Usage:
        @rate_limit(requests_per_minute=30)
        def _create_debate(self, handler):
            ...

        @rate_limit(requests_per_minute=10, burst=2, limiter_name="expensive")
        def _run_deep_analysis(self, path, query_params, handler):
            ...

        @rate_limit(requests_per_minute=100, tenant_aware=True)
        def _tenant_operation(self, path, query_params, handler):
            ...
    """

    # Support both rpm and requests_per_minute for compatibility
    effective_rpm = (
        requests_per_minute if requests_per_minute is not None else (rpm if rpm is not None else 30)
    )

    # Determine whether to use distributed limiter
    should_use_distributed = (
        use_distributed if use_distributed is not None else USE_DISTRIBUTED_LIMITER
    )

    def decorator(func: Callable) -> Callable:
        name = limiter_name or func.__name__

        # Get appropriate limiter based on configuration
        limiter: RateLimiter | RedisRateLimiter | Any
        if should_use_distributed:
            from .distributed import get_distributed_limiter

            limiter = get_distributed_limiter()
            # Configure endpoint on distributed limiter
            endpoint_key = f"/{name}"
            limiter.configure_endpoint(endpoint_key, effective_rpm, burst, key_type)
        else:
            limiter = get_rate_limiter(name, effective_rpm, burst)

        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = _extract_handler(*args, **kwargs)

            # Get client key
            client_key = limiter.get_client_key(handler)

            # Extract endpoint path if available
            endpoint = None
            if args and len(args) > 1 and isinstance(args[1], str):
                endpoint = args[1]  # path is usually second arg

            # Extract tenant_id if tenant_aware
            tenant_id = None
            if tenant_aware:
                tenant_id = _extract_tenant_id(handler, kwargs)

            # Check rate limit
            if should_use_distributed:
                result = limiter.allow(
                    client_ip=client_key,
                    endpoint=endpoint,
                    tenant_id=tenant_id,
                )
            else:
                result = limiter.allow(client_key, endpoint=endpoint)

            if not result.allowed:
                logger.warning(f"Rate limit exceeded for {client_key} on {func.__name__}")
                return _error_response(
                    "Rate limit exceeded. Please try again later.",
                    429,
                    rate_limit_headers(result),
                )

            # Call handler and add rate limit headers to response
            response = func(*args, **kwargs)

            # Add headers to response if possible
            if hasattr(response, "headers") and isinstance(response.headers, dict):
                response.headers.update({k: v for k, v in rate_limit_headers(result).items()})

            return response

        # Mark wrapper as rate limited for detection by default_limiter
        setattr(wrapper, "_rate_limited", True)
        setattr(wrapper, "_rate_limiter", limiter)
        setattr(wrapper, "_rate_limit_distributed", should_use_distributed)

        return wrapper

    return decorator


def _extract_tenant_id(handler: Any, kwargs: dict) -> str | None:
    """Extract tenant ID from request handler or kwargs.

    Looks for tenant_id in:
    1. kwargs['tenant_id']
    2. handler.tenant_id
    3. X-Tenant-ID header
    4. query params

    Returns:
        Tenant ID string or None
    """
    # Check kwargs first
    if "tenant_id" in kwargs:
        return str(kwargs["tenant_id"])

    # Check handler attribute
    if hasattr(handler, "tenant_id") and handler.tenant_id:
        return str(handler.tenant_id)

    # Check headers
    if hasattr(handler, "headers"):
        tenant_header = handler.headers.get("X-Tenant-ID") or handler.headers.get("x-tenant-id")
        if tenant_header:
            return str(tenant_header)

    return None


def user_rate_limit(
    action: str = "default",
    user_store_factory: Optional[Callable[[], Any]] = None,
):
    """
    Decorator for per-user rate limiting.

    Args:
        action: Action name for rate limit lookup.
        user_store_factory: Optional callable to get UserStore instance.

    Usage:
        @user_rate_limit(action="debate_create")
        def _create_debate(self, handler):
            ...

        @user_rate_limit(action="vote", user_store_factory=get_user_store)
        def _submit_vote(self, path, query_params, handler):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = _extract_handler(*args, **kwargs)
            user_store = user_store_factory() if user_store_factory else None

            result = check_user_rate_limit(handler, user_store, action)

            if not result.allowed:
                logger.warning(f"User rate limit exceeded for {result.key} on {action}")
                return _error_response(
                    f"Rate limit exceeded for {action}. Please try again later.",
                    429,
                    rate_limit_headers(result),
                )

            response = func(*args, **kwargs)

            # Add headers to response if possible
            if hasattr(response, "headers") and isinstance(response.headers, dict):
                response.headers.update(rate_limit_headers(result))

            return response

        # Mark wrapper as rate limited for detection by default_limiter
        setattr(wrapper, "_rate_limited", True)

        return wrapper

    return decorator


__all__ = [
    "rate_limit_headers",
    "rate_limit",
    "user_rate_limit",
]
