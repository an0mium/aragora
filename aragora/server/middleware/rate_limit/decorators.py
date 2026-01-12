"""
Rate limiting decorators.

Provides decorator functions for applying rate limits to handler methods.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

from .limiter import RateLimitResult
from .registry import get_rate_limiter
from .user_limiter import check_user_rate_limit

if TYPE_CHECKING:
    from aragora.server.handlers.base import HandlerResult

logger = logging.getLogger(__name__)


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


def _error_response(
    message: str, status: int, headers: Dict[str, str]
) -> "HandlerResult":
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
                logger.warning(
                    f"User rate limit exceeded for {result.key} on {action}"
                )
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

        return wrapper

    return decorator


__all__ = [
    "rate_limit_headers",
    "rate_limit",
    "user_rate_limit",
]
