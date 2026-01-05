"""
Base handler utilities for modular endpoint handlers.

Provides common response formatting, error handling, and utilities
shared across all endpoint modules.
"""

import json
import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, Tuple
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)


# Simple TTL cache for expensive operations
_cache: dict[str, tuple[float, Any]] = {}


def ttl_cache(ttl_seconds: float = 60.0, key_prefix: str = ""):
    """
    Decorator for caching function results with TTL expiry.

    Args:
        ttl_seconds: How long to cache results (default 60s)
        key_prefix: Prefix for cache key to namespace different functions

    Usage:
        @ttl_cache(ttl_seconds=300, key_prefix="leaderboard")
        def get_expensive_data(limit: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key from function name, args and kwargs
            cache_key = f"{key_prefix}:{func.__name__}:{args}:{sorted(kwargs.items())}"

            now = time.time()
            if cache_key in _cache:
                cached_time, cached_value = _cache[cache_key]
                if now - cached_time < ttl_seconds:
                    logger.debug(f"Cache hit for {cache_key}")
                    return cached_value

            # Cache miss or expired
            result = func(*args, **kwargs)
            _cache[cache_key] = (now, result)
            logger.debug(f"Cache miss, stored {cache_key}")
            return result
        return wrapper
    return decorator


def clear_cache(key_prefix: str = None) -> int:
    """Clear cached entries, optionally filtered by prefix.

    Returns number of entries cleared.
    """
    global _cache
    if key_prefix is None:
        count = len(_cache)
        _cache = {}
        return count
    else:
        keys_to_remove = [k for k in _cache if k.startswith(key_prefix)]
        for k in keys_to_remove:
            del _cache[k]
        return len(keys_to_remove)


@dataclass
class HandlerResult:
    """Result of handling an HTTP request."""
    status_code: int
    content_type: str
    body: bytes
    headers: dict = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


def json_response(data: Any, status: int = 200) -> HandlerResult:
    """Create a JSON response."""
    body = json.dumps(data, default=str).encode('utf-8')
    return HandlerResult(
        status_code=status,
        content_type="application/json",
        body=body,
    )


def error_response(message: str, status: int = 400) -> HandlerResult:
    """Create an error response."""
    return json_response({"error": message}, status=status)


def parse_query_params(query_string: str) -> dict:
    """Parse query string into a dictionary."""
    if not query_string:
        return {}
    params = parse_qs(query_string)
    # Convert single-value lists to just values
    return {k: v[0] if len(v) == 1 else v for k, v in params.items()}


def get_int_param(params: dict, key: str, default: int = 0) -> int:
    """Safely get an integer parameter."""
    try:
        return int(params.get(key, default))
    except (ValueError, TypeError):
        return default


def get_float_param(params: dict, key: str, default: float = 0.0) -> float:
    """Safely get a float parameter."""
    try:
        return float(params.get(key, default))
    except (ValueError, TypeError):
        return default


def get_bool_param(params: dict, key: str, default: bool = False) -> bool:
    """Safely get a boolean parameter."""
    value = params.get(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


class BaseHandler:
    """
    Base class for endpoint handlers.

    Subclasses implement specific endpoint groups and register
    their routes via the `routes` class attribute.
    """

    def __init__(self, server_context: dict):
        """
        Initialize with server context.

        Args:
            server_context: Dict containing shared server resources like
                           storage, elo_system, debate_embeddings, etc.
        """
        self.ctx = server_context

    def get_storage(self):
        """Get debate storage instance."""
        return self.ctx.get("storage")

    def get_elo_system(self):
        """Get ELO system instance."""
        return self.ctx.get("elo_system")

    def get_debate_embeddings(self):
        """Get debate embeddings database."""
        return self.ctx.get("debate_embeddings")

    def get_critique_store(self):
        """Get critique store instance."""
        return self.ctx.get("critique_store")

    def get_nomic_dir(self):
        """Get nomic directory path."""
        return self.ctx.get("nomic_dir")

    def handle(self, path: str, query_params: dict) -> Optional[HandlerResult]:
        """
        Handle a request. Override in subclasses.

        Args:
            path: The request path
            query_params: Parsed query parameters

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None
