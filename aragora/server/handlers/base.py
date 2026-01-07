"""
Base handler utilities for modular endpoint handlers.

Provides common response formatting, error handling, and utilities
shared across all endpoint modules.
"""

import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, Tuple
from urllib.parse import parse_qs

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.server.error_utils import safe_error_message
from aragora.server.prometheus import record_cache_hit, record_cache_miss
from aragora.server.validation import (
    SAFE_ID_PATTERN,
    SAFE_AGENT_PATTERN,
    SAFE_SLUG_PATTERN,
    # Re-export validation functions for backwards compatibility
    validate_path_segment,
    validate_agent_name,
    validate_debate_id,
)

# Re-export DB_TIMEOUT_SECONDS for backwards compatibility
__all__ = [
    "DB_TIMEOUT_SECONDS", "require_auth", "require_storage", "error_response",
    "json_response", "handle_errors", "log_request", "ttl_cache", "clear_cache",
    "get_cache_stats", "invalidate_cache", "CACHE_INVALIDATION_MAP", "rate_limit",
    "RateLimiter",
]

logger = logging.getLogger(__name__)


# Cache configuration from environment
CACHE_MAX_ENTRIES = int(os.environ.get("ARAGORA_CACHE_MAX_ENTRIES", "1000"))
CACHE_EVICT_PERCENT = float(os.environ.get("ARAGORA_CACHE_EVICT_PERCENT", "0.1"))


class BoundedTTLCache:
    """
    Thread-safe TTL cache with bounded size and LRU eviction.

    Prevents memory leaks by limiting the number of entries and
    evicting oldest entries when the limit is reached.
    """

    def __init__(self, max_entries: int = CACHE_MAX_ENTRIES, evict_percent: float = CACHE_EVICT_PERCENT):
        self._cache: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._max_entries = max_entries
        self._evict_count = max(1, int(max_entries * evict_percent))
        self._hits = 0
        self._misses = 0

    def get(self, key: str, ttl_seconds: float) -> tuple[bool, Any]:
        """
        Get a value from cache if not expired.

        Returns:
            Tuple of (hit, value). If hit is False, value is None.
        """
        now = time.time()

        if key in self._cache:
            cached_time, cached_value = self._cache[key]
            if now - cached_time < ttl_seconds:
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                self._hits += 1
                return True, cached_value
            else:
                # Expired - remove it
                del self._cache[key]

        self._misses += 1
        return False, None

    def set(self, key: str, value: Any) -> None:
        """Store a value in cache, evicting old entries if necessary."""
        now = time.time()

        # If key exists, update and move to end
        if key in self._cache:
            self._cache[key] = (now, value)
            self._cache.move_to_end(key)
            return

        # Check if we need to evict
        if len(self._cache) >= self._max_entries:
            self._evict_oldest()

        # Add new entry
        self._cache[key] = (now, value)

    def _evict_oldest(self) -> int:
        """Evict oldest entries to make room. Returns count evicted."""
        evicted = 0
        for _ in range(self._evict_count):
            if self._cache:
                self._cache.popitem(last=False)
                evicted += 1
        if evicted > 0:
            logger.debug(f"Cache evicted {evicted} entries (size: {len(self._cache)})")
        return evicted

    def clear(self, key_prefix: str | None = None) -> int:
        """Clear entries, optionally filtered by prefix."""
        if key_prefix is None:
            count = len(self._cache)
            self._cache.clear()
            return count
        else:
            keys_to_remove = [k for k in self._cache if k.startswith(key_prefix)]
            for k in keys_to_remove:
                del self._cache[k]
            return len(keys_to_remove)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def items(self):
        """Iterate over cache items."""
        return self._cache.items()

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "max_entries": self._max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


# Global bounded cache instance
_cache = BoundedTTLCache()


def ttl_cache(ttl_seconds: float = 60.0, key_prefix: str = "", skip_first: bool = True):
    """
    Decorator for caching function results with TTL expiry.

    Args:
        ttl_seconds: How long to cache results (default 60s)
        key_prefix: Prefix for cache key to namespace different functions
        skip_first: If True, skip first arg (self) when building cache key for methods.
                   Default is True since most usage is on class methods.
                   Set to False when decorating standalone functions.

    Usage:
        @ttl_cache(ttl_seconds=300, key_prefix="leaderboard")
        def _get_leaderboard(self, limit: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Skip 'self' when building cache key for methods
            cache_args = args[1:] if skip_first and args else args
            # Build cache key from function name, args and kwargs
            cache_key = f"{key_prefix}:{func.__name__}:{cache_args}:{sorted(kwargs.items())}"

            hit, cached_value = _cache.get(cache_key, ttl_seconds)
            if hit:
                record_cache_hit(key_prefix or func.__name__)
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value

            # Cache miss or expired
            record_cache_miss(key_prefix or func.__name__)
            result = func(*args, **kwargs)
            _cache.set(cache_key, result)
            logger.debug(f"Cache miss, stored {cache_key}")
            return result
        return wrapper
    return decorator


def clear_cache(key_prefix: str | None = None) -> int:
    """Clear cached entries, optionally filtered by prefix.

    Returns number of entries cleared.
    """
    return _cache.clear(key_prefix)


def get_cache_stats() -> dict:
    """Get cache statistics for monitoring."""
    return _cache.stats


# Cache invalidation registry - maps data sources to cache key prefixes
CACHE_INVALIDATION_MAP: dict[str, list[str]] = {
    "elo": [
        "lb_rankings", "lb_matches", "lb_reputation", "lb_stats",
        "leaderboard", "agent_profile", "agent_h2h", "lb_teams",
    ],
    "calibration": [
        "calibration_lb", "lb_introspection", "agent_flips",
        "flips_recent", "flips_summary",
    ],
    "memory": [
        "replays_list", "learning_evolution", "meta_learning_stats",
    ],
    "consensus": [
        "consensus_similar", "consensus_settled", "consensus_stats",
        "recent_dissents", "contrarian_views", "risk_warnings",
    ],
    "debates": [
        "dashboard_debates", "analytics_disagreement", "analytics_roles",
        "analytics_early_stop", "analytics_ranking", "analytics_debates",
        "analytics_memory",
    ],
}


def invalidate_cache(data_source: str) -> int:
    """Invalidate all caches related to a data source.

    Args:
        data_source: One of 'elo', 'calibration', 'memory', 'consensus', 'debates'

    Returns:
        Total number of cache entries cleared.
    """
    prefixes = CACHE_INVALIDATION_MAP.get(data_source, [])
    total = 0
    for prefix in prefixes:
        cleared = clear_cache(prefix)
        total += cleared
    if total > 0:
        logger.debug(f"Cache invalidated for '{data_source}': {total} entries cleared")
    return total


# =============================================================================
# Rate Limiting
# =============================================================================


class RateLimiter:
    """
    Token bucket rate limiter for API endpoints.

    Provides per-client rate limiting with configurable requests per minute
    and burst capacity. Uses sliding window for accurate rate tracking.

    Example:
        limiter = RateLimiter(requests_per_minute=30, burst=5)
        if not limiter.allow(client_ip):
            return error_response("Rate limit exceeded", 429)
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst: int = 10,
        key_prefix: str = "",
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per client
            burst: Additional burst capacity above the base rate
            key_prefix: Prefix for client keys (to namespace different endpoints)
        """
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.key_prefix = key_prefix
        self._tokens: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_update)
        self._refill_rate = requests_per_minute / 60.0  # tokens per second

    def allow(self, client_key: str) -> tuple[bool, dict]:
        """
        Check if request is allowed for a client.

        Args:
            client_key: Unique client identifier (e.g., IP address, API key)

        Returns:
            Tuple of (allowed, info) where info contains rate limit headers
        """
        now = time.time()
        full_key = f"{self.key_prefix}:{client_key}" if self.key_prefix else client_key

        # Get or create token bucket for this client
        if full_key in self._tokens:
            tokens, last_update = self._tokens[full_key]
            # Refill tokens based on time elapsed
            elapsed = now - last_update
            tokens = min(
                self.requests_per_minute + self.burst,
                tokens + elapsed * self._refill_rate
            )
        else:
            tokens = self.requests_per_minute + self.burst

        # Rate limit info for headers
        info = {
            "X-RateLimit-Limit": str(self.requests_per_minute),
            "X-RateLimit-Remaining": str(max(0, int(tokens) - 1)),
            "X-RateLimit-Reset": str(int(now + 60)),
        }

        # Check if request is allowed
        if tokens >= 1:
            self._tokens[full_key] = (tokens - 1, now)
            return True, info
        else:
            # Calculate retry-after
            tokens_needed = 1 - tokens
            retry_after = int(tokens_needed / self._refill_rate) + 1
            info["Retry-After"] = str(retry_after)
            self._tokens[full_key] = (tokens, now)
            return False, info

    def get_client_key(self, handler) -> str:
        """Extract client key from request handler.

        Uses X-Forwarded-For if behind proxy, otherwise REMOTE_ADDR.
        Falls back to 'anonymous' if neither available.
        """
        if handler is None:
            return "anonymous"

        # Check for forwarded IP (behind proxy)
        if hasattr(handler, 'headers'):
            forwarded = handler.headers.get('X-Forwarded-For', '')
            if forwarded:
                # Take first IP in chain (original client)
                return forwarded.split(',')[0].strip()

        # Check for direct connection
        if hasattr(handler, 'client_address'):
            addr = handler.client_address
            if isinstance(addr, tuple) and len(addr) >= 1:
                return str(addr[0])

        return "anonymous"

    def cleanup(self, max_age_seconds: int = 300) -> int:
        """Remove stale entries older than max_age_seconds.

        Returns number of entries removed.
        """
        now = time.time()
        stale_keys = [
            key for key, (_, last_update) in self._tokens.items()
            if now - last_update > max_age_seconds
        ]
        for key in stale_keys:
            del self._tokens[key]
        return len(stale_keys)


# Global rate limiters for different endpoint categories
_rate_limiters: dict[str, RateLimiter] = {}


def get_rate_limiter(
    name: str,
    requests_per_minute: int = 60,
    burst: int = 10,
) -> RateLimiter:
    """Get or create a named rate limiter.

    Args:
        name: Unique name for this limiter (e.g., "debate_create", "probe_run")
        requests_per_minute: Max requests per minute
        burst: Burst capacity

    Returns:
        RateLimiter instance
    """
    if name not in _rate_limiters:
        _rate_limiters[name] = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst=burst,
            key_prefix=name,
        )
    return _rate_limiters[name]


def rate_limit(
    requests_per_minute: int = 30,
    burst: int = 5,
    limiter_name: Optional[str] = None,
):
    """
    Decorator for rate limiting handler methods.

    Applies token bucket rate limiting per client. Returns 429 Too Many Requests
    when limit exceeded.

    Args:
        requests_per_minute: Maximum requests per minute per client
        burst: Additional burst capacity
        limiter_name: Optional name to share limiter across handlers

    Usage:
        @rate_limit(requests_per_minute=30, burst=5)
        def _run_capability_probe(self, handler):
            ...

        @rate_limit(requests_per_minute=10, burst=2, limiter_name="expensive")
        def _run_deep_analysis(self, path, query_params, handler):
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Get or create limiter
        name = limiter_name or func.__name__
        limiter = get_rate_limiter(name, requests_per_minute, burst)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract handler from args/kwargs
            handler = kwargs.get('handler')
            if handler is None:
                for arg in args:
                    if hasattr(arg, 'headers'):
                        handler = arg
                        break

            # Get client key and check rate limit
            client_key = limiter.get_client_key(handler)
            allowed, info = limiter.allow(client_key)

            if not allowed:
                logger.warning(
                    f"Rate limit exceeded for {client_key} on {func.__name__}"
                )
                return error_response(
                    "Rate limit exceeded. Please try again later.",
                    status=429,
                    headers=info,
                )

            # Call handler and add rate limit headers to response
            result = func(*args, **kwargs)

            # Add headers to response if possible
            if hasattr(result, 'headers') and isinstance(result.headers, dict):
                result.headers.update({
                    k: v for k, v in info.items()
                    if k.startswith('X-RateLimit')
                })

            return result
        return wrapper
    return decorator


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


def json_response(
    data: Any,
    status: int = 200,
    headers: Optional[dict] = None,
) -> HandlerResult:
    """Create a JSON response."""
    body = json.dumps(data, default=str).encode('utf-8')
    return HandlerResult(
        status_code=status,
        content_type="application/json",
        body=body,
        headers=headers or {},
    )


def error_response(
    message: str,
    status: int = 400,
    headers: Optional[dict] = None,
) -> HandlerResult:
    """Create an error response."""
    return json_response({"error": message}, status=status, headers=headers)


# ============================================================================
# Centralized Exception Handling
# ============================================================================

def generate_trace_id() -> str:
    """Generate a unique trace ID for request tracking."""
    return str(uuid.uuid4())[:8]


# Exception to HTTP status code mapping
_EXCEPTION_STATUS_MAP = {
    "FileNotFoundError": 404,
    "KeyError": 404,
    "ValueError": 400,
    "TypeError": 400,
    "json.JSONDecodeError": 400,
    "PermissionError": 403,
    "TimeoutError": 504,
    "asyncio.TimeoutError": 504,
    "ConnectionError": 502,
    "OSError": 500,
}


def _map_exception_to_status(e: Exception, default: int = 500) -> int:
    """Map exception type to appropriate HTTP status code."""
    error_type = type(e).__name__
    return _EXCEPTION_STATUS_MAP.get(error_type, default)


def handle_errors(context: str, default_status: int = 500):
    """
    Decorator for consistent exception handling with tracing.

    Wraps handler methods to:
    - Generate unique trace IDs for debugging
    - Log full exception details server-side
    - Return sanitized error messages to clients
    - Map exceptions to appropriate HTTP status codes

    Args:
        context: Description of the operation (e.g., "debate creation")
        default_status: Default HTTP status for unrecognized exceptions

    Usage:
        @handle_errors("leaderboard retrieval")
        def _get_leaderboard(self, query_params: dict) -> HandlerResult:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = generate_trace_id()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"[{trace_id}] Error in {context}: {type(e).__name__}: {e}",
                    exc_info=True,
                )
                status = _map_exception_to_status(e, default_status)
                message = safe_error_message(e, context)
                return error_response(
                    message,
                    status=status,
                    headers={"X-Trace-Id": trace_id},
                )
        return wrapper
    return decorator


def log_request(context: str, log_response: bool = False):
    """
    Decorator for structured request/response logging.

    Logs request start, completion time, and status code for debugging
    and observability. Use on POST/PUT handlers where detailed logging
    is valuable.

    Args:
        context: Description of the operation (e.g., "debate creation")
        log_response: If True, also log response body (use cautiously for
                     privacy/size reasons)

    Usage:
        @log_request("debate creation")
        def _create_debate(self, path, query_params, handler) -> HandlerResult:
            ...

        @log_request("plugin execution", log_response=True)
        def _run_plugin(self, plugin_name, handler) -> HandlerResult:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            trace_id = generate_trace_id()
            start_time = time.time()
            logger.info(f"[{trace_id}] {context}: started")

            try:
                result = func(*args, **kwargs)
                duration_ms = round((time.time() - start_time) * 1000, 2)

                # Extract status code from result
                status_code = getattr(result, 'status_code', 200) if result else 200

                log_msg = f"[{trace_id}] {context}: {status_code} in {duration_ms}ms"
                if status_code >= 400:
                    logger.warning(log_msg)
                else:
                    logger.info(log_msg)

                if log_response and result:
                    body = getattr(result, 'body', b'')
                    if body and len(body) < 1000:  # Only log small responses
                        logger.debug(f"[{trace_id}] Response: {body.decode('utf-8', errors='ignore')[:500]}")

                return result

            except Exception as e:
                duration_ms = round((time.time() - start_time) * 1000, 2)
                logger.error(
                    f"[{trace_id}] {context}: failed in {duration_ms}ms - {type(e).__name__}: {e}",
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator


def require_auth(func: Callable) -> Callable:
    """
    Decorator that ALWAYS requires authentication, regardless of auth_config.enabled.

    Use this for sensitive endpoints that must never run without authentication,
    even in development/testing environments where global auth may be disabled.

    Examples of sensitive endpoints:
    - Plugin execution (/api/plugins/*/run)
    - Capability probing (/api/probes/run)
    - Laboratory experiments (/api/laboratory/*)

    Usage:
        @require_auth
        def _run_plugin(self, plugin_name: str, handler) -> HandlerResult:
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.server.auth import auth_config

        # Extract handler from kwargs or args
        handler = kwargs.get('handler')
        if handler is None and args:
            # Handler is often the last positional arg
            for arg in args:
                if hasattr(arg, 'headers'):
                    handler = arg
                    break

        if handler is None:
            logger.warning("require_auth: No handler provided, denying access")
            return error_response("Authentication required", 401)

        # Extract auth token from Authorization header
        auth_header = None
        if hasattr(handler, 'headers'):
            auth_header = handler.headers.get('Authorization', '')

        token = None
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header[7:]

        # Check that API token is configured
        if not auth_config.api_token:
            logger.warning(
                "require_auth: No API token configured, denying access to sensitive endpoint"
            )
            return error_response(
                "Authentication required. Set ARAGORA_API_TOKEN environment variable.",
                401
            )

        # Validate the provided token
        if not token or not auth_config.validate_token(token):
            return error_response("Invalid or missing authentication token", 401)

        return func(*args, **kwargs)
    return wrapper


def require_storage(func: Callable) -> Callable:
    """
    Decorator that requires storage to be available before executing the handler.

    Eliminates boilerplate storage availability checks from handler methods.
    Returns 503 Service Unavailable if storage is not configured.

    Usage:
        @require_storage
        def _list_debates(self, handler, limit: int) -> HandlerResult:
            storage = self.get_storage()  # Guaranteed to be non-None
            ...
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        storage = self.get_storage()
        if not storage:
            return error_response("Storage not available", 503)
        return func(self, *args, **kwargs)
    return wrapper


def with_error_recovery(
    fallback_value: Any = None,
    log_errors: bool = True,
    metrics_key: Optional[str] = None,
):
    """
    Decorator for graceful error recovery with fallback values.

    Unlike handle_errors which returns HTTP error responses, this decorator
    returns a fallback value allowing the caller to continue operation.
    Useful for non-critical operations where partial failure is acceptable.

    Args:
        fallback_value: Value to return on error (default: None)
        log_errors: Whether to log errors (default: True)
        metrics_key: Optional key for recording error metrics

    Usage:
        @with_error_recovery(fallback_value=[], log_errors=True)
        def get_optional_data():
            # May fail, but caller gets empty list instead of crash
            ...

        @with_error_recovery(fallback_value={"error": True})
        def fetch_external_service():
            # Returns error dict on failure instead of raising
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}: {type(e).__name__}: {e}",
                        exc_info=True,
                    )
                if metrics_key:
                    try:
                        from aragora.server.prometheus import record_error
                        record_error(metrics_key, str(type(e).__name__))
                    except ImportError:
                        pass
                return fallback_value
        return wrapper
    return decorator


# Note: Validation functions moved to aragora.server.validation
# Use: from aragora.server.validation import validate_against_schema, ValidationResult


def parse_query_params(query_string: str) -> dict:
    """Parse query string into a dictionary."""
    if not query_string:
        return {}
    params = parse_qs(query_string)
    # Convert single-value lists to just values
    return {k: v[0] if len(v) == 1 else v for k, v in params.items()}


def get_int_param(params: dict, key: str, default: int = 0) -> int:
    """Safely get an integer parameter, handling list values from query strings."""
    try:
        value = params.get(key, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return int(value)
    except (ValueError, TypeError):
        return default


def get_float_param(params: dict, key: str, default: float = 0.0) -> float:
    """Safely get a float parameter, handling list values from query strings."""
    try:
        value = params.get(key, default)
        if isinstance(value, list):
            value = value[0] if value else default
        return float(value)
    except (ValueError, TypeError):
        return default


def get_bool_param(params: dict, key: str, default: bool = False) -> bool:
    """Safely get a boolean parameter."""
    value = params.get(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_string_param(params: dict, key: str, default: str | None = None) -> Optional[str]:
    """Safely get a string parameter, handling list values from query strings."""
    value = params.get(key, default)
    if value is None:
        return default
    if isinstance(value, list):
        return value[0] if value else default
    return str(value)


def get_clamped_int_param(
    params: dict,
    key: str,
    default: int,
    min_val: int,
    max_val: int,
) -> int:
    """Get integer parameter clamped to [min_val, max_val].

    Args:
        params: Query parameters dict
        key: Parameter key to look up
        default: Default value if key not found
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Integer value clamped to the specified range
    """
    val = get_int_param(params, key, default)
    return min(max(val, min_val), max_val)


def get_bounded_float_param(
    params: dict,
    key: str,
    default: float,
    min_val: float,
    max_val: float,
) -> float:
    """Get float parameter bounded to [min_val, max_val].

    Args:
        params: Query parameters dict
        key: Parameter key to look up
        default: Default value if key not found
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Float value bounded to the specified range
    """
    val = get_float_param(params, key, default)
    return min(max(val, min_val), max_val)


def get_bounded_string_param(
    params: dict,
    key: str,
    default: str | None = None,
    max_length: int = 500,
) -> Optional[str]:
    """Get string parameter with length limit.

    Args:
        params: Query parameters dict
        key: Parameter key to look up
        default: Default value if key not found
        max_length: Maximum allowed string length

    Returns:
        String value truncated to max_length, or None
    """
    val = get_string_param(params, key, default)
    if val is None:
        return None
    return val[:max_length]


# Note: SAFE_ID_PATTERN, SAFE_AGENT_PATTERN, SAFE_SLUG_PATTERN imported from validation.py
# Path segment validation functions (validate_path_segment, validate_agent_name,
# validate_debate_id) are in aragora.server.validation


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

    # === POST Body Parsing Support ===

    # Maximum request body size (10MB default)
    MAX_BODY_SIZE = 10 * 1024 * 1024

    def read_json_body(self, handler, max_size: int = None) -> Optional[dict]:
        """Read and parse JSON body from request handler.

        Args:
            handler: The HTTP request handler with headers and rfile
            max_size: Maximum body size to accept (default: MAX_BODY_SIZE)

        Returns:
            Parsed JSON dict, empty dict for no content, or None for parse errors
        """
        max_size = max_size or self.MAX_BODY_SIZE
        try:
            content_length = int(handler.headers.get('Content-Length', 0))
            if content_length <= 0:
                return {}
            if content_length > max_size:
                return None  # Body too large
            body = handler.rfile.read(content_length)
            return json.loads(body) if body else {}
        except (json.JSONDecodeError, ValueError):
            return None

    def validate_content_length(self, handler, max_size: int = None) -> Optional[int]:
        """Validate Content-Length header.

        Args:
            handler: The HTTP request handler
            max_size: Maximum allowed size (default: MAX_BODY_SIZE)

        Returns:
            Content length if valid, None if invalid
        """
        max_size = max_size or self.MAX_BODY_SIZE
        try:
            content_length = int(handler.headers.get('Content-Length', '0'))
        except ValueError:
            return None

        if content_length < 0 or content_length > max_size:
            return None

        return content_length

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
