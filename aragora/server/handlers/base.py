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
from aragora.server.validation import SAFE_ID_PATTERN, SAFE_AGENT_PATTERN, SAFE_SLUG_PATTERN

# Re-export DB_TIMEOUT_SECONDS for backwards compatibility
__all__ = ["DB_TIMEOUT_SECONDS"]

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


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    is_valid: bool
    error: Optional[str] = None
    validated_params: Optional[dict] = None


def validate_against_schema(params: dict, schema: dict) -> ValidationResult:
    """
    Validate request parameters against a simple schema.

    Schema format:
        {
            "param_name": {
                "type": "int" | "float" | "string" | "bool",
                "required": True | False,
                "min": <number>,  # For int/float
                "max": <number>,  # For int/float
                "pattern": <regex>,  # For strings
                "choices": [<list>],  # Allowed values
                "default": <value>,  # Default if not provided
            },
            ...
        }

    Args:
        params: Dictionary of request parameters
        schema: Validation schema

    Returns:
        ValidationResult with is_valid, error message, and validated params
    """
    validated = {}
    errors = []

    for param_name, rules in schema.items():
        value = params.get(param_name)
        required = rules.get("required", False)
        default = rules.get("default")
        param_type = rules.get("type", "string")

        # Handle missing values
        if value is None:
            if required:
                errors.append(f"Missing required parameter: {param_name}")
                continue
            elif default is not None:
                value = default
            else:
                continue

        # Type coercion and validation
        try:
            if param_type == "int":
                value = int(value)
                if "min" in rules and value < rules["min"]:
                    errors.append(f"{param_name} must be >= {rules['min']}")
                if "max" in rules and value > rules["max"]:
                    errors.append(f"{param_name} must be <= {rules['max']}")

            elif param_type == "float":
                value = float(value)
                if "min" in rules and value < rules["min"]:
                    errors.append(f"{param_name} must be >= {rules['min']}")
                if "max" in rules and value > rules["max"]:
                    errors.append(f"{param_name} must be <= {rules['max']}")

            elif param_type == "bool":
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                else:
                    value = bool(value)

            elif param_type == "string":
                value = str(value)
                if "pattern" in rules:
                    pattern = rules["pattern"]
                    if isinstance(pattern, str):
                        pattern = re.compile(pattern)
                    if not pattern.match(value):
                        errors.append(f"{param_name} has invalid format")

            # Check choices
            if "choices" in rules and value not in rules["choices"]:
                errors.append(f"{param_name} must be one of: {rules['choices']}")

            validated[param_name] = value

        except (ValueError, TypeError) as e:
            errors.append(f"Invalid {param_name}: {e}")

    if errors:
        return ValidationResult(is_valid=False, error="; ".join(errors))

    return ValidationResult(is_valid=True, validated_params=validated)


def validate_params(schema: dict):
    """
    Decorator to validate request parameters against a schema.

    Validates query_params (second argument after self) and returns
    error response if validation fails.

    Args:
        schema: Validation schema (see validate_against_schema for format)

    Usage:
        @validate_params({
            "limit": {"type": "int", "min": 1, "max": 100, "default": 20},
            "agent": {"type": "string", "required": True, "pattern": SAFE_AGENT_PATTERN},
        })
        def _get_agent_data(self, path, query_params, handler):
            limit = query_params.get("limit")  # Already validated and converted
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, path, query_params, *args, **kwargs):
            result = validate_against_schema(query_params, schema)
            if not result.is_valid:
                return error_response(result.error, 400)
            # Merge validated params back (with type conversions applied)
            if result.validated_params:
                query_params.update(result.validated_params)
            return func(self, path, query_params, *args, **kwargs)
        return wrapper
    return decorator


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


# Note: SAFE_ID_PATTERN, SAFE_AGENT_PATTERN, SAFE_SLUG_PATTERN imported from validation.py


def validate_path_segment(
    value: str,
    name: str,
    pattern: re.Pattern = SAFE_ID_PATTERN,
) -> Tuple[bool, Optional[str]]:
    """Validate a path segment against a pattern.

    Args:
        value: The value to validate
        name: Name of the segment for error messages
        pattern: Regex pattern to match against

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not value:
        return False, f"Missing {name}"
    if '..' in value or '/' in value:
        return False, f"Invalid {name}: path traversal not allowed"
    if not pattern.match(value):
        return False, f"Invalid {name} format"
    return True, None


def validate_agent_name(agent: str) -> Tuple[bool, Optional[str]]:
    """Validate an agent name.

    Args:
        agent: Agent name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    return validate_path_segment(agent, "agent name", SAFE_AGENT_PATTERN)


def validate_debate_id(debate_id: str) -> Tuple[bool, Optional[str]]:
    """Validate a debate ID.

    Args:
        debate_id: Debate ID to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    return validate_path_segment(debate_id, "debate ID", SAFE_SLUG_PATTERN)


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
