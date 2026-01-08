"""
Base handler utilities for modular endpoint handlers.

Provides common response formatting, error handling, and utilities
shared across all endpoint modules.
"""

import json
import logging
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Generator, Optional, Tuple
from urllib.parse import parse_qs

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.server.error_utils import safe_error_message
from aragora.server.handlers.cache import (
    BoundedTTLCache,
    CACHE_INVALIDATION_MAP,
    CACHE_MAX_ENTRIES,
    CACHE_EVICT_PERCENT,
    _cache,  # Re-export for backwards compatibility
    clear_cache,
    get_cache_stats,
    get_handler_cache,
    invalidate_agent_cache,
    invalidate_cache,
    invalidate_debate_cache,
    invalidate_leaderboard_cache,
    invalidate_on_event,
    ttl_cache,
)
from aragora.server.validation import (
    SAFE_ID_PATTERN,
    SAFE_AGENT_PATTERN,
    SAFE_SLUG_PATTERN,
    # Re-export validation functions for backwards compatibility
    validate_path_segment,
    validate_agent_name,
    validate_debate_id,
)

# Rate limiting is available from aragora.server.middleware.rate_limit
# or aragora.server.rate_limit for backward compatibility

# Re-export DB_TIMEOUT_SECONDS for backwards compatibility
__all__ = [
    "DB_TIMEOUT_SECONDS", "require_auth", "require_storage", "require_feature",
    "error_response", "json_response", "handle_errors", "log_request", "ttl_cache",
    "clear_cache", "get_cache_stats", "CACHE_INVALIDATION_MAP", "invalidate_cache",
    "invalidate_on_event", "invalidate_leaderboard_cache", "invalidate_agent_cache",
    "invalidate_debate_cache", "PathMatcher", "RouteDispatcher", "safe_fetch",
    "get_db_connection", "table_exists", "safe_get", "safe_get_nested", "safe_json_parse",
    "get_host_header", "get_agent_name",
    "SAFE_ID_PATTERN", "SAFE_SLUG_PATTERN", "SAFE_AGENT_PATTERN",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Database Connection Helper
# =============================================================================

@contextmanager
def get_db_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection with proper cleanup.

    Shared utility for handlers that need direct database access.
    Uses DB_TIMEOUT_SECONDS for consistent timeout handling.

    Args:
        db_path: Path to the SQLite database file

    Yields:
        sqlite3.Connection with timeout configured

    Example:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
            rows = cursor.fetchall()
    """
    conn = sqlite3.connect(db_path, timeout=DB_TIMEOUT_SECONDS)
    try:
        yield conn
    finally:
        conn.close()


def table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    """Check if a table exists in the SQLite database.

    Args:
        cursor: Active database cursor
        table_name: Name of the table to check

    Returns:
        True if the table exists, False otherwise

    Example:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            if not table_exists(cursor, "agent_relationships"):
                return json_response({"error": "Table not found"})
    """
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


# =============================================================================
# Dict Access Helpers
# =============================================================================

def safe_get(data: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict-like object.

    Handles None values, missing keys, and non-dict inputs gracefully.

    Args:
        data: Dict or dict-like object (can be None)
        key: Key to look up
        default: Default value if key not found or data is None

    Returns:
        The value at key, or default

    Example:
        # Before:
        value = data.get("key", []) if data else []

        # After:
        value = safe_get(data, "key", [])
    """
    if data is None:
        return default
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


def safe_get_nested(data: Any, keys: list[str], default: Any = None) -> Any:
    """Safely navigate nested dict structures.

    Args:
        data: Root dict or dict-like object (can be None)
        keys: List of keys to traverse
        default: Default value if any key not found

    Returns:
        The nested value, or default

    Example:
        # Before:
        value = data.get("outer", {}).get("inner", {}).get("deep", [])

        # After:
        value = safe_get_nested(data, ["outer", "inner", "deep"], [])
    """
    current = data
    for key in keys:
        if current is None:
            return default
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


def safe_json_parse(data: Any, default: Any = None) -> Any:
    """Safely parse JSON string or return dict/list as-is.

    Handles the common pattern where a value might be stored as either
    a JSON string or already parsed as a dict/list.

    Args:
        data: JSON string, dict, list, or None
        default: Default value if parsing fails or data is None

    Returns:
        Parsed dict/list, or default on failure

    Example:
        # Before (6 lines):
        raw = debate.get("grounded_verdict")
        if isinstance(raw, str):
            try:
                verdict = json.loads(raw)
            except json.JSONDecodeError:
                verdict = None

        # After (1 line):
        verdict = safe_json_parse(debate.get("grounded_verdict"))
    """
    if data is None:
        return default
    if isinstance(data, (dict, list)):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, ValueError):
            return default
    return default


def get_host_header(handler, default: str = 'localhost:8080') -> str:
    """Extract Host header from request handler.

    Args:
        handler: HTTP request handler with headers attribute
        default: Default value if handler is None or Host header missing

    Returns:
        Host header value or default

    Example:
        # Before (repeated 5+ times):
        host = handler.headers.get('Host', 'localhost:8080') if handler else 'localhost:8080'

        # After:
        host = get_host_header(handler)
    """
    if handler is None:
        return default
    return handler.headers.get('Host', default) if hasattr(handler, 'headers') else default


def get_agent_name(agent: Any) -> Optional[str]:
    """Extract agent name from dict or object.

    Handles the common pattern where agent data might be either
    a dict with 'name'/'agent_name' key or an object with name attribute.

    Args:
        agent: Dict or object containing agent name

    Returns:
        Agent name string or None if not found

    Example:
        # Before (repeated 4+ times):
        name = agent.get("name") if isinstance(agent, dict) else getattr(agent, "name", None)

        # After:
        name = get_agent_name(agent)
    """
    if agent is None:
        return None
    if isinstance(agent, dict):
        return agent.get("agent_name") or agent.get("name")
    return getattr(agent, "agent_name", None) or getattr(agent, "name", None)


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


def handle_errors(context: str, default_status: int = 500) -> Callable[[Callable], Callable]:
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

    Returns:
        Decorator function that wraps handler methods with error handling.

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


def log_request(context: str, log_response: bool = False) -> Callable[[Callable], Callable]:
    """
    Decorator for structured request/response logging.

    Logs request start, completion time, and status code for debugging
    and observability. Use on POST/PUT handlers where detailed logging
    is valuable.

    Args:
        context: Description of the operation (e.g., "debate creation")
        log_response: If True, also log response body (use cautiously for
                     privacy/size reasons)

    Returns:
        Decorator function that wraps handler methods with logging.

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


def require_feature(
    feature_check: Callable[[], bool],
    feature_name: str,
    status_code: int = 503,
):
    """
    Decorator that requires a feature to be available before executing the handler.

    Eliminates repetitive feature availability checks from handler methods.
    Returns an error response if the feature is not available.

    Args:
        feature_check: Callable that returns True if feature is available
        feature_name: Human-readable name for error message
        status_code: HTTP status code to return if unavailable (default 503)

    Usage:
        @require_feature(lambda: CONSENSUS_AVAILABLE, "Consensus memory")
        def _get_similar_debates(self, topic: str, limit: int) -> HandlerResult:
            # CONSENSUS_AVAILABLE is guaranteed True here
            ...

        # Or with a more complex check:
        @require_feature(lambda: self.ctx.get("embeddings") is not None, "Embeddings")
        def _search_debates(self, query: str) -> HandlerResult:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not feature_check():
                return error_response(f"{feature_name} not available", status_code)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def safe_fetch(
    data_dict: dict,
    errors_dict: dict,
    key: str,
    fallback: Any,
    log_errors: bool = True,
):
    """
    Context manager for safe data fetching with graceful fallback.

    Consolidates the common pattern of:
    1. Try to fetch data
    2. Store in data dict on success
    3. Store fallback and record error on failure

    Args:
        data_dict: Dict to store successful result
        errors_dict: Dict to store error messages
        key: Key for both dicts
        fallback: Value to use on error
        log_errors: Whether to log errors (default: True)

    Usage:
        data = {}
        errors = {}

        # Before (4 lines repeated 6+ times):
        try:
            data["rankings"] = self._fetch_rankings(limit)
        except Exception as e:
            errors["rankings"] = str(e)
            data["rankings"] = {"agents": [], "count": 0}

        # After (2 lines):
        with safe_fetch(data, errors, "rankings", {"agents": [], "count": 0}):
            data["rankings"] = self._fetch_rankings(limit)

    Note: The fetch operation should store directly into data_dict[key].
    On exception, the context manager stores the fallback value.
    """
    from contextlib import contextmanager

    @contextmanager
    def _safe_fetch():
        try:
            yield
        except Exception as e:
            if log_errors:
                logger.warning(f"safe_fetch '{key}' failed: {type(e).__name__}: {e}")
            errors_dict[key] = str(e)
            data_dict[key] = fallback

    return _safe_fetch()


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


class PathMatcher:
    """Utility for matching URL paths against patterns.

    Simplifies the common pattern of parsing path segments and dispatching
    to handler methods.

    Example:
        matcher = PathMatcher("/api/agent/{name}/{action}")
        result = matcher.match("/api/agent/claude/profile")
        # result = {"name": "claude", "action": "profile"}

        matcher = PathMatcher("/api/debates")
        result = matcher.match("/api/debates")
        # result = {}  (empty dict = matched)

        result = matcher.match("/api/other")
        # result = None  (None = no match)
    """

    def __init__(self, pattern: str):
        """Initialize with a URL pattern.

        Args:
            pattern: URL pattern with {param} placeholders for path segments
        """
        self.pattern = pattern
        self.parts = pattern.strip("/").split("/")
        self.param_indices: dict[str, int] = {}

        for i, part in enumerate(self.parts):
            if part.startswith("{") and part.endswith("}"):
                param_name = part[1:-1]
                self.param_indices[param_name] = i

    def match(self, path: str) -> dict | None:
        """Match a path against this pattern.

        Returns:
            Dict of extracted parameters if matched, None otherwise
        """
        path_parts = path.strip("/").split("/")

        if len(path_parts) != len(self.parts):
            return None

        params = {}
        for i, (pattern_part, path_part) in enumerate(zip(self.parts, path_parts)):
            if pattern_part.startswith("{") and pattern_part.endswith("}"):
                param_name = pattern_part[1:-1]
                params[param_name] = path_part
            elif pattern_part != path_part:
                return None

        return params

    def matches(self, path: str) -> bool:
        """Check if a path matches this pattern."""
        return self.match(path) is not None


class RouteDispatcher:
    """Dispatcher for routing paths to handler methods.

    Simplifies the common pattern of if/elif chains in handle() methods.

    Example:
        dispatcher = RouteDispatcher()
        dispatcher.add_route("/api/agents", self._list_agents)
        dispatcher.add_route("/api/agent/{name}/profile", self._get_profile)
        dispatcher.add_route("/api/agent/{name}/history", self._get_history)

        # In handle() method:
        result = dispatcher.dispatch(path, query_params)
        if result is not None:
            return result
    """

    def __init__(self):
        self.routes: list[tuple[PathMatcher, callable]] = []

    def add_route(self, pattern: str, handler: callable) -> "RouteDispatcher":
        """Add a route pattern with its handler.

        Args:
            pattern: URL pattern with {param} placeholders
            handler: Callable that receives (params_dict, query_params)
                     or just () if no path params

        Returns:
            Self for chaining
        """
        self.routes.append((PathMatcher(pattern), handler))
        return self

    def dispatch(self, path: str, query_params: dict = None) -> any:
        """Dispatch a path to its handler.

        Args:
            path: URL path to dispatch
            query_params: Query parameters dict

        Returns:
            Handler result if matched, None otherwise
        """
        query_params = query_params or {}

        for matcher, handler in self.routes:
            params = matcher.match(path)
            if params is not None:
                # Call handler with path params and query params
                if params:
                    return handler(params, query_params)
                else:
                    return handler(query_params)

        return None

    def can_handle(self, path: str) -> bool:
        """Check if any route can handle this path."""
        return any(matcher.matches(path) for matcher, _ in self.routes)


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

    def extract_path_param(
        self,
        path: str,
        segment_index: int,
        param_name: str,
        pattern: re.Pattern = None,
    ) -> Tuple[Optional[str], Optional[HandlerResult]]:
        """Extract and validate a path segment parameter.

        Consolidates the common pattern of:
        1. Split path into parts
        2. Check segment exists at index
        3. Validate against pattern
        4. Return error response if invalid

        Args:
            path: URL path to extract from
            segment_index: Index of segment to extract (0-based)
            param_name: Human-readable name for error messages
            pattern: Regex pattern to validate against (default: SAFE_ID_PATTERN)

        Returns:
            Tuple of (value, error_response):
            - (value, None) on success
            - (None, HandlerResult) on failure

        Example:
            # Before:
            parts = path.split("/")
            if len(parts) < 4:
                return error_response("Invalid path", 400)
            domain = parts[3]
            is_valid, err = validate_path_segment(domain, "domain", SAFE_ID_PATTERN)
            if not is_valid:
                return error_response(err, 400)

            # After:
            domain, err = self.extract_path_param(path, 3, "domain")
            if err:
                return err
        """
        pattern = pattern or SAFE_ID_PATTERN
        parts = path.strip("/").split("/")

        if segment_index >= len(parts):
            return None, error_response(f"Missing {param_name} in path", 400)

        value = parts[segment_index]
        if not value:
            return None, error_response(f"Empty {param_name}", 400)

        is_valid, err_msg = validate_path_segment(value, param_name, pattern)
        if not is_valid:
            return None, error_response(err_msg, 400)

        return value, None

    def extract_path_params(
        self,
        path: str,
        param_specs: list[Tuple[int, str, Optional[re.Pattern]]],
    ) -> Tuple[Optional[dict], Optional[HandlerResult]]:
        """Extract and validate multiple path parameters at once.

        Args:
            path: URL path to extract from
            param_specs: List of (segment_index, param_name, pattern) tuples.
                        If pattern is None, SAFE_ID_PATTERN is used.

        Returns:
            Tuple of (params_dict, error_response):
            - ({"name": value, ...}, None) on success
            - (None, HandlerResult) on first failure

        Example:
            # Extract agent_a and agent_b from /api/agents/compare/claude/gpt4
            params, err = self.extract_path_params(path, [
                (3, "agent_a", SAFE_AGENT_PATTERN),
                (4, "agent_b", SAFE_AGENT_PATTERN),
            ])
            if err:
                return err
            # params = {"agent_a": "claude", "agent_b": "gpt4"}
        """
        result = {}
        for segment_index, param_name, pattern in param_specs:
            value, err = self.extract_path_param(
                path, segment_index, param_name, pattern
            )
            if err:
                return None, err
            result[param_name] = value
        return result, None

    def get_storage(self) -> Optional[Any]:
        """Get debate storage instance."""
        return self.ctx.get("storage")

    def get_elo_system(self) -> Optional[Any]:
        """Get ELO system instance."""
        return self.ctx.get("elo_system")

    def get_debate_embeddings(self) -> Optional[Any]:
        """Get debate embeddings database."""
        return self.ctx.get("debate_embeddings")

    def get_critique_store(self) -> Optional[Any]:
        """Get critique store instance."""
        return self.ctx.get("critique_store")

    def get_nomic_dir(self) -> Optional[Any]:
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
