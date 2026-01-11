"""
Base handler utilities for modular endpoint handlers.

Provides common response formatting, error handling, and utilities
shared across all endpoint modules.

Note: Some utilities have been extracted to handlers/utils/ for better
organization. They are re-exported here for backwards compatibility.

Authentication Requirements
---------------------------
Aragora uses header-based authentication (Bearer tokens in Authorization header).
This approach is inherently immune to CSRF attacks because:

1. Tokens are sent via Authorization header, not cookies
2. JavaScript must explicitly set the header (not automatic like cookies)
3. Cross-origin requests cannot access the header

Endpoint Authentication Patterns:
- Public endpoints: No authentication required (e.g., /api/plans, /api/health)
- Protected endpoints: Require valid Bearer token
- Write operations: Always require authentication

Handler classes should use `extract_user_from_request()` for authentication:

    from aragora.billing.jwt_auth import extract_user_from_request

    def _handle_protected(self, handler) -> HandlerResult:
        user_store = self._get_user_store()
        auth_ctx = extract_user_from_request(handler, user_store)
        if not auth_ctx.is_authenticated:
            return error_response("Not authenticated", 401)
        # ... proceed with authenticated logic

Rate limiting is applied via the @rate_limit decorator from utils.rate_limit.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Generator, Optional, Tuple
from urllib.parse import parse_qs

from aragora.config import DB_TIMEOUT_SECONDS

# Import from extracted utility modules (re-exported for backwards compatibility)
from aragora.server.handlers.utils.safe_data import (
    safe_get,
    safe_get_nested,
    safe_json_parse,
)
from aragora.server.handlers.utils.database import (
    get_db_connection,
    table_exists,
)
from aragora.server.handlers.utils.params import (
    parse_query_params,
    get_int_param,
    get_float_param,
    get_bool_param,
    get_string_param,
    get_clamped_int_param,
    get_bounded_float_param,
    get_bounded_string_param,
)
from aragora.server.handlers.utils.routing import (
    PathMatcher,
    RouteDispatcher,
)
from aragora.server.error_utils import safe_error_message
from aragora.server.handlers.cache import (
    BoundedTTLCache,
    CACHE_INVALIDATION_MAP,
    CACHE_MAX_ENTRIES,
    CACHE_EVICT_PERCENT,
    _cache,  # Re-export for backwards compatibility
    async_ttl_cache,
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
    "DB_TIMEOUT_SECONDS", "require_auth", "require_user_auth", "require_storage", "require_feature",
    "error_response", "json_response", "handle_errors", "auto_error_response", "log_request", "ttl_cache",
    "safe_error_message", "safe_error_response",
    "async_ttl_cache", "clear_cache", "get_cache_stats", "CACHE_INVALIDATION_MAP", "invalidate_cache",
    "invalidate_on_event", "invalidate_leaderboard_cache", "invalidate_agent_cache",
    "invalidate_debate_cache", "PathMatcher", "RouteDispatcher", "safe_fetch",
    "get_db_connection", "table_exists", "safe_get", "safe_get_nested", "safe_json_parse",
    "get_host_header", "get_agent_name", "agent_to_dict", "validate_params",
    "SAFE_ID_PATTERN", "SAFE_SLUG_PATTERN", "SAFE_AGENT_PATTERN",
    "feature_unavailable_response",
    # Parameter extraction helpers
    "get_int_param", "get_float_param", "get_bool_param", "get_string_param",
    "get_clamped_int_param", "get_bounded_float_param", "get_bounded_string_param",
    # Handler mixins
    "PaginatedHandlerMixin", "CachedHandlerMixin", "AuthenticatedHandlerMixin",
    "BaseHandler",
]


def feature_unavailable_response(
    feature_id: str,
    message: Optional[str] = None,
) -> "HandlerResult":
    """
    Create a standardized response for unavailable features.

    This should be used by all handlers when an optional feature is missing.
    Provides consistent error format with helpful installation hints.

    Args:
        feature_id: The feature identifier (e.g., "pulse", "genesis")
        message: Optional custom message (defaults to feature description)

    Returns:
        HandlerResult with 503 status and helpful information

    Example:
        if not self._pulse_manager:
            return feature_unavailable_response("pulse")
    """
    # Import here to avoid circular imports
    from aragora.server.handlers.features import (
        feature_unavailable_response as _feature_unavailable,
    )
    return _feature_unavailable(feature_id, message)

logger = logging.getLogger(__name__)


# =============================================================================
# Database Connection Helper (imported from utils/database.py)
# =============================================================================
# get_db_connection and table_exists are now in aragora.server.handlers.utils.database
# and re-exported above for backwards compatibility


# =============================================================================
# Dict Access Helpers (imported from utils/safe_data.py)
# =============================================================================
# safe_get, safe_get_nested, safe_json_parse are now in
# aragora.server.handlers.utils.safe_data and re-exported above


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


def agent_to_dict(agent: Any, include_name: bool = True) -> dict:
    """Convert agent object or dict to standardized dict with ELO fields.

    Handles the common pattern where agent data might be either a dict
    or an AgentRating object, extracting standard fields with safe defaults.

    Args:
        agent: Dict or object containing agent data
        include_name: Whether to include name/agent_name fields (default: True)

    Returns:
        Dict with standardized ELO-related fields

    Example:
        # Before (repeated 40+ times across handlers):
        agent_dict = {
            "name": getattr(agent, "name", "unknown"),
            "elo": getattr(agent, "elo", 1500),
            "wins": getattr(agent, "wins", 0),
            "losses": getattr(agent, "losses", 0),
            ...
        }

        # After:
        agent_dict = agent_to_dict(agent)
    """
    if agent is None:
        return {}

    if isinstance(agent, dict):
        return agent.copy()

    # Extract standard ELO fields from object
    name = get_agent_name(agent) or "unknown"
    result = {
        "elo": getattr(agent, "elo", 1500),
        "wins": getattr(agent, "wins", 0),
        "losses": getattr(agent, "losses", 0),
        "draws": getattr(agent, "draws", 0),
        "win_rate": getattr(agent, "win_rate", 0.0),
        "games": getattr(agent, "games_played", getattr(agent, "games", 0)),
        "matches": getattr(agent, "matches", 0),
    }

    if include_name:
        result["name"] = name
        result["agent_name"] = name

    return result


# =============================================================================
# Response Builders (imported from utils/responses.py)
# =============================================================================
# Core response utilities are now in aragora.server.handlers.utils.responses
# and re-exported here for backwards compatibility

from aragora.server.handlers.utils.responses import (
    HandlerResult,
    json_response,
    error_response,
    html_response,
    redirect_response,
)


def safe_error_response(
    exception: Exception,
    context: str,
    status: int = 500,
    handler: Any = None,
) -> HandlerResult:
    """Create an error response with sanitized message.

    Logs full exception details server-side for debugging, but returns only
    a generic, safe message to the client. This prevents information disclosure
    of internal paths, stack traces, and sensitive configuration.

    Args:
        exception: The exception that occurred
        context: Context for logging (e.g., "debate creation", "agent lookup")
        status: HTTP status code (default: 500)
        handler: Optional request handler for extracting trace_id

    Returns:
        HandlerResult with sanitized error message

    Example:
        try:
            result = do_something()
        except Exception as e:
            return safe_error_response(e, "debate creation", 500, handler)
    """
    from aragora.server.error_utils import ErrorFormatter

    # Generate or extract trace ID
    trace_id = None
    if handler is not None:
        # Try to get existing trace_id from handler
        if hasattr(handler, "trace_id"):
            trace_id = handler.trace_id
        elif hasattr(handler, "headers") and handler.headers:
            trace_id = handler.headers.get("X-Request-ID") or handler.headers.get("X-Trace-ID")
    if not trace_id:
        trace_id = generate_trace_id()

    # Format error with sanitization (logs full details server-side)
    error_dict = ErrorFormatter.format_server_error(
        exception, context=context, trace_id=trace_id
    )

    return json_response(error_dict, status=status)


# ============================================================================
# Centralized Exception Handling
# ============================================================================

def generate_trace_id() -> str:
    """Generate a unique trace ID for request tracking."""
    return str(uuid.uuid4())[:8]


# Exception to HTTP status code mapping
# Maps exception type names to appropriate HTTP status codes.
# Includes both Python built-ins and Aragora-specific exceptions.
_EXCEPTION_STATUS_MAP = {
    # Python built-in exceptions
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
    # Aragora validation errors (400 Bad Request)
    "ValidationError": 400,
    "InputValidationError": 400,
    "SchemaValidationError": 400,
    "DebateConfigurationError": 400,
    "AgentConfigurationError": 400,
    "ModeConfigurationError": 400,
    "ConvergenceThresholdError": 400,
    "CacheKeyError": 400,
    # Aragora not found errors (404)
    "DebateNotFoundError": 404,
    "AgentNotFoundError": 404,
    "RecordNotFoundError": 404,
    "ModeNotFoundError": 404,
    "PluginNotFoundError": 404,
    "CheckpointNotFoundError": 404,
    # Aragora auth errors
    "AuthenticationError": 401,
    "TokenExpiredError": 401,
    "AuthorizationError": 403,
    "RateLimitExceededError": 429,
    # Aragora storage errors (500/503)
    "StorageError": 500,
    "DatabaseError": 500,
    "DatabaseConnectionError": 503,
    "MemoryStorageError": 500,
    "CheckpointSaveError": 500,
    # Aragora agent errors
    "AgentTimeoutError": 504,
    "AgentRateLimitError": 429,
    "AgentConnectionError": 502,
    "AgentCircuitOpenError": 503,
    # Aragora verification/convergence errors
    "VerificationTimeoutError": 504,
    "Z3NotAvailableError": 503,
    "ConvergenceBackendError": 503,
}


def _map_exception_to_status(e: Exception, default: int = 500) -> int:
    """Map exception type to appropriate HTTP status code."""
    error_type = type(e).__name__
    return _EXCEPTION_STATUS_MAP.get(error_type, default)


def validate_params(
    param_specs: dict[str, tuple],
    query_params_arg: str = "query_params",
) -> Callable[[Callable], Callable]:
    """
    Decorator for declarative query parameter validation.

    Validates and extracts query parameters before calling the handler method.
    If validation fails, returns an error response without calling the handler.

    Args:
        param_specs: Dict of param_name -> (type, default, min_val, max_val)
                    - type: int, float, str, or bool
                    - default: Default value if param not provided
                    - min_val: Minimum value (for int/float) or None
                    - max_val: Maximum value (for int/float/str length) or None
        query_params_arg: Name of the query_params argument in the function

    Returns:
        Decorator that validates params and passes extracted values.

    Usage:
        @validate_params({
            "limit": (int, 20, 1, 100),
            "offset": (int, 0, 0, None),
            "domain": (str, None, None, 50),
        })
        @handle_errors("list debates")
        def _list_debates(self, query_params: dict, limit: int, offset: int, domain: str):
            # limit, offset, domain are already validated and typed
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find query_params in kwargs or positional args
            params = kwargs.get(query_params_arg)
            if params is None:
                # Try to find it in positional args by introspection
                import inspect
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                if query_params_arg in param_names:
                    idx = param_names.index(query_params_arg)
                    # Account for 'self' being first arg
                    if idx < len(args):
                        params = args[idx]

            if params is None:
                params = {}

            # Validate and extract each parameter
            extracted = {}
            for name, spec in param_specs.items():
                param_type, default, min_val, max_val = spec

                if param_type == int:
                    val = get_int_param(params, name, default)
                    if min_val is not None:
                        val = max(val, min_val)
                    if max_val is not None:
                        val = min(val, max_val)
                elif param_type == float:
                    val = get_float_param(params, name, default)
                    if min_val is not None:
                        val = max(val, min_val)
                    if max_val is not None:
                        val = min(val, max_val)
                elif param_type == bool:
                    val = get_bool_param(params, name, default)
                elif param_type == str:
                    val = get_string_param(params, name, default)
                    if val is not None and max_val is not None:
                        val = val[:max_val]
                else:
                    val = params.get(name, default)

                extracted[name] = val

            # Merge extracted params into kwargs
            kwargs.update(extracted)
            return func(*args, **kwargs)
        return wrapper
    return decorator


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


def auto_error_response(
    operation: str,
    log_level: str = "error",
    include_traceback: bool = True,
) -> Callable:
    """
    Decorator to automatically wrap handler methods with error handling.

    This is a simpler alternative to @handle_errors for common cases where you
    want to consolidate the repetitive pattern:

        try:
            return json_response(data)
        except Exception as e:
            logger.error(f"Failed to {operation}: {e}")
            return error_response(f"Failed to {operation}: {e}", 500)

    The decorator handles specific exception types with appropriate status codes:
    - sqlite3.OperationalError -> 503 (Database unavailable)
    - PermissionError -> 403 (Access denied)
    - ValueError -> 400 (Invalid request)
    - Other exceptions -> 500 (Internal server error)

    Args:
        operation: Human-readable description for error messages
                  (e.g., "fetch trending topics", "update agent settings")
        log_level: "error", "warning", or None to skip logging
        include_traceback: Whether to include exc_info in logs

    Returns:
        Decorated function that catches exceptions and returns error responses

    Usage:
        # Before (11 lines):
        def _get_trending_topics(self, limit: int) -> HandlerResult:
            try:
                from aragora.pulse.ingestor import PulseManager
                manager = PulseManager()
                topics = manager.get_trending_topics(limit)
                return json_response({"topics": topics})
            except Exception as e:
                logger.error(f"Failed to fetch trending topics: {e}")
                return error_response(f"Failed to fetch trending topics: {e}", 500)

        # After (6 lines):
        @auto_error_response("fetch trending topics")
        def _get_trending_topics(self, limit: int) -> HandlerResult:
            from aragora.pulse.ingestor import PulseManager
            manager = PulseManager()
            topics = manager.get_trending_topics(limit)
            return json_response({"topics": topics})

    Note:
        For endpoints that need trace IDs, structured errors, or custom
        exception mapping, use @handle_errors instead.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> HandlerResult:
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                logger.error(f"Database error in {operation}: {e}")
                return error_response("Database unavailable", 503)
            except PermissionError:
                return error_response("Access denied", 403)
            except ValueError as e:
                # Log full details but return sanitized message
                logger.warning(f"Invalid request in {operation}: {e}")
                return error_response("Invalid request", 400)
            except Exception as e:
                if log_level == "error":
                    logger.error(
                        f"Failed to {operation}: {e}",
                        exc_info=include_traceback,
                    )
                elif log_level == "warning":
                    logger.warning(f"Failed to {operation}: {e}")
                # Return sanitized error message to client
                return error_response(safe_error_message(e, operation), 500)
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


def require_user_auth(func: Callable) -> Callable:
    """
    Decorator that requires JWT/API key user authentication.

    Uses the billing JWT auth system (aragora.billing.jwt_auth) to validate
    Bearer tokens and API keys. Returns UserAuthContext with user info.

    The authenticated user context is passed as 'user' keyword argument.

    Usage:
        @require_user_auth
        def _protected_endpoint(self, handler, user: UserAuthContext) -> HandlerResult:
            # user.user_id, user.org_id, user.email available
            return json_response({"user_id": user.user_id})
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.billing.jwt_auth import extract_user_from_request, UserAuthContext

        # Extract handler from kwargs or args
        handler = kwargs.get('handler')
        if handler is None and args:
            for arg in args:
                if hasattr(arg, 'headers'):
                    handler = arg
                    break

        if handler is None:
            logger.warning("require_user_auth: No handler provided")
            return error_response("Authentication required", 401)

        # Get user store from handler or context
        user_store = None
        if hasattr(handler, 'user_store'):
            user_store = handler.user_store
        elif hasattr(handler.__class__, 'user_store'):
            user_store = handler.__class__.user_store

        # Extract user from request
        user_ctx = extract_user_from_request(handler, user_store)

        if not user_ctx.is_authenticated:
            error_msg = user_ctx.error_reason or "Authentication required"
            return error_response(error_msg, 401)

        # Inject user context into kwargs
        kwargs['user'] = user_ctx
        return func(*args, **kwargs)

    return wrapper


def require_auth(func: Callable) -> Callable:
    """
    Decorator that ALWAYS requires authentication, regardless of auth_config.enabled.

    Use this for sensitive endpoints that must never run without authentication,
    even in development/testing environments where global auth may be disabled.

    NOTE: For JWT/API key authentication, use @require_user_auth instead.
    This decorator uses the legacy ARAGORA_API_TOKEN system.

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

# Note: PathMatcher and RouteDispatcher moved to aragora.server.handlers.utils.routing
# Note: Parameter helpers moved to aragora.server.handlers.utils.params
# Note: SAFE_ID_PATTERN, SAFE_AGENT_PATTERN, SAFE_SLUG_PATTERN imported from validation.py
# Path segment validation functions (validate_path_segment, validate_agent_name,
# validate_debate_id) are in aragora.server.validation


# =============================================================================
# Handler Mixins
# =============================================================================
# These mixins provide reusable patterns for common handler operations.
# Handlers can inherit from these in addition to BaseHandler to get
# standardized implementations of common operations.


class PaginatedHandlerMixin:
    """Mixin for standardized pagination handling.

    Provides consistent limit/offset extraction with validation and defaults.

    Usage:
        class MyHandler(BaseHandler, PaginatedHandlerMixin):
            def handle(self, path, query_params, handler):
                limit, offset = self.get_pagination(query_params)
                results = self.get_data(limit=limit, offset=offset)
                return self.paginated_response(results, total=100, limit=limit, offset=offset)
    """

    DEFAULT_LIMIT = 20
    MAX_LIMIT = 100
    DEFAULT_OFFSET = 0

    def get_pagination(
        self,
        query_params: dict,
        default_limit: int | None = None,
        max_limit: int | None = None,
    ) -> tuple[int, int]:
        """Extract and validate pagination parameters.

        Args:
            query_params: Query parameters dict
            default_limit: Override default limit (default: DEFAULT_LIMIT)
            max_limit: Override max limit (default: MAX_LIMIT)

        Returns:
            Tuple of (limit, offset) with validated bounds
        """
        default_limit = default_limit or self.DEFAULT_LIMIT
        max_limit = max_limit or self.MAX_LIMIT

        limit = get_int_param(query_params, 'limit', default_limit)
        offset = get_int_param(query_params, 'offset', self.DEFAULT_OFFSET)

        # Clamp values
        limit = max(1, min(limit, max_limit))
        offset = max(0, offset)

        return limit, offset

    def paginated_response(
        self,
        items: list,
        total: int,
        limit: int,
        offset: int,
        items_key: str = "items",
    ) -> "HandlerResult":
        """Create a standardized paginated response.

        Args:
            items: List of items for this page
            total: Total count of all items
            limit: Page size used
            offset: Starting offset
            items_key: Key name for items in response (default: "items")

        Returns:
            JSON response with pagination metadata
        """
        return json_response({
            items_key: items,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(items) < total,
        })


class CachedHandlerMixin:
    """Mixin for cached response generation.

    Provides a simple interface for caching handler responses with TTL.

    Usage:
        class MyHandler(BaseHandler, CachedHandlerMixin):
            def _get_data(self, key: str):
                return self.cached_response(
                    cache_key=f"mydata:{key}",
                    ttl_seconds=300,
                    generator=lambda: expensive_computation(key),
                )
    """

    def cached_response(
        self,
        cache_key: str,
        ttl_seconds: float,
        generator: Callable[[], Any],
    ) -> Any:
        """Get or generate a cached response.

        Args:
            cache_key: Unique key for this cached item
            ttl_seconds: How long to cache the result
            generator: Callable that generates the value if not cached

        Returns:
            Cached or freshly generated value
        """
        cache = get_handler_cache()
        hit, cached_value = cache.get(cache_key, ttl_seconds)

        if hit:
            return cached_value

        value = generator()
        cache.set(cache_key, value)
        return value

    async def async_cached_response(
        self,
        cache_key: str,
        ttl_seconds: float,
        generator: Callable[[], Any],
    ) -> Any:
        """Async version of cached_response.

        Args:
            cache_key: Unique key for this cached item
            ttl_seconds: How long to cache the result
            generator: Async callable that generates the value if not cached

        Returns:
            Cached or freshly generated value
        """
        cache = get_handler_cache()
        hit, cached_value = cache.get(cache_key, ttl_seconds)

        if hit:
            return cached_value

        value = await generator()
        cache.set(cache_key, value)
        return value


class AuthenticatedHandlerMixin:
    """Mixin for requiring authenticated access.

    Provides standardized authentication extraction and error handling.

    Usage:
        class MyHandler(BaseHandler, AuthenticatedHandlerMixin):
            def handle_post(self, path, query_params, handler):
                user = self.require_auth(handler)
                if isinstance(user, tuple):  # Error response
                    return user
                # user is now the authenticated context
                return json_response({"user_id": user.user_id})
    """

    def require_auth(self, handler) -> Any:
        """Require authentication and return user context or error.

        Args:
            handler: HTTP request handler with headers

        Returns:
            UserAuthContext if authenticated,
            or HandlerResult with 401 error if not
        """
        # This method is typically overridden or uses BaseHandler's method
        # When used with BaseHandler, call require_auth_or_error instead
        if hasattr(self, 'require_auth_or_error'):
            user, err = self.require_auth_or_error(handler)
            if err:
                return err
            return user

        # Fallback implementation
        from aragora.billing.jwt_auth import extract_user_from_request
        user_store = getattr(self, 'ctx', {}).get('user_store')
        if hasattr(self.__class__, 'user_store'):
            user_store = self.__class__.user_store

        user_ctx = extract_user_from_request(handler, user_store)
        if not user_ctx.is_authenticated:
            return error_response("Authentication required", 401)
        return user_ctx


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
        # Check class attribute first (set by unified_server), then ctx
        if hasattr(self.__class__, 'elo_system') and self.__class__.elo_system is not None:
            return self.__class__.elo_system
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

    def get_current_user(self, handler: Any) -> Optional[Any]:
        """Get authenticated user from request, if any.

        Unlike @require_user_auth decorator which requires authentication,
        this method allows optional authentication - returning None if
        no valid auth is provided. Useful for endpoints that work for
        anonymous users but have enhanced features when authenticated.

        Args:
            handler: HTTP request handler with headers

        Returns:
            UserAuthContext if authenticated, None otherwise

        Example:
            def handle(self, path, query_params, handler):
                user = self.get_current_user(handler)
                if user:
                    # Show personalized content
                    return json_response({"user": user.email, "debates": ...})
                else:
                    # Show public content
                    return json_response({"debates": ...})
        """
        from aragora.billing.jwt_auth import extract_user_from_request

        user_store = None
        if hasattr(handler, 'user_store'):
            user_store = handler.user_store
        elif hasattr(self.__class__, 'user_store'):
            user_store = self.__class__.user_store

        user_ctx = extract_user_from_request(handler, user_store)
        return user_ctx if user_ctx.is_authenticated else None

    def require_auth_or_error(self, handler: Any) -> Tuple[Optional[Any], Optional["HandlerResult"]]:
        """Require authentication and return user or error response.

        Alternative to @require_user_auth decorator for cases where you need
        the user context inline without using a decorator.

        Args:
            handler: HTTP request handler with headers

        Returns:
            Tuple of (UserAuthContext, None) if authenticated,
            or (None, HandlerResult) with 401 error if not

        Example:
            def handle_post(self, path, query_params, handler):
                user, err = self.require_auth_or_error(handler)
                if err:
                    return err
                # user is now guaranteed to be authenticated
                return json_response({"created_by": user.user_id})
        """
        user = self.get_current_user(handler)
        if user is None:
            return None, error_response("Authentication required", 401)
        return user, None

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

    def handle(self, path: str, query_params: dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """
        Handle a GET request. Override in subclasses.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None

    def handle_post(self, path: str, query_params: dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """
        Handle a POST request. Override in subclasses that support POST.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None

    def handle_delete(self, path: str, query_params: dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """
        Handle a DELETE request. Override in subclasses that support DELETE.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None

    def handle_patch(self, path: str, query_params: dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """
        Handle a PATCH request. Override in subclasses that support PATCH.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None

    def handle_put(self, path: str, query_params: dict[str, Any], handler: Any) -> Optional[HandlerResult]:
        """
        Handle a PUT request. Override in subclasses that support PUT.

        Args:
            path: The request path
            query_params: Parsed query parameters
            handler: HTTP request handler for accessing request context

        Returns:
            HandlerResult if handled, None if not handled by this handler
        """
        return None
