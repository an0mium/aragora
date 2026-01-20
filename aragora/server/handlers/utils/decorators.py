"""
Handler decorators for authentication, validation, and error handling.

Provides reusable decorators for HTTP handlers including:
- Parameter validation (@validate_params)
- Error handling (@handle_errors, @auto_error_response)
- Request logging (@log_request)
- Authentication (@require_auth, @require_user_auth, @require_permission)
- Feature gating (@require_storage, @require_feature)
- Error recovery (@with_error_recovery)
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

from aragora.server.errors import safe_error_message
from aragora.server.handlers.utils.params import (
    get_bool_param,
    get_float_param,
    get_int_param,
    get_string_param,
)
from aragora.server.handlers.utils.responses import HandlerResult, error_response

logger = logging.getLogger(__name__)


# =============================================================================
# Trace ID Generation
# =============================================================================


def generate_trace_id() -> str:
    """Generate a unique trace ID for request tracking."""
    return str(uuid.uuid4())[:8]


# =============================================================================
# Exception Handling
# =============================================================================

# Exception to HTTP status code mapping
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
    # Handler-specific exceptions (from aragora.server.handlers.exceptions)
    "HandlerError": 500,
    "HandlerValidationError": 400,
    "HandlerNotFoundError": 404,
    "HandlerAuthorizationError": 403,
    "HandlerConflictError": 409,
    "HandlerRateLimitError": 429,
    "HandlerExternalServiceError": 502,
    "HandlerDatabaseError": 500,
}


def map_exception_to_status(e: Exception, default: int = 500) -> int:
    """Map exception type to appropriate HTTP status code."""
    error_type = type(e).__name__
    return _EXCEPTION_STATUS_MAP.get(error_type, default)


# =============================================================================
# Parameter Validation Decorator
# =============================================================================


def validate_params(
    param_specs: dict[str, tuple],
    query_params_arg: str = "query_params",
) -> Callable[[Callable], Callable]:
    """
    Decorator for automatic query parameter validation and extraction.

    Validates and extracts query parameters into typed function arguments,
    eliminating boilerplate validation code from handler methods.

    Args:
        param_specs: Dict mapping param names to (type, default, min, max) tuples.
                    Types supported: int, float, bool, str.
                    Use None for unbounded min/max.
        query_params_arg: Name of the query_params argument in the function.

    Returns:
        Decorator function that adds validated params to kwargs.
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
                    if idx < len(args):
                        params = args[idx]

            if params is None:
                params = {}

            # Validate and extract each parameter
            extracted = {}
            for name, spec in param_specs.items():
                param_type, default, min_val, max_val = spec

                if param_type is int:
                    val = get_int_param(params, name, default)
                    if min_val is not None:
                        val = max(val, min_val)
                    if max_val is not None:
                        val = min(val, max_val)
                elif param_type is float:
                    val = get_float_param(params, name, default)
                    if min_val is not None:
                        val = max(val, min_val)
                    if max_val is not None:
                        val = min(val, max_val)
                elif param_type is bool:
                    val = get_bool_param(params, name, default)
                elif param_type is str:
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


# =============================================================================
# Error Handling Decorators
# =============================================================================


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
                status = map_exception_to_status(e, default_status)
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
) -> Callable[[Callable], Callable]:
    """
    Decorator for automatic error response generation.

    Like @handle_errors but with configurable logging levels.

    Args:
        operation: Description of the operation being performed
        log_level: One of "error", "warning"
        include_traceback: Include stack trace in error log

    Returns:
        Decorator function that wraps handler methods.
    """
    import sqlite3

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

                # Extract status code from result (supports HandlerResult and dicts)
                status_code = getattr(result, "status_code", 200) if result else 200
                if isinstance(result, dict):
                    status_code = result.get("status", 200)

                log_msg = f"[{trace_id}] {context}: {status_code} in {duration_ms}ms"
                if status_code >= 400:
                    logger.warning(log_msg)
                else:
                    logger.info(log_msg)

                if log_response and result:
                    body = getattr(result, "body", b"")
                    if body and len(body) < 1000:  # Only log small responses
                        logger.debug(
                            f"[{trace_id}] Response: {body.decode('utf-8', errors='ignore')[:500]}"
                        )

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


# =============================================================================
# Permission / RBAC
# =============================================================================

# Role-Based Access Control permission matrix
# Permission -> list of roles that have access
# Role hierarchy: owner > admin > member (higher roles inherit lower permissions)
PERMISSION_MATRIX: dict[str, list[str]] = {
    # Debate permissions
    "debates:read": ["member", "admin", "owner"],
    "debates:create": ["member", "admin", "owner"],
    "debates:update": ["admin", "owner"],
    "debates:delete": ["admin", "owner"],
    "debates:export": ["member", "admin", "owner"],
    # Agent permissions
    "agents:read": ["member", "admin", "owner"],
    "agents:create": ["admin", "owner"],
    "agents:update": ["admin", "owner"],
    "agents:delete": ["admin", "owner"],
    # Organization permissions
    "org:read": ["member", "admin", "owner"],
    "org:settings": ["admin", "owner"],
    "org:members": ["admin", "owner"],
    "org:invite": ["admin", "owner"],
    "org:billing": ["owner"],
    "org:delete": ["owner"],
    # Plugin permissions
    "plugins:read": ["member", "admin", "owner"],
    "plugins:install": ["admin", "owner"],
    "plugins:configure": ["admin", "owner"],
    "plugins:uninstall": ["admin", "owner"],
    "plugins:run": ["member", "admin", "owner"],
    "plugins:execute": ["admin", "owner"],
    "plugins:manage": ["admin", "owner"],
    # Laboratory (experimental features)
    "laboratory:read": ["member", "admin", "owner"],
    "laboratory:execute": ["admin", "owner"],
    # Control Plane permissions
    "controlplane:read": ["member", "admin", "owner"],
    "controlplane:agents": ["admin", "owner"],
    "controlplane:tasks": ["admin", "owner"],
    "controlplane:manage": ["owner"],
    # Training permissions
    "training:read": ["member", "admin", "owner"],
    "training:create": ["admin", "owner"],
    "training:export": ["admin", "owner"],
    # ML permissions
    "ml:read": ["member", "admin", "owner"],
    "ml:train": ["admin", "owner"],
    "ml:deploy": ["admin", "owner"],
    "ml:delete": ["admin", "owner"],
    # Connector permissions
    "connectors:read": ["member", "admin", "owner"],
    "connectors:create": ["admin", "owner"],
    "connectors:delete": ["admin", "owner"],
    "connectors:configure": ["admin", "owner"],
    # Admin permissions
    "admin:*": ["owner"],
    "admin:audit": ["admin", "owner"],
    "admin:system": ["owner"],
    "admin:metrics": ["admin", "owner"],
    "admin:users": ["owner"],
    # API key management
    "apikeys:read": ["member", "admin", "owner"],
    "apikeys:create": ["member", "admin", "owner"],
    "apikeys:delete": ["member", "admin", "owner"],
    "apikeys:manage": ["admin", "owner"],
}


def has_permission(role: str, permission: str) -> bool:
    """
    Check if a role has a specific permission.

    Args:
        role: User's role (member, admin, owner)
        permission: Permission string (e.g., "debates:create")

    Returns:
        True if role has the permission, False otherwise
    """
    if not role or not permission:
        return False

    # Check exact permission
    allowed_roles = PERMISSION_MATRIX.get(permission, [])
    if role in allowed_roles:
        return True

    # Check wildcard permission (e.g., "admin:*")
    permission_category = permission.split(":")[0]
    wildcard = f"{permission_category}:*"
    wildcard_roles = PERMISSION_MATRIX.get(wildcard, [])
    if role in wildcard_roles:
        return True

    return False


def require_permission(permission: str) -> Callable[[Callable], Callable]:
    """
    Decorator that requires a specific permission.

    First authenticates the user, then checks if they have the required
    permission based on their role.

    Args:
        permission: Required permission (e.g., "debates:create")
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            from aragora.billing.jwt_auth import extract_user_from_request

            handler = kwargs.get("handler")
            if handler is None and args:
                for arg in args:
                    if hasattr(arg, "headers"):
                        handler = arg
                        break

            if handler is None:
                logger.warning(f"require_permission({permission}): No handler provided")
                return error_response("Authentication required", 401)

            user_store = None
            if hasattr(handler, "user_store"):
                user_store = handler.user_store
            elif hasattr(handler.__class__, "user_store"):
                user_store = handler.__class__.user_store

            user_ctx = extract_user_from_request(handler, user_store)

            if not user_ctx.is_authenticated:
                error_msg = user_ctx.error_reason or "Authentication required"
                return error_response(error_msg, 401)

            if not has_permission(user_ctx.role, permission):
                logger.warning(
                    f"Permission denied: user={user_ctx.user_id} role={user_ctx.role} "
                    f"permission={permission}"
                )
                return error_response(f"Permission denied: requires '{permission}'", 403)

            kwargs["user"] = user_ctx
            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Authentication Decorators
# =============================================================================


def require_user_auth(func: Callable) -> Callable:
    """
    Decorator that requires JWT/API key user authentication.

    Uses the billing JWT auth system to validate Bearer tokens and API keys.
    The authenticated user context is passed as 'user' keyword argument.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.billing.jwt_auth import extract_user_from_request

        handler = kwargs.get("handler")
        if handler is None and args:
            for arg in args:
                if hasattr(arg, "headers"):
                    handler = arg
                    break

        if handler is None:
            logger.warning("require_user_auth: No handler provided")
            return error_response("Authentication required", 401)

        user_store = None
        if hasattr(handler, "user_store"):
            user_store = handler.user_store
        elif hasattr(handler.__class__, "user_store"):
            user_store = handler.__class__.user_store

        user_ctx = extract_user_from_request(handler, user_store)

        if not user_ctx.is_authenticated:
            error_msg = user_ctx.error_reason or "Authentication required"
            return error_response(error_msg, 401)

        kwargs["user"] = user_ctx
        return func(*args, **kwargs)

    return wrapper


def require_auth(func: Callable) -> Callable:
    """
    Decorator that ALWAYS requires authentication via ARAGORA_API_TOKEN.

    Use this for sensitive endpoints that must never run without authentication.
    For JWT/API key authentication, use @require_user_auth instead.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        from aragora.server.auth import auth_config

        handler = kwargs.get("handler")
        if handler is None and args:
            for arg in args:
                if hasattr(arg, "headers"):
                    handler = arg
                    break

        if handler is None:
            logger.warning("require_auth: No handler provided, denying access")
            return error_response("Authentication required", 401)

        auth_header = None
        if hasattr(handler, "headers"):
            auth_header = handler.headers.get("Authorization", "")

        token = None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]

        if not auth_config.api_token:
            logger.warning("require_auth: No API token configured, denying access")
            return error_response(
                "Authentication required. Set ARAGORA_API_TOKEN environment variable.", 401
            )

        if not token or not auth_config.validate_token(token):
            return error_response("Invalid or missing authentication token", 401)

        return func(*args, **kwargs)

    return wrapper


# =============================================================================
# Feature Gating Decorators
# =============================================================================


def require_storage(func: Callable) -> Callable:
    """
    Decorator that requires storage to be available.

    Returns 503 Service Unavailable if storage is not configured.
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
) -> Callable[[Callable], Callable]:
    """
    Decorator that requires a feature to be available.

    Args:
        feature_check: Callable that returns True if feature is available
        feature_name: Human-readable name for error message
        status_code: HTTP status code to return if unavailable
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not feature_check():
                return error_response(f"{feature_name} not available", status_code)
            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Error Recovery
# =============================================================================


def safe_fetch(
    data_dict: dict,
    errors_dict: dict,
    key: str,
    fallback: Any,
    log_errors: bool = True,
):
    """
    Context manager for safe data fetching with graceful fallback.

    Usage:
        with safe_fetch(data, errors, "rankings", {"agents": [], "count": 0}):
            data["rankings"] = self._fetch_rankings(limit)
    """

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
) -> Callable[[Callable], Callable]:
    """
    Decorator for graceful error recovery with fallback values.

    Unlike handle_errors which returns HTTP error responses, this decorator
    returns a fallback value on error, allowing partial success.

    Args:
        fallback_value: Value to return on error
        log_errors: Whether to log errors
        metrics_key: Optional key for metrics tracking
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(
                        f"with_error_recovery '{func.__name__}' failed: {type(e).__name__}: {e}"
                    )
                return fallback_value

        return wrapper

    return decorator


# =============================================================================
# Deprecation Decorator
# =============================================================================


def deprecated_endpoint(
    replacement: Optional[str] = None,
    sunset_date: Optional[str] = None,
    message: Optional[str] = None,
) -> Callable[[Callable], Callable]:
    """
    Decorator for marking endpoints as deprecated.

    Adds RFC 8594 deprecation headers to responses and logs usage warnings.
    Use this on endpoints scheduled for removal.

    Args:
        replacement: URL path of the replacement endpoint (e.g., "/api/v2/debates")
        sunset_date: ISO 8601 date when endpoint will be removed (e.g., "2025-06-01")
        message: Custom deprecation message for logging

    Returns:
        Decorated function that adds deprecation headers to responses.

    Example:
        @deprecated_endpoint(replacement="/api/debates", sunset_date="2025-06-01")
        def _create_debate_legacy(self, handler, user=None):
            ...

    Response Headers Added:
        - Deprecation: true
        - Sunset: Sat, 01 Jun 2025 00:00:00 GMT (if sunset_date provided)
        - Link: </api/debates>; rel="successor-version" (if replacement provided)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log deprecation warning
            endpoint_name = func.__name__
            log_msg = message or f"Deprecated endpoint used: {endpoint_name}"
            if replacement:
                log_msg += f". Use {replacement} instead."
            logger.warning(log_msg)

            # Execute the handler
            result = func(*args, **kwargs)

            # Add deprecation headers to result
            if result is not None and isinstance(result, dict):
                # Get or create headers dict
                headers = result.get("headers", {})

                # Add Deprecation header (RFC 8594)
                headers["Deprecation"] = "true"

                # Add Sunset header if date provided
                if sunset_date:
                    try:
                        from datetime import datetime

                        # Parse ISO date and format as HTTP-date
                        dt = datetime.fromisoformat(sunset_date)
                        # Format: Sat, 01 Jun 2025 00:00:00 GMT
                        headers["Sunset"] = dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
                    except ValueError:
                        logger.warning(f"Invalid sunset_date format: {sunset_date}")

                # Add Link header for replacement
                if replacement:
                    headers["Link"] = f'<{replacement}>; rel="successor-version"'

                result["headers"] = headers

            return result

        return wrapper

    return decorator


__all__ = [
    # Trace ID
    "generate_trace_id",
    # Exception handling
    "map_exception_to_status",
    # Parameter validation
    "validate_params",
    # Error handling decorators
    "handle_errors",
    "auto_error_response",
    "log_request",
    # Permission/RBAC
    "PERMISSION_MATRIX",
    "has_permission",
    "require_permission",
    # Authentication
    "require_user_auth",
    "require_auth",
    # Feature gating
    "require_storage",
    "require_feature",
    # Error recovery
    "safe_fetch",
    "with_error_recovery",
    # Deprecation
    "deprecated_endpoint",
]
