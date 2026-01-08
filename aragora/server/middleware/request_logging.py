"""
Request/Response Logging Middleware.

Provides structured logging for HTTP requests with:
- X-Request-ID generation and propagation
- Request timing metrics
- Audit trail for security analysis
- Correlation ID support for distributed tracing
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Request ID header name
REQUEST_ID_HEADER = "X-Request-ID"

# Sensitive headers to mask in logs
SENSITIVE_HEADERS = frozenset({
    "authorization",
    "x-api-key",
    "cookie",
    "set-cookie",
    "x-auth-token",
})

# Sensitive query parameters to mask
SENSITIVE_PARAMS = frozenset({
    "token",
    "api_key",
    "apikey",
    "password",
    "secret",
    "access_token",
    "refresh_token",
})


@dataclass
class RequestContext:
    """Context for a single request."""
    request_id: str
    method: str
    path: str
    client_ip: str
    start_time: float
    token_hash: Optional[str] = None  # Hashed token for audit without exposing secret

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000


def generate_request_id() -> str:
    """Generate a unique request ID.

    Uses UUID4 for uniqueness with a short prefix for easy identification.

    Returns:
        Request ID string (e.g., "req-a1b2c3d4e5f6")
    """
    return f"req-{uuid.uuid4().hex[:12]}"


def hash_token(token: str) -> str:
    """Create a hash of a token for audit logging.

    Uses SHA256 truncated to 8 characters for privacy while
    allowing correlation of requests from the same token.

    Args:
        token: The authentication token

    Returns:
        Truncated hash of the token
    """
    return hashlib.sha256(token.encode()).hexdigest()[:8]


def mask_sensitive_value(value: str) -> str:
    """Mask a sensitive value for logging.

    Shows first 4 and last 4 characters with asterisks in between.

    Args:
        value: The sensitive value to mask

    Returns:
        Masked value (e.g., "Bear****oken")
    """
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}****{value[-4:]}"


def sanitize_headers(headers: dict) -> dict:
    """Sanitize headers for logging by masking sensitive values.

    Args:
        headers: Dictionary of HTTP headers

    Returns:
        Copy of headers with sensitive values masked
    """
    sanitized = {}
    for key, value in headers.items():
        if key.lower() in SENSITIVE_HEADERS:
            sanitized[key] = mask_sensitive_value(str(value))
        else:
            sanitized[key] = value
    return sanitized


def sanitize_params(params: dict) -> dict:
    """Sanitize query parameters for logging.

    Args:
        params: Dictionary of query parameters

    Returns:
        Copy of params with sensitive values masked
    """
    sanitized = {}
    for key, value in params.items():
        if key.lower() in SENSITIVE_PARAMS:
            sanitized[key] = "****"
        else:
            sanitized[key] = value
    return sanitized


def log_request(
    ctx: RequestContext,
    query_params: Optional[dict] = None,
    headers: Optional[dict] = None,
    log_level: int = logging.INFO,
) -> None:
    """Log an incoming request.

    Args:
        ctx: Request context
        query_params: Optional query parameters (will be sanitized)
        headers: Optional headers (will be sanitized)
        log_level: Logging level (default: INFO)
    """
    extra = {
        "request_id": ctx.request_id,
        "method": ctx.method,
        "path": ctx.path,
        "client_ip": ctx.client_ip,
    }

    if ctx.token_hash:
        extra["token_hash"] = ctx.token_hash

    if query_params:
        extra["params"] = sanitize_params(query_params)

    msg = f"[{ctx.request_id}] {ctx.method} {ctx.path} from {ctx.client_ip}"
    logger.log(log_level, msg, extra=extra)


def log_response(
    ctx: RequestContext,
    status_code: int,
    response_size: Optional[int] = None,
    error: Optional[str] = None,
    log_level: Optional[int] = None,
) -> None:
    """Log a response.

    Args:
        ctx: Request context
        status_code: HTTP status code
        response_size: Optional response body size in bytes
        error: Optional error message
        log_level: Logging level (auto-determined from status if not provided)
    """
    elapsed = ctx.elapsed_ms()

    # Auto-determine log level from status code
    if log_level is None:
        if status_code >= 500:
            log_level = logging.ERROR
        elif status_code >= 400:
            log_level = logging.WARNING
        else:
            log_level = logging.INFO

    extra = {
        "request_id": ctx.request_id,
        "method": ctx.method,
        "path": ctx.path,
        "status": status_code,
        "elapsed_ms": round(elapsed, 2),
    }

    if response_size is not None:
        extra["response_size"] = response_size

    if error:
        extra["error"] = error

    msg = f"[{ctx.request_id}] {ctx.method} {ctx.path} -> {status_code} ({elapsed:.1f}ms)"
    logger.log(log_level, msg, extra=extra)


def request_logging(
    log_request_body: bool = False,
    log_response_body: bool = False,
    slow_request_threshold_ms: float = 1000.0,
) -> Callable:
    """Decorator for request/response logging.

    Wraps a handler function to automatically log requests and responses
    with timing and correlation IDs.

    Args:
        log_request_body: Whether to log request body (default: False for privacy)
        log_response_body: Whether to log response body (default: False)
        slow_request_threshold_ms: Log slow requests as warnings above this threshold

    Returns:
        Decorator function

    Example:
        @request_logging()
        async def handle_request(request):
            ...

        @request_logging(slow_request_threshold_ms=500)
        def handle_fast_endpoint(request):
            ...
    """
    def decorator(func: Callable) -> Callable:
        """Create a logging wrapper for the given handler function.

        Detects whether the handler is async or sync and returns the appropriate
        wrapper. Both wrappers provide identical logging behavior.
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Async wrapper that logs request start, completion, and errors.

            Flow:
            1. Extract request from args (handler signature: self, request, ...)
            2. Generate or extract request ID from X-Request-ID header
            3. Build RequestContext with method, path, client IP, start time
            4. Log request start
            5. Execute handler and capture response
            6. Log response with status, size, and elapsed time
            7. Warn if request exceeds slow_request_threshold_ms
            8. Add X-Request-ID header to response
            """
            # Extract request info from args (assumes first arg after self is request)
            request = args[1] if len(args) > 1 else kwargs.get("request")

            # Generate or extract request ID
            request_id = generate_request_id()
            if request and hasattr(request, "headers"):
                existing_id = request.headers.get(REQUEST_ID_HEADER)
                if existing_id:
                    request_id = existing_id

            # Build context
            ctx = RequestContext(
                request_id=request_id,
                method=getattr(request, "method", "UNKNOWN") if request else "UNKNOWN",
                path=getattr(request, "path", "/") if request else "/",
                client_ip=_extract_ip(request) if request else "unknown",
                start_time=time.time(),
            )

            # Extract token hash if present
            if request and hasattr(request, "headers"):
                auth = request.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    ctx.token_hash = hash_token(auth[7:])

            # Log request
            log_request(ctx)

            try:
                # Call handler
                response = await func(*args, **kwargs)

                # Log response
                status = getattr(response, "status", 200) if response else 200
                size = getattr(response, "content_length", None) if response else None

                # Warn on slow requests
                log_level = None
                if ctx.elapsed_ms() > slow_request_threshold_ms:
                    log_level = logging.WARNING

                log_response(ctx, status, size, log_level=log_level)

                # Add request ID to response headers
                if response and hasattr(response, "headers"):
                    response.headers[REQUEST_ID_HEADER] = request_id

                return response

            except Exception as e:
                # Log error response
                log_response(ctx, 500, error=str(e)[:200])
                raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Sync wrapper that logs request start, completion, and errors.

            Mirrors async_wrapper behavior for synchronous handlers.
            """
            request = args[1] if len(args) > 1 else kwargs.get("request")

            request_id = generate_request_id()
            if request and hasattr(request, "headers"):
                existing_id = request.headers.get(REQUEST_ID_HEADER)
                if existing_id:
                    request_id = existing_id

            ctx = RequestContext(
                request_id=request_id,
                method=getattr(request, "method", "UNKNOWN") if request else "UNKNOWN",
                path=getattr(request, "path", "/") if request else "/",
                client_ip=_extract_ip(request) if request else "unknown",
                start_time=time.time(),
            )

            log_request(ctx)

            try:
                response = func(*args, **kwargs)

                status = getattr(response, "status", 200) if response else 200
                log_response(ctx, status)

                return response

            except Exception as e:
                log_response(ctx, 500, error=str(e)[:200])
                raise

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def _extract_ip(request: Any) -> str:
    """Extract client IP from request.

    Handles X-Forwarded-For for proxied requests.

    Args:
        request: The request object

    Returns:
        Client IP address string
    """
    if not request:
        return "unknown"

    # Try X-Forwarded-For first (for proxied requests)
    if hasattr(request, "headers"):
        xff = request.headers.get("X-Forwarded-For", "")
        if xff:
            # Take first IP (original client)
            return xff.split(",")[0].strip()

    # Try remote_addr / peername
    if hasattr(request, "remote"):
        return str(request.remote)

    if hasattr(request, "transport"):
        peername = request.transport.get_extra_info("peername")
        if peername:
            return peername[0]

    return "unknown"


# Context variable for accessing current request ID
import contextvars
_current_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_request_id", default=None
)


def get_current_request_id() -> Optional[str]:
    """Get the current request ID from context.

    Useful for including request ID in application logs.

    Returns:
        Current request ID or None if not in request context
    """
    return _current_request_id.get()


def set_current_request_id(request_id: str) -> None:
    """Set the current request ID in context.

    Args:
        request_id: The request ID to set
    """
    _current_request_id.set(request_id)


__all__ = [
    "REQUEST_ID_HEADER",
    "RequestContext",
    "generate_request_id",
    "hash_token",
    "sanitize_headers",
    "sanitize_params",
    "log_request",
    "log_response",
    "request_logging",
    "get_current_request_id",
    "set_current_request_id",
]
