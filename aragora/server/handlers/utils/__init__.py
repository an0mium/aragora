"""Handler utilities module.

Provides reusable utilities for HTTP handlers including:
- Rate limiting (token bucket algorithm)
- Query parameter extraction and validation
- URL routing and pattern matching
- Database connection helpers
- Safe data access utilities
- Handler decorators (auth, validation, error handling)
"""

from .rate_limit import RateLimiter, rate_limit, get_client_ip
from .params import (
    parse_query_params,
    get_int_param,
    get_float_param,
    get_bool_param,
    get_string_param,
    get_clamped_int_param,
    get_bounded_float_param,
    get_bounded_string_param,
)
from .routing import PathMatcher, RouteDispatcher
from .database import get_db_connection, table_exists
from .safe_data import safe_get, safe_get_nested, safe_json_parse
from .responses import (
    HandlerResult,
    json_response,
    error_response,
    html_response,
    redirect_response,
)
from .decorators import (
    generate_trace_id,
    map_exception_to_status,
    validate_params,
    handle_errors,
    auto_error_response,
    log_request,
    PERMISSION_MATRIX,
    has_permission,
    require_permission,
    require_user_auth,
    require_auth,
    require_storage,
    require_feature,
    safe_fetch,
    with_error_recovery,
)

__all__ = [
    # Rate limiting
    "RateLimiter",
    "rate_limit",
    "get_client_ip",
    # Parameter extraction
    "parse_query_params",
    "get_int_param",
    "get_float_param",
    "get_bool_param",
    "get_string_param",
    "get_clamped_int_param",
    "get_bounded_float_param",
    "get_bounded_string_param",
    # Routing
    "PathMatcher",
    "RouteDispatcher",
    # Database
    "get_db_connection",
    "table_exists",
    # Safe data access
    "safe_get",
    "safe_get_nested",
    "safe_json_parse",
    # Response builders
    "HandlerResult",
    "json_response",
    "error_response",
    "html_response",
    "redirect_response",
    # Decorators
    "generate_trace_id",
    "map_exception_to_status",
    "validate_params",
    "handle_errors",
    "auto_error_response",
    "log_request",
    "PERMISSION_MATRIX",
    "has_permission",
    "require_permission",
    "require_user_auth",
    "require_auth",
    "require_storage",
    "require_feature",
    "safe_fetch",
    "with_error_recovery",
]
