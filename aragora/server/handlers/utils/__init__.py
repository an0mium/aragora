"""Handler utilities module.

Provides reusable utilities for HTTP handlers including:
- Rate limiting (token bucket algorithm)
- Query parameter extraction and validation
- URL routing and pattern matching
- Database connection helpers
- Safe data access utilities
- Handler decorators (auth, validation, error handling)
"""

from .database import get_db_connection, table_exists
from .decorators import (
    PERMISSION_MATRIX,
    auto_error_response,
    generate_trace_id,
    handle_errors,
    has_permission,
    log_request,
    map_exception_to_status,
    require_auth,
    require_feature,
    require_permission,
    require_storage,
    require_user_auth,
    safe_fetch,
    validate_params,
    with_error_recovery,
)
from .params import (
    get_bool_param,
    get_bounded_float_param,
    get_bounded_string_param,
    get_clamped_int_param,
    get_float_param,
    get_int_param,
    get_string_param,
    parse_query_params,
)
from .rate_limit import RateLimiter, get_client_ip, rate_limit
from .responses import (
    HandlerResult,
    error_response,
    html_response,
    json_response,
    redirect_response,
)
from .routing import PathMatcher, RouteDispatcher
from .safe_data import safe_get, safe_get_nested, safe_json_parse

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
