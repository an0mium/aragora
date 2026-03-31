"""Handler utilities module.

Keeps package import side effects minimal so callers can import specific utility
submodules without eagerly loading auth, RBAC, or other heavyweight graphs.
Public exports remain available via module-level lazy loading.
"""

from __future__ import annotations

import importlib
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    # Database
    "get_db_connection": ("aragora.server.handlers.utils.database", "get_db_connection"),
    "table_exists": ("aragora.server.handlers.utils.database", "table_exists"),
    # Lazy stores
    "LazyStoreFactory": ("aragora.server.handlers.utils.lazy_stores", "LazyStoreFactory"),
    "LazyStoreRegistry": ("aragora.server.handlers.utils.lazy_stores", "LazyStoreRegistry"),
    # Safe fetch
    "safe_fetch_with_fallback": ("aragora.server.handlers.utils.safe_fetch", "safe_fetch"),
    "safe_fetch_async": ("aragora.server.handlers.utils.safe_fetch", "safe_fetch_async"),
    "SafeFetchContext": ("aragora.server.handlers.utils.safe_fetch", "SafeFetchContext"),
    "fetch_multiple": ("aragora.server.handlers.utils.safe_fetch", "fetch_multiple"),
    "fetch_multiple_async": ("aragora.server.handlers.utils.safe_fetch", "fetch_multiple_async"),
    # Auth helpers
    "ForbiddenError": ("aragora.server.handlers.utils.auth", "ForbiddenError"),
    "UnauthorizedError": ("aragora.server.handlers.utils.auth", "UnauthorizedError"),
    "get_auth_context": ("aragora.server.handlers.utils.auth", "get_auth_context"),
    "require_authenticated": (
        "aragora.server.handlers.utils.auth",
        "require_authenticated",
    ),
    # Auth mixins
    "SecureEndpointMixin": (
        "aragora.server.handlers.utils.auth_mixins",
        "SecureEndpointMixin",
    ),
    "AuthenticatedHandlerMixin": (
        "aragora.server.handlers.utils.auth_mixins",
        "AuthenticatedHandlerMixin",
    ),
    "require_permission_mixin": (
        "aragora.server.handlers.utils.auth_mixins",
        "require_permission",
    ),
    "require_any_permission": (
        "aragora.server.handlers.utils.auth_mixins",
        "require_any_permission",
    ),
    "require_all_permissions": (
        "aragora.server.handlers.utils.auth_mixins",
        "require_all_permissions",
    ),
    # Decorators
    "PERMISSION_MATRIX": ("aragora.server.handlers.utils.decorators", "PERMISSION_MATRIX"),
    "auto_error_response": (
        "aragora.server.handlers.utils.decorators",
        "auto_error_response",
    ),
    "generate_trace_id": ("aragora.server.handlers.utils.decorators", "generate_trace_id"),
    "handle_errors": ("aragora.server.handlers.utils.decorators", "handle_errors"),
    "has_permission": ("aragora.server.handlers.utils.decorators", "has_permission"),
    "log_request": ("aragora.server.handlers.utils.decorators", "log_request"),
    "map_exception_to_status": (
        "aragora.server.handlers.utils.decorators",
        "map_exception_to_status",
    ),
    "require_auth": ("aragora.server.handlers.utils.decorators", "require_auth"),
    "require_feature": ("aragora.server.handlers.utils.decorators", "require_feature"),
    "require_permission": (
        "aragora.server.handlers.utils.decorators",
        "require_permission",
    ),
    "require_storage": ("aragora.server.handlers.utils.decorators", "require_storage"),
    "require_user_auth": (
        "aragora.server.handlers.utils.decorators",
        "require_user_auth",
    ),
    "safe_fetch": ("aragora.server.handlers.utils.decorators", "safe_fetch"),
    "validate_params": ("aragora.server.handlers.utils.decorators", "validate_params"),
    "with_error_recovery": (
        "aragora.server.handlers.utils.decorators",
        "with_error_recovery",
    ),
    # Params
    "parse_query_params": ("aragora.server.handlers.utils.params", "parse_query_params"),
    "get_int_param": ("aragora.server.handlers.utils.params", "get_int_param"),
    "get_float_param": ("aragora.server.handlers.utils.params", "get_float_param"),
    "get_bool_param": ("aragora.server.handlers.utils.params", "get_bool_param"),
    "get_string_param": ("aragora.server.handlers.utils.params", "get_string_param"),
    "get_clamped_int_param": (
        "aragora.server.handlers.utils.params",
        "get_clamped_int_param",
    ),
    "get_bounded_float_param": (
        "aragora.server.handlers.utils.params",
        "get_bounded_float_param",
    ),
    "get_bounded_string_param": (
        "aragora.server.handlers.utils.params",
        "get_bounded_string_param",
    ),
    # Rate limiting
    "RateLimiter": ("aragora.server.handlers.utils.rate_limit", "RateLimiter"),
    "rate_limit": ("aragora.server.handlers.utils.rate_limit", "rate_limit"),
    "get_client_ip": ("aragora.server.handlers.utils.rate_limit", "get_client_ip"),
    # Responses
    "web_error_response": (
        "aragora.server.handlers.utils.aiohttp_responses",
        "web_error_response",
    ),
    "HandlerResult": ("aragora.server.handlers.utils.responses", "HandlerResult"),
    "json_response": ("aragora.server.handlers.utils.responses", "json_response"),
    "error_response": ("aragora.server.handlers.utils.responses", "error_response"),
    "html_response": ("aragora.server.handlers.utils.responses", "html_response"),
    "redirect_response": ("aragora.server.handlers.utils.responses", "redirect_response"),
    "paginated_response": (
        "aragora.server.handlers.utils.responses",
        "paginated_response",
    ),
    "parse_pagination_params": (
        "aragora.server.handlers.utils.responses",
        "parse_pagination_params",
    ),
    "normalize_pagination_response": (
        "aragora.server.handlers.utils.responses",
        "normalize_pagination_response",
    ),
    # Routing and data helpers
    "PathMatcher": ("aragora.server.handlers.utils.routing", "PathMatcher"),
    "RouteDispatcher": ("aragora.server.handlers.utils.routing", "RouteDispatcher"),
    "safe_get": ("aragora.server.handlers.utils.safe_data", "safe_get"),
    "safe_get_nested": ("aragora.server.handlers.utils.safe_data", "safe_get_nested"),
    "safe_json_parse": ("aragora.server.handlers.utils.safe_data", "safe_json_parse"),
    "parse_json_body": ("aragora.server.handlers.utils.json_body", "parse_json_body"),
    "parse_json_body_allow_array": (
        "aragora.server.handlers.utils.json_body",
        "parse_json_body_allow_array",
    ),
    # Sanitization
    "RESPONSE_SENSITIVE_FIELDS": (
        "aragora.server.handlers.utils.sanitization",
        "RESPONSE_SENSITIVE_FIELDS",
    ),
    "sanitize_response": ("aragora.server.handlers.utils.sanitization", "sanitize_response"),
    "sanitize_user_response": (
        "aragora.server.handlers.utils.sanitization",
        "sanitize_user_response",
    ),
    "sanitize_integration_response": (
        "aragora.server.handlers.utils.sanitization",
        "sanitize_integration_response",
    ),
    "sanitize_payment_response": (
        "aragora.server.handlers.utils.sanitization",
        "sanitize_payment_response",
    ),
    "sanitize_output": ("aragora.server.handlers.utils.sanitization", "sanitize_output"),
    # RBAC guard
    "rbac_available": ("aragora.server.handlers.utils.rbac_guard", "rbac_available"),
    "rbac_fail_closed": ("aragora.server.handlers.utils.rbac_guard", "rbac_fail_closed"),
    "is_production_env": (
        "aragora.server.handlers.utils.rbac_guard",
        "is_production_env",
    ),
    # File validation
    "validate_file_upload": (
        "aragora.server.handlers.utils.file_validation",
        "validate_file_upload",
    ),
    "validate_file_size": (
        "aragora.server.handlers.utils.file_validation",
        "validate_file_size",
    ),
    "validate_mime_type": (
        "aragora.server.handlers.utils.file_validation",
        "validate_mime_type",
    ),
    "validate_extension": (
        "aragora.server.handlers.utils.file_validation",
        "validate_extension",
    ),
    "validate_filename_security": (
        "aragora.server.handlers.utils.file_validation",
        "validate_filename_security",
    ),
    "sanitize_filename": (
        "aragora.server.handlers.utils.file_validation",
        "sanitize_filename",
    ),
    "get_max_file_size": (
        "aragora.server.handlers.utils.file_validation",
        "get_max_file_size",
    ),
    "get_max_file_size_mb": (
        "aragora.server.handlers.utils.file_validation",
        "get_max_file_size_mb",
    ),
    "FileValidationResult": (
        "aragora.server.handlers.utils.file_validation",
        "FileValidationResult",
    ),
    "FileValidationError": (
        "aragora.server.handlers.utils.file_validation",
        "FileValidationError",
    ),
    "FileValidationErrorCode": (
        "aragora.server.handlers.utils.file_validation",
        "FileValidationErrorCode",
    ),
    "ALLOWED_MIME_TYPES": (
        "aragora.server.handlers.utils.file_validation",
        "ALLOWED_MIME_TYPES",
    ),
    "ALLOWED_EXTENSIONS": (
        "aragora.server.handlers.utils.file_validation",
        "ALLOWED_EXTENSIONS",
    ),
    "MAX_FILE_SIZE": ("aragora.server.handlers.utils.file_validation", "MAX_FILE_SIZE"),
    "MAX_FILENAME_LENGTH": (
        "aragora.server.handlers.utils.file_validation",
        "MAX_FILENAME_LENGTH",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_path)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
