"""
Standard error codes for Aragora API responses.

This module defines a consistent set of error codes that should be used
across all API handlers. Using standard codes improves:
- Client error handling (can switch on error codes)
- API documentation clarity
- Debugging and log analysis
- Localization of error messages

Usage:
    from aragora.server.error_codes import ErrorCode
    from aragora.server.handlers.utils.responses import error_response

    return error_response(
        "Invalid email format",
        status=400,
        code=ErrorCode.VALIDATION_ERROR,
        details={"field": "email"}
    )
"""

from __future__ import annotations


class ErrorCode:
    """
    Standard error codes for API responses.

    Error codes are uppercase strings with underscores, grouped by category.
    Use these codes in error_response() for consistent error handling.
    """

    # =========================================================================
    # Authentication Errors (401)
    # =========================================================================

    AUTH_REQUIRED = "AUTH_REQUIRED"
    """Authentication is required but no credentials were provided."""

    AUTH_INVALID_TOKEN = "AUTH_INVALID_TOKEN"
    """The provided authentication token is invalid or malformed."""

    AUTH_EXPIRED = "AUTH_EXPIRED"
    """The authentication token has expired."""

    AUTH_INVALID_CREDENTIALS = "AUTH_INVALID_CREDENTIALS"
    """The provided credentials (email/password) are incorrect."""

    AUTH_ACCOUNT_LOCKED = "AUTH_ACCOUNT_LOCKED"
    """The account is temporarily locked due to too many failed attempts."""

    AUTH_MFA_REQUIRED = "AUTH_MFA_REQUIRED"
    """Multi-factor authentication is required for this account."""

    AUTH_MFA_INVALID = "AUTH_MFA_INVALID"
    """The provided MFA code is invalid or expired."""

    # =========================================================================
    # Authorization Errors (403)
    # =========================================================================

    FORBIDDEN = "FORBIDDEN"
    """The user does not have permission for this action."""

    PERMISSION_DENIED = "PERMISSION_DENIED"
    """The user lacks the required permissions for this resource."""

    ORG_ACCESS_DENIED = "ORG_ACCESS_DENIED"
    """The user is not a member of the required organization."""

    ROLE_INSUFFICIENT = "ROLE_INSUFFICIENT"
    """The user's role does not have sufficient privileges."""

    # =========================================================================
    # Validation Errors (400)
    # =========================================================================

    VALIDATION_ERROR = "VALIDATION_ERROR"
    """Request data failed validation (general validation failure)."""

    INVALID_REQUEST = "INVALID_REQUEST"
    """The request format is invalid or malformed."""

    INVALID_JSON = "INVALID_JSON"
    """The request body is not valid JSON."""

    MISSING_FIELD = "MISSING_FIELD"
    """A required field is missing from the request."""

    INVALID_FIELD = "INVALID_FIELD"
    """A field value is invalid (wrong type, format, etc.)."""

    INVALID_QUERY_PARAM = "INVALID_QUERY_PARAM"
    """A query parameter is invalid."""

    INVALID_PATH = "INVALID_PATH"
    """The request path is invalid or malformed."""

    # =========================================================================
    # Resource Errors (404, 409, 410)
    # =========================================================================

    NOT_FOUND = "NOT_FOUND"
    """The requested resource was not found."""

    DEBATE_NOT_FOUND = "DEBATE_NOT_FOUND"
    """The specified debate does not exist."""

    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    """The specified agent does not exist."""

    USER_NOT_FOUND = "USER_NOT_FOUND"
    """The specified user does not exist."""

    ORG_NOT_FOUND = "ORG_NOT_FOUND"
    """The specified organization does not exist."""

    CONFLICT = "CONFLICT"
    """The request conflicts with current state (e.g., duplicate creation)."""

    ALREADY_EXISTS = "ALREADY_EXISTS"
    """The resource already exists."""

    GONE = "GONE"
    """The resource existed but has been deleted."""

    # =========================================================================
    # Rate Limiting & Quota Errors (429)
    # =========================================================================

    RATE_LIMITED = "RATE_LIMITED"
    """Too many requests in a short time period."""

    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    """Monthly quota has been exceeded."""

    QUOTA_INSUFFICIENT = "QUOTA_INSUFFICIENT"
    """Not enough quota remaining for this operation."""

    # =========================================================================
    # Server Errors (500, 503)
    # =========================================================================

    INTERNAL_ERROR = "INTERNAL_ERROR"
    """An unexpected internal error occurred."""

    DATABASE_ERROR = "DATABASE_ERROR"
    """A database error occurred."""

    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    """An external service (API, etc.) failed."""

    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    """The service is temporarily unavailable."""

    FEATURE_UNAVAILABLE = "FEATURE_UNAVAILABLE"
    """The requested feature is not available or not configured."""

    MAINTENANCE = "MAINTENANCE"
    """The service is under maintenance."""

    # =========================================================================
    # Content Errors (413, 415, 422)
    # =========================================================================

    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    """The request body exceeds the size limit."""

    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    """The Content-Type is not supported."""

    UNPROCESSABLE_ENTITY = "UNPROCESSABLE_ENTITY"
    """The request was syntactically correct but semantically invalid."""


# HTTP status code mapping for common error codes
ERROR_CODE_STATUS_MAP: dict[str, int] = {
    # 400 Bad Request
    ErrorCode.VALIDATION_ERROR: 400,
    ErrorCode.INVALID_REQUEST: 400,
    ErrorCode.INVALID_JSON: 400,
    ErrorCode.MISSING_FIELD: 400,
    ErrorCode.INVALID_FIELD: 400,
    ErrorCode.INVALID_QUERY_PARAM: 400,
    ErrorCode.INVALID_PATH: 400,
    # 401 Unauthorized
    ErrorCode.AUTH_REQUIRED: 401,
    ErrorCode.AUTH_INVALID_TOKEN: 401,
    ErrorCode.AUTH_EXPIRED: 401,
    ErrorCode.AUTH_INVALID_CREDENTIALS: 401,
    ErrorCode.AUTH_ACCOUNT_LOCKED: 401,
    ErrorCode.AUTH_MFA_REQUIRED: 401,
    ErrorCode.AUTH_MFA_INVALID: 401,
    # 403 Forbidden
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.PERMISSION_DENIED: 403,
    ErrorCode.ORG_ACCESS_DENIED: 403,
    ErrorCode.ROLE_INSUFFICIENT: 403,
    # 404 Not Found
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.DEBATE_NOT_FOUND: 404,
    ErrorCode.AGENT_NOT_FOUND: 404,
    ErrorCode.USER_NOT_FOUND: 404,
    ErrorCode.ORG_NOT_FOUND: 404,
    # 409 Conflict
    ErrorCode.CONFLICT: 409,
    ErrorCode.ALREADY_EXISTS: 409,
    # 410 Gone
    ErrorCode.GONE: 410,
    # 413 Payload Too Large
    ErrorCode.PAYLOAD_TOO_LARGE: 413,
    # 415 Unsupported Media Type
    ErrorCode.UNSUPPORTED_MEDIA_TYPE: 415,
    # 422 Unprocessable Entity
    ErrorCode.UNPROCESSABLE_ENTITY: 422,
    # 429 Too Many Requests
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.QUOTA_EXCEEDED: 429,
    ErrorCode.QUOTA_INSUFFICIENT: 429,
    # 500 Internal Server Error
    ErrorCode.INTERNAL_ERROR: 500,
    ErrorCode.DATABASE_ERROR: 500,
    ErrorCode.EXTERNAL_SERVICE_ERROR: 500,
    # 503 Service Unavailable
    ErrorCode.SERVICE_UNAVAILABLE: 503,
    ErrorCode.FEATURE_UNAVAILABLE: 503,
    ErrorCode.MAINTENANCE: 503,
}


def get_status_for_code(code: str) -> int:
    """Get the recommended HTTP status code for an error code.

    Args:
        code: Error code string (e.g., ErrorCode.VALIDATION_ERROR)

    Returns:
        HTTP status code (defaults to 400 if code not found)
    """
    return ERROR_CODE_STATUS_MAP.get(code, 400)


__all__ = ["ErrorCode", "ERROR_CODE_STATUS_MAP", "get_status_for_code"]
