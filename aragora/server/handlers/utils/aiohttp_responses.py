"""
aiohttp-compatible response helpers.

Provides standardized error response formatting for handlers that use
aiohttp's web.Response instead of HandlerResult.

These helpers mirror the format of the HandlerResult-based error_response()
in responses.py, ensuring consistent error formats across both handler types.
"""

from __future__ import annotations

from typing import Any

from aiohttp import web


def web_error_response(
    message: str,
    status: int = 400,
    *,
    code: str | None = None,
    details: dict[str, Any] | None = None,
) -> web.Response:
    """Create a standardized aiohttp error response.

    Args:
        message: Human-readable error message
        status: HTTP status code (default: 400)
        code: Optional machine-readable error code (e.g., "VALIDATION_ERROR")
        details: Optional additional error details

    Returns:
        aiohttp web.Response with JSON error body

    Examples:
        # Simple error
        return web_error_response("Invalid input", 400)
        # -> {"error": "Invalid input"}

        # Structured error with code
        return web_error_response("Not found", 404, code="NOT_FOUND")
        # -> {"error": {"code": "NOT_FOUND", "message": "Not found"}}
    """
    if code or details:
        error_obj: dict[str, Any] = {"message": message}
        if code:
            error_obj["code"] = code
        if details:
            error_obj["details"] = details
        payload: dict[str, Any] = {"error": error_obj}
    else:
        payload = {"error": message}

    return web.json_response(payload, status=status)
