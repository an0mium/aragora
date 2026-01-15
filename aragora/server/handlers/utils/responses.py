"""
Response builders for HTTP handlers.

Provides standardized JSON response formatting with consistent error handling.

Functions:
    json_response: Create a JSON response with any data
    error_response: Create a structured error response

Classes:
    HandlerResult: Dataclass representing an HTTP response
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class HandlerResult:
    """Result of handling an HTTP request.

    Attributes:
        status_code: HTTP status code (200, 400, 500, etc.)
        content_type: MIME type of the response body
        body: Raw bytes of the response body
        headers: Optional HTTP headers to include
    """

    status_code: int
    content_type: str
    body: bytes
    headers: dict | None = None

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = {}


def json_response(
    data: Any,
    status: int = 200,
    headers: Optional[dict] = None,
) -> HandlerResult:
    """Create a JSON response.

    Args:
        data: Any JSON-serializable data (uses str() for non-serializable types)
        status: HTTP status code (default: 200)
        headers: Optional additional headers

    Returns:
        HandlerResult with JSON body and application/json content type

    Example:
        return json_response({"items": items, "count": len(items)})
        return json_response({"created": True}, status=201)
    """
    body = json.dumps(data, default=str).encode("utf-8")
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
    *,
    code: Optional[str] = None,
    trace_id: Optional[str] = None,
    suggestion: Optional[str] = None,
    details: Optional[dict] = None,
    structured: bool = False,
) -> HandlerResult:
    """Create an error response.

    Args:
        message: Human-readable error message
        status: HTTP status code (default: 400)
        headers: Optional additional headers
        code: Optional machine-readable error code (e.g., "VALIDATION_ERROR")
        trace_id: Optional request trace ID for debugging
        suggestion: Optional suggestion for resolving the error
        details: Optional additional error details
        structured: If True, always use structured format. If False,
                   only include additional fields if provided.

    Returns:
        HandlerResult with error JSON

    Examples:
        # Simple error (backward compatible)
        error_response("Invalid input", 400)
        # -> {"error": "Invalid input"}

        # Structured error with code and suggestion
        error_response(
            "Field 'name' is required",
            400,
            code="VALIDATION_ERROR",
            suggestion="Include 'name' in request body"
        )
        # -> {"error": {"code": "VALIDATION_ERROR", "message": "...", "suggestion": "..."}}
    """
    # Build error payload
    if structured or code or trace_id or suggestion or details:
        # Use structured format
        error_obj: dict[str, Any] = {"message": message}
        if code:
            error_obj["code"] = code
        if trace_id:
            error_obj["trace_id"] = trace_id
        if suggestion:
            error_obj["suggestion"] = suggestion
        if details:
            error_obj["details"] = details
        payload: dict[str, Any] = {"error": error_obj}
    else:
        # Simple format for backward compatibility
        payload = {"error": message}

    return json_response(payload, status=status, headers=headers)


def html_response(
    content: str,
    status: int = 200,
    headers: Optional[dict] = None,
) -> HandlerResult:
    """Create an HTML response.

    Args:
        content: HTML string content
        status: HTTP status code (default: 200)
        headers: Optional additional headers

    Returns:
        HandlerResult with HTML body and text/html content type
    """
    return HandlerResult(
        status_code=status,
        content_type="text/html; charset=utf-8",
        body=content.encode("utf-8"),
        headers=headers or {},
    )


def redirect_response(
    location: str,
    status: int = 302,
    headers: Optional[dict] = None,
) -> HandlerResult:
    """Create a redirect response.

    Args:
        location: URL to redirect to
        status: HTTP status code (default: 302 Found)
        headers: Optional additional headers

    Returns:
        HandlerResult with Location header for redirect
    """
    all_headers = headers.copy() if headers else {}
    all_headers["Location"] = location
    return HandlerResult(
        status_code=status,
        content_type="text/plain",
        body=b"",
        headers=all_headers,
    )
