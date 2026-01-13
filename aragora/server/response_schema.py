"""
Standard Response Schema helpers for Aragora API.

Provides consistent response formatting for paginated lists, single items,
and error responses. Follows REST conventions where HTTP status codes
indicate success/failure.

Response Patterns:
-----------------
1. Single item: {"id": ..., "name": ..., ...}
2. List with pagination: {"items": [...], "total": N, "offset": 0, "limit": 20}
3. Simple list: {"debates": [...], "total": N}
4. Error: {"error": "message", "code": "ERROR_CODE", "trace_id": "abc123"}
5. Health check: {"status": "healthy", ...}

Usage:
------
    from aragora.server.response_schema import (
        paginated_response,
        list_response,
        item_response,
        success_response,
    )

    # Paginated list
    return paginated_response(
        items=debates[:limit],
        total=total_count,
        offset=offset,
        limit=limit,
        item_key="debates",
    )

    # Simple success
    return success_response({"message": "Created"}, status=201)
"""

from typing import Any, Optional
from .handlers.base import json_response, HandlerResult


def paginated_response(
    items: list,
    total: int,
    offset: int = 0,
    limit: int = 20,
    item_key: str = "items",
    extra: Optional[dict] = None,
) -> HandlerResult:
    """Create a paginated list response.

    Args:
        items: List of items for current page
        total: Total count of items (before pagination)
        offset: Current offset
        limit: Page size
        item_key: Key name for items array (default: "items")
        extra: Additional fields to include in response

    Returns:
        HandlerResult with JSON response:
        {
            "<item_key>": [...],
            "total": N,
            "offset": 0,
            "limit": 20,
            "has_more": true/false
        }
    """
    data = {
        item_key: items,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(items) < total,
    }
    if extra:
        data.update(extra)
    return json_response(data)


def list_response(
    items: list,
    item_key: str = "items",
    include_total: bool = True,
    extra: Optional[dict] = None,
) -> HandlerResult:
    """Create a simple list response (non-paginated).

    Args:
        items: List of items
        item_key: Key name for items array
        include_total: Whether to include total count
        extra: Additional fields to include

    Returns:
        HandlerResult with JSON response:
        {
            "<item_key>": [...],
            "total": N  # if include_total
        }
    """
    data: dict[str, Any] = {item_key: items}
    if include_total:
        data["total"] = len(items)
    if extra:
        data.update(extra)
    return json_response(data)


def item_response(
    item: Any,
    status: int = 200,
    extra: Optional[dict] = None,
) -> HandlerResult:
    """Create a single item response.

    For dict items, optionally merges extra fields.
    For non-dict items, wraps in {"data": item}.

    Args:
        item: The item to return
        status: HTTP status code
        extra: Additional fields to merge (for dict items)

    Returns:
        HandlerResult with the item data
    """
    if isinstance(item, dict):
        data = {**item}
        if extra:
            data.update(extra)
    else:
        data = {"data": item}
        if extra:
            data.update(extra)
    return json_response(data, status=status)


def success_response(
    data: Optional[dict] = None,
    message: Optional[str] = None,
    status: int = 200,
) -> HandlerResult:
    """Create a success response for operations.

    Args:
        data: Optional response data
        message: Optional success message
        status: HTTP status code (default 200)

    Returns:
        HandlerResult with success response:
        {"message": "...", ...data}
    """
    response = data or {}
    if message:
        response["message"] = message
    return json_response(response, status=status)


def created_response(
    item: dict,
    id_field: str = "id",
) -> HandlerResult:
    """Create a 201 Created response.

    Args:
        item: The created item
        id_field: Name of the ID field (for Location header)

    Returns:
        HandlerResult with 201 status and item data
    """
    headers = {}
    if id_field in item:
        headers["Location"] = f"/api/{item[id_field]}"
    return json_response(item, status=201, headers=headers)


def deleted_response(
    message: str = "Deleted successfully",
) -> HandlerResult:
    """Create a delete success response.

    Args:
        message: Success message

    Returns:
        HandlerResult with 200 status
    """
    return json_response({"message": message, "deleted": True})


def not_found_response(
    resource: str = "Resource",
    identifier: Optional[str] = None,
) -> HandlerResult:
    """Create a 404 Not Found response.

    Args:
        resource: Type of resource not found
        identifier: Optional identifier that was searched

    Returns:
        HandlerResult with 404 status
    """
    message = f"{resource} not found"
    if identifier:
        message = f"{resource} '{identifier}' not found"
    return json_response({"error": message, "code": "NOT_FOUND"}, status=404)


def validation_error_response(
    errors: list[str] | str,
    field: Optional[str] = None,
) -> HandlerResult:
    """Create a 400 validation error response.

    Args:
        errors: List of error messages or single message
        field: Optional field name that failed validation

    Returns:
        HandlerResult with 400 status
    """
    if isinstance(errors, str):
        errors = [errors]

    data = {
        "error": "Validation failed",
        "code": "VALIDATION_ERROR",
        "details": errors,
    }
    if field:
        data["field"] = field
    return json_response(data, status=400)


def rate_limit_response(
    retry_after: int = 60,
) -> HandlerResult:
    """Create a 429 rate limit response.

    Args:
        retry_after: Seconds until client can retry

    Returns:
        HandlerResult with 429 status and Retry-After header
    """
    return json_response(
        {
            "error": "Rate limit exceeded",
            "code": "RATE_LIMITED",
            "retry_after": retry_after,
        },
        status=429,
        headers={"Retry-After": str(retry_after)},
    )


def server_error_response(
    trace_id: Optional[str] = None,
    message: str = "Internal server error",
) -> HandlerResult:
    """Create a 500 server error response.

    Args:
        trace_id: Optional trace ID for debugging
        message: Error message (sanitized)

    Returns:
        HandlerResult with 500 status
    """
    data = {
        "error": message,
        "code": "INTERNAL_ERROR",
    }
    if trace_id:
        data["trace_id"] = trace_id
    return json_response(data, status=500)


# Common response type hints for documentation
ResponseType = HandlerResult
PaginatedResponse = HandlerResult
ListResponse = HandlerResult
ItemResponse = HandlerResult
ErrorResponse = HandlerResult


# =============================================================================
# V2 Response Envelope Support
# =============================================================================


def v2_envelope(
    data: Any,
    meta: Optional[dict] = None,
    status: int = 200,
    headers: Optional[dict] = None,
) -> HandlerResult:
    """Wrap response data in V2 envelope format.

    V2 API responses follow the structure:
        {
            "data": {...},      # Actual response data
            "meta": {...}       # Metadata (pagination, timing, etc.)
        }

    Args:
        data: The response data to wrap
        meta: Optional metadata dict (pagination cursor, total, etc.)
        status: HTTP status code
        headers: Optional additional headers

    Returns:
        HandlerResult with V2 envelope structure
    """
    envelope = {"data": data}
    if meta:
        envelope["meta"] = meta
    return json_response(envelope, status=status, headers=headers)


def v2_paginated_response(
    items: list,
    total: int,
    cursor: Optional[str] = None,
    next_cursor: Optional[str] = None,
    limit: int = 20,
    item_key: str = "items",
    extra_meta: Optional[dict] = None,
) -> HandlerResult:
    """Create a V2 paginated response with cursor-based pagination.

    V2 uses cursor-based pagination instead of offset:
        {
            "data": {
                "<item_key>": [...]
            },
            "meta": {
                "total": N,
                "limit": 20,
                "cursor": "current_cursor",
                "next_cursor": "next_page_cursor",  # null if no more pages
                "has_more": true/false
            }
        }

    Args:
        items: List of items for current page
        total: Total count of items
        cursor: Current cursor value
        next_cursor: Cursor for next page (None if last page)
        limit: Page size
        item_key: Key name for items array
        extra_meta: Additional metadata fields

    Returns:
        HandlerResult with V2 paginated response
    """
    data = {item_key: items}
    meta: dict[str, Any] = {
        "total": total,
        "limit": limit,
        "has_more": next_cursor is not None,
    }
    if cursor:
        meta["cursor"] = cursor
    if next_cursor:
        meta["next_cursor"] = next_cursor
    if extra_meta:
        meta.update(extra_meta)

    return v2_envelope(data, meta=meta)


def v2_list_response(
    items: list,
    item_key: str = "items",
    extra_meta: Optional[dict] = None,
) -> HandlerResult:
    """Create a V2 list response (non-paginated).

    Args:
        items: List of items
        item_key: Key name for items array
        extra_meta: Additional metadata

    Returns:
        HandlerResult with V2 envelope:
        {
            "data": {"<item_key>": [...]},
            "meta": {"total": N}
        }
    """
    data = {item_key: items}
    meta = {"total": len(items)}
    if extra_meta:
        meta.update(extra_meta)
    return v2_envelope(data, meta=meta)


def v2_item_response(
    item: Any,
    status: int = 200,
    meta: Optional[dict] = None,
) -> HandlerResult:
    """Create a V2 single item response.

    Args:
        item: The item to return
        status: HTTP status code
        meta: Optional metadata

    Returns:
        HandlerResult with V2 envelope:
        {"data": item, "meta": {...}}
    """
    return v2_envelope(item, meta=meta, status=status)


def v2_success_response(
    message: str,
    data: Optional[dict] = None,
    status: int = 200,
) -> HandlerResult:
    """Create a V2 success response.

    Args:
        message: Success message
        data: Optional additional data
        status: HTTP status code

    Returns:
        HandlerResult with V2 envelope:
        {"data": {"message": "...", ...}, "meta": {}}
    """
    response_data = {"message": message}
    if data:
        response_data.update(data)
    return v2_envelope(response_data, status=status)


def v2_error_response(
    code: str,
    message: str,
    status: int = 400,
    details: Optional[dict] = None,
    trace_id: Optional[str] = None,
) -> HandlerResult:
    """Create a V2 structured error response.

    V2 error format:
        {
            "error": {
                "code": "ERROR_CODE",
                "message": "Human readable message",
                "details": {...},  # optional
                "trace_id": "abc123"  # optional
            }
        }

    Args:
        code: Error code (e.g., "VALIDATION_ERROR", "NOT_FOUND")
        message: Human-readable error message
        status: HTTP status code
        details: Optional additional error details
        trace_id: Optional trace ID for debugging

    Returns:
        HandlerResult with V2 error structure
    """
    error: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if details:
        error["details"] = details
    if trace_id:
        error["trace_id"] = trace_id

    return json_response({"error": error}, status=status)
