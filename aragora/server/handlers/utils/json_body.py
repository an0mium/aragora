"""
JSON body parsing utilities for async HTTP handlers.

Provides safe JSON body parsing with standardized error responses.
Replaces unguarded `await request.json()` calls that can crash with 500 on malformed input.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


async def parse_json_body(
    request: web.Request,
    *,
    allow_empty: bool = False,
    context: str = "request",
) -> tuple[dict[str, Any] | None, web.Response | None]:
    """Parse JSON body from an aiohttp request with error handling.

    Returns a tuple of (parsed_body, error_response). If parsing succeeds,
    error_response is None. If parsing fails, parsed_body is None and
    error_response contains a 400 response to return to the client.

    Args:
        request: The aiohttp web.Request object.
        allow_empty: If True, empty body returns ({}, None). If False, returns error.
        context: Context string for error logging (e.g., "webhook", "create_user").

    Returns:
        Tuple of (parsed_data, error_response).
        - If parsing succeeds: (dict, None)
        - If parsing fails: (None, web.Response with 400 error)
        - If body is empty and allow_empty: ({}, None)
        - If body is empty and not allow_empty: (None, web.Response with 400 error)

    Example:
        async def handle_create(self, request):
            body, err = await parse_json_body(request, context="create_user")
            if err:
                return err
            # body is guaranteed to be a dict here
            username = body.get("username")
            ...
    """
    try:
        content_length = getattr(request, "content_length", None)
        if content_length is None:
            json_attr = getattr(request, "json", None)
            if json_attr is not None:
                try:
                    parsed = await json_attr() if callable(json_attr) else json_attr
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.debug("Invalid JSON in %s: %s", context, e)
                    return None, web.json_response(
                        {"error": "Invalid JSON body", "code": "INVALID_JSON"},
                        status=400,
                    )
            else:
                body_bytes = await request.read()
                if not body_bytes:
                    if allow_empty:
                        return {}, None
                    logger.debug("Empty body in %s", context)
                    return None, web.json_response(
                        {"error": "Empty request body", "code": "EMPTY_BODY"},
                        status=400,
                    )
                try:
                    parsed = json.loads(body_bytes)
                except json.JSONDecodeError as e:
                    logger.debug("Invalid JSON in %s: %s", context, e)
                    return None, web.json_response(
                        {"error": "Invalid JSON body", "code": "INVALID_JSON"},
                        status=400,
                    )
        elif content_length == 0:
            body_bytes = await request.read()
            if not body_bytes:
                if allow_empty:
                    return {}, None
                logger.debug("Empty body in %s", context)
                return None, web.json_response(
                    {"error": "Empty request body", "code": "EMPTY_BODY"},
                    status=400,
                )
            try:
                parsed = json.loads(body_bytes)
            except json.JSONDecodeError as e:
                logger.debug("Invalid JSON in %s: %s", context, e)
                return None, web.json_response(
                    {"error": "Invalid JSON body", "code": "INVALID_JSON"},
                    status=400,
                )
        else:
            try:
                json_attr = getattr(request, "json", None)
                if json_attr is not None and not callable(json_attr):
                    parsed = json_attr
                else:
                    parsed = await request.json()
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("Invalid JSON in %s: %s", context, e)
                return None, web.json_response(
                    {"error": "Invalid JSON body", "code": "INVALID_JSON"},
                    status=400,
                )

        # Validate it's a dict (most handlers expect this)
        if not isinstance(parsed, dict):
            logger.debug("JSON body is not an object in %s: got %s", context, type(parsed).__name__)
            return None, web.json_response(
                {"error": "Request body must be a JSON object", "code": "INVALID_BODY_TYPE"},
                status=400,
            )

        return parsed, None

    except Exception as e:
        # Catch any other unexpected errors (e.g., encoding issues)
        logger.warning("Unexpected error parsing JSON body in %s: %s", context, e)
        return None, web.json_response(
            {"error": "Failed to parse request body", "code": "PARSE_ERROR"},
            status=400,
        )


async def parse_json_body_allow_array(
    request: web.Request,
    *,
    allow_empty: bool = False,
    context: str = "request",
) -> tuple[dict[str, Any] | list[Any] | None, web.Response | None]:
    """Parse JSON body allowing both objects and arrays.

    Similar to parse_json_body but allows the body to be a JSON array.
    Use this for endpoints that accept batch operations.

    Args:
        request: The aiohttp web.Request object.
        allow_empty: If True, empty body returns ({}, None). If False, returns error.
        context: Context string for error logging.

    Returns:
        Tuple of (parsed_data, error_response).
        - If parsing succeeds: (dict or list, None)
        - If parsing fails: (None, web.Response with 400 error)
    """
    try:
        content_length = getattr(request, "content_length", None)
        if content_length is None:
            json_attr = getattr(request, "json", None)
            if json_attr is not None:
                try:
                    parsed = await json_attr() if callable(json_attr) else json_attr
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.debug("Invalid JSON in %s: %s", context, e)
                    return None, web.json_response(
                        {"error": "Invalid JSON body", "code": "INVALID_JSON"},
                        status=400,
                    )
            else:
                body_bytes = await request.read()
                if not body_bytes:
                    if allow_empty:
                        return {}, None
                    return None, web.json_response(
                        {"error": "Empty request body", "code": "EMPTY_BODY"},
                        status=400,
                    )
                try:
                    parsed = json.loads(body_bytes)
                except json.JSONDecodeError as e:
                    logger.debug("Invalid JSON in %s: %s", context, e)
                    return None, web.json_response(
                        {"error": "Invalid JSON body", "code": "INVALID_JSON"},
                        status=400,
                    )
        elif content_length == 0:
            body_bytes = await request.read()
            if not body_bytes:
                if allow_empty:
                    return {}, None
                return None, web.json_response(
                    {"error": "Empty request body", "code": "EMPTY_BODY"},
                    status=400,
                )
            try:
                parsed = json.loads(body_bytes)
            except json.JSONDecodeError as e:
                logger.debug("Invalid JSON in %s: %s", context, e)
                return None, web.json_response(
                    {"error": "Invalid JSON body", "code": "INVALID_JSON"},
                    status=400,
                )
        else:
            try:
                json_attr = getattr(request, "json", None)
                if json_attr is not None and not callable(json_attr):
                    parsed = json_attr
                else:
                    parsed = await request.json()
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                logger.debug("Invalid JSON in %s: %s", context, e)
                return None, web.json_response(
                    {"error": "Invalid JSON body", "code": "INVALID_JSON"},
                    status=400,
                )

        if not isinstance(parsed, (dict, list)):
            logger.debug("JSON body is not an object or array in %s", context)
            return None, web.json_response(
                {
                    "error": "Request body must be a JSON object or array",
                    "code": "INVALID_BODY_TYPE",
                },
                status=400,
            )

        return parsed, None

    except Exception as e:
        logger.warning("Unexpected error parsing JSON body in %s: %s", context, e)
        return None, web.json_response(
            {"error": "Failed to parse request body", "code": "PARSE_ERROR"},
            status=400,
        )


__all__ = [
    "parse_json_body",
    "parse_json_body_allow_array",
]
