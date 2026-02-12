"""
Request Validation Middleware for FastAPI.

Provides input validation for all API requests:
- Request body size limits
- JSON nesting depth limits
- Array/object size limits

These are defense-in-depth measures to prevent abuse and DoS via
deeply nested JSON, oversized payloads, or excessively large arrays.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class ValidationLimits:
    """Configurable limits for request validation.

    Attributes:
        max_body_size: Maximum request body size in bytes (default 10MB)
        max_json_depth: Maximum nesting depth for JSON objects/arrays (default 10)
        max_array_items: Maximum items in any single JSON array (default 1000)
        max_object_keys: Maximum keys in any single JSON object (default 500)
        blocking_mode: If True, return 400 errors for invalid requests.
                      If False, only log warnings and allow request through.
                      Default True for production, use False for migration.
    """

    max_body_size: int = 10_485_760  # 10MB
    max_json_depth: int = 10
    max_array_items: int = 1000
    max_object_keys: int = 500
    blocking_mode: bool = True


def _check_json_depth(obj: Any, max_depth: int, max_array: int, max_keys: int) -> str | None:
    """Check JSON structure depth and collection sizes.

    Returns an error message string if limits are exceeded, None if valid.
    """
    stack: list[tuple[Any, int]] = [(obj, 0)]

    while stack:
        current, depth = stack.pop()

        if depth > max_depth:
            return f"JSON nesting depth exceeds limit of {max_depth}"

        if isinstance(current, dict):
            if len(current) > max_keys:
                return f"JSON object has {len(current)} keys (max {max_keys})"
            for value in current.values():
                if isinstance(value, (dict, list)):
                    stack.append((value, depth + 1))

        elif isinstance(current, list):
            if len(current) > max_array:
                return f"JSON array has {len(current)} items (max {max_array})"
            for item in current:
                if isinstance(item, (dict, list)):
                    stack.append((item, depth + 1))

    return None


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware that validates request body size and JSON structure.

    Rejects requests that exceed configured limits for body size,
    JSON nesting depth, or collection sizes.
    """

    def __init__(self, app: Any, limits: ValidationLimits | None = None):
        super().__init__(app)
        self.limits = limits or ValidationLimits()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip validation for GET, HEAD, OPTIONS, DELETE (typically no body)
        if request.method in ("GET", "HEAD", "OPTIONS", "DELETE"):
            return await call_next(request)

        # Check Content-Length header first (fast path)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                length = int(content_length)
                if length > self.limits.max_body_size:
                    logger.warning(
                        f"Request body too large: {length} bytes "
                        f"(max {self.limits.max_body_size}) for {request.method} {request.url.path}"
                    )
                    if self.limits.blocking_mode:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "error": "Request body too large",
                                "code": "payload_too_large",
                                "details": {
                                    "max_bytes": self.limits.max_body_size,
                                    "received_bytes": length,
                                },
                            },
                        )
            except ValueError:
                pass

        # For JSON requests, validate structure
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body = await request.body()

                # Check actual body size
                if len(body) > self.limits.max_body_size:
                    logger.warning(
                        f"Request body too large: {len(body)} bytes for "
                        f"{request.method} {request.url.path}"
                    )
                    if self.limits.blocking_mode:
                        return JSONResponse(
                            status_code=413,
                            content={
                                "error": "Request body too large",
                                "code": "payload_too_large",
                            },
                        )

                if body:
                    parsed = json.loads(body)
                    error = _check_json_depth(
                        parsed,
                        max_depth=self.limits.max_json_depth,
                        max_array=self.limits.max_array_items,
                        max_keys=self.limits.max_object_keys,
                    )
                    if error:
                        logger.warning(
                            f"JSON structure validation failed for "
                            f"{request.method} {request.url.path}: {error}"
                        )
                        if self.limits.blocking_mode:
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "error": error,
                                    "code": "invalid_request_structure",
                                },
                            )

            except json.JSONDecodeError:
                # Let FastAPI handle JSON decode errors downstream
                pass
            except Exception as e:
                logger.error(f"Validation middleware error: {e}")
                # Don't block on unexpected errors in validation itself

        return await call_next(request)
