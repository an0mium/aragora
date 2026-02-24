"""
Request validation limits for the Aragora HTTP server.

Protects against:
- Oversized request bodies (default 10 MB)
- Deeply nested JSON payloads (default 20 levels)
- Excessive query parameters (default 50)

All limits are configurable via ``RequestLimitsConfig``.

Usage::

    from aragora.server.security.request_limits import RequestLimitsMiddleware

    middleware = RequestLimitsMiddleware()

    # Check body size before reading
    ok, err = middleware.check_content_length(headers)
    if not ok:
        return error_response(413, err)

    # After parsing JSON
    ok, err = middleware.check_json_depth(parsed_body)
    if not ok:
        return error_response(400, err)

    # After parsing query string
    ok, err = middleware.check_query_params(params)
    if not ok:
        return error_response(400, err)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_BODY_BYTES: int = 10 * 1024 * 1024  # 10 MB
DEFAULT_MAX_JSON_DEPTH: int = 20
DEFAULT_MAX_QUERY_PARAMS: int = 50

HTTP_PAYLOAD_TOO_LARGE: int = 413
HTTP_BAD_REQUEST: int = 400


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RequestLimitsConfig:
    """Configurable limits for incoming requests."""

    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    max_json_depth: int = DEFAULT_MAX_JSON_DEPTH
    max_query_params: int = DEFAULT_MAX_QUERY_PARAMS

    # Per-path body-size overrides (path prefix -> max bytes).
    path_body_overrides: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure check functions (usable without middleware instance)
# ---------------------------------------------------------------------------


def check_json_depth(
    obj: Any,
    max_depth: int = DEFAULT_MAX_JSON_DEPTH,
) -> tuple[bool, str]:
    """Iteratively verify that *obj* does not exceed *max_depth* nesting levels.

    Returns ``(True, "")`` on success, ``(False, reason)`` on violation.
    """
    # Iterative approach avoids Python recursion-limit issues on malicious input.
    stack: list[tuple[Any, int]] = [(obj, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            return False, f"JSON nesting depth exceeds maximum of {max_depth}"
        if isinstance(current, dict):
            for v in current.values():
                if isinstance(v, (dict, list)):
                    stack.append((v, depth + 1))
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    stack.append((item, depth + 1))
    return True, ""


def check_query_params(
    query_string: str,
    max_params: int = DEFAULT_MAX_QUERY_PARAMS,
) -> tuple[bool, str]:
    """Check that a query string does not contain more than *max_params* keys.

    Returns ``(True, "")`` on success, ``(False, reason)`` on violation.
    """
    if not query_string:
        return True, ""
    params = parse_qs(query_string, keep_blank_values=True)
    total = sum(len(v) for v in params.values())
    if total > max_params:
        return False, f"Too many query parameters ({total} exceeds maximum {max_params})"
    return True, ""


# ---------------------------------------------------------------------------
# Middleware class
# ---------------------------------------------------------------------------


class RequestLimitsMiddleware:
    """Middleware that enforces request size and complexity limits.

    All check methods return ``(allowed: bool, error_message: str)`` tuples.
    """

    def __init__(self, config: RequestLimitsConfig | None = None) -> None:
        self.config = config or RequestLimitsConfig()

    # -- body size ---------------------------------------------------------

    def _max_body_for(self, path: str) -> int:
        """Resolve the effective body-size limit for *path*."""
        for prefix, limit in self.config.path_body_overrides.items():
            if path.startswith(prefix):
                return limit
        return self.config.max_body_bytes

    def check_content_length(
        self,
        headers: dict[str, str],
        path: str = "",
    ) -> tuple[bool, str]:
        """Validate the ``Content-Length`` header against the configured limit.

        Returns ``(True, "")`` if acceptable, ``(False, message)`` otherwise.
        """
        raw = headers.get("Content-Length") or headers.get("content-length")
        if not raw:
            return True, ""

        try:
            length = int(raw)
        except ValueError:
            logger.warning("Invalid Content-Length header value")
            return False, "Invalid Content-Length header"

        if length < 0:
            return False, "Content-Length must not be negative"

        max_allowed = self._max_body_for(path)
        if length > max_allowed:
            logger.warning(
                "Request body too large",
                extra={
                    "content_length": length,
                    "max_allowed": max_allowed,
                    "path": path,
                },
            )
            return False, "Request body exceeds maximum allowed size"
        return True, ""

    # -- JSON depth --------------------------------------------------------

    def check_json_depth(self, obj: Any) -> tuple[bool, str]:
        """Check that a parsed JSON payload does not exceed the depth limit."""
        return check_json_depth(obj, self.config.max_json_depth)

    # -- query params ------------------------------------------------------

    def check_query_params(self, query_string: str) -> tuple[bool, str]:
        """Check that the query string does not contain too many parameters."""
        return check_query_params(query_string, self.config.max_query_params)

    # -- convenience -------------------------------------------------------

    def validate_request(
        self,
        headers: dict[str, str],
        path: str = "",
        query_string: str = "",
        parsed_body: Any = None,
    ) -> tuple[bool, int, str]:
        """Run all applicable checks for an incoming request.

        Returns ``(ok, status_code, error_message)``.  ``status_code`` is
        only meaningful when ``ok`` is ``False``.
        """
        ok, msg = self.check_content_length(headers, path)
        if not ok:
            return False, HTTP_PAYLOAD_TOO_LARGE, msg

        ok, msg = self.check_query_params(query_string)
        if not ok:
            return False, HTTP_BAD_REQUEST, msg

        if parsed_body is not None:
            ok, msg = self.check_json_depth(parsed_body)
            if not ok:
                return False, HTTP_BAD_REQUEST, msg

        return True, 200, ""


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MAX_JSON_DEPTH",
    "DEFAULT_MAX_QUERY_PARAMS",
    "HTTP_PAYLOAD_TOO_LARGE",
    "HTTP_BAD_REQUEST",
    "RequestLimitsConfig",
    "RequestLimitsMiddleware",
    "check_json_depth",
    "check_query_params",
]
