"""
Request utility methods for the unified server.

This module provides the RequestUtilsMixin class with methods for:
- Safe parameter parsing (_safe_int, _safe_float, _safe_string)
- Path segment extraction (_extract_path_segment)
- Content length validation (_validate_content_length)

These methods are extracted from UnifiedHandler to improve modularity
and allow easier testing of request parsing logic.
"""

import re
from typing import Any

from aragora.server.validation import safe_query_float, safe_query_int

# DoS protection limits
MAX_JSON_CONTENT_LENGTH: int = 10 * 1024 * 1024  # 10MB for JSON API


class RequestUtilsMixin:
    """Mixin providing request parsing and validation utility methods.

    This mixin expects the following from the parent class:
    - headers: HTTP headers dict (for content length validation)

    And these methods from ResponseHelpersMixin:
    - _send_json(data, status): Send JSON response
    """

    # Type stubs for attributes expected from parent class
    headers: Any

    # Type stubs for methods expected from parent class
    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        """Send JSON response - provided by ResponseHelpersMixin."""
        ...

    def _safe_int(self, query: dict[str, Any], key: str, default: int, max_val: int = 100) -> int:
        """Safely parse integer query param with bounds checking.

        Delegates to shared safe_query_int from validation module.

        Args:
            query: Query parameters dict
            key: Parameter key to extract
            default: Default value if key not found or invalid
            max_val: Maximum allowed value (default 100)

        Returns:
            Parsed and bounded integer value
        """
        return safe_query_int(query, key, default, min_val=1, max_val=max_val)

    def _safe_float(
        self,
        query: dict[str, Any],
        key: str,
        default: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
    ) -> float:
        """Safely parse float query param with bounds checking.

        Delegates to shared safe_query_float from validation module.

        Args:
            query: Query parameters dict
            key: Parameter key to extract
            default: Default value if key not found or invalid
            min_val: Minimum allowed value (default 0.0)
            max_val: Maximum allowed value (default 1.0)

        Returns:
            Parsed and bounded float value
        """
        return safe_query_float(query, key, default, min_val=min_val, max_val=max_val)

    def _safe_string(
        self, value: str, max_len: int = 500, pattern: str | None = None
    ) -> str | None:
        """Safely validate string parameter with length and pattern checks.

        Args:
            value: The string to validate
            max_len: Maximum allowed length (default 500)
            pattern: Optional regex pattern to match (e.g., r'^[a-zA-Z0-9_-]+$')

        Returns:
            Validated string or None if invalid
        """
        if not value or not isinstance(value, str):
            return None
        # Truncate to max length
        value = value[:max_len]
        # Validate pattern if provided
        if pattern and not re.match(pattern, value):
            return None
        return value

    def _extract_path_segment(self, path: str, index: int, segment_name: str = "id") -> str | None:
        """Safely extract path segment with bounds checking.

        Returns None and sends 400 error if segment is missing.

        Args:
            path: URL path to extract from (e.g., "/api/debates/123/messages")
            index: Index of segment to extract (0-based after split)
            segment_name: Human-readable name for error messages

        Returns:
            Extracted segment value or None if missing (error response sent)
        """
        parts = path.split("/")
        if len(parts) <= index or not parts[index]:
            self._send_json({"error": f"Missing {segment_name} in path"}, status=400)
            return None
        return parts[index]

    def _validate_content_length(self, max_size: int | None = None) -> int | None:
        """Validate Content-Length header for DoS protection.

        Returns content length if valid, None if invalid (error already sent).

        Args:
            max_size: Maximum allowed content length in bytes.
                     Defaults to MAX_JSON_CONTENT_LENGTH (10MB).

        Returns:
            Content length as int if valid, None if invalid (error response sent)
        """
        max_size = max_size or MAX_JSON_CONTENT_LENGTH

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length header"}, status=400)
            return None

        if content_length < 0:
            self._send_json({"error": "Invalid Content-Length: negative value"}, status=400)
            return None

        if content_length > max_size:
            size_mb = max_size / (1024 * 1024)
            self._send_json({"error": f"Content too large. Max: {size_mb:.0f}MB"}, status=413)
            return None

        return content_length


__all__ = ["RequestUtilsMixin", "MAX_JSON_CONTENT_LENGTH"]
