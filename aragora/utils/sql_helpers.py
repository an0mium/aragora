"""SQL helper utilities for safe query construction.

This module provides utilities for safely handling SQL patterns,
particularly for LIKE clause escaping to prevent injection.
"""

from __future__ import annotations


def escape_like_pattern(value: str) -> str:
    """Escape special characters in SQL LIKE patterns to prevent injection.

    SQLite LIKE uses % and _ as wildcards. This function escapes them
    using backslash so they are treated as literal characters.

    IMPORTANT: When using this function, you must also specify the
    ESCAPE clause in your SQL query: LIKE ? ESCAPE '\\\\'

    Args:
        value: The string to escape for use in a LIKE pattern.

    Returns:
        The escaped string safe for use in LIKE patterns.

    Raises:
        TypeError: If ``value`` is not a string.

    Examples:
        >>> escape_like_pattern("100%")
        '100\\\\%'
        >>> escape_like_pattern("test_value")
        'test\\\\_value'
        >>> escape_like_pattern(r"folder\\name")
        'folder\\\\\\\\name'
        >>> escape_like_pattern(r"100%_match\\path")
        '100\\\\%\\\\_match\\\\\\\\path'

    Usage in SQL:
        escaped = escape_like_pattern(user_input)
        cursor.execute(
            "SELECT * FROM table WHERE name LIKE ? ESCAPE '\\\\'",
            (f"%{escaped}%",)
        )
    """
    if not isinstance(value, str):
        raise TypeError("value must be a string")

    # Escape backslash first (it's the escape character itself)
    value = value.replace("\\", "\\\\")
    # Escape LIKE metacharacters
    value = value.replace("%", "\\%")
    value = value.replace("_", "\\_")
    return value


def _escape_like_pattern(value: str) -> str:
    """Backward-compatible wrapper around ``escape_like_pattern``."""
    return escape_like_pattern(value)


__all__ = ["escape_like_pattern", "_escape_like_pattern"]
