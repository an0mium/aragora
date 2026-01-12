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

    Example:
        >>> escape_like_pattern("100%")
        '100\\\\%'
        >>> escape_like_pattern("test_value")
        'test\\\\_value'

    Usage in SQL:
        escaped = escape_like_pattern(user_input)
        cursor.execute(
            "SELECT * FROM table WHERE name LIKE ? ESCAPE '\\\\'",
            (f"%{escaped}%",)
        )
    """
    # Escape backslash first (it's the escape character itself)
    value = value.replace('\\', '\\\\')
    # Escape LIKE metacharacters
    value = value.replace('%', '\\%')
    value = value.replace('_', '\\_')
    return value


# Alias for backwards compatibility with existing imports
_escape_like_pattern = escape_like_pattern


__all__ = ["escape_like_pattern", "_escape_like_pattern"]
