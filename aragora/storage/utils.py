"""
Database utility functions for safe query operations.

Provides helper functions to prevent common database access errors
like NoneType subscripting when queries return empty results.
"""

__all__ = [
    "fetch_scalar",
    "fetch_scalar_or_none",
    "fetch_row_or_default",
]

from typing import Any, Optional


def fetch_scalar(cursor, default: Any = 0) -> Any:
    """
    Safely fetch a single scalar value from a database cursor.

    Prevents TypeError when cursor.fetchone() returns None (empty result set).

    Args:
        cursor: Database cursor after executing a query
        default: Value to return if no rows are returned (default: 0)

    Returns:
        The first column of the first row, or the default value

    Example:
        cursor.execute("SELECT COUNT(*) FROM users")
        count = fetch_scalar(cursor, default=0)

        # Instead of dangerous:
        # count = cursor.fetchone()[0]  # Crashes if no rows!
    """
    row = cursor.fetchone()
    return row[0] if row else default


def fetch_scalar_or_none(cursor) -> Optional[Any]:
    """
    Fetch a single scalar value, returning None if no rows.

    Use this when None is a meaningful value distinct from a default.

    Args:
        cursor: Database cursor after executing a query

    Returns:
        The first column of the first row, or None
    """
    row = cursor.fetchone()
    return row[0] if row else None


def fetch_row_or_default(cursor, default: tuple = ()) -> tuple:
    """
    Safely fetch a single row from a database cursor.

    Args:
        cursor: Database cursor after executing a query
        default: Value to return if no rows are returned

    Returns:
        The first row as a tuple, or the default value
    """
    row = cursor.fetchone()
    return row if row else default
