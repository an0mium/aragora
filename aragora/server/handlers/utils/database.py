"""
Database utilities for handler modules.

Provides database connection management and utility functions
for handlers that need direct database access.
"""

import sqlite3
from contextlib import contextmanager
from typing import Generator

from aragora.config import DB_TIMEOUT_SECONDS


@contextmanager
def get_db_connection(db_path: str) -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection with proper cleanup.

    Shared utility for handlers that need direct database access.
    Uses DatabaseManager for connection pooling and WAL mode.

    Args:
        db_path: Path to the SQLite database file

    Yields:
        sqlite3.Connection with WAL mode and timeout configured

    Example:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM table")
            rows = cursor.fetchall()
    """
    from aragora.storage.schema import DatabaseManager
    manager = DatabaseManager.get_instance(db_path, DB_TIMEOUT_SECONDS)
    with manager.fresh_connection() as conn:
        yield conn


def table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    """Check if a table exists in the SQLite database.

    Args:
        cursor: Active database cursor
        table_name: Name of the table to check

    Returns:
        True if the table exists, False otherwise

    Example:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            if not table_exists(cursor, "agent_relationships"):
                return json_response({"error": "Table not found"})
    """
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


__all__ = [
    "get_db_connection",
    "table_exists",
]
