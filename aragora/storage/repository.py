"""
Base repository pattern for SQLite database access.

Provides:
- Consistent connection management with WAL mode
- Standard CRUD operations
- Batch query support
- Built-in error handling
- Cache invalidation hooks
"""

import logging
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, Optional, TypeVar, Union

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.storage.schema import get_wal_connection

logger = logging.getLogger(__name__)

# SQL identifier validation pattern (alphanumeric + underscore, starts with letter/underscore)
_SQL_IDENTIFIER_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _validate_column_name(name: str) -> str:
    """Validate column name to prevent SQL injection.

    Args:
        name: Column name to validate

    Returns:
        The validated column name

    Raises:
        ValueError: If column name is invalid
    """
    if not name or len(name) > 64:
        raise ValueError(f"Invalid column name length: {name!r}")
    if not _SQL_IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"Invalid column name: {name!r}")
    return name


T = TypeVar("T")


class DatabaseRepository:
    """
    Base class for SQLite database repositories.

    Provides common patterns for database access with:
    - WAL mode connections
    - Consistent error handling
    - Batch query support
    - Cache invalidation hooks

    Subclasses should:
    1. Set TABLE_NAME class attribute
    2. Override _init_schema() for table creation
    3. Implement domain-specific methods
    """

    TABLE_NAME: str = ""  # Override in subclasses

    def __init__(self, db_path: Union[str, Path], auto_init: bool = True):
        """
        Initialize the repository.

        Args:
            db_path: Path to SQLite database file
            auto_init: If True, initialize schema on construction
        """
        self.db_path = str(db_path)
        self._on_change_callbacks: list[Callable[[str], None]] = []

        if auto_init:
            self._init_schema()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get a database connection with WAL mode.

        Usage:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        """
        conn = get_wal_connection(self.db_path, timeout=DB_TIMEOUT_SECONDS)
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """
        Initialize database schema.

        Override in subclasses to create tables.
        """
        pass

    def _notify_change(self, operation: str) -> None:
        """
        Notify registered callbacks of a data change.

        Args:
            operation: Type of operation (e.g., "insert", "update", "delete")
        """
        for callback in self._on_change_callbacks:
            try:
                callback(operation)
            except Exception as e:
                logger.warning(f"Change callback error: {e}")

    def on_change(self, callback: Callable[[str], None]) -> None:
        """
        Register a callback for data change notifications.

        Args:
            callback: Function to call on change with operation type
        """
        self._on_change_callbacks.append(callback)

    # =========================================================================
    # Standard Query Methods
    # =========================================================================

    def exists(self, id_value: Any, id_column: str = "id") -> bool:
        """
        Check if a record exists.

        Args:
            id_value: Value to search for
            id_column: Column name to search in

        Returns:
            True if record exists
        """
        safe_col = _validate_column_name(id_column)
        with self.connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT 1 FROM {self.TABLE_NAME} WHERE {safe_col} = ? LIMIT 1"  # nosec B608
            cursor.execute(query, (id_value,))
            return cursor.fetchone() is not None

    def count(self, where: str = "", params: tuple = ()) -> int:
        """
        Count records matching criteria.

        Args:
            where: Optional WHERE clause (without 'WHERE' keyword)
            params: Parameters for the WHERE clause

        Returns:
            Number of matching records
        """
        # nosec B608: TABLE_NAME is class constant, where is internal with parameterized values
        query = f"SELECT COUNT(*) FROM {self.TABLE_NAME}"  # nosec B608
        if where:
            query += f" WHERE {where}"  # nosec B608

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_by_id(self, id_value: Any, id_column: str = "id") -> Optional[dict]:
        """
        Get a single record by ID.

        Args:
            id_value: Value to search for
            id_column: Column name to search in

        Returns:
            Record as dict or None if not found
        """
        safe_col = _validate_column_name(id_column)
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = f"SELECT * FROM {self.TABLE_NAME} WHERE {safe_col} = ? LIMIT 1"  # nosec B608
            cursor.execute(query, (id_value,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "",
        where: str = "",
        params: tuple = (),
    ) -> list[dict]:
        """
        Get multiple records with pagination.

        Args:
            limit: Maximum records to return
            offset: Number of records to skip
            order_by: ORDER BY clause (without 'ORDER BY' keyword).
                Format: "column [ASC|DESC][, column [ASC|DESC]]..."
            where: Optional WHERE clause (without 'WHERE' keyword)
            params: Parameters for the WHERE clause

        Returns:
            List of records as dicts

        Raises:
            ValueError: If order_by contains invalid column names or directions
        """
        # nosec B608: TABLE_NAME is class constant, where is internal with parameterized values
        query = f"SELECT * FROM {self.TABLE_NAME}"  # nosec B608
        if where:
            query += f" WHERE {where}"  # nosec B608
        if order_by:
            # Validate order_by to prevent SQL injection
            # Format: "column [ASC|DESC][, column [ASC|DESC]]..."
            order_parts = []
            for part in order_by.split(","):
                part = part.strip()
                if not part:
                    continue
                tokens = part.split()
                col_name = _validate_column_name(tokens[0])  # Raises if invalid
                direction = ""
                if len(tokens) > 1:
                    direction = tokens[1].upper()
                    if direction not in ("ASC", "DESC"):
                        raise ValueError(f"Invalid sort direction: {tokens[1]}")
                order_parts.append(f"{col_name} {direction}".strip())
            if order_parts:
                # nosec B608: order_parts columns are regex-validated via _validate_column_name
                query += f" ORDER BY {', '.join(order_parts)}"  # nosec B608
        query += f" LIMIT {int(limit)} OFFSET {int(offset)}"

        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def batch_get(self, id_values: list[Any], id_column: str = "id") -> list[dict]:
        """
        Get multiple records by IDs in a single query.

        More efficient than multiple get_by_id calls.

        Args:
            id_values: List of values to search for
            id_column: Column name to search in

        Returns:
            List of matching records as dicts
        """
        if not id_values:
            return []

        safe_col = _validate_column_name(id_column)
        placeholders = ",".join("?" * len(id_values))
        # nosec B608: TABLE_NAME is class constant, safe_col is regex-validated, id_values are parameterized
        query = f"SELECT * FROM {self.TABLE_NAME} WHERE {safe_col} IN ({placeholders})"  # nosec B608

        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, id_values)
            return [dict(row) for row in cursor.fetchall()]

    def delete_by_id(self, id_value: Any, id_column: str = "id") -> bool:
        """
        Delete a record by ID.

        Args:
            id_value: Value to search for
            id_column: Column name to search in

        Returns:
            True if a record was deleted
        """
        safe_col = _validate_column_name(id_column)
        with self.connection() as conn:
            cursor = conn.cursor()
            # nosec B608: TABLE_NAME is class constant, safe_col is regex-validated, id_value is parameterized
            cursor.execute(f"DELETE FROM {self.TABLE_NAME} WHERE {safe_col} = ?", (id_value,))  # nosec B608
            conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            self._notify_change("delete")

        return deleted

    def delete_where(self, where: str, params: tuple) -> int:
        """
        Delete records matching criteria.

        Args:
            where: WHERE clause (without 'WHERE' keyword)
            params: Parameters for the WHERE clause

        Returns:
            Number of deleted records
        """
        # nosec B608: TABLE_NAME is class constant, where is internal with parameterized values
        query = f"DELETE FROM {self.TABLE_NAME} WHERE {where}"  # nosec B608

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            count = cursor.rowcount

        if count > 0:
            self._notify_change("delete")

        return count

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def execute(self, query: str, params: tuple = ()) -> list[dict]:
        """
        Execute a custom query and return results.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            List of result rows as dicts
        """
        with self.connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def execute_write(self, query: str, params: tuple = ()) -> int:
        """
        Execute a write query (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Number of affected rows
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

    def table_exists(self) -> bool:
        """Check if the repository's table exists."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.TABLE_NAME,)
            )
            return cursor.fetchone() is not None
