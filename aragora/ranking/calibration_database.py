"""
Database abstraction for the calibration engine.

Provides thread-safe database access by delegating to DatabaseManager
with per-operation connections for concurrent access patterns.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.storage.schema import DatabaseManager

logger = logging.getLogger(__name__)


class CalibrationDatabase:
    """
    Database wrapper for calibration system operations.

    Provides thread-safe access via DatabaseManager.fresh_connection(),
    which creates a new connection per operation. Uses WAL mode for
    better concurrent read/write performance.

    Usage:
        db = CalibrationDatabase("/path/to/calibration.db")

        # Context manager with auto-commit/rollback
        with db.connection() as conn:
            conn.execute("INSERT INTO ...")

        # Convenience methods
        row = db.fetch_one("SELECT * FROM predictions WHERE id = ?", ("123",))
        rows = db.fetch_all("SELECT * FROM predictions ORDER BY timestamp DESC")
    """

    def __init__(self, db_path: Union[str, Path], timeout: float = DB_TIMEOUT_SECONDS):
        """Initialize the CalibrationDatabase wrapper.

        Args:
            db_path: Path to the SQLite database file
            timeout: Connection timeout in seconds
        """
        self.db_path = Path(db_path)
        self._timeout = timeout
        self._manager = DatabaseManager.get_instance(db_path, timeout)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database operations with automatic commit/rollback.

        Creates a fresh connection per operation for thread safety.
        Commits on success, rolls back on exception.

        Yields:
            sqlite3.Connection for database operations
        """
        with self._manager.fresh_connection() as conn:
            yield conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Explicit transaction context manager.

        Creates a fresh connection with explicit BEGIN/COMMIT for clarity.

        Yields:
            sqlite3.Connection within a transaction
        """
        with self._manager.fresh_connection() as conn:
            conn.execute("BEGIN")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception as e:
                logger.warning(
                    f"Non-database exception during transaction, rolling back: {type(e).__name__}: {e}"
                )
                conn.execute("ROLLBACK")
                raise

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Execute query and fetch single row.

        Args:
            sql: SQL query to execute
            params: Query parameters

        Returns:
            Single row as tuple, or None if no results
        """
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()
            # Convert sqlite3.Row to tuple for consistent return type
            return tuple(row) if row is not None else None

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Execute query and fetch all rows.

        Args:
            sql: SQL query to execute
            params: Query parameters

        Returns:
            List of rows as tuples
        """
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            # Convert sqlite3.Row objects to tuples for consistent return type
            return [tuple(row) for row in cursor.fetchall()]

    def execute_write(self, sql: str, params: tuple = ()) -> None:
        """Execute a write operation with auto-commit.

        Args:
            sql: SQL statement to execute
            params: Statement parameters
        """
        with self.connection() as conn:
            conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement to execute
            params_list: List of parameter tuples
        """
        with self.connection() as conn:
            conn.executemany(sql, params_list)

    def __repr__(self) -> str:
        return f"CalibrationDatabase({self.db_path!r})"
