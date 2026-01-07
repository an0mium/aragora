"""
Database connection resilience with retry logic.

Provides ResilientConnection for automatic retry on transient SQLite errors
like "database is locked" and "database is busy".
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Optional, TypeVar

from aragora.config import DB_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# Type variable for generic return type
T = TypeVar("T")

# Transient errors that can be retried (lowercase for comparison)
TRANSIENT_ERRORS = (
    "database is locked",
    "database is busy",
    "unable to open database file",
    "disk i/o error",
)


def is_transient_error(error: Exception) -> bool:
    """Check if an error is transient and can be retried."""
    error_msg = str(error).lower()
    return any(te in error_msg for te in TRANSIENT_ERRORS)


class ResilientConnection:
    """
    SQLite connection wrapper with automatic retry on transient errors.

    Provides exponential backoff retry logic for handling temporary failures
    like locked databases, busy connections, and I/O errors.

    Usage:
        conn = ResilientConnection("/path/to/db.sqlite")

        # Using transaction context manager
        with conn.transaction() as cursor:
            cursor.execute("SELECT * FROM users")
            rows = cursor.fetchall()

        # Using execute with automatic retry
        result = conn.execute("INSERT INTO logs (msg) VALUES (?)", ("hello",))
    """

    def __init__(
        self,
        db_path: str,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 2.0,
        timeout: float = DB_TIMEOUT_SECONDS,
    ):
        """
        Initialize resilient connection.

        Args:
            db_path: Path to SQLite database file
            max_retries: Maximum number of retry attempts (default: 3)
            base_delay: Initial delay between retries in seconds (default: 0.1)
            max_delay: Maximum delay between retries in seconds (default: 2.0)
            timeout: SQLite busy timeout in seconds (default: DB_TIMEOUT_SECONDS)
        """
        self.db_path = db_path
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with timeout configured."""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        return conn

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions with auto-retry.

        Automatically retries on transient errors with exponential backoff.
        Commits on success, rolls back on error.

        Yields:
            sqlite3.Cursor: Database cursor for executing queries

        Raises:
            sqlite3.Error: If all retry attempts fail
        """
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            conn: Optional[sqlite3.Connection] = None
            try:
                conn = self._create_connection()
                cursor = conn.cursor()
                yield cursor
                conn.commit()
                return
            except sqlite3.Error as e:
                last_error = e
                if conn:
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass

                if not is_transient_error(e) or attempt >= self.max_retries:
                    logger.error(
                        f"Database error after {attempt + 1} attempts: {e}"
                    )
                    raise

                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Transient database error (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                time.sleep(delay)
            finally:
                if conn:
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass

        # Should not reach here, but just in case
        if last_error:
            raise last_error

    def execute(
        self,
        query: str,
        params: tuple = (),
        fetch: bool = False,
    ):
        """
        Execute a single query with automatic retry.

        Args:
            query: SQL query to execute
            params: Query parameters
            fetch: If True, return fetched rows; if False, return lastrowid

        Returns:
            If fetch=True: List of sqlite3.Row objects
            If fetch=False: Last row ID from insert
        """
        with self.transaction() as cursor:
            cursor.execute(query, params)
            if fetch:
                return cursor.fetchall()
            return cursor.lastrowid

    def executemany(
        self,
        query: str,
        params_list: list[tuple],
    ) -> int:
        """
        Execute a query with multiple parameter sets.

        Args:
            query: SQL query to execute
            params_list: List of parameter tuples

        Returns:
            Number of rows affected
        """
        with self.transaction() as cursor:
            cursor.executemany(query, params_list)
            return cursor.rowcount


def with_retry(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 2.0,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding retry logic to database operations.

    Automatically retries functions that raise transient SQLite errors.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds

    Usage:
        @with_retry(max_retries=3)
        def save_record(db_path: str, data: dict):
            with sqlite3.connect(db_path) as conn:
                conn.execute("INSERT INTO records (data) VALUES (?)", (json.dumps(data),))
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except sqlite3.Error as e:
                    last_error = e

                    if not is_transient_error(e) or attempt >= max_retries:
                        logger.error(
                            f"Database error in {func.__name__} after {attempt + 1} attempts: {e}"
                        )
                        raise

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Transient error in {func.__name__} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)

            # Should not reach here
            if last_error:
                raise last_error
            raise RuntimeError("Unexpected retry loop exit")

        return wrapper
    return decorator


class ConnectionPool:
    """
    Simple connection pool for SQLite with health checking.

    Maintains a pool of reusable connections to reduce connection overhead.
    Automatically removes stale connections and creates new ones as needed.
    """

    def __init__(
        self,
        db_path: str,
        max_connections: int = 5,
        timeout: float = DB_TIMEOUT_SECONDS,
    ):
        """
        Initialize connection pool.

        Args:
            db_path: Path to SQLite database
            max_connections: Maximum pool size
            timeout: SQLite busy timeout
        """
        self.db_path = db_path
        self.max_connections = max_connections
        self.timeout = timeout
        self._pool: list[sqlite3.Connection] = []
        self._in_use: set[sqlite3.Connection] = set()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new connection."""
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        return conn

    def _is_connection_healthy(self, conn: sqlite3.Connection) -> bool:
        """Check if connection is still usable."""
        try:
            conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.

        Yields:
            sqlite3.Connection: A database connection

        The connection is automatically returned to the pool when done.
        """
        conn: Optional[sqlite3.Connection] = None

        # Try to get an existing connection from pool
        while self._pool:
            candidate = self._pool.pop()
            if self._is_connection_healthy(candidate):
                conn = candidate
                break
            else:
                try:
                    candidate.close()
                except sqlite3.Error:
                    pass

        # Create new connection if needed
        if conn is None:
            conn = self._create_connection()

        self._in_use.add(conn)
        try:
            yield conn
        finally:
            self._in_use.discard(conn)
            # Return to pool if not at max
            if len(self._pool) < self.max_connections:
                self._pool.append(conn)
            else:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass

    def close_all(self):
        """Close all connections in the pool."""
        for conn in self._pool:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._pool.clear()

        for conn in list(self._in_use):
            try:
                conn.close()
            except sqlite3.Error:
                pass
        self._in_use.clear()
