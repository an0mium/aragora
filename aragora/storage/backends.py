"""
Database backend abstraction for SQLite and PostgreSQL.

Provides a unified interface for database operations that works with both
SQLite (default) and PostgreSQL (for production scale).

Usage:
    from aragora.storage.backends import get_database_backend

    # Get configured backend (based on settings)
    db = get_database_backend()

    # Use context manager for operations
    with db.connection() as conn:
        conn.execute("INSERT INTO ...")

    # Or use convenience methods
    row = db.fetch_one("SELECT * FROM table WHERE id = %s", (id,))
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ContextManager, Optional, Union

logger = logging.getLogger(__name__)

# Optional PostgreSQL support
try:
    import psycopg2
    from psycopg2 import pool as pg_pool
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    psycopg2 = None  # type: ignore
    pg_pool = None  # type: ignore
    RealDictCursor = None  # type: ignore
    POSTGRESQL_AVAILABLE = False


class DatabaseBackend(ABC):
    """
    Abstract base class for database backends.

    Provides a unified interface that works with both SQLite and PostgreSQL.
    """

    @abstractmethod
    def connection(self) -> ContextManager[Any]:
        """Context manager for database operations with automatic commit/rollback."""
        pass

    @abstractmethod
    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Execute query and fetch single row."""
        pass

    @abstractmethod
    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Execute query and fetch all rows."""
        pass

    @abstractmethod
    def execute_write(self, sql: str, params: tuple = ()) -> None:
        """Execute a write operation."""
        pass

    @abstractmethod
    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a SQL statement with multiple parameter sets."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection/pool."""
        pass

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """Return the backend type ('sqlite' or 'postgresql')."""
        pass

    def convert_placeholder(self, sql: str) -> str:
        """
        Convert SQL placeholders between SQLite (?) and PostgreSQL (%s).

        This default implementation returns SQL unchanged.
        Subclasses should override if needed.
        """
        return sql


class SQLiteBackend(DatabaseBackend):
    """
    SQLite database backend.

    Provides thread-safe access using fresh connections per operation
    with WAL mode for better concurrent read/write performance.
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        timeout: float = 30.0,
    ):
        """
        Initialize SQLite backend.

        Args:
            db_path: Path to the SQLite database file.
            timeout: Connection timeout in seconds.
        """
        self.db_path = Path(db_path)
        self.timeout = timeout

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize with WAL mode
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database with optimal settings."""
        conn = sqlite3.connect(str(self.db_path), timeout=self.timeout)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a fresh database connection."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager  # type: ignore[override]
    def connection(self):  # type: ignore[override]
        """Context manager for database operations."""
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error:
            conn.rollback()
            raise
        finally:
            conn.close()

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Execute query and fetch single row."""
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchone()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Execute query and fetch all rows."""
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()

    def execute_write(self, sql: str, params: tuple = ()) -> None:
        """Execute a write operation."""
        with self.connection() as conn:
            conn.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a SQL statement with multiple parameter sets."""
        with self.connection() as conn:
            conn.executemany(sql, params_list)

    def close(self) -> None:
        """SQLite uses fresh connections, nothing to close."""
        pass

    @property
    def backend_type(self) -> str:
        return "sqlite"

    def __repr__(self) -> str:
        return f"SQLiteBackend({self.db_path!r})"


class PostgreSQLBackend(DatabaseBackend):
    """
    PostgreSQL database backend.

    Uses connection pooling for efficient concurrent access.
    Requires psycopg2 to be installed.
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 5,
        pool_max_overflow: int = 10,
    ):
        """
        Initialize PostgreSQL backend.

        Args:
            database_url: PostgreSQL connection URL.
            pool_size: Minimum connections in pool.
            pool_max_overflow: Maximum overflow connections.
        """
        if not POSTGRESQL_AVAILABLE:
            raise ImportError(
                "psycopg2 is required for PostgreSQL support. "
                "Install with: pip install psycopg2-binary"
            )

        self.database_url = database_url
        self.pool_size = pool_size
        self.pool_max_overflow = pool_max_overflow

        # Create connection pool
        self._pool = pg_pool.ThreadedConnectionPool(
            minconn=pool_size,
            maxconn=pool_size + pool_max_overflow,
            dsn=database_url,
        )
        logger.info(f"PostgreSQL connection pool created (size={pool_size})")

    @contextmanager  # type: ignore[override]
    def connection(self):  # type: ignore[override]
        """Context manager for database operations."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except psycopg2.Error:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Execute query and fetch single row."""
        sql = self.convert_placeholder(sql)
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchone()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Execute query and fetch all rows."""
        sql = self.convert_placeholder(sql)
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                return cursor.fetchall()

    def execute_write(self, sql: str, params: tuple = ()) -> None:
        """Execute a write operation."""
        sql = self.convert_placeholder(sql)
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> None:
        """Execute a SQL statement with multiple parameter sets."""
        sql = self.convert_placeholder(sql)
        with self.connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(sql, params_list)

    def convert_placeholder(self, sql: str) -> str:
        """Convert SQLite ? placeholders to PostgreSQL %s."""
        # Simple conversion - doesn't handle ? inside strings
        return sql.replace("?", "%s")

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.closeall()
            logger.info("PostgreSQL connection pool closed")

    @property
    def backend_type(self) -> str:
        return "postgresql"

    def __repr__(self) -> str:
        # Hide credentials in URL
        url_parts = self.database_url.split("@")
        safe_url = f"***@{url_parts[-1]}" if len(url_parts) > 1 else "***"
        return f"PostgreSQLBackend({safe_url})"


# Global backend instance (protected by _backend_lock)
_backend: Optional[DatabaseBackend] = None
_backend_initialized: bool = False
_backend_lock = threading.Lock()  # Protects initialization


def get_database_backend(
    force_sqlite: bool = False,
    db_path: Optional[str] = None,
) -> DatabaseBackend:
    """
    Get the configured database backend.

    Uses settings to determine whether to use SQLite or PostgreSQL.
    Caches the backend instance for reuse.

    Args:
        force_sqlite: Force SQLite backend regardless of settings.
        db_path: Optional SQLite database path (for testing).

    Returns:
        DatabaseBackend instance.
    """
    global _backend, _backend_initialized

    # Fast path: already initialized (no lock needed for read)
    if _backend_initialized and _backend is not None and not force_sqlite:
        return _backend

    # Slow path: acquire lock for initialization
    with _backend_lock:
        # Double-check after acquiring lock (another thread may have initialized)
        if _backend_initialized and _backend is not None and not force_sqlite:
            return _backend

        # Get configuration
        try:
            from aragora.config.settings import get_settings
            settings = get_settings()
            db_settings = settings.database
        except Exception as e:
            logger.warning(f"Could not load settings, using SQLite: {e}")
            db_settings = None

        # Determine backend
        if force_sqlite:
            backend_type = "sqlite"
        elif db_settings and db_settings.is_postgresql:
            backend_type = "postgresql"
        else:
            backend_type = "sqlite"

        # Create backend
        if backend_type == "postgresql" and db_settings:
            try:
                _backend = PostgreSQLBackend(
                    database_url=db_settings.url,
                    pool_size=db_settings.pool_size,
                    pool_max_overflow=db_settings.pool_max_overflow,
                )
                logger.info("Using PostgreSQL backend")
            except ImportError as e:
                logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")
                _backend = SQLiteBackend(
                    db_path=db_path or str(Path(db_settings.nomic_dir) / "aragora.db"),
                    timeout=db_settings.timeout_seconds,
                )
        else:
            # SQLite backend
            if db_path:
                sqlite_path = db_path
            elif db_settings:
                sqlite_path = str(Path(db_settings.nomic_dir) / "aragora.db")
            else:
                sqlite_path = ".nomic/aragora.db"

            _backend = SQLiteBackend(db_path=sqlite_path)
            logger.info(f"Using SQLite backend: {sqlite_path}")

        _backend_initialized = True
        return _backend


def reset_database_backend() -> None:
    """Reset the database backend (for testing)."""
    global _backend, _backend_initialized

    with _backend_lock:
        if _backend is not None:
            _backend.close()
            _backend = None

        _backend_initialized = False


__all__ = [
    "DatabaseBackend",
    "SQLiteBackend",
    "PostgreSQLBackend",
    "get_database_backend",
    "reset_database_backend",
    "POSTGRESQL_AVAILABLE",
]
