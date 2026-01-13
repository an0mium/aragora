"""
Database backend abstraction layer.

Provides a unified interface for SQLite and PostgreSQL, enabling
seamless switching between development (SQLite) and production (PostgreSQL).

Features:
- Protocol-based abstraction for type safety
- Connection pooling for both backends
- Automatic parameter placeholder translation (? <-> $N)
- Context managers for transaction handling
- Health checks and connection validation
"""

import logging
import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional, Protocol, Union

from aragora.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# Type aliases
Params = Union[tuple, dict]
Row = tuple[Any, ...]


class ConnectionProtocol(Protocol):
    """Protocol for database connections."""

    def execute(self, sql: str, params: Params = ()) -> Any:
        """Execute a SQL statement."""
        ...

    def executemany(self, sql: str, params_list: list[Params]) -> Any:
        """Execute a SQL statement with multiple parameter sets."""
        ...

    def commit(self) -> None:
        """Commit the current transaction."""
        ...

    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class CursorProtocol(Protocol):
    """Protocol for database cursors."""

    def fetchone(self) -> Optional[Row]:
        """Fetch a single row."""
        ...

    def fetchall(self) -> list[Row]:
        """Fetch all rows."""
        ...

    def fetchmany(self, size: int = 100) -> list[Row]:
        """Fetch up to size rows."""
        ...

    @property
    def rowcount(self) -> int:
        """Number of affected rows."""
        ...


@dataclass
class DatabaseConfig:
    """Database configuration."""

    # Backend type: 'sqlite' or 'postgres'
    backend: str = "sqlite"

    # SQLite settings
    sqlite_path: str = "aragora.db"

    # PostgreSQL settings
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "aragora"
    pg_user: str = "aragora"
    pg_password: str = ""
    pg_ssl_mode: str = "prefer"

    # Connection pool settings
    pool_size: int = 10
    pool_max_overflow: int = 5
    pool_timeout: float = 30.0

    # Raw connection URL (takes precedence over individual settings)
    database_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables.

        Supports two configuration methods:
        1. DATABASE_URL: Standard connection string (preferred for managed PostgreSQL)
           Example: postgresql://user:pass@host:5432/dbname?sslmode=require
        2. Individual ARAGORA_PG_* variables for fine-grained control
        """
        # Check for DATABASE_URL first (standard for managed services)
        database_url = os.getenv("DATABASE_URL") or os.getenv("ARAGORA_DATABASE_URL")

        # Auto-detect backend from URL if provided
        backend = os.getenv("ARAGORA_DB_BACKEND", "sqlite")
        if database_url:
            if database_url.startswith(("postgresql://", "postgres://")):
                backend = "postgres"
            logger.info(f"Using DATABASE_URL, backend auto-detected as: {backend}")

        config = cls(
            backend=backend,
            database_url=database_url,
            sqlite_path=os.getenv("ARAGORA_SQLITE_PATH", "aragora.db"),
            pg_host=os.getenv("ARAGORA_PG_HOST", "localhost"),
            pg_port=int(os.getenv("ARAGORA_PG_PORT", "5432")),
            pg_database=os.getenv("ARAGORA_PG_DATABASE", "aragora"),
            pg_user=os.getenv("ARAGORA_PG_USER", "aragora"),
            pg_password=os.getenv("ARAGORA_PG_PASSWORD", ""),
            pg_ssl_mode=os.getenv("ARAGORA_PG_SSL_MODE", "require"),  # Default to require for managed
            pool_size=int(os.getenv("ARAGORA_DB_POOL_SIZE", "10")),
            pool_max_overflow=int(os.getenv("ARAGORA_DB_POOL_MAX_OVERFLOW", "5")),
            pool_timeout=float(os.getenv("ARAGORA_DB_POOL_TIMEOUT", "30.0")),
        )

        # Parse DATABASE_URL if provided
        if database_url:
            config._parse_database_url(database_url)

        return config

    def _parse_database_url(self, url: str) -> None:
        """Parse DATABASE_URL and populate individual fields."""
        import urllib.parse

        # Normalize postgres:// to postgresql://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)

        try:
            parsed = urllib.parse.urlparse(url)

            if parsed.hostname:
                self.pg_host = parsed.hostname
            if parsed.port:
                self.pg_port = parsed.port
            if parsed.path and parsed.path != "/":
                self.pg_database = parsed.path.lstrip("/")
            if parsed.username:
                self.pg_user = urllib.parse.unquote(parsed.username)
            if parsed.password:
                self.pg_password = urllib.parse.unquote(parsed.password)

            # Parse query parameters for SSL mode
            query_params = urllib.parse.parse_qs(parsed.query)
            if "sslmode" in query_params:
                self.pg_ssl_mode = query_params["sslmode"][0]

            logger.debug(
                f"Parsed DATABASE_URL: host={self.pg_host}, port={self.pg_port}, "
                f"database={self.pg_database}, sslmode={self.pg_ssl_mode}"
            )
        except Exception as e:
            logger.warning(f"Failed to parse DATABASE_URL: {e}")

    @property
    def pg_dsn(self) -> str:
        """PostgreSQL connection string."""
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_database} "
            f"user={self.pg_user} password={self.pg_password} sslmode={self.pg_ssl_mode}"
        )


class DatabaseBackend(ABC):
    """Abstract base class for database backends."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._lock = threading.Lock()

    @abstractmethod
    def connect(self) -> ConnectionProtocol:
        """Create a new database connection."""
        pass

    @abstractmethod
    @contextmanager
    def connection(self) -> Generator[ConnectionProtocol, None, None]:
        """Context manager for database connections with auto-commit/rollback."""
        pass

    @abstractmethod
    def execute(
        self, sql: str, params: Params = (), *, fetch: bool = False
    ) -> Union[CursorProtocol, list[Row]]:
        """Execute a SQL statement."""
        pass

    @abstractmethod
    def executemany(self, sql: str, params_list: list[Params]) -> int:
        """Execute a SQL statement with multiple parameter sets."""
        pass

    @abstractmethod
    def fetch_one(self, sql: str, params: Params = ()) -> Optional[Row]:
        """Execute query and fetch single row."""
        pass

    @abstractmethod
    def fetch_all(self, sql: str, params: Params = ()) -> list[Row]:
        """Execute query and fetch all rows."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if the database is accessible."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close all connections."""
        pass

    def translate_sql(self, sql: str) -> str:
        """Translate SQL for this backend (override if needed)."""
        return sql

    def translate_params(self, params: Params) -> Params:
        """Translate parameters for this backend (override if needed)."""
        return params

    @property
    @abstractmethod
    def placeholder(self) -> str:
        """Parameter placeholder for this backend (? or %s)."""
        pass


class SQLiteBackend(DatabaseBackend):
    """SQLite database backend with WAL mode and connection pooling."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._pool: list[sqlite3.Connection] = []
        self._pool_lock = threading.Lock()
        self._db_path = Path(config.sqlite_path)

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def placeholder(self) -> str:
        return "?"

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with WAL mode."""
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=self.config.pool_timeout,
            check_same_thread=False,
        )
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(f"PRAGMA busy_timeout = {int(self.config.pool_timeout * 1000)}")
        return conn

    def connect(self) -> sqlite3.Connection:
        """Get a connection from the pool or create a new one."""
        with self._pool_lock:
            while self._pool:
                conn = self._pool.pop()
                try:
                    conn.execute("SELECT 1")
                    return conn
                except sqlite3.Error:
                    # Connection is broken, discard
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass

        return self._create_connection()

    def _return_to_pool(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        with self._pool_lock:
            if len(self._pool) < self.config.pool_size:
                self._pool.append(conn)
                return

        # Pool is full, close the connection
        try:
            conn.close()
        except sqlite3.Error:
            pass

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"SQLite error: {e}")
            conn.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error in database operation: {e}", exc_info=True)
            conn.rollback()
            raise
        finally:
            self._return_to_pool(conn)

    def execute(
        self, sql: str, params: Params = (), *, fetch: bool = False
    ) -> Union[sqlite3.Cursor, list[Row]]:
        """Execute a SQL statement."""
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            if fetch:
                return cursor.fetchall()
            return cursor

    def executemany(self, sql: str, params_list: list[Params]) -> int:
        """Execute a SQL statement with multiple parameter sets."""
        with self.connection() as conn:
            cursor = conn.executemany(sql, params_list)
            return cursor.rowcount

    def fetch_one(self, sql: str, params: Params = ()) -> Optional[Row]:
        """Execute query and fetch single row."""
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchone()

    def fetch_all(self, sql: str, params: Params = ()) -> list[Row]:
        """Execute query and fetch all rows."""
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()

    def health_check(self) -> bool:
        """Check if the database is accessible."""
        try:
            with self.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"SQLite health check failed: {e}")
            return False

    def close(self) -> None:
        """Close all pooled connections."""
        with self._pool_lock:
            for conn in self._pool:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
            self._pool.clear()


class PostgresBackend(DatabaseBackend):
    """PostgreSQL database backend with asyncpg connection pooling.

    Falls back to psycopg2 for synchronous operations if asyncpg is not available.
    """

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._pool = None
        self._sync_pool: Any = None
        self._psycopg2: Any = None
        self._initialized = False

        # Try to import psycopg2
        try:
            import psycopg2
            import psycopg2.pool

            self._psycopg2 = psycopg2
            self._sync_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=config.pool_size,
                host=config.pg_host,
                port=config.pg_port,
                dbname=config.pg_database,
                user=config.pg_user,
                password=config.pg_password,
            )
            self._initialized = True
            logger.info(
                f"PostgreSQL backend initialized (psycopg2) - {config.pg_host}:{config.pg_port}"
            )
        except ImportError:
            logger.warning("psycopg2 not installed. Install with: pip install psycopg2-binary")

    @property
    def placeholder(self) -> str:
        return "%s"

    def translate_sql(self, sql: str) -> str:
        """Convert SQLite-style ? placeholders to PostgreSQL %s."""
        # Simple conversion - replace ? with %s
        # Note: This doesn't handle ? inside strings, but our SQL is parameterized
        return sql.replace("?", "%s")

    def connect(self) -> Any:
        """Get a connection from the pool."""
        if not self._initialized or self._sync_pool is None:
            raise ConfigurationError(
                component="PostgreSQL", reason="Backend not initialized. Call initialize() first"
            )

        return self._sync_pool.getconn()

    def _return_to_pool(self, conn: Any) -> None:
        """Return a connection to the pool."""
        if self._sync_pool:
            self._sync_pool.putconn(conn)

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Context manager for database connections."""
        if not self._initialized:
            raise ConfigurationError(
                component="PostgreSQL", reason="Backend not initialized. Call initialize() first"
            )

        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            logger.error(f"PostgreSQL error: {e}")
            conn.rollback()
            raise
        finally:
            self._return_to_pool(conn)

    def execute(
        self, sql: str, params: Params = (), *, fetch: bool = False
    ) -> Union[Any, list[Row]]:
        """Execute a SQL statement."""
        sql = self.translate_sql(sql)
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            if fetch:
                return cursor.fetchall()
            return cursor

    def executemany(self, sql: str, params_list: list[Params]) -> int:
        """Execute a SQL statement with multiple parameter sets."""
        sql = self.translate_sql(sql)
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(sql, params_list)
            return cursor.rowcount

    def fetch_one(self, sql: str, params: Params = ()) -> Optional[Row]:
        """Execute query and fetch single row."""
        sql = self.translate_sql(sql)
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchone()

    def fetch_all(self, sql: str, params: Params = ()) -> list[Row]:
        """Execute query and fetch all rows."""
        sql = self.translate_sql(sql)
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            return cursor.fetchall()

    def health_check(self) -> bool:
        """Check if the database is accessible."""
        if not self._initialized:
            return False

        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL health check failed: {e}")
            return False

    def close(self) -> None:
        """Close all connections."""
        if self._sync_pool:
            self._sync_pool.closeall()


# Global database instance
_database: Optional[DatabaseBackend] = None
_database_lock = threading.Lock()


def configure_database(config: Optional[DatabaseConfig] = None) -> DatabaseBackend:
    """Configure and return the global database instance.

    Args:
        config: Database configuration. If None, uses environment variables.

    Returns:
        Configured DatabaseBackend instance.
    """
    global _database

    if config is None:
        config = DatabaseConfig.from_env()

    with _database_lock:
        if _database is not None:
            _database.close()

        if config.backend == "postgres":
            _database = PostgresBackend(config)
        else:
            _database = SQLiteBackend(config)

        logger.info(f"Database configured: {config.backend}")
        return _database


def get_database() -> DatabaseBackend:
    """Get the global database instance.

    Returns the configured database backend, initializing with defaults if needed.

    Returns:
        DatabaseBackend instance.
    """
    global _database

    if _database is None:
        with _database_lock:
            if _database is None:
                _database = configure_database()

    return _database
