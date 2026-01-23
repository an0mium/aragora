"""
Schema versioning for SQLite databases.

Provides a simple migration framework for tracking and upgrading
database schemas across versions.

Usage:
    from aragora.storage.schema import SchemaManager

    manager = SchemaManager(conn, "my_module", current_version=2)

    manager.register_migration(1, 2, '''
        ALTER TABLE my_table ADD COLUMN new_field TEXT;
    ''')

    manager.ensure_schema()  # Runs any pending migrations
"""

__all__ = [
    "VALID_COLUMN_TYPES",
    "DB_TIMEOUT",
    "get_wal_connection",
    "Migration",
    "SchemaManager",
    "safe_add_column",
    "DatabaseManager",
    "PERFORMANCE_INDEXES",
    "create_performance_indexes",
    "analyze_tables",
    "ConnectionPool",
]

import logging
import re
import sqlite3
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator, Optional, Union

from aragora.exceptions import DatabaseError
from aragora.utils.timeouts import timed_lock

logger = logging.getLogger(__name__)


def _safe_log(level: int, msg: str) -> None:
    """Log a message safely, handling Python shutdown gracefully.

    During interpreter shutdown, sys.meta_path becomes None and logging
    may fail. This helper silently ignores such errors.
    """
    if sys.meta_path is None:
        # Python is shutting down, don't try to log
        return
    try:
        logger.log(level, msg)
    except Exception:
        # Logging failed (likely during shutdown), ignore silently
        pass


# Valid SQL column types (whitelist)
VALID_COLUMN_TYPES = frozenset(
    {
        "TEXT",
        "INTEGER",
        "REAL",
        "BLOB",
        "NUMERIC",
        "VARCHAR",
        "CHAR",
        "BOOLEAN",
        "DATETIME",
        "TIMESTAMP",
    }
)


def _validate_sql_identifier(name: str) -> bool:
    """Validate SQL identifier to prevent injection.

    Only allows alphanumeric characters and underscores.
    Must start with a letter or underscore.
    Maximum length of 128 characters.
    """
    if not name or len(name) > 128:
        return False
    return bool(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name))


def _validate_column_type(col_type: str) -> bool:
    """Validate column type against whitelist."""
    # Normalize and check base type (handles "VARCHAR(255)" etc.)
    base_type = col_type.split("(")[0].strip().upper()
    return base_type in VALID_COLUMN_TYPES


def _validate_default_value(default: str) -> bool:
    """Validate default value to prevent injection.

    Allows:
    - NULL
    - Numeric literals (integers, floats)
    - Single-quoted strings (properly escaped)
    - SQL functions: CURRENT_TIMESTAMP, CURRENT_DATE, CURRENT_TIME
    """
    if default is None:
        return True

    default_upper = default.strip().upper()

    # Allow NULL
    if default_upper == "NULL":
        return True

    # Allow common SQL functions
    if default_upper in ("CURRENT_TIMESTAMP", "CURRENT_DATE", "CURRENT_TIME"):
        return True

    # Allow numeric literals (integers and floats)
    if re.match(r"^-?\d+(\.\d+)?$", default.strip()):
        return True

    # Allow single-quoted strings (basic check - no embedded quotes)
    if re.match(r"^'[^']*'$", default.strip()):
        return True

    return False


# Default database connection timeout in seconds
DB_TIMEOUT = 30.0


def get_wal_connection(
    db_path: Union[str, Path],
    timeout: float = DB_TIMEOUT,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode enabled for better concurrency.

    WAL (Write-Ahead Logging) mode allows:
    - Multiple readers to operate concurrently with a single writer
    - Better performance for write-heavy workloads
    - Reduced lock contention in multi-threaded scenarios

    Args:
        db_path: Path to the SQLite database file
        timeout: Connection timeout in seconds (default: 30.0)
        check_same_thread: If False, allows connection use across threads (default: True)

    Returns:
        A sqlite3.Connection configured for WAL mode
    """
    conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=check_same_thread)
    # Enable dict-style row access (row["column_name"])
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    # Use NORMAL synchronous mode (safe with WAL, faster than FULL)
    conn.execute("PRAGMA synchronous=NORMAL")
    # Set busy timeout in milliseconds
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return conn


@dataclass
class Migration:
    """A database migration from one version to another."""

    from_version: int
    to_version: int
    sql: Optional[str] = None
    function: Optional[Callable[[sqlite3.Connection], None]] = None
    description: str = ""

    def apply(self, conn: sqlite3.Connection) -> None:
        """Apply this migration to the database."""
        if self.sql:
            conn.executescript(self.sql)
        elif self.function:
            self.function(conn)
        else:
            raise ValueError("Migration must have either sql or function")


class SchemaManager:
    """
    Manages schema versioning and migrations for a SQLite database.

    Tracks version in a _schema_versions table and runs pending migrations
    to bring the database up to the current version.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        module_name: str,
        current_version: int = 1,
    ):
        """
        Initialize schema manager.

        Args:
            conn: SQLite connection
            module_name: Unique identifier for this schema (e.g., "elo", "memory")
            current_version: The version this code expects
        """
        self.conn = conn
        self.module_name = module_name
        self.current_version = current_version
        self.migrations: list[Migration] = []

        self._ensure_version_table()

    def _ensure_version_table(self) -> None:
        """Create the schema versions table if it doesn't exist."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS _schema_versions (
                module TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.commit()

    def get_version(self) -> int:
        """Get the current schema version for this module."""
        cursor = self.conn.execute(
            "SELECT version FROM _schema_versions WHERE module = ?", (self.module_name,)
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def set_version(self, version: int) -> None:
        """Set the schema version for this module."""
        # Use INSERT OR REPLACE for SQLite 3.7 compatibility (EC2)
        # This requires module to be the primary key, which it is
        self.conn.execute(
            """
            INSERT OR REPLACE INTO _schema_versions (module, version, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        """,
            (self.module_name, version),
        )
        self.conn.commit()

    def register_migration(
        self,
        from_version: int,
        to_version: int,
        sql: Optional[str] = None,
        function: Optional[Callable[[sqlite3.Connection], None]] = None,
        description: str = "",
    ) -> None:
        """
        Register a migration between versions.

        Args:
            from_version: Version to migrate from
            to_version: Version to migrate to
            sql: SQL script to execute (either sql or function required)
            function: Python function to execute
            description: Human-readable description
        """
        migration = Migration(
            from_version=from_version,
            to_version=to_version,
            sql=sql,
            function=function,
            description=description,
        )
        self.migrations.append(migration)
        # Keep migrations sorted by from_version
        self.migrations.sort(key=lambda m: m.from_version)

    def get_pending_migrations(self) -> list[Migration]:
        """Get list of migrations needed to reach current version."""
        current = self.get_version()
        pending = []

        for migration in self.migrations:
            if migration.from_version >= current and migration.to_version <= self.current_version:
                pending.append(migration)

        return pending

    def ensure_schema(self, initial_schema: Optional[str] = None) -> bool:
        """
        Ensure the database schema is up to date.

        Args:
            initial_schema: SQL to create initial tables (version 1)

        Returns:
            True if migrations were applied, False if already up to date
        """
        current = self.get_version()

        applied = False

        if current == 0 and initial_schema:
            # Fresh database - create initial schema
            logger.info(f"[{self.module_name}] Creating initial schema (v1)")
            try:
                self.conn.executescript(initial_schema)
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                logger.error(f"[{self.module_name}] Schema initialization failed: {e}")
                raise
            self.set_version(1)
            current = 1
            applied = True

        if current == self.current_version:
            return applied  # Already at target version

        if current > self.current_version:
            logger.warning(
                f"[{self.module_name}] Database version ({current}) is newer than "
                f"code version ({self.current_version}). Skipping migrations."
            )
            return False

        # Apply pending migrations
        pending = self.get_pending_migrations()
        if not pending:
            # No registered migrations, just update version
            self.set_version(self.current_version)
            return True

        for migration in pending:
            if migration.from_version == current:
                desc = (
                    migration.description or f"v{migration.from_version} -> v{migration.to_version}"
                )
                logger.info(f"[{self.module_name}] Running migration: {desc}")
                try:
                    migration.apply(self.conn)
                    self.conn.commit()
                    current = migration.to_version
                    self.set_version(current)
                except Exception as e:
                    self.conn.rollback()
                    logger.error(
                        f"[{self.module_name}] Migration to v{migration.to_version} failed: {e}"
                    )
                    raise

        return True

    def validate_schema(self, expected_tables: list[str]) -> dict:
        """
        Validate that expected tables exist.

        Args:
            expected_tables: List of table names that should exist

        Returns:
            Dict with validation results
        """
        cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing = {row[0] for row in cursor.fetchall()}

        missing = [t for t in expected_tables if t not in existing]
        extra = [t for t in existing if t not in expected_tables and not t.startswith("_")]

        return {
            "valid": len(missing) == 0,
            "missing": missing,
            "extra": extra,
            "version": self.get_version(),
        }


def safe_add_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    column_type: str,
    default: Optional[str] = None,
) -> bool:
    """
    Safely add a column to a table if it doesn't exist.

    Validates all parameters to prevent SQL injection.

    Args:
        conn: SQLite connection
        table: Table name (alphanumeric and underscores only)
        column: Column name to add (alphanumeric and underscores only)
        column_type: SQL type (e.g., "TEXT", "INTEGER") - must be whitelisted
        default: Optional default value (numeric, quoted string, or SQL function)

    Returns:
        True if column was added, False if it already existed

    Raises:
        ValueError: If any parameter fails validation
    """
    # Validate all parameters to prevent SQL injection
    if not _validate_sql_identifier(table):
        raise ValueError(f"Invalid table name: {table!r}")
    if not _validate_sql_identifier(column):
        raise ValueError(f"Invalid column name: {column!r}")
    if not _validate_column_type(column_type):
        raise ValueError(f"Invalid column type: {column_type!r}")
    if default is not None and not _validate_default_value(default):
        raise ValueError(f"Invalid default value: {default!r}")

    # Check if column exists
    cursor = conn.execute(f"PRAGMA table_info({table})")
    columns = {row[1] for row in cursor.fetchall()}

    if column in columns:
        return False

    # Add the column (safe after validation)
    sql = f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
    if default is not None:
        sql += f" DEFAULT {default}"

    conn.execute(sql)
    conn.commit()
    logger.debug(f"Added column {column} to {table}")
    return True


class DatabaseManager:
    """
    Centralized database connection manager with singleton pattern.

    Provides:
    - Single instance per database path (thread-safe)
    - WAL mode for better concurrency
    - Connection reuse to avoid repeated open/close overhead
    - Automatic cleanup of idle connections
    - Context manager support for transactions

    Usage:
        # Get manager instance (singleton per path)
        manager = DatabaseManager.get_instance("/path/to/db.db")

        # Use context manager for automatic commit/rollback
        with manager.connection() as conn:
            conn.execute("INSERT INTO ...")

        # Or get raw connection for manual management
        conn = manager.get_connection()
        try:
            conn.execute("...")
            conn.commit()
        finally:
            # Connection is managed by DatabaseManager, no need to close
            pass
    """

    _instances: dict[str, "DatabaseManager"] = {}
    _instances_lock = threading.Lock()

    # Default pool size for connection pooling
    # Increased from 5 to 20 for better production concurrency
    DEFAULT_POOL_SIZE = 20

    def __init__(
        self,
        db_path: Union[str, Path],
        timeout: float = DB_TIMEOUT,
        pool_size: int = DEFAULT_POOL_SIZE,
    ):
        """Initialize the DatabaseManager.

        Note: Use get_instance() instead of direct instantiation to ensure
        singleton behavior per database path.

        Args:
            db_path: Path to the SQLite database file
            timeout: Connection timeout in seconds
            pool_size: Maximum number of pooled connections (default 20)
        """
        self.db_path = str(Path(db_path).resolve())
        self.timeout = timeout
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._thread_local = threading.local()

        # Connection pool for fresh_connection()
        self._pool: list[sqlite3.Connection] = []
        self._pool_size = pool_size
        self._pool_lock = threading.Lock()
        self._pool_stats = {"hits": 0, "misses": 0, "returns": 0, "created": 0, "closed": 0}

    def _get_pooled_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool or create a new one.

        Thread-safe. If the pool has idle connections, returns one.
        Otherwise creates a new WAL-mode connection.

        Returns:
            sqlite3.Connection configured for WAL mode
        """
        with self._pool_lock:
            while self._pool:
                conn = self._pool.pop()
                # Validate connection is still usable
                try:
                    conn.execute("SELECT 1")
                    self._pool_stats["hits"] += 1
                    return conn
                except sqlite3.Error as e:
                    # Connection is broken, discard and try next
                    logger.debug(f"Pooled connection validation failed, discarding: {e}")
                    self._pool_stats["closed"] += 1
                    try:
                        conn.close()
                    except sqlite3.Error as close_err:
                        logger.debug(f"Error closing broken pooled connection: {close_err}")

            # No valid pooled connections, create new one
            self._pool_stats["misses"] += 1
            self._pool_stats["created"] += 1

        # Create outside lock to avoid holding it during I/O
        return get_wal_connection(self.db_path, self.timeout, check_same_thread=False)

    def _return_to_pool(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool if space available.

        Thread-safe. If the pool is full, the connection is closed instead.

        Args:
            conn: Connection to return to the pool
        """
        with self._pool_lock:
            if len(self._pool) < self._pool_size:
                self._pool.append(conn)
                self._pool_stats["returns"] += 1
                return

        # Pool is full, close the connection
        self._pool_stats["closed"] += 1
        try:
            conn.close()
        except sqlite3.Error as e:
            logger.debug(f"Error closing excess pooled connection: {e}")

    def pool_stats(self) -> dict[str, int]:
        """Get connection pool statistics.

        Returns:
            Dict with pool stats:
            - hits: Connections retrieved from pool
            - misses: Times pool was empty (new connection created)
            - returns: Connections returned to pool
            - created: Total connections created
            - closed: Connections closed (broken or pool full)
            - pool_size: Current number of idle connections
            - max_pool_size: Maximum pool capacity
        """
        with self._pool_lock:
            return {
                **self._pool_stats.copy(),
                "pool_size": len(self._pool),
                "max_pool_size": self._pool_size,
            }

    @classmethod
    def get_instance(
        cls, db_path: Union[str, Path], timeout: float = DB_TIMEOUT
    ) -> "DatabaseManager":
        """Get or create a DatabaseManager instance for the given path.

        This is the recommended way to obtain a DatabaseManager. It ensures
        only one manager exists per database path (singleton pattern).

        Args:
            db_path: Path to the SQLite database file
            timeout: Connection timeout in seconds

        Returns:
            DatabaseManager instance for the given path

        Raises:
            TimeoutError: If unable to acquire the instance lock within 30 seconds
        """
        resolved_path = str(Path(db_path).resolve())

        with timed_lock(cls._instances_lock, timeout=30.0, name="DatabaseManager.instances"):
            if resolved_path not in cls._instances:
                cls._instances[resolved_path] = cls(db_path, timeout)
                logger.debug(f"Created DatabaseManager for {resolved_path}")
            return cls._instances[resolved_path]

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached instances. Useful for testing.

        Raises:
            TimeoutError: If unable to acquire the instance lock within 30 seconds
        """
        with timed_lock(cls._instances_lock, timeout=30.0, name="DatabaseManager.instances"):
            for manager in cls._instances.values():
                manager.close()
            cls._instances.clear()
            logger.debug("Cleared all DatabaseManager instances")

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection.

        Returns a connection configured for WAL mode. The connection is
        managed by this DatabaseManager and should not be closed manually.

        Returns:
            sqlite3.Connection configured for WAL mode
        """
        with self._lock:
            if self._conn is None:
                self._conn = get_wal_connection(self.db_path, self.timeout)
                logger.debug(f"Opened connection to {self.db_path}")
            return self._conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database operations with automatic commit/rollback.

        Commits on success, rolls back on exception.

        Usage:
            with manager.connection() as conn:
                conn.execute("INSERT INTO ...")
                # Auto-commits on exit

        Yields:
            sqlite3.Connection for database operations
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error in DatabaseManager.connection(): {e}", exc_info=True)
            conn.rollback()
            raise
        except Exception as e:
            # Rollback on any exception from user code, then re-raise unchanged
            logger.warning(
                f"Non-database exception in DatabaseManager.connection(), rolling back: {type(e).__name__}: {e}"
            )
            conn.rollback()
            raise

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Explicit transaction context manager.

        Same as connection() but makes the transaction intent clearer.

        Yields:
            sqlite3.Connection within a transaction
        """
        conn = self.get_connection()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.execute("COMMIT")
        except sqlite3.Error as e:
            logger.error(f"Database error in DatabaseManager.transaction(): {e}", exc_info=True)
            conn.execute("ROLLBACK")
            raise
        except Exception as e:
            # Rollback on any exception from user code, then re-raise unchanged
            logger.warning(
                f"Non-database exception in transaction context, rolling back: {type(e).__name__}: {e}"
            )
            conn.execute("ROLLBACK")
            raise

    @contextmanager
    def fresh_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for a fresh per-operation connection.

        Uses connection pooling to avoid repeated open/close overhead.
        Thread-safe and suitable for multi-threaded access patterns.

        Connections are automatically returned to the pool after use.
        For single-threaded use with connection reuse, use connection() instead.

        Yields:
            sqlite3.Connection for database operations (returned to pool on exit)
        """
        conn = self._get_pooled_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            logger.error(
                f"Database error in DatabaseManager.fresh_connection(): {e}", exc_info=True
            )
            conn.rollback()
            raise
        except Exception as e:
            logger.warning(
                f"Non-database exception in fresh_connection context, rolling back: {type(e).__name__}: {e}"
            )
            conn.rollback()
            raise
        finally:
            self._return_to_pool(conn)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a SQL statement.

        Convenience method for simple queries. For transactions, use
        the connection() context manager instead.

        Args:
            sql: SQL statement to execute
            params: Parameters for the SQL statement

        Returns:
            sqlite3.Cursor with the results
        """
        return self.get_connection().execute(sql, params)

    def executemany(self, sql: str, params_list: list[tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement to execute
            params_list: List of parameter tuples

        Returns:
            sqlite3.Cursor with the results
        """
        return self.get_connection().executemany(sql, params_list)

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Execute query and fetch single row.

        Convenience method for simple SELECT queries that expect one row.

        Args:
            sql: SQL query to execute
            params: Parameters for the SQL statement

        Returns:
            Single row as tuple, or None if no results
        """
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchone()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        """Execute query and fetch all rows.

        Convenience method for SELECT queries returning multiple rows.

        Args:
            sql: SQL query to execute
            params: Parameters for the SQL statement

        Returns:
            List of rows as tuples
        """
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()

    def fetch_many(self, sql: str, params: tuple = (), size: int = 100) -> list[tuple]:
        """Execute query and fetch up to 'size' rows.

        Convenience method for paginated queries.

        Args:
            sql: SQL query to execute
            params: Parameters for the SQL statement
            size: Maximum number of rows to return

        Returns:
            List of up to 'size' rows as tuples
        """
        with self.connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchmany(size)

    def close(self) -> None:
        """Close all database connections including pooled ones.

        This is called automatically when the manager is garbage collected,
        but can be called manually if needed. Uses _safe_log to handle
        Python shutdown gracefully.
        """
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                    _safe_log(logging.DEBUG, f"Closed connection to {self.db_path}")
                except sqlite3.Error as e:
                    _safe_log(logging.WARNING, f"Error closing connection to {self.db_path}: {e}")
                finally:
                    self._conn = None

        # Close all pooled connections
        with self._pool_lock:
            pool_size = len(self._pool)
            for conn in self._pool:
                try:
                    conn.close()
                except sqlite3.Error as e:
                    _safe_log(logging.DEBUG, f"Error closing pooled connection: {e}")
            self._pool.clear()
            if pool_size > 0:
                _safe_log(logging.DEBUG, f"Closed {pool_size} pooled connections to {self.db_path}")

    def __del__(self) -> None:
        """Ensure connection is closed on garbage collection.

        Note: During interpreter shutdown, logging handlers may already be closed,
        so we wrap in try-except to avoid errors when the logging system is torn down.
        """
        try:
            self.close()
        except Exception:
            # Silently ignore errors during shutdown - logging may be unavailable
            # and we just want to close connections without raising
            pass

    def __repr__(self) -> str:
        return f"DatabaseManager({self.db_path!r})"


# ============================================================================
# Performance Indexes
# ============================================================================

# Index definitions for commonly queried columns
# Format: (table_name, index_name, column_expression)
PERFORMANCE_INDEXES = [
    # Memory store indexes for agent/debate lookups
    ("memory_store", "idx_memory_agent_debate", "agent_name, debate_id"),
    ("memory_store", "idx_memory_timestamp", "timestamp"),
    # Continuum memory indexes for time-based queries
    ("continuum_memory", "idx_continuum_timestamp", "timestamp"),
    ("continuum_memory", "idx_continuum_tier", "tier"),
    # Votes table indexes
    ("votes", "idx_votes_agent_debate", "agent_name, debate_id"),
    ("votes", "idx_votes_debate_round", "debate_id, round_num"),
    # ELO matches for history lookups
    ("matches", "idx_matches_agent", "agent_name"),
    ("matches", "idx_matches_timestamp", "timestamp"),
    # Debates table for listing
    ("debates", "idx_debates_created", "created_at"),
    ("debates", "idx_debates_status", "status"),
    # Composite index for filtering by status and sorting by time (common pattern)
    ("debates", "idx_debates_status_created", "status, created_at"),
    # Consensus memory for debate lookups
    ("consensus_memory", "idx_consensus_debate", "debate_id"),
    # ELO ratings for domain-specific queries
    ("ratings", "idx_ratings_elo", "elo"),
]


def create_performance_indexes(
    conn: sqlite3.Connection, tables_to_index: list[str] | None = None
) -> dict:
    """
    Create performance indexes on frequently-queried columns.

    This function is idempotent - it uses CREATE INDEX IF NOT EXISTS
    so it's safe to call multiple times.

    Args:
        conn: SQLite connection
        tables_to_index: Optional list of table names to index. If None, indexes all.

    Returns:
        Dict with results:
        {
            "created": [...],  # Indexes created
            "skipped": [...],  # Indexes that already existed
            "errors": [...],   # Tables that don't exist or had errors
        }
    """
    created = []
    skipped = []
    errors = []

    # Get list of existing tables
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    existing_tables = {row[0] for row in cursor.fetchall()}

    # Get list of existing indexes
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    existing_indexes = {row[0] for row in cursor.fetchall()}

    for table, index_name, columns in PERFORMANCE_INDEXES:
        # Filter by tables_to_index if specified
        if tables_to_index and table not in tables_to_index:
            continue

        # Skip if table doesn't exist
        if table not in existing_tables:
            errors.append(f"{index_name}: table '{table}' does not exist")
            continue

        # Skip if index already exists
        if index_name in existing_indexes:
            skipped.append(index_name)
            continue

        # Validate column names before creating index
        for col in columns.split(","):
            col = col.strip()
            if not _validate_sql_identifier(col):
                errors.append(f"{index_name}: invalid column name '{col}'")
                continue

        # Create the index
        try:
            sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table}({columns})"
            conn.execute(sql)
            created.append(index_name)
            logger.info(f"Created index {index_name} on {table}({columns})")
        except sqlite3.Error as e:
            errors.append(f"{index_name}: {e}")

    conn.commit()

    return {
        "created": created,
        "skipped": skipped,
        "errors": errors,
    }


def analyze_tables(conn: sqlite3.Connection) -> None:
    """
    Run ANALYZE on all tables to update query planner statistics.

    Should be called after bulk inserts or after creating indexes
    to help SQLite choose optimal query plans.
    """
    conn.execute("ANALYZE")
    conn.commit()
    logger.info("Ran ANALYZE on database")


class ConnectionPool:
    """
    Thread-safe SQLite connection pool.

    Maintains a pool of reusable connections for high-concurrency scenarios.
    Each connection is configured with WAL mode for better concurrent access.

    Usage:
        pool = ConnectionPool("/path/to/db.db", max_connections=10)

        # Acquire and release connections
        with pool.connection() as conn:
            conn.execute("SELECT ...")

        # Or manually
        conn = pool.acquire()
        try:
            conn.execute("...")
        finally:
            pool.release(conn)

        # Get pool statistics
        stats = pool.stats()
        print(f"Active: {stats['active']}, Idle: {stats['idle']}")
    """

    def __init__(
        self,
        db_path: Union[str, Path],
        max_connections: int = 10,
        timeout: float = DB_TIMEOUT,
    ):
        """Initialize the connection pool.

        Args:
            db_path: Path to the SQLite database file
            max_connections: Maximum number of connections in the pool
            timeout: Connection timeout in seconds
        """
        self.db_path = str(Path(db_path).resolve())
        self.max_connections = max_connections
        self.timeout = timeout

        self._idle: list[sqlite3.Connection] = []
        self._active: set[int] = set()  # Track active connection ids
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False

    def acquire(self, timeout: Optional[float] = None) -> sqlite3.Connection:
        """Acquire a connection from the pool.

        Blocks if no connections are available and pool is at max capacity.

        Args:
            timeout: Max time to wait for a connection (None = wait forever)

        Returns:
            A sqlite3.Connection from the pool

        Raises:
            TimeoutError: If timeout is reached while waiting for a connection
            RuntimeError: If pool is closed
        """
        wait_timeout = timeout or self.timeout

        with self._condition:
            if self._closed:
                raise DatabaseError("Connection pool is closed")

            # Wait for an available connection
            waited = 0.0

            while True:
                # Try to get an idle connection
                if self._idle:
                    conn = self._idle.pop()
                    # Validate connection is still usable
                    try:
                        conn.execute("SELECT 1")
                        self._active.add(id(conn))
                        return conn
                    except sqlite3.Error:
                        # Connection is broken, discard it
                        logger.debug(f"Discarded broken pooled connection to {self.db_path}")
                        continue

                # Create a new connection if under limit
                if len(self._active) < self.max_connections:
                    # Use check_same_thread=False since pool is designed for multi-threaded use
                    conn = get_wal_connection(self.db_path, self.timeout, check_same_thread=False)
                    self._active.add(id(conn))
                    logger.debug(
                        f"Created new pooled connection to {self.db_path} "
                        f"(active: {len(self._active)}/{self.max_connections})"
                    )
                    return conn

                # Wait for a connection to be released
                remaining = wait_timeout - waited
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timeout waiting for connection to {self.db_path} "
                        f"(active: {len(self._active)}, max: {self.max_connections})"
                    )

                self._condition.wait(timeout=min(remaining, 1.0))
                waited += 1.0

    def release(self, conn: sqlite3.Connection) -> None:
        """Release a connection back to the pool.

        Args:
            conn: Connection to release
        """
        with self._condition:
            conn_id = id(conn)
            if conn_id not in self._active:
                logger.warning("Attempted to release connection not from this pool")
                return

            self._active.discard(conn_id)

            if self._closed:
                # Pool is closed, close the connection
                try:
                    conn.close()
                except sqlite3.Error as e:
                    logger.debug(f"Error closing connection on pool release: {e}")
            else:
                # Return to idle pool
                self._idle.append(conn)
                self._condition.notify()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for acquiring and releasing connections.

        Automatically handles commit/rollback semantics.

        Yields:
            sqlite3.Connection from the pool
        """
        conn = self.acquire()
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Database error in ConnectionPool.connection(): {e}", exc_info=True)
            conn.rollback()
            raise
        except Exception as e:
            # Rollback on any exception from user code, then re-raise unchanged
            logger.warning(
                f"Non-database exception in ConnectionPool.connection(), rolling back: {type(e).__name__}: {e}"
            )
            conn.rollback()
            raise
        finally:
            self.release(conn)

    def stats(self) -> dict[str, int]:
        """Get pool statistics.

        Returns:
            Dict with 'active', 'idle', and 'total' counts
        """
        with self._lock:
            return {
                "active": len(self._active),
                "idle": len(self._idle),
                "total": len(self._active) + len(self._idle),
                "max": self.max_connections,
            }

    def close(self) -> None:
        """Close all connections in the pool.

        Uses _safe_log to handle Python shutdown gracefully.
        """
        with self._condition:
            self._closed = True

            # Close all idle connections
            for conn in self._idle:
                try:
                    conn.close()
                except sqlite3.Error as e:
                    _safe_log(logging.WARNING, f"Error closing pooled connection: {e}")

            self._idle.clear()
            _safe_log(
                logging.DEBUG,
                f"Closed connection pool for {self.db_path} "
                f"(active: {len(self._active)} connections may still be in use)",
            )

            # Notify waiters so they can fail
            self._condition.notify_all()

    def __del__(self) -> None:
        """Ensure pool is closed on garbage collection.

        Note: During interpreter shutdown, modules may already be torn down,
        so we wrap in try-except to avoid errors.
        """
        try:
            if not self._closed:
                self.close()
        except Exception:
            # Silently ignore errors during shutdown
            pass

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"ConnectionPool({self.db_path!r}, "
            f"active={stats['active']}, idle={stats['idle']}, max={self.max_connections})"
        )
