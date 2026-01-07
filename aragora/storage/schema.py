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

import logging
import re
import sqlite3
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Generator, Optional, Union

logger = logging.getLogger(__name__)


# Valid SQL column types (whitelist)
VALID_COLUMN_TYPES = frozenset({
    "TEXT", "INTEGER", "REAL", "BLOB", "NUMERIC",
    "VARCHAR", "CHAR", "BOOLEAN", "DATETIME", "TIMESTAMP",
})


def _validate_sql_identifier(name: str) -> bool:
    """Validate SQL identifier to prevent injection.

    Only allows alphanumeric characters and underscores.
    Must start with a letter or underscore.
    Maximum length of 128 characters.
    """
    if not name or len(name) > 128:
        return False
    return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name))


def _validate_column_type(col_type: str) -> bool:
    """Validate column type against whitelist."""
    # Normalize and check base type (handles "VARCHAR(255)" etc.)
    base_type = col_type.split('(')[0].strip().upper()
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
    if re.match(r'^-?\d+(\.\d+)?$', default.strip()):
        return True

    # Allow single-quoted strings (basic check - no embedded quotes)
    if re.match(r"^'[^']*'$", default.strip()):
        return True

    return False


# Default database connection timeout in seconds
DB_TIMEOUT = 30.0


def get_wal_connection(db_path: Union[str, Path], timeout: float = DB_TIMEOUT) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode enabled for better concurrency.

    WAL (Write-Ahead Logging) mode allows:
    - Multiple readers to operate concurrently with a single writer
    - Better performance for write-heavy workloads
    - Reduced lock contention in multi-threaded scenarios

    Args:
        db_path: Path to the SQLite database file
        timeout: Connection timeout in seconds (default: 30.0)

    Returns:
        A sqlite3.Connection configured for WAL mode
    """
    conn = sqlite3.connect(db_path, timeout=timeout)
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
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS _schema_versions (
                module TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def get_version(self) -> int:
        """Get the current schema version for this module."""
        cursor = self.conn.execute(
            "SELECT version FROM _schema_versions WHERE module = ?",
            (self.module_name,)
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def set_version(self, version: int) -> None:
        """Set the schema version for this module."""
        self.conn.execute("""
            INSERT INTO _schema_versions (module, version, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(module) DO UPDATE SET
                version = excluded.version,
                updated_at = CURRENT_TIMESTAMP
        """, (self.module_name, version))
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
                desc = migration.description or f"v{migration.from_version} -> v{migration.to_version}"
                logger.info(f"[{self.module_name}] Running migration: {desc}")
                try:
                    migration.apply(self.conn)
                    self.conn.commit()
                    current = migration.to_version
                    self.set_version(current)
                except Exception as e:
                    self.conn.rollback()
                    logger.error(f"[{self.module_name}] Migration to v{migration.to_version} failed: {e}")
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
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
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

    def __init__(self, db_path: Union[str, Path], timeout: float = DB_TIMEOUT):
        """Initialize the DatabaseManager.

        Note: Use get_instance() instead of direct instantiation to ensure
        singleton behavior per database path.

        Args:
            db_path: Path to the SQLite database file
            timeout: Connection timeout in seconds
        """
        self.db_path = str(Path(db_path).resolve())
        self.timeout = timeout
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._thread_local = threading.local()

    @classmethod
    def get_instance(cls, db_path: Union[str, Path], timeout: float = DB_TIMEOUT) -> "DatabaseManager":
        """Get or create a DatabaseManager instance for the given path.

        This is the recommended way to obtain a DatabaseManager. It ensures
        only one manager exists per database path (singleton pattern).

        Args:
            db_path: Path to the SQLite database file
            timeout: Connection timeout in seconds

        Returns:
            DatabaseManager instance for the given path
        """
        resolved_path = str(Path(db_path).resolve())

        with cls._instances_lock:
            if resolved_path not in cls._instances:
                cls._instances[resolved_path] = cls(db_path, timeout)
                logger.debug(f"Created DatabaseManager for {resolved_path}")
            return cls._instances[resolved_path]

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all cached instances. Useful for testing."""
        with cls._instances_lock:
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
        except Exception:
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
        except Exception:
            conn.execute("ROLLBACK")
            raise

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

    def close(self) -> None:
        """Close the database connection.

        This is called automatically when the manager is garbage collected,
        but can be called manually if needed.
        """
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                    logger.debug(f"Closed connection to {self.db_path}")
                except Exception as e:
                    logger.warning(f"Error closing connection to {self.db_path}: {e}")
                finally:
                    self._conn = None

    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        self.close()

    def __repr__(self) -> str:
        return f"DatabaseManager({self.db_path!r})"
