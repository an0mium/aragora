"""
PostgreSQL store implementation for Aragora.

Provides async PostgreSQL-backed storage with connection pooling for production
deployments that need horizontal scaling and concurrent writes.

Usage:
    from aragora.storage.postgres_store import PostgresStore, get_postgres_pool

    # Initialize pool at startup
    pool = await get_postgres_pool()

    class MyStore(PostgresStore):
        SCHEMA_NAME = "my_store"
        SCHEMA_VERSION = 1

        INITIAL_SCHEMA = '''
            CREATE TABLE IF NOT EXISTS items (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_items_name ON items(name);
        '''

    store = MyStore(pool)
    await store.initialize()
"""

import logging
import os
from abc import ABC
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# Optional asyncpg import - graceful degradation
try:
    import asyncpg
    from asyncpg import Connection, Pool

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None
    Pool = Any
    Connection = Any
    ASYNCPG_AVAILABLE = False
    logger.debug("asyncpg not available, PostgreSQL backend disabled")


# Global pool singleton
_pool: Optional["Pool"] = None


async def get_postgres_pool(
    dsn: Optional[str] = None,
    min_size: int = 5,
    max_size: int = 20,
    command_timeout: float = 60.0,
    statement_timeout: int = 60,
    pool_recycle: int = 1800,
) -> "Pool":
    """
    Get or create the global PostgreSQL connection pool.

    Args:
        dsn: PostgreSQL connection string. If not provided, uses
             ARAGORA_POSTGRES_DSN or DATABASE_URL env vars.
        min_size: Minimum pool connections (default 5)
        max_size: Maximum pool connections (default 20)
        command_timeout: Max time (seconds) for any single command (default 60)
        statement_timeout: PostgreSQL statement_timeout in seconds (default 60)
        pool_recycle: Recycle connections older than this (seconds, default 1800)

    Returns:
        Connection pool instance

    Raises:
        RuntimeError: If asyncpg is not installed or connection fails
    """
    global _pool

    if not ASYNCPG_AVAILABLE:
        raise RuntimeError(
            "PostgreSQL backend requires 'asyncpg' package. "
            "Install with: pip install aragora[postgres] or pip install asyncpg"
        )

    if _pool is not None:
        return _pool

    # Get DSN from environment or secrets manager if not provided
    if dsn is None:
        dsn = os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")

        # Try secrets manager as fallback
        if not dsn:
            try:
                from aragora.config.secrets import get_secret

                dsn = get_secret("ARAGORA_POSTGRES_DSN") or get_secret("DATABASE_URL")
            except ImportError:
                pass  # secrets module not available

    if not dsn:
        raise RuntimeError(
            "PostgreSQL DSN not configured. Set ARAGORA_POSTGRES_DSN or DATABASE_URL "
            "in environment, AWS Secrets Manager, or pass dsn parameter."
        )

    # Connection initialization callback to set session parameters
    async def init_connection(conn):
        # Set statement_timeout to prevent runaway queries
        await conn.execute(f"SET statement_timeout = '{statement_timeout}s'")
        # Set idle_in_transaction_session_timeout to prevent abandoned transactions
        await conn.execute("SET idle_in_transaction_session_timeout = '300s'")

    logger.info(
        f"Creating PostgreSQL pool (min={min_size}, max={max_size}, "
        f"command_timeout={command_timeout}s, statement_timeout={statement_timeout}s)"
    )
    _pool = await asyncpg.create_pool(
        dsn,
        min_size=min_size,
        max_size=max_size,
        command_timeout=command_timeout,
        max_inactive_connection_lifetime=pool_recycle,
        init=init_connection,
    )
    return _pool


async def get_postgres_pool_from_settings() -> "Pool":
    """
    Get or create PostgreSQL connection pool using centralized settings.

    This is the recommended way to get the pool in application code,
    as it uses the settings from aragora.config.settings.DatabaseSettings.

    Returns:
        Connection pool instance

    Raises:
        RuntimeError: If asyncpg is not installed or PostgreSQL not configured

    Example:
        from aragora.storage.postgres_store import get_postgres_pool_from_settings

        pool = await get_postgres_pool_from_settings()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM debates")
    """
    from aragora.config import get_settings

    settings = get_settings()
    db_settings = settings.database

    if not db_settings.is_postgresql:
        raise RuntimeError(
            "PostgreSQL backend not configured. Set ARAGORA_DB_BACKEND=postgresql "
            "and DATABASE_URL environment variable."
        )

    return await get_postgres_pool(
        dsn=db_settings.url,
        min_size=max(1, db_settings.pool_size // 4),  # min is ~25% of max
        max_size=db_settings.pool_size,
    )


async def close_postgres_pool() -> None:
    """Close the global PostgreSQL connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL pool closed")


class PostgresStore(ABC):
    """
    Base class for PostgreSQL-backed stores with schema management.

    Provides:
    - Async connection management via connection pool
    - Schema versioning and migrations
    - Common CRUD helpers

    Subclasses must define:
    - SCHEMA_NAME: Unique identifier for schema versioning
    - SCHEMA_VERSION: Current schema version number
    - INITIAL_SCHEMA: SQL for initial table/index creation

    Example:
        class TodoStore(PostgresStore):
            SCHEMA_NAME = "todo_store"
            SCHEMA_VERSION = 1

            INITIAL_SCHEMA = '''
                CREATE TABLE IF NOT EXISTS todos (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    completed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
            '''

            async def add_todo(self, todo_id: str, title: str) -> None:
                async with self.connection() as conn:
                    await conn.execute(
                        "INSERT INTO todos (id, title) VALUES ($1, $2)",
                        todo_id, title
                    )
    """

    # Subclasses must define these
    SCHEMA_NAME: str = ""
    SCHEMA_VERSION: int = 1
    INITIAL_SCHEMA: str = ""

    def __init__(self, pool: "Pool"):
        """
        Initialize the store with a connection pool.

        Args:
            pool: asyncpg connection pool (from get_postgres_pool())
        """
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError("asyncpg is required for PostgresStore")

        self._pool = pool
        self._initialized = False

        # Validate subclass configuration
        if not self.SCHEMA_NAME:
            raise ValueError(f"{self.__class__.__name__} must define SCHEMA_NAME")
        if not self.INITIAL_SCHEMA:
            raise ValueError(f"{self.__class__.__name__} must define INITIAL_SCHEMA")

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates tables, indexes, and runs any pending migrations.
        Must be called before using the store.
        """
        if self._initialized:
            return

        async with self.connection() as conn:
            # Create schema version tracking table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS _schema_versions (
                    module TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """
            )

            # Check current version
            row = await conn.fetchrow(
                "SELECT version FROM _schema_versions WHERE module = $1", self.SCHEMA_NAME
            )
            current_version = row["version"] if row else 0

            if current_version == 0:
                # New database - run initial schema
                logger.info(f"[{self.SCHEMA_NAME}] Creating initial schema v{self.SCHEMA_VERSION}")
                await conn.execute(self.INITIAL_SCHEMA)
                await conn.execute(
                    """
                    INSERT INTO _schema_versions (module, version)
                    VALUES ($1, $2)
                    ON CONFLICT (module) DO UPDATE SET version = $2, updated_at = NOW()
                """,
                    self.SCHEMA_NAME,
                    self.SCHEMA_VERSION,
                )

            elif current_version < self.SCHEMA_VERSION:
                # Run migrations
                logger.info(
                    f"[{self.SCHEMA_NAME}] Migrating from v{current_version} to v{self.SCHEMA_VERSION}"
                )
                await self._run_migrations(conn, current_version)
                await conn.execute(
                    """
                    UPDATE _schema_versions SET version = $1, updated_at = NOW()
                    WHERE module = $2
                """,
                    self.SCHEMA_VERSION,
                    self.SCHEMA_NAME,
                )

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized at version {self.SCHEMA_VERSION}")

    async def _run_migrations(self, conn: "Connection", from_version: int) -> None:
        """
        Run migrations from current version to target version.

        Override in subclasses to define migrations.

        Args:
            conn: Database connection
            from_version: Current schema version
        """
        # Default: re-run initial schema (safe due to IF NOT EXISTS)
        await conn.execute(self.INITIAL_SCHEMA)

    @asynccontextmanager
    async def connection(self) -> AsyncGenerator["Connection", None]:
        """
        Context manager for database operations.

        Acquires a connection from the pool and returns it after use.

        Yields:
            asyncpg Connection for database operations
        """
        async with self._pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator["Connection", None]:
        """
        Context manager for transactional operations.

        Starts a transaction and commits on success, rolls back on error.

        Yields:
            asyncpg Connection within a transaction
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    # =========================================================================
    # Common Query Helpers
    # =========================================================================

    async def fetch_one(
        self,
        sql: str,
        *args: Any,
    ) -> Optional[Any]:  # Returns asyncpg.Record when available
        """
        Execute query and fetch single row.

        Args:
            sql: SQL query with $1, $2, etc placeholders
            *args: Query parameters

        Returns:
            Single row as asyncpg.Record, or None if no results
        """
        async with self.connection() as conn:
            return await conn.fetchrow(sql, *args)

    async def fetch_all(
        self,
        sql: str,
        *args: Any,
    ) -> list[Any]:  # Returns list[asyncpg.Record] when available
        """
        Execute query and fetch all rows.

        Args:
            sql: SQL query with $1, $2, etc placeholders
            *args: Query parameters

        Returns:
            List of rows as asyncpg.Record
        """
        async with self.connection() as conn:
            return await conn.fetch(sql, *args)

    async def execute(
        self,
        sql: str,
        *args: Any,
    ) -> str:
        """
        Execute a write operation.

        Args:
            sql: SQL statement with $1, $2, etc placeholders
            *args: Statement parameters

        Returns:
            Command status string (e.g., "INSERT 0 1")
        """
        async with self.connection() as conn:
            return await conn.execute(sql, *args)

    async def executemany(
        self,
        sql: str,
        args_list: list[tuple[Any, ...]],
    ) -> None:
        """
        Execute a SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement with $1, $2, etc placeholders
            args_list: List of parameter tuples
        """
        async with self.connection() as conn:
            await conn.executemany(sql, args_list)

    # =========================================================================
    # Common CRUD Helpers
    # =========================================================================

    async def exists(self, table: str, id_column: str, id_value: str) -> bool:
        """
        Check if a record exists.

        Args:
            table: Table name (must be alphanumeric/underscore only)
            id_column: Column name for ID lookup
            id_value: Value to check

        Returns:
            True if record exists
        """
        # Validate table/column names to prevent injection
        if not table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table}")
        if not id_column.replace("_", "").isalnum():
            raise ValueError(f"Invalid column name: {id_column}")

        query = f"SELECT 1 FROM {table} WHERE {id_column} = $1"  # nosec B608
        row = await self.fetch_one(query, id_value)
        return row is not None

    async def count(
        self,
        table: str,
        where: str = "",
        *args: Any,
    ) -> int:
        """
        Count records in a table.

        Args:
            table: Table name
            where: Optional WHERE clause (without "WHERE" keyword)
            *args: Parameters for WHERE clause

        Returns:
            Record count
        """
        if not table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table}")

        sql = f"SELECT COUNT(*) FROM {table}"  # nosec B608
        if where:
            sql += f" WHERE {where}"  # nosec B608

        row = await self.fetch_one(sql, *args)
        return row[0] if row else 0

    async def delete_by_id(
        self,
        table: str,
        id_column: str,
        id_value: str,
    ) -> bool:
        """
        Delete a record by ID.

        Args:
            table: Table name
            id_column: Column name for ID
            id_value: ID value to delete

        Returns:
            True if a record was deleted
        """
        if not table.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name: {table}")
        if not id_column.replace("_", "").isalnum():
            raise ValueError(f"Invalid column name: {id_column}")

        query = f"DELETE FROM {table} WHERE {id_column} = $1"  # nosec B608
        result = await self.execute(query, id_value)
        # Result is like "DELETE 1" or "DELETE 0"
        return result.endswith(" 0") is False

    async def get_schema_version(self) -> int:
        """
        Get the current schema version from database.

        Returns:
            Current schema version, or 0 if not yet initialized
        """
        try:
            row = await self.fetch_one(
                "SELECT version FROM _schema_versions WHERE module = $1", self.SCHEMA_NAME
            )
            return row[0] if row else 0
        except (OSError, RuntimeError) as e:
            # Table may not exist yet, or connection issue
            logger.debug(f"Could not fetch schema version: {e}")
            return 0


__all__ = [
    "PostgresStore",
    "get_postgres_pool",
    "get_postgres_pool_from_settings",
    "close_postgres_pool",
    "ASYNCPG_AVAILABLE",
]
