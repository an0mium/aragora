"""
Unified Database Backend Adapter for Aragora.

Bridges the two existing database abstraction layers into a single coherent
interface that the rest of the codebase can use to transparently work with
either SQLite or PostgreSQL.

The codebase has two abstraction layers:
1. **Sync layer** (``aragora.storage.backends``): ``DatabaseBackend`` ABC with
   ``SQLiteBackend`` and ``PostgreSQLBackend`` using psycopg2. Used by the
   migration runner and lightweight sync operations.
2. **Async layer** (``aragora.storage.postgres_store``): ``PostgresStore`` base
   class using asyncpg pools. Used by production store implementations that
   need high-throughput concurrent access.

This module provides:
- ``UnifiedBackend``: Facade that holds both sync and async backends
- ``get_unified_backend()``: Factory that auto-detects the right backends
- ``BackendCapabilities``: Introspection of what the current backend supports

Usage::

    from aragora.persistence.db_backend import get_unified_backend

    backend = get_unified_backend()

    # Sync operations (migrations, admin)
    row = backend.sync.fetch_one("SELECT count(*) FROM debates")

    # Check capabilities
    if backend.capabilities.supports_concurrent_writes:
        # Safe to run parallel workers
        ...

    if backend.capabilities.supports_advisory_locks:
        # Use PostgreSQL advisory locks for coordination
        ...

    # Get the async pool for high-throughput operations
    pool = await backend.get_async_pool()
    if pool:
        async with pool.acquire() as conn:
            await conn.fetch("SELECT * FROM debates LIMIT 10")

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (triggers PG backend)
    ARAGORA_POSTGRES_DSN: Alternative PG connection string
    ARAGORA_DB_BACKEND: Force backend ("sqlite", "postgres", "auto")
    ARAGORA_DATA_DIR: Base directory for SQLite files (default: ".nomic" or "data" if present)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from asyncpg import Pool

    from aragora.storage.backends import DatabaseBackend

logger = logging.getLogger(__name__)


class BackendType(str, Enum):
    """Supported database backend types."""

    SQLITE = "sqlite"
    POSTGRES = "postgres"


@dataclass(frozen=True)
class BackendCapabilities:
    """Describes what the current database backend supports.

    Use this to make runtime decisions about concurrency, locking, and
    feature availability without checking backend type strings directly.
    """

    backend_type: BackendType

    #: True if the backend can handle multiple concurrent writers without
    #: serialization. SQLite's WAL mode allows concurrent reads but only
    #: one writer at a time.
    supports_concurrent_writes: bool = False

    #: True if the backend supports advisory locks for distributed
    #: coordination (e.g., migration locking across pods).
    supports_advisory_locks: bool = False

    #: True if the backend supports LISTEN/NOTIFY for real-time event
    #: notification between connections.
    supports_listen_notify: bool = False

    #: True if the backend supports JSONB columns with indexed operators.
    supports_jsonb: bool = False

    #: True if the backend has full-text search support (tsvector/tsquery
    #: for PostgreSQL, FTS5 for SQLite).
    supports_full_text_search: bool = True

    #: True if the backend supports read replicas for load distribution.
    supports_read_replicas: bool = False

    #: True if the backend supports savepoints for nested transactions.
    supports_savepoints: bool = True

    #: True if the backend supports RETURNING clauses on INSERT/UPDATE.
    supports_returning: bool = False

    #: True if an async connection pool is available for high-throughput ops.
    has_async_pool: bool = False

    #: Maximum recommended concurrent connections. For SQLite this is
    #: effectively 1 writer; for PostgreSQL it depends on pool size.
    max_concurrent_connections: int = 1

    @classmethod
    def for_sqlite(cls) -> BackendCapabilities:
        """Capabilities for SQLite backend."""
        return cls(
            backend_type=BackendType.SQLITE,
            supports_concurrent_writes=False,
            supports_advisory_locks=False,
            supports_listen_notify=False,
            supports_jsonb=False,
            supports_full_text_search=True,  # FTS5
            supports_read_replicas=False,
            supports_savepoints=True,
            supports_returning=False,
            has_async_pool=False,
            max_concurrent_connections=1,
        )

    @classmethod
    def for_postgres(
        cls,
        has_async_pool: bool = False,
        max_connections: int = 20,
    ) -> BackendCapabilities:
        """Capabilities for PostgreSQL backend."""
        return cls(
            backend_type=BackendType.POSTGRES,
            supports_concurrent_writes=True,
            supports_advisory_locks=True,
            supports_listen_notify=True,
            supports_jsonb=True,
            supports_full_text_search=True,  # tsvector/tsquery
            supports_read_replicas=True,
            supports_savepoints=True,
            supports_returning=True,
            has_async_pool=has_async_pool,
            max_concurrent_connections=max_connections,
        )


@dataclass
class SchemaInfo:
    """Information about a database table's schema.

    Normalized representation that works for both SQLite and PostgreSQL,
    abstracting away differences in type naming and introspection.
    """

    table_name: str
    columns: list[ColumnInfo] = field(default_factory=list)

    def column_names(self) -> list[str]:
        """Get list of column names."""
        return [c.name for c in self.columns]

    def has_column(self, name: str) -> bool:
        """Check if table has a column with the given name."""
        return any(c.name == name for c in self.columns)


@dataclass
class ColumnInfo:
    """Normalized column information."""

    name: str
    type: str  # Normalized type name
    nullable: bool = True
    default: str | None = None
    is_primary_key: bool = False

    # Original type as reported by the database
    raw_type: str = ""


# SQLite -> PostgreSQL type mapping for schema translation
SQLITE_TO_PG_TYPE_MAP: dict[str, str] = {
    "TEXT": "TEXT",
    "INTEGER": "BIGINT",
    "REAL": "DOUBLE PRECISION",
    "BLOB": "BYTEA",
    "BOOLEAN": "BOOLEAN",
    "DATETIME": "TIMESTAMPTZ",
    "TIMESTAMP": "TIMESTAMPTZ",
    "JSON": "JSONB",
    "FLOAT": "DOUBLE PRECISION",
    "DOUBLE": "DOUBLE PRECISION",
    "VARCHAR": "TEXT",
    "CHAR": "TEXT",
    "NUMERIC": "NUMERIC",
}


def normalize_sqlite_type(raw_type: str) -> str:
    """Normalize a SQLite type declaration to a canonical form.

    SQLite is very permissive with types. This normalizes them to
    consistent names that can be mapped to PostgreSQL.

    Args:
        raw_type: Type as declared in SQLite schema (e.g., "VARCHAR(255)")

    Returns:
        Normalized type name
    """
    if not raw_type:
        return "TEXT"

    upper = raw_type.upper().strip()

    # Strip parenthesized length/precision (e.g., VARCHAR(255) -> VARCHAR)
    base = upper.split("(")[0].strip()

    # Direct lookup
    if base in SQLITE_TO_PG_TYPE_MAP:
        return SQLITE_TO_PG_TYPE_MAP[base]

    # Affinity rules (SQLite documentation section 3.1)
    if "INT" in upper:
        return "BIGINT"
    if "CHAR" in upper or "CLOB" in upper or "TEXT" in upper:
        return "TEXT"
    if "BLOB" in upper:
        return "BYTEA"
    if "REAL" in upper or "FLOA" in upper or "DOUB" in upper:
        return "DOUBLE PRECISION"
    if "BOOL" in upper:
        return "BOOLEAN"
    if "TIME" in upper or "DATE" in upper:
        return "TIMESTAMPTZ"
    if "JSON" in upper:
        return "JSONB"

    # Default to TEXT (SQLite stores everything as text anyway)
    return "TEXT"


class UnifiedBackend:
    """Facade that provides access to both sync and async database operations.

    Holds references to the sync ``DatabaseBackend`` (for migrations, admin
    operations) and optionally an async ``asyncpg.Pool`` (for high-throughput
    store operations).

    Attributes:
        sync: The synchronous database backend (always available)
        capabilities: What the current backend supports
    """

    def __init__(
        self,
        sync_backend: DatabaseBackend,
        capabilities: BackendCapabilities,
        async_pool_factory: Any = None,
    ):
        """Initialize the unified backend.

        Args:
            sync_backend: Synchronous database backend
            capabilities: Backend capabilities descriptor
            async_pool_factory: Optional callable that returns an asyncpg Pool.
                This is deferred to avoid creating pools at import time.
        """
        self.sync = sync_backend
        self.capabilities = capabilities
        self._async_pool_factory = async_pool_factory
        self._async_pool: Pool | None = None

    @property
    def backend_type(self) -> BackendType:
        """Get the current backend type."""
        return self.capabilities.backend_type

    @property
    def is_postgres(self) -> bool:
        """Check if currently using PostgreSQL."""
        return self.capabilities.backend_type == BackendType.POSTGRES

    @property
    def is_sqlite(self) -> bool:
        """Check if currently using SQLite."""
        return self.capabilities.backend_type == BackendType.SQLITE

    async def get_async_pool(self) -> Pool | None:
        """Get the asyncpg connection pool, if available.

        For PostgreSQL backends, this returns a connection pool suitable
        for high-throughput async operations. For SQLite, returns None.

        The pool is created lazily on first call and cached thereafter.

        Returns:
            asyncpg Pool or None if not available
        """
        if self._async_pool is not None:
            return self._async_pool

        if self._async_pool_factory is None:
            return None

        try:
            self._async_pool = await self._async_pool_factory()
            return self._async_pool
        except Exception as e:
            logger.warning(f"Failed to create async pool: {e}")
            return None

    def get_schema_info(self, table_name: str) -> SchemaInfo:
        """Get normalized schema information for a table.

        Works with both SQLite and PostgreSQL, returning a consistent
        representation.

        Args:
            table_name: Name of the table to introspect

        Returns:
            SchemaInfo with normalized column information
        """
        raw_columns = self.sync.get_table_columns(table_name)
        columns = []

        for col in raw_columns:
            raw_type = col.get("type", "TEXT")

            if self.is_sqlite:
                normalized_type = normalize_sqlite_type(raw_type)
            else:
                normalized_type = raw_type.upper()

            columns.append(
                ColumnInfo(
                    name=col["name"],
                    type=normalized_type,
                    nullable=not col.get("notnull", False),
                    default=col.get("default"),
                    is_primary_key=bool(col.get("pk", False)),
                    raw_type=raw_type,
                )
            )

        return SchemaInfo(table_name=table_name, columns=columns)

    def translate_sql(self, sql: str) -> str:
        """Translate SQL between backends.

        Converts placeholder syntax and basic type differences:
        - SQLite ``?`` placeholders become PostgreSQL ``%s``
        - ``AUTOINCREMENT`` becomes ``SERIAL`` / ``GENERATED``

        For complex schema translation, use the migration tools instead.

        Args:
            sql: SQL string (using ``?`` placeholders)

        Returns:
            Translated SQL for the current backend
        """
        return self.sync.convert_placeholder(sql)

    def close(self) -> None:
        """Close all connections and pools."""
        try:
            self.sync.close()
        except Exception as e:
            logger.warning(f"Error closing sync backend: {e}")

    async def close_async(self) -> None:
        """Close async pool if it exists."""
        if self._async_pool is not None:
            try:
                await self._async_pool.close()
                self._async_pool = None
            except Exception as e:
                logger.warning(f"Error closing async pool: {e}")

    def __repr__(self) -> str:
        pool_status = "pool=yes" if self._async_pool else "pool=no"
        return f"UnifiedBackend(type={self.capabilities.backend_type.value}, {pool_status})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Module-level singleton
_unified_backend: UnifiedBackend | None = None


def get_unified_backend(
    force_backend: BackendType | None = None,
    db_path: str | None = None,
) -> UnifiedBackend:
    """Get or create the unified database backend.

    Determines the appropriate backend based on environment configuration
    and creates a ``UnifiedBackend`` that provides both sync and async access.

    Resolution order:
    1. If ``force_backend`` is specified, use that
    2. Check ``ARAGORA_DB_BACKEND`` environment variable
    3. Check if ``DATABASE_URL`` or ``ARAGORA_POSTGRES_DSN`` is set
    4. Fall back to SQLite

    Args:
        force_backend: Override auto-detection with a specific backend type
        db_path: SQLite database path override (for testing)

    Returns:
        Configured UnifiedBackend instance
    """
    global _unified_backend

    if _unified_backend is not None and force_backend is None:
        return _unified_backend

    # Determine backend type
    backend_type = force_backend

    if backend_type is None:
        env_backend = os.environ.get("ARAGORA_DB_BACKEND", "auto").lower()
        if env_backend in ("postgres", "postgresql"):
            backend_type = BackendType.POSTGRES
        elif env_backend == "sqlite":
            backend_type = BackendType.SQLITE
        else:
            # Auto-detect: check for PostgreSQL configuration
            has_pg = bool(
                os.environ.get("DATABASE_URL")
                or os.environ.get("ARAGORA_POSTGRES_DSN")
                or os.environ.get("SUPABASE_POSTGRES_DSN")
            )
            backend_type = BackendType.POSTGRES if has_pg else BackendType.SQLITE

    if backend_type == BackendType.POSTGRES:
        _unified_backend = _create_postgres_backend()
    else:
        _unified_backend = _create_sqlite_backend(db_path)

    return _unified_backend


def _create_sqlite_backend(db_path: str | None = None) -> UnifiedBackend:
    """Create a SQLite-based unified backend."""
    from aragora.storage.backends import SQLiteBackend
    from aragora.config import resolve_db_path

    if db_path is None:
        db_path = resolve_db_path("aragora.db")
    else:
        db_path = resolve_db_path(db_path)

    sync_backend = SQLiteBackend(db_path)
    capabilities = BackendCapabilities.for_sqlite()

    logger.info(f"Unified backend: SQLite ({db_path})")
    return UnifiedBackend(
        sync_backend=sync_backend,
        capabilities=capabilities,
    )


def _create_postgres_backend() -> UnifiedBackend:
    """Create a PostgreSQL-based unified backend."""
    from aragora.storage.backends import POSTGRESQL_AVAILABLE, PostgreSQLBackend

    # Resolve DSN
    dsn = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("ARAGORA_POSTGRES_DSN")
        or os.environ.get("SUPABASE_POSTGRES_DSN")
    )

    if not dsn:
        # Try to derive from Supabase config
        try:
            from aragora.storage.connection_factory import get_supabase_postgres_dsn

            dsn = get_supabase_postgres_dsn()
        except ImportError:
            pass

    if not dsn:
        logger.warning("PostgreSQL requested but no DSN configured, falling back to SQLite")
        return _create_sqlite_backend()

    if not POSTGRESQL_AVAILABLE:
        logger.warning("PostgreSQL requested but psycopg2 not installed, falling back to SQLite")
        return _create_sqlite_backend()

    try:
        sync_backend = PostgreSQLBackend(dsn)
    except Exception as e:
        logger.warning(f"PostgreSQL connection failed ({e}), falling back to SQLite")
        return _create_sqlite_backend()

    # Pool size from environment
    max_connections = int(os.environ.get("ARAGORA_POOL_MAX_SIZE", "20"))

    async def _pool_factory() -> Any:
        """Lazy factory for async pool creation."""
        try:
            from aragora.storage.pool_manager import get_shared_pool, is_pool_initialized

            if is_pool_initialized():
                pool = get_shared_pool()
                if pool is not None:
                    return pool
        except ImportError:
            pass

        # Fall back to creating a new pool
        try:
            from aragora.storage.postgres_store import get_postgres_pool

            return await get_postgres_pool(dsn=dsn)
        except Exception as e:
            logger.warning(f"Failed to create async PostgreSQL pool: {e}")
            return None

    capabilities = BackendCapabilities.for_postgres(
        has_async_pool=True,
        max_connections=max_connections,
    )

    # Mask credentials in log
    safe_dsn = dsn.split("@")[-1] if "@" in dsn else "***"
    logger.info(f"Unified backend: PostgreSQL ({safe_dsn})")

    return UnifiedBackend(
        sync_backend=sync_backend,
        capabilities=capabilities,
        async_pool_factory=_pool_factory,
    )


def reset_unified_backend() -> None:
    """Reset the unified backend singleton (for testing)."""
    global _unified_backend
    if _unified_backend is not None:
        _unified_backend.close()
        _unified_backend = None


__all__ = [
    "BackendCapabilities",
    "BackendType",
    "ColumnInfo",
    "SchemaInfo",
    "UnifiedBackend",
    "get_unified_backend",
    "normalize_sqlite_type",
    "reset_unified_backend",
    "SQLITE_TO_PG_TYPE_MAP",
]
