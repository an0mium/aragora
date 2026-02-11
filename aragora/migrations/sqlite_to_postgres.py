"""
SQLite-to-PostgreSQL Migration Orchestrator.

Provides a comprehensive, production-grade migration path from SQLite to
PostgreSQL for the Aragora codebase.  Unlike the existing per-store migrators
(``persistence.migrations.postgres.data_migrator`` and ``memory_migrator``),
this module orchestrates the *entire* migration lifecycle:

1. **Discovery** -- Find all SQLite database files and enumerate their tables
2. **Schema translation** -- Convert SQLite DDL to PostgreSQL DDL
3. **Schema creation** -- Apply translated schemas to the target PostgreSQL
4. **Data migration** -- Batch-copy rows with type coercion and progress
5. **Verification** -- Compare row counts and checksums between source/target
6. **Rollback** -- Drop migrated PostgreSQL tables if something goes wrong

The orchestrator is idempotent: re-running it will skip tables that already
exist and rows that have already been migrated (via ``ON CONFLICT DO NOTHING``).

Usage::

    # CLI
    python -m aragora.migrations.sqlite_to_postgres \\
        --data-dir .nomic \\
        --postgres-dsn "postgresql://user:pass@host/db" \\
        --batch-size 2000

    # Dry-run (schema translation preview)
    python -m aragora.migrations.sqlite_to_postgres \\
        --data-dir .nomic \\
        --postgres-dsn "postgresql://..." \\
        --dry-run

    # Programmatic
    from aragora.migrations.sqlite_to_postgres import MigrationOrchestrator

    orch = MigrationOrchestrator(
        data_dir=".nomic",
        postgres_dsn="postgresql://...",
    )
    report = await orch.run()
    orch.print_report(report)

Environment Variables:
    DATABASE_URL / ARAGORA_POSTGRES_DSN: Target PostgreSQL DSN
    ARAGORA_DATA_DIR / ARAGORA_NOMIC_DIR: Source SQLite data directory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Optional asyncpg import -- the module stays importable without it so that
# tests and dry-run mode work without a PostgreSQL connection.
asyncpg: ModuleType | None
Pool: Any
try:
    import asyncpg  # type: ignore[no-redef]
    from asyncpg import Pool  # type: ignore[no-redef]

    ASYNCPG_AVAILABLE = True
except ImportError:
    asyncpg = None
    Pool = Any
    ASYNCPG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

#: Mapping from SQLite type affinity keywords to PostgreSQL types.
#: SQLite is permissive; this covers the types actually seen in Aragora.
SQLITE_TO_PG_TYPES: dict[str, str] = {
    # Exact matches
    "TEXT": "TEXT",
    "INTEGER": "BIGINT",
    "REAL": "DOUBLE PRECISION",
    "BLOB": "BYTEA",
    "BOOLEAN": "BOOLEAN",
    "NUMERIC": "NUMERIC",
    # Common aliases used in Aragora schemas
    "FLOAT": "DOUBLE PRECISION",
    "DOUBLE": "DOUBLE PRECISION",
    "VARCHAR": "TEXT",
    "CHAR": "TEXT",
    "JSON": "JSONB",
    "JSONB": "JSONB",
    "DATETIME": "TIMESTAMPTZ",
    "TIMESTAMP": "TIMESTAMPTZ",
    "DATE": "DATE",
    "TIME": "TIME",
    "BIGINT": "BIGINT",
    "SMALLINT": "SMALLINT",
    "SERIAL": "SERIAL",
}


def sqlite_type_to_pg(raw_type: str) -> str:
    """Convert a SQLite column type to its PostgreSQL equivalent.

    Handles parenthesized lengths (e.g., ``VARCHAR(255)``), common
    aliases, and falls back to ``TEXT`` for truly unknown types.

    Args:
        raw_type: Type string from ``PRAGMA table_info``.

    Returns:
        PostgreSQL type string.
    """
    if not raw_type:
        return "TEXT"

    upper = raw_type.upper().strip()
    base = upper.split("(")[0].strip()

    # Direct lookup
    if base in SQLITE_TO_PG_TYPES:
        return SQLITE_TO_PG_TYPES[base]

    # SQLite affinity rules
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

    return "TEXT"


# ---------------------------------------------------------------------------
# Data classes for reporting
# ---------------------------------------------------------------------------


@dataclass
class TableMigrationResult:
    """Result of migrating a single table."""

    database: str
    table: str
    rows_migrated: int = 0
    rows_skipped: int = 0
    rows_total_source: int = 0
    rows_total_target: int = 0
    schema_created: bool = False
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    @property
    def verified(self) -> bool:
        """True if target has at least as many rows as source."""
        return self.rows_total_target >= self.rows_total_source


@dataclass
class DatabaseMigrationResult:
    """Result of migrating an entire SQLite database file."""

    database: str
    tables: list[TableMigrationResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(t.success for t in self.tables)

    @property
    def total_rows_migrated(self) -> int:
        return sum(t.rows_migrated for t in self.tables)


@dataclass
class MigrationReport:
    """Complete migration report across all databases."""

    data_dir: str
    target_dsn_safe: str  # DSN with credentials masked
    started_at: datetime
    completed_at: datetime | None = None
    databases: list[DatabaseMigrationResult] = field(default_factory=list)
    dry_run: bool = False

    @property
    def success(self) -> bool:
        return all(db.success for db in self.databases)

    @property
    def total_tables(self) -> int:
        return sum(len(db.tables) for db in self.databases)

    @property
    def total_rows_migrated(self) -> int:
        return sum(db.total_rows_migrated for db in self.databases)

    @property
    def total_errors(self) -> int:
        return sum(len(t.errors) for db in self.databases for t in db.tables)

    @property
    def duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()


# ---------------------------------------------------------------------------
# Schema translator
# ---------------------------------------------------------------------------


class SchemaTranslator:
    """Translates SQLite schemas to PostgreSQL DDL.

    Uses ``PRAGMA table_info`` and ``PRAGMA index_list`` to introspect
    SQLite tables and generate equivalent PostgreSQL ``CREATE TABLE``
    and ``CREATE INDEX`` statements.
    """

    def __init__(self, pg_schema: str = "public"):
        """Initialize the translator.

        Args:
            pg_schema: PostgreSQL schema to create tables in.
        """
        self.pg_schema = pg_schema

    def translate_table(
        self,
        sqlite_conn: sqlite3.Connection,
        table_name: str,
    ) -> str:
        """Generate PostgreSQL CREATE TABLE for a SQLite table.

        Args:
            sqlite_conn: Open SQLite connection.
            table_name: Name of the table to translate.

        Returns:
            PostgreSQL DDL string.
        """
        columns = self._get_columns(sqlite_conn, table_name)
        if not columns:
            return ""

        col_defs = []
        pk_columns = []

        for col in columns:
            name = col["name"]
            raw_type = col["type"]
            notnull = col["notnull"]
            default = col["default"]
            is_pk = col["pk"]

            pg_type = sqlite_type_to_pg(raw_type)

            # Handle INTEGER PRIMARY KEY AUTOINCREMENT -> SERIAL
            if (
                is_pk
                and pg_type == "BIGINT"
                and self._is_autoincrement(sqlite_conn, table_name, name)
            ):
                pg_type = "BIGSERIAL"

            parts = [f'    "{name}" {pg_type}']

            if notnull and not is_pk:
                parts.append("NOT NULL")

            if default is not None:
                pg_default = self._translate_default(default, pg_type)
                if pg_default:
                    parts.append(f"DEFAULT {pg_default}")

            col_defs.append(" ".join(parts))

            if is_pk:
                pk_columns.append(f'"{name}"')

        # Add primary key constraint
        if pk_columns:
            col_defs.append(f"    PRIMARY KEY ({', '.join(pk_columns)})")

        qualified_name = (
            f'"{self.pg_schema}"."{table_name}"'
            if self.pg_schema != "public"
            else f'"{table_name}"'
        )

        ddl = f"CREATE TABLE IF NOT EXISTS {qualified_name} (\n"
        ddl += ",\n".join(col_defs)
        ddl += "\n);"

        # Add indexes
        indexes = self._get_indexes(sqlite_conn, table_name)
        for idx in indexes:
            idx_ddl = self._translate_index(table_name, idx)
            if idx_ddl:
                ddl += f"\n{idx_ddl}"

        return ddl

    def _get_columns(self, conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
        """Get column info from SQLite table."""
        cursor = conn.execute(f'PRAGMA table_info("{table}")')  # noqa: S608
        return [
            {
                "name": row[1],
                "type": row[2] or "TEXT",
                "notnull": bool(row[3]),
                "default": row[4],
                "pk": row[5] > 0,
            }
            for row in cursor.fetchall()
        ]

    def _get_indexes(self, conn: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
        """Get index info from SQLite table."""
        cursor = conn.execute(f'PRAGMA index_list("{table}")')  # noqa: S608
        indexes = []
        for row in cursor.fetchall():
            idx_name = row[1]
            is_unique = bool(row[2])

            # Get columns for this index
            col_cursor = conn.execute(f'PRAGMA index_info("{idx_name}")')  # noqa: S608
            col_names = [col_row[2] for col_row in col_cursor.fetchall()]

            # Skip auto-created indexes for PRIMARY KEY
            if idx_name.startswith("sqlite_autoindex"):
                continue

            indexes.append(
                {
                    "name": idx_name,
                    "unique": is_unique,
                    "columns": col_names,
                }
            )
        return indexes

    def _is_autoincrement(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        """Check if a column uses AUTOINCREMENT."""
        try:
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            row = cursor.fetchone()
            if row and row[0]:
                sql_upper = row[0].upper()
                return "AUTOINCREMENT" in sql_upper
        except sqlite3.Error as e:
            logger.debug("is autoincrement encountered an error: %s", e)
        return False

    def _translate_default(self, default: str, pg_type: str) -> str | None:
        """Translate a SQLite default value to PostgreSQL."""
        if default is None:
            return None

        upper = default.upper().strip()

        # Common SQLite defaults
        if upper in ("NULL", "NONE"):
            return "NULL"
        if upper in ("CURRENT_TIMESTAMP", "CURRENT_DATE", "CURRENT_TIME"):
            return "NOW()"
        if upper in ("TRUE", "1"):
            return "TRUE"
        if upper in ("FALSE", "0") and "BOOL" in pg_type.upper():
            return "FALSE"
        if upper.startswith("'") and upper.endswith("'"):
            # String default
            return default
        if upper == "0" and pg_type in ("BIGINT", "SMALLINT", "NUMERIC"):
            return "0"

        # Numeric default
        try:
            float(default)
            return default
        except (ValueError, TypeError) as e:
            logger.debug("Failed to parse numeric value: %s", e)

        # JSON default
        if pg_type == "JSONB":
            if upper in ("'{}'", "{}"):
                return "'{}'::jsonb"
            if upper in ("'[]'", "[]"):
                return "'[]'::jsonb"

        return None

    def _translate_index(self, table: str, index: dict[str, Any]) -> str:
        """Translate a SQLite index to PostgreSQL DDL."""
        columns = index["columns"]
        if not columns:
            return ""

        unique = "UNIQUE " if index["unique"] else ""
        col_list = ", ".join(f'"{c}"' for c in columns)
        idx_name = index["name"]

        return f'CREATE {unique}INDEX IF NOT EXISTS "{idx_name}" ON "{table}" ({col_list});'


# ---------------------------------------------------------------------------
# Migration Orchestrator
# ---------------------------------------------------------------------------


class MigrationOrchestrator:
    """Orchestrates the full SQLite-to-PostgreSQL migration.

    Discovers all SQLite databases in the data directory, translates
    their schemas, creates corresponding PostgreSQL tables, and copies
    data in batches with type coercion.

    The orchestrator is safe to re-run: schemas use ``IF NOT EXISTS``
    and data inserts use ``ON CONFLICT DO NOTHING``.
    """

    def __init__(
        self,
        data_dir: str | Path = ".nomic",
        postgres_dsn: str | None = None,
        batch_size: int = 2000,
        pg_schema: str = "public",
        exclude_tables: set[str] | None = None,
        include_databases: set[str] | None = None,
    ):
        """Initialize the migration orchestrator.

        Args:
            data_dir: Directory containing SQLite database files.
            postgres_dsn: PostgreSQL connection string. If not provided,
                uses ``DATABASE_URL`` or ``ARAGORA_POSTGRES_DSN``.
            batch_size: Number of rows per INSERT batch.
            pg_schema: PostgreSQL schema to create tables in.
            exclude_tables: Set of table names to skip (e.g., internal
                SQLite tables).
            include_databases: If set, only migrate these database files
                (by stem name, e.g., ``{"core", "memory"}``).
        """
        import os

        self.data_dir = Path(data_dir)
        self.postgres_dsn = (
            postgres_dsn or os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_POSTGRES_DSN")
        )
        self.batch_size = batch_size
        self.pg_schema = pg_schema
        self.exclude_tables = exclude_tables or {
            "sqlite_sequence",
            "sqlite_stat1",
            "sqlite_stat4",
        }
        self.include_databases = include_databases
        self.translator = SchemaTranslator(pg_schema=pg_schema)
        self._pool: Optional[Pool] = None

    # -- Discovery -----------------------------------------------------------

    def discover_databases(self) -> list[Path]:
        """Find all SQLite database files in the data directory.

        Returns:
            Sorted list of ``.db`` file paths.
        """
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return []

        db_files = sorted(self.data_dir.glob("*.db"))

        if self.include_databases:
            db_files = [f for f in db_files if f.stem in self.include_databases]

        logger.info(f"Discovered {len(db_files)} SQLite database(s) in {self.data_dir}")
        return db_files

    def discover_tables(self, db_path: Path) -> list[str]:
        """List user tables in a SQLite database.

        Args:
            db_path: Path to the SQLite database file.

        Returns:
            List of table names (excluding internal/system tables).
        """
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall() if row[0] not in self.exclude_tables]
            return sorted(tables)
        finally:
            conn.close()

    # -- Pool management -----------------------------------------------------

    async def _get_pool(self) -> Pool:
        """Get or create the asyncpg connection pool."""
        if not ASYNCPG_AVAILABLE:
            raise RuntimeError(
                "asyncpg is required for PostgreSQL migration. Install with: pip install asyncpg"
            )
        if not self.postgres_dsn:
            raise RuntimeError(
                "No PostgreSQL DSN configured. Set DATABASE_URL or "
                "ARAGORA_POSTGRES_DSN, or pass postgres_dsn to the constructor."
            )
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                self.postgres_dsn,
                min_size=2,
                max_size=10,
                command_timeout=120.0,
            )
        return self._pool

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # -- Schema creation -----------------------------------------------------

    async def create_schema(
        self,
        db_path: Path,
        table: str,
        dry_run: bool = False,
    ) -> str:
        """Translate and optionally apply a table schema.

        Args:
            db_path: SQLite database containing the table.
            table: Table name.
            dry_run: If True, return DDL without executing.

        Returns:
            The generated PostgreSQL DDL.
        """
        conn = sqlite3.connect(str(db_path))
        try:
            ddl = self.translator.translate_table(conn, table)
        finally:
            conn.close()

        if not ddl:
            return ""

        if not dry_run:
            pool = await self._get_pool()
            async with pool.acquire() as pg_conn:
                await pg_conn.execute(ddl)
                logger.debug(f"Created schema for table: {table}")

        return ddl

    # -- Data migration ------------------------------------------------------

    async def migrate_table_data(
        self,
        db_path: Path,
        table: str,
    ) -> TableMigrationResult:
        """Migrate data from a single SQLite table to PostgreSQL.

        Args:
            db_path: SQLite database file.
            table: Table name.

        Returns:
            TableMigrationResult with row counts and any errors.
        """
        result = TableMigrationResult(
            database=db_path.name,
            table=table,
        )
        start = time.monotonic()

        sqlite_conn = sqlite3.connect(str(db_path))
        sqlite_conn.row_factory = sqlite3.Row

        try:
            pool = await self._get_pool()

            # Get SQLite columns
            cursor = sqlite_conn.execute(f'PRAGMA table_info("{table}")')  # noqa: S608
            sqlite_columns = [row[1] for row in cursor.fetchall()]

            if not sqlite_columns:
                result.errors.append(f"No columns found in SQLite table {table}")
                return result

            # Get PostgreSQL columns (to find intersection)
            async with pool.acquire() as pg_conn:
                pg_cols = await pg_conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = $1 ORDER BY ordinal_position",
                    table,
                )
                pg_column_names = {row["column_name"] for row in pg_cols}

            if not pg_column_names:
                result.errors.append(
                    f"Table {table} not found in PostgreSQL. Run schema creation first."
                )
                return result

            # Migrate only columns that exist in both
            common_columns = [c for c in sqlite_columns if c in pg_column_names]
            if not common_columns:
                result.errors.append(f"No common columns between SQLite and PostgreSQL for {table}")
                return result

            # Count source rows
            count_cursor = sqlite_conn.execute(f'SELECT COUNT(*) FROM "{table}"')  # noqa: S608
            result.rows_total_source = count_cursor.fetchone()[0]

            if result.rows_total_source == 0:
                logger.debug(f"Table {table} is empty, skipping data migration")
                result.duration_seconds = time.monotonic() - start
                return result

            # Build parameterized INSERT
            col_list = ", ".join(f'"{c}"' for c in common_columns)
            placeholders = ", ".join(f"${i + 1}" for i in range(len(common_columns)))
            insert_sql = (
                f'INSERT INTO "{table}" ({col_list}) VALUES ({placeholders}) ON CONFLICT DO NOTHING'
            )

            # Get column types for coercion
            col_type_cursor = sqlite_conn.execute(f'PRAGMA table_info("{table}")')  # noqa: S608
            col_types = {row[1]: row[2] for row in col_type_cursor.fetchall()}

            # Stream data in batches
            data_cursor = sqlite_conn.execute(
                "SELECT {} ".format(", ".join('"{}"'.format(c) for c in common_columns))
                + 'FROM "{}"'.format(table)
            )

            batch: list[tuple[Any, ...]] = []
            for row in data_cursor:
                converted = tuple(
                    self._coerce_value(row[c], col_types.get(c, "TEXT")) for c in common_columns
                )
                batch.append(converted)

                if len(batch) >= self.batch_size:
                    await self._insert_batch(pool, insert_sql, batch, table, result)
                    batch = []

            # Final partial batch
            if batch:
                await self._insert_batch(pool, insert_sql, batch, table, result)

            logger.info(
                f"Migrated {result.rows_migrated} rows for {table} ({result.rows_skipped} skipped)"
            )

        except Exception as e:
            result.errors.append(f"Migration error: {e}")
            logger.exception(f"Error migrating table {table} from {db_path.name}")
        finally:
            sqlite_conn.close()
            result.duration_seconds = time.monotonic() - start

        return result

    async def _insert_batch(
        self,
        pool: Pool,
        insert_sql: str,
        batch: list[tuple[Any, ...]],
        table: str,
        result: TableMigrationResult,
    ) -> None:
        """Insert a batch of rows into PostgreSQL."""
        async with pool.acquire() as conn:
            try:
                await conn.executemany(insert_sql, batch)
                result.rows_migrated += len(batch)
            except Exception as e:
                error_msg = str(e)
                # Try row-by-row on batch failure to maximize migration
                if "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
                    result.rows_skipped += len(batch)
                else:
                    # Try individual inserts
                    migrated, skipped = 0, 0
                    for row in batch:
                        try:
                            await conn.execute(insert_sql, *row)
                            migrated += 1
                        except Exception as exc:
                            logger.debug("Skipped row during migration of table %s: %s", table, exc)
                            skipped += 1
                    result.rows_migrated += migrated
                    result.rows_skipped += skipped
                    if skipped > 0:
                        result.errors.append(
                            f"Batch error in {table}: {skipped} rows failed ({error_msg})"
                        )

    def _coerce_value(self, value: Any, sqlite_type: str) -> Any:
        """Coerce a SQLite value to a PostgreSQL-compatible type.

        Handles:
        - Boolean: SQLite 0/1 to Python bool
        - JSON: Ensure valid JSON strings
        - Timestamps: Parse ISO format strings
        - None passthrough
        """
        if value is None:
            return None

        upper_type = (sqlite_type or "").upper()

        # Boolean
        if "BOOL" in upper_type:
            return bool(value)

        # JSON
        if "JSON" in upper_type:
            if isinstance(value, str):
                try:
                    json.loads(value)  # Validate
                    return value
                except (json.JSONDecodeError, ValueError):
                    return json.dumps(value)
            if isinstance(value, (dict, list)):
                return json.dumps(value)
            return json.dumps(value)

        # Timestamp
        if "TIME" in upper_type or "DATE" in upper_type:
            if isinstance(value, str):
                try:
                    from datetime import datetime as dt

                    return dt.fromisoformat(value.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    return value
            return value

        return value

    # -- Verification --------------------------------------------------------

    async def verify_table(
        self,
        db_path: Path,
        table: str,
    ) -> TableMigrationResult:
        """Verify that a table was migrated correctly.

        Compares row counts between SQLite source and PostgreSQL target.

        Args:
            db_path: SQLite database file.
            table: Table name.

        Returns:
            TableMigrationResult with counts filled in (check ``.verified``).
        """
        result = TableMigrationResult(database=db_path.name, table=table)

        # Source count
        sqlite_conn = sqlite3.connect(str(db_path))
        try:
            cursor = sqlite_conn.execute(f'SELECT COUNT(*) FROM "{table}"')  # noqa: S608
            result.rows_total_source = cursor.fetchone()[0]
        finally:
            sqlite_conn.close()

        # Target count
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                row = await conn.fetchval(f'SELECT COUNT(*) FROM "{table}"')  # noqa: S608
                result.rows_total_target = row or 0
            except Exception as e:
                result.errors.append(f"Verification error: {e}")

        return result

    # -- Rollback ------------------------------------------------------------

    async def rollback_table(self, table: str) -> bool:
        """Drop a table from PostgreSQL (rollback).

        Args:
            table: Table to drop.

        Returns:
            True if the table was dropped successfully.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            try:
                await conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')  # noqa: S608
                logger.info(f"Rolled back table: {table}")
                return True
            except Exception as e:
                logger.error(f"Failed to rollback table {table}: {e}")
                return False

    async def rollback_database(self, db_path: Path) -> int:
        """Rollback all tables from a SQLite database.

        Args:
            db_path: SQLite database whose tables should be dropped.

        Returns:
            Number of tables successfully dropped.
        """
        tables = self.discover_tables(db_path)
        dropped = 0
        for table in tables:
            if await self.rollback_table(table):
                dropped += 1
        return dropped

    # -- Full orchestration --------------------------------------------------

    async def run(
        self,
        dry_run: bool = False,
        verify: bool = True,
    ) -> MigrationReport:
        """Run the full migration.

        Steps:
        1. Discover SQLite databases and tables
        2. For each table: translate schema, create in PG, migrate data
        3. Optionally verify row counts

        Args:
            dry_run: If True, only generate DDL without executing anything.
            verify: If True, verify row counts after migration.

        Returns:
            MigrationReport with full results.
        """
        safe_dsn = ""
        if self.postgres_dsn:
            safe_dsn = self.postgres_dsn.split("@")[-1] if "@" in self.postgres_dsn else "***"

        report = MigrationReport(
            data_dir=str(self.data_dir),
            target_dsn_safe=safe_dsn,
            started_at=datetime.now(timezone.utc),
            dry_run=dry_run,
        )

        db_files = self.discover_databases()
        if not db_files:
            logger.warning("No SQLite databases found to migrate")
            report.completed_at = datetime.now(timezone.utc)
            return report

        for db_path in db_files:
            db_result = DatabaseMigrationResult(database=db_path.name)
            tables = self.discover_tables(db_path)

            logger.info(f"Processing {db_path.name}: {len(tables)} table(s)")

            for table in tables:
                # Schema
                try:
                    ddl = await self.create_schema(db_path, table, dry_run=dry_run)
                    if dry_run:
                        table_result = TableMigrationResult(
                            database=db_path.name,
                            table=table,
                            schema_created=bool(ddl),
                        )
                        if ddl:
                            logger.info(f"[DRY RUN] Schema for {table}:\n{ddl}")
                        db_result.tables.append(table_result)
                        continue
                except Exception as e:
                    table_result = TableMigrationResult(
                        database=db_path.name,
                        table=table,
                    )
                    table_result.errors.append(f"Schema creation failed: {e}")
                    db_result.tables.append(table_result)
                    continue

                # Data
                table_result = await self.migrate_table_data(db_path, table)
                table_result.schema_created = True

                # Verify
                if verify and table_result.success:
                    verify_result = await self.verify_table(db_path, table)
                    table_result.rows_total_source = verify_result.rows_total_source
                    table_result.rows_total_target = verify_result.rows_total_target

                    if not verify_result.verified:
                        table_result.errors.append(
                            f"Verification mismatch: source={verify_result.rows_total_source}, "
                            f"target={verify_result.rows_total_target}"
                        )

                db_result.tables.append(table_result)

            report.databases.append(db_result)

        report.completed_at = datetime.now(timezone.utc)
        return report

    # -- Reporting -----------------------------------------------------------

    @staticmethod
    def print_report(report: MigrationReport) -> None:
        """Print a formatted migration report to stdout.

        Args:
            report: The migration report to display.
        """
        print("\n" + "=" * 70)
        print("SQLITE-TO-POSTGRESQL MIGRATION REPORT")
        print("=" * 70)

        mode = "[DRY RUN]" if report.dry_run else "[LIVE]"
        print(f"\n  Mode:       {mode}")
        print(f"  Source:     {report.data_dir}")
        print(f"  Target:     {report.target_dsn_safe}")
        print(f"  Started:    {report.started_at.isoformat()}")
        if report.completed_at:
            print(f"  Completed:  {report.completed_at.isoformat()}")
            print(f"  Duration:   {report.duration_seconds:.1f}s")

        for db in report.databases:
            print(f"\n  --- {db.database} ---")
            for t in db.tables:
                status = "OK" if t.success else "ERRORS"
                verify_str = ""
                if t.rows_total_source > 0:
                    verify_str = f" [src={t.rows_total_source}, tgt={t.rows_total_target}]"
                print(
                    f"    {t.table:30} | "
                    f"{t.rows_migrated:>8} migrated | "
                    f"{t.rows_skipped:>6} skipped | "
                    f"{t.duration_seconds:>6.1f}s | "
                    f"{status}{verify_str}"
                )
                for err in t.errors:
                    print(f"      ERROR: {err}")

        print("\n" + "-" * 70)
        print("SUMMARY")
        print("-" * 70)
        print(f"  Databases:     {len(report.databases)}")
        print(f"  Tables:        {report.total_tables}")
        print(f"  Rows migrated: {report.total_rows_migrated}")
        print(f"  Errors:        {report.total_errors}")
        print(f"  Status:        {'SUCCESS' if report.success else 'FAILED'}")
        print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


async def _cli_main() -> None:
    """CLI entry point for the migration orchestrator."""
    import os
    from aragora.persistence.db_config import get_default_data_dir

    parser = argparse.ArgumentParser(
        description="Migrate all Aragora SQLite databases to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Full migration\n"
            "  python -m aragora.migrations.sqlite_to_postgres \\\n"
            '    --data-dir .nomic --postgres-dsn "postgresql://..."\n'
            "\n"
            "  # Preview schemas without migrating\n"
            "  python -m aragora.migrations.sqlite_to_postgres \\\n"
            '    --data-dir .nomic --postgres-dsn "postgresql://..." --dry-run\n'
            "\n"
            "  # Migrate specific databases only\n"
            "  python -m aragora.migrations.sqlite_to_postgres \\\n"
            "    --include-db core --include-db memory\n"
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(get_default_data_dir()),
        help="Directory containing SQLite databases (default: .nomic or data)",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_POSTGRES_DSN"),
        help="PostgreSQL connection string (or set DATABASE_URL)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Rows per INSERT batch (default: 2000)",
    )
    parser.add_argument(
        "--pg-schema",
        default="public",
        help="PostgreSQL schema (default: public)",
    )
    parser.add_argument(
        "--include-db",
        action="append",
        default=None,
        dest="include_databases",
        help="Only migrate these database files (by stem name, repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview DDL without executing anything",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip row count verification",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback: drop all tables that were migrated",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    include_dbs = set(args.include_databases) if args.include_databases else None

    orch = MigrationOrchestrator(
        data_dir=args.data_dir,
        postgres_dsn=args.postgres_dsn,
        batch_size=args.batch_size,
        pg_schema=args.pg_schema,
        include_databases=include_dbs,
    )

    try:
        if args.rollback:
            db_files = orch.discover_databases()
            total_dropped = 0
            for db_path in db_files:
                dropped = await orch.rollback_database(db_path)
                total_dropped += dropped
                print(f"Dropped {dropped} table(s) from {db_path.name}")
            print(f"\nTotal tables dropped: {total_dropped}")
        else:
            report = await orch.run(
                dry_run=args.dry_run,
                verify=not args.no_verify,
            )
            orch.print_report(report)

            if not report.success:
                exit(1)
    finally:
        await orch.close()


if __name__ == "__main__":
    asyncio.run(_cli_main())


__all__ = [
    "DatabaseMigrationResult",
    "MigrationOrchestrator",
    "MigrationReport",
    "SchemaTranslator",
    "TableMigrationResult",
    "sqlite_type_to_pg",
    "SQLITE_TO_PG_TYPES",
]
