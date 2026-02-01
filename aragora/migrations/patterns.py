"""
Zero-Downtime Migration Patterns.

Provides helpers for safe database schema changes that maintain availability:
- Expand/Contract pattern: Add new columns nullable, backfill, then make non-null
- Safe column operations: Avoid exclusive locks on large tables
- Backward compatibility: Ensure old code can still work during migration

Usage:
    from aragora.migrations.patterns import (
        safe_add_column,
        safe_add_nullable_column,
        safe_drop_column,
        safe_rename_column,
        backfill_column,
        validate_migration_safety,
    )

    # In a migration up_fn:
    def up(backend):
        # Phase 1: Expand - add nullable column
        safe_add_nullable_column(backend, "users", "email_verified", "BOOLEAN")

        # Phase 2: Backfill (can be done in batches)
        backfill_column(backend, "users", "email_verified", "FALSE", batch_size=1000)

        # Phase 3: Contract - make non-null (in a separate migration!)
        # safe_set_not_null(backend, "users", "email_verified", default="FALSE")
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.storage.backends import DatabaseBackend

logger = logging.getLogger(__name__)

# SQL identifier: letters, digits, underscores; must start with letter/underscore
_VALID_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_identifier(name: str, kind: str = "identifier") -> str:
    """Validate a SQL identifier to prevent injection.

    Args:
        name: The identifier to validate
        kind: Description for error messages (e.g. 'table', 'column')

    Returns:
        The validated identifier

    Raises:
        ValueError: If the identifier is invalid
    """
    if not name:
        raise ValueError(f"Empty {kind}")
    if len(name) > 128:
        raise ValueError(f"{kind} too long (max 128): {name!r}")
    if not _VALID_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid {kind}: must start with letter/underscore, "
            f"contain only letters, digits, underscores. Got: {name!r}"
        )
    return name


def quote_identifier(name: str, kind: str = "identifier") -> str:
    """Validate and quote a SQL identifier (defense-in-depth).

    First validates the name, then wraps in double quotes with proper escaping.

    Args:
        name: The identifier to quote
        kind: Description for error messages

    Returns:
        The quoted identifier (e.g. '"my_table"')
    """
    validate_identifier(name, kind)
    return '"' + name.replace('"', '""') + '"'


class MigrationRisk(Enum):
    """Risk level for a migration operation."""

    LOW = "low"  # No downtime expected
    MEDIUM = "medium"  # Brief lock, may cause slowdown
    HIGH = "high"  # Extended lock, may cause downtime
    CRITICAL = "critical"  # Will cause downtime, avoid in production


@dataclass
class MigrationValidation:
    """Result of migration safety validation."""

    safe: bool
    risk_level: MigrationRisk
    warnings: list[str]
    recommendations: list[str]


def is_postgresql(backend: "DatabaseBackend") -> bool:
    """Check if backend is PostgreSQL."""
    return backend.__class__.__name__ == "PostgreSQLBackend"


def get_table_row_count(backend: "DatabaseBackend", table: str) -> int:
    """Get approximate row count for a table.

    Uses pg_class for PostgreSQL (fast estimate) or COUNT(*) for SQLite.
    """
    qt = quote_identifier(table, "table")
    if is_postgresql(backend):
        # Fast estimate from pg_class (doesn't lock table)
        result = backend.fetch_one(
            """
            SELECT reltuples::bigint
            FROM pg_class
            WHERE relname = %s
            """,
            (table,),
        )
        return int(result[0]) if result and result[0] >= 0 else 0
    else:
        # SQLite - COUNT(*) is the only option
        result = backend.fetch_one(f"SELECT COUNT(*) FROM {qt}")
        return int(result[0]) if result else 0


def safe_add_column(
    backend: "DatabaseBackend",
    table: str,
    column: str,
    data_type: str,
    nullable: bool = True,
    default: str | None = None,
) -> None:
    """Safely add a column to a table.

    For PostgreSQL:
    - Uses IF NOT EXISTS to make operation idempotent
    - For large tables, adds column as nullable first (no table rewrite)

    Args:
        backend: Database backend
        table: Table name
        column: Column name
        data_type: SQL data type (e.g., "VARCHAR(255)", "INTEGER")
        nullable: Whether column allows NULL (True = safer for large tables)
        default: Default value expression (e.g., "0", "'pending'", "NOW()")
    """
    qt = quote_identifier(table, "table")
    qc = quote_identifier(column, "column")
    # Validate base type keyword (the part before parenthesized size e.g. VARCHAR(255))
    base_type = data_type.split("(")[0].strip()
    validate_identifier(base_type, "data_type")

    if is_postgresql(backend):
        # PostgreSQL: Use ADD COLUMN IF NOT EXISTS (idempotent)
        parts = [f"ALTER TABLE {qt} ADD COLUMN IF NOT EXISTS {qc} {data_type}"]

        if default is not None:
            parts.append(f"DEFAULT {default}")

        if not nullable:
            # Warning: NOT NULL on existing table may require table rewrite
            parts.append("NOT NULL")

        sql = " ".join(parts)
        backend.execute_write(sql)
        logger.info(f"Added column {table}.{column} ({data_type})")
    else:
        # SQLite: Check if column exists first
        columns = backend.fetch_all(f"PRAGMA table_info({qt})")
        existing_columns = {row[1] for row in columns}

        if column not in existing_columns:
            parts = [f"ALTER TABLE {qt} ADD COLUMN {qc} {data_type}"]

            if default is not None:
                parts.append(f"DEFAULT {default}")

            sql = " ".join(parts)
            backend.execute_write(sql)
            logger.info(f"Added column {table}.{column} ({data_type})")
        else:
            logger.debug(f"Column {table}.{column} already exists")


def safe_add_nullable_column(
    backend: "DatabaseBackend",
    table: str,
    column: str,
    data_type: str,
    default: str | None = None,
) -> None:
    """Add a nullable column (safest option for large tables).

    This is the "expand" phase of expand/contract pattern.
    The column can be made NOT NULL later after backfilling.

    Args:
        backend: Database backend
        table: Table name
        column: Column name
        data_type: SQL data type
        default: Default value for new rows (optional)
    """
    safe_add_column(backend, table, column, data_type, nullable=True, default=default)


def safe_drop_column(
    backend: "DatabaseBackend",
    table: str,
    column: str,
    verify_unused: bool = True,
) -> None:
    """Safely drop a column from a table.

    This is the "contract" phase - only call after all code stops using the column.

    Args:
        backend: Database backend
        table: Table name
        column: Column to drop
        verify_unused: If True, checks that column is not referenced by indexes/constraints
    """
    qt = quote_identifier(table, "table")
    qc = quote_identifier(column, "column")

    if is_postgresql(backend):
        if verify_unused:
            # Check for indexes using this column
            indexes = backend.fetch_all(
                """
                SELECT indexname
                FROM pg_indexes
                WHERE tablename = %s
                AND indexdef LIKE %s
                """,
                (table, f"%{column}%"),
            )
            if indexes:
                logger.warning(
                    f"Column {table}.{column} is used by indexes: "
                    f"{[idx[0] for idx in indexes]}. Drop indexes first."
                )
                raise ValueError(f"Column {column} is still referenced by indexes")

        # PostgreSQL: Use IF EXISTS for idempotency
        backend.execute_write(f"ALTER TABLE {qt} DROP COLUMN IF EXISTS {qc}")
        logger.info(f"Dropped column {table}.{column}")
    else:
        # SQLite: Check if column exists
        columns = backend.fetch_all(f"PRAGMA table_info({qt})")
        existing_columns = {row[1] for row in columns}

        if column in existing_columns:
            # SQLite 3.35+ supports DROP COLUMN
            backend.execute_write(f"ALTER TABLE {qt} DROP COLUMN {qc}")
            logger.info(f"Dropped column {table}.{column}")
        else:
            logger.debug(f"Column {table}.{column} does not exist")


def safe_rename_column(
    backend: "DatabaseBackend",
    table: str,
    old_name: str,
    new_name: str,
) -> None:
    """Safely rename a column.

    Note: This requires a brief lock. For zero-downtime, consider:
    1. Add new column (expand)
    2. Copy data in batches
    3. Update code to use new column
    4. Drop old column (contract)

    Args:
        backend: Database backend
        table: Table name
        old_name: Current column name
        new_name: New column name
    """
    qt = quote_identifier(table, "table")
    qo = quote_identifier(old_name, "column")
    qn = quote_identifier(new_name, "column")

    if is_postgresql(backend):
        backend.execute_write(f"ALTER TABLE {qt} RENAME COLUMN {qo} TO {qn}")
    else:
        backend.execute_write(f"ALTER TABLE {qt} RENAME COLUMN {qo} TO {qn}")
    logger.info(f"Renamed column {table}.{old_name} to {new_name}")


def backfill_column(
    backend: "DatabaseBackend",
    table: str,
    column: str,
    value: str,
    where_clause: str | None = None,
    batch_size: int = 1000,
    sleep_between_batches: float = 0.1,
) -> int:
    """Backfill a column in batches to avoid long locks.

    Args:
        backend: Database backend
        table: Table name
        column: Column to backfill
        value: Value expression to set (e.g., "0", "NOW()", "other_column * 2")
        where_clause: Optional WHERE condition (e.g., "status = 'pending'")
        batch_size: Number of rows to update per batch
        sleep_between_batches: Seconds to sleep between batches (reduces load)

    Returns:
        Total number of rows updated
    """
    qt = quote_identifier(table, "table")
    qc = quote_identifier(column, "column")
    total_updated = 0

    # Build base query
    condition = f"{qc} IS NULL"
    if where_clause:
        condition = f"{condition} AND ({where_clause})"

    if is_postgresql(backend):
        # PostgreSQL: Use ctid for efficient batching
        while True:
            result = backend.fetch_one(
                f"""
                WITH batch AS (
                    SELECT ctid
                    FROM {qt}
                    WHERE {condition}
                    LIMIT {batch_size}
                    FOR UPDATE SKIP LOCKED
                )
                UPDATE {qt}
                SET {qc} = {value}
                WHERE ctid IN (SELECT ctid FROM batch)
                RETURNING 1
                """
            )

            if not result:
                # No more rows to update
                count = backend.fetch_one(f"SELECT COUNT(*) FROM {qt} WHERE {condition}")
                if count and count[0] == 0:
                    break
                # Retry if there were locked rows
                time.sleep(sleep_between_batches)
                continue

            batch_count = result[0] if result else 0
            if batch_count == 0:
                break

            total_updated += batch_count
            logger.debug(f"Backfilled {batch_count} rows in {table}.{column}")

            if batch_count < batch_size:
                break

            time.sleep(sleep_between_batches)
    else:
        # SQLite: Simple batch update using rowid
        while True:
            backend.execute_write(
                f"""
                UPDATE {qt}
                SET {qc} = {value}
                WHERE rowid IN (
                    SELECT rowid FROM {qt}
                    WHERE {condition}
                    LIMIT {batch_size}
                )
                """
            )

            # Check how many were updated (SQLite doesn't have RETURNING in older versions)
            remaining = backend.fetch_one(f"SELECT COUNT(*) FROM {qt} WHERE {condition}")

            if not remaining or remaining[0] == 0:
                break

            total_updated += batch_size
            time.sleep(sleep_between_batches)

    logger.info(f"Backfilled {total_updated} rows in {table}.{column}")
    return total_updated


def safe_set_not_null(
    backend: "DatabaseBackend",
    table: str,
    column: str,
    default: str | None = None,
) -> None:
    """Set a column to NOT NULL after backfilling.

    This is the final "contract" step after:
    1. Adding nullable column (expand)
    2. Backfilling all NULL values

    Warning: This may lock the table briefly. Ensure all NULLs are backfilled first.

    Args:
        backend: Database backend
        table: Table name
        column: Column to make NOT NULL
        default: Default value for any remaining NULLs (safety)
    """
    qt = quote_identifier(table, "table")
    qc = quote_identifier(column, "column")

    # First, ensure no NULLs remain
    null_count = backend.fetch_one(f"SELECT COUNT(*) FROM {qt} WHERE {qc} IS NULL")
    if null_count and null_count[0] > 0:
        if default is not None:
            # Backfill remaining NULLs with default
            backend.execute_write(f"UPDATE {qt} SET {qc} = {default} WHERE {qc} IS NULL")
            logger.info(f"Backfilled {null_count[0]} remaining NULLs in {table}.{column}")
        else:
            raise ValueError(
                f"Cannot set NOT NULL: {null_count[0]} NULL values remain in {table}.{column}"
            )

    if is_postgresql(backend):
        backend.execute_write(f"ALTER TABLE {qt} ALTER COLUMN {qc} SET NOT NULL")
    else:
        # SQLite doesn't support ALTER COLUMN, would need table recreation
        logger.warning(
            f"SQLite doesn't support ALTER COLUMN SET NOT NULL. "
            f"Column {table}.{column} remains nullable."
        )
        return

    logger.info(f"Set {table}.{column} to NOT NULL")


def safe_create_index(
    backend: "DatabaseBackend",
    index_name: str,
    table: str,
    columns: list[str],
    unique: bool = False,
    concurrently: bool = True,
) -> None:
    """Create an index without blocking writes.

    For PostgreSQL, uses CONCURRENTLY option to avoid blocking.

    Args:
        backend: Database backend
        index_name: Name for the index
        table: Table to index
        columns: Columns to include in index
        unique: Whether to create a unique index
        concurrently: Use CONCURRENTLY option (PostgreSQL only)
    """
    columns_str = ", ".join(columns)
    unique_str = "UNIQUE " if unique else ""

    if is_postgresql(backend) and concurrently:
        # PostgreSQL: CREATE INDEX CONCURRENTLY doesn't block writes
        # Note: Cannot be run inside a transaction
        backend.execute_write(
            f"CREATE {unique_str}INDEX CONCURRENTLY IF NOT EXISTS {index_name} "
            f"ON {table} ({columns_str})"
        )
    else:
        backend.execute_write(
            f"CREATE {unique_str}INDEX IF NOT EXISTS {index_name} ON {table} ({columns_str})"
        )

    logger.info(f"Created index {index_name} on {table}({columns_str})")


def safe_drop_index(
    backend: "DatabaseBackend",
    index_name: str,
    concurrently: bool = True,
) -> None:
    """Drop an index without blocking writes.

    Args:
        backend: Database backend
        index_name: Index to drop
        concurrently: Use CONCURRENTLY option (PostgreSQL only)
    """
    if is_postgresql(backend) and concurrently:
        backend.execute_write(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name}")
    else:
        backend.execute_write(f"DROP INDEX IF EXISTS {index_name}")

    logger.info(f"Dropped index {index_name}")


def validate_migration_safety(
    backend: "DatabaseBackend",
    operations: list[dict[str, Any]],
) -> MigrationValidation:
    """Validate whether migration operations are safe for production.

    Args:
        backend: Database backend
        operations: List of operation dicts with type and parameters

    Returns:
        MigrationValidation with risk assessment and recommendations

    Example:
        operations = [
            {"type": "add_column", "table": "users", "column": "email", "nullable": False},
            {"type": "drop_column", "table": "users", "column": "legacy_field"},
        ]
        result = validate_migration_safety(backend, operations)
        if not result.safe:
            for warning in result.warnings:
                logger.warning(warning)
    """
    warnings: list[str] = []
    recommendations: list[str] = []
    max_risk = MigrationRisk.LOW

    for op in operations:
        op_type = op.get("type", "unknown")
        table = op.get("table", "unknown")

        # Get table size for risk assessment
        row_count = 0
        try:
            row_count = get_table_row_count(backend, table)
        except Exception as e:
            logger.debug("Failed to get row count for table %s: %s", table, e)

        is_large_table = row_count > 100_000

        if op_type == "add_column":
            nullable = op.get("nullable", True)
            if not nullable and is_large_table:
                warnings.append(
                    f"Adding NOT NULL column to large table {table} ({row_count:,} rows) "
                    "may cause extended lock"
                )
                recommendations.append(
                    "Use expand/contract pattern: add nullable column first, backfill, "
                    "then add NOT NULL constraint"
                )
                max_risk = max(max_risk, MigrationRisk.HIGH, key=lambda x: x.value)

        elif op_type == "drop_column":
            if is_large_table:
                recommendations.append(
                    f"Ensure {table}.{op.get('column')} is not referenced by any code "
                    f"before dropping"
                )
                max_risk = max(max_risk, MigrationRisk.MEDIUM, key=lambda x: x.value)

        elif op_type == "create_index":
            if not op.get("concurrently", True) and is_large_table:
                warnings.append(
                    f"Creating index on large table {table} without CONCURRENTLY will block writes"
                )
                recommendations.append("Use concurrently=True for CREATE INDEX")
                max_risk = max(max_risk, MigrationRisk.HIGH, key=lambda x: x.value)

        elif op_type == "alter_column":
            warnings.append(
                f"ALTER COLUMN on {table} may require table rewrite depending on change"
            )
            max_risk = max(max_risk, MigrationRisk.MEDIUM, key=lambda x: x.value)

    safe = max_risk in (MigrationRisk.LOW, MigrationRisk.MEDIUM)

    return MigrationValidation(
        safe=safe,
        risk_level=max_risk,
        warnings=warnings,
        recommendations=recommendations,
    )


__all__ = [
    "MigrationRisk",
    "MigrationValidation",
    "safe_add_column",
    "safe_add_nullable_column",
    "safe_drop_column",
    "safe_rename_column",
    "backfill_column",
    "safe_set_not_null",
    "safe_create_index",
    "safe_drop_index",
    "validate_migration_safety",
    "get_table_row_count",
]
