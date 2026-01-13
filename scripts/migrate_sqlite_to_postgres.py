#!/usr/bin/env python3
"""
SQLite to PostgreSQL Migration Script

Migrates data from SQLite databases to PostgreSQL for production deployment.
Handles schema translation, data type conversion, and batch transfers.

Usage:
    # Dry run (shows what would be migrated)
    python scripts/migrate_sqlite_to_postgres.py --dry-run

    # Migrate all databases
    python scripts/migrate_sqlite_to_postgres.py

    # Migrate specific database
    python scripts/migrate_sqlite_to_postgres.py --database debates

    # With custom paths
    python scripts/migrate_sqlite_to_postgres.py --source-dir .nomic --target-url postgresql://...
"""

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# SQLite to PostgreSQL type mapping
TYPE_MAP = {
    "INTEGER": "BIGINT",
    "INT": "BIGINT",
    "REAL": "DOUBLE PRECISION",
    "FLOAT": "DOUBLE PRECISION",
    "TEXT": "TEXT",
    "BLOB": "BYTEA",
    "BOOLEAN": "BOOLEAN",
    "DATETIME": "TIMESTAMP WITH TIME ZONE",
    "TIMESTAMP": "TIMESTAMP WITH TIME ZONE",
    "JSON": "JSONB",
}


@dataclass
class MigrationStats:
    """Statistics for a migration run."""

    tables_migrated: int = 0
    rows_migrated: int = 0
    tables_failed: int = 0
    rows_failed: int = 0
    duration_seconds: float = 0.0


def sqlite_type_to_postgres(sqlite_type: str) -> str:
    """Convert SQLite type to PostgreSQL type."""
    # Normalize type (remove size constraints like VARCHAR(255))
    base_type = re.sub(r"\([^)]+\)", "", sqlite_type.upper()).strip()

    # Handle PRIMARY KEY AUTOINCREMENT
    if "AUTOINCREMENT" in sqlite_type.upper():
        return "BIGSERIAL"

    return TYPE_MAP.get(base_type, "TEXT")


def translate_create_table(sqlite_sql: str) -> str:
    """Translate SQLite CREATE TABLE to PostgreSQL syntax."""
    # Replace SQLite-specific syntax
    sql = sqlite_sql

    # Remove IF NOT EXISTS (we'll handle this separately)
    sql = re.sub(r"IF NOT EXISTS\s+", "", sql, flags=re.IGNORECASE)

    # Replace AUTOINCREMENT with SERIAL
    sql = re.sub(
        r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT",
        "BIGSERIAL PRIMARY KEY",
        sql,
        flags=re.IGNORECASE,
    )

    # Replace remaining type names
    for sqlite_type, pg_type in TYPE_MAP.items():
        sql = re.sub(rf"\b{sqlite_type}\b", pg_type, sql, flags=re.IGNORECASE)

    # Handle datetime defaults
    sql = re.sub(
        r"DEFAULT\s+CURRENT_TIMESTAMP",
        "DEFAULT CURRENT_TIMESTAMP",
        sql,
        flags=re.IGNORECASE,
    )

    # Handle boolean defaults
    sql = re.sub(r"DEFAULT\s+1\b", "DEFAULT TRUE", sql, flags=re.IGNORECASE)
    sql = re.sub(r"DEFAULT\s+0\b", "DEFAULT FALSE", sql, flags=re.IGNORECASE)

    return sql


def get_sqlite_databases(source_dir: Path) -> list[Path]:
    """Find all SQLite database files in the source directory."""
    databases = []

    # Known database locations
    known_dbs = [
        "aragora.db",
        "debates.db",
        "memory.db",
        "consensus.db",
        "elo.db",
        "calibration.db",
        "evolution.db",
        "sessions/default_telemetry/telemetry.db",
    ]

    for db_name in known_dbs:
        db_path = source_dir / db_name
        if db_path.exists():
            databases.append(db_path)

    # Also search for any .db files
    for db_path in source_dir.rglob("*.db"):
        if db_path not in databases and db_path.stat().st_size > 0:
            databases.append(db_path)

    return databases


def get_table_schema(sqlite_conn: sqlite3.Connection, table_name: str) -> str:
    """Get the CREATE TABLE statement for a SQLite table."""
    cursor = sqlite_conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    )
    row = cursor.fetchone()
    return row[0] if row else ""


def get_table_names(sqlite_conn: sqlite3.Connection) -> list[str]:
    """Get all table names from a SQLite database."""
    cursor = sqlite_conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )
    return [row[0] for row in cursor.fetchall()]


def convert_value(value: Any, col_type: str) -> Any:
    """Convert a SQLite value to PostgreSQL compatible format."""
    if value is None:
        return None

    col_type_upper = col_type.upper()

    # Handle JSON/JSONB
    if "JSON" in col_type_upper:
        if isinstance(value, str):
            try:
                # Validate it's valid JSON
                json.loads(value)
                return value
            except json.JSONDecodeError:
                return json.dumps(value)
        return json.dumps(value)

    # Handle boolean
    if "BOOL" in col_type_upper:
        if isinstance(value, int):
            return value != 0
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)

    # Handle datetime/timestamp
    if "TIMESTAMP" in col_type_upper or "DATETIME" in col_type_upper:
        if isinstance(value, str):
            # Try to parse various datetime formats
            for fmt in [
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S",
            ]:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return value

    return value


def migrate_table(
    sqlite_conn: sqlite3.Connection,
    pg_conn: Any,
    table_name: str,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Migrate a single table from SQLite to PostgreSQL."""
    logger.info(f"Migrating table: {table_name}")

    # Get schema
    sqlite_schema = get_table_schema(sqlite_conn, table_name)
    if not sqlite_schema:
        logger.warning(f"Could not get schema for table: {table_name}")
        return 0, 0

    # Translate schema
    pg_schema = translate_create_table(sqlite_schema)

    if dry_run:
        logger.info(f"Would create table with schema:\n{pg_schema}")
    else:
        # Create table in PostgreSQL
        pg_cursor = pg_conn.cursor()
        try:
            # Drop existing table if present
            pg_cursor.execute(f'DROP TABLE IF EXISTS "{table_name}" CASCADE')
            pg_cursor.execute(pg_schema)
            pg_conn.commit()
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            pg_conn.rollback()
            return 0, 1

    # Get column info
    sqlite_cursor = sqlite_conn.execute(f'PRAGMA table_info("{table_name}")')
    columns = [(row[1], row[2]) for row in sqlite_cursor.fetchall()]
    col_names = [col[0] for col in columns]
    col_types = [col[1] for col in columns]

    # Count rows
    count_cursor = sqlite_conn.execute(f'SELECT COUNT(*) FROM "{table_name}"')
    total_rows = count_cursor.fetchone()[0]

    if dry_run:
        logger.info(f"Would migrate {total_rows} rows from {table_name}")
        return total_rows, 0

    # Migrate data in batches
    rows_migrated = 0
    rows_failed = 0

    placeholders = ", ".join(["%s"] * len(col_names))
    insert_sql = f'INSERT INTO "{table_name}" ({", ".join(col_names)}) VALUES ({placeholders})'

    pg_cursor = pg_conn.cursor()

    # Fetch and insert in batches
    data_cursor = sqlite_conn.execute(f'SELECT * FROM "{table_name}"')

    while True:
        batch = data_cursor.fetchmany(batch_size)
        if not batch:
            break

        converted_batch = []
        for row in batch:
            converted_row = tuple(
                convert_value(val, col_types[i]) for i, val in enumerate(row)
            )
            converted_batch.append(converted_row)

        try:
            pg_cursor.executemany(insert_sql, converted_batch)
            pg_conn.commit()
            rows_migrated += len(converted_batch)
            logger.info(f"  Migrated {rows_migrated}/{total_rows} rows")
        except Exception as e:
            logger.error(f"Failed to insert batch in {table_name}: {e}")
            pg_conn.rollback()
            rows_failed += len(converted_batch)

    return rows_migrated, rows_failed


def migrate_database(
    sqlite_path: Path,
    pg_conn: Any,
    prefix: str = "",
    dry_run: bool = False,
) -> MigrationStats:
    """Migrate an entire SQLite database to PostgreSQL."""
    logger.info(f"Opening SQLite database: {sqlite_path}")

    stats = MigrationStats()
    start_time = datetime.now()

    try:
        sqlite_conn = sqlite3.connect(str(sqlite_path))
        tables = get_table_names(sqlite_conn)

        logger.info(f"Found {len(tables)} tables to migrate")

        for table_name in tables:
            # Apply prefix if needed
            pg_table_name = f"{prefix}{table_name}" if prefix else table_name

            rows, failed = migrate_table(
                sqlite_conn, pg_conn, table_name, dry_run=dry_run
            )

            if failed:
                stats.tables_failed += 1
                stats.rows_failed += failed
            else:
                stats.tables_migrated += 1
                stats.rows_migrated += rows

        sqlite_conn.close()

    except Exception as e:
        logger.error(f"Failed to migrate database {sqlite_path}: {e}")
        stats.tables_failed += 1

    stats.duration_seconds = (datetime.now() - start_time).total_seconds()
    return stats


def create_postgres_connection(database_url: str) -> Any:
    """Create a PostgreSQL connection from DATABASE_URL."""
    try:
        import psycopg2

        return psycopg2.connect(database_url)
    except ImportError:
        logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate SQLite databases to PostgreSQL"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(".nomic"),
        help="Directory containing SQLite databases (default: .nomic)",
    )
    parser.add_argument(
        "--target-url",
        type=str,
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection URL (default: DATABASE_URL env var)",
    )
    parser.add_argument(
        "--database",
        type=str,
        help="Migrate specific database only (e.g., 'debates')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of rows to insert per batch (default: 1000)",
    )

    args = parser.parse_args()

    if not args.target_url:
        logger.error("No PostgreSQL URL provided. Set DATABASE_URL or use --target-url")
        sys.exit(1)

    if not args.source_dir.exists():
        logger.error(f"Source directory does not exist: {args.source_dir}")
        sys.exit(1)

    # Find databases
    databases = get_sqlite_databases(args.source_dir)

    if args.database:
        databases = [db for db in databases if args.database in db.name]

    if not databases:
        logger.warning("No SQLite databases found to migrate")
        sys.exit(0)

    logger.info(f"Found {len(databases)} database(s) to migrate:")
    for db in databases:
        logger.info(f"  - {db}")

    if args.dry_run:
        logger.info("DRY RUN - No changes will be made")

    # Connect to PostgreSQL
    pg_conn: Optional[Any] = None
    if not args.dry_run:
        logger.info("Connecting to PostgreSQL...")
        pg_conn = create_postgres_connection(args.target_url)
        logger.info("Connected successfully")

    # Migrate each database
    total_stats = MigrationStats()

    for db_path in databases:
        # Use database filename as table prefix for non-main databases
        prefix = ""
        if db_path.name != "aragora.db":
            prefix = db_path.stem.replace("-", "_") + "_"

        stats = migrate_database(db_path, pg_conn, prefix=prefix, dry_run=args.dry_run)

        total_stats.tables_migrated += stats.tables_migrated
        total_stats.rows_migrated += stats.rows_migrated
        total_stats.tables_failed += stats.tables_failed
        total_stats.rows_failed += stats.rows_failed
        total_stats.duration_seconds += stats.duration_seconds

    # Close connection
    if pg_conn:
        pg_conn.close()

    # Print summary
    logger.info("=" * 60)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tables migrated:   {total_stats.tables_migrated}")
    logger.info(f"Rows migrated:     {total_stats.rows_migrated}")
    logger.info(f"Tables failed:     {total_stats.tables_failed}")
    logger.info(f"Rows failed:       {total_stats.rows_failed}")
    logger.info(f"Duration:          {total_stats.duration_seconds:.2f}s")

    if total_stats.tables_failed > 0:
        logger.warning("Some tables failed to migrate. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
