#!/usr/bin/env python3
"""
SQLite to Supabase (PostgreSQL) Migration Script.

Migrates data from SQLite-based stores to PostgreSQL for production deployment.

Usage:
    # Migrate all stores
    python scripts/migrate_sqlite_to_supabase.py --all

    # Migrate specific stores
    python scripts/migrate_sqlite_to_supabase.py --stores workflow jobs integrations

    # Dry run (show what would be migrated)
    python scripts/migrate_sqlite_to_supabase.py --all --dry-run

    # Custom paths
    python scripts/migrate_sqlite_to_supabase.py --all \\
        --data-dir /path/to/sqlite/data \\
        --postgres-dsn "postgresql://user:pass@host:5432/db"

Requirements:
    - asyncpg: pip install asyncpg
    - Set DATABASE_URL environment variable or use --postgres-dsn

Note:
    This script performs INSERT ... ON CONFLICT DO NOTHING to avoid
    overwriting existing data. Run multiple times safely.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def get_pool(dsn: str) -> Any:
    """Create asyncpg connection pool."""
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg not installed. Run: pip install asyncpg")
        sys.exit(1)

    return await asyncpg.create_pool(dsn, min_size=2, max_size=10)


def unix_to_iso(timestamp: float | None) -> str | None:
    """Convert Unix timestamp to ISO format."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).isoformat()


async def migrate_workflow_store(
    sqlite_path: Path,
    pool: Any,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Migrate finding_workflows table.

    Returns:
        Tuple of (migrated_count, error_count)
    """
    if not sqlite_path.exists():
        logger.warning(f"SQLite database not found: {sqlite_path}")
        return 0, 0

    logger.info(f"Migrating workflow store from: {sqlite_path}")

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='finding_workflows'"
    )
    if not cursor.fetchone():
        logger.info("No finding_workflows table found, skipping")
        conn.close()
        return 0, 0

    cursor.execute("SELECT COUNT(*) FROM finding_workflows")
    total = cursor.fetchone()[0]
    logger.info(f"Found {total} workflow records to migrate")

    if dry_run:
        conn.close()
        return total, 0

    cursor.execute("SELECT * FROM finding_workflows")
    rows = cursor.fetchall()

    migrated = 0
    errors = 0

    # Initialize schema
    async with pool.acquire() as pg_conn:
        await pg_conn.execute("""
            CREATE TABLE IF NOT EXISTS finding_workflows (
                finding_id TEXT PRIMARY KEY,
                current_state TEXT NOT NULL DEFAULT 'open',
                assigned_to TEXT,
                assigned_by TEXT,
                assigned_at TIMESTAMPTZ,
                priority INTEGER DEFAULT 3,
                due_date TIMESTAMPTZ,
                parent_finding_id TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                data_json JSONB NOT NULL
            )
        """)
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_workflow_assigned_to ON finding_workflows(assigned_to)"
        )
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_workflow_state ON finding_workflows(current_state)"
        )
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_workflow_due_date ON finding_workflows(due_date)"
        )

        for row in rows:
            try:
                await pg_conn.execute(
                    """
                    INSERT INTO finding_workflows
                    (finding_id, current_state, assigned_to, assigned_by,
                     assigned_at, priority, due_date, parent_finding_id,
                     created_at, updated_at, data_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
                    ON CONFLICT (finding_id) DO NOTHING
                    """,
                    row["finding_id"],
                    row["current_state"],
                    row["assigned_to"],
                    row["assigned_by"],
                    unix_to_iso(row["assigned_at"]) if row["assigned_at"] else None,
                    row["priority"],
                    row["due_date"],
                    row["parent_finding_id"],
                    unix_to_iso(row["created_at"]),
                    unix_to_iso(row["updated_at"]),
                    row["data_json"],
                )
                migrated += 1
            except Exception as e:
                logger.error(f"Error migrating workflow {row['finding_id']}: {e}")
                errors += 1

    conn.close()
    logger.info(f"Workflow migration complete: {migrated} migrated, {errors} errors")
    return migrated, errors


async def migrate_job_queue_store(
    sqlite_path: Path,
    pool: Any,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Migrate job_queue table.

    Returns:
        Tuple of (migrated_count, error_count)
    """
    if not sqlite_path.exists():
        logger.warning(f"SQLite database not found: {sqlite_path}")
        return 0, 0

    logger.info(f"Migrating job queue store from: {sqlite_path}")

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='job_queue'")
    if not cursor.fetchone():
        logger.info("No job_queue table found, skipping")
        conn.close()
        return 0, 0

    cursor.execute("SELECT COUNT(*) FROM job_queue")
    total = cursor.fetchone()[0]
    logger.info(f"Found {total} job records to migrate")

    if dry_run:
        conn.close()
        return total, 0

    cursor.execute("SELECT * FROM job_queue")
    rows = cursor.fetchall()

    migrated = 0
    errors = 0

    # Initialize schema
    async with pool.acquire() as pg_conn:
        await pg_conn.execute("""
            CREATE TABLE IF NOT EXISTS job_queue (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                payload_json JSONB,
                status TEXT NOT NULL DEFAULT 'pending',
                priority INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                started_at TIMESTAMPTZ,
                completed_at TIMESTAMPTZ,
                attempts INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 3,
                worker_id TEXT,
                error TEXT,
                result_json JSONB,
                user_id TEXT,
                workspace_id TEXT
            )
        """)
        await pg_conn.execute("CREATE INDEX IF NOT EXISTS idx_job_status ON job_queue(status)")
        await pg_conn.execute("CREATE INDEX IF NOT EXISTS idx_job_type ON job_queue(job_type)")
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_priority ON job_queue(priority DESC)"
        )
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_pending ON job_queue(status, priority DESC, created_at)"
        )

        for row in rows:
            try:
                await pg_conn.execute(
                    """
                    INSERT INTO job_queue
                    (id, job_type, payload_json, status, priority, created_at, updated_at,
                     started_at, completed_at, attempts, max_attempts, worker_id, error,
                     result_json, user_id, workspace_id)
                    VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb, $15, $16)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    row["id"],
                    row["job_type"],
                    row["payload_json"],
                    row["status"],
                    row["priority"],
                    unix_to_iso(row["created_at"]),
                    unix_to_iso(row["updated_at"]),
                    unix_to_iso(row["started_at"]) if row["started_at"] else None,
                    unix_to_iso(row["completed_at"]) if row["completed_at"] else None,
                    row["attempts"],
                    row["max_attempts"],
                    row["worker_id"],
                    row["error"],
                    row["result_json"],
                    row["user_id"],
                    row["workspace_id"],
                )
                migrated += 1
            except Exception as e:
                logger.error(f"Error migrating job {row['id']}: {e}")
                errors += 1

    conn.close()
    logger.info(f"Job queue migration complete: {migrated} migrated, {errors} errors")
    return migrated, errors


async def migrate_integration_store(
    sqlite_path: Path,
    pool: Any,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Migrate integrations and user_id_mappings tables.

    Returns:
        Tuple of (migrated_count, error_count)
    """
    if not sqlite_path.exists():
        logger.warning(f"SQLite database not found: {sqlite_path}")
        return 0, 0

    logger.info(f"Migrating integration store from: {sqlite_path}")

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='integrations'")
    if not cursor.fetchone():
        logger.info("No integrations table found, skipping")
        conn.close()
        return 0, 0

    cursor.execute("SELECT COUNT(*) FROM integrations")
    total_integrations = cursor.fetchone()[0]

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='user_id_mappings'"
    )
    total_mappings = 0
    if cursor.fetchone():
        cursor.execute("SELECT COUNT(*) FROM user_id_mappings")
        total_mappings = cursor.fetchone()[0]

    logger.info(f"Found {total_integrations} integrations, {total_mappings} mappings to migrate")

    if dry_run:
        conn.close()
        return total_integrations + total_mappings, 0

    migrated = 0
    errors = 0

    # Initialize schema
    async with pool.acquire() as pg_conn:
        await pg_conn.execute("""
            CREATE TABLE IF NOT EXISTS integrations (
                integration_type TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'default',
                enabled BOOLEAN NOT NULL DEFAULT TRUE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                notify_on_consensus BOOLEAN DEFAULT TRUE,
                notify_on_debate_end BOOLEAN DEFAULT TRUE,
                notify_on_error BOOLEAN DEFAULT FALSE,
                notify_on_leaderboard BOOLEAN DEFAULT FALSE,
                settings_json JSONB,
                messages_sent INTEGER DEFAULT 0,
                errors_24h INTEGER DEFAULT 0,
                last_activity TIMESTAMPTZ,
                last_error TEXT,
                workspace_id TEXT,
                PRIMARY KEY (user_id, integration_type)
            )
        """)
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_integrations_user ON integrations(user_id)"
        )
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_integrations_type ON integrations(integration_type)"
        )

        await pg_conn.execute("""
            CREATE TABLE IF NOT EXISTS user_id_mappings (
                email TEXT NOT NULL,
                platform TEXT NOT NULL,
                platform_user_id TEXT NOT NULL,
                display_name TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                user_id TEXT NOT NULL DEFAULT 'default',
                PRIMARY KEY (user_id, platform, email)
            )
        """)
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mappings_email ON user_id_mappings(email)"
        )
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mappings_platform ON user_id_mappings(platform)"
        )

        # Migrate integrations
        cursor.execute("SELECT * FROM integrations")
        for row in cursor.fetchall():
            try:
                await pg_conn.execute(
                    """
                    INSERT INTO integrations
                    (integration_type, user_id, enabled, created_at, updated_at,
                     notify_on_consensus, notify_on_debate_end, notify_on_error,
                     notify_on_leaderboard, settings_json, messages_sent, errors_24h,
                     last_activity, last_error, workspace_id)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11, $12, $13, $14, $15)
                    ON CONFLICT (user_id, integration_type) DO NOTHING
                    """,
                    row["integration_type"],
                    row["user_id"],
                    bool(row["enabled"]),
                    unix_to_iso(row["created_at"]),
                    unix_to_iso(row["updated_at"]),
                    bool(row["notify_on_consensus"]),
                    bool(row["notify_on_debate_end"]),
                    bool(row["notify_on_error"]),
                    bool(row["notify_on_leaderboard"]),
                    row["settings_json"],
                    row["messages_sent"],
                    row["errors_24h"],
                    unix_to_iso(row["last_activity"]) if row["last_activity"] else None,
                    row["last_error"],
                    row["workspace_id"],
                )
                migrated += 1
            except Exception as e:
                logger.error(f"Error migrating integration {row['integration_type']}: {e}")
                errors += 1

        # Migrate user_id_mappings
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='user_id_mappings'"
        )
        if cursor.fetchone():
            cursor.execute("SELECT * FROM user_id_mappings")
            for row in cursor.fetchall():
                try:
                    await pg_conn.execute(
                        """
                        INSERT INTO user_id_mappings
                        (email, platform, platform_user_id, display_name,
                         created_at, updated_at, user_id)
                        VALUES ($1, $2, $3, $4, $5, $6, $7)
                        ON CONFLICT (user_id, platform, email) DO NOTHING
                        """,
                        row["email"],
                        row["platform"],
                        row["platform_user_id"],
                        row["display_name"],
                        unix_to_iso(row["created_at"]),
                        unix_to_iso(row["updated_at"]),
                        row["user_id"],
                    )
                    migrated += 1
                except Exception as e:
                    logger.error(f"Error migrating mapping {row['email']}: {e}")
                    errors += 1

    conn.close()
    logger.info(f"Integration migration complete: {migrated} migrated, {errors} errors")
    return migrated, errors


async def migrate_checkpoint_store(
    sqlite_path: Path,
    pool: Any,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Migrate workflow_checkpoints table (if using file-based).

    Note: Checkpoint store may use file-based storage. This handles
    SQLite-based checkpoints if present.

    Returns:
        Tuple of (migrated_count, error_count)
    """
    if not sqlite_path.exists():
        logger.warning(f"SQLite database not found: {sqlite_path}")
        return 0, 0

    logger.info(f"Migrating checkpoint store from: {sqlite_path}")

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if table exists
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_checkpoints'"
    )
    if not cursor.fetchone():
        logger.info("No workflow_checkpoints table found, skipping")
        conn.close()
        return 0, 0

    cursor.execute("SELECT COUNT(*) FROM workflow_checkpoints")
    total = cursor.fetchone()[0]
    logger.info(f"Found {total} checkpoint records to migrate")

    if dry_run:
        conn.close()
        return total, 0

    cursor.execute("SELECT * FROM workflow_checkpoints")
    rows = cursor.fetchall()

    migrated = 0
    errors = 0

    # Initialize schema
    async with pool.acquire() as pg_conn:
        await pg_conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                definition_id TEXT NOT NULL,
                current_step TEXT,
                completed_steps TEXT[] DEFAULT '{}',
                step_outputs JSONB DEFAULT '{}',
                context_state JSONB DEFAULT '{}',
                checksum TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wf_checkpoints_workflow_id ON workflow_checkpoints(workflow_id)"
        )
        await pg_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_wf_checkpoints_created_at ON workflow_checkpoints(created_at DESC)"
        )

        for row in rows:
            try:
                # Parse completed_steps from JSON if stored as string
                completed_steps = row.get("completed_steps", [])
                if isinstance(completed_steps, str):
                    completed_steps = json.loads(completed_steps)

                await pg_conn.execute(
                    """
                    INSERT INTO workflow_checkpoints
                    (id, workflow_id, definition_id, current_step, completed_steps,
                     step_outputs, context_state, checksum, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8, $9, $10)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    row["id"],
                    row["workflow_id"],
                    row["definition_id"],
                    row.get("current_step"),
                    completed_steps,
                    row.get("step_outputs", "{}"),
                    row.get("context_state", "{}"),
                    row.get("checksum"),
                    unix_to_iso(row.get("created_at")),
                    unix_to_iso(row.get("updated_at")),
                )
                migrated += 1
            except Exception as e:
                logger.error(f"Error migrating checkpoint {row['id']}: {e}")
                errors += 1

    conn.close()
    logger.info(f"Checkpoint migration complete: {migrated} migrated, {errors} errors")
    return migrated, errors


async def run_migration(
    data_dir: Path,
    postgres_dsn: str,
    stores: list[str],
    dry_run: bool = False,
) -> dict[str, tuple[int, int]]:
    """
    Run migration for specified stores.

    Args:
        data_dir: Directory containing SQLite databases
        postgres_dsn: PostgreSQL connection string
        stores: List of stores to migrate
        dry_run: If True, only count records without migrating

    Returns:
        Dict mapping store name to (migrated, errors) tuple
    """
    results: dict[str, tuple[int, int]] = {}

    if dry_run:
        logger.info("=== DRY RUN MODE - No data will be modified ===")

    pool = await get_pool(postgres_dsn)

    try:
        if "workflow" in stores:
            db_path = data_dir / "finding_workflows.db"
            results["workflow"] = await migrate_workflow_store(db_path, pool, dry_run)

        if "jobs" in stores:
            db_path = data_dir / "job_queue.db"
            results["jobs"] = await migrate_job_queue_store(db_path, pool, dry_run)

        if "integrations" in stores:
            db_path = data_dir / "integrations.db"
            results["integrations"] = await migrate_integration_store(db_path, pool, dry_run)

        if "checkpoints" in stores:
            db_path = data_dir / "checkpoints.db"
            results["checkpoints"] = await migrate_checkpoint_store(db_path, pool, dry_run)

    finally:
        await pool.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate SQLite stores to PostgreSQL (Supabase)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Migrate all stores
    python scripts/migrate_sqlite_to_supabase.py --all

    # Dry run to see what would be migrated
    python scripts/migrate_sqlite_to_supabase.py --all --dry-run

    # Migrate specific stores
    python scripts/migrate_sqlite_to_supabase.py --stores workflow jobs

    # Custom paths
    python scripts/migrate_sqlite_to_supabase.py --all \\
        --data-dir /var/lib/aragora \\
        --postgres-dsn "postgresql://user:pass@host:5432/db"
        """,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Migrate all stores",
    )
    parser.add_argument(
        "--stores",
        nargs="+",
        choices=["workflow", "jobs", "integrations", "checkpoints"],
        help="Specific stores to migrate",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.getenv("ARAGORA_DATA_DIR", "/tmp/aragora")),
        help="Directory containing SQLite databases",
    )
    parser.add_argument(
        "--postgres-dsn",
        default=os.getenv("DATABASE_URL") or os.getenv("ARAGORA_POSTGRES_DSN"),
        help="PostgreSQL connection string (default: DATABASE_URL env var)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count records without migrating",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.all and not args.stores:
        parser.error("Either --all or --stores must be specified")

    if not args.postgres_dsn:
        parser.error(
            "PostgreSQL connection string required. Set DATABASE_URL environment variable or use --postgres-dsn"
        )

    stores = ["workflow", "jobs", "integrations", "checkpoints"] if args.all else args.stores

    print("=" * 60)
    print(" SQLite to PostgreSQL Migration")
    print("=" * 60)
    print(f"\nData directory: {args.data_dir}")
    print(f"Stores to migrate: {', '.join(stores)}")
    print(f"Dry run: {args.dry_run}")
    print()

    results = asyncio.run(
        run_migration(
            args.data_dir,
            args.postgres_dsn,
            stores,
            args.dry_run,
        )
    )

    # Summary
    print("\n" + "=" * 60)
    print(" Migration Summary")
    print("=" * 60)

    total_migrated = 0
    total_errors = 0

    for store, (migrated, errors) in results.items():
        total_migrated += migrated
        total_errors += errors
        status = "OK" if errors == 0 else "ERRORS"
        print(f"  {store:<15} {migrated:>6} migrated, {errors:>3} errors [{status}]")

    print("-" * 60)
    print(f"  {'TOTAL':<15} {total_migrated:>6} migrated, {total_errors:>3} errors")

    if args.dry_run:
        print("\n[DRY RUN] No data was modified. Remove --dry-run to perform migration.")

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
