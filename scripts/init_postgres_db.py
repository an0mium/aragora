#!/usr/bin/env python3
"""
Initialize PostgreSQL database for Aragora.

This script sets up the PostgreSQL database with all required tables and schemas.
It can be run multiple times safely - tables are created with IF NOT EXISTS.

Usage:
    # Using environment variable
    export ARAGORA_POSTGRES_DSN="postgresql://user:pass@localhost:5432/aragora"
    python scripts/init_postgres_db.py

    # Or with explicit DSN
    python scripts/init_postgres_db.py --dsn "postgresql://user:pass@localhost:5432/aragora"

    # Verify setup
    python scripts/init_postgres_db.py --verify

    # Run from SQL file directly
    psql -U postgres -d aragora -f aragora/db/schema/postgres_schema.sql

Requirements:
    pip install asyncpg
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add aragora to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def get_pool(dsn: str):
    """Create connection pool."""
    try:
        import asyncpg
    except ImportError:
        logger.error("asyncpg not installed. Run: pip install asyncpg")
        sys.exit(1)

    return await asyncpg.create_pool(
        dsn,
        min_size=1,
        max_size=5,
        command_timeout=60,
    )


async def run_schema_file(pool, schema_path: Path) -> bool:
    """Run SQL schema file."""
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False

    schema_sql = schema_path.read_text()

    async with pool.acquire() as conn:
        try:
            await conn.execute(schema_sql)
            logger.info(f"Successfully executed schema from {schema_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to execute schema: {e}")
            return False


async def init_all_stores(pool) -> dict[str, bool]:
    """Initialize all storage modules that have PostgreSQL implementations."""
    results = {}

    # Import and initialize each store
    stores_to_init = [
        ("user_store", "aragora.storage.user_store", "PostgresUserStore"),
        ("governance_store", "aragora.storage.governance_store", "PostgresGovernanceStore"),
        ("marketplace_store", "aragora.storage.marketplace_store", "PostgresMarketplaceStore"),
        ("integration_store", "aragora.storage.integration_store", "PostgresIntegrationStore"),
        (
            "webhook_config_store",
            "aragora.storage.webhook_config_store",
            "PostgresWebhookConfigStore",
        ),
        ("webhook_store", "aragora.storage.webhook_store", "PostgresWebhookStore"),
        ("gmail_token_store", "aragora.storage.gmail_token_store", "PostgresGmailTokenStore"),
        ("job_queue_store", "aragora.storage.job_queue_store", "PostgresJobQueueStore"),
        (
            "federation_registry_store",
            "aragora.storage.federation_registry_store",
            "PostgresFederationRegistryStore",
        ),
        ("gauntlet_run_store", "aragora.storage.gauntlet_run_store", "PostgresGauntletRunStore"),
        (
            "finding_workflow_store",
            "aragora.storage.finding_workflow_store",
            "PostgresFindingWorkflowStore",
        ),
        (
            "approval_request_store",
            "aragora.storage.approval_request_store",
            "PostgresApprovalRequestStore",
        ),
    ]

    for name, module_path, class_name in stores_to_init:
        try:
            module = __import__(module_path, fromlist=[class_name])
            store_class = getattr(module, class_name)
            store = store_class(pool)

            # Initialize schema
            if hasattr(store, "initialize"):
                await store.initialize()
            elif hasattr(store, "_ensure_schema"):
                await store._ensure_schema()

            results[name] = True
            logger.info(f"Initialized {name}")
        except ImportError as e:
            logger.warning(f"Could not import {name}: {e}")
            results[name] = False
        except Exception as e:
            logger.error(f"Failed to initialize {name}: {e}")
            results[name] = False

    return results


async def verify_tables(pool) -> dict[str, bool]:
    """Verify that essential tables exist."""
    essential_tables = [
        "users",
        "organizations",
        "audit_log",
        "approval_requests",
        "integrations",
        "webhook_configs",
        "job_queue",
        "marketplace_templates",
        "_schema_versions",
    ]

    results = {}

    async with pool.acquire() as conn:
        for table in essential_tables:
            try:
                # Check in both public and aragora schemas
                exists = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = $1
                        AND (table_schema = 'public' OR table_schema = 'aragora')
                    )
                    """,
                    table,
                )
                results[table] = exists
                status = "OK" if exists else "MISSING"
                logger.info(f"  {table}: {status}")
            except Exception as e:
                results[table] = False
                logger.error(f"  {table}: ERROR - {e}")

    return results


async def main():
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL database for Aragora")
    parser.add_argument(
        "--dsn",
        help="PostgreSQL connection string (or set ARAGORA_POSTGRES_DSN)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify tables exist, don't create",
    )
    parser.add_argument(
        "--schema-file",
        action="store_true",
        help="Use consolidated schema file instead of individual stores",
    )
    parser.add_argument(
        "--alembic",
        action="store_true",
        help="Run Alembic migrations after initialization (requires alembic)",
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

    # Get DSN
    dsn = args.dsn or os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        logger.error(
            "No PostgreSQL DSN configured.\n"
            "Set ARAGORA_POSTGRES_DSN or use --dsn argument.\n"
            "Example: postgresql://user:pass@localhost:5432/aragora"
        )
        return 1

    # Redact password for logging
    dsn_redacted = dsn
    if "@" in dsn:
        parts = dsn.split("@")
        user_pass = parts[0].rsplit(":", 1)
        if len(user_pass) == 2 and "://" in user_pass[0]:
            dsn_redacted = f"{user_pass[0]}:***@{parts[1]}"

    logger.info(f"Connecting to: {dsn_redacted}")

    try:
        pool = await get_pool(dsn)
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return 1

    try:
        if args.verify:
            logger.info("Verifying database tables...")
            results = await verify_tables(pool)
            missing = [t for t, exists in results.items() if not exists]
            if missing:
                logger.warning(f"Missing tables: {', '.join(missing)}")
                return 1
            logger.info("All essential tables exist!")
            return 0

        if args.schema_file:
            # Use consolidated schema file
            schema_path = (
                Path(__file__).parent.parent / "aragora" / "db" / "schema" / "postgres_schema.sql"
            )
            logger.info(f"Initializing from schema file: {schema_path}")
            success = await run_schema_file(pool, schema_path)
            if not success:
                return 1
        else:
            # Initialize individual stores
            logger.info("Initializing storage modules...")
            results = await init_all_stores(pool)
            failed = [name for name, success in results.items() if not success]
            if failed:
                logger.warning(f"Some stores failed to initialize: {', '.join(failed)}")

        # Verify
        logger.info("\nVerifying tables...")
        verify_results = await verify_tables(pool)
        missing = [t for t, exists in verify_results.items() if not exists]

        if missing:
            logger.warning(f"Some tables are missing: {', '.join(missing)}")
            logger.info("You may need to run individual store initializations.")
        else:
            logger.info("\nDatabase initialization complete!")

        return 0

    finally:
        await pool.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
