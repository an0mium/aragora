#!/usr/bin/env python3
"""
PostgreSQL Database Initialization Script

Initializes all PostgreSQL tables for Aragora production deployment.
Can be run standalone or imported as a module.

Usage:
    # Initialize database using environment variables
    python scripts/init_postgres_db.py

    # Initialize with custom DSN
    python scripts/init_postgres_db.py --dsn "postgresql://user:pass@host:5432/aragora"

    # Verify tables exist (dry run)
    python scripts/init_postgres_db.py --verify

Environment Variables:
    ARAGORA_POSTGRES_DSN or DATABASE_URL: PostgreSQL connection string

Example:
    export ARAGORA_POSTGRES_DSN="postgresql://aragora:password@localhost:5432/aragora"
    python scripts/init_postgres_db.py
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def init_all_stores(dsn: str | None = None) -> dict[str, bool]:
    """
    Initialize all PostgreSQL stores.

    Args:
        dsn: Optional PostgreSQL connection string. If not provided,
             uses ARAGORA_POSTGRES_DSN or DATABASE_URL from environment.

    Returns:
        Dictionary mapping store names to initialization status (True = success)
    """
    from aragora.storage.postgres_store import get_postgres_pool

    results: dict[str, bool] = {}

    try:
        pool = await get_postgres_pool(dsn=dsn)
        logger.info("Connected to PostgreSQL")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return {"_connection": False}

    # Import all PostgreSQL stores
    stores_to_init = []

    try:
        from aragora.storage.webhook_config_store import PostgresWebhookConfigStore

        stores_to_init.append(("webhook_configs", PostgresWebhookConfigStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresWebhookConfigStore: {e}")

    try:
        from aragora.storage.integration_store import PostgresIntegrationStore

        stores_to_init.append(("integrations", PostgresIntegrationStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresIntegrationStore: {e}")

    try:
        from aragora.storage.gmail_token_store import PostgresGmailTokenStore

        stores_to_init.append(("gmail_tokens", PostgresGmailTokenStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresGmailTokenStore: {e}")

    try:
        from aragora.storage.finding_workflow_store import PostgresFindingWorkflowStore

        stores_to_init.append(("finding_workflows", PostgresFindingWorkflowStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresFindingWorkflowStore: {e}")

    try:
        from aragora.storage.gauntlet_run_store import PostgresGauntletRunStore

        stores_to_init.append(("gauntlet_runs", PostgresGauntletRunStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresGauntletRunStore: {e}")

    try:
        from aragora.storage.job_queue_store import PostgresJobQueueStore

        stores_to_init.append(("job_queue", PostgresJobQueueStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresJobQueueStore: {e}")

    try:
        from aragora.storage.governance_store import PostgresGovernanceStore

        stores_to_init.append(("governance", PostgresGovernanceStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresGovernanceStore: {e}")

    try:
        from aragora.storage.marketplace_store import PostgresMarketplaceStore

        stores_to_init.append(("marketplace", PostgresMarketplaceStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresMarketplaceStore: {e}")

    try:
        from aragora.storage.federation_registry_store import PostgresFederationRegistryStore

        stores_to_init.append(("federation_registry", PostgresFederationRegistryStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresFederationRegistryStore: {e}")

    try:
        from aragora.storage.approval_request_store import PostgresApprovalRequestStore

        stores_to_init.append(("approval_requests", PostgresApprovalRequestStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresApprovalRequestStore: {e}")

    try:
        from aragora.storage.token_blacklist_store import PostgresBlacklist

        stores_to_init.append(("token_blacklist", PostgresBlacklist(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresBlacklist: {e}")

    try:
        from aragora.storage.user_store import PostgresUserStore

        stores_to_init.append(("users", PostgresUserStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresUserStore: {e}")

    try:
        from aragora.storage.webhook_store import PostgresWebhookStore

        stores_to_init.append(("webhooks", PostgresWebhookStore(pool)))
    except ImportError as e:
        logger.warning(f"Could not import PostgresWebhookStore: {e}")

    # Initialize each store
    for name, store in stores_to_init:
        try:
            await store.initialize()
            results[name] = True
            logger.info(f"Initialized {name}")
        except Exception as e:
            results[name] = False
            logger.error(f"Failed to initialize {name}: {e}")

    return results


async def verify_tables(dsn: str | None = None) -> dict[str, bool]:
    """
    Verify that all required tables exist.

    Args:
        dsn: Optional PostgreSQL connection string.

    Returns:
        Dictionary mapping table names to existence status
    """
    from aragora.storage.postgres_store import get_postgres_pool

    tables = [
        "webhook_configs",
        "integrations",
        "user_id_mappings",
        "gmail_tokens",
        "finding_workflows",
        "gauntlet_runs",
        "job_queue",
        "governance_approvals",
        "governance_verifications",
        "governance_decisions",
        "marketplace_templates",
        "federation_nodes",
        "approval_requests",
        "token_blacklist",
        "users",
        "organizations",
        "webhook_events",
    ]

    results: dict[str, bool] = {}

    try:
        pool = await get_postgres_pool(dsn=dsn)
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL: {e}")
        return {table: False for table in tables}

    async with pool.acquire() as conn:
        for table in tables:
            try:
                row = await conn.fetchrow(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = $1
                    )
                    """,
                    table,
                )
                exists = row[0] if row else False
                results[table] = exists
                status = "exists" if exists else "MISSING"
                logger.info(f"Table {table}: {status}")
            except Exception as e:
                results[table] = False
                logger.error(f"Error checking table {table}: {e}")

    return results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL database for Aragora"
    )
    parser.add_argument(
        "--dsn",
        help="PostgreSQL connection string (default: from environment)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify tables exist, don't create",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dsn = args.dsn or os.environ.get("ARAGORA_POSTGRES_DSN") or os.environ.get("DATABASE_URL")

    if not dsn:
        logger.error(
            "No PostgreSQL DSN provided. Set ARAGORA_POSTGRES_DSN or DATABASE_URL, "
            "or use --dsn argument."
        )
        return 1

    if args.verify:
        logger.info("Verifying PostgreSQL tables...")
        results = asyncio.run(verify_tables(dsn))
        missing = [table for table, exists in results.items() if not exists]
        if missing:
            logger.warning(f"Missing tables: {', '.join(missing)}")
            return 1
        logger.info("All tables verified!")
        return 0

    logger.info("Initializing PostgreSQL database...")
    results = asyncio.run(init_all_stores(dsn))

    # Check results
    failed = [name for name, success in results.items() if not success]
    if failed:
        logger.error(f"Failed to initialize: {', '.join(failed)}")
        return 1

    succeeded = len([s for s in results.values() if s])
    logger.info(f"Successfully initialized {succeeded} stores")
    return 0


if __name__ == "__main__":
    sys.exit(main())
