"""
Channel and Governance Storage Migration.

This migration adds persistent storage tables for:

1. integration_configs - Channel/integration configurations (Slack, Teams, etc.)
2. gmail_tokens - Gmail OAuth tokens and sync state
3. finding_workflows - Audit finding workflow state and assignments
4. federation_registry - Knowledge mound federation region configs

These tables support multi-instance deployments with SQLite and PostgreSQL.
"""

import logging

from aragora.migrations.runner import Migration
from aragora.storage.backends import DatabaseBackend, PostgreSQLBackend

logger = logging.getLogger(__name__)


def _table_exists(backend: DatabaseBackend, table: str) -> bool:
    """Check if a table exists."""
    try:
        if isinstance(backend, PostgreSQLBackend):
            rows = backend.fetch_all(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_name = %s
                """,
                (table,),
            )
        else:
            # SQLite
            rows = backend.fetch_all(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
        return len(rows) > 0
    except Exception:  # noqa: BLE001
        return False


def up_fn(backend: DatabaseBackend) -> None:
    """Apply the migration."""
    is_postgres = isinstance(backend, PostgreSQLBackend)

    # =========================================================================
    # 1. Integration Configs Table
    # =========================================================================
    if not _table_exists(backend, "integration_configs"):
        logger.info("Creating integration_configs table")
        if is_postgres:
            backend.execute_write(
                """
                CREATE TABLE integration_configs (
                    integration_type TEXT NOT NULL,
                    integration_id TEXT NOT NULL,
                    config_data JSONB NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (integration_type, integration_id)
                )
            """
            )
        else:
            backend.execute_write(
                """
                CREATE TABLE integration_configs (
                    integration_type TEXT NOT NULL,
                    integration_id TEXT NOT NULL,
                    config_data TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (integration_type, integration_id)
                )
            """
            )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_integrations_type ON integration_configs(integration_type)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_integrations_enabled ON integration_configs(enabled)"
        )

    # =========================================================================
    # 2. Gmail Tokens Table
    # =========================================================================
    if not _table_exists(backend, "gmail_tokens"):
        logger.info("Creating gmail_tokens table")
        if is_postgres:
            backend.execute_write(
                """
                CREATE TABLE gmail_tokens (
                    user_id TEXT PRIMARY KEY,
                    state_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """
            )
        else:
            backend.execute_write(
                """
                CREATE TABLE gmail_tokens (
                    user_id TEXT PRIMARY KEY,
                    state_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    # =========================================================================
    # 3. Gmail Sync Jobs Table
    # =========================================================================
    if not _table_exists(backend, "gmail_sync_jobs"):
        logger.info("Creating gmail_sync_jobs table")
        if is_postgres:
            backend.execute_write(
                """
                CREATE TABLE gmail_sync_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    state_data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """
            )
        else:
            backend.execute_write(
                """
                CREATE TABLE gmail_sync_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_gmail_jobs_user ON gmail_sync_jobs(user_id)"
        )

    # =========================================================================
    # 4. Finding Workflows Table
    # =========================================================================
    if not _table_exists(backend, "finding_workflows"):
        logger.info("Creating finding_workflows table")
        if is_postgres:
            backend.execute_write(
                """
                CREATE TABLE finding_workflows (
                    finding_id TEXT PRIMARY KEY,
                    workflow_data JSONB NOT NULL,
                    current_state TEXT NOT NULL DEFAULT 'open',
                    assigned_to TEXT,
                    priority INTEGER DEFAULT 3,
                    due_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """
            )
        else:
            backend.execute_write(
                """
                CREATE TABLE finding_workflows (
                    finding_id TEXT PRIMARY KEY,
                    workflow_data TEXT NOT NULL,
                    current_state TEXT NOT NULL DEFAULT 'open',
                    assigned_to TEXT,
                    priority INTEGER DEFAULT 3,
                    due_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_workflows_state ON finding_workflows(current_state)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_workflows_assignee ON finding_workflows(assigned_to)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_workflows_due ON finding_workflows(due_date)"
        )

    # =========================================================================
    # 5. Federation Registry Table
    # =========================================================================
    if not _table_exists(backend, "federation_registry"):
        logger.info("Creating federation_registry table")
        if is_postgres:
            backend.execute_write(
                """
                CREATE TABLE federation_registry (
                    region_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL DEFAULT '__global__',
                    config_data JSONB NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    last_sync_at TIMESTAMP,
                    last_push_at TIMESTAMP,
                    last_pull_at TIMESTAMP,
                    last_sync_error TEXT,
                    total_pushes INTEGER DEFAULT 0,
                    total_pulls INTEGER DEFAULT 0,
                    total_nodes_synced INTEGER DEFAULT 0,
                    total_sync_errors INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    PRIMARY KEY (region_id, workspace_id)
                )
            """
            )
        else:
            backend.execute_write(
                """
                CREATE TABLE federation_registry (
                    region_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL DEFAULT '__global__',
                    config_data TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    last_sync_at TIMESTAMP,
                    last_push_at TIMESTAMP,
                    last_pull_at TIMESTAMP,
                    last_sync_error TEXT,
                    total_pushes INTEGER DEFAULT 0,
                    total_pulls INTEGER DEFAULT 0,
                    total_nodes_synced INTEGER DEFAULT 0,
                    total_sync_errors INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (region_id, workspace_id)
                )
            """
            )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_federation_workspace ON federation_registry(workspace_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_federation_enabled ON federation_registry(enabled)"
        )

    logger.info("Channel and governance storage migration applied successfully")


def down_fn(backend: DatabaseBackend) -> None:
    """Rollback the migration."""
    # Drop indexes
    backend.execute_write("DROP INDEX IF EXISTS idx_federation_enabled")
    backend.execute_write("DROP INDEX IF EXISTS idx_federation_workspace")
    backend.execute_write("DROP INDEX IF EXISTS idx_workflows_due")
    backend.execute_write("DROP INDEX IF EXISTS idx_workflows_assignee")
    backend.execute_write("DROP INDEX IF EXISTS idx_workflows_state")
    backend.execute_write("DROP INDEX IF EXISTS idx_gmail_jobs_user")
    backend.execute_write("DROP INDEX IF EXISTS idx_integrations_enabled")
    backend.execute_write("DROP INDEX IF EXISTS idx_integrations_type")

    # Drop tables
    backend.execute_write("DROP TABLE IF EXISTS federation_registry")
    backend.execute_write("DROP TABLE IF EXISTS finding_workflows")
    backend.execute_write("DROP TABLE IF EXISTS gmail_sync_jobs")
    backend.execute_write("DROP TABLE IF EXISTS gmail_tokens")
    backend.execute_write("DROP TABLE IF EXISTS integration_configs")

    logger.info("Channel and governance storage migration rolled back")


migration = Migration(
    version=20260120000000,
    name="Channel and governance persistent stores",
    up_fn=up_fn,
    down_fn=down_fn,
)
