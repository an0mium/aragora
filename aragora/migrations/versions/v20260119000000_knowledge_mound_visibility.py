"""
Knowledge Mound Visibility and Access Grants Migration.

This migration adds visibility control and access grant support to the
Knowledge Mound system:

1. Adds visibility columns to knowledge_nodes table
2. Creates access_grants table for fine-grained permissions
3. Creates federated_regions table for multi-region sync
4. Adds indexes for efficient querying
"""

import logging

from aragora.migrations.runner import Migration
from aragora.storage.backends import DatabaseBackend, PostgreSQLBackend

logger = logging.getLogger(__name__)


def _column_exists(backend: DatabaseBackend, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    try:
        if isinstance(backend, PostgreSQLBackend):
            rows = backend.fetch_all(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND column_name = %s
                """,
                (table, column),
            )
        else:
            # SQLite
            rows = backend.fetch_all(f"PRAGMA table_info({table})")
            return any(row[1] == column for row in rows)
        return len(rows) > 0
    except Exception:
        return False


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
    except Exception:
        return False


def up_fn(backend: DatabaseBackend) -> None:
    """Apply the migration using Python for safer column additions."""
    is_postgres = isinstance(backend, PostgreSQLBackend)

    # Add visibility columns to knowledge_nodes if table exists
    if _table_exists(backend, "knowledge_nodes"):
        if not _column_exists(backend, "knowledge_nodes", "visibility"):
            logger.info("Adding visibility column to knowledge_nodes")
            backend.execute_write(
                "ALTER TABLE knowledge_nodes ADD COLUMN visibility TEXT DEFAULT 'workspace'"
            )

        if not _column_exists(backend, "knowledge_nodes", "visibility_set_by"):
            logger.info("Adding visibility_set_by column to knowledge_nodes")
            backend.execute_write("ALTER TABLE knowledge_nodes ADD COLUMN visibility_set_by TEXT")

        if not _column_exists(backend, "knowledge_nodes", "is_discoverable"):
            logger.info("Adding is_discoverable column to knowledge_nodes")
            if is_postgres:
                backend.execute_write(
                    "ALTER TABLE knowledge_nodes ADD COLUMN is_discoverable BOOLEAN DEFAULT TRUE"
                )
            else:
                backend.execute_write(
                    "ALTER TABLE knowledge_nodes ADD COLUMN is_discoverable INTEGER DEFAULT 1"
                )

    # Create access_grants table
    if is_postgres:
        backend.execute_write(
            """
            CREATE TABLE IF NOT EXISTS access_grants (
                id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                grantee_type TEXT NOT NULL CHECK (grantee_type IN ('user', 'role', 'workspace', 'organization')),
                grantee_id TEXT NOT NULL,
                permissions TEXT[] DEFAULT '{read}',
                granted_by TEXT,
                granted_at TIMESTAMP DEFAULT NOW(),
                expires_at TIMESTAMP,
                workspace_id TEXT,
                UNIQUE(item_id, grantee_type, grantee_id)
            )
        """
        )
    else:
        backend.execute_write(
            """
            CREATE TABLE IF NOT EXISTS access_grants (
                id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                grantee_type TEXT NOT NULL CHECK (grantee_type IN ('user', 'role', 'workspace', 'organization')),
                grantee_id TEXT NOT NULL,
                permissions TEXT DEFAULT '["read"]',
                granted_by TEXT,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                workspace_id TEXT,
                UNIQUE(item_id, grantee_type, grantee_id)
            )
        """
        )

    # Create indexes for access_grants
    backend.execute_write("CREATE INDEX IF NOT EXISTS idx_grants_item_id ON access_grants(item_id)")
    backend.execute_write(
        "CREATE INDEX IF NOT EXISTS idx_grants_grantee ON access_grants(grantee_type, grantee_id)"
    )
    backend.execute_write(
        "CREATE INDEX IF NOT EXISTS idx_grants_workspace ON access_grants(workspace_id)"
    )
    backend.execute_write(
        "CREATE INDEX IF NOT EXISTS idx_grants_expires ON access_grants(expires_at)"
    )

    # Create federated_regions table
    if is_postgres:
        backend.execute_write(
            """
            CREATE TABLE IF NOT EXISTS federated_regions (
                region_id TEXT PRIMARY KEY,
                endpoint_url TEXT NOT NULL,
                api_key_hash TEXT NOT NULL,
                mode TEXT DEFAULT 'bidirectional' CHECK (mode IN ('push', 'pull', 'bidirectional', 'none')),
                sync_scope TEXT DEFAULT 'summary' CHECK (sync_scope IN ('full', 'metadata', 'summary')),
                enabled BOOLEAN DEFAULT TRUE,
                last_sync_at TIMESTAMP,
                last_sync_error TEXT,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """
        )
    else:
        backend.execute_write(
            """
            CREATE TABLE IF NOT EXISTS federated_regions (
                region_id TEXT PRIMARY KEY,
                endpoint_url TEXT NOT NULL,
                api_key_hash TEXT NOT NULL,
                mode TEXT DEFAULT 'bidirectional' CHECK (mode IN ('push', 'pull', 'bidirectional', 'none')),
                sync_scope TEXT DEFAULT 'summary' CHECK (sync_scope IN ('full', 'metadata', 'summary')),
                enabled INTEGER DEFAULT 1,
                last_sync_at TIMESTAMP,
                last_sync_error TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

    # Create index for federation status queries
    backend.execute_write(
        "CREATE INDEX IF NOT EXISTS idx_federation_enabled ON federated_regions(enabled)"
    )

    # Create visibility indexes on knowledge_nodes if table exists
    if _table_exists(backend, "knowledge_nodes"):
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_nodes_visibility ON knowledge_nodes(visibility)"
        )
        if is_postgres:
            # PostgreSQL supports partial indexes
            try:
                backend.execute_write(
                    "CREATE INDEX IF NOT EXISTS idx_nodes_discoverable ON knowledge_nodes(is_discoverable) WHERE is_discoverable = TRUE"
                )
            except Exception:
                pass  # Index may already exist
        else:
            # SQLite partial index syntax
            try:
                backend.execute_write(
                    "CREATE INDEX IF NOT EXISTS idx_nodes_discoverable ON knowledge_nodes(is_discoverable) WHERE is_discoverable = 1"
                )
            except Exception:
                pass  # Index may already exist

    logger.info("Knowledge Mound visibility migration applied successfully")


def down_fn(backend: DatabaseBackend) -> None:
    """Rollback the migration."""
    # Drop indexes
    backend.execute_write("DROP INDEX IF EXISTS idx_nodes_discoverable")
    backend.execute_write("DROP INDEX IF EXISTS idx_nodes_visibility")
    backend.execute_write("DROP INDEX IF EXISTS idx_federation_enabled")
    backend.execute_write("DROP INDEX IF EXISTS idx_grants_expires")
    backend.execute_write("DROP INDEX IF EXISTS idx_grants_workspace")
    backend.execute_write("DROP INDEX IF EXISTS idx_grants_grantee")
    backend.execute_write("DROP INDEX IF EXISTS idx_grants_item_id")

    # Drop tables
    backend.execute_write("DROP TABLE IF EXISTS federated_regions")
    backend.execute_write("DROP TABLE IF EXISTS access_grants")

    logger.info("Knowledge Mound visibility migration rolled back")


migration = Migration(
    version=20260119000000,
    name="Knowledge Mound visibility and access grants",
    up_fn=up_fn,
    down_fn=down_fn,
)
