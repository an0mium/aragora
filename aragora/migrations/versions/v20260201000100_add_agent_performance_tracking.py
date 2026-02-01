"""
Add agent performance tracking table.

Migration created: 2026-02-01

This migration creates the agent_performance table for tracking:
- Individual agent response times
- Success/failure rates
- Token usage statistics
- ELO rating snapshots

The table supports the PerformanceMonitor and agent selection algorithms.
"""

import logging

from aragora.migrations.runner import Migration
from aragora.migrations.patterns import safe_create_index, safe_drop_index
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
            rows = backend.fetch_all(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
        return len(rows) > 0
    except Exception:  # noqa: BLE001
        return False


def up_fn(backend: DatabaseBackend) -> None:
    """Create agent_performance table."""
    logger.info("Creating agent_performance table")

    is_postgres = isinstance(backend, PostgreSQLBackend)

    if not _table_exists(backend, "agent_performance"):
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE agent_performance (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    debate_id TEXT,
                    operation TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    duration_ms REAL,
                    success BOOLEAN DEFAULT TRUE,
                    error_type TEXT,
                    error_message TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd REAL,
                    elo_rating REAL,
                    model_version TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT NOW(),
                    workspace_id TEXT,
                    org_id TEXT
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE agent_performance (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    agent_type TEXT NOT NULL,
                    debate_id TEXT,
                    operation TEXT NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    duration_ms REAL,
                    success INTEGER DEFAULT 1,
                    error_type TEXT,
                    error_message TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd REAL,
                    elo_rating REAL,
                    model_version TEXT,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    workspace_id TEXT,
                    org_id TEXT
                )
            """)
        logger.info("Created agent_performance table")

        # Create indexes for common query patterns
        safe_create_index(
            backend,
            "idx_agent_perf_agent_time",
            "agent_performance",
            ["agent_id", "created_at"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_agent_perf_type",
            "agent_performance",
            ["agent_type"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_agent_perf_debate",
            "agent_performance",
            ["debate_id"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_agent_perf_operation",
            "agent_performance",
            ["operation", "success"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_agent_perf_workspace",
            "agent_performance",
            ["workspace_id", "created_at"],
            concurrently=True,
        )
        logger.info("Created indexes on agent_performance")

    logger.info("Migration 20260201000100 applied successfully")


def down_fn(backend: DatabaseBackend) -> None:
    """Drop agent_performance table."""
    logger.info("Dropping agent_performance table")

    # Drop indexes first
    safe_drop_index(backend, "idx_agent_perf_workspace", concurrently=True)
    safe_drop_index(backend, "idx_agent_perf_operation", concurrently=True)
    safe_drop_index(backend, "idx_agent_perf_debate", concurrently=True)
    safe_drop_index(backend, "idx_agent_perf_type", concurrently=True)
    safe_drop_index(backend, "idx_agent_perf_agent_time", concurrently=True)

    # Drop table
    backend.execute_write("DROP TABLE IF EXISTS agent_performance")

    logger.info("Migration 20260201000100 rolled back successfully")


migration = Migration(
    version=20260201000100,
    name="Add agent performance tracking table",
    up_fn=up_fn,
    down_fn=down_fn,
)
