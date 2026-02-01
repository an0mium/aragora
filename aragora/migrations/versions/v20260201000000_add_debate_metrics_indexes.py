"""
Add performance indexes for debate metrics queries.

Migration created: 2026-02-01

This migration adds optimized indexes for common debate query patterns:
- Debate results by status and completion time
- Agent performance lookups
- Consensus tracking queries

These indexes support the performance monitoring and analytics dashboards.

Zero-Downtime Strategy:
- All indexes created with CONCURRENTLY option on PostgreSQL
- No table locks during index creation
- Safe to run during production traffic
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
    """Create performance indexes for debate queries."""
    logger.info("Creating performance indexes for debate metrics")

    # Index for gauntlet_results queries by status and time
    if _table_exists(backend, "gauntlet_results"):
        safe_create_index(
            backend,
            "idx_gauntlet_results_verdict_created",
            "gauntlet_results",
            ["verdict", "created_at"],
            concurrently=True,
        )
        # Index for confidence score queries
        safe_create_index(
            backend,
            "idx_gauntlet_results_confidence",
            "gauntlet_results",
            ["confidence"],
            concurrently=True,
        )
        # Index for robustness score analysis
        safe_create_index(
            backend,
            "idx_gauntlet_results_robustness",
            "gauntlet_results",
            ["robustness_score"],
            concurrently=True,
        )
        logger.info("Created indexes on gauntlet_results")

    # Index for job_queue performance
    if _table_exists(backend, "job_queue"):
        # Composite index for job scheduling queries
        safe_create_index(
            backend,
            "idx_job_queue_pending_priority",
            "job_queue",
            ["status", "priority", "scheduled_at"],
            concurrently=True,
        )
        logger.info("Created indexes on job_queue")

    # Index for audit_log queries
    if _table_exists(backend, "audit_log"):
        # Index for resource-specific audit lookups
        safe_create_index(
            backend,
            "idx_audit_log_resource_time",
            "audit_log",
            ["resource_type", "resource_id", "timestamp"],
            concurrently=True,
        )
        logger.info("Created indexes on audit_log")

    logger.info("Migration 20260201000000 applied successfully")


def down_fn(backend: DatabaseBackend) -> None:
    """Remove performance indexes."""
    logger.info("Removing performance indexes for debate metrics")

    # Drop indexes in reverse order
    safe_drop_index(backend, "idx_audit_log_resource_time", concurrently=True)
    safe_drop_index(backend, "idx_job_queue_pending_priority", concurrently=True)
    safe_drop_index(backend, "idx_gauntlet_results_robustness", concurrently=True)
    safe_drop_index(backend, "idx_gauntlet_results_confidence", concurrently=True)
    safe_drop_index(backend, "idx_gauntlet_results_verdict_created", concurrently=True)

    logger.info("Migration 20260201000000 rolled back successfully")


migration = Migration(
    version=20260201000000,
    name="Add debate metrics performance indexes",
    up_fn=up_fn,
    down_fn=down_fn,
)
