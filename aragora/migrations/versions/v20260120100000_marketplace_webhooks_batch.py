"""
Marketplace, Webhooks, and Batch Explainability Migration.

This migration adds persistent storage tables for:

1. marketplace_templates - Workflow template storage with versioning
2. marketplace_reviews - Template reviews and ratings
3. webhook_registrations - Webhook endpoint configurations
4. webhook_delivery_receipts - Delivery tracking and audit trail
5. batch_explainability_jobs - Batch job tracking and status
6. batch_explainability_results - Individual debate explanation results

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
    # 1. Marketplace Templates Table
    # =========================================================================
    if not _table_exists(backend, "marketplace_templates"):
        logger.info("Creating marketplace_templates table")
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE marketplace_templates (
                    template_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL DEFAULT 'workflow',
                    visibility TEXT NOT NULL DEFAULT 'public',
                    author_id TEXT NOT NULL,
                    version TEXT NOT NULL DEFAULT '1.0.0',
                    template_data JSONB NOT NULL,
                    tags TEXT[],
                    downloads INTEGER DEFAULT 0,
                    avg_rating REAL DEFAULT 0.0,
                    rating_count INTEGER DEFAULT 0,
                    featured BOOLEAN DEFAULT FALSE,
                    deprecated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE marketplace_templates (
                    template_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT NOT NULL DEFAULT 'workflow',
                    visibility TEXT NOT NULL DEFAULT 'public',
                    author_id TEXT NOT NULL,
                    version TEXT NOT NULL DEFAULT '1.0.0',
                    template_data TEXT NOT NULL,
                    tags TEXT,
                    downloads INTEGER DEFAULT 0,
                    avg_rating REAL DEFAULT 0.0,
                    rating_count INTEGER DEFAULT 0,
                    featured INTEGER DEFAULT 0,
                    deprecated INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_templates_category ON marketplace_templates(category)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_templates_visibility ON marketplace_templates(visibility)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_templates_author ON marketplace_templates(author_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_templates_featured ON marketplace_templates(featured)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_templates_avg_rating ON marketplace_templates(avg_rating DESC)"
        )

    # =========================================================================
    # 2. Marketplace Reviews Table
    # =========================================================================
    if not _table_exists(backend, "marketplace_reviews"):
        logger.info("Creating marketplace_reviews table")
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE marketplace_reviews (
                    review_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    title TEXT,
                    content TEXT,
                    helpful_votes INTEGER DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'approved',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(template_id, user_id)
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE marketplace_reviews (
                    review_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    title TEXT,
                    content TEXT,
                    helpful_votes INTEGER DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'approved',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(template_id, user_id)
                )
            """)
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_reviews_template ON marketplace_reviews(template_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_reviews_user ON marketplace_reviews(user_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_reviews_rating ON marketplace_reviews(rating)"
        )

    # =========================================================================
    # 3. Webhook Registrations Table
    # =========================================================================
    if not _table_exists(backend, "webhook_registrations"):
        logger.info("Creating webhook_registrations table")
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE webhook_registrations (
                    webhook_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    secret TEXT,
                    event_types TEXT[] NOT NULL,
                    enabled BOOLEAN DEFAULT TRUE,
                    description TEXT,
                    metadata JSONB,
                    failure_count INTEGER DEFAULT 0,
                    last_triggered_at TIMESTAMP,
                    last_success_at TIMESTAMP,
                    last_failure_at TIMESTAMP,
                    last_failure_reason TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE webhook_registrations (
                    webhook_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    secret TEXT,
                    event_types TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    description TEXT,
                    metadata TEXT,
                    failure_count INTEGER DEFAULT 0,
                    last_triggered_at TIMESTAMP,
                    last_success_at TIMESTAMP,
                    last_failure_at TIMESTAMP,
                    last_failure_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_webhooks_owner ON webhook_registrations(owner_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_webhooks_enabled ON webhook_registrations(enabled)"
        )

    # =========================================================================
    # 4. Webhook Delivery Receipts Table
    # =========================================================================
    if not _table_exists(backend, "webhook_delivery_receipts"):
        logger.info("Creating webhook_delivery_receipts table")
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE webhook_delivery_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    webhook_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    delivery_status TEXT NOT NULL DEFAULT 'pending',
                    http_status INTEGER,
                    response_body TEXT,
                    attempt_count INTEGER DEFAULT 0,
                    latency_ms REAL,
                    error_message TEXT,
                    delivered_at TIMESTAMP,
                    next_retry_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE webhook_delivery_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    webhook_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    delivery_status TEXT NOT NULL DEFAULT 'pending',
                    http_status INTEGER,
                    response_body TEXT,
                    attempt_count INTEGER DEFAULT 0,
                    latency_ms REAL,
                    error_message TEXT,
                    delivered_at TIMESTAMP,
                    next_retry_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_receipts_webhook ON webhook_delivery_receipts(webhook_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_receipts_status ON webhook_delivery_receipts(delivery_status)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_receipts_event ON webhook_delivery_receipts(event_type, event_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_receipts_retry ON webhook_delivery_receipts(next_retry_at) WHERE delivery_status = 'pending'"
        )

    # =========================================================================
    # 5. Batch Explainability Jobs Table
    # =========================================================================
    if not _table_exists(backend, "batch_explainability_jobs"):
        logger.info("Creating batch_explainability_jobs table")
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE batch_explainability_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    total_debates INTEGER NOT NULL DEFAULT 0,
                    processed_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    options JSONB,
                    error_message TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE batch_explainability_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    total_debates INTEGER NOT NULL DEFAULT 0,
                    processed_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    options TEXT,
                    error_message TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_batch_jobs_user ON batch_explainability_jobs(user_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_explainability_jobs(status)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_batch_jobs_created ON batch_explainability_jobs(created_at DESC)"
        )

    # =========================================================================
    # 6. Batch Explainability Results Table
    # =========================================================================
    if not _table_exists(backend, "batch_explainability_results"):
        logger.info("Creating batch_explainability_results table")
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE batch_explainability_results (
                    result_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    debate_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    explanation JSONB,
                    processing_time_ms REAL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(job_id, debate_id)
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE batch_explainability_results (
                    result_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    debate_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    explanation TEXT,
                    processing_time_ms REAL,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(job_id, debate_id)
                )
            """)
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_results_job ON batch_explainability_results(job_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_results_debate ON batch_explainability_results(debate_id)"
        )
        backend.execute_write(
            "CREATE INDEX IF NOT EXISTS idx_results_status ON batch_explainability_results(status)"
        )

    logger.info("Marketplace, webhooks, and batch explainability migration applied successfully")


def down_fn(backend: DatabaseBackend) -> None:
    """Rollback the migration."""
    # Drop indexes
    backend.execute_write("DROP INDEX IF EXISTS idx_results_status")
    backend.execute_write("DROP INDEX IF EXISTS idx_results_debate")
    backend.execute_write("DROP INDEX IF EXISTS idx_results_job")
    backend.execute_write("DROP INDEX IF EXISTS idx_batch_jobs_created")
    backend.execute_write("DROP INDEX IF EXISTS idx_batch_jobs_status")
    backend.execute_write("DROP INDEX IF EXISTS idx_batch_jobs_user")
    backend.execute_write("DROP INDEX IF EXISTS idx_receipts_retry")
    backend.execute_write("DROP INDEX IF EXISTS idx_receipts_event")
    backend.execute_write("DROP INDEX IF EXISTS idx_receipts_status")
    backend.execute_write("DROP INDEX IF EXISTS idx_receipts_webhook")
    backend.execute_write("DROP INDEX IF EXISTS idx_webhooks_enabled")
    backend.execute_write("DROP INDEX IF EXISTS idx_webhooks_owner")
    backend.execute_write("DROP INDEX IF EXISTS idx_reviews_rating")
    backend.execute_write("DROP INDEX IF EXISTS idx_reviews_user")
    backend.execute_write("DROP INDEX IF EXISTS idx_reviews_template")
    backend.execute_write("DROP INDEX IF EXISTS idx_templates_avg_rating")
    backend.execute_write("DROP INDEX IF EXISTS idx_templates_featured")
    backend.execute_write("DROP INDEX IF EXISTS idx_templates_author")
    backend.execute_write("DROP INDEX IF EXISTS idx_templates_visibility")
    backend.execute_write("DROP INDEX IF EXISTS idx_templates_category")

    # Drop tables
    backend.execute_write("DROP TABLE IF EXISTS batch_explainability_results")
    backend.execute_write("DROP TABLE IF EXISTS batch_explainability_jobs")
    backend.execute_write("DROP TABLE IF EXISTS webhook_delivery_receipts")
    backend.execute_write("DROP TABLE IF EXISTS webhook_registrations")
    backend.execute_write("DROP TABLE IF EXISTS marketplace_reviews")
    backend.execute_write("DROP TABLE IF EXISTS marketplace_templates")

    logger.info("Marketplace, webhooks, and batch explainability migration rolled back")


migration = Migration(
    version=20260120100000,
    name="Marketplace, webhooks, and batch explainability storage",
    up_fn=up_fn,
    down_fn=down_fn,
)
