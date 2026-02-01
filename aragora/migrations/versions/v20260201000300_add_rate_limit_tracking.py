"""
Add rate limit tracking table and indexes.

Migration created: 2026-02-01

This migration creates infrastructure for distributed rate limiting:
- rate_limit_entries: Track rate limit state per key
- rate_limit_violations: Log rate limit violations for analysis

Supports the Redis-based rate limiter with fallback to database.
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
    """Create rate limiting tables."""
    logger.info("Creating rate limiting tables")

    is_postgres = isinstance(backend, PostgreSQLBackend)

    # Rate limit entries table
    if not _table_exists(backend, "rate_limit_entries"):
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE rate_limit_entries (
                    key TEXT PRIMARY KEY,
                    bucket_type TEXT NOT NULL,
                    tokens_remaining REAL NOT NULL,
                    last_refill_at TIMESTAMP NOT NULL,
                    window_start_at TIMESTAMP,
                    request_count INTEGER DEFAULT 0,
                    expires_at TIMESTAMP NOT NULL,
                    metadata JSONB DEFAULT '{}'
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE rate_limit_entries (
                    key TEXT PRIMARY KEY,
                    bucket_type TEXT NOT NULL,
                    tokens_remaining REAL NOT NULL,
                    last_refill_at TIMESTAMP NOT NULL,
                    window_start_at TIMESTAMP,
                    request_count INTEGER DEFAULT 0,
                    expires_at TIMESTAMP NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
        logger.info("Created rate_limit_entries table")

        # Index for cleanup of expired entries
        safe_create_index(
            backend,
            "idx_rate_limit_expires",
            "rate_limit_entries",
            ["expires_at"],
            concurrently=True,
        )
        # Index for bucket type queries
        safe_create_index(
            backend,
            "idx_rate_limit_bucket",
            "rate_limit_entries",
            ["bucket_type"],
            concurrently=True,
        )

    # Rate limit violations table (for analytics)
    if not _table_exists(backend, "rate_limit_violations"):
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE rate_limit_violations (
                    id BIGSERIAL PRIMARY KEY,
                    key TEXT NOT NULL,
                    bucket_type TEXT NOT NULL,
                    violated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    client_ip TEXT,
                    user_id TEXT,
                    api_key_id TEXT,
                    endpoint TEXT,
                    request_count INTEGER,
                    limit_value INTEGER,
                    retry_after_seconds INTEGER,
                    metadata JSONB DEFAULT '{}'
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE rate_limit_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key TEXT NOT NULL,
                    bucket_type TEXT NOT NULL,
                    violated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    client_ip TEXT,
                    user_id TEXT,
                    api_key_id TEXT,
                    endpoint TEXT,
                    request_count INTEGER,
                    limit_value INTEGER,
                    retry_after_seconds INTEGER,
                    metadata TEXT DEFAULT '{}'
                )
            """)
        logger.info("Created rate_limit_violations table")

        # Indexes for violation analysis
        safe_create_index(
            backend,
            "idx_violations_time",
            "rate_limit_violations",
            ["violated_at"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_violations_key",
            "rate_limit_violations",
            ["key", "violated_at"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_violations_user",
            "rate_limit_violations",
            ["user_id"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_violations_ip",
            "rate_limit_violations",
            ["client_ip"],
            concurrently=True,
        )

    logger.info("Migration 20260201000300 applied successfully")


def down_fn(backend: DatabaseBackend) -> None:
    """Drop rate limiting tables."""
    logger.info("Dropping rate limiting tables")

    # Drop violation indexes
    safe_drop_index(backend, "idx_violations_ip", concurrently=True)
    safe_drop_index(backend, "idx_violations_user", concurrently=True)
    safe_drop_index(backend, "idx_violations_key", concurrently=True)
    safe_drop_index(backend, "idx_violations_time", concurrently=True)

    # Drop entries indexes
    safe_drop_index(backend, "idx_rate_limit_bucket", concurrently=True)
    safe_drop_index(backend, "idx_rate_limit_expires", concurrently=True)

    # Drop tables
    backend.execute_write("DROP TABLE IF EXISTS rate_limit_violations")
    backend.execute_write("DROP TABLE IF EXISTS rate_limit_entries")

    logger.info("Migration 20260201000300 rolled back successfully")


migration = Migration(
    version=20260201000300,
    name="Add rate limit tracking tables",
    up_fn=up_fn,
    down_fn=down_fn,
)
