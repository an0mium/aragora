"""
Add session management tables.

Migration created: 2026-02-01

This migration creates infrastructure for user session management:
- user_sessions: Active session tracking
- session_events: Session activity logging for security

Supports the authentication system and security monitoring.
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
    """Create session management tables."""
    logger.info("Creating session management tables")

    is_postgres = isinstance(backend, PostgreSQLBackend)

    # User sessions table
    if not _table_exists(backend, "user_sessions"):
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMP NOT NULL,
                    last_activity_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    ip_address TEXT,
                    user_agent TEXT,
                    device_fingerprint TEXT,
                    auth_method TEXT NOT NULL DEFAULT 'password',
                    mfa_verified BOOLEAN DEFAULT FALSE,
                    is_active BOOLEAN DEFAULT TRUE,
                    revoked_at TIMESTAMP,
                    revoked_reason TEXT,
                    metadata JSONB DEFAULT '{}',
                    workspace_id TEXT,
                    org_id TEXT
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    last_activity_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    device_fingerprint TEXT,
                    auth_method TEXT NOT NULL DEFAULT 'password',
                    mfa_verified INTEGER DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    revoked_at TIMESTAMP,
                    revoked_reason TEXT,
                    metadata TEXT DEFAULT '{}',
                    workspace_id TEXT,
                    org_id TEXT
                )
            """)
        logger.info("Created user_sessions table")

        # Indexes for session management
        safe_create_index(
            backend,
            "idx_sessions_user",
            "user_sessions",
            ["user_id", "is_active"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_sessions_expires",
            "user_sessions",
            ["expires_at"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_sessions_activity",
            "user_sessions",
            ["last_activity_at"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_sessions_ip",
            "user_sessions",
            ["ip_address"],
            concurrently=True,
        )

    # Session events table (for security audit)
    if not _table_exists(backend, "session_events"):
        if is_postgres:
            backend.execute_write("""
                CREATE TABLE session_events (
                    id BIGSERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_time TIMESTAMP NOT NULL DEFAULT NOW(),
                    ip_address TEXT,
                    user_agent TEXT,
                    resource_type TEXT,
                    resource_id TEXT,
                    success BOOLEAN DEFAULT TRUE,
                    failure_reason TEXT,
                    metadata JSONB DEFAULT '{}'
                )
            """)
        else:
            backend.execute_write("""
                CREATE TABLE session_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT,
                    resource_type TEXT,
                    resource_id TEXT,
                    success INTEGER DEFAULT 1,
                    failure_reason TEXT,
                    metadata TEXT DEFAULT '{}'
                )
            """)
        logger.info("Created session_events table")

        # Indexes for event queries
        safe_create_index(
            backend,
            "idx_session_events_session",
            "session_events",
            ["session_id", "event_time"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_session_events_user",
            "session_events",
            ["user_id", "event_time"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_session_events_type",
            "session_events",
            ["event_type", "event_time"],
            concurrently=True,
        )
        safe_create_index(
            backend,
            "idx_session_events_time",
            "session_events",
            ["event_time"],
            concurrently=True,
        )

    logger.info("Migration 20260201000400 applied successfully")


def down_fn(backend: DatabaseBackend) -> None:
    """Drop session management tables."""
    logger.info("Dropping session management tables")

    # Drop event indexes
    safe_drop_index(backend, "idx_session_events_time", concurrently=True)
    safe_drop_index(backend, "idx_session_events_type", concurrently=True)
    safe_drop_index(backend, "idx_session_events_user", concurrently=True)
    safe_drop_index(backend, "idx_session_events_session", concurrently=True)

    # Drop session indexes
    safe_drop_index(backend, "idx_sessions_ip", concurrently=True)
    safe_drop_index(backend, "idx_sessions_activity", concurrently=True)
    safe_drop_index(backend, "idx_sessions_expires", concurrently=True)
    safe_drop_index(backend, "idx_sessions_user", concurrently=True)

    # Drop tables
    backend.execute_write("DROP TABLE IF EXISTS session_events")
    backend.execute_write("DROP TABLE IF EXISTS user_sessions")

    logger.info("Migration 20260201000400 rolled back successfully")


migration = Migration(
    version=20260201000400,
    name="Add session management tables",
    up_fn=up_fn,
    down_fn=down_fn,
)
