"""
Migration 002: Add lockout fields

Adds fields for account lockout support:
- locked_until: Timestamp when lockout expires
- failed_login_count: Track failed login attempts
- lockout_reason: Why the account was locked

Created: 2024-01-02
"""

import sqlite3

from aragora.storage.schema import safe_add_column


def upgrade(conn: sqlite3.Connection) -> None:
    """Apply this migration - add lockout fields to users table."""
    # Add lockout fields using safe_add_column to handle existing columns
    safe_add_column(conn, "users", "locked_until", "TEXT", "NULL")
    safe_add_column(conn, "users", "failed_login_count", "INTEGER", "0")
    safe_add_column(conn, "users", "lockout_reason", "TEXT", "NULL")

    # Add last_activity tracking for churn prevention
    safe_add_column(conn, "users", "last_activity_at", "TEXT", "NULL")
    safe_add_column(conn, "users", "last_debate_at", "TEXT", "NULL")


def downgrade(conn: sqlite3.Connection) -> None:
    """Reverse this migration (for development only)."""
    # SQLite doesn't support DROP COLUMN directly
    # Would need to recreate the table without these columns
    # Not recommended in production
    pass
