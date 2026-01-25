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
    """Reverse this migration - remove lockout fields from users table.

    SQLite doesn't support DROP COLUMN directly, so we use table recreation:
    1. Create backup table with current data
    2. Drop original table
    3. Recreate table with original schema (without new columns)
    4. Copy data back from backup (only original columns)
    5. Drop backup table
    6. Recreate indexes
    """
    conn.executescript(
        """
        -- Step 1: Create backup table with all current data
        CREATE TABLE IF NOT EXISTS users_backup AS SELECT * FROM users;

        -- Step 2: Drop original users table
        DROP TABLE IF EXISTS users;

        -- Step 3: Recreate users table with original schema (without lockout columns)
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            name TEXT DEFAULT '',
            org_id TEXT,
            role TEXT DEFAULT 'member',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login_at TEXT,
            is_active INTEGER DEFAULT 1,
            email_verified INTEGER DEFAULT 0,
            avatar_url TEXT,
            preferences TEXT DEFAULT '{}'
        );

        -- Step 4: Copy data back from backup (only original columns)
        INSERT INTO users (
            id, email, password_hash, password_salt, name, org_id, role,
            created_at, updated_at, last_login_at, is_active, email_verified,
            avatar_url, preferences
        )
        SELECT
            id, email, password_hash, password_salt, name, org_id, role,
            created_at, updated_at, last_login_at, is_active, email_verified,
            avatar_url, preferences
        FROM users_backup;

        -- Step 5: Drop backup table
        DROP TABLE IF EXISTS users_backup;

        -- Step 6: Recreate indexes
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id);
    """
    )
