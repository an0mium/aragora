"""
Migration 001: Initial schema

Creates the base tables for user management:
- users: User accounts
- organizations: Multi-tenant organizations
- usage_events: Usage tracking
- oauth_providers: Social auth providers
- audit_log: Security audit trail
- org_invitations: Team invitation system

Created: 2024-01-01
"""

import sqlite3


def upgrade(conn: sqlite3.Connection) -> None:
    """Apply this migration - create initial schema."""
    conn.executescript(
        """
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
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

        -- Organizations table
        CREATE TABLE IF NOT EXISTS organizations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            tier TEXT DEFAULT 'free',
            owner_id TEXT,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            settings TEXT DEFAULT '{}',
            FOREIGN KEY (owner_id) REFERENCES users(id)
        );

        -- Usage events table
        CREATE TABLE IF NOT EXISTS usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            org_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            metadata TEXT DEFAULT '{}',
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (org_id) REFERENCES organizations(id)
        );

        -- OAuth providers table
        CREATE TABLE IF NOT EXISTS oauth_providers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            provider_user_id TEXT NOT NULL,
            email TEXT,
            access_token TEXT,
            refresh_token TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(provider, provider_user_id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- Audit log table
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_id TEXT,
            org_id TEXT,
            action TEXT NOT NULL,
            resource_type TEXT,
            resource_id TEXT,
            details TEXT DEFAULT '{}',
            ip_address TEXT,
            user_agent TEXT
        );

        -- Organization invitations table
        CREATE TABLE IF NOT EXISTS org_invitations (
            id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT DEFAULT 'member',
            token TEXT UNIQUE NOT NULL,
            invited_by TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            expires_at TEXT,
            accepted_at TEXT,
            FOREIGN KEY (org_id) REFERENCES organizations(id),
            FOREIGN KEY (invited_by) REFERENCES users(id)
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_org ON users(org_id);
        CREATE INDEX IF NOT EXISTS idx_orgs_slug ON organizations(slug);
        CREATE INDEX IF NOT EXISTS idx_orgs_stripe ON organizations(stripe_customer_id);
        CREATE INDEX IF NOT EXISTS idx_usage_org ON usage_events(org_id);
        CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_org ON audit_log(org_id);
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_invitations_token ON org_invitations(token);
        CREATE INDEX IF NOT EXISTS idx_invitations_email ON org_invitations(email);
    """
    )


def downgrade(conn: sqlite3.Connection) -> None:
    """Reverse this migration (for development only)."""
    # WARNING: This will delete all data!
    conn.executescript(
        """
        DROP TABLE IF EXISTS org_invitations;
        DROP TABLE IF EXISTS audit_log;
        DROP TABLE IF EXISTS oauth_providers;
        DROP TABLE IF EXISTS usage_events;
        DROP TABLE IF EXISTS organizations;
        DROP TABLE IF EXISTS users;
    """
    )
