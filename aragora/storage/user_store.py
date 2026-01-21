"""
UserStore - SQLite backend for user and organization persistence.

Provides CRUD operations for:
- Users (registration, authentication, API keys)
- Organizations (team management, billing)
- Usage tracking (debate counts, monthly resets)

This class serves as a facade over specialized repositories:
- UserRepository: User CRUD and authentication
- OrganizationRepository: Team management and billing
- OAuthRepository: Social login provider linking
- UsageRepository: Rate limiting and usage tracking
- AuditRepository: Audit logging for compliance
- InvitationRepository: Organization invitation workflow
- SecurityRepository: Account lockout and login security

The repositories are created internally and methods delegate to them,
maintaining backward compatibility while enabling modular testing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    from asyncpg import Pool

from aragora.billing.models import Organization, OrganizationInvitation, SubscriptionTier, User
from aragora.storage.repositories import (
    AuditRepository,
    InvitationRepository,
    OAuthRepository,
    OrganizationRepository,
    SecurityRepository,
    UsageRepository,
    UserRepository,
)

logger = logging.getLogger(__name__)


class UserStore:
    """
    SQLite-backed storage for users and organizations.

    Thread-safe with connection pooling via thread-local storage.
    Uses WAL mode for better read/write concurrency.
    """

    # Class-level lock for schema migrations to prevent race conditions
    _schema_lock = threading.Lock()

    # Lockout policy constants (delegated from SecurityRepository for backward compatibility)
    LOCKOUT_THRESHOLD_1 = SecurityRepository.LOCKOUT_THRESHOLD_1
    LOCKOUT_THRESHOLD_2 = SecurityRepository.LOCKOUT_THRESHOLD_2
    LOCKOUT_THRESHOLD_3 = SecurityRepository.LOCKOUT_THRESHOLD_3
    LOCKOUT_DURATION_1 = SecurityRepository.LOCKOUT_DURATION_1
    LOCKOUT_DURATION_2 = SecurityRepository.LOCKOUT_DURATION_2
    LOCKOUT_DURATION_3 = SecurityRepository.LOCKOUT_DURATION_3

    def __init__(self, db_path: Path | str):
        """
        Initialize UserStore.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

        # Initialize specialized repositories
        self._user_repo = UserRepository(self._transaction, self._get_connection)
        self._org_repo = OrganizationRepository(self._transaction, self._row_to_user)
        self._oauth_repo = OAuthRepository(self._transaction)
        self._usage_repo = UsageRepository(self._transaction, self.get_organization_by_id)
        self._audit_repo = AuditRepository(self._transaction)
        self._invitation_repo = InvitationRepository(self._transaction)
        self._security_repo = SecurityRepository(self._transaction)

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.connection = conn
        return self._local.connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for database transactions."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.debug("Transaction rolled back due to: %s", e)
            raise

    def _init_schema(self) -> None:
        """Initialize database schema with migration support."""
        with self._schema_lock, self._transaction() as cursor:
            # Users table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    org_id TEXT,
                    role TEXT DEFAULT 'member',
                    is_active INTEGER DEFAULT 1,
                    email_verified INTEGER DEFAULT 0,
                    api_key TEXT UNIQUE,
                    api_key_hash TEXT UNIQUE,
                    api_key_prefix TEXT,
                    api_key_created_at TEXT,
                    api_key_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT,
                    FOREIGN KEY (org_id) REFERENCES organizations(id)
                )
            """
            )

            self._migrate_api_key_columns(cursor)

            # Organizations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS organizations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    slug TEXT UNIQUE NOT NULL,
                    tier TEXT DEFAULT 'free',
                    owner_id TEXT,
                    stripe_customer_id TEXT,
                    stripe_subscription_id TEXT,
                    debates_used_this_month INTEGER DEFAULT 0,
                    billing_cycle_start TEXT NOT NULL,
                    settings TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (owner_id) REFERENCES users(id)
                )
            """
            )

            # Usage events table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    count INTEGER DEFAULT 1,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (org_id) REFERENCES organizations(id)
                )
            """
            )

            # OAuth providers table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS oauth_providers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    provider_user_id TEXT NOT NULL,
                    email TEXT,
                    linked_at TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    UNIQUE(provider, provider_user_id)
                )
            """
            )

            # Audit log table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    org_id TEXT,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    old_value TEXT,
                    new_value TEXT,
                    metadata TEXT DEFAULT '{}',
                    ip_address TEXT,
                    user_agent TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id),
                    FOREIGN KEY (org_id) REFERENCES organizations(id)
                )
            """
            )

            # Organization invitations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS org_invitations (
                    id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    email TEXT NOT NULL,
                    role TEXT DEFAULT 'member',
                    token TEXT UNIQUE NOT NULL,
                    invited_by TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    accepted_at TEXT,
                    FOREIGN KEY (org_id) REFERENCES organizations(id),
                    FOREIGN KEY (invited_by) REFERENCES users(id)
                )
            """
            )

            # Create indexes
            self._create_indexes(cursor)

        logger.info(f"UserStore initialized: {self.db_path}")

    def _create_indexes(self, cursor: sqlite3.Cursor) -> None:
        """Create database indexes."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_api_key_hash ON users(api_key_hash)",
            "CREATE INDEX IF NOT EXISTS idx_users_org_id ON users(org_id)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_slug ON organizations(slug)",
            "CREATE INDEX IF NOT EXISTS idx_orgs_stripe ON organizations(stripe_customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_usage_org ON usage_events(org_id)",
            "CREATE INDEX IF NOT EXISTS idx_oauth_user ON oauth_providers(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_oauth_provider ON oauth_providers(provider, provider_user_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_org ON audit_log(org_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)",
            "CREATE INDEX IF NOT EXISTS idx_invitations_org ON org_invitations(org_id)",
            "CREATE INDEX IF NOT EXISTS idx_invitations_email ON org_invitations(email)",
            "CREATE INDEX IF NOT EXISTS idx_invitations_token ON org_invitations(token)",
            "CREATE INDEX IF NOT EXISTS idx_invitations_status ON org_invitations(status)",
            # Composite indexes for common query patterns
            "CREATE INDEX IF NOT EXISTS idx_users_org_role ON users(org_id, role)",
            "CREATE INDEX IF NOT EXISTS idx_users_email_active ON users(email, is_active)",
            "CREATE INDEX IF NOT EXISTS idx_usage_org_type ON usage_events(org_id, event_type)",
        ]
        for idx in indexes:
            cursor.execute(idx)

    def _migrate_api_key_columns(self, cursor: sqlite3.Cursor) -> None:
        """Migrate schema to add new columns if they don't exist."""
        cursor.execute("PRAGMA table_info(users)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        migrations = [
            ("api_key_hash", "ALTER TABLE users ADD COLUMN api_key_hash TEXT"),
            ("api_key_prefix", "ALTER TABLE users ADD COLUMN api_key_prefix TEXT"),
            ("api_key_expires_at", "ALTER TABLE users ADD COLUMN api_key_expires_at TEXT"),
            ("mfa_secret", "ALTER TABLE users ADD COLUMN mfa_secret TEXT"),
            ("mfa_enabled", "ALTER TABLE users ADD COLUMN mfa_enabled INTEGER DEFAULT 0"),
            ("mfa_backup_codes", "ALTER TABLE users ADD COLUMN mfa_backup_codes TEXT"),
            ("token_version", "ALTER TABLE users ADD COLUMN token_version INTEGER DEFAULT 1"),
            (
                "failed_login_attempts",
                "ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0",
            ),
            ("lockout_until", "ALTER TABLE users ADD COLUMN lockout_until TEXT"),
            ("last_failed_login_at", "ALTER TABLE users ADD COLUMN last_failed_login_at TEXT"),
            ("preferences", "ALTER TABLE users ADD COLUMN preferences TEXT DEFAULT '{}'"),
        ]

        for col_name, sql in migrations:
            if col_name not in existing_columns:
                cursor.execute(sql)
                logger.info(f"Migration: Added {col_name} column")

    def migrate_plaintext_api_keys(self) -> int:
        """Migrate existing plaintext API keys to hashed storage."""
        migrated = 0
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT id, api_key, api_key_created_at
                FROM users
                WHERE api_key IS NOT NULL AND api_key_hash IS NULL
            """
            )

            for row in cursor.fetchall():
                user_id, api_key, _ = row[0], row[1], row[2]
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                prefix = api_key[:12]
                expires_at = (datetime.now(timezone.utc) + timedelta(days=365)).isoformat()

                cursor.execute(
                    """
                    UPDATE users
                    SET api_key_hash = ?, api_key_prefix = ?, api_key_expires_at = ?, updated_at = ?
                    WHERE id = ?
                """,
                    (key_hash, prefix, expires_at, datetime.now(timezone.utc).isoformat(), user_id),
                )
                migrated += 1
                logger.info(f"Migrated API key for user {user_id}")

        logger.info(f"API key migration complete: {migrated} keys migrated")
        return migrated

    # =========================================================================
    # User Operations (delegated to UserRepository)
    # =========================================================================

    def create_user(
        self,
        email: str,
        password_hash: str,
        password_salt: str,
        name: str = "",
        org_id: Optional[str] = None,
        role: str = "member",
    ) -> User:
        """Create a new user."""
        return self._user_repo.create(email, password_hash, password_salt, name, org_id, role)

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._user_repo.get_by_id(user_id)

    def get_users_batch(self, user_ids: list[str]) -> dict[str, User]:
        """Fetch multiple users in a single query."""
        return self._user_repo.get_batch(user_ids)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        return self._user_repo.get_by_email(email)

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key."""
        return self._user_repo.get_by_api_key(api_key)

    def update_user(self, user_id: str, **fields) -> bool:
        """Update user fields."""
        return self._user_repo.update(user_id, **fields)

    def update_users_batch(self, updates: list[dict[str, Any]]) -> int:
        """Update multiple users in a single transaction."""
        return self._user_repo.update_batch(updates)

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        return self._user_repo.delete(user_id)

    def get_user_preferences(self, user_id: str) -> Optional[dict]:
        """Get user preferences."""
        return self._user_repo.get_preferences(user_id)

    def set_user_preferences(self, user_id: str, preferences: dict) -> bool:
        """Set user preferences."""
        return self._user_repo.set_preferences(user_id, preferences)

    def increment_token_version(self, user_id: str) -> int:
        """Increment token version, invalidating all existing tokens."""
        return self._user_repo.increment_token_version(user_id)

    def _row_to_user(self, row: sqlite3.Row) -> User:
        """Convert database row to User object."""
        return UserRepository._row_to_user(row)

    # =========================================================================
    # Organization Operations (delegated to OrganizationRepository)
    # =========================================================================

    def create_organization(
        self,
        name: str,
        owner_id: str,
        slug: Optional[str] = None,
        tier: SubscriptionTier = SubscriptionTier.FREE,
    ) -> Organization:
        """Create a new organization."""
        return self._org_repo.create(name, owner_id, slug, tier)

    def get_organization_by_id(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        return self._org_repo.get_by_id(org_id)

    def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        return self._org_repo.get_by_slug(slug)

    def get_organization_by_stripe_customer(
        self, stripe_customer_id: str
    ) -> Optional[Organization]:
        """Get organization by Stripe customer ID."""
        return self._org_repo.get_by_stripe_customer(stripe_customer_id)

    def get_organization_by_subscription(self, subscription_id: str) -> Optional[Organization]:
        """Get organization by Stripe subscription ID."""
        return self._org_repo.get_by_subscription(subscription_id)

    def reset_org_usage(self, org_id: str) -> bool:
        """Reset monthly usage for a single organization."""
        return self._org_repo.reset_usage(org_id)

    def update_organization(self, org_id: str, **fields) -> bool:
        """Update organization fields."""
        return self._org_repo.update(org_id, **fields)

    def add_user_to_org(self, user_id: str, org_id: str, role: str = "member") -> bool:
        """Add user to organization."""
        return self._org_repo.add_member(user_id, org_id, role)

    def remove_user_from_org(self, user_id: str) -> bool:
        """Remove user from organization."""
        return self._org_repo.remove_member(user_id)

    def get_org_members(self, org_id: str) -> list[User]:
        """Get all members of an organization."""
        return self._org_repo.get_members(org_id)

    def get_org_members_eager(self, org_id: str) -> tuple[Optional[Organization], list[User]]:
        """Get organization and all its members in a single query operation."""
        return self._org_repo.get_with_members(org_id)

    def get_orgs_with_members_batch(
        self, org_ids: list[str]
    ) -> dict[str, tuple[Organization, list[User]]]:
        """Get multiple organizations with their members in optimized queries."""
        return self._org_repo.get_batch_with_members(org_ids)

    def _row_to_org(self, row: sqlite3.Row) -> Organization:
        """Convert database row to Organization object."""
        return OrganizationRepository._row_to_org(row)

    # =========================================================================
    # Organization Invitations (delegated to InvitationRepository)
    # =========================================================================

    def create_invitation(self, invitation: OrganizationInvitation) -> bool:
        """Create a new organization invitation."""
        return self._invitation_repo.create_invitation(invitation)

    def get_invitation_by_id(self, invitation_id: str) -> Optional[OrganizationInvitation]:
        """Get invitation by ID."""
        return self._invitation_repo.get_by_id(invitation_id)

    def get_invitation_by_token(self, token: str) -> Optional[OrganizationInvitation]:
        """Get invitation by token."""
        return self._invitation_repo.get_by_token(token)

    def get_invitation_by_email(self, org_id: str, email: str) -> Optional[OrganizationInvitation]:
        """Get pending invitation by org and email."""
        return self._invitation_repo.get_by_email(org_id, email)

    def get_invitations_for_org(self, org_id: str) -> list[OrganizationInvitation]:
        """Get all invitations for an organization."""
        return self._invitation_repo.get_for_org(org_id)

    def get_pending_invitations_by_email(self, email: str) -> list[OrganizationInvitation]:
        """Get all pending invitations for an email address."""
        return self._invitation_repo.get_pending_by_email(email)

    def update_invitation_status(
        self,
        invitation_id: str,
        status: str,
        accepted_at: Optional[datetime] = None,
    ) -> bool:
        """Update invitation status."""
        return self._invitation_repo.update_status(invitation_id, status, accepted_at)

    def delete_invitation(self, invitation_id: str) -> bool:
        """Delete an invitation."""
        return self._invitation_repo.delete(invitation_id)

    def cleanup_expired_invitations(self) -> int:
        """Mark expired invitations as expired."""
        return self._invitation_repo.cleanup_expired()

    # =========================================================================
    # Usage Tracking (delegated to UsageRepository)
    # =========================================================================

    def increment_usage(self, org_id: str, count: int = 1) -> int:
        """Increment debate usage for an organization."""
        return self._usage_repo.increment(org_id, count)

    def record_usage_event(
        self,
        org_id: str,
        event_type: str,
        count: int = 1,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a usage event for analytics."""
        self._usage_repo.record_event(org_id, event_type, count, metadata)

    def reset_monthly_usage(self) -> int:
        """Reset monthly usage for all organizations."""
        return self._usage_repo.reset_all_monthly()

    def get_usage_summary(self, org_id: str) -> dict:
        """Get usage summary for an organization."""
        return self._usage_repo.get_summary(org_id)

    # =========================================================================
    # OAuth Provider Operations (delegated to OAuthRepository)
    # =========================================================================

    def link_oauth_provider(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str,
        email: Optional[str] = None,
    ) -> bool:
        """Link an OAuth provider to a user account."""
        return self._oauth_repo.link_provider(user_id, provider, provider_user_id, email)

    def unlink_oauth_provider(self, user_id: str, provider: str) -> bool:
        """Unlink an OAuth provider from a user account."""
        return self._oauth_repo.unlink_provider(user_id, provider)

    def get_user_by_oauth(self, provider: str, provider_user_id: str) -> Optional[User]:
        """Get user by OAuth provider ID."""
        user_id = self._oauth_repo.get_user_id_by_provider(provider, provider_user_id)
        return self.get_user_by_id(user_id) if user_id else None

    def get_user_oauth_providers(self, user_id: str) -> list[dict]:
        """Get all OAuth providers linked to a user."""
        return self._oauth_repo.get_providers_for_user(user_id)

    # =========================================================================
    # Audit Logging Operations (delegated to AuditRepository)
    # =========================================================================

    def log_audit_event(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        old_value: Optional[dict] = None,
        new_value: Optional[dict] = None,
        metadata: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> int:
        """Log an audit event."""
        return self._audit_repo.log_event(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            org_id=org_id,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata,
            ip_address=ip_address,
            user_agent=user_agent,
        )

    def get_audit_log(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query audit log entries."""
        return self._audit_repo.get_log(
            org_id=org_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            since=since,
            until=until,
            limit=limit,
            offset=offset,
        )

    def get_audit_log_count(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> int:
        """Get count of audit log entries matching filters."""
        return self._audit_repo.get_log_count(
            org_id=org_id,
            user_id=user_id,
            action=action,
            resource_type=resource_type,
        )

    # =========================================================================
    # Account Lockout Methods (delegated to SecurityRepository)
    # =========================================================================

    def is_account_locked(self, email: str) -> tuple[bool, Optional[datetime], int]:
        """Check if an account is currently locked."""
        return self._security_repo.is_account_locked(email)

    def record_failed_login(self, email: str) -> tuple[int, Optional[datetime]]:
        """Record a failed login attempt and potentially lock the account."""
        return self._security_repo.record_failed_login(email)

    def reset_failed_login_attempts(self, email: str) -> bool:
        """Reset failed login attempts after successful login."""
        return self._security_repo.reset_failed_login_attempts(email)

    def get_lockout_info(self, email: str) -> dict:
        """Get detailed lockout information for an account."""
        return self._security_repo.get_lockout_info(email)

    # =========================================================================
    # Admin Methods (for admin panel)
    # =========================================================================

    def list_all_organizations(
        self,
        limit: int = 50,
        offset: int = 0,
        tier_filter: Optional[str] = None,
    ) -> tuple[list[Organization], int]:
        """List all organizations with pagination."""
        conn = self._get_connection()
        cursor = conn.cursor()

        where_clause = ""
        params: list[Any] = []
        if tier_filter:
            where_clause = "WHERE tier = ?"
            params.append(tier_filter)

        # nosec B608: where_clause is built from constants, tier_filter is parameterized
        cursor.execute(f"SELECT COUNT(*) FROM organizations {where_clause}", params)  # nosec B608
        total = cursor.fetchone()[0]

        # nosec B608: where_clause is built from constants, all values are parameterized
        query = f"""
            SELECT * FROM organizations
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """  # nosec B608
        params.extend([limit, offset])
        cursor.execute(query, params)

        return [self._row_to_org(row) for row in cursor.fetchall()], total

    def list_all_users(
        self,
        limit: int = 50,
        offset: int = 0,
        org_id_filter: Optional[str] = None,
        role_filter: Optional[str] = None,
        active_only: bool = False,
    ) -> tuple[list[User], int]:
        """List all users with pagination and filtering."""
        conn = self._get_connection()
        cursor = conn.cursor()

        conditions = []
        params: list[Any] = []

        if org_id_filter:
            conditions.append("org_id = ?")
            params.append(org_id_filter)
        if role_filter:
            conditions.append("role = ?")
            params.append(role_filter)
        if active_only:
            conditions.append("is_active = 1")

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # nosec B608: where_clause is built from constants, all filter values are parameterized
        cursor.execute(f"SELECT COUNT(*) FROM users {where_clause}", params.copy())  # nosec B608
        total = cursor.fetchone()[0]

        # nosec B608: where_clause is built from constants, all values are parameterized
        query = f"""
            SELECT * FROM users
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """  # nosec B608
        params.extend([limit, offset])
        cursor.execute(query, params)

        return [self._row_to_user(row) for row in cursor.fetchall()], total

    def get_admin_stats(self) -> dict:
        """Get system-wide statistics for admin dashboard."""
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        cursor.execute("SELECT COUNT(*) FROM users")
        stats["total_users"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
        stats["active_users"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM organizations")
        stats["total_organizations"] = cursor.fetchone()[0]

        cursor.execute("SELECT tier, COUNT(*) as count FROM organizations GROUP BY tier")
        stats["tier_distribution"] = {row["tier"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT SUM(debates_used_this_month) as total FROM organizations")
        result = cursor.fetchone()
        stats["total_debates_this_month"] = result["total"] or 0

        cursor.execute("SELECT COUNT(*) FROM users WHERE last_login_at > datetime('now', '-1 day')")
        stats["users_active_24h"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE created_at > datetime('now', '-7 days')")
        stats["new_users_7d"] = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM organizations WHERE created_at > datetime('now', '-7 days')"
        )
        stats["new_orgs_7d"] = cursor.fetchone()[0]

        return stats

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection


class PostgresUserStore:
    """
    PostgreSQL-backed storage for users and organizations.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent writes.
    """

    SCHEMA_NAME = "user_store"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            password_salt TEXT NOT NULL,
            name TEXT DEFAULT '',
            org_id TEXT,
            role TEXT DEFAULT 'member',
            is_active BOOLEAN DEFAULT TRUE,
            email_verified BOOLEAN DEFAULT FALSE,
            api_key TEXT UNIQUE,
            api_key_hash TEXT UNIQUE,
            api_key_prefix TEXT,
            api_key_created_at TIMESTAMPTZ,
            api_key_expires_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_login_at TIMESTAMPTZ,
            mfa_secret TEXT,
            mfa_enabled BOOLEAN DEFAULT FALSE,
            mfa_backup_codes TEXT,
            token_version INTEGER DEFAULT 1,
            failed_login_attempts INTEGER DEFAULT 0,
            lockout_until TIMESTAMPTZ,
            last_failed_login_at TIMESTAMPTZ,
            preferences JSONB DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
        CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_api_key_hash ON users(api_key_hash);
        CREATE INDEX IF NOT EXISTS idx_users_org_id ON users(org_id);
        CREATE INDEX IF NOT EXISTS idx_users_org_role ON users(org_id, role);
        CREATE INDEX IF NOT EXISTS idx_users_email_active ON users(email, is_active);

        CREATE TABLE IF NOT EXISTS organizations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            tier TEXT DEFAULT 'free',
            owner_id TEXT,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            debates_used_this_month INTEGER DEFAULT 0,
            billing_cycle_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            settings JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_orgs_slug ON organizations(slug);
        CREATE INDEX IF NOT EXISTS idx_orgs_stripe ON organizations(stripe_customer_id);

        CREATE TABLE IF NOT EXISTS usage_events (
            id SERIAL PRIMARY KEY,
            org_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            count INTEGER DEFAULT 1,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_usage_org ON usage_events(org_id);
        CREATE INDEX IF NOT EXISTS idx_usage_org_type ON usage_events(org_id, event_type);

        CREATE TABLE IF NOT EXISTS oauth_providers (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            provider_user_id TEXT NOT NULL,
            email TEXT,
            linked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(provider, provider_user_id)
        );
        CREATE INDEX IF NOT EXISTS idx_oauth_user ON oauth_providers(user_id);
        CREATE INDEX IF NOT EXISTS idx_oauth_provider ON oauth_providers(provider, provider_user_id);

        CREATE TABLE IF NOT EXISTS audit_log (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            user_id TEXT,
            org_id TEXT,
            action TEXT NOT NULL,
            resource_type TEXT NOT NULL,
            resource_id TEXT,
            old_value JSONB,
            new_value JSONB,
            metadata JSONB DEFAULT '{}',
            ip_address TEXT,
            user_agent TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_org ON audit_log(org_id);
        CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action);

        CREATE TABLE IF NOT EXISTS org_invitations (
            id TEXT PRIMARY KEY,
            org_id TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT DEFAULT 'member',
            token TEXT UNIQUE NOT NULL,
            invited_by TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at TIMESTAMPTZ NOT NULL,
            accepted_at TIMESTAMPTZ
        );
        CREATE INDEX IF NOT EXISTS idx_invitations_org ON org_invitations(org_id);
        CREATE INDEX IF NOT EXISTS idx_invitations_email ON org_invitations(email);
        CREATE INDEX IF NOT EXISTS idx_invitations_token ON org_invitations(token);
        CREATE INDEX IF NOT EXISTS idx_invitations_status ON org_invitations(status);
    """

    # Lockout policy constants
    LOCKOUT_THRESHOLD_1 = 3
    LOCKOUT_THRESHOLD_2 = 6
    LOCKOUT_THRESHOLD_3 = 10
    LOCKOUT_DURATION_1 = timedelta(minutes=5)
    LOCKOUT_DURATION_2 = timedelta(minutes=30)
    LOCKOUT_DURATION_3 = timedelta(hours=24)

    def __init__(self, pool: "Pool"):
        """
        Initialize PostgresUserStore.

        Args:
            pool: asyncpg connection pool
        """
        self._pool = pool
        self._initialized = False
        logger.info("PostgresUserStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

    # =========================================================================
    # User Operations
    # =========================================================================

    def create_user(
        self,
        email: str,
        password_hash: str,
        password_salt: str,
        name: str = "",
        org_id: Optional[str] = None,
        role: str = "member",
    ) -> User:
        """Create a new user (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.create_user_async(email, password_hash, password_salt, name, org_id, role)
        )

    async def create_user_async(
        self,
        email: str,
        password_hash: str,
        password_salt: str,
        name: str = "",
        org_id: Optional[str] = None,
        role: str = "member",
    ) -> User:
        """Create a new user asynchronously."""
        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO users
                   (id, email, password_hash, password_salt, name, org_id, role,
                    is_active, email_verified, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, TRUE, FALSE, $8, $8)""",
                user_id, email, password_hash, password_salt, name, org_id, role, now,
            )

        return User(
            id=user_id,
            email=email,
            password_hash=password_hash,
            password_salt=password_salt,
            name=name,
            org_id=org_id,
            role=role,
            is_active=True,
            email_verified=False,
            created_at=now,
            updated_at=now,
        )

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_user_by_id_async(user_id))

    async def get_user_by_id_async(self, user_id: str) -> Optional[User]:
        """Get user by ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, email, password_hash, password_salt, name, org_id, role,
                          is_active, email_verified, api_key, api_key_hash, api_key_prefix,
                          api_key_created_at, api_key_expires_at, created_at, updated_at,
                          last_login_at, mfa_secret, mfa_enabled, mfa_backup_codes,
                          token_version, failed_login_attempts, lockout_until,
                          last_failed_login_at, preferences
                   FROM users WHERE id = $1""",
                user_id,
            )
            if row:
                return self._row_to_user(row)
            return None

    def get_users_batch(self, user_ids: list[str]) -> dict[str, User]:
        """Fetch multiple users in a single query (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_users_batch_async(user_ids))

    async def get_users_batch_async(self, user_ids: list[str]) -> dict[str, User]:
        """Fetch multiple users asynchronously."""
        if not user_ids:
            return {}

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, email, password_hash, password_salt, name, org_id, role,
                          is_active, email_verified, api_key, api_key_hash, api_key_prefix,
                          api_key_created_at, api_key_expires_at, created_at, updated_at,
                          last_login_at, mfa_secret, mfa_enabled, mfa_backup_codes,
                          token_version, failed_login_attempts, lockout_until,
                          last_failed_login_at, preferences
                   FROM users WHERE id = ANY($1)""",
                user_ids,
            )
            return {row["id"]: self._row_to_user(row) for row in rows}

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_user_by_email_async(email))

    async def get_user_by_email_async(self, email: str) -> Optional[User]:
        """Get user by email asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, email, password_hash, password_salt, name, org_id, role,
                          is_active, email_verified, api_key, api_key_hash, api_key_prefix,
                          api_key_created_at, api_key_expires_at, created_at, updated_at,
                          last_login_at, mfa_secret, mfa_enabled, mfa_backup_codes,
                          token_version, failed_login_attempts, lockout_until,
                          last_failed_login_at, preferences
                   FROM users WHERE email = $1""",
                email,
            )
            if row:
                return self._row_to_user(row)
            return None

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """Get user by API key (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_user_by_api_key_async(api_key))

    async def get_user_by_api_key_async(self, api_key: str) -> Optional[User]:
        """Get user by API key asynchronously."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, email, password_hash, password_salt, name, org_id, role,
                          is_active, email_verified, api_key, api_key_hash, api_key_prefix,
                          api_key_created_at, api_key_expires_at, created_at, updated_at,
                          last_login_at, mfa_secret, mfa_enabled, mfa_backup_codes,
                          token_version, failed_login_attempts, lockout_until,
                          last_failed_login_at, preferences
                   FROM users WHERE api_key_hash = $1 OR api_key = $2""",
                key_hash, api_key,
            )
            if row:
                return self._row_to_user(row)
            return None

    def update_user(self, user_id: str, **fields: Any) -> bool:
        """Update user fields (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.update_user_async(user_id, **fields)
        )

    async def update_user_async(self, user_id: str, **fields: Any) -> bool:
        """Update user fields asynchronously."""
        if not fields:
            return False

        updates: list[str] = []
        params: list[Any] = []
        param_idx = 1

        for key, value in fields.items():
            updates.append(f"{key} = ${param_idx}")
            params.append(value)
            param_idx += 1

        updates.append(f"updated_at = ${param_idx}")
        params.append(datetime.now(timezone.utc))
        param_idx += 1
        params.append(user_id)

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = ${param_idx}",
                *params,
            )
            return result != "UPDATE 0"

    def update_users_batch(self, updates: list[dict[str, Any]]) -> int:
        """Update multiple users in a single transaction (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.update_users_batch_async(updates))

    async def update_users_batch_async(self, updates: list[dict[str, Any]]) -> int:
        """Update multiple users asynchronously."""
        count = 0
        for update in updates:
            user_id = update.pop("user_id", None) or update.pop("id", None)
            if user_id and update:
                if await self.update_user_async(user_id, **update):
                    count += 1
        return count

    def delete_user(self, user_id: str) -> bool:
        """Delete a user (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.delete_user_async(user_id))

    async def delete_user_async(self, user_id: str) -> bool:
        """Delete a user asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute("DELETE FROM users WHERE id = $1", user_id)
            return result != "DELETE 0"

    def get_user_preferences(self, user_id: str) -> Optional[dict]:
        """Get user preferences (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_user_preferences_async(user_id)
        )

    async def get_user_preferences_async(self, user_id: str) -> Optional[dict]:
        """Get user preferences asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT preferences FROM users WHERE id = $1", user_id
            )
            if row and row["preferences"]:
                prefs = row["preferences"]
                return json.loads(prefs) if isinstance(prefs, str) else prefs
            return None

    def set_user_preferences(self, user_id: str, preferences: dict) -> bool:
        """Set user preferences (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.set_user_preferences_async(user_id, preferences)
        )

    async def set_user_preferences_async(self, user_id: str, preferences: dict) -> bool:
        """Set user preferences asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE users SET preferences = $1, updated_at = $2 WHERE id = $3",
                json.dumps(preferences), datetime.now(timezone.utc), user_id,
            )
            return result != "UPDATE 0"

    def increment_token_version(self, user_id: str) -> int:
        """Increment token version (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.increment_token_version_async(user_id)
        )

    async def increment_token_version_async(self, user_id: str) -> int:
        """Increment token version asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """UPDATE users SET token_version = token_version + 1, updated_at = $1
                   WHERE id = $2 RETURNING token_version""",
                datetime.now(timezone.utc), user_id,
            )
            return row["token_version"] if row else 1

    def _row_to_user(self, row: Any) -> User:
        """Convert database row to User object."""
        return User(
            id=row["id"],
            email=row["email"],
            password_hash=row["password_hash"],
            password_salt=row["password_salt"],
            name=row["name"] or "",
            org_id=row["org_id"],
            role=row["role"] or "member",
            is_active=bool(row["is_active"]),
            email_verified=bool(row["email_verified"]),
            api_key=row["api_key"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_login_at=row["last_login_at"],
        )

    # =========================================================================
    # Organization Operations
    # =========================================================================

    def create_organization(
        self,
        name: str,
        owner_id: str,
        slug: Optional[str] = None,
        tier: SubscriptionTier = SubscriptionTier.FREE,
    ) -> Organization:
        """Create a new organization (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.create_organization_async(name, owner_id, slug, tier)
        )

    async def create_organization_async(
        self,
        name: str,
        owner_id: str,
        slug: Optional[str] = None,
        tier: SubscriptionTier = SubscriptionTier.FREE,
    ) -> Organization:
        """Create a new organization asynchronously."""
        org_id = str(uuid.uuid4())
        if not slug:
            slug = name.lower().replace(" ", "-")[:50] + "-" + org_id[:8]
        now = datetime.now(timezone.utc)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO organizations
                   (id, name, slug, tier, owner_id, billing_cycle_start, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $6, $6)""",
                org_id, name, slug, tier.value if hasattr(tier, 'value') else str(tier),
                owner_id, now,
            )
            # Update owner's org_id
            await conn.execute(
                "UPDATE users SET org_id = $1, role = 'owner', updated_at = $2 WHERE id = $3",
                org_id, now, owner_id,
            )

        return Organization(
            id=org_id,
            name=name,
            slug=slug,
            tier=tier,
            owner_id=owner_id,
            billing_cycle_start=now,
            created_at=now,
            updated_at=now,
        )

    def get_organization_by_id(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_organization_by_id_async(org_id)
        )

    async def get_organization_by_id_async(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, name, slug, tier, owner_id, stripe_customer_id,
                          stripe_subscription_id, debates_used_this_month,
                          billing_cycle_start, settings, created_at, updated_at
                   FROM organizations WHERE id = $1""",
                org_id,
            )
            if row:
                return self._row_to_org(row)
            return None

    def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_organization_by_slug_async(slug)
        )

    async def get_organization_by_slug_async(self, slug: str) -> Optional[Organization]:
        """Get organization by slug asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, name, slug, tier, owner_id, stripe_customer_id,
                          stripe_subscription_id, debates_used_this_month,
                          billing_cycle_start, settings, created_at, updated_at
                   FROM organizations WHERE slug = $1""",
                slug,
            )
            if row:
                return self._row_to_org(row)
            return None

    def get_organization_by_stripe_customer(self, stripe_customer_id: str) -> Optional[Organization]:
        """Get organization by Stripe customer ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_organization_by_stripe_customer_async(stripe_customer_id)
        )

    async def get_organization_by_stripe_customer_async(
        self, stripe_customer_id: str
    ) -> Optional[Organization]:
        """Get organization by Stripe customer ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, name, slug, tier, owner_id, stripe_customer_id,
                          stripe_subscription_id, debates_used_this_month,
                          billing_cycle_start, settings, created_at, updated_at
                   FROM organizations WHERE stripe_customer_id = $1""",
                stripe_customer_id,
            )
            if row:
                return self._row_to_org(row)
            return None

    def get_organization_by_subscription(self, subscription_id: str) -> Optional[Organization]:
        """Get organization by Stripe subscription ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_organization_by_subscription_async(subscription_id)
        )

    async def get_organization_by_subscription_async(
        self, subscription_id: str
    ) -> Optional[Organization]:
        """Get organization by Stripe subscription ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT id, name, slug, tier, owner_id, stripe_customer_id,
                          stripe_subscription_id, debates_used_this_month,
                          billing_cycle_start, settings, created_at, updated_at
                   FROM organizations WHERE stripe_subscription_id = $1""",
                subscription_id,
            )
            if row:
                return self._row_to_org(row)
            return None

    def update_organization(self, org_id: str, **fields: Any) -> bool:
        """Update organization fields (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.update_organization_async(org_id, **fields)
        )

    async def update_organization_async(self, org_id: str, **fields: Any) -> bool:
        """Update organization fields asynchronously."""
        if not fields:
            return False

        updates: list[str] = []
        params: list[Any] = []
        param_idx = 1

        for key, value in fields.items():
            if key == "settings" and isinstance(value, dict):
                value = json.dumps(value)
            updates.append(f"{key} = ${param_idx}")
            params.append(value)
            param_idx += 1

        updates.append(f"updated_at = ${param_idx}")
        params.append(datetime.now(timezone.utc))
        param_idx += 1
        params.append(org_id)

        async with self._pool.acquire() as conn:
            result = await conn.execute(
                f"UPDATE organizations SET {', '.join(updates)} WHERE id = ${param_idx}",
                *params,
            )
            return result != "UPDATE 0"

    def reset_org_usage(self, org_id: str) -> bool:
        """Reset monthly usage for a single organization (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.reset_org_usage_async(org_id))

    async def reset_org_usage_async(self, org_id: str) -> bool:
        """Reset monthly usage asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE organizations SET debates_used_this_month = 0,
                   billing_cycle_start = $1, updated_at = $1 WHERE id = $2""",
                datetime.now(timezone.utc), org_id,
            )
            return result != "UPDATE 0"

    def add_user_to_org(self, user_id: str, org_id: str, role: str = "member") -> bool:
        """Add user to organization (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.add_user_to_org_async(user_id, org_id, role)
        )

    async def add_user_to_org_async(self, user_id: str, org_id: str, role: str = "member") -> bool:
        """Add user to organization asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE users SET org_id = $1, role = $2, updated_at = $3 WHERE id = $4",
                org_id, role, datetime.now(timezone.utc), user_id,
            )
            return result != "UPDATE 0"

    def remove_user_from_org(self, user_id: str) -> bool:
        """Remove user from organization (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.remove_user_from_org_async(user_id))

    async def remove_user_from_org_async(self, user_id: str) -> bool:
        """Remove user from organization asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "UPDATE users SET org_id = NULL, role = 'member', updated_at = $1 WHERE id = $2",
                datetime.now(timezone.utc), user_id,
            )
            return result != "UPDATE 0"

    def get_org_members(self, org_id: str) -> list[User]:
        """Get all members of an organization (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_org_members_async(org_id))

    async def get_org_members_async(self, org_id: str) -> list[User]:
        """Get all members of an organization asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, email, password_hash, password_salt, name, org_id, role,
                          is_active, email_verified, api_key, api_key_hash, api_key_prefix,
                          api_key_created_at, api_key_expires_at, created_at, updated_at,
                          last_login_at, mfa_secret, mfa_enabled, mfa_backup_codes,
                          token_version, failed_login_attempts, lockout_until,
                          last_failed_login_at, preferences
                   FROM users WHERE org_id = $1 ORDER BY created_at""",
                org_id,
            )
            return [self._row_to_user(row) for row in rows]

    def get_org_members_eager(self, org_id: str) -> tuple[Optional[Organization], list[User]]:
        """Get organization and all its members (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_org_members_eager_async(org_id))

    async def get_org_members_eager_async(
        self, org_id: str
    ) -> tuple[Optional[Organization], list[User]]:
        """Get organization and all its members asynchronously."""
        org = await self.get_organization_by_id_async(org_id)
        if not org:
            return None, []
        members = await self.get_org_members_async(org_id)
        return org, members

    def get_orgs_with_members_batch(
        self, org_ids: list[str]
    ) -> dict[str, tuple[Organization, list[User]]]:
        """Get multiple organizations with their members (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_orgs_with_members_batch_async(org_ids)
        )

    async def get_orgs_with_members_batch_async(
        self, org_ids: list[str]
    ) -> dict[str, tuple[Organization, list[User]]]:
        """Get multiple organizations with their members asynchronously."""
        result: dict[str, tuple[Organization, list[User]]] = {}
        for org_id in org_ids:
            org, members = await self.get_org_members_eager_async(org_id)
            if org:
                result[org_id] = (org, members)
        return result

    def _row_to_org(self, row: Any) -> Organization:
        """Convert database row to Organization object."""
        settings = row["settings"]
        if isinstance(settings, str):
            settings = json.loads(settings)

        return Organization(
            id=row["id"],
            name=row["name"],
            slug=row["slug"],
            tier=SubscriptionTier(row["tier"]) if row["tier"] else SubscriptionTier.FREE,
            owner_id=row["owner_id"],
            stripe_customer_id=row["stripe_customer_id"],
            stripe_subscription_id=row["stripe_subscription_id"],
            debates_used_this_month=row["debates_used_this_month"] or 0,
            billing_cycle_start=row["billing_cycle_start"],
            settings=settings or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # =========================================================================
    # Usage Tracking
    # =========================================================================

    def increment_usage(self, org_id: str, count: int = 1) -> int:
        """Increment debate usage for an organization (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.increment_usage_async(org_id, count)
        )

    async def increment_usage_async(self, org_id: str, count: int = 1) -> int:
        """Increment debate usage asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """UPDATE organizations SET debates_used_this_month = debates_used_this_month + $1,
                   updated_at = $2 WHERE id = $3 RETURNING debates_used_this_month""",
                count, datetime.now(timezone.utc), org_id,
            )
            return row["debates_used_this_month"] if row else 0

    def record_usage_event(
        self,
        org_id: str,
        event_type: str,
        count: int = 1,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a usage event for analytics (sync wrapper)."""
        asyncio.get_event_loop().run_until_complete(
            self.record_usage_event_async(org_id, event_type, count, metadata)
        )

    async def record_usage_event_async(
        self,
        org_id: str,
        event_type: str,
        count: int = 1,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a usage event asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO usage_events (org_id, event_type, count, metadata, created_at)
                   VALUES ($1, $2, $3, $4, $5)""",
                org_id, event_type, count, json.dumps(metadata or {}), datetime.now(timezone.utc),
            )

    def reset_monthly_usage(self) -> int:
        """Reset monthly usage for all organizations (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.reset_monthly_usage_async())

    async def reset_monthly_usage_async(self) -> int:
        """Reset monthly usage asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE organizations SET debates_used_this_month = 0,
                   billing_cycle_start = $1, updated_at = $1""",
                datetime.now(timezone.utc),
            )
            # Parse "UPDATE N" to get count
            parts = result.split()
            return int(parts[1]) if len(parts) > 1 else 0

    def get_usage_summary(self, org_id: str) -> dict:
        """Get usage summary for an organization (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_usage_summary_async(org_id))

    async def get_usage_summary_async(self, org_id: str) -> dict:
        """Get usage summary asynchronously."""
        org = await self.get_organization_by_id_async(org_id)
        if not org:
            return {}

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT event_type, SUM(count) as total
                   FROM usage_events WHERE org_id = $1 GROUP BY event_type""",
                org_id,
            )

        return {
            "org_id": org_id,
            "debates_used_this_month": org.debates_used_this_month,
            "billing_cycle_start": org.billing_cycle_start.isoformat() if org.billing_cycle_start else None,
            "events": {row["event_type"]: row["total"] for row in rows},
        }

    # =========================================================================
    # OAuth Provider Operations
    # =========================================================================

    def link_oauth_provider(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str,
        email: Optional[str] = None,
    ) -> bool:
        """Link an OAuth provider to a user account (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.link_oauth_provider_async(user_id, provider, provider_user_id, email)
        )

    async def link_oauth_provider_async(
        self,
        user_id: str,
        provider: str,
        provider_user_id: str,
        email: Optional[str] = None,
    ) -> bool:
        """Link an OAuth provider asynchronously."""
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """INSERT INTO oauth_providers (user_id, provider, provider_user_id, email, linked_at)
                       VALUES ($1, $2, $3, $4, $5)
                       ON CONFLICT (provider, provider_user_id) DO NOTHING""",
                    user_id, provider, provider_user_id, email, datetime.now(timezone.utc),
                )
                return True
            except Exception:
                return False

    def unlink_oauth_provider(self, user_id: str, provider: str) -> bool:
        """Unlink an OAuth provider from a user account (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.unlink_oauth_provider_async(user_id, provider)
        )

    async def unlink_oauth_provider_async(self, user_id: str, provider: str) -> bool:
        """Unlink an OAuth provider asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM oauth_providers WHERE user_id = $1 AND provider = $2",
                user_id, provider,
            )
            return result != "DELETE 0"

    def get_user_by_oauth(self, provider: str, provider_user_id: str) -> Optional[User]:
        """Get user by OAuth provider ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_user_by_oauth_async(provider, provider_user_id)
        )

    async def get_user_by_oauth_async(self, provider: str, provider_user_id: str) -> Optional[User]:
        """Get user by OAuth provider ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id FROM oauth_providers WHERE provider = $1 AND provider_user_id = $2",
                provider, provider_user_id,
            )
            if row:
                return await self.get_user_by_id_async(row["user_id"])
            return None

    def get_user_oauth_providers(self, user_id: str) -> list[dict]:
        """Get all OAuth providers linked to a user (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_user_oauth_providers_async(user_id)
        )

    async def get_user_oauth_providers_async(self, user_id: str) -> list[dict]:
        """Get all OAuth providers asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT provider, provider_user_id, email, linked_at FROM oauth_providers WHERE user_id = $1",
                user_id,
            )
            return [
                {
                    "provider": row["provider"],
                    "provider_user_id": row["provider_user_id"],
                    "email": row["email"],
                    "linked_at": row["linked_at"].isoformat() if row["linked_at"] else None,
                }
                for row in rows
            ]

    # =========================================================================
    # Audit Logging Operations
    # =========================================================================

    def log_audit_event(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        old_value: Optional[dict] = None,
        new_value: Optional[dict] = None,
        metadata: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> int:
        """Log an audit event (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.log_audit_event_async(
                action, resource_type, resource_id, user_id, org_id,
                old_value, new_value, metadata, ip_address, user_agent
            )
        )

    async def log_audit_event_async(
        self,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        org_id: Optional[str] = None,
        old_value: Optional[dict] = None,
        new_value: Optional[dict] = None,
        metadata: Optional[dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> int:
        """Log an audit event asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """INSERT INTO audit_log
                   (timestamp, user_id, org_id, action, resource_type, resource_id,
                    old_value, new_value, metadata, ip_address, user_agent)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                   RETURNING id""",
                datetime.now(timezone.utc), user_id, org_id, action, resource_type, resource_id,
                json.dumps(old_value) if old_value else None,
                json.dumps(new_value) if new_value else None,
                json.dumps(metadata or {}),
                ip_address, user_agent,
            )
            return row["id"] if row else 0

    def get_audit_log(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query audit log entries (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_audit_log_async(
                org_id, user_id, action, resource_type, since, until, limit, offset
            )
        )

    async def get_audit_log_async(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Query audit log entries asynchronously."""
        query = "SELECT * FROM audit_log WHERE 1=1"
        params: list[Any] = []
        param_idx = 1

        if org_id:
            query += f" AND org_id = ${param_idx}"
            params.append(org_id)
            param_idx += 1
        if user_id:
            query += f" AND user_id = ${param_idx}"
            params.append(user_id)
            param_idx += 1
        if action:
            query += f" AND action = ${param_idx}"
            params.append(action)
            param_idx += 1
        if resource_type:
            query += f" AND resource_type = ${param_idx}"
            params.append(resource_type)
            param_idx += 1
        if since:
            query += f" AND timestamp >= ${param_idx}"
            params.append(since)
            param_idx += 1
        if until:
            query += f" AND timestamp <= ${param_idx}"
            params.append(until)
            param_idx += 1

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
        params.extend([limit, offset])

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                    "user_id": row["user_id"],
                    "org_id": row["org_id"],
                    "action": row["action"],
                    "resource_type": row["resource_type"],
                    "resource_id": row["resource_id"],
                    "old_value": json.loads(row["old_value"]) if row["old_value"] else None,
                    "new_value": json.loads(row["new_value"]) if row["new_value"] else None,
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "ip_address": row["ip_address"],
                    "user_agent": row["user_agent"],
                }
                for row in rows
            ]

    def get_audit_log_count(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> int:
        """Get count of audit log entries (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_audit_log_count_async(org_id, user_id, action, resource_type)
        )

    async def get_audit_log_count_async(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> int:
        """Get count of audit log entries asynchronously."""
        query = "SELECT COUNT(*) FROM audit_log WHERE 1=1"
        params: list[Any] = []
        param_idx = 1

        if org_id:
            query += f" AND org_id = ${param_idx}"
            params.append(org_id)
            param_idx += 1
        if user_id:
            query += f" AND user_id = ${param_idx}"
            params.append(user_id)
            param_idx += 1
        if action:
            query += f" AND action = ${param_idx}"
            params.append(action)
            param_idx += 1
        if resource_type:
            query += f" AND resource_type = ${param_idx}"
            params.append(resource_type)
            param_idx += 1

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, *params)
            return row[0] if row else 0

    # =========================================================================
    # Organization Invitations
    # =========================================================================

    def create_invitation(self, invitation: OrganizationInvitation) -> bool:
        """Create a new organization invitation (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.create_invitation_async(invitation)
        )

    async def create_invitation_async(self, invitation: OrganizationInvitation) -> bool:
        """Create a new organization invitation asynchronously."""
        async with self._pool.acquire() as conn:
            try:
                await conn.execute(
                    """INSERT INTO org_invitations
                       (id, org_id, email, role, token, invited_by, status, created_at, expires_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                    invitation.id, invitation.org_id, invitation.email, invitation.role,
                    invitation.token, invitation.invited_by, invitation.status,
                    invitation.created_at, invitation.expires_at,
                )
                return True
            except Exception:
                return False

    def get_invitation_by_id(self, invitation_id: str) -> Optional[OrganizationInvitation]:
        """Get invitation by ID (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_invitation_by_id_async(invitation_id)
        )

    async def get_invitation_by_id_async(self, invitation_id: str) -> Optional[OrganizationInvitation]:
        """Get invitation by ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM org_invitations WHERE id = $1", invitation_id
            )
            if row:
                return self._row_to_invitation(row)
            return None

    def get_invitation_by_token(self, token: str) -> Optional[OrganizationInvitation]:
        """Get invitation by token (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_invitation_by_token_async(token)
        )

    async def get_invitation_by_token_async(self, token: str) -> Optional[OrganizationInvitation]:
        """Get invitation by token asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM org_invitations WHERE token = $1", token
            )
            if row:
                return self._row_to_invitation(row)
            return None

    def get_invitation_by_email(self, org_id: str, email: str) -> Optional[OrganizationInvitation]:
        """Get pending invitation by org and email (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_invitation_by_email_async(org_id, email)
        )

    async def get_invitation_by_email_async(
        self, org_id: str, email: str
    ) -> Optional[OrganizationInvitation]:
        """Get pending invitation asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM org_invitations
                   WHERE org_id = $1 AND email = $2 AND status = 'pending'""",
                org_id, email,
            )
            if row:
                return self._row_to_invitation(row)
            return None

    def get_invitations_for_org(self, org_id: str) -> list[OrganizationInvitation]:
        """Get all invitations for an organization (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_invitations_for_org_async(org_id)
        )

    async def get_invitations_for_org_async(self, org_id: str) -> list[OrganizationInvitation]:
        """Get all invitations for an organization asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM org_invitations WHERE org_id = $1 ORDER BY created_at DESC",
                org_id,
            )
            return [self._row_to_invitation(row) for row in rows]

    def get_pending_invitations_by_email(self, email: str) -> list[OrganizationInvitation]:
        """Get all pending invitations for an email address (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_pending_invitations_by_email_async(email)
        )

    async def get_pending_invitations_by_email_async(
        self, email: str
    ) -> list[OrganizationInvitation]:
        """Get all pending invitations asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM org_invitations
                   WHERE email = $1 AND status = 'pending' ORDER BY created_at DESC""",
                email,
            )
            return [self._row_to_invitation(row) for row in rows]

    def update_invitation_status(
        self,
        invitation_id: str,
        status: str,
        accepted_at: Optional[datetime] = None,
    ) -> bool:
        """Update invitation status (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.update_invitation_status_async(invitation_id, status, accepted_at)
        )

    async def update_invitation_status_async(
        self,
        invitation_id: str,
        status: str,
        accepted_at: Optional[datetime] = None,
    ) -> bool:
        """Update invitation status asynchronously."""
        async with self._pool.acquire() as conn:
            if accepted_at:
                result = await conn.execute(
                    "UPDATE org_invitations SET status = $1, accepted_at = $2 WHERE id = $3",
                    status, accepted_at, invitation_id,
                )
            else:
                result = await conn.execute(
                    "UPDATE org_invitations SET status = $1 WHERE id = $2",
                    status, invitation_id,
                )
            return result != "UPDATE 0"

    def delete_invitation(self, invitation_id: str) -> bool:
        """Delete an invitation (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.delete_invitation_async(invitation_id)
        )

    async def delete_invitation_async(self, invitation_id: str) -> bool:
        """Delete an invitation asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM org_invitations WHERE id = $1", invitation_id
            )
            return result != "DELETE 0"

    def cleanup_expired_invitations(self) -> int:
        """Mark expired invitations as expired (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.cleanup_expired_invitations_async()
        )

    async def cleanup_expired_invitations_async(self) -> int:
        """Mark expired invitations asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE org_invitations SET status = 'expired'
                   WHERE status = 'pending' AND expires_at < $1""",
                datetime.now(timezone.utc),
            )
            parts = result.split()
            return int(parts[1]) if len(parts) > 1 else 0

    def _row_to_invitation(self, row: Any) -> OrganizationInvitation:
        """Convert database row to OrganizationInvitation object."""
        return OrganizationInvitation(
            id=row["id"],
            org_id=row["org_id"],
            email=row["email"],
            role=row["role"] or "member",
            token=row["token"],
            invited_by=row["invited_by"],
            status=row["status"] or "pending",
            created_at=row["created_at"],
            expires_at=row["expires_at"],
            accepted_at=row["accepted_at"],
        )

    # =========================================================================
    # Account Lockout Methods
    # =========================================================================

    def is_account_locked(self, email: str) -> tuple[bool, Optional[datetime], int]:
        """Check if an account is currently locked (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.is_account_locked_async(email))

    async def is_account_locked_async(self, email: str) -> tuple[bool, Optional[datetime], int]:
        """Check if an account is currently locked asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT failed_login_attempts, lockout_until FROM users WHERE email = $1",
                email,
            )
            if not row:
                return False, None, 0

            lockout_until = row["lockout_until"]
            attempts = row["failed_login_attempts"] or 0

            if lockout_until and lockout_until > datetime.now(timezone.utc):
                return True, lockout_until, attempts
            return False, None, attempts

    def record_failed_login(self, email: str) -> tuple[int, Optional[datetime]]:
        """Record a failed login attempt (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.record_failed_login_async(email))

    async def record_failed_login_async(self, email: str) -> tuple[int, Optional[datetime]]:
        """Record a failed login attempt asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """UPDATE users SET
                   failed_login_attempts = failed_login_attempts + 1,
                   last_failed_login_at = $1,
                   updated_at = $1
                   WHERE email = $2
                   RETURNING failed_login_attempts""",
                datetime.now(timezone.utc), email,
            )
            if not row:
                return 0, None

            attempts = row["failed_login_attempts"]
            lockout_until = None

            # Determine lockout duration based on attempts
            if attempts >= self.LOCKOUT_THRESHOLD_3:
                lockout_until = datetime.now(timezone.utc) + self.LOCKOUT_DURATION_3
            elif attempts >= self.LOCKOUT_THRESHOLD_2:
                lockout_until = datetime.now(timezone.utc) + self.LOCKOUT_DURATION_2
            elif attempts >= self.LOCKOUT_THRESHOLD_1:
                lockout_until = datetime.now(timezone.utc) + self.LOCKOUT_DURATION_1

            if lockout_until:
                await conn.execute(
                    "UPDATE users SET lockout_until = $1 WHERE email = $2",
                    lockout_until, email,
                )

            return attempts, lockout_until

    def reset_failed_login_attempts(self, email: str) -> bool:
        """Reset failed login attempts (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.reset_failed_login_attempts_async(email)
        )

    async def reset_failed_login_attempts_async(self, email: str) -> bool:
        """Reset failed login attempts asynchronously."""
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """UPDATE users SET
                   failed_login_attempts = 0, lockout_until = NULL,
                   last_failed_login_at = NULL, updated_at = $1
                   WHERE email = $2""",
                datetime.now(timezone.utc), email,
            )
            return result != "UPDATE 0"

    def get_lockout_info(self, email: str) -> dict:
        """Get detailed lockout information (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_lockout_info_async(email))

    async def get_lockout_info_async(self, email: str) -> dict:
        """Get detailed lockout information asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT failed_login_attempts, lockout_until, last_failed_login_at
                   FROM users WHERE email = $1""",
                email,
            )
            if not row:
                return {"exists": False}

            return {
                "exists": True,
                "failed_attempts": row["failed_login_attempts"] or 0,
                "lockout_until": row["lockout_until"].isoformat() if row["lockout_until"] else None,
                "last_failed_at": row["last_failed_login_at"].isoformat() if row["last_failed_login_at"] else None,
                "is_locked": bool(row["lockout_until"] and row["lockout_until"] > datetime.now(timezone.utc)),
            }

    # =========================================================================
    # Admin Methods
    # =========================================================================

    def list_all_organizations(
        self,
        limit: int = 50,
        offset: int = 0,
        tier_filter: Optional[str] = None,
    ) -> tuple[list[Organization], int]:
        """List all organizations with pagination (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_all_organizations_async(limit, offset, tier_filter)
        )

    async def list_all_organizations_async(
        self,
        limit: int = 50,
        offset: int = 0,
        tier_filter: Optional[str] = None,
    ) -> tuple[list[Organization], int]:
        """List all organizations asynchronously."""
        async with self._pool.acquire() as conn:
            if tier_filter:
                total_row = await conn.fetchrow(
                    "SELECT COUNT(*) FROM organizations WHERE tier = $1", tier_filter
                )
                rows = await conn.fetch(
                    """SELECT * FROM organizations WHERE tier = $1
                       ORDER BY created_at DESC LIMIT $2 OFFSET $3""",
                    tier_filter, limit, offset,
                )
            else:
                total_row = await conn.fetchrow("SELECT COUNT(*) FROM organizations")
                rows = await conn.fetch(
                    """SELECT * FROM organizations
                       ORDER BY created_at DESC LIMIT $1 OFFSET $2""",
                    limit, offset,
                )

            total = total_row[0] if total_row else 0
            return [self._row_to_org(row) for row in rows], total

    def list_all_users(
        self,
        limit: int = 50,
        offset: int = 0,
        org_id_filter: Optional[str] = None,
        role_filter: Optional[str] = None,
        active_only: bool = False,
    ) -> tuple[list[User], int]:
        """List all users with pagination (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_all_users_async(limit, offset, org_id_filter, role_filter, active_only)
        )

    async def list_all_users_async(
        self,
        limit: int = 50,
        offset: int = 0,
        org_id_filter: Optional[str] = None,
        role_filter: Optional[str] = None,
        active_only: bool = False,
    ) -> tuple[list[User], int]:
        """List all users asynchronously."""
        query = "SELECT * FROM users WHERE 1=1"
        count_query = "SELECT COUNT(*) FROM users WHERE 1=1"
        params: list[Any] = []
        param_idx = 1

        if org_id_filter:
            query += f" AND org_id = ${param_idx}"
            count_query += f" AND org_id = ${param_idx}"
            params.append(org_id_filter)
            param_idx += 1
        if role_filter:
            query += f" AND role = ${param_idx}"
            count_query += f" AND role = ${param_idx}"
            params.append(role_filter)
            param_idx += 1
        if active_only:
            query += " AND is_active = TRUE"
            count_query += " AND is_active = TRUE"

        async with self._pool.acquire() as conn:
            total_row = await conn.fetchrow(count_query, *params)
            total = total_row[0] if total_row else 0

            query += f" ORDER BY created_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            params.extend([limit, offset])
            rows = await conn.fetch(query, *params)

            return [self._row_to_user(row) for row in rows], total

    def get_admin_stats(self) -> dict:
        """Get system-wide statistics (sync wrapper)."""
        return asyncio.get_event_loop().run_until_complete(self.get_admin_stats_async())

    async def get_admin_stats_async(self) -> dict:
        """Get system-wide statistics asynchronously."""
        async with self._pool.acquire() as conn:
            stats: dict[str, Any] = {}

            row = await conn.fetchrow("SELECT COUNT(*) FROM users")
            stats["total_users"] = row[0] if row else 0

            row = await conn.fetchrow("SELECT COUNT(*) FROM users WHERE is_active = TRUE")
            stats["active_users"] = row[0] if row else 0

            row = await conn.fetchrow("SELECT COUNT(*) FROM organizations")
            stats["total_organizations"] = row[0] if row else 0

            rows = await conn.fetch(
                "SELECT tier, COUNT(*) as count FROM organizations GROUP BY tier"
            )
            stats["tier_distribution"] = {row["tier"]: row["count"] for row in rows}

            row = await conn.fetchrow(
                "SELECT SUM(debates_used_this_month) as total FROM organizations"
            )
            stats["total_debates_this_month"] = row["total"] or 0 if row else 0

            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM users WHERE last_login_at > NOW() - INTERVAL '1 day'"
            )
            stats["users_active_24h"] = row[0] if row else 0

            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM users WHERE created_at > NOW() - INTERVAL '7 days'"
            )
            stats["new_users_7d"] = row[0] if row else 0

            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM organizations WHERE created_at > NOW() - INTERVAL '7 days'"
            )
            stats["new_orgs_7d"] = row[0] if row else 0

            return stats

    def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


# Singleton instance for global access
_user_store_instance: Optional[UserStore] = None
_postgres_user_store_instance: Optional[PostgresUserStore] = None


def get_user_store() -> Optional[UserStore | PostgresUserStore]:
    """
    Get or create the user store.

    Uses environment variables to configure:
    - ARAGORA_DB_BACKEND: "sqlite" or "postgres" (selects database backend)
    - ARAGORA_DATA_DIR: Directory for SQLite database
    - ARAGORA_POSTGRES_DSN or DATABASE_URL: PostgreSQL connection string

    Returns:
        Configured UserStore or PostgresUserStore instance
    """
    global _user_store_instance, _postgres_user_store_instance

    # Check if PostgreSQL backend is requested
    backend_type = os.environ.get("ARAGORA_DB_BACKEND", "sqlite").lower()

    if backend_type in ("postgres", "postgresql"):
        if _postgres_user_store_instance is not None:
            return _postgres_user_store_instance

        logger.info("Using PostgreSQL user store")
        try:
            from aragora.storage.postgres_store import get_postgres_pool

            pool = asyncio.get_event_loop().run_until_complete(get_postgres_pool())
            store = PostgresUserStore(pool)
            asyncio.get_event_loop().run_until_complete(store.initialize())
            _postgres_user_store_instance = store
            return store
        except Exception as e:
            logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")
            # Fall through to SQLite

    # Default: SQLite
    if _user_store_instance is not None:
        return _user_store_instance

    # Get data directory for SQLite
    try:
        from aragora.config.legacy import DATA_DIR

        data_dir = DATA_DIR
    except ImportError:
        env_dir = os.environ.get("ARAGORA_DATA_DIR") or os.environ.get("ARAGORA_NOMIC_DIR")
        data_dir = Path(env_dir or ".nomic")

    db_path = data_dir / "users.db"
    logger.info(f"Using SQLite user store: {db_path}")
    _user_store_instance = UserStore(db_path)
    return _user_store_instance


def set_user_store(store: UserStore | PostgresUserStore) -> None:
    """Set the global UserStore singleton instance."""
    global _user_store_instance, _postgres_user_store_instance
    if isinstance(store, PostgresUserStore):
        _postgres_user_store_instance = store
    else:
        _user_store_instance = store


def reset_user_store() -> None:
    """Reset the global user store (for testing)."""
    global _user_store_instance, _postgres_user_store_instance
    _user_store_instance = None
    _postgres_user_store_instance = None


__all__ = [
    "UserStore",
    "PostgresUserStore",
    "get_user_store",
    "set_user_store",
    "reset_user_store",
]
