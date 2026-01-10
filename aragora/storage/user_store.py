"""
UserStore - SQLite backend for user and organization persistence.

Provides CRUD operations for:
- Users (registration, authentication, API keys)
- Organizations (team management, billing)
- Usage tracking (debate counts, monthly resets)
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from aragora.billing.models import Organization, SubscriptionTier, User

logger = logging.getLogger(__name__)


class UserStore:
    """
    SQLite-backed storage for users and organizations.

    Thread-safe with connection pooling via thread-local storage.
    """

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

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
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
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._transaction() as cursor:
            # Users table
            cursor.execute("""
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
            """)

            # Migration: Add new columns if they don't exist
            self._migrate_api_key_columns(cursor)

            # Organizations table
            cursor.execute("""
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
            """)

            # Usage events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    org_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    count INTEGER DEFAULT 1,
                    metadata TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (org_id) REFERENCES organizations(id)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_api_key_hash ON users(api_key_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_org_id ON users(org_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orgs_slug ON organizations(slug)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orgs_stripe ON organizations(stripe_customer_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_org ON usage_events(org_id)")

        logger.info(f"UserStore initialized: {self.db_path}")

    def _migrate_api_key_columns(self, cursor: sqlite3.Cursor) -> None:
        """Migrate schema to add new API key columns if they don't exist."""
        # Check which columns exist
        cursor.execute("PRAGMA table_info(users)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add new columns if missing
        if "api_key_hash" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN api_key_hash TEXT UNIQUE")
            logger.info("Migration: Added api_key_hash column")

        if "api_key_prefix" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN api_key_prefix TEXT")
            logger.info("Migration: Added api_key_prefix column")

        if "api_key_expires_at" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN api_key_expires_at TEXT")
            logger.info("Migration: Added api_key_expires_at column")

    def migrate_plaintext_api_keys(self) -> int:
        """
        Migrate existing plaintext API keys to hashed storage.

        Call this once during deployment to migrate existing keys.
        After migration, plaintext keys will continue to work but will
        be validated against the hash.

        Returns:
            Number of keys migrated
        """
        migrated = 0
        with self._transaction() as cursor:
            # Find users with plaintext keys but no hash
            cursor.execute("""
                SELECT id, api_key, api_key_created_at
                FROM users
                WHERE api_key IS NOT NULL
                  AND api_key_hash IS NULL
            """)

            from datetime import timedelta
            for row in cursor.fetchall():
                user_id = row[0]
                api_key = row[1]
                created_at = row[2]

                # Generate hash from plaintext
                key_hash = hashlib.sha256(api_key.encode()).hexdigest()
                prefix = api_key[:12]

                # Set expiration to 1 year from now for existing keys
                expires_at = (datetime.utcnow() + timedelta(days=365)).isoformat()

                cursor.execute("""
                    UPDATE users
                    SET api_key_hash = ?,
                        api_key_prefix = ?,
                        api_key_expires_at = ?,
                        updated_at = ?
                    WHERE id = ?
                """, (key_hash, prefix, expires_at, datetime.utcnow().isoformat(), user_id))

                migrated += 1
                logger.info(f"Migrated API key for user {user_id}")

        logger.info(f"API key migration complete: {migrated} keys migrated")
        return migrated

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
        """
        Create a new user.

        Args:
            email: User email (must be unique)
            password_hash: Hashed password
            password_salt: Password salt
            name: Display name
            org_id: Organization ID
            role: Role in organization

        Returns:
            Created User object

        Raises:
            ValueError: If email already exists
        """
        user = User(
            email=email,
            password_hash=password_hash,
            password_salt=password_salt,
            name=name,
            org_id=org_id,
            role=role,
        )

        try:
            with self._transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (
                        id, email, password_hash, password_salt, name, org_id, role,
                        is_active, email_verified, api_key, api_key_created_at,
                        created_at, updated_at, last_login_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user.id,
                        user.email,
                        user.password_hash,
                        user.password_salt,
                        user.name,
                        user.org_id,
                        user.role,
                        1 if user.is_active else 0,
                        1 if user.email_verified else 0,
                        user.api_key,
                        user.api_key_created_at.isoformat() if user.api_key_created_at else None,
                        user.created_at.isoformat(),
                        user.updated_at.isoformat(),
                        user.last_login_at.isoformat() if user.last_login_at else None,
                    ),
                )
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: users.email" in str(e):
                raise ValueError(f"Email already exists: {email}")
            raise

        logger.info(f"user_created id={user.id} email={email}")
        return user

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)
        return None

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM users WHERE email = ?", (email.lower(),))
            row = cursor.fetchone()
            if row:
                return self._row_to_user(row)
        return None

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        """
        Get user by API key.

        Supports both hash-based lookup (preferred) and legacy plaintext
        lookup for backward compatibility during migration.

        Args:
            api_key: The plaintext API key

        Returns:
            User if found and key is valid/not expired, None otherwise
        """
        # Compute hash for lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        with self._transaction() as cursor:
            # Try hash-based lookup first (preferred)
            cursor.execute("SELECT * FROM users WHERE api_key_hash = ?", (key_hash,))
            row = cursor.fetchone()

            if row:
                user = self._row_to_user(row)
                # Check expiration
                if user.api_key_expires_at and datetime.utcnow() > user.api_key_expires_at:
                    logger.debug(f"API key expired for user {user.id}")
                    return None
                return user

            # Fall back to legacy plaintext lookup
            cursor.execute("SELECT * FROM users WHERE api_key = ?", (api_key,))
            row = cursor.fetchone()
            if row:
                user = self._row_to_user(row)
                logger.warning(
                    f"Legacy plaintext API key lookup for user {user.id}. "
                    "Run migrate_plaintext_api_keys() to upgrade."
                )
                return user

        return None

    def update_user(self, user_id: str, **fields) -> bool:
        """
        Update user fields.

        Args:
            user_id: User ID
            **fields: Fields to update

        Returns:
            True if user was updated
        """
        if not fields:
            return False

        # Map field names to columns
        column_map = {
            "email": "email",
            "password_hash": "password_hash",
            "password_salt": "password_salt",
            "name": "name",
            "org_id": "org_id",
            "role": "role",
            "is_active": "is_active",
            "email_verified": "email_verified",
            "api_key": "api_key",  # Legacy, kept for migration
            "api_key_hash": "api_key_hash",
            "api_key_prefix": "api_key_prefix",
            "api_key_created_at": "api_key_created_at",
            "api_key_expires_at": "api_key_expires_at",
            "last_login_at": "last_login_at",
        }

        updates = []
        values = []
        for field, value in fields.items():
            if field in column_map:
                updates.append(f"{column_map[field]} = ?")
                if isinstance(value, bool):
                    values.append(1 if value else 0)
                elif isinstance(value, datetime):
                    values.append(value.isoformat())
                else:
                    values.append(value)

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(user_id)

        with self._transaction() as cursor:
            cursor.execute(
                f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            return cursor.rowcount > 0

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cursor.rowcount > 0

    def _row_to_user(self, row: sqlite3.Row) -> User:
        """Convert database row to User object."""
        # Helper to safely get column that may not exist yet
        def safe_get(name: str, default=None):
            try:
                return row[name]
            except (IndexError, KeyError):
                return default

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
            api_key=row["api_key"],  # Legacy field
            api_key_hash=safe_get("api_key_hash"),
            api_key_prefix=safe_get("api_key_prefix"),
            api_key_created_at=datetime.fromisoformat(row["api_key_created_at"])
            if row["api_key_created_at"]
            else None,
            api_key_expires_at=datetime.fromisoformat(safe_get("api_key_expires_at"))
            if safe_get("api_key_expires_at")
            else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_login_at=datetime.fromisoformat(row["last_login_at"])
            if row["last_login_at"]
            else None,
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
        """
        Create a new organization.

        Args:
            name: Organization name
            owner_id: User ID of owner
            slug: URL-friendly slug (auto-generated if not provided)
            tier: Subscription tier

        Returns:
            Created Organization object
        """
        if slug is None:
            slug = name.lower().replace(" ", "-").replace("_", "-")
            # Make unique by appending random suffix if needed
            import secrets
            base_slug = slug
            for _ in range(10):
                with self._transaction() as cursor:
                    cursor.execute("SELECT 1 FROM organizations WHERE slug = ?", (slug,))
                    if not cursor.fetchone():
                        break
                    slug = f"{base_slug}-{secrets.token_hex(4)}"

        org = Organization(
            name=name,
            slug=slug,
            tier=tier,
            owner_id=owner_id,
        )

        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO organizations (
                    id, name, slug, tier, owner_id, stripe_customer_id,
                    stripe_subscription_id, debates_used_this_month,
                    billing_cycle_start, settings, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    org.id,
                    org.name,
                    org.slug,
                    org.tier.value,
                    org.owner_id,
                    org.stripe_customer_id,
                    org.stripe_subscription_id,
                    org.debates_used_this_month,
                    org.billing_cycle_start.isoformat(),
                    json.dumps(org.settings),
                    org.created_at.isoformat(),
                    org.updated_at.isoformat(),
                ),
            )

        # Update owner's org_id
        self.update_user(owner_id, org_id=org.id, role="owner")

        logger.info(f"organization_created id={org.id} name={name} owner={owner_id}")
        return org

    def get_organization_by_id(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM organizations WHERE id = ?", (org_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_org(row)
        return None

    def get_organization_by_slug(self, slug: str) -> Optional[Organization]:
        """Get organization by slug."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM organizations WHERE slug = ?", (slug,))
            row = cursor.fetchone()
            if row:
                return self._row_to_org(row)
        return None

    def get_organization_by_stripe_customer(self, stripe_customer_id: str) -> Optional[Organization]:
        """Get organization by Stripe customer ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM organizations WHERE stripe_customer_id = ?",
                (stripe_customer_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_org(row)
        return None

    def get_organization_by_subscription(self, subscription_id: str) -> Optional[Organization]:
        """Get organization by Stripe subscription ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM organizations WHERE stripe_subscription_id = ?",
                (subscription_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_org(row)
        return None

    def reset_org_usage(self, org_id: str) -> bool:
        """Reset monthly usage for a single organization."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE organizations
                SET debates_used_this_month = 0,
                    billing_cycle_start = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), datetime.utcnow().isoformat(), org_id),
            )
            return cursor.rowcount > 0

    def update_organization(self, org_id: str, **fields) -> bool:
        """Update organization fields."""
        if not fields:
            return False

        column_map = {
            "name": "name",
            "slug": "slug",
            "tier": "tier",
            "owner_id": "owner_id",
            "stripe_customer_id": "stripe_customer_id",
            "stripe_subscription_id": "stripe_subscription_id",
            "debates_used_this_month": "debates_used_this_month",
            "billing_cycle_start": "billing_cycle_start",
            "settings": "settings",
        }

        updates = []
        values = []
        for field, value in fields.items():
            if field in column_map:
                updates.append(f"{column_map[field]} = ?")
                if field == "tier" and isinstance(value, SubscriptionTier):
                    values.append(value.value)
                elif field == "settings" and isinstance(value, dict):
                    values.append(json.dumps(value))
                elif isinstance(value, datetime):
                    values.append(value.isoformat())
                else:
                    values.append(value)

        if not updates:
            return False

        updates.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.append(org_id)

        with self._transaction() as cursor:
            cursor.execute(
                f"UPDATE organizations SET {', '.join(updates)} WHERE id = ?",
                values,
            )
            return cursor.rowcount > 0

    def add_user_to_org(self, user_id: str, org_id: str, role: str = "member") -> bool:
        """Add user to organization."""
        return self.update_user(user_id, org_id=org_id, role=role)

    def remove_user_from_org(self, user_id: str) -> bool:
        """Remove user from organization."""
        return self.update_user(user_id, org_id=None, role="member")

    def get_org_members(self, org_id: str) -> list[User]:
        """Get all members of an organization."""
        with self._transaction() as cursor:
            cursor.execute("SELECT * FROM users WHERE org_id = ?", (org_id,))
            return [self._row_to_user(row) for row in cursor.fetchall()]

    def _row_to_org(self, row: sqlite3.Row) -> Organization:
        """Convert database row to Organization object."""
        return Organization(
            id=row["id"],
            name=row["name"],
            slug=row["slug"],
            tier=SubscriptionTier(row["tier"]),
            owner_id=row["owner_id"],
            stripe_customer_id=row["stripe_customer_id"],
            stripe_subscription_id=row["stripe_subscription_id"],
            debates_used_this_month=row["debates_used_this_month"],
            billing_cycle_start=datetime.fromisoformat(row["billing_cycle_start"]),
            settings=json.loads(row["settings"]) if row["settings"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # =========================================================================
    # Usage Tracking
    # =========================================================================

    def increment_usage(self, org_id: str, count: int = 1) -> int:
        """
        Increment debate usage for an organization.

        Args:
            org_id: Organization ID
            count: Number of debates to add

        Returns:
            New total debates used this month
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE organizations
                SET debates_used_this_month = debates_used_this_month + ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (count, datetime.utcnow().isoformat(), org_id),
            )
            cursor.execute(
                "SELECT debates_used_this_month FROM organizations WHERE id = ?",
                (org_id,),
            )
            row = cursor.fetchone()
            return row[0] if row else 0

    def record_usage_event(
        self,
        org_id: str,
        event_type: str,
        count: int = 1,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a usage event for analytics."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO usage_events (org_id, event_type, count, metadata, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    org_id,
                    event_type,
                    count,
                    json.dumps(metadata or {}),
                    datetime.utcnow().isoformat(),
                ),
            )

    def reset_monthly_usage(self) -> int:
        """
        Reset monthly usage for all organizations.

        Returns:
            Number of organizations reset
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE organizations
                SET debates_used_this_month = 0,
                    billing_cycle_start = ?,
                    updated_at = ?
                """,
                (datetime.utcnow().isoformat(), datetime.utcnow().isoformat()),
            )
            return cursor.rowcount

    def get_usage_summary(self, org_id: str) -> dict:
        """Get usage summary for an organization."""
        org = self.get_organization_by_id(org_id)
        if not org:
            return {}

        return {
            "org_id": org_id,
            "tier": org.tier.value,
            "debates_used": org.debates_used_this_month,
            "debates_limit": org.limits.debates_per_month,
            "debates_remaining": org.debates_remaining,
            "is_at_limit": org.is_at_limit,
            "billing_cycle_start": org.billing_cycle_start.isoformat(),
        }

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            del self._local.connection


__all__ = ["UserStore"]
