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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, Optional

from aragora.billing.models import Organization, OrganizationInvitation, SubscriptionTier, User

logger = logging.getLogger(__name__)


class UserStore:
    """
    SQLite-backed storage for users and organizations.

    Thread-safe with connection pooling via thread-local storage.
    Uses WAL mode for better read/write concurrency.
    """

    # Class-level lock for schema migrations to prevent race conditions
    _schema_lock = threading.Lock()

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
        """Get thread-local database connection.

        Note: check_same_thread=False is safe here because we store connections
        in thread-local storage, ensuring each thread uses its own connection.
        """
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
        except Exception as e:
            conn.rollback()
            logger.debug("Transaction rolled back due to: %s", e)
            raise

    def _init_schema(self) -> None:
        """Initialize database schema with migration support."""
        # Use class-level lock to prevent concurrent schema modifications
        with self._schema_lock, self._transaction() as cursor:
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

            # OAuth providers table (for SSO)
            cursor.execute("""
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
            """)

            # Audit log table (for billing/subscription changes)
            cursor.execute("""
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
            """)

            # Organization invitations table
            cursor.execute("""
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
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)")
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_api_key_hash ON users(api_key_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_org_id ON users(org_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orgs_slug ON organizations(slug)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orgs_stripe ON organizations(stripe_customer_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_org ON usage_events(org_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_oauth_user ON oauth_providers(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_oauth_provider ON oauth_providers(provider, provider_user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_org ON audit_log(org_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_invitations_org ON org_invitations(org_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_invitations_email ON org_invitations(email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_invitations_token ON org_invitations(token)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_invitations_status ON org_invitations(status)")

        logger.info(f"UserStore initialized: {self.db_path}")

    def _migrate_api_key_columns(self, cursor: sqlite3.Cursor) -> None:
        """Migrate schema to add new API key columns if they don't exist."""
        # Check which columns exist
        cursor.execute("PRAGMA table_info(users)")
        existing_columns = {row[1] for row in cursor.fetchall()}

        # Add new columns if missing
        # Note: SQLite doesn't support ALTER TABLE ADD COLUMN with UNIQUE constraint
        # when the table has data. We add the column without constraint, then create
        # a unique index separately (done in _init_schema after migrations).
        if "api_key_hash" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN api_key_hash TEXT")
            logger.info("Migration: Added api_key_hash column")

        if "api_key_prefix" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN api_key_prefix TEXT")
            logger.info("Migration: Added api_key_prefix column")

        if "api_key_expires_at" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN api_key_expires_at TEXT")
            logger.info("Migration: Added api_key_expires_at column")

        # MFA columns
        if "mfa_secret" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN mfa_secret TEXT")
            logger.info("Migration: Added mfa_secret column")

        if "mfa_enabled" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN mfa_enabled INTEGER DEFAULT 0")
            logger.info("Migration: Added mfa_enabled column")

        if "mfa_backup_codes" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN mfa_backup_codes TEXT")
            logger.info("Migration: Added mfa_backup_codes column")

        # Token revocation support
        if "token_version" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN token_version INTEGER DEFAULT 1")
            logger.info("Migration: Added token_version column")

        # Account lockout columns
        if "failed_login_attempts" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0")
            logger.info("Migration: Added failed_login_attempts column")

        if "lockout_until" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN lockout_until TEXT")
            logger.info("Migration: Added lockout_until column")

        if "last_failed_login_at" not in existing_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN last_failed_login_at TEXT")
            logger.info("Migration: Added last_failed_login_at column")

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
            "mfa_secret": "mfa_secret",
            "mfa_enabled": "mfa_enabled",
            "mfa_backup_codes": "mfa_backup_codes",
        }

        updates: list[str] = []
        values: list[Any] = []
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

    def increment_token_version(self, user_id: str) -> int:
        """
        Increment a user's token version, invalidating all existing tokens.

        This is used for "logout all devices" functionality. When the token
        version is incremented, all existing JWT tokens (which contain the
        old version) will fail validation.

        Args:
            user_id: User ID to increment token version for

        Returns:
            The new token version, or 0 if user not found
        """
        with self._transaction() as cursor:
            # Increment and get new version in one query
            cursor.execute(
                """
                UPDATE users
                SET token_version = COALESCE(token_version, 1) + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), user_id),
            )

            if cursor.rowcount == 0:
                logger.warning(f"increment_token_version: user {user_id} not found")
                return 0

            # Get the new version
            cursor.execute(
                "SELECT token_version FROM users WHERE id = ?",
                (user_id,),
            )
            row = cursor.fetchone()
            new_version = row[0] if row else 1

            logger.info(f"token_version_incremented user_id={user_id} new_version={new_version}")
            return new_version

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
            mfa_secret=safe_get("mfa_secret"),
            mfa_enabled=bool(safe_get("mfa_enabled", 0)),
            mfa_backup_codes=safe_get("mfa_backup_codes"),
            token_version=safe_get("token_version", 1) or 1,
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
    # Organization Invitations
    # =========================================================================

    def create_invitation(self, invitation: "OrganizationInvitation") -> bool:
        """
        Create a new organization invitation.

        Args:
            invitation: OrganizationInvitation instance

        Returns:
            True if created successfully
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO org_invitations
                (id, org_id, email, role, token, invited_by, status, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    invitation.id,
                    invitation.org_id,
                    invitation.email.lower(),
                    invitation.role,
                    invitation.token,
                    invitation.invited_by,
                    invitation.status,
                    invitation.created_at.isoformat(),
                    invitation.expires_at.isoformat(),
                ),
            )
        return True

    def get_invitation_by_id(self, invitation_id: str) -> Optional["OrganizationInvitation"]:
        """Get invitation by ID."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM org_invitations WHERE id = ?",
                (invitation_id,),
            )
            row = cursor.fetchone()
            return self._row_to_invitation(row) if row else None

    def get_invitation_by_token(self, token: str) -> Optional["OrganizationInvitation"]:
        """Get invitation by token."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM org_invitations WHERE token = ?",
                (token,),
            )
            row = cursor.fetchone()
            return self._row_to_invitation(row) if row else None

    def get_invitation_by_email(
        self, org_id: str, email: str
    ) -> Optional["OrganizationInvitation"]:
        """Get pending invitation by org and email."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT * FROM org_invitations
                WHERE org_id = ? AND email = ? AND status = 'pending'
                ORDER BY created_at DESC LIMIT 1
                """,
                (org_id, email.lower()),
            )
            row = cursor.fetchone()
            return self._row_to_invitation(row) if row else None

    def get_invitations_for_org(self, org_id: str) -> list["OrganizationInvitation"]:
        """Get all invitations for an organization."""
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT * FROM org_invitations WHERE org_id = ? ORDER BY created_at DESC",
                (org_id,),
            )
            return [self._row_to_invitation(row) for row in cursor.fetchall()]

    def get_pending_invitations_by_email(self, email: str) -> list["OrganizationInvitation"]:
        """Get all pending invitations for an email address."""
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT * FROM org_invitations
                WHERE email = ? AND status = 'pending'
                ORDER BY created_at DESC
                """,
                (email.lower(),),
            )
            return [self._row_to_invitation(row) for row in cursor.fetchall()]

    def update_invitation_status(
        self,
        invitation_id: str,
        status: str,
        accepted_at: Optional[datetime] = None,
    ) -> bool:
        """
        Update invitation status.

        Args:
            invitation_id: Invitation ID
            status: New status (pending, accepted, expired, revoked)
            accepted_at: Timestamp when accepted (for accepted status)

        Returns:
            True if updated
        """
        with self._transaction() as cursor:
            if accepted_at:
                cursor.execute(
                    """
                    UPDATE org_invitations
                    SET status = ?, accepted_at = ?
                    WHERE id = ?
                    """,
                    (status, accepted_at.isoformat(), invitation_id),
                )
            else:
                cursor.execute(
                    "UPDATE org_invitations SET status = ? WHERE id = ?",
                    (status, invitation_id),
                )
            return cursor.rowcount > 0

    def delete_invitation(self, invitation_id: str) -> bool:
        """Delete an invitation."""
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM org_invitations WHERE id = ?",
                (invitation_id,),
            )
            return cursor.rowcount > 0

    def cleanup_expired_invitations(self) -> int:
        """
        Mark expired invitations as expired.

        Returns:
            Number of invitations marked as expired
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE org_invitations
                SET status = 'expired'
                WHERE status = 'pending' AND expires_at < ?
                """,
                (datetime.utcnow().isoformat(),),
            )
            return cursor.rowcount

    def _row_to_invitation(self, row: sqlite3.Row) -> "OrganizationInvitation":
        """Convert database row to OrganizationInvitation object."""
        from aragora.billing.models import OrganizationInvitation

        return OrganizationInvitation(
            id=row["id"],
            org_id=row["org_id"],
            email=row["email"],
            role=row["role"],
            token=row["token"],
            invited_by=row["invited_by"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            accepted_at=datetime.fromisoformat(row["accepted_at"]) if row["accepted_at"] else None,
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
        """
        Link an OAuth provider to a user account.

        Args:
            user_id: User ID to link to
            provider: OAuth provider name (e.g., 'google', 'github')
            provider_user_id: User ID from the OAuth provider
            email: Email from OAuth provider (optional)

        Returns:
            True if linked successfully
        """
        try:
            with self._transaction() as cursor:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO oauth_providers
                    (user_id, provider, provider_user_id, email, linked_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        provider.lower(),
                        provider_user_id,
                        email,
                        datetime.utcnow().isoformat(),
                    ),
                )
            logger.info(f"OAuth linked: user={user_id} provider={provider}")
            return True
        except Exception as e:
            logger.error(f"Failed to link OAuth: {e}")
            return False

    def unlink_oauth_provider(self, user_id: str, provider: str) -> bool:
        """
        Unlink an OAuth provider from a user account.

        Args:
            user_id: User ID to unlink from
            provider: OAuth provider name

        Returns:
            True if unlinked successfully
        """
        with self._transaction() as cursor:
            cursor.execute(
                "DELETE FROM oauth_providers WHERE user_id = ? AND provider = ?",
                (user_id, provider.lower()),
            )
            if cursor.rowcount > 0:
                logger.info(f"OAuth unlinked: user={user_id} provider={provider}")
                return True
        return False

    def get_user_by_oauth(self, provider: str, provider_user_id: str) -> Optional[User]:
        """
        Get user by OAuth provider ID.

        Args:
            provider: OAuth provider name
            provider_user_id: User ID from the OAuth provider

        Returns:
            User if found, None otherwise
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT user_id FROM oauth_providers
                WHERE provider = ? AND provider_user_id = ?
                """,
                (provider.lower(), provider_user_id),
            )
            row = cursor.fetchone()
            if row:
                return self.get_user_by_id(row[0])
        return None

    def get_user_oauth_providers(self, user_id: str) -> list[dict]:
        """
        Get all OAuth providers linked to a user.

        Args:
            user_id: User ID

        Returns:
            List of linked providers with their details
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT provider, provider_user_id, email, linked_at
                FROM oauth_providers
                WHERE user_id = ?
                """,
                (user_id,),
            )
            return [
                {
                    "provider": row[0],
                    "provider_user_id": row[1],
                    "email": row[2],
                    "linked_at": row[3],
                }
                for row in cursor.fetchall()
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
        """
        Log an audit event.

        Args:
            action: Action performed (e.g., 'subscription.created', 'tier.changed')
            resource_type: Type of resource (e.g., 'subscription', 'user', 'organization')
            resource_id: ID of the affected resource
            user_id: User who performed the action
            org_id: Organization context
            old_value: Previous value (for changes)
            new_value: New value (for changes)
            metadata: Additional context
            ip_address: Client IP address
            user_agent: Client user agent

        Returns:
            Audit log entry ID
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO audit_log
                (timestamp, user_id, org_id, action, resource_type, resource_id,
                 old_value, new_value, metadata, ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    user_id,
                    org_id,
                    action,
                    resource_type,
                    resource_id,
                    json.dumps(old_value) if old_value else None,
                    json.dumps(new_value) if new_value else None,
                    json.dumps(metadata or {}),
                    ip_address,
                    user_agent,
                ),
            )
            return cursor.lastrowid

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
        """
        Query audit log entries.

        Args:
            org_id: Filter by organization
            user_id: Filter by user
            action: Filter by action (supports prefix match with *)
            resource_type: Filter by resource type
            since: Filter entries after this time
            until: Filter entries before this time
            limit: Maximum entries to return
            offset: Pagination offset

        Returns:
            List of audit log entries
        """
        conditions: list[str] = []
        params: list[Any] = []

        if org_id:
            conditions.append("org_id = ?")
            params.append(org_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            if action.endswith("*"):
                conditions.append("action LIKE ?")
                params.append(action[:-1] + "%")
            else:
                conditions.append("action = ?")
                params.append(action)
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if until:
            conditions.append("timestamp <= ?")
            params.append(until.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        with self._transaction() as cursor:
            cursor.execute(
                f"""
                SELECT id, timestamp, user_id, org_id, action, resource_type,
                       resource_id, old_value, new_value, metadata, ip_address, user_agent
                FROM audit_log
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
                """,
                params,
            )
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "user_id": row[2],
                    "org_id": row[3],
                    "action": row[4],
                    "resource_type": row[5],
                    "resource_id": row[6],
                    "old_value": json.loads(row[7]) if row[7] else None,
                    "new_value": json.loads(row[8]) if row[8] else None,
                    "metadata": json.loads(row[9]) if row[9] else {},
                    "ip_address": row[10],
                    "user_agent": row[11],
                }
                for row in cursor.fetchall()
            ]

    def get_audit_log_count(
        self,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
    ) -> int:
        """Get count of audit log entries matching filters."""
        conditions = []
        params = []

        if org_id:
            conditions.append("org_id = ?")
            params.append(org_id)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if action:
            if action.endswith("*"):
                conditions.append("action LIKE ?")
                params.append(action[:-1] + "%")
            else:
                conditions.append("action = ?")
                params.append(action)
        if resource_type:
            conditions.append("resource_type = ?")
            params.append(resource_type)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with self._transaction() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) FROM audit_log WHERE {where_clause}",
                params,
            )
            return cursor.fetchone()[0]

    # =========================================================================
    # Account Lockout Methods
    # =========================================================================

    # Lockout policy constants
    LOCKOUT_THRESHOLD_1 = 5   # 5 attempts -> 15 min lockout
    LOCKOUT_THRESHOLD_2 = 10  # 10 attempts -> 1 hour lockout
    LOCKOUT_THRESHOLD_3 = 20  # 20 attempts -> 24 hour lockout

    LOCKOUT_DURATION_1 = 15 * 60       # 15 minutes
    LOCKOUT_DURATION_2 = 60 * 60       # 1 hour
    LOCKOUT_DURATION_3 = 24 * 60 * 60  # 24 hours

    def is_account_locked(self, email: str) -> tuple[bool, Optional[datetime], int]:
        """
        Check if an account is currently locked.

        Args:
            email: User's email address

        Returns:
            Tuple of (is_locked, lockout_until, failed_attempts)
        """
        with self._transaction() as cursor:
            cursor.execute(
                """
                SELECT failed_login_attempts, lockout_until
                FROM users
                WHERE email = ?
                """,
                (email,),
            )
            row = cursor.fetchone()

            if not row:
                return False, None, 0

            failed_attempts = row[0] or 0
            lockout_until_str = row[1]

            if not lockout_until_str:
                return False, None, failed_attempts

            lockout_until = datetime.fromisoformat(lockout_until_str)
            now = datetime.now()

            if now < lockout_until:
                return True, lockout_until, failed_attempts
            else:
                # Lockout expired
                return False, None, failed_attempts

    def record_failed_login(self, email: str) -> tuple[int, Optional[datetime]]:
        """
        Record a failed login attempt and potentially lock the account.

        Args:
            email: User's email address

        Returns:
            Tuple of (new_attempt_count, lockout_until_if_locked)
        """
        now = datetime.now()

        with self._transaction() as cursor:
            # Increment failed attempts
            cursor.execute(
                """
                UPDATE users
                SET failed_login_attempts = COALESCE(failed_login_attempts, 0) + 1,
                    last_failed_login_at = ?,
                    updated_at = ?
                WHERE email = ?
                """,
                (now.isoformat(), now.isoformat(), email),
            )

            # Get new count
            cursor.execute(
                "SELECT failed_login_attempts FROM users WHERE email = ?",
                (email,),
            )
            row = cursor.fetchone()

            if not row:
                return 0, None

            failed_attempts = row[0]
            lockout_until = None

            # Determine if lockout is needed
            if failed_attempts >= self.LOCKOUT_THRESHOLD_3:
                lockout_until = now + timedelta(seconds=self.LOCKOUT_DURATION_3)
            elif failed_attempts >= self.LOCKOUT_THRESHOLD_2:
                lockout_until = now + timedelta(seconds=self.LOCKOUT_DURATION_2)
            elif failed_attempts >= self.LOCKOUT_THRESHOLD_1:
                lockout_until = now + timedelta(seconds=self.LOCKOUT_DURATION_1)

            if lockout_until:
                cursor.execute(
                    """
                    UPDATE users
                    SET lockout_until = ?
                    WHERE email = ?
                    """,
                    (lockout_until.isoformat(), email),
                )
                logger.warning(
                    f"Account locked: email={email}, attempts={failed_attempts}, "
                    f"locked_until={lockout_until.isoformat()}"
                )

            return failed_attempts, lockout_until

    def reset_failed_login_attempts(self, email: str) -> bool:
        """
        Reset failed login attempts after successful login.

        Args:
            email: User's email address

        Returns:
            True if reset was successful
        """
        now = datetime.now()

        with self._transaction() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET failed_login_attempts = 0,
                    lockout_until = NULL,
                    last_login_at = ?,
                    updated_at = ?
                WHERE email = ?
                """,
                (now.isoformat(), now.isoformat(), email),
            )
            return cursor.rowcount > 0

    def get_lockout_info(self, email: str) -> dict:
        """
        Get detailed lockout information for an account.

        Args:
            email: User's email address

        Returns:
            Dict with lockout details
        """
        is_locked, lockout_until, failed_attempts = self.is_account_locked(email)

        info = {
            "email": email,
            "is_locked": is_locked,
            "failed_attempts": failed_attempts,
            "lockout_until": lockout_until.isoformat() if lockout_until else None,
        }

        # Calculate remaining lockout time
        if is_locked and lockout_until:
            remaining = (lockout_until - datetime.now()).total_seconds()
            info["lockout_remaining_seconds"] = max(0, int(remaining))
            info["lockout_remaining_minutes"] = max(0, int(remaining / 60))

        # Warn if approaching lockout
        if not is_locked:
            if failed_attempts >= self.LOCKOUT_THRESHOLD_1 - 2:
                info["warning"] = f"Account will be locked after {self.LOCKOUT_THRESHOLD_1 - failed_attempts} more failed attempts"

        return info


__all__ = ["UserStore"]
