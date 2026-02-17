"""
External Identity Repository.

Maps external identity providers (Azure AD, Slack, etc.) to Aragora users.
Enables SSO and identity federation across platforms.

Schema:
    CREATE TABLE external_identities (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,           -- Aragora user ID
        provider TEXT NOT NULL,          -- 'azure_ad', 'slack', 'google'
        external_id TEXT NOT NULL,       -- ID from external provider
        tenant_id TEXT,                  -- External tenant/workspace ID
        email TEXT,                      -- Email from external provider
        display_name TEXT,               -- Display name from provider
        raw_claims TEXT,                 -- JSON of raw claims/metadata
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        last_seen_at REAL,
        is_active INTEGER DEFAULT 1,
        UNIQUE(provider, external_id, tenant_id)
    );
"""

from __future__ import annotations

import contextvars
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aragora.config import resolve_db_path

logger = logging.getLogger(__name__)


@dataclass
class ExternalIdentity:
    """Represents a mapping from external identity to Aragora user."""

    id: str
    user_id: str  # Aragora user ID
    provider: str  # azure_ad, slack, google, github, etc.
    external_id: str  # ID from external provider (aadObjectId, etc.)
    tenant_id: str | None = None  # External tenant/workspace ID
    email: str | None = None
    display_name: str | None = None
    raw_claims: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_seen_at: float | None = None
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "provider": self.provider,
            "external_id": self.external_id,
            "tenant_id": self.tenant_id,
            "email": self.email,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "created_at_iso": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "updated_at": self.updated_at,
            "last_seen_at": self.last_seen_at,
            "is_active": self.is_active,
        }


class ExternalIdentityRepository:
    """
    Repository for managing external identity mappings.

    Thread-safe SQLite backend for mapping external identities
    (Azure AD, Slack, etc.) to Aragora users.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS external_identities (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        provider TEXT NOT NULL,
        external_id TEXT NOT NULL,
        tenant_id TEXT,
        email TEXT,
        display_name TEXT,
        raw_claims TEXT,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        last_seen_at REAL,
        is_active INTEGER DEFAULT 1,
        UNIQUE(provider, external_id, tenant_id)
    );

    CREATE INDEX IF NOT EXISTS idx_external_identities_user
        ON external_identities(user_id);

    CREATE INDEX IF NOT EXISTS idx_external_identities_provider
        ON external_identities(provider, external_id);

    CREATE INDEX IF NOT EXISTS idx_external_identities_email
        ON external_identities(email);
    """

    def __init__(self, db_path: str | None = None):
        """Initialize the repository.

        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            db_path = "external_identities.db"

        self._db_path = resolve_db_path(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        # ContextVar for per-async-context connection (async-safe replacement for threading.local)
        self._conn_var: contextvars.ContextVar[sqlite3.Connection | None] = contextvars.ContextVar(
            f"externalidentityrepo_conn_{id(self)}", default=None
        )
        # Track all connections for proper cleanup
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._init_lock = threading.Lock()
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get per-context database connection (async-safe)."""
        conn = self._conn_var.get()
        if conn is None:
            db_dir = Path(self._db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            self._conn_var.set(conn)
            with self._connections_lock:
                self._connections.add(conn)

        return conn

    def close(self) -> None:
        """Close all database connections."""
        with self._connections_lock:
            for conn in self._connections:
                try:
                    conn.close()
                except (OSError, RuntimeError, ValueError) as e:
                    logger.debug("Error closing connection: %s", e)
                    pass
            self._connections.clear()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Ensure database schema exists."""
        with self._init_lock:
            if not self._initialized:
                conn.executescript(self.SCHEMA)
                conn.commit()
                self._initialized = True

    def _generate_id(self) -> str:
        """Generate a unique ID."""
        import uuid

        return f"ext-{uuid.uuid4().hex[:12]}"

    def create(self, identity: ExternalIdentity) -> ExternalIdentity:
        """Create a new external identity mapping.

        Args:
            identity: ExternalIdentity to create

        Returns:
            Created ExternalIdentity with ID
        """
        conn = self._get_connection()

        if not identity.id:
            identity.id = self._generate_id()

        now = time.time()
        identity.created_at = now
        identity.updated_at = now

        conn.execute(
            """
            INSERT INTO external_identities
            (id, user_id, provider, external_id, tenant_id, email,
             display_name, raw_claims, created_at, updated_at,
             last_seen_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                identity.id,
                identity.user_id,
                identity.provider,
                identity.external_id,
                identity.tenant_id,
                identity.email,
                identity.display_name,
                json.dumps(identity.raw_claims),
                identity.created_at,
                identity.updated_at,
                identity.last_seen_at,
                1 if identity.is_active else 0,
            ),
        )
        conn.commit()

        logger.info(
            f"Created external identity: {identity.provider}/{identity.external_id} -> {identity.user_id}"
        )
        return identity

    def get(self, identity_id: str) -> ExternalIdentity | None:
        """Get an external identity by ID."""
        conn = self._get_connection()

        cursor = conn.execute(
            "SELECT * FROM external_identities WHERE id = ?",
            (identity_id,),
        )
        row = cursor.fetchone()

        if row:
            return self._row_to_identity(row)
        return None

    def get_by_external_id(
        self, provider: str, external_id: str, tenant_id: str | None = None
    ) -> ExternalIdentity | None:
        """Get identity by external provider ID.

        Args:
            provider: Provider name (azure_ad, slack, etc.)
            external_id: ID from external provider
            tenant_id: Optional tenant/workspace ID

        Returns:
            ExternalIdentity if found
        """
        conn = self._get_connection()

        if tenant_id:
            cursor = conn.execute(
                """
                SELECT * FROM external_identities
                WHERE provider = ? AND external_id = ? AND tenant_id = ?
                    AND is_active = 1
                """,
                (provider, external_id, tenant_id),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM external_identities
                WHERE provider = ? AND external_id = ? AND is_active = 1
                """,
                (provider, external_id),
            )

        row = cursor.fetchone()
        if row:
            return self._row_to_identity(row)
        return None

    def get_by_user_id(self, user_id: str, provider: str | None = None) -> list[ExternalIdentity]:
        """Get all external identities for a user.

        Args:
            user_id: Aragora user ID
            provider: Optional filter by provider

        Returns:
            List of external identities
        """
        conn = self._get_connection()

        if provider:
            cursor = conn.execute(
                """
                SELECT * FROM external_identities
                WHERE user_id = ? AND provider = ? AND is_active = 1
                ORDER BY created_at DESC
                """,
                (user_id, provider),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM external_identities
                WHERE user_id = ? AND is_active = 1
                ORDER BY created_at DESC
                """,
                (user_id,),
            )

        return [self._row_to_identity(row) for row in cursor.fetchall()]

    def get_by_email(self, email: str, provider: str | None = None) -> list[ExternalIdentity]:
        """Get external identities by email.

        Args:
            email: Email address
            provider: Optional filter by provider

        Returns:
            List of matching external identities
        """
        conn = self._get_connection()

        if provider:
            cursor = conn.execute(
                """
                SELECT * FROM external_identities
                WHERE email = ? AND provider = ? AND is_active = 1
                ORDER BY created_at DESC
                """,
                (email.lower(), provider),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM external_identities
                WHERE email = ? AND is_active = 1
                ORDER BY created_at DESC
                """,
                (email.lower(),),
            )

        return [self._row_to_identity(row) for row in cursor.fetchall()]

    def update(self, identity: ExternalIdentity) -> bool:
        """Update an external identity.

        Args:
            identity: ExternalIdentity with updated fields

        Returns:
            True if updated
        """
        conn = self._get_connection()

        identity.updated_at = time.time()

        result = conn.execute(
            """
            UPDATE external_identities
            SET user_id = ?, email = ?, display_name = ?,
                raw_claims = ?, updated_at = ?, last_seen_at = ?,
                is_active = ?
            WHERE id = ?
            """,
            (
                identity.user_id,
                identity.email,
                identity.display_name,
                json.dumps(identity.raw_claims),
                identity.updated_at,
                identity.last_seen_at,
                1 if identity.is_active else 0,
                identity.id,
            ),
        )
        conn.commit()

        return result.rowcount > 0

    def update_last_seen(self, identity_id: str) -> bool:
        """Update the last_seen_at timestamp."""
        conn = self._get_connection()

        now = time.time()
        result = conn.execute(
            """
            UPDATE external_identities
            SET last_seen_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (now, now, identity_id),
        )
        conn.commit()

        return result.rowcount > 0

    def deactivate(self, identity_id: str) -> bool:
        """Deactivate an external identity."""
        conn = self._get_connection()

        result = conn.execute(
            """
            UPDATE external_identities
            SET is_active = 0, updated_at = ?
            WHERE id = ?
            """,
            (time.time(), identity_id),
        )
        conn.commit()

        return result.rowcount > 0

    def delete(self, identity_id: str) -> bool:
        """Permanently delete an external identity."""
        conn = self._get_connection()

        result = conn.execute(
            "DELETE FROM external_identities WHERE id = ?",
            (identity_id,),
        )
        conn.commit()

        return result.rowcount > 0

    def link_or_update(
        self,
        user_id: str,
        provider: str,
        external_id: str,
        tenant_id: str | None = None,
        email: str | None = None,
        display_name: str | None = None,
        raw_claims: dict[str, Any] | None = None,
    ) -> ExternalIdentity:
        """Link an external identity to a user, updating if exists.

        This is the primary method for syncing external identities.
        Creates a new mapping if one doesn't exist, updates if it does.

        Args:
            user_id: Aragora user ID
            provider: Provider name
            external_id: External provider ID
            tenant_id: Optional tenant/workspace ID
            email: Optional email
            display_name: Optional display name
            raw_claims: Optional raw claims/metadata

        Returns:
            Created or updated ExternalIdentity
        """
        existing = self.get_by_external_id(provider, external_id, tenant_id)

        if existing:
            # Update existing mapping
            existing.user_id = user_id
            existing.email = email
            existing.display_name = display_name
            existing.raw_claims = raw_claims or existing.raw_claims
            existing.last_seen_at = time.time()
            existing.is_active = True
            self.update(existing)
            return existing
        else:
            # Create new mapping
            identity = ExternalIdentity(
                id="",
                user_id=user_id,
                provider=provider,
                external_id=external_id,
                tenant_id=tenant_id,
                email=email,
                display_name=display_name,
                raw_claims=raw_claims or {},
                last_seen_at=time.time(),
            )
            return self.create(identity)

    def _row_to_identity(self, row: sqlite3.Row) -> ExternalIdentity:
        """Convert database row to ExternalIdentity."""
        raw_claims = {}
        if row["raw_claims"]:
            try:
                raw_claims = json.loads(row["raw_claims"])
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSON data: %s", e)

        return ExternalIdentity(
            id=row["id"],
            user_id=row["user_id"],
            provider=row["provider"],
            external_id=row["external_id"],
            tenant_id=row["tenant_id"],
            email=row["email"],
            display_name=row["display_name"],
            raw_claims=raw_claims,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            last_seen_at=row["last_seen_at"],
            is_active=bool(row["is_active"]),
        )


# Singleton instance
_repository: ExternalIdentityRepository | None = None


def get_external_identity_repository(
    db_path: str | None = None,
) -> ExternalIdentityRepository:
    """Get or create the external identity repository singleton."""
    global _repository
    if _repository is None:
        _repository = ExternalIdentityRepository(db_path)
    return _repository
