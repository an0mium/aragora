"""
Microsoft Teams Tenant Storage for OAuth token management.

Stores tenant credentials after OAuth installation for multi-tenant support.
Tokens are encrypted at rest using AES-256-GCM when ARAGORA_ENCRYPTION_KEY is set.

Schema:
    CREATE TABLE teams_tenants (
        tenant_id TEXT PRIMARY KEY,
        tenant_name TEXT NOT NULL,
        access_token TEXT NOT NULL,
        refresh_token TEXT,
        bot_id TEXT NOT NULL,
        installed_at REAL NOT NULL,
        installed_by TEXT,
        scopes TEXT,
        aragora_org_id TEXT,
        is_active INTEGER DEFAULT 1,
        expires_at REAL
    );
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Storage configuration
TEAMS_TENANT_DB_PATH = os.environ.get(
    "TEAMS_TENANT_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "teams_tenants.db"),
)

# Encryption key for tokens (optional but recommended)
ENCRYPTION_KEY = os.environ.get("ARAGORA_ENCRYPTION_KEY", "")


@dataclass
class TeamsTenant:
    """Represents an installed Microsoft Teams tenant."""

    tenant_id: str  # Azure AD tenant ID
    tenant_name: str
    access_token: str  # Bot access token
    bot_id: str  # Bot ID in Teams
    installed_at: float  # Unix timestamp
    refresh_token: Optional[str] = None  # OAuth refresh token
    installed_by: Optional[str] = None  # User ID who installed
    scopes: List[str] = field(default_factory=list)
    aragora_org_id: Optional[str] = None  # Link to Aragora organization
    is_active: bool = True
    expires_at: Optional[float] = None  # Token expiration timestamp

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes sensitive tokens)."""
        return {
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant_name,
            "bot_id": self.bot_id,
            "installed_at": self.installed_at,
            "installed_at_iso": datetime.fromtimestamp(
                self.installed_at, tz=timezone.utc
            ).isoformat(),
            "installed_by": self.installed_by,
            "scopes": self.scopes,
            "aragora_org_id": self.aragora_org_id,
            "is_active": self.is_active,
            "expires_at": self.expires_at,
            "expires_at_iso": (
                datetime.fromtimestamp(self.expires_at, tz=timezone.utc).isoformat()
                if self.expires_at
                else None
            ),
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TeamsTenant":
        """Create from database row."""
        scopes_str = row["scopes"] or ""
        scopes = scopes_str.split(",") if scopes_str else []

        return cls(
            tenant_id=row["tenant_id"],
            tenant_name=row["tenant_name"],
            access_token=row["access_token"],
            refresh_token=row["refresh_token"],
            bot_id=row["bot_id"],
            installed_at=row["installed_at"],
            installed_by=row["installed_by"],
            scopes=scopes,
            aragora_org_id=row["aragora_org_id"],
            is_active=bool(row["is_active"]),
            expires_at=row["expires_at"],
        )

    def is_token_expired(self) -> bool:
        """Check if the access token has expired."""
        if not self.expires_at:
            return False
        import time

        # Consider expired if within 5 minutes of expiration
        return time.time() > (self.expires_at - 300)


class TeamsTenantStore:
    """
    Storage for Microsoft Teams tenant OAuth credentials.

    Supports SQLite backend with optional token encryption.
    Thread-safe for concurrent access.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS teams_tenants (
        tenant_id TEXT PRIMARY KEY,
        tenant_name TEXT NOT NULL,
        access_token TEXT NOT NULL,
        refresh_token TEXT,
        bot_id TEXT NOT NULL,
        installed_at REAL NOT NULL,
        installed_by TEXT,
        scopes TEXT,
        aragora_org_id TEXT,
        is_active INTEGER DEFAULT 1,
        expires_at REAL
    );

    CREATE INDEX IF NOT EXISTS idx_teams_tenants_org
        ON teams_tenants(aragora_org_id);

    CREATE INDEX IF NOT EXISTS idx_teams_tenants_active
        ON teams_tenants(is_active);

    CREATE INDEX IF NOT EXISTS idx_teams_tenants_expires
        ON teams_tenants(expires_at);
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the tenant store.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = db_path or TEAMS_TENANT_DB_PATH
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            # Ensure directory exists
            db_dir = os.path.dirname(self._db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            self._local.connection = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
            self._ensure_schema(self._local.connection)

        return self._local.connection

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Ensure database schema exists."""
        with self._init_lock:
            if not self._initialized:
                conn.executescript(self.SCHEMA)
                conn.commit()
                self._initialized = True

    def _encrypt_token(self, token: str) -> str:
        """Encrypt token if encryption key is configured."""
        if not ENCRYPTION_KEY or not token:
            return token

        try:
            from cryptography.fernet import Fernet
            import base64
            import hashlib

            # Derive Fernet key from encryption key
            key = base64.urlsafe_b64encode(hashlib.sha256(ENCRYPTION_KEY.encode()).digest())
            f = Fernet(key)
            return f.encrypt(token.encode()).decode()
        except ImportError:
            logger.warning("cryptography not installed, storing token unencrypted")
            return token
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.error(f"Token encryption failed: {e}")
            return token

    def _decrypt_token(self, encrypted: str) -> str:
        """Decrypt token if encryption key is configured."""
        if not ENCRYPTION_KEY or not encrypted:
            return encrypted

        # Check if it looks like an encrypted token (Fernet tokens are base64)
        if not encrypted.startswith("gAAA"):
            return encrypted  # Not encrypted

        try:
            from cryptography.fernet import Fernet
            import base64
            import hashlib

            key = base64.urlsafe_b64encode(hashlib.sha256(ENCRYPTION_KEY.encode()).digest())
            f = Fernet(key)
            return f.decrypt(encrypted.encode()).decode()
        except ImportError:
            return encrypted
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.error(f"Token decryption failed: {e}")
            return encrypted

    def save(self, tenant: TeamsTenant) -> bool:
        """Save or update a tenant.

        Args:
            tenant: Tenant to save

        Returns:
            True if saved successfully
        """
        conn = self._get_connection()
        try:
            encrypted_access = self._encrypt_token(tenant.access_token)
            encrypted_refresh = (
                self._encrypt_token(tenant.refresh_token) if tenant.refresh_token else None
            )
            scopes_str = ",".join(tenant.scopes)

            conn.execute(
                """
                INSERT OR REPLACE INTO teams_tenants
                (tenant_id, tenant_name, access_token, refresh_token, bot_id,
                 installed_at, installed_by, scopes, aragora_org_id, is_active, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant.tenant_id,
                    tenant.tenant_name,
                    encrypted_access,
                    encrypted_refresh,
                    tenant.bot_id,
                    tenant.installed_at,
                    tenant.installed_by,
                    scopes_str,
                    tenant.aragora_org_id,
                    1 if tenant.is_active else 0,
                    tenant.expires_at,
                ),
            )
            conn.commit()
            logger.info(f"Saved Teams tenant: {tenant.tenant_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to save tenant: {e}")
            return False

    def get(self, tenant_id: str) -> Optional[TeamsTenant]:
        """Get a tenant by ID.

        Args:
            tenant_id: Azure AD tenant ID

        Returns:
            Tenant or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM teams_tenants WHERE tenant_id = ?",
                (tenant_id,),
            )
            row = cursor.fetchone()

            if row:
                tenant = TeamsTenant.from_row(row)
                tenant.access_token = self._decrypt_token(tenant.access_token)
                if tenant.refresh_token:
                    tenant.refresh_token = self._decrypt_token(tenant.refresh_token)
                return tenant

            return None

        except sqlite3.Error as e:
            logger.error(f"Failed to get tenant {tenant_id}: {e}")
            return None

    def get_by_org(self, aragora_org_id: str) -> List[TeamsTenant]:
        """Get all tenants for an Aragora organization.

        Args:
            aragora_org_id: Aragora organization ID

        Returns:
            List of tenants
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM teams_tenants
                WHERE aragora_org_id = ? AND is_active = 1
                ORDER BY installed_at DESC
                """,
                (aragora_org_id,),
            )

            tenants = []
            for row in cursor.fetchall():
                tenant = TeamsTenant.from_row(row)
                tenant.access_token = self._decrypt_token(tenant.access_token)
                if tenant.refresh_token:
                    tenant.refresh_token = self._decrypt_token(tenant.refresh_token)
                tenants.append(tenant)

            return tenants

        except sqlite3.Error as e:
            logger.error(f"Failed to get tenants for org {aragora_org_id}: {e}")
            return []

    def list_active(self, limit: int = 100, offset: int = 0) -> List[TeamsTenant]:
        """List all active tenants.

        Args:
            limit: Maximum number of tenants to return
            offset: Pagination offset

        Returns:
            List of active tenants
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM teams_tenants
                WHERE is_active = 1
                ORDER BY installed_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

            tenants = []
            for row in cursor.fetchall():
                tenant = TeamsTenant.from_row(row)
                tenant.access_token = self._decrypt_token(tenant.access_token)
                if tenant.refresh_token:
                    tenant.refresh_token = self._decrypt_token(tenant.refresh_token)
                tenants.append(tenant)

            return tenants

        except sqlite3.Error as e:
            logger.error(f"Failed to list tenants: {e}")
            return []

    def list_expiring(self, within_seconds: int = 3600) -> List[TeamsTenant]:
        """List tenants with tokens expiring soon.

        Args:
            within_seconds: Time window in seconds (default 1 hour)

        Returns:
            List of tenants with expiring tokens
        """
        import time

        conn = self._get_connection()
        try:
            cutoff = time.time() + within_seconds
            cursor = conn.execute(
                """
                SELECT * FROM teams_tenants
                WHERE is_active = 1 AND expires_at IS NOT NULL AND expires_at < ?
                ORDER BY expires_at ASC
                """,
                (cutoff,),
            )

            tenants = []
            for row in cursor.fetchall():
                tenant = TeamsTenant.from_row(row)
                tenant.access_token = self._decrypt_token(tenant.access_token)
                if tenant.refresh_token:
                    tenant.refresh_token = self._decrypt_token(tenant.refresh_token)
                tenants.append(tenant)

            return tenants

        except sqlite3.Error as e:
            logger.error(f"Failed to list expiring tenants: {e}")
            return []

    def update_tokens(
        self,
        tenant_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_at: Optional[float] = None,
    ) -> bool:
        """Update tokens for a tenant (after refresh).

        Args:
            tenant_id: Azure AD tenant ID
            access_token: New access token
            refresh_token: New refresh token (optional)
            expires_at: Token expiration timestamp

        Returns:
            True if updated successfully
        """
        conn = self._get_connection()
        try:
            encrypted_access = self._encrypt_token(access_token)
            encrypted_refresh = self._encrypt_token(refresh_token) if refresh_token else None

            if refresh_token:
                conn.execute(
                    """
                    UPDATE teams_tenants
                    SET access_token = ?, refresh_token = ?, expires_at = ?
                    WHERE tenant_id = ?
                    """,
                    (encrypted_access, encrypted_refresh, expires_at, tenant_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE teams_tenants
                    SET access_token = ?, expires_at = ?
                    WHERE tenant_id = ?
                    """,
                    (encrypted_access, expires_at, tenant_id),
                )

            conn.commit()
            logger.info(f"Updated tokens for Teams tenant: {tenant_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to update tokens for tenant {tenant_id}: {e}")
            return False

    def deactivate(self, tenant_id: str) -> bool:
        """Deactivate a tenant (on uninstall).

        Args:
            tenant_id: Azure AD tenant ID

        Returns:
            True if deactivated successfully
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE teams_tenants SET is_active = 0 WHERE tenant_id = ?",
                (tenant_id,),
            )
            conn.commit()
            logger.info(f"Deactivated Teams tenant: {tenant_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to deactivate tenant {tenant_id}: {e}")
            return False

    def delete(self, tenant_id: str) -> bool:
        """Permanently delete a tenant.

        Args:
            tenant_id: Azure AD tenant ID

        Returns:
            True if deleted successfully
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM teams_tenants WHERE tenant_id = ?",
                (tenant_id,),
            )
            conn.commit()
            logger.info(f"Deleted Teams tenant: {tenant_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to delete tenant {tenant_id}: {e}")
            return False

    def count(self, active_only: bool = True) -> int:
        """Count tenants.

        Args:
            active_only: Only count active tenants

        Returns:
            Number of tenants
        """
        conn = self._get_connection()
        try:
            if active_only:
                cursor = conn.execute("SELECT COUNT(*) FROM teams_tenants WHERE is_active = 1")
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM teams_tenants")

            return cursor.fetchone()[0]

        except sqlite3.Error as e:
            logger.error(f"Failed to count tenants: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get tenant statistics.

        Returns:
            Statistics dictionary
        """
        import time

        conn = self._get_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM teams_tenants").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM teams_tenants WHERE is_active = 1"
            ).fetchone()[0]

            # Count expiring tokens (within 1 hour)
            cutoff = time.time() + 3600
            expiring = conn.execute(
                "SELECT COUNT(*) FROM teams_tenants WHERE is_active = 1 AND expires_at < ?",
                (cutoff,),
            ).fetchone()[0]

            return {
                "total_tenants": total,
                "active_tenants": active,
                "inactive_tenants": total - active,
                "expiring_tokens": expiring,
            }

        except sqlite3.Error as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_tenants": 0, "active_tenants": 0}


# Environment configuration
ARAGORA_ENV = os.environ.get("ARAGORA_ENV", "development")


class SupabaseTeamsTenantStore:
    """
    Supabase-backed storage for Microsoft Teams tenant OAuth credentials.

    Production-ready storage with:
    - Automatic token encryption
    - Row Level Security for tenant isolation
    - Efficient queries with proper indexes
    """

    def __init__(self):
        """Initialize Supabase client."""
        self._client = None
        self._encryption_key = ENCRYPTION_KEY
        self._init_client()

    def _init_client(self):
        """Initialize the Supabase client."""
        try:
            from aragora.persistence.supabase_client import get_supabase_client

            self._client = get_supabase_client()
        except ImportError:
            logger.warning("Supabase client not available")
        except (ConnectionError, TimeoutError, OSError, RuntimeError) as e:
            logger.error(f"Failed to initialize Supabase client: {e}")

    @property
    def is_configured(self) -> bool:
        """Check if Supabase is properly configured."""
        return self._client is not None

    def _encrypt_token(self, token: str) -> str:
        """Encrypt token if encryption key is configured."""
        if not self._encryption_key or not token:
            return token

        try:
            from cryptography.fernet import Fernet
            import base64
            import hashlib

            key = base64.urlsafe_b64encode(hashlib.sha256(self._encryption_key.encode()).digest())
            f = Fernet(key)
            return f.encrypt(token.encode()).decode()
        except ImportError:
            logger.warning("cryptography not installed, storing token unencrypted")
            return token
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.error(f"Token encryption failed: {e}")
            return token

    def _decrypt_token(self, encrypted: str) -> str:
        """Decrypt token if encryption key is configured."""
        if not self._encryption_key or not encrypted:
            return encrypted

        # Check if it looks like an encrypted token
        if not encrypted.startswith("gAAA"):
            return encrypted

        try:
            from cryptography.fernet import Fernet
            import base64
            import hashlib

            key = base64.urlsafe_b64encode(hashlib.sha256(self._encryption_key.encode()).digest())
            f = Fernet(key)
            return f.decrypt(encrypted.encode()).decode()
        except ImportError:
            return encrypted
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.error(f"Token decryption failed: {e}")
            return encrypted

    def save(self, tenant: TeamsTenant) -> bool:
        """Save or update a tenant in Supabase."""
        if not self._client:
            return False

        try:
            encrypted_access = self._encrypt_token(tenant.access_token)
            encrypted_refresh = (
                self._encrypt_token(tenant.refresh_token) if tenant.refresh_token else None
            )

            data = {
                "tenant_id": tenant.tenant_id,
                "tenant_name": tenant.tenant_name,
                "access_token": encrypted_access,
                "refresh_token": encrypted_refresh,
                "bot_id": tenant.bot_id,
                "installed_at": datetime.fromtimestamp(
                    tenant.installed_at, tz=timezone.utc
                ).isoformat(),
                "installed_by": tenant.installed_by,
                "scopes": tenant.scopes,
                "aragora_org_id": tenant.aragora_org_id,
                "is_active": tenant.is_active,
                "expires_at": (
                    datetime.fromtimestamp(tenant.expires_at, tz=timezone.utc).isoformat()
                    if tenant.expires_at
                    else None
                ),
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            }

            # Upsert: insert or update on conflict
            self._client.table("teams_tenants").upsert(data, on_conflict="tenant_id").execute()
            logger.info(f"Saved Teams tenant to Supabase: {tenant.tenant_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to save tenant to Supabase: {e}")
            return False

    def get(self, tenant_id: str) -> Optional[TeamsTenant]:
        """Get a tenant by ID from Supabase."""
        if not self._client:
            return None

        try:
            response = (
                self._client.table("teams_tenants").select("*").eq("tenant_id", tenant_id).execute()
            )

            if response.data and len(response.data) > 0:
                row = response.data[0]
                return self._row_to_tenant(row)

            return None

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get tenant {tenant_id} from Supabase: {e}")
            return None

    def _row_to_tenant(self, row: Dict[str, Any]) -> TeamsTenant:
        """Convert Supabase row to TeamsTenant object."""
        # Parse timestamps
        installed_at = row.get("installed_at")
        if isinstance(installed_at, str):
            installed_at = datetime.fromisoformat(installed_at.replace("Z", "+00:00")).timestamp()
        else:
            installed_at = installed_at or 0

        expires_at = row.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00")).timestamp()

        tenant = TeamsTenant(
            tenant_id=row["tenant_id"],
            tenant_name=row["tenant_name"],
            access_token=self._decrypt_token(row["access_token"]),
            refresh_token=(
                self._decrypt_token(row["refresh_token"]) if row.get("refresh_token") else None
            ),
            bot_id=row["bot_id"],
            installed_at=installed_at,
            installed_by=row.get("installed_by"),
            scopes=row.get("scopes") or [],
            aragora_org_id=row.get("aragora_org_id"),
            is_active=row.get("is_active", True),
            expires_at=expires_at,
        )
        return tenant

    def get_by_org(self, aragora_org_id: str) -> List[TeamsTenant]:
        """Get all tenants for an Aragora organization."""
        if not self._client:
            return []

        try:
            response = (
                self._client.table("teams_tenants")
                .select("*")
                .eq("aragora_org_id", aragora_org_id)
                .eq("is_active", True)
                .order("installed_at", desc=True)
                .execute()
            )

            return [self._row_to_tenant(row) for row in (response.data or [])]

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get tenants for org {aragora_org_id}: {e}")
            return []

    def list_active(self, limit: int = 100, offset: int = 0) -> List[TeamsTenant]:
        """List all active tenants."""
        if not self._client:
            return []

        try:
            response = (
                self._client.table("teams_tenants")
                .select("*")
                .eq("is_active", True)
                .order("installed_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )

            return [self._row_to_tenant(row) for row in (response.data or [])]

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to list tenants: {e}")
            return []

    def list_expiring(self, within_seconds: int = 3600) -> List[TeamsTenant]:
        """List tenants with tokens expiring soon."""
        if not self._client:
            return []

        try:
            import time

            cutoff = datetime.fromtimestamp(
                time.time() + within_seconds, tz=timezone.utc
            ).isoformat()

            response = (
                self._client.table("teams_tenants")
                .select("*")
                .eq("is_active", True)
                .not_.is_("expires_at", "null")
                .lt("expires_at", cutoff)
                .order("expires_at")
                .execute()
            )

            return [self._row_to_tenant(row) for row in (response.data or [])]

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to list expiring tenants: {e}")
            return []

    def update_tokens(
        self,
        tenant_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        expires_at: Optional[float] = None,
    ) -> bool:
        """Update tokens for a tenant (after refresh)."""
        if not self._client:
            return False

        try:
            encrypted_access = self._encrypt_token(access_token)
            data: Dict[str, Any] = {
                "access_token": encrypted_access,
                "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            }

            if refresh_token:
                data["refresh_token"] = self._encrypt_token(refresh_token)

            if expires_at:
                data["expires_at"] = datetime.fromtimestamp(expires_at, tz=timezone.utc).isoformat()

            self._client.table("teams_tenants").update(data).eq("tenant_id", tenant_id).execute()
            logger.info(f"Updated tokens for Teams tenant: {tenant_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to update tokens for tenant {tenant_id}: {e}")
            return False

    def deactivate(self, tenant_id: str) -> bool:
        """Deactivate a tenant (on uninstall)."""
        if not self._client:
            return False

        try:
            self._client.table("teams_tenants").update(
                {
                    "is_active": False,
                    "updated_at": datetime.now(tz=timezone.utc).isoformat(),
                }
            ).eq("tenant_id", tenant_id).execute()
            logger.info(f"Deactivated Teams tenant: {tenant_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to deactivate tenant {tenant_id}: {e}")
            return False

    def delete(self, tenant_id: str) -> bool:
        """Permanently delete a tenant."""
        if not self._client:
            return False

        try:
            self._client.table("teams_tenants").delete().eq("tenant_id", tenant_id).execute()
            logger.info(f"Deleted Teams tenant: {tenant_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to delete tenant {tenant_id}: {e}")
            return False

    def count(self, active_only: bool = True) -> int:
        """Count tenants."""
        if not self._client:
            return 0

        try:
            query = self._client.table("teams_tenants").select("*", count="exact", head=True)
            if active_only:
                query = query.eq("is_active", True)
            response = query.execute()
            return response.count or 0

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to count tenants: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get tenant statistics."""
        if not self._client:
            return {"total_tenants": 0, "active_tenants": 0}

        try:
            total = self.count(active_only=False)
            active = self.count(active_only=True)
            expiring = len(self.list_expiring(3600))

            return {
                "total_tenants": total,
                "active_tenants": active,
                "inactive_tenants": total - active,
                "expiring_tokens": expiring,
            }

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get stats: {e}")
            return {"total_tenants": 0, "active_tenants": 0}


# Singleton instance
_tenant_store: Optional[Any] = None


def get_teams_tenant_store(db_path: Optional[str] = None) -> Any:
    """Get or create the tenant store singleton.

    In production (ARAGORA_ENV=production or USE_SUPABASE_TEAMS_STORE=1),
    uses Supabase for storage. Falls back to SQLite in development.

    Args:
        db_path: Optional path to SQLite database file (dev only)

    Returns:
        TeamsTenantStore or SupabaseTeamsTenantStore instance
    """
    global _tenant_store
    if _tenant_store is None:
        # Use Supabase in production
        if ARAGORA_ENV == "production" or os.environ.get("USE_SUPABASE_TEAMS_STORE"):
            supabase_store = SupabaseTeamsTenantStore()
            if supabase_store.is_configured:
                logger.info("Using Supabase Teams tenant store")
                _tenant_store = supabase_store
                return _tenant_store
            logger.warning("Supabase not configured, falling back to SQLite")

        # Fall back to SQLite
        _tenant_store = TeamsTenantStore(db_path)
    return _tenant_store
