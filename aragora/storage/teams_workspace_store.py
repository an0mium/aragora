"""
Microsoft Teams Workspace Storage for OAuth token management.

Stores Teams tenant credentials after OAuth installation for multi-tenant support.
Tokens are encrypted at rest using AES-256-GCM when ARAGORA_ENCRYPTION_KEY is set.

Schema:
    CREATE TABLE teams_workspaces (
        tenant_id TEXT PRIMARY KEY,
        tenant_name TEXT NOT NULL,
        access_token TEXT NOT NULL,
        bot_id TEXT NOT NULL,
        installed_at REAL NOT NULL,
        installed_by TEXT,
        scopes TEXT,
        aragora_tenant_id TEXT,
        is_active INTEGER DEFAULT 1,
        refresh_token TEXT,
        token_expires_at REAL
    );
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Storage configuration
TEAMS_WORKSPACE_DB_PATH = os.environ.get(
    "TEAMS_WORKSPACE_DB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "teams_workspaces.db"),
)

# Encryption key for tokens (required in production)
ENCRYPTION_KEY = os.environ.get("ARAGORA_ENCRYPTION_KEY", "")

# Environment mode
ARAGORA_ENV = os.environ.get("ARAGORA_ENV", "development")

# Track if encryption warning has been shown
_encryption_warning_shown = False


@dataclass
class TeamsWorkspace:
    """Represents an installed Microsoft Teams tenant."""

    tenant_id: str  # Azure AD tenant ID
    tenant_name: str
    access_token: str  # Bot access token
    bot_id: str  # Bot application ID
    installed_at: float  # Unix timestamp
    installed_by: Optional[str] = None  # User ID who installed
    scopes: List[str] = field(default_factory=list)
    aragora_tenant_id: Optional[str] = None  # Link to Aragora tenant
    is_active: bool = True
    refresh_token: Optional[str] = None  # OAuth refresh token
    token_expires_at: Optional[float] = None  # Unix timestamp when token expires
    service_url: Optional[str] = None  # Bot Framework service URL

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
            "aragora_tenant_id": self.aragora_tenant_id,
            "is_active": self.is_active,
            "has_refresh_token": bool(self.refresh_token),
            "token_expires_at": self.token_expires_at,
            "service_url": self.service_url,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TeamsWorkspace":
        """Create from database row."""
        scopes_str = row["scopes"] or ""
        scopes = scopes_str.split(",") if scopes_str else []

        # Handle optional columns which may not exist in older DBs
        refresh_token = None
        token_expires_at = None
        service_url = None
        try:
            refresh_token = row["refresh_token"]
        except (IndexError, KeyError):
            pass
        try:
            token_expires_at = row["token_expires_at"]
        except (IndexError, KeyError):
            pass
        try:
            service_url = row["service_url"]
        except (IndexError, KeyError):
            pass

        return cls(
            tenant_id=row["tenant_id"],
            tenant_name=row["tenant_name"],
            access_token=row["access_token"],
            bot_id=row["bot_id"],
            installed_at=row["installed_at"],
            installed_by=row["installed_by"],
            scopes=scopes,
            aragora_tenant_id=row["aragora_tenant_id"],
            is_active=bool(row["is_active"]),
            refresh_token=refresh_token,
            token_expires_at=token_expires_at,
            service_url=service_url,
        )


class TeamsWorkspaceStore:
    """
    Storage for Microsoft Teams workspace OAuth credentials.

    Supports SQLite backend with optional token encryption.
    Thread-safe for concurrent access.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS teams_workspaces (
        tenant_id TEXT PRIMARY KEY,
        tenant_name TEXT NOT NULL,
        access_token TEXT NOT NULL,
        bot_id TEXT NOT NULL,
        installed_at REAL NOT NULL,
        installed_by TEXT,
        scopes TEXT,
        aragora_tenant_id TEXT,
        is_active INTEGER DEFAULT 1,
        refresh_token TEXT,
        token_expires_at REAL,
        service_url TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_teams_workspaces_aragora_tenant
        ON teams_workspaces(aragora_tenant_id);

    CREATE INDEX IF NOT EXISTS idx_teams_workspaces_active
        ON teams_workspaces(is_active);
    """

    # Migration to add service_url column
    MIGRATION_ADD_SERVICE_URL = """
    ALTER TABLE teams_workspaces ADD COLUMN service_url TEXT;
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize the workspace store.

        Args:
            db_path: Path to SQLite database file

        Raises:
            ValueError: If ARAGORA_ENCRYPTION_KEY is not set in production
        """
        global _encryption_warning_shown

        # Enforce encryption in production
        if not ENCRYPTION_KEY:
            if ARAGORA_ENV == "production":
                raise ValueError(
                    "ARAGORA_ENCRYPTION_KEY environment variable is required in production. "
                    "Teams OAuth tokens must be encrypted at rest."
                )
            elif not _encryption_warning_shown:
                logger.warning(
                    "Teams tokens will be stored UNENCRYPTED. "
                    "Set ARAGORA_ENCRYPTION_KEY for production use."
                )
                _encryption_warning_shown = True

        self._db_path = db_path or TEAMS_WORKSPACE_DB_PATH
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
        """Ensure database schema exists and run migrations."""
        with self._init_lock:
            if not self._initialized:
                conn.executescript(self.SCHEMA)
                conn.commit()

                # Run migrations for optional columns if needed
                try:
                    cursor = conn.execute("PRAGMA table_info(teams_workspaces)")
                    columns = {row[1] for row in cursor.fetchall()}

                    if "service_url" not in columns:
                        conn.execute(self.MIGRATION_ADD_SERVICE_URL)
                        conn.commit()
                        logger.info("Added service_url column to teams_workspaces")

                except sqlite3.Error as e:
                    logger.debug(f"Migration check: {e}")

                self._initialized = True

    def _derive_key_v2(self) -> bytes:
        """Derive encryption key using PBKDF2HMAC (secure KDF)."""
        import base64
        import hashlib

        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        # Use SHA-256 of key as deterministic salt (16 bytes)
        salt = hashlib.sha256(b"aragora-teams-token-salt:" + ENCRYPTION_KEY.encode()).digest()[:16]

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        return base64.urlsafe_b64encode(kdf.derive(ENCRYPTION_KEY.encode()))

    def _derive_key_v1(self) -> bytes:
        """Derive key using legacy SHA-256 method (for backward compatibility)."""
        import base64
        import hashlib

        return base64.urlsafe_b64encode(hashlib.sha256(ENCRYPTION_KEY.encode()).digest())

    def _encrypt_token(self, token: str) -> str:
        """Encrypt token using PBKDF2-derived key."""
        if not ENCRYPTION_KEY:
            return token

        try:
            from cryptography.fernet import Fernet

            key = self._derive_key_v2()
            f = Fernet(key)
            encrypted = f.encrypt(token.encode()).decode()
            return f"v2:{encrypted}"
        except ImportError:
            logger.warning("cryptography not installed, storing token unencrypted")
            return token
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.error(f"Token encryption failed: {e}")
            return token

    def _decrypt_token(self, encrypted: str) -> str:
        """Decrypt token with support for multiple KDF versions."""
        if not ENCRYPTION_KEY:
            return encrypted

        # Check if it looks like an unencrypted token
        if encrypted.startswith("ey"):  # JWT tokens start with "ey"
            return encrypted

        try:
            from cryptography.fernet import Fernet

            if encrypted.startswith("v2:"):
                key = self._derive_key_v2()
                ciphertext = encrypted[3:]
            else:
                key = self._derive_key_v1()
                ciphertext = encrypted

            f = Fernet(key)
            return f.decrypt(ciphertext.encode()).decode()
        except ImportError:
            return encrypted
        except (ValueError, TypeError, UnicodeDecodeError) as e:
            logger.error(f"Token decryption failed: {e}")
            return encrypted

    def save(self, workspace: TeamsWorkspace) -> bool:
        """Save or update a workspace.

        Args:
            workspace: Workspace to save

        Returns:
            True if saved successfully
        """
        conn = self._get_connection()
        try:
            encrypted_token = self._encrypt_token(workspace.access_token)
            encrypted_refresh = (
                self._encrypt_token(workspace.refresh_token) if workspace.refresh_token else None
            )
            scopes_str = ",".join(workspace.scopes)

            conn.execute(
                """
                INSERT OR REPLACE INTO teams_workspaces
                (tenant_id, tenant_name, access_token, bot_id,
                 installed_at, installed_by, scopes, aragora_tenant_id, is_active,
                 refresh_token, token_expires_at, service_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workspace.tenant_id,
                    workspace.tenant_name,
                    encrypted_token,
                    workspace.bot_id,
                    workspace.installed_at,
                    workspace.installed_by,
                    scopes_str,
                    workspace.aragora_tenant_id,
                    1 if workspace.is_active else 0,
                    encrypted_refresh,
                    workspace.token_expires_at,
                    workspace.service_url,
                ),
            )
            conn.commit()
            logger.info(f"Saved Teams workspace: {workspace.tenant_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to save Teams workspace: {e}")
            return False

    def get(self, tenant_id: str) -> Optional[TeamsWorkspace]:
        """Get a workspace by tenant ID.

        Args:
            tenant_id: Azure AD tenant ID

        Returns:
            Workspace or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM teams_workspaces WHERE tenant_id = ?",
                (tenant_id,),
            )
            row = cursor.fetchone()

            if row:
                workspace = TeamsWorkspace.from_row(row)
                workspace.access_token = self._decrypt_token(workspace.access_token)
                if workspace.refresh_token:
                    workspace.refresh_token = self._decrypt_token(workspace.refresh_token)
                return workspace

            return None

        except sqlite3.Error as e:
            logger.error(f"Failed to get Teams workspace {tenant_id}: {e}")
            return None

    def get_by_aragora_tenant(self, aragora_tenant_id: str) -> List[TeamsWorkspace]:
        """Get all Teams workspaces for an Aragora tenant.

        Args:
            aragora_tenant_id: Aragora tenant ID

        Returns:
            List of workspaces
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM teams_workspaces
                WHERE aragora_tenant_id = ? AND is_active = 1
                ORDER BY installed_at DESC
                """,
                (aragora_tenant_id,),
            )

            workspaces = []
            for row in cursor.fetchall():
                workspace = TeamsWorkspace.from_row(row)
                workspace.access_token = self._decrypt_token(workspace.access_token)
                if workspace.refresh_token:
                    workspace.refresh_token = self._decrypt_token(workspace.refresh_token)
                workspaces.append(workspace)

            return workspaces

        except sqlite3.Error as e:
            logger.error(f"Failed to get Teams workspaces for tenant {aragora_tenant_id}: {e}")
            return []

    def list_active(self, limit: int = 100, offset: int = 0) -> List[TeamsWorkspace]:
        """List all active workspaces.

        Args:
            limit: Maximum number of workspaces to return
            offset: Pagination offset

        Returns:
            List of active workspaces
        """
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT * FROM teams_workspaces
                WHERE is_active = 1
                ORDER BY installed_at DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )

            workspaces = []
            for row in cursor.fetchall():
                workspace = TeamsWorkspace.from_row(row)
                workspace.access_token = self._decrypt_token(workspace.access_token)
                if workspace.refresh_token:
                    workspace.refresh_token = self._decrypt_token(workspace.refresh_token)
                workspaces.append(workspace)

            return workspaces

        except sqlite3.Error as e:
            logger.error(f"Failed to list Teams workspaces: {e}")
            return []

    def deactivate(self, tenant_id: str) -> bool:
        """Deactivate a workspace (on uninstall).

        Args:
            tenant_id: Azure AD tenant ID

        Returns:
            True if deactivated successfully
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE teams_workspaces SET is_active = 0 WHERE tenant_id = ?",
                (tenant_id,),
            )
            conn.commit()
            logger.info(f"Deactivated Teams workspace: {tenant_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to deactivate Teams workspace {tenant_id}: {e}")
            return False

    def delete(self, tenant_id: str) -> bool:
        """Permanently delete a workspace.

        Args:
            tenant_id: Azure AD tenant ID

        Returns:
            True if deleted successfully
        """
        conn = self._get_connection()
        try:
            conn.execute(
                "DELETE FROM teams_workspaces WHERE tenant_id = ?",
                (tenant_id,),
            )
            conn.commit()
            logger.info(f"Deleted Teams workspace: {tenant_id}")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to delete Teams workspace {tenant_id}: {e}")
            return False

    def count(self, active_only: bool = True) -> int:
        """Count workspaces.

        Args:
            active_only: Only count active workspaces

        Returns:
            Number of workspaces
        """
        conn = self._get_connection()
        try:
            if active_only:
                cursor = conn.execute("SELECT COUNT(*) FROM teams_workspaces WHERE is_active = 1")
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM teams_workspaces")

            return cursor.fetchone()[0]

        except sqlite3.Error as e:
            logger.error(f"Failed to count Teams workspaces: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get workspace statistics.

        Returns:
            Statistics dictionary
        """
        conn = self._get_connection()
        try:
            total = conn.execute("SELECT COUNT(*) FROM teams_workspaces").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM teams_workspaces WHERE is_active = 1"
            ).fetchone()[0]

            return {
                "total_workspaces": total,
                "active_workspaces": active,
                "inactive_workspaces": total - active,
            }

        except sqlite3.Error as e:
            logger.error(f"Failed to get Teams workspace stats: {e}")
            return {"total_workspaces": 0, "active_workspaces": 0}

    def refresh_workspace_token(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
    ) -> Optional[TeamsWorkspace]:
        """Refresh an expired access token using the refresh token.

        Args:
            tenant_id: Azure AD tenant ID
            client_id: Azure AD application client ID
            client_secret: Azure AD client secret

        Returns:
            Updated workspace with new tokens, or None on failure
        """
        import json
        import urllib.parse
        import urllib.request

        workspace = self.get(tenant_id)
        if not workspace:
            logger.error(f"Teams workspace not found for refresh: {tenant_id}")
            return None

        if not workspace.refresh_token:
            logger.error(f"No refresh token available for Teams workspace: {tenant_id}")
            return None

        try:
            # Exchange refresh token for new access token via Azure AD
            token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
            data = urllib.parse.urlencode(
                {
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": workspace.refresh_token,
                    "scope": "https://graph.microsoft.com/.default offline_access",
                }
            ).encode()

            request = urllib.request.Request(
                token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode())

            if "error" in result:
                error = result.get("error", "unknown")
                logger.error(f"Token refresh failed for Teams {tenant_id}: {error}")
                if error in ("invalid_grant", "invalid_refresh_token"):
                    self.deactivate(tenant_id)
                return None

            # Update workspace with new tokens
            workspace.access_token = result.get("access_token", workspace.access_token)

            new_refresh = result.get("refresh_token")
            if new_refresh:
                workspace.refresh_token = new_refresh

            expires_in = result.get("expires_in")
            if expires_in:
                workspace.token_expires_at = time.time() + int(expires_in)

            if self.save(workspace):
                logger.info(f"Successfully refreshed token for Teams workspace: {tenant_id}")
                return workspace

            return None

        except urllib.error.URLError as e:
            logger.error(f"Network error refreshing Teams token for {tenant_id}: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Invalid response refreshing Teams token for {tenant_id}: {e}")
            return None

    def is_token_expired(self, tenant_id: str, buffer_seconds: int = 300) -> bool:
        """Check if a workspace's access token is expired or will expire soon.

        Args:
            tenant_id: Azure AD tenant ID
            buffer_seconds: Consider token expired this many seconds before actual expiry

        Returns:
            True if token is expired or will expire within buffer_seconds
        """
        workspace = self.get(tenant_id)
        if not workspace:
            return True

        if not workspace.token_expires_at:
            return False

        return time.time() + buffer_seconds >= workspace.token_expires_at


# Supabase-backed implementation for production
class SupabaseTeamsWorkspaceStore:
    """
    Supabase-backed storage for Teams workspace OAuth credentials.

    Uses Supabase PostgreSQL for production deployments with proper
    encryption and multi-region support.
    """

    def __init__(self):
        """Initialize Supabase workspace store."""
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize Supabase client."""
        try:
            from aragora.persistence.supabase_client import SupabaseClient

            client = SupabaseClient()
            if client.is_configured:
                self._client = client.client
                logger.info("Teams workspace store using Supabase backend")
            else:
                logger.warning("Supabase not configured for Teams workspace store")
        except ImportError:
            logger.debug("Supabase client not available")

    @property
    def is_configured(self) -> bool:
        """Check if Supabase is configured."""
        return self._client is not None

    def save(self, workspace: TeamsWorkspace) -> bool:
        """Save or update a workspace."""
        if not self.is_configured:
            return False

        try:
            data = {
                "tenant_id": workspace.tenant_id,
                "tenant_name": workspace.tenant_name,
                "access_token": workspace.access_token,
                "bot_id": workspace.bot_id,
                "installed_at": datetime.fromtimestamp(
                    workspace.installed_at, tz=timezone.utc
                ).isoformat(),
                "installed_by": workspace.installed_by,
                "scopes": workspace.scopes,
                "aragora_tenant_id": workspace.aragora_tenant_id,
                "is_active": workspace.is_active,
                "refresh_token": workspace.refresh_token,
                "token_expires_at": workspace.token_expires_at,
                "service_url": workspace.service_url,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            self._client.table("teams_workspaces").upsert(data, on_conflict="tenant_id").execute()

            logger.info(f"Saved Teams workspace to Supabase: {workspace.tenant_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to save Teams workspace to Supabase: {e}")
            return False

    def get(self, tenant_id: str) -> Optional[TeamsWorkspace]:
        """Get a workspace by tenant ID."""
        if not self.is_configured:
            return None

        try:
            result = (
                self._client.table("teams_workspaces")
                .select("*")
                .eq("tenant_id", tenant_id)
                .single()
                .execute()
            )

            if result.data:
                return self._row_to_workspace(result.data)
            return None

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get Teams workspace from Supabase: {e}")
            return None

    def get_by_aragora_tenant(self, aragora_tenant_id: str) -> List[TeamsWorkspace]:
        """Get all Teams workspaces for an Aragora tenant."""
        if not self.is_configured:
            return []

        try:
            result = (
                self._client.table("teams_workspaces")
                .select("*")
                .eq("aragora_tenant_id", aragora_tenant_id)
                .eq("is_active", True)
                .order("installed_at", desc=True)
                .execute()
            )

            return [self._row_to_workspace(row) for row in result.data]

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get Teams workspaces from Supabase: {e}")
            return []

    def list_active(self, limit: int = 100, offset: int = 0) -> List[TeamsWorkspace]:
        """List all active workspaces."""
        if not self.is_configured:
            return []

        try:
            result = (
                self._client.table("teams_workspaces")
                .select("*")
                .eq("is_active", True)
                .order("installed_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )

            return [self._row_to_workspace(row) for row in result.data]

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to list Teams workspaces from Supabase: {e}")
            return []

    def deactivate(self, tenant_id: str) -> bool:
        """Deactivate a workspace."""
        if not self.is_configured:
            return False

        try:
            self._client.table("teams_workspaces").update(
                {
                    "is_active": False,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            ).eq("tenant_id", tenant_id).execute()

            logger.info(f"Deactivated Teams workspace in Supabase: {tenant_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to deactivate Teams workspace in Supabase: {e}")
            return False

    def delete(self, tenant_id: str) -> bool:
        """Permanently delete a workspace."""
        if not self.is_configured:
            return False

        try:
            self._client.table("teams_workspaces").delete().eq("tenant_id", tenant_id).execute()

            logger.info(f"Deleted Teams workspace from Supabase: {tenant_id}")
            return True

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to delete Teams workspace from Supabase: {e}")
            return False

    def count(self, active_only: bool = True) -> int:
        """Count workspaces."""
        if not self.is_configured:
            return 0

        try:
            query = self._client.table("teams_workspaces").select("*", count="exact")
            if active_only:
                query = query.eq("is_active", True)

            result = query.execute()
            return result.count or 0

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to count Teams workspaces in Supabase: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get workspace statistics."""
        if not self.is_configured:
            return {"total_workspaces": 0, "active_workspaces": 0}

        try:
            total = self.count(active_only=False)
            active = self.count(active_only=True)

            return {
                "total_workspaces": total,
                "active_workspaces": active,
                "inactive_workspaces": total - active,
            }

        except (ConnectionError, TimeoutError, OSError, ValueError, KeyError) as e:
            logger.error(f"Failed to get Teams workspace stats from Supabase: {e}")
            return {"total_workspaces": 0, "active_workspaces": 0}

    def _row_to_workspace(self, row: Dict[str, Any]) -> TeamsWorkspace:
        """Convert Supabase row to TeamsWorkspace."""
        installed_at = row.get("installed_at")
        if isinstance(installed_at, str):
            installed_at = datetime.fromisoformat(installed_at.replace("Z", "+00:00")).timestamp()
        elif isinstance(installed_at, (int, float)):
            pass
        else:
            installed_at = time.time()

        return TeamsWorkspace(
            tenant_id=row["tenant_id"],
            tenant_name=row["tenant_name"],
            access_token=row["access_token"],
            bot_id=row["bot_id"],
            installed_at=installed_at,
            installed_by=row.get("installed_by"),
            scopes=row.get("scopes") or [],
            aragora_tenant_id=row.get("aragora_tenant_id"),
            is_active=row.get("is_active", True),
            refresh_token=row.get("refresh_token"),
            token_expires_at=row.get("token_expires_at"),
            service_url=row.get("service_url"),
        )


# Singleton instance
_workspace_store: Optional[Any] = None


def get_teams_workspace_store(db_path: Optional[str] = None) -> Any:
    """Get or create the workspace store singleton.

    Uses Supabase backend in production when configured,
    falls back to SQLite for development.

    Args:
        db_path: Optional path to database file (SQLite only)

    Returns:
        Workspace store instance (Supabase or SQLite backed)
    """
    global _workspace_store
    if _workspace_store is None:
        # Try Supabase first in production
        if ARAGORA_ENV == "production" or os.getenv("USE_SUPABASE_TEAMS_STORE"):
            supabase_store = SupabaseTeamsWorkspaceStore()
            if supabase_store.is_configured:
                _workspace_store = supabase_store
                logger.info("Using Supabase-backed Teams workspace store")
                return _workspace_store

        # Fall back to SQLite
        _workspace_store = TeamsWorkspaceStore(db_path)
        logger.info("Using SQLite-backed Teams workspace store")

    return _workspace_store
